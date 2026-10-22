"""Pixel decoder (design B) — deformable neck + PatchDecode upsample tail + surface gate.

The design-B replacement for the UNet :class:`PixelDecoder`, following the Dinovol author's surface predictor
(``docs/dinovol_sheet_architecture.md`` §3):

  1. **Neck (part A)** — optional :class:`DeformableNeck` cross-scale refinement over {1/8 tokens, 1/4 s2}.
  2. **Upsample tail (part B)** — the author's **PatchDecode**: SAM-style ConvTranspose3d upsampling 1/8 → 1/1
     with **no skips** (the voxel detail is relearned by the convs, not carried by stem skips). Two 1×1 heads off
     the 1/1 trunk: a **surface head** (``n_sem_classes``, supervised DIRECTLY with the thin-surface BCE stack,
     author-style) and a **mask-feature head** (``mask_dim`` = 16, the shared embeddings for the instance head).
  3. **Surface concat** — ``instance_features = concat([softmax(surface), mask_feat])``: the per-voxel surface
     *probabilities* (``n_sem_classes`` dims; bg→≈(1,0), sheet→≈(0,1)) are **concatenated** onto the mask
     features rather than multiplied in. This puts "is this a sheet" into a **direction** (the two surface
     prototypes) instead of a magnitude — robust to upstream normalisation, and it gives the instance head's
     single null query a shared bg-direction to key on (background is heterogeneous in the encoder/mask
     embedding but homogeneous in ``p_bg``). The instance head is then left to do ID (direction) only; the
     surface head carries the "find the medial" job on its own. See ``docs/dinovol_sheet_architecture.md`` §3.

Deep supervision (optional): surface logits also at 1/2 and 1/4, so ``semantic_loss`` can supervise all three.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .pixel_decoder import _conv_block
from .neck import DeformableNeck


def _gn_groups(c: int) -> int:
    """Largest group count (≤8) that divides ``c`` — for GroupNorm on the final decoder block."""
    for g in (8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1


def _conv_block_gn(in_ch: int, out_ch: int) -> nn.Sequential:
    """Like :func:`_conv_block` but GroupNorm (not InstanceNorm) — used for the LAST block feeding the mask/
    surface heads. InstanceNorm re-scales every channel independently over the window (per-channel, spatially),
    which rotates the feature direction and makes magnitude window-relative; GroupNorm normalises channel groups
    together, preserving relative channel magnitudes so the mask-feature *direction* (which now carries instance
    identity) survives (``docs/dinovol_sheet_architecture.md`` §3)."""
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(_gn_groups(out_ch), out_ch, affine=True),
        nn.LeakyReLU(inplace=True),
    )


class PatchDecoder(nn.Module):
    """Deformable neck → PatchDecode upsample (1/8→1/1, skipless) → surface head + gated mask features.

    Args:
        token_dim: 1/8 token width (288).
        s2_dim: 1/4 stem-skip width (128) — fed to the neck only.
        mask_dim: shared mask-feature width (16).
        n_sem_classes: surface classes (2 = background / sheet-surface).
        neck: ``"deformable"`` (part A) or ``"none"`` (author-exact: tokens straight into the upsample).
        up_channels: PatchDecode widths at (1/8, 1/4, 1/2, 1/1).
        surface_gate: gate the mask features by σ(surface) for the instance head.
        deep_supervision: also emit surface logits at 1/2 and 1/4.
    """

    def __init__(
        self,
        *,
        token_dim: int = 288,
        s2_dim: int = 128,
        mask_dim: int = 16,
        n_sem_classes: int = 2,
        neck: str = "deformable",
        neck_layers: int = 6,
        neck_heads: int = 4,
        neck_points: int = 4,
        up_channels=(128, 64, 32, 16),
        surface_gate: bool = True,
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        if neck not in ("deformable", "none"):
            raise ValueError(f"neck must be 'deformable' or 'none', got {neck!r}.")
        self.neck = (DeformableNeck(token_dim, s2_dim, n_layers=neck_layers, n_heads=neck_heads,
                                    n_points=neck_points) if neck == "deformable" else None)
        c0, c1, c2, c3 = (int(c) for c in up_channels)
        self.proj8 = _conv_block(token_dim, c0)                           # 1/8  (16³)
        self.up4 = nn.ConvTranspose3d(c0, c1, kernel_size=2, stride=2)    # 1/4  (32³)
        self.block4 = _conv_block(c1, c1)
        self.up2 = nn.ConvTranspose3d(c1, c2, kernel_size=2, stride=2)    # 1/2  (64³)
        self.block2 = _conv_block(c2, c2)
        self.up1 = nn.ConvTranspose3d(c2, c3, kernel_size=2, stride=2)    # 1/1  (128³)
        self.block1 = _conv_block_gn(c3, c3)                             # GroupNorm (feeds mask+surface heads; §3)
        self.surface_head = nn.Conv3d(c3, int(n_sem_classes), kernel_size=1)
        self.mask_head = nn.Conv3d(c3, int(mask_dim), kernel_size=1)
        self.deep_supervision = bool(deep_supervision)
        if self.deep_supervision:
            self.surface_head_2 = nn.Conv3d(c2, int(n_sem_classes), kernel_size=1)   # 1/2
            self.surface_head_4 = nn.Conv3d(c1, int(n_sem_classes), kernel_size=1)   # 1/4
        self.surface_gate = bool(surface_gate)
        self.n_sem_classes = int(n_sem_classes)
        self.mask_dim = int(mask_dim)
        # the per-voxel feature the instance head reads: mask_dim, plus the n_sem surface-prob dims when concat is on
        self.instance_feat_dim = int(mask_dim) + (int(n_sem_classes) if self.surface_gate else 0)

    def forward(self, tokens_grid: torch.Tensor, s2: torch.Tensor) -> dict:
        """``tokens_grid`` (B,token_dim,16³); ``s2`` (B,s2_dim,32³). Returns ``{"sem_logits": [full,(1/2,1/4)],
        "mask_features": (B,mask_dim,128³), "instance_features": (B,n_sem+mask_dim,128³ when concat)}``."""
        n8 = self.neck(tokens_grid, s2) if self.neck is not None else tokens_grid
        x8 = self.proj8(n8)                                              # c0 @16³
        x4 = self.block4(self.up4(x8))                                   # c1 @32³
        x2 = self.block2(self.up2(x4))                                   # c2 @64³
        x1 = self.block1(self.up1(x2))                                   # c3 @128³
        sem = self.surface_head(x1)
        sem_logits = [sem]
        if self.deep_supervision:
            sem_logits.append(self.surface_head_2(x2))
            sem_logits.append(self.surface_head_4(x4))
        mf = self.mask_head(x1)
        if self.surface_gate:
            # CONCAT (not gate): surface probs become extra feature dims so sheet/no-sheet is a DIRECTION
            # (bg→≈(1,0), sheet→≈(0,1)), not a magnitude. §3.
            p = torch.softmax(sem.float(), dim=1).to(mf.dtype)          # (B, n_sem, 128³) full simplex
            inst_feat = torch.cat([p, mf], dim=1)                        # (B, n_sem+mask_dim, 128³)
        else:
            inst_feat = mf
        return {"sem_logits": sem_logits, "mask_features": mf, "instance_features": inst_feat}

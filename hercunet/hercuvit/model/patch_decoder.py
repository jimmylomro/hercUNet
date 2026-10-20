"""Pixel decoder (design B) — deformable neck + PatchDecode upsample tail + surface gate.

The design-B replacement for the UNet :class:`PixelDecoder`, following the Dinovol author's surface predictor
(``docs/dinovol_sheet_architecture.md`` §3):

  1. **Neck (part A)** — optional :class:`DeformableNeck` cross-scale refinement over {1/8 tokens, 1/4 s2}.
  2. **Upsample tail (part B)** — the author's **PatchDecode**: SAM-style ConvTranspose3d upsampling 1/8 → 1/1
     with **no skips** (the voxel detail is relearned by the convs, not carried by stem skips). Two 1×1 heads off
     the 1/1 trunk: a **surface head** (``n_sem_classes``, supervised DIRECTLY with the thin-surface BCE stack,
     author-style) and a **mask-feature head** (``mask_dim`` = 16, the shared embeddings for the instance head).
  3. **Surface gate** — ``gated = mask_feat · σ(surface)``: the per-voxel surface probability gates the mask
     features, so the instance head reasons on **surface-aware** features and the surface head carries the
     "find the medial" job on its own (the instance head is left to do ID only). Differentiable soft gate.

Deep supervision (optional): surface logits also at 1/2 and 1/4, so ``semantic_loss`` can supervise all three.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .pixel_decoder import _conv_block
from .neck import DeformableNeck


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
        self.block1 = _conv_block(c3, c3)
        self.surface_head = nn.Conv3d(c3, int(n_sem_classes), kernel_size=1)
        self.mask_head = nn.Conv3d(c3, int(mask_dim), kernel_size=1)
        self.deep_supervision = bool(deep_supervision)
        if self.deep_supervision:
            self.surface_head_2 = nn.Conv3d(c2, int(n_sem_classes), kernel_size=1)   # 1/2
            self.surface_head_4 = nn.Conv3d(c1, int(n_sem_classes), kernel_size=1)   # 1/4
        self.surface_gate = bool(surface_gate)
        self.n_sem_classes = int(n_sem_classes)
        self.mask_dim = int(mask_dim)

    def forward(self, tokens_grid: torch.Tensor, s2: torch.Tensor) -> dict:
        """``tokens_grid`` (B,token_dim,16³); ``s2`` (B,s2_dim,32³). Returns ``{"sem_logits": [full,(1/2,1/4)],
        "mask_features": (B,mask_dim,128³), "gated_features": (B,mask_dim,128³)}``."""
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
            p = torch.softmax(sem.float(), dim=1)[:, 1:2].to(mf.dtype)   # P(surface) per voxel, (B,1,128³)
            gated = mf * p
        else:
            gated = mf
        return {"sem_logits": sem_logits, "mask_features": mf, "gated_features": gated}

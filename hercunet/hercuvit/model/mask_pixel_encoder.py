"""Design-C pixel encoder — a true Mask2Former / Deformable-DETR multi-scale encoder with the 1/4 scale as a
FULL level (``docs/hercuvit-mask-architecture.md``).

Replaces the design-B neck (which built 1/8+1/4 tokens, refined both, then **discarded** the 1/4) + the skipless
upsample tail. Here all four levels — 1/4 (stem s2), 1/8 (ViT, projected 288→128), 1/16 and 1/32 (learned
stride-2 downsamples of 1/8) — are flattened into ONE token sequence, every token (incl. all 1/4 tokens) is a
query, and 6 multi-scale deformable-attention layers refine all of them jointly. The encoder then **keeps** all
four refined maps:

  * ``mask_features`` = refined **1/4** → **linear** 1×1 (128→16), **no activation** (dot-product readout).
  * ``sem_logits``    = refined **1/4** → a small upsample head to 1/1 (+ deep-sup at 1/2, 1/4) for the surface
                        stack (fg-weighted CE + Focal-Tversky + clDice + separation — unchanged).
  * ``memory``        = refined **1/32, 1/16, 1/8** (128-d) for the decoder's round-robin masked attention.

All four levels are 128-d; the decoder hidden width is 128 too, so the memory needs no projection.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .neck import NeckLayer, _split_to_grids
from .instance_head import _coord_grid
from .patch_decoder import _gn_groups


def _sine_pe_3d(shape: Sequence[int], dim: int, device, dtype) -> torch.Tensor:
    """Parameter-free 3-D sine positional encoding → ``(d·h·w, dim)`` (raster z,y,x). ``dim`` split across the 3 axes."""
    dz, dy, dx = shape
    coords = _coord_grid(shape, device, torch.float32)                    # (G,3) in [-1,1]
    per = dim // 3
    freqs = torch.arange(per // 2, device=device, dtype=torch.float32)
    freqs = 1.0 / (10000.0 ** (2 * freqs / max(per, 1)))                   # (per/2,)
    outs = []
    for a in range(3):
        ang = coords[:, a:a + 1] * freqs.unsqueeze(0)                      # (G, per/2)
        outs.append(torch.cat([ang.sin(), ang.cos()], dim=-1))            # (G, per)
    pe = torch.cat(outs, dim=-1)                                          # (G, 3·per)
    if pe.shape[1] < dim:                                                 # pad remainder
        pe = F.pad(pe, (0, dim - pe.shape[1]))
    return pe.to(dtype)


def _conv_gn(in_ch: int, out_ch: int, *, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(_gn_groups(out_ch), out_ch),
        nn.GELU(),
    )


class MaskPixelEncoder(nn.Module):
    """4-level (1/4·1/8·1/16·1/32) multi-scale deformable pixel encoder. Returns refined maps + mask features +
    surface logits (``docs/hercuvit-mask-architecture.md`` §1-§4)."""

    def __init__(self, *, token_dim: int = 288, s2_dim: int = 128, dim: int = 128, n_layers: int = 6,
                 n_heads: int = 8, n_points: int = 4, mlp_ratio: int = 4, n_sem_classes: int = 2,
                 mask_dim: int = 16, deep_supervision: bool = True) -> None:
        super().__init__()
        self.dim = int(dim)
        self.n_sem_classes = int(n_sem_classes)
        self.mask_dim = int(mask_dim)
        self.deep_supervision = bool(deep_supervision)
        self.proj8 = nn.Conv3d(token_dim, dim, kernel_size=1)             # ViT 1/8: 288 → 128
        self.proj4 = nn.Conv3d(s2_dim, dim, kernel_size=1) if s2_dim != dim else nn.Identity()  # stem 1/4: 128
        self.down16 = _conv_gn(dim, dim, stride=2)                        # 1/8 → 1/16
        self.down32 = _conv_gn(dim, dim, stride=2)                        # 1/16 → 1/32
        self.level_embed = nn.Parameter(torch.zeros(4, dim))             # learned per-level embedding
        nn.init.normal_(self.level_embed, std=0.02)
        self.layers = nn.ModuleList(
            [NeckLayer(dim, n_scales=4, n_heads=n_heads, n_points=n_points, mlp_ratio=mlp_ratio)
             for _ in range(int(n_layers))]
        )
        self.mask_proj = nn.Linear(dim, int(mask_dim))                    # refined 1/4 → mask features; NO activation
        # mask_features are L2-normalized jointly over all `mask_dim` dims (in forward) and SPLIT into subspaces:
        #   dims [0:surf_dim)  = SURFACE subspace — the (nonlinear) surface head reads ONLY these.
        #   dims [surf_dim:]   = INSTANCE subspace — the surface head never touches these.
        # The instance mask readout still dots the FULL mask_dim vector, so the surface dims act as a shared fg/bg
        # axis the null query keys on, while the surface loss can't collapse the instance dims (no competition tax).
        md = int(mask_dim)
        self.surf_dim = min(3, md)                                       # surface subspace width (dims 0..surf_dim-1)
        sd = self.surf_dim
        # surface head STAYS at the subspace width `sd` — never re-expanded to mask_dim (that would defeat the
        # bottleneck). Light learnable upsample 1/4→1/1 with a 3×3 refine + GELU per level; no GroupNorm (the
        # source mask_features are already L2-normed, and GN on sd=3 channels would be near per-channel).
        self.s_up2 = nn.ConvTranspose3d(sd, sd, kernel_size=2, stride=2)  # 1/4 → 1/2
        self.s_block2 = nn.Conv3d(sd, sd, kernel_size=3, padding=1)
        self.s_up1 = nn.ConvTranspose3d(sd, sd, kernel_size=2, stride=2)  # 1/2 → 1/1
        self.s_block1 = nn.Conv3d(sd, sd, kernel_size=3, padding=1)
        self.s_act = nn.GELU()
        self.surf1 = nn.Conv3d(sd, n_sem_classes, kernel_size=1)         # 1/1
        if self.deep_supervision:
            self.surf2 = nn.Conv3d(sd, n_sem_classes, kernel_size=1)     # 1/2
            self.surf4 = nn.Conv3d(sd, n_sem_classes, kernel_size=1)     # 1/4 (direct on surface subspace)

    def forward(self, tokens_grid: torch.Tensor, s2: torch.Tensor) -> dict:
        """``tokens_grid`` (B,token_dim,16³) ViT 1/8; ``s2`` (B,s2_dim,32³) stem 1/4. Returns dict with
        ``memory`` [1/32,1/16,1/8], ``feat4`` (refined 1/4), ``mask_features`` (B,mask_dim,32³), ``sem_logits``."""
        B = tokens_grid.shape[0]
        f8 = self.proj8(tokens_grid)                                     # 128 @ 16³
        f4 = self.proj4(s2)                                             # 128 @ 32³
        f16 = self.down16(f8)                                          # 128 @ 8³
        f32 = self.down32(f16)                                         # 128 @ 4³
        grids = [f4, f8, f16, f32]                                     # level order 1/4,1/8,1/16,1/32
        shapes = [tuple(g.shape[2:]) for g in grids]
        dev, dt = f8.device, f8.dtype
        ref = torch.cat([_coord_grid(s, dev, dt) for s in shapes], dim=0).unsqueeze(0).expand(B, -1, -1)  # (B,N,3)
        seq_parts, ref_check = [], 0
        for li, g in enumerate(grids):
            tok = g.flatten(2).transpose(1, 2)                          # (B, n, dim)
            pe = _sine_pe_3d(shapes[li], self.dim, dev, dt).unsqueeze(0)  # (1,n,dim)
            seq_parts.append(tok + pe + self.level_embed[li].view(1, 1, -1))
            ref_check += tok.shape[1]
        seq = torch.cat(seq_parts, dim=1)                              # (B, N, dim)
        for layer in self.layers:
            seq = layer(seq, ref, shapes)
        r4, r8, r16, r32 = _split_to_grids(seq, shapes, self.dim)       # refined per-level maps
        mask_features = self.mask_proj(r4.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(
            B, self.mask_dim, *shapes[0])                               # (B, mask_dim, 32³) — NO activation
        mask_features = F.normalize(mask_features, dim=1)              # L2-norm jointly over all mask_dim (per voxel)
        surf_sub = mask_features[:, :self.surf_dim]                     # surface subspace (B, surf_dim, 32³)
        xs2 = self.s_act(self.s_block2(self.s_up2(surf_sub)))         # 1/2 — surface head reads ONLY surf subspace (GELU)
        xs1 = self.s_act(self.s_block1(self.s_up1(xs2)))             # 1/1 (GELU)
        sem_logits = [self.surf1(xs1)]
        if self.deep_supervision:
            sem_logits.append(self.surf2(xs2))
            sem_logits.append(self.surf4(surf_sub))
        # monitor: share of the unit norm in the surface subspace (expect ~surf_dim/mask_dim ≈ 0.19; →1 = collapse)
        surf_share = surf_sub.pow(2).sum(1).mean()
        return {"memory": [r32, r16, r8], "feat4": r4, "mask_features": mask_features,
                "sem_logits": sem_logits, "surf_share": surf_share}

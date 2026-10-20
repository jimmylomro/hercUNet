"""Deformable neck (design B, part A) — Mask2Former-style multi-scale deformable-attention refinement.

A cross-scale refinement stack between the frozen encoder and the upsample tail (``PatchDecoder``). It runs
``n_layers`` of **3-D multi-scale deformable self-attention** over the two coarse scales the stem gives us —
1/8 (the 288-d tokens, 16³) and 1/4 (the s2 stem skip, 128@32³) — so each location mixes cross-scale context
*before* the convolutional upsample. This is the M2F pixel-decoder NECK (MSDeformAttn encoder), not a mask-head
change (``docs/dinovol_sheet_architecture.md`` §3).

**Pure-torch, grid_sample-based** (like :class:`DeformableCrossAttn`): no custom MSDeformAttn CUDA kernel, so it
builds on Blackwell (sm_120) / torch 2.8 where the Deformable-DETR kernel is fragile (§13.4). Deformable
attention is **linear** in the token count, so the 1/8+1/4 scales (~37k tokens) never materialise an N² matrix —
the memory win that keeps the neck affordable at 1/4.

The neck starts as an **identity** (output projections zero-init, residual branches), so a run on the frozen
encoder is undisturbed at step 0 and the neck learns its refinement from there.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .instance_head import _coord_grid


class MSDeformAttn3D(nn.Module):
    """Multi-scale 3-D deformable attention (Deformable-DETR encoder style). Each query predicts ``n_points``
    sampling offsets per (head, scale) around its own reference point and attends over the bilinearly-sampled
    values, softmax-normalised over all (scale × point) samples.

    Built on :func:`torch.nn.functional.grid_sample` — a first-class CUDA kernel (verified on Blackwell sm_120) —
    exactly as detectron2/mmcv's ``multi_scale_deformable_attention_pytorch`` reference (there is no fused *3-D*
    MSDeformAttn kernel; the Deformable-DETR CUDA op is 2-D and sm_120-fragile to compile). Memory-lean: all heads
    are folded into the batch dim so each scale is ONE grid_sample call, and levels accumulate (no stacked
    ``(B,N,h,L,k,hd)`` buffer); kept in the autocast dtype (bf16) throughout."""

    offset_scale: float = 0.1                                                     # normalised offset range around ref

    def __init__(self, dim: int, n_scales: int, n_heads: int = 4, n_points: int = 4) -> None:
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} not divisible by n_heads {n_heads}"
        self.dim, self.L, self.h, self.k = dim, int(n_scales), int(n_heads), int(n_points)
        self.hd = dim // n_heads
        self.sampling_offsets = nn.Linear(dim, n_heads * n_scales * n_points * 3)
        self.attn_weights = nn.Linear(dim, n_heads * n_scales * n_points)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.sampling_offsets.weight); nn.init.zeros_(self.sampling_offsets.bias)
        nn.init.zeros_(self.attn_weights.weight); nn.init.zeros_(self.attn_weights.bias)
        nn.init.xavier_uniform_(self.value_proj.weight); nn.init.zeros_(self.value_proj.bias)
        nn.init.zeros_(self.out_proj.weight); nn.init.zeros_(self.out_proj.bias)   # identity start (residual)

    def forward(self, q: torch.Tensor, ref: torch.Tensor, value_maps: Sequence[torch.Tensor]) -> torch.Tensor:
        """``q`` (B,N,dim) queries; ``ref`` (B,N,3) reference points in [-1,1] (z,y,x); ``value_maps`` list of
        ``(B,dim,d,h,w)`` per scale. Returns ``(B,N,dim)``."""
        B, N, _ = q.shape
        h, hd, k = self.h, self.hd, self.k
        off = self.sampling_offsets(q).view(B, N, h, self.L, k, 3)
        aw = self.attn_weights(q).view(B, N, h, self.L * k)
        aw = torch.softmax(aw, dim=-1).view(B, N, h, self.L, k)                   # over all (scale,point)
        acc = q.new_zeros(B * h, hd, N)                                          # (B·h, hd, N) accumulated over levels
        for l, vm in enumerate(value_maps):
            d, hh_, w = vm.shape[2:]
            val = self.value_proj(vm.flatten(2).transpose(1, 2))                  # (B, M, dim)
            val = val.view(B, d * hh_ * w, h, hd).permute(0, 2, 3, 1).reshape(B * h, hd, d, hh_, w)  # heads→batch
            loc = ref.view(B, N, 1, 1, 3) + self.offset_scale * off[:, :, :, l]   # (B,N,h,k,3)
            grid = loc.permute(0, 2, 1, 3, 4).reshape(B * h, N, k, 1, 3).flip(-1)  # (B·h,N,k,1,3) grid wants (x,y,z)
            samp = F.grid_sample(val, grid, mode="bilinear", padding_mode="border",
                                 align_corners=True).squeeze(-1)                  # (B·h, hd, N, k)
            w_l = aw[:, :, :, l].permute(0, 2, 1, 3).reshape(B * h, 1, N, k)       # (B·h,1,N,k)
            acc = acc + (samp * w_l).sum(dim=-1)                                  # (B·h, hd, N)
        out = acc.view(B, h * hd, N).transpose(1, 2)                             # (B, N, dim)
        return self.out_proj(out)


class NeckLayer(nn.Module):
    """One deformable-attention encoder layer: pre-norm MS-deformable self-attention + pre-norm FFN, both residual.
    The value maps are rebuilt from the (normed) query sequence each layer, so refinement compounds across layers."""

    def __init__(self, dim: int, n_scales: int, n_heads: int = 4, n_points: int = 4, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.dim = int(dim)
        self.attn = MSDeformAttn3D(dim, n_scales, n_heads, n_points)
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)    # identity start (residual)

    def forward(self, seq: torch.Tensor, ref: torch.Tensor, shapes: Sequence[Sequence[int]]) -> torch.Tensor:
        x = self.n1(seq)
        vmaps = _split_to_grids(x, shapes, self.dim)
        seq = seq + self.attn(x, ref, vmaps)
        seq = seq + self.mlp(self.n2(seq))
        return seq


def _split_to_grids(seq: torch.Tensor, shapes: Sequence[Sequence[int]], dim: int) -> list[torch.Tensor]:
    """Split a flat multi-scale token sequence ``(B,N,dim)`` back into per-scale grids ``(B,dim,d,h,w)``."""
    B = seq.shape[0]
    out, off = [], 0
    for (d, h, w) in shapes:
        n = d * h * w
        out.append(seq[:, off:off + n].transpose(1, 2).reshape(B, dim, d, h, w))
        off += n
    return out


class DeformableNeck(nn.Module):
    """Cross-scale deformable-attention neck over {1/8 tokens, 1/4 s2}. Projects both scales to a common width,
    runs ``n_layers`` deformable self-attention layers, and returns the refined **1/8** grid projected back to
    ``token_dim`` for the upsample tail. Identity at init.

    Args:
        token_dim: 1/8 token width (288 for Dinovol).
        s2_dim: 1/4 stem-skip width (128).
        dim: internal neck width (default = ``token_dim``).
    """

    def __init__(self, token_dim: int = 288, s2_dim: int = 128, *, dim: int | None = None,
                 n_layers: int = 6, n_heads: int = 4, n_points: int = 4, mlp_ratio: int = 4) -> None:
        super().__init__()
        dim = int(dim or token_dim)
        self.dim = dim
        self.in8 = nn.Conv3d(token_dim, dim, kernel_size=1)
        self.in4 = nn.Conv3d(s2_dim, dim, kernel_size=1)
        self.out8 = nn.Conv3d(dim, token_dim, kernel_size=1)
        nn.init.zeros_(self.out8.weight); nn.init.zeros_(self.out8.bias)          # identity start: out8 adds 0
        self.layers = nn.ModuleList(
            [NeckLayer(dim, n_scales=2, n_heads=n_heads, n_points=n_points, mlp_ratio=mlp_ratio)
             for _ in range(int(n_layers))]
        )

    def forward(self, tokens_grid: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        """``tokens_grid`` (B,token_dim,16³); ``s2`` (B,s2_dim,32³). Returns the refined 1/8 grid (B,token_dim,16³)."""
        f8 = self.in8(tokens_grid)
        f4 = self.in4(s2)
        shapes = [tuple(f8.shape[2:]), tuple(f4.shape[2:])]
        dev, dt = f8.device, f8.dtype
        ref8 = _coord_grid(shapes[0], dev, dt)
        ref4 = _coord_grid(shapes[1], dev, dt)
        ref = torch.cat([ref8, ref4], dim=0).unsqueeze(0).expand(f8.shape[0], -1, -1)   # (B,N,3)
        seq = torch.cat([f8.flatten(2).transpose(1, 2), f4.flatten(2).transpose(1, 2)], dim=1)  # (B,N,dim)
        for layer in self.layers:
            seq = layer(seq, ref, shapes)
        n8 = shapes[0][0] * shapes[0][1] * shapes[0][2]
        f8_out = seq[:, :n8].transpose(1, 2).reshape(f8.shape[0], self.dim, *shapes[0])
        return tokens_grid + self.out8(f8_out)                                    # residual onto the raw tokens

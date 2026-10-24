"""Small, architecture-agnostic decoder primitives shared by the decoupled pixel encoder + mask decoder.

Extracted verbatim (math-identical) from the original design-A/B/C modules so the decoupled path is self-contained
after those experimental modules were removed: the dot-product mask readouts, a plain MLP, the Mask2Former decoder
layer, and the parameter-free 3-D sine positional encoding (with its coordinate grid).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _coord_grid(shape: Sequence[int], device, dtype) -> torch.Tensor:
    """``(G, 3)`` cell-centre coordinates of a ``shape`` grid in [-1, 1] (z, y, x), raster order (x fastest)."""
    dz, dy, dx = shape
    zs = (torch.arange(dz, device=device, dtype=dtype) + 0.5) / dz * 2 - 1
    ys = (torch.arange(dy, device=device, dtype=dtype) + 0.5) / dy * 2 - 1
    xs = (torch.arange(dx, device=device, dtype=dtype) + 0.5) / dx * 2 - 1
    gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
    return torch.stack([gz.reshape(-1), gy.reshape(-1), gx.reshape(-1)], dim=-1)  # (G, 3)


def mask_logits_from_embed(mask_embed: torch.Tensor, mask_features: torch.Tensor) -> torch.Tensor:
    """Full-resolution instance mask logits. ``mask_embed`` (B,Q,C) · ``mask_features`` (B,C,D,H,W) → (B,Q,D,H,W)."""
    return torch.einsum("bqc,bcdhw->bqdhw", mask_embed, mask_features)


def mask_logits_at_points(mask_embed: torch.Tensor, sampled_features: torch.Tensor) -> torch.Tensor:
    """Point-sampled mask logits. ``mask_embed`` (B,Q,C) · ``sampled_features`` (B,C,P) → (B,Q,P)."""
    return torch.einsum("bqc,bcp->bqp", mask_embed, sampled_features)


def _mlp(dim: int, hidden: int, out: int, n: int = 3) -> nn.Sequential:
    layers, d = [], dim
    for i in range(n - 1):
        layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]; d = hidden
    layers += [nn.Linear(d, out)]
    return nn.Sequential(*layers)


class QKNormAttention(nn.Module):
    """Multi-head attention with **per-head QK-LayerNorm** — query and key are LayerNorm'd over the head dimension
    *before* the scaled dot — computed through the fused ``F.scaled_dot_product_attention`` kernel.

    Drop-in replacement for ``nn.MultiheadAttention(dim, n_heads, batch_first=True)`` as used in the mask decoder,
    with one structural change: the normalisation of q and k (ViT-22B QK-LayerNorm [Dehghani+ 2023]; Wortsman+ 2024).
    PyTorch's ``nn.MultiheadAttention`` cannot host this — with ``need_weights=False`` it hands q,k,v straight to the
    compiled ``scaled_dot_product_attention`` kernel, exposing no hook between the in-projection and the dot. Here we
    normalise q,k ourselves (cheap elementwise ops) and keep SDPA for the heavy O(N²) dot·softmax·value. Because SDPA
    keeps its internal ``1/√head_dim`` scaling while q,k are unit-RMS, each logit is O(1) *regardless of the magnitude
    of the projected keys/queries* — the structural fix for the mask-decoder attention-logit explosion (an unbounded
    key norm, ‖k‖→3025, drove the cross-attention logits to ~20000 and collapsed training).

    ``attn_mask`` follows the decoder's convention used by :class:`MaskDecoderLayer`: a **bool** tensor of shape
    ``(B·n_heads, Lq, Lk)`` with ``True`` = MASKED (not allowed to attend), or ``None`` for full attention. A fully
    masked query row is reset to attend everywhere (SDPA returns ``NaN`` for an all-``-inf`` row), matching the
    Mask2Former masked-attention guard.
    """

    def __init__(self, dim: int, n_heads: int, bias: bool = True) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} not divisible by n_heads {n_heads}")
        self.dim = int(dim); self.n_heads = int(n_heads); self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.q_norm = nn.LayerNorm(self.head_dim)          # QK-LayerNorm (per head, over head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:     # (B,L,dim) → (B,H,L,head_dim)
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, query, key, value, attn_mask=None):
        B, Lq, _ = query.shape
        q = self.q_norm(self._heads(self.q_proj(query)))   # (B,H,Lq,hd)  ← normalised before the dot
        k = self.k_norm(self._heads(self.k_proj(key)))     # (B,H,Lk,hd)  ← normalised before the dot
        v = self._heads(self.v_proj(value))                # (B,H,Lk,hd)
        bias = None
        if attn_mask is not None:                          # (B·H,Lq,Lk) bool True=masked → (B,H,Lq,Lk) float bias
            m = attn_mask.view(B, self.n_heads, Lq, attn_mask.shape[-1])
            m = m & ~m.all(dim=-1, keepdim=True)           # fully-masked row → attend everywhere (no NaN)
            bias = torch.zeros_like(m, dtype=q.dtype).masked_fill_(m, float("-inf"))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)   # fused kernel; internal 1/√head_dim scale
        out = out.transpose(1, 2).reshape(B, Lq, self.dim)
        return self.out_proj(out)


class MaskDecoderLayer(nn.Module):
    """Mask2Former decoder layer: masked cross-attn (to one memory level) → self-attn → FFN (pre-norm, residual).
    Both attentions use :class:`QKNormAttention` (QK-LayerNorm) for training stability — see its docstring."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.n_heads = int(n_heads)
        self.cross = QKNormAttention(dim, n_heads)
        self.self_attn = QKNormAttention(dim, n_heads)
        self.n1 = nn.LayerNorm(dim); self.n2 = nn.LayerNorm(dim); self.n3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))

    def forward(self, q, qpos, mem_k, mem_v, cross_mask, self_mask):
        qn = self.n1(q)
        q = q + self.cross(qn + qpos, mem_k, mem_v, attn_mask=cross_mask)
        qn = self.n2(q)
        q = q + self.self_attn(qn + qpos, qn + qpos, qn, attn_mask=self_mask)
        q = q + self.mlp(self.n3(q))
        return q


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

"""Pillar 3 — the Mask2Former-style instance (sheet-ID) head.

Set-prediction instance masks for sheets. Design ``docs/hercuvit.md`` §7:

- **Two-stage query selection.** A linear objectness head scores all 16³ tokens; the top-K become queries,
  each carrying an **anchor** (its token-cell centre in [-1,1], refined per layer) and a **content** vector
  (token feature projected 288→``dim``).
- **Query decoder.** ``n_layers`` layers; each = self-attention among queries + **3D multi-scale deformable
  cross-attention** (reusing :class:`hercunet.tracer.decoder.DeformableCrossAttn`) around the anchor over the
  1/8 tokens and the 1/4 decoder features + an FFN; an MLP refines the anchor.
- **Outputs per query:** a sheet / no-object class logit, a refined anchor, and a ``mask_dim`` mask embedding.
  The instance mask is ``sigmoid(mask_embed · mask_features)`` against the shared full-resolution mask features
  (pillar 2). Masks are computed on demand — full at inference, at importance-sampled points in training — so
  the head never materialises 64×128³ masks during training (keeps Mask2Former point-sampling affordable).

Identity is positional, not appearance-based: anchoring each query to an encoder-chosen location makes the
near-identical sheets separable and avoids the query-swapping instability of free DETR queries (design §7).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ...tracer.decoder import DeformableCrossAttn


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


class QuerySelector(nn.Module):
    """Objectness over the token grid → top-K anchored queries (two-stage selection)."""

    def __init__(self, token_dim: int, dim: int, n_queries: int) -> None:
        super().__init__()
        self.n_queries = int(n_queries)
        self.obj = nn.Linear(token_dim, 1)
        self.content = nn.Linear(token_dim, dim)

    def forward(self, tokens_seq: torch.Tensor, coords: torch.Tensor):
        """``tokens_seq`` (B,G,token_dim); ``coords`` (G,3). Returns (content (B,Q,dim), anchors (B,Q,3), obj (B,G), idx (B,Q))."""
        B, G, _ = tokens_seq.shape
        obj_logits = self.obj(tokens_seq).squeeze(-1)                      # (B, G)
        k = min(self.n_queries, G)
        idx = obj_logits.topk(k, dim=1).indices                           # (B, Q)
        gather = idx.unsqueeze(-1).expand(-1, -1, tokens_seq.shape[-1])
        content = self.content(torch.gather(tokens_seq, 1, gather))       # (B, Q, dim)
        anchors = coords.unsqueeze(0).expand(B, -1, -1).gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))  # (B,Q,3)
        return content, anchors, obj_logits, idx


class QueryDecoderLayer(nn.Module):
    """Self-attn among queries + multi-scale deformable cross-attn around anchors + FFN + anchor refinement."""

    def __init__(self, dim: int, n_heads: int, n_points: int, n_scales: int, mlp_ratio: int = 4,
                 anchor_step: float = 0.25) -> None:
        super().__init__()
        self.anchor_step = float(anchor_step)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.cross = nn.ModuleList([DeformableCrossAttn(dim, n_heads, n_points) for _ in range(n_scales)])
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.n3 = nn.LayerNorm(dim)
        self.anchor_mlp = nn.Linear(dim, 3)
        nn.init.zeros_(self.anchor_mlp.weight)
        nn.init.zeros_(self.anchor_mlp.bias)

    def forward(self, q: torch.Tensor, anchors: torch.Tensor, feats: Sequence[torch.Tensor]):
        s = self.n1(q)
        q = q + self.self_attn(s, s, s, need_weights=False)[0]
        qn = self.n2(q)
        cross = sum(c(qn, anchors, f) for c, f in zip(self.cross, feats))
        q = q + cross
        q = q + self.mlp(self.n3(q))
        anchors = (anchors + self.anchor_step * torch.tanh(self.anchor_mlp(q))).clamp(-1.0, 1.0)
        return q, anchors


class InstanceHead(nn.Module):
    """Two-stage queries + query decoder → per-query class logits, mask embeddings and refined anchors.

    forward returns a dict:
      ``objectness`` (B,G) token-cell objectness logits (for the objectness loss);
      ``query_idx`` (B,Q) selected token indices;
      ``layers`` list (len ``n_layers``) of ``{class_logits (B,Q,2), mask_embed (B,Q,mask_dim), anchors (B,Q,3)}``
      — the last is the prediction, the earlier ones feed the auxiliary (per-layer) losses.

    Masks are *not* materialised here; use :func:`mask_logits_from_embed` (inference) or
    :func:`mask_logits_at_points` (training) with ``mask_features`` from the pixel decoder.
    """

    def __init__(
        self,
        *,
        token_dim: int = 288,
        feat_14_dim: int = 128,
        dim: int = 64,
        n_queries: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        n_points: int = 8,
        mask_dim: int = 32,
    ) -> None:
        super().__init__()
        self.selector = QuerySelector(token_dim, dim, n_queries)
        self.proj8 = nn.Conv3d(token_dim, dim, kernel_size=1)   # 1/8 token grid -> query width
        self.proj4 = nn.Conv3d(feat_14_dim, dim, kernel_size=1)  # 1/4 decoder features -> query width
        self.layers = nn.ModuleList(
            [QueryDecoderLayer(dim, n_heads, n_points, n_scales=2) for _ in range(int(n_layers))]
        )
        self.class_head = nn.Linear(dim, 2)        # sheet vs no-object
        self.mask_embed = nn.Linear(dim, int(mask_dim))

    def forward(self, tokens_grid: torch.Tensor, feat_14: torch.Tensor) -> dict:
        """``tokens_grid`` (B, token_dim, 16,16,16); ``feat_14`` (B, feat_14_dim, 32,32,32)."""
        B, C, dz, dy, dx = tokens_grid.shape
        tokens_seq = tokens_grid.flatten(2).transpose(1, 2)                 # (B, G, token_dim), raster (z,y,x)
        coords = _coord_grid((dz, dy, dx), tokens_grid.device, tokens_grid.dtype)
        content, anchors, objectness, idx = self.selector(tokens_seq, coords)

        feats = [self.proj8(tokens_grid), self.proj4(feat_14)]              # multi-scale deformable keys/values
        q = content
        layers_out = []
        for layer in self.layers:
            q, anchors = layer(q, anchors, feats)
            layers_out.append(
                {
                    "class_logits": self.class_head(q),
                    "mask_embed": self.mask_embed(q),
                    "anchors": anchors,
                }
            )
        return {"objectness": objectness, "query_idx": idx, "layers": layers_out}

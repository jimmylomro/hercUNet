"""The cross-window stitching decoder (docs/hercuvit-stitching.md §6).

A decoder-only transformer: fragment **queries** self-attend, cross-attend to a fixed **spatial memory** ``Z`` of
pooled window tokens, and pass through a feed-forward block, ``L`` times. Position + source-window slot are added
into the K **and** V of every cross-attention (the "where" a fragment sits must flow into the values — §6.1), and a
shared cosine affinity head turns each layer's refined queries into a pairwise same-object logit matrix for deep
supervision. The spatial memory is embedded once and reused by every layer (no spatial encoder).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import StitchConfig


def fourier_features(P: torch.Tensor, bands: int) -> torch.Tensor:
    """``P: (..., 3)`` in ``[-1, 1]`` → ``(..., 6F)`` = concat over ``k = 0 … F-1`` of ``sin(2^k π P), cos(2^k π P)``
    (docs §6.1). 3 axes × 2 (sin/cos) × ``F`` bands = ``6F``."""
    freqs = (2.0 ** torch.arange(bands, device=P.device, dtype=P.dtype)) * math.pi   # (F,)
    a = P[..., None] * freqs                                                          # (..., 3, F)
    emb = torch.cat([torch.sin(a), torch.cos(a)], dim=-1)                             # (..., 3, 2F)
    return emb.reshape(*P.shape[:-1], 3 * 2 * bands)                                  # (..., 6F)


class _DecoderLayer(nn.Module):
    """One pre-norm decoder layer (§6.3): self-attn over fragments → cross-attn to the spatial memory → FFN. Weights
    are NOT shared between layers. Uses full scaled-dot-product attention (no windowing needed at these sizes)."""

    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.ln_sa = nn.LayerNorm(d)
        self.q1, self.k1, self.v1, self.o1 = (nn.Linear(d, d) for _ in range(4))
        self.ln_ca = nn.LayerNorm(d)
        self.q2, self.k2, self.v2, self.o2 = (nn.Linear(d, d) for _ in range(4))
        self.ln_ff = nn.LayerNorm(d)
        self.ff1, self.ff2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.h, self.dh).transpose(1, 2)        # (B, h, N, d_h)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        B, _, N, _ = x.shape
        return x.transpose(1, 2).reshape(B, N, self.h * self.dh)    # (B, N, d)

    def forward(self, Q: torch.Tensor, Z: torch.Tensor, keep_q: torch.Tensor, keep_s: torch.Tensor) -> torch.Tensor:
        # self-attention over fragments (padded queries masked out as keys)
        A = self.ln_sa(Q)
        a = F.scaled_dot_product_attention(self._heads(self.q1(A)), self._heads(self.k1(A)),
                                           self._heads(self.v1(A)), attn_mask=keep_q)
        Q = Q + self.o1(self._merge(a))
        # cross-attention, queries → spatial tokens (invalid spatial tokens masked out as keys)
        A = self.ln_ca(Q)
        a = F.scaled_dot_product_attention(self._heads(self.q2(A)), self._heads(self.k2(Z)),
                                           self._heads(self.v2(Z)), attn_mask=keep_s)
        Q = Q + self.o2(self._merge(a))
        # feed-forward
        return Q + self.ff2(F.gelu(self.ff1(self.ln_ff(Q))))


class StitchDecoder(nn.Module):
    """The stitching network (§6). ``forward`` returns a LIST of ``L`` logit matrices ``Λ^ℓ`` (B×M×M), one per
    decoder layer, for the deep-supervised loss; inference reads the last one."""

    def __init__(self, cfg: StitchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, D = cfg.d_model, cfg.pca_dim
        self.slot = nn.Embedding(8, d)                             # E_slot — shared by spatial tokens AND queries (§4)
        self.lin_s = nn.Linear(D, d)                              # spatial token embedding
        self.lin_p = nn.Linear(6 * cfg.fourier_bands, d)         # position embedding
        self.ln_z = nn.LayerNorm(d)
        self.lin_q = nn.Linear(D, d)                             # query embedding
        self.layers = nn.ModuleList([_DecoderLayer(d, cfg.heads) for _ in range(cfg.layers)])
        self.ln_head = nn.LayerNorm(d)                           # affinity head, SHARED by all layers (§6.4)
        self.head = nn.Linear(d, d)
        self.log_s = nn.Parameter(torch.tensor(math.log(10.0)))  # s = exp(t), init 10, clamped ≤ 100
        self.b = nn.Parameter(torch.tensor(-5.0))                # decision bias, init -5

    def spatial_memory(self, Xs, Ps, slot_s):
        """``Z = LayerNorm(Lin(Xs) + Lin(Fourier(Ps)) + E_slot[slot_s])`` (§6.1) — computed once, reused every layer."""
        pos = fourier_features(Ps, self.cfg.fourier_bands)
        return self.ln_z(self.lin_s(Xs) + self.lin_p(pos) + self.slot(slot_s))

    def forward(self, Xs, Ps, slot_s, valid_s, Xq, slot_q, valid_q):
        """Inputs (B = corners in the batch, S = spatial tokens, M = query fragments, D = ``cfg.pca_dim``):
          ``Xs (B,S,D)``, ``Ps (B,S,3)`` in [-1,1], ``slot_s (B,S)`` 0..7, ``valid_s (B,S)`` bool;
          ``Xq (B,M,D)``, ``slot_q (B,M)`` 0..7, ``valid_q (B,M)`` bool.
        Returns a list of ``L`` dicts ``{"logit": Λ^ℓ (B,M,M), "emb": Ô^ℓ (B,M,d)}`` — the cosine-affinity logits
        (for the focal BCE) and the L2-normalised query embeddings (for the SupCon contrastive term), one per layer."""
        Z = self.spatial_memory(Xs, Ps, slot_s)                                      # (B,S,d)
        Q = self.lin_q(Xq) + self.slot(slot_q)                                       # (B,M,d) = Q⁰
        # boolean key-masks (True = attend): broadcast (B,1,1,N) over (B,h,M,N) scores — mask padded/invalid KEYS.
        keep_q = valid_q[:, None, None, :]
        keep_s = valid_s[:, None, None, :]
        s = self.log_s.exp().clamp(max=100.0)
        out = []
        for layer in self.layers:
            Q = layer(Q, Z, keep_q, keep_s)
            O = F.normalize(self.head(self.ln_head(Q)), dim=-1)                       # (B,M,d) L2-normalised rows
            C = O @ O.transpose(-1, -2)                                               # (B,M,M) cosine in [-1,1]
            out.append({"logit": s * C + self.b, "emb": O})                          # Λ = s·C + b, plus embeddings
        return out

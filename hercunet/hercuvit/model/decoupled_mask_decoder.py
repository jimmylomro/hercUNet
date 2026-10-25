"""Decoupled mask decoder (architecture v2) — vanilla Mask2Former, built PRECISE to ``hercuvit-mask-architecture.md``
§10. NO null query, NO class head, NO cosine/temperature — the decoupled surface head (pixel encoder) is the
objectness, so the decoder only answers "which sheet owns this voxel" via a per-voxel softmax over the Q queries.

Differences from the design-C :class:`MaskDecoder` (which it deliberately does NOT reuse — see memory
``build-precise-to-spec-graph``):

  * **Learned queries + denoising (DN)** — persistent ``nn.Embedding`` content/pos queries (DETR/M2F standard),
    plus a train-only fixed-assigned DN group seeded from the GT sheet masks. NO surface-objectness FPS seeding.
  * **Plain dot readout** — ``mask_logits = einsum(mask_embed[Q,C], mask_features=U1[C,128³])`` directly at 1/1,
    with NO ``F.normalize`` and NO ``logit_scale`` (the decoupling removes the null-magnitude cheat those fixed).
  * **No null, no class head** — each layer emits only ``mask_embed``; the match is Hungarian on ownership-CE cost
    and the supervision is a no-null softmax-over-Q ownership CE on foreground voxels (see
    ``train.losses.instance_loss_decoupled``). Unmatched queries are suppressed for free by the softmax.

The decoder ATTENDS only to the ≤1/8 ``memory`` pyramid (round-robin 1/32·1/16·1/8). The dense 1/1 ``mask_logits``
are materialised only (detached) to build the next layer's attention mask (``sigmoid<0.5`` → max-pool → mem level).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .decoder_common import mask_logits_from_embed, MaskDecoderLayer, _mlp, _sine_pe_3d


class DecoupledMaskDecoder(nn.Module):
    """Learned-query masked-attention decoder (no null, no class). ``n_layers`` must be a multiple of 3 (round-robin
    over the 3 memory levels 1/32, 1/16, 1/8)."""

    def __init__(self, *, dim: int = 128, mask_dim: int = 32, n_heads: int = 8, n_queries: int = 64,
                 n_layers: int = 6, dn_groups: int = 2) -> None:
        super().__init__()
        assert n_layers % 3 == 0, "n_layers must be a multiple of 3 (round-robin over 1/32,1/16,1/8)"
        self.dim = int(dim); self.mask_dim = int(mask_dim); self.n_heads = int(n_heads)
        self.n_queries = int(n_queries); self.n_layers = int(n_layers); self.dn_groups = int(dn_groups)

        self.query_feat = nn.Embedding(self.n_queries, dim)             # learned content queries (DETR/M2F)
        self.query_pos = nn.Embedding(self.n_queries, dim)             # learned positional queries
        self.mem_level_embed = nn.Parameter(torch.zeros(3, dim)); nn.init.normal_(self.mem_level_embed, std=0.02)
        self.dn_feat = nn.Parameter(torch.zeros(dim)); nn.init.normal_(self.dn_feat, std=0.02)   # DN content seed
        self.dn_pos = nn.Parameter(torch.zeros(dim)); nn.init.normal_(self.dn_pos, std=0.02)     # DN positional seed

        self.layers = nn.ModuleList([MaskDecoderLayer(dim, n_heads) for _ in range(n_layers)])
        self.mask_embed = _mlp(dim, dim, mask_dim, n=3)                 # dim→dim→dim→mask_dim
        self.me_norm = nn.LayerNorm(mask_dim)                           # bound the query operand of the readout dot:
        # the mask_embed MLP output was unbounded (only its input q is LayerNorm'd), so under the softmax-CE readout
        # pressure its norm ran away (→1e5) alongside U1. With U1 LayerNorm'd in the pixel encoder and mask_embed
        # LayerNorm'd here, mask_logits = dot(me, U1) land in ~±mask_dim before the softmax/sigmoid that already follow.
        self.decoder_norm = nn.LayerNorm(dim)

    # ---- helpers -------------------------------------------------------------------------------------------
    def _predict(self, q, mask_features):
        """Mask embedding (grad) + dense 1/1 mask LOGITS (DETACHED, kept in the autocast dtype — attention-mask use
        only; the loss recomputes the masks from ``mask_embed``). Plain dot: NO L2-norm, NO temperature."""
        me = self.me_norm(self.mask_embed(self.decoder_norm(q)))       # (B,Q,mask_dim) — LayerNorm'd (bounds readout)
        mlog = mask_logits_from_embed(me.detach(), mask_features.detach())   # (B,Q,D,H,W) plain dot (bf16 under autocast)
        return me, mlog

    def _attn_mask(self, mlog, side_mem):
        """M2F attention mask for the next layer. Max-pool the 1/1 LOGITS to ``side_mem`` and threshold at 0 —
        exact: ``sigmoid(x) > 0.5 ⟺ x > 0`` and max-pool commutes with the monotonic sigmoid, so this is the
        ``sigmoid<0.5`` feedback WITHOUT materialising the sigmoid. Empty rows attend everywhere. Returns
        (B·heads, Q, N_mem) bool (True = masked)."""
        B, Q = mlog.shape[:2]
        k = mlog.shape[-1] // side_mem
        pooled = F.max_pool3d(mlog, kernel_size=k, stride=k)          # (B,Q,side³) max logit per block
        keep = (pooled.flatten(2) > 0.0)                             # (B,Q,N) True = attend (sigmoid>0.5)
        keep[keep.sum(-1) == 0] = True                              # empty → attend everywhere
        amask = ~keep
        return amask.unsqueeze(1).expand(B, self.n_heads, Q, amask.shape[-1]).reshape(B * self.n_heads, Q, -1)

    def _build_dn(self, mask_features, dn_owner):
        """Train-only denoise queries from the thin GT owner @1/1. Each GT sheet → ``dn_groups`` fixed queries whose
        INITIAL mask is the dilated GT sheet mask (the denoising signal); content/pos are the learned DN seeds.
        Returns ``(content, qpos, init_mlog (B,n_dn,D,H,W), valid (B,n_dn), sheet (B,n_dn))`` or ``None``."""
        B, _C, D, H, W = mask_features.shape
        dev = mask_features.device
        per = [torch.unique(dn_owner[b][dn_owner[b] > 0]) for b in range(B)]
        max_s = max((int(p.numel()) for p in per), default=0)
        if max_s == 0:
            return None
        n_dn = max_s * self.dn_groups
        content = self.dn_feat.view(1, 1, -1).expand(B, n_dn, self.dim).contiguous()
        qpos = self.dn_pos.view(1, 1, -1).expand(B, n_dn, self.dim).contiguous()
        init = mask_features.new_zeros(B, n_dn, D, H, W)
        valid = torch.zeros(B, n_dn, dtype=torch.bool, device=dev)
        sheet = torch.zeros(B, n_dn, dtype=torch.long, device=dev)
        for b in range(B):
            for si, s in enumerate(per[b]):
                gt = (dn_owner[b] == s).float().view(1, 1, D, H, W)
                dil = (F.max_pool3d(gt, kernel_size=3, stride=1, padding=1) > 0).float().view(D, H, W)
                for g in range(self.dn_groups):
                    slot = si * self.dn_groups + g
                    init[b, slot] = dil * 6.0 - 3.0                   # logit-ish: inside ≈ +3, outside ≈ -3
                    valid[b, slot] = True; sheet[b, slot] = int(s)
        return content, qpos, init, valid, sheet

    # ---- forward -------------------------------------------------------------------------------------------
    def forward(self, memory: Sequence[torch.Tensor], mask_features, dn_owner=None) -> dict:
        """``memory`` = [1/32, 1/16, 1/8] (B,dim,·); ``mask_features`` (B,mask_dim,128³) = U1; ``dn_owner`` (B,128³)
        thin GT owner (train only). Emits ``{"layers": [{"mask_embed"}], "n_match", "dn_valid", "dn_sheet"}``."""
        B = mask_features.shape[0]
        content = self.query_feat.weight.unsqueeze(0).expand(B, -1, -1)           # (B,Q,dim) learned content
        qpos = self.query_pos.weight.unsqueeze(0).expand(B, -1, -1)              # (B,Q,dim) learned positional
        n_match = self.n_queries

        dn = self._build_dn(mask_features, dn_owner) if (dn_owner is not None and self.training) else None
        self_mask = None
        if dn is not None:
            dn_content, dn_qpos, dn_init, dn_valid, dn_sheet = dn
            content = torch.cat([content, dn_content], dim=1); qpos = torch.cat([qpos, dn_qpos], dim=1)
            L = content.shape[1]
            self_mask = torch.zeros(L, L, dtype=torch.bool, device=content.device)
            self_mask[:n_match, n_match:] = True                                 # matching can't see DN
            self_mask[n_match:, :n_match] = True                                 # DN can't see matching
            self_mask = self_mask.unsqueeze(0).expand(B * self.n_heads, L, L)

        # memory keys/values (+ sine PE + level embed on keys), coarse→fine [1/32,1/16,1/8]
        mem_kv = []
        for li, m in enumerate(memory):
            tok = m.flatten(2).transpose(1, 2)                                   # (B,N,dim)
            pe = _sine_pe_3d(tuple(m.shape[2:]), self.dim, m.device, m.dtype).unsqueeze(0)
            mem_kv.append((tok + pe + self.mem_level_embed[li].view(1, 1, -1), tok, m.shape[-1]))

        q = content
        me0, mlog0 = self._predict(q, mask_features)                             # pre-layer-0 prediction → first attn mask
        if dn is not None:
            mlog0 = mlog0.clone()
            mlog0[:, n_match:] = dn_init                                          # DN init mask overrides at init
        layers_out = [{"mask_embed": me0}]
        mlog = mlog0
        for i, layer in enumerate(self.layers):
            lv = i % 3                                                            # round-robin 1/32,1/16,1/8
            mem_k, mem_v, side = mem_kv[lv]
            cmask = self._attn_mask(mlog, side)
            q = layer(q, qpos, mem_k, mem_v, cmask, self_mask)
            me, mlog = self._predict(q, mask_features)
            layers_out.append({"mask_embed": me})
        out = {"layers": layers_out, "n_match": n_match, "objectness": None, "query_idx": None}
        if dn is not None:
            out["dn_valid"], out["dn_sheet"] = dn_valid, dn_sheet
        return out

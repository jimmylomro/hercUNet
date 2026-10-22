"""Design-C mask decoder — seeded, masked-attention transformer decoder (``docs/hercuvit-mask-architecture.md``
§5-§8). Round-robin **coarse→fine** memory (1/32, 1/16, 1/8), masked cross-attention with max-pooled mask
feedback, surface-seeded matching queries + training-only denoising queries.

Emits the SAME per-layer dict as the design-A head (``{class_logits, mask_embed}`` + ``n_match``/``query_idx``/
``dn_valid``/``dn_sheet``), so ``train.losses.instance_loss`` (softmax ownership-CE) is reused unchanged — just at
1/4 resolution (``mask_features`` and ``owner`` are both 1/4)."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .instance_head import InstanceHead, _coord_grid, mask_logits_from_embed
from .mask_pixel_encoder import _sine_pe_3d


def _mlp(dim: int, hidden: int, out: int, n: int = 3) -> nn.Sequential:
    layers, d = [], dim
    for i in range(n - 1):
        layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]; d = hidden
    layers += [nn.Linear(d, out)]
    return nn.Sequential(*layers)


class MaskDecoderLayer(nn.Module):
    """Mask2Former decoder layer: masked cross-attn (to one memory level) → self-attn → FFN (pre-norm, residual)."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.n_heads = int(n_heads)
        self.cross = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.n1 = nn.LayerNorm(dim); self.n2 = nn.LayerNorm(dim); self.n3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))

    def forward(self, q, qpos, mem_k, mem_v, cross_mask, self_mask):
        qn = self.n1(q)
        q = q + self.cross(qn + qpos, mem_k, mem_v, attn_mask=cross_mask, need_weights=False)[0]
        qn = self.n2(q)
        q = q + self.self_attn(qn + qpos, qn + qpos, qn, attn_mask=self_mask, need_weights=False)[0]
        q = q + self.mlp(self.n3(q))
        return q


class MaskDecoder(nn.Module):
    """Seeded masked-attention decoder. ``n_layers`` must be a multiple of 3 (round-robin over 3 memory levels)."""

    def __init__(self, *, dim: int = 128, mask_dim: int = 16, n_heads: int = 8, n_queries: int = 64,
                 n_layers: int = 9, add_null: bool = True, dn_groups: int = 2, seed_thresh: float = 0.3) -> None:
        super().__init__()
        assert n_layers % 3 == 0, "n_layers must be a multiple of 3 (round-robin over 1/32,1/16,1/8)"
        self.dim = int(dim); self.mask_dim = int(mask_dim); self.n_heads = int(n_heads)
        self.n_queries = int(n_queries); self.n_layers = int(n_layers)
        self.add_null = bool(add_null); self.dn_groups = int(dn_groups); self.seed_thresh = float(seed_thresh)
        self.content_proj = nn.Linear(dim, dim)                          # seed feature → content query
        self.pos_proj = nn.Linear(3, dim)                               # seed coords → positional query
        self.mem_level_embed = nn.Parameter(torch.zeros(3, dim)); nn.init.normal_(self.mem_level_embed, std=0.02)
        if self.add_null:
            self.null_content = nn.Parameter(torch.zeros(dim)); self.null_pos = nn.Parameter(torch.zeros(dim))
        self.layers = nn.ModuleList([MaskDecoderLayer(dim, n_heads) for _ in range(n_layers)])
        self.class_head = nn.Linear(dim, 2)
        self.mask_embed = _mlp(dim, dim, mask_dim, n=3)                  # 128→128→128→16
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(10.0)))  # cosine-classifier temperature (logit = exp(s)·cos)
        self.decoder_norm = nn.LayerNorm(dim)

    # ---- helpers -------------------------------------------------------------------------------------------
    def _predict(self, q, mask_features):
        """Class logits + mask embedding + dense 1/4 mask logits for one set of queries."""
        qn = self.decoder_norm(q)
        cls = self.class_head(qn)
        me = self.mask_embed(qn)                                        # (B,Q,mask_dim)
        # cosine-classifier readout: unit-normalize the query embedding (mask_features are already unit), so a logit
        # is a pure cosine NO query — including the null — can win by magnitude; a learnable temperature exp(s)
        # (init 10, capped 100) restores the softmax sharpness that bounded cosines would otherwise lose.
        me = F.normalize(me, dim=-1) * self.logit_scale.exp().clamp(max=100.0)
        mlog = mask_logits_from_embed(me.float(), mask_features.float())  # (B,Q,32³) = temp·cos
        return cls, me, mlog

    def _attn_mask(self, mlog, side_mem):
        """Mask2Former attention mask for the next layer: sigmoid(1/4 logits) → max-pool to ``side_mem`` → <0.5 is
        masked out; all-empty rows → attend everywhere. Returns (B·heads, Q, N_mem) bool (True = masked)."""
        B, Q = mlog.shape[:2]
        k = mlog.shape[-1] // side_mem
        prob = torch.sigmoid(mlog.detach())                            # (B,Q,32³)
        pooled = F.max_pool3d(prob, kernel_size=k, stride=k)            # (B,Q,side³)
        keep = (pooled.flatten(2) > 0.5)                               # (B,Q,N) True = attend
        keep[keep.sum(-1) == 0] = True                                 # empty → attend everywhere
        amask = ~keep                                                  # True = masked out
        return amask.unsqueeze(1).expand(B, self.n_heads, Q, amask.shape[-1]).reshape(B * self.n_heads, Q, -1)

    def _seed(self, feat4, surf_prob4):
        """Surface-objectness FPS seeding at 1/4: pick ``n_queries`` spatially-spread high-surface voxels. Returns
        (content (B,Q,dim), qpos (B,Q,dim), coords (B,Q,3), idx (B,Q))."""
        B, C = feat4.shape[0], feat4.shape[1]
        side = feat4.shape[-1]
        coords = _coord_grid((side, side, side), feat4.device, feat4.dtype)          # (G,3)
        G = coords.shape[0]
        feat_seq = feat4.flatten(2).transpose(1, 2)                                  # (B,G,C)
        valid = (surf_prob4.flatten(1) > self.seed_thresh)                           # (B,G) high-surface pool
        cg = coords.unsqueeze(0).expand(B, -1, -1)                                   # (B,G,3)
        idx = InstanceHead._fps_indices(cg, self.n_queries, valid=valid)             # spatial FPS over surface voxels
        gi = idx.unsqueeze(-1)
        content = self.content_proj(torch.gather(feat_seq, 1, gi.expand(-1, -1, C)))
        scoord = torch.gather(cg, 1, gi.expand(-1, -1, 3))                           # (B,Q,3)
        qpos = self.pos_proj(scoord)
        return content, qpos, scoord, idx

    def _build_dn(self, feat4, dn_owner):
        """Train-only denoise queries from GT sheets @1/4: content = pooled feat, qpos from centroid, initial 1/4
        mask = dilated GT sheet mask. Returns (content, qpos, init_mlog (B,n_dn,32³), valid (B,n_dn), sheet (B,n_dn))."""
        B, C, S = feat4.shape[0], feat4.shape[1], feat4.shape[-1]
        coords = _coord_grid((S, S, S), feat4.device, feat4.dtype)                   # (G,3)
        feat_seq = feat4.flatten(2).transpose(1, 2)                                  # (B,G,C)
        per = [torch.unique(dn_owner[b][dn_owner[b] > 0]) for b in range(B)]
        max_s = max((int(p.numel()) for p in per), default=0)
        if max_s == 0:
            return None
        n_dn = max_s * self.dn_groups
        content = feat4.new_zeros(B, n_dn, self.dim); qpos = feat4.new_zeros(B, n_dn, self.dim)
        init = feat4.new_zeros(B, n_dn, S * S * S); valid = torch.zeros(B, n_dn, dtype=torch.bool, device=feat4.device)
        sheet = torch.zeros(B, n_dn, dtype=torch.long, device=feat4.device)
        for b in range(B):
            of = dn_owner[b].reshape(-1)
            for si, s in enumerate(per[b]):
                m = (of == s)                                                        # (G,) GT sheet voxels @1/4
                pooled = feat_seq[b][m].mean(0)
                cen = coords[m].mean(0)
                gt = m.float().reshape(1, 1, S, S, S)
                dil = (F.max_pool3d(gt, kernel_size=3, stride=1, padding=1) > 0).reshape(-1).float()  # dilated GT mask
                for g in range(self.dn_groups):
                    slot = si * self.dn_groups + g
                    content[b, slot] = self.content_proj(pooled + 0.3 * pooled.detach().std().clamp_min(1e-4) * torch.randn_like(pooled))
                    qpos[b, slot] = self.pos_proj(cen + 0.1 * torch.randn_like(cen))
                    init[b, slot] = (dil * 6.0 - 3.0)                                 # logit-ish: inside≈+3, outside≈-3
                    valid[b, slot] = True; sheet[b, slot] = int(s)
        return content, qpos, init, valid, sheet

    # ---- forward -------------------------------------------------------------------------------------------
    def forward(self, memory: Sequence[torch.Tensor], feat4, mask_features, surf_prob4, dn_owner=None) -> dict:
        """``memory`` = [1/32,1/16,1/8] (B,dim,·); ``feat4`` (B,dim,32³); ``mask_features`` (B,mask_dim,32³);
        ``surf_prob4`` (B,32³) surface prob for seeding; ``dn_owner`` (B,32³) 1/4 GT (train only)."""
        B = feat4.shape[0]
        content, qpos, _, idx = self._seed(feat4, surf_prob4)                        # matching queries
        if self.add_null:
            content = torch.cat([self.null_content.expand(B, 1, -1), content], dim=1)
            qpos = torch.cat([self.null_pos.expand(B, 1, -1), qpos], dim=1)
            idx = torch.cat([idx.new_full((B, 1), -1), idx], dim=1)
        n_match = content.shape[1]
        dn = self._build_dn(feat4, dn_owner) if (dn_owner is not None and self.training) else None
        self_mask = None
        dn_init = None
        if dn is not None:
            dn_content, dn_qpos, dn_init, dn_valid, dn_sheet = dn
            content = torch.cat([content, dn_content], dim=1); qpos = torch.cat([qpos, dn_qpos], dim=1)
            L = content.shape[1]
            self_mask = torch.zeros(L, L, dtype=torch.bool, device=content.device)
            self_mask[:n_match, n_match:] = True                                     # matching can't see DN
            self_mask[n_match:, :n_match] = True                                     # DN can't see matching
            self_mask = self_mask.unsqueeze(0).expand(B * self.n_heads, L, L)

        # precompute memory keys/values (+ sine PE + level embed on keys)
        mem_kv = []
        for li, m in enumerate(memory):
            tok = m.flatten(2).transpose(1, 2)                                       # (B,N,dim)
            pe = _sine_pe_3d(tuple(m.shape[2:]), self.dim, m.device, m.dtype).unsqueeze(0)
            mem_kv.append((tok + pe + self.mem_level_embed[li].view(1, 1, -1), tok, m.shape[-1]))

        q = content
        # initial prediction (before layer 0) → first attention mask
        cls0, me0, mlog0 = self._predict(q, mask_features)
        if dn is not None:                                                          # DN init mask overrides at init
            mlog0 = mlog0.clone()
            mlog0[:, n_match:] = dn_init.reshape(B, dn_init.shape[1], *mask_features.shape[2:])
        layers_out = [{"class_logits": cls0, "mask_embed": me0}]
        mlog = mlog0
        for i, layer in enumerate(self.layers):
            lv = i % 3                                                              # round-robin 1/32,1/16,1/8
            mem_k, mem_v, side = mem_kv[lv]
            cmask = self._attn_mask(mlog, side)
            q = layer(q, qpos, mem_k, mem_v, cmask, self_mask)
            cls, me, mlog = self._predict(q, mask_features)
            layers_out.append({"class_logits": cls, "mask_embed": me})
        out = {"objectness": None, "query_idx": idx, "layers": layers_out, "n_match": n_match,
               "seed_grid": feat4.shape[-1]}
        if dn is not None:
            out["dn_valid"], out["dn_sheet"] = dn_valid, dn_sheet
        return out

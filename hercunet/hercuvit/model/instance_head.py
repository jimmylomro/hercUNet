"""Pillar 3 — the Mask2Former-style instance (sheet-ID) head.

Set-prediction instance masks for sheets. Design ``docs/hercuvit.md`` §7:

- **Two-stage query selection.** A linear objectness head scores all 16³ tokens; the top-K become queries,
  each carrying an **anchor** (its token-cell centre in [-1,1], refined per layer) and a **content** vector
  (token feature projected 288→``dim``).
- **Query decoder.** ``n_layers`` layers; each = self-attention among queries + **3D multi-scale deformable
  cross-attention** (:class:`DeformableCrossAttn`, defined below) around the anchor over the
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

class DeformableCrossAttn(nn.Module):
    """Single-scale 3-D deformable cross-attention: each query samples ``n_points`` locations per head around
    its reference point on the node-feature grid and attends over them. Self-contained (torch only)."""

    def __init__(self, dim, n_heads=4, n_points=8):
        super().__init__()
        assert dim % n_heads == 0
        self.dim, self.h, self.k = dim, n_heads, n_points
        self.hd = dim // n_heads
        self.sampling_offsets = nn.Linear(dim, n_heads * n_points * 3)
        self.attn_weights = nn.Linear(dim, n_heads * n_points)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.sampling_offsets.weight); nn.init.zeros_(self.sampling_offsets.bias)
        nn.init.zeros_(self.attn_weights.weight); nn.init.zeros_(self.attn_weights.bias)

    def forward(self, q, ref, node_feat):
        """q:(B,Q,dim) queries; ref:(B,Q,3) reference points in [-1,1] (z,y,x); node_feat:(B,dim,nd,nh,nw)."""
        B, Q, _ = q.shape
        nd, nh, nw = node_feat.shape[2:]
        val = self.value_proj(node_feat.flatten(2).transpose(1, 2))          # (B, N, dim)
        val = val.transpose(1, 2).reshape(B, self.dim, nd, nh, nw)
        off = self.sampling_offsets(q).view(B, Q, self.h, self.k, 3)
        aw = self.attn_weights(q).view(B, Q, self.h, self.k)
        aw = torch.softmax(aw, dim=-1)
        # sampling locations in [-1,1]; offsets are in normalised units (small, zero-init => start at ref)
        loc = ref.view(B, Q, 1, 1, 3) + 0.1 * off                            # (B,Q,h,k,3)
        out = q.new_zeros(B, Q, self.dim)
        for hh in range(self.h):
            vh = val[:, hh * self.hd:(hh + 1) * self.hd]                      # (B,hd,nd,nh,nw)
            # grid_sample wants grid last-dim order (x,y,z); our coords are (z,y,x) -> flip
            g = loc[:, :, hh].flip(-1).reshape(B, Q, self.k, 1, 3)            # (B,Q,k,1,3)
            samp = F.grid_sample(vh, g, mode="bilinear", padding_mode="border",
                                 align_corners=True)                          # (B,hd,Q,k,1)
            samp = samp.squeeze(-1).permute(0, 2, 3, 1)                       # (B,Q,k,hd)
            wsum = (samp * aw[:, :, hh].unsqueeze(-1)).sum(dim=2)             # (B,Q,hd)
            out[:, :, hh * self.hd:(hh + 1) * self.hd] = wsum
        return self.out_proj(out)


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


def m2f_semantic_prob(layer_out: dict, mask_features: torch.Tensor, *, reduce: str = "max") -> torch.Tensor:
    """Mask2Former **semantic** readout — sheet-surface probability from the instance head (no separate dense head).

    Mask2Former does semantic segmentation as mask classification: combine per-query (class-prob × mask-prob). Here
    the single foreground class is "sheet", so ``sheet_prob(voxel) = reduce_q [ P(class_q=sheet) · σ(mask_q(voxel)) ]``
    with ``mask_q = mask_embed_q · mask_features`` (``mask_features`` is the full-res pixel-decoder output that
    consumes the stem skips). ``reduce="max"`` (is any sheet here) or ``"sum"`` (M2F's soft-union logit). Returns
    ``(B, D, H, W)`` in [0,1].
    """
    cls = torch.softmax(layer_out["class_logits"].float(), dim=-1)[..., 1]            # (B,Q) P(sheet)
    masks = torch.sigmoid(mask_logits_from_embed(layer_out["mask_embed"].float(), mask_features.float()))  # (B,Q,D,H,W)
    weighted = cls[..., None, None, None] * masks
    sem = weighted.amax(dim=1) if reduce == "max" else weighted.sum(dim=1)
    return sem.clamp(0.0, 1.0)


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
    """Self-attn among queries + multi-scale cross-attn + FFN + anchor refinement. Two cross-attention modes:

    * ``"deformable"`` — DETR/DINO-style: each query samples ``n_points`` around its **anchor** (``DeformableCrossAttn``).
    * ``"masked"`` — Mask2Former-style: full multi-head cross-attn over the feature tokens, **masked to the query's
      own predicted-mask foreground** (``cross_masks`` built by the caller from the previous layer's mask). Anchor-free
      localisation — the right fit for shapeless, window-spanning sheets (docs §13.6). The anchor is still carried +
      refined (for logging / FPS init) but does not drive attention in this mode."""

    def __init__(self, dim: int, n_heads: int, n_points: int, n_scales: int, mlp_ratio: int = 4,
                 anchor_step: float = 0.25, cross_attn: str = "deformable") -> None:
        super().__init__()
        self.cross_attn = cross_attn
        self.n_heads = int(n_heads)
        self.anchor_step = float(anchor_step)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        if cross_attn == "deformable":
            self.cross = nn.ModuleList([DeformableCrossAttn(dim, n_heads, n_points) for _ in range(n_scales)])
        else:                                                           # masked: one MHA per scale over feature tokens
            self.cross = nn.ModuleList([nn.MultiheadAttention(dim, n_heads, batch_first=True) for _ in range(n_scales)])
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.n3 = nn.LayerNorm(dim)
        self.anchor_mlp = nn.Linear(dim, 3)
        nn.init.zeros_(self.anchor_mlp.weight)
        nn.init.zeros_(self.anchor_mlp.bias)

    def forward(self, q: torch.Tensor, anchors: torch.Tensor, feats: Sequence[torch.Tensor],
                attn_mask: torch.Tensor | None = None, cross_masks: Sequence | None = None):
        s = self.n1(q)
        q = q + self.self_attn(s, s, s, need_weights=False, attn_mask=attn_mask)[0]
        qn = self.n2(q)
        if self.cross_attn == "deformable":
            cross = sum(c(qn, anchors, f) for c, f in zip(self.cross, feats))
        else:                                                           # masked cross-attn over flattened scale tokens
            cross = qn.new_zeros(qn.shape)
            for i, (c, f) in enumerate(zip(self.cross, feats)):
                kv = f.flatten(2).transpose(1, 2)                       # (B,N,dim) keys/values at this scale
                m = cross_masks[i] if cross_masks is not None else None  # (B·heads,Q,N) bool; None => attend all
                cross = cross + c(qn, kv, kv, need_weights=False, attn_mask=m)[0]
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
        add_null: bool = False,
        learned_queries: bool = False,
        fps_queries: bool = False,
        cross_attn: str = "deformable",
        gated_kv: bool = False,
    ) -> None:
        super().__init__()
        self.add_null = bool(add_null)
        self.learned_queries = bool(learned_queries)
        self.fps_queries = bool(fps_queries)                   # option 2 (§13.3): embedding-FPS anchored query init
        self.cross_attn = str(cross_attn)                      # "deformable" | "masked" (Mask2Former, docs §13.6)
        if self.cross_attn not in ("deformable", "masked"):
            raise ValueError(f"cross_attn must be 'deformable' or 'masked', got {cross_attn!r}.")
        if self.learned_queries and self.fps_queries:
            raise ValueError("learned_queries and fps_queries are mutually exclusive query-init modes.")
        if self.add_null:
            # reserved learned NULL / no-sheet query (index 0), HercUNet sheet-seg convention: excluded from the
            # Hungarian match, used as the background (owner==0) CE target so off-sheet voxels drain to null.
            self.null_query = nn.Parameter(torch.zeros(dim))
            self.null_anchor = nn.Parameter(torch.zeros(3))
        if self.learned_queries:
            # option 1: fixed persistent learned queries (DETR-style) + learned reference points — they specialise
            # (each query owns a sheet across steps) for stable Hungarian, instead of per-forward top-K selection.
            self.query_embed = nn.Parameter(torch.randn(int(n_queries), dim) * 0.02)
            self.query_ref = nn.Parameter(torch.rand(int(n_queries), 3) * 2 - 1)   # spread in [-1,1]
        self.gated_kv = bool(gated_kv)                         # design B: K/V + FPS seeds from surface-GATED features
        emb_dim = int(mask_dim) if self.gated_kv else int(token_dim)   # the FPS/denoise/content embedding source width
        self.selector = QuerySelector(token_dim, dim, n_queries)
        if self.gated_kv:
            # K/V + FPS seeds come from the surface-gated mask features (NOT raw frozen tokens — the bypass that made
            # queries lean on unadaptable frozen features; docs/dinovol_sheet_architecture.md §3 design B).
            self.kv_proj8 = nn.Conv3d(int(mask_dim), dim, kernel_size=1)   # gated feats @1/8 -> query width
            self.kv_proj4 = nn.Conv3d(int(mask_dim), dim, kernel_size=1)   # gated feats @1/4 -> query width
            self.gated_content = nn.Linear(int(mask_dim), dim)            # FPS query content from gated embeddings
        else:
            self.proj8 = nn.Conv3d(token_dim, dim, kernel_size=1)   # 1/8 token grid -> query width
            self.proj4 = nn.Conv3d(feat_14_dim, dim, kernel_size=1)  # 1/4 decoder features -> query width
        self.dn_content = nn.Linear(emb_dim, dim)              # denoise query content from pooled sheet embeddings (§13.2)
        self.layers = nn.ModuleList(
            [QueryDecoderLayer(dim, n_heads, n_points, n_scales=2, cross_attn=self.cross_attn)
             for _ in range(int(n_layers))]
        )
        self.class_head = nn.Linear(dim, 2)        # sheet vs no-object
        self.mask_embed = nn.Linear(dim, int(mask_dim))
        self.token_dim = int(token_dim)
        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.mask_dim = int(mask_dim)

    @staticmethod
    def _fps_indices(feat_n: torch.Tensor, k: int) -> torch.Tensor:
        """Pure-torch farthest-point sampling (§13.4) on L2-normalised features ``feat_n`` (B,G,d) → ``idx`` (B,K)
        long. Greedy: seed = the highest-norm... here features are unit, so seed = token farthest from the batch-item
        mean (most distinctive); then repeatedly add the point maximising the min distance to the chosen set. Squared
        euclidean on unit vectors == 2−2·cos, so this hops between cosine clusters → ≈ one seed per sheet. NON-
        differentiable (argmax over distances); run detached, like a top-K. In-graph on GPU (short serial loop)."""
        B, G, _ = feat_n.shape
        k = min(int(k), G)
        dev = feat_n.device
        idx = torch.zeros(B, k, dtype=torch.long, device=dev)
        # seed: farthest token from the per-item mean direction (deterministic, train/infer identical)
        mean = feat_n.mean(dim=1, keepdim=True)                              # (B,1,d)
        d0 = ((feat_n - mean) ** 2).sum(-1)                                  # (B,G)
        last = d0.argmax(dim=1)                                              # (B,)
        idx[:, 0] = last
        min_d = feat_n.new_full((B, G), float("inf"))
        bar = torch.arange(B, device=dev)
        for i in range(1, k):
            cur = feat_n[bar, last]                                         # (B,d) last chosen
            d = ((feat_n - cur.unsqueeze(1)) ** 2).sum(-1)                  # (B,G) sq-dist to last
            min_d = torch.minimum(min_d, d)
            last = min_d.argmax(dim=1)                                      # farthest from the chosen set
            idx[:, i] = last
        return idx

    def _fps_select(self, emb_seq: torch.Tensor, coords: torch.Tensor, content_proj):
        """Embedding-FPS anchored query init (§13.3): pick ``n_queries`` cells by FPS in NORMALISED-embedding
        (cosine) space — one anchor per sheet cluster — then DIFFERENTIABLY gather each one's content (``content_proj``:
        embed → query width) + anchor (cell centre). ``emb_seq`` is the raw frozen tokens (default) or the pooled
        surface-gated features (design B). GT-free: identical at train and inference. Returns
        ``(content (B,Q,dim), anchors (B,Q,3), objectness (B,G) zeros, idx (B,Q))`` to match the selector."""
        B, G, _ = emb_seq.shape
        with torch.no_grad():                                               # selection is non-differentiable (detached)
            feat_n = F.normalize(emb_seq.float(), dim=-1)                   # cosine ⇒ sq-euclidean on unit vectors
            idx = self._fps_indices(feat_n, self.selector.n_queries)       # (B,Q)
        gather = idx.unsqueeze(-1).expand(-1, -1, emb_seq.shape[-1])
        content = content_proj(torch.gather(emb_seq, 1, gather))            # (B,Q,dim) differentiable gather
        anchors = coords.unsqueeze(0).expand(B, -1, -1).gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))  # (B,Q,3)
        objectness = emb_seq.new_zeros(B, G)                               # unused (no two-stage objectness)
        return content, anchors, objectness, idx

    def _build_denoise(self, tokens_seq, coords, dn_owner, *, n_groups=2, content_noise=0.3, anchor_jitter=0.15):
        """Build a TRAIN-ONLY denoise query group (§13.2): one anchored query per GT sheet, × ``n_groups`` noised
        copies, initialised from that sheet's pooled token embedding (→ ``dn_content``) + its token centroid, with
        Gaussian content noise + anchor jitter. Returns ``(content (B,n_dn,dim), anchors (B,n_dn,3), valid (B,n_dn)
        bool, sheet (B,n_dn) long)`` or ``None`` if the batch has no sheets. Each denoise query is FIXED-assigned to
        its sheet in the loss — a direct 'reconstruct this sheet' signal that bypasses Hungarian."""
        B, G, _ = tokens_seq.shape
        side = round(G ** (1.0 / 3.0))
        k = max(1, dn_owner.shape[-1] // side)
        tok_owner = F.max_pool3d(dn_owner.float()[:, None], kernel_size=k, stride=k)[:, 0].reshape(B, G).long()  # (B,G)
        per_item = [torch.unique(tok_owner[b][tok_owner[b] > 0]) for b in range(B)]
        max_s = max((int(s.numel()) for s in per_item), default=0)
        if max_s == 0:
            return None
        n_dn = max_s * int(n_groups)
        dev = tokens_seq.device
        content = tokens_seq.new_zeros(B, n_dn, self.dim)
        anchors = tokens_seq.new_zeros(B, n_dn, 3)
        valid = torch.zeros(B, n_dn, dtype=torch.bool, device=dev)
        sheet = torch.zeros(B, n_dn, dtype=torch.long, device=dev)
        for b in range(B):
            for si, s in enumerate(per_item[b]):
                cells = tok_owner[b] == s
                pooled = tokens_seq[b][cells].mean(0)                       # (token_dim,) sheet token centroid-embed
                cen = coords[cells].mean(0)                                 # (3,) sheet token-space centroid
                c = self.dn_content(pooled)                                 # (dim,)
                cstd = c.detach().std().clamp_min(1e-4)
                for g in range(int(n_groups)):
                    slot = si * int(n_groups) + g
                    content[b, slot] = c + content_noise * cstd * torch.randn_like(c)
                    anchors[b, slot] = (cen + anchor_jitter * (torch.rand(3, device=dev) * 2 - 1)).clamp(-1.0, 1.0)
                    valid[b, slot] = True
                    sheet[b, slot] = int(s)
        return content, anchors, valid, sheet

    def _pool_mask_features(self, mask_features: torch.Tensor, sides: Sequence[int]):
        """Avg-pool the full-res ``mask_features`` (B, mask_dim, D,H,W) down to each attention scale's ``side`` →
        list of (B, mask_dim, N_s) flattened low-res pixel-embeddings — the keys for the Mask2Former attention mask
        (so it never materialises a full-res 128³ mask just to gate attention; §13.6)."""
        D = mask_features.shape[-1]
        out = []
        for side in sides:
            k = max(1, D // int(side))
            out.append(F.avg_pool3d(mask_features, kernel_size=k, stride=k).flatten(2))   # (B,mask_dim,N_s)
        return out

    def _cross_masks(self, mask_embed_q: torch.Tensor, mf_scales: Sequence[torch.Tensor]):
        """Masked-attention masks (Mask2Former) from the current queries' predicted mask: block a (query, token) pair
        wherever the query's low-res mask prob < 0.5. ``mask_embed_q`` (B,Q,mask_dim); ``mf_scales`` list of
        (B,mask_dim,N_s). Returns list of ``(B·heads, Q, N_s)`` bool masks (True = blocked). Detached (a gate, not a
        gradient path); a fully-blocked query falls back to attend-all (no NaN softmax), as in Mask2Former."""
        masks = []
        h = self.n_heads
        for mf in mf_scales:
            with torch.no_grad():
                logits = torch.einsum("bqc,bcn->bqn", mask_embed_q.float(), mf.float())   # (B,Q,N_s) low-res mask
                block = torch.sigmoid(logits) < 0.5                      # True = background => not attended
                block = block & ~block.all(dim=-1, keepdim=True)         # empty mask => attend everywhere
            B, Q, N = block.shape
            masks.append(block.unsqueeze(1).expand(B, h, Q, N).reshape(B * h, Q, N))
        return masks

    def forward(self, tokens_grid: torch.Tensor, feat_14: torch.Tensor,
                mask_features: torch.Tensor | None = None, dn_owner: torch.Tensor | None = None) -> dict:
        """``tokens_grid`` (B, token_dim, 16,16,16); ``feat_14`` (B, feat_14_dim, 32,32,32); ``mask_features``
        (B, mask_dim, 128³) — REQUIRED for ``cross_attn="masked"`` (it gates each query's cross-attention by its own
        predicted mask; §13.6), unused for ``"deformable"``. ``dn_owner`` (B,D,H,W) thin-surface owner ids — TRAIN
        ONLY: appends a fixed-assignment denoise query group (§13.2), isolated in the self-attention; ``None`` at
        inference (no denoise group)."""
        B, C, dz, dy, dx = tokens_grid.shape
        tokens_seq = tokens_grid.flatten(2).transpose(1, 2)                 # (B, G, token_dim), raster (z,y,x)
        coords = _coord_grid((dz, dy, dx), tokens_grid.device, tokens_grid.dtype)
        # Embedding source for FPS/denoise/content + the cross-attn K/V: the surface-GATED mask features (design B,
        # pooled to 1/8 + 1/4) or the raw frozen tokens + 1/4 decoder features (default).
        if self.gated_kv:
            if mask_features is None:
                raise ValueError("gated_kv needs mask_features (the surface-gated pixel-decoder features).")
            D = mask_features.shape[-1]
            k8, k4 = max(1, D // dz), max(1, D // (dz * 2))
            g8 = F.avg_pool3d(mask_features, kernel_size=k8, stride=k8)      # (B,mask_dim,16³)
            g4 = F.avg_pool3d(mask_features, kernel_size=k4, stride=k4)      # (B,mask_dim,32³)
            emb_seq = g8.flatten(2).transpose(1, 2)                         # (B,G,mask_dim)
            content_proj = self.gated_content
            feats = [self.kv_proj8(g8), self.kv_proj4(g4)]                   # multi-scale K/V from gated features
        else:
            emb_seq = tokens_seq
            content_proj = self.selector.content
            feats = [self.proj8(tokens_grid), self.proj4(feat_14)]          # multi-scale K/V (16³ tokens, 32³ feat_14)
        if self.learned_queries:                                            # option 1: fixed learned queries
            content = self.query_embed.to(tokens_seq.dtype).expand(B, -1, -1)
            anchors = self.query_ref.clamp(-1.0, 1.0).to(tokens_seq.dtype).expand(B, -1, -1)
            objectness = tokens_grid.new_zeros(B, dz * dy * dx)              # unused (no two-stage selection)
            idx = torch.full((B, self.selector.n_queries), -1, dtype=torch.long, device=tokens_grid.device)
        elif self.fps_queries:                                              # option 2 (§13.3): embedding-FPS init
            content, anchors, objectness, idx = self._fps_select(emb_seq, coords, content_proj)
        else:
            content, anchors, objectness, idx = self.selector(tokens_seq, coords)
        if self.add_null:                                                   # prepend the reserved null query (index 0)
            content = torch.cat([self.null_query.to(content.dtype).expand(B, 1, -1), content], dim=1)
            anchors = torch.cat([self.null_anchor.to(anchors.dtype).expand(B, 1, -1), anchors], dim=1)
            idx = torch.cat([idx.new_full((B, 1), -1), idx], dim=1)         # null has no source token

        n_match = content.shape[1]                                          # matching (+ null) queries
        dn = self._build_denoise(emb_seq, coords, dn_owner) if (dn_owner is not None and self.training) else None
        attn_mask = None
        if dn is not None:                                                  # append the denoise group, isolate it
            dn_content, dn_anchors, dn_valid, dn_sheet = dn
            content = torch.cat([content, dn_content.to(content.dtype)], dim=1)
            anchors = torch.cat([anchors, dn_anchors.to(anchors.dtype)], dim=1)
            L = content.shape[1]
            attn_mask = torch.zeros(L, L, dtype=torch.bool, device=content.device)
            attn_mask[:n_match, n_match:] = True                            # matching can't see denoise
            attn_mask[n_match:, :n_match] = True                            # denoise can't see matching

        masked = self.cross_attn == "masked"
        if masked:
            if mask_features is None:
                raise ValueError("cross_attn='masked' needs mask_features (the pixel decoder's full-res embeddings).")
            sides = [int(f.shape[-1]) for f in feats]                         # scale sides from the K/V maps (16³, 32³)
            mf_scales = self._pool_mask_features(mask_features, sides)        # low-res keys for the mask
        q = content
        layers_out = []
        for layer in self.layers:
            # masked mode (§13.6): gate THIS layer's cross-attn by the current queries' predicted-mask foreground.
            cross_masks = self._cross_masks(self.mask_embed(q), mf_scales) if masked else None
            q, anchors = layer(q, anchors, feats, attn_mask=attn_mask, cross_masks=cross_masks)
            layers_out.append(
                {"class_logits": self.class_head(q), "mask_embed": self.mask_embed(q), "anchors": anchors}
            )
        out = {"objectness": objectness, "query_idx": idx, "layers": layers_out, "n_match": n_match}
        if dn is not None:
            out["dn_valid"], out["dn_sheet"] = dn_valid, dn_sheet
        return out

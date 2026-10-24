"""HercuViT losses — semantic (Dice + CE, deep supervision) and Mask2Former-style instance set prediction.

Semantic (``docs/hercuvit.md`` §6/§7): soft-Dice + cross-entropy on the full-res sheet-surface logits, with the
same loss at 1/2 and 1/4 (targets max-pooled to each scale so the thin surface survives the downsample).

Instance (§7, v2 DECOUPLED — ``instance_loss_decoupled``): set prediction over the ``n_queries`` instance
queries with NO class head and NO null query (surface presence is the separate surface head's job) —
  * **Hungarian matching** (scipy ``linear_sum_assignment``) per decoder layer on the ownership cross-entropy
    cost, evaluated on **importance-sampled points** (not all 128³) via ``mask_logits_at_points``;
  * loss: fg-gated **ownership cross-entropy** (``w_mask_bce``) over the matched queries at the sheet voxels;
  * **auxiliary** losses after every decoder layer (``aux_weight``).
The legacy M2F-with-class terms (``w_class``/``no_object_weight``) survive only in the gated
``m2f_symskel_mask_loss`` alternative; the decoupled path above does not use them.

Per-sample gating (§7): instance losses are computed only for samples with ``has_identity``; for a sample
without identity the instance loss is a **zero that is still grad-connected** to the instance outputs (so the
optimiser step is well defined and DDP sees every parameter), never a bare ``0.0``.

Point sampling: a cheap, CPU-light importance mix — half the points drawn from GT sheet voxels (guarantees
positives for every sheet), half uniform (the background / boundary). This approximates PointRend/Mask2Former
uncertainty sampling without a first forward; recorded as a simplification in ``docs/hercuvit.md`` §13.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import torch.nn.functional as F

from ..model.decoder_common import mask_logits_at_points, mask_logits_from_embed


# --------------------------------------------------------------------------- semantic
def _downsample_target(tgt01: torch.Tensor, side: int) -> torch.Tensor:
    """Max-pool a ``(B,1,D,H,W)`` {0,1} target down to spatial ``side`` (preserves the thin surface)."""
    cur = tgt01.shape[-1]
    if cur == side:
        return tgt01
    k = cur // side
    return F.max_pool3d(tgt01, kernel_size=k, stride=k)


def semantic_loss(sem_logits, sem_target: torch.Tensor, recipe) -> dict:
    """``sem_logits`` = [full, 1/2, 1/4]; ``sem_target`` = (B,D,H,W) long. Thin medial-surface stack (ported from
    HercUNet, see ``train/surface_loss.py``): foreground-weighted CE + Focal-Tversky + clDice skeleton-recall +
    separation penalty, with deep supervision. Returns ``{sem, ce, tversky, skel, sep, dice}`` (``dice``=Tversky,
    kept for log compatibility)."""
    from .surface_loss import surface_terms
    weights = recipe.deep_sup_weights
    sem_f = sem_target[:, None].float()                                 # (B,1,D,H,W) {0 bg,1 surface,2 ignore}
    total = sem_logits[0].new_zeros(())
    acc = {k: sem_logits[0].new_zeros(()) for k in ("ce", "tversky", "skel", "sep")}
    for i, logits in enumerate(sem_logits):
        w = float(weights[i]) if i < len(weights) else 0.0
        side = logits.shape[-1]
        tl = F.interpolate(sem_f, size=(side, side, side), mode="nearest")[:, 0].long()  # nearest keeps {0,1,2}
        terms = surface_terms(
            logits.float(), tl,
            tversky_alpha=recipe.tversky_alpha, tversky_beta=recipe.tversky_beta, tversky_gamma=recipe.tversky_gamma,
            skel_iters=recipe.skel_iters, sep_radius=recipe.sep_radius, ignore_label=2,
        )
        total = total + w * (recipe.w_sem_ce * terms["ce"] + recipe.w_sem_tversky * terms["tversky"]
                             + recipe.w_sem_skel * terms["skel"] + recipe.w_sem_sep * terms["sep"])
        for k in acc:
            acc[k] = acc[k] + w * terms[k]
    return {"sem": total, "ce": acc["ce"].detach(), "tversky": acc["tversky"].detach(),
            "skel": acc["skel"].detach(), "sep": acc["sep"].detach(), "dice": acc["tversky"].detach()}


# --------------------------------------------------------------------------- SymSkel directly on matched masks (optional)
def m2f_symskel_mask_loss(instance: dict, mask_features: torch.Tensor, sem: torch.Tensor, owner: torch.Tensor,
                          has_identity, recipe, *, seed: int = 0, use_softmax: bool = False) -> tuple:
    """OPTIONAL per-query thinness add-on (SymSkel, oldest-HercUNet supervision) over the plain-M2F default. Apply
    the SymSkel thin-surface stack (fg-weighted CE precision + Focal-Tversky + clDice skeleton-recall) DIRECTLY to
    each matched query's mask —
    GT = that sheet's **thin medial surface** (``sem==1 & owner==k``). So each query's mask *is* a thin sheet, its
    index *is* the membership, and the M2F readout ``max_q[P(sheet_q)·σ(mask_q)]`` is thin for free. Pair with
    ``affinity_malis_loss`` for the SEPARATION that keeps neighbouring sigmoid masks from leaking into each other
    (classical HercUNet's job for AffMALIS).

    ``use_softmax=False`` (default, the chosen path): **independent per-query sigmoid** masks — the vanilla-M2F
    readout that overfits cleanly; separation is carried by AffMALIS, not by exclusivity. ``use_softmax=True`` makes
    the per-query mask the softmax-over-queries OWNERSHIP channel — DEAD END: the background-dominated null channel
    starves the thin targets and everything collapses (``tv~0.97``). The fg-weighted CE is the precision term
    (without it the recall-only skeleton term collapses the mask to all-foreground). Reuses ``surface_loss``
    primitives + DETR class CE; Hungarian match on a cheap point-sampled dice cost. Returns ``(loss, terms)``."""
    from scipy.optimize import linear_sum_assignment
    from .surface_loss import focal_tversky, skeleton_recall

    lay = instance["layers"][-1]
    cls_all = lay["class_logits"]; me_all = lay["mask_embed"]
    B, Q = cls_all.shape[:2]
    C = mask_features.shape[1]
    dev = mask_features.device
    gen = torch.Generator(device=dev); gen.manual_seed(int(seed))
    tot_cls = mask_features.new_zeros(()); tot_tv = mask_features.new_zeros(())
    tot_sk = mask_features.new_zeros(()); tot_ce = mask_features.new_zeros(())
    fgw = float(recipe.sem_fg_weight)
    n_item = 0; n_mask = 0
    cw = torch.tensor([recipe.no_object_weight, 1.0], device=dev)
    for b in range(B):
        if not bool(has_identity[b]):
            tot_cls = tot_cls + cls_all[b].sum() * 0.0 + me_all[b].sum() * 0.0
            continue
        me = me_all[b].float()                                              # (Q,C)
        sem_b = sem[b]; own_b = owner[b]
        surf = (sem_b == 1); valid = (sem_b != 2)
        ids = [int(i) for i in torch.unique(own_b[surf]).tolist() if i > 0]  # sheets that carry thin surface
        if not ids:
            tot_cls = tot_cls + cls_all[b].sum() * 0.0 + me.sum() * 0.0
            continue
        # per-query masks: SOFTMAX ownership (exclusive) or independent sigmoid. (Q,D,H,W)
        mlog_all = torch.einsum("qc,cdhw->qdhw", me, mask_features[b].float())
        p_all = (torch.softmax(mlog_all, dim=0) if use_softmax else torch.sigmoid(mlog_all))
        p_flat = p_all.reshape(Q, -1)
        # --- Hungarian on a cheap point-sampled dice cost (thin-surface targets), on the prob channels ---
        sflat = surf.reshape(-1); oflat = own_b.reshape(-1); vflat = valid.reshape(-1)
        pos = sflat.nonzero(as_tuple=False).squeeze(1)
        neg = (vflat & ~sflat).nonzero(as_tuple=False).squeeze(1)
        take = lambda idx, k: idx[torch.randint(0, int(idx.numel()), (k,), generator=gen, device=dev)] if idx.numel() else idx
        pts = torch.cat([take(pos, min(1024, int(pos.numel()))), take(neg, min(1024, int(neg.numel())))], 0)
        gt_pts = torch.stack([((oflat == k) & sflat)[pts].float() for k in ids], 0)   # (G,P)
        with torch.no_grad():
            pp = p_flat[:, pts]                                             # (Q,P) prob
            num = 2.0 * torch.einsum("qp,gp->qg", pp, gt_pts)
            den = pp.sum(1).unsqueeze(1) + gt_pts.sum(1).unsqueeze(0)
            cost = (1.0 - (num + 1.0) / (den + 1.0)).cpu().numpy()
            rows, cols = linear_sum_assignment(cost)
        cls_t = torch.zeros(Q, dtype=torch.long, device=dev); cls_t[torch.as_tensor(rows, device=dev)] = 1
        tot_cls = tot_cls + F.cross_entropy(cls_all[b], cls_t, weight=cw)
        # --- SymSkel per matched ownership channel vs its sheet's THIN surface (dense, full-res) ---
        v = valid.float()[None, None]
        for r, c in zip(rows.tolist(), cols.tolist()):
            p = p_all[r].float().unsqueeze(0).unsqueeze(0).clamp(1e-6, 1.0 - 1e-6)   # (1,1,D,H,W) fp32
            gt = ((own_b == ids[c]) & surf).float()[None, None]
            wmask = v * (1.0 + (fgw - 1.0) * gt)                            # fg-weighted CE = PRECISION term
            bce = -(gt * torch.log(p) + (1.0 - gt) * torch.log1p(-p))
            tot_ce = tot_ce + (bce * wmask).sum() / wmask.sum().clamp_min(1.0)
            tot_tv = tot_tv + focal_tversky(p * v, gt, alpha=recipe.tversky_alpha, beta=recipe.tversky_beta,
                                            gamma=recipe.tversky_gamma)
            tot_sk = tot_sk + skeleton_recall(p, gt, iters=recipe.skel_iters, mask=v)
            n_mask += 1
        n_item += 1
    di = max(n_item, 1); dm = max(n_mask, 1)
    loss = (recipe.w_class * (tot_cls / di) + recipe.w_sem_ce * (tot_ce / dm)
            + recipe.w_sem_tversky * (tot_tv / dm) + recipe.w_sem_skel * (tot_sk / dm))
    return loss, {"class": (tot_cls / di).detach(), "ce": (tot_ce / dm).detach(),
                  "tversky": (tot_tv / dm).detach(), "skel": (tot_sk / dm).detach(), "n_mask": float(n_mask)}


# --------------------------------------------------------------------------- membership-affinity MALIS (AffMALIS)
def _membership_affinity(p: torch.Tensor, offsets) -> torch.Tensor:
    """Soft same-owner affinity from a per-voxel membership ``p`` (Q,D,H,W softmax over queries). For each offset
    ``δ`` returns ``aff_δ(v) = Σ_q p(q|v)·p(q|v+δ)`` ∈ [0,1] = P(v and v+δ share an owner). Out-of-bounds → 0.
    Richer than a fixed axis-offset affinity head: the membership defines affinity at ANY offset, consistent with
    the instance assignment. Returns (C,D,H,W)."""
    Q, D, H, W = p.shape
    out = p.new_zeros((len(offsets), D, H, W))
    for c, (dz, dy, dx) in enumerate(offsets):
        z0, z1 = max(0, -dz), D - max(0, dz)
        y0, y1 = max(0, -dy), H - max(0, dy)
        x0, x1 = max(0, -dx), W - max(0, dx)
        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            continue
        a = p[:, z0:z1, y0:y1, x0:x1]
        b = p[:, z0 + dz:z1 + dz, y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        out[c, z0:z1, y0:y1, x0:x1] = (a * b).sum(0)
    return out


def affinity_malis_loss(instance: dict, mask_features: torch.Tensor, owner: torch.Tensor, has_identity, recipe,
                        *, rng=None) -> tuple:
    """AffMALIS on the M2F readout's MEMBERSHIP (option B) — REUSES ``hercunet.losses.malis.ConstrainedMalisLoss``.

    Per identity sample: crop a random ``recipe.malis_crop``³ cube, form the per-voxel membership
    ``p = softmax_Q(mask_embed · mask_features)`` on it, derive the soft same-owner affinity
    ``aff_δ = Σ_q p(q|v)·p(q|v+δ)`` over ``recipe.malis_offsets``, and run constrained MALIS against ``owner`` —
    the maximin attractive/repulsive passes that give HercUNet its clean sheet separation. Gradient flows through
    the affinity into the M2F ownership (mask_embed + mask_features). Gated by ``has_identity``. Returns
    ``(loss, terms)`` with ``terms = {malis, grow, sep, n_identity}``."""
    import numpy as np
    from ...losses.malis import ConstrainedMalisLoss

    rng = rng or np.random.default_rng()
    offsets = [tuple(o) for o in recipe.malis_offsets]
    crop = int(recipe.malis_crop)
    malis = ConstrainedMalisLoss(offsets=offsets, crop=None)   # we crop the features ourselves (cheaper)
    lay = instance["layers"][-1]
    n_match = instance.get("n_match")                          # exclude the train-only DN queries from the membership
    me = lay["mask_embed"]                                     # (B, Q(+DN), C)
    me = (me[:, :n_match] if n_match is not None else me).float()   # matching queries only (matches inference)
    B = mask_features.shape[0]
    total = mask_features.sum() * 0.0                          # grad-connected zero (no-identity / empty batch)
    grow = sep = 0.0
    n = 0
    for b in range(B):
        if not bool(has_identity[b]):
            total = total + me[b].sum() * 0.0
            continue
        mf_b = mask_features[b].float()                        # (C, D, H, W)
        own_b = owner[b]
        C, D, H, W = mf_b.shape
        cz, cy, cx = min(crop, D), min(crop, H), min(crop, W)
        z = int(rng.integers(0, D - cz + 1)); y = int(rng.integers(0, H - cy + 1)); x = int(rng.integers(0, W - cx + 1))
        mf_c = mf_b[:, z:z + cz, y:y + cy, x:x + cx]
        own_c = own_b[z:z + cz, y:y + cy, x:x + cx]
        own_logits = torch.einsum("qc,cdhw->qdhw", me[b], mf_c)          # (Q, cz, cy, cx)
        p = torch.softmax(own_logits, dim=0)                            # membership
        aff = _membership_affinity(p, offsets).unsqueeze(0)            # (1, C, cz, cy, cx)
        total = total + malis(aff, own_c.unsqueeze(0), rng)
        grow += float(getattr(malis, "last_grow", 0.0)); sep += float(getattr(malis, "last_sep", 0.0))
        n += 1
    denom = max(n, 1)
    return total / denom, {"malis": (total / denom).detach(), "grow": grow / denom, "sep": sep / denom,
                           "n_identity": float(n)}


# --------------------------------------------------------------------------- instance helpers
def _sample_points(owner_flat: torch.Tensor, ignore_flat: torch.Tensor, n: int, rng: torch.Generator) -> torch.Tensor:
    """``n`` flat voxel indices: ~half from sheet voxels (owner>0), the rest from **confident background**
    (owner==0 AND not ignore). Ignore voxels (``ignore_flat``) are never sampled — they are neither positive nor
    negative, matching the confidence/collapsed-sheet masking. Returns (n,) long."""
    dev = owner_flat.device
    pos = (owner_flat > 0).nonzero(as_tuple=False).squeeze(1)
    neg = ((owner_flat == 0) & (~ignore_flat)).nonzero(as_tuple=False).squeeze(1)
    n_pos = min(n // 2, int(pos.numel()))
    out = []
    if n_pos > 0:
        out.append(pos[torch.randint(0, int(pos.numel()), (n_pos,), generator=rng, device=dev)])
    n_neg = n - n_pos
    if n_neg > 0 and int(neg.numel()) > 0:
        out.append(neg[torch.randint(0, int(neg.numel()), (n_neg,), generator=rng, device=dev)])
    elif n_neg > 0 and int(pos.numel()) > 0:                     # degenerate: no clean bg → fall back to positives
        out.append(pos[torch.randint(0, int(pos.numel()), (n_neg,), generator=rng, device=dev)])
    return torch.cat(out, 0) if out else torch.zeros(0, dtype=torch.long, device=dev)


# --------------------------------------------------------------------------- v2 decoupled instance loss
def _decoupled_item(me_L, feat_pts, owner_pts, ids, mf_flat, owner_flat, recipe):
    """v2 no-null ownership losses for ONE item, ALL L layers batched. ``me_L`` (L,Q,C); ``feat_pts`` (C,P) /
    ``owner_pts`` (P,) = point-sampled features/owner (Hungarian COST only); ``mf_flat`` (C,N) = mask_features @1/1;
    ``owner_flat`` (N,) = THIN owner ids (0 = bg/ignore).

    Match: per-layer Hungarian on **ownership-CE cost ONLY** (``cost[q,s] = -mean_{v∈s} log softmax_Q(dot)[q]``) —
    no Dice, no class (they would compete with the decoupled surface head). Supervise: **fg-gated no-null
    softmax-over-Q ownership CE** — each sheet voxel → its matched query, softmax over the Q queries; background is
    NOT a class (the surface head gates it at inference). Returns the aux-weighted CE + final-layer metrics.
    """
    Lx, Q, C = me_L.shape
    device = me_L.device
    wl = torch.tensor([1.0 if li == Lx - 1 else recipe.aux_weight for li in range(Lx)], device=device)
    z = me_L.sum() * 0.0
    if len(ids) == 0:                                                   # no sheets → no ownership supervision
        return {"ce": z, "own_correct": z, "own_fg": z, "n_fire": z}

    # --- per-layer Hungarian on the point-sampled ownership-CE cost (one GPU→CPU transfer for all L) ---
    S = len(ids)
    pred_pts = torch.einsum("lqc,cp->lqp", me_L.float(), feat_pts.float())          # (L,Q,P)
    logp = F.log_softmax(pred_pts, dim=1)                                           # softmax OVER QUERIES
    gt_pts = torch.stack([(owner_pts == i).float() for i in ids], 0)                # (S,P)
    cnt = gt_pts.sum(-1).clamp_min(1.0)                                             # (S,)
    with torch.no_grad():
        cost = -torch.einsum("lqp,sp->lqs", logp, gt_pts) / cnt[None, None, :]      # (L,Q,S) ownership-CE cost
        from scipy.optimize import linear_sum_assignment
        cc = cost.float().cpu().numpy()                                             # ONE transfer
        rows_L = [None] * Lx; cols_L = [None] * Lx
        for li in range(Lx):
            rows_L[li], cols_L[li] = linear_sum_assignment(cc[li])
    qmap = torch.zeros(Lx, S, dtype=torch.long, device=device)                      # sheet-index → matched query, per layer
    for li in range(Lx):
        rt = torch.as_tensor(rows_L[li], dtype=torch.long, device=device)
        ct = torch.as_tensor(cols_L[li], dtype=torch.long, device=device)
        qmap[li, ct] = rt

    # --- fg-gated no-null ownership CE (gather ONLY the thin sheet voxels — bg is never materialised) ---
    fg_idx = (owner_flat > 0).nonzero(as_tuple=False).squeeze(1)                    # (n_fg,) thin sheet voxels
    mf_fg = mf_flat[:, fg_idx].float()                                             # (C, n_fg)
    owner_fg = owner_flat[fg_idx]                                                  # (n_fg,)
    ids_t = torch.as_tensor(ids, dtype=owner_flat.dtype, device=device)            # (S,)
    sidx = (owner_fg[:, None] == ids_t[None, :]).float().argmax(1)                 # (n_fg,) sheet index per fg voxel
    mlog_fg = torch.einsum("lqc,cf->lqf", me_L.float(), mf_fg)                     # (L,Q,n_fg) ownership logits
    tgt = qmap[:, sidx]                                                            # (L,n_fg) matched query per fg voxel
    ce = F.cross_entropy(mlog_fg, tgt, reduction="none")                           # (L,n_fg) softmax over Q
    loss_ce = (ce.mean(1) * wl).sum()
    with torch.no_grad():                                                          # metrics from the FINAL layer
        pred = mlog_fg[-1].argmax(0)                                               # winning query per fg voxel
        own_correct = (pred == tgt[-1]).sum().float()
        own_fg = torch.tensor(float(tgt.shape[1]), device=device)
        used = torch.zeros(Q, dtype=torch.bool, device=device)
        if pred.numel():
            used[pred.unique()] = True
        n_fire = used.sum().float()
    return {"ce": loss_ce, "own_correct": own_correct, "own_fg": own_fg, "n_fire": n_fire}


def _decoupled_dn(dn_me_L, mf_flat, owner_flat, dn_sheet_b, dn_valid_b, recipe):
    """DN group for ONE item, ALL L layers. Fixed assignment (no Hungarian): each sheet voxel → its FIXED dn query,
    same fg-gated no-null softmax-over-the-DN-queries CE. Returns the aux-weighted CE (grad-connected zero if none)."""
    Lx = dn_me_L.shape[0]
    device = dn_me_L.device
    wl = torch.tensor([1.0 if li == Lx - 1 else recipe.aux_weight for li in range(Lx)], device=device)
    vq = dn_valid_b.nonzero(as_tuple=False).squeeze(1)
    if vq.numel() == 0:
        return dn_me_L.sum() * 0.0
    me = dn_me_L[:, vq].float()                                                    # (L,M,C) the valid DN queries
    sheets = dn_sheet_b[vq]                                                        # (M,) their fixed sheet ids
    fg_idx = (owner_flat > 0).nonzero(as_tuple=False).squeeze(1)                   # thin sheet voxels
    if fg_idx.numel() == 0:
        return dn_me_L.sum() * 0.0
    mf_fg = mf_flat[:, fg_idx].float()                                            # (C,n_fg)
    owner_fg = owner_flat[fg_idx]                                                 # (n_fg,)
    match = owner_fg[:, None] == sheets[None, :]                                  # (n_fg,M)
    has = match.any(dim=1)
    pref = torch.arange(sheets.numel(), device=device).float() + 1.0             # last-wins on duplicate groups
    qi = (match.float() * pref[None, :]).argmax(1)                                # (n_fg,) dn-query index
    mlog = torch.einsum("lmc,cf->lmf", me, mf_fg)                                 # (L,M,n_fg) over the DN queries
    tgt = torch.where(has, qi, torch.full_like(qi, -100))                         # (n_fg,) non-dn-sheet voxels → -100
    ce = F.cross_entropy(mlog, tgt[None, :].expand(Lx, -1), ignore_index=-100, reduction="none")  # (L,n_fg)
    ce = ce.sum(1) / (tgt != -100).sum().clamp_min(1)                             # mean over supervised voxels
    return (ce * wl).sum()


def instance_loss_decoupled(instance: dict, mask_features: torch.Tensor, owner: torch.Tensor,
                            has_identity, recipe, *, ignore: torch.Tensor | None = None, seed: int = 0) -> dict:
    """v2 DECOUPLED set-prediction loss (``docs/hercuvit-mask-architecture.md`` §10). No null, no class head, no
    objectness (the surface head is the objectness), no mask-Dice. ``mask_features`` and ``owner`` are @1/1; ``owner``
    is the THIN owner (sheet id only on the medial, 0 elsewhere), which is naturally fg-gated and ignore-free.

    Returns ``inst`` (= weighted total) plus per-term entries for the trainer's logging (``class``/``mask_dice``/``obj``
    are 0 — not used by the decoupled head); the ownership CE sits in the ``mask_bce`` slot (weight ``w_mask_bce``)."""
    B = mask_features.shape[0]
    C = mask_features.shape[1]
    layers = instance["layers"]
    n_match = instance.get("n_match")
    dn_valid = instance.get("dn_valid"); dn_sheet = instance.get("dn_sheet")
    has_dn = dn_valid is not None
    device = mask_features.device
    rng = torch.Generator(device=device); rng.manual_seed(int(seed))

    tot_ce = mask_features.new_zeros(())
    tot_dn = mask_features.new_zeros(())
    own_correct = mask_features.new_zeros(()); own_fg = mask_features.new_zeros(()); fire_sum = mask_features.new_zeros(())
    n_ident = 0
    for b in range(B):
        if not bool(has_identity[b]):
            z = layers[0]["mask_embed"][b].sum() * 0.0
            for lay in layers:
                z = z + lay["mask_embed"][b].sum() * 0.0
            tot_ce = tot_ce + z
            continue
        n_ident += 1
        owner_flat = owner[b].reshape(-1)
        ign_b = (ignore[b] if ignore is not None else torch.zeros_like(owner[b], dtype=torch.bool)).bool()
        ids = [int(i) for i in torch.unique(owner_flat).tolist() if i > 0]
        pts = _sample_points(owner_flat, ign_b.reshape(-1), recipe.point_sample, rng)   # Hungarian COST only
        mf_flat = mask_features[b].reshape(C, -1)
        feat_pts = mf_flat[:, pts]; owner_pts = owner_flat[pts]
        me_L = torch.stack([lay["mask_embed"][b][:n_match] for lay in layers])          # (L,Q,C)
        out = _decoupled_item(me_L, feat_pts, owner_pts, ids, mf_flat, owner_flat, recipe)
        tot_ce = tot_ce + out["ce"]
        own_correct = own_correct + out["own_correct"]; own_fg = own_fg + out["own_fg"]; fire_sum = fire_sum + out["n_fire"]
        if has_dn:
            dn_me_L = torch.stack([lay["mask_embed"][b][n_match:] for lay in layers])
            tot_dn = tot_dn + _decoupled_dn(dn_me_L, mf_flat, owner_flat, dn_sheet[b], dn_valid[b], recipe)

    denom = max(n_ident, 1)
    dn_w = float(getattr(recipe, "dn_weight", 1.0))
    z = mask_features.new_zeros(())
    inst = (recipe.w_mask_bce * tot_ce + dn_w * recipe.w_mask_bce * tot_dn) / denom
    return {
        "inst": inst,
        "class": z, "mask_bce": (tot_ce / denom).detach(), "mask_dice": z, "obj": z,
        "dn": (tot_dn / denom).detach(),
        "own_correct": own_correct.detach(), "own_fg": own_fg.detach(),
        "n_fire": (fire_sum / denom).detach(),
    }

"""HercuViT losses — semantic (Dice + CE, deep supervision) and Mask2Former-style instance set prediction.

Semantic (``docs/hercuvit.md`` §6/§7): soft-Dice + cross-entropy on the full-res sheet-surface logits, with the
same loss at 1/2 and 1/4 (targets max-pooled to each scale so the thin surface survives the downsample).

Instance (§7): DETR/Mask2Former set prediction over the ``n_queries`` instance queries —
  * **Hungarian matching** (scipy ``linear_sum_assignment``) per decoder layer on ``cost_class + cost_mask_bce +
    cost_mask_dice``, evaluated on **importance-sampled points** (not all 128³) via ``mask_logits_at_points``;
  * matched queries: mask **BCE + Dice** at those points; all queries: sheet / no-object **classification**;
  * **token objectness** two-stage loss (which 16³ tokens sit on a sheet);
  * **auxiliary** losses after every decoder layer.

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

from ..model.instance_head import mask_logits_at_points, mask_logits_from_embed


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
            fg_weight=recipe.sem_fg_weight,
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
    """AffMALIS on the M2F readout's MEMBERSHIP (option B) — REUSES ``hercunet.training.malis.ConstrainedMalisLoss``.

    Per identity sample: crop a random ``recipe.malis_crop``³ cube, form the per-voxel membership
    ``p = softmax_Q(mask_embed · mask_features)`` on it, derive the soft same-owner affinity
    ``aff_δ = Σ_q p(q|v)·p(q|v+δ)`` over ``recipe.malis_offsets``, and run constrained MALIS against ``owner`` —
    the maximin attractive/repulsive passes that give HercUNet its clean sheet separation. Gradient flows through
    the affinity into the M2F ownership (mask_embed + mask_features). Gated by ``has_identity``. Returns
    ``(loss, terms)`` with ``terms = {malis, grow, sep, n_identity}``."""
    import numpy as np
    from ...training.malis import ConstrainedMalisLoss

    rng = rng or np.random.default_rng()
    offsets = [tuple(o) for o in recipe.malis_offsets]
    crop = int(recipe.malis_crop)
    malis = ConstrainedMalisLoss(offsets=offsets, crop=None)   # we crop the features ourselves (cheaper)
    lay = instance["layers"][-1]
    me = lay["mask_embed"].float()                             # (B, Q, C)
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


def _bce_cost(pred_logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean-over-points BCE cost matrix. ``pred_logits`` (Q,P), ``gt`` (G,P) → (Q,G)."""
    Q, P = pred_logits.shape
    G = gt.shape[0]
    pl = pred_logits.unsqueeze(1).expand(Q, G, P)
    gg = gt.unsqueeze(0).expand(Q, G, P)
    return F.binary_cross_entropy_with_logits(pl, gg, reduction="none").mean(-1)


def _dice_cost(prob: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Soft-Dice cost matrix. ``prob`` (Q,P) in [0,1], ``gt`` (G,P) → (Q,G)."""
    num = 2.0 * torch.einsum("qp,gp->qg", prob, gt)
    den = prob.sum(-1).unsqueeze(1) + gt.sum(-1).unsqueeze(0)
    return 1.0 - (num + 1.0) / (den + 1.0)


def _instance_item(layer_out_b, feat_pts, owner_pts, ids, mask_features_b, owner_b, valid_b, recipe):
    """Matched set-prediction losses for ONE item at ONE decoder layer. Hungarian assignment uses the cheap
    point-sampled cost, but the **mask loss is DENSE** — the matched queries' full-resolution masks (``mask_embed
    · mask_features``) are supervised against the GT sheet over every valid (non-ignore) voxel. Dense is essential
    here: sparse point-sampled mask loss leaves the full-res mask features sheet-uncorrelated (measured AUC≈0.5).

    ``feat_pts`` (C,P) / ``owner_pts`` (P,) = sampled features/owner (cost only); ``mask_features_b`` (C,D,H,W),
    ``owner_b`` (D,H,W), ``valid_b`` (D,H,W bool) = dense. Returns ``{class, mask_bce, mask_dice}``.
    """
    class_logits = layer_out_b["class_logits"]                         # (Q,2)
    mask_embed = layer_out_b["mask_embed"]                             # (Q,C)
    Q = class_logits.shape[0]
    device = class_logits.device
    G = len(ids)
    if G == 0:
        cls_target = torch.zeros(Q, dtype=torch.long, device=device)   # all no-object
        cw = torch.tensor([recipe.no_object_weight, 1.0], device=device)
        loss_class = F.cross_entropy(class_logits, cls_target, weight=cw)
        zero = mask_embed.sum() * 0.0
        return {"class": loss_class, "mask_bce": zero, "mask_dice": zero}

    # --- Hungarian assignment on the cheap sampled cost ---
    pred_pts = mask_logits_at_points(mask_embed.unsqueeze(0), feat_pts.unsqueeze(0))[0]  # (Q,P)
    gt_pts = torch.stack([(owner_pts == i).float() for i in ids], 0)   # (G,P)
    with torch.no_grad():
        cls_prob = torch.softmax(class_logits, dim=-1)
        cost = (recipe.cost_class * (-cls_prob[:, 1].unsqueeze(1).expand(Q, G))
                + recipe.cost_mask_bce * _bce_cost(pred_pts, gt_pts)
                + recipe.cost_mask_dice * _dice_cost(torch.sigmoid(pred_pts), gt_pts))
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(cost.detach().float().cpu().numpy())
    rows_t = torch.as_tensor(rows, dtype=torch.long, device=device)

    cls_target = torch.zeros(Q, dtype=torch.long, device=device)
    cls_target[rows_t] = 1                                             # matched queries → sheet
    cw = torch.tensor([recipe.no_object_weight, 1.0], device=device)
    loss_class = F.cross_entropy(class_logits, cls_target, weight=cw)

    # --- DENSE mask loss on the matched queries (full-res, ignore-masked) ---
    me_m = mask_embed[rows_t]                                          # (M,C)
    mlog = mask_logits_from_embed(me_m.unsqueeze(0), mask_features_b.unsqueeze(0))[0]     # (M,D,H,W)
    gt_m = torch.stack([(owner_b == int(ids[c])).float() for c in cols.tolist()], 0)     # (M,D,H,W)
    v = valid_b.float().unsqueeze(0)                                   # (1,D,H,W)
    bce = F.binary_cross_entropy_with_logits(mlog, gt_m, reduction="none")
    loss_bce = (bce * v).sum() / (v.sum() * mlog.shape[0]).clamp_min(1.0)
    p = torch.sigmoid(mlog)
    num = 2.0 * (p * gt_m * v).sum(dim=(1, 2, 3)) + 1.0
    den = (p * v).sum(dim=(1, 2, 3)) + (gt_m * v).sum(dim=(1, 2, 3)) + 1.0
    loss_dice = (1.0 - num / den).mean()
    return {"class": loss_class, "mask_bce": loss_bce, "mask_dice": loss_dice}


def instance_loss(instance: dict, mask_features: torch.Tensor, owner: torch.Tensor,
                  has_identity, recipe, *, ignore: torch.Tensor | None = None, seed: int = 0) -> dict:
    """Mask2Former set-prediction + objectness loss over the batch. Returns ``{inst, class, mask_bce, mask_dice,
    obj}`` (``inst`` = the weighted total). Per item without identity → a grad-connected zero (no supervision).

    ``ignore`` (B,D,H,W bool, from the surface label ``== 2``): confidence-/collapsed-sheet-masked voxels, excluded
    from the point-sampled negatives and from the objectness target — never supervised as background."""
    B = mask_features.shape[0]
    C = mask_features.shape[1]
    layers = instance["layers"]
    objectness = instance["objectness"]                               # (B,G_tok)
    device = mask_features.device
    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed))

    tot_class = mask_features.new_zeros(())
    tot_bce = mask_features.new_zeros(())
    tot_dice = mask_features.new_zeros(())
    tot_obj = mask_features.new_zeros(())
    n_ident = 0

    for b in range(B):
        if not bool(has_identity[b]):
            # grad-connected zero over EVERY instance output so the step is well-defined (DDP-safe).
            z = objectness[b].sum() * 0.0
            for lay in layers:
                z = z + lay["class_logits"][b].sum() * 0.0 + lay["mask_embed"][b].sum() * 0.0
            tot_class = tot_class + z
            continue
        n_ident += 1

        owner_flat = owner[b].reshape(-1)
        ign_b = (ignore[b] if ignore is not None else torch.zeros_like(owner[b], dtype=torch.bool)).bool()
        ignore_flat = ign_b.reshape(-1)
        valid_b = ~ign_b                                              # (D,H,W) non-ignore voxels (dense mask loss)
        ids = [int(i) for i in torch.unique(owner_flat).tolist() if i > 0]
        pts = _sample_points(owner_flat, ignore_flat, recipe.point_sample, rng)  # for the Hungarian COST only
        mf_flat = mask_features[b].reshape(C, -1)                     # (C, Nvox)
        feat_pts = mf_flat[:, pts]                                    # (C,P)
        owner_pts = owner_flat[pts]                                   # (P,)

        for li, lay in enumerate(layers):
            lay_b = {"class_logits": lay["class_logits"][b], "mask_embed": lay["mask_embed"][b]}
            is_final = (li == len(layers) - 1)
            out = _instance_item(lay_b, feat_pts, owner_pts, ids, mask_features[b], owner[b], valid_b, recipe)
            w = 1.0 if is_final else recipe.aux_weight
            tot_class = tot_class + w * out["class"]
            tot_bce = tot_bce + w * out["mask_bce"]
            tot_dice = tot_dice + w * out["mask_dice"]

        # token objectness: 16³ cell is positive iff a sheet voxel falls in it; cells that are >=50% ignore are
        # excluded from the objectness BCE (not forced to either class).
        side = round(objectness.shape[1] ** (1 / 3))
        k = owner[b].shape[-1] // side
        pool = (owner[b] > 0).float()[None, None]
        obj_target = (F.max_pool3d(pool, kernel_size=k, stride=k) > 0).reshape(-1).float()
        obj_valid = (F.avg_pool3d(ign_b.float()[None, None], kernel_size=k, stride=k) < 0.5).reshape(-1).float()
        obj_bce = F.binary_cross_entropy_with_logits(objectness[b], obj_target, reduction="none")
        tot_obj = tot_obj + (obj_bce * obj_valid).sum() / obj_valid.sum().clamp_min(1.0)

    denom = max(n_ident, 1)
    inst = (recipe.w_class * tot_class + recipe.w_mask_bce * tot_bce
            + recipe.w_mask_dice * tot_dice + recipe.w_objectness * tot_obj) / denom
    return {
        "inst": inst,
        "class": (tot_class / denom).detach(),
        "mask_bce": (tot_bce / denom).detach(),
        "mask_dice": (tot_dice / denom).detach(),
        "obj": (tot_obj / denom).detach(),
    }

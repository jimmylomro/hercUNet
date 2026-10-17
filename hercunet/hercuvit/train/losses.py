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

from ..model.instance_head import mask_logits_at_points


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
    tgt01 = (sem_target > 0).float().unsqueeze(1)                       # (B,1,D,H,W) for downsampling
    total = sem_logits[0].new_zeros(())
    acc = {k: sem_logits[0].new_zeros(()) for k in ("ce", "tversky", "skel", "sep")}
    for i, logits in enumerate(sem_logits):
        w = float(weights[i]) if i < len(weights) else 0.0
        side = logits.shape[-1]
        tl = (_downsample_target(tgt01, side).squeeze(1) > 0.5).long()  # (B,side,side,side) {0,1}
        terms = surface_terms(
            logits.float(), tl,
            fg_weight=recipe.sem_fg_weight,
            tversky_alpha=recipe.tversky_alpha, tversky_beta=recipe.tversky_beta, tversky_gamma=recipe.tversky_gamma,
            skel_iters=recipe.skel_iters, sep_radius=recipe.sep_radius, ignore_label=None,
        )
        total = total + w * (recipe.w_sem_ce * terms["ce"] + recipe.w_sem_tversky * terms["tversky"]
                             + recipe.w_sem_skel * terms["skel"] + recipe.w_sem_sep * terms["sep"])
        for k in acc:
            acc[k] = acc[k] + w * terms[k]
    return {"sem": total, "ce": acc["ce"].detach(), "tversky": acc["tversky"].detach(),
            "skel": acc["skel"].detach(), "sep": acc["sep"].detach(), "dice": acc["tversky"].detach()}


# --------------------------------------------------------------------------- instance helpers
def _sample_points(owner_flat: torch.Tensor, n: int, rng: torch.Generator) -> torch.Tensor:
    """``n`` flat voxel indices: ~half from sheet voxels (owner>0), the rest uniform. Returns (n,) long."""
    nvox = owner_flat.numel()
    pos = (owner_flat > 0).nonzero(as_tuple=False).squeeze(1)
    n_pos = min(n // 2, int(pos.numel()))
    out = []
    if n_pos > 0:
        pick = torch.randint(0, int(pos.numel()), (n_pos,), generator=rng, device=owner_flat.device)
        out.append(pos[pick])
    n_rand = n - n_pos
    if n_rand > 0:
        out.append(torch.randint(0, nvox, (n_rand,), generator=rng, device=owner_flat.device))
    return torch.cat(out, 0) if out else torch.zeros(0, dtype=torch.long, device=owner_flat.device)


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


def _instance_item(layer_out_b, feat_pts, owner_pts, ids, recipe, *, is_final: bool):
    """Matched set-prediction losses for ONE item at ONE decoder layer.

    ``layer_out_b`` = {class_logits (Q,2), mask_embed (Q,C), anchors (Q,3)} for this item;
    ``feat_pts`` (C,P) point-sampled mask features; ``owner_pts`` (P,) owner id per point; ``ids`` the GT ids.
    Returns ``{class, mask_bce, mask_dice}`` scalar tensors (grad-connected).
    """
    class_logits = layer_out_b["class_logits"]                         # (Q,2)
    mask_embed = layer_out_b["mask_embed"]                             # (Q,C)
    Q = class_logits.shape[0]
    device = class_logits.device

    pred_pts = mask_logits_at_points(mask_embed.unsqueeze(0), feat_pts.unsqueeze(0))[0]  # (Q,P)
    G = len(ids)
    if G == 0:
        cls_target = torch.zeros(Q, dtype=torch.long, device=device)   # all no-object
        cw = torch.tensor([recipe.no_object_weight, 1.0], device=device)
        loss_class = F.cross_entropy(class_logits, cls_target, weight=cw)
        zero = pred_pts.sum() * 0.0
        return {"class": loss_class, "mask_bce": zero, "mask_dice": zero}

    gt_pts = torch.stack([(owner_pts == i).float() for i in ids], 0)   # (G,P)
    with torch.no_grad():
        cls_prob = torch.softmax(class_logits, dim=-1)                 # (Q,2)
        cost_class = -cls_prob[:, 1].unsqueeze(1).expand(Q, G)
        cost_bce = _bce_cost(pred_pts, gt_pts)
        cost_dice = _dice_cost(torch.sigmoid(pred_pts), gt_pts)
        cost = (recipe.cost_class * cost_class + recipe.cost_mask_bce * cost_bce
                + recipe.cost_mask_dice * cost_dice)
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(cost.detach().float().cpu().numpy())
    rows_t = torch.as_tensor(rows, dtype=torch.long, device=device)
    cols_t = torch.as_tensor(cols, dtype=torch.long, device=device)

    cls_target = torch.zeros(Q, dtype=torch.long, device=device)
    cls_target[rows_t] = 1                                             # matched queries → sheet
    cw = torch.tensor([recipe.no_object_weight, 1.0], device=device)
    loss_class = F.cross_entropy(class_logits, cls_target, weight=cw)

    mp = pred_pts[rows_t]                                              # (M,P)
    mg = gt_pts[cols_t]                                                # (M,P)
    loss_bce = F.binary_cross_entropy_with_logits(mp, mg)
    prob = torch.sigmoid(mp)
    num = 2.0 * (prob * mg).sum(-1) + 1.0
    den = prob.sum(-1) + mg.sum(-1) + 1.0
    loss_dice = (1.0 - num / den).mean()
    return {"class": loss_class, "mask_bce": loss_bce, "mask_dice": loss_dice}


def instance_loss(instance: dict, mask_features: torch.Tensor, owner: torch.Tensor,
                  has_identity, recipe, *, seed: int = 0) -> dict:
    """Mask2Former set-prediction + objectness loss over the batch. Returns ``{inst, class, mask_bce, mask_dice,
    obj}`` (``inst`` = the weighted total). Per item without identity → a grad-connected zero (no supervision)."""
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
        ids = [int(i) for i in torch.unique(owner_flat).tolist() if i > 0]
        pts = _sample_points(owner_flat, recipe.point_sample, rng)
        mf_flat = mask_features[b].reshape(C, -1)                     # (C, Nvox)
        feat_pts = mf_flat[:, pts]                                    # (C,P)
        owner_pts = owner_flat[pts]                                   # (P,)

        for li, lay in enumerate(layers):
            lay_b = {"class_logits": lay["class_logits"][b], "mask_embed": lay["mask_embed"][b]}
            is_final = (li == len(layers) - 1)
            out = _instance_item(lay_b, feat_pts, owner_pts, ids, recipe, is_final=is_final)
            w = 1.0 if is_final else recipe.aux_weight
            tot_class = tot_class + w * out["class"]
            tot_bce = tot_bce + w * out["mask_bce"]
            tot_dice = tot_dice + w * out["mask_dice"]

        # token objectness: 16³ cell is positive iff a sheet voxel falls in it (max-pool owner>0 by patch=8).
        side = round(objectness.shape[1] ** (1 / 3))
        pool = (owner[b] > 0).float()[None, None]
        k = pool.shape[-1] // side
        obj_target = (F.max_pool3d(pool, kernel_size=k, stride=k) > 0).reshape(-1).float()
        tot_obj = tot_obj + F.binary_cross_entropy_with_logits(objectness[b], obj_target)

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

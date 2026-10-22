"""Validation metrics for HercuViT — per-sheet instance coverage (§13.5), the quantity that actually matters.

Mask-Dice can fall while the real failure (sheets left as no-object) persists, so validation reports **per-sheet
coverage**: for every GT sheet, the best-query IoU, how many sheets are captured (IoU over a threshold), and how
many of the queries fire. This is the §13.1 measurement, run on held-out windows and logged each val interval.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..model.instance_head import mask_logits_from_embed


@torch.no_grad()
def surface_metrics(net, loader, device, *, max_windows: int = 64) -> dict:
    """Surface-head validation (surface-only training): run ``net`` with the instance head OFF over up to
    ``max_windows`` windows and score the predicted thin surface (``argmax`` of the full-res surface logits == 1)
    against the GT thin medial (``sem==1``), ignoring ``sem==2``. Reports the author-style **foreground Dice**
    (``2·TP/(P+T)``, per-window mean) plus a **1-voxel-tolerant surface Dice** (F1 of ±1-dilated precision/recall)
    — fairer to 1-voxel medials where strict Dice is punished by sub-voxel misalignment. Returns
    ``{windows, fg_dice, sdice_tol1, precision, recall, nfg}`` (windows with any GT surface)."""
    was_training = net.training
    net.eval()
    dt = device.type if hasattr(device, "type") else "cuda"
    dices, tol1s, precs, recs, n_win = [], [], [], [], 0
    for b in loader:
        ct = b["ct"].to(device); sem = b["sem"].to(device)
        with torch.autocast(device_type=dt, dtype=torch.bfloat16):
            out = net(ct, dn_owner=None, run_instance=False)               # surface forward only (no instance head)
        pred = out["sem_logits"][0].float().argmax(1)                      # (B,D,H,W) argmax class
        for i in range(pred.shape[0]):
            valid = sem[i] != 2
            p = (pred[i] == 1) & valid                                     # predicted surface
            t = (sem[i] == 1) & valid                                      # GT thin medial
            ps, ts, tp = p.sum().float(), t.sum().float(), (p & t).sum().float()
            if float(ts) == 0:                                            # no GT surface here → Dice undefined
                continue
            dices.append(float(2 * tp / (ps + ts).clamp_min(1.0)))
            dil_t = F.max_pool3d(t[None, None].float(), 3, 1, 1)[0, 0] > 0.5   # ±1-voxel tolerance band
            dil_p = F.max_pool3d(p[None, None].float(), 3, 1, 1)[0, 0] > 0.5
            prec = float((p & dil_t).sum().float() / ps.clamp_min(1.0))
            rec = float((t & dil_p).sum().float() / ts.clamp_min(1.0))
            precs.append(prec); recs.append(rec)
            tol1s.append(2 * prec * rec / max(prec + rec, 1e-6))
        n_win += 1
        if n_win >= int(max_windows):
            break
    if was_training:
        net.train()
    if not dices:
        return {"windows": n_win, "fg_dice": 0.0, "sdice_tol1": 0.0, "precision": 0.0, "recall": 0.0, "nfg": 0}
    return {"windows": n_win, "fg_dice": float(np.mean(dices)), "sdice_tol1": float(np.mean(tol1s)),
            "precision": float(np.mean(precs)), "recall": float(np.mean(recs)), "nfg": len(dices)}


@torch.no_grad()
def coverage_metrics(net, loader, device, *, max_windows: int = 64, prob_thresh: float = 0.5,
                     fire_thresh: float = 0.5, mask_mode: str = "sigmoid") -> dict:
    """Run ``net`` in INFERENCE mode (no denoise group) over up to ``max_windows`` identity windows of ``loader`` and
    return per-sheet coverage. For each GT sheet (thin surface, ``owner & surf``): best-query IoU of the binarised
    predicted instance mask. ``mask_mode="sigmoid"`` → ``σ(mask_embed·mask_features) > prob_thresh`` (independent);
    ``"softmax"`` → ``argmax_q(mask_embed·mask_features) == q`` excluding the null query 0 (exclusive ownership).
    Returns ``{windows, sheets, iou_mean, iou_median, iou_p90, cap010, cap050, fire_mean, fire_min, fire_max}``
    (fractions in [0,1]); empty-safe."""
    was_training = net.training
    net.eval()
    best, fires, n_win = [], [], 0
    cap010 = cap050 = n_sheets = 0
    for b in loader:
        if not any(bool(x) for x in b["has_identity"]):
            continue
        ct = b["ct"].to(device); sem = b["sem"].to(device); owner = b["owner"].to(device)
        owner_thin = owner * (sem == 1).to(owner.dtype)
        ids = [int(i) for i in torch.unique(owner_thin[0]).tolist() if i > 0]
        if not ids:
            continue
        with torch.autocast(device_type=device.type if hasattr(device, "type") else "cuda", dtype=torch.bfloat16):
            out = net(ct, dn_owner=None)                                   # inference: no denoise queries
        lay = out["instance"]["layers"][-1]
        cls = torch.softmax(lay["class_logits"][0].float(), dim=-1)[:, 1]  # (Q,) P(sheet)
        mlog = mask_logits_from_embed(lay["mask_embed"][0:1].float(), out["mask_features"][0:1].float())[0]  # (Q,D,H,W)
        # masks live at mask_features resolution (1/4 in design C) — bring the GT to the same grid (nearest/max-pool)
        mside = out["mask_features"].shape[-1]
        if mside != owner_thin.shape[-1]:
            r = owner_thin.shape[-1] // mside
            owner_g = F.max_pool3d(owner_thin.float()[:, None], r, r)[:, 0].long()
            surf = F.max_pool3d((sem == 1).float()[:, None], r, r)[:, 0] > 0.5
        else:
            owner_g, surf = owner_thin, (sem == 1)
        ids = [int(i) for i in torch.unique(owner_g[0]).tolist() if i > 0]
        if mask_mode == "softmax":                                     # exclusive ownership: argmax query owns the voxel
            owner_q = mlog.argmax(0)                                   # (D,H,W)
            masks = torch.stack([(owner_q == q) & (q != 0) for q in range(mlog.shape[0])], 0)  # null (0) = background
        else:
            masks = torch.sigmoid(mlog) > prob_thresh                  # (Q,D,H,W) independent
        fires.append(int((masks.flatten(1).any(1)).sum()) if mask_mode == "softmax" else int((cls > fire_thresh).sum()))
        surf = surf[0]
        for sid in ids:
            gt = (owner_g[0] == sid) & surf
            inter = (masks & gt).sum(dim=(1, 2, 3)).float()
            union = (masks | gt).sum(dim=(1, 2, 3)).float().clamp_min(1.0)
            bi = float((inter / union).max())
            best.append(bi); n_sheets += 1
            cap010 += bi > 0.10; cap050 += bi > 0.50
        n_win += 1
        if n_win >= int(max_windows):
            break
    if was_training:
        net.train()
    if n_sheets == 0:
        return {"windows": n_win, "sheets": 0, "iou_mean": 0.0, "iou_median": 0.0, "iou_p90": 0.0,
                "cap010": 0.0, "cap050": 0.0, "fire_mean": 0.0, "fire_min": 0, "fire_max": 0}
    a = np.asarray(best)
    return {
        "windows": n_win, "sheets": n_sheets,
        "iou_mean": float(a.mean()), "iou_median": float(np.median(a)), "iou_p90": float(np.percentile(a, 90)),
        "cap010": cap010 / n_sheets, "cap050": cap050 / n_sheets,
        "fire_mean": float(np.mean(fires)), "fire_min": int(min(fires)), "fire_max": int(max(fires)),
    }

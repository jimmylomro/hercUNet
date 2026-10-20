"""Validation metrics for HercuViT — per-sheet instance coverage (§13.5), the quantity that actually matters.

Mask-Dice can fall while the real failure (sheets left as no-object) persists, so validation reports **per-sheet
coverage**: for every GT sheet, the best-query IoU, how many sheets are captured (IoU over a threshold), and how
many of the queries fire. This is the §13.1 measurement, run on held-out windows and logged each val interval.
"""
from __future__ import annotations

import numpy as np
import torch

from ..model.instance_head import mask_logits_from_embed


@torch.no_grad()
def coverage_metrics(net, loader, device, *, max_windows: int = 64, prob_thresh: float = 0.5,
                     fire_thresh: float = 0.5) -> dict:
    """Run ``net`` in INFERENCE mode (no denoise group) over up to ``max_windows`` identity windows of ``loader`` and
    return per-sheet coverage. For each GT sheet (thin surface, ``owner & surf``): best-query IoU of the binarised
    predicted mask ``σ(mask_embed·mask_features) > prob_thresh``. Returns ``{windows, sheets, iou_mean, iou_median,
    iou_p90, cap010, cap050, fire_mean, fire_min, fire_max}`` (fractions in [0,1]); empty-safe."""
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
        masks = torch.sigmoid(mask_logits_from_embed(lay["mask_embed"][0:1].float(),
                                                     out["mask_features"][0:1].float())[0]) > prob_thresh  # (Q,D,H,W)
        fires.append(int((cls > fire_thresh).sum()))
        surf = (sem[0] == 1)
        for sid in ids:
            gt = (owner_thin[0] == sid) & surf
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

"""Surface-head confidence probe: is the decoupled surface/objectness head under-confident?

Over N held-out windows it loads a trained HercuViT checkpoint, reads the soft surface probability
``P(surface) = softmax(surface_logits)[:, 1]`` at 1/1, and reports (i) the distribution of P on true medial
voxels vs. background, and (ii) precision / recall / Dice of ``P > τ`` against the thin GT medial over a
threshold sweep — so we can tell an *under-confident but correct* surface (mass at 0.3–0.5, best-Dice τ well
below 0.5) from a *genuinely missed* surface (mass near 0). Ignore (label 2) is excluded everywhere.

Run::

    python -m hercunet.hercuvit.probes.surface_confidence --ckpt <run>/checkpoint_latest.pth --nwin 40
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description="Surface-head confidence / calibration probe.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--nwin", type=int, default=40)
    ap.add_argument("--cache", default="/root/train_nano/train_cache")
    ap.add_argument("--recipe", default="hercunet/hercuvit/recipes/full_decoupled.yml")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)

    from .. import netbuild
    from ..train import recipe as R
    from ..train.chain import resolve_encoder
    from ..data.dataset import LocalHercuvitDataset, split_cache
    rec = R.load(args.recipe)
    enc = resolve_encoder(encoder_hf="scrollprize/dinovol_ps8_9.6um_101626")
    net = netbuild.build(enc, device=device, arch_override=rec.encoder_arch(), global_blocks=rec.global_blocks,
        window_size=rec.window_size, n_sem_classes=rec.n_sem_classes, mask_dim=rec.mask_dim, n_queries=rec.n_queries,
        query_dim=rec.query_dim, instance_layers=rec.instance_layers, deep_supervision=rec.deep_supervision,
        with_instance=False, add_null_query=False, learned_queries=rec.learned_queries, fps_queries=rec.fps_queries,
        cross_attn=rec.cross_attn, decoder=rec.decoder, neck=rec.neck, neck_layers=rec.neck_layers,
        neck_heads=rec.neck_heads, neck_points=rec.neck_points, surface_gate=rec.surface_gate)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False); sd = ck["model"]; cur = net.state_dict()
    sd = {k: v for k, v in sd.items() if k in cur and tuple(cur[k].shape) == tuple(v.shape)}
    net.load_state_dict(sd, strict=False); net.eval()
    print(f"loaded {len(sd)} tensors @ ep {ck.get('epoch')}", flush=True)

    tr, va = split_cache(args.cache, fold=rec.val_fold)
    ds = LocalHercuvitDataset(args.cache, augment=False, target_voxel_um=rec.target_voxel_um, crop=rec.window,
                              files=va[:args.nwin])
    import torch.nn.functional as F
    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    tp = {t: 0.0 for t in taus}; fp = {t: 0.0 for t in taus}; fn = {t: 0.0 for t in taus}
    tol = {t: {"tp_p": 0.0, "np": 0.0, "tp_r": 0.0, "ng": 0.0} for t in (0.4, 0.5)}   # ±1-voxel tolerant P/R
    p_on_gt, p_on_bg = [], []

    def dil(x):  # ±1-voxel dilation
        return F.max_pool3d(torch.as_tensor(x).float()[None, None], 3, 1, 1)[0, 0].numpy() > 0.5
    for wi in range(len(ds)):
        it = ds[wi]; sem = torch.as_tensor(it["sem"])
        if int((sem == 1).sum()) < 200:
            continue
        ct = torch.as_tensor(it["ct"]); ctb = (ct.unsqueeze(0) if ct.ndim == 4 else ct.unsqueeze(0).unsqueeze(0)).float().to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            out = net(ctb, dn_owner=None, run_instance=False)
        p = torch.softmax(out["sem_logits"][0][0].float(), dim=0)[1].cpu().numpy()   # P(surface) @1/1
        del ctb
        if device.type == "cuda":
            torch.cuda.empty_cache()
        s = sem.numpy(); gt = (s == 1); valid = (s != 2); bg = (s == 0)
        p_on_gt.append(p[gt]); p_on_bg.append(p[bg][::17])
        for t in taus:
            pr = (p > t) & valid
            tp[t] += float((pr & gt).sum()); fp[t] += float((pr & ~gt & valid).sum()); fn[t] += float((~pr & gt).sum())
        for t in tol:                                              # ±1-voxel tolerant precision/recall
            pr = (p > t) & valid; dgt = dil(gt) & valid; dpr = dil(pr) & valid
            tol[t]["tp_p"] += float((pr & dgt).sum()); tol[t]["np"] += float(pr.sum())
            tol[t]["tp_r"] += float((gt & dpr).sum()); tol[t]["ng"] += float(gt.sum())
    p_on_gt = np.concatenate(p_on_gt); p_on_bg = np.concatenate(p_on_bg)
    res = {"epoch": int(ck.get("epoch", -1)), "n_windows": len(ds),
           "P_on_GTsurface_mean": float(p_on_gt.mean()), "P_on_GTsurface_median": float(np.median(p_on_gt)),
           "P_on_GTsurface_p10": float(np.percentile(p_on_gt, 10)), "P_on_GTsurface_p90": float(np.percentile(p_on_gt, 90)),
           "P_on_bg_mean": float(p_on_bg.mean()),
           "frac_GTsurface_below_0.5": float((p_on_gt < 0.5).mean()),
           "sweep": {}}
    best = (0.0, 0.5)
    for t in taus:
        prec = tp[t] / max(tp[t] + fp[t], 1); rec_ = tp[t] / max(tp[t] + fn[t], 1)
        dice = 2 * tp[t] / max(2 * tp[t] + fp[t] + fn[t], 1)
        res["sweep"][str(t)] = {"precision": round(prec, 3), "recall": round(rec_, 3), "dice": round(dice, 3)}
        if dice > best[0]:
            best = (dice, t)
    res["best_dice"] = round(best[0], 3); res["best_dice_tau"] = best[1]
    res["tolerant_pm1"] = {}
    for t in tol:
        pr = tol[t]["tp_p"] / max(tol[t]["np"], 1); rc = tol[t]["tp_r"] / max(tol[t]["ng"], 1)
        res["tolerant_pm1"][str(t)] = {"precision": round(pr, 3), "recall": round(rc, 3),
                                       "dice": round(2 * pr * rc / max(pr + rc, 1e-6), 3)}
    print(json.dumps(res, indent=2), flush=True)
    print(f"SUMMARY ep{res['epoch']}: P(surf) on GT medial mean={res['P_on_GTsurface_mean']:.2f} "
          f"median={res['P_on_GTsurface_median']:.2f}, {res['frac_GTsurface_below_0.5']*100:.0f}% below 0.5 | "
          f"strict best Dice={res['best_dice']} @τ{res['best_dice_tau']} | "
          f"tolerant±1 @0.5 Dice={res['tolerant_pm1']['0.5']['dice']} "
          f"(P={res['tolerant_pm1']['0.5']['precision']} R={res['tolerant_pm1']['0.5']['recall']})", flush=True)


if __name__ == "__main__":
    main()

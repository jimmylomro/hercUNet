"""Mask-feature (decoder U1) representation-geometry probe — the VOXEL-SCALE analog of
:mod:`hercunet.hercuvit.probes.frozen_encoder`.

The pixel decoder's final output ``U1`` (@1/1, ``mask_dim``=32) is a LEARNED UPSAMPLE of the encoder's 1/8 tokens.
Where the frozen-encoder probe measures the sheet primitive at the coarse 76.8 µm token grid (and says, explicitly,
that the grid inflates every number and the decoder must still sharpen it to the voxel medial), THIS probe measures
the SAME four things on ``U1`` of a TRAINED network, at **per-voxel 9.6 µm resolution** — so it quantifies how much
the decoder sharpens the coarse-token signal:

  (A) linear decodability of sheet vs background on U1 (held-out Fisher-LDA AUC/Dice + per-window prototype Dice);
  (B) intrinsic dimensionality of the 32-d U1 (participation ratio / effective rank, and held-out fg/bg AUC vs #dims);
  (C) sheet identity (within- vs between-sheet cosine on U1, and its ranking AUC + dimensionality);
  (D) U1 norms (directional vs magnitude).

It reuses the measurement + figure code from ``frozen_encoder`` unchanged (``analyse`` is feature-matrix-agnostic;
``figures`` takes label overrides); only the feature/label EXTRACTION differs — the full trained net's U1 at 1/1 with
PER-VOXEL labels (no 8³ pooling). Default encoder = the 288 teacher (run B, unfrozen); pass ``--encoder-hf`` /
``--encoder-file`` for the 480.

Run::

    python -m hercunet.hercuvit.probes.mask_features --ckpt /root/train_v2_unfrozen/checkpoint_latest.pth \
        --out /root/mf_stats --cache /root/train_nano/train_cache \
        --recipe hercunet/hercuvit/recipes/full_decoupled.yml
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from . import stats as st
from .frozen_encoder import analyse, figures   # reused unchanged (analyse is feature-agnostic; figures takes labels)

_ENC_288 = "scrollprize/dinovol_ps8_9.6um_101626"


def build_net(recipe_path: str, ckpt_path: str, device: torch.device, encoder_hf: str, encoder_file=None):
    """Build the FULL decoupled net (``with_instance=False`` — we only read U1 = ``mask_features``) and load the
    trained checkpoint (shape-filtered). Returns ``(net, recipe, epoch)``."""
    from .. import netbuild
    from ..train import recipe as R
    from ..train.chain import resolve_encoder
    rec = R.load(recipe_path)
    ckpt = resolve_encoder(encoder_hf=encoder_hf, hf_file=encoder_file)
    net = netbuild.build(
        ckpt, device=device, arch_override=rec.encoder_arch(), global_blocks=rec.global_blocks,
        window_size=rec.window_size, n_sem_classes=rec.n_sem_classes, mask_dim=rec.mask_dim, n_queries=rec.n_queries,
        query_dim=rec.query_dim, instance_layers=rec.instance_layers, deep_supervision=rec.deep_supervision,
        with_instance=False, add_null_query=rec.add_null_query, learned_queries=rec.learned_queries,
        fps_queries=rec.fps_queries, cross_attn=rec.cross_attn, decoder=rec.decoder, neck=rec.neck,
        neck_layers=rec.neck_layers, neck_heads=rec.neck_heads, neck_points=rec.neck_points,
        surface_gate=rec.surface_gate)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model"]; cur = net.state_dict()
    sd = {k: v for k, v in sd.items() if not (k in cur and tuple(cur[k].shape) != tuple(v.shape))}
    m, u = net.load_state_dict(sd, strict=False)
    net.eval()
    print(f"loaded trained net @ ep {ck.get('epoch')}  (mask_dim={rec.mask_dim}; missing={len(m)} unexpected={len(u)})",
          flush=True)
    return net, rec, ck.get("epoch")


def extract(net, rec, cache: str, nwin: int, cap: int, device: torch.device, seed: int = 0, per_win: int = 3000):
    """Iterate windows, forward the net, take U1 = ``mask_features`` (@1/1, mask_dim) and accumulate PER-VOXEL
    sheet/background features (sub-sampled per window), their owner + window ids, norms, and the per-window prototype
    separability — the same dict shape :func:`frozen_encoder.analyse` consumes (``patch=1`` ⇒ 9.6 µm voxel scale)."""
    from ..data.dataset import LocalHercuvitDataset, split_cache
    rng = np.random.default_rng(seed)
    tr, va = split_cache(cache, fold=rec.val_fold)
    files = (va + tr)[:nwin]
    ds = LocalHercuvitDataset(cache, augment=False, target_voxel_um=rec.target_voxel_um, crop=rec.window, files=files)
    S, B, OWN, WID, SID, pw_dice, pw_auc, norm_s, norm_b = [], [], [], [], [], [], [], [], []
    for wi in range(len(ds)):
        it = ds[wi]
        sem = np.asarray(it["sem"]).reshape(-1); owner = np.asarray(it["owner"]).reshape(-1)
        if int((sem == 1).sum()) < 200:
            continue
        ct = torch.as_tensor(it["ct"])
        ctb = (ct.unsqueeze(0) if ct.ndim == 4 else ct.unsqueeze(0).unsqueeze(0)).float().to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            u1 = net(ctb, run_instance=False)["mask_features"][0].float()          # (C, D,H,W) @1/1
        C = u1.shape[0]
        u1f = u1.reshape(C, -1)                                                    # (C, N) on device
        sheet_i = np.where(sem == 1)[0]; bg_i = np.where(sem == 0)[0]              # per-voxel (no pooling); ignore sem==2
        if len(sheet_i) > per_win:
            sheet_i = rng.choice(sheet_i, per_win, replace=False)
        if len(bg_i) > per_win:
            bg_i = rng.choice(bg_i, per_win, replace=False)
        sv = u1f[:, torch.as_tensor(sheet_i, device=device)].T.cpu().numpy()      # (n_sheet, C)
        bv = u1f[:, torch.as_tensor(bg_i, device=device)].T.cpu().numpy()         # (n_bg, C)
        del ctb, u1, u1f
        if device.type == "cuda":
            torch.cuda.empty_cache()
        ow = owner[sheet_i]                                                        # per-voxel sheet owner (0 = none)
        e = st.unit(np.concatenate([sv, bv]))                                     # per-window prototype linear probe
        es, eb = e[:len(sv)], e[len(sv):]
        if len(sv) >= 5 and len(bv) >= 20:
            proto = es.mean(0); proto /= (np.linalg.norm(proto) + 1e-9)
            cos = np.concatenate([es @ proto, eb @ proto])
            lab = np.concatenate([np.ones(len(sv), bool), np.zeros(len(bv), bool)])
            d, _ = st.best_dice(cos, lab); pw_dice.append(d)
            pw_auc.append(st.auc(es @ proto, eb @ proto))
        norm_s.append(np.linalg.norm(sv, axis=1)); norm_b.append(np.linalg.norm(bv, axis=1))
        S.append(sv); WID += [wi] * len(sv); OWN += ow.tolist(); SID += [wi] * len(sv); B.append(bv)
        if wi % 25 == 0:
            print(f"  win {wi}: sheet_vox={len(sv)} bg_vox={len(bv)}", flush=True)
    S = np.concatenate(S).astype(np.float32); B = np.concatenate(B).astype(np.float32)
    OWN = np.array(OWN); WID = np.array(WID); SID = np.array(SID)
    norm_s = np.concatenate(norm_s); norm_b = np.concatenate(norm_b)
    if len(S) > cap:
        k = rng.choice(len(S), cap, replace=False); S, OWN, WID, SID = S[k], OWN[k], WID[k], SID[k]
    if len(B) > cap:
        B = B[rng.choice(len(B), cap, replace=False)]
    print(f"collected sheet_vox={len(S)} bg_vox={len(B)} (windows used={len(pw_dice)})", flush=True)
    return dict(S=S, B=B, OWN=OWN, WID=WID, SID=SID, norm_s=norm_s, norm_b=norm_b,
                pw_dice=pw_dice, pw_auc=pw_auc, n_windows=len(ds), patch=1)


def main():
    ap = argparse.ArgumentParser(description="Trained-decoder U1 (mask-feature) representation-geometry probe.")
    ap.add_argument("--ckpt", required=True, help="trained HercuViT checkpoint (e.g. run B unfrozen)")
    ap.add_argument("--nwin", type=int, default=250)
    ap.add_argument("--cap", type=int, default=120000, help="per-class voxel cap")
    ap.add_argument("--per-win", dest="per_win", type=int, default=3000, help="voxels sampled per class per window")
    ap.add_argument("--out", default="mf_stats")
    ap.add_argument("--cache", default="/root/train_nano/train_cache")
    ap.add_argument("--recipe", default="hercunet/hercuvit/recipes/full_decoupled.yml")
    ap.add_argument("--encoder-hf", dest="encoder_hf", default=_ENC_288)
    ap.add_argument("--encoder-file", dest="encoder_file", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    net, rec, epoch = build_net(args.recipe, args.ckpt, device, args.encoder_hf, args.encoder_file)
    data = extract(net, rec, args.cache, args.nwin, args.cap, device, seed=args.seed, per_win=args.per_win)
    np.savez_compressed(f"{args.out}/voxels.npz", S=data["S"], B=data["B"], OWN=data["OWN"],
                        WID=data["WID"], SID=data["SID"])          # cache arrays so re-analysis needs no GPU
    res, aux = analyse(data, seed=args.seed)
    res["checkpoint_epoch"] = epoch
    json.dump(res, open(f"{args.out}/stats.json", "w"), indent=2)
    print(json.dumps(res, indent=2), flush=True)
    figures(res, aux, args.out, prefix="maskfeat", feat_label="trained decoder (U1 @1/1)", unit="voxel")
    print("FIGS_DONE", flush=True)


if __name__ == "__main__":
    main()

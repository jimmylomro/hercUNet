"""Frozen-encoder representation-geometry probe for the HercuViT submission.

Runs the FROZEN Dinovol encoder over many held-out windows, labels each 1/8 token (patch 8) as sheet-surface /
background / ignore, and measures: (A) linear decodability of sheet vs background (per-window prototype Dice, a
held-out Fisher-LDA probe); (B) intrinsic dimensionality (participation ratio / effective rank of the on-sheet
vs background clouds, and held-out fg/bg AUC as a function of #PCA directions); (C) sheet identity (within- vs
between-sheet cosine, and its ranking AUC); (D) embedding norms. Writes ``stats.json`` + paper JPG figures.

Run::

    python -m hercunet.hercuvit.probes.frozen_encoder --nwin 250 --out /root/enc_stats \
        --cache /root/train_nano/train_cache --recipe hercunet/hercuvit/recipes/full_decoupled.yml
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from . import stats as st


def build_encoder(recipe_path: str, device: torch.device):
    """Build the frozen encoder from the recipe's Dinovol spec; returns ``(backbone, recipe)``."""
    from .. import netbuild
    from ..train import recipe as R
    from ..train.chain import resolve_encoder
    rec = R.load(recipe_path)
    ckpt = resolve_encoder(encoder_hf="scrollprize/dinovol_ps8_9.6um_101626")
    net = netbuild.build(
        ckpt, device=device, arch_override=rec.encoder_arch(), global_blocks=rec.global_blocks,
        window_size=rec.window_size, n_sem_classes=2, mask_dim=rec.mask_dim, n_queries=rec.n_queries,
        query_dim=rec.query_dim, instance_layers=rec.instance_layers, deep_supervision=False, with_instance=False,
        add_null_query=False, learned_queries=rec.learned_queries, fps_queries=rec.fps_queries,
        cross_attn=rec.cross_attn, decoder=rec.decoder, neck=rec.neck, neck_layers=rec.neck_layers,
        neck_heads=rec.neck_heads, neck_points=rec.neck_points, surface_gate=rec.surface_gate)
    net.eval()
    return net.backbone, rec


def extract(backbone, rec, cache: str, nwin: int, cap: int, device: torch.device, seed: int = 0):
    """Iterate windows, pool token labels, accumulate on-sheet/background token embeddings (bg subsampled) and
    per-window prototype separability. Returns a dict of arrays + per-window lists."""
    from ..data.dataset import LocalHercuvitDataset, split_cache
    rng = np.random.default_rng(seed)
    tr, va = split_cache(cache, fold=rec.val_fold)
    files = (va + tr)[:nwin]
    ds = LocalHercuvitDataset(cache, augment=False, target_voxel_um=rec.target_voxel_um, crop=rec.window, files=files)
    patch = int(backbone.patch_size[0]) if hasattr(backbone, "patch_size") else 8
    S, B, OWN, WID, SID, pw_dice, pw_auc, norm_s, norm_b = [], [], [], [], [], [], [], [], []
    for wi in range(len(ds)):
        it = ds[wi]
        sem = np.asarray(it["sem"]); owner = np.asarray(it["owner"])
        if int((sem == 1).sum()) < 200:
            continue
        ct = torch.as_tensor(it["ct"])
        ctb = (ct.unsqueeze(0) if ct.ndim == 4 else ct.unsqueeze(0).unsqueeze(0)).float().to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            tok = backbone.forward_features(ctb, view_kind="global")["x_norm_patchtokens"][0].float().cpu().numpy()
        del ctb
        if device.type == "cuda":
            torch.cuda.empty_cache()
        is_sheet, is_bg, owner_tok = st.token_labels(sem, owner, patch=patch)
        e = st.unit(tok)
        if is_sheet.sum() >= 5 and is_bg.sum() >= 20:                      # per-window prototype linear probe
            proto = e[is_sheet].mean(0); proto /= (np.linalg.norm(proto) + 1e-9)
            cos = e @ proto; m = is_sheet | is_bg
            d, _ = st.best_dice(cos[m], is_sheet[m]); pw_dice.append(d)
            pw_auc.append(st.auc(cos[is_sheet], cos[is_bg]))
        norm_s.append(np.linalg.norm(tok[is_sheet], axis=1)); norm_b.append(np.linalg.norm(tok[is_bg], axis=1))
        si = np.where(is_sheet)[0]; bi = np.where(is_bg)[0]
        if len(bi) > 400:
            bi = rng.choice(bi, 400, replace=False)
        S.append(tok[si]); WID += [wi] * len(si); OWN += owner_tok[si].tolist(); SID += [wi] * len(si)
        B.append(tok[bi])
        if wi % 25 == 0:
            print(f"  win {wi}: sheet_tok={int(is_sheet.sum())} bg_tok={int(is_bg.sum())}", flush=True)
    S = np.concatenate(S).astype(np.float32); B = np.concatenate(B).astype(np.float32)
    OWN = np.array(OWN); WID = np.array(WID); SID = np.array(SID)
    norm_s = np.concatenate(norm_s); norm_b = np.concatenate(norm_b)
    if len(S) > cap:
        k = rng.choice(len(S), cap, replace=False); S, OWN, WID, SID = S[k], OWN[k], WID[k], SID[k]
    if len(B) > cap:
        B = B[rng.choice(len(B), cap, replace=False)]
    print(f"collected sheet_tok={len(S)} bg_tok={len(B)} (windows used={len(pw_dice)})", flush=True)
    return dict(S=S, B=B, OWN=OWN, WID=WID, SID=SID, norm_s=norm_s, norm_b=norm_b,
                pw_dice=pw_dice, pw_auc=pw_auc, n_windows=len(ds), patch=patch)


def analyse(d: dict, seed: int = 0) -> dict:
    """All scalar statistics from the extracted arrays."""
    rng = np.random.default_rng(seed)
    S, B, OWN, WID, SID = d["S"], d["B"], d["OWN"], d["WID"], d["SID"]
    res = {"n_windows": d["n_windows"], "embed_dim": int(S.shape[1]), "n_sheet_tok": int(len(S)),
           "n_bg_tok": int(len(B)), "patch_um": round(9.6 * d["patch"], 1),
           "pw_proto_dice_mean": float(np.mean(d["pw_dice"])), "pw_proto_dice_median": float(np.median(d["pw_dice"])),
           "pw_proto_auc_mean": float(np.mean(d["pw_auc"])),
           "norm_sheet_mean": float(d["norm_s"].mean()), "norm_bg_mean": float(d["norm_b"].mean()),
           "norm_sheet_std": float(d["norm_s"].std()), "norm_bg_std": float(d["norm_b"].std())}

    wins = np.unique(WID); rng.shuffle(wins); trw = set(wins[: int(0.7 * len(wins))].tolist())
    Str = st.unit(S[np.array([w in trw for w in WID])]); Ste = st.unit(S[np.array([w not in trw for w in WID])])
    bsplit = rng.random(len(B)) < 0.7; Btr, Bte = st.unit(B[bsplit]), st.unit(B[~bsplit])
    w = st.lda_dir(Str, Btr)
    res["lda_fgbg_auc_heldout"] = st.auc(Ste @ w, Bte @ w)
    res["lda_fgbg_auc_train"] = st.auc(Str @ w, Btr @ w)
    res["lda_fgbg_dice_heldout"], _ = st.best_dice(np.concatenate([Ste @ w, Bte @ w]),
                                                   np.concatenate([np.ones(len(Ste)), np.zeros(len(Bte))]))
    eig_s, pr_s, er_s = st.spectrum(S); eig_b, pr_b, er_b = st.spectrum(B)
    _, pr_all, er_all = st.spectrum(np.concatenate([S, B]))
    res.update({"pr_sheet": pr_s, "pr_bg": pr_b, "pr_all": pr_all,
                "erank_sheet": er_s, "erank_bg": er_b, "erank_all": er_all})
    mu = np.concatenate([Str, Btr]).mean(0); _, _, Vt = np.linalg.svd(np.concatenate([Str, Btr]) - mu, full_matrices=False)
    ks = [1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 288]
    auc_k = []
    for k in ks:
        Pk = Vt[:k]
        wk = st.lda_dir((Str - mu) @ Pk.T, (Btr - mu) @ Pk.T)
        auc_k.append(st.auc(((Ste - mu) @ Pk.T) @ wk, ((Bte - mu) @ Pk.T) @ wk))
    res["auc_vs_k"] = {"k": ks, "auc": [float(a) for a in auc_k]}
    ceil = auc_k[-1] - 0.5
    for frac, key in [(0.90, "dims_for_90pct_fgbg"), (0.99, "dims_for_99pct_fgbg")]:
        res[key] = int(next((ks[i] for i, a in enumerate(auc_k) if (a - 0.5) >= frac * ceil), ks[-1]))
    # identity: within vs between sheet cosine + IDENTITY DIMENSIONALITY (same/diff-sheet AUC vs #PCA dims).
    # Pairs are sampled WITHIN a window (same illumination/context) and efficiently (pick b from a's window), so the
    # same-window constraint never bottlenecks the sampler.
    Su = st.unit(S); gid = SID.astype(np.int64) * 100000 + np.clip(OWN, 0, None)
    from collections import defaultdict
    by_win = defaultdict(list)
    for i in np.where(OWN > 0)[0]:
        by_win[int(SID[i])].append(i)
    by_win = {w: np.array(v) for w, v in by_win.items() if len(v) >= 4}
    wkeys = list(by_win.keys())
    A, Bp, lab = [], [], []
    while len(A) < 80000 and wkeys:
        toks = by_win[wkeys[rng.integers(len(wkeys))]]
        a = int(toks[rng.integers(len(toks))]); b = int(toks[rng.integers(len(toks))])
        if a == b:
            continue
        A.append(a); Bp.append(b); lab.append(bool(gid[a] == gid[b]))
    A = np.array(A); Bp = np.array(Bp); lab = np.array(lab)
    cfull = (Su[A] * Su[Bp]).sum(1)
    res["within_sheet_cos_mean"] = float(cfull[lab].mean()); res["between_sheet_cos_mean"] = float(cfull[~lab].mean())
    res["identity_cos_auc"] = st.auc(cfull[lab], cfull[~lab])
    res["n_identity_pairs"] = int(len(A)); res["frac_same_sheet_pairs"] = float(lab.mean())
    muS = S.mean(0); _, _, VtS = np.linalg.svd(S - muS, full_matrices=False)       # PCA of the sheet cloud
    ks_id = [1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 288]
    auc_id = []
    for k in ks_id:
        Sk = st.unit((S - muS) @ VtS[:k].T)
        c = (Sk[A] * Sk[Bp]).sum(1)
        auc_id.append(st.auc(c[lab], c[~lab]))
    res["identity_auc_vs_k"] = {"k": ks_id, "auc": [float(a) for a in auc_id]}
    ceil_id = auc_id[-1] - 0.5
    for frac, key in [(0.90, "dims_for_90pct_identity"), (0.99, "dims_for_99pct_identity")]:
        res[key] = int(next((ks_id[i] for i, a in enumerate(auc_id) if (a - 0.5) >= frac * ceil_id), ks_id[-1]))
    return res, dict(Ste=Ste, Bte=Bte, w=w, eig_s=eig_s, eig_b=eig_b, pr_s=pr_s, pr_b=pr_b, er_s=er_s, er_b=er_b,
                     ks=ks, auc_k=auc_k, ks_id=ks_id, auc_id=auc_id,
                     wi_cos=cfull[lab].tolist(), bt_cos=cfull[~lab].tolist())


def figures(res: dict, aux: dict, out: str):
    """Write the four paper JPGs (small, publication-ready)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25})
    jpg = {"pil_kwargs": {"quality": 90, "optimize": True}}
    C_S, C_B = "#c0392b", "#2c6fbb"

    sp, bn = aux["Ste"] @ aux["w"], aux["Bte"] @ aux["w"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].hist(bn, bins=60, color=C_B, alpha=0.65, density=True, label="background")
    ax[0].hist(sp, bins=60, color=C_S, alpha=0.65, density=True, label="sheet surface")
    ax[0].set_title(f"frozen-encoder sheet/bg (held-out)\nLDA AUC={res['lda_fgbg_auc_heldout']:.3f}  Dice={res['lda_fgbg_dice_heldout']:.3f}")
    ax[0].set_xlabel("linear projection (held-out windows)"); ax[0].set_ylabel("density"); ax[0].legend()
    alls = np.concatenate([sp, bn]); yb = np.concatenate([np.ones(len(sp)), np.zeros(len(bn))])
    order = np.argsort(-alls); tp = np.cumsum(yb[order]); fp = np.cumsum(1 - yb[order])
    ax[1].plot(alls[order], 2 * tp / (tp + fp + yb.sum() + 1e-9), color="#555")
    ax[1].set_title("Dice vs decision threshold"); ax[1].set_xlabel("threshold"); ax[1].set_ylabel("Dice"); ax[1].set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(f"{out}/frozen-fgbg-separability.jpg", **jpg); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.semilogx(aux["ks"], aux["auc_k"], "o-", color="#8e44ad",
                label=f"sheet vs. background  (90% @ {res['dims_for_90pct_fgbg']} dims)")
    ax.semilogx(aux["ks_id"], aux["auc_id"], "s-", color="#c0392b",
                label=f"sheet identity  (90% @ {res['dims_for_90pct_identity']} dims)")
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_xlabel("# PCA directions"); ax.set_ylabel("held-out AUC")
    ax.set_title("how many directions carry the signal"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}/frozen-dims-fgbg.jpg", **jpg); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.semilogy(aux["eig_s"] / aux["eig_s"].sum(), color=C_S, label=f"on-sheet  (PR={aux['pr_s']:.1f}, erank={aux['er_s']:.1f})")
    ax.semilogy(aux["eig_b"] / aux["eig_b"].sum(), color=C_B, label=f"background (PR={aux['pr_b']:.1f}, erank={aux['er_b']:.1f})")
    ax.set_xlabel("principal component"); ax.set_ylabel("normalised variance")
    ax.set_title("token embedding spectra (288-d)"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{out}/frozen-spectra.jpg", **jpg); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.hist(aux["bt_cos"], bins=60, color=C_B, alpha=0.65, density=True, label="different sheets")
    ax.hist(aux["wi_cos"], bins=60, color=C_S, alpha=0.65, density=True, label="same sheet")
    ax.set_title(f"sheet identity in frozen features\nwithin={res['within_sheet_cos_mean']:.2f} between={res['between_sheet_cos_mean']:.2f} AUC={res['identity_cos_auc']:.3f}")
    ax.set_xlabel("cosine similarity (token pairs, same window)"); ax.set_ylabel("density"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{out}/frozen-identity.jpg", **jpg); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Frozen Dinovol encoder representation-geometry probe.")
    ap.add_argument("--nwin", type=int, default=250)
    ap.add_argument("--cap", type=int, default=120000, help="per-class token cap")
    ap.add_argument("--out", default="enc_stats")
    ap.add_argument("--cache", default="/root/train_nano/train_cache")
    ap.add_argument("--recipe", default="hercunet/hercuvit/recipes/full_decoupled.yml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    backbone, rec = build_encoder(args.recipe, device)
    print(f"encoder embed_dim={backbone.embed_dim}", flush=True)
    data = extract(backbone, rec, args.cache, args.nwin, args.cap, device, seed=args.seed)
    np.savez_compressed(f"{args.out}/tokens.npz", S=data["S"], B=data["B"], OWN=data["OWN"],
                        WID=data["WID"], SID=data["SID"])      # cache the arrays so re-analysis needs no GPU
    res, aux = analyse(data, seed=args.seed)
    json.dump(res, open(f"{args.out}/stats.json", "w"), indent=2)
    print(json.dumps(res, indent=2), flush=True)
    figures(res, aux, args.out)
    print("FIGS_DONE", flush=True)


if __name__ == "__main__":
    main()

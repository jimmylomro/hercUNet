"""Per-window panel render — the 10-panel CT + representation-geometry + prediction visualiser for a trained
decoupled HercuViT (no null query, masks at 1/1).

Per z-slice, 10 panels:
  CT | encoder {PCA, sheet-LDA, fg/bg-LDA} | mask-feature U1 {PCA, sheet-LDA, fg/bg-LDA} | surface prediction |
  predicted sheets | GT sheets.

PCA panels = top-3 max-variance directions (RGB). sheet-LDA = the 3 Fisher directions that best SEPARATE the GT
sheets among the foreground (so sheets fill the colour gamut). fg/bg-LDA = the single Fisher direction separating
surface from background (coolwarm). "predicted sheets" = argmax-query ownership gated by the surface head
(``sigmoid(surf)>thresh``) — the decoupled inference. The companion of :mod:`frozen_encoder` / :mod:`mask_features`:
those quantify the geometry as scalars; this shows it per window.

Run::

    python -m hercunet.hercuvit.probes.render_panels --ckpt .../checkpoint_latest.pth --out renders \
        --cache /root/train_nano/train_cache --recipe hercunet/hercuvit/recipes/full_decoupled.yml --thresh 0.4 --k 10
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.linalg import eigh as _eigh

_ENC_288 = "scrollprize/dinovol_ps8_9.6um_101626"

# Maximally-distinct categorical colours (saturated; distinct on a grey CT background). Masks are coloured by RANK
# among the ids PRESENT in a window (not id%N), so the few sheets in a window take the first, most-separated entries.
_PALETTE = np.array([mcolors.to_rgb(c) for c in [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6",
    "#bfef45", "#469990", "#dcbeff", "#9A6324", "#800000", "#808000", "#000075", "#fabed4",
    "#00ff7f", "#ff1493", "#00ced1", "#ffa500", "#7cfc00", "#8a2be2", "#ff4500", "#1e90ff"]])


def _colmap(ids, drop_zero=True):
    vals = [int(u) for u in np.unique(ids) if (u > 0 or not drop_zero)]
    return {u: _PALETTE[i % len(_PALETTE)] for i, u in enumerate(vals)}


def parse_loc(path):
    """Scroll id + global L0 centre coords from a window filename like ``win_sample_s5_L0_z1008_y5826_x3738.npz``
    (``org`` is the window CENTRE at ``level``; L0 coord = org·2**level). Returns ``(human_str, filename_tag)``."""
    b = os.path.basename(path)
    m = re.search(r"_s([^_]+)_L(\d+)_z(\d+)_y(\d+)_x(\d+)", b)
    if not m:
        return (b, "win")
    sid = m.group(1); L = int(m.group(2)); z, y, x = (int(m.group(i)) * (2 ** L) for i in (3, 4, 5))
    return (f"scroll {sid}  |  L0 centre  z={z} y={y} x={x}" + (f"  [src L{L}]" if L else ""), f"s{sid}_z{z}_y{y}_x{x}")


def pca_rgb(feat, nsub=40000):
    """Top-3 max-variance (PCA) directions → RGB, percentile-stretched."""
    N = feat.shape[0]; mu = feat.mean(0, keepdims=True); x = feat - mu
    sub = x[np.random.default_rng(0).choice(N, min(nsub, N), replace=False)] if N > nsub else x
    _, _, vt = np.linalg.svd(sub, full_matrices=False); c = x @ vt[:3].T
    lo = np.percentile(c, 2, 0); hi = np.percentile(c, 98, 0)
    return np.clip((c - lo) / (hi - lo + 1e-6), 0, 1)


def up(a, f):
    """Nearest-neighbour upsample a (D,H,W,·) volume by integer factor ``f`` on the 3 spatial axes."""
    return a if f == 1 else np.repeat(np.repeat(np.repeat(a, f, 0), f, 1), f, 2)


def sep_rgb(feat, lab):
    """RGB from the 3 directions that best SEPARATE sheets among the foreground (Fisher LDA, GT sheet id = class),
    rather than max-variance PCA. ``feat`` (N,C); ``lab`` (N,) sheet id (0=bg). Falls back to PCA with <2 sheets."""
    C = feat.shape[1]; fg = lab > 0; ids = np.unique(lab[fg])
    if len(ids) < 2:
        return pca_rgb(feat)
    X = feat[fg]; y = lab[fg]; mu = X.mean(0)
    Sw = np.zeros((C, C)); Sb = np.zeros((C, C))
    for c in ids:
        Xc = X[y == c]; mc = Xc.mean(0); d = Xc - mc
        Sw += d.T @ d; Sb += len(Xc) * np.outer(mc - mu, mc - mu)
    Sw += 1e-2 * np.trace(Sw) / C * np.eye(C)
    _, V = _eigh(Sb, Sw)                       # generalised symmetric Sb v = w Sw v, w ascending
    W = V[:, ::-1][:, :3]                      # top-3 most-separating directions
    c3 = feat @ W; lo = np.percentile(c3[fg], 2, 0); hi = np.percentile(c3[fg], 98, 0)
    return np.clip((c3 - lo) / (hi - lo + 1e-6), 0, 1)


def fgbg_rgb(feat, fg, bg):
    """Diverging RGB from the single Fisher direction best separating foreground (surface) from background →
    coolwarm. ``feat`` (N,C); ``fg``, ``bg`` (N,) bool."""
    C = feat.shape[1]
    if fg.sum() < 5 or bg.sum() < 5:
        return np.zeros((len(feat), 3))
    Xf = feat[fg]; bidx = np.where(bg)[0]
    if len(bidx) > 50000:
        bidx = np.random.default_rng(0).choice(bidx, 50000, replace=False)
    Xb = feat[bidx]
    Sw = np.cov(Xf.T) * len(Xf) + np.cov(Xb.T) * len(Xb); Sw += 1e-2 * np.trace(Sw) / C * np.eye(C)
    w = np.linalg.solve(Sw, Xf.mean(0) - Xb.mean(0)); w /= np.linalg.norm(w) + 1e-9
    pr = feat @ w; lo, hi = np.percentile(pr, 2), np.percentile(pr, 98)
    return plt.get_cmap("coolwarm")(np.clip((pr - lo) / (hi - lo + 1e-9), 0, 1))[:, :3]


def build_net(recipe_path, ckpt_path, device, encoder_hf, encoder_file=None):
    """Build the full decoupled net (``with_instance=True``) and load the trained checkpoint (shape-filtered)."""
    from .. import netbuild
    from ..train import recipe as R
    from ..train.chain import resolve_encoder
    rec = R.load(recipe_path)
    enc = resolve_encoder(encoder_hf=encoder_hf, hf_file=encoder_file)
    net = netbuild.build(
        enc, device=device, arch_override=rec.encoder_arch(), global_blocks=rec.global_blocks,
        window_size=rec.window_size, n_sem_classes=rec.n_sem_classes, mask_dim=rec.mask_dim, n_queries=rec.n_queries,
        query_dim=rec.query_dim, instance_layers=rec.instance_layers, deep_supervision=rec.deep_supervision,
        with_instance=True, add_null_query=rec.add_null_query, learned_queries=rec.learned_queries,
        fps_queries=rec.fps_queries, cross_attn=rec.cross_attn, decoder=rec.decoder, neck=rec.neck,
        neck_layers=rec.neck_layers, neck_heads=rec.neck_heads, neck_points=rec.neck_points, surface_gate=rec.surface_gate)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model"]; cur = net.state_dict()
    sd = {k: v for k, v in sd.items() if not (k in cur and tuple(cur[k].shape) != tuple(v.shape))}
    net.load_state_dict(sd, strict=False); net.eval()
    print(f"loaded @ ep {ck.get('epoch')}", flush=True)
    return net, rec, ck.get("epoch")


@torch.no_grad()
def render(net, rec, cache, files, out, thresh, k, device, epoch, tag="val"):
    from ..data.dataset import LocalHercuvitDataset, split_cache  # noqa: F401 (split_cache used by caller)
    from ..model.decoder_common import mask_logits_from_embed
    os.makedirs(out, exist_ok=True)
    ds = LocalHercuvitDataset(cache, augment=False, target_voxel_um=rec.target_voxel_um, crop=rec.window, files=files)
    picked = 0
    for i in range(len(ds)):
        it = ds[i]; sem = torch.as_tensor(it["sem"]); loc_h, loc_s = parse_loc(it.get("path", f"win{i}"))
        if int((sem == 1).sum()) < 400:
            continue
        ct = torch.as_tensor(it["ct"])
        ctb = (ct.unsqueeze(0) if ct.ndim == 4 else ct.unsqueeze(0).unsqueeze(0)).float().to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            feats = net.backbone.forward_features(ctb, view_kind="global"); tok = feats["x_norm_patchtokens"][0].float()
            out_d = net(ctb, dn_owner=None, run_instance=True)                         # inference: no DN queries
        D = 128
        owt = (torch.as_tensor(it["owner"]) * (sem == 1)).cpu().numpy(); semn = sem.numpy()   # thin GT sheet id @1/1
        tok_lab = owt.reshape(16, 8, 16, 8, 16, 8).max(axis=(1, 3, 5)).reshape(-1)             # per-token sheet id @1/8
        s6 = (semn == 1).reshape(16, 8, 16, 8, 16, 8).sum((1, 3, 5)).reshape(-1)
        i6 = (semn == 2).reshape(16, 8, 16, 8, 16, 8).sum((1, 3, 5)).reshape(-1)
        tok_fg = (s6 >= 2) & (i6 < 64); tok_bg = (s6 == 0) & (i6 == 0)                          # token fg/bg @1/8
        vox_fg = (semn.reshape(-1) == 1); vox_bg = (semn.reshape(-1) == 0)                      # voxel fg/bg @1/1
        tokn = tok.cpu().numpy()
        enc_pca = up(pca_rgb(tokn).reshape(16, 16, 16, 3), 8)
        enc_sld = up(sep_rgb(tokn, tok_lab).reshape(16, 16, 16, 3), 8)
        enc_fld = up(fgbg_rgb(tokn, tok_fg, tok_bg).reshape(16, 16, 16, 3), 8)
        mf = out_d["mask_features"][0].float(); S4 = mf.shape[-1]; mfn = mf.cpu().numpy().reshape(mf.shape[0], -1).T
        mask_pca = up(pca_rgb(mfn).reshape(S4, S4, S4, 3), D // S4)
        mask_sld = up(sep_rgb(mfn, owt.reshape(-1)).reshape(S4, S4, S4, 3), D // S4)
        mask_fld = up(fgbg_rgb(mfn, vox_fg, vox_bg).reshape(S4, S4, S4, 3), D // S4)
        lay = out_d["instance"]["layers"][-1]
        mlog = mask_logits_from_embed(lay["mask_embed"][0:1].float(), mf.unsqueeze(0))[0]
        owner = up(mlog.argmax(0).cpu().numpy(), D // S4)
        surfp = torch.softmax(out_d["sem_logits"][0][0].float(), dim=0)[1].cpu().numpy()
        gate = up((surfp > thresh), D // S4); psurf = gate
        ctn = (ct[0].cpu().numpy() if ct.ndim == 4 else ct.cpu().numpy()); gt = (semn == 1); gto = owt
        zc = gt.sum(axis=(1, 2)); zs = sorted(int(z) for z in np.argsort(zc)[::-1][:2])
        nqg = len(np.unique(owner[gate])); nsh = int((np.unique(gto) > 0).sum())
        qmap = _colmap(owner[gate], drop_zero=False); smap = _colmap(gto, drop_zero=True)
        cols = [("encoder PCA", enc_pca), ("encoder sheet-LDA", enc_sld), ("encoder fg/bg-LDA", enc_fld),
                ("mask-feat PCA", mask_pca), ("mask-feat sheet-LDA", mask_sld), ("mask-feat fg/bg-LDA", mask_fld)]
        fig, ax = plt.subplots(len(zs), 10, figsize=(34, 3.4 * len(zs))); ax = ax if len(zs) > 1 else ax[None, :]
        for r, z in enumerate(zs):
            base = ctn[z]; base = (base - base.min()) / (np.ptp(base) + 1e-6); rgbb = np.stack([base] * 3, -1)
            ax[r, 0].imshow(base, cmap="gray"); ax[r, 0].set_title(f"CT z={z}")
            for c, (ttl, img) in enumerate(cols):
                ax[r, 1 + c].imshow(img[z]); ax[r, 1 + c].set_title(ttl)
            ov = rgbb.copy(); ov[psurf[z]] = [0, 1, 1]; ax[r, 7].imshow(ov); ax[r, 7].set_title("surface prediction")
            om = owner[z]; g2 = gate[z]; d2 = rgbb.copy()
            for u in np.unique(om[g2]):
                if int(u) in qmap:
                    d2[(om == u) & g2] = qmap[int(u)]
            ax[r, 8].imshow(d2); ax[r, 8].set_title(f"predicted sheets ({nqg} q)")
            omg = gto[z]; d3 = rgbb.copy()
            for u in np.unique(omg):
                if u > 0:
                    d3[omg == u] = smap[int(u)]
            ax[r, 9].imshow(d3); ax[r, 9].set_title(f"GT sheets ({nsh})")
            for c in range(10):
                ax[r, c].axis("off")
        fig.suptitle(f"{tag}  win#{i}   {loc_h}   ({int(gt.sum())} GT-surf vx)  ep {epoch}  surf-gate P>{thresh}\n"
                     f"— PCA | sheet-LDA | fg/bg-LDA  for encoder & mask-feats ;  col9 predicted  vs  col10 GT")
        fig.tight_layout(); fp = os.path.join(out, f"val_{picked:02d}_{loc_s}.png"); fig.savefig(fp, dpi=85); plt.close(fig)
        print(f"  {fp}  pred-q={nqg} gt-sheets={nsh}", flush=True); picked += 1
        if picked >= k:
            break


def main():
    ap = argparse.ArgumentParser(description="10-panel CT + PCA/LDA + prediction render for a trained decoupled HercuViT.")
    ap.add_argument("--ckpt", required=True, help="trained HercuViT checkpoint")
    ap.add_argument("--out", default="render_panels", help="output dir for the PNGs")
    ap.add_argument("--cache", default="/root/train_nano/train_cache")
    ap.add_argument("--recipe", default="hercunet/hercuvit/recipes/full_decoupled.yml")
    ap.add_argument("--encoder-hf", dest="encoder_hf", default=_ENC_288)
    ap.add_argument("--encoder-file", dest="encoder_file", default=None)
    ap.add_argument("--thresh", type=float, default=0.4, help="surface-gate probability threshold")
    ap.add_argument("--k", type=int, default=10, help="number of windows to render")
    ap.add_argument("--split", choices=("val", "train"), default="val")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    from ..data.dataset import split_cache
    device = torch.device(args.device)
    net, rec, epoch = build_net(args.recipe, args.ckpt, device, args.encoder_hf, args.encoder_file)
    tr, va = split_cache(args.cache, fold=rec.val_fold)
    files = va if args.split == "val" else tr
    render(net, rec, args.cache, files, args.out, args.thresh, args.k, device, epoch, tag=args.split)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

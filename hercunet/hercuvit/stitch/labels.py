"""Stitcher pseudo-label extraction (``hercuvit labels create``).

At each **corner** (a point where 8 inference windows meet) run HercuViT on the 8 octant windows plus one window
**centred on the corner**, and emit one self-supervised stitching training sample — with **no ground truth**:

  * spatial tokens — each octant window's per-window normalised-PCA (``D``-dim) tokens inside the 2p cube,
    overlap kept, tagged by source-window slot + position (docs/hercuvit-stitching.md §3.2/§3.4/§5.1);
  * queries — one per (octant, owner) fragment: its **median** ``D``-dim embedding (median taken AFTER the PCA),
    slot, and mean confidence (§3.3/§5.3);
  * labels — a pairwise same-object matrix ``Y`` from overlap-voting each octant fragment onto the centred pass's
    single consistent segmentation (§7.1): 1 = same, 0 = different, -1 = ignore (no reliable overlap).

Corner coordinates come from one of (mirroring ``herculabels create``): ``--count`` random (material-gated, seeded),
``--coords-file`` manifest (``SCROLL,Z,Y,X`` per line), ``--corpus`` (the herculabel windows' own centres), or a single
``--coords`` + ``--scroll``. The herculabels themselves are never read as labels — only their locations are reused.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from ..infer.loader import load_hercuvit
from ..data.dataset import ScrollCtReader, resample_to_crop
from ..data.normalize import dinovol_normalise

CROP = 128
POOL = 8
G = CROP // POOL                                                   # 1/8 token grid side (16)
OCT = [(sz, sy, sx) for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)]   # the 8 octant windows


# ----------------------------------------------------------------------------- HercuViT forward
@torch.no_grad()
def _infer(net, ct128, device):
    """One 128³ normalised CT -> 1/8 encoder tokens, per-voxel owner (argmax query), surface prob."""
    ctb = torch.from_numpy(np.ascontiguousarray(ct128)).float()[None, None].to(device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        tok = net.backbone.forward_features(ctb, view_kind="global")["x_norm_patchtokens"][0].float()
        out = net(ctb, run_instance=True)
    mf = out["mask_features"][0].float()
    me = out["instance"]["layers"][-1]["mask_embed"][0].float()
    mff = mf.reshape(mf.shape[0], -1)
    best = torch.full((mff.shape[1],), -1e30, device=device)
    owner = torch.zeros((mff.shape[1],), dtype=torch.long, device=device)
    for q0 in range(0, me.shape[0], 8):                            # argmax query per voxel, in Q-chunks (8 GB-safe)
        mx, ix = (me[q0:q0 + 8] @ mff).max(0)
        u = mx > best
        owner[u] = (q0 + ix)[u]
        best[u] = mx[u]
    r = dict(enc=tok.reshape(G, G, G, -1).cpu().numpy(),
             own=owner.reshape(CROP, CROP, CROP).to(torch.int16).cpu().numpy(),
             surf=torch.softmax(out["sem_logits"][0][0].float(), 0)[1].cpu().numpy().astype(np.float32))
    del ctb, mf, me, mff, owner, best
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return r


def _reduce(enc, surf8, D):
    """Per-window normalised PCA -> (G,G,G,D). z-score + PCA fit on this window's SURFACE cells, project EVERY cell.
    The query median (below) is then taken over these projected vectors — median AFTER the PCA (docs §3.4)."""
    X = enc.reshape(-1, enc.shape[-1])
    s = surf8.reshape(-1) > 0.5
    fit = X[s] if int(s.sum()) > D + 8 else X
    mu = fit.mean(0); sd = fit.std(0) + 1e-6
    Z = (fit - mu) / sd
    zc = Z.mean(0)
    _, _, vt = np.linalg.svd(Z - zc, full_matrices=False)
    V = vt[:D]
    emb = (((X - mu) / sd) - zc) @ V.T
    return emb.reshape(G, G, G, D).astype(np.float32)


def _ct128(reader, scroll, centre, fov, vu, target_um):
    ctn = reader(scroll, 0, centre, (fov, fov, fov)).astype(np.float32)
    ct, _ = resample_to_crop(dinovol_normalise(ctn), {}, vu, target_voxel_um=target_um, crop=CROP)
    return ct


# ----------------------------------------------------------------------------- one corner -> one sample
def extract_corner(net, reader, scroll, coord, target_um, device, *, D=3, overlap=0.1,
                   min_vote=25, min_pts=20, stream_workers=8):
    vu = float(reader._vol(scroll).meta.voxel_size_um)
    fov = int(round(CROP * target_um / vu)); nsp = fov / CROP
    core = int(round(CROP * (1 - overlap) * target_um / vu)); h = core // 2   # octant centres at coord ± h

    # CT prefetch: stream the 9 windows (centred + 8 octants) CONCURRENTLY — the reads are latency-bound (many small
    # chunk GETs), so overlapping them collapses ~9× sequential latency into ~one read's worth. The volume is already
    # open+cached above, so there is no open race; the GPU forwards still run one at a time (unchanged VRAM).
    centres = [coord] + [(coord[0] + k[0] * h, coord[1] + k[1] * h, coord[2] + k[2] * h) for k in OCT]
    read = lambda c: _ct128(reader, scroll, c, fov, vu, target_um)
    if stream_workers > 1:
        with ThreadPoolExecutor(max_workers=min(stream_workers, len(centres))) as ex:
            cts = list(ex.map(read, centres))
    else:
        cts = [read(c) for c in centres]
    centre = _infer(net, cts[0], device)
    W = {k: _infer(net, cts[1 + i], device) for i, k in enumerate(OCT)}
    emb = {k: _reduce(W[k]["enc"], W[k]["surf"][POOL // 2::POOL, POOL // 2::POOL, POOL // 2::POOL], D) for k in OCT}

    # --- queries (fragments) + vote each onto the centred pass (global id = centred owner) -------------------------
    rng = np.random.default_rng(0)
    qslot, qown, qemb, qconf, qgid = [], [], [], [], []
    for k in OCT:
        o8 = W[k]["own"][POOL // 2::POOL, POOL // 2::POOL, POOL // 2::POOL]
        s8 = W[k]["surf"][POOL // 2::POOL, POOL // 2::POOL, POOL // 2::POOL] > 0.5
        slot = (k[0] > 0) * 4 + (k[1] > 0) * 2 + (k[2] > 0)
        off = np.array([k[0] * h, k[1] * h, k[2] * h])
        lo = np.maximum(-fov / 2, off - fov / 2); hi = np.minimum(fov / 2, off + fov / 2)
        pts = rng.uniform(lo, hi, (12000, 3))
        vc = pts / nsp + CROP / 2; vk = (pts - off) / nsp + CROP / 2
        m = ((vc >= 0) & (vc <= CROP - 1)).all(1) & ((vk >= 0) & (vk <= CROP - 1)).all(1)
        ic = np.clip(np.round(vc[m]).astype(int), 0, CROP - 1); ik = np.clip(np.round(vk[m]).astype(int), 0, CROP - 1)
        vs = (centre["surf"][ic[:, 0], ic[:, 1], ic[:, 2]] > 0.5) & (W[k]["surf"][ik[:, 0], ik[:, 1], ik[:, 2]] > 0.5)
        oc_ov = centre["own"][ic[vs, 0], ic[vs, 1], ic[vs, 2]]; ok_ov = W[k]["own"][ik[vs, 0], ik[vs, 1], ik[vs, 2]]
        confw = W[k]["surf"][POOL // 2::POOL, POOL // 2::POOL, POOL // 2::POOL]
        for o in np.unique(o8[s8]):
            cells = (o8 == o) & s8
            if int(cells.sum()) < min_pts:
                continue
            med = np.median(emb[k][cells], 0)                     # median AFTER the PCA projection (docs §3.4)
            cand = ok_ov == o
            if int(cand.sum()) >= min_vote:
                vals, cnts = np.unique(oc_ov[cand], return_counts=True)
                gid = int(vals[cnts.argmax()])                    # centred-pass owner = the global id
            else:
                gid = -1                                          # no reliable overlap -> ignore this fragment's pairs
            qslot.append(slot); qown.append(int(o)); qemb.append(med.astype(np.float16))
            qconf.append(float(confw[cells].mean())); qgid.append(gid)

    M = len(qslot)
    if M < 2:
        return None
    qgid = np.asarray(qgid)
    Y = np.full((M, M), -1, np.int8)                              # -1 ignore
    known = qgid >= 0
    same = (qgid[:, None] == qgid[None, :]) & known[:, None] & known[None, :]
    both = known[:, None] & known[None, :]
    Y[both] = 0; Y[same] = 1; np.fill_diagonal(Y, -1)

    # --- spatial tokens: each octant window's 1/8 tokens inside the ±fov/2 cube (overlap kept, slot-tagged) --------
    ii, jj, kk = np.meshgrid(np.arange(G), np.arange(G), np.arange(G), indexing="ij")
    Xs, Ps, Ss = [], [], []
    for k in OCT:
        off = np.array([k[0] * h, k[1] * h, k[2] * h]); slot = (k[0] > 0) * 4 + (k[1] > 0) * 2 + (k[2] > 0)
        d = np.stack([(ii * POOL + POOL / 2 - CROP / 2) * nsp + off[0],
                      (jj * POOL + POOL / 2 - CROP / 2) * nsp + off[1],
                      (kk * POOL + POOL / 2 - CROP / 2) * nsp + off[2]], -1)     # native offset from the corner
        inside = (np.abs(d) <= fov / 2).all(-1)
        Xs.append(emb[k][inside]); Ps.append((d[inside] / (fov / 2)).astype(np.float16))
        Ss.append(np.full(int(inside.sum()), slot, np.uint8))

    return dict(scroll=str(scroll), coord=np.asarray(coord, np.int64), D=np.int64(D), overlap=np.float32(overlap),
                Xq=np.asarray(qemb, np.float16), slot_q=np.asarray(qslot, np.uint8),
                conf_q=np.asarray(qconf, np.float16), own_q=np.asarray(qown, np.int16), Y=Y,
                Xs=np.concatenate(Xs).astype(np.float16), Ps=np.concatenate(Ps),
                slot_s=np.concatenate(Ss))


# ------------------------------------------------------- corner sources — REUSE herculabels' window machinery --
# No reinvented parsing / sampling: `_resolve_coords` calls herculabels' own `parse_coords_file`, the seeded
# `sample_windows` stream (material-bbox aware), and `window_material_frac` (the coarse-mask himat screen).
def _validate(a):
    """Mirror `herculabels create`: --count is required unless --coords-file / --coords; --coords needs --scroll + --count 1."""
    if a.coords_file:
        if a.count is not None:
            raise SystemExit("labels create: cannot use --count with --coords-file — the file defines the corners")
    elif a.coords is not None:
        if not a.scroll:
            raise SystemExit("labels create: --coords requires --scroll")
        if a.count not in (None, 1):
            raise SystemExit("labels create: --coords targets one exact corner, so requires --count 1")
    elif a.count is None:
        raise SystemExit("labels create: --count is required (unless --coords-file or --coords)")
    if a.himat is not None and not (0.0 <= a.himat <= 1.0):
        raise SystemExit("labels create: --himat must be a fraction in [0, 1]")


def _resolve_coords(a):
    """Corner list, reusing herculabels' exact window machinery (no duplicated parsing/sampling)."""
    from ...herculabels.grind import parse_coords_file
    if a.coords_file:
        return [(s, (z, y, x)) for s, z, y, x in parse_coords_file(a.coords_file, a.scroll)]
    if a.coords is not None:
        z, y, x = (int(v) for v in a.coords.split(","))
        return [(a.scroll, (z, y, x))]
    # --count: herculabels' seeded window stream + coarse-mask himat screen (same corpus-set logic as `create`)
    from ...config import Config
    from ...data import get_backend
    from ...herculabels._brick import sample_windows, window_material_frac, find_scroll, in_band_scrolls
    from ...herculabels.create import _resolve_scrolls
    be = get_backend(Config.from_env())
    scrolls = _resolve_scrolls(be, a.scroll, 7.5, 9.5, find_scroll, in_band_scrolls)
    himat = a.himat if a.himat is not None else 0.15
    out, seen, cap = [], 0, a.count * 60
    for sid, wc in sample_windows(be, scrolls, level=0, seed=(a.seed or 0)):
        if len(out) >= a.count or seen > cap:
            break
        seen += 1
        try:
            if window_material_frac(be, find_scroll(be, sid), wc) < himat:
                continue
        except Exception:
            continue
        out.append((sid, tuple(int(v) for v in wc)))
    return out


# ----------------------------------------------------------------------------- driver
def create_run(a):
    _validate(a)
    device = torch.device(a.device)
    net, recipe, _ = load_hercuvit(model=a.model, model_hf=a.model_hf, ckpt=a.ckpt, device=a.device,
                                   with_instance=True)
    target_um = float(recipe.target_voxel_um)
    reader = ScrollCtReader()
    os.makedirs(a.out, exist_ok=True)

    coords = _resolve_coords(a)
    coords = coords[a.shard::a.nshards]                           # simple per-process sharding for a multi-GPU run
    print(f"[labels] {len(coords)} corners (shard {a.shard}/{a.nshards}) -> {a.out}", flush=True)

    t0 = time.time(); done = 0; skipped = 0; failed = 0
    for i, (scroll, coord) in enumerate(coords):
        tag = f"s{scroll}_z{coord[0]}_y{coord[1]}_x{coord[2]}"
        fp = os.path.join(a.out, f"stitch_{tag}.npz")
        if os.path.exists(fp) and not a.overwrite:
            skipped += 1; continue
        try:
            s = extract_corner(net, reader, scroll, coord, target_um, device, D=a.pca_dim, overlap=a.overlap,
                               min_vote=a.min_vote, min_pts=a.min_pts, stream_workers=a.stream_workers)
            if s is None:
                skipped += 1
            else:
                np.savez_compressed(fp, **s)
                done += 1
        except Exception as e:  # one bad corner never aborts the sweep
            failed += 1
            print(f"[labels] {tag} FAILED: {repr(e)[:160]}", flush=True)
        if (i + 1) % 20 == 0 or i + 1 == len(coords):
            dt = time.time() - t0
            print(f"[labels] {i+1}/{len(coords)}  done={done} skip={skipped} fail={failed}  "
                  f"{dt/(i+1):.2f}s/corner", flush=True)
    print(f"[labels] DONE: {done} written, {skipped} skipped, {failed} failed in {time.time()-t0:.0f}s", flush=True)

"""``hercuvit stitcher infer`` — apply a trained stitcher to a volume's stored HercuViT artifacts (docs §8).

Consumes the two artifacts a ``hercuvit infer --store-embeddings`` run emits (no HercuViT re-run here):

  1. a **pooled-window-embedding zarr** (``emb (N,G,G,G,D)`` + ``origin (N,3)``) — each window's per-window
     normalised-PCA tokens (``stitch.embed.reduce_normalised_pca``), the stitcher's spatial memory ``Z`` source; and
  2. a **per-sheet median embedding** in each ``.asv`` sheet's ``annotation.emb`` — the after-PCA
     ``median_after_pca`` descriptor in the same per-window basis, the stitcher's queries.

Per corner (a point where 8 core-grid windows meet) it gathers those 8 windows' tokens from the zarr and the
touching sheets' medians from the ``.asv``, assembles the sample with the SAME placement used in training
(:func:`hercunet.hercuvit.stitch.embed.corner_spatial_tokens`) and the SAME batch collation
(:func:`hercunet.hercuvit.stitch.dataset.stitch_collate`), runs :class:`StitchDecoder`, and writes the pairwise
affinities straight into the ``.asv`` as an ``affinities`` edge table (:func:`hercunet.meshes.asv.write_affinities`),
so one file carries both the sheets and their cross-window affinities. Both directions ``(A,B)`` and ``(B,A)`` are
stored, so every neighbour of a sheet is a single indexed lookup. Agglomeration into a fragment→object lookup (§9)
is a later, separate step over this table.
"""
from __future__ import annotations

import re

import numpy as np


def load_stitcher(path_or_repo: str, *, device="cuda"):
    """Rebuild a trained :class:`StitchDecoder` from a checkpoint (local ``.pth``/dir or an HF repo) → ``(model, cfg)``.
    Mirrors :func:`hercunet.hercuvit.infer.loader.load_hercuvit`: the config is stored in the checkpoint, so the
    decoder is rebuilt exactly and the weights loaded over it. Ready for the infer wiring + ``hercuvit infer
    --stitcher``."""
    import os

    import torch

    from .config import config_from_dict
    from .model import StitchDecoder

    root = path_or_repo
    if not os.path.exists(root):                                   # treat a non-path as an HF repo id
        from huggingface_hub import snapshot_download
        print(f"[stitcher] downloading stitcher from HF {root} …", flush=True)
        root = snapshot_download(root)
    if os.path.isdir(root):
        for name in ("checkpoint_best.pth", "checkpoint_final.pth", "checkpoint_latest.pth"):
            if os.path.isfile(os.path.join(root, name)):
                root = os.path.join(root, name); break
        else:
            raise SystemExit(f"hercuvit stitcher: no checkpoint_*.pth under {root}.")
    ck = torch.load(root, map_location="cpu", weights_only=False)
    cfg = config_from_dict(ck.get("config", {}))
    model = StitchDecoder(cfg)
    model.load_state_dict(ck["model"])
    model.eval().to(device)
    print(f"[stitcher] loaded {root} ({cfg.summary()})", flush=True)
    return model, cfg


_WIN_RE = re.compile(r"z(-?\d+)_y(-?\d+)_x(-?\d+)")


def _parse_window(s: str) -> tuple[int, int, int]:
    """``'z11500_y2875_x3450'`` → ``(11500, 2875, 3450)`` (global-L0 window origin, z, y, x) — the zarr ``origin`` row."""
    m = _WIN_RE.match(s)
    if not m:
        raise ValueError(f"unparseable sheet window id {s!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _load_sheets(asv_path: str, D: int):
    """Read every sheet's id, source-window origin, median embedding and bbox from the ``.asv`` → a dict mapping each
    window origin ``(z,y,x)`` to a list of its sheets ``{id, emb(D,), conf, lo(x,y,z), hi(x,y,z)}``. Sheets with no /
    wrong-dim ``annotation.emb`` are skipped (they cannot be queries)."""
    import json
    import sqlite3

    db = sqlite3.connect(f"file:{asv_path}?mode=ro", uri=True)
    try:
        rows = db.execute("SELECT id,window,annotation,min_x,max_x,min_y,max_y,min_z,max_z,confidence "
                          "FROM sheets").fetchall()
    finally:
        db.close()
    by_win: dict[tuple, list] = {}
    skipped = 0
    for sid, win, ann, x0, x1, y0, y1, z0, z1, conf in rows:
        try:
            emb = json.loads(ann).get("emb")
        except Exception:
            emb = None
        if emb is None or len(emb) != D:
            skipped += 1
            continue
        by_win.setdefault(_parse_window(win), []).append(
            dict(id=int(sid), emb=np.asarray(emb, np.float32), conf=float(conf or 0.0),
                 lo=(float(x0), float(y0), float(z0)), hi=(float(x1), float(y1), float(z1))))
    return by_win, len(rows), skipped


def infer_run(a) -> None:
    import time

    import torch
    import zarr

    from ...meshes.asv import write_affinities
    from .dataset import stitch_collate
    from .embed import corner_spatial_tokens

    if not (a.embeddings_zarr and a.asv and a.stitcher):
        raise SystemExit("hercuvit stitcher infer: need --embeddings-zarr, --asv and --stitcher.")

    device = torch.device(a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model, cfg = load_stitcher(a.stitcher, device=device.type)
    D = int(cfg.pca_dim)
    top_k = int(getattr(a, "top_k", 0) or 0)
    batch = max(1, int(getattr(a, "batch", 16)))

    # --- stored artifacts: the per-window embedding grids (spatial memory) + their global-L0 origins -----------------
    g = zarr.open(a.embeddings_zarr, mode="r")
    if g["emb"].shape[-1] != D:
        raise SystemExit(f"stitcher infer: zarr emb dim {g['emb'].shape[-1]} != stitcher pca_dim {D} — "
                         f"the stitcher and the `hercuvit infer --emb-dim` must match.")
    emb_grid = np.asarray(g["emb"][:])                         # (N,G,G,G,D) f16 — small; load once
    origin = np.asarray(g["origin"][:])                       # (N,3) int z,y,x
    N, G = emb_grid.shape[0], emb_grid.shape[1]
    CROP, POOL = 128, 128 // G
    overlap = float(g.attrs.get("overlap", 0.1))
    orig_key = {tuple(int(v) for v in origin[n]): n for n in range(N)}

    # lattice geometry (one regular core-grid): stride = window-centre spacing; fov = window side; nsp = native/grid step
    uz, uy, ux = (np.unique(origin[:, i]) for i in range(3))
    diffs = np.concatenate([np.diff(uz), np.diff(uy), np.diff(ux)])
    stride = float(np.median(diffs)) if diffs.size else float(CROP) * (1.0 - overlap)
    fov = stride / (1.0 - overlap)
    nsp = fov / CROP
    h = stride / 2.0                                          # octant-window centre offset from the corner (±h)
    half = fov / 2.0

    by_win, n_sheets, n_skipped = _load_sheets(a.asv, D)
    print(f"[stitcher] {N} windows (grid {uz.size}x{uy.size}x{ux.size}, stride {stride:.0f}, fov {fov:.0f}), "
          f"{n_sheets} sheets ({n_skipped} without a {D}-d median skipped) -> affinities in "
          f"{a.out or a.asv}", flush=True)

    OCT = [(sz, sy, sx) for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)]   # 8 octant windows of a corner

    # --- build one sample per interior corner (8 present windows) ---------------------------------------------------
    items, metas, corners_meta = [], [], []
    cidx = 0
    for zi in range(uz.size - 1):
        for yi in range(uy.size - 1):
            for xi in range(ux.size - 1):
                # the 8 windows of this corner, keyed by octant sign; skip the corner unless all 8 exist
                emb_by_oct, win_origin = {}, {}
                ok = True
                for (sz, sy, sx) in OCT:
                    o = (int(uz[zi + (sz > 0)]), int(uy[yi + (sy > 0)]), int(ux[xi + (sx > 0)]))
                    n = orig_key.get(o)
                    if n is None:
                        ok = False
                        break
                    emb_by_oct[(sz, sy, sx)] = emb_grid[n].astype(np.float32)
                    win_origin[(sz, sy, sx)] = o
                if not ok:
                    continue
                # corner position (global L0, z,y,x) = midpoint of the two centres along each axis
                cz = (uz[zi] + uz[zi + 1]) / 2.0 + half
                cy = (uy[yi] + uy[yi + 1]) / 2.0 + half
                cx = (ux[xi] + ux[xi + 1]) / 2.0 + half
                clo = (cx - half, cy - half, cz - half)      # x,y,z — to test against sheet bbox (x,y,z)
                chi = (cx + half, cy + half, cz + half)

                # queries: each octant window's sheets whose bbox reaches the corner cube (docs §5.3, bbox proxy)
                Xq, slot_q, conf_q, qid = [], [], [], []
                for (sz, sy, sx) in OCT:
                    slot = (sz > 0) * 4 + (sy > 0) * 2 + (sx > 0)
                    for s in by_win.get(win_origin[(sz, sy, sx)], ()):
                        if all(s["hi"][i] >= clo[i] and s["lo"][i] <= chi[i] for i in range(3)):
                            Xq.append(s["emb"]); slot_q.append(slot); conf_q.append(s["conf"]); qid.append(s["id"])
                if len(qid) < 2:
                    continue

                Xs_a, Ps_a, Ss_a = corner_spatial_tokens(emb_by_oct, h=h, fov=fov, nsp=nsp, CROP=CROP, POOL=POOL)
                M = len(qid)
                items.append(dict(
                    Xs=torch.from_numpy(Xs_a.astype(np.float32)), Ps=torch.from_numpy(Ps_a.astype(np.float32)),
                    slot_s=torch.from_numpy(Ss_a.astype(np.int64)),
                    Xq=torch.from_numpy(np.asarray(Xq, np.float32)),
                    slot_q=torch.tensor(slot_q, dtype=torch.long), conf_q=torch.tensor(conf_q, dtype=torch.float32),
                    Y=torch.zeros(M, M, dtype=torch.long), path=""))          # Y unused at infer; keeps collate reuse
                metas.append((cidx, qid))
                corners_meta.append((cidx, int(round(cz)), int(round(cy)), int(round(cx)), M))
                cidx += 1

    if not items:
        raise SystemExit("hercuvit stitcher infer: no interior corner had ≥2 queryable sheets — nothing to write.")

    # --- batched forward + collect affinity rows (both directions) --------------------------------------------------
    t0 = time.time()
    rows: list[tuple] = []
    model.eval()
    with torch.no_grad():
        for b0 in range(0, len(items), batch):
            chunk = items[b0:b0 + batch]
            bt = stitch_collate(chunk)
            out = model(bt["Xs"].to(device), bt["Ps"].to(device), bt["slot_s"].to(device),
                        bt["valid_s"].to(device), bt["Xq"].to(device), bt["slot_q"].to(device),
                        bt["valid_q"].to(device))
            logit = out[-1]["logit"].float().cpu().numpy()               # (B,Mmax,Mmax) — last layer (§8)
            for bi, (cid, qid) in enumerate(metas[b0:b0 + batch]):
                M = len(qid)
                L = logit[bi, :M, :M]
                prob = 1.0 / (1.0 + np.exp(-L))
                for i in range(M):
                    order = np.argsort(-L[i])
                    order = order[order != i]
                    if top_k > 0:
                        order = order[:top_k]
                    for j in order:
                        rows.append((qid[i], qid[j], float(L[i, j]), float(prob[i, j]), cid))
            if (b0 // batch) % 20 == 0 or b0 + batch >= len(items):
                done = min(b0 + batch, len(items))
                print(f"[stitcher] {done}/{len(items)} corners · {len(rows)} edges · "
                      f"{done / max(1e-6, time.time() - t0):.0f} corner/s", flush=True)

    out_asv = a.asv
    if a.out:
        import shutil
        shutil.copy2(a.asv, a.out)
        out_asv = a.out
    n = write_affinities(out_asv, rows, corners=corners_meta)
    print(f"[stitcher] DONE: wrote {n} affinity rows (both directions) over {len(items)} corners into {out_asv} "
          f"in {time.time() - t0:.0f}s", flush=True)

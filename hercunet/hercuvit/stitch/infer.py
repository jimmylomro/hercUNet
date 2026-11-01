"""``hercuvit stitcher infer`` — apply a trained stitcher to a volume's stored HercuViT artifacts (docs §8).

Consumes the two artifacts a ``hercuvit infer --store-embeddings`` (or ``--stitcher``) run emits (no HercuViT re-run
here):

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

The stitching core is :func:`stitch_asv`, shared by this CLI AND by ``hercuvit infer --stitcher`` (which runs it in
the leader process straight after it has merged the per-window sheets + embeddings).

**Memory:** the work is **streamed one z-plane at a time** — a feeder pool reads each plane's pooled grids
(``emb_arr.oindex`` over just that plane's rows) and its sheets' medians (one indexed ``sheets.window`` lookup per
window), and the per-corner samples are forwarded and written in batches. Only a sliding two-plane window and one
GPU batch are resident, so the leader never holds the whole embedding array or the whole sheet table (which, for a
scroll, is several GB) — see :func:`stitch_asv`.
"""
from __future__ import annotations

import re

import numpy as np

#: Default published stitcher — resolved like every other inference model, as ``hf://<repo>``.
_STITCHER_HF = "jimmylomro/hercuvit-stitcher"


def resolve_stitcher_uri(uri: str) -> str:
    """Resolve a stitcher location to a LOCAL path (a dir or a ``.pth``), accepting the same ``<hf|file>://`` scheme
    the other inference tools take: ``hf://<repo>`` → ``snapshot_download`` (public, cached); ``file://<path>`` → the
    local path; an existing local path → itself; and a bare non-path → treated as an HF repo id (back-compat)."""
    import os

    u = str(uri)
    if u.startswith("hf://"):
        from huggingface_hub import snapshot_download
        repo = u[len("hf://"):]
        print(f"[stitcher] downloading stitcher from HF {repo} …", flush=True)
        return snapshot_download(repo)
    if u.startswith("file://"):
        return u[len("file://"):]
    if os.path.exists(u):
        return u
    from huggingface_hub import snapshot_download                       # a bare, non-existent value = an HF repo id
    print(f"[stitcher] downloading stitcher from HF {u} …", flush=True)
    return snapshot_download(u)


def load_stitcher(path_or_uri: str, *, device="cuda"):
    """Rebuild a trained :class:`StitchDecoder` from a checkpoint → ``(model, cfg)``. ``path_or_uri`` takes the
    ``<hf|file>://`` scheme (or a bare local path / HF repo id); a directory resolves ``checkpoint_best`` → ``final``
    → ``latest``. The config is stored in the checkpoint, so the decoder is rebuilt exactly and the weights loaded
    over it (mirrors :func:`hercunet.hercuvit.infer.loader.load_hercuvit`)."""
    import os

    import torch

    from .config import config_from_dict
    from .model import StitchDecoder

    root = resolve_stitcher_uri(path_or_uri)
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
    """``'z11500_y2875_x3450'`` → ``(11500, 2875, 3450)`` (global-L0 window origin, z, y, x) — the zarr ``origin`` row.
    The inverse (and single source) of the format is :func:`hercunet.meshes.extract.window_name`."""
    m = _WIN_RE.match(s)
    if not m:
        raise ValueError(f"unparseable sheet window id {s!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


# 8 octant windows of a corner, keyed by octant sign (sz,sy,sx); slot = 4·(sz>0)+2·(sy>0)+(sx>0).
_OCT = [(sz, sy, sx) for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)]


def _corner_item(emb_by_oct, sheets_by_oct, *, half, h, fov, nsp, CROP, POOL, cz, cy, cx):
    """Build ONE corner's training-shaped sample from its 8 octant windows' pooled grids (``emb_by_oct``) and the
    sheets touching the corner cube (``sheets_by_oct``: octant-sign → list of ``{id,emb,conf,lo,hi}``). Returns
    ``(item_dict, qid)`` or ``None`` when fewer than 2 sheets reach the corner. Placement + collation shapes are
    IDENTICAL to the eager path and to ``labels create`` — only the FEEDING of ``emb_by_oct``/``sheets_by_oct``
    changes between the streamed and whole-array paths, so the samples (and thus the affinities) are the same."""
    import torch

    from .embed import corner_spatial_tokens

    clo = (cx - half, cy - half, cz - half)                  # x,y,z — to test against sheet bbox (x,y,z)
    chi = (cx + half, cy + half, cz + half)
    Xq, slot_q, conf_q, qid = [], [], [], []
    for (sz, sy, sx) in _OCT:
        slot = (sz > 0) * 4 + (sy > 0) * 2 + (sx > 0)
        for s in sheets_by_oct.get((sz, sy, sx), ()):
            if all(s["hi"][i] >= clo[i] and s["lo"][i] <= chi[i] for i in range(3)):
                Xq.append(s["emb"]); slot_q.append(slot); conf_q.append(s["conf"]); qid.append(s["id"])
    if len(qid) < 2:
        return None
    Xs_a, Ps_a, Ss_a = corner_spatial_tokens(emb_by_oct, h=h, fov=fov, nsp=nsp, CROP=CROP, POOL=POOL)
    M = len(qid)
    item = dict(
        Xs=torch.from_numpy(Xs_a.astype(np.float32)), Ps=torch.from_numpy(Ps_a.astype(np.float32)),
        slot_s=torch.from_numpy(Ss_a.astype(np.int64)),
        Xq=torch.from_numpy(np.asarray(Xq, np.float32)),
        slot_q=torch.tensor(slot_q, dtype=torch.long), conf_q=torch.tensor(conf_q, dtype=torch.float32),
        Y=torch.zeros(M, M, dtype=torch.long), path="")      # Y unused at infer; keeps collate reuse
    return item, qid


def stitch_asv(embeddings_zarr: str, asv: str, stitcher: str, *, out: str | None = None,
               top_k: int = 0, batch: int = 16, device: str = "cuda",
               feed_workers: int = 3, readahead: int = 2) -> tuple[str, int]:
    """Stitch one volume's stored artifacts: read the pooled-window embeddings (``embeddings_zarr``) and the per-sheet
    medians in ``asv``, run the trained ``stitcher`` (a ``<hf|file>://`` location, local path, or HF repo id) at every
    interior core-grid corner, and write the pairwise affinities into the ``.asv`` (in place, or into ``out`` if
    given). ``top_k`` = neighbours kept per sheet per corner (0 = all in-range). Returns ``(out_asv, n_rows)``.

    STREAMED, bounded memory. The only thing loaded whole is the tiny ``(N,3)`` ``origin`` index (~12 B/window); it
    drives the lattice geometry AND groups window rows by z-plane. Then, plane by plane along z, ``feed_workers``
    threads prefetch (up to ``readahead`` planes ahead) each plane's pooled grids — ``emb_arr.oindex`` over just that
    plane's rows, so a plane's chunks are read once — and its sheets' medians (one indexed ``sheets.window`` lookup
    per existing window, each feeder on its own read-only connection). Corners of row ``zi`` use the sliding pair of
    planes ``{uz[zi], uz[zi+1]}``; the lower plane is evicted when the row completes. Per-corner samples are
    forwarded in batches of ``batch`` and their affinity rows streamed straight into ``write_affinities`` (which
    consumes the row generator with ``executemany``), so neither the whole embedding array, the whole sheet table,
    nor the full row list is ever resident — only a two-plane window and one GPU batch.

    This is the ONE stitching implementation — the ``hercuvit stitcher infer`` CLI and ``hercuvit infer --stitcher``
    (leader process) both call it, so the two paths never diverge."""
    import json
    import shutil
    import sqlite3
    import time
    from concurrent.futures import ThreadPoolExecutor

    import torch
    import zarr

    from ...meshes.asv import write_affinities
    from ...meshes.extract import window_name
    from .dataset import stitch_collate

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    model, cfg = load_stitcher(stitcher, device=dev.type)
    model.eval()
    D = int(cfg.pca_dim)
    top_k = int(top_k or 0)
    batch = max(1, int(batch))
    feed_workers = max(1, int(feed_workers))
    readahead = max(0, int(readahead))

    # --- stored artifacts: the per-window embedding grid is read a z-PLANE AT A TIME (never the whole array). Only
    # the tiny (N,3) origin index is loaded whole — it drives the lattice geometry AND tells us which window rows
    # belong to each z-plane, so the feeder can oindex exactly a plane's rows. -----------------------------------
    g = zarr.open(embeddings_zarr, mode="r")
    emb_arr = g["emb"]
    if emb_arr.shape[-1] != D:
        raise SystemExit(f"stitcher infer: zarr emb dim {emb_arr.shape[-1]} != stitcher pca_dim {D} — "
                         f"the stitcher and the `hercuvit infer --emb-dim` must match.")
    origin = np.asarray(g["origin"][:])                       # (N,3) int z,y,x — small; the only whole-array load
    N, G = int(emb_arr.shape[0]), int(emb_arr.shape[1])
    CROP, POOL = 128, 128 // G
    overlap = float(g.attrs.get("overlap", 0.1))

    # lattice geometry (one regular core-grid): stride = window-centre spacing; fov = window side; nsp = native/grid
    # step. Derived from the whole origin index — IDENTICAL to the un-streamed computation.
    uz, uy, ux = (np.unique(origin[:, i]) for i in range(3))
    diffs = np.concatenate([np.diff(uz), np.diff(uy), np.diff(ux)])
    stride = float(np.median(diffs)) if diffs.size else float(CROP) * (1.0 - overlap)
    fov = stride / (1.0 - overlap)
    nsp = fov / CROP
    h = stride / 2.0                                          # octant-window centre offset from the corner (±h)
    half = fov / 2.0

    # which emb rows live on each z-plane (so a plane is one oindex read, not N scattered single-row reads)
    plane_ns: dict[int, list[int]] = {}
    for n in range(N):
        plane_ns.setdefault(int(origin[n, 0]), []).append(n)

    out_asv = asv
    if out:
        shutil.copy2(asv, out)
        out_asv = out

    # Index sheets by their window id so each window's medians are an O(log S) lookup while streaming (we open the
    # .asv read-write here anyway — the affinities table is written into it at the end). Without this the per-window
    # `WHERE window=?` would be a full scan per window => O(windows × sheets).
    idb = sqlite3.connect(out_asv)
    try:
        idb.execute("CREATE INDEX IF NOT EXISTS sheets_window ON sheets(window)")
        n_sheets = int(idb.execute("SELECT COUNT(*) FROM sheets").fetchone()[0])
        idb.commit()
    finally:
        idb.close()

    if uz.size < 2 or uy.size < 2 or ux.size < 2:
        raise SystemExit(f"hercuvit stitcher infer: grid {uz.size}x{uy.size}x{ux.size} has no interior corner "
                         f"(need ≥2 windows on each axis) — nothing to stitch.")

    print(f"[stitcher] {N} windows (grid {uz.size}x{uy.size}x{ux.size}, stride {stride:.0f}, fov {fov:.0f}), "
          f"{n_sheets} sheets -> affinities in {out_asv} "
          f"(streamed: {feed_workers} feed worker(s), {readahead}-plane readahead)", flush=True)

    def load_plane(zval: int):
        """Feeder task: one z-plane's pooled-embedding grids + its sheets' median descriptors. Opens its OWN
        read-only sqlite connection so feeder threads never share a cursor. Returns ``(emb_dict, sheet_dict,
        n_skipped)`` keyed by window origin ``(z,y,x)``."""
        ns = plane_ns.get(int(zval))
        emb_dict: dict[tuple, np.ndarray] = {}
        sheet_dict: dict[tuple, list] = {}
        skipped = 0
        if not ns:
            return emb_dict, sheet_dict, skipped
        ns_arr = np.asarray(sorted(ns), dtype=np.int64)
        embs = np.asarray(emb_arr.oindex[ns_arr]).astype(np.float32)      # (len,G,G,G,D) — this plane's chunks, once
        for i, n in enumerate(ns_arr):
            emb_dict[(int(origin[n, 0]), int(origin[n, 1]), int(origin[n, 2]))] = embs[i]
        db = sqlite3.connect(f"file:{out_asv}?mode=ro", uri=True)
        try:
            db.execute("PRAGMA busy_timeout=60000")                       # tolerate the writer's final commit
            for (oz, oy, ox) in emb_dict:
                for sid, ann, x0, x1, y0, y1, z0, z1, conf in db.execute(
                        "SELECT id,annotation,min_x,max_x,min_y,max_y,min_z,max_z,confidence FROM sheets "
                        "WHERE window=?", (window_name(oz, oy, ox),)):
                    try:
                        emb = json.loads(ann).get("emb")
                    except Exception:
                        emb = None
                    if emb is None or len(emb) != D:
                        skipped += 1
                        continue
                    sheet_dict.setdefault((oz, oy, ox), []).append(
                        dict(id=int(sid), emb=np.asarray(emb, np.float32), conf=float(conf or 0.0),
                             lo=(float(x0), float(y0), float(z0)), hi=(float(x1), float(y1), float(z1))))
        finally:
            db.close()
        return emb_dict, sheet_dict, skipped

    # streaming producer: yields affinity rows and fills `corners_meta` + `stats` as a side effect. write_affinities
    # consumes this generator FULLY (executemany) BEFORE it reads `corners`, so corners_meta is complete by then.
    corners_meta: list[tuple] = []
    stats = dict(corners=0, edges=0, skipped=0, t0=time.time())

    def _rows():
        ex = ThreadPoolExecutor(max_workers=feed_workers)
        futures: dict[int, object] = {}
        cache: dict[int, tuple] = {}

        def prefetch(zi: int):
            if 0 <= zi < uz.size:
                zv = int(uz[zi])
                if zv not in futures and zv not in cache:
                    futures[zv] = ex.submit(load_plane, zv)

        def get(zi: int):
            zv = int(uz[zi])
            if zv not in cache:
                fut = futures.pop(zv, None) or ex.submit(load_plane, zv)
                emb_d, sh_d, sk = fut.result()
                cache[zv] = (emb_d, sh_d)
                stats["skipped"] += sk
            return cache[zv]

        buf_items: list = []
        buf_meta: list = []
        cidx = 0

        def run_batch():
            if not buf_items:
                return []
            bt = stitch_collate(buf_items)
            with torch.no_grad():
                out_l = model(bt["Xs"].to(dev), bt["Ps"].to(dev), bt["slot_s"].to(dev),
                              bt["valid_s"].to(dev), bt["Xq"].to(dev), bt["slot_q"].to(dev),
                              bt["valid_q"].to(dev))
            logit = out_l[-1]["logit"].float().cpu().numpy()             # (B,Mmax,Mmax) — last layer (§8)
            out_rows = []
            for bi, (cid, qid) in enumerate(buf_meta):
                M = len(qid)
                Lm = logit[bi, :M, :M]
                prob = 1.0 / (1.0 + np.exp(-Lm))
                for i in range(M):
                    order = np.argsort(-Lm[i])
                    order = order[order != i]
                    if top_k > 0:
                        order = order[:top_k]
                    for j in order:
                        out_rows.append((qid[i], qid[j], float(Lm[i, j]), float(prob[i, j]), cid))
            stats["corners"] += len(buf_meta)
            stats["edges"] += len(out_rows)
            buf_items.clear(); buf_meta.clear()
            return out_rows

        try:
            for zi0 in range(min(readahead + 2, uz.size)):                # prime the feeder pipeline
                prefetch(zi0)
            for zi in range(uz.size - 1):
                prefetch(zi + readahead + 1)
                plane = {-1: get(zi), 1: get(zi + 1)}                     # sz<0 → plane uz[zi]; sz>0 → plane uz[zi+1]
                cz = (uz[zi] + uz[zi + 1]) / 2.0 + half                   # corner z (global L0) for this row
                for yi in range(uy.size - 1):
                    cy = (uy[yi] + uy[yi + 1]) / 2.0 + half
                    for xi in range(ux.size - 1):
                        cx = (ux[xi] + ux[xi + 1]) / 2.0 + half
                        emb_by_oct, sheets_by_oct, ok = {}, {}, True
                        for (sz, sy, sx) in _OCT:
                            o = (int(uz[zi + (sz > 0)]), int(uy[yi + (sy > 0)]), int(ux[xi + (sx > 0)]))
                            e = plane[sz][0].get(o)
                            if e is None:                                # an octant window is missing (air) → skip corner
                                ok = False; break
                            emb_by_oct[(sz, sy, sx)] = e
                            sheets_by_oct[(sz, sy, sx)] = plane[sz][1].get(o, ())
                        if not ok:
                            continue
                        built = _corner_item(emb_by_oct, sheets_by_oct, half=half, h=h, fov=fov, nsp=nsp,
                                             CROP=CROP, POOL=POOL, cz=cz, cy=cy, cx=cx)
                        if built is None:                                # <2 queryable sheets at this corner
                            continue
                        item, qid = built
                        buf_items.append(item)
                        buf_meta.append((cidx, qid))
                        corners_meta.append((cidx, int(round(cz)), int(round(cy)), int(round(cx)), len(qid)))
                        cidx += 1
                        if len(buf_items) >= batch:
                            yield from run_batch()
                            if stats["corners"] and (stats["corners"] // batch) % 20 == 0:
                                print(f"[stitcher] {stats['corners']} corners · {stats['edges']} edges · "
                                      f"{stats['corners'] / max(1e-6, time.time() - stats['t0']):.0f} corner/s",
                                      flush=True)
                cache.pop(int(uz[zi]), None)                              # row done → evict lower plane (zi+1 stays)
            yield from run_batch()                                        # tail batch
        finally:
            ex.shutdown(wait=True)

    n = write_affinities(out_asv, _rows(), corners=corners_meta)
    dt = time.time() - stats["t0"]
    print(f"[stitcher] DONE: wrote {n} affinity rows (both directions) over {stats['corners']} corners "
          f"({stats['skipped']} sheets without a {D}-d median skipped) into {out_asv} in {dt:.0f}s", flush=True)
    return out_asv, n


def infer_run(a) -> None:
    """``hercuvit stitcher infer`` CLI entry — validates the flags and calls :func:`stitch_asv`."""
    if not (a.embeddings_zarr and a.asv):
        raise SystemExit("hercuvit stitcher infer: need --embeddings-zarr and --asv.")
    stitch_asv(a.embeddings_zarr, a.asv, a.stitcher, out=a.out,
               top_k=getattr(a, "top_k", 0), batch=getattr(a, "batch", 16), device=a.device,
               feed_workers=getattr(a, "feed_workers", 3), readahead=getattr(a, "readahead", 2))

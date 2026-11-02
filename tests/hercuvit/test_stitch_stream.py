"""The streamed :func:`hercunet.hercuvit.stitch.infer.stitch_asv` must produce exactly the affinities an eager
whole-array gather would, and must be invariant to the feeder knobs.

The rewrite reads the pooled-embedding zarr and the sheet medians a z-plane at a time (feeder pool + sliding
two-plane cache) instead of loading the whole ``emb`` array and the whole ``sheets`` table into RAM. That is a
pure data-FEEDING change: the per-corner sample assembly (``_corner_item`` → ``corner_spatial_tokens`` →
``stitch_collate``) and the decoder forward are unchanged, so the set of affinity edges must be identical. These
tests pin that: (1) streamed == an independent eager oracle over the same model; (2) streamed is invariant to
``feed_workers``/``readahead`` (guards the cache/prefetch/eviction); (3) the emb rows are stored in SHUFFLED order,
so the plane-grouping can't accidentally depend on storage order.
"""
from __future__ import annotations

import dataclasses
import os
import sqlite3

import numpy as np
import pytest

torch = pytest.importorskip("torch")
zarr = pytest.importorskip("zarr")

from hercunet.hercuvit.stitch.config import StitchConfig
from hercunet.hercuvit.stitch.dataset import stitch_collate
from hercunet.hercuvit.stitch.infer import _OCT, _corner_item, stitch_asv
from hercunet.hercuvit.stitch.model import StitchDecoder
from hercunet.meshes.asv import AsvWriter
from hercunet.meshes.extract import window_name

D = 3
G = 4                                                     # pooled grid per window (emb shape (N,G,G,G,D))
AXIS = (0, 100, 200)                                     # 3 window starts per axis -> a 3x3x3 grid, 2x2x2 corners
OVERLAP = 0.1


def _build_checkpoint(path):
    """A tiny but real StitchDecoder saved exactly as training saves it (weights + asdict config)."""
    cfg = StitchConfig(pca_dim=D, d_model=16, heads=2, layers=2, fourier_bands=4)
    torch.manual_seed(0)
    model = StitchDecoder(cfg)
    torch.save({"model": model.state_dict(), "config": dataclasses.asdict(cfg)}, path)
    return cfg


def _build_artifacts(tmp_path):
    """A synthetic emb zarr + .asv on a 3^3 window grid: one sheet per window (median emb + a bbox wide enough to
    reach every corner it touches). Emb rows are stored in SHUFFLED order to exercise the plane grouping."""
    coords = [(z, y, x) for z in AXIS for y in AXIS for x in AXIS]
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((len(coords), G, G, G, D)).astype(np.float16)
    order = rng.permutation(len(coords))                 # shuffle storage order (origin[n] != sorted)
    coords = [coords[i] for i in order]
    emb = emb[order]

    zpath = str(tmp_path / "emb.zarr")
    zg = zarr.open_group(zpath, mode="w")
    zg.create_dataset("emb", data=emb, chunks=(16, G, G, G, D), dtype="f2")
    zg.create_dataset("origin", data=np.asarray(coords, np.int32), chunks=(8192, 3), dtype="int32")
    zg.attrs["overlap"] = OVERLAP

    apath = str(tmp_path / "meshes.asv")
    w = AsvWriter(apath, frame={"vc_open_data_coordinate_space": "TEST/0"}, voxel_um=1.0, grid_step_um=30.0)
    foot = np.ones((2, 2), bool)
    for (z, y, x) in coords:
        # bbox [o-10, o+120] on each axis -> overlaps every corner cube this window is an octant of
        verts = np.array([[x - 10, y - 10, z - 10], [x + 120, y - 10, z - 10],
                          [x - 10, y + 120, z - 10], [x + 120, y + 120, z + 120]], np.float32)
        med = rng.standard_normal(D).astype(np.float32)
        w.add_sheet(foot, verts, name=f"{window_name(z, y, x)}_s0", window=window_name(z, y, x),
                    annotation={"emb": [float(v) for v in med]})
    w.close()
    return zpath, apath


def _affinity_multiset(asv_path):
    """The written edges as a sortable multiset, IGNORING the corner-id (a provenance counter whose value depends on
    enumeration order); an edge is (src, dst, logit, prob) rounded."""
    db = sqlite3.connect(f"file:{asv_path}?mode=ro", uri=True)
    try:
        rows = db.execute("SELECT src,dst,logit,prob FROM affinities").fetchall()
    finally:
        db.close()
    return sorted((int(s), int(d), round(float(l), 4), round(float(p), 4)) for (s, d, l, p) in rows)


def _eager_oracle(zpath, apath, ckpt, tmp_path):
    """Independent whole-array reference: load the emb grid and ALL sheet medians up front, enumerate every interior
    corner with the SAME geometry + sample builder the streamed path uses, forward the model in one batch, and emit
    the same rows. If the streamed feeding is correct, its edges equal these."""
    from hercunet.hercuvit.stitch.infer import load_stitcher

    model, cfg = load_stitcher(ckpt, device="cpu")
    model.eval()
    g = zarr.open(zpath, mode="r")
    emb_all = np.asarray(g["emb"][:]).astype(np.float32)
    origin = np.asarray(g["origin"][:])
    orig_key = {tuple(int(v) for v in origin[n]): n for n in range(origin.shape[0])}
    uz, uy, ux = (np.unique(origin[:, i]) for i in range(3))
    diffs = np.concatenate([np.diff(uz), np.diff(uy), np.diff(ux)])
    stride = float(np.median(diffs))
    fov = stride / (1.0 - OVERLAP); nsp = fov / 128.0; h = stride / 2.0; half = fov / 2.0
    CROP, POOL = 128, 128 // G

    by_win: dict[tuple, list] = {}
    db = sqlite3.connect(f"file:{apath}?mode=ro", uri=True)
    import json
    for sid, win, ann, x0, x1, y0, y1, z0, z1, conf in db.execute(
            "SELECT id,window,annotation,min_x,max_x,min_y,max_y,min_z,max_z,confidence FROM sheets"):
        emb = json.loads(ann).get("emb")
        from hercunet.hercuvit.stitch.infer import _parse_window
        by_win.setdefault(_parse_window(win), []).append(
            dict(id=int(sid), emb=np.asarray(emb, np.float32), conf=float(conf or 0.0),
                 lo=(float(x0), float(y0), float(z0)), hi=(float(x1), float(y1), float(z1))))
    db.close()

    items, metas = [], []
    cidx = 0
    for zi in range(uz.size - 1):
        cz = (uz[zi] + uz[zi + 1]) / 2.0 + half
        for yi in range(uy.size - 1):
            cy = (uy[yi] + uy[yi + 1]) / 2.0 + half
            for xi in range(ux.size - 1):
                cx = (ux[xi] + ux[xi + 1]) / 2.0 + half
                emb_by_oct, sheets_by_oct, ok = {}, {}, True
                for (sz, sy, sx) in _OCT:
                    o = (int(uz[zi + (sz > 0)]), int(uy[yi + (sy > 0)]), int(ux[xi + (sx > 0)]))
                    n = orig_key.get(o)
                    if n is None:
                        ok = False; break
                    emb_by_oct[(sz, sy, sx)] = emb_all[n]
                    sheets_by_oct[(sz, sy, sx)] = by_win.get(o, ())
                if not ok:
                    continue
                built = _corner_item(emb_by_oct, sheets_by_oct, half=half, h=h, fov=fov, nsp=nsp,
                                     CROP=CROP, POOL=POOL, cz=cz, cy=cy, cx=cx)
                if built is None:
                    continue
                item, qid = built
                items.append(item); metas.append((cidx, qid)); cidx += 1

    rows = []
    bt = stitch_collate(items)
    with torch.no_grad():
        out_l = model(bt["Xs"], bt["Ps"], bt["slot_s"], bt["valid_s"], bt["Xq"], bt["slot_q"], bt["valid_q"])
    logit = out_l[-1]["logit"].float().numpy()
    for bi, (cid, qid) in enumerate(metas):
        M = len(qid)
        Lm = logit[bi, :M, :M]
        prob = 1.0 / (1.0 + np.exp(-Lm))
        for i in range(M):
            order = np.argsort(-Lm[i])
            order = order[order != i]
            for j in order:
                rows.append((int(qid[i]), int(qid[j]), round(float(Lm[i, j]), 4), round(float(prob[i, j]), 4)))
    return sorted(rows)


def test_streamed_matches_eager_oracle(tmp_path):
    ckpt = str(tmp_path / "stitcher.pth")
    _build_checkpoint(ckpt)
    zpath, apath = _build_artifacts(tmp_path)

    out = str(tmp_path / "streamed.asv")
    _, n = stitch_asv(zpath, apath, ckpt, out=out, device="cpu", batch=4, feed_workers=3, readahead=2)

    streamed = _affinity_multiset(out)
    eager = _eager_oracle(zpath, apath, ckpt, tmp_path)
    assert n == len(streamed) > 0
    assert streamed == eager                                       # same edges, same logits — feeding change only

    # both directions of every pair are present (the table stores (A,B) and (B,A))
    pairs = {(s, d) for (s, d, _, _) in streamed}
    assert all((d, s) in pairs for (s, d) in pairs)


def test_streamed_invariant_to_feeder_knobs(tmp_path):
    ckpt = str(tmp_path / "stitcher.pth")
    _build_checkpoint(ckpt)
    zpath, apath = _build_artifacts(tmp_path)

    out_a = str(tmp_path / "a.asv")
    out_b = str(tmp_path / "b.asv")
    stitch_asv(zpath, apath, ckpt, out=out_a, device="cpu", batch=1, feed_workers=1, readahead=0)
    stitch_asv(zpath, apath, ckpt, out=out_b, device="cpu", batch=8, feed_workers=4, readahead=3)
    assert _affinity_multiset(out_a) == _affinity_multiset(out_b)


def test_stitch_leaves_no_wal_sidecar(tmp_path):
    """The medians are read up front and the .asv is written by a single writer with no concurrent reader, so the
    output stays a self-contained single file — no ``-wal``/``-shm`` sidecar left beside it to be missed on upload."""
    ckpt = str(tmp_path / "stitcher.pth")
    _build_checkpoint(ckpt)
    zpath, apath = _build_artifacts(tmp_path)
    out = str(tmp_path / "idx.asv")
    stitch_asv(zpath, apath, ckpt, out=out, device="cpu")
    assert not os.path.exists(out + "-wal") and not os.path.exists(out + "-shm")
    db = sqlite3.connect(f"file:{out}?mode=ro", uri=True)
    try:
        assert int(db.execute("SELECT COUNT(*) FROM affinities").fetchone()[0]) > 0
    finally:
        db.close()

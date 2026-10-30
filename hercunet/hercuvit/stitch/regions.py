"""Region-based pseudo-label extraction (efficient, local).

The per-corner streaming path is latency-bound (thousands of tiny chunk GETs) and leaves the GPU ~idle — the wrong
tool for scale. Instead: per volume, download a few BIG regions ONCE to a local **compressed zarr** (uint8, blosc —
not the opaque chunk cache, not decompressed npy), load each into RAM, and extract every corner by **slicing windows
from RAM**. That removes all per-corner network latency, so the GPU is actually busy. :class:`RegionReader` is a
drop-in for :class:`ScrollCtReader`, so :func:`hercunet.hercuvit.stitch.labels.extract_corner` is reused unchanged.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

from .labels import CROP, OCT


class _Meta:
    def __init__(self, vu):
        self.voxel_size_um = float(vu)


class _Vol:
    def __init__(self, vu):
        self.meta = _Meta(vu)


class RegionReader:
    """Drop-in for ``ScrollCtReader`` backed by an in-RAM region array (uint8, level-0) at global ``origin`` (z,y,x).
    ``__call__(scroll, level, org, shape)`` slices the window (``org`` = CENTRE, same convention as ScrollCtReader)
    straight from RAM — no network. Windows are sampled with a margin so they never run off the region."""

    def __init__(self, scroll, array: np.ndarray, origin, voxel_um: float) -> None:
        self.scroll = str(scroll)
        self.arr = array
        self.oz, self.oy, self.ox = (int(o) for o in origin)
        self._m = _Vol(voxel_um)

    def _vol(self, scroll):
        return self._m

    def __call__(self, scroll, level, org, shape) -> np.ndarray:
        zc, yc, xc = (int(o) for o in org)
        dz, dy, dx = (int(s) for s in shape)
        z0, y0, x0 = zc - dz // 2 - self.oz, yc - dy // 2 - self.oy, xc - dx // 2 - self.ox
        a = self.arr
        blk = a[max(0, z0):z0 + dz, max(0, y0):y0 + dy, max(0, x0):x0 + dx]
        if blk.shape != (dz, dy, dx):                                    # margin guarantees this is rare (edge corners)
            out = np.zeros((dz, dy, dx), a.dtype)
            out[:blk.shape[0], :blk.shape[1], :blk.shape[2]] = blk
            blk = out
        return blk


def download_region(scroll, bbox, *, save_dir: str | None = None) -> tuple[np.ndarray, tuple, float]:
    """Read a big region (``bbox = (z0,z1,y0,y1,x0,x1)`` at L0) from the remote volume ONCE into a RAM uint8 array.
    Optionally persist it as a local compressed zarr (blosc-zstd, 128³ chunks) — the safe, compressed local copy the
    extraction reads from. Returns ``(array, origin(z0,y0,x0), voxel_um)``."""
    from ...infer.nnunet_infer import open_source
    v = open_source(str(scroll))
    vu = float(v.meta.voxel_size_um)
    z0, z1, y0, y1, x0, x1 = (int(b) for b in bbox)
    t = time.time()
    blk, _ = v.read_window(0, z0, z1, y0, y1, x0, x1)
    arr = np.ascontiguousarray(np.asarray(blk), np.uint8)
    dt = time.time() - t
    print(f"[region] {scroll} {arr.shape} {arr.nbytes / 1e6:.0f}MB read in {dt:.1f}s "
          f"({arr.nbytes / 1e6 / max(dt, 1e-6):.0f} MB/s)", flush=True)
    if save_dir:
        import zarr
        import numcodecs
        os.makedirs(save_dir, exist_ok=True)
        zp = os.path.join(save_dir, f"{scroll}_z{z0}_y{y0}_x{x0}.zarr")
        zarr.save_array(zp, arr, chunks=(128, 128, 128), compressor=numcodecs.Blosc("zstd", clevel=5, shuffle=1))
        comp = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(zp) for f in fs)
        print(f"[region] saved {zp}  ({comp / 1e6:.0f}MB compressed, {arr.nbytes / max(comp, 1):.1f}×)", flush=True)
    return arr, (z0, y0, x0), vu


def sample_region_corners(arr, origin, voxel_um, target_um, *, n, himat=0.15, overlap=0.1, min_dist=None,
                          exclude=None, seed=0):
    """Sample up to ``n`` corner coords (global z,y,x) inside the region whose 9 windows all fit, material-gated
    (herculabels-style: fraction of the centred window above the region's 55th-percentile threshold ≥ ``himat``)
    and **spaced ≥ ``min_dist``** apart so their windows don't overlap — default ``min_dist = fov`` (adjacent
    corners' central windows just touch, no overlap). ``exclude`` = global (z,y,x) coords of ALREADY-extracted
    corners (any scroll's full set); a candidate within ``min_dist`` of one is rejected, so a new run never repeats
    or overlaps a prior one. Returns a list of ``(global_coord, material_frac)``."""
    fov = int(round(CROP * target_um / voxel_um))
    core = int(round(CROP * (1 - overlap) * target_um / voxel_um))
    reach = core // 2 + fov // 2 + 2                                    # a window reaches coord ± (h + fov/2)
    md = int(fov if min_dist is None else min_dist)
    oz, oy, ox = origin
    Z, Y, X = arr.shape
    lo = np.array([reach, reach, reach])
    hi = np.array([Z - reach, Y - reach, X - reach])
    if np.any(hi <= lo):
        return []
    thr = float(np.percentile(arr[::4, ::4, ::4], 55))                 # material threshold (subsampled, fast)
    rng = np.random.default_rng(seed)
    acc = np.empty((0, 3), np.int64)                                   # accepted LOCAL coords (for the spacing check)
    if exclude is not None and len(exclude):                          # seed the spacing check with prior corners (local)
        loc = np.asarray(exclude, np.int64) - np.array([oz, oy, ox])
        inb = ((loc >= -md) & (loc <= np.array([Z, Y, X]) + md)).all(1)   # only those near this region can block
        if inb.any():
            acc = loc[inb]                                            # never emitted, only blocks candidates
    out, tries, cap = [], 0, n * 60
    while len(out) < n and tries < cap:
        tries += 1
        p = rng.integers(lo, hi)                                        # local index
        if acc.shape[0] and int(((acc - p) ** 2).sum(1).min()) < md * md:   # too close to an accepted corner
            continue
        w = arr[p[0] - fov // 2:p[0] + fov // 2, p[1] - fov // 2:p[1] + fov // 2, p[2] - fov // 2:p[2] + fov // 2]
        frac = float((w > thr).mean()) if w.size else 0.0
        if frac >= himat:
            acc = np.vstack([acc, p])
            out.append(((int(p[0] + oz), int(p[1] + oy), int(p[2] + ox)), frac))
    return out


# --------------------------------------------------------------------- region placement + the driver --
_STITCH_RX = None


def _load_exclusions(dirs):
    """Parse ``stitch_s<scroll>_z<z>_y<y>_x<x>.npz`` filenames under each dir into ``{scroll: (N,3) int64 coords}`` —
    the corners already extracted, which a new run must avoid. ``dirs`` may be None, a str, or a list of dirs."""
    import glob
    import re
    if not dirs:
        return {}
    if isinstance(dirs, str):
        dirs = [dirs]
    rx = re.compile(r"stitch_s(.+?)_z(\d+)_y(\d+)_x(\d+)\.npz$")
    acc: dict[str, list] = {}
    for d in dirs:
        for fp in glob.glob(os.path.join(d, "stitch_s*.npz")):
            m = rx.search(os.path.basename(fp))
            if m:
                acc.setdefault(m.group(1), []).append((int(m.group(2)), int(m.group(3)), int(m.group(4))))
    return {s: np.asarray(v, np.int64) for s, v in acc.items()}


def place_regions(be, scroll, side, n, rng):
    """Place up to ``n`` region bboxes (z0,z1,y0,y1,x0,x1) of edge ``side`` spread across the scroll's material bbox,
    min-spaced ~``side`` so they sample DIFFERENT parts of the volume (sparse, not one blob). Reuses herculabels'
    ``scroll_geom`` for the (voxel, shape, XY material bbox)."""
    from ...herculabels._brick import scroll_geom
    vu, (Z, Y, X), (by0, by1, bx0, bx1) = scroll_geom(be, scroll, 0, {})
    hs = side // 2
    lo = np.array([hs, max(by0 + hs, hs), max(bx0 + hs, hs)])
    hi = np.array([Z - hs, min(by1 - hs, Y - hs), min(bx1 - hs, X - hs)])
    if np.any(hi <= lo):
        return [], float(vu)
    acc, out, tries = [], [], 0
    while len(out) < n and tries < n * 40:
        tries += 1
        c = rng.integers(lo, hi)
        if acc and float(((np.array(acc) - c) ** 2).sum(1).min()) ** 0.5 < side * 0.8:   # keep regions apart
            continue
        acc.append(tuple(int(v) for v in c))
        out.append((c[0] - hs, c[0] + hs, c[1] - hs, c[1] + hs, c[2] - hs, c[2] + hs))
    return out, float(vu)


def create_run(a):
    """``hercuvit labels create --regions`` driver: per volume, place + download a few regions, extract non-overlapping
    himat corners from RAM (batched), write one npz per label + a provenance manifest (JSONL). Freed region-by-region.
    Sharded across scrolls by ``--shard/--nshards`` for a multi-GPU box."""
    import json
    import shutil
    import time

    import torch

    from ...config import Config
    from ...data import get_backend
    from ...herculabels._brick import find_scroll, in_band_scrolls
    from ...herculabels.create import _resolve_scrolls
    from ..infer.loader import load_hercuvit
    from .labels import extract_corner

    dev = torch.device(a.device)
    net, recipe, _ = load_hercuvit(model=a.model, model_hf=a.model_hf, ckpt=a.ckpt, device=a.device, with_instance=True)
    target_um = float(recipe.target_voxel_um)
    os.makedirs(a.out, exist_ok=True)
    cache = a.region_cache or os.path.join(a.out, "_regions")
    os.makedirs(cache, exist_ok=True)

    be = get_backend(Config.from_env())
    scrolls = _resolve_scrolls(be, a.scroll, 7.5, 9.5, find_scroll, in_band_scrolls)
    sids = sorted(scrolls)[a.shard::a.nshards]                          # shard by scroll across GPUs
    per_vol = max(1, a.count // max(1, len(sorted(scrolls))))
    rng = np.random.default_rng((a.seed or 0) + a.shard)
    man = open(os.path.join(a.out, f"manifest_{a.shard}.jsonl"), "a")
    exclmap = _load_exclusions(getattr(a, "exclude_dir", None))       # {scroll: (N,3) coords} of already-extracted corners
    print(f"[regions] shard {a.shard}/{a.nshards}: {len(sids)} scrolls, target {per_vol}/scroll "
          f"({a.count} total), region {a.region_side}³, infer_batch {a.infer_batch}"
          f"{f', excluding {sum(len(v) for v in exclmap.values())} prior corners' if exclmap else ''} -> {a.out}", flush=True)

    total, t0 = 0, time.time()
    for sid in sids:
        got = 0
        bboxes, vu = place_regions(be, sid, a.region_side, a.regions_per_volume, rng)
        for bbox in bboxes:
            if got >= per_vol:
                break
            try:
                arr, origin, vu = download_region(sid, bbox, save_dir=cache)
            except Exception as e:
                print(f"[regions] {sid} region download FAILED: {repr(e)[:140]}", flush=True)
                continue
            reader = RegionReader(sid, arr, origin, vu)
            want = min(a.per_region, per_vol - got)
            himat = a.himat if a.himat is not None else 0.15          # herculabels default
            for coord, frac in sample_region_corners(arr, origin, vu, target_um, n=want, himat=himat,
                                                     overlap=a.overlap, min_dist=a.min_dist,
                                                     exclude=exclmap.get(str(sid)),
                                                     seed=int(rng.integers(1 << 31))):
                try:
                    s = extract_corner(net, reader, sid, coord, target_um, dev, D=a.pca_dim, overlap=a.overlap,
                                       min_vote=a.min_vote, min_pts=a.min_pts, infer_batch=a.infer_batch)
                except Exception as e:
                    print(f"[regions] corner {coord} FAILED: {repr(e)[:120]}", flush=True)
                    continue
                if s is None:
                    continue
                tag = f"s{sid}_z{coord[0]}_y{coord[1]}_x{coord[2]}"
                np.savez_compressed(os.path.join(a.out, f"stitch_{tag}.npz"), **s)
                Y = s["Y"]; iu = np.triu_indices(Y.shape[0], 1); yy = Y[iu]
                man.write(json.dumps(dict(npz=f"stitch_{tag}.npz", scroll=str(sid), region=[int(b) for b in bbox],
                                          coord=[int(c) for c in coord], himat=round(float(frac), 3),
                                          M=int(len(s["slot_q"])), npos=int((yy == 1).sum()),
                                          nneg=int((yy == 0).sum()), D=int(a.pca_dim))) + "\n")
                man.flush()
                got += 1; total += 1
            del arr, reader
            if not a.keep_regions:                                     # free the region (disk tight; corpus is the keeper)
                shutil.rmtree(os.path.join(cache, f"{sid}_z{bbox[0]}_y{bbox[2]}_x{bbox[4]}.zarr"), ignore_errors=True)
            print(f"[regions] {sid}: {got}/{per_vol}  ({total} total, {(time.time()-t0)/max(total,1):.1f}s/label)", flush=True)
            if a.count and total >= a.count:
                break
        if a.count and total >= a.count:
            break
    man.close()
    print(f"[regions] DONE shard {a.shard}: {total} labels written in {time.time()-t0:.0f}s", flush=True)

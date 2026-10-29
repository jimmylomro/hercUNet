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


def sample_region_corners(arr, origin, voxel_um, target_um, *, n, himat=0.15, overlap=0.1, min_dist=None, seed=0):
    """Sample up to ``n`` corner coords (global z,y,x) inside the region whose 9 windows all fit, material-gated
    (herculabels-style: fraction of the centred window above the region's 55th-percentile threshold ≥ ``himat``)
    and **spaced ≥ ``min_dist``** apart so their windows don't overlap — default ``min_dist = fov`` (adjacent
    corners' central windows just touch, no overlap). Returns a list of ``(global_coord, material_frac)``."""
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

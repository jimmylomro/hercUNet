"""Reusable material-aware tiling over a :class:`SegmentVolume` — the shared "smart" read layer.

Every consumer (tractability maps, seam assembly, the 3-D brick pipeline) needs the same thing: find
where the material is, skip air, and stream the tiles/bricks of a region at full resolution with I/O
overlapped by compute. That logic lives HERE so it is written once:

  * :func:`locate_material` — cheap coarse read (the pyramid's purpose) → material bbox + a coarse
    mask + scale factors, so callers know where to work WITHOUT downloading the air corners.
  * :func:`iter_material_tiles` — yields the full-resolution tiles overlapping material, prefetched
    (via :func:`prefetch.iter_windows`) so the GPU never idles on downloads. The chunk cache
    underneath means neighbouring tiles / depths / re-runs are free.
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass

import numpy as np

from .prefetch import iter_windows

Tile = namedtuple("Tile", "wy wx ry0 rx0 oz block")  # placement (wy,wx), haloed read origin, depth off, data


@dataclass
class MaterialRegion:
    bbox: tuple            # (bx0, bx1, by0, by1) at the working level
    mask: np.ndarray       # coarse material mask (bool)
    scales: tuple          # (syr, sxr): working-level px per coarse px
    coarse_slice: np.ndarray
    coarse_level: int


def _otsu(a: np.ndarray) -> float:
    v = a[a > 0]
    if v.size == 0:
        return 0.0
    hist, edges = np.histogram(v, bins=256)
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    denom = omega * (1 - omega)
    denom[denom == 0] = 1e-12
    sigma_b = (mu[-1] * omega - mu) ** 2 / denom
    return float(edges[int(np.argmax(sigma_b))])


def locate_material(vol, level: int, z: int, bbox_levels: int = 2):
    """Material bbox at ``level`` for depth ``z``, found from ONE cheap coarse slice (level +
    ``bbox_levels``). Returns a :class:`MaterialRegion` (or None if the slice is empty). This is
    location only — never the computation, which the caller does at full ``level``."""
    m = vol.meta
    Z, Y, X = m.level_shapes[level]
    Lc = min(m.num_levels - 1, level + bbox_levels)
    Zc, Yc, Xc = m.level_shapes[Lc]
    zcc = int(np.clip(z * Zc / Z, 0, Zc - 1))
    slc, _ = vol.read_window(Lc, zcc, zcc + 1, 0, Yc, 0, Xc)
    slc = slc[0]
    mask = slc > _otsu(slc)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    syr, sxr = Y / Yc, X / Xc
    bbox = (int(xs.min() * sxr), int(xs.max() * sxr), int(ys.min() * syr), int(ys.max() * syr))
    return MaterialRegion(bbox=bbox, mask=mask, scales=(syr, sxr), coarse_slice=slc, coarse_level=Lc)


def iter_material_tiles(vol, level, z0, z1, region: MaterialRegion, tile=512, halo=8,
                        min_material=0.02, readahead=4, workers=4):
    """Yield full-resolution :class:`Tile`s covering the material of ``region`` at ``level`` over the
    depth slab ``[z0, z1)``. Air tiles are skipped up front on the coarse mask (no download at all);
    the survivors are streamed with read-ahead so downloads overlap the caller's per-tile compute."""
    bx0, bx1, by0, by1 = region.bbox
    syr, sxr = region.scales
    matc = region.mask
    _, Y, X = vol.meta.level_shapes[level]

    def has_material(wy, wx):
        cy0, cy1 = int(wy / syr), int(min(by1, wy + tile) / syr) + 1
        cx0, cx1 = int(wx / sxr), int(min(bx1, wx + tile) / sxr) + 1
        sub = matc[cy0:cy1, cx0:cx1]
        return sub.size > 0 and sub.mean() > min_material

    metas, reqs = [], []
    for wy in range(by0, by1, tile):
        for wx in range(bx0, bx1, tile):
            if not has_material(wy, wx):
                continue
            ry0, ry1 = max(0, wy - halo), min(Y, wy + tile + halo)
            rx0, rx1 = max(0, wx - halo), min(X, wx + tile + halo)
            metas.append((wy, wx, ry0, rx0))
            reqs.append((level, z0, z1, ry0, ry1, rx0, rx1))
    for idx, (_req, blk, org) in enumerate(iter_windows(vol, reqs, readahead=readahead, workers=workers)):
        wy, wx, ry0, rx0 = metas[idx]
        yield Tile(wy=wy, wx=wx, ry0=ry0, rx0=rx0, oz=int(org[0]), block=blk)


def n_material_tiles(vol, level, region: MaterialRegion, tile=512, min_material=0.02) -> int:
    """Count of material tiles (for progress) — mirrors the skip logic in :func:`iter_material_tiles`."""
    bx0, bx1, by0, by1 = region.bbox
    syr, sxr = region.scales
    matc = region.mask
    n = 0
    for wy in range(by0, by1, tile):
        for wx in range(bx0, bx1, tile):
            cy0, cy1 = int(wy / syr), int(min(by1, wy + tile) / syr) + 1
            cx0, cx1 = int(wx / sxr), int(min(bx1, wx + tile) / sxr) + 1
            sub = matc[cy0:cy1, cx0:cx1]
            if sub.size > 0 and sub.mean() > min_material:
                n += 1
    return n

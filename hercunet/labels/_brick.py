"""Scroll resolution, window sampling, and the cleaned brick.

``build_brick`` reads a window from a scroll volume (via the data layer) and coherence-enhancing-
diffuses it into the ``bced`` substrate the pipeline runs on. It does the CED directly (the research
pipeline routed it through a per-slice tracing step whose output was discarded anyway), keeping the
experimental streamline/layer-potential subtree out of the label pipeline.
"""

from __future__ import annotations

import time

import numpy as np

from ..data.tiling import locate_material


def find_scroll(be, name: str):
    """Resolve a scroll id (e.g. ``PHerc1447`` or ``1``) to its :class:`ScrollInfo` via the backend."""
    key = name.lower().replace(" ", "")
    return next(s for s in be.list_scrolls()
                if s.extra.get("fragment") is None
                and (f"Scroll {name}" in s.name or key in s.name.lower().replace(" ", "")
                     or key in str(s.extra.get("pherc", "")).lower()))


def in_band_scrolls(be, vmin: float, vmax: float, restrict=None) -> dict:
    """In-band FULL scrolls from one cached ``list_scrolls()`` — pure in-memory voxel-size filter, no
    volume opens. Returns ``{scroll_id: voxel_um}``. ``restrict`` optionally limits to a set of ids."""
    out, seen = {}, set()
    for s in be.list_scrolls():
        sid = s.scroll_id
        if s.extra.get("fragment") is not None or sid in seen:
            continue
        seen.add(sid)
        if restrict and sid not in restrict:
            continue
        vu = s.resolution_um
        if vu is None or not (vmin <= vu <= vmax):
            continue
        out[sid] = float(vu)
    return out


def scroll_geom(be, sid: str, level: int, cache: dict):
    """Lazy per-scroll geometry (voxel, shape, XY material bbox) so window centres land on papyrus,
    not air corners. Opens the volume once (~7 s) and caches. Returns (voxel_um, (Z,Y,X), (by0,by1,bx0,bx1))."""
    if sid in cache:
        return cache[sid]
    vol = be.open_scroll_volume(find_scroll(be, sid))
    m = vol.meta
    L = int(np.clip(level, 0, m.num_levels - 1))
    Z, Y, X = m.level_shapes[L]
    reg = locate_material(vol, L, Z // 2, bbox_levels=2)
    if reg is not None:
        bx0, bx1, by0, by1 = reg.bbox
        bbox = (int(by0), int(by1), int(bx0), int(bx1))
    else:
        bbox = (0, int(Y), 0, int(X))
    geom = (float(m.voxel_size_um), (int(Z), int(Y), int(X)), bbox)
    cache[sid] = geom
    return geom


def sample_windows(be, scrolls: dict, *, level=0, margin_um=900.0, seed=0):
    """Yield ``(scroll_id, (z, y, x))`` picks indefinitely from a seeded stream over ``scrolls``
    ({id: voxel_um}), each centre inside the scroll's material bbox and ``margin_um`` from the faces.
    The caller stops when it has enough non-air windows (air is only known after the brick is built)."""
    rng = np.random.default_rng(seed)
    geom_cache: dict = {}
    tbl = dict(scrolls)
    while tbl:
        sid = str(rng.choice(list(tbl)))
        try:
            vu, (Z, Y, X), (by0, by1, bx0, bx1) = scroll_geom(be, sid, level, geom_cache)
        except Exception as e:
            print(f"[skip-scroll] {sid}: geom failed {type(e).__name__}: {str(e)[:60]} — dropping", flush=True)
            tbl.pop(sid, None)
            continue
        mpx = margin_um / vu
        zc = int(rng.integers(mpx, max(mpx + 1, Z - mpx)))
        ylo, yhi = by0 + mpx, by1 - mpx
        xlo, xhi = bx0 + mpx, bx1 - mpx
        yc = int(rng.integers(ylo, max(ylo + 1, yhi))) if yhi > ylo else int((by0 + by1) // 2)
        xc = int(rng.integers(xlo, max(xlo + 1, xhi))) if xhi > xlo else int((bx0 + bx1) // 2)
        yield sid, (zc, yc, xc)


def build_brick(be, target, coords, *, level=0, depth_um=1500.0, gpu=True) -> dict:
    """Read the window around ``coords`` from ``target``'s volume and CED-clean it to the ``bced``
    substrate. Returns ``{bced, voxel_um, org}``."""
    from hercunet.labels.fields.anisotropic import coherence_enhancing_diffusion
    from hercunet.labels.common.scales import mesh_params, window_px, window_z_px

    vol = be.open_scroll_volume(target)
    m = vol.meta
    L = int(np.clip(level, 0, m.num_levels - 1))
    Z, Y, X = m.level_shapes[L]
    voxel_um = m.voxel_size_um * (m.level_shapes[0][0] / Z)
    S, margin_px = window_px(voxel_um), int(round(120.0 / voxel_um))
    n_slices = max(1, int(round(depth_um / voxel_um)))
    Sz = max(window_z_px(voxel_um), n_slices + 2 * margin_px)
    zc, yc, xc = coords
    z0 = int(np.clip(zc - Sz // 2, 0, max(0, Z - Sz)))
    y0 = int(np.clip(yc - S // 2, 0, max(0, Y - S)))
    x0 = int(np.clip(xc - S // 2, 0, max(0, X - S)))
    t = time.time()
    block, _ = vol.read_window(L, z0, z0 + Sz, y0, y0 + S, x0, x0 + S)
    mp = mesh_params(voxel_um)
    bced = coherence_enhancing_diffusion(block, iters=mp["ced_iters"], mode=mp["ced_mode"],
                                         sigma_tensor=mp["sigma_tensor"], recompute_every=0)
    print(f"[brick] {target.name} L{L} vox≈{voxel_um:.2f}µm block {Sz}×{S}×{S} centre z{zc} y{yc} x{xc} "
          f"({time.time()-t:.0f}s)", flush=True)
    return dict(bced=np.ascontiguousarray(bced, np.float32), voxel_um=voxel_um, org=(zc, yc, xc),
                corner=(z0, y0, x0))                             # block origin in GLOBAL scroll voxels


_MASK_CACHE: dict = {}                                            # scroll_id -> (mask5, l0_per_l5, (Z5,Y5,X5))


def scroll_material_mask(be, target, *, mask_level=5):
    """Read the whole scroll at a COARSE pyramid level ONCE and threshold it into a material mask (dense
    voxels vs air), the fast way the m7 himat scout finds material (scripts/mine_m7_labels.py). Cached per
    scroll. With it, a window's material fill is an instant numpy slice-mean — no full-res read per window."""
    key = target.scroll_id
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]
    vol = be.open_scroll_volume(target)
    m = vol.meta
    L = int(np.clip(mask_level, 0, m.num_levels - 1))
    Z5, Y5, X5 = m.level_shapes[L]
    ct5, _ = vol.read_window(L, 0, Z5, 0, Y5, 0, X5)             # coarse level → tiny read
    ct5 = np.asarray(ct5)
    thr = float(np.percentile(ct5, 55))                         # material = above the scroll's 55th percentile
    mask = np.ascontiguousarray(ct5 > thr)
    fac = float(m.level_shapes[0][0] / Z5)                      # L0 voxels per coarse voxel (~2**mask_level)
    out = (mask, fac, (int(Z5), int(Y5), int(X5)))
    _MASK_CACHE[key] = out
    return out


def window_material_frac(be, target, coords, *, mask_level=5):
    """Material fill fraction of the window centred at ``coords`` (L0 voxels), estimated from the cached
    coarse mask — instant, no full-res read. Same window footprint as :func:`build_brick`."""
    from hercunet.labels.common.scales import window_px, window_z_px
    mask, fac, (Z5, Y5, X5) = scroll_material_mask(be, target, mask_level=mask_level)
    voxel_um = be.open_scroll_volume(target).meta.voxel_size_um
    Sxy, Sz = window_px(voxel_um), window_z_px(voxel_um)
    zc, yc, xc = coords

    def rng(c, s, n5):
        a = int(np.clip((c - s / 2) / fac, 0, n5))
        b = int(np.clip((c + s / 2) / fac, 0, n5))
        return a, max(a + 1, b)

    z0, z1 = rng(zc, Sz, Z5)
    y0, y1 = rng(yc, Sxy, Y5)
    x0, x1 = rng(xc, Sxy, X5)
    sub = mask[z0:z1, y0:y1, x0:x1]
    return float(sub.mean()) if sub.size else 0.0

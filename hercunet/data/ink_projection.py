"""Project a segment's 2-D ink prediction back into native scroll-volume coordinates.

The inverse of unwrapping: a ``.tifxyz`` mesh stores, for every surface pixel, its (x, y, z)
voxel coordinate in a named scroll volume. Sampling the segment's ink raster on that same
surface grid gives, for each inked pixel, exactly where the ink sits in the raw 3-D volume —
so ink can be examined against the native fiber structure instead of a flattened render.

We deliberately use the ``.tifxyz`` whose scan-id matches the volume being displayed, so the
coordinates are already in that volume's voxel grid (no cross-scan registration needed).
"""

from __future__ import annotations

import io
import re

import numpy as np

_BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"


def _list(prefix: str) -> list[str]:
    import requests

    url = f"{_BUCKET}/?list-type=2&delimiter=/&prefix={prefix}"
    r = requests.get(url, timeout=30)
    return re.findall(r"<Prefix>([^<]+)</Prefix>", r.text) if r.status_code == 200 else []


def _keys(prefix: str) -> list[str]:
    import requests

    url = f"{_BUCKET}/?list-type=2&prefix={prefix}"
    r = requests.get(url, timeout=30)
    return re.findall(r"<Key>([^<]+)</Key>", r.text) if r.status_code == 200 else []


def _tif(url: str) -> np.ndarray:
    import requests
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    return np.asarray(Image.open(io.BytesIO(requests.get(url, timeout=180).content)))


def choose_aligned_scan(seg_prefix: str):
    """Pick the finest scan whose *volume actually contains* this segment's mesh.

    A segment ships a ``.tifxyz`` per scan; the matching ``/volumes/`` zarr is the frame that
    mesh's coordinates live in — but some are cropped derivatives that don't contain the whole
    segment (mesh bbox exceeds the volume). We keep only scans whose volume dims contain the
    mesh bbox, then take the finest voxel. Returns ``(scan_id, volume_url, voxel_um)`` or None.
    """
    import json

    import requests

    rel = seg_prefix[len(_BUCKET) + 1:].rstrip("/")
    pherc = rel.split("/segments/")[0]
    meshes = {}
    for m in _list(rel + "/mesh/"):
        mm = m.rstrip("/")
        if not mm.endswith(".tifxyz"):
            continue
        sid = re.search(r"-on-(\d{14})-", mm.split("/")[-1])
        if sid:
            meshes[sid.group(1)] = mm
    volumes = {}
    for v in _list(pherc + "/volumes/"):
        vv = v.rstrip("/")
        if not vv.endswith(".zarr"):
            continue
        sid = re.match(r"(\d{14})", vv.split("/")[-1])
        if sid:
            volumes.setdefault(sid.group(1), vv)  # one masked volume per scan

    best = None
    for sid, mesh in meshes.items():
        vv = volumes.get(sid)
        if vv is None:
            continue
        try:
            bb = json.loads(requests.get(f"{_BUCKET}/{mesh}/meta.json", timeout=30).text)["bbox"]
            shp = json.loads(requests.get(f"{_BUCKET}/{vv}/0/.zarray", timeout=30).text)["shape"]
        except (requests.RequestException, KeyError, ValueError):
            continue
        z, y, x = shp[0], shp[1], shp[2]
        if bb[1][2] <= z and bb[1][1] <= y and bb[1][0] <= x:  # mesh bbox fits the volume
            m = re.search(r"(\d+(?:\.\d+)?)\s*um", vv.split("/")[-1])
            voxel = float(m.group(1)) if m else 1e9
            if best is None or voxel < best[2]:
                best = (sid, f"{_BUCKET}/{vv}", voxel)
    return best


def load_zmap(seg_prefix: str, scan_id: str, max_side: int = 1600):
    """Return the segment's per-surface-pixel scroll-z map (from ``z.tif`` of the ``scan_id``
    mesh), downsampled, with invalid (-1) pixels as NaN. Used to draw the "current scroll
    depth" iso-line on the flattened segment. Returns a 2-D float32 array or None."""
    rel = seg_prefix[len(_BUCKET) + 1:].rstrip("/")
    meshes = [m for m in _list(rel + "/mesh/") if scan_id in m and m.rstrip("/").endswith(".tifxyz")]
    if not meshes:
        return None
    z = _tif(f"{_BUCKET}/{meshes[0].rstrip('/')}/z.tif").astype(np.float32)
    z[z <= 0] = np.nan
    step = max(1, -(-max(z.shape) // max_side))  # ceil-downsample the longest side to ~max_side
    return z[::step, ::step] if step > 1 else z


def load_xyz(seg_prefix: str, scan_id: str, max_side: int = 1600):
    """Return (x, y, z) per-surface-pixel scroll-voxel maps for the ``scan_id`` mesh,
    downsampled, invalid (-1) as NaN. Lets the volume mark where a segment starts/ends at a
    depth (extremes of the iso-z contour) and draw the depth-link line. Returns None if absent."""
    rel = seg_prefix[len(_BUCKET) + 1:].rstrip("/")
    meshes = [m for m in _list(rel + "/mesh/") if scan_id in m and m.rstrip("/").endswith(".tifxyz")]
    if not meshes:
        return None
    mesh = meshes[0].rstrip("/")
    maps = []
    for nm in ("x.tif", "y.tif", "z.tif"):
        a = _tif(f"{_BUCKET}/{mesh}/{nm}").astype(np.float32)
        a[a <= 0] = np.nan
        maps.append(a)
    x, y, z = maps
    step = max(1, -(-max(z.shape) // max_side))
    if step > 1:
        x, y, z = x[::step, ::step], y[::step, ::step], z[::step, ::step]
    return x, y, z


def project_segment_ink(
    seg_prefix: str,
    scan_id: str,
    ink_threshold: float = 0.10,
    max_points: int = 1_200_000,
):
    """Return ink locations in the scan's voxel grid.

    ``seg_prefix`` is the segment's bucket URL; ``scan_id`` is the displayed volume's scan id
    (e.g. ``20260608103018``) so we pick the matching ``.tifxyz``. Returns an ``(N, 4)`` float32
    array of ``(z, y, x, ink)`` (ink in 0..1), or None if no mesh/ink is available.
    """
    rel = seg_prefix[len(_BUCKET) + 1:].rstrip("/")
    meshes = [m for m in _list(rel + "/mesh/") if scan_id in m and m.rstrip("/").endswith(".tifxyz")]
    if not meshes:
        return None
    mesh = meshes[0].rstrip("/")
    xs = _tif(f"{_BUCKET}/{mesh}/x.tif").astype(np.float32)
    ys = _tif(f"{_BUCKET}/{mesh}/y.tif").astype(np.float32)
    zs = _tif(f"{_BUCKET}/{mesh}/z.tif").astype(np.float32)

    ink = _load_ink(rel, scan_id)
    if ink is None:
        return None
    ink = _resample_to(ink, xs.shape).astype(np.float32)
    ink = ink / (ink.max() or 1.0)

    valid = (xs > 0) & (ys > 0) & (zs > 0) & (ink >= ink_threshold)
    if not valid.any():
        return None
    pts = np.stack([zs[valid], ys[valid], xs[valid], ink[valid]], axis=1).astype(np.float32)
    if pts.shape[0] > max_points:
        # Uniform (deterministic) subsample — preserves the faint/strong distribution, unlike
        # keeping only the strongest, which erased faint ink before it could be drawn.
        keep = np.random.default_rng(0).choice(pts.shape[0], max_points, replace=False)
        pts = pts[keep]
    return pts


def _load_ink(rel: str, scan_id: str) -> np.ndarray | None:
    """Prefer the small downsampled ink preview (fast); it is only used for per-point value."""
    previews = [
        k for k in _keys(rel + "/ink-detection/downsampled/")
        if k.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    match = [p for p in previews if scan_id in p] or previews
    if match:
        return _tif(f"{_BUCKET}/{match[0]}")
    full = [k for k in _keys(rel + "/ink-detection/") if k.lower().endswith(".tif") and scan_id in k]
    return _tif(f"{_BUCKET}/{full[0]}") if full else None


def _resample_to(img: np.ndarray, shape) -> np.ndarray:
    from PIL import Image

    if img.ndim == 3:
        img = img[..., 0]
    if img.shape == tuple(shape):
        return img
    im = Image.fromarray(img).resize((shape[1], shape[0]), Image.BILINEAR)
    return np.asarray(im)

"""Labelled reference fragments — the only data with genuine human/IR ink ground truth.

Detached fragments were µCT-scanned AND photographed in infrared (where the carbon ink is
visible). The volume-cartographer ``.volpkg`` for each holds the raw 3-D CT scan
(``volumes_zarr/…``) plus, in the ``working/…_exposed_surface`` dir, the flattened
exposed-surface render (``result.png``), the IR-derived binary ink mask (``inklabels.png``),
and — crucially — ``result.ppm``, a per-surface-pixel map giving each rendered pixel's (x, y,
z) coordinate back in the raw CT volume. That ppm is the fragment's analogue of a segment's
``.tifxyz`` mesh, so we can back-project the ink mask into native CT voxels exactly like the
scroll "See ink" path and view it in depth (the raw CT carries the through-sheet morphology
the flat render throws away).

Public server: ``https://dl.ash2txt.org/fragments/`` (no auth needed for these). These three
are the classic Kaggle ink-detection training fragments.
"""

from __future__ import annotations

import io
import os

import numpy as np

from .segment import SegmentVolume
from .types import SegmentMeta
from .zarr_reader import ZarrSegment

_ROOT = "https://dl.ash2txt.org/fragments"

# (id, display name, volpkg, exposed-surface subdir, ct-zarr name). The 54 keV scan is
# 3.24 µm isotropic; the raw CT zarr is a standard OME-Zarr v2 pyramid (scale 1.0, so the
# voxel size must be forced from the name — see voxel_override below).
FRAGMENTS = [
    ("Frag1", "Fragment 1 (PHerc Paris 2 Fr47)", "PHercParis2Fr47.volpkg", "54keV_exposed_surface"),
    ("Frag2", "Fragment 2 (PHerc Paris 2 Fr143)", "PHercParis2Fr143.volpkg", "54keV_exposed_surface"),
    ("Frag3", "Fragment 3 (PHerc Paris 1 Fr34)", "PHercParis1Fr34.volpkg", "54keV_exposed_surface"),
]
_VOXEL_UM = 3.24
_CT_ZARR = "volumes_zarr/54keV_3.24um_.zarr"

_SESSION = None


def _http():
    """A shared requests Session that retries with backoff on rate-limits/5xx. The fragment
    host (dl.ash2txt.org) throttles bursts with HTTP 429 (unlike the CDN-backed scroll bucket),
    so every fragment fetch goes through this — it honours the server's Retry-After header."""
    global _SESSION
    if _SESSION is None:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retry = Retry(
            total=6, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]), respect_retry_after_header=True,
            raise_on_status=False,
        )
        s = requests.Session()
        adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=16)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _SESSION = s
    return _SESSION


def fragment_catalog() -> list[dict]:
    """Metadata rows for the labelled fragments, for the scroll list."""
    rows = []
    for fid, name, volpkg, surf in FRAGMENTS:
        base = f"{_ROOT}/{fid}/{volpkg}/working/{surf}"
        rows.append({
            "id": fid, "name": name, "voxel_um": _VOXEL_UM,
            "ct_zarr": f"{_ROOT}/{fid}/{volpkg}/{_CT_ZARR}",
            "ct": f"{base}/result.png", "ink": f"{base}/inklabels.png",
            "ppm": f"{base}/result.ppm", "ir": f"{base}/ir.png", "mask": f"{base}/mask.png",
        })
    return rows


def _fetch_image(url: str) -> np.ndarray:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None  # these renders are large but trusted
    r = _http().get(url, timeout=180)
    r.raise_for_status()
    return np.asarray(Image.open(io.BytesIO(r.content)))


def open_fragment_ct(frag: dict) -> SegmentVolume:
    """Open the fragment's raw 3-D CT scan (OME-Zarr) so it can be viewed in depth and rotated
    through the depth axis. Its broad exposed face lies in the X–Z plane (Y is the thin
    through-thickness axis), so the viewer defaults this one to depth = Y."""
    return ZarrSegment(frag["ct_zarr"], _VOXEL_UM, voxel_override=_VOXEL_UM)


def _projection_cache_path(cache_dir, frag_id: str):
    if not cache_dir:
        return None
    try:
        d = os.path.join(str(cache_dir), "projections")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"frag_{frag_id}_v1.npy")
    except OSError:
        return None


def project_fragment_ink(
    frag: dict,
    cache_dir: str | None = None,
    ink_threshold: float = 0.10,
    max_rows: int = 440,
    max_cols: int = 440,
    max_points: int = 800_000,
):
    """Back-project the fragment's IR ink label into raw-CT voxels via ``result.ppm``.

    Returns ``(N, 4)`` float32 ``[z, y, x, ink]`` in the CT zarr's voxel grid — the same format
    as :func:`ink_projection.project_segment_ink`, so it feeds the identical viewer path. The
    ppm is ~2.5 GB, so we read only a strided grid of rows with HTTP range requests and cache
    the resulting point cloud to disk; the heavy read then happens once per fragment.
    """
    cache_file = _projection_cache_path(cache_dir, frag["id"])
    if cache_file and os.path.exists(cache_file):
        try:
            return np.load(cache_file)
        except (OSError, ValueError):
            pass

    import requests

    sess = _http()  # shared retry/backoff session (dl.ash2txt.org 429-throttles bursts)
    ppm_url = frag["ppm"]
    try:
        resp = sess.get(ppm_url, headers={"Range": "bytes=0-255"}, timeout=30)
        if resp.status_code not in (200, 206):
            return None
        head = resp.content
    except requests.RequestException:
        return None
    marker = head.find(b"<>\n")
    if marker < 0:
        return None
    data_start = marker + 3
    dims = {}
    for line in head[:marker].decode("ascii", "replace").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            dims[k.strip()] = v.strip()
    try:
        width, height = int(dims["width"]), int(dims["height"])
    except (KeyError, ValueError):
        return None
    rowbytes = width * 6 * 8  # dim=6 float64 per pixel: (x, y, z, nx, ny, nz)

    ink = _fetch_image(frag["ink"])
    if ink.ndim == 3:
        ink = ink[..., 0]
    ink = ink.astype(np.float32) / (float(ink.max()) or 1.0)
    if ink.shape[:2] != (height, width):  # guard an unexpected size mismatch
        from PIL import Image
        ink = np.asarray(
            Image.fromarray((ink * 255).astype(np.uint8)).resize((width, height))
        ).astype(np.float32) / 255.0

    row_stride = max(1, -(-height // max_rows))
    col_stride = max(1, -(-width // max_cols))
    cols = np.arange(0, width, col_stride)
    rows = list(range(0, height, row_stride))

    def fetch_row(r: int):
        start = data_start + r * rowbytes
        try:
            resp = sess.get(
                ppm_url, headers={"Range": f"bytes={start}-{start + rowbytes - 1}"}, timeout=60
            )
        except requests.RequestException:
            return None
        if resp.status_code not in (200, 206):  # gave up after retries (e.g. still 429)
            return None
        content = resp.content
        if len(content) < rowbytes:
            return None
        a = np.frombuffer(content, dtype="<f8").reshape(width, 6)[::col_stride, :3]
        xv, yv, zv = a[:, 0], a[:, 1], a[:, 2]     # ppm stores (x, y, z) in CT voxel coords
        iv = ink[r, cols]
        valid = (xv > 0) & (yv > 0) & (zv > 0) & (iv >= ink_threshold)
        if not valid.any():
            return None
        return np.stack([zv[valid], yv[valid], xv[valid], iv[valid]], axis=1).astype(np.float32)

    from concurrent.futures import ThreadPoolExecutor

    chunks = []
    # Keep concurrency modest: dl.ash2txt.org 429-throttles bursts, and the retry/backoff above
    # only helps if we aren't hammering it with 16 parallel range reads.
    with ThreadPoolExecutor(max_workers=6) as ex:
        for res in ex.map(fetch_row, rows):
            if res is not None and len(res):
                chunks.append(res)
    if not chunks:
        return None
    pts = np.concatenate(chunks, axis=0)
    if pts.shape[0] > max_points:
        keep = np.random.default_rng(0).choice(pts.shape[0], max_points, replace=False)
        pts = pts[keep]
    if cache_file:
        try:
            np.save(cache_file, pts)
        except OSError:
            pass
    return pts


def load_fragment_ir(frag: dict, max_side: int = 2400):
    """The fragment's rendered infrared scan (``ir.png``) — the flattened exposed surface
    photographed in IR, where the carbon ink is directly visible — as a downsampled grayscale
    array for the side pane, or None. This is the ground truth the CT ink label derives from."""
    ir = _fetch_image(frag["ir"])
    if ir.ndim == 3:
        ir = ir[..., :3].mean(axis=2)  # IR photo → single channel
    ir = ir.astype(np.float32)
    step = max(1, -(-max(ir.shape) // max_side))  # ceil-downsample the longest side
    return ir[::step, ::step] if step > 1 else ir


def load_fragment_ink(frag: dict):
    """The human/IR ink label for the fragment, as a grayscale raster (downsampled for the
    overlay), registered to the CT render. Returned like a model prediction but it is TRUTH."""
    ink = _fetch_image(frag["ink"])
    if ink.ndim == 3:
        ink = ink[..., 0]
    ink = (ink > 0).astype(np.uint8) * 255
    # Downsample so the RGBA overlay stays light; it is stretched to the full extent anyway.
    step = max(1, -(-max(ink.shape) // 4096))  # ceil: guarantees the longest side ≤ ~4096
    if step > 1:
        ink = ink[::step, ::step]
    return ink, "IR ground-truth ink label (human)"


class FragmentImageVolume(SegmentVolume):
    """A single-layer surface volume backed by an in-memory 2-D image, with a small in-RAM
    pyramid for smooth zoom-out. Depth is 1 (a fragment's exposed surface is one render)."""

    def __init__(self, img2d: np.ndarray, voxel_um: float, n_levels: int = 4):
        img = np.ascontiguousarray(img2d)
        self._levels = [img]
        for _ in range(n_levels - 1):
            prev = self._levels[-1]
            if min(prev.shape[:2]) <= 512:
                break
            h, w = (prev.shape[0] // 2) * 2, (prev.shape[1] // 2) * 2
            ds = prev[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3)).astype(prev.dtype)
            self._levels.append(ds)
        self.source_id = None  # in-memory; no disk L2 needed (reads are already instant)
        level_shapes = tuple((1, lv.shape[0], lv.shape[1]) for lv in self._levels)
        super().__init__(SegmentMeta(level_shapes, str(img.dtype), float(voxel_um)))

    def depth_chunk(self, level: int) -> int:
        return 1

    def _read_block(self, level, z0, z1, y0, y1, x0, x1) -> np.ndarray:
        return self._levels[level][y0:y1, x0:x1][None, ...]  # (1, dy, dx)

    def _read_points(self, level, zs, ys, xs) -> np.ndarray:
        return self._levels[level][ys, xs]

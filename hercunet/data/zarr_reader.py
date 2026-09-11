"""Robust direct reader for multiscale OME-Zarr stores on the Vesuvius data servers.

Why this exists: ``vesuvius.Volume(type="zarr", …)`` opens a zarr *group* and then iterates
its members to find the pyramid arrays — but over plain HTTP a zarr v2 group cannot list
its members, so the opener reports "No arrays found" for perfectly good stores and the
scroll/segment would be silently unopenable. We instead address the level arrays by their
known sub-paths (``0``, ``1``, …), discovered from the multiscales metadata or by probing.

Auto-detection order (first that yields ≥1 array wins):
  1. a bare array at the root (``.zarray`` present);
  2. ``multiscales`` datasets listed in the group's ``.zattrs``;
  3. numeric level probing (``0/.zarray``, ``1/.zarray``, …).

Everything is best-effort but never silent: a store that yields no 3D array raises, and the
caller surfaces the error in the UI.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .segment import SegmentVolume
from .types import SegmentMeta

# Shared on-disk block cache for ALL reads (UI + offline scripts), so a re-run or a parameter sweep
# over the same region skips S3 entirely. Lazily created; disable with HERCUNET_BLOCK_CACHE=0.
_BLOCK_CACHE = None
_BLOCK_CACHE_INIT = False

# Shared thread pool that fetches the sub-stripes of one block concurrently — the data layer pulls a
# requested slice in parallel so callers don't have to prefetch themselves.
_READ_POOL = None


def _read_pool():
    global _READ_POOL
    if _READ_POOL is None:
        _READ_POOL = ThreadPoolExecutor(
            max_workers=int(os.environ.get("HERCUNET_READ_WORKERS", "12"))
        )
    return _READ_POOL


def _block_cache():
    global _BLOCK_CACHE, _BLOCK_CACHE_INIT
    if not _BLOCK_CACHE_INIT:
        _BLOCK_CACHE_INIT = True
        if os.environ.get("HERCUNET_BLOCK_CACHE", "1") != "0":
            try:
                from .disk_cache import SlabDiskCache
                root = os.environ.get("HERCUNET_DISK_CACHE") or None
                mb = int(os.environ.get("HERCUNET_DISK_CACHE_MB", "65536"))
                _BLOCK_CACHE = SlabDiskCache(root, mb * 1024 * 1024)
            except Exception:
                _BLOCK_CACHE = None
    return _BLOCK_CACHE

_UM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*um", re.IGNORECASE)
_MAX_LEVELS = 16


def parse_voxel_um(url: str, default: float | None = None) -> float | None:
    """Extract voxel size in micrometres from a zarr name like ``…-7.910um-…`` if present."""
    m = _UM_RE.search(url)
    return float(m.group(1)) if m else default


_TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})


def _http_get(url: str, timeout: int = 15, tries: int = 3, base: float = 0.5):
    """GET a metadata object with transient-retry, mirroring the chunk-read :func:`_retry`. The open-time
    metadata reads (``discover_levels``) hit the public HTTPS bucket, which occasionally hiccups; a single-shot
    read there turned a blip into a spurious 'no zarr arrays found'. A 404 is a DEFINITIVE 'absent' (the
    level-probe loop relies on it to stop) and returns immediately — only connection errors, timeouts and
    5xx/429 are retried with exponential backoff. Returns the final ``requests.Response`` (200 or 404), or
    ``None`` if every attempt errored/was transient."""
    import requests
    import time

    resp = None
    for i in range(tries):
        try:
            resp = requests.get(url, timeout=timeout)
        except requests.RequestException:
            resp = None
        else:
            if resp.status_code == 200 or resp.status_code == 404:
                return resp
        if i + 1 < tries:
            time.sleep(base * (2 ** i))
    return resp


def _http_ok(url: str, timeout: int = 15) -> bool:
    r = _http_get(url, timeout=timeout)
    return r is not None and r.status_code == 200


_UNIT_TO_UM = {
    "micrometer": 1.0, "micron": 1.0, "um": 1.0, "µm": 1.0,
    "nanometer": 1e-3, "nm": 1e-3,
    "millimeter": 1e3, "mm": 1e3,
    "meter": 1e6, "m": 1e6,
}


def voxel_um_from_metadata(base: str) -> float | None:
    """Authoritative in-plane voxel size (µm) from the OME-Zarr level-0 scale transform.

    Filenames lie (e.g. ``…1.129um…-L1.zarr`` is actually 2.258 µm because ``-L1`` is a
    downsampled level), so the ``multiscales`` ``coordinateTransformations`` scale is the
    source of truth. Returns the x-axis scale converted to micrometres, or None.
    """
    attrs = _http_json(base.rstrip("/") + "/.zattrs")
    if not attrs:
        return None
    try:
        ms = attrs["multiscales"][0]
        scale = None
        for t in ms["datasets"][0]["coordinateTransformations"]:
            if t.get("type") == "scale":
                scale = t["scale"]
                break
        if not scale:
            return None
        axes = ms.get("axes", [])
        unit = (axes[-1].get("unit", "micrometer") if axes else "micrometer") or "micrometer"
        return float(scale[-1]) * _UNIT_TO_UM.get(str(unit).lower(), 1.0)
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _http_json(url: str, timeout: int = 15):
    r = _http_get(url, timeout=timeout)
    if r is None or r.status_code != 200:
        return None
    try:
        return r.json()
    except ValueError:
        return None


def discover_levels(base: str) -> list[str]:
    """Return the ordered array sub-paths of a multiscale store (``""`` == bare root array)."""
    base = base.rstrip("/")
    # local filesystem path (a downloaded OME-Zarr) — enumerate via os.path, no HTTP
    root = base[7:] if base.startswith("file://") else base
    if os.path.isdir(root):
        if os.path.exists(f"{root}/.zarray"):
            return [""]
        ap = f"{root}/.zattrs"
        if os.path.exists(ap):
            try:
                import json
                ds = json.load(open(ap))["multiscales"][0]["datasets"]
                paths = [str(d["path"]) for d in ds if os.path.exists(f'{root}/{d["path"]}/.zarray')]
                if paths:
                    return paths
            except (KeyError, IndexError, TypeError, ValueError):
                pass
        paths, i = [], 0
        while i < _MAX_LEVELS and os.path.exists(f"{root}/{i}/.zarray"):
            paths.append(str(i)); i += 1
        return paths
    if _http_ok(base + "/.zarray"):
        return [""]  # a single array, not a group

    attrs = _http_json(base + "/.zattrs")
    if attrs:
        try:
            datasets = attrs["multiscales"][0]["datasets"]
            paths = [str(d["path"]) for d in datasets]
            paths = [p for p in paths if _http_ok(f"{base}/{p}/.zarray")]
            if paths:
                return paths
        except (KeyError, IndexError, TypeError):
            pass

    paths, i = [], 0
    while i < _MAX_LEVELS and _http_ok(f"{base}/{i}/.zarray"):
        paths.append(str(i))
        i += 1
    return paths


class ZarrSegment(SegmentVolume):
    """A :class:`SegmentVolume` backed by a multiscale OME-Zarr read directly over HTTP."""

    def __init__(self, base_url: str, voxel_size_um: float | None, voxel_override: float | None = None):
        import zarr

        base = base_url.rstrip("/")
        paths = discover_levels(base)
        if not paths:
            raise ValueError(f"no zarr arrays found under {base}")

        arrays = []
        for p in paths:
            url = base if p == "" else f"{base}/{p}"
            try:
                arr = zarr.open(url, mode="r")
            except Exception:
                break  # levels are contiguous; stop at the first gap
            if getattr(arr, "ndim", 0) >= 3:
                arrays.append(arr)
        if not arrays:
            raise ValueError(f"no 3D arrays opened under {base}")

        self._arrays = arrays
        self.source_id = base  # store URL — the L2 disk-cache identity
        level_shapes = tuple(tuple(int(s) for s in a.shape[:3]) for a in arrays)
        # ``voxel_override`` wins outright. Otherwise the OME-Zarr scale metadata is authoritative
        # UNLESS it is the trivial identity (scale == 1.0 µm), which the standardized/masked scroll
        # volumes store as a voxel-space placeholder — a 1.0 µm CT voxel is physically implausible
        # here, so it means "no real scale", and the name's µm (e.g. ``…-2.399um``) is truer. Without
        # this, scrolls like PHerc 813 report a 1 µm voxel → a nonsensical ~8 mm diameter.
        if voxel_override and voxel_override > 0:
            voxel = float(voxel_override)
        else:
            meta_vox = voxel_um_from_metadata(base)
            name_vox = parse_voxel_um(base)
            if meta_vox is not None and meta_vox > 0 and abs(meta_vox - 1.0) > 1e-6:
                voxel = meta_vox                            # real physical scale
            elif name_vox and name_vox > 0:
                voxel = name_vox                            # trivial scale → trust the name
            else:
                voxel = float(voxel_size_um or 7.91)
        super().__init__(
            SegmentMeta(
                level_shapes=level_shapes,
                dtype=str(arrays[0].dtype),
                voxel_size_um=voxel,
            )
        )

    def depth_chunk(self, level: int) -> int:
        arr = self._arrays[level]
        chunks = getattr(arr, "chunks", None)
        dz = self.meta.level_shapes[level][0]
        if chunks and chunks[0]:
            return int(min(dz, max(int(chunks[0]), 1)))
        return super().depth_chunk(level)

    def _read_block(self, level, z0, z1, y0, y1, x0, x1) -> np.ndarray:
        """Chunk-aligned, parallel, cached read. The requested rect is decomposed into the array's
        native chunk grid; each chunk is served from the disk cache or fetched (in parallel) and
        cached WHOLE. Because whole chunks are the cache unit, any overlapping request — a prefetched
        full slice, a small tile, a neighbouring window — reuses the same cached chunks."""
        arr = self._arrays[level]
        cache = _block_cache()
        chunks = getattr(arr, "chunks", None)
        if cache is None or not chunks:
            return _retry(lambda: np.asarray(arr[z0:z1, y0:y1, x0:x1]))
        Zl, Yl, Xl = (int(s) for s in arr.shape[:3])
        cz, cy, cx = (int(c) for c in chunks[:3])
        out = np.empty((z1 - z0, y1 - y0, x1 - x0), dtype=arr.dtype)

        spans = []                                          # one per chunk overlapping the rect
        for ciz in range((z0 // cz), (z1 - 1) // cz + 1):
            az0, az1 = ciz * cz, min((ciz + 1) * cz, Zl)
            for ciy in range((y0 // cy), (y1 - 1) // cy + 1):
                ay0, ay1 = ciy * cy, min((ciy + 1) * cy, Yl)
                for cix in range((x0 // cx), (x1 - 1) // cx + 1):
                    ax0, ax1 = cix * cx, min((cix + 1) * cx, Xl)
                    spans.append((az0, az1, ay0, ay1, ax0, ax1, (level, ciz, ciy, cix)))

        fill = arr.fill_value if arr.fill_value is not None else 0

        def fetch(sp):
            az0, az1, ay0, ay1, ax0, ax1, key = sp
            ch = cache.get(self.source_id, key)
            if ch is not None:
                return sp, ch
            try:
                ch = _retry(lambda: np.asarray(arr[az0:az1, ay0:ay1, ax0:ax1]))
            except Exception as e:
                if not _is_missing_chunk(e):
                    raise
                # Retries exhausted on a 403/404: zero-fill THIS read but do NOT cache it, so a genuinely
                # air chunk stays cheap yet a transient throttle re-fetches next time (never poisons the cache).
                return sp, np.full((az1 - az0, ay1 - ay0, ax1 - ax0), fill, dtype=arr.dtype)
            try:
                cache.put(self.source_id, key, ch)                  # cache real data only
            except Exception:
                pass
            return sp, ch

        for sp, ch in _read_pool().map(fetch, spans):
            az0, az1, ay0, ay1, ax0, ax1, _ = sp
            iz0, iz1 = max(z0, az0), min(z1, az1)
            iy0, iy1 = max(y0, ay0), min(y1, ay1)
            ix0, ix1 = max(x0, ax0), min(x1, ax1)
            out[iz0 - z0:iz1 - z0, iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = \
                ch[iz0 - az0:iz1 - az0, iy0 - ay0:iy1 - ay0, ix0 - ax0:ix1 - ax0]
        return out

    def _read_points(self, level, zs, ys, xs) -> np.ndarray:
        # Vectorised point ("vindex") read: fetches only the chunks the points fall in,
        # so sampling a line costs O(chunks-on-the-line), not the bounding box.
        return _retry(lambda: np.asarray(self._arrays[level].vindex[zs, ys, xs]))


def _is_missing_chunk(e) -> bool:
    """True if the error means the chunk OBJECT is absent (air): a region-restricted ``finalize_pyramid``
    leaves air chunks unwritten, and a public-read S3 bucket without ListBucket returns 403 (not 404) for a
    missing key. Such chunks are legitimately empty and should read as fill, not crash the whole window."""
    status = getattr(e, "status", None) or getattr(getattr(e, "response", None), "status", None)
    if status in (403, 404):
        return True
    if isinstance(e, (KeyError, FileNotFoundError)):
        return True
    s = str(e)
    return "403" in s or "404" in s or "Forbidden" in s or "Not Found" in s or "NoSuchKey" in s


def _retry(fn, tries: int = 6, base: float = 0.6):
    """Run a chunk read, retrying transient failures with exponential backoff. Requested ranges are always
    valid (pre-clipped to the array shape), so an in-grid chunk that errors is present-but-throttled: S3
    returns 403 for BOTH a throttled request and a genuinely-absent key, and present chunks otherwise return
    200 — so a 403 on a requested chunk is far more likely transient throttling than air. Retry EVERYTHING
    (403 included) with backoff; only after `tries` are exhausted does the error propagate, letting the caller
    zero-fill THIS read (without caching it, so a later read retries). This prevents a throttle burst from
    poisoning the cache with permanent air chunks."""
    import time

    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(base * (2 ** i))

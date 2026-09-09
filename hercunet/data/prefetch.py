"""Transparent background prefetching for a :class:`SegmentVolume`.

Reads already cache their chunks (see ``zarr_reader._read_block``), so "prefetching" is simply issuing
the anticipated ``read_window`` calls on background threads — they warm the shared chunk cache, and the
foreground request that follows becomes a cache hit. This mirrors the UI's prefetcher but lives in the
data layer so offline scripts (tractability, seam assembly, the 3-D brick pipeline) get it for free.

Strategies:
  * ``window``       — a specific block;
  * ``slice``        — a whole (y, x) region at one depth slab (the current view);
  * ``neighbours``   — the same region at neighbouring DEPTHS and neighbouring RESOLUTIONS, so
                       scrolling z or zooming in/out is already buffered.
De-duplicated: the same block is never enqueued twice.
"""

from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor


def iter_windows(vol, requests, readahead: int = 3, workers: int = 3):
    """Yield ``(request, block, origin)`` for each read ``request`` (a ``read_window`` arg tuple
    ``(level, z0, z1, y0, y1, x0, x1)``) IN ORDER, while reading up to ``readahead`` ahead on a small
    thread pool. This overlaps S3 I/O with the caller's per-block compute (GPU), so the GPU stops
    idling on downloads — the caller just iterates. The natural iteration primitive for the 3-D brick
    pipeline. Each read still parallelises its own chunks and hits the chunk cache underneath."""
    reqs = iter(list(requests))
    ex = ThreadPoolExecutor(max_workers=workers)

    def rd(req):
        blk, org = vol.read_window(*req)
        return req, blk, org

    try:
        inflight: deque = deque()
        for _ in range(max(1, readahead)):
            r = next(reqs, None)
            if r is None:
                break
            inflight.append(ex.submit(rd, r))
        while inflight:
            fut = inflight.popleft()
            r = next(reqs, None)
            if r is not None:
                inflight.append(ex.submit(rd, r))
            yield fut.result()
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


class VolumePrefetcher:
    def __init__(self, vol, max_workers: int = 6):
        self.vol = vol
        self.ex = ThreadPoolExecutor(max_workers=max_workers)
        self._seen: set = set()
        self._lock = threading.Lock()
        self._futs: list = []

    def _warm(self, level, z0, z1, y0, y1, x0, x1) -> None:
        Zl, Yl, Xl = self.vol.meta.level_shapes[level]
        z0, z1 = max(0, int(z0)), min(int(Zl), int(z1))
        y0, y1 = max(0, int(y0)), min(int(Yl), int(y1))
        x0, x1 = max(0, int(x0)), min(int(Xl), int(x1))
        if z1 <= z0 or y1 <= y0 or x1 <= x0:
            return
        key = (level, z0, z1, y0, y1, x0, x1)
        with self._lock:
            if key in self._seen:
                return
            self._seen.add(key)
        self._futs.append(self.ex.submit(self._read, key))

    def _read(self, key) -> None:
        try:
            self.vol.read_window(*key)          # populates the chunk cache; result discarded
        except Exception:
            pass

    # -- strategies -------------------------------------------------------------------------------
    def window(self, level, z0, z1, y0, y1, x0, x1) -> None:
        """Warm one exact block."""
        self._warm(level, z0, z1, y0, y1, x0, x1)

    def slice(self, level, z, y0, y1, x0, x1, halfdepth: int = 0) -> None:
        """Warm a whole (y, x) region at depth ``z`` (± ``halfdepth`` slices)."""
        self._warm(level, z - halfdepth, z + halfdepth + 1, y0, y1, x0, x1)

    def _scaled(self, level, tgt, z0, z1, y0, y1, x0, x1):
        Zl, Yl, Xl = self.vol.meta.level_shapes[level]
        Zt, Yt, Xt = self.vol.meta.level_shapes[tgt]
        sz, sy, sx = Zt / Zl, Yt / Yl, Xt / Xl
        return (int(z0 * sz), int(z1 * sz), int(y0 * sy), int(y1 * sy), int(x0 * sx), int(x1 * sx))

    def neighbours(self, level, z0, z1, y0, y1, x0, x1,
                   depth_steps=(-1, 1), level_steps=(-1, 1)) -> None:
        """Warm the same region at neighbouring DEPTHS (shifting the z-slab by its own thickness) and
        neighbouring RESOLUTIONS (the same physical box at ``level`` ± steps). Depth neighbours within
        one 128-deep chunk are usually already cached; the resolution neighbours are the real win for
        zoom, and coarser levels are cheap to pre-buffer."""
        dz = z1 - z0
        for s in depth_steps:
            self._warm(level, z0 + s * dz, z1 + s * dz, y0, y1, x0, x1)
        nlev = self.vol.meta.num_levels
        for s in level_steps:
            tgt = level + s
            if 0 <= tgt < nlev:
                self._warm(tgt, *self._scaled(level, tgt, z0, z1, y0, y1, x0, x1))

    # -- lifecycle --------------------------------------------------------------------------------
    def drain(self) -> None:
        """Block until all enqueued prefetches finish (e.g. before timing a foreground pass)."""
        futs, self._futs = self._futs, []
        for f in futs:
            f.result()

    def shutdown(self) -> None:
        self.ex.shutdown(wait=False, cancel_futures=True)

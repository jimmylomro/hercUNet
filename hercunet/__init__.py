"""hercunet — cross-volume surface (sheet) detection for carbonised Herculaneum scrolls.

The package is organised as a three-stage pipeline over a shared streaming data layer:

  * :mod:`hercunet.data`   — THE DATA LAYER. Streaming multiscale OME-Zarr reads with a chunk-aligned
                             parallel fetch, an on-disk L2 slab cache, transparent prefetch, and
                             material tiling. The only stage implemented so far.
  * :mod:`hercunet.labels` — STAGE 1 (skeleton): pseudo-label generation.
  * :mod:`hercunet.refine` — STAGE 2 (skeleton): the iterative HercUNet refiner.
  * :mod:`hercunet.infer`  — STAGE 3 (skeleton): single-instance full-volume inference.
  * :mod:`hercunet.viz`    — per-stage visual tools (skeleton).

Everything downstream is written against the data layer's abstractions
(:class:`~hercunet.data.SegmentSource`, :class:`~hercunet.data.SegmentVolume`), so a concrete
backend (streaming ``vesuvius`` or procedural ``demo``) is interchangeable.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .data import (
    SegmentSource,
    SegmentVolume,
    ScrollInfo,
    SegmentInfo,
    SegmentMeta,
    get_backend,
)

__all__ = [
    "__version__",
    "SegmentSource",
    "SegmentVolume",
    "ScrollInfo",
    "SegmentInfo",
    "SegmentMeta",
    "get_backend",
]

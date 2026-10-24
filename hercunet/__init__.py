"""hercunet — cross-volume surface (sheet) detection for carbonised Herculaneum scrolls.

Three products, each its own package, over a set of shared top-level modules:

  * :mod:`hercunet.herculabels` — **HercuLabels**: stage-1 pseudo-label generation + the Qt editing UI (``viz``).
  * :mod:`hercunet.hercunet`    — **HercUNet**: the nnU-Net iterative surface refiner (training + recipes).
  * :mod:`hercunet.hercuvit`    — **HercuViT**: Dinovol-based full-resolution sheet segmentation.

Shared top-level modules (used by more than one product):

  * :mod:`hercunet.data`    — THE DATA LAYER. Streaming multiscale OME-Zarr reads with a chunk-aligned parallel
                              fetch, an on-disk L2 slab cache, transparent prefetch, and material tiling.
  * :mod:`hercunet.infer`   — the model-agnostic full-volume inference engine.
  * :mod:`hercunet.losses`  — constrained-MALIS + affinity losses.
  * :mod:`hercunet.labels`  — label GEOMETRY primitives (corpus, io, mesh, meshlets, embedding, fields, selection).
  * :mod:`hercunet.pretrained`, :mod:`hercunet.config` — warm-start resolution + config.

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

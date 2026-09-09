"""Backend interface + selection.

A backend enumerates scrolls and their segments and opens a segment as a
:class:`SegmentVolume`. The UI is written against this interface only. Two concrete
backends ship: ``vesuvius`` (streaming, real data) and ``demo`` (procedural, offline).
"""

from __future__ import annotations

import abc

from ..config import Config
from .segment import SegmentVolume
from .types import ScrollInfo, SegmentInfo


class SegmentSource(abc.ABC):
    """Enumerate scrolls, open a scroll's raw volume, and list/open its surface segments."""

    @abc.abstractmethod
    def list_scrolls(self) -> list[ScrollInfo]:
        ...

    @abc.abstractmethod
    def open_scroll_volume(self, scroll: ScrollInfo) -> SegmentVolume:
        """Open a scroll's raw µCT volume (depth axis = scan slices)."""

    @abc.abstractmethod
    def list_segments(self, scroll: ScrollInfo) -> list[SegmentInfo]:
        ...

    @abc.abstractmethod
    def open_segment(self, segment: SegmentInfo) -> SegmentVolume:
        """Open a segment's flattened surface volume (depth axis = the ~65 surface layers)."""

    def load_ink_overlay(self, segment: SegmentInfo, volume_url: str | None = None):
        """Return ``(image, label)`` for the segment's ink-detection **model prediction**
        (a 2D raster on the flattened surface, NOT a human/IR ground-truth mask), or None
        if none is published. ``volume_url`` (the opened surface-volume) lets the backend
        pick the prediction rendered at the matching resolution. Default: no overlay."""
        return None

    def load_segment_xyz(self, segment: SegmentInfo, scan_id: str):
        """Return per-surface-pixel ``(x, y, z)`` scroll-voxel coordinate maps for the segment
        (from its ``.tifxyz`` mesh for ``scan_id``), or None. Drives the depth-link line and
        the start/end markers. Default: unavailable."""
        return None

    def load_segment_zmap(self, segment: SegmentInfo, scan_id: str):
        """Per-surface-pixel scroll-z map (the z of :meth:`load_segment_xyz`), or None."""
        xyz = self.load_segment_xyz(segment, scan_id)
        return None if xyz is None else xyz[2]

    def project_fragment_ink(self, scroll: ScrollInfo, cache_dir: str | None = None):
        """Back-project a labelled fragment's IR ink into raw-CT voxels; ``(N,4)`` [z,y,x,ink]
        or None. Only meaningful for the labelled fragments. Default: unavailable."""
        return None

    def load_fragment_ir(self, scroll: ScrollInfo):
        """The fragment's rendered IR scan as a 2D array, or None. Default: unavailable."""
        return None


def get_backend(config: Config) -> SegmentSource:
    """Instantiate the configured backend, falling back to demo if S3/zarr deps are absent."""
    if config.backend == "demo":
        from .demo_backend import DemoBackend

        return DemoBackend()

    try:
        import requests  # noqa: F401
        import zarr  # noqa: F401

        from .s3_backend import S3Backend

        return S3Backend(config)
    except Exception as exc:  # missing deps at import time
        import warnings

        warnings.warn(
            f"S3 backend unavailable ({exc}); using demo backend. "
            f"Install the 'data' extra (zarr, requests, fsspec) or set "
            f"HERCUNET_BACKEND=demo to silence.",
            stacklevel=2,
        )
        from .demo_backend import DemoBackend

        return DemoBackend()

"""Data access layer: catalog of scrolls/segments and streaming multiscale reads.

The rest of the pipeline depends only on the abstractions here (:class:`SegmentSource`,
:class:`SegmentVolume`, and the info dataclasses), never on a concrete backend, so the
``vesuvius`` streaming backend and the procedural ``demo`` backend are interchangeable.

Beyond those abstractions this module exposes the streaming primitives directly — the
chunk-aligned cached reader (:class:`ZarrSegment`), the on-disk L2 slab cache
(:class:`SlabDiskCache`), the prefetch iterator (:func:`iter_windows`) and prefetcher
(:class:`VolumePrefetcher`), and material tiling (:func:`locate_material`,
:func:`iter_material_tiles`) — so offline scripts get the same fast, cached, prefetched
access the viewer uses.
"""

from .backend import SegmentSource, get_backend
from .segment import SegmentVolume
from .types import ScrollInfo, SegmentInfo, SegmentMeta
from .zarr_reader import ZarrSegment, parse_voxel_um
from .disk_cache import SlabDiskCache
from .prefetch import iter_windows, VolumePrefetcher
from .tiling import MaterialRegion, locate_material, iter_material_tiles, n_material_tiles

__all__ = [
    # abstractions
    "SegmentSource",
    "get_backend",
    "SegmentVolume",
    "ScrollInfo",
    "SegmentInfo",
    "SegmentMeta",
    # streaming primitives
    "ZarrSegment",
    "parse_voxel_um",
    "SlabDiskCache",
    "iter_windows",
    "VolumePrefetcher",
    "MaterialRegion",
    "locate_material",
    "iter_material_tiles",
    "n_material_tiles",
]

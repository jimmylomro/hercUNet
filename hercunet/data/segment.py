"""SegmentVolume: multiscale, viewport-oriented reads of one segment surface volume.

The UI never reads a whole segment (they are large and scrolls are teravoxel-scale). It
asks for a single depth layer over a clipped (y, x) window at a chosen pyramid level;
:meth:`read_region` streams only that region. Concrete backends supply the pixel reads by
implementing :meth:`_read_level_region`.
"""

from __future__ import annotations

import abc

import numpy as np

from .types import SegmentMeta


class SegmentVolume(abc.ABC):
    """Abstract multiscale surface volume. One instance == one opened segment."""

    # A stable identity for the disk (L2) slab cache: the store URL plus anything that
    # changes the bytes a (level, z, ty, tx) read returns (e.g. depth orientation). None
    # disables disk caching for this volume (used by the procedural demo backend).
    source_id: str | None = None

    def __init__(self, meta: SegmentMeta):
        self.meta = meta

    # ---- to be provided by backends -------------------------------------------------
    @abc.abstractmethod
    def _read_block(
        self, level: int, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int
    ) -> np.ndarray:
        """Return a 3D (dz, dy, dx) array for the given *level* coordinate ranges."""

    @abc.abstractmethod
    def _read_points(self, level: int, zs, ys, xs) -> np.ndarray:
        """Return a 1D array of values at the given *level* coordinate points (vectorised)."""

    def _read_level_region(
        self, level: int, z: int, ly0: int, ly1: int, lx0: int, lx1: int
    ) -> np.ndarray:
        """A single depth layer over a (y, x) window — a 2D plane out of :meth:`_read_block`."""
        return self._read_block(level, z, z + 1, ly0, ly1, lx0, lx1)[0]

    # ---- line sampling (for the click-through profile / spectrum / cepstrum) ---------
    def sample_line(self, level: int, zs, ys, xs) -> np.ndarray:
        """Sample intensity at arbitrary (z, y, x) points in *level* coordinates."""
        z = np.clip(np.rint(zs).astype(np.int64), 0, self.meta.level_shapes[level][0] - 1)
        y = np.clip(np.rint(ys).astype(np.int64), 0, self.meta.level_shapes[level][1] - 1)
        x = np.clip(np.rint(xs).astype(np.int64), 0, self.meta.level_shapes[level][2] - 1)
        return np.asarray(self._read_points(level, z, y, x))

    # ---- shared logic ---------------------------------------------------------------
    def pick_level(self, view_width_fullpx: float, pane_width_px: int) -> int:
        """Choose the coarsest level that still gives ~1 data pixel per screen pixel."""
        if pane_width_px <= 0 or view_width_fullpx <= 0:
            return 0
        want = view_width_fullpx / pane_width_px  # desired downsample factor
        best = 0
        for lvl in range(self.meta.num_levels):
            if self.meta.downsample_x(lvl) <= max(1.0, want):
                best = lvl
        return best

    def read_region(
        self, level: int, z: int, y0: float, y1: float, x0: float, x1: float
    ):
        """Read a window at ``level`` for full-resolution depth layer ``z``.

        ``z`` and ``y*``/``x*`` are in *full-resolution* coordinates. Because every axis
        (including depth) is downsampled per level, we map them into this level's grid.
        Returns ``(array, (fy0, fx0, fy1, fx1))`` where the bbox is the full-resolution
        extent the returned array covers, so the caller can place it in one shared full-res
        coordinate space (the space the 0.5 mm square lives in).
        """
        level = int(np.clip(level, 0, self.meta.num_levels - 1))
        Z0, Y0, X0 = self.meta.level_shapes[0]
        Zl, Yl, Xl = self.meta.level_shapes[level]
        sy, sx, sz = Y0 / Yl, X0 / Xl, Z0 / Zl  # full-res units per level pixel

        # Map the full-res depth layer into this level's (coarser) depth axis.
        z = int(np.clip(z, 0, Z0 - 1))
        zc = int(np.clip(round(z / sz), 0, Zl - 1))

        # Clip requested window to the full-res volume, then map to level pixels.
        fy0 = float(np.clip(np.floor(y0), 0, Y0))
        fy1 = float(np.clip(np.ceil(y1), 0, Y0))
        fx0 = float(np.clip(np.floor(x0), 0, X0))
        fx1 = float(np.clip(np.ceil(x1), 0, X0))
        if fy1 <= fy0 or fx1 <= fx0:
            return np.zeros((0, 0), dtype=self.meta.dtype), (fy0, fx0, fy0, fx0)

        ly0, ly1 = int(fy0 / sy), min(Yl, int(-(-fy1 // sy)))  # ceil upper bound
        lx0, lx1 = int(fx0 / sx), min(Xl, int(-(-fx1 // sx)))
        arr = self._read_level_region(level, zc, ly0, ly1, lx0, lx1)

        # Full-res extent of what we actually read (level-grid snapped).
        return arr, (ly0 * sy, lx0 * sx, ly1 * sy, lx1 * sx)

    # ---- tile access (fixed grid, for the viewer's tiled cache) ---------------------
    def tile_grid(self, level: int, tile: int) -> tuple[int, int]:
        """Number of (rows, cols) tiles of side ``tile`` (level pixels) at ``level``."""
        _, Yl, Xl = self.meta.level_shapes[level]
        return (-(-Yl // tile), -(-Xl // tile))  # ceil division

    def read_tile(self, level: int, z: int, ty: int, tx: int, tile: int):
        """Read one fixed grid tile ``(ty, tx)`` of a level for full-res depth ``z``.

        Returns ``(array, (fy0, fx0, fy1, fx1))`` with the full-resolution extent, so tiles
        from any level share one coordinate space. Tiles are the unit the viewer caches, so
        overlapping views reuse them instead of re-reading.
        """
        level = int(np.clip(level, 0, self.meta.num_levels - 1))
        Z0, Y0, X0 = self.meta.level_shapes[0]
        Zl, Yl, Xl = self.meta.level_shapes[level]
        sy, sx, sz = Y0 / Yl, X0 / Xl, Z0 / Zl

        z = int(np.clip(z, 0, Z0 - 1))
        zc = int(np.clip(round(z / sz), 0, Zl - 1))
        ly0, ly1 = ty * tile, min(Yl, (ty + 1) * tile)
        lx0, lx1 = tx * tile, min(Xl, (tx + 1) * tile)
        if ly1 <= ly0 or lx1 <= lx0:
            return np.zeros((0, 0), dtype=self.meta.dtype), (0.0, 0.0, 0.0, 0.0)

        arr = self._read_level_region(level, zc, ly0, ly1, lx0, lx1)
        return arr, (ly0 * sy, lx0 * sx, ly1 * sy, lx1 * sx)

    # ---- depth slabs (amortise chunk fetches over many layers) ----------------------
    def depth_chunk(self, level: int) -> int:
        """How many depth layers to read in one fetch. Reading a single z-slice of a chunked
        store still downloads the whole z-chunk, so we read (and cache) the slab at once —
        making depth scrubbing within a slab free. Backends override with the real chunk
        depth; the default reads the full (shallow) depth or a 128-slab for deep volumes."""
        dz = self.meta.level_shapes[level][0]
        return dz if dz <= 192 else 128

    def slab_bbox(self, level: int, ty: int, tx: int, tile: int):
        """Full-resolution ``(fy0, fx0, fy1, fx1)`` extent of tile ``(ty, tx)`` at ``level``.

        Deterministic from geometry alone, so a disk-cache hit (which stores only the pixel
        block) can recover the tile's placement without any read.
        """
        level = int(np.clip(level, 0, self.meta.num_levels - 1))
        _, Y0, X0 = self.meta.level_shapes[0]
        _, Yl, Xl = self.meta.level_shapes[level]
        sy, sx = Y0 / Yl, X0 / Xl
        ly0, ly1 = ty * tile, min(Yl, (ty + 1) * tile)
        lx0, lx1 = tx * tile, min(Xl, (tx + 1) * tile)
        return (ly0 * sy, lx0 * sx, ly1 * sy, lx1 * sx)

    def read_window(self, level: int, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int):
        """Read an arbitrary ``(z, y, x)`` sub-block at ``level`` (level-native coords).

        Returns ``(block (dz, dy, dx), (z0c, y0c, x0c))`` — the block plus the clipped
        level-native origins, so a caller can place it. Used by the structure-tensor frame
        field, which needs a z-slab around the current depth (not the full column)."""
        level = int(np.clip(level, 0, self.meta.num_levels - 1))
        Zl, Yl, Xl = self.meta.level_shapes[level]
        z0, z1 = int(np.clip(z0, 0, Zl)), int(np.clip(z1, 0, Zl))
        y0, y1 = int(np.clip(y0, 0, Yl)), int(np.clip(y1, 0, Yl))
        x0, x1 = int(np.clip(x0, 0, Xl)), int(np.clip(x1, 0, Xl))
        if z1 <= z0 or y1 <= y0 or x1 <= x0:
            return np.zeros((0, 0, 0), dtype=self.meta.dtype), (z0, y0, x0)
        return self._read_block(level, z0, z1, y0, y1, x0, x1), (z0, y0, x0)

    def read_depth_column(self, level: int, y0: int, y1: int, x0: int, x1: int):
        """Read the FULL depth stack over a ``(y, x)`` window at ``level``: ``(dz, dy, dx)``.

        The depth axis is the whole column, so for a flattened surface volume (depth = the
        through-sheet layers) this is exactly the material to measure papyrus thickness from.
        ``y*``/``x*`` are level-native coords; returns ``(block, (fy0, fx0, fy1, fx1))`` full-res.
        """
        level = int(np.clip(level, 0, self.meta.num_levels - 1))
        _, Y0, X0 = self.meta.level_shapes[0]
        Zl, Yl, Xl = self.meta.level_shapes[level]
        sy, sx = Y0 / Yl, X0 / Xl
        ly0, ly1 = int(np.clip(y0, 0, Yl)), int(np.clip(y1, 0, Yl))
        lx0, lx1 = int(np.clip(x0, 0, Xl)), int(np.clip(x1, 0, Xl))
        if ly1 <= ly0 or lx1 <= lx0:
            return np.zeros((0, 0, 0), dtype=self.meta.dtype), (0.0, 0.0, 0.0, 0.0)
        block = self._read_block(level, 0, Zl, ly0, ly1, lx0, lx1)
        return block, (ly0 * sy, lx0 * sx, ly1 * sy, lx1 * sx)

    def read_tile_slab(self, level: int, lz0: int, lz1: int, ty: int, tx: int, tile: int):
        """Read a depth slab ``[lz0, lz1)`` (level-native z) of tile ``(ty, tx)``.

        Returns ``(block (dz, h, w), (fy0, fx0, fy1, fx1))``. The caller caches each layer.
        """
        level = int(np.clip(level, 0, self.meta.num_levels - 1))
        Z0, Y0, X0 = self.meta.level_shapes[0]
        Zl, Yl, Xl = self.meta.level_shapes[level]
        sy, sx = Y0 / Yl, X0 / Xl
        lz0, lz1 = int(np.clip(lz0, 0, Zl)), int(np.clip(lz1, 0, Zl))
        ly0, ly1 = ty * tile, min(Yl, (ty + 1) * tile)
        lx0, lx1 = tx * tile, min(Xl, (tx + 1) * tile)
        if lz1 <= lz0 or ly1 <= ly0 or lx1 <= lx0:
            return np.zeros((0, 0, 0), dtype=self.meta.dtype), (0.0, 0.0, 0.0, 0.0)
        block = self._read_block(level, lz0, lz1, ly0, ly1, lx0, lx1)
        return block, (ly0 * sy, lx0 * sx, ly1 * sy, lx1 * sx)


# Depth-axis orientations: which base axis becomes "depth" (axis 0 of the oriented view).
# The tuple is the transpose permutation mapping oriented axes -> base (z, y, x) axes, so
# oriented_axis i reads base axis perm[i]. perm[0] is therefore the base axis used as depth.
#   "z" -> depth is Z; the plane shown is (Y rows, X cols)   [identity]
#   "y" -> depth is Y; the plane shown is (Z rows, X cols)   [swap Z and Y]
#   "x" -> depth is X; the plane shown is (Y rows, Z cols)   [reverse]
_PERM = {"z": (0, 1, 2), "y": (1, 0, 2), "x": (2, 1, 0)}


class OrientedVolume(SegmentVolume):
    """A view of a base :class:`SegmentVolume` with a chosen axis as depth.

    Wraps reads so the viewer can treat any orientation uniformly (depth == axis 0). ``z``
    shows XY planes and scrubs through the scan/​surface axis; ``y`` shows XZ planes and
    scrubs through Y — useful for looking *across* the winding of a scroll.
    """

    def __init__(self, base: SegmentVolume, axis: str):
        if axis not in _PERM:
            raise ValueError(f"axis must be one of {list(_PERM)}, got {axis!r}")
        self.base = base
        self.axis = axis
        self._perm = _PERM[axis]
        # Oriented reads transpose the bytes, so key the disk cache separately per axis.
        self.source_id = f"{base.source_id}|axis={axis}" if base.source_id else None
        oriented_shapes = tuple(
            tuple(shape[p] for p in self._perm) for shape in base.meta.level_shapes
        )
        super().__init__(
            SegmentMeta(
                level_shapes=oriented_shapes,
                dtype=base.meta.dtype,
                voxel_size_um=base.meta.voxel_size_um,
            )
        )

    def _to_base_ranges(self, d0, d1, a0, a1, b0, b1):
        """Map oriented (depth, a, b) ranges back to base (z, y, x) ranges."""
        oriented = [(d0, d1), (a0, a1), (b0, b1)]
        base = [None, None, None]
        for oriented_axis, base_axis in enumerate(self._perm):
            base[base_axis] = oriented[oriented_axis]
        return base  # [(z0,z1),(y0,y1),(x0,x1)]

    def _read_block(self, level, z0, z1, y0, y1, x0, x1) -> np.ndarray:
        (bz, by, bx) = self._to_base_ranges(z0, z1, y0, y1, x0, x1)
        block = self.base._read_block(level, bz[0], bz[1], by[0], by[1], bx[0], bx[1])
        return np.transpose(block, self._perm)  # base (z,y,x) -> oriented (depth,a,b)

    def _read_points(self, level, zs, ys, xs) -> np.ndarray:
        oriented = [zs, ys, xs]
        base = [None, None, None]
        for oriented_axis, base_axis in enumerate(self._perm):
            base[base_axis] = oriented[oriented_axis]
        return self.base._read_points(level, base[0], base[1], base[2])

"""Plain data records shared across backends and UI.

Kept dependency-free (stdlib only) so they can be imported anywhere, including tests,
without pulling in Qt, pyqtgraph, or the vesuvius library.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScrollInfo:
    """A scannable scroll and the scan variant we will read it at."""

    scroll_id: str            # canonical id, e.g. "1", "5", "PHerc172"
    name: str                 # human label, e.g. "Scroll 1 (PHerc. Paris 4)"
    energy: int | None = None       # scan energy in keV, when known
    resolution_um: float | None = None  # voxel size in micrometres of this scan variant
    status: str | None = None       # reading status, e.g. "read", "unread"
    notes: str | None = None        # freeform description (history, contents)
    n_segments: int | None = None   # number of published segments in the catalog
    category: str = ""              # "read" | "ink" | "first_letters" (drives list colour)
    extra: dict = field(default_factory=dict)  # backend-specific handles (e.g. PHerc id)

    @property
    def label(self) -> str:
        bits = [self.name]
        if self.resolution_um:
            bits.append(f"{self.resolution_um:g} um")
        if self.energy:
            bits.append(f"{self.energy} keV")
        return "  ·  ".join(bits)

    @property
    def tooltip(self) -> str:
        """Rich-text hover summary for the scroll list."""
        lines = [f"<b>{self.name}</b>"]
        specs = []
        if self.resolution_um:
            specs.append(f"{self.resolution_um:g} µm/voxel")
        if self.energy:
            specs.append(f"{self.energy} keV")
        if specs:
            lines.append(" · ".join(specs))
        if self.status:
            lines.append(f"Status: <b>{self.status}</b>")
        if self.n_segments is not None:
            noun = "segment" if self.n_segments == 1 else "segments"
            has = self.n_segments or "no"
            lines.append(f"{has} published {noun}")
        if self.notes:
            lines.append(f"<span style='color:#9fb0c9'>{self.notes}</span>")
        body = "<br>".join(lines)
        return f"<div style='max-width: 340px'>{body}</div>"


@dataclass(frozen=True)
class SegmentInfo:
    """A published segment (flattened surface) of a scroll."""

    segment_id: str
    scroll: ScrollInfo
    resolution_um: float          # voxel size; drives physical-scale overlays
    area_cm2: float | None = None
    author: str | None = None
    extra: dict = field(default_factory=dict)  # backend-specific handle/paths

    @property
    def voxel_size_um(self) -> float:
        return self.resolution_um

    @property
    def label(self) -> str:
        bits = [self.segment_id]
        if self.area_cm2:
            bits.append(f"{self.area_cm2:g} cm²")
        if self.author:
            bits.append(self.author)
        return "  ·  ".join(bits)


@dataclass(frozen=True)
class SegmentMeta:
    """Physical + array metadata for an opened segment surface volume.

    Axis convention (matches the surface-volume tensor in the project brief):
        axis 0 = z / depth  (the flattened layers through the papyrus surface)
        axis 1 = y / vertical
        axis 2 = x / horizontal

    Real OME-Zarr pyramids downsample ALL three axes per level (e.g. Scroll 1 segment:
    65×9163×5048 at level 0, 33×4582×2524 at level 1, …), and the factors are not exact
    powers of two. So we store the actual per-level ``(Z, Y, X)`` shapes rather than a
    single downsample factor, and derive coordinate mappings from them.
    """

    level_shapes: tuple[tuple[int, int, int], ...]  # per level (Z, Y, X); level 0 = full res
    dtype: str
    voxel_size_um: float                            # full-resolution voxel size

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.level_shapes[0]

    @property
    def num_layers(self) -> int:
        return self.level_shapes[0][0]

    @property
    def num_levels(self) -> int:
        return len(self.level_shapes)

    def level_shape(self, level: int) -> tuple[int, int, int]:
        return self.level_shapes[level]

    def downsample_x(self, level: int) -> float:
        """Horizontal downsample factor of ``level`` vs full res (usually ~2**level)."""
        return self.level_shapes[0][2] / self.level_shapes[level][2]

    @property
    def level_downsamples(self) -> tuple[int, ...]:
        """Approximate integer X downsample per level, for display/inspection."""
        x0 = self.level_shapes[0][2]
        return tuple(round(x0 / s[2]) for s in self.level_shapes)

    def window_cap_px(self, window_mm: float) -> float:
        """Side length, in full-resolution pixels, of a ``window_mm`` physical square."""
        return (window_mm * 1000.0) / self.voxel_size_um

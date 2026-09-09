"""Procedural, offline backend for development and tests.

Synthesizes a plausible surface volume *on demand* for any requested region and level, so
the full UI (scroll picker -> segment picker -> zoom/pan/depth + 0.5 mm square) runs with
no network and no ``vesuvius`` install, and so tests have deterministic pixels.

The synthetic signal deliberately mirrors the project's physics: an oriented high-
frequency *fibre* carrier, and a low-frequency *ink* envelope whose concentration follows
the fibre field multiplicatively (``observed ≈ base·(1 − ink·fibre)``), peaking at the
centre depth layer so scrolling through z makes the "writing" fade in and out.
"""

from __future__ import annotations

import math

import numpy as np

from .backend import SegmentSource
from .segment import SegmentVolume
from .types import ScrollInfo, SegmentInfo, SegmentMeta


def _pyramid(z: int, y: int, x: int, n_levels: int) -> tuple[tuple[int, int, int], ...]:
    """Build per-level (Z, Y, X) shapes halving every axis, like a real OME-Zarr pyramid."""
    shapes = []
    for _ in range(n_levels):
        shapes.append((z, y, x))
        z, y, x = max(1, -(-z // 2)), max(1, -(-y // 2)), max(1, -(-x // 2))
    return tuple(shapes)


class DemoBackend(SegmentSource):
    def list_scrolls(self) -> list[ScrollInfo]:
        return [
            ScrollInfo("demo-1", "Demo Scroll 1", energy=54, resolution_um=7.91,
                       status="synthetic", n_segments=4, category="read",
                       notes="Procedurally generated fibre + depth-modulated ink. Offline."),
            ScrollInfo("demo-5", "Demo Scroll 5 (lead-rich)", energy=53, resolution_um=3.24,
                       status="synthetic", n_segments=3, category="ink",
                       notes="Higher-resolution synthetic variant for testing the LOD path."),
        ]

    def list_segments(self, scroll: ScrollInfo) -> list[SegmentInfo]:
        n = 4 if scroll.scroll_id == "demo-1" else 3
        return [
            SegmentInfo(
                segment_id=f"{scroll.scroll_id}-seg{100 + i}",
                scroll=scroll,
                resolution_um=scroll.resolution_um or 7.91,
                area_cm2=round(1.5 + 0.7 * i, 1),
                author="demo",
                extra={"seed": i},
            )
            for i in range(n)
        ]

    def open_segment(self, segment: SegmentInfo) -> SegmentVolume:
        return _DemoSegment(segment)

    def load_ink_overlay(self, segment: SegmentInfo, volume_url: str | None = None):
        """A synthetic 'ink prediction' preview so the overlay path works offline: bright
        strokes on a low-res grid, roughly aligned to the depth-modulated ink in _synth."""
        seed = int(segment.extra.get("seed", 0))
        yy, xx = np.mgrid[0:750, 0:500].astype(np.float32)
        yf, xf = yy * 8.0, xx * 8.0  # this preview is ~8x downsampled from the surface grid
        lines = 0.5 + 0.5 * np.sin(2 * np.pi * yf / 140.0)
        strokes = 0.5 + 0.5 * np.tanh(
            3.0 * np.sin(2 * np.pi * xf / 55.0 + 2.0 * np.sin(yf / 13.0) + seed)
        )
        pred = (np.clip(lines * strokes, 0, 1) * 255).astype(np.uint8)
        return pred, f"demo synthetic prediction (seed {seed})"

    def open_scroll_volume(self, scroll: ScrollInfo) -> SegmentVolume:
        # Stand-in "raw volume" for the demo: a synthetic segment with many depth layers.
        info = SegmentInfo(f"{scroll.scroll_id}-volume", scroll,
                           scroll.resolution_um or 7.91, extra={"seed": 0})
        return _DemoSegment(info, deep=True)


class _DemoSegment(SegmentVolume):
    def __init__(self, segment: SegmentInfo, deep: bool = False):
        # A large virtual extent that is never materialised — regions are synthesized.
        # Depth is downsampled per level too, like real OME-Zarr pyramids. ``deep`` mimics a
        # raw scroll volume (hundreds of slices) vs a ~65-layer surface segment.
        z0 = 400 if deep else 65
        level_shapes = _pyramid(z0, 6000, 4000, n_levels=4)
        super().__init__(
            SegmentMeta(
                level_shapes=level_shapes,
                dtype="uint8",
                voxel_size_um=segment.voxel_size_um,
            )
        )
        seed = int(segment.extra.get("seed", 0))
        z0, y0, x0 = level_shapes[0]
        self._theta = 0.35 + 0.25 * seed           # fibre orientation (radians)
        self._fibre_period = 6.0 + 1.5 * seed      # full-res px per fibre (near Nyquist)
        self._center_z = z0 // 2
        self._cx, self._cy = x0 / 2.0, y0 / 2.0    # "scroll centre" for the winding
        self._wrap_pitch = 48.0                    # full-res px between wraps (the cepstrum peak)

    def _synth(self, zf, yf, xf, atten: float) -> np.ndarray:
        """Procedural intensity at full-res coords (broadcastable arrays). uint8."""
        ct, st = math.cos(self._theta), math.sin(self._theta)
        u = xf * ct + yf * st  # across-fibre axis
        fibre = 0.5 + 0.5 * atten * np.sin(2 * np.pi * u / self._fibre_period)
        base = 0.55 + 0.10 * np.sin(2 * np.pi * xf / 1500.0) * np.sin(2 * np.pi * yf / 1700.0)

        zc = (zf - self._center_z) / 8.0
        depth_gain = np.exp(-0.5 * zc * zc)
        lines = 0.5 + 0.5 * np.sin(2 * np.pi * yf / 140.0)
        strokes = 0.5 + 0.5 * np.tanh(3.0 * np.sin(2 * np.pi * xf / 55.0 + 2.0 * np.sin(yf / 13.0)))
        ink = depth_gain * lines * strokes

        # Wound-sheet radial periodicity: concentric wraps about (cx, cy). This is the signal
        # a radial profile's cepstrum should peak on, at quefrency == wrap pitch.
        r = np.sqrt((xf - self._cx) ** 2 + (yf - self._cy) ** 2)
        wraps = 0.5 + 0.5 * np.sin(2 * np.pi * r / self._wrap_pitch)

        observed = base * (1.0 - 0.6 * ink * fibre) * (0.6 + 0.4 * wraps)
        return (np.clip(observed, 0.0, 1.0) * 255).astype(np.uint8)

    def _read_block(self, level, z0, z1, y0, y1, x0, x1) -> np.ndarray:
        (Z0, Y0, X0), (Zl, Yl, Xl) = self.meta.level_shapes[0], self.meta.level_shapes[level]
        sy, sx, sz = Y0 / Yl, X0 / Xl, Z0 / Zl
        zf = (np.arange(z0, z1) * sz).reshape(-1, 1, 1).astype(np.float32)
        yf = (np.arange(y0, y1) * sy).reshape(1, -1, 1).astype(np.float32)
        xf = (np.arange(x0, x1) * sx).reshape(1, 1, -1).astype(np.float32)
        if zf.size == 0 or yf.size == 0 or xf.size == 0:
            return np.zeros((max(0, z1 - z0), max(0, y1 - y0), max(0, x1 - x0)), np.uint8)
        return self._synth(zf, yf, xf, atten=Xl / X0)

    def _read_points(self, level, zs, ys, xs) -> np.ndarray:
        (Z0, Y0, X0), (Zl, Yl, Xl) = self.meta.level_shapes[0], self.meta.level_shapes[level]
        zf = np.asarray(zs, np.float32) * (Z0 / Zl)
        yf = np.asarray(ys, np.float32) * (Y0 / Yl)
        xf = np.asarray(xs, np.float32) * (X0 / Xl)
        return self._synth(zf, yf, xf, atten=Xl / X0)

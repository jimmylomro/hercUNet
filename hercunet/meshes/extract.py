"""Per-window sheet-mesh extraction — turn one window's instance output into fitted sheet meshes, and write them.

The bridge from HercuViT's per-window prediction (``owner`` = argmax-query ownership, ``gate`` = surface mask) to the
geometry stored in an .asv. The fit is the fast KNN-free field solver (:func:`hercunet.meshes.field_fit.fit_sheets_field`
— per-sheet blurred-mask field + explicit gradient ascent), NOT the herculabels slab fit; it runs on the GPU next to
inference and returns ``{sheet_id: mesh}`` in window-local voxel coordinates (z,y,x).
"""
from __future__ import annotations

import numpy as np

from .field_fit import fit_sheets_field


def extract_window_sheets(owner, gate, *, voxel_um: float, sigma: float = 3.0, iters: int = 40, lr: float = 1.0,
                          lam: float = 0.3, step_um: float = 30.0, min_pts: int = 400, device: str = "cuda"):
    """Fit sheet meshes for one window from ``owner`` (D,H,W int sheet ids) and ``gate`` (D,H,W bool surface keep).
    Thin wrapper over the hercuvit field fit (see :func:`fit_sheets_field` for the parameters and the min-points /
    degenerate drops). Returns ``{sheet_id: mesh}`` (empty if no fittable sheet); verts are window-local voxel
    coords (z,y,x)."""
    return fit_sheets_field(owner, gate, voxel_um=voxel_um, sigma=sigma, iters=iters, lr=lr, lam=lam,
                            step_um=step_um, min_pts=min_pts, device=device)


class MeshSink:
    """Write per-window sheet meshes into an open :class:`~hercunet.meshes.asv.AsvWriter`, converting window-local
    voxel coords (z,y,x, at the working level) to global native-L0 XYZ. ``downsample`` is the working level's L0
    downsample factor (1.0 at level 0). One sink per worker shard; the engine calls :meth:`add_window` per window."""

    def __init__(self, writer, downsample: float = 1.0, name_prefix: str = ""):
        self.w = writer
        self.ds = float(downsample)
        self.pre = name_prefix
        self.n_windows = 0
        self.n_sheets = 0

    def add_window(self, origin_zyx, meshes) -> None:
        """``origin_zyx`` = the window's (z,y,x) start on the working-level grid; ``meshes`` = ``{sid: mesh}`` from
        :func:`extract_window_sheets` (verts window-local z,y,x). Each sheet is stored in global L0 XYZ."""
        oz, oy, ox = (float(v) for v in origin_zyx)
        off = np.array([oz, oy, ox], np.float32)
        wname = f"z{int(oz)}_y{int(oy)}_x{int(ox)}"
        self.n_windows += 1
        for sid, m in meshes.items():
            g_zyx = (m["verts"] + off) * self.ds                                     # window-local -> global L0 (z,y,x)
            self.w.add_sheet(m["foot"], g_zyx[:, ::-1], name=f"{self.pre}{wname}_s{int(sid)}", window=wname)
            self.n_sheets += 1

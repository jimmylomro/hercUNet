"""One orthogonal cross-section pane: a zoomable / pannable / slice-scrollable CT view with a
streamlet-point overlay and a debounced hover read-out.

Lean in-memory counterpart of the research ``hercureader.ui.viz_pane`` (which streams a segment in
tiles) — the window brick is a small array already in RAM, so this uses a plain pyqtgraph ViewBox +
ImageItem (the slice) + ScatterPlotItem (streamlet points near the slice). Plain wheel scrolls
through slices; Ctrl+wheel and right-drag zoom; left-drag pans. Fit-to-view is the maximum zoom-out.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QVBoxLayout, QWidget

# axis -> (along_dim, row_dim, col_dim) in the (z,y,x) block. np.take(block, s, along) yields a slice
# whose two remaining axes are (row_dim, col_dim); scatter uses (x=col, y=row).
_AX = {"z": (0, 1, 2), "y": (1, 0, 2), "x": (2, 0, 1)}


class _SmoothImageItem(pg.ImageItem):
    """ImageItem that upscales with bilinear smoothing so the streamlet-point overlay stays smooth
    and round when the view is zoomed in — instead of the chunky nearest-neighbour blocks a plain
    ImageItem shows. Applied ONLY to the point overlay; the CT slice keeps its own crisp ImageItem."""

    def paint(self, painter, *args):
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        super().paint(painter, *args)


class _SliceViewBox(pg.ViewBox):
    """ViewBox where a plain wheel emits a slice step; Ctrl+wheel falls through to normal zoom."""

    wheelStep = Signal(int)

    def wheelEvent(self, ev, axis=None):
        if ev.modifiers() & Qt.ControlModifier:
            f = 0.8 if ev.delta() > 0 else 1.25              # zoom in / out about the VIEW CENTRE (no drift to mouse)
            c = self.viewRect().center()
            self.scaleBy(s=(f, f), center=pg.Point(c.x(), c.y()))
            ev.accept()
            return
        self.wheelStep.emit(1 if ev.delta() > 0 else -1)
        ev.accept()


class SlicePane(QWidget):
    """Cross-section pane for one axis. Emit :attr:`hovered` (a formatted string) as the cursor
    moves (debounced) and :attr:`sliceChanged` when the depth changes."""

    hovered = Signal(str, int, str)                             # coords text, sheet id (-1 noise, -2 none), colour hex
    sliceChanged = Signal(str, int)                              # axis, LOCAL slice index (for the 3D section plane)
    picked = Signal(int, bool)                                   # click: sheet id (-2 = air gap / noise), ctrl held

    def __init__(self, axis: str = "z"):
        super().__init__()
        assert axis in _AX
        self.axis = axis
        self._along, self._row, self._col = _AX[axis]
        self._block = None
        self._corner = (0, 0, 0)                                 # global (z0,y0,x0)
        self._voxel_um = 1.0
        self._slice = 0
        self._pts = None                                         # [N,3] local (z,y,x) — unsorted, for hover
        self._plab = None                                        # per-point cluster id — unsorted, for hover
        # points pre-SORTED by the along-axis coord so each scroll is a searchsorted band (not an O(N) mask):
        self._a_s = self._xs_s = self._ys_s = self._plab_s = None
        self._rgba_s = None                                      # per-point RGBA (sorted) post-cluster, else None
        self._uniform_rgba = np.array([200, 200, 205, 150], np.uint8)  # pre-cluster colour (all one)
        self._cluster_hex = None                                 # {cid: '#rrggbb'} for the hover read-out
        self._band = 1
        self._step = 1                                           # slices advanced per wheel notch
        self._dot_radius_vox = 1.0                               # streamlet-dot radius in voxels
        self._dot_alpha = 220
        self._stamp = None                                       # cached (ss, dr, dc, coverage) round stamp
        self._sel = None                                         # selected sheet ids (others fade), or None

        self._glw = pg.GraphicsLayoutWidget()
        self._vb = _SliceViewBox(lockAspect=True, invertY=True, enableMenu=False)
        self._vb.setMouseMode(pg.ViewBox.PanMode)                # left-drag pans, right-drag zooms
        self._vb.wheelStep.connect(self._step_slice)
        self._vb.sigResized.connect(self._fit)                   # re-fill the pane on every resize
        self._glw.addItem(self._vb, row=0, col=0, colspan=2)
        self._img = pg.ImageItem(axisOrder="row-major")
        self._vb.addItem(self._img)
        self._conf = None                                        # low-confidence region volume (deleted sheets)
        self._conf_img = _SmoothImageItem(axisOrder="row-major")  # red translucent region; hidden unless in conf mode
        self._conf_img.setVisible(False)
        self._vb.addItem(self._conf_img)
        # streamlet points are rasterised into an RGBA overlay (one setImage) — ScatterPlotItem is far too
        # slow at ~10^5 pts/slice (~70-140ms/scroll). This is O(visible) numpy + one texture upload (~5ms).
        # The overlay is SUPERSAMPLED and drawn with round soft-alpha dots + a smoothing ImageItem so the
        # points stay round and crisp when zoomed (a plain native-res square-stamp overlay pixelates badly).
        self._pts_img = _SmoothImageItem(axisOrder="row-major")
        self._vb.addItem(self._pts_img)

        self._title = pg.LabelItem(justify="left")
        self._glw.addItem(self._title, row=1, col=0)
        hint = pg.LabelItem(justify="right")                     # controls, obvious per spec
        hint.setText("scroll: wheel  ·  zoom: ⌃+wheel / right-drag  ·  pan: drag", size="8pt", color="#888888")
        self._glw.addItem(hint, row=1, col=1)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._glw)

        self._vb.scene().sigMouseMoved.connect(self._on_mouse)
        self._vb.scene().sigMouseClicked.connect(self._on_click)  # click a sheet → select (see picked)
        self._hover_timer = QTimer(self)                          # debounce hover work
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(40)
        self._hover_timer.timeout.connect(self._emit_hover)
        self._last_scene_pos = None

    # ---- data ----------------------------------------------------------------
    def set_volume(self, block, corner, voxel_um) -> None:
        self._block = np.asarray(block, np.float32)
        self._corner = tuple(int(c) for c in corner)
        self._voxel_um = float(voxel_um)
        self._slice = self._block.shape[self._along] // 2
        self._render_slice()
        self._fit()

    def set_points(self, pts, plab=None, brush_by=None, uniform_brush=None) -> None:
        """``pts`` [N,3] local (z,y,x). Pre-cluster: pass ``uniform_brush`` (one colour). Post-cluster:
        pass ``plab`` (per-point cluster id) + ``brush_by`` ({cid: QBrush}). Points are sorted ONCE by
        this pane's slice axis so scrolling gathers a slice band by searchsorted; rendering rasterises."""
        if uniform_brush is not None:
            self._uniform_rgba = np.array(uniform_brush.color().getRgb(), np.uint8)
        pts = None if pts is None else np.asarray(pts, np.float32)
        plab = None if plab is None else np.asarray(plab)
        if plab is not None and pts is not None:                 # drop noise (-1): never rendered OR picked
            keep = plab >= 0
            pts, plab = pts[keep], plab[keep]
        if pts is None or len(pts) == 0:
            self._pts = self._plab = self._a_s = self._xs_s = self._ys_s = self._plab_s = self._rgba_s = None
            self._pts_img.clear()
            return
        self._pts = pts
        self._plab = plab
        a = np.round(pts[:, self._along]).astype(np.int32)       # sort by the along-axis integer slice
        order = np.argsort(a, kind="stable")
        self._a_s = a[order]
        self._xs_s = np.ascontiguousarray(pts[order, self._col])
        self._ys_s = np.ascontiguousarray(pts[order, self._row])
        self._plab_s = None if self._plab is None else self._plab[order]
        if brush_by is not None and self._plab_s is not None:    # per-point RGBA via a cid→rgba LUT (vectorised)
            maxc = int(self._plab_s.max()) if len(self._plab_s) else 0
            lut = np.zeros((maxc + 2, 4), np.uint8)              # row = cid + 1  (noise -1 → row 0)
            for cid, br in brush_by.items():
                if -1 <= int(cid) <= maxc:
                    lut[int(cid) + 1] = br.color().getRgb()
            self._rgba_s = lut[self._plab_s + 1]
        else:
            self._rgba_s = None                                 # uniform colour
        self._render_points()

    def set_cluster_colours(self, hex_by_cid: dict) -> None:
        self._cluster_hex = dict(hex_by_cid)

    def set_streamlets_visible(self, on: bool) -> None:
        self._pts_img.setVisible(bool(on))

    def set_confidence(self, field) -> None:
        """Set the low-confidence (uncertainty) volume — a block-shaped float array in [0,1] (0 = certain,
        1 = fully uncertain), or None. Rendered per-slice as graded translucent red in confidence mode."""
        self._conf = None if field is None else np.asarray(field, np.float32)
        self._render_conf()

    def set_confidence_visible(self, on: bool) -> None:
        self._conf_img.setVisible(bool(on))
        self._render_conf()

    def _render_conf(self) -> None:
        if self._conf is None or self._block is None or not self._conf_img.isVisible():
            self._conf_img.clear()
            return
        s = int(np.clip(self._slice, 0, self._conf.shape[self._along] - 1))
        sl = np.take(self._conf, s, axis=self._along).astype(np.float32)  # [row, col] uncertainty 0..1
        H, W = self._block.shape[self._row], self._block.shape[self._col]
        overlay = np.zeros((H, W, 4), np.uint8)
        if sl.shape == (H, W):
            overlay[..., 0] = 224                                # constant red; graded alpha (no smoothing halo)
            overlay[..., 1] = 48
            overlay[..., 2] = 48
            overlay[..., 3] = (np.clip(sl, 0.0, 1.0) * 165).astype(np.uint8)
        self._conf_img.setImage(overlay)

    def set_scroll_step(self, n: int) -> None:
        self._step = max(1, int(n))

    def set_selection(self, sel) -> None:
        """Highlight only these sheet ids (others fade to a faint alpha). ``None``/empty = no selection."""
        self._sel = list(sel) if sel else None
        self._render_points()

    # ---- rendering -----------------------------------------------------------
    def _render_slice(self) -> None:
        if self._block is None:
            return
        s = int(np.clip(self._slice, 0, self._block.shape[self._along] - 1))
        self._slice = s
        img = np.take(self._block, s, axis=self._along)
        lo, hi = np.percentile(img, [1, 99])
        norm = np.clip((img - lo) / (hi - lo + 1e-9), 0.0, 1.0).astype(np.float32)
        self._img.setImage(norm, levels=(0.0, 1.0))              # float image needs explicit levels
        gz = self._corner[self._along] + s
        self._title.setText(f"{self.axis}-slice — global {'zyx'[self._along]}={gz}", size="9pt")
        self._render_points()
        self._render_conf()
        self.sliceChanged.emit(self.axis, s)

    def _fit(self) -> None:
        """Fill the pane: aspect-locked fit so the slice's short side spans the pane's short side, with
        max-zoom-out clamped to this fit and a sensible max zoom-in. Re-run on every pane resize."""
        if self._block is None:
            return
        # Clear ALL stale limits first (pan bounds + range caps): otherwise autoRange is constrained by the
        # PREVIOUS fit's limits (a different pane aspect) and can't reach the true fit — which then locks in
        # too-tight (zoom-out sticks). Clear → fit unconstrained → re-set limits to the real fit rectangle.
        self._vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None, maxXRange=None, maxYRange=None)
        self._vb.autoRange(padding=0)                            # aspect-locked → fills the short side, no margin
        (x0, x1), (y0, y1) = self._vb.viewRange()
        xs, ys = x1 - x0, y1 - y0
        # pan bounds = the fit rectangle itself → the image can NEVER be dragged out of the pane (at fit it's
        # pinned; zoomed in you pan only within the fitted image). maxXRange/maxYRange clamp zoom-out to fit.
        self._vb.setLimits(xMin=x0, xMax=x1, yMin=y0, yMax=y1,
                           maxXRange=xs, maxYRange=ys,
                           minXRange=max(4.0, xs / 50.0), minYRange=max(4.0, ys / 50.0))  # max zoom-in

    def _dot_stamp(self, ss: int):
        """Round soft-edged dot footprint at supersample ``ss``: integer (dr, dc) offsets + a per-offset
        uint8 alpha (anti-aliased rim), sorted so the opaque centre is written LAST (wins over rims).
        Cached per ``ss`` — it only changes when the pane size (hence ss) changes."""
        if self._stamp is not None and self._stamp[0] == ss:
            return self._stamp[1:]
        R = self._dot_radius_vox * ss
        r = int(np.ceil(R))
        dr, dc, al = [], [], []
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                cov = R + 0.5 - (i * i + j * j) ** 0.5           # 1 inside, ramps to 0 across the last voxel
                if cov > 0.0:
                    dr.append(i); dc.append(j); al.append(min(1.0, cov))
        order = np.argsort(al)                                   # ascending → opaque centre painted last
        dr = np.asarray(dr, np.intp)[order]
        dc = np.asarray(dc, np.intp)[order]
        cov = np.asarray(al, np.float32)[order]                  # coverage 0..1 (alpha applied per-render)
        self._stamp = (ss, dr, dc, cov)
        return dr, dc, cov

    def _render_points(self) -> None:
        if self._a_s is None or self._block is None or len(self._a_s) == 0:
            self._pts_img.clear()
            return
        lo = int(np.searchsorted(self._a_s, self._slice - self._band, "left"))   # band = a contiguous run
        hi = int(np.searchsorted(self._a_s, self._slice + self._band, "right"))
        H, W = self._block.shape[self._row], self._block.shape[self._col]
        ss = int(np.clip(1800 // max(H, W, 1), 1, 3))            # supersample so dots are round, not blocky
        Hs, Ws = H * ss, W * ss
        overlay = np.zeros((Hs, Ws, 4), np.uint8)               # supersampled RGBA overlay, mapped back to (W,H)
        if hi > lo:
            rows = np.clip(np.round(self._ys_s[lo:hi] * ss).astype(np.intp), 0, Hs - 1)
            cols = np.clip(np.round(self._xs_s[lo:hi] * ss).astype(np.intp), 0, Ws - 1)
            rgb = (self._rgba_s[lo:hi, :3] if self._rgba_s is not None
                   else self._uniform_rgba[None, :3])           # dot colour (RGB); alpha comes from the stamp
            dim = None                                          # per-point alpha multiplier when a selection is active
            if self._sel and self._plab_s is not None:
                lab = self._plab_s[lo:hi]
                dim = np.where(np.isin(lab, self._sel), 1.0, 0.45).astype(np.float32)  # non-selected fade
                o = np.argsort(dim, kind="stable")             # draw selected LAST so they win any overlap
                rows, cols, rgb, dim = rows[o], cols[o], rgb[o], dim[o]
            dr, dc, cov = self._dot_stamp(ss)
            for k in range(len(dr)):                            # round stamp: one vectorised write per rim ring
                rr = np.clip(rows + dr[k], 0, Hs - 1)
                cc = np.clip(cols + dc[k], 0, Ws - 1)
                overlay[rr, cc, :3] = rgb                       # constant RGB across the footprint → no dark halo
                a = cov[k] * self._dot_alpha
                overlay[rr, cc, 3] = np.uint8(a) if dim is None else (a * dim).astype(np.uint8)
        self._pts_img.setImage(overlay)
        self._pts_img.setRect(0.0, 0.0, float(W), float(H))     # map the supersampled overlay onto the slice

    def _step_slice(self, d: int) -> None:
        self._slice += d * self._step
        self._render_slice()

    # ---- picking (shared by hover + click) -----------------------------------
    def _view_xy(self, scene_pos):
        """Scene point → (xi, yi) col/row in the slice, or None if outside the image plane."""
        if self._block is None or scene_pos is None:
            return None
        if not self._vb.sceneBoundingRect().contains(scene_pos):
            return None
        p = self._vb.mapSceneToView(scene_pos)
        xi, yi = int(round(p.x())), int(round(p.y()))            # col, row in the slice
        loc = [0, 0, 0]
        loc[self._along] = self._slice
        loc[self._row] = yi
        loc[self._col] = xi
        shp = self._block.shape
        if not (0 <= loc[0] < shp[0] and 0 <= loc[1] < shp[1] and 0 <= loc[2] < shp[2]):
            return None
        return xi, yi

    def _sheet_under(self, xi: int, yi: int, r2: float = 25.0) -> int:
        """Sheet id of the nearest streamlet point within ``sqrt(r2)`` px of (xi,yi) on this slice,
        else -2 (nothing near = air gap). Used for the hover read-out and for click-selection."""
        if self._pts is None or self._plab is None or not len(self._pts):
            return -2
        d2 = ((self._pts[:, self._row] - yi) ** 2 + (self._pts[:, self._col] - xi) ** 2
              + (self._pts[:, self._along] - self._slice) ** 2)
        j = int(np.argmin(d2))
        return int(self._plab[j]) if d2[j] <= r2 else -2

    # ---- hover ---------------------------------------------------------------
    def _on_mouse(self, scene_pos) -> None:
        self._last_scene_pos = scene_pos
        self._hover_timer.start()

    def _emit_hover(self) -> None:
        xy = self._view_xy(self._last_scene_pos)
        if xy is None:
            return
        xi, yi = xy
        loc = [0, 0, 0]
        loc[self._along] = self._slice
        loc[self._row] = yi
        loc[self._col] = xi
        gz = self._corner[0] + loc[0]; gy = self._corner[1] + loc[1]; gx = self._corner[2] + loc[2]
        coords = f"global  z={gz}   y={gy}   x={gx}"
        sheet_id = self._sheet_under(xi, yi, 25.0)               # within ~5px of a streamlet point
        colour = ""
        if sheet_id >= 0 and self._cluster_hex:
            colour = self._cluster_hex.get(sheet_id, "")
        self.hovered.emit(coords, sheet_id, colour)

    # ---- click-to-select -----------------------------------------------------
    def _on_click(self, ev) -> None:
        if self._block is None or ev.button() != Qt.LeftButton:
            return
        ctrl = bool(ev.modifiers() & Qt.ControlModifier)
        xy = self._view_xy(ev.scenePos())
        if xy is None:                                           # clicked outside the image → treat as air gap
            self.picked.emit(-2, ctrl)
            ev.accept()
            return
        self.picked.emit(self._sheet_under(*xy, 64.0), ctrl)     # generous ~8px click radius
        ev.accept()

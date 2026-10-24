"""3D pane: the fitted cluster sheet-meshes in an orbit/pan/zoom GL view.

Stays in a loading state (its own overlay) from the moment the window opens until the meshes arrive —
i.e. through the whole cluster/mesh computation. Hover reports the cluster id of the nearest sheet
(screen-space nearest-centroid pick; no coordinates, per spec).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.opengl.shaders import FragmentShader, ShaderProgram, VertexShader
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QVector3D
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from .overlay import LoadingOverlay
from .toggle import SegmentedToggle

# A brighter replacement for pyqtgraph's 'shaded' shader (which is colour*(0.2 + 0.8*max(dot,0)) — a single
# light, so back-facing sheets go to 20%). This lights BOTH sides (abs(dot)) with a high ambient floor, so
# every sheet reads clearly. Vertex stage copied from the built-in 'shaded'; only the fragment differs.
_BRIGHT_SHADER = ShaderProgram('hercu_bright', [
    VertexShader("""
        uniform mat4 u_mvp;
        uniform mat3 u_normal;
        attribute vec4 a_position;
        attribute vec3 a_normal;
        attribute vec4 a_color;
        varying vec4 v_color;
        varying vec3 v_normal;
        void main() {
            v_normal = normalize(u_normal * a_normal);
            v_color = a_color;
            gl_Position = u_mvp * a_position;
        }
    """),
    FragmentShader("""
        #ifdef GL_ES
        precision mediump float;
        #endif
        varying vec4 v_color;
        varying vec3 v_normal;
        void main() {
            float p = abs(dot(v_normal, normalize(vec3(1.0, -1.0, -1.0))));
            vec3 rgb = v_color.rgb * clamp(0.6 + 0.55 * p, 0.0, 1.0);
            gl_FragColor = vec4(rgb, v_color.a);
        }
    """),
])


class _OrbitView(gl.GLViewWidget):
    """GLViewWidget that reports mouse position for hover picking, and a click (a press+release that
    did NOT orbit/drag) for sheet selection."""

    moved = Signal(float, float)
    clicked = Signal(float, float, object)                       # x, y, keyboard modifiers

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._press = None

    def mouseMoveEvent(self, ev):
        super().mouseMoveEvent(ev)
        self.moved.emit(ev.position().x(), ev.position().y())

    def mousePressEvent(self, ev):
        self._press = ev.position()
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        if self._press is not None:
            dx = ev.position().x() - self._press.x()
            dy = ev.position().y() - self._press.y()
            if dx * dx + dy * dy <= 9.0:                         # moved < 3px → a click, not an orbit
                self.clicked.emit(ev.position().x(), ev.position().y(), ev.modifiers())
            self._press = None


class MeshPane(QWidget):
    hovered = Signal(str, int, str)                             # info text, sheet id (-2 none), colour hex
    picked = Signal(int, bool)                                  # click: sheet id (-2 = empty space), ctrl held

    def __init__(self):
        super().__init__()
        self._view = _OrbitView()
        self._view.setMouseTracking(True)                       # hover events without a pressed button (else no update)
        self._view.setCameraPosition(distance=300)
        self._view.setBackgroundColor((36, 40, 48))              # lighter than near-black so sheets read brighter
        self._items = []
        self._base_cols = []                                     # base (r,g,b,a) per item — restored on deselect
        self._sel = None                                         # selected sheet ids (others dimmed), or None
        self._centroids = np.zeros((0, 3), np.float32)           # (x,y,z) per cluster
        self._cids = []                                          # cluster id per centroid (aligned with _items)
        self._hex = {}                                           # cluster id -> '#rrggbb' for the hover swatch
        self._pick_ok = True                                     # 3D hover pick; disabled after one loud failure
        self._shape = None                                       # (Z,Y,X) block extent
        self._planes = {}                                        # axis -> GLMeshItem (faint red section plane)
        self._plane_faces = np.array([[0, 1, 2], [0, 2, 3]])
        self._slices = {"z": 0, "y": 0, "x": 0}
        self._planes_visible = False
        # ---- view mode: sheet meshes ↔ 3-D PCA of the embedding ----
        self._mode = 0                                          # 0 = sheets, 1 = embeddings
        self._scatter = None                                   # GLScatterPlotItem for the embedding cloud
        self._emb_pts = None                                   # [n_units, 3] PCA positions
        self._emb_lab = None                                   # per-unit cluster id
        self._emb_colours = {}                                 # {cid: (r,g,b,a)} for the cloud
        self._emb_cent = {}                                    # {cid: (x,y,z)} PCA centroid, for picking
        self._emb_src = None                                   # id() of the E the PCA was computed from
        self._sheet_center = None                             # cached camera framing for the sheets view
        self._sheet_span = 100.0

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._view, 1)
        band = QWidget()                                       # black bottom band, like the slice panes
        band.setStyleSheet("background:#000000;")
        bl = QHBoxLayout(band)
        bl.setContentsMargins(8, 4, 8, 4)
        self._view_mode = SegmentedToggle(["sheets", "embeddings"], current=0, seg_width=94)
        bl.addStretch(1)
        bl.addWidget(self._view_mode)
        bl.addStretch(1)
        lay.addWidget(band)
        self._view_mode.changed.connect(self._on_view_mode)
        self._overlay = LoadingOverlay(self)                   # covers ONLY the GL view (not the band below)
        self._show_overlay("computing streamlets + clusters + sheets…")

        self._view.moved.connect(self._on_move)
        self._view.clicked.connect(self._on_click)
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(50)
        self._hover_timer.timeout.connect(self._emit_hover)
        self._last_xy = None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._overlay.setGeometry(self._view.geometry())         # cover the GL view only, leave the band clear

    def _show_overlay(self, text: str) -> None:
        self._overlay.set_message(text)
        self._overlay.setGeometry(self._view.geometry())
        self._overlay.raise_()
        self._overlay.show()

    def set_status(self, text: str) -> None:
        self._overlay.set_message(text)

    def reset(self) -> None:
        """Clear all sheet meshes + the embedding cloud and re-show the overlay — for a new window.
        Keeps the sheets/embeddings mode (the toggle position) across windows."""
        for it in self._items:
            self._view.removeItem(it)
        self._items = []
        self._base_cols = []
        self._sel = None
        self._centroids = np.zeros((0, 3), np.float32)
        self._cids = []
        self._hex = {}
        if self._scatter is not None:
            self._view.removeItem(self._scatter)
            self._scatter = None
        self._emb_pts = self._emb_lab = self._emb_src = None
        self._emb_colours = {}
        self._emb_cent = {}
        self._sheet_center = None
        self._show_overlay("computing streamlets + clusters + sheets…")

    # ---- cross-section planes ----
    def set_extent(self, shape) -> None:
        """Give the 3D view the block extent (Z,Y,X) so it can draw the three cross-section planes.
        GL axes map to block (x,y,z): x∈[0,X], y∈[0,Y], z∈[0,Z] (meshes are plotted the same way)."""
        self._shape = tuple(int(s) for s in shape)
        self._slices = {"z": self._shape[0] // 2, "y": self._shape[1] // 2, "x": self._shape[2] // 2}
        for it in self._planes.values():
            self._view.removeItem(it)
        self._planes = {}
        for ax in ("z", "y", "x"):
            it = gl.GLMeshItem(vertexes=self._plane_verts(ax, self._slices[ax]), faces=self._plane_faces,
                               smooth=False, drawEdges=True, edgeColor=(1.0, 0.25, 0.25, 0.55),
                               color=(1.0, 0.12, 0.12, 0.13), glOptions="translucent")
            it.setVisible(self._planes_visible)
            self._view.addItem(it)
            self._planes[ax] = it

    def _plane_verts(self, axis: str, s: int):
        Z, Y, X = self._shape
        if axis == "z":
            v = [[0, 0, s], [X, 0, s], [X, Y, s], [0, Y, s]]
        elif axis == "y":
            v = [[0, s, 0], [X, s, 0], [X, s, Z], [0, s, Z]]
        else:                                                    # x
            v = [[s, 0, 0], [s, Y, 0], [s, Y, Z], [s, 0, Z]]
        return np.array(v, np.float32)

    def set_plane(self, axis: str, s: int) -> None:
        """Move one section plane to the slice ``s`` (called as a 2D pane is scrolled)."""
        if self._shape is None or axis not in self._planes:
            return
        self._slices[axis] = int(s)
        self._planes[axis].setMeshData(vertexes=self._plane_verts(axis, int(s)), faces=self._plane_faces)

    def set_planes_visible(self, on: bool) -> None:
        self._planes_visible = bool(on)
        for it in self._planes.values():
            it.setVisible(self._planes_visible and self._mode == 0)   # section planes are a sheets-view thing only

    def set_meshes(self, meshes: dict, colours: dict, recenter: bool = True) -> None:
        """``meshes`` = {cluster_id: mesh_dict with 'V' [nu,nv,3] and 'tris'}; ``colours`` =
        {cluster_id: (r,g,b,a) 0..1}. Builds one GLMeshItem per cluster and drops the overlay.
        ``recenter`` re-frames the camera on the meshes — set False on an edit to keep the user's orbit."""
        for it in self._items:
            self._view.removeItem(it)
        self._items = []
        self._base_cols = []
        self._sel = None
        self._hex = {int(c): "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
                     for c, (r, g, b, a) in colours.items()}
        cents, cids = [], []
        allv = []
        for cid, m in meshes.items():
            V = np.asarray(m["V"], np.float32).reshape(-1, 3)    # (z,y,x)
            tris = np.asarray(m["tris"])
            if len(tris) == 0 or len(V) == 0:
                continue
            verts = np.stack([V[:, 2], V[:, 1], V[:, 0]], axis=1)  # → (x,y,z) for GL
            col = colours.get(int(cid), (0.8, 0.8, 0.8, 1.0))
            item = gl.GLMeshItem(vertexes=verts, faces=tris, smooth=True, drawEdges=False,
                                 color=col, shader=_BRIGHT_SHADER, glOptions="opaque")
            self._view.addItem(item)
            self._items.append(item)
            self._base_cols.append(tuple(float(x) for x in col))
            used = verts[np.unique(tris)]
            cents.append(used.mean(0)); cids.append(int(cid)); allv.append(used)
        if allv:
            allv = np.concatenate(allv, 0)
            self._sheet_center = allv.mean(0)
            self._sheet_span = float(np.linalg.norm(allv.max(0) - allv.min(0))) or 100.0
            if recenter and self._mode == 0:
                self._view.opts["center"] = QVector3D(*self._sheet_center)
                self._view.setCameraPosition(distance=1.4 * self._sheet_span)
        for it in self._items:
            it.setVisible(self._mode == 0)                      # hidden in embeddings mode
        self._centroids = np.asarray(cents, np.float32) if cents else np.zeros((0, 3), np.float32)
        self._cids = cids
        self._overlay.hide()

    # ---- embedding (3-D PCA) view ----
    def set_embedding(self, E, ulab, colours) -> None:
        """Provide the per-unit embedding ``E`` [n,≥1] + current per-unit cluster ids ``ulab`` +
        ``{cid: (r,g,b,a)}`` colours. Reduces ``E`` to 3-D by PCA (cached per ``E``) and (re)builds the
        embedding cloud, coloured by cluster — so it tracks split/merge/delete/undo like the sheets do."""
        E = np.asarray(E, np.float32)
        ulab = np.asarray(ulab)
        if E.ndim != 2 or len(E) == 0:                          # empty/edge window → nothing to show
            return
        recompute = self._emb_src != id(E) or self._emb_pts is None   # a NEW embedding (window), not an edit
        if recompute:
            mu = E.mean(0)
            try:
                _, _, Vt = np.linalg.svd(E - mu, full_matrices=False)
            except Exception:
                return
            p = (E - mu) @ Vt[:min(3, Vt.shape[0])].T
            if p.shape[1] < 3:
                p = np.pad(p, ((0, 0), (0, 3 - p.shape[1])))
            span = float(np.ptp(p, axis=0).max()) or 1.0
            self._emb_pts = (p / span * 120.0).astype(np.float32)  # comfortable view scale
            self._emb_src = id(E)
        self._emb_lab = ulab
        self._emb_colours = dict(colours)
        self._emb_cent = {int(c): self._emb_pts[ulab == c].mean(0)
                          for c in np.unique(ulab) if c >= 0}
        self._emb_render()
        if self._mode == 1 and recompute:                      # frame a NEW embedding only — keep the view on edits
            self._recenter_embedding()

    def _emb_point_colours(self):
        n = len(self._emb_pts)
        out = np.tile(np.array([0.45, 0.45, 0.45, 0.22], np.float32), (n, 1))  # noise/default faint grey
        for c, col in self._emb_colours.items():
            m = self._emb_lab == int(c)
            if not m.any():
                continue
            r, g, b = float(col[0]), float(col[1]), float(col[2])
            if self._sel is not None and int(c) not in self._sel:
                out[m] = (r * 0.32, g * 0.32, b * 0.32, 0.5)   # dim non-selected cluster
            else:
                out[m] = (r, g, b, 1.0)
        return out

    def _emb_render(self):
        if self._emb_pts is None:
            return
        cols = self._emb_point_colours()
        if self._scatter is None:
            self._scatter = gl.GLScatterPlotItem(pos=self._emb_pts, color=cols, size=6.0, pxMode=True)
            self._scatter.setGLOptions("translucent")
            self._view.addItem(self._scatter)
        else:
            self._scatter.setData(pos=self._emb_pts, color=cols)
        self._scatter.setVisible(self._mode == 1)

    def _recenter_embedding(self):
        if self._emb_pts is None or not len(self._emb_pts):
            return
        c = self._emb_pts.mean(0)
        self._view.opts["center"] = QVector3D(float(c[0]), float(c[1]), float(c[2]))
        span = float(np.linalg.norm(self._emb_pts.max(0) - self._emb_pts.min(0))) or 100.0
        self._view.setCameraPosition(distance=1.4 * span)

    def _recenter_sheets(self):
        if self._sheet_center is None:
            return
        self._view.opts["center"] = QVector3D(*[float(x) for x in self._sheet_center])
        self._view.setCameraPosition(distance=1.4 * self._sheet_span)

    def _on_view_mode(self, idx):
        self._mode = int(idx)
        for it in self._items:
            it.setVisible(self._mode == 0)
        for it in self._planes.values():
            it.setVisible(self._planes_visible and self._mode == 0)
        if self._scatter is not None:
            self._scatter.setVisible(self._mode == 1)
        if self._mode == 1:
            self._emb_render()
            self._recenter_embedding()
        else:
            self._recenter_sheets()

    # ---- selection ----
    def set_selection(self, sel) -> None:
        """Highlight only these sheet ids; dim the rest (darkened, since the opaque shader ignores alpha).
        ``None``/empty restores every sheet's base colour."""
        self._sel = set(int(c) for c in sel) if sel else None
        for it, cid, col in zip(self._items, self._cids, self._base_cols):
            if self._sel is None or int(cid) in self._sel:
                it.setColor(col)
            else:
                r, g, b, a = col
                it.setColor((r * 0.5, g * 0.5, b * 0.5, a))     # dimmed (darker) non-selected sheet
        if self._scatter is not None and self._emb_pts is not None:
            self._scatter.setData(color=self._emb_point_colours())  # same dimming for the embedding cloud

    def _pick_cid(self, mx: float, my: float) -> int:
        """Cluster id of the nearest cluster centroid within ~80px of screen (mx,my), else -2. Uses the
        sheet-mesh centroids in sheets mode, the PCA cluster centroids in embeddings mode."""
        if not self._pick_ok:
            return -2
        pairs = (list(self._emb_cent.items()) if self._mode == 1     # (cid, xyz)
                 else list(zip(self._cids, [tuple(c) for c in self._centroids])))
        if not pairs:
            return -2
        try:
            vp = self._view.getViewport()                        # (x,y,w,h); projectionMatrix needs region+viewport
            mvp = self._view.projectionMatrix(vp, vp) * self._view.viewMatrix()
        except Exception as e:
            # NOT silent: surface it once (a genuine API/GL problem), then disable to avoid per-move spam.
            self._pick_ok = False
            import sys
            print(f"[viz] 3D pick disabled — projection error: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            return -2
        w, h = self._view.width(), self._view.height()
        best, best_d = -2, 1e18
        for cid, c in pairs:
            clip = mvp.map(QVector3D(float(c[0]), float(c[1]), float(c[2])))
            sx = (clip.x() * 0.5 + 0.5) * w
            sy = (1.0 - (clip.y() * 0.5 + 0.5)) * h
            d = (sx - mx) ** 2 + (sy - my) ** 2
            if d < best_d:
                best_d, best = d, int(cid)
        return best if best_d <= 80.0 ** 2 else -2

    def _on_move(self, x, y):
        self._last_xy = (x, y)
        self._hover_timer.start()

    def _emit_hover(self):
        if self._last_xy is None:
            return
        cid = self._pick_cid(*self._last_xy)
        if cid >= 0:
            self.hovered.emit("", cid, self._hex.get(cid, ""))
        else:
            self.hovered.emit("", -2, "")                        # nothing under the cursor → clear the swatch

    def _on_click(self, x, y, mods):
        self.picked.emit(self._pick_cid(x, y), bool(mods & Qt.ControlModifier))

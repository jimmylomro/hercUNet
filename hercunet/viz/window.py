"""The interactive viewer window: three orthogonal cross-sections + a 3D sheet view + a log pane,
fed progressively by the pipeline worker over Qt signals.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThreadPool, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QComboBox, QGridLayout, QHBoxLayout, QLabel, QMainWindow,
                               QPlainTextEdit, QPushButton, QSizePolicy, QSplitter, QVBoxLayout,
                               QWidget)

from .mesh_pane import MeshPane
from .overlay import LoadingOverlay
from .slice_pane import SlicePane
from .swatch import SheetSwatch
from .toggle import LabelledToggle, SegmentedToggle
from .worker import EditWorker, PipelineWorker

_COMBO_QSS = """
QComboBox { background:#2b2f38; color:#c9ccd2; border:1px solid #3a3f4a; border-radius:13px;
            padding:2px 10px; min-height:22px; font-size:12px; }
QComboBox::drop-down { border:0; width:18px; }
QComboBox QAbstractItemView { background:#2b2f38; color:#c9ccd2; selection-background-color:#3b82f6; }
"""

_BTN_QSS = """
QPushButton { background:#2b2f38; color:#c9ccd2; border:1px solid #3a3f4a; border-radius:8px;
              padding:4px 14px; font-size:12px; font-weight:600; }
QPushButton:hover:enabled { background:#333844; }
QPushButton:disabled { color:#676c76; border-color:#2f333c; background:#262a32; }
"""

_DEL_QSS = """
QPushButton { background:#3a2226; color:#f0b6bd; border:1px solid #5a2a30; border-radius:8px;
              padding:4px 14px; font-size:12px; font-weight:600; }
QPushButton:hover:enabled { background:#4a2a30; }
QPushButton:disabled { color:#7a4a50; border-color:#4a2228; background:#301c20; }
"""

_GREY = pg.mkBrush(200, 200, 205, 150)


def _palette():
    """The 60-colour distinct palette (tab20 + tab20b + tab20c) — same as the CLI montage."""
    import matplotlib.pyplot as plt
    return [tuple(float(x) for x in c)
            for name in ("tab20", "tab20b", "tab20c") for c in plt.get_cmap(name).colors]


def _initial_colours(plab, pts, voxel_um):
    """Adjacency-aware initial colouring ``{cid: (r,g,b)}`` (touching sheets forced to different palette
    entries), excluding noise. Same ``cluster_colors`` graph-colouring the CLI montage uses. This map is
    then held PERSISTENT for the session so an edit keeps every existing sheet's colour."""
    from hercunet.labels.common.scales import cluster_colors
    raw = cluster_colors(np.asarray(plab), polylines=pts[:, None, :], voxel_um=voxel_um)
    return {int(c): (float(v[0]), float(v[1]), float(v[2])) for c, v in raw.items() if int(c) >= 0}


def _pick_new_colour(colours):
    """A palette colour not currently used by any sheet, so a freshly-created sheet is visually distinct
    from every existing one (falls back to a random colour only if the whole palette is in use)."""
    used = {tuple(round(float(x), 3) for x in c) for c in colours.values()}
    for col in _palette():
        if tuple(round(x, 3) for x in col) not in used:
            return col
    import random
    return (random.random(), random.random(), random.random())


def _brush_maps(colours):
    """``{cid: (r,g,b)}`` → (brush_by {cid:QBrush}, gl_by {cid:(r,g,b,1)}, hex_by {cid:'#rrggbb'}). Noise
    (-1) is intentionally ABSENT → its LUT row stays transparent, so noise/deleted points don't render
    (a deleted sheet lingering as grey points is confusing)."""
    brush_by = {}
    gl_by, hex_by = {}, {}
    for cid, col in colours.items():
        ci = int(cid)
        r, g, b = float(col[0]), float(col[1]), float(col[2])
        ir, ig, ib = int(r * 255), int(g * 255), int(b * 255)
        brush_by[ci] = pg.mkBrush(ir, ig, ib, 220)
        gl_by[ci] = (r, g, b, 1.0)
        hex_by[ci] = "#%02x%02x%02x" % (ir, ig, ib)
    return brush_by, gl_by, hex_by


class InteractiveWindow(QMainWindow):
    def __init__(self, session):
        super().__init__()
        self._session = session                                  # grind session: run_for(request) + save(...)
        self.setWindowTitle("HercuLabels")
        self.resize(1280, 980)

        self._panes = {"z": SlicePane("z"), "y": SlicePane("y"), "x": SlicePane("x")}
        self._mesh = MeshPane()
        grid = QGridLayout()
        grid.setContentsMargins(4, 4, 4, 4)
        grid.addWidget(self._panes["z"], 0, 0)
        grid.addWidget(self._panes["y"], 0, 1)
        grid.addWidget(self._panes["x"], 1, 0)
        grid.addWidget(self._mesh, 1, 1)
        for i in (0, 1):
            grid.setRowStretch(i, 1)
            grid.setColumnStretch(i, 1)
        grid_holder = QWidget()
        grid_holder.setLayout(grid)

        self._logs = QPlainTextEdit()
        self._logs.setReadOnly(True)
        self._logs.setMinimumHeight(60)                          # grows/shrinks with the resizable bottom section
        self._logs.setStyleSheet("font-family: monospace; font-size: 11px;")

        controls = QWidget()
        controls.setStyleSheet("background:#1e2127;")
        ch = QHBoxLayout(controls)
        ch.setContentsMargins(10, 5, 10, 5)
        ch.setSpacing(22)
        self._overlay_mode = SegmentedToggle(["streamlets", "none", "confidence", "samples"], current=0)
        self._t_planes = LabelledToggle("3D section planes", checked=False)
        ch.addWidget(self._overlay_mode)
        ch.addWidget(self._t_planes)
        step_lbl = QLabel("scroll step")
        step_lbl.setStyleSheet("color:#c9ccd2; font-size:12px;")
        self._step = QComboBox()
        self._step.addItems(["1", "2", "3", "5", "10", "20"])
        self._step.setStyleSheet(_COMBO_QSS)
        self._step.setFixedHeight(26)
        ch.addWidget(step_lbl)
        ch.addWidget(self._step)
        ch.addStretch(1)
        self._counter = QLabel("")                               # "window k" / "window k / N"
        self._counter.setStyleSheet("color:#c9ccd2; font-size:12px; font-weight:600;")
        self._btn_next = QPushButton("save + next →")            # grind to the next window (saves this one)
        self._btn_next.setStyleSheet(_BTN_QSS)
        self._btn_next.setEnabled(False)
        self._btn_next.setMinimumHeight(26)
        self._btn_next.clicked.connect(self._on_next)
        ch.addWidget(self._counter)
        ch.addWidget(self._btn_next)

        central = QWidget()
        v = QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self._swatch = SheetSwatch()                             # big hovered/selected-sheet colour + label
        self._sel_hint = QLabel("click a sheet to select")       # selection instruction, above the swatch box
        self._sel_hint.setAlignment(Qt.AlignCenter)
        self._sel_hint.setStyleSheet("color:#20242b; font-size:11px; font-weight:600;")
        self._btn_undo = QPushButton("undo")                     # sits under the sheet-id box
        self._btn_undo.setStyleSheet(_BTN_QSS)
        self._btn_undo.setEnabled(False)
        self._btn_undo.setFixedWidth(168)                        # match the swatch width
        self._btn_undo.setMinimumHeight(26)
        self._btn_undo.clicked.connect(self._undo)
        swatch_col = QWidget()
        scol = QVBoxLayout(swatch_col)
        scol.setContentsMargins(0, 0, 0, 0)
        scol.setSpacing(3)
        scol.addStretch(1)
        scol.addWidget(self._sel_hint, 0)
        scol.addWidget(self._swatch, 0, Qt.AlignHCenter)
        scol.addWidget(self._btn_undo, 0, Qt.AlignHCenter)
        scol.addStretch(1)

        # sheet actions — a vertical column of buttons on the far right, filling the logs-pane height.
        # Disabled for now (selection is wired; the split/merge/delete operations come next).
        self._btn_split = QPushButton("split")
        self._btn_merge = QPushButton("merge")
        self._btn_delete = QPushButton("delete")
        self._btn_split.setStyleSheet(_BTN_QSS)
        self._btn_merge.setStyleSheet(_BTN_QSS)
        self._btn_delete.setStyleSheet(_DEL_QSS)
        for b in (self._btn_split, self._btn_merge, self._btn_delete):
            b.setEnabled(False)
            b.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
            b.setMinimumHeight(28)
        btn_col = QWidget()
        bc = QVBoxLayout(btn_col)
        bc.setContentsMargins(0, 0, 0, 0)
        bc.setSpacing(6)
        bc.addWidget(self._btn_split)
        bc.addWidget(self._btn_merge)
        bc.addWidget(self._btn_delete)
        btn_col.setFixedWidth(100)

        bottom = QWidget()
        bottom.setMinimumHeight(112)                             # fits the three stacked action buttons
        bh = QHBoxLayout(bottom)
        bh.setContentsMargins(6, 4, 8, 6)
        bh.setSpacing(10)
        bh.addWidget(self._logs, 1)
        bh.addWidget(swatch_col, 0)
        bh.addWidget(btn_col, 0)

        # the grid (2×2 panes) over the bottom section, split so the bottom can be dragged taller/shorter.
        split = QSplitter(Qt.Vertical)
        split.addWidget(grid_holder)
        split.addWidget(bottom)
        split.setStretchFactor(0, 1)                             # extra space goes to the panes
        split.setStretchFactor(1, 0)
        split.setCollapsible(1, False)                           # bottom can't be dragged away entirely
        split.setSizes([820, 150])

        v.addWidget(controls)
        v.addWidget(split, 1)
        self.setCentralWidget(central)
        self._central = central

        # ---- persistent state + wiring (built once; per-window state is (re)set in _load_window) ----
        self._undo_stack = deque(maxlen=5)                       # last ≤5 pre-edit states, for undo
        self._req_index = 0                                      # index into a coords list (list mode)
        self._window_no = 1                                      # running window number (open-ended mode)
        self._grind_done = False                                 # coords list exhausted
        self._next_request = None                                # window to load once the save completes
        self._btn_split.clicked.connect(self._on_split)
        self._btn_merge.clicked.connect(self._on_merge)
        self._btn_delete.clicked.connect(self._on_delete)
        for keys, slot in (("S", self._on_split), ("M", self._on_merge), ("D", self._on_delete),
                           ("Ctrl+Z", self._undo), ("N", self._on_next), ("Right", self._on_next)):
            sc = QShortcut(QKeySequence(keys), self, activated=slot)  # s / m / d / ctrl+z / n / →
            sc.setContext(Qt.ApplicationShortcut)                # fire regardless of which pane has focus

        self._coords_lbl = QLabel()                              # global voxel coords read-out
        self.statusBar().addWidget(self._coords_lbl)
        for p in self._panes.values():
            p.hovered.connect(self._on_hover)
            p.sliceChanged.connect(self._mesh.set_plane)         # scroll a pane → move its 3D section plane
            p.picked.connect(self._on_pick)                      # click a sheet → (de)select
            p.sample_hover.connect(self._on_sample_hover)        # samples mode: highlight the point's pos/neg
        self._mesh.hovered.connect(self._on_hover)
        self._mesh.picked.connect(self._on_pick)
        self._overlay_mode.changed.connect(self._on_overlay_mode)
        self._t_planes.toggled.connect(self._mesh.set_planes_visible)
        self._step.currentTextChanged.connect(
            lambda s: [p.set_scroll_step(int(s)) for p in self._panes.values()])

        self._overlay = LoadingOverlay(central)

        if session.coords_list:                                  # list mode → first entry
            first = session.coords_list[0]
        else:                                                    # open-ended → an optional --coords start, else random
            first = getattr(session, "first", None)
        self._load_window(first)                                 # grind the first window

    # ---- per-window lifecycle ----
    def _reset_state(self) -> None:
        self._selection = []
        self._hex_by = {}
        self._gl_colours = {}
        self._meshes = {}                                        # {cid: mesh} (updated by edits)
        self._colours = {}                                       # persistent {cid: (r,g,b)} across edits
        self._conf_region = None                                 # accumulated delete regions
        self._conf_base = None                                   # pipeline low-conf (intersection·sharp2·dropped)
        self._edit_ctx = None
        self._busy = False
        self._pipeline_done = False
        self._edited = False                                     # any edit made → save on advance
        self._current_coords = None                              # (scroll,z,y,x) of the loaded window
        self._split_cid = None
        self._pos_nbr = None                                     # per-point positive sample table (live only)
        self._neg_nbr = None                                     # per-point negative sample table (live only)
        self._sample_pts = None                                  # full point cloud for the samples overlay
        self._undo_stack.clear()
        self._pending_undo = None

    def _load_window(self, request) -> None:
        """Reset all per-window state/UI and run the pipeline for ``request`` (None = random) off-thread."""
        self._reset_state()
        for p in self._panes.values():
            p.set_points(None)
            p.set_confidence(None)
            p.set_selection(None)
            p.set_sample_cloud(None)
            p.set_sample_highlight(-1, [], [])
        self._mesh.reset()
        self._swatch.clear()
        self._sel_hint.setText("click a sheet to select")
        self._coords_lbl.setText("")
        himat = getattr(self._session, "himat", None)            # searching for a high-material window?
        if request is None and himat is not None:
            self._overlay.show_over(f"searching for a window with {int(round(himat * 100))}% material fill…")
        else:
            self._overlay.show_over("reading window from the scroll…")
        self._update_counter()
        self._update_action_buttons()
        self._worker = PipelineWorker(self._session.run_for(request))
        s = self._worker.signals
        s.status.connect(self._log)
        s.brick.connect(self._on_brick)
        s.streamlets.connect(self._on_streamlets)
        s.clusters.connect(self._on_clusters)
        s.editctx.connect(self._on_editctx)
        s.confidence.connect(self._on_confidence)
        s.samples.connect(self._on_samples)
        s.meshes.connect(self._on_meshes)
        s.finished.connect(self._on_finished)
        s.failed.connect(self._on_failed)
        QThreadPool.globalInstance().start(self._worker)

    def _update_counter(self) -> None:
        cl = self._session.coords_list
        self._counter.setText(f"window {self._req_index + 1} / {len(cl)}" if cl
                              else f"window {self._window_no} / ?")   # open-ended: total unknown

    # ---- overlay tracks size ----
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._overlay.cover()

    # ---- slots (GUI thread) ----
    def _log(self, line: str) -> None:
        self._logs.appendPlainText(line)
        self._logs.verticalScrollBar().setValue(self._logs.verticalScrollBar().maximum())

    def _on_hover(self, coords: str, sheet_id: int, colour: str) -> None:
        if coords:                                               # 3D hover sends "" — don't blank the coords
            self._coords_lbl.setText(coords)
        if self._selection:                                      # an active selection pins the swatch
            return
        if sheet_id == -1:
            self._swatch.set_noise()
        elif sheet_id >= 0:
            self._swatch.set_sheet(sheet_id, colour)
        else:
            self._swatch.clear()

    def _on_pick(self, sheet_id: int, ctrl: bool) -> None:
        if sheet_id < 0:                                         # air gap / noise → clear the selection
            self._selection = []
        elif ctrl and len(self._selection) == 1 and self._selection[0] != sheet_id:
            self._selection = [self._selection[0], sheet_id]     # ctrl+click a second sheet → two selected
        else:
            self._selection = [sheet_id]                         # plain click → (re)select a single sheet
        self._apply_selection()

    def _apply_selection(self) -> None:
        sel = self._selection
        for p in self._panes.values():
            p.set_selection(sel)
        self._mesh.set_selection(sel)
        if len(sel) == 2:
            self._swatch.set_pair(sel[0], self._hex_by.get(sel[0], ""),
                                  sel[1], self._hex_by.get(sel[1], ""))
            self._sel_hint.setText("click a sheet to select")
        elif len(sel) == 1:
            self._swatch.set_sheet(sel[0], self._hex_by.get(sel[0], ""))
            self._sel_hint.setText("ctrl+click a second sheet")
        else:
            self._swatch.clear()                                 # nothing selected → hover drives the swatch again
            self._sel_hint.setText("click a sheet to select")
        self._update_action_buttons()

    def _update_action_buttons(self) -> None:
        """split needs exactly 1 selected sheet, merge exactly 2, delete ≥1 — with the base pipeline done
        and no edit running. undo needs a non-empty stack."""
        n = len(self._selection)
        ready = (self._edit_ctx is not None and self._pipeline_done and not self._busy)
        self._btn_split.setEnabled(ready and n == 1)
        self._btn_merge.setEnabled(ready and n == 2)
        self._btn_delete.setEnabled(ready and n >= 1)
        self._btn_undo.setEnabled(bool(self._undo_stack) and not self._busy)
        self._btn_next.setEnabled(self._pipeline_done and not self._busy and not self._grind_done)

    # ---- grind: save this window, advance to the next ----
    def _on_next(self) -> None:
        if self._busy or not self._pipeline_done or self._grind_done:
            return
        if not self._confirm_save_and_move():                   # "are you sure?" before saving + advancing
            return
        if self._session.coords_list is not None:               # list mode → next entry, no chooser
            self._advance(None)
        else:                                                    # open-ended → random-or-specify chooser
            self._next_menu()

    def _confirm_save_and_move(self) -> bool:
        from PySide6.QtWidgets import QMessageBox
        msg = ("Save this window's edits and move to the next window?" if self._edited
               else "Move to the next window? (no edits made — the base labels are already saved.)")
        return QMessageBox.question(self, "Save + next", msg,
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) == QMessageBox.Yes

    def _next_menu(self) -> None:
        # A modal popover (centred on the window, like the save confirmation) — NOT a drop-down — so the
        # confirm → choose flow reads as one sequence of dialogs. The advance runs after exec() returns,
        # i.e. once this dialog has fully closed (never from inside its modal loop).
        from PySide6.QtWidgets import QMessageBox
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Next window")
        box.setText("Grind to the next window:")
        rand = box.addButton("Random window", QMessageBox.AcceptRole)
        spec = box.addButton("Specify coords…", QMessageBox.ActionRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is rand:
            self._advance(None)
        elif clicked is spec:
            self._advance_specify()

    def _advance_specify(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        txt, ok = QInputDialog.getText(self, "Specify window", "Z,Y,X   (or  SCROLL,Z,Y,X):")
        if not ok or not txt.strip():
            return
        parts = txt.replace(",", " ").split()
        try:
            if len(parts) == 3:
                if not self._session.scroll:
                    self._log("specify coords needs a scroll — type SCROLL,Z,Y,X"); return
                req = (self._session.scroll, int(parts[0]), int(parts[1]), int(parts[2]))
            elif len(parts) == 4:
                req = (parts[0], int(parts[1]), int(parts[2]), int(parts[3]))
            else:
                self._log("coords must be 'Z,Y,X' or 'SCROLL,Z,Y,X'"); return
        except ValueError:
            self._log("coords must be integers"); return
        if req == self._current_coords:                          # same window you're already on → no-op recompute
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Same window",
                                    "Those are the current window's coordinates — choose a different window.")
            return
        self._advance(req)

    def _advance(self, request) -> None:
        """Save the current (edited) window off-thread, then load the next one."""
        self._next_request = request
        self._busy = True
        self._update_action_buttons()
        if self._edited and self._edit_ctx is not None:
            self._overlay.show_over("saving edited window…")
            ctx, meshes, region = self._edit_ctx, dict(self._meshes), self._conf_region
            self._save_worker = EditWorker(lambda progress: self._session.save(ctx, meshes, region))
            self._save_worker.signals.done.connect(self._on_saved)
            self._save_worker.signals.failed.connect(self._on_save_failed)
            QThreadPool.globalInstance().start(self._save_worker)
        else:
            self._on_saved(None)                                # unedited → generate's base .npz already saved

    def _on_saved(self, path) -> None:
        if path:
            self._log(f"saved → {path}")
        self._busy = False
        cl = self._session.coords_list
        if cl is not None:
            self._req_index += 1
            if self._req_index >= len(cl):
                self._grind_done = True
                self._overlay.hide()
                self._log("grind complete — all coords-file windows done. Close the viewer to exit.")
                self._update_action_buttons()
                return
        self._window_no += 1
        self._load_window(self._next_request if cl is None else cl[self._req_index])

    def _on_save_failed(self, message: str) -> None:
        self._busy = False
        self._overlay.hide()
        self._log(f"✗ save failed — {message} (staying on this window)")
        self._update_action_buttons()

    def _start_edit(self, fn, overlay_text: str, undo_label: str, done_slot) -> None:
        """Common launch for an interactive edit: snapshot the pre-edit state, freeze the controls behind
        the whole-window overlay, and run ``fn(progress)`` off-thread (its result goes to ``done_slot``)."""
        self._pending_undo = self._snapshot()
        self._pending_undo["label"] = undo_label
        self._busy = True
        self._update_action_buttons()
        self._overlay.show_over(overlay_text)
        self._edit_t0 = time.monotonic()
        self._edit_worker = EditWorker(fn)
        self._edit_worker.signals.status.connect(self._overlay.set_message)   # e.g. "updating sheet meshes…"
        self._edit_worker.signals.done.connect(done_slot)
        self._edit_worker.signals.failed.connect(self._on_edit_failed)
        QThreadPool.globalInstance().start(self._edit_worker)

    def _apply_labels(self, new_plab, *, remove_meshes, add_meshes, select) -> None:
        """Re-render the 2D panes + 3D meshes for an updated labelling, swap the affected meshes, and set
        the selection. ``self._colours`` must already be updated by the caller."""
        brush_by, gl_by, hex_by = _brush_maps(self._colours)
        self._gl_colours, self._hex_by = gl_by, hex_by
        pts = self._edit_ctx["pts"]
        for p in self._panes.values():
            p.set_points(pts, plab=new_plab, brush_by=brush_by)
            p.set_cluster_colours(hex_by)
        rm = {int(c) for c in remove_meshes}
        self._meshes = {k: v for k, v in self._meshes.items() if k not in rm}
        self._meshes.update(add_meshes)
        self._mesh.set_meshes(self._meshes, gl_by, recenter=False)
        self._refresh_embedding()                                # relabel/recolour the embeddings cloud too
        self._selection = list(select)
        self._apply_selection()

    def _edit_finish(self) -> None:
        """Reveal after an edit: hold the overlay a readable minimum (a fast edit like delete would else
        flash), then unfreeze the controls."""
        remaining = max(0, int(400 - (time.monotonic() - self._edit_t0) * 1000))
        QTimer.singleShot(remaining, self._edit_reveal)

    def _edit_reveal(self) -> None:
        self._busy = False
        self._overlay.hide()
        self._update_action_buttons()

    def _on_edit_failed(self, message: str) -> None:
        self._pending_undo = None                                # ctx unchanged on failure → discard snapshot
        self._log(f"✗ edit failed — {message}")
        self._edit_finish()

    def _snapshot(self) -> dict:
        """Restorable pre-edit state: the labelling (plab/ulab), the mesh set, and the sheet colours."""
        ctx = self._edit_ctx
        return {"plab": np.asarray(ctx["plab"]).copy(),
                "ulab": np.asarray(ctx["ulab"]).copy(),
                "meshes": dict(self._meshes),
                "colours": dict(self._colours),
                "conf_region": self._conf_region}                # accumulation makes new arrays → ref is safe

    def _undo(self) -> None:
        if self._busy or not self._undo_stack:
            return
        self._busy = True
        self._update_action_buttons()
        label = self._undo_stack[-1].get("label", "the last edit")   # what we're about to undo
        self._overlay.show_over(f"undoing {label}…")             # whole-window overlay (blocks interaction)
        self._undo_t0 = time.monotonic()
        QTimer.singleShot(0, self._undo_apply)                   # let the overlay paint, then restore

    def _undo_apply(self) -> None:
        snap = self._undo_stack.pop()
        ctx = self._edit_ctx
        ctx["plab"] = snap["plab"].copy()                        # copy so a later edit won't alias the stack
        ctx["ulab"] = snap["ulab"].copy()
        self._meshes = dict(snap["meshes"])
        self._colours = dict(snap["colours"])
        brush_by, gl_by, hex_by = _brush_maps(self._colours)
        self._gl_colours, self._hex_by = gl_by, hex_by
        plab, pts = ctx["plab"], ctx["pts"]
        for p in self._panes.values():
            p.set_points(pts, plab=plab, brush_by=brush_by)
            p.set_cluster_colours(hex_by)
        self._mesh.set_meshes(self._meshes, gl_by, recenter=False)
        self._refresh_embedding()                                # restore the embeddings cloud's colouring
        self._conf_region = snap.get("conf_region")              # restore the delete regions too
        self._update_confidence()
        self._selection = []
        self._apply_selection()                                  # re-render under the overlay (buttons still frozen)
        remaining = max(0, int(450 - (time.monotonic() - self._undo_t0) * 1000))  # keep overlay readable
        QTimer.singleShot(remaining, self._undo_finish)

    def _undo_finish(self) -> None:
        self._busy = False
        self._overlay.hide()
        self._update_action_buttons()
        self._log(f"undo — {len(self._undo_stack)} edit(s) left to undo")

    def _on_split(self) -> None:
        if self._busy or not self._pipeline_done or len(self._selection) != 1 or self._edit_ctx is None:
            return
        cid = int(self._selection[0])
        self._split_cid = cid
        ctx = self._edit_ctx
        from hercunet.labels.edit import split_sheet
        self._start_edit(lambda progress: split_sheet(ctx, cid, progress),
                         f"splitting sheet {cid}…", f"split of sheet {cid}", self._on_split_done)

    def _on_split_done(self, result) -> None:
        cid = self._split_cid
        if result is None:                                       # forced_split found no seam to cut
            self._pending_undo = None
            self._log(f"split: sheet {cid} could not be split (no seam found)")
            self._edit_finish()
            return
        self._undo_stack.append(self._pending_undo)              # commit the pre-split state
        self._pending_undo = None
        self._edited = True
        new_plab, new_meshes, (ia, ib) = result
        # id_a KEEPS the split sheet's id + colour; id_b is the ONE new sheet (fresh colour, auto-selected).
        kept = self._colours.pop(cid, None)
        self._colours[ia] = kept if kept is not None else _pick_new_colour(self._colours)
        self._colours[ib] = _pick_new_colour(self._colours)
        self._apply_labels(new_plab, remove_meshes=[cid], add_meshes=new_meshes, select=[ib])
        self._log(f"split sheet {cid} → kept sheet {ia} + new sheet {ib} (selected)")
        self._edit_finish()

    def _on_merge(self) -> None:
        if self._busy or not self._pipeline_done or len(self._selection) != 2 or self._edit_ctx is None:
            return
        a, b = int(self._selection[0]), int(self._selection[1])
        ctx = self._edit_ctx
        from hercunet.labels.edit import merge_sheets
        self._start_edit(lambda progress: merge_sheets(ctx, a, b, progress),
                         f"merging sheets {a} + {b}…", f"merge of sheets {a} + {b}", self._on_merge_done)

    def _on_merge_done(self, result) -> None:
        self._undo_stack.append(self._pending_undo)
        self._pending_undo = None
        self._edited = True
        new_plab, new_meshes, keep, drop = result
        self._colours.pop(drop, None)                            # keep the LARGER sheet's id + colour
        self._apply_labels(new_plab, remove_meshes=[keep, drop], add_meshes=new_meshes, select=[keep])
        self._log(f"merged sheets {keep} + {drop} → sheet {keep}")
        self._edit_finish()

    def _on_delete(self) -> None:
        if self._busy or not self._pipeline_done or not self._selection or self._edit_ctx is None:
            return
        ids = [int(c) for c in self._selection]
        ctx = self._edit_ctx
        from hercunet.labels.edit import delete_sheets
        lab = ", ".join(str(i) for i in ids)
        self._start_edit(lambda progress: delete_sheets(ctx, ids, progress),
                         f"deleting sheet(s) {lab}…", f"delete of sheet(s) {lab}", self._on_delete_done)

    def _on_delete_done(self, result) -> None:
        self._undo_stack.append(self._pending_undo)
        self._pending_undo = None
        self._edited = True
        new_plab, deleted, region = result
        for c in deleted:
            self._colours.pop(c, None)
        self._conf_region = region if self._conf_region is None else (self._conf_region | region)
        self._update_confidence()                                # grows the low-conf overlay (visible in conf mode)
        self._apply_labels(new_plab, remove_meshes=deleted, add_meshes={}, select=[])
        self._log(f"deleted sheet(s) {', '.join(str(c) for c in deleted)} → noise (low-confidence region)")
        self._edit_finish()

    def _on_overlay_mode(self, idx: int) -> None:
        """4-way overlay selector: 0 = streamlets, 1 = none, 2 = confidence, 3 = samples (hover a point to see
        its positives/negatives). 'samples' is populated only while generating a window (live create)."""
        for p in self._panes.values():
            p.set_streamlets_visible(idx == 0)
            p.set_confidence_visible(idx == 2)
            p.set_samples_visible(idx == 3)
        if idx == 3 and self._pos_nbr is None:
            self._log("samples overlay: pos/neg sampling is only shown while generating (create --interactive), "
                      "not for stored windows")

    def _on_brick(self, payload) -> None:
        bced, corner, vu = payload["bced"], payload["corner"], payload["voxel_um"]
        scroll, coords = payload.get("scroll"), payload.get("coords")
        if scroll and coords is not None:                        # resolved window (incl. a randomly-chosen one)
            cz, cy, cx = coords
            self._current_coords = (scroll, int(cz), int(cy), int(cx))   # to reject re-selecting this window
            self.setWindowTitle(f"HercuLabels · {scroll} z{cz} y{cy} x{cx}")
        self._voxel_um = float(vu)
        for p in self._panes.values():
            p.set_volume(bced, corner, vu)
        self._mesh.set_extent(bced.shape)                        # size the 3D section planes to the block
        self._overlay.hide()                                     # 2D ready; 3D keeps its own overlay
        self._mesh.set_status("computing streamlets + clusters + sheets…")

    def _on_streamlets(self, payload) -> None:
        pts = payload["pts"]
        for p in self._panes.values():
            p.set_points(pts, uniform_brush=_GREY)               # one colour until clustered (fast)
        self._mesh.set_status("computing clusters + sheets…")    # streamlets done → drop them from the message

    def _on_clusters(self, payload) -> None:
        pts, plab = payload["pts"], payload["plab"]
        self._sample_pts = np.asarray(pts, np.float32)           # FULL cloud (incl noise) — the samples tables index into it
        self._colours = _initial_colours(plab, pts, getattr(self, "_voxel_um", None))  # persistent from here
        brush_by, gl_by, hex_by = _brush_maps(self._colours)
        self._gl_colours = gl_by
        self._hex_by = hex_by
        self._selection = []                                     # fresh cluster ids → drop any stale selection
        for p in self._panes.values():
            p.set_points(pts, plab=plab, brush_by=brush_by)
            p.set_cluster_colours(hex_by)
            p.set_sample_cloud(self._sample_pts)
        n = len({int(x) for x in plab if x >= 0})
        self._mesh.set_status(f"computing {n} sheets…")          # clusters done → only sheet meshes left

    def _on_meshes(self, payload) -> None:
        self._meshes = dict(payload["meshes"])
        self._mesh.set_meshes(self._meshes, getattr(self, "_gl_colours", {}))
        self._pipeline_done = True                               # sheets are ready → unlock editing now
        self._update_action_buttons()

    def _on_editctx(self, ctx) -> None:
        self._edit_ctx = ctx                                     # enables the sheet-edit actions
        self._refresh_embedding()                                # feed the 3-D pane's embeddings view
        self._update_action_buttons()

    def _refresh_embedding(self) -> None:
        ctx = self._edit_ctx
        if ctx is not None and "E" in ctx:
            self._mesh.set_embedding(ctx["E"], ctx["ulab"], self._gl_colours)

    def _on_samples(self, payload) -> None:
        """Live per-point sample tables (positives = mutual slab, negatives = gap-gated), for the 'samples'
        overlay. Only emitted while generating a window; stored windows opened via ``edit`` have none."""
        self._pos_nbr = np.asarray(payload["pos_nbr"])
        self._neg_nbr = np.asarray(payload["neg_nbr"])
        self._log("samples ready — pick 'samples' and hover a point to see its positives (green) / "
                  "negatives (red)")

    def _on_sample_hover(self, idx: int) -> None:
        """Hover in samples mode: highlight the point's positives/negatives across all panes (or clear)."""
        if self._pos_nbr is None or idx < 0 or idx >= len(self._pos_nbr):
            for p in self._panes.values():
                p.set_sample_highlight(-1, [], [])
            return
        pos = self._pos_nbr[idx]; pos = pos[pos >= 0]
        neg = self._neg_nbr[idx]; neg = neg[neg >= 0]
        for p in self._panes.values():
            p.set_sample_highlight(idx, pos, neg)

    def _on_confidence(self, payload) -> None:
        """Pipeline low-confidence: combine the regional channels (intersection ⊕ sharp2 ⊕ dropped) into one
        UNCERTAINTY field = 1 − Π(confᵢ) (soft-OR of doubts, the pipeline's combine_confidence rule). ~0 in
        air/certain regions, high at intersections/ambiguous seams/dropped structure."""
        i = np.clip(np.asarray(payload["intersection"], np.float32), 0.0, 1.0)
        s = np.clip(np.asarray(payload["sharp2"], np.float32), 0.0, 1.0)
        d = np.clip(np.asarray(payload["dropped"], np.float32), 0.0, 1.0)
        self._conf_base = np.clip(1.0 - i * s * d, 0.0, 1.0)
        self._update_confidence()

    def _update_confidence(self) -> None:
        """Push the combined low-confidence field to the panes: the pipeline uncertainty ORed with the
        interactive delete regions (both are uncertainties, so element-wise max)."""
        base, reg = self._conf_base, self._conf_region
        if base is None and reg is None:
            field = None
        elif reg is None:
            field = base
        elif base is None:
            field = reg.astype(np.float32)
        else:
            field = np.maximum(base, reg.astype(np.float32))
        for p in self._panes.values():
            p.set_confidence(field)

    def _on_finished(self, result) -> None:
        meta = getattr(result, "meta", None)
        if meta is not None:
            self._log(f"✓ window ready — {meta.get('n_sheets', '?')} sheets (saved to corpus)")
        else:
            self._log("✓ done — no fittable sheet in this window")
        self._pipeline_done = True                               # base pipeline complete → edits are safe now
        self._update_action_buttons()

    def _on_failed(self, message: str) -> None:
        self._overlay.hide()
        self._mesh.set_status(f"failed: {message}")
        self._log(f"✗ FAILED — {message}  (use save + next to skip to another window)")
        self._busy = False
        self._pipeline_done = True                               # let 'next' skip a broken window (edits stay off)
        self._update_action_buttons()

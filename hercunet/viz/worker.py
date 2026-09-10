"""Off-thread pipeline runner for the interactive viewer.

The GUI must never run the (minutes-long, GPU) label pipeline on the event loop. This QRunnable runs
the SAME ``run(progress)`` closure the CLI's ``create()`` builds — it calls the real
``build_brick``/``generate``/``extract_window``; nothing about label generation is reimplemented here.
Its ``progress`` hook and captured stdout are turned into Qt signals delivered to the GUI thread
(the only safe way to touch widgets from another thread).
"""

from __future__ import annotations

import io
import sys
import traceback

from PySide6.QtCore import QObject, QRunnable, Signal


class _Signals(QObject):
    status = Signal(str)        # a log / status line
    brick = Signal(object)      # {bced, corner, voxel_um, scroll, coords}
    streamlets = Signal(object)  # {pts, kind}          (pre-cluster point cloud)
    clusters = Signal(object)   # {pts, plab, corner}  (labelled point cloud)
    editctx = Signal(object)    # interactive-edit context (labels arrays, by reference in-process)
    confidence = Signal(object)  # {intersection, sharp2, dropped} regional confidence volumes
    samples = Signal(object)    # {pos_nbr, neg_nbr}   per-point positive/negative sample tables (live only)
    meshes = Signal(object)     # {meshes}             (fitted sheet meshes)
    finished = Signal(object)   # result of run() (e.g. .npz path)
    failed = Signal(str)


class _LineEmitter(io.TextIOBase):
    """Tee stdout: forward to the real stream and emit each completed line as a status signal."""

    def __init__(self, real, emit):
        self._real = real
        self._emit = emit
        self._buf = ""

    def write(self, s):
        try:
            self._real.write(s)
        except Exception:
            pass
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                try:
                    self._emit(line.rstrip())
                except RuntimeError:
                    pass                                         # signals QObject gone (window closing)
        return len(s)

    def flush(self):
        try:
            self._real.flush()
        except Exception:
            pass


class PipelineWorker(QRunnable):
    """Runs ``run_fn(progress)`` in a pool thread. ``run_fn`` is the closure ``create()`` builds for
    the interactive path (resolve window → build_brick → generate), instrumented with the progress
    hook so we observe each stage without changing the computation."""

    def __init__(self, run_fn):
        super().__init__()
        self.setAutoDelete(False)                                # keep alive until signals delivered
        self._run_fn = run_fn
        self.signals = _Signals()

    def _progress(self, stage: str, payload) -> None:
        sig = getattr(self.signals, stage, None)
        if stage == "status":
            self.signals.status.emit(str(payload))
        elif sig is not None:
            sig.emit(payload)

    def run(self) -> None:
        old = sys.stdout
        sys.stdout = _LineEmitter(old, self.signals.status.emit)
        try:
            result = self._run_fn(self._progress)
            self._safe_emit(self.signals.finished, result)
        except Exception as exc:
            self._safe_emit(self.signals.status, traceback.format_exc())
            self._safe_emit(self.signals.failed, f"{type(exc).__name__}: {exc}")
        finally:
            sys.stdout = old

    @staticmethod
    def _safe_emit(sig, payload) -> None:
        try:
            sig.emit(payload)
        except RuntimeError:
            pass                                                 # window closed mid-run — nothing to update


class _EditSignals(QObject):
    status = Signal(str)        # overlay text updates while the edit runs ("updating sheet meshes…")
    done = Signal(object)       # the edit's return value (e.g. (new_plab, meshes, (ia, ib)) or None)
    failed = Signal(str)


class EditWorker(QRunnable):
    """Runs one interactive label edit — ``fn(progress)`` calling a ``hercunet.labels.edit`` function —
    off the GUI thread (a split does a GPU/CPU mesh refit). Its ``progress`` hook forwards ``split_status``
    payloads to :attr:`signals.status` so the loading overlay can update mid-edit."""

    def __init__(self, fn):
        super().__init__()
        self.setAutoDelete(False)
        self._fn = fn
        self.signals = _EditSignals()

    def _progress(self, stage: str, payload) -> None:
        if stage == "split_status":
            self._safe(self.signals.status, str(payload))

    def run(self) -> None:
        try:
            self._safe(self.signals.done, self._fn(self._progress))
        except Exception as exc:
            self._safe(self.signals.status, traceback.format_exc())
            self._safe(self.signals.failed, f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _safe(sig, payload) -> None:
        try:
            sig.emit(payload)
        except RuntimeError:
            pass

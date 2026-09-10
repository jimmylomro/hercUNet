"""Entry point for the interactive Qt viewer (``hercunet labels create --interactive (or edit)``).

Qt is imported lazily HERE so the rest of the package (and the CLI's core paths) never need it.
``launch_interactive`` blocks on the Qt event loop until the window is closed.
"""

from __future__ import annotations


def _require_qt():
    import importlib.util
    missing = [m for m in ("PySide6", "pyqtgraph", "OpenGL") if importlib.util.find_spec(m) is None]
    if missing:
        raise SystemExit(
            "hercunet labels create --interactive (or edit) needs the viewer extra. "
            'Install it with:  pip install -e ".[viz]"   (PySide6 + pyqtgraph + PyOpenGL). '
            f"(missing: {', '.join(missing)})"
        )


def launch_interactive(session):
    """Open the viewer and grind the ``session``'s windows one at a time — each runs the SAME
    build_brick→generate pipeline off-thread, streaming its stages into the UI. Blocks until closed."""
    _require_qt()
    import os
    import signal
    import sys

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from .window import InteractiveWindow

    app = QApplication.instance()
    owns = app is None
    if owns:
        app = QApplication(sys.argv[:1])
        # Clean Ctrl-C: Qt's C++ event loop otherwise swallows SIGINT. (1) Handle it with a hard exit —
        # the pool worker may be deep in a long GPU op that can't be joined cleanly (same reason
        # hercureader hard-exits). (2) A periodic no-op timer hands control back to Python often enough
        # for the signal to be delivered.
        signal.signal(signal.SIGINT, lambda *_: os._exit(130))
        _tick = QTimer()
        _tick.start(150)
        _tick.timeout.connect(lambda: None)
    win = InteractiveWindow(session)
    win.show()
    if owns:
        app.exec()
        _cleanup_pools()                                         # release loky/joblib pools before the hard exit
        os._exit(0)                                              # skip Qt/thread-pool teardown (can hang/segfault)
    return win


def _cleanup_pools():
    """os._exit skips atexit, so sklearn's loky process pool (HDBSCAN/kNN) would leak its semaphores and
    its /dev/shm joblib memmap folder — loky's resource_tracker then prints 'leaked semlock/folder
    objects' at shutdown. Shut the pool down (kills workers → releases semaphores), then UNREGISTER any
    memmap folder this process created so the tracker unlinks it cleanly and reports nothing leaked."""
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True, kill_workers=True)
    except Exception:
        pass
    try:
        import glob
        import os
        import tempfile

        from joblib.externals.loky.backend import resource_tracker as rt
        pid = os.getpid()                                        # joblib names the folder after its creator's pid
        seen = set()
        for base in ("/dev/shm", tempfile.gettempdir()):
            for folder in glob.glob(os.path.join(base, f"joblib_memmapping_folder_{pid}_*")):
                if folder in seen:
                    continue
                seen.add(folder)
                try:
                    rt.unregister(folder, "folder")             # tracker drops it from its cache + rmtrees it
                except Exception:
                    pass
    except Exception:
        pass

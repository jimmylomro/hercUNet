"""Interactive Qt viewer for stage-1 label generation.

``hercunet labels create --interactive (or edit)`` opens a window (three orthogonal cross-sections +
a 3D sheet view + a log pane) that streams the SAME pipeline the CLI runs — the viewer only observes
via a progress hook; it never reimplements label generation. Qt (PySide6 + pyqtgraph) is an optional
extra (``.[viz]`` / ``.[full]``) and is imported lazily by :mod:`hercunet.viz.interactive`, so
importing this package does not require it.
"""

__all__ = ["launch_interactive"]


def launch_interactive(run_fn, meta):
    from .interactive import launch_interactive as _launch
    return _launch(run_fn, meta)

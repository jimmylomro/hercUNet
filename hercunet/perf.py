"""Opt-in performance timing for the inference hot paths — **off by default, zero overhead**.

Set ``HERCUNET_PERF=1`` to turn it on: phase timings accumulate and are emitted to the ``hercunet.perf`` logger at
DEBUG level (configure logging to see them). When off, :func:`now`/:func:`add`/:func:`report` are no-ops and
:func:`sync` never calls ``torch.cuda.synchronize`` — important, because the per-phase syncs needed for accurate GPU
attribution would otherwise serialise and slow the real run.

This is the reusable form of the ad-hoc timers that located the mesh-fit bottleneck (the per-sheet cKDTree metric +
grid init running on the CPU beside an idle GPU). Typical use in a hot function::

    from ..perf import ENABLED, now, add, report, sync
    acc = globals().setdefault("_PERF_ACC", {})
    t = now(); ...work...; sync(); add(acc, "phase", t)
    report("fit", acc, n_sheets=N)        # throttle the call site if it is per-item
"""
from __future__ import annotations

import logging
import os
import time

ENABLED = bool(os.environ.get("HERCUNET_PERF"))
_log = logging.getLogger("hercunet.perf")


def now() -> float:
    """``perf_counter()`` when timing is on, else ``0.0`` (so ``add`` below is a no-op)."""
    return time.perf_counter() if ENABLED else 0.0


def sync() -> None:
    """CUDA sync for accurate GPU-phase attribution — ONLY when timing is on (and CUDA is present). A no-op otherwise,
    so the production path is never serialised by synchronisation."""
    if not ENABLED:
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def add(acc: dict, key: str, t0: float) -> None:
    """Accumulate ``perf_counter() - t0`` into ``acc[key]`` (no-op when timing is off)."""
    if ENABLED:
        acc[key] = acc.get(key, 0.0) + (time.perf_counter() - t0)


def report(prefix: str, acc: dict, **extra) -> None:
    """Emit the accumulated ``acc`` (plus any ``extra`` scalars) to the ``hercunet.perf`` logger at DEBUG."""
    if not ENABLED:
        return
    msg = " ".join(f"{k}={v:.1f}s" for k, v in acc.items())
    if extra:
        msg += " " + " ".join(f"{k}={v}" for k, v in extra.items())
    _log.debug("%s %s", prefix, msg)

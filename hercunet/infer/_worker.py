"""Per-GPU inference worker (internal). Spawned by :func:`hercunet.infer.run._run`, one process per GPU, each with
``CUDA_VISIBLE_DEVICES`` already pinned to its device (so ``device="cuda"`` resolves to that single GPU). Reads
its jacobi_refine kwargs from the JSON file named on argv, tags every line it prints with its worker id (so the
shared log is readable when several GPUs interleave), and runs a single worker of the shared claim-queue pass.

Not a user-facing command — the ``hercunet infer single-instance`` / ``multi-instance`` orchestrator invokes it.
"""
from __future__ import annotations

import json
import sys


class _TagStream:
    """Wrap a text stream so every printed LINE is prefixed with ``tag`` (e.g. ``[w0·gpu0·L] ``). Tracks line
    starts across writes, so ``print``'s split ``write(msg)`` + ``write("\\n")`` calls tag correctly and a single
    worker's multi-GPU log lines are attributable."""

    def __init__(self, stream, tag):
        self._s = stream
        self._tag = tag
        self._at_start = True

    def write(self, data):
        if not data:
            return
        out = []
        for ch in data:
            if self._at_start:
                out.append(self._tag)
                self._at_start = False
            out.append(ch)
            if ch == "\n":
                self._at_start = True
        self._s.write("".join(out))

    def flush(self):
        self._s.flush()

    def isatty(self):
        return False


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit("hercunet.infer._worker: expected exactly one argument (the args JSON file path)")
    with open(argv[0]) as f:
        kwargs = json.load(f)
    # worker identity (added by run._run) — used only for the log tag, not passed to jacobi_refine
    rank = kwargs.pop("rank", 0)
    gpu = kwargs.pop("gpu", None)
    tag = f"[w{rank}" + (f"·gpu{gpu}" if gpu is not None else "") + ("·L" if kwargs.get("leader") else "") + "] "
    sys.stdout = _TagStream(sys.stdout, tag)
    sys.stderr = _TagStream(sys.stderr, tag)
    region = kwargs.get("region")
    if region is not None:
        kwargs["region"] = tuple(region)                              # JSON lists → the tuple jacobi_refine expects
    from .jacobi_refine import jacobi_refine
    jacobi_refine(**kwargs)


if __name__ == "__main__":
    main()

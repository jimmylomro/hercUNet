"""Per-GPU inference worker (internal). Spawned by :func:`hercunet.infer.run._run`, one process per GPU, each with
``CUDA_VISIBLE_DEVICES`` already pinned to its device (so ``device="cuda"`` resolves to that single GPU). Reads
its jacobi_refine kwargs from the JSON file named on argv, runs a single worker of the shared claim-queue pass.

Not a user-facing command — the ``hercunet infer single-instance`` / ``multi-instance`` orchestrator invokes it.
"""
from __future__ import annotations

import json
import sys


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit("hercunet.infer._worker: expected exactly one argument (the args JSON file path)")
    with open(argv[0]) as f:
        kwargs = json.load(f)
    region = kwargs.get("region")
    if region is not None:
        kwargs["region"] = tuple(region)                              # JSON lists → the tuple jacobi_refine expects
    from .jacobi_refine import jacobi_refine
    jacobi_refine(**kwargs)


if __name__ == "__main__":
    main()

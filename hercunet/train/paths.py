"""Resolve nnU-Net's path roots for the training chain from a ``--nnunet-path`` flag (no env export needed).

nnU-Net addresses its data through three roots — ``nnUNet_raw`` / ``nnUNet_preprocessed`` / ``nnUNet_results``
— and its path layer (``nnunetv2.paths``) reads them **lazily** from ``os.environ`` on each access. So setting
them in-process *before* any nnU-Net call is equivalent to exporting them in the shell, and lets
``hercunet train`` take a single ``--nnunet-path <dir>`` instead of requiring you to export anything.

Precedence: an explicit ``--nnunet-path`` wins (it sets all three under that dir using the standard subdir
names); otherwise an already-exported env var is used as-is; if a root a command needs is set by neither, we
fail fast with a clear message instead of letting nnU-Net raise deeper. Split layouts (roots in different
places) are still supported the normal way — just export the env vars and omit the flag.
"""
from __future__ import annotations

import os
import sys

ROOTS = ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")


def set_nnunet_roots(nnunet_path=None, *, needs=ROOTS):
    """Point nnU-Net at its roots for this process, up front (called at the start of the command, before any
    nnU-Net call — so a missing root fails HERE, not deep inside nnU-Net).

    ``nnunet_path`` (if given) is the parent dir holding the standard ``nnUNet_raw/ nnUNet_preprocessed/
    nnUNet_results/`` subdirs — it sets all three (flag wins over any existing env). If it is not given:

      * the pre-existing ``nnUNet_*`` env vars are used — with a **warning** that ``--nnunet-path`` is
        preferred (so you are told which mechanism is in effect); or
      * if a root the command needs is set by neither, a ``SystemExit`` is raised naming the options.

    ``needs`` is the subset of roots the calling command actually uses."""
    if nnunet_path:
        base = os.path.abspath(os.path.expanduser(nnunet_path))
        for name in ROOTS:
            os.environ[name] = os.path.join(base, name)
        return

    missing = [n for n in needs if not os.environ.get(n)]
    if missing:
        raise SystemExit(
            "hercunet train: no nnU-Net path roots. Set --nnunet-path <dir> (preferred — a directory "
            "containing nnUNet_raw/ nnUNet_preprocessed/ nnUNet_results/), or export the environment "
            "variables " + ", ".join(needs) + f".\n  Missing: {', '.join(missing)}.")
    have = ", ".join(f"{n}={os.environ[n]}" for n in needs)
    print(f"[hercunet train] warning: --nnunet-path not set — using the nnUNet_* environment variables "
          f"({have}). Prefer --nnunet-path <dir> so you don't have to manage env vars.",
          file=sys.stderr, flush=True)

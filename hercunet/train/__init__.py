"""Stage-2 training chain (HercUNet refiner) as ``hercunet train <step>`` subcommands.

Mirrors the pod chain (``scripts/v3_chain.sh`` in the research repo) as separate, resumable
steps — export-labels -> preprocess -> export-owner -> fit — over a **pinned,
unmodified** nnU-Net (``nnunetv2==2.8.1``). No fork, no clone, no copy-paste of trainers into
the nnunetv2 package (see ``docs/stage2-port.md``):

- ``preprocess`` wraps three stock nnU-Net library functions (no CLI shell-out, no edits).
- ``fit`` launches a trainer by **direct instantiation** of an imported class — it never uses
  nnU-Net's string-name discovery, so trainers live in ``hercunet`` with nothing copied.

Heavy deps (torch, nnunetv2) are imported lazily inside each step, so importing ``hercunet``
elsewhere (viewer, label gen) never pulls them in.
"""

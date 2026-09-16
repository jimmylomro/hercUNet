"""Stage-2 training logic for the HercUNet iterative surface refiner (run301 / v0 recipe).

The nnU-Net *trainer classes* live in :mod:`hercunet.training.trainers`; all of their loss / prev-
channel / affinity / MALIS LOGIC lives here as plain modules so the trainers stay thin. These modules
import heavy training deps (``torch``, ``nnunetv2``) at load, so import them only on a training host —
never from the data layer, the viewer, or label export.

Unlike the research repo, the trainers are **not** copied into the ``nnunetv2`` package: they are
imported directly by :func:`hercunet.train.launch.fit` (direct instantiation, no ``-tr`` discovery),
and import their siblings by ``hercunet.training.trainers.*`` paths. The loss/affinity/MALIS math is
carried over **bit-identically** from the research pipeline — the ``nnunetv2==2.8.1`` pin is
load-bearing (the trainers reproduce nnU-Net's ``train_step`` internals verbatim).
"""

"""Focal-Tversky + SEPARATION penalty (anti-merge) — THIN SHIM.

Loss logic lives in :mod:`hercunet.training.losses` (SeparationPenaltyLoss = mean P(surface) in
morphologically-closed inter-sheet gaps; "a broken sheet >> a merged one"). This file only registers the
trainer class and flips the shared builder's ``include_sep``.
"""
from hercunet.training.trainers.focal_tversky import nnUNetTrainer_FocalTversky_500epochs
from hercunet.training.losses import SeparationPenaltyLoss

__all__ = ["nnUNetTrainer_FocalTverskySep_500epochs", "SeparationPenaltyLoss"]


class nnUNetTrainer_FocalTverskySep_500epochs(nnUNetTrainer_FocalTversky_500epochs):
    # separation term — lambda scales the anti-merge push vs the base loss; radius = widest gap (vx) protected
    # (closing fills <=~2r). Conservative default so it nudges without undoing the Tversky beta>alpha recall bias.
    include_sep = True
    sep_lambda = 0.5
    sep_radius = 2
    # _build_loss inherited from the parent — the shared builder handles include_sep.

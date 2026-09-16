"""Focal-Tversky + separation + SKELETON-RECALL — THIN SHIM.

Loss logic lives in :mod:`hercunet.training.losses` (SkeletonRecallLoss = recall-only clDice soft-skel
coverage of the GT medial; raises confidence where our label's skeleton is, punishes nothing beyond it so
Dice/CE/Tversky keep precision). This file only registers the trainer class and flips ``include_skel``.
"""
from hercunet.training.trainers.focal_tversky_sep import nnUNetTrainer_FocalTverskySep_500epochs
from hercunet.training.losses import SkeletonRecallLoss, soft_skel

__all__ = ["nnUNetTrainer_FocalTverskySepSkel_500epochs", "SkeletonRecallLoss", "soft_skel"]


class nnUNetTrainer_FocalTverskySepSkel_500epochs(nnUNetTrainer_FocalTverskySep_500epochs):
    # skeleton-recall coverage push vs the base loss; iters = soft-skel depth (our medial is already thin → 3).
    include_skel = True
    srec_lambda = 1.0
    srec_iters = 3
    # inherits include_sep + sep_lambda/sep_radius + Tversky α/β/γ; _build_loss from the shared builder.

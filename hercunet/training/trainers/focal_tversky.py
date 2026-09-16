"""Custom nnU-Net trainer: Focal-Tversky (+CE) loss for the extreme thin-positive imbalance.

THIN SHIM — all loss logic lives in :mod:`hercunet.training.losses`; this file only registers the
trainer class. See the library module for the rationale (Tversky alpha<beta + focal gamma for the ~2%
thin crest in ~80% papyrus bg). Imported directly by ``hercunet train fit`` (no copy into nnunetv2).
"""
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_500epochs

from hercunet.training.losses import (
    build_surface_loss, deep_supervision_weights, MemoryEfficientSoftTverskyLoss)

__all__ = ["nnUNetTrainer_FocalTversky_500epochs", "MemoryEfficientSoftTverskyLoss"]


class nnUNetTrainer_FocalTversky_500epochs(nnUNetTrainer_500epochs):
    # Tversky/focal hyperparameters — tuned for thin-positive imbalance; override in subclasses if needed.
    tversky_alpha = 0.3   # FP weight
    tversky_beta = 0.7    # FN weight (>alpha => penalize misses harder)
    tversky_gamma = 1.33  # focal exponent (>1 => focus on the hard crest)
    # separation / skeleton knobs consumed by subclasses via the shared builder (no effect at this level)
    include_sep = False
    include_skel = False
    sep_lambda = 0.5
    sep_radius = 2
    srec_lambda = 1.0
    srec_iters = 3

    def _build_loss(self):
        assert not self.label_manager.has_regions, "FocalTversky trainer assumes non-region labels"
        dsw = None
        if self.enable_deep_supervision:
            dsw = deep_supervision_weights(self._get_deep_supervision_scales(),
                                           self.is_ddp, self._do_i_compile())
        return build_surface_loss(
            batch_dice=self.configuration_manager.batch_dice, is_ddp=self.is_ddp,
            ignore_label=self.label_manager.ignore_label,
            tversky_alpha=self.tversky_alpha, tversky_beta=self.tversky_beta, tversky_gamma=self.tversky_gamma,
            include_sep=self.include_sep, sep_lambda=self.sep_lambda, sep_radius=self.sep_radius,
            include_skel=self.include_skel, srec_lambda=self.srec_lambda, srec_iters=self.srec_iters,
            deep_supervision_weights=dsw, compile_dc=self._do_i_compile())

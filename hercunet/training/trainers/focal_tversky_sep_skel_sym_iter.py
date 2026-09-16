"""ITERATIVE refiner trainer — SepSkelSym loss (UNCHANGED) on the 2-channel [CT, prev] input. THIN SHIM.

Same loss as ablA (inherits nnUNetTrainer_FocalTverskySepSkelSym). Two additions, both wiring (logic lives in
:mod:`hercunet.training`):
  1. **Network forced to 2 input channels** — the dataset stores ``1 + max_candidates`` channels (CT + candidate
     ∇φ slots) so the per-epoch transform can compose ``prev``; the NETWORK, however, takes [CT, prev] = 2
     channels (matches the expanded m7 warm-start and inference). We override build_network_architecture to hard
     -set num_input_channels=2.
  2. **ComposePrevTransform appended** to train/val transforms — collapses [CT, cand...] -> [CT, prev] each
     epoch (fresh octant "+"-seam corruption for train; deterministic for val).
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from hercunet.training.trainers.focal_tversky_sep_skel_sym import (
    nnUNetTrainer_FocalTverskySepSkelSym_500epochs)
from hercunet.training.prev_transform import compose_prev_transform

__all__ = ["nnUNetTrainer_FocalTverskySepSkelSym_Iter_500epochs"]

# The network takes CT + composed prev = 2 channels, regardless of how many candidate channels the data carries.
_NET_IN_CHANNELS = 2


def _append_prev(transforms, deterministic):
    """Append ComposePrevTransform to a ComposeTransforms (has .transforms) or a plain list."""
    t = compose_prev_transform(deterministic=deterministic)
    if hasattr(transforms, "transforms"):
        transforms.transforms.append(t)
    else:
        transforms.append(t)
    return transforms


class nnUNetTrainer_FocalTverskySepSkelSym_Iter_500epochs(nnUNetTrainer_FocalTverskySepSkelSym_500epochs):

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        # force 2 input channels (CT + prev) — data has 1+K channels but the transform collapses to 2.
        # Signature matches nnunetv2 2.8.1 (plans_manager, configuration_manager, ...); verified on the pod.
        return nnUNetTrainer.build_network_architecture(
            plans_manager, configuration_manager, _NET_IN_CHANNELS, num_output_channels, enable_deep_supervision)

    @staticmethod
    def get_training_transforms(*args, **kwargs):
        tr = nnUNetTrainer.get_training_transforms(*args, **kwargs)
        return _append_prev(tr, deterministic=False)

    @staticmethod
    def get_validation_transforms(*args, **kwargs):
        val = nnUNetTrainer.get_validation_transforms(*args, **kwargs)
        return _append_prev(val, deterministic=True)

"""nnU-Net training transform that composes the ``prev`` channel per-epoch from candidate ∇φ channels.

The iterative dataset stores ``[CT, cand0(base), cand1(aug), ..., cand{K-1}]`` (absent slots = the -1 sentinel).
This transform, appended LAST in the (train/val) transform list, replaces those channels with ``[CT, prev]``:
it drops the sentinel slots, runs :func:`hercunet.training.prev_channel.compose_prev` (octant "+"-seam over
{zero, base, augs} + bootstrap-zero) on a FRESH RNG each call, and concatenates ``[CT, prev]``. So every epoch a
window sees a different synthetic ``prev`` (no memorised prev→target pairing), and the network — forced to 2 input
channels by the trainer — gets exactly the [CT, prev] it will see at inference.

Candidates ride through nnU-Net's spatial (and intensity) augmentation as ordinary image channels first, so
``prev`` stays spatially consistent with the augmented CT. ``deterministic`` (validation) fixes the RNG so the val
metric is stable across epochs.
"""
from __future__ import annotations

import numpy as np
import torch

from .prev_channel import compose_prev

try:                                                                   # present on the training pod
    from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
except Exception:                                                      # allow import where bg2 is absent
    BasicTransform = object

_SENTINEL = -0.5                                                        # a candidate slot is "absent" if its max < this


class ComposePrevTransform(BasicTransform):
    """Collapse ``[CT, cand0..cand{K-1}]`` -> ``[CT, prev]`` by composing prev from the present candidate slots."""

    def __init__(self, p_boot=0.3, tiling="octant", aug_prob=0.5, zero_prob=0.15, noise_sigma=0.0,
                 frag_prob=0.3, frag_slab_max=60, frag_half_prob=0.4, deterministic=False, seed=0):
        super().__init__()
        self.p_boot = float(p_boot)
        self.tiling = str(tiling)
        self.aug_prob = float(aug_prob)
        self.zero_prob = float(zero_prob)
        self.noise_sigma = float(noise_sigma)
        self.frag_prob = float(frag_prob)
        self.frag_slab_max = int(frag_slab_max)
        self.frag_half_prob = float(frag_half_prob)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self._n = 0

    # BasicTransform calls apply(); define __call__ too so it also works if bg2 is absent (never used there).
    def __call__(self, **data_dict):
        return self.apply(data_dict, **self.get_parameters(**data_dict))

    def get_parameters(self, **data_dict):
        return {}

    def _rng(self):
        if self.deterministic:
            self._n += 1
            return np.random.default_rng(self.seed + self._n)          # stable across epochs for a fixed order
        return np.random.default_rng()                                 # fresh entropy each call

    def apply(self, data_dict, **params):
        img = data_dict["image"]                                       # torch [C, Z, Y, X], C = 1 + K
        is_torch = isinstance(img, torch.Tensor)
        arr = img.detach().cpu().numpy() if is_torch else np.asarray(img)
        ct = arr[0:1]
        cands = []
        for k in range(1, arr.shape[0]):
            c = arr[k]
            if float(c.max()) < _SENTINEL:                            # absent slot (sentinel) -> skip
                continue
            cands.append(np.clip(c, 0.0, 1.0).astype(np.float32))
        if not cands:                                                  # no candidates (e.g. their786) -> bootstrap
            prev = np.zeros(ct.shape[1:], np.float32)
        else:
            prev = compose_prev(cands, self._rng(), p_boot=self.p_boot, tiling=self.tiling,
                                aug_prob=self.aug_prob, zero_prob=self.zero_prob, noise_sigma=self.noise_sigma,
                                frag_prob=self.frag_prob, frag_slab_max=self.frag_slab_max,
                                frag_half_prob=self.frag_half_prob)
        out = np.concatenate([ct, prev[None].astype(ct.dtype)], axis=0)  # [2, Z, Y, X]
        data_dict["image"] = torch.from_numpy(out) if is_torch else out
        if is_torch:
            data_dict["image"] = data_dict["image"].to(img.dtype)
        return data_dict


def compose_prev_transform(deterministic=False, recipe=None):
    """Build a ComposePrevTransform from the HercUNet training recipe — the run301 defaults live in
    :mod:`hercunet.training.recipe` (no env vars). Pass a modified
    :class:`~hercunet.training.recipe.Recipe` to run a variant."""
    from hercunet.training.recipe import active
    r = recipe or active()
    return ComposePrevTransform(
        p_boot=r.prev_pboot, tiling=r.prev_tiling, aug_prob=r.prev_augprob, zero_prob=r.prev_zeroprob,
        noise_sigma=r.prev_noise, frag_prob=r.prev_fragprob, frag_slab_max=r.prev_fragslabmax,
        frag_half_prob=r.prev_fraghalfprob, deterministic=deterministic)

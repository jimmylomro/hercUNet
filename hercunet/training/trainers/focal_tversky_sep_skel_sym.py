"""FocalTverskySepSkel with SYMMETRIC Tversky (α=β=0.5 = soft-Dice) — THIN SHIM. **This is the ablA loss.**

Division of labour: skeleton-recall carries recall, separation carries anti-merge, so Tversky's β>α recall-hack
is redundant AND double-pushes recall → all-ones mush. Reverting Tversky to symmetric re-tasks it as the PRECISION
term. Only α/β change vs the parent (clean one-variable ablation). No loss logic here — inherited from the parent
shim + the shared library builder.
"""
from hercunet.training.trainers.focal_tversky_sep_skel import nnUNetTrainer_FocalTverskySepSkel_500epochs

__all__ = ["nnUNetTrainer_FocalTverskySepSkelSym_500epochs"]


class nnUNetTrainer_FocalTverskySepSkelSym_500epochs(nnUNetTrainer_FocalTverskySepSkel_500epochs):
    tversky_alpha = 0.5   # symmetric: FP now weighted == FN → Tversky is the precision term
    tversky_beta = 0.5    # (recall is skel-recall's job; separation is anti-merge's)
    # inherited unchanged: tversky_gamma=1.33, include_sep/sep_lambda/sep_radius, include_skel/srec_lambda/srec_iters

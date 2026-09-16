"""Single source of truth for the surface-model loss stack (Focal-Tversky + Separation + Skeleton-recall).

Loss logic lifted VERBATIM (bit-identical math) from the research pipeline so the trainer classes stay
thin shims. Both the single-channel trainers and the iterative 2-channel trainer import from here, so
they share ONE definition and stay in lock-step.

Composition (the champion = ``build_surface_loss(..., include_sep=True, include_skel=True, tversky α=β=0.5)``):

    DeepSupervisionWrapper(
        SkeletonRecallLoss(                      # recall: cover the GT medial skeleton (clDice soft-skel)
            SeparationPenaltyLoss(               # anti-merge: penalise P(surface) in inter-sheet gaps
                DC_and_CE_loss(CE + MemoryEfficientSoftTverskyLoss)   # precision (α=β=0.5) + CE, ignore-masked
            )))

Why symmetric Tversky in the champion: focal-Tversky's β>α recall-tilt AND skeleton-recall BOTH push
activations up → all-ones mush; reverting Tversky to α=β=0.5 re-tasks it as the precision term (recall =
skel, anti-merge = sep). See memory labels-vs-loss-ablation.

Imports nnunetv2/torch at module load — only import this on a training host (the pod).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.ddp_allgather import AllGatherGrad


# --------------------------------------------------------------------------------------------------
# Focal-Tversky (drop-in for MemoryEfficientSoftDiceLoss) — the precision/imbalance term inside DC_and_CE.
# --------------------------------------------------------------------------------------------------
class MemoryEfficientSoftTverskyLoss(nn.Module):
    """Tversky / Focal-Tversky, drop-in for MemoryEfficientSoftDiceLoss (same ctor signature + alpha/beta/gamma).

    TI = TP / (TP + alpha*FP + beta*FN);  loss = (1 - TI)^gamma  (gamma=1 -> plain Tversky loss)."""

    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1., ddp=True,
                 alpha=0.3, beta=0.7, gamma=1.0):
        super().__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        axes = tuple(range(2, x.ndim))
        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))
            if x.shape == y.shape:
                y_onehot = y.to(torch.float32)
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
                y_onehot.scatter_(1, y.long(), 1)
            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
            sum_gt = y_onehot.sum(axes, dtype=torch.float32) if loss_mask is None \
                else (y_onehot * loss_mask).sum(axes, dtype=torch.float32)
        if not self.do_bg:
            x = x[:, 1:]
        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes, dtype=torch.float32)
            sum_pred = x.sum(axes, dtype=torch.float32)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes, dtype=torch.float32)
            sum_pred = (x * loss_mask).sum(axes, dtype=torch.float32)
        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0, dtype=torch.float32)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0, dtype=torch.float32)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0, dtype=torch.float32)
            intersect = intersect.sum(0, dtype=torch.float32)
            sum_pred = sum_pred.sum(0, dtype=torch.float32)
            sum_gt = sum_gt.sum(0, dtype=torch.float32)
        tp = intersect
        fp = sum_pred - intersect
        fn = sum_gt - intersect
        ti = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth).clamp_min(1e-8)
        return (1.0 - ti).clamp_min(0.0).pow(self.gamma).mean()


# --------------------------------------------------------------------------------------------------
# Separation penalty — anti-merge: P(surface) inside morphologically-closed inter-sheet GAP voxels.
# --------------------------------------------------------------------------------------------------
class SeparationPenaltyLoss(nn.Module):
    """Wrap a base (pred, target) loss; return base + lambda * separation_penalty.

    separation_penalty = mean over INTER-SHEET GAP voxels of P(surface) = softmax(pred)[:, surface].
    gap = morphological closing(surface) MINUS surface, kept only where target == background(0). radius r sets
    the widest gap bridged/protected (closing fills gaps up to ~2r voxels). ignore(2) and surface(1) are excluded
    from gaps by construction, so we never penalize inside ignored (collapsed / crossing) regions."""

    def __init__(self, base, lam=1.0, radius=2, surface_channel=1):
        super().__init__()
        self.base = base
        self.lam = float(lam)
        self.r = int(radius)
        self.sc = int(surface_channel)

    def _gap(self, t):                                            # t: [B,1,...] float labels {0,1,2}
        surf = (t == 1).float()
        k = 2 * self.r + 1
        dil = F.max_pool3d(surf, k, 1, self.r)                    # dilation
        closed = -F.max_pool3d(-dil, k, 1, self.r)               # erosion(dilation) = closing
        return ((closed > 0.5) & (t == 0)).float()               # thin bg the closing bridged = inter-sheet gap

    def forward(self, pred, target):
        base = self.base(pred, target)
        t = target if target.ndim == pred.ndim else target.view(target.shape[0], 1, *target.shape[1:])
        gap = self._gap(t)
        surf_prob = torch.softmax(pred, 1)[:, self.sc:self.sc + 1]
        sep = (surf_prob * gap).sum() / gap.sum().clamp_min(1.0)
        return base + self.lam * sep


# --------------------------------------------------------------------------------------------------
# Skeleton-recall — recall-only clDice term: cover the GT medial skeleton without a precision half.
# --------------------------------------------------------------------------------------------------
def _soft_erode(x):  return -F.max_pool3d(-x, 3, 1, 1)
def _soft_dilate(x): return F.max_pool3d(x, 3, 1, 1)
def _soft_open(x):   return _soft_dilate(_soft_erode(x))


def soft_skel(img, iters):
    """clDice soft-skeleton (Shin et al.): iterated (img − open(img)) residuals. On our already-thin binary GT
    medial a few iters give ~the 1-vx centre-surface."""
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class SkeletonRecallLoss(nn.Module):
    """Wrap a base (pred,target) loss; return base + lam·(1 − skeleton-recall). recall = Σ(P_surface · skel(GT)) /
    Σ(skel(GT)), ignore-masked (villa's SoftSkeletonRecallLoss). Recall-only: rewards covering the GT medial
    skeleton, penalises nothing outside it (Dice/CE/Tversky keep precision)."""

    def __init__(self, base, lam=1.0, iters=3, surface_channel=1, ignore_label=2):
        super().__init__()
        self.base = base; self.lam = float(lam); self.iters = int(iters)
        self.sc = int(surface_channel); self.ig = ignore_label

    def forward(self, pred, target):
        base = self.base(pred, target)
        t = target if target.ndim == pred.ndim else target.view(target.shape[0], 1, *target.shape[1:])
        gt = (t == 1).float()
        mask = (t != self.ig).float() if self.ig is not None else torch.ones_like(gt)
        skel = soft_skel(gt * mask, self.iters)                     # skeletonize the (non-ignored) GT medial
        P = torch.softmax(pred, 1)[:, self.sc:self.sc + 1]
        rec = (P * skel * mask).sum() / (skel * mask).sum().clamp_min(1.0)
        return base + self.lam * (1.0 - rec)


# --------------------------------------------------------------------------------------------------
# Builder — composes the exact stack each trainer variant uses (one place, no per-trainer duplication).
# --------------------------------------------------------------------------------------------------
def build_surface_loss(*, batch_dice, is_ddp, ignore_label,
                       tversky_alpha, tversky_beta, tversky_gamma,
                       include_sep=False, sep_lambda=0.5, sep_radius=2,
                       include_skel=False, srec_lambda=1.0, srec_iters=3,
                       deep_supervision_weights=None, compile_dc=False):
    """Return the composed loss module. Mirrors the ``_build_loss`` bodies exactly:

      base = DC_and_CE_loss(CE + Focal-Tversky, ignore_label)  [dc optionally torch.compile'd]
      + SeparationPenaltyLoss              if include_sep
      + SkeletonRecallLoss                 if include_skel
      + DeepSupervisionWrapper             if deep_supervision_weights is not None

    ``deep_supervision_weights`` is a 1-D array (the trainer computes it from its DS scales and passes it in —
    that is trainer state, not loss logic). ``compile_dc`` wraps the Tversky/Dice term in torch.compile."""
    dc_ce = DC_and_CE_loss(
        {'batch_dice': batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': is_ddp,
         'alpha': tversky_alpha, 'beta': tversky_beta, 'gamma': tversky_gamma},
        {}, weight_ce=1, weight_dice=1, ignore_label=ignore_label,
        dice_class=MemoryEfficientSoftTverskyLoss)
    if compile_dc:
        dc_ce.dc = torch.compile(dc_ce.dc)
    loss = dc_ce
    if include_sep:
        loss = SeparationPenaltyLoss(loss, lam=sep_lambda, radius=sep_radius)
    if include_skel:
        loss = SkeletonRecallLoss(loss, lam=srec_lambda, iters=srec_iters, ignore_label=ignore_label)
    if deep_supervision_weights is not None:
        loss = DeepSupervisionWrapper(loss, deep_supervision_weights)
    return loss


def deep_supervision_weights(scales, is_ddp, do_compile):
    """The standard nnU-Net DS weight vector (1/2^i, last→0 unless ddp-without-compile), normalised.
    Factored out so every trainer shim computes it identically."""
    w = np.array([1 / (2 ** i) for i in range(len(scales))])
    w[-1] = 1e-6 if (is_ddp and not do_compile) else 0
    return w / w.sum()

"""Thin-surface loss primitives for HercuViT — self-contained (no nnU-Net), reusable.

The sheet-surface target is a ~1-voxel medial surface (~0.1-0.3% of voxels). A plain CE/Dice head collapses to
all-background on it; HercUNet solves this with Focal-Tversky + clDice skeleton-recall (+ a separation penalty for
anti-merge). Those loss terms are **pure torch**, but ``hercunet/training/losses.py`` imports nnU-Net at module
load (for ``DC_and_CE_loss`` / ``DeepSupervisionWrapper`` / ``AllGatherGrad``), so they cannot be imported on a
non-nnU-Net host. This module ports the **pure math verbatim** (single-GPU; no DDP all-gather) so HercuViT gets
the same proven thin-surface behaviour without the nnU-Net dependency.

Math lifted bit-identically from ``hercunet/training/losses.py`` (``MemoryEfficientSoftTverskyLoss``,
``soft_skel`` / ``SkeletonRecallLoss``, ``SeparationPenaltyLoss``) — see memory ``labels-vs-loss-ablation`` for the
champion config (symmetric Tversky α=β=0.5 + skeleton-recall + separation).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# --- morphology + clDice soft-skeleton (verbatim from hercunet/training/losses.py) ---
def _soft_erode(x):  return -F.max_pool3d(-x, 3, 1, 1)
def _soft_dilate(x): return F.max_pool3d(x, 3, 1, 1)
def _soft_open(x):   return _soft_dilate(_soft_erode(x))


def soft_skel(img: torch.Tensor, iters: int) -> torch.Tensor:
    """clDice soft-skeleton (Shin et al.): iterated (img − open(img)) residuals."""
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(int(iters)):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def focal_tversky(prob_fg: torch.Tensor, gt_fg: torch.Tensor, *, alpha: float, beta: float,
                  gamma: float, smooth: float = 1e-5, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Focal-Tversky loss on the foreground. ``prob_fg``/``gt_fg``: ``(B,1,D,H,W)``.

    ``TI = (TP+s)/(TP + α·FP + β·FN + s); loss = (1 − TI)^γ``. γ=1 is plain Tversky; α=β=0.5 = Dice-like precision
    (the champion), β>α tilts toward recall. Per-sample then mean (``batch_dice=False`` path). ``mask`` (``(B,1,…)``
    1=keep, 0=ignore) excludes ignore voxels from TP/FP/FN, matching nnU-Net's ``loss_mask`` — without it a surface
    prediction in an ignore region is wrongly counted as a false positive."""
    axes = tuple(range(2, prob_fg.ndim))
    m = 1.0 if mask is None else mask
    tp = (prob_fg * gt_fg * m).sum(axes, dtype=torch.float32)
    fp = (prob_fg * (1.0 - gt_fg) * m).sum(axes, dtype=torch.float32)
    fn = ((1.0 - prob_fg) * gt_fg * m).sum(axes, dtype=torch.float32)
    ti = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth).clamp_min(1e-8)
    return (1.0 - ti).clamp_min(0.0).pow(gamma).mean()


def skeleton_recall(prob_fg: torch.Tensor, gt_fg: torch.Tensor, *, iters: int = 3,
                    mask: torch.Tensor | None = None) -> torch.Tensor:
    """Recall-only clDice term (SkeletonRecallLoss): ``1 − Σ(P·skel(GT))/Σ(skel(GT))`` — rewards covering the GT
    medial skeleton, penalises nothing outside it (Tversky/CE keep precision). ``prob_fg``/``gt_fg``: ``(B,1,…)``."""
    g = gt_fg if mask is None else gt_fg * mask
    skel = soft_skel(g, iters)
    p = prob_fg if mask is None else prob_fg * mask
    sk = skel if mask is None else skel * mask            # re-mask the skeleton (soft_skel can bleed ≤1vx onto
    rec = (p * sk).sum() / sk.sum().clamp_min(1.0)        # ignore voxels); matches nnU-Net SkeletonRecallLoss exactly
    return 1.0 - rec


def separation_penalty(prob_fg: torch.Tensor, target: torch.Tensor, *, radius: int = 2) -> torch.Tensor:
    """Anti-merge: mean ``P(surface)`` inside morphologically-closed inter-sheet GAP voxels (closing(surface) −
    surface, kept where target==0). ``prob_fg``: ``(B,1,…)``; ``target``: ``(B,1,…)`` int {0,1,(2=ignore)}."""
    surf = (target == 1).float()
    k = 2 * int(radius) + 1
    dil = F.max_pool3d(surf, k, 1, radius)
    closed = -F.max_pool3d(-dil, k, 1, radius)                 # closing = erosion(dilation)
    gap = ((closed > 0.5) & (target == 0)).float()
    return (prob_fg * gap).sum() / gap.sum().clamp_min(1.0)


def surface_terms(logits: torch.Tensor, target: torch.Tensor, *,
                  tversky_alpha: float = 0.5, tversky_beta: float = 0.5, tversky_gamma: float = 1.0,
                  skel_iters: int = 3, sep_radius: int = 2, ignore_label: int | None = None) -> dict:
    """All surface terms for ONE resolution. ``logits`` ``(B,C,D,H,W)``, ``target`` ``(B,D,H,W)`` int.

    Returns ``{"ce", "tversky", "skel", "sep"}`` scalar tensors; the caller weights + sums them (and over
    deep-supervision scales). CE is PLAIN (unweighted) — matching HercUNet's ``build_surface_loss`` (``DC_and_CE_loss``
    with ``ce_kwargs={}``): imbalance is carried by Focal-Tversky (α=β=0.5) + skeleton-recall, NOT by a CE class
    weight. Tversky + skeleton-recall are the thin-surface precision/recall pair; separation is the anti-merge term.
    """
    tl = target.long()
    ce = F.cross_entropy(logits, tl,
                         ignore_index=(-100 if ignore_label is None else int(ignore_label)))
    prob_fg = torch.softmax(logits, 1)[:, 1:2]
    t1 = tl.unsqueeze(1)
    gt_fg = (t1 == 1).float()
    mask = (t1 != ignore_label).float() if ignore_label is not None else None
    return {
        "ce": ce,
        "tversky": focal_tversky(prob_fg, gt_fg, alpha=tversky_alpha, beta=tversky_beta, gamma=tversky_gamma, mask=mask),
        "skel": skeleton_recall(prob_fg, gt_fg, iters=skel_iters, mask=mask),
        "sep": separation_penalty(prob_fg, t1, radius=sep_radius),
    }

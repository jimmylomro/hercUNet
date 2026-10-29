"""Stitcher training loss (docs/hercuvit-stitching.md §7.7, extended) — deep-supervised, applied every decoder layer:

  * **focal BCE** on the sheet-to-sheet affinity logits — a pos-weighted sigmoid focal loss. With ~90% negative
    pairs, the auto ``pos_weight`` (≈ #neg/#pos) balances the classes and the focal ``gamma`` down-weights the easy
    pairs so the gradient concentrates on the hard near-misses; and
  * **SupCon contrastive** on the L2-normalised query embeddings — each fragment is pulled toward its same-sheet
    fragments and pushed from the different-sheet ones (same-octant pairs are free hard negatives). Ignore(-1) pairs
    are excluded from both the numerator and the denominator.

Both terms are summed over the ``L`` layers. Confidence weighting is dropped (queries are already gated to confident
sheets upstream); ``conf_q`` is accepted but unused.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _focal_bce(logits, targets, valid, pos_weight, gamma):
    """Pos-weighted sigmoid focal loss, mean over the valid pairs. ``logits``/``targets``/``valid`` are ``(B,M,M)``."""
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")       # -log p_t (per element)
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)                                       # prob of the true class
    focal = (1 - p_t).clamp(min=0.0) ** gamma * ce                                    # down-weight easy pairs
    w = targets * pos_weight + (1 - targets)                                          # up-weight the rare positives
    fw = focal * w
    return fw[valid].sum() / valid.sum().clamp(min=1)


def _supcon(emb, Y, valid_q, tau):
    """Supervised contrastive loss over the query embeddings (one anchor per fragment with ≥1 same-sheet partner).
    ``emb`` ``(B,M,d)`` L2-normalised; positives = Y==1, denominator = known pairs (Y>=0), ignore(-1)/self excluded."""
    B, M, _ = Y.shape
    dev = Y.device
    sim = (emb @ emb.transpose(-1, -2)) / tau                                         # (B,M,M) cosine / temperature
    eye = torch.eye(M, dtype=torch.bool, device=dev)[None]
    vpair = valid_q[:, :, None] & valid_q[:, None, :]
    known = vpair & ~eye & (Y >= 0)                                                   # columns that count (non-ignore, non-self)
    pos = known & (Y == 1)
    sim_k = sim.masked_fill(~known, torch.finfo(sim.dtype).min)
    log_prob = sim - torch.logsumexp(sim_k, dim=-1, keepdim=True)                     # log-softmax over known columns
    pos_count = pos.sum(-1)                                                           # (B,M)
    has_pos = pos_count > 0
    per_anchor = -(log_prob * pos).sum(-1) / pos_count.clamp(min=1)                   # mean over positives per anchor
    return per_anchor[has_pos].mean() if bool(has_pos.any()) else sim.new_zeros(())


def stitch_loss(out, Y, valid_q, conf_q=None, *, pos_weight: float = 0.0, gamma: float = 2.0,
                tau: float = 0.1, lambda_contrast: float = 1.0):
    """``out``: list of ``L`` dicts ``{"logit": (B,M,M), "emb": (B,M,d)}`` from :class:`StitchDecoder`.
    Returns ``(loss, stats)`` = Σ_layers [ focal_bce + lambda_contrast · supcon ]."""
    B, M, _ = Y.shape
    dev = Y.device
    vpair = valid_q[:, :, None] & valid_q[:, None, :]
    eye = torch.eye(M, dtype=torch.bool, device=dev)[None]
    valid = vpair & ~eye & (Y >= 0)                              # drop padding, diagonal, and ignore (-1) pairs
    Yf = (Y == 1).float()

    npos = (valid & (Y == 1)).sum()
    nneg = (valid & (Y == 0)).sum()
    pw = torch.tensor(float(pos_weight), device=dev) if (pos_weight and pos_weight > 0) \
        else (nneg.float() / npos.float().clamp(min=1.0)).clamp(1.0, 1000.0)

    focal_sum = out[0]["logit"].new_zeros(())
    contrast_sum = out[0]["logit"].new_zeros(())
    for lo in out:
        focal_sum = focal_sum + _focal_bce(lo["logit"].float(), Yf, valid, pw, gamma)
        if lambda_contrast > 0:
            contrast_sum = contrast_sum + _supcon(lo["emb"].float(), Y, valid_q, tau)
    total = focal_sum + lambda_contrast * contrast_sum

    stats = {"pos_weight": float(pw), "npos": int(npos), "nneg": int(nneg), "n_valid": int(valid.sum()),
             "focal": float(focal_sum.detach()), "contrast": float(contrast_sum.detach())}
    return total, stats

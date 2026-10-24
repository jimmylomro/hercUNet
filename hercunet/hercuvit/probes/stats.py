"""Pure-numpy statistics for the representation probes — no sklearn dependency.

AUC (Mann-Whitney), best-Dice threshold sweep, regularised Fisher linear discriminant, covariance spectrum
summaries (participation ratio, effective rank), and the 1/8 token labelling that maps a full-resolution
thin-surface + Voronoi-owner annotation onto the encoder's patch tokens.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auc(pos, neg) -> float:
    """Area under the ROC curve via the Mann-Whitney statistic: P(score(pos) > score(neg))."""
    pos = np.asarray(pos, dtype=np.float64); neg = np.asarray(neg, dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    n1 = len(pos)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(neg)))


def best_dice(score, y) -> tuple[float, float]:
    """Best Dice of ``(score > τ)`` against binary ``y`` over all thresholds; returns ``(dice, τ)``."""
    score = np.asarray(score); y = np.asarray(y, dtype=np.float64)
    order = np.argsort(-score); ys = y[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys); P = ys.sum()
    dice = 2 * tp / (tp + fp + P + 1e-9)
    i = int(np.argmax(dice))
    return float(dice[i]), float(score[order][i])


def unit(x) -> np.ndarray:
    """L2-normalise rows."""
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def lda_dir(Xp, Xn, reg: float = 1e-2) -> np.ndarray:
    """Regularised Fisher linear discriminant direction separating classes ``Xp`` (N,d) and ``Xn`` (M,d).
    ``reg`` adds a multiple of the mean within-class-scatter eigenvalue to the diagonal. Returns a unit vector."""
    d = Xp.shape[1]
    mp, mn = Xp.mean(0), Xn.mean(0)
    Sw = (np.atleast_2d(np.cov(Xp.T)) * len(Xp) + np.atleast_2d(np.cov(Xn.T)) * len(Xn)).reshape(d, d)
    Sw += reg * np.trace(Sw) / d * np.eye(d)
    w = np.linalg.solve(Sw, (mp - mn).reshape(d))
    return w / (np.linalg.norm(w) + 1e-9)


def spectrum(X) -> tuple[np.ndarray, float, float]:
    """Covariance eigenspectrum of ``X`` (N,d) with two effective-dimensionality summaries.

    Returns ``(eigenvalues desc, participation_ratio, effective_rank)`` where participation ratio
    ``(Σλ)²/Σλ²`` is an effective number of equally-weighted dimensions and effective rank is ``exp(entropy)``
    of the normalised spectrum."""
    Xc = X - X.mean(0)
    _, sv, _ = np.linalg.svd(Xc, full_matrices=False)
    eig = (sv ** 2) / (len(X) - 1)
    eig = eig[eig > 0]
    pr = float(eig.sum() ** 2 / (eig ** 2).sum())
    p = eig / eig.sum()
    erank = float(np.exp(-(p * np.log(p)).sum()))
    return eig, pr, erank


def token_labels(sem: np.ndarray, owner: np.ndarray, patch: int = 8):
    """Pool a full-resolution annotation onto the encoder's 1/8 patch tokens.

    ``sem`` (D,H,W) ∈ {0 bg, 1 surface, 2 ignore}; ``owner`` (D,H,W) Voronoi sheet id. Each ``patch``³ block
    becomes one token. Returns ``(is_sheet, is_bg, owner_tok)`` flat over the token grid (raster z,y,x): a token
    is **sheet** if its block holds ≥2 surface voxels and <64 ignore voxels; **background** if 0 surface and 0
    ignore; ``owner_tok`` is the modal nonzero thin-owner over the block (−1 where none)."""
    side = sem.shape[0] // patch
    s6 = (sem == 1).reshape(side, patch, side, patch, side, patch)
    i6 = (sem == 2).reshape(side, patch, side, patch, side, patch)
    surf_cnt = s6.sum(axis=(1, 3, 5)).reshape(-1)
    ign_cnt = i6.sum(axis=(1, 3, 5)).reshape(-1)
    is_sheet = (surf_cnt >= 2) & (ign_cnt < 64)
    is_bg = (surf_cnt == 0) & (ign_cnt == 0)
    own_thin = (owner * (sem == 1)).reshape(side, patch, side, patch, side, patch)
    blk = own_thin.transpose(0, 2, 4, 1, 3, 5).reshape(side ** 3, -1)       # (token, 512) owner values
    owner_tok = np.full(side ** 3, -1, dtype=np.int64)
    for t in np.where(is_sheet)[0]:
        v = blk[t]; v = v[v > 0]
        if len(v):
            vals, cnts = np.unique(v, return_counts=True)
            owner_tok[t] = int(vals[np.argmax(cnts)])
    return is_sheet, is_bg, owner_tok

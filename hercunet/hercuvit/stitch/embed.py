"""Per-window normalised-PCA embedding + median-after-PCA query descriptor (docs/hercuvit-stitching.md §3.4).

The ONE implementation of the stitcher embedding reduction, shared by every producer so the spatial tokens and the
per-fragment median queries always live in the same space:

  * ``hercuvit labels create`` (:mod:`hercunet.hercuvit.stitch.labels`) — self-supervised training-sample extraction;
  * (later) the ``hercuvit infer`` path — when it emits the pooled-embedding zarr + the per-sheet ``.asv`` median
    column, it calls these SAME functions (no second, divergent reduction).

Both steps are pure NumPy and per window: z-score each raw feature dim (statistics fit on the window's SURFACE
cells), project every cell onto the window's top-``D`` principal directions, and — for a fragment query — take the
per-dim **median across its already-projected cells**. The median is taken AFTER the projection on purpose:
``proj(median(raw)) != median(proj(raw))``, so median-then-project would land the query off the token manifold.
"""
from __future__ import annotations

import numpy as np


def reduce_normalised_pca(feats: np.ndarray, surf_mask: np.ndarray, D: int) -> np.ndarray:
    """Per-window normalised PCA → ``(..., D)`` (docs §3.4).

    ``feats``  : ``(..., C)`` raw per-cell features (e.g. a window's 1/8 encoder tokens, ``(G, G, G, C)``).
    ``surf_mask``: ``(...)`` truthy on the window's SURFACE cells — the PCA mean/std/basis are fit on these only
                   (the structure the stitcher cares about), then EVERY cell is projected.
    Returns ``(..., D)`` ``float32``. Fit falls back to all cells when too few surface cells exist.
    """
    lead = feats.shape[:-1]
    X = feats.reshape(-1, feats.shape[-1])
    s = np.asarray(surf_mask).reshape(-1) > 0.5
    fit = X[s] if int(s.sum()) > D + 8 else X
    mu = fit.mean(0)
    sd = fit.std(0) + 1e-6
    Z = (fit - mu) / sd
    zc = Z.mean(0)
    _, _, vt = np.linalg.svd(Z - zc, full_matrices=False)
    V = vt[:D]
    emb = (((X - mu) / sd) - zc) @ V.T
    return emb.reshape(*lead, D).astype(np.float32)


def median_after_pca(emb: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Per-dim median of a fragment's already-projected cells (docs §3.4) → ``(D,)``.

    ``emb``  : ``(..., D)`` from :func:`reduce_normalised_pca`.
    ``cells``: boolean mask (same leading shape) of the fragment's cells.
    Median AFTER the projection — never ``proj(median(raw))`` (the median does not commute with the projection).
    """
    return np.median(emb[np.asarray(cells)], 0)

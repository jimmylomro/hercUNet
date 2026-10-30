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


def corner_spatial_tokens(emb_by_oct, *, h, fov, nsp, CROP: int = 128, POOL: int = 8):
    """Assemble one corner's spatial memory from its 8 octant windows' per-window normalised-PCA grids (docs §5.1).

    ``emb_by_oct`` maps each octant sign-tuple ``k = (sz, sy, sx)`` with ``sa ∈ {-1, +1}`` to that source window's
    ``(G, G, G, D)`` embedding grid (``G = CROP // POOL``). Geometry (all in the native/global voxel units the grids
    live in): ``h`` = octant-window-centre offset from the corner along each axis (``±h``); ``fov`` = window side,
    so tokens are kept inside the ``±fov/2`` cube centred on the corner; ``nsp`` = native voxels per pooled-grid step.

    Returns ``(Xs (S, D) f32, Ps (S, 3) f16 in [-1, 1], slot_s (S,) u8)`` — each kept token's embedding, its position
    relative to the corner (normalised by ``fov/2``), and its source-window slot ``4·b_z + 2·b_y + b_x`` (§4). This is
    the ONE placement implementation, shared by ``labels create`` (training samples) and ``stitcher infer`` (stored
    artifacts), so spatial tokens are built identically on both paths.
    """
    G = CROP // POOL
    ii, jj, kk = np.meshgrid(np.arange(G), np.arange(G), np.arange(G), indexing="ij")
    base = (np.stack([ii, jj, kk], -1) * POOL + POOL / 2 - CROP / 2) * nsp         # (G,G,G,3) token offset from centre
    Xs, Ps, Ss = [], [], []
    for k, e in emb_by_oct.items():
        off = np.array([k[0] * h, k[1] * h, k[2] * h], np.float64)                 # window centre offset from corner
        slot = (k[0] > 0) * 4 + (k[1] > 0) * 2 + (k[2] > 0)
        d = base + off                                                            # (G,G,G,3) token offset from corner
        inside = (np.abs(d) <= fov / 2).all(-1)
        Xs.append(np.asarray(e)[inside])
        Ps.append((d[inside] / (fov / 2)).astype(np.float16))
        Ss.append(np.full(int(inside.sum()), slot, np.uint8))
    return np.concatenate(Xs), np.concatenate(Ps), np.concatenate(Ss)

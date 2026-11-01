"""The GPU/torch PCA reduction must match the numpy one it replaces — up to the LAPACK per-component sign.

``hercuvit infer --store-embeddings`` reduces each window's encoder tokens with
:func:`hercunet.hercuvit.stitch.embed.reduce_normalised_pca_torch` (on the GPU) instead of copying the whole token
tensor to the CPU for the numpy :func:`reduce_normalised_pca`. The stored embeddings must be the same the stitcher
was trained on. The sign of each principal direction is LAPACK-arbitrary in BOTH paths (and already independent
per window in training — a window's tokens and its median flip together), so the contract is: identical magnitudes,
equal up to a global per-component sign. These tests pin exactly that, so the GPU swap can't silently shift the
embedding distribution the trained stitcher expects.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hercunet.hercuvit.stitch.embed import reduce_normalised_pca, reduce_normalised_pca_torch


def _match_up_to_sign(a: np.ndarray, b: np.ndarray, D: int):
    af, bf = a.reshape(-1, D), b.reshape(-1, D)
    for d in range(D):                                            # each component equal up to ONE global sign flip
        assert np.allclose(af[:, d], bf[:, d], atol=1e-3) or np.allclose(af[:, d], -bf[:, d], atol=1e-3), \
            f"component {d} differs beyond a sign flip"
    assert np.allclose(np.abs(af), np.abs(bf), atol=1e-3)         # magnitudes identical regardless of sign


def test_torch_pca_matches_numpy_up_to_sign():
    rng = np.random.default_rng(0)
    G, C, D = 8, 32, 3
    feats = rng.standard_normal((G, G, G, C)).astype(np.float32)
    surf = rng.random((G, G, G)) > 0.5                           # ~half surface cells (> D+8, so fit on surface)
    a = reduce_normalised_pca(feats, surf, D)
    b = reduce_normalised_pca_torch(torch.from_numpy(feats), torch.from_numpy(surf), D).numpy()
    assert a.shape == b.shape == (G, G, G, D)
    _match_up_to_sign(a, b, D)


def test_torch_pca_fallback_when_few_surface_cells():
    """With too few surface cells (≤ D+8), both paths fit PCA on ALL cells — same fallback, same result up to sign."""
    rng = np.random.default_rng(1)
    G, C, D = 8, 16, 3
    feats = rng.standard_normal((G, G, G, C)).astype(np.float32)
    surf = np.zeros((G, G, G), bool)                             # no surface cells → fit on all
    a = reduce_normalised_pca(feats, surf, D)
    b = reduce_normalised_pca_torch(torch.from_numpy(feats), torch.from_numpy(surf), D).numpy()
    _match_up_to_sign(a, b, D)


def test_torch_pca_median_consistent():
    """The per-sheet median descriptor (median_after_pca over a cell mask) also matches up to the component sign."""
    from hercunet.hercuvit.stitch.embed import median_after_pca
    rng = np.random.default_rng(2)
    G, C, D = 8, 24, 3
    feats = rng.standard_normal((G, G, G, C)).astype(np.float32)
    surf = rng.random((G, G, G)) > 0.4
    cells = rng.random((G, G, G)) > 0.7
    a = median_after_pca(reduce_normalised_pca(feats, surf, D), cells)
    b = median_after_pca(reduce_normalised_pca_torch(torch.from_numpy(feats), torch.from_numpy(surf), D).numpy(), cells)
    assert np.allclose(np.abs(a), np.abs(b), atol=1e-3)

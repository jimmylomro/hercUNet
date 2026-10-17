"""Dinovol robust normalisation — shape preservation + median/MAD statistics."""
import numpy as np
import torch

from hercunet.hercuvit.data.normalize import DinovolNormalizer, dinovol_normalise, normalize_robust


def test_robust_shape_and_stats():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    img = rng.normal(100.0, 20.0, size=(16, 16, 16)).astype(np.float32)
    out = dinovol_normalise(img)
    assert out.shape == img.shape
    assert out.dtype == np.float32
    # robust centring: median ~0, MAD-scaled so median(|x|) ~ 0.6745 (the MAD of a unit normal).
    assert abs(float(np.median(out))) < 0.05
    assert 0.5 < float(np.median(np.abs(out))) < 0.85
    assert np.isfinite(out).all()


def test_robust_constant_volume_is_finite():
    # degenerate (flat) crop must not divide by ~0 — villa's fallback ladder → all-finite output.
    img = np.full((8, 8, 8), 7.0, np.float32)
    out = normalize_robust(img)
    assert np.isfinite(out).all()
    assert out.shape == img.shape


def test_normalizer_matches_function():
    rng = np.random.default_rng(1)
    img = rng.normal(0.0, 1.0, size=(8, 8, 8)).astype(np.float32)
    assert np.allclose(DinovolNormalizer().run(img), normalize_robust(img))


def test_clip_bounds_outliers():
    rng = np.random.default_rng(2)
    img = rng.normal(0.0, 1.0, size=(32, 32, 32)).astype(np.float32)
    img[0, 0, 0] = 1e6                      # a wild outlier must be clipped away (percentile 99)
    out = dinovol_normalise(img)
    assert np.isfinite(out).all()
    assert float(out.max()) < 100.0

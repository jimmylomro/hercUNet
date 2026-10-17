"""Dinovol intensity normalisation (replicated from villa's ``ssl_zarr_dataset`` / ``normalization.py``).

The Dinovol encoder (``scrollprize/dinovol_ps8_9.6um_101626``) was SSL-trained with villa's
``SSLZarrDataset``, which normalises every CT crop with ``get_normalization(config.normalization_scheme)``
and ``config.normalization_scheme`` **defaults to ``"robust"``**. The released checkpoint's ``config.dataset``
block carries **no** ``normalization_scheme`` key (verified by loading ``teacher_305k.pt``), so the encoder
saw the **robust** scheme: a percentile clip (1–99) followed by a median / MAD rescale. HercuViT must feed the
encoder CT normalised the *same* way, so this module ports villa's ``normalize_robust`` faithfully.

Source (fetched during implementation, MIT/Apache-2.0):
``raw.githubusercontent.com/ScrollPrize/villa/dinovol-updates/dinovol/dinovol_2/dataset/normalization.py``
and ``…/dataset/ssl_zarr_dataset.py`` (``_finalize_crop`` → ``self.normalizer.run(crop)`` then trilinear
resize to the crop target). villa applies ``gaussian_blur`` in its augmentation, which HercuViT deliberately
**omits** (preserve one-voxel sheet separation — ``docs/hercuvit.md`` §8).

Pure numpy; safe to import anywhere.
"""
from __future__ import annotations

import numpy as np

#: villa's ``normalization.EPS``.
EPS = 1e-8
#: The normalisation scheme the released Dinovol checkpoint trained with (its config omits the key → villa
#: default ``"robust"``). Recorded here so the choice is auditable and a future re-check is one grep away.
DINOVOL_NORMALIZATION_SCHEME = "robust"


def normalize_robust(
    image: np.ndarray,
    *,
    percentile_lower: float = 1.0,
    percentile_upper: float = 99.0,
    clip_values: bool = True,
    target_dtype=np.float32,
) -> np.ndarray:
    """Robust normalisation: percentile clip, then ``(x - median) / (1.4826 · MAD)``.

    A faithful port of villa ``normalize_robust`` (no mask — villa calls it without one in ``_finalize_crop``).
    The ``1.4826`` factor makes the MAD a consistent estimator of the standard deviation for Gaussian data;
    the fallback ladder (pre-clip std → half the percentile span → ``1.0``) matches villa exactly so a flat /
    degenerate crop never divides by ~0. Non-finite outputs are zeroed (villa ``np.nan_to_num``).
    """
    arr = np.asarray(image).astype(target_dtype, copy=True)
    valid = arr.reshape(-1)
    if valid.size == 0:
        return arr

    std_preclip = float(np.std(valid))
    min_preclip = float(np.min(valid))
    max_preclip = float(np.max(valid))

    lower_val = upper_val = None
    if clip_values:
        lower_val = float(np.percentile(valid, percentile_lower))
        upper_val = float(np.percentile(valid, percentile_upper))
        np.clip(arr, lower_val, upper_val, out=arr)
        valid = arr.reshape(-1)

    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    scaled_mad = 1.4826 * mad

    if not np.isfinite(scaled_mad) or scaled_mad < 1e-6:
        if lower_val is not None and upper_val is not None:
            percentile_span = upper_val - lower_val
        else:
            percentile_span = max_preclip - min_preclip
        candidates = []
        if np.isfinite(std_preclip):
            candidates.append(abs(std_preclip))
        if percentile_span is not None and np.isfinite(percentile_span):
            candidates.append(abs(percentile_span) / 2.0)
        scaled_mad = next((c for c in candidates if c >= 1e-6), None)
        if scaled_mad is None or not np.isfinite(scaled_mad):
            scaled_mad = 1.0

    arr -= median
    arr /= scaled_mad
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def dinovol_normalise(image: np.ndarray) -> np.ndarray:
    """The exact normalisation the Dinovol encoder expects — :func:`normalize_robust` with villa's defaults."""
    return normalize_robust(image)


class DinovolNormalizer:
    """Callable wrapper mirroring villa's ``RobustNormalization`` instance (``.run(image)``)."""

    def __init__(self, percentile_lower: float = 1.0, percentile_upper: float = 99.0,
                 clip_values: bool = True) -> None:
        self.percentile_lower = float(percentile_lower)
        self.percentile_upper = float(percentile_upper)
        self.clip_values = bool(clip_values)

    def run(self, image: np.ndarray) -> np.ndarray:
        return normalize_robust(image, percentile_lower=self.percentile_lower,
                                percentile_upper=self.percentile_upper, clip_values=self.clip_values)

"""3-D structure-tensor frame field — Stage-1 v0 substrate characterisation (classical).

Estimates local papyrus geometry per voxel straight from image gradients: no training, no
labels, immediately inspectable. It is the baseline the learned (equivariant, PyTorch/GPU)
frame estimator must beat, and it doubles as a pseudo-label source in high-coherence regions.

Scales (two, deliberately separate):
  * ``sigma_grad``   — DERIVATIVE-OF-GAUSSIAN differentiation scale. Gradients are convolutions
    with the analytic Gaussian derivative (smooth, band-limited), never finite differences.
    ~1 voxel keeps fibre edges while killing scan noise.
  * ``sigma_tensor`` — Gaussian integration scale of the tensor = the orientation-coherence
    neighbourhood. ~4 voxels (≈ one fibre bundle) gives fibre orientation; ~16 voxels
    (≈ sheet thickness) makes the largest-eigenvector track the through-sheet normal.

Eigen-decomposition of the 3×3 tensor per voxel (ascending eigenvalues λ0≤λ1≤λ2):
  * fibre direction = eigenvector of λ0 (least intensity change = along the fibre / in-plane),
  * sheet normal    = eigenvector of λ2 (most change = across the sheet),
  * coherence       = 1 − λ0/λ2 ∈ [0,1]  (0 = isotropic/crushed → the "unresolved" state).
Gradient sign is irrelevant (the tensor uses outer products), so no sign convention is needed.
Axis order is (z, y, x) throughout; eigenvector components follow the same order.

The same computation moves verbatim to ``torch.nn.functional.conv3d`` + ``torch.linalg.eigh``
on GPU for batched/dense processing; scipy is used here only for the interactive single-block.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def structure_tensor_frame(
    block: np.ndarray, sigma_grad: float = 1.0, sigma_tensor: float = 4.0
) -> dict:
    """Per-voxel frame field for an isotropic ``(dz, dy, dx)`` block. Returns a dict with
    ``fibre`` / ``normal`` ``(dz,dy,dx,3)`` unit vectors, ``coherence`` ``(dz,dy,dx)``, and the
    ascending ``eigvals`` ``(dz,dy,dx,3)``."""
    b = np.asarray(block, dtype=np.float32)
    # Derivative-of-Gaussian gradients (smoothed derivatives): order=1 on one axis, 0 elsewhere.
    gz = ndimage.gaussian_filter(b, sigma_grad, order=(1, 0, 0))
    gy = ndimage.gaussian_filter(b, sigma_grad, order=(0, 1, 0))
    gx = ndimage.gaussian_filter(b, sigma_grad, order=(0, 0, 1))

    def integ(a: np.ndarray) -> np.ndarray:
        return ndimage.gaussian_filter(a, sigma_tensor)

    jzz, jyy, jxx = integ(gz * gz), integ(gy * gy), integ(gx * gx)
    jzy, jzx, jyx = integ(gz * gy), integ(gz * gx), integ(gy * gx)

    j = np.empty(b.shape + (3, 3), np.float32)
    j[..., 0, 0], j[..., 1, 1], j[..., 2, 2] = jzz, jyy, jxx
    j[..., 0, 1] = j[..., 1, 0] = jzy
    j[..., 0, 2] = j[..., 2, 0] = jzx
    j[..., 1, 2] = j[..., 2, 1] = jyx

    w, v = np.linalg.eigh(j)              # ascending eigenvalues; v[..., :, i] = i-th eigenvector
    fibre = v[..., :, 0]                  # smallest λ → along fibre / within sheet plane
    normal = v[..., :, 2]                 # largest λ  → across the sheet
    coherence = 1.0 - w[..., 0] / (w[..., 2] + 1e-12)
    return {"fibre": fibre, "normal": normal, "coherence": coherence, "eigvals": w}


def compression_field(
    block: np.ndarray, voxel_um: float = 2.4, sigma_pre: float = 5.0, downsample: int = 3,
    sigma_grad: float = 1.0, sigma_tensor: float = 16.0,
) -> dict:
    """Dense per-voxel layering period (µm) = the fluid "streamwise wavenumber" of the sheet
    stack: how tightly sheets are packed. Small period = compressed/pressed; large = loose/
    delaminated. No patch-FFT — it is the local frequency of the intensity along the sheet
    normal, estimated densely as ω = √(⟨(∂²I/∂n²)²⟩ / ⟨(∂I/∂n)²⟩), period = 2π/ω.

    The block is LOW-PASSED (``sigma_pre``) to strip the fibre texture (else the ω⁴ weight in
    the 2nd derivative locks onto the fibre period) and then DOWNSAMPLED (``downsample``): the
    period field is smooth at the sheet scale, so it is computed coarse and upsampled back —
    ~D³ cheaper. Returns period/coherence fields at the input block's resolution."""
    b = np.asarray(block, dtype=np.float32)
    blow = ndimage.gaussian_filter(b, sigma_pre)
    d0 = max(1, int(downsample))
    if d0 > 1:
        blow = blow[::d0, ::d0, ::d0]
        vox = voxel_um * d0
        st = max(2.0, sigma_tensor / d0)
    else:
        vox, st = voxel_um, sigma_tensor

    ff = structure_tensor_frame(blow, sigma_grad, st)            # coarse normal on the sheet band
    n = ff["normal"]
    nz, ny, nx = n[..., 0], n[..., 1], n[..., 2]

    def d(order):
        return ndimage.gaussian_filter(blow, sigma_grad, order=order)

    gz, gy, gx = d((1, 0, 0)), d((0, 1, 0)), d((0, 0, 1))
    hzz, hyy, hxx = d((2, 0, 0)), d((0, 2, 0)), d((0, 0, 2))
    hzy, hzx, hyx = d((1, 1, 0)), d((1, 0, 1)), d((0, 1, 1))
    d1 = gz * nz + gy * ny + gx * nx                             # ∂I/∂n
    d2 = (nz * nz * hzz + ny * ny * hyy + nx * nx * hxx          # ∂²I/∂n²
          + 2 * (nz * ny * hzy + nz * nx * hzx + ny * nx * hyx))
    e1 = ndimage.gaussian_filter(d1 * d1, st)
    e2 = ndimage.gaussian_filter(d2 * d2, st)
    omega = np.sqrt(e2 / np.maximum(e1, 1e-12))
    period_um = (2.0 * np.pi / np.maximum(omega, 1e-6) * vox).astype(np.float32)
    coherence = ff["coherence"].astype(np.float32)
    if d0 > 1:  # upsample the (smooth) fields back to the input resolution
        zoom = tuple(bs / cs for bs, cs in zip(b.shape, blow.shape))
        period_um = ndimage.zoom(period_um, zoom, order=1)
        coherence = ndimage.zoom(coherence, zoom, order=1)
    return {"period_um": period_um, "coherence": coherence}



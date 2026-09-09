"""GPU structure-tensor frame field (torch) — same math as ``structure_tensor.py``, batched.

Derivative-of-Gaussian gradients as separable ``conv3d`` and the per-voxel 3×3 eigen-decomposition
as one batched ``torch.linalg.eigh`` — so the whole dense field is computed on the GPU in a
fraction of the scipy time, which is what makes a *large* window at a *fine* scale interactive.
Falls back to CPU torch if CUDA is absent. Returns numpy arrays (z,y,x axis order), matching the
scipy version so it's a drop-in. This is also the core the Rung-0 batch export / Rung-1 reuse.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:  # torch optional
    _HAS_TORCH = False


def torch_available() -> bool:
    return _HAS_TORCH


def _kernel(sigma: float, order: int, device):
    r = max(1, int(np.ceil(3 * sigma)))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    g = torch.exp(-x * x / (2 * sigma * sigma))
    g = g / g.sum()
    if order == 0:
        return g
    if order == 1:                                   # sign irrelevant (tensor uses ∇⊗∇)
        return -(x / (sigma * sigma)) * g
    return ((x * x - sigma * sigma) / sigma ** 4) * g


def _conv_axis(v, k, axis):
    """1-D convolution of (1,1,D,H,W) with kernel ``k`` along spatial ``axis`` (2/3/4), reflect."""
    r = (k.numel() - 1) // 2
    pad = [0, 0, 0, 0, 0, 0]
    slot = {2: (4, 5), 3: (2, 3), 4: (0, 1)}[axis]   # F.pad orders last dim first (W,H,D)
    pad[slot[0]] = r
    pad[slot[1]] = r
    mode = "reflect" if r < v.shape[axis] else "replicate"  # reflect needs pad < dim
    v = F.pad(v, pad, mode=mode)
    shape = [1, 1, 1, 1, 1]
    shape[axis] = k.numel()
    return F.conv3d(v, k.view(shape))


def _sym3x3_normal_coh(a00, a11, a22, a01, a02, a12, torch):
    """Closed-form largest eigenvector (across-sheet NORMAL) + coherence (1 - λmin/λmax) of the symmetric
    structure tensor, per voxel — Cardano eigenvalues + the (A-λ2 I)(A-λ3 I) column for λmax. Fully parallel
    elementwise (no batched eigh → ~10× faster than MAGMA on 3×3). Robust when the two SMALLER eigenvalues
    are degenerate (a sheet's isotropic tangent plane) — which is exactly our ∇φ case, and we never need the
    then-arbitrary in-plane eigenvectors. Validated vs numpy eigh to machine precision. Returns (normal[N,3], coh[N])."""
    eps = 1e-20
    p1 = a01 * a01 + a02 * a02 + a12 * a12
    q = (a00 + a11 + a22) / 3.0
    p2 = (a00 - q) ** 2 + (a11 - q) ** 2 + (a22 - q) ** 2 + 2.0 * p1
    p = torch.sqrt(torch.clamp(p2 / 6.0, min=eps))
    b00, b11, b22 = (a00 - q) / p, (a11 - q) / p, (a22 - q) / p
    b01, b02, b12 = a01 / p, a02 / p, a12 / p
    detB = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02)
    r = torch.clamp(detB / 2.0, -1.0, 1.0)
    phi = torch.arccos(r) / 3.0
    e1 = q + 2.0 * p * torch.cos(phi)                          # largest
    e3 = q + 2.0 * p * torch.cos(phi + 2.0 * float(np.pi) / 3.0)   # smallest
    e2 = 3.0 * q - e1 - e3                                     # middle
    diag = p1 < eps                                            # diagonal tensor → eigenvalues are the diagonal
    if bool(diag.any()):
        dmax = torch.maximum(torch.maximum(a00, a11), a22)
        dmin = torch.minimum(torch.minimum(a00, a11), a22)
        e1 = torch.where(diag, dmax, e1)
        e3 = torch.where(diag, dmin, e3)
        e2 = torch.where(diag, a00 + a11 + a22 - dmax - dmin, e2)
    a0, a1, a2 = a00 - e2, a11 - e2, a22 - e2                  # A - e2 I
    c0, c1, c2 = a00 - e3, a11 - e3, a22 - e3                  # A - e3 I  (off-diagonals shared)
    M00 = a0 * c0 + a01 * a01 + a02 * a02
    M10 = a01 * c0 + a1 * a01 + a12 * a02
    M20 = a02 * c0 + a12 * a01 + a2 * a02
    M01 = a0 * a01 + a01 * c1 + a02 * a12
    M11 = a01 * a01 + a1 * c1 + a12 * a12
    M21 = a02 * a01 + a12 * c1 + a2 * a12
    M02 = a0 * a02 + a01 * a12 + a02 * c2
    M12 = a01 * a02 + a1 * a12 + a12 * c2
    M22 = a02 * a02 + a12 * a12 + a2 * c2
    col0 = torch.stack([M00, M10, M20], -1)                   # columns of M are ∝ the λmax eigenvector
    col1 = torch.stack([M01, M11, M21], -1)
    col2 = torch.stack([M02, M12, M22], -1)
    n0 = (col0 * col0).sum(-1); n1 = (col1 * col1).sum(-1); n2 = (col2 * col2).sum(-1)
    v = torch.where(((n0 >= n1) & (n0 >= n2))[:, None], col0,
                    torch.where((n1 >= n2)[:, None], col1, col2))    # max-norm column (best-conditioned)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-20)
    coh = 1.0 - e3 / (e1 + 1e-12)
    return v, coh


def structure_tensor_frame_torch(
    block, sigma_grad: float = 1.0, sigma_tensor: float = 4.0, device=None, want_fibre: bool = True
) -> dict:
    """Per-voxel frame field on the GPU. Same outputs as :func:`structure_tensor_frame`. With
    ``want_fibre=False`` the (degenerate, unused for ∇φ) in-plane fibre is skipped and the normal+coherence
    come from the closed-form :func:`_sym3x3_normal_coh` instead of batched eigh — ~10× faster."""
    if not _HAS_TORCH:
        raise RuntimeError("torch not available")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        v = torch.as_tensor(np.ascontiguousarray(block, np.float32), device=device)[None, None]
        g0 = _kernel(sigma_grad, 0, device)
        g1 = _kernel(sigma_grad, 1, device)
        gz = _conv_axis(_conv_axis(_conv_axis(v, g1, 2), g0, 3), g0, 4)
        gy = _conv_axis(_conv_axis(_conv_axis(v, g0, 2), g1, 3), g0, 4)
        gx = _conv_axis(_conv_axis(_conv_axis(v, g0, 2), g0, 3), g1, 4)
        gt = _kernel(sigma_tensor, 0, device)

        def integ(a):
            return _conv_axis(_conv_axis(_conv_axis(a, gt, 2), gt, 3), gt, 4)[0, 0]

        jzz, jyy, jxx = integ(gz * gz), integ(gy * gy), integ(gx * gx)
        jzy, jzx, jyx = integ(gz * gy), integ(gz * gx), integ(gy * gx)
        del gz, gy, gx, v
        shp = jzz.shape
        n = jzz.numel()
        # Keep only the 6 unique component vectors resident (6·N·4B), and build the small (chunk,3,3)
        # J per chunk inside the eigh loop — NEVER materialise the full (N,3,3) tensor (9·N·4B ≈ 7GB on a
        # 1cm³ L1 window), which would blow an 8GB card. Peak stays ~6·N·4B + one chunk.
        jzz, jyy, jxx = jzz.reshape(-1), jyy.reshape(-1), jxx.reshape(-1)
        jzy, jzx, jyx = jzy.reshape(-1), jzx.reshape(-1), jyx.reshape(-1)

        fibre = np.empty((n, 3), np.float32) if want_fibre else None
        normal = np.empty((n, 3), np.float32)
        coh = np.empty(n, np.float32)
        chunk = 1_000_000 if want_fibre else 8_000_000        # closed-form is cheap → bigger chunks
        if want_fibre and jzz.is_cuda:
            # cuSOLVER's batched syev raises CUSOLVER_STATUS_INVALID_VALUE on 3×3 batches (torch 2.8/CUDA12.8
            # Blackwell/Ada); MAGMA handles them and is bundled in the cu124/cu128 wheels.
            try:
                torch.backends.cuda.preferred_linalg_library("magma")
            except Exception:
                pass
        for i in range(0, n, chunk):
            sl = slice(i, i + chunk)
            if want_fibre:
                jc = torch.stack(                              # (chunk,3,3), built per-chunk only
                    [torch.stack([jzz[sl], jzy[sl], jzx[sl]], dim=-1),
                     torch.stack([jzy[sl], jyy[sl], jyx[sl]], dim=-1),
                     torch.stack([jzx[sl], jyx[sl], jxx[sl]], dim=-1)], dim=-2)
                w, vec = torch.linalg.eigh(jc)                 # ascending eigenvalues
                fibre[sl] = vec[..., :, 0].cpu().numpy()       # smallest λ → along fibre
                normal[sl] = vec[..., :, 2].cpu().numpy()      # largest λ  → across sheet
                coh[sl] = (1.0 - w[..., 0] / (w[..., 2] + 1e-12)).cpu().numpy()
                del jc, w, vec
            else:                                              # closed-form: normal + coherence only, ~10× faster
                v, ch = _sym3x3_normal_coh(jzz[sl], jyy[sl], jxx[sl], jzy[sl], jzx[sl], jyx[sl], torch)
                normal[sl] = v.cpu().numpy()
                coh[sl] = ch.cpu().numpy()
                del v, ch
    if device == "cuda":
        torch.cuda.empty_cache()
    out = {"normal": normal.reshape(*shp, 3), "coherence": coh.reshape(*shp)}
    if want_fibre:
        out["fibre"] = fibre.reshape(*shp, 3)
    return out



"""Coherence/sheet-enhancing anisotropic diffusion (Weickert-style), 3-D, on the GPU.

A first-class substrate cleaner, orthogonal to the layer-potential: evolve the volume with a
diffusion tensor that lets intensity flow FREELY *within* the sheet plane and almost not at all
*across* the sheet normal. The result is a "tube-like" map — each sheet smoothed into a clean band,
the inter-sheet gaps widened and sharpened — which improves the normals, the layer potential, the
streamlines and (eventually) the joint model's input representation.

Diffusion tensor in closed form from the sheet normal n̂ (unit):
    D = I − (1−α)·n̂ n̂ᵀ          →  eigenvalue 1 along the two in-plane tangents, α (≪1) along n̂.
So the flux is  f = ∇u − (1−α)·n̂ (n̂·∇u)  — no full 3×3 tensor to store. Explicit update
u ← u + dt·div(f), dt ≤ 1/6 for 3-D stability. n̂ comes from the structure tensor; recompute it
every ``recompute_every`` steps (0 = fix it from the input) so the diffusion re-steers as edges
sharpen.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def _pad(u):
    return F.pad(u, (1, 1, 1, 1, 1, 1), mode="replicate")


def _grad(u):
    """Central-difference gradient of (D,H,W) tensor → (gz, gy, gx), replicate edges."""
    up = _pad(u[None, None])[0, 0]
    gz = (up[2:, 1:-1, 1:-1] - up[:-2, 1:-1, 1:-1]) * 0.5
    gy = (up[1:-1, 2:, 1:-1] - up[1:-1, :-2, 1:-1]) * 0.5
    gx = (up[1:-1, 1:-1, 2:] - up[1:-1, 1:-1, :-2]) * 0.5
    return gz, gy, gx


def _div(fz, fy, fx):
    """Central-difference divergence of a vector field, replicate edges."""
    fzp = _pad(fz[None, None])[0, 0]
    fyp = _pad(fy[None, None])[0, 0]
    fxp = _pad(fx[None, None])[0, 0]
    dz = (fzp[2:, 1:-1, 1:-1] - fzp[:-2, 1:-1, 1:-1]) * 0.5
    dy = (fyp[1:-1, 2:, 1:-1] - fyp[1:-1, :-2, 1:-1]) * 0.5
    dx = (fxp[1:-1, 1:-1, 2:] - fxp[1:-1, 1:-1, :-2]) * 0.5
    return dz + dy + dx


def coherence_enhancing_diffusion(
    block, iters: int = 10, dt: float = 0.12, alpha: float = 0.02,
    sigma_grad: float = 1.0, sigma_tensor: float = 4.0, recompute_every: int = 0,
    mode: str = "sheet", device=None,
) -> np.ndarray:
    """Anisotropic diffusion. Returns the cleaned block (float32 numpy, same shape). GPU when
    torch/CUDA present. Both modes are DESTRUCTIVE — use them only to derive a cleaner frame
    field, never as detection input.

    ``mode``:
      * ``"sheet"`` — diffuse in the sheet PLANE (⊥ normal): clean slabs, kills in-plane fibres.
        Flux f = ∇u − (1−α)·n̂(n̂·∇u).  For the φ / segmentation path.
      * ``"fibre"`` — diffuse ALONG the fibre (∥ smallest-eigenvector tangent): connects fibres,
        bridges delamination gaps, preserves them. Flux f = α·∇u + (1−α)·t̂(t̂·∇u). Run this
        BEFORE estimating the sheet normal to rescue broken/merging sheets.

    ``alpha`` is the small cross-direction diffusivity; ``recompute_every`` re-estimates the
    steering field every N steps (0 = once)."""
    if not _HAS_TORCH:
        raise RuntimeError("torch required for anisotropic diffusion")
    if mode not in ("sheet", "fibre"):
        raise ValueError(f"mode must be 'sheet' or 'fibre', got {mode!r}")
    from hercunet.labels.fields.structure_tensor_torch import structure_tensor_frame_torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    u = torch.as_tensor(np.ascontiguousarray(block, np.float32), device=device)
    vz = vy = vx = None
    key = "normal" if mode == "sheet" else "fibre"

    def refresh_dir(cur):
        ff = structure_tensor_frame_torch(cur.detach().cpu().numpy(), sigma_grad, sigma_tensor,
                                          device=device)
        v = torch.as_tensor(ff[key], device=device)      # (D,H,W,3) z,y,x
        return v[..., 0], v[..., 1], v[..., 2]

    with torch.no_grad():
        for it in range(iters):
            if vz is None or (recompute_every and it % recompute_every == 0):
                vz, vy, vx = refresh_dir(u)
            gz, gy, gx = _grad(u)
            vdotg = vz * gz + vy * gy + vx * gx
            if mode == "sheet":                          # diffuse ⊥ n̂ (in the sheet plane)
                k = 1.0 - alpha
                fz, fy, fx = gz - k * vz * vdotg, gy - k * vy * vdotg, gx - k * vx * vdotg
            else:                                        # diffuse ∥ t̂ (along the fibre)
                k = 1.0 - alpha
                fz, fy, fx = alpha * gz + k * vz * vdotg, alpha * gy + k * vy * vdotg, \
                    alpha * gx + k * vx * vdotg
            u = u + dt * _div(fz, fy, fx)
    out = u.detach().cpu().numpy().astype(np.float32)
    if device == "cuda":
        torch.cuda.empty_cache()
    return out

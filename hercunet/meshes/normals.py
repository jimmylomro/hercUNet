"""GPU per-mask DoG/gradient normals — shared mesh-prep geometry.

For sheet mesh fitting we need a per-voxel across-sheet **normal** at each detected-mask voxel. Computing it as the
gradient of a Gaussian-blurred mask is far smoother than per-point PCA, and doing it **per sheet** (each sheet's
mask blurred alone) stops touching sheets from contaminating each other's normals. At scale this must run on the
GPU next to inference — not a CPU scipy pass per window — so this is a torch implementation: a separable 3-D
Gaussian (three ``conv3d``) + finite-difference gradient, looped over the sheets present in the window.

Consume as::

    N = dog_normals_gpu(owner, gate, sigma=5.0)     # (D,H,W,3) unit normals, zero off-mask
    # per sheet: pts = mask voxels, normals = N[mask]  ->  hercunet.meshes.fit.fit_sheet_meshes(..., jac=0)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian1d(sigma: float, device, dtype=torch.float32):
    r = int(max(1, round(3.0 * sigma)))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum(), r


def _separable_blur(vol: torch.Tensor, k: torch.Tensor, r: int) -> torch.Tensor:
    """Separable 3-D Gaussian on a (1,1,D,H,W) volume via three axis-wise conv3d (reflect-padded)."""
    K = k.numel()
    for shape, pad in (((1, 1, K, 1, 1), (r, 0, 0)), ((1, 1, 1, K, 1), (0, r, 0)), ((1, 1, 1, 1, K), (0, 0, r))):
        vol = F.pad(vol, (pad[2], pad[2], pad[1], pad[1], pad[0], pad[0]), mode="replicate")
        vol = F.conv3d(vol, k.view(shape))
    return vol


@torch.no_grad()
def dog_normals_gpu(owner: torch.Tensor, gate: torch.Tensor, sigma: float = 5.0) -> torch.Tensor:
    """Per-voxel across-sheet unit normals from the detected masks, on the GPU.

    ``owner`` (D,H,W) long = per-voxel sheet id (argmax query); ``gate`` (D,H,W) bool = surface gate (keep). For each
    sheet present in ``owner[gate]``: blur its binary mask with a separable Gaussian (``sigma``), take the gradient
    (= across-sheet direction), normalise, and write it at that sheet's gated voxels. Returns (D,H,W,3) float, zero
    off-mask. Per-sheet so touching sheets don't bleed. ~a handful of conv3d per sheet — cheap next to the forward."""
    assert owner.ndim == 3 and gate.shape == owner.shape
    dev = owner.device
    k, r = _gaussian1d(sigma, dev)
    D, H, W = owner.shape
    N = torch.zeros(D, H, W, 3, device=dev)
    sheets = torch.unique(owner[gate])
    for sid in sheets.tolist():
        m = (owner == sid) & gate
        if int(m.sum()) == 0:
            continue
        mg = _separable_blur(m.float()[None, None], k, r)[0, 0]            # blurred mask (D,H,W)
        g = torch.gradient(mg)                                            # (gz, gy, gx)
        n = torch.stack(g, dim=-1)                                        # (D,H,W,3)
        n = n / (n.norm(dim=-1, keepdim=True) + 1e-9)
        N[m] = n[m]
    return N

"""Fast KNN-free sheet fit for HercuViT inference — fit each sheet's quad grid to a scalar FIELD by explicit
(autograd-free) gradient ascent.

Why not the herculabels slab fit (:func:`hercunet.meshes.fit.fit_sheet_meshes`)? That builds a per-vertex k-NN to
the mask point cloud every Adam step (a ``[verts × k × 3]`` tensor), which costs the same order as the whole network
forward — absurd for fitting a few-thousand-vertex grid. Here the data term is a precomputed per-sheet **field**: the
Gaussian-blurred binary mask, which peaks on the sheet's medial surface (and, being per-sheet, cannot be contaminated
by neighbours). Each step is then just ``grid_sample`` of the field gradient at the current vertices — O(verts), pure
GPU, no k-NN, no KDTree, no CPU round-trips, no autograd graph, no optimiser state.

The step moves each vertex ONLY along its local grid normal (toward the ridge — no in-plane drift) and applies a
Laplacian smoothing; a handful of iterations suffice. Validated against the slab fit: ~half-a-voxel vertex agreement
(negligible at an 8.6 µm voxel / 30 µm grid) at a fraction of the cost. herculabels' label-quality fit is untouched.

Returns the SAME ``{sheet_id: mesh}`` dict shape as :func:`fit_sheet_meshes` (``V``/``foot``/``nu``/``nv``/``verts``/
``tris``) so it is a drop-in for :func:`hercunet.meshes.extract.extract_window_sheets` and the .asv writer.
"""
from __future__ import annotations

import numpy as np

from .fit import grid_triangles, init_sheet_grid
from .normals import _gaussian1d, _separable_blur


def fit_sheets_field(owner, gate, *, voxel_um: float, sigma: float = 3.0, iters: int = 40, lr: float = 1.0,
                     lam: float = 0.3, step_um: float = 30.0, min_pts: int = 400, min_grid: int = 3,
                     device: str = "cuda") -> dict:
    """Fit every sheet in ``owner`` (D,H,W int ids) gated by ``gate`` (D,H,W bool). Drops sheets with < ``min_pts``
    gated voxels or a degenerate grid (< ``min_grid`` either axis, or < 3 footprint cells) — the field-fit analogue
    of the slab fit's min-points / failure drops. ``sigma`` is the field blur (voxels); ``iters``/``lr``/``lam`` the
    gradient-ascent steps / step size / Laplacian weight. Returns ``{sid: mesh}`` (verts in window-local z,y,x)."""
    import torch
    import torch.nn.functional as F
    dev = device
    owner_t = torch.as_tensor(owner, device=dev)
    gate_t = torch.as_tensor(gate, device=dev, dtype=torch.bool)
    owner_np = owner_t.cpu().numpy(); gate_np = gate_t.cpu().numpy()
    D, H, W = owner_np.shape
    step_px = float(step_um) / float(voxel_um)

    # --- per-sheet grid init + drops -------------------------------------------------------------------------------
    grids, masks = [], []
    for sid in [int(u) for u in np.unique(owner_np[gate_np])]:
        M = (owner_np == sid) & gate_np
        if int(M.sum()) < min_pts:
            continue
        V0, foot, (nu, nv) = init_sheet_grid(np.argwhere(M).astype(np.float32), step_px)
        if nu < min_grid or nv < min_grid or int(foot.sum()) < 3:          # degenerate grid -> failure drop
            continue
        grids.append((V0, foot, nu, nv, sid)); masks.append(M)
    if not grids:
        return {}
    N = len(grids)

    # --- per-sheet field (blurred mask) + its gradient, batched ----------------------------------------------------
    with torch.no_grad():
        Mst = torch.stack([torch.as_tensor(m, device=dev, dtype=torch.float32) for m in masks])[:, None]   # [N,1,D,H,W]
        kg, r = _gaussian1d(sigma, dev)
        fld = _separable_blur(Mst, kg, r)
        fld = fld / fld.amax(dim=(2, 3, 4), keepdim=True).clamp(min=1e-6)
        GF = torch.cat(torch.gradient(fld, dim=(2, 3, 4)), dim=1)          # [N,3,D,H,W] grad in (z,y,x)

        # --- flat verts, concatenated Laplacian graph, padded sample index + inverse -------------------------------
        V0_all, lc_l, ln_l = [], [], []
        voff = 0
        Vmax = max(nu * nv for _, _, nu, nv, _ in grids)
        pidx = np.zeros((N, Vmax), np.int64); padm = np.zeros((N, Vmax), np.float32)
        for n, (V0, foot, nu, nv, sid) in enumerate(grids):
            V0_all.append(V0.reshape(-1, 3))
            gi = np.arange(nu * nv).reshape(nu, nv) + voff
            lc_l.append(gi[1:-1, 1:-1].ravel())
            ln_l.append(np.stack([gi[2:, 1:-1].ravel(), gi[:-2, 1:-1].ravel(),
                                  gi[1:-1, 2:].ravel(), gi[1:-1, :-2].ravel()], 1))
            cnt = nu * nv
            pidx[n, :cnt] = np.arange(voff, voff + cnt); padm[n, :cnt] = 1.0; voff += cnt
        V = torch.tensor(np.concatenate(V0_all), device=dev)
        lc = torch.tensor(np.concatenate(lc_l), device=dev)
        ln = torch.tensor(np.concatenate(ln_l), device=dev)
        pidxt = torch.tensor(pidx, device=dev)
        pos = np.arange(N)[:, None] * Vmax + np.arange(Vmax)[None, :]
        inv = np.zeros(voff, np.int64); inv[pidx[padm > 0]] = pos[padm > 0]
        invt = torch.tensor(inv, device=dev)
        scale = torch.tensor([W - 1, H - 1, D - 1], device=dev, dtype=torch.float32)   # for (x,y,z) normalisation

        # --- explicit gradient-ascent loop (no autograd) ----------------------------------------------------------
        for _ in range(iters):
            Vp = V[pidxt]                                                  # [N,Vmax,3] (z,y,x)
            gn = torch.stack([2 * Vp[..., 2] / scale[0] - 1,              # x
                              2 * Vp[..., 1] / scale[1] - 1,              # y
                              2 * Vp[..., 0] / scale[2] - 1], -1)         # z
            g = F.grid_sample(GF, gn.view(N, Vmax, 1, 1, 3), align_corners=True,
                              padding_mode="border").view(N, 3, Vmax)
            gflat = g.permute(0, 2, 1).reshape(N * Vmax, 3)[invt]          # [voff,3] field gradient per vertex
            tu = V[ln[:, 0]] - V[ln[:, 1]]; tv = V[ln[:, 2]] - V[ln[:, 3]]
            nrm = torch.cross(tu, tv, dim=1)
            nrm = nrm / (nrm.norm(dim=1, keepdim=True) + 1e-9)            # local grid normal (interior)
            upd = torch.zeros_like(V)
            upd[lc] = lr * (gflat[lc] * nrm).sum(1, keepdim=True) * nrm + lam * (V[ln].mean(1) - V[lc])
            V = V + upd

        Vnp = V.cpu().numpy()

    from scipy.spatial import cKDTree
    out, voff = {}, 0
    for n, (V0, foot, nu, nv, sid) in enumerate(grids):
        Vf = Vnp[voff:voff + nu * nv].reshape(nu, nv, 3); fm = foot.reshape(-1)
        verts = Vf.reshape(-1, 3)[fm]                                           # (nvert,3) window-local z,y,x voxels
        # Fit quality vs the mask: nearest-vertex distance from each gated mask voxel to the mesh. fit_rms_um is the
        # RMS displacement (um); coverage is the fraction of mask voxels within one grid step of the mesh.
        pts = np.argwhere(masks[n]).astype(np.float32)
        d, _ = cKDTree(verts).query(pts, k=1)
        fit_rms_um = float(np.sqrt(np.mean(d * d))) * float(voxel_um)
        coverage = float(np.mean(d <= step_px))
        out[sid] = dict(V=Vf, foot=foot, nu=nu, nv=nv, verts=verts, tris=grid_triangles(foot),
                        fit_rms_um=fit_rms_um, coverage=coverage)
        voff += nu * nv
    return out

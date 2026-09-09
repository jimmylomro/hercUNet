"""Centre-surface mesh fitting for per-cluster sheets — the unsupervised pseudo-label GEOMETRY stage.

Runs AFTER streamlet clustering (``meshlets.streamlets3d.cluster_streamlets``). For each cluster we:

  1. build an ACCUMULATED SURFACE-SLAB potential that peaks at the sheet centre
     and BRIDGES within-sheet delamination holes. Within a cluster a density gap is a within-sheet hole to
     bridge, NOT an inter-sheet boundary, so we use a FIXED SYMMETRIC sigma_n (never the density-measured
     sigma_n, which collapses at delaminations and punches holes) + a generous in-plane band.
  2. fit a quadmesh to the CENTRE of that potential (``fit_sheet_mesh``), lasagna-style: Adam maximises the
     sampled slab at the vertices, regularised by Laplacian smoothness + even edge length. The mesh sits at the
     sheet's medial surface, fit to the POTENTIAL (never to raw CT / material).

Grid spacing is PHYSICAL (``step_um``), so vertex density is constant per unit surface area and the vertex
count scales with (projected) sheet area rather than a fixed cell count. Triangulation is IMPLICIT in the
regular grid + footprint mask — faces are reconstructable and need not be stored.

The fitted mesh is the sheet skeleton from which downstream we derive the winding field grad-phi (normal from
mesh faces, magnitude bump) and per-vertex width. Everything here is unsupervised and upstream of any UNet.

Pure numpy + scipy + (optional) torch — no plotting. Viz lives in the calling scripts.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


def _pick_device(device):
    if device is not None:
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def slab_logscore_batch(v, n, J, ss2, sn2p, sn2m):
    """Vectorised ``sheet_membership.surface_slab_logscore`` over a (M,k) neighbour axis.

    v chord [M,k,3], n normal [M,k,3], J normal-jacobian [M,k,3,3]; ss2 in-plane var, sn2p/sn2m across-normal
    var on the +/- side of the sheet. Curvature-corrected (Weingarten kappa) across-normal residual r. Returns
    the log slab score [M,k]. Mirrors the per-row kernel in sheet_membership.py so the accumulation matches the
    clustering slab exactly.
    """
    across = (v * n).sum(-1)                                        # (M,k)
    vip = v - across[..., None] * n
    inp2 = np.maximum((vip * vip).sum(-1), 0.0)
    Jv = np.einsum("mki,mkij->mkj", vip, J)
    kappa = -(vip * Jv).sum(-1) / np.maximum(inp2, 1e-9)
    r = (across - 0.5 * kappa * inp2) / np.sqrt(1.0 + kappa ** 2 * inp2)
    sn2 = np.where(across >= 0.0, sn2p, sn2m)
    return -(inp2 / ss2 + r ** 2 / sn2)


def init_sheet_grid(P, step_px):
    """Quad grid over the cluster's PCA (u,v) plane; init height = binned-mean across-normal coord (curved seed).

    Returns ``V0 [nu,nv,3]`` (world voxel z,y,x), ``foot [nu,nv]`` footprint (occupied cells, closed+hole-filled),
    ``(nu,nv)``. The height seed is refined freely in 3-D by the fit; the grid only provides connectivity.
    """
    mu = P.mean(0); Q = P - mu
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    e = Vt                                                          # e0,e1 in-plane; e2 across-normal
    uv = Q @ e[:2].T; wv = Q @ e[2]
    umin, umax = uv[:, 0].min(), uv[:, 0].max(); vmin, vmax = uv[:, 1].min(), uv[:, 1].max()
    gu = np.arange(umin, umax + step_px, step_px); gv = np.arange(vmin, vmax + step_px, step_px)
    nu, nv = len(gu), len(gv)
    ui = np.clip(((uv[:, 0] - umin) / step_px).astype(int), 0, nu - 1)
    vi = np.clip(((uv[:, 1] - vmin) / step_px).astype(int), 0, nv - 1)
    wsum = np.zeros((nu, nv)); cnt = np.zeros((nu, nv))
    np.add.at(wsum, (ui, vi), wv); np.add.at(cnt, (ui, vi), 1.0)
    occ = cnt > 0
    w0 = np.zeros((nu, nv)); w0[occ] = wsum[occ] / cnt[occ]
    if (~occ).any() and occ.any():
        ind = ndi.distance_transform_edt(~occ, return_distances=False, return_indices=True)
        w0 = w0[tuple(ind)]                                         # fill empty cells by nearest (seed only)
    foot = ndi.binary_fill_holes(ndi.binary_closing(occ, iterations=1))
    U, Vg = np.meshgrid(gu, gv, indexing="ij")
    world = mu[None, None, :] + U[..., None] * e[0] + Vg[..., None] * e[1] + w0[..., None] * e[2]
    return world.astype(np.float32), foot, (nu, nv)


def mesh_vertex_normals(Vg):
    """Unit surface normal at each grid vertex from the grid tangents (central differences of ∂u, ∂v, then
    cross product). Vg [nu,nv,3] fitted vertices. Returns [nu,nv,3]. Orientation is consistent across the grid
    (global sign is arbitrary but irrelevant to width, which is symmetric)."""
    tu = np.empty_like(Vg); tv = np.empty_like(Vg)
    tu[1:-1] = Vg[2:] - Vg[:-2]; tu[0] = Vg[1] - Vg[0]; tu[-1] = Vg[-1] - Vg[-2]
    tv[:, 1:-1] = Vg[:, 2:] - Vg[:, :-2]; tv[:, 0] = Vg[:, 1] - Vg[:, 0]; tv[:, -1] = Vg[:, -1] - Vg[:, -2]
    n = np.cross(tu.reshape(-1, 3), tv.reshape(-1, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    return n.reshape(Vg.shape)


def compute_mesh_width(mesh, anchors, voxel_um, *, r_in_um=60.0, t_max_um=150.0, lo=2.0, hi=98.0,
                       min_pts=6, max_per=2000, seed=0, anchor_cap=100000, device=None):
    """Per-vertex sheet WIDTH (µm) + recentred vertices, JOINTLY from ONE column gather. NO k-NN, NO kernel, NO
    σ_n, NO descale — anchors are gathered in the in-plane column by RADIUS (in-plane <= r_in_um, |across-normal|
    <= t_max_um) and the two faces are read DIRECTLY as the extreme percentiles of the column's across-normal
    coordinate d_n:
        WIDTH  = (p_hi - p_lo)·voxel_um     (p_lo/p_hi = the two faces; the interior — incl. a delamination gap —
                                             sits between them, so this runs THROUGH the delamination by construction)
        CENTRE = (p_hi + p_lo)/2            (the vertex is moved here — true mid-sheet, off-centre-safe)
    No convolution kernel means no additive σ_n floor (which a traced-slab decay + global rescale could not remove)
    and the real thin-vs-thick contrast is preserved. Radius gather on the CPU KD-tree over anchors subsampled to
    ``anchor_cap`` (uniform → d_n distribution unbiased). ``device`` is accepted for API symmetry but the readout
    is light CPU work done in the gather loop. < min_pts column anchors -> NaN.

    Returns (width_um [nvert], offset_um [nvert] signed centre shift, verts_centered [nvert,3])."""
    Vg = mesh["V"]; fm = mesh["foot"].reshape(-1)
    Nf = mesh_vertex_normals(Vg).reshape(-1, 3)[fm]
    Vf = Vg.reshape(-1, 3)[fm].astype(np.float64)
    A = np.ascontiguousarray(anchors, np.float64)
    if len(A) > anchor_cap:                                                  # subsample for a cheap gather (unbiased)
        A = A[np.random.default_rng(seed).integers(0, len(A), anchor_cap)]
    r_in = r_in_um / voxel_um; t_max = t_max_um / voxel_um
    tree = cKDTree(A)
    lists = tree.query_ball_point(Vf, float(np.hypot(r_in, t_max)), workers=-1)
    rng = np.random.default_rng(seed)
    width = np.full(len(Vf), np.nan, np.float32); offset = np.full(len(Vf), np.nan, np.float32)
    Vc = Vf.copy()
    for i, l in enumerate(lists):
        if not l:
            continue
        o = A[l] - Vf[i]; dn = o @ Nf[i]
        ip2 = np.maximum((o * o).sum(1) - dn * dn, 0.0)
        m = (ip2 <= r_in * r_in) & (np.abs(dn) <= t_max)
        if m.sum() < min_pts:
            continue
        d = dn[m]
        if d.size > max_per:
            d = d[rng.integers(0, d.size, max_per)]                          # unbiased subsample
        p_lo, p_hi = np.percentile(d, [lo, hi])                              # the two faces
        width[i] = (p_hi - p_lo) * voxel_um
        c = 0.5 * (p_lo + p_hi)
        offset[i] = c * voxel_um
        Vc[i] = Vf[i] + c * Nf[i]                                            # recentre to true mid-sheet
    return width, offset, Vc.astype(np.float32)


def grid_triangles(foot):
    """Triangles (each quad -> 2 tris) whose 3 corners are ALL in the footprint. Indices into the flat grid."""
    nu, nv = foot.shape
    idx = np.arange(nu * nv).reshape(nu, nv)
    a = idx[:-1, :-1]; b = idx[1:, :-1]; c = idx[:-1, 1:]; d = idx[1:, 1:]
    fa = foot[:-1, :-1]; fb = foot[1:, :-1]; fc = foot[:-1, 1:]; fd = foot[1:, 1:]
    t1 = np.stack([a, b, c], -1)[fa & fb & fc]
    t2 = np.stack([b, d, c], -1)[fb & fd & fc]
    return np.concatenate([t1, t2], 0)


def _slab_pot_torch(v, n, J, ss2, sn2):
    """Differentiable slab potential ``sum_k exp(logscore)`` at each vertex — the SAME kernel as
    ``slab_logscore_batch`` / ``sheet_membership.surface_slab_logscore``, but evaluated in torch at the
    continuous vertex positions (no rasterised volume). v [N,k,3] chord to the k nearest same-cluster anchors,
    n [N,k,3] anchor normals, J [N,k,3,3] anchor normal-jacobians; ss2 in-plane var, sn2 across-normal var
    (fixed symmetric). Returns pot [N] — high where the vertex sits on the sheet's mid-surface."""
    import torch
    across = (v * n).sum(-1)                                       # (N,k)
    vip = v - across.unsqueeze(-1) * n
    inp2 = (vip * vip).sum(-1).clamp(min=0.0)
    Jv = torch.einsum("nki,nkij->nkj", vip, J)
    kappa = -(vip * Jv).sum(-1) / inp2.clamp(min=1e-9)
    r = (across - 0.5 * kappa * inp2) / torch.sqrt(1.0 + kappa ** 2 * inp2)
    logS = -(inp2 / ss2 + r * r / sn2)
    return torch.exp(logS.clamp(max=0.0)).sum(1)                   # (N,)


def fit_sheet_mesh(P, normals, jac, voxel_um, *, step_px, sigma_s_um=180.0, sigma_n_um=25.0, scale=2.0,
                   k=64, iters=250, lr=0.5, lam_smooth=2.0, lam_step=1.0, knn_every=50, device=None):
    """On-the-fly Adam fit of ONE sheet's quadmesh to the CENTRE of the slab potential, computed DIRECTLY from
    the cluster points (no materialised volume, no grid_sample, no blur).

    loss = -<slab potential at vertices> (footprint-masked, per-cluster normalised) + lam_smooth*Laplacian^2
    + lam_step*(edge-len error)^2. Each vertex queries its ``k`` nearest same-cluster anchors (KDTree built once
    on the fixed anchors; re-queried every ``knn_every`` iters as the mesh moves) and the analytic slab is
    differentiated w.r.t. the vertex position. Returns the mesh dict:
      V/V0 [nu,nv,3] fitted/init verts (voxel z,y,x); foot [nu,nv]; verts [nvert,3] footprint-only; tris; nu; nv.
    """
    import torch
    dev = _pick_device(device)
    ss2 = 2.0 * (sigma_s_um / voxel_um) ** 2
    sn2 = 2.0 * (scale * sigma_n_um / voxel_um) ** 2               # fixed symmetric across-normal var
    V0, foot, (nu, nv) = init_sheet_grid(P, step_px)
    A = np.ascontiguousarray(P, np.float64)
    tree = cKDTree(A)
    An = torch.tensor(A, dtype=torch.float32, device=dev)
    nn = torch.tensor(np.asarray(normals), dtype=torch.float32, device=dev)
    JJ = torch.tensor(np.asarray(jac), dtype=torch.float32, device=dev)
    V = torch.tensor(V0.reshape(-1, 3), dtype=torch.float32, device=dev, requires_grad=True)
    footm = torch.tensor(foot.reshape(-1), device=dev)
    opt = torch.optim.Adam([V], lr=lr)
    kq = min(k, len(A))
    idx = None; potscale = None
    for it in range(iters):
        if it % knn_every == 0:
            _, ii = tree.query(V.detach().cpu().numpy(), k=kq, workers=-1)
            if ii.ndim == 1:
                ii = ii[:, None]
            idx = torch.tensor(ii, dtype=torch.long, device=dev)
        opt.zero_grad()
        v = V[:, None, :] - An[idx]                                # (N,k,3)
        pot = _slab_pot_torch(v, nn[idx], JJ[idx], ss2, sn2)
        if potscale is None:
            potscale = pot.detach().max().clamp(min=1e-6)          # per-cluster loss scale (like acc.max())
        Vg = V.reshape(nu, nv, 3)
        data = -((pot / potscale) * footm).sum() / footm.sum().clamp(min=1)
        lap = (Vg[2:, 1:-1] + Vg[:-2, 1:-1] + Vg[1:-1, 2:] + Vg[1:-1, :-2] - 4 * Vg[1:-1, 1:-1])
        smooth = (lap ** 2).sum(-1).mean()
        du = (Vg[1:, :] - Vg[:-1, :]).norm(dim=-1); dv = (Vg[:, 1:] - Vg[:, :-1]).norm(dim=-1)
        stepl = (((du - step_px) / step_px) ** 2).mean() + (((dv - step_px) / step_px) ** 2).mean()
        loss = data + lam_smooth * smooth + lam_step * stepl
        loss.backward(); opt.step()
    Vf = V.detach().cpu().numpy().reshape(nu, nv, 3)
    fm = foot.reshape(-1)
    return dict(V=Vf, V0=V0, foot=foot, nu=nu, nv=nv,
                verts=Vf.reshape(-1, 3)[fm], tris=grid_triangles(foot))


def _fit_all_gpu(items, *, step_px, ss2, sn2, k, iters, lr, lam_smooth, lam_step, knn_every, dev):
    """Batched on-the-fly fit of ALL clusters in ONE Adam optimisation (GPU). Vertices of every sheet form one
    global tensor; a concatenated edge/Laplacian graph carries the per-cluster regularisers; per-cluster KDTrees
    (built once) map each vertex to its own cluster's k nearest anchors. Collapses B sequential fits into one."""
    import torch
    A_l, n_l, J_l, V0_l, foot_l, edges, lap_c, lap_n = [], [], [], [], [], [], [], []
    trees, vslices, aoffs = [], [], []
    voff = aoff = 0
    for m in items:
        P = np.ascontiguousarray(m["P"], np.float64); nu, nv = m["nu"], m["nv"]
        A_l.append(P); n_l.append(np.asarray(m["normals"])); J_l.append(np.asarray(m["jac"]))
        V0_l.append(m["V0"].reshape(-1, 3)); foot_l.append(m["foot"].reshape(-1))
        gi = np.arange(nu * nv).reshape(nu, nv) + voff
        edges.append(np.stack([gi[:-1, :].ravel(), gi[1:, :].ravel()], 1))
        edges.append(np.stack([gi[:, :-1].ravel(), gi[:, 1:].ravel()], 1))
        lap_c.append(gi[1:-1, 1:-1].ravel())
        lap_n.append(np.stack([gi[2:, 1:-1].ravel(), gi[:-2, 1:-1].ravel(),
                               gi[1:-1, 2:].ravel(), gi[1:-1, :-2].ravel()], 1))
        trees.append(cKDTree(P)); vslices.append((voff, voff + nu * nv)); aoffs.append(aoff)
        voff += nu * nv; aoff += len(P)
    An = torch.tensor(np.concatenate(A_l), dtype=torch.float32, device=dev)
    nn = torch.tensor(np.concatenate(n_l), dtype=torch.float32, device=dev)
    JJ = torch.tensor(np.concatenate(J_l), dtype=torch.float32, device=dev)
    V = torch.tensor(np.concatenate(V0_l), dtype=torch.float32, device=dev, requires_grad=True)
    footm = torch.tensor(np.concatenate(foot_l), device=dev)
    edg = torch.tensor(np.concatenate(edges), dtype=torch.long, device=dev)
    lc = torch.tensor(np.concatenate(lap_c), dtype=torch.long, device=dev)
    ln = torch.tensor(np.concatenate(lap_n), dtype=torch.long, device=dev)
    kq = min([k] + [len(m["P"]) for m in items])
    opt = torch.optim.Adam([V], lr=lr)
    idxt = None; potscale = None
    idx_np = np.empty((voff, kq), np.int64)
    for it in range(iters):
        if it % knn_every == 0:
            Vnp = V.detach().cpu().numpy()
            for tr, (vs, ve), ao in zip(trees, vslices, aoffs):
                _, ii = tr.query(Vnp[vs:ve], k=kq, workers=-1)
                idx_np[vs:ve] = (ii[:, None] if ii.ndim == 1 else ii) + ao
            idxt = torch.tensor(idx_np, dtype=torch.long, device=dev)
        opt.zero_grad()
        v = V[:, None, :] - An[idxt]
        pot = _slab_pot_torch(v, nn[idxt], JJ[idxt], ss2, sn2)
        if potscale is None:
            potscale = torch.ones(voff, device=dev)
            for vs, ve in vslices:
                potscale[vs:ve] = pot[vs:ve].detach().max().clamp(min=1e-6)   # per-cluster normalisation
        data = -((pot / potscale) * footm).sum() / footm.sum().clamp(min=1)
        lap = V[ln].sum(1) - 4.0 * V[lc]
        smooth = (lap ** 2).sum(-1).mean()
        d = (V[edg[:, 0]] - V[edg[:, 1]]).norm(dim=-1)
        stepl = (((d - step_px) / step_px) ** 2).mean()
        loss = data + lam_smooth * smooth + lam_step * stepl
        loss.backward(); opt.step()
    Vnp = V.detach().cpu().numpy()
    out = {}
    for m, (vs, ve) in zip(items, vslices):
        Vf = Vnp[vs:ve].reshape(m["nu"], m["nv"], 3); fm = m["foot"].reshape(-1)
        out[m["c"]] = dict(V=Vf, V0=m["V0"], foot=m["foot"], nu=m["nu"], nv=m["nv"],
                           verts=Vf.reshape(-1, 3)[fm], tris=grid_triangles(m["foot"]))
    return out


def mesh_vertex_jacobian(Vg):
    """Per-vertex normal Jacobian J = ∂n/∂x (3×3, the shape operator embedded in 3-D), from the grid: fit the
    normal's change against the surface tangents, dN·T = [∂N/∂u, ∂N/∂v], J = [∂N/∂u,∂N/∂v]·pinv([t_u,t_v]). This
    is the SAME curvature object ``slab_logscore_batch`` expects, but read off the clean mesh (not the noisy
    structure tensor). Vg [nu,nv,3] → [nu,nv,3,3]."""
    N = mesh_vertex_normals(Vg)
    tu = np.empty_like(Vg); tv = np.empty_like(Vg); nu_ = np.empty_like(Vg); nv_ = np.empty_like(Vg)
    tu[1:-1] = Vg[2:] - Vg[:-2]; tu[0] = Vg[1] - Vg[0]; tu[-1] = Vg[-1] - Vg[-2]
    tv[:, 1:-1] = Vg[:, 2:] - Vg[:, :-2]; tv[:, 0] = Vg[:, 1] - Vg[:, 0]; tv[:, -1] = Vg[:, -1] - Vg[:, -2]
    nu_[1:-1] = N[2:] - N[:-2]; nu_[0] = N[1] - N[0]; nu_[-1] = N[-1] - N[-2]
    nv_[:, 1:-1] = N[:, 2:] - N[:, :-2]; nv_[:, 0] = N[:, 1] - N[:, 0]; nv_[:, -1] = N[:, -1] - N[:, -2]
    guu = (tu * tu).sum(-1); guv = (tu * tv).sum(-1); gvv = (tv * tv).sum(-1)
    det = guu * gvv - guv * guv
    det = np.where(np.abs(det) < 1e-9, 1e-9, det)
    a = gvv / det; b = -guv / det; c = guu / det                     # inverse of the 2×2 metric
    pinv0 = a[..., None] * tu + b[..., None] * tv                     # rows of pinv([t_u,t_v]) (each [nu,nv,3])
    pinv1 = b[..., None] * tu + c[..., None] * tv
    return (np.einsum("...i,...j->...ij", nu_, pinv0)
            + np.einsum("...i,...j->...ij", nv_, pinv1)).astype(np.float32)


def _dense_medials(mesh, voxel_um, upsample):
    """Sub-voxel medial samples for one sheet by bilinearly upsampling the fitted (u,v) grid — so the nearest-
    medial lookup approximates the nearest SURFACE point, not the nearest 30µm vertex (which facets the bump).
    Returns (med [m,3], nrm [m,3], halfw_px [m])."""
    Vg = mesh["V"]; foot = mesh["foot"]; nu, nv = mesh["nu"], mesh["nv"]
    wgrid = np.full(nu * nv, np.nan, np.float32); wgrid[foot.reshape(-1)] = mesh["width"]
    wgrid = wgrid.reshape(nu, nv)
    good = np.isfinite(wgrid)
    wgrid = np.where(good, wgrid, np.nanmedian(wgrid[good]) if good.any() else 30.0)   # fill holes before zoom
    if upsample > 1:
        Vg = ndi.zoom(Vg, (upsample, upsample, 1), order=1)
        wgrid = ndi.zoom(wgrid, (upsample, upsample), order=1)
        footz = ndi.zoom(foot.astype(np.float32), (upsample, upsample), order=1) > 0.5
    else:
        footz = foot
    Nf = mesh_vertex_normals(Vg).reshape(-1, 3)
    J = mesh_vertex_jacobian(Vg).reshape(-1, 3, 3)
    fm = footz.reshape(-1)
    return Vg.reshape(-1, 3)[fm], Nf[fm], (0.5 * wgrid.reshape(-1)[fm] / voxel_um), J[fm]


def _slab_bump_torch(cand, med, nrm, jac, halfw, L, idx, own_lab, sn_cap, ss2, sfw, chunk=40000):
    """GPU port of ``build_gradphi``'s per-voxel slab accumulation — the SAME curvature-corrected kernel as
    ``slab_logscore_batch`` / ``_slab_pot_torch`` (float32 on CUDA). KDTree/Voronoi stay on CPU; only this
    O(Ncand·kk·3×3) inner loop moves to the GPU (the ~75s single-window bottleneck). Returns bump [Ncand]."""
    import torch
    dev = "cuda"
    med_t = torch.as_tensor(med, dtype=torch.float32, device=dev)
    nrm_t = torch.as_tensor(nrm, dtype=torch.float32, device=dev)
    jac_t = torch.as_tensor(jac, dtype=torch.float32, device=dev)
    halfw_t = torch.as_tensor(halfw, dtype=torch.float32, device=dev)
    L_t = torch.as_tensor(L, device=dev)
    cand_t = torch.as_tensor(cand, dtype=torch.float32, device=dev)
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=dev)
    own_t = torch.as_tensor(own_lab, device=dev)
    cap_t = torch.as_tensor(sn_cap, dtype=torch.float32, device=dev)
    out = np.empty(len(cand), np.float32)
    for s in range(0, len(cand), chunk):
        e = min(s + chunk, len(cand))
        ix = idx_t[s:e]
        same = (L_t[ix] == own_t[s:e, None])
        V = cand_t[s:e, None, :] - med_t[ix]
        n = nrm_t[ix]; J = jac_t[ix]
        across = (V * n).sum(-1)
        vip = V - across.unsqueeze(-1) * n
        inp2 = (vip * vip).sum(-1).clamp(min=0.0)
        Jv = torch.einsum("cki,ckij->ckj", vip, J)
        kappa = -(vip * Jv).sum(-1) / inp2.clamp(min=1e-9)
        r = (across - 0.5 * kappa * inp2) / torch.sqrt(1.0 + kappa ** 2 * inp2)
        sn = torch.minimum(sfw * halfw_t[ix], cap_t[s:e, None]).clamp(min=0.5)
        sn2v = 2.0 * sn * sn
        logS = -(inp2 / ss2 + r * r / sn2v)
        out[s:e] = (torch.exp(logS.clamp(min=-60.0, max=0.0)) * same).sum(1).float().cpu().numpy()
    return out


def build_gradphi(meshes, shape, voxel_um, *, mode="slab", sigma_frac=0.65, width_frac=0.5, power=3.0,
                  valley_frac=0.7, in_plane_um=40.0, reach_mult=2.5, kk=48, upsample=4, chunk=100000, min_mag=0.05,
                  use_gpu=None, device=None):
    """Dense ∇φ field from the fitted sheet meshes + per-vertex widths (the oracle emission).

    Every mesh footprint vertex is a medial point with a clean normal and a half-width. Over ALL clusters:
      - NEAREST-MEDIAL VORONOI (a voxel belongs to the sheet of its nearest medial) → a sheet's support is
        clipped to its own cell, so ∇φ CANNOT cross into another sheet; the boundary is the midplane.
      - |∇φ| = super-Gaussian exp(-(|d|/σ)^power) of the across-normal distance d to the owner medial's plane,
        peaked at the sheet CENTRE. σ = min(sigma_frac·(width_frac·half_width), valley_frac·½·spacing) with
        spacing = distance to the nearest DIFFERENT-sheet medial. ``width_frac`` (default 0.5) intentionally
        builds the bump over the INNER half of the sheet, NOT its full physical width: a full-width slab decays
        slowly and gets ABRUPTLY chopped where it runs into the neighbour's cell (the valley cap binds), leaving
        a hard edge; a half-width bump decays to ~0 on its own before the midplane, so the cap is non-binding
        and |∇φ| is smooth. Width is still encoded — σ stays PROPORTIONAL to width (thicker sheet → wider bump)
        and the physical width lives on its own channel/artifact. We do not need to paint the full width.
      - direction = the owner medial's mesh normal.

    Also returns the geometric INTERSECTION signal for the confidence channel: the two sheet envelopes overlap
    where ``spacing < half_own + half_other``; ``overlap`` ∈ [0,1] is that degree (1 = fully intersecting). It is
    soft-OR'd with the embedding uncertainty downstream — never modulates |∇φ|.

    Returns (mag [Z,H,W], lab3 [Z,H,W] int sheet id / -1, nrm3 [3,Z,H,W] float16, overlap [Z,H,W] float32)."""
    Z, H, W = shape
    med, nrm, halfw, jac, L = [], [], [], [], []
    for c, m in meshes.items():
        dm, dn, dhw, dj = _dense_medials(m, voxel_um, upsample)        # sub-voxel medials → smooth (no faceting)
        med.append(dm); nrm.append(dn); halfw.append(dhw); jac.append(dj)
        L.append(np.full(len(dm), c, np.int64))
    med = np.concatenate(med).astype(np.float64); nrm = np.concatenate(nrm).astype(np.float64)
    jac = np.concatenate(jac).astype(np.float64)
    halfw = np.maximum(np.concatenate(halfw), 0.5); L = np.concatenate(L)

    tree = cKDTree(med)
    occ = np.zeros(shape, bool)
    mi = np.clip(np.round(med).astype(int), 0, [Z - 1, H - 1, W - 1])
    occ[mi[:, 0], mi[:, 1], mi[:, 2]] = True
    reach = max(1, int(np.ceil(reach_mult * float(np.percentile(halfw, 90)))))
    cand = np.argwhere(ndi.binary_dilation(occ, iterations=reach)).astype(np.float64)

    dist, idx = tree.query(cand, k=min(kk, len(med)), workers=-1)
    if idx.ndim == 1:
        idx = idx[:, None]; dist = dist[:, None]
    lab_k = L[idx]; own = idx[:, 0]; own_lab = lab_k[:, 0]
    diff = lab_k != own_lab[:, None]; has = diff.any(1)
    fd = np.where(has, diff.argmax(1), idx.shape[1] - 1)
    other = idx[np.arange(len(cand)), fd]                              # nearest different-sheet medial
    spacing = np.where(has, dist[np.arange(len(cand)), fd], np.inf)

    env = halfw[own] + np.where(has, halfw[other], 0.0)               # sum of the two half-widths
    overlap = np.where(has, np.clip((env - spacing) / np.maximum(env, 1e-6), 0.0, 1.0), 0.0).astype(np.float32)
    sn_cap = valley_frac * 0.5 * spacing                             # σ cap → guaranteed valley in the gap

    if mode == "supergauss":                                          # distance to the owner medial's plane
        v = cand - med[own]
        d = np.abs(np.einsum("mi,mi->m", v, nrm[own]))
        sigma = np.maximum(np.minimum(sigma_frac * width_frac * halfw[own], sn_cap), 0.5)
        bump = np.exp(-(d / sigma) ** power).astype(np.float32)
    else:                                                            # "slab": the SAME surface-slab kernel
        ss2 = 2.0 * (in_plane_um / voxel_um) ** 2                     # (slab_logscore_batch, curvature-corrected)
        dev = _pick_device(device) if use_gpu is None else ("cuda" if use_gpu else "cpu")
        if dev == "cuda":                                            # GPU port of the per-voxel slab gather
            bump = _slab_bump_torch(cand, med, nrm, jac, halfw, L, idx, own_lab, sn_cap, ss2,
                                    sigma_frac * width_frac)
        else:
            bump = np.empty(len(cand), np.float32)                    # accumulated over the sheet's OWN dense medials
            for s in range(0, len(cand), chunk):
                e = min(s + chunk, len(cand))
                ix = idx[s:e]; same = (L[ix] == own_lab[s:e, None])
                V = cand[s:e, None, :] - med[ix]                      # (cb,kk,3)
                sn = np.maximum(np.minimum(sigma_frac * width_frac * halfw[ix], sn_cap[s:e, None]), 0.5)
                sn2v = 2.0 * sn * sn                                  # across-normal variance from HALF the WIDTH
                logS = slab_logscore_batch(V, nrm[ix], jac[ix], ss2, sn2v, sn2v)  # curvature-corrected slab
                contrib = np.exp(np.clip(logS, -60.0, 0.0)) * same
                bump[s:e] = contrib.sum(1).astype(np.float32)         # accumulate (raw slab)
        bump /= max(float(np.percentile(bump[bump > 0], 99)) if (bump > 0).any() else 1.0, 1e-6)
        bump = np.clip(bump, 0.0, 1.0)

    on = bump >= min_mag                                              # a voxel is COVERED only where the slab is real
    mag = np.zeros(shape, np.float32); lab3 = np.full(shape, -1, np.int32)
    nrm3 = np.zeros((3,) + shape, np.float16); ovl = np.zeros(shape, np.float32)
    ci = cand.astype(int); cz, cy, cx = ci[:, 0], ci[:, 1], ci[:, 2]
    mag[cz, cy, cx] = bump                                            # magnitude kept everywhere (≈0 in the treads)
    lab3[cz[on], cy[on], cx[on]] = own_lab[on]                        # ownership/colour = the slab extent, not the band
    ovl[cz[on], cy[on], cx[on]] = overlap[on]
    no = nrm[own]
    nrm3[0, cz[on], cy[on], cx[on]] = no[on, 0]
    nrm3[1, cz[on], cy[on], cx[on]] = no[on, 1]
    nrm3[2, cz[on], cy[on], cx[on]] = no[on, 2]
    return mag, lab3, nrm3, ovl


def intersection_confidence(overlap, lab3, *, smooth_px=6.0):
    """REGIONAL confidence from the sheet-INTERSECTION signal (mesh-derived): footprint-aware Gaussian-smooth the
    per-voxel ``overlap`` over the covered region and return ``conf = 1 − smoothed_overlap`` ∈ [0,1]. Low where
    sheet envelopes intersect. This is a SEPARATE channel — it never modulates |∇φ| — and is meant to be soft-OR'd
    with the embedding-derived confidence (co-membership + skinnydip dip-seam; see scripts/gradphi_viz.py). Also
    returns ``cover`` (the sheet mask) so callers can grey-out the air/tread regions."""
    cover = lab3 >= 0
    cf = cover.astype(np.float32)
    ov_s = ndi.gaussian_filter(overlap, smooth_px) / (ndi.gaussian_filter(cf, smooth_px) + 1e-6)
    return np.clip(1.0 - ov_s, 0.0, 1.0).astype(np.float32), cover


# --- Embedding-derived confidence (moved from scripts/gradphi_viz.py) -------------------------------------------
# The clustering's trust field: low where the embedding says the sheet assignment is ambiguous. Soft-OR'd (as
# uncertainties) with the mesh-derived intersection_confidence — combined conf = product of the conf fields.


def build_uncertainty(prob, glosh, comemb, seam, use_comembership=True, use_dipseam=True,
                      use_membership=False, use_glosh=False):
    """Soft-OR of the active per-meshlet signals: uncertainty = 1 − Π(1 − sᵢ). Any one doubt drops confidence.
    Default = the 'sharp-2' set (co-membership + skinnydip dip-seam), the promising one."""
    sig = []
    if use_comembership:
        sig.append(np.clip(comemb, 0, 1))
    if use_dipseam:
        sig.append(np.clip(seam, 0, 1))
    if use_membership:
        sig.append(np.clip(1.0 - prob, 0, 1))
    if use_glosh:
        sig.append(np.clip(glosh, 0, 1))
    if not sig:
        return np.zeros(len(prob), np.float32)
    keep = np.ones(len(prob), np.float32)
    for s in sig:
        keep *= (1.0 - s)
    return (1.0 - keep).astype(np.float32)


def confidence_volume(pts, u_point, shape, smooth_px=6.0):
    """REGIONAL (spatial, NOT per-sheet) embedding confidence: splat per-point uncertainty into 3-D, Gaussian-
    smooth a weighted average → confidence = 1 − smoothed uncertainty. Low-confidence regions (bridges, seams,
    boundaries) span space wherever the uncertain meshlets are, owned by no single sheet. Returns (conf, cover)."""
    Z, H, W = shape
    z = np.clip(np.round(pts[:, 0]).astype(int), 0, Z - 1)
    y = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
    x = np.clip(np.round(pts[:, 2]).astype(int), 0, W - 1)
    usum = np.zeros(shape, np.float32); csum = np.zeros(shape, np.float32)
    np.add.at(usum, (z, y, x), u_point); np.add.at(csum, (z, y, x), 1.0)
    us = ndi.gaussian_filter(usum, smooth_px); cs = ndi.gaussian_filter(csum, smooth_px)
    conf = 1.0 - us / (cs + 1e-6)
    cover = cs > (0.02 * cs.max() + 1e-9)
    return conf.astype(np.float32), cover


def combine_confidence(*conf_fields):
    """Soft-OR (as UNCERTAINTIES) of two-or-more REGIONAL confidence fields → one combined confidence.

    Each input is a confidence in [0,1] (1 = certain). We OR the DOUBTS: uᵢ = 1 − confᵢ, combined uncertainty
    = 1 − Π(1 − uᵢ), so ANY one field's doubt drops the combined confidence — the SAME 'any one doubt' rule
    ``build_uncertainty`` applies across per-meshlet signals, here applied across the confidence CHANNELS
    (mesh-derived ``intersection_confidence`` ⊕ embedding-derived 'sharp-2' ``confidence_volume``).
    Algebraically the soft-OR of the doubts is just the PRODUCT of the confidences: combined = Π confᵢ.
    This is a SEPARATE channel and NEVER modulates |∇φ|."""
    if not conf_fields:
        raise ValueError("combine_confidence needs at least one field")
    out = np.ones_like(np.asarray(conf_fields[0], np.float32))
    for c in conf_fields:
        out = out * np.clip(np.asarray(c, np.float32), 0.0, 1.0)       # Π confᵢ  ==  1 − softOR(uᵢ)
    return out.astype(np.float32)


def mesh_field_congruence(meshes, ff_normal, ff_coherence, *, scale=(1.0, 1.0, 1.0), offset=(0.0, 0.0, 0.0),
                          coh_gate=0.05):
    """QUALITY score: how well each sheet MESH normal agrees with the structure-tensor FIBER-FIELD normal at its
    vertices — ``congruence = |n_mesh · n_field|`` (both across-sheet; sign-free), coherence-weighted so unreliable
    (air/degenerate) field regions don't count. A collapsed/wrong sheet is geometrically "off" → its normal
    disagrees with the reliable field → low score. Returns ``(window_score, per_sheet)`` where per_sheet[c] =
    {score, n_verts, coh}; window = coherence-weighted mean over ALL vertices (bigger sheets dominate).

    ``ff_normal`` (Z,Y,X,3) / ``ff_coherence`` (Z,Y,X) are the field arrays; vertices are mapped to field voxels by
    ``field_idx = round(vert*scale + offset)`` per axis — identity for a full-res field sampled in the SAME window
    frame (the worker case, no re-read), or (1/r, origin) for a coarse field (the post-hoc filter)."""
    N = np.asarray(ff_normal, np.float32); C = np.asarray(ff_coherence, np.float32)
    Z, Y, X = N.shape[:3]
    (sz, sy, sx), (oz, oy, ox) = scale, offset
    per_sheet = {}
    tot_w = 0.0; tot_cw = 0.0
    for c, m in meshes.items():
        verts = np.asarray(m["verts"], np.float32)
        nm = mesh_vertex_normals(np.asarray(m["V"], np.float32))[np.asarray(m["foot"], bool)]  # grid normals → verts
        iz = np.clip(np.round(verts[:, 0] * sz + oz).astype(int), 0, Z - 1)
        iy = np.clip(np.round(verts[:, 1] * sy + oy).astype(int), 0, Y - 1)
        ix = np.clip(np.round(verts[:, 2] * sx + ox).astype(int), 0, X - 1)
        nf = N[iz, iy, ix]; cv = C[iz, iy, ix]
        nf /= (np.linalg.norm(nf, axis=-1, keepdims=True) + 1e-9)
        cong = np.abs(np.sum(nm * nf, axis=1))
        w = np.clip(cv, 0.0, 1.0) * (cv > coh_gate)
        sw = float(w.sum())
        sc = float((cong * w).sum() / sw) if sw > 1e-6 else float(cong.mean())
        per_sheet[int(c)] = dict(score=sc, n_verts=int(len(verts)), coh=float(cv.mean()))
        tot_w += sw; tot_cw += float((cong * w).sum())
    return (float(tot_cw / tot_w) if tot_w > 1e-6 else 0.0), per_sheet


def fit_sheet_meshes(pts, plab, bced, ff, voxel_um, *, normals=None, jac=None, step_um=30.0, min_points=200,
                     sigma_s_um=180.0, sigma_n_um=25.0, scale=2.0, k=64, iters=250, lr=0.5, lam_smooth=2.0,
                     lam_step=1.0, knn_every=50, device=None, batch=None, n_jobs=1, width=True, verbose=True):
    """Fit a centre-surface mesh for every cluster in a clustered streamlet point cloud (on-the-fly potential).

    pts [N,3] voxel coords, plab [N] per-point cluster label (-1 = noise), ff frame field (for per-point normals
    when ``normals``/``jac`` not supplied), voxel_um. ``bced`` is accepted for API stability but not used by the
    fit (the on-the-fly potential needs only the points + frames). ``batch`` (default: True on cuda) runs one
    batched Adam over all clusters on GPU; otherwise clusters are fit with a thread pool of ``n_jobs`` workers
    (the KDTree / torch-CPU ops release the GIL). ``width`` computes per-vertex sheet width from the cluster's
    own points (``compute_mesh_width``).

    Returns ``{cluster_id: mesh_dict}`` — each ``fit_sheet_mesh`` output plus ``n_points``, ``width`` [nvert] µm
    (or None) and ``offset`` [nvert] µm (centering QC). Vertices are in voxel (z,y,x) coords.
    """
    from hercunet.labels.selection.membership import normal_jacobian_batch
    pts = np.asarray(pts, np.float32); plab = np.asarray(plab)
    if normals is None or jac is None:
        nf = ff["normal"] / (np.linalg.norm(ff["normal"], axis=-1, keepdims=True) + 1e-9)
        Z, Y, X = nf.shape[:3]
        pi = np.round(pts).astype(int)
        pi[:, 0] = pi[:, 0].clip(0, Z - 1); pi[:, 1] = pi[:, 1].clip(0, Y - 1); pi[:, 2] = pi[:, 2].clip(0, X - 1)
        normals = nf[pi[:, 0], pi[:, 1], pi[:, 2]]
        jac = normal_jacobian_batch(nf, pts, normals)
    normals = np.asarray(normals, np.float32); jac = np.asarray(jac, np.float32)
    step_px = step_um / voxel_um
    dev = _pick_device(device)
    if batch is None:
        batch = dev == "cuda"

    # collect the eligible clusters + their point/frame slices
    items = []
    for c in sorted(int(x) for x in np.unique(plab) if x >= 0):
        sel = plab == c
        P = pts[sel].astype(np.float64)
        if len(P) < min_points:
            if verbose:
                print(f"  [mesh] cluster {c}: {len(P)} pts < {min_points}, skip", flush=True)
            continue
        V0, foot, (nu, nv) = init_sheet_grid(P, step_px)
        items.append(dict(c=c, P=P, normals=normals[sel], jac=jac[sel], V0=V0, foot=foot, nu=nu, nv=nv))

    ss2 = 2.0 * (sigma_s_um / voxel_um) ** 2
    sn2 = 2.0 * (scale * sigma_n_um / voxel_um) ** 2
    meshes = {}
    if batch and dev == "cuda" and items:                          # one batched Adam over all sheets (GPU)
        fitted = _fit_all_gpu(items, step_px=step_px, ss2=ss2, sn2=sn2, k=k, iters=iters, lr=lr,
                              lam_smooth=lam_smooth, lam_step=lam_step, knn_every=knn_every, dev=dev)
        for m in items:
            meshes[m["c"]] = {**fitted[m["c"]], "n_points": int(len(m["P"])), "width": None}
    else:                                                          # per-cluster fits, thread-parallel on CPU
        def _one(m):
            r = fit_sheet_mesh(m["P"], m["normals"], m["jac"], voxel_um, step_px=step_px, sigma_s_um=sigma_s_um,
                               sigma_n_um=sigma_n_um, scale=scale, k=k, iters=iters, lr=lr, lam_smooth=lam_smooth,
                               lam_step=lam_step, knn_every=knn_every, device=dev)
            return m["c"], {**r, "n_points": int(len(m["P"])), "width": None}
        if n_jobs > 1 and len(items) > 1:
            import torch
            from concurrent.futures import ThreadPoolExecutor
            n_threads0 = torch.get_num_threads()
            try:
                import os as _os
                torch.set_num_threads(max(1, (_os.cpu_count() or n_jobs) // n_jobs))  # avoid intra-op oversubscription
                with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                    for c, r in ex.map(_one, items):
                        meshes[c] = r
            finally:
                torch.set_num_threads(n_threads0)
        else:
            for m in items:
                c, r = _one(m); meshes[c] = r
    if width:                                                      # per-vertex width from the cluster's own points
        by_c = {m["c"]: m for m in items}
        for c, mesh in meshes.items():
            w, off, vc = compute_mesh_width(mesh, by_c[c]["P"], voxel_um, device=dev)
            mesh["width"] = w; mesh["offset"] = off; mesh["verts"] = vc   # verts recentred to true mid-sheet
            Vgrid = mesh["V"].reshape(-1, 3).copy()                       # write recentre back into the grid too
            Vgrid[mesh["foot"].reshape(-1)] = vc
            mesh["V"] = Vgrid.reshape(mesh["nu"], mesh["nv"], 3)
    if verbose:
        for c in sorted(meshes):
            m = meshes[c]
            wtxt = ""
            if m.get("width") is not None:
                w = m["width"]; ok = np.isfinite(w)
                wtxt = (f", width µm p50 {np.nanmedian(w):.0f} [{np.nanpercentile(w[ok],10):.0f}–"
                        f"{np.nanpercentile(w[ok],90):.0f}] ({100*ok.mean():.0f}% defined)") if ok.any() else ", width n/a"
            print(f"  [mesh] cluster {c}: {m['n_points']} pts -> {len(m['verts'])} verts, "
                  f"{len(m['tris'])} tris{wtxt}", flush=True)
    return meshes

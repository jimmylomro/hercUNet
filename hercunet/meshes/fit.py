"""Centre-surface mesh FITTING — shared geometry, reusable by any product (HercuLabels, HercuViT).

Fit a quad-mesh to the medial surface of a per-cluster point cloud: an on-the-fly accumulated surface-slab
potential (peaks at the sheet centre, bridges within-sheet holes) is maximised at the mesh vertices by Adam,
regularised by Laplacian smoothness + even edge length. Grid spacing is PHYSICAL (``step_um``); triangulation is
IMPLICIT in the regular grid + footprint mask (faces reconstructable, need not be stored).

Pure numpy + scipy + (optional, lazy) torch — NO dependency on labels / herculabels / hercuvit. The caller passes
the per-point normals + Jacobian from whatever source it has; see ``fit_sheet_meshes``.
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


def fit_sheet_meshes(pts, plab, normals, jac, voxel_um, *, step_um=30.0, min_points=200,
                     sigma_s_um=180.0, sigma_n_um=25.0, scale=2.0, k=64, iters=250, lr=0.5, lam_smooth=2.0,
                     lam_step=1.0, knn_every=50, device=None, batch=None, n_jobs=1, width=True, verbose=True):
    """Fit a centre-surface mesh for every cluster in a clustered point cloud (on-the-fly slab potential).

    ``pts`` [N,3] voxel coords, ``plab`` [N] per-point cluster label (-1 = noise), ``normals`` [N,3] per-point unit
    sheet normals and ``jac`` [N,3,3] their spatial Jacobian (curvature; pass zeros for a flat-tangent slab). The
    caller supplies normals/jac from whatever source it has (frame field, local PCA, structure tensor); this helper
    is source-agnostic and has NO dependency on any product package. ``batch`` (default True on cuda) runs one
    batched Adam over all clusters; otherwise a thread pool of ``n_jobs``. ``width`` adds per-vertex sheet width.

    Returns ``{cluster_id: mesh_dict}`` — each ``fit_sheet_mesh`` output plus ``n_points``, ``width`` [nvert] µm
    (or None) and ``offset`` [nvert] µm. Vertices are voxel (z,y,x) coords; triangulation is implicit in the grid.
    """
    pts = np.asarray(pts, np.float32); plab = np.asarray(plab)
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

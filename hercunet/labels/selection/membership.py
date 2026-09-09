"""Streamline → sheet-membership embeddings (per-window).

One trainable vector per LINE; points only SAMPLE the training objective. NO precomputed [N,N] affinity,
NO arc reduction, NO derived sheet-spacing, NO density valleys. Each iteration draws anchors ∝ arc length
and, among the candidates within MAX_SHEET_DIST, SOFT-samples one positive and one hard-negative via a
multinomial over the LOCAL SHEET-SLAB model with MUTUAL membership (see ``_neighbor_tables``): every point
is a thin parabolic slab in its own frame (σ_s along the tangent, σ_n across the normal, bent by its
signed curvature), and a pair is a positive only if EACH point's slab contains the OTHER
(``pos_w = S_p(q)·S_q(p)``) — symmetric by construction, and rejecting crossing / adjacent sheets
structurally. Plus minority far-negatives; abstain on degenerate. A triplet-margin loss integrates the
point-wise statistics over iterations, so a pair that is same-sheet over most of its length but grazes at
a point nets out correctly.

Only physical constants: ``MAX_SHEET_DIST`` (candidate radius, NOT pitch) and the substrate scales σ_n
(≈ sheet across-normal thickness) / σ_s (along reach). In 3-D (whole-volume) a world-z cap σ_z limits reach
along the noisy z axis. 2D ceiling (by design): locally parallel + equidistant + equal-curvature windings
cannot be separated — that needs 3D context (3-D CED / verso streamlets), not tuning.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree


def _slab_logscore(v, t, n, kappa, ss2, sn2, m=None, sm2=None):
    """log S = −a²/2σ_s² − r²/2σ_n² (− m_comp²/2σ_m²): the thin parabolic slab in the frame (t, n[, m]).
    ``v`` chord (…,D), ``t/n/m`` unit frame vectors (…,D), ``kappa`` signed (…,). r = TRUE perpendicular
    distance to the osculating Monge patch. All broadcast over the (P,K) neighbour grid.

    ``sn2`` is 2σ_n². Pass a scalar/array for a SYMMETRIC slab, or a tuple ``(sn2_plus, sn2_minus)`` for a
    TWO-SIDED slab whose across-normal width differs per side — ``sn2_plus`` on the +normal side, ``sn2_minus``
    on the −normal side (selected by the sign of the across-normal component). This lets σ_n squish toward
    a density valley (the gap to the next winding) while staying wide toward the point's own sheet body."""
    a = (v * t).sum(-1)
    across = (v * n).sum(-1)
    r = (across - 0.5 * kappa * a ** 2) / np.sqrt(1.0 + (kappa * a) ** 2)
    if isinstance(sn2, tuple):
        sn2_plus, sn2_minus = sn2
        sn2 = np.where(across >= 0.0, sn2_plus, sn2_minus)         # per-side across-normal width
    ex = a ** 2 / ss2 + r ** 2 / sn2
    if m is not None and sm2:
        ex = ex + (v * m).sum(-1) ** 2 / sm2
    return -ex


def measure_sigma_n(img, pts, tans, voxel_um, r_um=90.0, valley_frac=0.6, frac=0.5,
                    sigma_min_um=8.0, sigma_max_um=25.0):
    """TWO-SIDED across-normal σ_n at each point, measured from the along-normal intensity profile of the
    (CED'd) slice ``img``. At each point we sample the intensity along ±normal (leftward n = perp(tangent),
    matching the slab's convention) out to ``r_um``; the first drop below ``valley_frac``·(peak at the point)
    marks the local ridge edge on that side (the gap to the next winding). σ_n(side) = ``frac`` · (edge
    distance) — deliberately SMALLER than the measured half-width so the Gaussian slab dies WITHIN the sheet,
    not at the gap — clamped to ``[sigma_min_um, sigma_max_um]``. ``sigma_max_um`` defaults to the OLD fixed
    σ_n (25 µm): adaptive σ_n is only ever ≤ what it used to be, never wider — it can only tighten (edge
    streamlets, small gaps) or, with no valley in reach, fall back to the old value. Vectorised. ``pts``
    [P,2] xy, ``tans`` [P,2] unit tangents. Returns ``sp_um, sm_um`` ([P] each)."""
    R = max(1.0, r_um / voxel_um)
    step = 0.5
    ks = np.arange(0.0, R + step, step)                           # [K] outward offsets (px)
    nx = -tans[:, 1]
    ny = tans[:, 0]                                               # leftward normal
    yp = pts[:, 1][:, None] + ks[None, :] * ny[:, None]           # +side rows [P,K]
    xp = pts[:, 0][:, None] + ks[None, :] * nx[:, None]
    ym = pts[:, 1][:, None] - ks[None, :] * ny[:, None]           # −side rows
    xm = pts[:, 0][:, None] - ks[None, :] * nx[:, None]
    ip = map_coordinates(img, [yp.ravel(), xp.ravel()], order=1, mode="nearest").reshape(yp.shape)
    im = map_coordinates(img, [ym.ravel(), xm.ravel()], order=1, mode="nearest").reshape(ym.shape)
    thr = valley_frac * np.maximum(ip[:, 0], 1e-6)[:, None]       # ip[:,0]=im[:,0]= intensity at the point

    def edge_um(prof):
        below = prof < thr
        idx = np.argmax(below, axis=1)                           # first True (0 if none — col 0 never below)
        idx = np.where(below.any(axis=1), idx, prof.shape[1] - 1)   # no valley → full reach
        return idx * step * voxel_um
    sp = np.clip(frac * edge_um(ip), sigma_min_um, sigma_max_um)
    sm = np.clip(frac * edge_um(im), sigma_min_um, sigma_max_um)
    return sp, sm


def measure_sigma_n_3d(vol, pts, normals, voxel_um, r_um=90.0, valley_frac=0.6, frac=0.5,
                       sigma_min_um=8.0, sigma_max_um=25.0, use_gpu=False, chunk=400000):
    """TWO-SIDED across-normal σ_n at 3-D points — the volumetric lift of :func:`measure_sigma_n`, for the
    native-3-D (paperlet) slab. At each point sample the CED'd VOLUME ``vol`` along ±n̂ (the 3-D
    structure-tensor normal) out to ``r_um``; the first drop below ``valley_frac``·(intensity at the point)
    marks the local ridge edge on that side (the gap to the next winding). σ_n(side) = ``frac``·(edge
    distance), clamped to ``[sigma_min_um, sigma_max_um]`` — so adaptive σ_n only ever TIGHTENS below the fixed
    cap (edge sheets / small gaps), never widens; with no valley in reach it falls back to ``sigma_max_um``.
    Vectorised over all points × offsets. ``pts`` [P,3] (z,y,x); ``normals`` [P,3] unit (n_z,n_y,n_x) — SAME
    order as positions, i.e. sampled straight from ``ff['normal']``. Returns ``sp_um, sm_um`` ([P] each, +side
    along +n̂ / −side along −n̂ — matching the slab's ``across≥0`` +side convention)."""
    R = max(1.0, r_um / voxel_um)
    step = 0.5
    ks = np.arange(0.0, R + step, step)                           # [K] outward offsets (px)
    n = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)
    pp = pts[:, :, None] + ks[None, None, :] * n[:, :, None]       # +side [P,3,K]
    pm = pts[:, :, None] - ks[None, None, :] * n[:, :, None]       # −side
    K = len(ks)
    P = len(pts)
    _torch = None
    if use_gpu:
        from hercunet.labels.meshlets.streamlets3d import _torch_cuda, _field5d, _gsample
        _torch = _torch_cuda()
    if _torch is not None:                                         # ---- GPU path: same trace via grid_sample ----
        torch = _torch; dev = torch.device("cuda"); shape = vol.shape
        vol5 = _field5d(vol, torch, dev).to(torch.float32)         # scalar field (1,1,Z,Y,X)
        ks_t = torch.as_tensor(ks, dtype=torch.float32, device=dev)
        ip = np.empty((P, K), np.float32); im = np.empty((P, K), np.float32)
        for lo in range(0, P, chunk):                             # tile → bounded GPU memory
            hi = min(lo + chunk, P); c = hi - lo
            pt = torch.as_tensor(pts[lo:hi], dtype=torch.float32, device=dev)
            nt = torch.as_tensor(n[lo:hi], dtype=torch.float32, device=dev)
            base = pt[:, None, :]; off = ks_t[None, :, None] * nt[:, None, :]      # [c,K,3]
            ip[lo:hi] = _gsample(vol5, (base + off).reshape(-1, 3), shape, torch).reshape(c, K).cpu().numpy()
            im[lo:hi] = _gsample(vol5, (base - off).reshape(-1, 3), shape, torch).reshape(c, K).cpu().numpy()
    else:
        ip = map_coordinates(vol, [pp[:, 0].ravel(), pp[:, 1].ravel(), pp[:, 2].ravel()],
                             order=1, mode="nearest").reshape(P, K)
        im = map_coordinates(vol, [pm[:, 0].ravel(), pm[:, 1].ravel(), pm[:, 2].ravel()],
                             order=1, mode="nearest").reshape(P, K)
    thr = valley_frac * np.maximum(ip[:, 0], 1e-6)[:, None]        # ip[:,0]=im[:,0]= intensity at the point

    def edge_um(prof):
        below = prof < thr
        idx = np.argmax(below, axis=1)                            # first True (0 if none — col 0 never below)
        idx = np.where(below.any(axis=1), idx, prof.shape[1] - 1)   # no valley → full reach
        return idx * step * voxel_um
    sp = np.clip(frac * edge_um(ip), sigma_min_um, sigma_max_um)
    sm = np.clip(frac * edge_um(im), sigma_min_um, sigma_max_um)
    return sp, sm


def _normal_jacobian(nf, p, n0, h=0.5):
    """Local Jacobian ``J = ∂n̂/∂x`` (3×3; column ``ax`` = ∂n̂/∂(z,y,x)[ax]) of the normal field at a SINGLE
    anchor ``p`` (z,y,x), each neighbour sample sign-aligned to the anchor normal ``n0`` (n≡−n) before the
    central difference so a field sign-flip doesn't fake a huge derivative. Trilinear. This is the shape
    operator for the surface slab's curvature term — LOCAL (per-anchor), never a global/averaged normal."""
    def samp(q):
        c = np.asarray(q, float).reshape(3, 1)
        s = np.array([map_coordinates(nf[..., i], c, order=1, mode="nearest")[0] for i in range(3)])
        s /= (np.linalg.norm(s) + 1e-9)
        return -s if float(s @ n0) < 0 else s
    cols = []
    for ax in range(3):
        e = np.zeros(3)
        e[ax] = h
        cols.append((samp(p + e) - samp(p - e)) / (2.0 * h))
    return np.stack(cols, axis=1)                                 # J[:, ax] = ∂n̂/∂(axis)


def normal_jacobian_batch(nf, pts, n0, h=0.5):
    """VECTORISED local Jacobian ``J = ∂n̂/∂x`` [P,3,3] at many anchors ``pts`` [P,3] (z,y,x) — the batched
    equivalent of :func:`_normal_jacobian` (each neighbour sample sign-aligned to that anchor's ``n0`` [P,3]
    before the central diff). One vectorised ``map_coordinates`` per (offset, component) instead of a Python
    loop per point — the difference between ~1M interpolation calls and ~18. ``J[:, :, ax] = ∂n̂/∂(axis)``."""
    def samp(q, ref):                                            # q [P,3], ref [P,3] → sign-aligned unit n̂ [P,3]
        c = [q[:, 0], q[:, 1], q[:, 2]]
        s = np.stack([map_coordinates(nf[..., i], c, order=1, mode="nearest") for i in range(3)], axis=1)
        s /= (np.linalg.norm(s, axis=1, keepdims=True) + 1e-9)
        flip = (s * ref).sum(1) < 0
        s[flip] *= -1.0
        return s
    cols = []
    for ax in range(3):
        e = np.zeros(3)
        e[ax] = h
        cols.append((samp(pts + e, n0) - samp(pts - e, n0)) / (2.0 * h))
    return np.stack(cols, axis=2)                                # [P,3,3], column ax = ∂n̂/∂(axis)


def surface_slab_logscore(v, n, J, ss2, sn2):
    """log S of the 3-D thin SURFACE slab of a paperlet anchor: ISOTROPIC in-plane (σ_s in both tangent
    directions — matches the pipeline default σ_m=σ_s) and thin across the normal (σ_n), curvature-corrected by
    the local shape operator so the slab HUGS the curved sheet instead of a flat pancake bulging off it.
    ``v`` chord (…,3) anchor→query (z,y,x); ``n`` anchor unit normal (3,); ``J`` = ∂n̂/∂x (3×3,
    :func:`_normal_jacobian`). The sheet's across-normal height at in-plane displacement ρ·û is ``0.5·κ·ρ²``
    with normal curvature ``κ = −(û·Jû)`` (Weingarten: the tangential part of ∂_û n̂ is −Bû, so the height
    Hessian is B), giving the SAME Monge form as :func:`_slab_logscore` with the in-plane displacement
    magnitude in place of the along-tangent coordinate. ``sn2`` scalar (=2σ_n²) or a tuple ``(2σ⁺²,2σ⁻²)`` for
    the adaptive TWO-SIDED width (σ⁺ on the +n̂ side by the sign of ``across``). Returns log S (…,)."""
    across = (v * n).sum(-1)
    vip = v - across[..., None] * n                               # in-plane part of the chord
    inp2 = np.maximum((vip * vip).sum(-1), 0.0)                   # ρ²
    Jv = vip @ J.T
    kappa = -(vip * Jv).sum(-1) / np.maximum(inp2, 1e-9)         # κ = −(û·Jû), û = v_ip/ρ
    r = (across - 0.5 * kappa * inp2) / np.sqrt(1.0 + kappa ** 2 * inp2)   # ⟂ dist to the osculating sheet
    if isinstance(sn2, tuple):
        sn2 = np.where(across >= 0.0, sn2[0], sn2[1])             # per-side across-normal width
    return -(inp2 / ss2 + r ** 2 / sn2)


def averaged_normal_grid_3d(pts, nrm, cell_px):
    """Coarse 3-D grid of LOCALLY-AVERAGED (sign-free) normals — the 3-D lift of the per-cell normal in
    :func:`_corridor_negatives` (``meshlet_clustering``). Bin ``pts`` [P,3] (z,y,x) into ``cell_px`` cubes;
    per occupied cell average the ORIENTATION via the tensor Σ n̂n̂ᵀ (raw-vector averaging cancels n vs −n —
    the same sign trap the 2-D grid avoids) and take its principal eigenvector as the cell normal. Also keep
    one REPRESENTATIVE point per cell (nearest the cell centre). ``nrm`` [P,3] unit (z,y,x), same order as
    ``pts``. Returns ``(occ_cells [C], cell_ctr [C,3], cell_nrm [C,3], reppt [C], cell_of_pt [P])`` — C =
    number of occupied cells, ``reppt`` indexes into ``pts``, ``cell_of_pt`` maps each point to its row in
    ``occ_cells`` (−1 never happens; every point occupies a cell). LOCAL per-cell only, never a window normal."""
    org = pts.min(0)
    g = np.floor((pts - org) / cell_px).astype(np.int64)
    gZ, gY, gX = int(g[:, 0].max()) + 1, int(g[:, 1].max()) + 1, int(g[:, 2].max()) + 1
    flat = (g[:, 0] * gY + g[:, 1]) * gX + g[:, 2]
    nC = gZ * gY * gX
    # per-cell orientation tensor Σ n̂n̂ᵀ (6 unique comps), sign-free
    comps = {}
    for a, b, key in ((0, 0, "zz"), (1, 1, "yy"), (2, 2, "xx"), (0, 1, "zy"), (0, 2, "zx"), (1, 2, "yx")):
        comps[key] = np.bincount(flat, weights=nrm[:, a] * nrm[:, b], minlength=nC)
    cnt = np.bincount(flat, minlength=nC)
    occ = np.where(cnt > 0)[0]
    C = len(occ)
    Tn = np.zeros((C, 3, 3))
    Tn[:, 0, 0], Tn[:, 1, 1], Tn[:, 2, 2] = comps["zz"][occ], comps["yy"][occ], comps["xx"][occ]
    Tn[:, 0, 1] = Tn[:, 1, 0] = comps["zy"][occ]
    Tn[:, 0, 2] = Tn[:, 2, 0] = comps["zx"][occ]
    Tn[:, 1, 2] = Tn[:, 2, 1] = comps["yx"][occ]
    w, V = np.linalg.eigh(Tn)                                     # ascending; principal = last column
    cell_nrm = V[:, :, -1]                                        # [C,3] sign-free averaged normal
    ci = np.arange(nC)
    cz = (ci // (gY * gX))
    cy = (ci % (gY * gX)) // gX
    cx = ci % gX
    cell_ctr_all = np.stack([cz, cy, cx], 1).astype(np.float64) * cell_px + org + 0.5 * cell_px
    cell_ctr = cell_ctr_all[occ]
    remap = -np.ones(nC, np.int64)
    remap[occ] = np.arange(C)
    cell_of_pt = remap[flat]
    # representative point per occupied cell = nearest its centre
    order = np.argsort(cell_of_pt, kind="stable")
    bnd = np.searchsorted(cell_of_pt[order], np.arange(C + 1))
    reppt = np.empty(C, np.int64)
    for c in range(C):
        members = order[bnd[c]:bnd[c + 1]]
        reppt[c] = members[np.argmin(((pts[members] - cell_ctr[c]) ** 2).sum(1))]
    return occ, cell_ctr, cell_nrm, reppt, cell_of_pt


def corridor_negatives_3d(pts, nrm, cell_px, dist_px, kneg, seed=0, max_cells_search=4096):
    """3-D lift of :func:`_corridor_negatives` — per-point hard-negative candidates that lie ACROSS the sheet
    normal (the adjacent winding), never along the fibre. Build the averaged-normal grid
    (:func:`averaged_normal_grid_3d`); for each occupied cell, project every other occupied cell centre onto
    THIS cell's (sign-free) averaged normal and keep those with ``|proj| ≥ dist_px`` (≥ ~one winding across —
    same-sheet-far cells project ≈0 → auto-excluded), then subsample ``kneg`` at random. A point inherits its
    cell's far cells; the negative is that far cell's representative point. Returns ``neg_nbr [P,kneg]``
    (indices into ``pts``, −1 pad) + ``neg_absr [P,kneg]`` (|along-normal projection| px). NO walk, NO valley,
    NO meshlet-id, NO global normal; LOCAL per-cell. ``max_cells_search`` caps the per-cell scan for huge C
    (subsample the other-cell pool first — logged by the caller, not silent)."""
    P = len(pts)
    neg_nbr = np.full((P, kneg), -1, np.int64)
    neg_absr = np.zeros((P, kneg), np.float32)
    if P < 4:
        return neg_nbr, neg_absr
    occ, cell_ctr, cell_nrm, reppt, cell_of_pt = averaged_normal_grid_3d(pts, nrm, cell_px)
    C = len(occ)
    rng = np.random.default_rng(seed)
    pool = np.arange(C)
    if C > max_cells_search:                                      # cap the other-cell pool (not silent)
        pool = rng.choice(C, max_cells_search, replace=False)
    ctr_pool = cell_ctr[pool]
    order = np.argsort(cell_of_pt, kind="stable")
    bnd = np.searchsorted(cell_of_pt[order], np.arange(C + 1))
    per_cell_neg = [np.empty(0, np.int64)] * C
    per_cell_absr = [np.empty(0, np.float32)] * C
    for c in range(C):
        proj = (ctr_pool - cell_ctr[c]) @ cell_nrm[c]            # project pool cell centres onto THIS normal
        far_local = np.where(np.abs(proj) >= dist_px)[0]
        if not len(far_local):
            continue
        if len(far_local) > kneg:
            far_local = far_local[rng.choice(len(far_local), kneg, replace=False)]
        far_cells = pool[far_local]
        per_cell_neg[c] = reppt[far_cells]
        per_cell_absr[c] = np.abs(proj[far_local]).astype(np.float32)
    for c in range(C):
        reps, absr = per_cell_neg[c], per_cell_absr[c]
        if not len(reps):
            continue
        pidx = order[bnd[c]:bnd[c + 1]]
        neg_nbr[np.ix_(pidx, np.arange(len(reps)))] = reps[None, :]
        neg_absr[np.ix_(pidx, np.arange(len(absr)))] = absr[None, :]
    return neg_nbr, neg_absr


def gap_gated_negatives_3d(pts, nrm, bced, voxel_um, pap_id=None, anchor_idx=None, kneg=32,
                           sigma_max_um=25.0, tau=0.30, tau_nb=0.50, margin_k=2.5,
                           gate_floor_um=15.0, reach_um=120.0, sigma_ip_um=40.0,
                           lam_decay_um=60.0, depth_scale=0.30, kq=48, chunk=40000, seed=0,
                           use_gpu=False, sn2_plus=None, sn2_minus=None):
    """GAP-GATED negative slab — the validated ``neg_probe`` kernel, vectorised for training.

    For each anchor, along BOTH normal sides: require a boundary (σ_n < σ_max) AND a density
    VALLEY (rel < ``tau``) past a conservative gate (``margin_k``·σ_n, floored) AND a neighbour
    rise (rel ≥ ``tau_nb``) after the valley — i.e. an ACTUAL gap then the adjacent winding.
    Where a side fires, candidate points in the blob band (gate < along-normal < reach, in-plane
    < ~σ_ip) become negatives, weighted::

        neg_w = exp(-depth/depth_scale) · exp(-½(perp/σ_ip)²) · exp(-proj/λ)

    deep gap + on-axis + near ⇒ strong push; shallow (same-sheet) fires ⇒ ~0 (fold-safe). The
    two sides are combined by max weight; SAME-streamlet candidates are excluded. Returns
    ``neg_nbr [A,kneg]`` (indices into ``pts``, −1 pad) and ``neg_w [A,kneg]`` — the exact mirror
    of ``pos_nbr/pos_w``. ``A`` = len(anchor_idx) (or P), matching the positive table's rows."""
    P = len(pts)
    aidx = np.arange(P) if anchor_idx is None else np.asarray(anchor_idx)
    A = len(aidx)
    neg_nbr = np.full((A, kneg), -1, np.int64)
    neg_w = np.zeros((A, kneg), np.float32)
    if P < 8 or A == 0:
        return neg_nbr, neg_w
    nrm = nrm / (np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9)
    an = nrm[aidx].astype(np.float64)
    ap = pts[aidx].astype(np.float64)
    a_sid = None if pap_id is None else np.asarray(pap_id)[aidx]
    sid_all = None if pap_id is None else np.asarray(pap_id)
    if sn2_plus is not None and sn2_minus is not None:                # reuse σ_n already measured for positives
        sp = np.sqrt(np.maximum(np.asarray(sn2_plus, float)[aidx], 0.0) / 2.0) * voxel_um
        sm = np.sqrt(np.maximum(np.asarray(sn2_minus, float)[aidx], 0.0) / 2.0) * voxel_um
    else:
        sp, sm = measure_sigma_n_3d(bced, ap, an, voxel_um, sigma_max_um=sigma_max_um)
    tree = cKDTree(pts)
    t_um = np.arange(0.0, reach_um + 1e-6, 2.0)
    t_vx = t_um / voxel_um
    T = len(t_um)
    knn = min(int(kq), P)                                              # candidates gathered AT THE BLOB CENTRE
    torch = None
    if use_gpu:
        from hercunet.labels.meshlets.streamlets3d import _torch_cuda, _field5d, _gsample
        torch = _torch_cuda()

    def _topk(side_ids, side_w, n):                                    # pool both sides, keep top-kneg by weight
        ids = np.concatenate(side_ids, axis=1); wts = np.concatenate(side_w, axis=1)
        kk = ids.shape[1]; kne = min(kneg, kk)
        top = np.argpartition(-wts, kne - 1, axis=1)[:, :kne] if kk > kne else np.tile(np.arange(kk), (n, 1))
        rows = np.arange(n)[:, None]
        return np.where(wts[rows, top] > 0, ids[rows, top], -1), wts[rows, top]

    if torch is not None:                                             # ---- GPU path (trace+gate+band on CUDA) ----
        dev = torch.device("cuda"); f64 = torch.float32              # fp32: 3.2× over fp64 on GeForce, parity ~1e-6
        bced5 = _field5d(bced, torch, dev).to(f64); shape = bced.shape   # field dtype must match fp32 coords
        t_um_t = torch.as_tensor(t_um, dtype=f64, device=dev)
        t_vx_t = torch.as_tensor(t_vx, dtype=f64, device=dev)
        ar_T = torch.arange(T, device=dev)
        for lo in range(0, A, chunk):
            hi = min(lo + chunk, A); n = hi - lo
            Pa_t = torch.as_tensor(ap[lo:hi], dtype=f64, device=dev)
            Na_t = torch.as_tensor(an[lo:hi], dtype=f64, device=dev)
            side_ids, side_w = [], []
            for sign, sig in [(1.0, sp[lo:hi]), (-1.0, sm[lo:hi])]:
                axis = sign * Na_t
                sig_t = torch.as_tensor(sig, dtype=f64, device=dev)
                coords = Pa_t[:, None, :] + t_vx_t[None, :, None] * axis[:, None, :]   # [n,T,3]
                d = _gsample(bced5, coords.reshape(-1, 3), shape, torch).reshape(n, T)
                rel = d / torch.clamp(d[:, 0:1], min=1e-6)
                gate = torch.clamp(margin_k * sig_t, min=gate_floor_um)
                boundary_ok = sig_t < (sigma_max_um - 1e-3)
                past = t_um_t[None, :] >= gate[:, None]
                below = (rel < tau) & past
                has_valley = below.any(1)
                vi = torch.where(has_valley, below.to(f64).argmax(1),
                                 torch.full((n,), T, device=dev, dtype=torch.long))
                after = ar_T[None, :] >= vi[:, None]
                nb_mask = (rel >= tau_nb) & after
                has_nb = nb_mask.any(1)
                fires = boundary_ok & has_valley & has_nb
                depth = torch.where(past, rel, torch.full_like(rel, float("inf"))).min(1).values
                depth_w = torch.exp(-torch.clamp(depth, min=0.0) / depth_scale)
                ni = torch.where(has_nb, nb_mask.to(f64).argmax(1), torch.zeros(n, device=dev, dtype=torch.long))
                D = t_um_t[ni]
                center = (Pa_t + (D[:, None] / voxel_um) * axis).cpu().numpy()
                dq, iq = tree.query(center, k=knn, workers=-1)         # KDTree stays on CPU (like positives)
                dq, iq = np.atleast_2d(dq), np.atleast_2d(iq)
                valc_np = (iq < P) & np.isfinite(dq)
                jc = np.where(valc_np, iq, 0)
                if a_sid is not None:
                    valc_np = valc_np & (a_sid[lo:hi][:, None] != sid_all[jc])
                q = torch.as_tensor(pts[jc], dtype=f64, device=dev)   # [n,knn,3]
                off = q - Pa_t[:, None, :]
                proj = (off * axis[:, None, :]).sum(2)
                perp = torch.linalg.norm(off - proj[:, :, None] * axis[:, None, :], dim=2)
                proj_um, perp_um = proj * voxel_um, perp * voxel_um
                valc = torch.as_tensor(valc_np, device=dev)
                in_band = (valc & fires[:, None] & (proj_um > gate[:, None])
                           & (proj_um < reach_um) & (perp_um < 2.5 * sigma_ip_um))
                w = (in_band.to(f64) * depth_w[:, None]
                     * torch.exp(-0.5 * (perp_um / sigma_ip_um) ** 2)
                     * torch.exp(-proj_um / lam_decay_um)).cpu().numpy().astype(np.float32)
                side_ids.append(np.where(w > 0, jc, -1)); side_w.append(w)
            neg_nbr[lo:hi], neg_w[lo:hi] = _topk(side_ids, side_w, n)
        del bced5; torch.cuda.empty_cache()
    else:                                                             # ---- CPU path (numpy + map_coordinates) ----
        for lo in range(0, A, chunk):
            hi = min(lo + chunk, A); n = hi - lo
            Pa, Na = ap[lo:hi], an[lo:hi]
            side_ids, side_w = [], []
            for sign, sig in [(1.0, sp[lo:hi]), (-1.0, sm[lo:hi])]:
                axis = sign * Na
                coords = Pa[:, None, :] + t_vx[None, :, None] * axis[:, None, :]
                d = map_coordinates(bced, coords.reshape(-1, 3).T, order=1, mode="nearest").reshape(n, T)
                rel = d / np.maximum(d[:, 0], 1e-6)[:, None]
                gate = np.maximum(margin_k * sig, gate_floor_um)
                boundary_ok = sig < (sigma_max_um - 1e-3)
                past = t_um[None, :] >= gate[:, None]
                below = (rel < tau) & past
                has_valley = below.any(1)
                vi = np.where(has_valley, below.argmax(1), T)
                after = np.arange(T)[None, :] >= vi[:, None]
                nb_mask = (rel >= tau_nb) & after
                has_nb = nb_mask.any(1)
                fires = boundary_ok & has_valley & has_nb
                depth = np.where(past, rel, np.inf).min(1)
                depth_w = np.exp(-np.clip(depth, 0.0, None) / depth_scale)
                D = t_um[np.where(has_nb, nb_mask.argmax(1), 0)]
                center = Pa + (D[:, None] / voxel_um) * axis
                dq, iq = tree.query(center, k=knn, workers=-1)
                dq, iq = np.atleast_2d(dq), np.atleast_2d(iq)
                valc = (iq < P) & np.isfinite(dq)
                jc = np.where(valc, iq, 0)
                if a_sid is not None:
                    valc = valc & (a_sid[lo:hi][:, None] != sid_all[jc])
                off = pts[jc].astype(np.float64) - Pa[:, None, :]
                proj = (off * axis[:, None, :]).sum(2)
                perp = np.linalg.norm(off - proj[:, :, None] * axis[:, None, :], axis=2)
                proj_um, perp_um = proj * voxel_um, perp * voxel_um
                in_band = (valc & fires[:, None] & (proj_um > gate[:, None])
                           & (proj_um < reach_um) & (perp_um < 2.5 * sigma_ip_um))
                w = (in_band * depth_w[:, None]
                     * np.exp(-0.5 * (perp_um / sigma_ip_um) ** 2)
                     * np.exp(-proj_um / lam_decay_um)).astype(np.float32)
                side_ids.append(np.where(w > 0, jc, -1)); side_w.append(w)
            neg_nbr[lo:hi], neg_w[lo:hi] = _topk(side_ids, side_w, n)
    pos = neg_w > 0                                                   # normalise weight scale: a confident gap
    if pos.any():                                                    # negative ≈ 1 (comparable to soft-neg weight
        s = float(np.percentile(neg_w[pos], 95))                     # of 1), so it isn't swamped in the triplet.
        if s > 0:
            neg_w = np.clip(neg_w / s, 0.0, 1.0).astype(np.float32)
    return neg_nbr, neg_w


def _neighbor_tables(pos, line_id, tan, kappa, max_px, sigma_s_px, sigma_n_px, sigma_z_px, kcap,
                     nrm3=None, sigma_m_px=None, min_tilt_deg=0.0, pos_threshold=0.0, neg_below=1.0,
                     pt_weight=None, sn2_plus=None, sn2_minus=None,
                     poll_px=None, sigma_s_poll_px=None, sigma_n_poll_px=None, poll_k=512):
    """For every point, its ≤``kcap`` in-range neighbours + SOFT sampling weights ``pos_w, neg_w [P,K]``
    from the LOCAL SHEET-SLAB model with MUTUAL membership. Each point is a thin parabolic slab in its own
    frame — long along the tangent (σ_s), thin across the normal (σ_n), bent by its signed curvature. A
    pair is a positive only if EACH point's slab contains the OTHER: ``pos_w = S_p(q)·S_q(p)`` (symmetric
    by construction → no moving target; rejects crossing/adjacent sheets structurally). Hard negatives =
    close but low mutual: ``neg_w = 1 − mutual``.

    ``nrm3`` None → **2-D model**: normal = perp(tangent), in-plane. ``nrm3`` given → **3-D model**:
    normal = the 3-D structure-tensor normal (xyz), plus the in-sheet ``mid = normal×tangent`` (σ_m) and a
    NEAR-VERTICAL gate (``min_tilt_deg``: a normal within that angle of straight-up ≈ flat-sheet/z-noise →
    abstain). In 3-D positions a world-z cap ``exp(−dz²/2σ_z²)`` limits reach along the noisy z axis.

    **Candidate selection (``poll_px``).** Default (``poll_px=None``): the ``kcap`` candidates are the nearest
    by RAW EUCLIDEAN distance within ``max_px`` — isotropic, mismatched with the anisotropic slab (the nearest
    slots get eaten by across-winding points the slab rejects, so far-along-fibre positives never become
    candidates; audited to clip >50% of positive mass). When ``poll_px`` is set, do a TWO-STAGE anisotropic
    selection: gather up to ``poll_k`` within the (larger) ``poll_px`` radius, score each with an EXPANDED
    one-sided slab (long ``sigma_s_poll_px`` along-fibre + WIDE ``sigma_n_poll_px`` across — wide enough to keep
    the adjacent winding as a hard-negative candidate), and keep the top-``kcap`` by that poll score. The strict
    (adaptive-σ_n) mutual slab below then does the actual agreement/split on that better-chosen set. In-plane
    poll frame (2-D model); a coarse pre-filter for the 3-D model too."""
    p = len(pos)
    if p < 3:
        return np.zeros((p, 1), int), np.zeros((p, 1)), np.zeros((p, 1))
    if poll_px is not None:                                       # two-stage anisotropic candidate selection
        kq = min(poll_k, p)
        dd, ii = cKDTree(pos).query(pos, k=kq, distance_upper_bound=poll_px, workers=-1)
        dd, ii = np.atleast_2d(dd), np.atleast_2d(ii)
        si = np.arange(p)[:, None]
        vm = (ii < p) & np.isfinite(dd) & (ii != si) & (line_id[:, None] != line_id[np.where(ii < p, ii, 0)])
        jp = np.where(vm, ii, 0)
        xy = pos[:, :2]
        vpoll = xy[jp] - xy[:, None, :]
        tp = tan[:, None, :]
        npn = np.stack([-tp[..., 1], tp[..., 0]], axis=-1)        # leftward normal (in-plane)
        a_poll = (vpoll * tp).sum(-1)
        r_poll = (vpoll * npn).sum(-1)
        ss2p = 2.0 * (sigma_s_poll_px or sigma_s_px) ** 2
        sn2p = 2.0 * (sigma_n_poll_px or sigma_n_px) ** 2
        score = -(a_poll ** 2 / ss2p + r_poll ** 2 / sn2p)        # expanded one-sided log-slab
        score = np.where(vm, score, -1e18)                        # invalid/self/same-line last
        kc = min(kcap, kq)
        if kc < kq:
            top = np.argpartition(-score, kc - 1, axis=1)[:, :kc]  # top-kcap by poll score (unordered)
            rows = np.arange(p)[:, None]
            ii, dd = ii[rows, top], dd[rows, top]
        self_idx = np.arange(p)[:, None]
        valid = (ii < p) & np.isfinite(dd) & (ii != self_idx)
        jj = np.where(valid, ii, 0)
        valid &= line_id[:, None] != line_id[jj]
        return _neighbor_tables_score(pos, tan, kappa, valid, jj, sigma_s_px, sigma_n_px, sigma_z_px, nrm3,
                                      sigma_m_px, min_tilt_deg, pos_threshold, neg_below, pt_weight,
                                      sn2_plus, sn2_minus)
    k = min(kcap, p)
    dd, ii = cKDTree(pos).query(pos, k=k, distance_upper_bound=max_px, workers=-1)
    dd = np.atleast_2d(dd)
    ii = np.atleast_2d(ii)
    self_idx = np.arange(p)[:, None]
    valid = (ii < p) & np.isfinite(dd) & (ii != self_idx)
    jj = np.where(valid, ii, 0)
    valid &= line_id[:, None] != line_id[jj]                      # only DIFFERENT-line pairs
    return _neighbor_tables_score(pos, tan, kappa, valid, jj, sigma_s_px, sigma_n_px, sigma_z_px, nrm3,
                                  sigma_m_px, min_tilt_deg, pos_threshold, neg_below, pt_weight,
                                  sn2_plus, sn2_minus)


def _neighbor_tables_score(pos, tan, kappa, valid, jj, sigma_s_px, sigma_n_px, sigma_z_px, nrm3, sigma_m_px,
                           min_tilt_deg, pos_threshold, neg_below, pt_weight, sn2_plus, sn2_minus):
    """Strict mutual-slab scoring on a PRE-SELECTED candidate set ``jj`` [P,K] (with ``valid`` mask), shared
    by both candidate-selection paths of :func:`_neighbor_tables`. Returns ``jj, pos_w, neg_w``."""
    p = len(pos)
    ss2, sn2 = 2.0 * sigma_s_px ** 2, 2.0 * sigma_n_px ** 2
    # Two-sided across-normal width: when per-point ``sn2_plus/sn2_minus`` (=2σ⁺²/2σ⁻² px²) are given, the
    # slab uses σ⁺ on the +normal side and σ⁻ on the −normal side (squished toward density valleys). Pass
    # them to _slab_logscore as (anchor[:,None], neighbour[jj]) tuples so each point uses its OWN pair.
    sij_n = (sn2_plus[:, None], sn2_minus[:, None]) if sn2_plus is not None else sn2
    sji_n = (sn2_plus[jj], sn2_minus[jj]) if sn2_plus is not None else sn2
    has_z = pos.shape[1] == 3
    if nrm3 is None:                                              # ---- 2-D model: normal = perp(tangent)
        xy = pos[:, :2]
        v = xy[jj] - xy[:, None, :]
        ti = tan[:, None, :]
        ni = np.stack([-ti[..., 1], ti[..., 0]], axis=-1)        # leftward normal (matches signed κ)
        tj = tan[jj]
        nj = np.stack([-tj[..., 1], tj[..., 0]], axis=-1)
        log_sij = _slab_logscore(v, ti, ni, kappa[:, None], ss2, sij_n)
        log_sji = _slab_logscore(-v, tj, nj, kappa[jj], ss2, sji_n)
    else:                                                        # ---- 3-D model: tensor normal + in-sheet
        sm2 = 2.0 * (sigma_m_px if sigma_m_px else sigma_s_px) ** 2
        if min_tilt_deg > 0:
            ok = np.abs(nrm3[:, 2]) <= np.cos(np.radians(min_tilt_deg))   # z-comp (xyz) near vertical
            valid &= ok[:, None] & ok[jj]
        p3 = pos if has_z else np.concatenate([pos, np.zeros((p, 1))], axis=1)
        t3 = np.concatenate([tan, np.zeros((p, 1))], axis=1)     # streamline tangent, z = 0
        m3 = np.cross(nrm3, t3)
        m3 /= np.linalg.norm(m3, axis=1, keepdims=True) + 1e-9
        v = p3[jj] - p3[:, None, :]
        log_sij = _slab_logscore(v, t3[:, None], nrm3[:, None], kappa[:, None], ss2, sij_n, m3[:, None], sm2)
        log_sji = _slab_logscore(-v, t3[jj], nrm3[jj], kappa[jj], ss2, sji_n, m3[jj], sm2)
    logm = log_sij + log_sji
    if has_z and sigma_z_px:
        dz = pos[jj, 2] - pos[:, None, 2]
        logm = logm - dz ** 2 / (2.0 * sigma_z_px ** 2)          # world-z cap (noisy axis)
    mutual = np.exp(np.clip(logm, -60.0, 0.0))                    # S_p(q)·S_q(p) ∈ (0,1]
    if pt_weight is not None:                                     # down-weight unreliable frames
        mutual = mutual * pt_weight[:, None] * pt_weight[jj]      # (e.g. coherence at BOTH points)
    vw = valid.astype(np.float64)
    pos_w = vw * mutual
    if pos_threshold > 0:                                         # positives must be CONFIDENTLY same-sheet
        pos_w = pos_w * (mutual >= pos_threshold)
    neg_w = vw * (1.0 - mutual)
    if neg_below < 1.0:                                           # negatives must be CONFIDENTLY different
        neg_w = neg_w * (mutual <= neg_below)
    return jj.astype(np.int64), pos_w.astype(np.float32), neg_w.astype(np.float32)


def splat_slab_field(pts, normals, jac, sn2p, sn2m, ss2, shape, R, chunk=300):
    """Whole-volume accumulation of the curvature-corrected surface slab (§5.3, the slab-field negatives).

    Each point contributes mass shaped like its own thin asymmetric slab (the §4 anisotropic Gaussian in
    the Darboux frame, curvature-followed), scatter-added into a shared [Z,Y,X] volume on the GPU. Because
    the slabs are thin across the normal, compressed inter-wrap space that reads merely grey in the raw
    density becomes a DEEP valley in this field — the anisotropy manufactures the contrast the raw density
    lacks, so the gap-gate can then fire where it could not before. ``ss2`` is the in-plane 2σ_s² (px²),
    ``R`` the splat half-window (px). Returns the accumulated field as a float32 [Z,Y,X] numpy array."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Z, Y, X = shape
    acc = torch.zeros(Z * Y * X, device=dev)
    rr = torch.arange(-R, R + 1, device=dev)
    oz, oy, ox = torch.meshgrid(rr, rr, rr, indexing="ij")
    off = torch.stack([oz.reshape(-1), oy.reshape(-1), ox.reshape(-1)], 1).float()   # [W3,3]
    offl = off.long()
    P = torch.as_tensor(pts, dtype=torch.float32, device=dev)
    N = torch.as_tensor(normals, dtype=torch.float32, device=dev)
    J = torch.as_tensor(jac, dtype=torch.float32, device=dev)
    SNP = torch.as_tensor(sn2p, dtype=torch.float32, device=dev)
    SNM = torch.as_tensor(sn2m, dtype=torch.float32, device=dev)
    for s in range(0, len(pts), chunk):
        p, n, j = P[s:s + chunk], N[s:s + chunk], J[s:s + chunk]
        snp, snm = SNP[s:s + chunk, None], SNM[s:s + chunk, None]
        v = off[None]                                            # [1,W3,3]
        across = (v * n[:, None, :]).sum(-1)                    # [c,W3]
        vip = v - across[..., None] * n[:, None, :]
        inp2 = (vip * vip).sum(-1).clamp(min=0)
        Jv = torch.einsum("cij,cwj->cwi", j, vip)
        kap = -(vip * Jv).sum(-1) / inp2.clamp(min=1e-9)
        r = (across - 0.5 * kap * inp2) / torch.sqrt(1 + kap ** 2 * inp2)
        sn2 = torch.where(across >= 0, snp, snm)
        S = torch.exp((-(inp2 / ss2 + r ** 2 / sn2)).clamp(min=-60))
        base = p.round().long()[:, None, :] + offl[None]        # [c,W3,3]
        iz, iy, ix = base[..., 0], base[..., 1], base[..., 2]
        ok = (iz >= 0) & (iz < Z) & (iy >= 0) & (iy < Y) & (ix >= 0) & (ix < X)
        flat = ((iz * Y + iy) * X + ix).clamp(0, Z * Y * X - 1)
        acc.scatter_add_(0, torch.where(ok, flat, 0).reshape(-1),
                         torch.where(ok, S, torch.zeros_like(S)).reshape(-1))
    return acc.reshape(Z, Y, X).cpu().numpy()


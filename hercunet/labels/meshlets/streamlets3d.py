"""2.5-D meshlet primitives and the streamlet-clustering entry point.

The live primitive is a 2.5-D meshlet grown on the structure-tensor frame by RK2 streamline
integration: a spine (:func:`grow_2p5d_streamlets`) grown along the in-plane fibre direction and kept
flat to the mid-scale eigenvector so it hugs a single sheet, plus perpendicular ribs
(:func:`grow_ribs_2p5d`) that sweep across the spine to cover the local surface patch. Points sample
the meshlets at a physical ``sample_um`` (:func:`sample_streamlet_points`).

:func:`cluster_streamlets` is the entry the per-window pipeline calls: it builds the mutual-slab /
gap-gated tables and trains the contrastive embedding (``embedding.paperlet``), then labels that
embedding by probeom (``embedding.affinity``). One embedding vector per meshlet — a meshlet's points
move together as one unit. All lengths are µm→px via ``voxel_um``; see ``docs/pseudo-labels.md`` §3–4.
"""

from __future__ import annotations

import os

import numpy as np


# The three orthogonal trace-planes, and the single canonical recto/verso grouping used everywhere:
# recto = the in-(xy)-plane family (drawn as lines in a z-slice), verso = the cross-plane families
# (drawn as dots where they pierce the z-slice).
TRACE_AXES = ("xy", "zx", "zy")
RECTO_AXES = ("xy",)
VERSO_AXES = ("zx", "zy")


def _torch_cuda():
    """Return the ``torch`` module iff a CUDA device is present, else ``None`` (caller falls back to numpy)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch
    except Exception:
        pass
    return None


def _field5d(field, torch, dev, dtype=None):
    """Pack a (Z,Y,X,3) vector field or (Z,Y,X) scalar field into a (1,C,Z,Y,X) tensor for ``grid_sample``
    (channels = vector components, so one call samples all 3 at once). ``dtype`` defaults to float64 (parity
    with the numpy/``map_coordinates`` path); pass float32 to halve GPU memory for large windows — the
    coords/grid in :func:`_gsample` must match this dtype."""
    dtype = dtype or torch.float64
    a = np.ascontiguousarray(field)
    if a.ndim == 4:
        return torch.as_tensor(a, dtype=dtype, device=dev).permute(3, 0, 1, 2)[None]
    return torch.as_tensor(a, dtype=dtype, device=dev)[None, None]


def _gsample(field5d, coords, shape, torch):
    """Trilinear field lookup at ``coords`` [N,3] (z,y,x) — the ``grid_sample`` equivalent of
    ``scipy.ndimage.map_coordinates(order=1, mode='nearest')``: bilinear/trilinear with align_corners=True
    (integer voxel k ↔ normalised 2k/(size-1)-1) and padding_mode='border' (clamp to edge = scipy 'nearest'
    boundary). ``grid_sample`` wants normalised (x,y,z) in [-1,1]; our field/coords are (z,y,x). Returns
    [N,C] (C=3 for a vector field, 1 for scalar)."""
    import torch.nn.functional as F
    Z, Y, X = shape
    z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]
    gx = 2.0 * x / (X - 1) - 1.0                                  # normalise in the COORDS dtype (fp32) for
    gy = 2.0 * y / (Y - 1) - 1.0                                  # sub-voxel precision, THEN cast the [-1,1]
    gz = 2.0 * z / (Z - 1) - 1.0                                  # grid to the field dtype (fp16 keeps ~0.001
    grid = torch.stack([gx, gy, gz], dim=1).view(1, -1, 1, 1, 3)  # here → ~0.3vox, vs ~0.5vox on raw indices)
    if grid.dtype != field5d.dtype:
        grid = grid.to(field5d.dtype)
    out = F.grid_sample(field5d, grid, mode="bilinear", align_corners=True, padding_mode="border")
    out = out[0, :, :, 0, 0].transpose(0, 1)                      # (N, C)
    return out if out.dtype == coords.dtype else out.to(coords.dtype)   # fp16 field → fp32 for the integrator


def _frame_field(block, sigma_tensor, use_gpu):
    if use_gpu:
        try:
            from hercunet.labels.fields.structure_tensor_torch import structure_tensor_frame_torch, torch_available
            if torch_available():
                return structure_tensor_frame_torch(block, 1.0, sigma_tensor)
        except Exception:
            pass
    from hercunet.labels.fields.structure_tensor import structure_tensor_frame
    return structure_tensor_frame(block, 1.0, sigma_tensor)


_AXIS_CODE = {"xy": 0, "zx": 1, "zy": 2}


def grow_streamlets_3d(material, fibre, coherence, voxel_um, phi=None, *,
                       seed_stride_um=40.0, step_um=8.0, max_len_um=450.0, min_coh=0.15,
                       gap_tol=8, max_phi_drift_px=12.0, seed_coh=0.3, momentum=0.0,
                       chunk_seeds=20000, seed=0):
    """TRUE-3-D streamlets, VECTORISED on CPU: grow a streamline along the 3-D fibre tangent ``fibre``
    [Z,Y,X,3] (structure-tensor λ0 eigenvector, in-plane), UNCONFINED to any slice plane — the 3-D lift of
    :func:`tractography._trace_one`. All seeds step in lockstep.

    Regime ported from the validated 2-D tracer: BOTH directions from each seed; RK2 midpoint; the fibre is a
    LINE not a vector, so each step is SIGN-COHERENT (flip the sampled tangent where ``t·prev<0`` — the
    hyperstreamline sign problem); GAP-SHY — coast through up to ``gap_tol`` low ``coherence×material`` steps to
    bridge delamination cracks, trailing coast trimmed; NON-CROSSING via the layer potential ``phi`` (∇φ=n̂), so
    ``|φ−φ_seed| > max_phi_drift_px`` (≈ half a sheet pitch, in voxels) = reached the next winding → stop.

    ``material`` bool [Z,Y,X]; ``coherence`` [Z,Y,X]; ``phi`` [Z,Y,X] or None (no non-crossing gate). Seeds =
    material voxels on a ``seed_stride_um`` lattice with coherence ≥ ``seed_coh``. Returns a list of polylines,
    each ``[M,3]`` (z,y,x float)."""
    from scipy.ndimage import map_coordinates
    Z, Y, X = material.shape
    step = step_um / voxel_um
    max_steps = int(np.ceil(max_len_um / step_um))
    stride = max(1, int(round(seed_stride_um / voxel_um)))
    matf = material.astype(np.float32)

    zz, yy, xx = np.where(material & (coherence >= seed_coh))
    keep = (zz % stride == 0) & (yy % stride == 0) & (xx % stride == 0)
    seeds = np.stack([zz[keep], yy[keep], xx[keep]], 1).astype(np.float64)
    S = len(seeds)
    if S == 0:
        return []

    def s3(vol3, p):
        return np.stack([map_coordinates(vol3[..., c], [p[:, 0], p[:, 1], p[:, 2]], order=1, mode="nearest")
                         for c in range(3)], 1)

    def s1(vol, p):
        return map_coordinates(vol, [p[:, 0], p[:, 1], p[:, 2]], order=1, mode="nearest")

    def integrate(sgn):
        pos = seeds.copy()
        prev = np.zeros((S, 3))
        miss = np.zeros(S)
        active = np.ones(S, bool)
        phi_seed = s1(phi, pos) if phi is not None else None
        traj = np.full((S, max_steps, 3), np.nan)
        onsheet = np.zeros((S, max_steps), bool)
        first = True
        for it in range(max_steps):
            idx = np.where(active)[0]
            if not len(idx):
                break
            p = pos[idx]
            t = s3(fibre, p)
            t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
            if first:
                d = t * sgn
            else:
                d = t.copy()
                flip = (d * prev[idx]).sum(1) < 0
                d[flip] *= -1.0
            pm = p + 0.5 * step * d                                   # RK2 midpoint
            tm = s3(fibre, pm)
            tm /= np.linalg.norm(tm, axis=1, keepdims=True) + 1e-9
            fl = (tm * d).sum(1) < 0
            tm[fl] *= -1.0
            d = tm
            if momentum > 0.0 and not first:                          # stiffness: resist noisy fibre turns
                d = (1.0 - momentum) * d + momentum * prev[idx]       #   (kills coiling in low-anisotropy)
                d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
            g = s1(coherence, p) * s1(matf, p)
            good = g >= min_coh
            traj[idx, it] = p
            onsheet[idx, it] = good
            miss[idx] = np.where(good, 0.0, miss[idx] + 1.0)
            newp = p + step * d
            oob = ((newp < 0).any(1) | (newp[:, 0] > Z - 1) | (newp[:, 1] > Y - 1) | (newp[:, 2] > X - 1))
            cross = np.zeros(len(p), bool)
            if phi is not None:
                cross = np.abs(s1(phi, p) - phi_seed[idx]) > max_phi_drift_px
            die = (miss[idx] > gap_tol) | oob | cross
            prev[idx] = d
            pos[idx] = newp
            active[idx[die]] = False
            first = False
        return traj, onsheet

    tb, ob = integrate(-1.0)
    tf, of = integrate(+1.0)

    def run(traj, on):
        m = ~np.isnan(traj[:, 0])
        pts, onm = traj[m], on[m]
        w = np.where(onm)[0]
        return pts[:w[-1] + 1] if len(w) else np.zeros((0, 3))

    out = []
    for s in range(S):
        b = run(tb[s], ob[s])[::-1]
        f = run(tf[s], of[s])
        pl = np.concatenate([b, f[1:]], 0) if (len(b) and len(f)) else (b if len(b) else f)
        if len(pl) >= 2:
            out.append(pl)
    return out


def grow_2p5d_streamlets(material, normal, fibre, coherence, voxel_um, *,
                         seed_stride_um=40.0, step_um=8.0, max_len_um=450.0, min_coh=0.15,
                         gap_tol=8, seed_coh=0.3, momentum=0.0, seed=0, use_gpu=False,
                         seed_mat=0.5, seed_bbox=None, gpu_fp16=True):
    """TRULY-2.5-D streamlets: a planar spine that traces the intersection of the sheet with a plane Π
    FROZEN at the seed. Π = span(seed normal e₀, seed fibre e₂); its plane-normal is the MIDDLE eigenvector
    e₁ = ``normal × fibre``. The spine may PITCH (bend within Π, following the sheet undulating along e₀) but
    can NEVER YAW (leave Π along e₁) — so projected onto the seed's tangent plane it is a STRAIGHT line along
    the fibre. Unlike :func:`grow_streamlets_3d` (which re-reads the fibre eigenvector each step and wanders
    with its yaw at low-congruency spots), the step direction here uses only the stable LOCAL NORMAL:

        d = normalize( localNormal(p) × n_Π )            (n_Π = normal_seed × fibre_seed, fixed per seed)

    which is ⟂ localNormal (on the sheet surface) AND ⟂ n_Π (inside Π). Where the sheet is flat d → fibre;
    as it undulates d pitches along e₀. This is a 2.5-D per-slice trace cut on a LOCALLY-oriented plane
    (not the global z-slice), so it powers straight through incongruent stretches at ANY sheet orientation.

    BOTH directions from each seed; RK2 midpoint; SIGN-COHERENT; GAP-SHY (coast ``gap_tol`` low
    coherence×material steps). No φ. ``normal``/``fibre`` [Z,Y,X,3]. Returns a list of polylines ``[M,3]``."""
    from scipy.ndimage import map_coordinates
    Z, Y, X = material.shape
    step = step_um / voxel_um
    max_steps = int(np.ceil(max_len_um / step_um))
    stride = max(1, int(round(seed_stride_um / voxel_um)))
    matf = material.astype(np.float32)

    # material may be a CONTINUOUS field (e.g. ∇φ surface prob 0..1) OR a bool mask; seed where it clears
    # ``seed_mat`` (bool masks pass at seed_mat<=1). ``seed_bbox`` = (z0,z1,y0,y1,x0,x1) in field voxels
    # restricts seed PLACEMENT to a sub-box (e.g. a tile's core) while tracing still uses the full field/halo.
    seedmask = (matf >= seed_mat) & (coherence >= seed_coh)
    if seed_bbox is not None:
        bz0, bz1, by0, by1, bx0, bx1 = seed_bbox
        bb = np.zeros(seedmask.shape, bool)
        bb[bz0:bz1, by0:by1, bx0:bx1] = True
        seedmask &= bb
    zz, yy, xx = np.where(seedmask)
    keep = (zz % stride == 0) & (yy % stride == 0) & (xx % stride == 0)
    seeds = np.stack([zz[keep], yy[keep], xx[keep]], 1).astype(np.float64)
    S = len(seeds)
    if S == 0:
        return []

    def s3(vol3, p):
        return np.stack([map_coordinates(vol3[..., c], [p[:, 0], p[:, 1], p[:, 2]], order=1, mode="nearest")
                         for c in range(3)], 1)

    def s1(vol, p):
        return map_coordinates(vol, [p[:, 0], p[:, 1], p[:, 2]], order=1, mode="nearest")

    # Π's plane-normal per seed, frozen: n_Π = normal_seed × fibre_seed (= middle eigenvector e₁)
    n0 = s3(normal, seeds); n0 /= np.linalg.norm(n0, axis=1, keepdims=True) + 1e-9
    f0 = s3(fibre, seeds); f0 /= np.linalg.norm(f0, axis=1, keepdims=True) + 1e-9
    n_pi = np.cross(n0, f0); n_pi /= np.linalg.norm(n_pi, axis=1, keepdims=True) + 1e-9

    def indir(p, npi):
        """step direction confined to Π: d = localNormal(p) × n_Π (projected/normalized)."""
        m = s3(normal, p); m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
        d = np.cross(m, npi)
        d -= (d * npi).sum(1, keepdims=True) * npi                # strict Π projection (kill float drift)
        nn = np.linalg.norm(d, axis=1, keepdims=True)
        return d, nn[:, 0]

    def integrate(sgn):
        pos = seeds.copy()
        prev = np.zeros((S, 3))
        miss = np.zeros(S)
        active = np.ones(S, bool)
        traj = np.full((S, max_steps, 3), np.nan)
        onsheet = np.zeros((S, max_steps), bool)
        first = True
        for it in range(max_steps):
            idx = np.where(active)[0]
            if not len(idx):
                break
            p = pos[idx]
            npi = n_pi[idx]
            d, nd = indir(p, npi)
            deg = nd < 1e-6                                       # sheet ⟂ Π (edge-on): coast on prev
            if first:
                d *= sgn
                d[deg] = 0.0
            else:
                flip = (d * prev[idx]).sum(1) < 0
                d[flip] *= -1.0
                d[deg] = prev[idx][deg]
            pm = p + 0.5 * step * d                               # RK2 midpoint
            dm, ndm = indir(pm, npi)
            fl = (dm * d).sum(1) < 0
            dm[fl] *= -1.0
            ok = ndm >= 1e-6
            d[ok] = dm[ok]
            if momentum > 0.0 and not first:                      # stiffness (kills jitter in flat runs)
                d = (1.0 - momentum) * d + momentum * prev[idx]
                d -= (d * npi).sum(1, keepdims=True) * npi
                d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
            g = s1(coherence, p) * s1(matf, p)
            good = g >= min_coh
            traj[idx, it] = p
            onsheet[idx, it] = good
            miss[idx] = np.where(good, 0.0, miss[idx] + 1.0)
            newp = p + step * d
            oob = ((newp < 0).any(1) | (newp[:, 0] > Z - 1) | (newp[:, 1] > Y - 1) | (newp[:, 2] > X - 1))
            die = (miss[idx] > gap_tol) | oob
            prev[idx] = d
            pos[idx] = newp
            active[idx[die]] = False
            first = False
        return traj, onsheet

    def integrate_torch(sgn, torch):
        """GPU twin of :func:`integrate`: same 2.5-D step (localNormal × frozen n_Π, RK2, sign-coherent,
        gap-shy), all S streamlets stepped in lockstep with an ``active`` mask; ``grid_sample`` replaces
        ``map_coordinates``. Returns numpy ``(traj, onsheet)`` so the CPU polyline assembly below is reused."""
        dev = torch.device("cuda")
        ft = torch.float32                                               # coords/integration stay fp32 (precision)
        fdt = torch.float16 if gpu_fp16 else torch.float32              # fields fp16 → half GPU mem (bigger windows)
        shape = (Z, Y, X)
        normal5 = _field5d(normal, torch, dev, fdt)
        coh5 = _field5d(coherence, torch, dev, fdt)
        mat5 = _field5d(matf, torch, dev, fdt)
        npi_t = torch.as_tensor(n_pi, dtype=ft, device=dev)              # frozen per-seed Π-normal (numpy reuse)
        pos = torch.as_tensor(seeds, dtype=ft, device=dev)
        prev = torch.zeros((S, 3), dtype=ft, device=dev)
        miss = torch.zeros(S, dtype=ft, device=dev)
        active = torch.ones(S, dtype=torch.bool, device=dev)
        traj = torch.full((S, max_steps, 3), float("nan"), dtype=ft, device=dev)
        onsheet = torch.zeros((S, max_steps), dtype=torch.bool, device=dev)

        def indir_t(p):
            m = _gsample(normal5, p, shape, torch)
            m = m / (m.norm(dim=1, keepdim=True) + 1e-9)
            d = torch.linalg.cross(m, npi_t, dim=1)
            d = d - (d * npi_t).sum(1, keepdim=True) * npi_t
            return d, d.norm(dim=1)

        first = True
        for it in range(max_steps):
            if not bool(active.any()):
                break
            d, nd = indir_t(pos)
            deg = nd < 1e-6
            if first:
                d = d * sgn
                d = torch.where(deg[:, None], torch.zeros_like(d), d)
            else:
                flip = (d * prev).sum(1) < 0
                d = torch.where(flip[:, None], -d, d)
                d = torch.where(deg[:, None], prev, d)
            pm = pos + 0.5 * step * d
            dm, ndm = indir_t(pm)
            fl = (dm * d).sum(1) < 0
            dm = torch.where(fl[:, None], -dm, dm)
            ok = ndm >= 1e-6
            d = torch.where(ok[:, None], dm, d)
            g = _gsample(coh5, pos, shape, torch)[:, 0] * _gsample(mat5, pos, shape, torch)[:, 0]
            good = g >= min_coh
            traj[:, it] = torch.where(active[:, None], pos, traj[:, it])
            onsheet[:, it] = torch.where(active, good, onsheet[:, it])
            miss = torch.where(active, torch.where(good, torch.zeros_like(miss), miss + 1.0), miss)
            newp = pos + step * d
            oob = (newp < 0).any(1) | (newp[:, 0] > Z - 1) | (newp[:, 1] > Y - 1) | (newp[:, 2] > X - 1)
            die = (miss > gap_tol) | oob
            prev = torch.where(active[:, None], d, prev)
            pos = torch.where(active[:, None], newp, pos)
            active = active & ~die
            first = False
        return traj.cpu().numpy(), onsheet.cpu().numpy()

    torch = _torch_cuda() if use_gpu else None
    if torch is not None:
        tb, ob = integrate_torch(-1.0, torch)
        tf, of = integrate_torch(+1.0, torch)
    else:
        tb, ob = integrate(-1.0)
        tf, of = integrate(+1.0)

    def run(traj, on):
        m = ~np.isnan(traj[:, 0])
        pts, onm = traj[m], on[m]
        w = np.where(onm)[0]
        return pts[:w[-1] + 1] if len(w) else np.zeros((0, 3))

    out = []
    for s in range(S):
        b = run(tb[s], ob[s])[::-1]
        f = run(tf[s], of[s])
        pl = np.concatenate([b, f[1:]], 0) if (len(b) and len(f)) else (b if len(b) else f)
        if len(pl) >= 2:
            out.append(pl)
    return out


def grow_ribs_2p5d(spines, normal, fibre, gate_field, material, voxel_um, *, rib_spacing_um=20.0,
                   step_um=8.0, max_len_um=100.0, min_gate=0.0, gap_tol=12, momentum=0.0, use_gpu=False,
                   gpu_fp16=True):
    """RIBS = the dual of the spine. From points sampled every ``rib_spacing_um`` along each spine, grow a
    2.5-D streamlet along the MIDDLE eigenvector e₁ (across-fibre, perpendicular to the spine's e₂ heading),
    confined to Π_rib = span(e₀ normal, e₁) whose plane-normal is the FIBRE e₂. Same stable rule as the spine
    with the plane-normal swapped: ``d = localNormal(p) × fibre_seed`` — flat → e₁, pitches along e₀, never
    yaws. Gate on ``gate_field`` (pass a PLANARITY field, not fibre-coherence, so ribs are LESS SHY — they
    follow the sheet plane straight across low fibre-congruency). Returns ``(ribs, owner)`` where ``owner[i]``
    is the parent-spine index. ``normal``/``fibre`` [Z,Y,X,3]; polylines ``[M,3]`` (z,y,x)."""
    from scipy.ndimage import map_coordinates
    Z, Y, X = material.shape
    step = step_um / voxel_um
    max_steps = int(np.ceil(max_len_um / step_um))
    matf = material.astype(np.float32)

    d0 = max(1e-3, rib_spacing_um / voxel_um)
    seed_list, owner = [], []
    for si, p in enumerate(spines):                                  # rib seeds = arc-length samples of the spine
        p = np.asarray(p, np.float64)
        if len(p) < 2:
            continue
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        targ = np.arange(0.0, max(arc[-1], 1e-6), d0)
        qs = np.stack([np.interp(targ, arc, p[:, c]) for c in range(3)], 1)
        seed_list.append(qs)
        owner.extend([si] * len(qs))
    if not seed_list:
        return [], []
    seeds = np.concatenate(seed_list, 0)
    owner = np.asarray(owner)
    S = len(seeds)

    def s3(vol3, p):
        return np.stack([map_coordinates(vol3[..., c], [p[:, 0], p[:, 1], p[:, 2]], order=1, mode="nearest")
                         for c in range(3)], 1)

    def s1(vol, p):
        return map_coordinates(vol, [p[:, 0], p[:, 1], p[:, 2]], order=1, mode="nearest")

    n_pi = s3(fibre, seeds)                                           # Π_rib plane-normal = fibre e₂ (frozen)
    n_pi /= np.linalg.norm(n_pi, axis=1, keepdims=True) + 1e-9

    def indir(p, npi):
        m = s3(normal, p); m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
        d = np.cross(m, npi)                                          # ⟂ localNormal (on sheet) & ⟂ fibre (in Π_rib)
        d -= (d * npi).sum(1, keepdims=True) * npi
        nn = np.linalg.norm(d, axis=1, keepdims=True)
        return d, nn[:, 0]

    def integrate(sgn):
        pos = seeds.copy()
        prev = np.zeros((S, 3))
        miss = np.zeros(S)
        active = np.ones(S, bool)
        traj = np.full((S, max_steps, 3), np.nan)
        onsheet = np.zeros((S, max_steps), bool)
        first = True
        for it in range(max_steps):
            idx = np.where(active)[0]
            if not len(idx):
                break
            p = pos[idx]
            npi = n_pi[idx]
            d, nd = indir(p, npi)
            deg = nd < 1e-6
            if first:
                d *= sgn
                d[deg] = 0.0
            else:
                flip = (d * prev[idx]).sum(1) < 0
                d[flip] *= -1.0
                d[deg] = prev[idx][deg]
            pm = p + 0.5 * step * d
            dm, ndm = indir(pm, npi)
            fl = (dm * d).sum(1) < 0
            dm[fl] *= -1.0
            ok = ndm >= 1e-6
            d[ok] = dm[ok]
            if momentum > 0.0 and not first:
                d = (1.0 - momentum) * d + momentum * prev[idx]
                d -= (d * npi).sum(1, keepdims=True) * npi
                d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
            g = s1(gate_field, p) * s1(matf, p)
            good = g >= min_gate
            traj[idx, it] = p
            onsheet[idx, it] = good
            miss[idx] = np.where(good, 0.0, miss[idx] + 1.0)
            newp = p + step * d
            oob = ((newp < 0).any(1) | (newp[:, 0] > Z - 1) | (newp[:, 1] > Y - 1) | (newp[:, 2] > X - 1))
            die = (miss[idx] > gap_tol) | oob
            prev[idx] = d
            pos[idx] = newp
            active[idx[die]] = False
            first = False
        return traj, onsheet

    def integrate_torch(sgn, torch):
        """GPU twin of :func:`integrate` for ribs: step along localNormal × frozen fibre-normal, gated on
        ``gate_field`` (min_gate). All S rib-seeds in lockstep; ``grid_sample`` field lookups. Returns numpy."""
        dev = torch.device("cuda")
        ft = torch.float32                                               # coords/integration stay fp32 (precision)
        fdt = torch.float16 if gpu_fp16 else torch.float32              # fields fp16 → half GPU mem (bigger windows)
        shape = (Z, Y, X)
        normal5 = _field5d(normal, torch, dev, fdt)
        gate5 = _field5d(gate_field, torch, dev, fdt)
        mat5 = _field5d(matf, torch, dev, fdt)
        npi_t = torch.as_tensor(n_pi, dtype=ft, device=dev)              # frozen per-seed Π_rib normal (=fibre)
        pos = torch.as_tensor(seeds, dtype=ft, device=dev)
        prev = torch.zeros((S, 3), dtype=ft, device=dev)
        miss = torch.zeros(S, dtype=ft, device=dev)
        active = torch.ones(S, dtype=torch.bool, device=dev)
        traj = torch.full((S, max_steps, 3), float("nan"), dtype=ft, device=dev)
        onsheet = torch.zeros((S, max_steps), dtype=torch.bool, device=dev)

        def indir_t(p):
            m = _gsample(normal5, p, shape, torch)
            m = m / (m.norm(dim=1, keepdim=True) + 1e-9)
            d = torch.linalg.cross(m, npi_t, dim=1)
            d = d - (d * npi_t).sum(1, keepdim=True) * npi_t
            return d, d.norm(dim=1)

        first = True
        for it in range(max_steps):
            if not bool(active.any()):
                break
            d, nd = indir_t(pos)
            deg = nd < 1e-6
            if first:
                d = d * sgn
                d = torch.where(deg[:, None], torch.zeros_like(d), d)
            else:
                flip = (d * prev).sum(1) < 0
                d = torch.where(flip[:, None], -d, d)
                d = torch.where(deg[:, None], prev, d)
            pm = pos + 0.5 * step * d
            dm, ndm = indir_t(pm)
            fl = (dm * d).sum(1) < 0
            dm = torch.where(fl[:, None], -dm, dm)
            ok = ndm >= 1e-6
            d = torch.where(ok[:, None], dm, d)
            g = _gsample(gate5, pos, shape, torch)[:, 0] * _gsample(mat5, pos, shape, torch)[:, 0]
            good = g >= min_gate
            traj[:, it] = torch.where(active[:, None], pos, traj[:, it])
            onsheet[:, it] = torch.where(active, good, onsheet[:, it])
            miss = torch.where(active, torch.where(good, torch.zeros_like(miss), miss + 1.0), miss)
            newp = pos + step * d
            oob = (newp < 0).any(1) | (newp[:, 0] > Z - 1) | (newp[:, 1] > Y - 1) | (newp[:, 2] > X - 1)
            die = (miss > gap_tol) | oob
            prev = torch.where(active[:, None], d, prev)
            pos = torch.where(active[:, None], newp, pos)
            active = active & ~die
            first = False
        return traj.cpu().numpy(), onsheet.cpu().numpy()

    torch = _torch_cuda() if use_gpu else None
    if torch is not None:
        tb, ob = integrate_torch(-1.0, torch)
        tf, of = integrate_torch(+1.0, torch)
    else:
        tb, ob = integrate(-1.0)
        tf, of = integrate(+1.0)

    def run(traj, on):
        m = ~np.isnan(traj[:, 0])
        pts, onm = traj[m], on[m]
        w = np.where(onm)[0]
        return pts[:w[-1] + 1] if len(w) else np.zeros((0, 3))

    ribs, own = [], []
    for s in range(S):
        b = run(tb[s], ob[s])[::-1]
        f = run(tf[s], of[s])
        pl = np.concatenate([b, f[1:]], 0) if (len(b) and len(f)) else (b if len(b) else f)
        if len(pl) >= 2:
            ribs.append(pl)
            own.append(int(owner[s]))
    return ribs, own


def sample_streamlet_points(streamlets, voxel_um, sample_um=20.0):
    """Arc-length resample each streamlet at ``sample_um`` spacing → ``pts [P,3]`` (z,y,x), ``sid [P]``
    (streamlet id — the per-unit binding: all points of one streamlet share its embedding vector), and
    ``cent [S,3]`` per-streamlet centroid."""
    d = max(1e-3, sample_um / voxel_um)
    pts, sid = [], []
    cent = np.zeros((len(streamlets), 3))
    for i, p in enumerate(streamlets):
        p = np.asarray(p, np.float64)
        cent[i] = p.mean(0)
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        if arc[-1] < d:
            s = p[:1]
        else:
            targ = np.arange(0.0, arc[-1], d)
            s = np.stack([np.interp(targ, arc, p[:, c]) for c in range(3)], 1)
        pts.append(s)
        sid.append(np.full(len(s), i))
    return np.concatenate(pts), np.concatenate(sid).astype(int), cent


def cluster_streamlets(streamlets, bced, ff, voxel_um, *, sample_um=20.0, sigma_n_um=25.0,
                       dim=32, pos_pull=1.0, min_cluster_size=20, min_samples=None, cluster_selection_method="eom",
                       presampled=None, n_units=None, ptype=None, recto_anchor=False, seed=0,
                       loss_mode="triplet", vic_inv=25.0, vic_var=25.0, vic_cov=1.0,
                       samples_per_point=150, use_gpu=False, verbose=True, **table_kwargs):
    """End-to-end: 3-D streamlets → per-STREAMLET embedding → eom. Points sample the streamlets; the streamlet
    id binds each unit (one vector per streamlet — its points move together as one, the disambiguation). Cross-
    streamlet positives from the mutual 3-D surface slab (``paperlet_tables``, sheet-based so wobble-robust),
    corridor across-normal negatives, ordinal triplet + restored unconditional ``pos_pull``. Returns
    ``(labels [S], n_clusters, info)`` with ``info['E']`` and the sampled ``pts/sid``."""
    import time as _time
    from hercunet.labels.embedding.paperlet import paperlet_tables, train_paperlet_embedding
    from hercunet.labels.embedding.affinity import labels_from_embedding
    from hercunet.labels.selection.membership import measure_sigma_n_3d, normal_jacobian_batch
    if presampled is not None:                                       # ribbon points (pts, rid, cent) supplied
        pts, sid, cent = presampled
        S = int(n_units if n_units is not None else (sid.max() + 1))
    else:
        S = len(streamlets)
        pts, sid, cent = sample_streamlet_points(streamlets, voxel_um, sample_um)
    if len(pts) < 8:
        return np.full(S, -1, int), 0, {"n_points": len(pts)}
    nf = ff["normal"] / (np.linalg.norm(ff["normal"], axis=-1, keepdims=True) + 1e-9)
    Z, Y, X = nf.shape[:3]
    ib = ((pts[:, 0] >= 0) & (pts[:, 0] <= Z - 1) & (pts[:, 1] >= 0) & (pts[:, 1] <= Y - 1)
          & (pts[:, 2] >= 0) & (pts[:, 2] <= X - 1))                   # ribbon offsets can leave the block
    pts, sid = pts[ib], sid[ib]
    if ptype is not None:
        ptype = np.asarray(ptype)[ib]
    anchor_idx = (np.where(ptype == 0)[0] if (ptype is not None and recto_anchor) else None)  # recto-only anchor rows

    _t = _time.time()
    pi = np.round(pts).astype(int)
    pi[:, 0] = pi[:, 0].clip(0, Z - 1); pi[:, 1] = pi[:, 1].clip(0, Y - 1); pi[:, 2] = pi[:, 2].clip(0, X - 1)
    normals = nf[pi[:, 0], pi[:, 1], pi[:, 2]]
    jac = normal_jacobian_batch(nf, pts, normals)
    sp, sm = measure_sigma_n_3d(bced, pts, normals, voxel_um, sigma_max_um=sigma_n_um, use_gpu=use_gpu)
    sn2p, sn2m = 2.0 * np.maximum(sp / voxel_um, 1.0) ** 2, 2.0 * np.maximum(sm / voxel_um, 1.0) ** 2
    if verbose:
        print(f"  [streamlet-sample] {len(pts)} pts from {S} streamlets ({sample_um:.0f}µm)  "
              f"[t] frame-prep {_time.time()-_t:.1f}s", flush=True)
    _t = _time.time()
    pos_nbr, pos_w, neg_nbr, neg_w = paperlet_tables(pts, sid, normals, jac, sn2p, sn2m, voxel_um,
                                                     seed=seed, anchor_idx=anchor_idx, use_gpu=use_gpu,
                                                     bced=(bced if loss_mode != "vicreg" else None),
                                                     **table_kwargs)
    _t_tables = _time.time() - _t
    if verbose:
        _nw = neg_w[neg_nbr >= 0]
        print(f"  [streamlet-tables] pos cov {100*float((pos_w.sum(1)>0).mean()):.0f}%  "
              f"neg cov {100*float((neg_nbr>=0).any(1).mean()):.0f}%  "
              f"neg_w p50 {float(np.median(_nw)) if _nw.size else 0:.3f}  [t] tables {_t_tables:.1f}s", flush=True)
    n_pos_avail = n_pos_scored = None
    if os.environ.get("HERCU_POS_AVAIL"):                             # DEBUG: pre-cap positive pool per point (not training)
        from hercunet.labels.embedding.paperlet import pos_available_count
        _tau = float(os.environ.get("HERCU_POS_TAU", "0.05")); _ak = int(os.environ.get("HERCU_AVAIL_K", "256"))
        _tk = {k: table_kwargs[k] for k in ("max_sheet_um", "sigma_s_um", "width_lambda_um", "width_steep_um") if k in table_kwargs}
        n_pos_avail, n_pos_scored = pos_available_count(pts, sid, normals, jac, sn2p, sn2m, voxel_um,
                                                        tau=_tau, avail_k=_ak, anchor_idx=anchor_idx, use_gpu=use_gpu, **_tk)
        if verbose:
            _na = n_pos_avail[n_pos_avail >= 0]
            print(f"  [pos-avail τ={_tau} k={_ak}] pool/anchor p50 {np.median(_na):.0f} p90 {np.percentile(_na,90):.0f} "
                  f"max {int(_na.max())} | saturated(=k) {100*float((n_pos_scored>=_ak).mean()):.0f}%", flush=True)
    _t = _time.time()
    E = train_paperlet_embedding(S, pts, sid, pos_nbr, pos_w, neg_nbr, neg_w, cent, voxel_um,
                                 dim=dim, pos_pull=pos_pull, seed=seed, verbose=verbose,
                                 ptype=ptype, anchor_pts=anchor_idx, loss_mode=loss_mode,
                                 samples_per_point=samples_per_point,
                                 vic_inv=vic_inv, vic_var=vic_var, vic_cov=vic_cov)
    _t_train = _time.time() - _t; _t = _time.time()
    prop_dist = np.zeros(len(E), np.float32)
    if cluster_selection_method == "probeom":                        # size-penalized EOM + embedding propagation
        from hercunet.labels.embedding.affinity import probeom_labels
        labels, prop_dist = probeom_labels(E, min_cluster_size=min_cluster_size, min_samples=min_samples)
        sel_name = "probeom"
    else:
        labels = labels_from_embedding(E, min_cluster_size=min_cluster_size, min_samples=min_samples,
                                       cluster_selection_method=cluster_selection_method)
        sel_name = cluster_selection_method
    if verbose:
        print(f"  [t] train {_t_train:.1f}s   {sel_name} {_time.time()-_t:.1f}s", flush=True)
    n = len(set(int(x) for x in labels if x >= 0))
    return labels, n, {"E": E, "pts": pts, "sid": sid, "cent": cent, "n_points": len(pts),
                       "neg_nbr": neg_nbr, "pos_nbr": pos_nbr, "prop_dist": prop_dist,
                       "ptype": (np.asarray(ptype) if ptype is not None else None),  # 0=recto/spine 1=verso/rib (ib-masked)
                       "n_pos_avail": n_pos_avail, "n_pos_scored": n_pos_scored}  # DEBUG pre-cap positive pool (env-gated)


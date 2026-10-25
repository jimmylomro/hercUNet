"""Mesh → volume rasterisers for the HercuLabels pseudo-label GEOMETRY stage: ∇φ / owner-field builders
(``build_gradphi`` / ``build_gradphi_edt``), confidence volumes, and mesh↔frame-field congruence.

The mesh FITTING primitives moved to the shared :mod:`hercunet.meshes.fit` (re-exported below for back-compat);
this module keeps only the herculabels-side rasterisers that turn fitted meshes into the dense label fields.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from hercunet.meshes.fit import (  # noqa: F401 (re-export: fitting primitives live in the shared module now)
    _pick_device, slab_logscore_batch, init_sheet_grid, mesh_vertex_normals, compute_mesh_width,
    grid_triangles, _slab_pot_torch, fit_sheet_mesh, _fit_all_gpu, mesh_vertex_jacobian, fit_sheet_meshes)


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
    own_t = torch.as_tensor(own_lab, device=dev)
    cap_t = torch.as_tensor(sn_cap, dtype=torch.float32, device=dev)
    # NB: the full neighbour-index array `idx` [Ncand, kk] int64 is ~2.3GB for a whole window — do NOT move it
    # to the GPU wholesale (that was the build_gradphi OOM). Keep it on CPU; move only each chunk's slice.
    out = np.empty(len(cand), np.float32)
    for s in range(0, len(cand), chunk):
        e = min(s + chunk, len(cand))
        ix = torch.as_tensor(idx[s:e], dtype=torch.long, device=dev)
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


def build_gradphi_edt(meshes, shape, voxel_um, *, sigma_vox=3.0, power=3.0, upsample=3,
                      band_sigmas=3.0, min_mag=0.05, return_owner_full=False):
    """FAST ∇φ decoder — **the training-label emitter** consumed by ``hercunet train export-labels``.

    Same stored ``meshes`` as :func:`build_gradphi`, but ~10× faster (1–1.8 s vs 9–14 s), **CPU-only** (EDT +
    numpy, NO KDTree, NO GPU → runs on dataloader workers without touching the training GPU), and cleaner:

      • **Fixed σ** (``sigma_vox``, width-INDEPENDENT) → ``|∇φ|`` is a genuine phase gradient. The slab decoder
        scaled σ with sheet width, so ``∫|∇φ|`` across a sheet grew with thickness (not a real dφ/dn) AND the
        width-scaled ``reach`` dilated candidates to ~97% of the volume (the slowness). Here thickness lives on
        its OWN channel (``width3``), not in ``|∇φ|``.
      • **Non-crossing single owner** — a Euclidean distance transform assigns every voxel to its NEAREST medial
        surface; the boundary between two sheets is the equidistant midplane. Guaranteed by construction (the
        only failure mode is an incomplete surface rasterization → keep ``upsample`` dense enough to be hole-free).

    Rasterize each sheet's dense medials → occupancy + owner (label/normal/half-width); ``distance_transform_edt``
    gives, at every voxel, distance ``d`` to the nearest surface and that surface's coords (→ gather the owner).
    ``|∇φ| = exp(-(d/σ)^power)`` inside ``d ≤ band_sigmas·σ``.

    Returns ``(mag [Z,H,W] float32, lab3 [Z,H,W] int32 sheet-id/-1, nrm3 [3,Z,H,W] float16,
    width3 [Z,H,W] float32 µm)``; with ``return_owner_full`` also the UNBOUNDED nearest-medial Voronoi id
    ``owner_full [Z,H,W] int32`` (every voxel), used by the surface-label collapsed-sheet ignore. Intersection /
    low-confidence is NOT recomputed here — it comes from the stored ``conf`` channels (σ-independent)."""
    Z, H, W = shape
    occ = np.zeros(shape, bool)
    ownlab = np.full(shape, -1, np.int32)
    ownnrm = np.zeros((3,) + shape, np.float32)
    ownhw = np.zeros(shape, np.float32)
    for c, m in meshes.items():
        dm, dn, dhw, _ = _dense_medials(m, voxel_um, upsample)
        vi = np.clip(np.round(dm).astype(int), 0, [Z - 1, H - 1, W - 1])
        z, y, x = vi[:, 0], vi[:, 1], vi[:, 2]
        occ[z, y, x] = True; ownlab[z, y, x] = int(c); ownhw[z, y, x] = dhw
        dn = dn / (np.linalg.norm(dn, axis=1, keepdims=True) + 1e-9)
        ownnrm[0, z, y, x] = dn[:, 0]; ownnrm[1, z, y, x] = dn[:, 1]; ownnrm[2, z, y, x] = dn[:, 2]
    d, (iz, iy, ix) = ndi.distance_transform_edt(~occ, return_indices=True)
    owner_full = ownlab[iz, iy, ix].astype(np.int32)                 # nearest medial for EVERY voxel (UNBOUNDED Voronoi)
    on = (d <= band_sigmas * sigma_vox)
    mag = np.where(on, np.exp(-(d / sigma_vox) ** power), 0.0).astype(np.float32)
    on = on & (mag >= min_mag)
    lab3 = np.where(on, owner_full, -1).astype(np.int32)             # band-limited owner (= owner_full inside the band)
    width3 = np.where(on, ownhw[iz, iy, ix] * 2.0 * voxel_um, 0.0).astype(np.float32)
    nrm3 = (ownnrm[:, iz, iy, ix] * on[None]).astype(np.float16)
    if return_owner_full:
        return mag, lab3, nrm3, width3, owner_full
    return mag, lab3, nrm3, width3


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

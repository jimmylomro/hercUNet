"""Contrastive embedding of the meshlet primitives — the neighbour tables and the training loop.

One 8-D embedding vector per meshlet (points only SAMPLE it). :func:`paperlet_tables` builds the
tables: positives from the MUTUAL 3-D surface slab (``selection.membership.surface_slab_logscore``
with the adaptive two-sided σ_n from ``measure_sigma_n_3d``), hard negatives across the sheet normal
(``selection.membership`` gap-gated / corridor grid). :func:`train_paperlet_embedding` then trains the
embedding with an ordinal triplet objective plus the faint VICReg variance/covariance terms and a
recto/verso positive balance.

The trained embedding ``E`` is clustered into per-sheet labels downstream by ``embedding.affinity``
(probeom). Euclidean and ordinal by construction — never spherical (a sphere collapses the ordinal
magnitude ladder). See ``submission/writeup/herculabels.md`` §6 for the method.
"""
from __future__ import annotations

import numpy as np


def _surface_slab_batch(v, n, J, ss2, sn2):
    """Batched 3-D surface-slab log S over a [P,K] candidate grid — the [P,K] lift of
    ``sheet_membership.surface_slab_logscore``. ``v`` [P,K,3] anchor→candidate (z,y,x); ``n`` [P,3] anchor
    normal; ``J`` [P,3,3] anchor Jacobian ∂n̂/∂x; ``sn2`` = ``(sn2_plus [P], sn2_minus [P])`` two-sided across
    width (2σ² px²) or a scalar. Isotropic σ_s in-plane, curvature-corrected across the normal (Weingarten
    κ=−(û·Jû)). Returns log S [P,K]."""
    across = np.einsum("pkc,pc->pk", v, n)                        # signed across-normal
    vip = v - across[..., None] * n[:, None, :]                  # in-plane part
    inp2 = np.maximum((vip * vip).sum(-1), 0.0)                  # ρ²
    Jv = np.einsum("pij,pkj->pki", J, vip)                       # J·vip per row
    kappa = -(vip * Jv).sum(-1) / np.maximum(inp2, 1e-9)        # normal curvature κ=−(û·Jû)
    r = (across - 0.5 * kappa * inp2) / np.sqrt(1.0 + kappa ** 2 * inp2)
    if isinstance(sn2, tuple):
        sn2 = np.where(across >= 0.0, sn2[0][:, None], sn2[1][:, None])
    return -(inp2 / ss2 + r ** 2 / sn2)


def _surface_slab_batch_torch(v, n, J, ss2, sn2, torch):
    """Torch/CUDA twin of :func:`_surface_slab_batch` — identical einsum/elementwise math (float64) so the
    GPU positive tables match the numpy path to machine precision. ``sn2`` = scalar or ``(sn2_plus, sn2_minus)``
    tensors [P]."""
    across = torch.einsum("pkc,pc->pk", v, n)
    vip = v - across[..., None] * n[:, None, :]
    inp2 = torch.clamp((vip * vip).sum(-1), min=0.0)
    Jv = torch.einsum("pij,pkj->pki", J, vip)
    kappa = -(vip * Jv).sum(-1) / torch.clamp(inp2, min=1e-9)
    r = (across - 0.5 * kappa * inp2) / torch.sqrt(1.0 + kappa ** 2 * inp2)
    if isinstance(sn2, tuple):
        sn2 = torch.where(across >= 0.0, sn2[0][:, None], sn2[1][:, None])
    return -(inp2 / ss2 + r ** 2 / sn2)


def paperlet_tables(pts, pap_id, normals, jac, sn2_plus, sn2_minus, voxel_um,
                    max_sheet_um=90.0, sigma_s_um=90.0, kcap=48,
                    neg_cell_um=75.0, neg_dist_um=200.0, kneg=32, seed=0, chunk=50000,
                    poll_um=None, sigma_s_poll_um=None, sigma_n_poll_um=60.0, poll_k=256,
                    anchor_idx=None, width_lambda_um=None, width_steep_um=20.0, use_gpu=False,
                    bced=None, neg_tau=0.30, neg_reach_um=120.0, neg_sigma_ip_um=40.0,
                    neg_lam_decay_um=60.0, neg_margin_k=2.5, neg_gate_floor_um=15.0,
                    slab_negatives=False):
    """Positive (mutual 3-D surface slab) + negative (averaged-normal corridor) sampling tables for the
    paperlet embedding. Positives: KDTree candidates within ``max_sheet_um``; a pair is positive ∝
    ``S_i(j)·S_j(i)`` (mutual — each slab must contain the other, symmetric, rejects the adjacent winding
    structurally), DIFFERENT paperlet only. Negatives: :func:`corridor_negatives_3d` (cell across the normal ⇒
    adjacent winding). Returns ``pos_nbr [P,kcap], pos_w [P,kcap], neg_nbr [P,kneg], neg_absr [P,kneg]`` (nbr
    index into ``pts``; −1 pad). ``normals`` [P,3] unit (z,y,x); ``jac`` [P,3,3]; ``sn2_plus/minus`` [P] px².

    ``poll_um`` enables the 2.5-D TWO-STAGE anisotropic candidate selection (ported from
    ``sheet_membership._neighbor_tables``): gather up to ``poll_k`` within the larger ``poll_um`` radius, rank
    them with an EXPANDED poll slab (LONG in-plane ``sigma_s_poll_um`` + WIDE across-normal ``sigma_n_poll_um``
    ≈1 pitch — wide enough to keep the adjacent winding visible so it can be beaten, not silently grabbed), keep
    the top-``kcap``, THEN strict-score those. Default (None) = the leak-prone nearest-``kcap`` selection."""
    from scipy.spatial import cKDTree
    from hercunet.labels.selection.membership import corridor_negatives_3d, gap_gated_negatives_3d
    P = len(pts)
    aidx = np.arange(P) if anchor_idx is None else np.asarray(anchor_idx)  # ROWS = anchors (recto-only in v1) —
    A = len(aidx)                                                          # verso rows never anchor, so skip them
    apts, anrm, ajac = pts[aidx], normals[aidx], jac[aidx]                 # ANCHOR-side frames; NEIGHBOURS (jc)
    asnp, asnm, apap = sn2_plus[aidx], sn2_minus[aidx], pap_id[aidx]       # still index the FULL cloud
    max_px, ss2 = max_sheet_um / voxel_um, 2.0 * (sigma_s_um / voxel_um) ** 2
    tree = cKDTree(pts)
    if poll_um is not None:                                       # TWO-STAGE anisotropic candidate selection
        poll_px = poll_um / voxel_um
        ss2_poll = 2.0 * ((sigma_s_poll_um or max_sheet_um) / voxel_um) ** 2   # LONG in-plane reach
        sn2_poll = 2.0 * (sigma_n_poll_um / voxel_um) ** 2                     # WIDE across (~1 pitch)
        kq = min(int(poll_k) + 1, P)
        dq, iq = tree.query(apts, k=kq, distance_upper_bound=poll_px, workers=-1)
        dq, iq = np.atleast_2d(dq)[:, 1:], np.atleast_2d(iq)[:, 1:]
        kc = min(kcap, iq.shape[1])
        dd = np.full((A, kc), np.inf)
        ii = np.full((A, kc), P, np.int64)                       # P = invalid sentinel (matches iic<P test below)
        for lo in range(0, A, chunk):
            hi = min(lo + chunk, A); sl = slice(lo, hi)
            iqc, dqc = iq[sl], dq[sl]
            valc = (iqc < P) & np.isfinite(dqc)
            jc = np.where(valc, iqc, 0)
            vc = pts[jc] - apts[sl, None, :]                      # [c,poll_k,3]
            score = _surface_slab_batch(vc, anrm[sl], ajac[sl], ss2_poll, sn2_poll)  # one-sided poll rank
            score = np.where(valc, score, -np.inf)
            top = np.argpartition(-score, kc - 1, axis=1)[:, :kc]  # top-kc by poll score
            rows = np.arange(hi - lo)[:, None]
            ii[sl] = iqc[rows, top]
            dd[sl] = dqc[rows, top]
    else:
        k = min(kcap + 1, P)
        dd, ii = tree.query(apts, k=k, distance_upper_bound=max_px, workers=-1)
        dd, ii = np.atleast_2d(dd)[:, 1:], np.atleast_2d(ii)[:, 1:]  # drop self (col 0)
    kc = ii.shape[1]
    pos_nbr = np.full((A, kc), -1, np.int64)
    pos_w = np.zeros((A, kc), np.float32)
    _dn_all, _wf_all = [], []                                   # width-sigmoid diagnostics
    _torch = None
    if use_gpu:                                                 # GPU slab tables (numerically-equivalent, float64)
        try:
            import torch as _torch_mod
            if _torch_mod.cuda.is_available():
                _torch = _torch_mod
        except Exception:
            _torch = None
    if _torch is not None:
        torch = _torch
        dev = torch.device("cuda")
        f64 = torch.float32                                              # fp32: 3.2× over fp64 on GeForce, parity ~1e-6
        pts_t = torch.as_tensor(pts, dtype=f64, device=dev)              # full cloud (indexed by jc) on GPU once
        normals_t = torch.as_tensor(normals, dtype=f64, device=dev)
        jac_t = torch.as_tensor(jac, dtype=f64, device=dev)
        sn2p_t = torch.as_tensor(sn2_plus, dtype=f64, device=dev)
        sn2m_t = torch.as_tensor(sn2_minus, dtype=f64, device=dev)
        apts_t = torch.as_tensor(apts, dtype=f64, device=dev)            # anchor-side frames
        anrm_t = torch.as_tensor(anrm, dtype=f64, device=dev)
        ajac_t = torch.as_tensor(ajac, dtype=f64, device=dev)
        asnp_t = torch.as_tensor(asnp, dtype=f64, device=dev)
        asnm_t = torch.as_tensor(asnm, dtype=f64, device=dev)
        for lo in range(0, A, chunk):                          # same tiling; slab math on CUDA
            hi = min(lo + chunk, A)
            sl = slice(lo, hi)
            ddc, iic = dd[sl], ii[sl]
            validc = (iic < P) & np.isfinite(ddc)
            jc = np.where(validc, iic, 0)
            validc &= apap[sl, None] != pap_id[jc]              # KDTree candidate bookkeeping stays on CPU
            jc_t = torch.as_tensor(jc, dtype=torch.long, device=dev)
            valid_t = torch.as_tensor(validc, device=dev)
            vc = pts_t[jc_t] - apts_t[sl][:, None, :]           # [c,kc,3]
            sij = _surface_slab_batch_torch(vc, anrm_t[sl], ajac_t[sl], ss2, (asnp_t[sl], asnm_t[sl]), torch)
            nj = normals_t[jc_t]                                # j's slab at i (evaluate at −v with j's frame)
            acr = torch.einsum("pkc,pkc->pk", -vc, nj)
            vip = -vc - acr[..., None] * nj
            inp2 = torch.clamp((vip * vip).sum(-1), min=0.0)
            Jvv = torch.einsum("pkij,pkj->pki", jac_t[jc_t], vip)
            kap = -(vip * Jvv).sum(-1) / torch.clamp(inp2, min=1e-9)
            rr = (acr - 0.5 * kap * inp2) / torch.sqrt(1.0 + kap ** 2 * inp2)
            sn2j = torch.where(acr >= 0.0, sn2p_t[jc_t], sn2m_t[jc_t])
            sji = -(inp2 / ss2 + rr ** 2 / sn2j)
            mutual = torch.exp(torch.clamp(sij + sji, -60.0, 0.0))
            pw = valid_t.to(f64) * mutual
            if width_lambda_um is not None:                    # SHEET-WIDTH sigmoid: down-weight (never push)
                dn_um = torch.abs(torch.einsum("pkc,pc->pk", vc, anrm_t[sl])) * voxel_um
                wfac = 1.0 / (1.0 + torch.exp((dn_um - width_lambda_um) / max(width_steep_um, 1e-3)))
                pw = pw * wfac
                _dn_all.append(dn_um[valid_t].cpu().numpy()); _wf_all.append(wfac[valid_t].cpu().numpy())
            pos_w[sl] = pw.cpu().numpy().astype(np.float32)
            pos_nbr[sl] = np.where(validc, jc, -1)
        del pts_t, normals_t, jac_t, sn2p_t, sn2m_t
        torch.cuda.empty_cache()
    else:
        for lo in range(0, A, chunk):                          # TILE rows → bounded peak memory (numpy path)
            hi = min(lo + chunk, A)
            sl = slice(lo, hi)
            ddc, iic = dd[sl], ii[sl]
            validc = (iic < P) & np.isfinite(ddc)
            jc = np.where(validc, iic, 0)
            validc &= apap[sl, None] != pap_id[jc]
            vc = pts[jc] - apts[sl, None, :]                       # [c,kc,3]
            sij = _surface_slab_batch(vc, anrm[sl], ajac[sl], ss2, (asnp[sl], asnm[sl]))
            nj = normals[jc]                                       # j's slab at i (evaluate at −v with j's frame)
            acr = np.einsum("pkc,pkc->pk", -vc, nj)
            vip = -vc - acr[..., None] * nj
            inp2 = np.maximum((vip * vip).sum(-1), 0.0)
            Jvv = np.einsum("pkij,pkj->pki", jac[jc], vip)
            kap = -(vip * Jvv).sum(-1) / np.maximum(inp2, 1e-9)
            rr = (acr - 0.5 * kap * inp2) / np.sqrt(1.0 + kap ** 2 * inp2)
            sn2j = np.where(acr >= 0.0, sn2_plus[jc], sn2_minus[jc])
            sji = -(inp2 / ss2 + rr ** 2 / sn2j)
            mutual = np.exp(np.clip(sij + sji, -60.0, 0.0))
            pos_w[sl] = validc.astype(np.float32) * mutual.astype(np.float32)
            if width_lambda_um is not None:                        # SHEET-WIDTH sigmoid: down-weight (never push)
                dn_um = np.abs(np.einsum("pkc,pc->pk", vc, anrm[sl])) * voxel_um  # across-normal reach at anchor frame
                wfac = 1.0 / (1.0 + np.exp((dn_um - width_lambda_um) / max(width_steep_um, 1e-3)))
                pos_w[sl] *= wfac.astype(np.float32)
                _dn_all.append(dn_um[validc]); _wf_all.append(wfac[validc])
            pos_nbr[sl] = np.where(validc, jc, -1)
    if width_lambda_um is not None and _dn_all:
        dn, wf = np.concatenate(_dn_all), np.concatenate(_wf_all)
        print(f"  [width] λ={width_lambda_um:.0f}µm steep={width_steep_um:.0f}: d_n(µm) p50 {np.median(dn):.0f} "
              f"p90 {np.percentile(dn,90):.0f}   width-weight p50 {np.median(wf):.2f} mean {wf.mean():.2f}", flush=True)
    if bced is not None and slab_negatives:                    # §5.3: gate on the aggregated slab FIELD + decoupled boundary
        from hercunet.labels.selection.membership import splat_slab_field
        gate_bnd = 40.0                                                 # boundary σ_max (µm), decoupled from field width
        sig_s_f = 40.0                                                  # field in-plane reach (µm) — sharp local sheets
        sub = 600000                                                    # cap points splatted (GPU memory)
        ss2_f = 2.0 * (sig_s_f / voxel_um) ** 2
        R = int(np.ceil(3.0 * sig_s_f / voxel_um))
        sel = (np.random.RandomState(0).choice(len(pts), sub, replace=False) if len(pts) > sub else np.arange(len(pts)))
        acc = splat_slab_field(pts[sel], normals[sel], jac[sel], sn2_plus[sel], sn2_minus[sel], ss2_f, bced.shape, R)
        acc = (acc / max(float(acc.max()), 1e-9)).astype(np.float32)    # sharp field (σn from positives ~25) = deep gaps
        gate_tau = 0.5                                                  # shallower-gap threshold (sweep: 0.3→0.5 = 6× negs)
        # REUSE the sharp cap-25 σn (already measured for positives): gate distance = margin_k·σn stays tight (~60µm),
        # and boundary σmax=gate_bnd(40) > σn(≤25) so the guard never rejects thick compressed sheets. No 2nd measure.
        neg_nbr, neg_w = gap_gated_negatives_3d(pts, normals, acc, voxel_um, pap_id=pap_id,
                                                anchor_idx=anchor_idx, kneg=kneg, tau=gate_tau,
                                                sigma_max_um=gate_bnd, reach_um=neg_reach_um, sigma_ip_um=neg_sigma_ip_um,
                                                lam_decay_um=neg_lam_decay_um, margin_k=neg_margin_k,
                                                gate_floor_um=neg_gate_floor_um, seed=seed,
                                                use_gpu=use_gpu, sn2_plus=sn2_plus, sn2_minus=sn2_minus)
        print(f"  [slab-gate] field {len(sel)}pts R={R}px σs={sig_s_f} boundary={gate_bnd} tau={gate_tau} → "
              f"neg cov {100*float((neg_nbr>=0).any(1).mean()):.0f}% (was ~8-12%)", flush=True)
        return pos_nbr, pos_w, neg_nbr, neg_w
    if bced is not None:                                       # GAP-GATED negative slab (mirror of positives)
        neg_nbr, neg_w = gap_gated_negatives_3d(pts, normals, bced, voxel_um, pap_id=pap_id,
                                                anchor_idx=anchor_idx, kneg=kneg, tau=neg_tau,
                                                reach_um=neg_reach_um, sigma_ip_um=neg_sigma_ip_um,
                                                lam_decay_um=neg_lam_decay_um, margin_k=neg_margin_k,
                                                gate_floor_um=neg_gate_floor_um, seed=seed,
                                                use_gpu=use_gpu, sn2_plus=sn2_plus, sn2_minus=sn2_minus)
        return pos_nbr, pos_w, neg_nbr, neg_w                  # neg_w = per-negative push weight
    neg_nbr, neg_absr = corridor_negatives_3d(pts, normals, neg_cell_um / voxel_um,   # legacy fallback
                                              neg_dist_um / voxel_um, kneg, seed=seed)
    if anchor_idx is not None:                                  # restrict negative ROWS to the anchors too
        neg_nbr, neg_absr = neg_nbr[aidx], neg_absr[aidx]
    return pos_nbr, pos_w, neg_nbr, (neg_nbr >= 0).astype(np.float32)  # uniform weight (corridor)


def pos_available_count(pts, pap_id, normals, jac, sn2_plus, sn2_minus, voxel_um,
                        max_sheet_um=220.0, sigma_s_um=180.0, width_lambda_um=30.0, width_steep_um=12.0,
                        tau=0.05, avail_k=256, anchor_idx=None, chunk=40000, use_gpu=False):
    """DEBUG (not in the training path): per anchor, the number of DIFFERENT-paperlet neighbours within
    ``max_sheet_um`` whose mutual surface-slab weight ``pos_w = exp(sij+sji)·width`` exceeds ``tau`` — i.e. the
    TRUE positive pool BEFORE the ``kcap`` truncation. Reveals the huge candidate pool in compressed regions.
    Reuses the exact mutual-slab math of :func:`paperlet_tables` (numpy or torch twin), tiled with a per-tile
    expanded gather so memory stays bounded (never materialises [P,avail_k] globally). Returns
    ``n_avail [P] int32`` and ``n_scored [P] int32`` (valid candidates scored; ==avail_k ⇒ pool may saturate)."""
    from scipy.spatial import cKDTree
    P = len(pts)
    aidx = np.arange(P) if anchor_idx is None else np.asarray(anchor_idx)
    A = len(aidx)
    max_px = max_sheet_um / voxel_um
    ss2 = 2.0 * (sigma_s_um / voxel_um) ** 2
    tree = cKDTree(pts)
    n_avail = np.zeros(P, np.int32); n_scored = np.zeros(P, np.int32)
    kq = min(int(avail_k) + 1, P)
    torch = None
    if use_gpu:
        try:
            import torch as _t
            if _t.cuda.is_available():
                torch = _t
        except Exception:
            torch = None
    dev = torch.device("cuda") if torch is not None else None
    f = torch.float32 if torch is not None else None
    for lo in range(0, A, chunk):
        hi = min(lo + chunk, A); rows = aidx[lo:hi]
        apts, anrm, ajac = pts[rows], normals[rows], jac[rows]
        asnp, asnm, apap = sn2_plus[rows], sn2_minus[rows], pap_id[rows]
        dq, iq = tree.query(apts, k=kq, distance_upper_bound=max_px, workers=-1)
        dq, iq = np.atleast_2d(dq)[:, 1:], np.atleast_2d(iq)[:, 1:]        # drop self
        valid = (iq < P) & np.isfinite(dq)
        jc = np.where(valid, iq, 0)
        valid &= apap[:, None] != pap_id[jc]                              # DIFFERENT paperlet only
        if torch is not None:
            vc = torch.as_tensor(pts[jc] - apts[:, None, :], dtype=f, device=dev)
            anrm_t = torch.as_tensor(anrm, dtype=f, device=dev); ajac_t = torch.as_tensor(ajac, dtype=f, device=dev)
            asnp_t = torch.as_tensor(asnp, dtype=f, device=dev); asnm_t = torch.as_tensor(asnm, dtype=f, device=dev)
            sij = _surface_slab_batch_torch(vc, anrm_t, ajac_t, ss2, (asnp_t, asnm_t), torch)
            nj = torch.as_tensor(normals[jc], dtype=f, device=dev)
            acr = torch.einsum("pkc,pkc->pk", -vc, nj)
            vip = -vc - acr[..., None] * nj
            inp2 = torch.clamp((vip * vip).sum(-1), min=0.0)
            Jvv = torch.einsum("pkij,pkj->pki", torch.as_tensor(jac[jc], dtype=f, device=dev), vip)
            kap = -(vip * Jvv).sum(-1) / torch.clamp(inp2, min=1e-9)
            rr = (acr - 0.5 * kap * inp2) / torch.sqrt(1.0 + kap ** 2 * inp2)
            sn2j = torch.as_tensor(np.where(acr.cpu().numpy() >= 0.0, sn2_plus[jc], sn2_minus[jc]), dtype=f, device=dev)
            sji = -(inp2 / ss2 + rr ** 2 / sn2j)
            pw = torch.exp(torch.clamp(sij + sji, -60.0, 0.0))
            if width_lambda_um is not None:
                dn = torch.abs(torch.einsum("pkc,pc->pk", vc, anrm_t)) * voxel_um
                pw = pw / (1.0 + torch.exp((dn - width_lambda_um) / max(width_steep_um, 1e-3)))
            pw = pw.cpu().numpy()
        else:
            vc = pts[jc] - apts[:, None, :]
            sij = _surface_slab_batch(vc, anrm, ajac, ss2, (asnp, asnm))
            nj = normals[jc]
            acr = np.einsum("pkc,pkc->pk", -vc, nj)
            vip = -vc - acr[..., None] * nj
            inp2 = np.maximum((vip * vip).sum(-1), 0.0)
            Jvv = np.einsum("pkij,pkj->pki", jac[jc], vip)
            kap = -(vip * Jvv).sum(-1) / np.maximum(inp2, 1e-9)
            rr = (acr - 0.5 * kap * inp2) / np.sqrt(1.0 + kap ** 2 * inp2)
            sn2j = np.where(acr >= 0.0, sn2_plus[jc], sn2_minus[jc])
            sji = -(inp2 / ss2 + rr ** 2 / sn2j)
            pw = np.exp(np.clip(sij + sji, -60.0, 0.0))
            if width_lambda_um is not None:
                dn = np.abs(np.einsum("pkc,pc->pk", vc, anrm)) * voxel_um
                pw = pw / (1.0 + np.exp((dn - width_lambda_um) / max(width_steep_um, 1e-3)))
        n_avail[rows] = ((pw > tau) & valid).sum(1).astype(np.int32)
        n_scored[rows] = valid.sum(1).astype(np.int32)
    return n_avail, n_scored


def train_paperlet_embedding(n_pap, pts, pap_id, pos_nbr, pos_w, neg_nbr, neg_w, pap_centroid, voxel_um,
                             dim=32, lr=0.05, margin=1.0, margin_scale=0.004, n_soft=8, samples_per_point=150,
                             pos_tau=1.0, pos_pull=0.0, batch=65536, seed=0, log_every_s=60.0, verbose=True,
                             ptype=None, anchor_pts=None, loss_mode="triplet",
                             vic_inv=25.0, vic_var=25.0, vic_cov=1.0):
    """One embedding vector per paperlet (points sample). Per anchor point: a positive paperlet (drawn ∝
    ``pos_w**(1/pos_tau)`` — the WARMTH τ: 1=∝agreement, >1 flattens toward uniform so weak boundary positives
    get drawn and stop starving the seams, <1 sharpens), the corridor hard-negative paperlet + ``n_soft`` random
    soft-negatives, ORDINAL margin ``margin + margin_scale·d_fibre`` (µm between paperlet centroids). Constant
    gradient budget (``samples_per_point`` draws/point). Progress logged every ``log_every_s``. Returns
    ``E [n_pap, dim]`` (un-pulled paperlets keep random init → eom noise = abstain)."""
    import time as _time

    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    E = (0.01 * torch.randn(n_pap, dim, device=dev)).requires_grad_(True)
    P = len(pts)
    if P < 8:
        return E.detach().cpu().numpy()
    pid = torch.as_tensor(pap_id, dtype=torch.long, device=dev)
    pn = torch.as_tensor(pos_nbr, dtype=torch.long, device=dev)
    pw = torch.as_tensor(np.nan_to_num(pos_w), device=dev)
    nn = torch.as_tensor(neg_nbr, dtype=torch.long, device=dev)
    nwt = torch.as_tensor(np.nan_to_num(neg_w), dtype=torch.float32, device=dev)  # per-negative push weight
    # FORCED recto/verso balance: ptype[P] (0=recto/spine, 1=verso/rib). Draw one positive of EACH type per
    # anchor with equal loss weight, so the 90%-verso mass can't starve the co-planar recto links (v1 parity).
    balance = ptype is not None
    ptt = torch.as_tensor(np.asarray(ptype), dtype=torch.long, device=dev) if balance else None  # [P] 0=recto 1=verso
    apts_idx = torch.as_tensor(np.arange(P) if anchor_pts is None else np.asarray(anchor_pts),
                               dtype=torch.long, device=dev)             # table ROW -> anchor POINT index
    A_rows = pn.shape[0]                                                  # anchors = table rows (recto-only in v1)
    eff = min(int(batch), A_rows)
    n_iter = max(200, int(np.ceil(samples_per_point * A_rows / eff)))    # budget scales to ANCHOR count
    opt = torch.optim.Adam([E], lr=lr)
    if verbose:
        print(f"    [train] {P} pts / {A_rows} anchors, {n_pap} paperlets, {n_iter} iters (batch {eff}, "
              f"~{samples_per_point}/pt) on {dev.type}", flush=True)
    t0 = _time.time()
    tlast = t0
    run_loss = 0.0
    seen = 0
    for it in range(n_iter):
        r = torch.randint(0, A_rows, (eff,), device=dev)                 # table ROW = anchor (recto-only in v1)
        a_pap = pid[apts_idx[r]]
        pnr = pn[r]
        w = pw[r].clone()
        pv = w.sum(1) > 0
        if pos_tau != 1.0:                                                # warmth: >1 flattens toward uniform
            w = w.clamp(min=0.0) ** (1.0 / pos_tau)

        def _draw(wmat):                                                  # sample one positive paperlet ∝ wmat
            has = wmat.sum(1) > 0
            wf = wmat.clone(); wf[~has] = 1.0
            j = torch.multinomial(wf, 1).squeeze(1)
            pt = pnr.gather(1, j[:, None]).squeeze(1).clamp(min=0)
            return pid[pt], has

        za = E[a_pap]
        if balance:                                                      # recto + verso positive draws
            rt = ptt[pnr.clamp(min=0)]
            wmats = (w * (rt == 0).float(), w * (rt == 1).float())
        else:
            wu = w.clone(); wu[~pv] = 1.0
            wmats = (wu,)
        draws = [_draw(wm) for wm in wmats]                              # [(pos_pap, has), ...]

        if loss_mode == "vicreg":                                        # VICReg: NO negatives, NO ordinality
            inv, nz, zbatch = 0.0, 0, [za]
            for pap, has in draws:
                zp = E[pap]; zbatch.append(zp)
                if has.any():
                    inv = inv + ((za[has] - zp[has]) ** 2).sum(1).mean(); nz += 1
            if nz == 0:
                continue
            Zall = torch.cat(zbatch, 0)
            Zc = Zall - Zall.mean(0)
            std = torch.sqrt(Zc.var(0) + 1e-4)
            var_loss = torch.relu(1.0 - std).mean()                      # keep per-dim spread → no collapse
            cov = (Zc.t() @ Zc) / (Zall.shape[0] - 1)                     # decorrelate dims
            cov_loss = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / dim
            loss = vic_inv * (inv / nz) + vic_var * var_loss + vic_cov * cov_loss
        else:                                                            # PULL (invariance) + VARIANCE anti-collapse + gap negatives PUSH
            nrow = nn[r]; nwr = nwt[r]; nvalid = nrow >= 0; nany = nvalid.any(1)
            wsel = (nwr * nvalid.float()).clamp(min=0.0) + 1e-9           # draw one gap negative ∝ neg_w
            nsel = torch.multinomial(wsel, 1).squeeze(1)
            hw = nwr.gather(1, nsel[:, None]).squeeze(1)                  # its push weight [eff]
            neg_pap = pid[nrow.gather(1, nsel[:, None]).squeeze(1).clamp(min=0)]
            dneg = torch.sqrt(((za - E[neg_pap]) ** 2).sum(1) + 1e-9)     # [eff]
            inv, nz, push, zbatch = 0.0, 0, [], [za]
            for pap, has in draws:
                zp = E[pap]; zbatch.append(zp)
                if not has.any():
                    continue
                inv = inv + ((za[has] - zp[has]) ** 2).sum(1).mean(); nz += 1       # UNCONDITIONAL positive pull
                okn = (nany & has).float()                                          # push where a gap negative exists
                if okn.sum() > 0:
                    t = torch.relu(margin - dneg)                                   # hinge: separate the adjacent winding
                    push.append((t * hw * okn).sum() / (hw * okn).sum().clamp(min=1e-6))  # neg_w-weighted
            if nz == 0:
                continue
            Zall = torch.cat(zbatch, 0)                                   # VARIANCE anti-collapse: counterbalances the pull so
            Zc = Zall - Zall.mean(0)                                      # low negative-coverage windows can't implode to a blob
            std = torch.sqrt(Zc.var(0) + 1e-4)
            var_loss = torch.relu(1.0 - std).mean()
            cov = (Zc.t() @ Zc) / (Zall.shape[0] - 1)
            cov_loss = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / dim
            loss = (vic_inv * (inv / nz) + vic_var * var_loss + vic_cov * cov_loss
                    + (sum(push) / len(push) if push else 0.0))
        opt.zero_grad()
        loss.backward()
        opt.step()
        run_loss += float(loss.detach())
        seen += 1
        now = _time.time()
        if verbose and (now - tlast >= log_every_s or it == n_iter - 1):
            print(f"    [train] iter {it+1}/{n_iter}  loss {run_loss/max(1,seen):.4f}  "
                  f"{1000*(now-t0)/(it+1):.1f}ms/step  elapsed {now-t0:.0f}s", flush=True)
            tlast, run_loss, seen = now, 0.0, 0
    return E.detach().cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────────────────────────────────────
# SupCon-on-Louvain bootstrap (deep clustering / PCL) — GUARDED so it MERGES fragmented same-sheet communities
# and only PUSHES real (across-normal) adjacent windings apart. The Louvain pseudo-labels CONTAIN the current
# errors (a sheet is FRAGMENTED across communities; an adjacent winding may be MERGED into one), so the triplets
# are arbitrated by PHYSICS, never by the raw labels:
#   positives  = same-community  ∪  slab-continuous-along-sheet across communities   → MERGE fragments
#   negatives  = corridor / ACROSS-NORMAL (applied even within a community)          → SPLIT merged windings
#                (a slab-continuous pair is NEVER a negative — POS wins; random soft-negs are cross-community only)
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────


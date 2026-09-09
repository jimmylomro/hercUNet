"""Embedding-space label augmentations for ∇φ pseudo-label training — synthetic MERGES and SPLITS.

The contrastive embedding is an ORDINAL manifold (adjacent windings sit adjacent, §7a of
docs/sheet_membership_embeddings_v3.md). That lets us synthesise the two error modes a downstream detector
must survive, DIRECTLY from the de-noised clustering — no CT, no training:

  MERGE  — relabel an ordinal-adjacent cluster pair (A,B) to one id → one winding where there were two.
  SPLIT  — cut a single cluster along a borderline internal bimodality → a false fragmentation seam.

We generate ONLY in the AMBIGUOUS regime the embedding itself flags (a merge where A,B are genuinely
intermixed in E; a split where the cluster is borderline-bimodal), because an OBVIOUS perturbation marked
low-confidence would teach the confidence head to cry wolf on easy configs. The perturbed LOCUS (the interface
band) is the low-confidence region: it down-weights the deliberately-wrong ∇φ AND supervises the confidence
head ("this topology is untrustworthy"). See the handover §5.4 confidence discussion.

Pure numpy + scipy (+ sklearn for the split's 2-component GMM). No CT, no torch, no plotting.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def per_unit_labels(sid, plab, n_units=None):
    """Collapse the per-POINT label ``plab`` (indexed like ``pts``) to a per-UNIT label indexed like ``E``.
    A unit (streamlet) has a single cluster label, so any of its points' labels serves."""
    sid = np.asarray(sid); plab = np.asarray(plab)
    n = int(sid.max() + 1) if n_units is None else n_units
    ulab = np.full(n, -1, np.int64)
    ulab[sid] = plab                                                   # consistent within a unit → order irrelevant
    return ulab


def _embedding_centroids(E, ulab):
    labs = sorted(int(x) for x in np.unique(ulab) if x >= 0)
    cent = {c: E[ulab == c].mean(0) for c in labs}
    return labs, cent


def ordinal_merge_sets(E, ulab, *, n_merge=3, min_units_frac=0.3, max_sets=4):
    """Select MEANINGFUL merges: fold N CONSECUTIVE windings along the ordinal axis, restricted to LARGE
    clusters. Rationale (Jaime, 2026-08-08): the CLOSEST ordinal pair is usually a skinnydip OVER-SPLIT (a small
    fragment beside its big parent), so merging it FIXES the label — punishing the model for that is wrong. A
    genuine corruption conflates real, distinct windings, so we (a) drop small clusters (``min_units_frac``
    percentile of unit counts) and (b) merge a sliding window of ``n_merge`` clusters that are CONSECUTIVE along
    PC1 of the embedding (the winding ladder, ~47% var, §7a) — i.e. physically neighbouring real sheets.
    Returns (sets [tuple(cluster_ids), ...] biggest-first, non-overlapping, sizes dict)."""
    labs, cent = _embedding_centroids(E, ulab)
    if len(labs) < n_merge:
        return [], {c: int((ulab == c).sum()) for c in labs}
    sizes = {c: int((ulab == c).sum()) for c in labs}
    mu = E[ulab >= 0].mean(0)
    _, _, Vt = np.linalg.svd(E[ulab >= 0] - mu, full_matrices=False)
    pc1 = Vt[0]                                                        # ordinal winding axis
    order = sorted(labs, key=lambda c: float(cent[c] @ pc1))          # windings in ladder order
    thr = np.quantile([sizes[c] for c in labs], min_units_frac)
    big = [c for c in order if sizes[c] >= thr]                       # real sheets only (drop over-split fragments)
    sets = [tuple(int(x) for x in big[i:i + n_merge]) for i in range(len(big) - n_merge + 1)]
    sets = sorted(sets, key=lambda g: -sum(sizes[c] for c in g))      # biggest (most meaningful) first
    chosen, used = [], set()
    for g in sets:
        if used & set(g):
            continue
        chosen.append(g); used |= set(g)
        if len(chosen) >= max_sets:
            break
    return chosen, sizes


def relabel_merge_set(plab, ids):
    """Fold every cluster in ``ids`` into ``ids[0]``. Returns (plab_new, merged_id=ids[0])."""
    out = np.asarray(plab).copy()
    for c in ids[1:]:
        out[out == c] = ids[0]
    return out, int(ids[0])


def borderline_split(E, ulab, cluster, *, pcs=15, min_gap=1.0, max_pairs=None):
    """Propose a BORDERLINE split of one cluster: project its units onto the top-``pcs`` GLOBAL PCA axes of E
    (seams live on global axes), pick the most bimodal axis, fit a 2-component GMM, and accept only if the two
    modes are separated by ≥ ``min_gap`` pooled-σ AND GMM-2 beats GMM-1 by BIC (a plausible seam, not noise).
    Returns (unit_side [bool over the cluster's units], separation) or None. ``unit_side`` True = side B."""
    from sklearn.mixture import GaussianMixture
    sel = np.where(ulab == cluster)[0]
    if len(sel) < 40:
        return None
    Ec = E[sel]
    mu = E[ulab >= 0].mean(0)
    _, _, Vt = np.linalg.svd(E[ulab >= 0] - mu, full_matrices=False)
    ax = Vt[:min(pcs, Vt.shape[0])]
    proj = (Ec - mu) @ ax.T                                            # [n, pcs]
    best = None
    for j in range(proj.shape[1]):
        x = proj[:, j:j + 1]
        g1 = GaussianMixture(1, covariance_type="full", random_state=0).fit(x)
        g2 = GaussianMixture(2, covariance_type="full", random_state=0).fit(x)
        if g2.bic(x) >= g1.bic(x):
            continue
        m0, m1 = g2.means_[:, 0]
        s = np.sqrt(g2.covariances_[:, 0, 0]).mean()
        gap = abs(m0 - m1) / (s + 1e-9)
        if gap >= min_gap and (best is None or gap > best[0]):
            side = (g2.predict(x) == int(m1 > m0))
            best = (gap, side)
    if best is None:
        return None
    return best[1], float(best[0])


def dropped_lowconf(pts, plab, shape, *, smooth_px=3.0, norm_pct=99.0, gamma=2.5, region_eps=0.1):
    """SMOOTH but TIGHTLY-LOCALISED low-confidence from DROPPED points (plab < 0 = eom noise /
    sub-min_cluster_size fragments): structure we detected but couldn't confidently assign a sheet. Splat the
    noise points, Gaussian-smooth (small kernel → compact), normalise by a high percentile, then raise to
    ``gamma`` > 1 so uncertainty concentrates at the dense CORES and falls off fast (compact soft blobs, not a
    broad wash — and not a binary mask). Returns (conf = 1 − u, region mask u>region_eps for the display cover).
    On real sheets the dropped density ≈ 0 → conf ≈ 1, so surviving sheets stay confident."""
    drop = np.asarray(pts)[np.asarray(plab) < 0]
    if not len(drop):
        return np.ones(shape, np.float32), np.zeros(shape, bool)
    Z, H, W = shape
    z = np.clip(np.round(drop[:, 0]).astype(int), 0, Z - 1)
    y = np.clip(np.round(drop[:, 1]).astype(int), 0, H - 1)
    x = np.clip(np.round(drop[:, 2]).astype(int), 0, W - 1)
    cnt = np.zeros(shape, np.float32); np.add.at(cnt, (z, y, x), 1.0)
    sm = ndi.gaussian_filter(cnt, smooth_px)
    scale = np.percentile(sm[sm > 0], norm_pct) if (sm > 0).any() else 1.0
    u = np.clip(sm / (scale + 1e-9), 0.0, 1.0) ** gamma               # gamma>1 → tight cores, fast falloff
    return (1.0 - u).astype(np.float32), (u > region_eps)


def relabel_split(plab, sid, cluster, unit_side):
    """New per-point labelling splitting ``cluster`` into two fresh ids by the per-unit boolean ``unit_side``
    (indexed over the cluster's units in ascending unit id). Returns (plab_new, (id_a, id_b))."""
    plab = np.asarray(plab); sid = np.asarray(sid)
    id_a = int(plab.max()) + 1
    id_b = id_a + 1
    units = np.where(per_unit_labels(sid, plab) == cluster)[0]
    side_of_unit = {int(u): bool(s) for u, s in zip(units, unit_side)}
    out = plab.copy()
    pt = plab == cluster
    su = sid[pt]
    newlab = np.array([id_b if side_of_unit.get(int(u), False) else id_a for u in su])
    out[pt] = newlab
    return out, (id_a, id_b)


def interface_band(lab3, id_a, id_b, band_px=3):
    """The band of voxels straddling the ``id_a`` | ``id_b`` interface in a label volume — the perturbation
    locus (the real inter-winding gap for a merge; the injected cut plane for a split). Dilate each side's mask
    and intersect: the overlap is the seam. Returns a boolean volume."""
    a = lab3 == id_a; b = lab3 == id_b
    it = max(1, int(band_px))
    return (ndi.binary_dilation(a, iterations=it) & ndi.binary_dilation(b, iterations=it))


def low_confidence_from_mask(mask, cover, *, smooth_px=6.0):
    """Turn a perturbation-locus mask into a REGIONAL low-confidence field on the same footing as the oracle
    confidence (``sheet_mesh.confidence_volume``): footprint-smoothed, conf = 1 − smoothed(mask) inside the
    sheet cover. Combine (soft-OR of doubts) with the oracle confidence via ``sheet_mesh.combine_confidence``."""
    cf = cover.astype(np.float32)
    ms = ndi.gaussian_filter(mask.astype(np.float32), smooth_px) / (ndi.gaussian_filter(cf, smooth_px) + 1e-6)
    return np.clip(1.0 - ms, 0.0, 1.0).astype(np.float32)

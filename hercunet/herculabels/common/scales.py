"""Naive sheet segmentation from streamline features — a pre-ML probe of the fluid extraction.

Idea (before any learned model): trace many streamlines through the sheet-tangent field at a LOW
coherence threshold (deliberately over-tracing, sub-sheet fragments and all), turn each streamline
into a small feature vector, and cluster in feature space with DBSCAN (density-based, no fixed
cluster count). Sheet membership is *density-separated in feature space even when it is not spatially
separated* — which is exactly the delamination-robust "group by orientation + continuity, not
distance" principle.

The load-bearing feature is **curvature**: a scroll is a spiral, so a streamline's local curvature
κ ≈ 1/R where R is its radial distance from the umbilicus. Curvature is therefore monotonic in the
radial *layer index*, LOCAL, and umbilicus-free — outer sheets are naturally flatter. Radius from an
estimated umbilicus is kept only as a diagnostic (it smears, because a small window can't locate the
cm-away umbilicus).

This operates on a single depth slice's streamlines (the current 2-D tracer). The feature extraction
is written to lift unchanged to 3-D streamlines once those exist (curvature/coherence/density/tangent
all generalise; the 2-D normal-projection becomes a 3-D one).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree


def split_streamlines(xs: np.ndarray, ys: np.ndarray) -> list[np.ndarray]:
    """Split the NaN-separated flat ``(xs, ys)`` from :func:`trace_streamlines` into a list of
    ``(M, 2)`` polylines (columns = x, y). Drops empty/degenerate runs."""
    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    out: list[np.ndarray] = []
    n = len(xs)
    i = 0
    while i < n:
        if np.isnan(xs[i]):
            i += 1
            continue
        j = i
        while j < n and not np.isnan(xs[j]):
            j += 1
        if j - i >= 3:
            out.append(np.stack([xs[i:j], ys[i:j]], axis=1))
        i = j
    return out


def _resample_smooth(pts: np.ndarray, smooth_sigma: float = 2.0) -> np.ndarray:
    """Light Gaussian smoothing along arc-length so RK2 integration noise doesn't dominate the
    curvature estimate. Endpoints are held (reflect)."""
    if len(pts) < 5 or smooth_sigma <= 0:
        return pts
    x = gaussian_filter1d(pts[:, 0], smooth_sigma, mode="nearest")
    y = gaussian_filter1d(pts[:, 1], smooth_sigma, mode="nearest")
    return np.stack([x, y], axis=1)


def polyline_curvature(pts: np.ndarray, voxel_um: float) -> tuple[float, float]:
    """Length-weighted mean **signed** and **absolute** curvature of a polyline, in **rad/mm**.

    κ = dφ/ds via the signed turning angle between consecutive (central-difference) tangents over
    the local arc length. Signed κ also encodes concavity (which way the sheet bows → which side the
    umbilicus is); |κ| is the radial-index proxy. Coords are in voxels; converted to mm via
    ``voxel_um``."""
    p = _resample_smooth(pts)
    if len(p) < 5:
        return 0.0, 0.0
    d = np.diff(p, axis=0)                                   # segment vectors
    seg = np.hypot(d[:, 0], d[:, 1])
    ok = seg > 1e-6
    if ok.sum() < 3:
        return 0.0, 0.0
    t = d[ok] / seg[ok, None]                                # unit tangents
    seg = seg[ok]
    # signed angle between successive tangents (cross for sign, dot for magnitude)
    cross = t[:-1, 0] * t[1:, 1] - t[:-1, 1] * t[1:, 0]
    dot = np.clip((t[:-1] * t[1:]).sum(1), -1.0, 1.0)
    dphi = np.arctan2(cross, dot)                            # rad, signed
    ds = 0.5 * (seg[:-1] + seg[1:]) * (voxel_um / 1000.0)    # mm
    ds = np.maximum(ds, 1e-6)
    kappa = dphi / ds                                        # rad/mm, signed
    w = ds
    signed = float(np.average(kappa, weights=w))
    absk = float(np.average(np.abs(kappa), weights=w))
    return signed, absk


def _bilinear_vec(field: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear-sample an ``(H, W, C)`` field at float ``xy`` points ``(N, 2)`` (x=col, y=row).
    Out-of-bounds points clamp to the edge."""
    h, w = field.shape[:2]
    x = np.clip(xy[:, 0], 0, w - 1.001)
    y = np.clip(xy[:, 1], 0, h - 1.001)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    return (field[y0, x0] * (1 - fy) * (1 - fx) + field[y0, x1] * (1 - fy) * fx
            + field[y1, x0] * fy * (1 - fx) + field[y1, x1] * fy * fx)


def estimate_umbilicus(centroids: np.ndarray, normals_xy: np.ndarray):
    """Least-squares intersection of the lines through ``centroids`` with in-plane directions
    ``normals_xy`` (the sheet normals all point roughly radially, so they converge on the
    umbilicus). Returns ``(center_xy, condition)`` where a large ``condition`` means the normals
    are near-parallel (small window, umbilicus far) and the centre is untrustworthy."""
    n = normals_xy / (np.linalg.norm(normals_xy, axis=1, keepdims=True) + 1e-9)
    # Each line contributes (I - n nᵀ); solve Σ(I-nnᵀ) c = Σ(I-nnᵀ) p.
    A = np.zeros((2, 2))
    b = np.zeros(2)
    for p, ni in zip(centroids, n):
        proj = np.eye(2) - np.outer(ni, ni)
        A += proj
        b += proj @ p
    try:
        center = np.linalg.solve(A, b)
        cond = float(np.linalg.cond(A))
    except np.linalg.LinAlgError:
        center = centroids.mean(0)
        cond = np.inf
    return center, cond


# Feature columns produced by :func:`streamline_features`, in order.
FEATURE_NAMES = [
    "abs_curvature",     # |κ| rad/mm — radial layer index proxy, separates stuck-together sheets
    "signed_curvature",  # signed κ rad/mm — concavity / winding side
    "normal_proj_mm",    # centroid on window mean-normal (mm) — robust LOCAL "which layer"
    "umbilicus_r_mm",    # radius from estimated umbilicus (mm) — separates gapped/radially-offset sheets
    "umbilicus_theta",   # angle around umbilicus (rad) — position along the sheet
    "tangent_angle",     # mean streamline orientation (rad, mod π) — local sheet orientation
    "fibre_z_frac",      # mean |fibre·ẑ| — RECTO/VERSO axis (recto ∥z reading face vs verso in-plane)
    "mean_coherence",    # mean structure-tensor coherence along the line
    "log_density",       # log local streamline-point density (gaps → low)
    "length_mm",         # streamline length (mm) — long=real sheet, short=noise
    "mean_phi_mm",       # mean layer-potential φ (mm) — the curvature-following 'which sheet' coord
]


def streamline_features(
    polylines: list[np.ndarray],
    normal_plane: np.ndarray,
    fibre_plane: np.ndarray,
    coherence: np.ndarray,
    voxel_um: float,
    density_radius_um: float = 120.0,
    umbilicus_center: np.ndarray | None = None,
    phi_field: np.ndarray | None = None,
):
    """Build the ``(N, len(FEATURE_NAMES))`` per-streamline feature matrix.

    ``normal_plane`` / ``fibre_plane`` are the frame normal / fibre eigenvectors on this slice as
    ``(H, W, 3)`` in (z, y, x); ``coherence`` is ``(H, W)``. Coords in ``polylines`` share that pixel
    frame. The umbilicus is estimated internally from the sheet normals (least-squares convergence
    point) unless ``umbilicus_center`` is given. Returns ``(X, names, extra)`` where ``extra`` holds
    centroids, mean normals, the umbilicus centre and its condition number."""
    all_pts = np.concatenate(polylines, axis=0) if polylines else np.zeros((0, 2))
    tree = cKDTree(all_pts) if len(all_pts) else None
    r_px = density_radius_um / voxel_um

    rows = []
    centroids = []
    mean_norms = []
    for pts in polylines:
        c = pts.mean(0)
        signed_k, abs_k = polyline_curvature(pts, voxel_um)
        nrm = _bilinear_vec(normal_plane, pts)              # (M,3) z,y,x
        nxy = nrm[:, 2:0:-1]                                # -> (x, y)
        mean_n = nxy.mean(0)
        mean_n = mean_n / (np.linalg.norm(mean_n) + 1e-9)
        fib = _bilinear_vec(fibre_plane, pts)               # (M,3) z,y,x
        fib = fib / (np.linalg.norm(fib, axis=1, keepdims=True) + 1e-9)
        fibre_z = float(np.abs(fib[:, 0]).mean())           # |fibre·ẑ| → recto/verso
        coh = float(_bilinear_vec(coherence[..., None], pts)[:, 0].mean())
        d = np.diff(pts, axis=0)
        seg = np.hypot(d[:, 0], d[:, 1])
        length_px = float(seg.sum())
        ang = np.arctan2(d[:, 1], d[:, 0])                  # tangent orientation mod π
        mang = 0.5 * np.arctan2(np.sin(2 * ang).mean(), np.cos(2 * ang).mean())
        dens = tree.query_ball_point(c, r_px, return_length=True) if tree is not None else 1
        if phi_field is not None:
            mean_phi = float(_bilinear_vec(phi_field[..., None], pts)[:, 0].mean()) * voxel_um / 1000.0
        else:
            mean_phi = 0.0
        rows.append([abs_k, signed_k, 0.0, 0.0, 0.0, mang, fibre_z, coh,
                     float(np.log1p(dens)), length_px * voxel_um / 1000.0, mean_phi])
        centroids.append(c)
        mean_norms.append(mean_n)

    X = np.asarray(rows, np.float64)
    centroids = np.asarray(centroids, np.float64)
    mean_norms = np.asarray(mean_norms, np.float64)
    center, cond = (umbilicus_center, 0.0)
    if len(X):
        # normal_proj: centroid on the WINDOW mean-normal axis (robust local radial coordinate).
        axis = mean_norms.mean(0)
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        X[:, 2] = (centroids @ axis) * voxel_um / 1000.0
        # umbilicus r / θ: separates genuinely-gapped or radially-offset sheets (complements
        # curvature, which separates stuck-together-but-differently-bent ones). Ill-conditioned in a
        # small window (normals near-parallel) — cond flags that; normal_proj is the safe fallback.
        if center is None:
            center, cond = estimate_umbilicus(centroids, mean_norms)
        rel = centroids - center
        X[:, 3] = np.hypot(rel[:, 0], rel[:, 1]) * voxel_um / 1000.0
        X[:, 4] = np.arctan2(rel[:, 1], rel[:, 0])
    extra = {"centroids": centroids, "mean_normals": mean_norms,
             "umbilicus": center, "umbilicus_cond": cond}
    return X, list(FEATURE_NAMES), extra


def cluster_streamlines(X: np.ndarray, feature_subset: list[str] | None = None,
                        eps: float = 0.6, min_samples: int = 4):
    """DBSCAN on standardized features. Returns ``(labels, used_names)``; label -1 = noise.

    ``feature_subset`` selects which columns to cluster on (default: the discriminative set that
    excludes raw length). Standardization puts every feature on comparable scale so ``eps`` is in
    z-score units."""
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    names = list(FEATURE_NAMES)
    if feature_subset is None:
        # Between-sheet separators + along-sheet connectivity:
        #  - umbilicus_r (which radial layer; separates gapped/offset sheets)
        #  - abs_curvature (≈1/R; separates stuck-together-but-differently-bent sheets)
        #  - tangent_angle (varies SMOOTHLY along a sheet → each sheet is a continuous density-
        #    connected filament DBSCAN can trace through hard bends; ≠ a fragmenting split)
        #  - mean_coherence (rejects the low-threshold noise streamlines)
        # Deliberately NOT fibre_z_frac: recto/verso is a DISCRETE intra-sheet split that would
        # fork one physical sheet into two clusters.
        feature_subset = ["abs_curvature", "umbilicus_r_mm", "tangent_angle", "mean_coherence"]
    cols = [names.index(f) for f in feature_subset]
    if len(X) == 0:
        return np.zeros(0, int), feature_subset
    Xs = StandardScaler().fit_transform(X[:, cols])
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Xs)
    return labels, feature_subset


# --------------------------------------------------------------------------------------------
# End-to-end helpers so the offline script and the in-app button run the SAME code.
# --------------------------------------------------------------------------------------------
def otsu_threshold(block: np.ndarray) -> float:
    """Otsu threshold over the non-zero voxels of ``block`` (papyrus vs air/background)."""
    v = block[block > 0]
    if v.size == 0:
        return 0.0
    hist, edges = np.histogram(v, bins=256)
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    denom = omega * (1 - omega)
    denom[denom == 0] = 1e-12
    sigma_b = (mu[-1] * omega - mu) ** 2 / denom
    return float(edges[int(np.argmax(sigma_b))])


def period_from_phi(phi, material, pmin_px=3.0, pmax_px=80.0, nbins=240):
    """Layering period (px) derived from φ itself: φ is the across-sheet coordinate (∇φ = unit
    normal → φ is in px), so the histogram of φ over material is periodic — peaks at sheet centres,
    spaced by the period. Return the first autocorrelation peak of that histogram. Data-driven,
    adaptive (tracks compression), level-agnostic; no physical constant, no fibre confusion."""
    v = phi[material]
    if v.size < 300:
        return float("nan")
    lo, hi = np.percentile(v, [1, 99])
    if hi - lo < pmin_px:
        return float("nan")
    hist, edges = np.histogram(v, bins=nbins, range=(lo, hi))
    binw = (hi - lo) / nbins                         # φ units (px) per bin
    hist = hist.astype(np.float64) - hist.mean()
    ac = np.correlate(hist, hist, "full")[len(hist) - 1:]
    lag_min = max(2, int(pmin_px / binw))
    lag_max = min(len(ac) - 2, int(pmax_px / binw))
    for k in range(lag_min, lag_max):
        if ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1] and ac[k] > 0:
            return float(k * binw)
    return float("nan")


def _line_phi_px(polylines, phi_field):
    """Mean layer-potential φ (in px) per streamline — the ONLY streamline_features column the φ /
    moment clustering needs. Computed directly so the full (expensive) feature matrix can be skipped."""
    if phi_field is None:
        return np.zeros(len(polylines))
    return np.array([float(_bilinear_vec(phi_field[..., None], pl)[:, 0].mean()) if len(pl) else 0.0
                     for pl in polylines])


def _phi_spreads(polylines, phi_field, voxel_um):
    """Per-streamline within-line φ spread (mm): how much φ varies ALONG each line. On a valid
    (single-sheet) line this is ~0; it is the LOCAL, measured 'same-sheet' scale — no period."""
    if phi_field is None:
        return np.zeros(len(polylines))
    out = []
    for pl in polylines:
        v = _bilinear_vec(phi_field[..., None], pl)[:, 0]
        out.append((float(v.max()) - float(v.min())) * voxel_um / 1000.0 if len(v) else 0.0)
    return np.asarray(out)


def label_by_flow(polylines, phi_px, spread_px, min_sep_px=3.5, gap_px=6.0,
                  phi_tol_k=2.0, max_angle_deg=30.0, max_span_px=None, contact_frac=0.4):
    """Period-FREE within-window sheet labelling by FLOW (not φ-value bucketing, which chains when
    fragments densely fill φ). Two fragments are the SAME sheet iff they are spatially ADJACENT
    (points within ``min_sep_px`` — smaller than the inter-sheet gap, so it cannot chain to a
    neighbour) AND tangent-aligned AND on the same φ level; plus end-to-end CONTINUATION (collinear
    endpoints within ``gap_px``) to bridge along-sheet breaks. The φ tolerance is set from the
    streamlines' OWN within-line φ spread (measured locally) × ``phi_tol_k`` — no period, no quantum.

    HARD non-crossing cap: ``max_span_px`` bounds a cluster's extent along the normal (= its φ-span,
    since φ is the across-sheet coordinate) to one papyrus-sheet THICKNESS (~300 µm, a squish-robust
    material constant, NOT a period). Any union that would push a component past the cap is refused,
    so a chain through a pinch point cannot swallow the neighbouring sheet. Connected components =
    sheets. Returns per-fragment labels (0..K-1)."""
    n = len(polylines)
    if n <= 1:
        return np.zeros(n, int)
    parent = list(range(n))
    pmin = np.asarray(phi_px, float).copy()
    pmax = np.asarray(phi_px, float).copy()

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if max_span_px is not None:
            lo, hi = min(pmin[ra], pmin[rb]), max(pmax[ra], pmax[rb])
            if hi - lo > max_span_px:
                return                                     # would exceed one sheet thickness → refuse
        parent[ra] = rb
        pmin[rb], pmax[rb] = min(pmin[ra], pmin[rb]), max(pmax[ra], pmax[rb])

    # per-fragment mean orientation (mod π)
    tang = np.zeros((n, 2))
    for i, pl in enumerate(polylines):
        d = np.diff(pl, axis=0)
        ang = np.arctan2(d[:, 1], d[:, 0])
        a2 = 0.5 * np.arctan2(np.sin(2 * ang).sum(), np.cos(2 * ang).sum())
        tang[i] = [np.cos(a2), np.sin(a2)]
    phi_tol = max(phi_tol_k * spread_px, 1.0)
    cos_a = np.cos(np.radians(max_angle_deg))

    # proximity edges with a DIVERGENCE loss: same-sheet fragments run alongside each other over a
    # SUSTAINED span (many points close); two sheets meeting at a pinch only touch briefly then
    # diverge. So require the close-contact count to be ≥ contact_frac of the shorter fragment — a
    # single-point touch (a pinch) is refused. Delamination (small, same-sheet) still passes.
    allpts = np.concatenate(polylines)
    owner = np.concatenate([np.full(len(pl), i) for i, pl in enumerate(polylines)])
    lens = np.array([len(pl) for pl in polylines])
    tree = cKDTree(allpts)
    contact: dict = {}
    for i, j in tree.query_pairs(min_sep_px):
        fi, fj = int(owner[i]), int(owner[j])
        if fi == fj:
            continue
        key = (fi, fj) if fi < fj else (fj, fi)
        contact[key] = contact.get(key, 0) + 1
    for (fi, fj), cnt in contact.items():
        if abs(phi_px[fi] - phi_px[fj]) > phi_tol:
            continue
        if abs(float(tang[fi] @ tang[fj])) < cos_a:
            continue
        if cnt < contact_frac * min(int(lens[fi]), int(lens[fj])):
            continue                                       # brief touch → diverging → different sheet
        union(fi, fj)

    # continuation edges (collinear endpoints bridge along-sheet breaks)
    ep_pos, ep_tan, ep_fid = [], [], []
    for i, pl in enumerate(polylines):
        k = min(4, len(pl) - 1)
        for p, q in ((pl[0], pl[k]), (pl[-1], pl[-1 - k])):
            tg = np.asarray(p, float) - np.asarray(q, float)
            nn = float(np.hypot(tg[0], tg[1]))
            if nn < 1e-6:
                continue
            ep_pos.append(np.asarray(p, float))
            ep_tan.append(tg / nn)
            ep_fid.append(i)
    if ep_pos:
        ep_pos = np.asarray(ep_pos)
        ep_tan = np.asarray(ep_tan)
        ep_fid = np.asarray(ep_fid)
        etree = cKDTree(ep_pos)
        for a, b in etree.query_pairs(gap_px):
            fa, fb = int(ep_fid[a]), int(ep_fid[b])
            if fa == fb or abs(phi_px[fa] - phi_px[fb]) > phi_tol:
                continue
            d = ep_pos[b] - ep_pos[a]
            g = float(np.hypot(d[0], d[1]))
            if g < 1e-6:
                union(fa, fb)
                continue
            dh = d / g
            if (ep_tan[a] @ dh > cos_a and ep_tan[b] @ (-dh) > cos_a
                    and ep_tan[a] @ (-ep_tan[b]) > cos_a):
                union(fa, fb)

    roots = [find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(sorted(set(roots)))}
    return np.array([remap[r] for r in roots], int)


def _streamlet_components(polylines, min_sep_px, gap_px, max_angle_deg, phi=None, max_dphi=None):
    """Connectivity graph of streamlet fragments → (ncomp, component-labels). Edges: side-by-side
    along-tangent adjacency (points within ``min_sep_px`` AND tangents aligned) + collinear MOMENTUM
    continuation (endpoints within ``gap_px``, end-directions pointing at each other). Merges are only
    ever allowed WITHIN a component, so spatially-disconnected fragments stay separate and nothing
    chains across the inter-sheet gap. If ``phi`` + ``max_dphi`` are given, an edge is REFUSED when the
    two fragments' φ differ by more than ``max_dphi`` — so the connectivity can't promote a segmentlet
    across a band even when it is spatially close (fix for band-jumping). Membership backbone = these
    components (topological)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(polylines)
    if n == 0:
        return 0, np.zeros(0, int)

    def band_ok(fa, fb):                                        # vectorized φ-band gate on fragment pairs
        if phi is None or max_dphi is None:
            return np.ones(len(fa), bool)
        return np.abs(np.asarray(phi)[fa] - np.asarray(phi)[fb]) <= max_dphi

    tang = np.zeros((n, 2))
    for i, pl in enumerate(polylines):
        d = np.diff(pl, axis=0)
        ang = np.arctan2(d[:, 1], d[:, 0]) if len(d) else np.zeros(1)
        a2 = 0.5 * np.arctan2(np.sin(2 * ang).sum(), np.cos(2 * ang).sum())
        tang[i] = (np.cos(a2), np.sin(a2))
    allpts = np.concatenate(polylines)
    owner = np.concatenate([np.full(len(pl), i) for i, pl in enumerate(polylines)])
    cos_a = np.cos(np.radians(max_angle_deg))
    r_all, c_all = [], []

    # side-by-side along-tangent adjacency — VECTORIZED (no python loop over the ~quadratic pair pile)
    pairs = cKDTree(allpts).query_pairs(min_sep_px, output_type="ndarray") if len(allpts) else \
        np.zeros((0, 2), int)
    if len(pairs):
        fa, fb = owner[pairs[:, 0]], owner[pairs[:, 1]]
        keep = (fa != fb)
        keep &= np.abs(np.einsum("ij,ij->i", tang[fa], tang[fb])) >= cos_a
        keep &= band_ok(fa, fb)
        r_all.append(fa[keep])
        c_all.append(fb[keep])

    # collinear MOMENTUM continuation edges — also vectorized
    ep_pos, ep_tan, ep_fid = [], [], []
    for i, pl in enumerate(polylines):
        k = min(4, len(pl) - 1)
        for p, q in ((pl[0], pl[k]), (pl[-1], pl[-1 - k])):
            tg = np.asarray(p, float) - np.asarray(q, float)
            nn = float(np.hypot(tg[0], tg[1]))
            if nn < 1e-6:
                continue
            ep_pos.append(np.asarray(p, float))
            ep_tan.append(tg / nn)
            ep_fid.append(i)
    if ep_pos:
        ep_pos, ep_tan, ep_fid = np.asarray(ep_pos), np.asarray(ep_tan), np.asarray(ep_fid)
        ep = cKDTree(ep_pos).query_pairs(gap_px, output_type="ndarray")
        if len(ep):
            fa, fb = ep_fid[ep[:, 0]], ep_fid[ep[:, 1]]
            d = ep_pos[ep[:, 1]] - ep_pos[ep[:, 0]]
            g = np.hypot(d[:, 0], d[:, 1])
            dh = d / (g[:, None] + 1e-12)
            ta, tb = ep_tan[ep[:, 0]], ep_tan[ep[:, 1]]
            collinear = (g < 1e-6) | ((np.einsum("ij,ij->i", ta, dh) > cos_a)
                                      & (np.einsum("ij,ij->i", tb, -dh) > cos_a)
                                      & (np.einsum("ij,ij->i", ta, -tb) > cos_a))
            keep = (fa != fb) & collinear & band_ok(fa, fb)
            r_all.append(fa[keep])
            c_all.append(fb[keep])

    rows = np.concatenate(r_all) if r_all else np.zeros(0, int)
    cols = np.concatenate(c_all) if c_all else np.zeros(0, int)
    data = np.ones(2 * len(rows))
    graph = coo_matrix((data, (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
                       shape=(n, n))
    return connected_components(graph, directed=False)


def cluster_by_phi_hac(polylines, phi_px, pitch_px, min_sep_px, gap_px=None, max_angle_deg=35.0,
                       cut_frac=0.85):
    """Connectivity-constrained HAC on the across-sheet φ (distance |Δφ|), complete-linkage cut at
    ``cut_frac × pitch`` per connected component. De-brittled two ways: (1) the cut sits ABOVE one
    sheet THICKNESS (``cut_frac`` near-pitch) so the two bread-faces of one sandwich stay one cluster,
    yet BELOW a full pitch so adjacent windings still split; (2) the connectivity is φ-band-gated
    (``max_dphi = ½ pitch``) so a segmentlet can't be promoted to the next band even when spatially
    close. φ is used only LOCALLY here (per connected component), which sidesteps its cumulative drift."""
    from scipy.cluster.hierarchy import fcluster, linkage

    n = len(polylines)
    if n == 0:
        return np.zeros(0, int), {"merge_heights": np.zeros(0), "pitch_px": pitch_px, "cut_px": 0.0}
    if gap_px is None:
        gap_px = 2.0 * min_sep_px
    phi = np.asarray(phi_px, float)
    ncomp, comp = _streamlet_components(polylines, min_sep_px, gap_px, max_angle_deg,
                                        phi=phi, max_dphi=0.5 * pitch_px)
    cut = cut_frac * pitch_px
    out = np.full(n, -1, int)
    nextlab = 0
    heights: list = []
    for c in range(ncomp):
        idx = np.where(comp == c)[0]
        if len(idx) == 1:
            out[idx[0]] = nextlab
            nextlab += 1
            continue
        z = linkage(phi[idx][:, None], method="complete")
        heights.extend(z[:, 2].tolist())
        sub = fcluster(z, t=cut, criterion="distance")
        for s in np.unique(sub):
            out[idx[sub == s]] = nextlab
            nextlab += 1
    return out, {"merge_heights": np.asarray(heights), "pitch_px": float(pitch_px), "cut_px": float(cut)}


def streamlet_moment_features(polylines, normal_plane, max_order=4):
    """Per-streamlet MOMENT coordinates (drift-free, unlike the path-integrated φ): the centroid's
    ACROSS-NORMAL position (`c·n̂` — the actual across-sheet position, so a bump on a neighbour can't
    shift it) plus scale-normalized central moments up to ``max_order`` (shape descriptors that
    discriminate touching windings). Returns ``(F (n, d), names)``. Central moments beyond ~4 are tail
    noise, so ``max_order`` ≈ 4 is the sweet spot."""
    n = len(polylines)
    pairs = [(p, o - p) for o in range(2, max_order + 1) for p in range(o + 1)]  # all p+q=o, 2≤o≤max
    names = ["c_normal"] + [f"eta{p}{q}" for (p, q) in pairs]
    H, W = normal_plane.shape[:2]
    feat = np.zeros((n, len(names)))
    for i, pl in enumerate(polylines):
        x, y = pl[:, 0], pl[:, 1]
        cx, cy = float(x.mean()), float(y.mean())
        yi = int(np.clip(round(cy), 0, H - 1))
        xi = int(np.clip(round(cx), 0, W - 1))
        nx, ny = float(normal_plane[yi, xi, 2]), float(normal_plane[yi, xi, 1])  # in-plane normal (x,y)
        nn = np.hypot(nx, ny) + 1e-9
        feat[i, 0] = (cx * nx + cy * ny) / nn                  # centroid across-normal position (px)
        dx, dy = x - cx, y - cy
        rms = float(np.sqrt((dx * dx + dy * dy).mean())) + 1e-6
        for j, (p, q) in enumerate(pairs):
            feat[i, 1 + j] = float((dx ** p * dy ** q).mean()) / (rms ** (p + q))
    return feat, names


def cluster_by_moments(polylines, normal_plane, min_sep_px, gap_px, max_order=4, cut_z=2.0,
                       max_angle_deg=35.0):
    """Cluster streamlets by MOMENT coordinates — the agreed membership basis. Connectivity-constrained
    per-component COMPLETE-linkage HAC on the STANDARDIZED feature vector [across-normal centroid +
    central moments ≤ ``max_order``]. The connectivity components are the topological membership
    backbone (along-tangent + momentum), and the moments split touching windings within a component
    (across-normal centroid separates parallel sheets, higher moments separate by shape). Replaces the
    path-integrated φ. Returns ``(labels, info)`` with merge heights + feature names."""
    from scipy.cluster.hierarchy import fcluster, linkage

    n = len(polylines)
    if n == 0:
        return np.zeros(0, int), {"merge_heights": np.zeros(0), "cut_z": cut_z, "feat_names": []}
    feat, names = streamlet_moment_features(polylines, normal_plane, max_order)
    fs = (feat - feat.mean(0)) / (feat.std(0) + 1e-9)          # standardize → comparable z-score units
    ncomp, comp = _streamlet_components(polylines, min_sep_px, gap_px, max_angle_deg)
    out = np.full(n, -1, int)
    nextlab = 0
    heights: list = []
    for c in range(ncomp):
        idx = np.where(comp == c)[0]
        if len(idx) == 1:
            out[idx[0]] = nextlab
            nextlab += 1
            continue
        z = linkage(fs[idx], method="complete", metric="euclidean")
        heights.extend(z[:, 2].tolist())
        sub = fcluster(z, t=cut_z, criterion="distance")
        for s in np.unique(sub):
            out[idx[sub == s]] = nextlab
            nextlab += 1
    return out, {"merge_heights": np.asarray(heights), "cut_z": float(cut_z), "feat_names": names}


def second_pass_merge(polylines, labels, phi_px, curv, voxel_um, gap_px=25.0, phi_tol_px=6.0,
                      max_angle_deg=35.0, curv_rel_tol=0.6, max_span_px=None):
    """SECOND within-window pass: re-associate the deliberately OVER-segmented flow clusters into
    coherent sheets using the FULL cluster descriptor — end-to-end CONTINUITY (nearest endpoints
    within ``gap_px``, collinear with both clusters' tangents) + TANGENT agreement + CURVATURE
    (κ≈1/R, the radial-layer index) + φ-LEVEL (same across-sheet coordinate). It bridges the longer
    along-sheet breaks that point-level flow could not, while the sheet-thickness φ-span cap
    (``max_span_px``) forbids merging into a neighbouring sheet. Different sheets sit at different φ,
    so the φ-level gate is the non-crossing guard. Returns new per-fragment labels."""
    labels = np.asarray(labels)
    uniq = [lb for lb in sorted(set(labels.tolist())) if lb >= 0]
    if len(uniq) <= 1:
        return labels
    cos_a = np.cos(np.radians(max_angle_deg))
    cl = {}
    for lb in uniq:
        idx = np.where(labels == lb)[0]
        pts = np.concatenate([polylines[i] for i in idx], axis=0)
        d = np.concatenate([np.diff(polylines[i], axis=0) for i in idx], axis=0)
        ang = np.arctan2(d[:, 1], d[:, 0]) if len(d) else np.zeros(1)
        a2 = 0.5 * np.arctan2(np.sin(2 * ang).sum(), np.cos(2 * ang).sum())
        t = np.array([np.cos(a2), np.sin(a2)])
        proj = pts @ t
        cl[lb] = {"phi": float(np.mean(phi_px[idx])), "curv": float(np.median(curv[idx])),
                  "pmin": float(np.min(phi_px[idx])), "pmax": float(np.max(phi_px[idx])),
                  "t": t, "lo": pts[int(np.argmin(proj))], "hi": pts[int(np.argmax(proj))]}

    parent = {lb: lb for lb in uniq}
    prange = {lb: [cl[lb]["pmin"], cl[lb]["pmax"]] for lb in uniq}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        lo, hi = min(prange[ra][0], prange[rb][0]), max(prange[ra][1], prange[rb][1])
        if max_span_px is not None and hi - lo > max_span_px:
            return
        parent[ra] = rb
        prange[rb] = [lo, hi]

    for a in range(len(uniq)):
        ca = cl[uniq[a]]
        for b in range(a + 1, len(uniq)):
            cb = cl[uniq[b]]
            if abs(ca["phi"] - cb["phi"]) > phi_tol_px:
                continue                                   # different φ level → different sheet
            if abs(ca["curv"] - cb["curv"]) / (abs(ca["curv"]) + abs(cb["curv"]) + 1e-6) > curv_rel_tol:
                continue
            if abs(float(ca["t"] @ cb["t"])) < cos_a:
                continue
            best = None
            for pa in (ca["lo"], ca["hi"]):
                for pb in (cb["lo"], cb["hi"]):
                    g = float(np.hypot(*(pb - pa)))
                    if best is None or g < best[0]:
                        best = (g, pb - pa)
            g, dv = best
            if g > gap_px:
                continue
            if g > 1e-6:
                dh = dv / g
                if abs(float(ca["t"] @ dh)) < cos_a or abs(float(cb["t"] @ dh)) < cos_a:
                    continue                               # not a collinear continuation
            union(uniq[a], uniq[b])

    newid: dict = {}
    out = labels.copy()
    for i, lb in enumerate(labels):
        if lb < 0:
            continue
        r = find(int(lb))
        if r not in newid:
            newid[r] = len(newid)
        out[i] = newid[r]
    return out


def _estimate_period_px(block, voxel_um, sigma_tensor):
    """Median layering period (px) over the block, from the compression/streamwise-wavenumber
    field. Used to set the φ→sheet-index quantum and the DBSCAN eps."""
    from hercunet.labels.fields.structure_tensor import compression_field
    try:
        cf = compression_field(block, voxel_um=voxel_um, sigma_tensor=sigma_tensor)
        per, coh = cf["period_um"], cf["coherence"]
        good = np.isfinite(per) & (coh > 0.3) & (per > 0)
        if good.any():
            return float(np.median(per[good])) / voxel_um
    except Exception:
        pass
    return 40.0  # fallback ~ 100 µm at 2.4 µm/vox


MESH_WINDOW_UM = 1500.0   # scale-aware physical window edge for sheet segmentation. ~1.5 mm because the
#                           grand-prize scans are COARSE (8.6 µm → 1.5 mm is only ~174 px; 0.5 mm was a
#                           useless 58 px). Fixed PIXELS aren't scale-aware; but a fixed PHYSICAL size
#                           explodes at fine resolution (1.5 mm @ 1.1 µm = 1329 px), so cap the budget.
MESH_WINDOW_MAX_PX = 224  # compute/DNN pixel cap; fine scrolls get a smaller physical window (still
#                           plenty of pixels), coarse scrolls get the full 1.5 mm. Larger windows also
#                           catch more bends → lean on the quality gate to mask those, not on shrinking.
MESH_WINDOW_Z_UM = 500.0  # the window is ANISOTROPIC: thin along z (across the normal — the weak,
#                           deform-prone axis) and wide in xy. 0.5 mm of z is enough to bootstrap, and
#                           z can be augmented by rotation. Fewer voxels → cheaper field.


def window_px(voxel_um):
    """Scale-aware in-plane (xy) window edge in voxels: ``MESH_WINDOW_UM`` at this resolution, clamped
    to ``[32, MESH_WINDOW_MAX_PX]`` so compute stays bounded on fine scrolls."""
    return int(np.clip(round(MESH_WINDOW_UM / voxel_um), 32, MESH_WINDOW_MAX_PX))


def window_z_px(voxel_um):
    """Scale-aware z (across-normal) window depth in voxels: ``MESH_WINDOW_Z_UM``, clamped to
    ``[16, 128]`` — the thin axis of the anisotropic window."""
    return int(np.clip(round(MESH_WINDOW_Z_UM / voxel_um), 16, 128))


def mesh_params(voxel_um):
    """Canonical 2-D fiber-mesh / sheet-clustering parameters — the SINGLE source of truth shared by
    the UI '2D fiber mesh' button and the render scripts so they can never drift. Pass as ``**kwargs``
    to :func:`compute_from_block`. All scale-aware (µm-derived), so it behaves the same at any level."""
    return {
        "sigma_tensor": max(3.0, 20.0 / voxel_um),
        # CED: 8 iters, steer ONCE (recompute_every=0, set at the call sites). Validated 2026-07-26 to be
        # MORE coherent than the old 24-iter/refresh-every-8 regime — fewer iters keep the fibre field from
        # over-diffusing sheets together, so clusters stop merging at dodgy junctions. The end-field is
        # re-estimated from the fully-diffused block anyway (structure_tensor_frame), so a light CED suffices.
        "ced_iters": 8,
        "ced_mode": "sheet",
        # Shorter, gap-shier streamlets (validated same day): capping length to ~300 µm and tightening the
        # φ non-crossing gate (cross_span 35→20 µm ≈ 2 px at 8.6 µm — the lever that actually bites, since
        # gap_tol floors at 2 px) stops lines jumping into neighbouring sheets. The per-line embedding
        # re-connects the resulting short fragments into long sheets with no trouble.
        "cross_span_um": 20.0,
        "max_len_um": 300.0,
        "gap_tol_um": 10.0,
        "seed_stride_um": 6.0,
        "trace_min_sep_um": 8.0,
        "min_coh": 0.12,
        "cluster_method": "hac",
        # momentum=0: the coasting/gap budget is CONSTANT (=gap_tol), not length-scaled. The old
        # length-scaled budget was a self-reinforcing loop (long line → jumps a gap → longer → jumps
        # more). The embedding now clusters fragments robustly, so short clean lines are fine.
        "momentum": 0.0,
    }


def compute_from_block(block, zc_in_block, voxel_um, sigma_tensor=10.0, min_coh=0.15,
                       eps=0.6, min_samples=4, use_gpu=True, ced_iters=24, use_phi=True,
                       phi_3d=False, ced_mode="sheet", cross_frac=0.15, second_pass=True,
                       max_normal_span_um=300.0, gap_tol_um=20.0, seed_stride_um=10.0,
                       trace_min_sep_um=15.0, flow_min_sep_um=17.0, flow_gap_um=30.0,
                       second_gap_um=120.0, second_phi_tol_um=30.0, cross_span_um=35.0,
                       momentum=0.0, max_len_frac=1.0, max_len_um=None, min_len_um=50.0,
                       moment_order=4, moment_cut_z=2.0,
                       cluster_method="hac", ff=None, phi_vol_in=None):
    """Full pipeline on a pre-read ``(dz, H, W)`` block. All spatial thresholds are PHYSICAL
    micrometres, converted to pixels via ``voxel_um`` — so one call behaves identically across scan
    resolutions (a 90 µm sheet is ~9 px at 9.4 µm but ~19 px at 4.8 µm; the µm params absorb that).
    Knobs:
      * ``ced_iters`` > 0 — pre-clean the block with sheet-enhancing anisotropic diffusion (CED).
      * ``use_phi`` — group streamlines by the layer-potential φ via period-FREE flow clustering
        (:func:`label_by_flow`) + a second-pass merge; the tracer uses φ for its non-crossing gate.
      * ``cross_frac`` — φ purity-rate for the tractography non-crossing gate (dimensionless → already
        resolution-independent). ``*_um`` — physical thresholds: coast gap, seed spacing, line
        separation, flow proximity/continuation, second-pass continuation + φ tolerance, and the
        sheet-thickness span cap.
    Returns a result dict consumable by :func:`build_figures`."""
    gap_tol = max(2, int(round(gap_tol_um / voxel_um)))
    seed_stride = max(1, int(round(seed_stride_um / voxel_um)))
    trace_min_sep = trace_min_sep_um / voxel_um
    from hercunet.labels.fields.structure_tensor import structure_tensor_frame
    from .tractography import ridge_mask, sheet_tangent_from_normal, trace_streamlines

    zi0 = int(np.clip(zc_in_block, 0, block.shape[0] - 1))
    raw_slice = np.array(block[zi0], np.float32)            # keep raw for the CED before/after
    # ``ff`` may be a PRECOMPUTED 3-D frame field (structure-tensor over the block) — pass it to segment
    # many slices of one block without recomputing the tensor per slice (CED assumed already applied).
    if ced_iters and ced_iters > 0 and ff is None:
        try:
            from hercunet.labels.fields.anisotropic import coherence_enhancing_diffusion
            block = coherence_enhancing_diffusion(block, iters=int(ced_iters), mode=ced_mode,
                                                  sigma_tensor=sigma_tensor, recompute_every=0)
        except Exception:
            pass

    if ff is None and use_gpu:
        try:
            from hercunet.labels.fields.structure_tensor_torch import structure_tensor_frame_torch, torch_available
            if torch_available():
                ff = structure_tensor_frame_torch(block, 1.0, sigma_tensor)
        except Exception:
            ff = None
    if ff is None:
        ff = structure_tensor_frame(block, 1.0, sigma_tensor)

    zi = int(np.clip(zc_in_block, 0, block.shape[0] - 1))
    normal, fibre, coh = ff["normal"][zi], ff["fibre"][zi], ff["coherence"][zi]
    material = block[zi] > otsu_threshold(block)
    tan = sheet_tangent_from_normal(normal)
    ridge = ridge_mask(material, normal)

    # φ FIRST (before tracing) so the tracer can use the period-free non-crossing gate: sheets are
    # level sets of φ, so a valid line keeps φ ≈ const; the tracer stops when dφ/ds says it is moving
    # ACROSS sheets rather than along one. This is what keeps lines from jumping into a neighbour.
    phi_field = None
    phi_vol = None
    if use_phi:
        from .layer_potential import layer_potential
        if phi_vol_in is not None:                          # precomputed 3-D φ (reused across slices)
            phi_field = phi_vol_in[zi]
        elif phi_3d:
            phi_vol, _ = layer_potential(ff["normal"])     # φ over the WHOLE 3-D block
            phi_field = phi_vol[zi]
        else:
            phi_field, _ = layer_potential(normal)

    # Streamlet max length. Prefer the SCALE-AWARE µm cap (``max_len_um``, resolution-robust): per-direction
    # steps = 0.5·max_len_um/voxel_um so the total physical length ≈ max_len_um at ANY scan resolution.
    # Fall back to the old window-fraction cap only when max_len_um is None.
    if max_len_um is not None:
        max_steps = int(max(30, 0.5 * max_len_um / voxel_um))       # per-direction; total ≈ max_len_um
    else:
        max_steps = int(max(30, max_len_frac * 0.5 * block.shape[2]))   # per-direction; total ≈ max_len_frac·W
    xs, ys = trace_streamlines(tan, coh, material=material, seed_mask=ridge, seed_stride=seed_stride,
                               min_coh=min_coh, min_sep=trace_min_sep, max_steps=max_steps,
                               gap_tol=gap_tol, phi=phi_field, cross_frac=cross_frac if use_phi else None,
                               max_phi_drift=(cross_span_um / voxel_um) if (use_phi and cross_span_um) else None,
                               momentum=momentum, min_len=min_len_um / voxel_um,
                               use_gpu=False)  # GPU tracer written but UNVALIDATED (benchmark hung) — keep
    #                                            the CPU reference path live until trace_streamlines_gpu is debugged
    polylines = split_streamlines(xs, ys)
    res = {"img": block[zi], "raw_slice": raw_slice, "ced_slice": np.array(block[zi], np.float32),
           "polylines": polylines, "voxel_um": voxel_um, "zi": zi,
           "n_streamlines": len(polylines), "names": list(FEATURE_NAMES),
           "normal_plane": normal, "fibre_plane": fibre, "coh_plane": coh, "phi_field": phi_field,
           "used_phi": bool(use_phi), "used_ced": bool(ced_iters and ced_iters > 0),
           "used_phi_3d": bool(use_phi and phi_3d)}
    if len(polylines) < 5:
        res.update(X=np.zeros((0, len(FEATURE_NAMES))), labels=np.zeros(0, int),
                   extra={"umbilicus": np.zeros(2), "umbilicus_cond": np.inf}, used=[])
        return res

    names, extra = list(FEATURE_NAMES), {"umbilicus": np.zeros(2), "umbilicus_cond": np.inf}
    if use_phi:
        # Only mean-φ per line is needed for the φ / moment clustering — compute it DIRECTLY and SKIP
        # the full streamline_features (umbilicus least-squares + per-line curvature) the HAC path never
        # uses. This is the per-slice speed-up (B). φ-value bucketing → flow; membership → hac/moments.
        phi_px = _line_phi_px(polylines, phi_field)
        res["phi_px"] = phi_px
        X = np.zeros((len(polylines), len(FEATURE_NAMES)))    # placeholder (features unused by φ/moments)
        if cluster_method == "moments":
            # MOMENT coordinates (drift-free): centroid-⊥ + central moments, connectivity-constrained HAC
            labels, minfo = cluster_by_moments(polylines, normal, flow_min_sep_um / voxel_um,
                                               flow_gap_um / voxel_um, max_order=moment_order,
                                               cut_z=moment_cut_z)
            res["moments"] = minfo
            res["period_px"] = float("nan")
            used = ["moments (centroid⊥ + central-moments≤%d, connectivity HAC)" % moment_order]
        elif cluster_method == "hac":
            # Connectivity-constrained HAC: membership coord = φ only, |Δφ| distance, along-tangent
            # adjacency components, complete-linkage cut < one pitch. Shared by UI / linker / scripts.
            pitch = period_from_phi(phi_field, material)
            if not np.isfinite(pitch) or pitch <= 0:
                pitch = _estimate_period_px(block, voxel_um, sigma_tensor)
            labels, hac_info = cluster_by_phi_hac(polylines, phi_px, pitch,
                                                  min_sep_px=flow_min_sep_um / voxel_um,
                                                  gap_px=flow_gap_um / voxel_um)
            res["hac"] = hac_info
            res["period_px"] = float(pitch)
            used = ["hac (φ-distance, along-tangent connectivity, complete-linkage, cut<pitch)"]
        else:
            spreads_mm = _phi_spreads(polylines, phi_field, voxel_um)
            spread_px = (float(np.median(spreads_mm)) * 1000.0 / voxel_um) if len(spreads_mm) else 0.0
            labels = label_by_flow(polylines, phi_px, spread_px,
                                   min_sep_px=flow_min_sep_um / voxel_um,
                                   gap_px=flow_gap_um / voxel_um,
                                   max_span_px=max_normal_span_um / voxel_um)
            if second_pass and len(polylines) > 1:
                curv_arr = np.array([abs(polyline_curvature(pl, voxel_um)[1]) if len(pl) >= 5 else 0.0
                                     for pl in polylines])
                labels = second_pass_merge(polylines, labels, phi_px, curv_arr, voxel_um,
                                           gap_px=second_gap_um / voxel_um,
                                           phi_tol_px=second_phi_tol_um / voxel_um,
                                           max_span_px=max_normal_span_um / voxel_um)
            res["period_px"] = float("nan")
            res["phi_spread_mm"] = float(np.median(spreads_mm)) if len(spreads_mm) else 0.0
            used = ["flow (proximity+continuation, φ-gated, period-free)"]
    else:
        X, names, extra = streamline_features(polylines, normal, fibre, coh, voxel_um,
                                              phi_field=phi_field)
        labels, used = cluster_streamlines(X, eps=eps, min_samples=min_samples)
        res["period_px"] = float("nan")
    res.update(X=X, labels=labels, extra=extra, used=used)
    return res


def cluster_colors(labels, polylines=None, voxel_um=None, adj_um=180.0):
    """Map cluster labels → RGBA. Noise (-1) = translucent grey. Uses a 60-colour DISTINCT palette
    (tab20 + tab20b + tab20c) instead of a 20-colour ramp, so 40–50 clusters no longer alias onto muddy
    near-duplicates. When ``polylines`` are given (aligned with ``labels``), colours are assigned by GREEDY
    GRAPH COLOURING on spatial adjacency: clusters whose streamlines come within ``adj_um`` µm (converted via
    ``voxel_um``; a point-cloud-extent fraction if not given) are forced to DIFFERENT palette entries, so two
    touching sheets can never share a colour and *look* merged. Without polylines, labels are scatter-assigned
    across the palette (stride) to avoid consecutive-label similarity."""
    import matplotlib.pyplot as plt
    labels = np.asarray(labels)
    pal = [tuple(c) for name in ("tab20", "tab20b", "tab20c") for c in plt.get_cmap(name).colors]
    colors = {-1: (0.6, 0.6, 0.6, 0.5)}
    uniq = [int(u) for u in sorted(set(labels.tolist())) if u != -1]
    if not uniq:
        return colors
    if polylines is None or len(polylines) != len(labels):
        for i, u in enumerate(uniq):
            colors[u] = pal[(i * 7) % len(pal)]                  # scatter → neighbouring labels differ
        return colors

    from collections import defaultdict
    from scipy.spatial import cKDTree
    gi_of = {u: i for i, u in enumerate(uniq)}
    pts, owner = [], []
    for idx, lb in enumerate(labels):
        if int(lb) in gi_of:
            xy = np.asarray(polylines[idx])[:, :2]
            pts.append(xy)
            owner.append(np.full(len(xy), gi_of[int(lb)]))
    ptsA = np.concatenate(pts)
    ownerA = np.concatenate(owner)
    if len(ptsA) > 4000:                                          # cap for a cheap query_pairs
        sel = np.random.default_rng(0).choice(len(ptsA), 4000, replace=False)
        ptsA, ownerA = ptsA[sel], ownerA[sel]
    adj_px = (adj_um / voxel_um) if voxel_um else max(8.0, 0.05 * float(np.ptp(ptsA, 0).max()))
    adj = defaultdict(set)
    prs = cKDTree(ptsA).query_pairs(adj_px, output_type="ndarray")
    if len(prs):
        for a, b in zip(ownerA[prs[:, 0]], ownerA[prs[:, 1]]):
            if a != b:
                adj[int(a)].add(int(b))
                adj[int(b)].add(int(a))
    # Assign a UNIQUE palette colour per cluster (not minimised — graph-colouring minimises colours, which
    # re-uses ~6 colours for 50 clusters and makes distinct sheets look identical). Largest clusters pick
    # first; each takes the still-unused colour MOST UNLIKE its already-coloured spatial neighbours (RGB
    # distance), so adjacency drives contrast, not colour re-use. Only when >palette clusters exist (rare)
    # is re-use allowed, and then still pushed away from neighbours.
    pal_rgb = np.array([c[:3] for c in pal])
    sizes = np.bincount(ownerA, minlength=len(uniq))
    assign, used = {}, set()
    for g in sorted(range(len(uniq)), key=lambda g: -sizes[g]):
        avail = [k for k in range(len(pal)) if k not in used] or list(range(len(pal)))
        nbr = [assign[nb] for nb in adj[g] if nb in assign]
        if nbr:
            d = np.linalg.norm(pal_rgb[avail][:, None, :] - pal_rgb[nbr][None, :, :], axis=2).min(1)
            pick = avail[int(np.argmax(d))]
        else:
            pick = avail[0]
        assign[g] = pick
        used.add(pick)
    for g, u in enumerate(uniq):
        colors[u] = pal[assign[g]]
    return colors


def _label_rgb(lab, plt):
    """Map an integer label array → RGB (tab20 modulo; -1 = black background)."""
    table = np.array([plt.get_cmap("tab20")(i)[:3] for i in range(20)], np.float32)
    idx = np.where(lab >= 0, lab % 20, 0)
    rgb = table[idx]
    rgb[lab < 0] = 0.0
    return rgb


def _label_vol_figures(res, plt):
    """One figure proving the segmentation is 3-D: orthogonal cross-sections through DEPTH (a sheet
    is a connected band across z) and three XY depth slices sharing the SAME labels (a sheet keeps
    its colour as it weaves in/out of the depth window)."""
    lab = res["label_vol"]                      # (D, H, W)
    D, H, W = lab.shape
    zi = res.get("zi", D // 2)
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, :2])
    ax.imshow(_label_rgb(lab[:, H // 2, :], plt), aspect="auto", origin="upper")
    ax.set_title("X–Z cross-section (y=mid): sheets as bands THROUGH depth — this is the 3-D cut")
    ax.set_xlabel("x")
    ax.set_ylabel("depth z")

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(_label_rgb(lab[:, :, W // 2], plt), aspect="auto", origin="upper")
    ax.set_title("Y–Z (x=mid)")
    ax.set_xlabel("y")
    ax.set_ylabel("depth z")

    dz = max(1, D // 4)
    for j, z in enumerate([max(0, zi - dz), zi, min(D - 1, zi + dz)]):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(_label_rgb(lab[z], plt), origin="upper")
        ax.set_title(f"XY @ z={z}" + ("  (viewing depth)" if z == zi else ""))
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("3-D sheet labels (φ over the whole block) — same colour = same sheet across depth")
    fig.tight_layout()
    return fig


def build_figures(res):
    """Build (do not save) the diagnostic matplotlib figures from a :func:`compute_from_block`
    result. Returns a list of Figures — the script saves them, the UI shows them."""
    import matplotlib.pyplot as plt

    img, polylines, X = res["img"], res["polylines"], res["X"]
    labels, names, extra = res["labels"], res["names"], res["extra"]
    center = np.asarray(extra.get("umbilicus", np.zeros(2)))
    cond = extra.get("umbilicus_cond", np.inf)
    colors = cluster_colors(labels)
    n_sheets = len(set(labels)) - (1 if -1 in labels else 0)
    use_phi = res.get("used_phi", False)
    ai = names.index("abs_curvature")
    figs = []

    method = ("φ layer-potential" if use_phi else "umbilicus/curvature DBSCAN")
    ced = " + CED" if res.get("used_ced") else ""

    # --- CED before/after (see the effect of anisotropic diffusion) ---
    if res.get("used_ced") and "raw_slice" in res:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(res["raw_slice"], cmap="gray")
        axes[0].set_title("raw slice")
        axes[1].imshow(res["ced_slice"], cmap="gray")
        axes[1].set_title(f"CED-cleaned (sheets tightened, gaps widened){ced}")
        for a in axes:
            a.set_xticks([])
            a.set_yticks([])
        fig.tight_layout()
        figs.append(fig)

    # --- the 3-D segmentation: φ solved over the whole block, following sheets through depth ---
    if res.get("label_vol") is not None:
        figs.append(_label_vol_figures(res, plt))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap="gray", origin="upper")
    for pl, lb in zip(polylines, labels):
        ax.plot(pl[:, 0], pl[:, 1], color=colors[lb], lw=1.2)
    if not use_phi and cond < 50:
        ax.plot(center[0], center[1], "r+", ms=16, mew=3, label="umbilicus est")
        ax.legend()
    ax.set_title(f"streamlines by sheet — {method}{ced}  ({n_sheets} sheets)")
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    fig.tight_layout()
    figs.append(fig)

    if len(X):
        if use_phi:
            pi = names.index("mean_phi_mm")
            fig, ax = plt.subplots(figsize=(8, 5))
            for lb in sorted(set(labels)):
                m = labels == lb
                ax.scatter(X[m, pi], X[m, ai], s=18, color=colors[lb],
                           label=("noise" if lb == -1 else f"sheet {lb}"))
            ax.set_xlabel("mean φ (mm) — layer coordinate (sheets = evenly-spaced bands)")
            ax.set_ylabel("|curvature| (rad/mm)")
            ax.set_title("φ vs curvature — each band = one sheet")
            ax.legend(fontsize=6, ncol=3)
            fig.tight_layout()
            figs.append(fig)

            fig, ax = plt.subplots(figsize=(8, 4))
            order = np.argsort(X[:, pi])
            ax.scatter(np.arange(len(order)), X[order, pi],
                       c=[colors[labels[i]] for i in order], s=14)
            ax.set_xlabel("streamline (sorted by φ)")
            ax.set_ylabel("mean φ (mm)")
            ax.set_title("φ staircase — flat steps = sheets, risers = inter-sheet gaps")
            fig.tight_layout()
            figs.append(fig)
        else:
            npj = names.index("normal_proj_mm")
            ri = names.index("umbilicus_r_mm")
            fig, ax = plt.subplots(figsize=(7, 5))
            for lb in sorted(set(labels)):
                m = labels == lb
                ax.scatter(X[m, npj], X[m, ai], s=16, color=colors[lb],
                           label=("noise" if lb == -1 else f"sheet {lb}"))
            ax.set_xlabel("normal projection (mm)")
            ax.set_ylabel("|curvature| (rad/mm)")
            ax.set_title("curvature vs normal-projection")
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            figs.append(fig)

            fig, ax = plt.subplots(figsize=(7, 5))
            for lb in sorted(set(labels)):
                m = labels == lb
                ax.scatter(X[m, ri], X[m, ai], s=16, color=colors[lb])
            ax.set_xlabel("umbilicus radius r (mm)")
            ax.set_ylabel("|curvature| (rad/mm)")
            ax.set_title(f"curvature vs radius (cond={cond:.0f})")
            fig.tight_layout()
            figs.append(fig)
    return figs

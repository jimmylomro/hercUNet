"""Density clustering of the contrastive embedding into per-sheet labels.

Two entries operate on the trained 8-D embedding ``E`` (from
``embedding.paperlet.train_paperlet_embedding``):

  * :func:`labels_from_embedding` — plain HDBSCAN. Finds the dense per-sheet cores; a thin low-density
    bridge between two cores does not merge them (chaining-proof), and sparse/ambiguous points fall to
    noise (-1). Purity over coverage; no ``k``, no distance threshold.

  * :func:`probeom_labels` — the probabilistic, size-penalized Excess-of-Mass selection with
    embedding-space propagation. It corrects plain-eom's over-merge bias (one mega-cluster swallowing
    several sheets) and returns a per-point propagation distance the pipeline reuses as a confidence
    signal. This is the selection the live per-window pipeline uses.

Membership is decided by DENSITY in a Euclidean, ordinal embedding — never by a path-integrated
coordinate (which aliases sheets at bends).
"""

from __future__ import annotations

import numpy as np


def labels_from_embedding(z, min_cluster_size=4, min_samples=None, cluster_selection_method="eom",
                          cluster_selection_epsilon=0.0):
    """ROBUST density clustering (HDBSCAN) of the contrastive embedding — robust because embeddings are
    never perfect. Finds the DENSE per-sheet cores; a thin bridge between two cores is LOW-DENSITY so it
    does NOT merge them (chaining-proof, unlike connected-components / average linkage which chain
    through any bridge). Sparse/ambiguous points become NOISE (-1) → masked (purity over coverage). No
    ``k``, no distance threshold — just a minimum cluster size. Returns integer labels (-1 = noise).

    ``cluster_selection_method``: 'eom' (Excess of Mass, sklearn default) tends to pick a few LARGE clusters
    (merge-prone); 'leaf' selects the condensed-tree LEAVES → many small homogeneous clusters (split-prone,
    resists gluing two ordinality-adjacent sheets). ``min_samples`` (None → = min_cluster_size) is the
    density/conservativeness knob: higher → sparser border points become noise (does NOT set cluster size).
    ``cluster_selection_epsilon`` (embedding-distance units, 0 = off): merges clusters whose separation is
    below it — the lever for reconnecting one sheet split across a THIN low-density bridge (the slim parts
    between clusters), without chaining through everything the way linkage would."""
    n = len(z)
    if n == 0:
        return np.zeros(0, int)
    if n <= min_cluster_size:
        return np.zeros(n, int)
    from sklearn.cluster import HDBSCAN

    # copy=True explicitly (the future sklearn default) — silences the FutureWarning at root and avoids
    # HDBSCAN mutating the embedding in place.
    hdb = HDBSCAN(min_cluster_size=max(2, int(min_cluster_size)), min_samples=min_samples,
                  cluster_selection_method=cluster_selection_method,
                  cluster_selection_epsilon=float(cluster_selection_epsilon), copy=True)
    return hdb.fit_predict(z).astype(int)


def probeom_labels(z, min_cluster_size=250, min_samples=None, soft_max_frac=1.5, strength=3.0,
                   temperature=1.0, propagate=True, prop_k=5, random_state=0):
    """PROBABILISTIC (size-penalized) EOM selection + embedding-space propagation — the fix for HDBSCAN eom's
    OVER-MERGE bias (one mega-cluster swallowing 3+ sheets), replacing plain eom AND the skinnydip 2nd pass.

    eom over-merges because a child's stability BEFORE it separates from its siblings is credited to the parent,
    so `stability(parent) ≥ Σstability(children)` is structurally biased toward keeping merged. Instead of a hard
    `max_cluster_size` cliff, we put a SMOOTH size pressure on the recursive subtree selection:
        keep-whole  iff  sigmoid( norm_diff/temperature − penalty ) > 0.5   (deterministic → reproducible)
        norm_diff = (stability − Σchild_stability) / (|stability| + |Σchild|)          # scale-invariant, in [−1,1]
        penalty   = max(0, size/soft_max − 1) · strength                                # 0 at soft_max, =strength at 2×
    Normalising the stability difference is ESSENTIAL — raw stability scales with point count, so an un-normalised
    penalty never bites. Splits land ONLY at the condensed tree's existing branches (= the weakest density seams /
    longest MST edges) and RECURSE → N-way peel (beats skinnydip's single bimodal cut). The root is never a
    cluster; when a parent is kept, its descendants are DESELECTED.

    Cores land at the tree leaves (peeled inter-sheet mass → noise, like `leaf`), so we then PROPAGATE: each noise
    unit joins the cluster its ``prop_k`` nearest LABELLED cores agree on **in embedding space only** (a Voronoi
    fill along the E-boundaries → recovers coverage WITHOUT re-merging the cores). ``soft_max`` is
    ``soft_max_frac`` × the median non-largest eom cluster size (a "typical single sheet" in units).

    Returns ``(labels, prop_dist)``: integer labels (−1 = noise, only if propagate=False); ``prop_dist`` = per-unit
    mean embedding distance to the k voted cores (0 for core units, larger = ambiguous filler) — the LOW-CONFIDENCE
    signal that replaces the skinnydip seam downstream."""
    import hdbscan
    n = len(z)
    if n <= max(2, int(min_cluster_size)):
        return np.zeros(n, int), np.zeros(n, np.float32)
    cl = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size), min_samples=min_samples,
                         cluster_selection_method="eom").fit(np.asarray(z, np.float64))
    eom = cl.labels_
    uc = np.unique(eom[eom >= 0], return_counts=True)[1]
    med = float(np.median(np.sort(uc)[:-1])) if len(uc) > 1 else (float(np.median(uc)) if len(uc) else 50.0)
    soft_max = max(soft_max_frac * med, 1.0)

    raw = cl.condensed_tree_.to_numpy()                          # structured array: parent, child, lambda_val, child_size
    root = int(raw["parent"].min())
    crows = raw[raw["child_size"] > 1]
    births = {root: 0.0}
    for r in crows:
        births[int(r["child"])] = float(r["lambda_val"])
    cluster_ids = [root] + sorted(int(c) for c in np.unique(crows["child"]))
    idset = set(cluster_ids)
    stability = {cid: float(((raw[raw["parent"] == cid]["lambda_val"] - births[cid])
                             * raw[raw["parent"] == cid]["child_size"]).sum()) for cid in cluster_ids}
    children_map = {cid: [] for cid in cluster_ids}
    size = {}
    for r in crows:
        p, c = int(r["parent"]), int(r["child"])
        if p in idset:
            children_map[p].append(c)
        size[c] = int(r["child_size"])
    size[root] = n
    selected = set()

    def descendants(cid):
        out, stack = [], list(children_map.get(cid, []))
        while stack:
            m = stack.pop(); out.append(m); stack.extend(children_map.get(m, []))
        return out

    def select(cid, is_root=False):
        children = children_map.get(cid, [])
        if not children:
            selected.add(cid); return stability[cid]
        child_total = sum(select(c) for c in children)
        if is_root:                                          # root can never be one cluster → always descend
            return child_total
        own = stability[cid]
        norm_diff = (own - child_total) / (abs(own) + abs(child_total) + 1e-9)
        penalty = max(0.0, size[cid] / soft_max - 1.0) * strength
        p_keep = 1.0 / (1.0 + np.exp(-(norm_diff / max(temperature, 1e-9) - penalty)))
        keep = (p_keep > 0.5)                                # deterministic base label (cliff-free but reproducible)
        if keep:
            for d in descendants(cid):
                selected.discard(d)                          # prune descendants
            selected.add(cid); return own
        return child_total

    select(root, is_root=True)

    cluster_parent = {int(r["child"]): int(r["parent"]) for r in crows}
    labels = np.full(n, -1, int)
    lut = {cid: i for i, cid in enumerate(sorted(selected))}
    for r in raw[raw["child_size"] == 1]:
        pid, cur = int(r["child"]), int(r["parent"])
        while cur is not None:
            if cur in selected:
                labels[pid] = lut[cur]; break
            cur = cluster_parent.get(cur)

    dist = np.zeros(n, np.float32)
    if propagate:
        known = labels >= 0
        if known.any() and not known.all():
            from sklearn.neighbors import NearestNeighbors
            kk = int(min(prop_k, known.sum()))
            d, idx = NearestNeighbors(n_neighbors=kk).fit(z[known]).kneighbors(z[~known])
            klab = labels[known][idx]
            labels[~known] = np.array([np.bincount(row).argmax() for row in klab])
            dist[~known] = d.mean(1)
    return labels, dist


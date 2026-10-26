"""Per-window sheet-mesh extraction — turn one window's instance output into fitted sheet meshes, and write them.

The bridge from HercuViT's per-window prediction (``owner`` = argmax-query ownership, ``gate`` = surface mask) to the
geometry stored in an .asv. The fit is the fast KNN-free field solver (:func:`hercunet.meshes.field_fit.fit_sheets_field`
— per-sheet blurred-mask field + explicit gradient ascent), NOT the herculabels slab fit; it runs on the GPU next to
inference and returns ``{sheet_id: mesh}`` in window-local voxel coordinates (z,y,x).
"""
from __future__ import annotations

import numpy as np

from .field_fit import fit_sheets_field


def prune_sheet_masks(owner, gate, memb, *, mass_frac: float = 0.10, conf_thr: float = 0.70):
    """Drop spurious sheet masks from ``gate`` BEFORE mesh fitting, returning a new gate with the dropped sheets'
    voxels cleared. A sheet (an id in ``owner`` within ``gate``) is dropped **iff it is both small and unconfident**:

        ``self_mass < mass_frac * max(self_mass)``  AND  ``mean_own_conf < conf_thr``

    where ``self_mass`` is the sum of per-voxel own-membership (``memb``, the max softmax-over-queries probability)
    over the sheet's gated voxels, and ``mean_own_conf = self_mass / n_voxels``. This is the confidence-weighted
    fragment filter: a confident thin sheet (high ``mean_own_conf``) survives even when small, and any large sheet
    (high ``self_mass``) survives even when its confidence is modest — the two-sided guard against pruning real thin
    sheets that a bare size/voxel cut would eat. The absolute/relative voxel floor is applied separately by the fit's
    ``min_pts``; this only removes confidence-weighted fragments."""
    import torch
    owner_t = torch.as_tensor(owner)
    gate_t = torch.as_tensor(gate, dtype=torch.bool)
    g = gate_t.reshape(-1)
    if not bool(g.any()):
        return gate_t
    ow = owner_t.reshape(-1)[g]
    mm = torch.as_tensor(memb, dtype=torch.float32, device=ow.device).reshape(-1)[g]
    ids, inv = torch.unique(ow, return_inverse=True)
    sm = torch.zeros(ids.numel(), device=mm.device).index_add_(0, inv, mm)                 # self-mass per sheet
    cnt = torch.zeros(ids.numel(), device=mm.device).index_add_(0, inv, torch.ones_like(mm))
    conf = sm / cnt.clamp_min(1.0)                                                         # mean own-confidence
    drop_ids = ids[(sm < mass_frac * sm.max()) & (conf < conf_thr)]                        # small AND unconfident
    if drop_ids.numel() == 0:
        return gate_t
    return gate_t & ~(torch.isin(owner_t, drop_ids) & gate_t)


def merge_confused_sheets(owner, gate, mask_logits, *, thresh: float = 1.0 / 3):
    """Merge sheet masks the decoder **over-split** — one physical wrap the model spread across two queries, each
    claiming ~half the voxels. The signature is *symmetric* cross-confidence: averaged over sheet i's gated voxels,
    the mean softmax-over-queries probability given to query j, ``M[i,j]``, is large **and** ``M[j,i]`` is large too
    (as opposed to a sliver, which leaks one-sidedly and is handled by :func:`prune_sheet_masks`). For every mutual
    pair with ``M[i,j] > thresh`` and ``M[j,i] > thresh`` we relabel the smaller-self-mass sheet into the larger.

    ``thresh = 1/3`` makes the merge **provably pairwise**: a row of ``M`` is a probability distribution summing to
    ≤ 1 and its diagonal (own membership) is the row max (a voxel's own query is its argmax), so no sheet can give
    more than 1/3 to *two* different others — mutual ``> 1/3`` edges therefore form a matching (disjoint pairs), never
    a 3-way chain. Returns the relabelled ``owner`` (unchanged if nothing merges). ``mask_logits`` is the per-query
    mask logits ``[Q, D, H, W]`` (the same tensor whose argmax is ``owner``)."""
    import torch
    owner_t = torch.as_tensor(owner)
    gate_t = torch.as_tensor(gate, dtype=torch.bool)
    g = gate_t.reshape(-1)
    if not bool(g.any()):
        return owner_t
    ml = torch.as_tensor(mask_logits)
    Q = ml.shape[0]
    ow = owner_t.reshape(-1)[g]
    ids = torch.unique(ow)
    K = int(ids.numel())
    if K < 2:
        return owner_t
    prob = torch.softmax(ml.reshape(Q, -1)[:, g], 0)                         # [Q, Ng] softmax over queries
    inv = torch.searchsorted(ids, ow)                                        # [Ng] -> 0..K-1
    onehot = torch.zeros(ow.numel(), K, device=ow.device, dtype=prob.dtype)
    onehot[torch.arange(ow.numel(), device=ow.device), inv] = 1.0
    cnt = onehot.sum(0).float().clamp_min(1.0)                              # [K] voxels per sheet
    Msum = (onehot.t() @ prob[ids].t()).float()                            # [K row sheet, K col query] (float: bf16-safe)
    M = (Msum / cnt[:, None]).cpu().numpy()                                 # mean cross-confidence
    self_mass = Msum.diagonal().cpu().numpy()                              # Σ own-membership per sheet
    parent = list(range(K))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for i in range(K):
        for j in range(i + 1, K):
            if M[i, j] > thresh and M[j, i] > thresh:                       # symmetric confusion -> same wrap
                ri, rj = find(i), find(j)
                if ri != rj:                                               # attach smaller-self-mass root to larger
                    big = ri if self_mass[ri] >= self_mass[rj] else rj
                    parent[rj if big == ri else ri] = big
    comp: dict[int, list[int]] = {}
    for k in range(K):
        comp.setdefault(find(k), []).append(k)
    relabel = {}
    for root, members in comp.items():
        if len(members) > 1:
            tgt = max(members, key=lambda k: self_mass[k])
            for k in members:
                if k != tgt:
                    relabel[int(ids[k])] = int(ids[tgt])
    if not relabel:
        return owner_t
    owner2 = owner_t.clone()
    for src, dst in relabel.items():
        owner2[owner_t == src] = dst
    return owner2


def extract_window_sheets(owner, gate, *, voxel_um: float, sigma: float = 3.0, iters: int = 40, lr: float = 1.0,
                          lam: float = 0.3, step_um: float = 30.0, min_pts: int = 400, device: str = "cuda"):
    """Fit sheet meshes for one window from ``owner`` (D,H,W int sheet ids) and ``gate`` (D,H,W bool surface keep).
    Thin wrapper over the hercuvit field fit (see :func:`fit_sheets_field` for the parameters and the min-points /
    degenerate drops). Returns ``{sheet_id: mesh}`` (empty if no fittable sheet); verts are window-local voxel
    coords (z,y,x)."""
    return fit_sheets_field(owner, gate, voxel_um=voxel_um, sigma=sigma, iters=iters, lr=lr, lam=lam,
                            step_um=step_um, min_pts=min_pts, device=device)


class MeshSink:
    """Write per-window sheet meshes into an open :class:`~hercunet.meshes.asv.AsvWriter`, converting window-local
    voxel coords (z,y,x, at the working level) to global native-L0 XYZ. ``downsample`` is the working level's L0
    downsample factor (1.0 at level 0). One sink per worker shard; the engine calls :meth:`add_window` per window."""

    def __init__(self, writer, downsample: float = 1.0, name_prefix: str = ""):
        self.w = writer
        self.ds = float(downsample)
        self.pre = name_prefix
        self.n_windows = 0
        self.n_sheets = 0

    def add_window(self, origin_zyx, meshes) -> None:
        """``origin_zyx`` = the window's (z,y,x) start on the working-level grid; ``meshes`` = ``{sid: mesh}`` from
        :func:`extract_window_sheets` (verts window-local z,y,x). Each sheet is stored in global L0 XYZ."""
        oz, oy, ox = (float(v) for v in origin_zyx)
        off = np.array([oz, oy, ox], np.float32)
        wname = f"z{int(oz)}_y{int(oy)}_x{int(ox)}"
        self.n_windows += 1
        for sid, m in meshes.items():
            g_zyx = (m["verts"] + off) * self.ds                                     # window-local -> global L0 (z,y,x)
            self.w.add_sheet(m["foot"], g_zyx[:, ::-1], name=f"{self.pre}{wname}_s{int(sid)}", window=wname)
            self.n_sheets += 1

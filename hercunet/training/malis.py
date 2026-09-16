"""Constrained MALIS loss for affinity-based sheet separation (winding_field_segmentation_v3 §4.1).

MALIS (Maximin Affinity Learning; Turaga et al. 2009) trains per-voxel **affinities** by their effect on the
**segmentation topology**: the predicted connection strength between two voxels is the *maximin* edge — the
weakest affinity on the strongest path between them. We use the **constrained** two-pass variant (Funke et al.
2018): a POSITIVE pass that pushes the maximin edge UP for voxel pairs in the SAME ground-truth sheet
(growth/completion, long-range), and a NEGATIVE pass that pushes it DOWN for pairs in DIFFERENT sheets
(cut the merge). Ground truth is our per-sheet instance label ``owner_full`` (mesh Voronoi id per voxel).

No external dependency (no `malis`/`affogato`/numba): the maximin edges are found with Kruskal's algorithm
(process edges in descending affinity, union-find), which is exact. Constraining makes each component
single-segment, so we only need to track per-component **labeled-node counts** — the number of pairs a merge
edge is maximin for is ``sizeA * sizeB``. Runs on CPU/numpy inside a ``torch.autograd.Function``; the whole
volume or a random sub-crop can be used per step (see ``crop``) to bound the O(E log E) sort + O(E α) union-find.

Edge convention: for offset ``δ`` (channel ``c``) and voxel ``u`` (in-bounds ``u+δ``), affinity
``aff[c, u]`` is the predicted "same-sheet-ness" of the ordered pair ``(u, u+δ)``. GT affinity is 1 iff
``seg[u] == seg[u+δ]`` and both are labeled (``>0``); ``seg <= 0`` is background/ignore (excluded from pairs).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


try:                                                     # numba JITs the maximin/union-find hot loop (~50-100x).
    from numba import njit                                # needs numpy <= 2.4 (pod is 2.1 -> works; local 2.5 -> no)
    _HAVE_NUMBA = True
except Exception:                                        # graceful fallback: run the same code in pure python
    _HAVE_NUMBA = False
    def njit(*a, **k):
        return a[0] if a and callable(a[0]) else (lambda f: f)


@njit(cache=True)
def _find(parent, x):
    r = x
    while parent[r] != r:
        r = parent[r]
    while parent[x] != r:                                # path compression
        nx = parent[x]; parent[x] = r; x = nx
    return r


@njit(cache=True)
def _uf_pass(order, u, v, aff, labeled, target, N, pre_u, pre_v):
    """Constrained-MALIS union-find pass (Kruskal, descending affinity). ``pre_*`` are pre-merged first
    (the negative pass fuses each GT segment). Returns (loss, per-edge grad). Numba-jitted; identical logic to
    the pure-python version this replaced (unit-tested: perfect->0, merge->push-down, split->push-up)."""
    parent = np.arange(N)
    rank = np.zeros(N, np.uint8)
    size = np.zeros(N, np.int64)
    for i in range(N):
        size[i] = labeled[i]

    for k in range(pre_u.shape[0]):                      # pre-merge within-segment (negative pass only)
        ra, rb = _find(parent, pre_u[k]), _find(parent, pre_v[k])
        if ra != rb:
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            size[ra] += size[rb]

    grad = np.zeros(aff.shape[0])
    loss = 0.0
    npairs_tot = 0
    for oi in range(order.shape[0]):
        e = order[oi]
        ra, rb = _find(parent, u[e]), _find(parent, v[e])
        if ra == rb:
            continue
        sa, sb = size[ra], size[rb]
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        size[ra] = sa + sb
        npairs = sa * sb                                 # this edge is maximin for sizeA*sizeB labeled pairs
        if npairs > 0:
            d = aff[e] - target
            loss += npairs * d * d
            grad[e] += 2.0 * npairs * d
            npairs_tot += npairs
    if npairs_tot > 0:
        loss /= npairs_tot
        for i in range(grad.shape[0]):
            grad[i] /= npairs_tot
    return loss, grad


def _edges(shape, offsets):
    """Return (u, v, c) flat-index edge arrays for every in-bounds (voxel, offset). c = offset channel."""
    D, H, W = shape
    zz, yy, xx = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij")
    us, vs, cs = [], [], []
    for c, (dz, dy, dx) in enumerate(offsets):
        z0, z1 = max(0, -dz), D - max(0, dz)
        y0, y1 = max(0, -dy), H - max(0, dy)
        x0, x1 = max(0, -dx), W - max(0, dx)
        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            continue
        zc, yc, xc = zz[z0:z1, y0:y1, x0:x1], yy[z0:z1, y0:y1, x0:x1], xx[z0:z1, y0:y1, x0:x1]
        u = (zc * H + yc) * W + xc
        v = ((zc + dz) * H + (yc + dy)) * W + (xc + dx)
        cc = np.full(u.size, c, np.int64)
        us.append(u.ravel()); vs.append(v.ravel()); cs.append(cc)
    return np.concatenate(us), np.concatenate(vs), np.concatenate(cs)


def _malis_pass(aff_c, seg_flat, u, v, c, *, positive):
    """One constrained MALIS pass. Returns (loss, grad_per_edge) where grad is dL/d(aff) for THIS edge set.

    aff_c: (E,) predicted affinity for each edge (already gathered by channel).  seg_flat: (N,) int labels.
    positive: True = attractive pass (same-segment pairs, target 1); False = repulsive (diff-segment, target 0).
    Loss is the pair-count-weighted squared error at each maximin edge (mean over total constrained pairs).
    """
    su, sv = seg_flat[u], seg_flat[v]
    same = (su == sv) & (su > 0) & (sv > 0)              # within-segment, both labeled
    labeled = (seg_flat > 0).astype(np.int64)
    N = seg_flat.shape[0]
    if positive:                                         # only within-segment edges retain predicted weight
        keep = same; target = 1.0
        pre_u = np.empty(0, np.int64); pre_v = np.empty(0, np.int64)
    else:                                                # NEGATIVE: pre-fuse each GT segment, eval between-segment
        pre_u = u[same].astype(np.int64); pre_v = v[same].astype(np.int64)
        keep = (~same) & (su > 0) & (sv > 0); target = 0.0
    ek = np.nonzero(keep)[0]
    if ek.size == 0:
        return 0.0, np.zeros_like(aff_c)
    order = ek[np.argsort(-aff_c[ek], kind="stable")].astype(np.int64)   # descending predicted affinity
    loss, grad = _uf_pass(order, u.astype(np.int64), v.astype(np.int64), aff_c.astype(np.float64),
                          labeled, float(target), N, pre_u, pre_v)
    return float(loss), grad.astype(aff_c.dtype)


_STATS = []                                                           # per-item (pos_loss, neg_loss) for logging


class _ConstrainedMalis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, aff, seg, offsets):
        """aff: (C, D, H, W) float in [0,1] on any device. seg: (D,H,W) int (owner_full). offsets: list."""
        dev, dt = aff.device, aff.dtype
        a = aff.detach().float().cpu().numpy()
        s = seg.detach().cpu().numpy().astype(np.int64).ravel()
        shape = a.shape[1:]
        u, v, c = _edges(shape, offsets)
        aff_c = a.reshape(a.shape[0], -1)[c, u]            # gather predicted affinity per edge
        lp, gp = _malis_pass(aff_c, s, u, v, c, positive=True)
        ln, gn = _malis_pass(aff_c, s, u, v, c, positive=False)
        g_edge = 0.5 * (gp + gn)                            # average the two passes
        grad = np.zeros_like(a).reshape(a.shape[0], -1)
        np.add.at(grad, (c, u), g_edge)                    # scatter edge grads back to (channel, voxel)
        ctx.grad = torch.from_numpy(grad.reshape(a.shape)).to(dev, dt)
        _STATS.append((float(lp), float(ln)))              # lp = attractive(grow), ln = repulsive(separate)
        return torch.tensor(0.5 * (lp + ln), device=dev, dtype=dt)

    @staticmethod
    def backward(ctx, grad_out):
        return ctx.grad * grad_out, None, None


class ConstrainedMalisLoss(nn.Module):
    """Constrained MALIS over per-sheet instance labels. See module docstring / v3 spec §4.1.

    offsets: list of (dz,dy,dx). Include the 3 nearest (guarantee within-segment connectivity) plus mid-range.
    crop:    if set, run MALIS on a random cube of this edge size per item (bounds the O(E log E) cost); None =
             whole volume. weight ramp is applied by the caller (pass-index dependent, v3 §5.1).
    """

    def __init__(self, offsets=None, crop=None):
        super().__init__()
        self.offsets = offsets or [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                                   (3, 0, 0), (0, 3, 0), (0, 0, 3),
                                   (9, 0, 0), (0, 9, 0), (0, 0, 9)]
        self.crop = crop

    def _crop(self, aff, seg, rng):
        if self.crop is None:
            return aff, seg
        C, D, H, W = aff.shape
        cz, cy, cx = min(self.crop, D), min(self.crop, H), min(self.crop, W)
        z = int(rng.integers(0, D - cz + 1)); y = int(rng.integers(0, H - cy + 1)); x = int(rng.integers(0, W - cx + 1))
        return aff[:, z:z+cz, y:y+cy, x:x+cx], seg[z:z+cz, y:y+cy, x:x+cx]

    def forward(self, aff, seg, rng=None):
        """aff: (B,C,D,H,W) in [0,1]; seg: (B,D,H,W) int owner_full. Returns scalar mean loss over the batch.
        After the call, ``self.last_grow`` / ``self.last_sep`` hold the mean attractive/repulsive pass losses
        (for per-term logging)."""
        rng = rng or np.random.default_rng()
        _STATS.clear()
        losses = []
        for b in range(aff.shape[0]):
            a, s = self._crop(aff[b], seg[b], rng)
            losses.append(_ConstrainedMalis.apply(a, s, self.offsets))
        self.last_grow = float(np.mean([s[0] for s in _STATS])) if _STATS else 0.0
        self.last_sep = float(np.mean([s[1] for s in _STATS])) if _STATS else 0.0
        return torch.stack(losses).mean()


def affinity_target(seg, offsets):
    """GT affinities (for a plain per-edge BCE auxiliary, and for tests). seg: (D,H,W) int owner_full.
    Returns (aff01, mask): aff01[c]=1 iff same labeled sheet across offset c; mask[c]=0 where either end is
    ignore (seg<=0). Shapes (C,D,H,W)."""
    seg = np.asarray(seg)
    D, H, W = seg.shape
    C = len(offsets)
    aff = np.zeros((C, D, H, W), np.float32)
    mask = np.zeros((C, D, H, W), np.float32)
    for c, (dz, dy, dx) in enumerate(offsets):
        z0, z1 = max(0, -dz), D - max(0, dz)
        y0, y1 = max(0, -dy), H - max(0, dy)
        x0, x1 = max(0, -dx), W - max(0, dx)
        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            continue
        a = seg[z0:z1, y0:y1, x0:x1]
        b = seg[z0+dz:z1+dz, y0+dy:y1+dy, x0+dx:x1+dx]
        both = (a > 0) & (b > 0)
        aff[c, z0:z1, y0:y1, x0:x1] = ((a == b) & both).astype(np.float32)
        mask[c, z0:z1, y0:y1, x0:x1] = both.astype(np.float32)
    return aff, mask

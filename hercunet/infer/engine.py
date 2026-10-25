"""Iterative surface refinement — double-buffered Jacobi passes with a half-window phase shift + Gaussian blend.

Each pass reads the PREVIOUS pass's whole surface-probability volume as the ``prev`` input channel and writes a
fresh one; pass 0 feeds ``prev = 0`` (bootstrap). Information propagates ~one window per pass (linear), which is
all the local structural refinement needs.

Two seam-healing modes:

* **Blend (default, ``overlap > 0``)** — each pass runs OVERLAPPING windows at stride ``S = P·(1-overlap)`` and
  merges them with the villa Gaussian importance blend in LOGIT space (same as single-pass full-volume
  inference), so there are NO grid artefacts. Between passes the grid shifts by ``S/2`` so pass p+1's window
  CENTRES land on pass p's overlap-centres. Output is tiled into fixed, chunk-aligned, disjoint tiles (a halo per
  tile), so tiles never blend across each other and multi-instance stays race-free. **HercUNet v0 uses
  ``overlap = 0.25``.**
* **Disjoint (``overlap = 0``)** — NON-overlapping windows at stride P written raw; the grid shifts by P/2 on odd
  passes so one pass's seams land at the CENTRE of the next pass's windows. Faster, but leaves faint grid seams —
  kept for reference / ablation.

Single instance by default (sequential blocks/tiles per pass). ``multi=True`` fans a pass across any number of
workers via the shared :mod:`hercunet.infer.claim_queue` (a faster worker steals more), separated by a per-pass
barrier + leader-finalize — used both for one box's per-GPU workers and for workers across pods sharing a network
volume. ``prev`` is fed raw [0,1] (NoNormalization); CT uses the fingerprint CTNormalization.
"""
from __future__ import annotations

import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import nnunet_infer as NI
from . import claim_queue as CQ


def _hms(seconds):
    """Compact human duration: ``45s``, ``12m03s``, ``1h04m`` (``??`` for inf/NaN)."""
    if seconds is None or seconds != seconds or seconds == float("inf"):
        return "??"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ----------------------------------------------------------------------------- multi-instance coordination
# The generic elastic claim-queue (atomic .claim / .done, barrier, orphan reclaim) lives in ``claim_queue`` and is
# shared with full-volume inference — nothing is copied here. Per-pass, each nb^3 prefetch BLOCK is one claim item
# (blocks are output-disjoint so parallel writes never race); passes are separated by a barrier + leader-finalize
# (a block of pass p+1 reads pass p output ONE window over, via the ½-shift). ``donedir`` must be on storage
# shared by all workers (a network volume across pods, or the local FS for multi-GPU on one box).
def _n_blocks(shape, P, offset, region, nb, stride=None):
    """Number of prefetch blocks a pass produces (for the barrier) — via ``pass_blocks`` so it can never drift
    from what ``run_pass`` actually claims."""
    return len(pass_blocks(shape, P, offset, region, nb, stride=stride)[0])


# ----------------------------------------------------------------------------- tiling (pure, testable)
def axis_tiles(dim, P, offset):
    """Disjoint owned segments covering [0,dim) for one axis, each with a P-wide window origin ``ws`` s.t.
    [ws, ws+P) ⊇ [o0,o1) and 0 <= ws <= max(0,dim-P). Boundaries at {0, offset, offset+P, ...} ∪ {dim} → every
    boundary is a multiple of P/2 when offset ∈ {0, P/2} (chunk-aligned for chunk=P/2)."""
    bs = {0, dim}
    k = 0
    while offset + k * P < dim:
        b = offset + k * P
        if 0 < b < dim:
            bs.add(b)
        k += 1
    bs = sorted(bs)
    out = []
    top = max(0, dim - P)
    for o0, o1 in zip(bs[:-1], bs[1:]):
        ws = min(o0, top)
        out.append((int(o0), int(o1), int(ws)))
    return out


def pass_tiles(shape, P, offset, region=None):
    """3-D disjoint tiles for one pass. Returns list of dict(owned=(z0,z1,y0,y1,x0,x1), win=(wz,wy,wx)).
    ``offset`` applies to all three axes (octant "+"-seam). ``region``=(z0,z1,y0,y1,x0,x1) restricts owned tiles
    (windows still clamp within the full volume)."""
    Z, Y, X = (int(s) for s in shape)
    tz, ty, tx = axis_tiles(Z, P, offset), axis_tiles(Y, P, offset), axis_tiles(X, P, offset)
    rz0, rz1, ry0, ry1, rx0, rx1 = region if region else (0, Z, 0, Y, 0, X)
    tiles = []
    for (oz0, oz1, wz) in tz:
        if oz1 <= rz0 or oz0 >= rz1:
            continue
        for (oy0, oy1, wy) in ty:
            if oy1 <= ry0 or oy0 >= ry1:
                continue
            for (ox0, ox1, wx) in tx:
                if ox1 <= rx0 or ox0 >= rx1:
                    continue
                tiles.append(dict(owned=(oz0, oz1, oy0, oy1, ox0, ox1), win=(wz, wy, wx)))
    return tiles


# ----------------------------------------------------------------------------- one pass (block + batch + prefetch)
def _window_grid(dim, P, offset, r0, r1, stride=None):
    """Window START positions on the stride-``S`` grid ``{offset + k*S}`` (``S`` defaults to the window size ``P`` —
    disjoint) whose [s, s+P) intersects [r0, r1), clamped to [0, dim-P] so every read is a full P^3 (no padding).
    ``stride < P`` gives OVERLAPPING windows, each still processed exactly once (unlike the blend tile-halo).
    Deduped + sorted."""
    import math
    S = int(stride) if stride else P
    if dim <= P:
        return [0]
    kmin = math.floor((r0 - offset - P) / S) + 1
    kmax = math.ceil((r1 - offset) / S) - 1
    return sorted({int(min(max(offset + k * S, 0), dim - P)) for k in range(kmin, kmax + 1)})


def pass_blocks(shape, P, offset, region, nb, level=0, stride=None):
    """The deterministic list of prefetch blocks for a pass (same on every worker → shared block ids). Each block
    = up to ``nb`` window-starts per axis; ``req`` is its CT read box (``level``-prefixed for ``read_window``), its
    ``wins`` the stride-P window origins. ``shape``/``region`` are in ``level``'s voxel grid."""
    Zf, Yf, Xf = (int(s) for s in shape)
    rz0, rz1, ry0, ry1, rx0, rx1 = region
    gz = _window_grid(Zf, P, offset, rz0, rz1, stride=stride)
    gy = _window_grid(Yf, P, offset, ry0, ry1, stride=stride)
    gx = _window_grid(Xf, P, offset, rx0, rx1, stride=stride)
    grp = lambda g: [g[i:i + nb] for i in range(0, len(g), nb)]
    blocks = []
    for zg in grp(gz):
        for yg in grp(gy):
            for xg in grp(gx):
                blocks.append(dict(
                    req=(level, zg[0], zg[-1] + P, yg[0], yg[-1] + P, xg[0], xg[-1] + P),
                    origin=(zg[0], yg[0], xg[0]),
                    wins=[(z, y, x) for z in zg for y in yg for x in xg]))
    return blocks, (len(gz), len(gy), len(gx))


# ----------------------------------------------------------------------------- OVERLAP mode (Gaussian blend)
# Each pass runs OVERLAPPING windows at stride S and merges them with the villa Gaussian importance blend
# (identical to the single-pass detector), instead of disjoint stride-P windows written raw. Between passes the
# grid shifts by S/2 so pass p+1's window CENTRES land on pass p's OVERLAP-centres (generalising the disjoint
# ½-shift, which is S/2 when S=P). The OUTPUT is tiled into fixed, chunk-aligned, disjoint tiles; each tile runs
# every window covering it (a halo) and accumulates locally, so tiles never blend across each other → multi-
# instance stays race-free.
def make_gaussian(P, sigma_scale=8.0):
    """Gaussian importance map, IDENTICAL to villa's ``generate_gaussian_map``: delta at the patch centre,
    gaussian_filter σ=P/8, normalise by max, clip ≥ 0. Peak at centre, ~0 at faces."""
    from scipy.ndimage import gaussian_filter
    tmp = np.zeros((P, P, P), np.float32)
    tmp[P // 2, P // 2, P // 2] = 1.0
    g = gaussian_filter(tmp, P / sigma_scale, 0, mode="constant", cval=0.0)
    g = g / g.max()
    g[g < 0] = 0.0
    return g.astype(np.float32)


def _overlap_grid(dim, P, S, offset):
    """Overlapping window START positions on the stride-S grid ``{offset + k*S}``, clamped to [0, dim-P] (so every
    read is a full P^3), deduped + sorted. Always includes 0 and dim-P (via clamp) so the whole axis is covered."""
    if dim <= P:
        return [0]
    import math
    kmin = math.floor((0 - offset) / S)
    kmax = math.ceil((dim - P - offset) / S)
    return sorted({int(min(max(offset + k * S, 0), dim - P)) for k in range(kmin, kmax + 1)})


def pass_blend_tiles(shape, P, S, offset, region, T, level=0):
    """Fixed, disjoint, chunk-aligned OUTPUT tiles (T per axis) each carrying the overlapping window starts that
    cover it (from the stride-S grid at ``offset``) + the CT read box (``level``-prefixed for ``read_window``;
    halo = min..max start + P). The output tiling is FIXED per volume; only the window grid moves with ``offset``.
    ``shape``/``region`` are in ``level``'s voxel grid. ``region`` restricts computed tiles."""
    Z, Y, X = (int(s) for s in shape)
    rz0, rz1, ry0, ry1, rx0, rx1 = region if region else (0, Z, 0, Y, 0, X)
    gz, gy, gx = _overlap_grid(Z, P, S, offset), _overlap_grid(Y, P, S, offset), _overlap_grid(X, P, S, offset)

    def cover(g, t0, t1):
        return [s for s in g if s < t1 and s + P > t0]

    tiles = []
    for tz in range(0, Z, T):
        tz1 = min(tz + T, Z)
        if tz1 <= rz0 or tz >= rz1:
            continue
        sz = cover(gz, tz, tz1)
        for ty in range(0, Y, T):
            ty1 = min(ty + T, Y)
            if ty1 <= ry0 or ty >= ry1:
                continue
            sy = cover(gy, ty, ty1)
            for tx in range(0, X, T):
                tx1 = min(tx + T, X)
                if tx1 <= rx0 or tx >= rx1:
                    continue
                sx = cover(gx, tx, tx1)
                if not (sz and sy and sx):
                    continue
                tiles.append(dict(owned=(tz, tz1, ty, ty1, tx, tx1),
                                  origin=(min(sz), min(sy), min(sx)),
                                  req=(level, min(sz), max(sz) + P, min(sy), max(sy) + P, min(sx), max(sx) + P),
                                  starts=(sz, sy, sx)))
    return tiles


# ----------------------------------------------------------------------------- forward + test-time augmentation
# TTA is per WINDOW: each window's batch is augmented, forwarded, un-augmented, and its logits averaged before the
# result is written (disjoint) or fed to the Gaussian blend. Two families, matching villa's tta.py — MIRRORING
# (torch.flip over z/y/x, up to 8 3-D variants) and ROTATION (torch.transpose axis-swaps, 2 extra variants) — and
# they combine (union of the two variant sets, deduping identity). The 8-ch orientation channels are recomputed
# in-forward from the CT (:func:`_seg_logits`), so under any flip/transpose the orientation is derived from the
# already-transformed CT and stays consistent — no manual channel handling.
def _resolve_tta(spec):
    """Parse ``--tta`` into a list of ``(flip_dims, transpose_pair)`` VARIANTS, identity first. Comma-separated
    tokens: ``none``/``off``/``""`` → off (identity only, the default); ``all`` → every mirror (z,y,x) + rotations;
    ``mirror`` → the 3 axis mirrors; ``rotate``/``rotation`` → the transpose variants; ``z``/``y``/``x`` → mirror
    over that axis. Tokens combine (e.g. ``mirror,rotate`` == ``all``; ``z,rotate``). Union semantics — villa runs
    mirroring and rotation as separate modes; we merge their variant SETS (not their product)."""
    dim = {"z": -3, "y": -2, "x": -1}
    toks = [t.strip() for t in str(spec).strip().lower().split(",") if t.strip()]
    if not toks or set(toks) <= {"none", "no", "off"}:
        return [((), None)]
    axes, rotate = [], False
    for t in toks:
        if t == "all":
            axes, rotate = ["z", "y", "x"], True
        elif t in ("mirror", "mirrors"):
            axes = [a for a in ("z", "y", "x")]
        elif t in ("rotate", "rotation", "rot"):
            rotate = True
        elif t in dim:
            if t not in axes:
                axes.append(t)
        else:
            raise SystemExit(f"hercunet infer: --tta got token {t!r}; expected a comma list of "
                             f"none|all|mirror|rotate or axes z,y,x (e.g. 'mirror,rotate' or 'z,y').")
    from itertools import combinations
    ds = [dim[a] for a in axes]
    variants = [((), None)]                                          # identity always first
    for r in range(1, len(ds) + 1):                                  # non-empty mirror combos (villa 'mirroring')
        for c in combinations(ds, r):
            variants.append((tuple(c), None))
    if rotate:                                                       # villa 'rotation' = 2 transpose swaps
        variants.append(((), (-3, -1)))
        variants.append(((), (-3, -2)))
    return variants


def _tta_surface(model, xb_base, variants):
    """Surface logits ``[B,2,P,P,P]`` with optional TTA, via ``model.forward``. Averages the un-augmented logits over
    each ``(flips, transpose)`` in ``variants`` (villa-style — average raw LOGITS, no softmax first). A single
    ``[((), None)]`` means TTA off (one forward). Each variant is applied (flip then transpose) to the base input and
    its inverse (transpose then flip — both self-inverse) to the output; the model recomputes any input-derived
    feature (e.g. structure-tensor orientation) from the transformed CT, so TTA stays correct."""
    import torch
    if len(variants) == 1:                                           # fast path: TTA off
        return model.forward(xb_base)[0]
    acc = None
    for flips, tp in variants:
        xa = xb_base
        if flips:
            xa = torch.flip(xa, flips)
        if tp:
            xa = torch.transpose(xa, *tp).contiguous()
        lg = model.forward(xa)[0]
        if tp:
            lg = torch.transpose(lg, *tp)                            # undo transpose (self-inverse)
        if flips:
            lg = torch.flip(lg, flips)                               # undo flip (self-inverse)
        acc = lg if acc is None else acc.add_(lg)
    return acc.div_(len(variants))


def run_pass(model, vol, prev_arr, cur_arr, region, offset, P, device,
             air=25, batch=4, nb=4, log_every=40, donedir=None, reclaim=False, claim_chunk=6,
             readahead=2, prefetch_workers=3, tta_variants=None, pass_idx=0, passes=1, level=0,
             mesh_sink=None, voxel_um=None, out_shape=None, win_stride=None):
    """One disjoint (raw-write) Jacobi pass — BLOCK-batched + prefetched (fp16). Reads CT in big blocks (``nb``*P
    per axis) through the prefetched :func:`iter_windows` (S3 I/O overlaps GPU), reads the matching ``prev`` block
    from the ``prev_arr`` zarr once per block (None on pass 0 → prev=0), runs the P^3 windows in each block in
    BATCHES of ``batch`` under fp16 autocast, and writes each window's surface-probability (uint8) into ``cur_arr``
    at its disjoint stride-P position. Air windows (CT max < ``air``) are skipped.

    ``donedir`` set → MULTI-INSTANCE: claim blocks from the shared ``claim_queue`` in chunks of ``claim_chunk``
    (dynamic work-stealing across workers), flushing + marking each block ``.done`` before moving on. Blocks are
    output-disjoint so parallel workers never race. This worker returns when it can claim no more blocks; the
    caller runs the barrier + leader-finalize. ``donedir`` None → single instance (all blocks, in order)."""
    import torch
    from ..data import iter_windows
    if not tta_variants:
        tta_variants = [((), None)]                                 # no TTA (single identity forward)
    blocks, (ngz, ngy, ngx) = pass_blocks(cur_arr.shape if cur_arr is not None else out_shape,
                                          P, offset, region, nb, level=level, stride=win_stride)
    total_win = sum(len(b["wins"]) for b in blocks)
    print(f"[pass] {len(blocks)} blocks / {total_win} windows (grid {ngz}x{ngy}x{ngx})"
          f"{' [claim-queue]' if donedir else ''}", flush=True)

    t0 = time.time(); done = [0]
    buf, meta = [], []

    def flush():
        if not buf:
            return
        xb = torch.from_numpy(np.stack(buf)).to(device, non_blocking=True)     # [B,Cbase,P,P,P]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            if mesh_sink is not None:                                         # one run_instance forward: surface + meshes
                logits, meshes = model.forward_meshes(xb, voxel_um)
            else:
                logits = _tta_surface(model, xb, tta_variants)                # [B,2,P,P,P] (TTA-averaged if on)
                meshes = None
            prob = torch.softmax(logits, 1)[:, 1]                             # [B,P,P,P]
        if cur_arr is not None:                                               # surface write (skipped in mesh-only runs)
            u8 = prob.mul_(255.0).round_().clamp_(0, 255).to(torch.uint8).cpu().numpy()
            for i, (z, y, x) in enumerate(meta):
                cur_arr[z:z + P, y:y + P, x:x + P] = u8[i]
        if mesh_sink is not None:                                             # per-window meshes -> .asv shard
            for i, (z, y, x) in enumerate(meta):
                mesh_sink.add_window((z, y, x), meshes[i])
        buf.clear(); meta.clear()

    def process_block(ct_blk, block):
        """Append this block's non-air windows to the batch buffer (flushing at ``batch``). Reads its prev slab."""
        bz0, by0, bx0 = block["origin"]
        pv_blk = None
        if prev_arr is not None:
            pv_blk = np.asarray(prev_arr[bz0:bz0 + ct_blk.shape[0], by0:by0 + ct_blk.shape[1],
                                         bx0:bx0 + ct_blk.shape[2]]).astype(np.float32) / 255.0
        for (z, y, x) in block["wins"]:
            sz, sy, sx = z - bz0, y - by0, x - bx0
            sub = ct_blk[sz:sz + P, sy:sy + P, sx:sx + P]
            if sub.shape != (P, P, P) or int(sub.max()) < air:               # off-block or all-air → skip
                continue
            ct_n = model.norm_ct(sub)
            pv = None if pv_blk is None else pv_blk[sz:sz + P, sy:sy + P, sx:sx + P]
            buf.append(model.base_input(ct_n, pv).astype(np.float32)); meta.append((z, y, x))
            done[0] += 1
            if len(buf) >= batch:
                flush()

    def rate():
        dt = time.time() - t0
        return f"{done[0]} win · {_hms(dt)} · {done[0]/max(dt,1e-9):.1f} win/s"

    if donedir is None:                                                       # ---- single instance: all blocks
        reqs = [b["req"] for b in blocks]
        for bi, (req, blk, org) in enumerate(iter_windows(vol, reqs, readahead=readahead, workers=prefetch_workers)):
            process_block(np.asarray(blk), blocks[bi])
            if log_every and bi and bi % log_every == 0:
                print(f"[pass] P{pass_idx}/{passes - 1} block {bi}/{len(blocks)} ({rate()})", flush=True)
        flush()
    else:                                                                     # ---- multi instance: claim-queue
        ntot = len(blocks); nproc = 0
        g0, tg0 = CQ.count_done(donedir), time.time()                         # baseline for the AGGREGATE rate/ETA
        for chunk in CQ.claim_chunks(donedir, range(ntot), chunk=claim_chunk, reclaim=reclaim):
            reqs = [blocks[g]["req"] for g in chunk]
            for (req, blk, org), gid in zip(iter_windows(vol, reqs, readahead=readahead, workers=prefetch_workers), chunk):
                process_block(np.asarray(blk), blocks[gid])
                flush()                                                       # persist the block BEFORE .done
                CQ.mark_done(donedir, gid)
                nproc += 1
            gd = CQ.count_done(donedir)                                        # GLOBAL blocks done (all workers)
            grate = (gd - g0) / max(time.time() - tg0, 1e-9)                   # aggregate blocks/s (this worker's view)
            eta = (ntot - gd) / grate if grate > 0 else float("inf")            # seconds
            full = ntot / grate if grate > 0 else float("inf")                  # whole-pass seconds at this rate
            run_eta = eta + max(0, passes - 1 - pass_idx) * full                # + the passes still to come
            pass_pct = 100 * gd // ntot
            run_pct = int(100 * (pass_idx + gd / ntot) / max(passes, 1))        # overall, across all passes
            print(f"[pass] P{pass_idx}/{passes - 1} | this worker: {nproc} blocks · {rate()} "
                  f"| all workers: {gd}/{ntot} blocks · {grate * 60:.1f} blk/min "
                  f"| pass ETA {_hms(eta)} ({pass_pct}%) · run ETA {_hms(run_eta)} ({run_pct}%)", flush=True)
    flush()
    print(f"[pass] DONE (this worker) {rate()}", flush=True)


def run_pass_blend(model, vol, prev_arr, cur_arr, region, offset, P, S, T, device,
                   air=25, batch=4, log_every=20, donedir=None, reclaim=False, claim_chunk=6,
                   readahead=2, prefetch_workers=3, tta_variants=None, aux_arr=None, pass_idx=0, passes=1, level=0,
                   mesh_sink=None, voxel_um=None, out_shape=None):
    """One OVERLAP-mode pass: like :func:`run_pass` (prefetched blocks, claim-queue, fp16) but each claim item is a
    disjoint OUTPUT TILE that runs every stride-S window covering it (halo) and merges them with the villa Gaussian
    blend in LOGIT space (accumulate ``(l1-l0)·g`` + ``g``, normalise, sigmoid), then writes only its owned region —
    so tiles never blend across each other and multi-instance stays race-free (same trick as full-volume infer).

    ``aux_arr`` set (a 4-D ``(channels,Z,Y,X)`` uint8 zarr, from ``model.aux``) → ALSO emit the model's aux output
    (e.g. affinity): its channels are Gaussian-blended in PROBABILITY space with the SAME weights as the surface,
    from the TTA-free identity ``model.forward``, and written to the tile's owned region. Final pass only."""
    import torch
    from ..data import iter_windows
    if not tta_variants:
        tta_variants = [((), None)]                                          # no TTA
    gauss_t = torch.from_numpy(make_gaussian(P)).to(device)
    tiles = pass_blend_tiles(cur_arr.shape if cur_arr is not None else out_shape, P, S, offset, region, T, level=level)
    total_win = sum(len(t["starts"][0]) * len(t["starts"][1]) * len(t["starts"][2]) for t in tiles)
    print(f"[pass-blend] {len(tiles)} tiles / {total_win} windows (S={S} T={T})"
          f"{' [claim-queue]' if donedir else ''}", flush=True)
    t0 = time.time(); done = [0]
    asm_ex = ThreadPoolExecutor(max_workers=max(1, prefetch_workers))              # off-thread batch assembly (CPU/IO)
    prev_ex = ThreadPoolExecutor(max_workers=2)                                    # off-thread prev-pass slab reads
    PREV_DEPTH = 2                                                                  # prev slabs in flight (big; 2 hides the read)

    def _read_prev(tile):
        """Read a tile's prev-pass slab (the block its ``req`` covers) AHEAD of the GPU, so passes 1-3 don't
        STALL on the synchronous read (the pass-1 slowdown). None on pass 0 (``prev_arr`` is None)."""
        if prev_arr is None:
            return None
        _, z0, z1, y0, y1, x0, x1 = tile["req"]
        return np.asarray(prev_arr[z0:z1, y0:y1, x0:x1]).astype(np.float32) / 255.0

    def _assemble_batch(chunk_st, ct_blk, pv_blk, bz0, by0, bx0):
        """Build one batch's [B,Cbase,P,P,P] net input (model.norm_ct + prev-gather + model.base_input) OFF the GPU
        thread, so the next batch's CPU/IO overlaps the current batch's forward (kills the 100%->0% starvation
        sawtooth). Returns (stacked_float32 | None, keep_coords); None when every window in the batch is air."""
        arrs, keep = [], []
        for (z, y, x) in chunk_st:
            oz, oy, ox = z - bz0, y - by0, x - bx0
            sub = ct_blk[oz:oz + P, oy:oy + P, ox:ox + P]
            if sub.shape != (P, P, P) or int(sub.max()) < air:                     # off-block or all-air → contributes nothing
                continue
            ct_n = model.norm_ct(sub)
            pv = None if pv_blk is None else pv_blk[oz:oz + P, oy:oy + P, ox:ox + P]
            arrs.append(model.base_input(ct_n, pv).astype(np.float32)); keep.append((z, y, x))
        return (np.stack(arrs) if arrs else None, keep)

    def process_tile(ct_blk, tile, pv_blk):
        oz0, oz1, oy0, oy1, ox0, ox1 = tile["owned"]
        bz0, by0, bx0 = tile["origin"]
        sz, sy, sx = tile["starts"]
        th, tw, tdp = oz1 - oz0, oy1 - oy0, ox1 - ox0
        acc = torch.zeros((th, tw, tdp), dtype=torch.float32, device=device)   # Σ (l1-l0)·g
        wsum = torch.zeros((th, tw, tdp), dtype=torch.float32, device=device)  # Σ g
        aux_acc = (torch.zeros((aux_arr.shape[0], th, tw, tdp), dtype=torch.float32, device=device)
                   if aux_arr is not None else None)                          # Σ aux·g (per channel)
        if pv_blk is not None:                                                     # prefetched by _read_prev; trim to the CT block
            pv_blk = pv_blk[:ct_blk.shape[0], :ct_blk.shape[1], :ct_blk.shape[2]]
        starts = [(z, y, x) for z in sz for y in sy for x in sx]
        chunks = [starts[i:i + batch] for i in range(0, len(starts), batch)]       # fixed batches, in order
        # Pipeline: assemble the next `depth` batches on the pool while the GPU runs the current one, so the
        # CPU work (norm_ct + prev-gather + stack) and the prev-slab reads overlap compute instead of stalling
        # it. Batches are consumed strictly in submit order, so the blend accumulation below is byte-identical
        # to the serial version — this is a pure scheduling change, no maths touched.
        depth = max(2, readahead)
        pend = deque(); nxt = 0
        while nxt < min(depth, len(chunks)):
            pend.append(asm_ex.submit(_assemble_batch, chunks[nxt], ct_blk, pv_blk, bz0, by0, bx0)); nxt += 1
        while pend:
            arrs_np, keep = pend.popleft().result()
            if nxt < len(chunks):                                                  # keep the queue full
                pend.append(asm_ex.submit(_assemble_batch, chunks[nxt], ct_blk, pv_blk, bz0, by0, bx0)); nxt += 1
            if arrs_np is None:                                                     # whole batch was air → nothing to do
                continue
            xb = torch.from_numpy(arrs_np).to(device, non_blocking=True)           # [B,Cbase,P,P,P]
            wmeshes = None
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                if mesh_sink is not None:
                    seg_logits, wmeshes = model.forward_meshes(xb, voxel_um)        # surface + per-window meshes (no TTA)
                    aux = None
                elif aux_acc is not None and len(tta_variants) == 1:
                    seg_logits, aux = model.forward(xb)                             # one forward gives both (no TTA)
                elif aux_acc is not None:
                    seg_logits = _tta_surface(model, xb, tta_variants)             # TTA'd surface …
                    _, aux = model.forward(xb)                                      # … + one identity forward for aux
                else:
                    seg_logits = _tta_surface(model, xb, tta_variants)             # [B,2,P,P,P] (TTA-avg if on)
                    aux = None
                diff = (seg_logits[:, 1] - seg_logits[:, 0]).float()               # [B,P,P,P] logit margin
                aux_f = aux.float() if aux is not None else None                   # [B,Caux,P,P,P] in [0,1]
            for bj, (z, y, x) in enumerate(keep):
                iz0, iz1 = max(z, oz0), min(z + P, oz1)                             # window ∩ owned (absolute)
                iy0, iy1 = max(y, oy0), min(y + P, oy1)
                ix0, ix1 = max(x, ox0), min(x + P, ox1)
                gp = gauss_t[iz0 - z:iz1 - z, iy0 - y:iy1 - y, ix0 - x:ix1 - x]     # gauss over that overlap (patch coords)
                acc[iz0 - oz0:iz1 - oz0, iy0 - oy0:iy1 - oy0, ix0 - ox0:ix1 - ox0] += \
                    diff[bj, iz0 - z:iz1 - z, iy0 - y:iy1 - y, ix0 - x:ix1 - x] * gp
                wsum[iz0 - oz0:iz1 - oz0, iy0 - oy0:iy1 - oy0, ix0 - ox0:ix1 - ox0] += gp
                if aux_acc is not None:
                    aux_acc[:, iz0 - oz0:iz1 - oz0, iy0 - oy0:iy1 - oy0, ix0 - ox0:ix1 - ox0] += \
                        aux_f[bj, :, iz0 - z:iz1 - z, iy0 - y:iy1 - y, ix0 - x:ix1 - x] * gp[None]
                done[0] += 1
            if mesh_sink is not None:                                               # per-window meshes (not blended)
                for bj, (z, y, x) in enumerate(keep):
                    mesh_sink.add_window((z, y, x), wmeshes[bj])
        if cur_arr is None:                                                        # mesh-only run (surf=off): no surface write
            return
        mask = wsum > 0
        wden = wsum.clamp(min=1e-12)                                                # shared denom (surface + aff)
        acc.div_(wden); acc.clamp_(-30.0, 30.0)
        torch.sigmoid_(acc); acc.mul_(mask)                                        # 0 where no window covered (air)
        u8 = acc.mul_(255.0).round_().clamp_(0, 255).to(torch.uint8).cpu().numpy()
        cur_arr[oz0:oz1, oy0:oy1, ox0:ox1] = u8
        if aux_acc is not None:
            aux_acc.div_(wden[None]); aux_acc.mul_(mask[None]); aux_acc.clamp_(0.0, 1.0)
            aux_u8 = aux_acc.mul_(255.0).round_().clamp_(0, 255).to(torch.uint8).cpu().numpy()
            aux_arr[:, oz0:oz1, oy0:oy1, ox0:ox1] = aux_u8

    def rate():
        dt = time.time() - t0
        return f"{done[0]} win · {_hms(dt)} · {done[0]/max(dt,1e-9):.1f} win/s"

    if donedir is None:                                                            # ---- single instance
        reqs = [t["req"] for t in tiles]
        pvq = deque(); j = 0                                                        # prev-slab prefetch (tiles order)
        while j < min(PREV_DEPTH, len(tiles)): pvq.append(prev_ex.submit(_read_prev, tiles[j])); j += 1
        for bi, (req, blk, org) in enumerate(iter_windows(vol, reqs, readahead=readahead, workers=prefetch_workers)):
            pv = pvq.popleft().result()
            if j < len(tiles): pvq.append(prev_ex.submit(_read_prev, tiles[j])); j += 1
            process_tile(np.asarray(blk), tiles[bi], pv)
            if log_every and bi and bi % log_every == 0:
                print(f"[pass-blend] P{pass_idx}/{passes - 1} tile {bi}/{len(tiles)} ({rate()})", flush=True)
    else:                                                                          # ---- multi instance: claim-queue
        ntot = len(tiles); nproc = 0
        g0, tg0 = CQ.count_done(donedir), time.time()
        for chunk in CQ.claim_chunks(donedir, range(ntot), chunk=claim_chunk, reclaim=reclaim):
            reqs = [tiles[g]["req"] for g in chunk]
            pvq = deque(); j = 0                                                    # prev-slab prefetch (chunk order)
            while j < min(PREV_DEPTH, len(chunk)): pvq.append(prev_ex.submit(_read_prev, tiles[chunk[j]])); j += 1
            for (req, blk, org), gid in zip(iter_windows(vol, reqs, readahead=readahead, workers=prefetch_workers), chunk):
                pv = pvq.popleft().result()
                if j < len(chunk): pvq.append(prev_ex.submit(_read_prev, tiles[chunk[j]])); j += 1
                process_tile(np.asarray(blk), tiles[gid], pv)
                CQ.mark_done(donedir, gid)
                nproc += 1
            gd = CQ.count_done(donedir)
            grate = (gd - g0) / max(time.time() - tg0, 1e-9)
            eta = (ntot - gd) / grate if grate > 0 else float("inf")            # seconds
            full = ntot / grate if grate > 0 else float("inf")                  # whole-pass seconds at this rate
            run_eta = eta + max(0, passes - 1 - pass_idx) * full                # + the passes still to come
            pass_pct = 100 * gd // ntot
            run_pct = int(100 * (pass_idx + gd / ntot) / max(passes, 1))        # overall, across all passes
            print(f"[pass-blend] P{pass_idx}/{passes - 1} | this worker: {nproc} tiles · {rate()} "
                  f"| all workers: {gd}/{ntot} tiles · {grate * 60:.1f} tile/min "
                  f"| pass ETA {_hms(eta)} ({pass_pct}%) · run ETA {_hms(run_eta)} ({run_pct}%)", flush=True)
    asm_ex.shutdown(wait=True); prev_ex.shutdown(wait=True)
    print(f"[pass-blend] DONE (this worker) {rate()}", flush=True)


# ----------------------------------------------------------------------------- driver + pass-selection specs
def _resolve_pass_spec(spec, passes, flag="--finalise"):
    """Resolve a pass-selection spec (shared by ``--finalise`` and ``--upload``) to a set of absolute pass indices.
    ``spec``: ``"last"``/``"-1"`` (default → only the final pass, the deliverable), ``"all"`` (every pass),
    ``"no"``/``"none"`` (none), or a comma-list of pass indices like ``"0,2,3"`` (negatives count from the end).
    ``flag`` names the CLI option in error messages."""
    s = str(spec).strip().lower()
    if s in ("no", "none", "off"):
        return set()
    if s == "all":
        return set(range(passes))
    if s in ("last", ""):
        return {passes - 1}
    out = set()
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            i = int(tok)
        except ValueError:
            raise SystemExit(f"hercunet infer: {flag} got {spec!r}; expected 'last', 'all', 'no', or "
                             f"comma-separated pass indices like '0,2,3' or '-1'.")
        if i < 0:
            i += passes
        if not (0 <= i < passes):
            raise SystemExit(f"hercunet infer: {flag} index {tok} out of range for {passes} passes (0..{passes-1}).")
        out.add(i)
    return out


def run_inference(source, out_prefix, *, model_type="nnunet-jacobi", model=None, model_hf=None,
                  ckpt="checkpoint_best.pth", device="cuda", model_opts=None, model_import=None,
                  air=25, region=None, level=0, keep_buffers=False, finalise="last", resume=False, batch=4, nb=4,
                  s3_prefix=None, upload="last", multi=False, reclaim=False, claim_chunk=6, leader=False,
                  readahead=2, prefetch_workers=3, tta="none", tta_passes="all",
                  surf_store="local", mesh_store="off"):
    """Model-agnostic inference engine. Build the model (:class:`hercunet.infer.model.InferenceModel`) named by
    ``model_type`` from ``model``/``model_hf`` + ``model_opts``, then run ``model.passes`` double-buffered passes over
    ``source`` and write per-pass surface-prob OME-Zarrs ``{out_prefix}_pass{p}.zarr``. Returns the final pass path.

    Single-pass models (``model.passes == 1``, e.g. HercuViT) just run pass 0 with prev=0. For Jacobi models
    (``passes>1``): pass 0 aligned grid, prev=0 (bootstrap); pass p>0 grid shifted on odd p, reads pass p-1.
    Buffers pruned to the last two unless ``keep_buffers``. ``finalise``/``upload`` select which passes get a pyramid /
    upload to ``s3_prefix`` (see :func:`_resolve_pass_spec`). ``region``=(z0,z1,y0,y1,x0,x1) restricts computed tiles.
    ``model.overlap`` > 0 → Gaussian-blend mode; 0 → disjoint. ``tta`` applies per-window TTA on ``tta_passes``.

    ``multi`` → MULTI-INSTANCE: run the identical worker on any number of processes sharing ``out_prefix`` storage;
    within a pass workers steal tiles via the ``claim_queue``; between passes a BARRIER + one ``leader`` builds the
    pyramid + uploads + writes ``_complete``. All model-specific behaviour (load, norm, input stack, forward, aux
    output) is in the model; the orchestration here is unchanged."""
    from .model import InferenceModel
    from . import models as _builtin  # noqa: F401 — registers the built-in models (nnunet-jacobi)
    if model_import:                                                    # an out-of-tree model module to register first
        import importlib
        importlib.import_module(model_import)                           # e.g. "hercunet.hercuvit.infer.model"
    m = InferenceModel.get(model_type).load(model=model, model_hf=model_hf, ckpt=ckpt, device=device,
                                            **(model_opts or {}))
    passes = int(m.passes)
    overlap = float(m.overlap)
    aux_spec = m.aux
    P = int(m.patch_size)
    chunk = P // 2                                                     # phase boundaries (0, P/2, P, ...) all aligned
    # OVERLAP mode: each pass is a Gaussian-blended overlapping-window inference; passes shift by S/2 so centres land
    # on the previous pass's overlap-centres. S forced even so S/2 is integral; output tile T is a multiple of the
    # zarr chunk (P/2) so parallel tile writes never share a chunk.
    blend = bool(overlap) and overlap > 0.0
    if aux_spec is not None and not blend:
        raise SystemExit("infer: this model's aux output requires overlap>0 (blend mode); rerun without --overlap 0.")
    S = Tt = None
    if blend:
        S = 2 * max(1, int(round(P * (1.0 - float(overlap)) / 2)))
        S = int(min(max(S, 2), P - 2))                                # 0 < S < P (real overlap AND real shift)
        Tt = max(P, int(nb) * (P // 2))                              # output tile per axis (chunk-aligned)
        print(f"[engine] OVERLAP mode: stride S={S} (overlap {1 - S / P:.2f}), tile T={Tt}, "
              f"inter-pass shift {S // 2} (centres land on prev overlap-centres)", flush=True)
    vol = NI.open_source(source)
    label = NI.source_label(source)
    # --- WORKING RESOLUTION: run on OME-Zarr pyramid `level` (default 0 = full res). Everything downstream — the
    # buffer canvas, the window grid, --region, the output voxel size — is expressed on THIS level's grid, so a
    # fine rescan can be run at the ~8.6 µm scale the model expects by picking the nearest level (e.g. 2.4 µm -> L2).
    nlev = int(vol.meta.num_levels)
    level = int(level)
    if not (0 <= level < nlev):
        raise SystemExit(f"hercunet infer --level {level}: source has levels 0..{nlev - 1} only ({label}).")
    shape = tuple(int(v) for v in vol.meta.level_shapes[level])
    vx = float(vol.meta.voxel_size_um) * float(vol.meta.level_downsamples[level])
    region_full = tuple(region) if region else (0, shape[0], 0, shape[1], 0, shape[2])
    # Big, unmissable warning if this level's voxel size is outside the model's training band (model.train_voxel_band;
    # None → skip). Outside the band the predictions will be unreliable.
    band = m.train_voxel_band
    if band is not None and not (band[0] <= vx <= band[1]):
        mid = 0.5 * (band[0] + band[1])
        best = min(range(nlev), key=lambda L: abs(vol.meta.voxel_size_um * vol.meta.level_downsamples[L] - mid))
        best_vx = float(vol.meta.voxel_size_um) * float(vol.meta.level_downsamples[best])
        bar = "!" * 92
        print(f"\n{bar}\n"
              f"!!  WARNING — OUT-OF-DOMAIN RESOLUTION: --level {level} of {label} is {vx:.2f} µm/voxel.\n"
              f"!!  {model_type} was trained near {band[0]}-{band[1]} µm; at {vx:.2f} µm it is OUT OF DISTRIBUTION\n"
              f"!!  and its surface predictions will be UNRELIABLE. Closest in-band level for this source is "
              f"--level {best} ({best_vx:.2f} µm).\n"
              f"!!  Re-run with --level {best} unless you specifically intend this resolution.\n"
              f"{bar}\n", flush=True)
    print(f"[engine] {model_type} {label} level={level} vx={vx:.3f}µm shape={shape} "
          f"(L0={tuple(int(v) for v in vol.meta.level_shapes[0])}) "
          f"P={P} chunk={chunk} passes={passes} region={region_full} batch={batch} nb={nb}", flush=True)

    # ---- surface/mesh output selection (surf_store / mesh_store ∈ {off, local, upload}). The surface volume is
    # created+finalised+uploaded through the existing path when stored (upload gated by the `upload` spec, which the
    # caller sets from surf_store); meshes are fit per-window and written to a per-worker .asv shard, merged by the
    # leader into {out}_meshes.asv (uploaded when mesh_store==upload). `want_mesh` needs a mesh-emitting model.
    want_surf = surf_store != "off"
    want_mesh = bool(getattr(m, "emits_meshes", False)) and mesh_store != "off"
    if mesh_store != "off" and not getattr(m, "emits_meshes", False):
        raise SystemExit(f"infer: --mesh {mesh_store} but model {model_type!r} emits no meshes.")
    if want_mesh and passes != 1:
        raise SystemExit("infer: mesh output is only supported for single-pass models (e.g. HercuViT).")
    if not want_surf and not want_mesh:
        raise SystemExit("infer: nothing to produce — both surf and mesh are off (or the model emits no meshes).")
    ds_level = float(vol.meta.level_downsamples[level])                  # working level -> L0 voxel scale
    l0_voxel = float(vol.meta.voxel_size_um)
    mesh_frame = {"vc_open_data_coordinate_space": label}
    # Meshes NEVER use the blend tile-halo (which re-runs each window per covering tile — huge duplication). When
    # the surface is NOT stored, drop blend entirely and run a simple strided window grid (each window ONCE); a sub-P
    # stride gives the requested small window overlap. When the surface IS blended, meshes ride those (overlapping)
    # windows as-is (per-window, redundant) — use --overlap 0 for a clean mesh store alongside a disjoint surface.
    use_blend = blend and want_surf
    win_stride = (int(round(P * (1.0 - overlap))) if (want_mesh and not use_blend and overlap > 0) else P)
    win_stride = int(min(max(win_stride, 1), P))
    print(f"[engine] outputs: surface={surf_store} mesh={mesh_store if want_mesh else 'off'} "
          f"(L0 voxel {l0_voxel:.3f}µm, level downsample {ds_level:.0f}; "
          f"{'blend' if use_blend else f'disjoint stride {win_stride}'})", flush=True)

    # ---- STORAGE WARNING: each pass writes a full-resolution uint8 surface-prob buffer. For a whole scroll that
    # is HUNDREDS of GB (up to ~500 GB); the number below is an UPPER BOUND over the computed box (only material /
    # non-air chunks are actually written, so real usage is the material fraction — less). By default buffers are
    # pruned to the last two, so budget ~2 on disk at once (a 3rd may briefly exist while one finalises);
    # --keep-buffers keeps ALL `passes`. Make sure --out has this much free space or a pass dies partway.
    rz0, rz1, ry0, ry1, rx0, rx1 = region_full
    box_gb = (rz1 - rz0) * (ry1 - ry0) * (rx1 - rx0) / 1e9            # uint8 = 1 byte/voxel
    buf_gb = box_gb * 1.15                                            # + max-pool pyramid (~1/8+1/64+… ≈ 14%)
    n_on_disk = passes if keep_buffers else min(passes, 2)
    print(f"[jacobi] STORAGE: up to ~{buf_gb:.0f} GB per pass buffer (uint8 L0+pyramid over the computed box; "
          f"actual = the material fraction, less — air chunks are not written). Keeping "
          f"{'ALL ' + str(passes) if keep_buffers else '~2'} buffer(s) → ensure roughly ~{buf_gb * n_on_disk:.0f} "
          f"GB is free where --out lives ({out_prefix}).", flush=True)

    finalise_set = _resolve_pass_spec(finalise, passes, "--finalise")   # which passes get an OME-Zarr pyramid
    upload_set = _resolve_pass_spec(upload, passes, "--upload") if s3_prefix else set()   # which passes upload to S3
    tta_all = _resolve_tta(tta)                                          # TTA variant list (identity only if off)
    tta_set = _resolve_pass_spec(tta_passes, passes, "--tta-passes") if len(tta_all) > 1 else set()  # passes w/ TTA
    if len(tta_all) > 1:
        print(f"[jacobi] TTA: {len(tta_all)} variants (--tta {tta}) on pass(es) "
              f"{sorted(tta_set) if tta_set else 'none'} (--tta-passes {tta_passes}) — {len(tta_all)}x forwards there",
              flush=True)
    print(f"[jacobi] finalise (build pyramid) for pass(es): "
          f"{sorted(finalise_set) if finalise_set else 'none'} (--finalise {finalise})", flush=True)
    if s3_prefix:
        print(f"[jacobi] upload to S3 for pass(es): {sorted(upload_set) if upload_set else 'none'} "
              f"(--upload {upload}) -> {s3_prefix}", flush=True)

    import threading
    import zarr
    if multi and leader:
        print("[jacobi] THIS worker is the designated LEADER (creates each pass zarr + finalises + uploads)", flush=True)
    fin_threads = {}                                                   # pass -> Thread doing background finalize+upload
    prev_arr, prev_path = None, None
    # --resume: find the HIGHEST pass with a _complete marker. Every pass at/below it is done — even if its buffer
    # was already freed (we keep only the last ~2). Resume from the next pass, wiring prev from that highest-complete
    # pass, whose buffer is always still present (nothing ran after it to free it). This lets a run stop/resume PAST
    # pass 0 without re-running freed passes. (The per-pass check inside the loop stays as a safety net.)
    start_pass = 0
    if resume:
        for _p in range(passes):
            _cp = f"{out_prefix}_pass{_p}.zarr"
            if os.path.isdir(_cp) and os.path.exists(os.path.join(_cp + ".done", "_complete")):
                start_pass, prev_path = _p + 1, _cp
        if prev_path is not None:
            prev_arr = zarr.open_group(prev_path, mode="r")["0"]
            print(f"[jacobi] resume: passes 0..{start_pass - 1} already complete; prev buffer={prev_path}; "
                  f"resuming at pass {start_pass}", flush=True)
    for p in range(start_pass, passes):
        offset = 0 if p % 2 == 0 else (S // 2 if blend else P // 2)
        cur_path = f"{out_prefix}_pass{p}.zarr"
        donedir = cur_path + ".done"
        complete = os.path.join(donedir, "_complete")
        if resume and os.path.isdir(cur_path) and os.path.exists(complete):
            print(f"[jacobi] pass {p}: already complete, skipping", flush=True)
            prev_arr = zarr.open_group(cur_path, mode="r")["0"]
            prev_path = cur_path
            continue
        tag = os.path.basename(out_prefix.rstrip("/")) or "iter"    # distinct volume name PER RUN so VC3D can tell
        os.makedirs(donedir, exist_ok=True)
        print(f"[jacobi] pass {p} offset={offset}{' [multi]' if multi else ''}", flush=True)

        # open/create the pass zarr — single: this proc; multi: ONE elected leader creates, the rest attach r+.
        # (create_surface_zarr opens-in-place when it already exists, so a resume never clobbers L0.)
        # VC3D volume name/uuid come from the OUTPUT prefix (distinctive per run), NOT the source CT basename —
        # a name that embeds the raw CT volume's name collides with the CT volume VC3D already has loaded, so the
        # prediction won't show as a separate attachable volume.
        vol_name = f"{tag}-p{p}"
        cur = None
        if want_surf:                                                  # surf=off (mesh-only) skips the surface volume
            if not multi or leader:
                cur = NI.create_surface_zarr(cur_path, shape, chunk, vx, tag, vol_name, overwrite=not resume)
                if multi:
                    open(os.path.join(donedir, "_created"), "w").close()   # signal followers the zarr is ready
            else:
                CQ.wait_for_path(os.path.join(donedir, "_created"), f"pass {p} zarr creation")
                cur = zarr.open_group(cur_path, mode="r+")["0"]

        # model.aux (e.g. affinity): on the FINAL pass only (most-settled prev → best, deliverable-consistent),
        # create a 4-D (channels,Z,Y,X) uint8 zarr and let run_pass_blend fill it from the identity forward.
        aux_arr = None
        emit_aux = aux_spec is not None and (p == passes - 1 or not aux_spec.final_pass_only)
        if emit_aux:
            aux_path = f"{out_prefix}_pass{p}_{aux_spec.name}.zarr"
            if not multi or leader:
                aux_arr = NI.create_affinity_zarr(aux_path, aux_spec.channels, shape, chunk, vx,
                                                  aux_spec.offsets, p, overwrite=not resume)
                if multi:
                    open(os.path.join(donedir, "_aux_created"), "w").close()
            else:
                CQ.wait_for_path(os.path.join(donedir, "_aux_created"), f"pass {p} aux zarr creation")
                aux_arr = zarr.open_group(aux_path, mode="r+")["0"]
            print(f"[engine] pass {p}: ALSO emitting {aux_spec.name} ({aux_spec.channels} ch, uint8) -> {aux_path}", flush=True)

        pass_variants = tta_all if p in tta_set else [((), None)]   # TTA only on the selected passes
        mesh_sink = mesh_writer = shard_path = None                  # per-worker .asv shard (this pid)
        if want_mesh:
            from ..meshes.asv import AsvWriter
            from ..meshes.extract import MeshSink
            shard_path = f"{out_prefix}_meshes.{os.getpid()}.asv"
            mesh_writer = AsvWriter(shard_path, frame=mesh_frame, voxel_um=l0_voxel, grid_step_um=30.0, overwrite=True)
            mesh_sink = MeshSink(mesh_writer, downsample=ds_level)
            if multi:
                open(os.path.join(donedir, f"_meshshard_{os.getpid()}.active"), "w").close()
            print(f"[engine] pass {p}: writing mesh shard -> {shard_path}", flush=True)
        if use_blend:
            run_pass_blend(m, vol, prev_arr, cur, region_full, offset, P, S, Tt, device, air=air,
                           batch=batch, donedir=(donedir if multi else None), reclaim=reclaim,
                           claim_chunk=claim_chunk,
                           readahead=readahead, prefetch_workers=prefetch_workers, tta_variants=pass_variants,
                           aux_arr=aux_arr, pass_idx=p, passes=passes, level=level, mesh_sink=mesh_sink, voxel_um=vx,
                           out_shape=shape)
        else:
            run_pass(m, vol, prev_arr, cur, region_full, offset, P, device, air=air, batch=batch, nb=nb,
                     donedir=(donedir if multi else None), reclaim=reclaim, claim_chunk=claim_chunk,
                     readahead=readahead, prefetch_workers=prefetch_workers, tta_variants=pass_variants,
                     pass_idx=p, passes=passes, level=level, mesh_sink=mesh_sink, voxel_um=vx, out_shape=shape,
                     win_stride=win_stride)
        if want_mesh:
            info = mesh_writer.close()
            print(f"[engine] pass {p}: mesh shard done ({info['sheet_count']} sheets, {info['vert_count']} verts)", flush=True)
            if multi:
                open(os.path.join(donedir, f"_meshshard_{os.getpid()}.done"), "w").close()

        # ---- pass p compute done. Multi: BARRIER (also the propagation guarantee — it polls until every L0 block
        # .done is VISIBLE, so L0 is readable cross-node by then). Write ``_complete`` (resume marker) at once — the
        # NEXT pass gates on the barrier / L0, NOT on the pyramid. finalise+upload (pyramid for VC3D + S3) is
        # per-pass OPTIONAL (see --finalise) and DECOUPLED into a background thread, so it can be limited to the
        # last pass (default) or skipped entirely (--finalise no) and run SEPARATELY on a cheap CPU box (finalise
        # needs no GPU). A bounded-backlog barrier keeps at most ONE finalise in flight, so a fast follower can't
        # outrun the leader and prune/reuse a buffer a finalise still reads.
        def _finalize_upload(cp=cur_path, pp=p):
            uri = f"{s3_prefix.rstrip('/')}/pass{pp}.zarr" if (s3_prefix and pp in upload_set) else None

            def _up(what):
                if not uri:
                    return
                try:                                                  # L0: cp whole tree; pyramid: sync (incremental)
                    NI.upload_zarr_to_s3(cp, uri, incremental=(what == "pyramid"))
                    print(f"[jacobi] PASS {pp} {what} ON S3 — ATTACH: {NI.s3_to_https(uri)}", flush=True)
                except Exception as e:
                    print(f"[jacobi] pass {pp} {what} upload FAILED: {e}", flush=True)

            _up("L0")                                                  # upload L0 FIRST — safe + directly viewable in
            if pp in finalise_set:                                     #   VC3D (single-level multiscales), independent
                try:                                                   #   of the pyramid; a finalise failure can't lose it
                    NI.finalize_pyramid(cp, region=region_full)        # region-restricted; per-pass pyramid for VC3D
                except Exception as e:
                    print(f"[jacobi] pass {pp} finalise FAILED: {e}", flush=True)
                else:
                    _up("pyramid")                                     # re-upload: sync adds only levels 1-5 + attrs

            # model.aux: ship the 4-D aux zarr alongside the deliverable (whole-tree cp; no pyramid — it is a 4-D
            # analysis field, not a VC3D volume). ``uri`` is truthy only when this pass is in --upload, so this gates
            # on the same s3_prefix + upload_set; it exists only on the aux-emitting pass(es).
            if aux_spec is not None and uri and (pp == passes - 1 or not aux_spec.final_pass_only):
                aux_lp = f"{out_prefix}_pass{pp}_{aux_spec.name}.zarr"
                aux_uri = f"{s3_prefix.rstrip('/')}/pass{pp}_{aux_spec.name}.zarr"
                if os.path.isdir(aux_lp):
                    try:
                        NI.upload_zarr_to_s3(aux_lp, aux_uri, incremental=False)   # same throttled streaming logs
                        print(f"[engine] PASS {pp} {aux_spec.name.upper()} ON S3 — {NI.s3_to_https(aux_uri)}", flush=True)
                    except Exception as e:
                        print(f"[engine] pass {pp} {aux_spec.name} upload FAILED: {e}", flush=True)

        if multi:
            nbk = (len(pass_blend_tiles(shape, P, S, offset, region_full, Tt)) if use_blend
                   else _n_blocks(shape, P, offset, region_full, nb, stride=win_stride))
            print(f"[jacobi] pass {p}: barrier — waiting for all {nbk} {'tiles' if blend else 'blocks'} .done", flush=True)
            CQ.wait_all_done(donedir, nbk, what=f"pass {p}")
            if leader:
                if want_surf and ((p in finalise_set) or (s3_prefix and p in upload_set)):  # decouple: BOUND to 1 in-flight
                    for q in sorted(k for k in fin_threads if k < p):  # (no unfinalised-buffer overwrite: a still-
                        fin_threads.pop(q).join()                      #  running finalise/upload's pass is never pruned/reused)
                    t = threading.Thread(target=_finalize_upload, name=f"finalise-p{p}"); t.start()
                    fin_threads[p] = t
                open(complete, "w").close()                            # pass L0 done → resume marker (instant)
            print(f"[jacobi] pass {p}: barrier cleared ({'leader' if leader else 'follower'} → pass {p + 1})", flush=True)
        else:
            if want_surf:
                _finalize_upload()                                     # single instance: inline
            open(complete, "w").close()

        if want_mesh and (leader or not multi):                       # leader merges per-worker shards -> one .asv
            import glob
            import time as _time
            from ..meshes.asv import merge_asv
            final = f"{out_prefix}_meshes.asv"
            if multi:                                                 # wait until every active shard has closed
                for _ in range(1800):
                    act = len(glob.glob(os.path.join(donedir, "_meshshard_*.active")))
                    dn = len(glob.glob(os.path.join(donedir, "_meshshard_*.done")))
                    if act >= 1 and dn >= act:
                        break
                    _time.sleep(1.0)
            shards = sorted(s for s in glob.glob(f"{out_prefix}_meshes.*.asv") if s != final)
            if shards:
                res = merge_asv(shards, final, frame=mesh_frame, voxel_um=l0_voxel, grid_step_um=30.0, overwrite=True)
                print(f"[engine] merged {len(shards)} mesh shard(s) -> {final} ({res['sheet_count']} sheets)", flush=True)
                for s in shards:
                    try:
                        os.remove(s)
                    except OSError:
                        pass
                if mesh_store == "upload" and s3_prefix:
                    uri = f"{s3_prefix.rstrip('/')}/meshes.asv"
                    try:
                        NI.upload_file_to_s3(final, uri)
                        print(f"[engine] MESHES ON S3 — {NI.s3_to_https(uri)}", flush=True)
                    except Exception as e:
                        print(f"[engine] mesh upload FAILED: {e}", flush=True)

        if want_surf and not keep_buffers and p >= 1 and prev_path is not None and (leader or not multi):
            import shutil                                              # p-1's finalize already joined above (bound=1)
            shutil.rmtree(prev_path, ignore_errors=True)
            shutil.rmtree(prev_path + ".done", ignore_errors=True)
        if want_surf:
            prev_arr = zarr.open_group(cur_path, mode="r")["0"]        # next pass reads this pass's L0 (done @ barrier)
            prev_path = cur_path

    for pp, t in sorted(fin_threads.items()):                          # leader: finish the last passes' finalize+upload
        print(f"[jacobi] waiting on background finalize/upload for pass {pp}", flush=True)
        t.join()

    print(f"[jacobi] DONE -> {prev_path}", flush=True)
    return prev_path

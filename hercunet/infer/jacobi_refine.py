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

import numpy as np

from . import nnunet_infer as NI
from . import claim_queue as CQ


# ----------------------------------------------------------------------------- multi-instance coordination
# The generic elastic claim-queue (atomic .claim / .done, barrier, orphan reclaim) lives in ``claim_queue`` and is
# shared with full-volume inference — nothing is copied here. Per-pass, each nb^3 prefetch BLOCK is one claim item
# (blocks are output-disjoint so parallel writes never race); passes are separated by a barrier + leader-finalize
# (a block of pass p+1 reads pass p output ONE window over, via the ½-shift). ``donedir`` must be on storage
# shared by all workers (a network volume across pods, or the local FS for multi-GPU on one box).
def _n_blocks(shape, P, offset, region, nb):
    """Number of prefetch blocks a pass produces (for the barrier) — via ``pass_blocks`` so it can never drift
    from what ``run_pass`` actually claims."""
    return len(pass_blocks(shape, P, offset, region, nb)[0])


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
def _window_grid(dim, P, offset, r0, r1):
    """Window START positions on the stride-P grid ``{offset + k*P}`` whose [s, s+P) intersects [r0, r1),
    clamped to [0, dim-P] so every read is a full P^3 (no padding). Deduped + sorted."""
    import math
    if dim <= P:
        return [0]
    kmin = math.floor((r0 - offset - P) / P) + 1
    kmax = math.ceil((r1 - offset) / P) - 1
    return sorted({int(min(max(offset + k * P, 0), dim - P)) for k in range(kmin, kmax + 1)})


def pass_blocks(shape, P, offset, region, nb):
    """The deterministic list of prefetch blocks for a pass (same on every worker → shared block ids). Each block
    = up to ``nb`` window-starts per axis; ``req`` is its CT read box, ``wins`` its stride-P window origins."""
    Zf, Yf, Xf = (int(s) for s in shape)
    rz0, rz1, ry0, ry1, rx0, rx1 = region
    gz = _window_grid(Zf, P, offset, rz0, rz1)
    gy = _window_grid(Yf, P, offset, ry0, ry1)
    gx = _window_grid(Xf, P, offset, rx0, rx1)
    grp = lambda g: [g[i:i + nb] for i in range(0, len(g), nb)]
    blocks = []
    for zg in grp(gz):
        for yg in grp(gy):
            for xg in grp(gx):
                blocks.append(dict(
                    req=(0, zg[0], zg[-1] + P, yg[0], yg[-1] + P, xg[0], xg[-1] + P),
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


def pass_blend_tiles(shape, P, S, offset, region, T):
    """Fixed, disjoint, chunk-aligned OUTPUT tiles (T per axis) each carrying the overlapping window starts that
    cover it (from the stride-S grid at ``offset``) + the CT read box (halo = min..max start + P). The output
    tiling is FIXED per volume; only the window grid moves with ``offset``. ``region`` restricts computed tiles."""
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
                                  req=(0, min(sz), max(sz) + P, min(sy), max(sy) + P, min(sx), max(sx) + P),
                                  starts=(sz, sy, sx)))
    return tiles


def run_pass(net, ctnorm, vol, prev_arr, cur_arr, region, offset, P, device,
             air=25, batch=4, nb=4, log_every=40, donedir=None, reclaim=False, claim_chunk=6, affinity=False,
             readahead=2, prefetch_workers=3):
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
    if affinity:
        from ..training.affinity import ct_orientation             # 8-ch model: [CT, prev] + 6 orient channels
    lo, hi, mean, std = ctnorm
    blocks, (ngz, ngy, ngx) = pass_blocks(cur_arr.shape, P, offset, region, nb)
    total_win = sum(len(b["wins"]) for b in blocks)
    print(f"[pass] {len(blocks)} blocks / {total_win} windows (grid {ngz}x{ngy}x{ngx})"
          f"{' [claim-queue]' if donedir else ''}", flush=True)

    t0 = time.time(); done = [0]
    buf, meta = [], []

    def flush():
        if not buf:
            return
        xb = torch.from_numpy(np.stack(buf)).to(device, non_blocking=True)     # [B,2,P,P,P]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            if affinity:                                                       # -> [B,8,P,P,P] with CT orientation
                orient = ct_orientation(xb[:, 0:1], sigma_grad=1.0, sigma_tensor=3.0)
                xb = torch.cat([xb, orient.to(xb.dtype)], dim=1)
            logits = net(xb)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]                                            # AffinityHeadNet base -> seg tensor
            prob = torch.softmax(logits, 1)[:, 1]                             # [B,P,P,P]
        u8 = prob.mul_(255.0).round_().clamp_(0, 255).to(torch.uint8).cpu().numpy()
        for i, (z, y, x) in enumerate(meta):
            cur_arr[z:z + P, y:y + P, x:x + P] = u8[i]
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
            ct_n = NI.norm_ct(sub, lo, hi, mean, std)
            pv = np.zeros((P, P, P), np.float32) if pv_blk is None else pv_blk[sz:sz + P, sy:sy + P, sx:sx + P]
            buf.append(np.stack([ct_n, pv]).astype(np.float32)); meta.append((z, y, x))
            done[0] += 1
            if len(buf) >= batch:
                flush()

    def rate():
        dt = time.time() - t0
        return f"{done[0]} win {dt:.0f}s {done[0]/max(dt,1e-9):.1f} win/s"

    if donedir is None:                                                       # ---- single instance: all blocks
        reqs = [b["req"] for b in blocks]
        for bi, (req, blk, org) in enumerate(iter_windows(vol, reqs, readahead=readahead, workers=prefetch_workers)):
            process_block(np.asarray(blk), blocks[bi])
            if log_every and bi and bi % log_every == 0:
                print(f"[pass] block {bi}/{len(blocks)} ({rate()})", flush=True)
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
            eta = (ntot - gd) / grate / 60.0 if grate > 0 else float("inf")
            print(f"[pass] worker {nproc}/{ntot} | GLOBAL {gd}/{ntot} ({100 * gd // ntot}%) "
                  f"{grate * 60:.1f} blk/min ETA {eta:.0f}m | {rate()}", flush=True)
    flush()
    print(f"[pass] DONE (this worker) {rate()}", flush=True)


def run_pass_blend(net, ctnorm, vol, prev_arr, cur_arr, region, offset, P, S, T, device,
                   air=25, batch=4, log_every=20, donedir=None, reclaim=False, claim_chunk=6, affinity=False,
                   readahead=2, prefetch_workers=3):
    """One OVERLAP-mode pass: like :func:`run_pass` (prefetched blocks, claim-queue, fp16) but each claim item is a
    disjoint OUTPUT TILE that runs every stride-S window covering it (halo) and merges them with the villa Gaussian
    blend in LOGIT space (accumulate ``(l1-l0)·g`` + ``g``, normalise, sigmoid), then writes only its owned region —
    so tiles never blend across each other and multi-instance stays race-free (same trick as full-volume infer)."""
    import torch
    from ..data import iter_windows
    if affinity:
        from ..training.affinity import ct_orientation
    lo, hi, mean, std = ctnorm
    gauss_t = torch.from_numpy(make_gaussian(P)).to(device)
    tiles = pass_blend_tiles(cur_arr.shape, P, S, offset, region, T)
    total_win = sum(len(t["starts"][0]) * len(t["starts"][1]) * len(t["starts"][2]) for t in tiles)
    print(f"[pass-blend] {len(tiles)} tiles / {total_win} windows (S={S} T={T})"
          f"{' [claim-queue]' if donedir else ''}", flush=True)
    t0 = time.time(); done = [0]

    def process_tile(ct_blk, tile):
        oz0, oz1, oy0, oy1, ox0, ox1 = tile["owned"]
        bz0, by0, bx0 = tile["origin"]
        sz, sy, sx = tile["starts"]
        th, tw, tdp = oz1 - oz0, oy1 - oy0, ox1 - ox0
        acc = torch.zeros((th, tw, tdp), dtype=torch.float32, device=device)   # Σ (l1-l0)·g
        wsum = torch.zeros((th, tw, tdp), dtype=torch.float32, device=device)  # Σ g
        pv_blk = None
        if prev_arr is not None:
            pv_blk = np.asarray(prev_arr[bz0:bz0 + ct_blk.shape[0], by0:by0 + ct_blk.shape[1],
                                         bx0:bx0 + ct_blk.shape[2]]).astype(np.float32) / 255.0
        starts = [(z, y, x) for z in sz for y in sy for x in sx]
        bi = 0
        while bi < len(starts):
            chunk_st = starts[bi:bi + batch]; bi += batch
            arrs, keep = [], []
            for (z, y, x) in chunk_st:
                oz, oy, ox = z - bz0, y - by0, x - bx0
                sub = ct_blk[oz:oz + P, oy:oy + P, ox:ox + P]
                if sub.shape != (P, P, P) or int(sub.max()) < air:            # off-block or all-air → contributes nothing
                    continue
                ct_n = NI.norm_ct(sub, lo, hi, mean, std)
                pv = np.zeros((P, P, P), np.float32) if pv_blk is None else pv_blk[oz:oz + P, oy:oy + P, ox:ox + P]
                arrs.append(np.stack([ct_n, pv]).astype(np.float32)); keep.append((z, y, x))
            if not arrs:
                continue
            xb = torch.from_numpy(np.stack(arrs)).to(device, non_blocking=True)     # [B,2,P,P,P]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                if affinity:
                    orient = ct_orientation(xb[:, 0:1], sigma_grad=1.0, sigma_tensor=3.0)
                    xb = torch.cat([xb, orient.to(xb.dtype)], dim=1)
                logits = net(xb)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                diff = (logits[:, 1] - logits[:, 0]).float()                        # [B,P,P,P] logit margin
            for bj, (z, y, x) in enumerate(keep):
                iz0, iz1 = max(z, oz0), min(z + P, oz1)                             # window ∩ owned (absolute)
                iy0, iy1 = max(y, oy0), min(y + P, oy1)
                ix0, ix1 = max(x, ox0), min(x + P, ox1)
                gp = gauss_t[iz0 - z:iz1 - z, iy0 - y:iy1 - y, ix0 - x:ix1 - x]     # gauss over that overlap (patch coords)
                acc[iz0 - oz0:iz1 - oz0, iy0 - oy0:iy1 - oy0, ix0 - ox0:ix1 - ox0] += \
                    diff[bj, iz0 - z:iz1 - z, iy0 - y:iy1 - y, ix0 - x:ix1 - x] * gp
                wsum[iz0 - oz0:iz1 - oz0, iy0 - oy0:iy1 - oy0, ix0 - ox0:ix1 - ox0] += gp
                done[0] += 1
        mask = wsum > 0
        acc.div_(wsum.clamp_(min=1e-12)); acc.clamp_(-30.0, 30.0)
        torch.sigmoid_(acc); acc.mul_(mask)                                        # 0 where no window covered (air)
        u8 = acc.mul_(255.0).round_().clamp_(0, 255).to(torch.uint8).cpu().numpy()
        cur_arr[oz0:oz1, oy0:oy1, ox0:ox1] = u8

    def rate():
        dt = time.time() - t0
        return f"{done[0]} win {dt:.0f}s {done[0]/max(dt,1e-9):.1f} win/s"

    if donedir is None:                                                            # ---- single instance
        reqs = [t["req"] for t in tiles]
        for bi, (req, blk, org) in enumerate(iter_windows(vol, reqs, readahead=readahead, workers=prefetch_workers)):
            process_tile(np.asarray(blk), tiles[bi])
            if log_every and bi and bi % log_every == 0:
                print(f"[pass-blend] tile {bi}/{len(tiles)} ({rate()})", flush=True)
    else:                                                                          # ---- multi instance: claim-queue
        ntot = len(tiles); nproc = 0
        g0, tg0 = CQ.count_done(donedir), time.time()
        for chunk in CQ.claim_chunks(donedir, range(ntot), chunk=claim_chunk, reclaim=reclaim):
            reqs = [tiles[g]["req"] for g in chunk]
            for (req, blk, org), gid in zip(iter_windows(vol, reqs, readahead=readahead, workers=prefetch_workers), chunk):
                process_tile(np.asarray(blk), tiles[gid])
                CQ.mark_done(donedir, gid)
                nproc += 1
            gd = CQ.count_done(donedir)
            grate = (gd - g0) / max(time.time() - tg0, 1e-9)
            eta = (ntot - gd) / grate / 60.0 if grate > 0 else float("inf")
            print(f"[pass-blend] worker {nproc}/{ntot} | GLOBAL {gd}/{ntot} ({100 * gd // ntot}%) "
                  f"{grate * 60:.1f} tile/min ETA {eta:.0f}m | {rate()}", flush=True)
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


def jacobi_refine(scroll, model, out_prefix, passes=3, ckpt="checkpoint_best.pth", device="cuda",
                  air=25, region=None, keep_buffers=False, finalise="last", resume=False, batch=4, nb=4,
                  s3_prefix=None, upload="last", multi=False, reclaim=False, claim_chunk=6, leader=False,
                  affinity=True, n_orient=6, overlap=0.25, readahead=2, prefetch_workers=3, local_vol=None):
    """Run ``passes`` double-buffered Jacobi refinement passes over ``scroll`` and write per-pass surface-prob
    OME-Zarrs ``{out_prefix}_pass{p}.zarr``. Returns the final pass path.

    pass 0: aligned grid, prev = 0 (bootstrap). pass p>0: grid shifted (all axes) on odd p, reads pass p-1.
    Buffers pruned to the last two unless ``keep_buffers``. ``finalise`` selects which passes get an OME-Zarr
    pyramid built and ``upload`` which passes upload to ``s3_prefix`` — both take the same spec (``"last"`` default
    = only the deliverable; ``"all"``; ``"no"``; or an index list — see :func:`_resolve_pass_spec`); ``upload`` is
    inert without ``s3_prefix``. ``region``=(z0,z1,y0,y1,x0,x1) restricts computed tiles (buffer canvas stays
    full-volume) — use it for a small-cube smoke test. ``overlap`` > 0 → Gaussian-blend mode (default 0.25,
    HercUNet v0); 0 → disjoint mode.

    ``multi`` → MULTI-INSTANCE. Run the identical worker on any number of processes sharing ``out_prefix`` storage
    (one per GPU on a box's local FS, or across pods on a network volume); extra workers can join mid-run. Within a
    pass, workers dynamically steal blocks/tiles via the ``claim_queue`` (a faster worker does more). Between passes
    there is a hard BARRIER (a pass-p+1 tile reads pass-p output one window over, so ALL of pass p must finish
    first): every worker waits for all items ``.done``, then ONE ``leader`` builds the pyramid + uploads + writes
    the ``_complete`` marker while the others wait on it. ``reclaim`` re-queues blocks orphaned by a crashed worker.
    Single instance (``multi=False``) is byte-identical to before."""
    if affinity:                                                       # 8-ch [CT, prev, orient(6)] AffinityMalis model
        net, cfg, lm, pm, dsj, num_in = NI.load_affinity_net(model, ckpt, device, n_orient=n_orient)
        assert num_in == 2 + n_orient, f"expected {2+n_orient}-ch affinity model, got {num_in}"
    else:
        net, cfg, lm, pm, dsj, num_in = NI.load_net(model, ckpt, device, deep_supervision=False, force_in_channels=2)
        assert num_in == 2, f"expected a 2-channel [CT, prev] model, got num_input_channels={num_in}"
    P = int(cfg.patch_size[0])
    assert list(cfg.patch_size) == [P, P, P], cfg.patch_size
    chunk = P // 2                                                     # phase boundaries (0, P/2, P, ...) all aligned
    # OVERLAP mode: each pass is a Gaussian-blended overlapping-window inference (like single-pass); passes shift
    # by S/2 so centres land on the previous pass's overlap-centres. S forced even so S/2 is integral; output tile
    # T is a multiple of the zarr chunk (P/2) so parallel tile writes never share a chunk.
    blend = bool(overlap) and overlap > 0.0
    S = Tt = None
    if blend:
        S = 2 * max(1, int(round(P * (1.0 - float(overlap)) / 2)))
        S = int(min(max(S, 2), P - 2))                                # 0 < S < P (real overlap AND real shift)
        Tt = max(P, int(nb) * (P // 2))                              # output tile per axis (chunk-aligned)
        print(f"[jacobi] OVERLAP mode: stride S={S} (overlap {1 - S / P:.2f}), tile T={Tt}, "
              f"inter-pass shift {S // 2} (centres land on prev overlap-centres)", flush=True)
    ctnorm = NI.ct_norm_params(model)
    vol = NI.open_scroll(scroll, local_vol=local_vol)
    shape = tuple(int(v) for v in vol.meta.level_shapes[0])
    vx = float(vol.meta.voxel_size_um)
    region_full = tuple(region) if region else (0, shape[0], 0, shape[1], 0, shape[2])
    print(f"[jacobi] {scroll} L0={shape} P={P} chunk={chunk} passes={passes} region={region_full} batch={batch} nb={nb}",
          flush=True)

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
    for p in range(passes):
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
        name = f"{scroll}-{tag}-p{p}"
        if not multi or leader:
            cur = NI.create_surface_zarr(cur_path, shape, chunk, vx, scroll, name, overwrite=not resume)
            if multi:
                open(os.path.join(donedir, "_created"), "w").close()   # signal followers the zarr is ready
        else:
            CQ.wait_for_path(os.path.join(donedir, "_created"), f"pass {p} zarr creation")
            cur = zarr.open_group(cur_path, mode="r+")["0"]

        if blend:
            run_pass_blend(net, ctnorm, vol, prev_arr, cur, region_full, offset, P, S, Tt, device, air=air,
                           batch=batch, donedir=(donedir if multi else None), reclaim=reclaim,
                           claim_chunk=claim_chunk, affinity=affinity,
                           readahead=readahead, prefetch_workers=prefetch_workers)
        else:
            run_pass(net, ctnorm, vol, prev_arr, cur, region_full, offset, P, device, air=air, batch=batch, nb=nb,
                     donedir=(donedir if multi else None), reclaim=reclaim, claim_chunk=claim_chunk, affinity=affinity,
                     readahead=readahead, prefetch_workers=prefetch_workers)

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

        if multi:
            nbk = (len(pass_blend_tiles(shape, P, S, offset, region_full, Tt)) if blend
                   else _n_blocks(shape, P, offset, region_full, nb))
            print(f"[jacobi] pass {p}: barrier — waiting for all {nbk} {'tiles' if blend else 'blocks'} .done", flush=True)
            CQ.wait_all_done(donedir, nbk, what=f"pass {p}")
            if leader:
                if (p in finalise_set) or (s3_prefix and p in upload_set):   # decouple: BOUND to 1 in-flight, then spawn
                    for q in sorted(k for k in fin_threads if k < p):  # (no unfinalised-buffer overwrite: a still-
                        fin_threads.pop(q).join()                      #  running finalise/upload's pass is never pruned/reused)
                    t = threading.Thread(target=_finalize_upload, name=f"finalise-p{p}"); t.start()
                    fin_threads[p] = t
                open(complete, "w").close()                            # pass L0 done → resume marker (instant)
            print(f"[jacobi] pass {p}: barrier cleared ({'leader' if leader else 'follower'} → pass {p + 1})", flush=True)
        else:
            _finalize_upload(); open(complete, "w").close()            # single instance: inline

        if not keep_buffers and p >= 1 and prev_path is not None and (leader or not multi):
            import shutil                                              # p-1's finalize already joined above (bound=1)
            shutil.rmtree(prev_path, ignore_errors=True)
            shutil.rmtree(prev_path + ".done", ignore_errors=True)
        prev_arr = zarr.open_group(cur_path, mode="r")["0"]            # next pass reads this pass's L0 (done @ barrier)
        prev_path = cur_path

    for pp, t in sorted(fin_threads.items()):                          # leader: finish the last passes' finalize+upload
        print(f"[jacobi] waiting on background finalize/upload for pass {pp}", flush=True)
        t.join()

    print(f"[jacobi] DONE -> {prev_path}", flush=True)
    return prev_path

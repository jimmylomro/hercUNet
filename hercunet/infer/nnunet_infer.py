"""Shared full-volume inference building blocks (net load, CT norm, shared-zarr buffer + pyramid, scroll open).

Reused from the proven research inference path (villa fallback net-load: build the stock network from
``plans.json`` and load ``network_weights`` — the custom loss-only trainer class need NOT be importable, so
inference works from a pinned ``nnunetv2==2.8.1`` alone). The HercUNet v0 model is 8-channel
``[CT, prev, orient(6)]`` (:func:`load_affinity_net`); a plain iterative model is 2-channel ``[CT, prev]``
(:func:`load_net`). The ``prev`` channel is NoNormalization (fed raw [0,1]); only the CT channel uses the
fingerprint CTNormalization.
"""
from __future__ import annotations

import json
import os

import numpy as np


def load_net(model_folder, ckpt_name="checkpoint_best.pth", device="cuda", deep_supervision=False,
             force_in_channels=None):
    """Build the stock network from plans.json + dataset.json and load the checkpoint weights.

    ``force_in_channels`` overrides the input-channel count. The iterative model was TRAINED with a network
    forced to 2 channels ([CT, prev]) even though its dataset.json declares 1+K candidate channels (the
    per-epoch transform collapses them) — so inference must build a 2-channel net to match the saved weights.
    Pass ``force_in_channels=2`` for the iterative model; leave None for a normal single-input model."""
    import torch
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

    plans = json.load(open(os.path.join(model_folder, "plans.json")))
    dataset_json = json.load(open(os.path.join(model_folder, "dataset.json")))
    pm = PlansManager(plans)
    ckpt = torch.load(os.path.join(model_folder, "fold_0", ckpt_name), map_location="cpu", weights_only=False)
    cfg = pm.get_configuration(ckpt["init_args"]["configuration"])
    num_in = int(force_in_channels) if force_in_channels else determine_num_input_channels(pm, cfg, dataset_json)
    lm = pm.get_label_manager(dataset_json)
    net = get_network_from_plans(
        cfg.network_arch_class_name, cfg.network_arch_init_kwargs,
        cfg.network_arch_init_kwargs_req_import, num_in, lm.num_segmentation_heads,
        allow_init=True, deep_supervision=deep_supervision)
    net.load_state_dict(ckpt["network_weights"])
    net.eval().to(device)
    return net, cfg, lm, pm, dataset_json, num_in


def load_affinity_net(model_folder, ckpt_name="checkpoint_best.pth", device="cuda",
                      configuration="3d_fullres", n_orient=6):
    """Load the SURFACE branch of an AffinityMalis (HercUNet v0) model for iterative inference.
    The trained net is ``AffinityHeadNet(base_nnunet, aff_head)`` with 8-ch input ``[CT, prev, orient(6)]``; for
    the surface-probability Jacobi passes we only need the base nnU-Net — build it at ``2 + n_orient`` channels and
    load ONLY the ``base.*`` weights (aff_head skipped). The custom checkpoint has no ``init_args``, so pass
    ``configuration`` explicitly. Same return tuple as :func:`load_net`; caller supplies the orient channels
    (recomputed from CT in-forward, so they stay consistent — see :func:`hercunet.training.affinity.ct_orientation`)."""
    import torch
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

    plans = json.load(open(os.path.join(model_folder, "plans.json")))
    dataset_json = json.load(open(os.path.join(model_folder, "dataset.json")))
    pm = PlansManager(plans)
    cfg = pm.get_configuration(configuration)
    lm = pm.get_label_manager(dataset_json)
    num_in = 2 + int(n_orient)
    net = get_network_from_plans(
        cfg.network_arch_class_name, cfg.network_arch_init_kwargs,
        cfg.network_arch_init_kwargs_req_import, num_in, lm.num_segmentation_heads,
        allow_init=True, deep_supervision=False)
    ckpt = torch.load(os.path.join(model_folder, "fold_0", ckpt_name), map_location="cpu", weights_only=False)
    sd = ckpt["network_weights"]
    base_sd = {k[len("base."):]: v for k, v in sd.items() if k.startswith("base.")}
    missing, unexpected = net.load_state_dict(base_sd, strict=False)
    missing = [m for m in missing if "aff_head" not in m]
    assert not missing and not unexpected, f"affinity base mismatch: missing={missing[:4]} unexpected={unexpected[:4]}"
    net.eval().to(device)
    print(f"[load-affinity] base nnU-Net {num_in}ch, {len(base_sd)} tensors (aff_head skipped), ep={ckpt.get('current_epoch')}",
          flush=True)
    return net, cfg, lm, pm, dataset_json, num_in


def ct_norm_params(model_folder, channel="0"):
    """CTNormalization params for a channel from the dataset fingerprint: (clip_lo, clip_hi, mean, std)."""
    fp = json.load(open(os.path.join(model_folder, "dataset_fingerprint.json")))
    ip = fp["foreground_intensity_properties_per_channel"][channel]
    return (float(ip["percentile_00_5"]), float(ip["percentile_99_5"]),
            float(ip["mean"]), float(ip["std"]))


def norm_ct(block, lo, hi, mean, std):
    """CTNormalization: clip to [lo,hi] then (x-mean)/std. Returns float32."""
    x = np.clip(np.asarray(block, np.float32), lo, hi)
    return (x - mean) / std


def s3_to_https(s3_uri, region="eu-west-1"):
    """s3://bucket/key -> https://bucket.s3.<region>.amazonaws.com/key (virtual-hosted, the form VC3D attaches)."""
    assert s3_uri.startswith("s3://"), s3_uri
    bucket, _, key = s3_uri[len("s3://"):].partition("/")
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key.rstrip('/')}"


def _s5cmd_bin():
    """Locate the s5cmd binary. PATH first, then next to the running interpreter (``sys.executable``'s dir — the
    venv's ``bin/`` where the s5cmd wheel installs its launcher). The fallback matters because calling a venv's
    entry point directly (``/path/to/venv/bin/hercunet`` under nohup / systemd, no ``activate``) does NOT put the
    venv bin on PATH, so a plain ``which`` would miss the s5cmd installed right beside us. Returns a path or None."""
    import shutil
    import sys
    p = shutil.which("s5cmd")
    if p:
        return p
    cand = os.path.join(os.path.dirname(sys.executable), "s5cmd")
    return cand if os.path.exists(cand) else None


def _count_files(local_dir):
    """Total number of files under ``local_dir`` — the upload-progress denominator (local FS metadata walk;
    a few seconds even for a ~470k-object L0)."""
    n = 0
    for _dp, _dirs, fns in os.walk(local_dir):
        n += len(fns)
    return n


def _s5cmd_stream(argv_tail, env, total=None, label="upload", poll=5.0):
    """Run ``s5cmd --json <argv_tail>`` and print a THROTTLED progress line (every ``poll`` s): completed-object
    count, percent of ``total`` when known, files/s, elapsed. Consumes the per-object JSON stream so the user gets
    real progress instead of one line per (100k+) chunk file. Raises RuntimeError on any failed object or a
    non-zero exit — a broken upload is never silent."""
    import json as _json
    import subprocess
    import time
    s5 = _s5cmd_bin()
    if s5 is None:
        raise RuntimeError("s5cmd not found — install the [infer] extra")
    proc = subprocess.Popen([s5, "--json", *argv_tail], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    done = 0
    errors = []                                                       # JSON events carrying an "error"
    noise = []                                                        # stray non-JSON lines (rare)
    t0 = last = time.time()
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            ev = _json.loads(line)
        except ValueError:
            noise.append(line)
            continue
        if ev.get("error"):
            errors.append(ev["error"])
            continue
        if ev.get("operation") == "cp":
            done += 1
        now = time.time()
        if now - last >= poll:
            rate = done / max(now - t0, 1e-9)
            pct = f" ({100 * done // total}%)" if total else ""
            print(f"[upload] {label}: {done}{f'/{total}' if total else ''} files{pct} "
                  f"{rate:.0f} files/s {now - t0:.0f}s", flush=True)
            last = now
    rc = proc.wait()
    if rc != 0 or errors:
        detail = " | ".join(str(e) for e in (errors or noise)[:3])
        raise RuntimeError(f"s5cmd {label} FAILED (exit {rc}): {detail}")
    print(f"[upload] {label}: DONE {done} files in {time.time() - t0:.0f}s", flush=True)


def upload_zarr_to_s3(local_dir, s3_uri, region="eu-west-1", workers=2048, incremental=False):
    """Upload a finished zarr dir to S3 with s5cmd (creds from env: AWS_ACCESS_KEY_ID/SECRET; AWS_REGION else
    ``region``). Streams a throttled progress line every few seconds (:func:`_s5cmd_stream`) — completed-object
    count + files/s — instead of the per-file spam, and raises on any failure. ``workers`` is set far above core
    count on purpose: each PUT is a latency-bound round-trip, so heavy oversubscription hides RTT and saturates
    bandwidth (upload is IO-bound, not CPU-bound).

    ``incremental=False`` (default) → ``cp``: uploads the whole tree (first / L0 upload). ``incremental=True`` →
    ``sync``: diffs source vs dest and sends only the new files (the pyramid re-upload after L0 — so L0's ~470k
    objects are not re-sent)."""
    if _s5cmd_bin() is None:
        raise RuntimeError("s5cmd not found — install the [infer] extra before using --s3-prefix")
    env = os.environ.copy()
    env.setdefault("AWS_REGION", region)
    src = local_dir.rstrip("/") + "/"
    dst = s3_uri.rstrip("/") + "/"
    op = "sync" if incremental else "cp"
    total = None if incremental else _count_files(local_dir)          # sync sends only new files → no denominator
    print(f"[upload] s5cmd {op} {src} -> {dst} "
          f"({f'{total} files' if total is not None else 'incremental'}, workers={workers})", flush=True)
    _s5cmd_stream(["--numworkers", str(workers), op, src, dst], env, total=total, label=op)


def upload_pyramid_levels_to_s3(local_dir, s3_uri, region="eu-west-1", workers=2048):
    """Upload ONLY the pyramid levels (1..N) + the rewritten multiscales ``.zattrs`` to S3, via targeted
    ``s5cmd cp`` — NOT a whole-tree ``sync``. Use this after ``finalize_pyramid`` when L0 is ALREADY on S3
    (the inference run uploaded it with ``cp``). L0 is ~470k tiny objects, so ``s5cmd sync`` would stall for
    minutes just diffing local-vs-remote before sending anything; finalize only ADDS levels 1..N (~35k files)
    and rewrites the root ``.zattrs``, so we push exactly those paths and nothing else. Each ``cp`` overwrites,
    so it's idempotent / re-runnable. Raises on a missing level dir or any s5cmd failure (never silent)."""
    import glob
    import subprocess
    s5 = _s5cmd_bin()
    if s5 is None:
        raise RuntimeError("s5cmd not found — install the [infer] extra before uploading")
    env = os.environ.copy()
    env.setdefault("AWS_REGION", region)
    base = local_dir.rstrip("/")
    dst = s3_uri.rstrip("/")
    levels = sorted(int(os.path.basename(d)) for d in glob.glob(os.path.join(base, "[0-9]*"))
                    if os.path.isdir(d) and os.path.basename(d).isdigit() and int(os.path.basename(d)) >= 1)
    if not levels:
        raise RuntimeError(f"no pyramid levels (>=1) under {base} — did finalize_pyramid run first?")
    zattrs = os.path.join(base, ".zattrs")
    if not os.path.exists(zattrs):
        raise RuntimeError(f"missing {zattrs} — the multiscales metadata; VC3D needs it to see the pyramid")
    print(f"[upload] levels-only: cp levels {levels} + .zattrs -> {dst}/ (workers={workers})", flush=True)
    for lv in levels:                                                   # each level dir: recursive cp (incl .zarray)
        subprocess.run([s5, "--log", "error", "--numworkers", str(workers),
                        "cp", f"{base}/{lv}/", f"{dst}/{lv}/"], check=True, env=env)
    subprocess.run([s5, "--log", "error", "--numworkers", str(workers),             # the rewritten multiscales attrs
                    "cp", zattrs, f"{dst}/.zattrs"], check=True, env=env)            # (single file) — MANDATORY, last
    print(f"[upload] levels-only DONE: {len(levels)} levels + .zattrs on S3", flush=True)


def preflight_s3(s3_prefix, region="eu-west-1"):
    """Fail fast BEFORE any inference: verify the AWS creds can actually WRITE to (and delete from) ``s3_prefix``
    with a tiny s5cmd round-trip, so a multi-hour pass never completes only to hit a permission error on the
    upload. Raises SystemExit with actionable guidance on failure (missing s5cmd, bad creds, no PutObject). A
    delete failure is a WARNING only — the write is what uploads need; it just leaves a tiny test object behind."""
    import subprocess
    import tempfile
    s5 = _s5cmd_bin()
    if s5 is None:
        raise SystemExit("hercunet infer: --s3-prefix needs the s5cmd binary (install the [infer] extra), "
                         "or drop --s3-prefix (results still save locally).")
    key = f"{s3_prefix.rstrip('/')}/.hercunet_write_test_{os.getpid()}"
    print(f"[preflight] checking S3 write access → {s3_prefix} …", flush=True)
    env = os.environ.copy()
    env.setdefault("AWS_REGION", region)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("hercunet s3 write preflight\n")
        local = f.name
    try:
        cp = subprocess.run([s5, "--log", "error", "cp", local, key],
                            env=env, capture_output=True, text=True)
        if cp.returncode != 0:
            raise SystemExit(
                f"hercunet infer: cannot WRITE to {s3_prefix} — the AWS creds lack PutObject, or the "
                f"bucket/prefix is wrong or unreachable.\ns5cmd: {cp.stderr.strip() or cp.stdout.strip()}\n"
                f"Fix the creds/bucket (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION) or drop "
                f"--s3-prefix (results still save locally, upload later).")
        rm = subprocess.run([s5, "--log", "error", "rm", key],
                            env=env, capture_output=True, text=True)
        if rm.returncode != 0:
            print(f"[preflight] WARNING: wrote OK but could not delete the test object {key} "
                  f"(DeleteObject denied?) — uploads will still work; remove it manually.\n"
                  f"           s5cmd: {rm.stderr.strip() or rm.stdout.strip()}", flush=True)
    finally:
        try:
            os.unlink(local)
        except OSError:
            pass
    print(f"[preflight] S3 write access OK → {s3_prefix}", flush=True)


def _find_scroll(be, sid):
    """The ScrollInfo whose id matches ``sid``. Exact ``scroll_id`` match first, then a tolerant substring over
    id / name / PHerc handle (mirrors ``hercunet.train.dataset._find_scroll``)."""
    key = str(sid).lower().replace(" ", "")
    scrolls = list(be.list_scrolls())
    for s in scrolls:                                                # exact id first
        if str(s.scroll_id).lower().replace(" ", "") == key:
            return s
    for s in scrolls:                                                # then tolerant substring
        for c in (s.scroll_id, s.name, s.extra.get("pherc", "")):
            if key and key in str(c).lower().replace(" ", ""):
                return s
    raise KeyError(f"scroll {sid} not found")


def open_scroll(name, local_vol=None):
    """Open a scroll's full-resolution OME-Zarr through the hercunet data layer (chunk-cached, prefetch-ready).
    Returns a volume exposing ``.meta.level_shapes`` / ``.meta.voxel_size_um`` and ``read_window(level, z0,z1,
    y0,y1, x0,x1) -> (block, origin)`` — the read primitive :func:`hercunet.data.iter_windows` drives.

    ``local_vol`` (the ``--local-vol`` flag): path to a LOCALLY-downloaded copy of the scroll's OME-Zarr (fast
    local NVMe / tmpfs). When given, open it directly and BYPASS the S3 backend — reads become local instead of a
    throttled cross-region S3 pull, turning the run from I/O-bound into GPU-bound. Decompressed chunks are
    byte-identical to S3's, so logits and the blend are byte-identical. ``parse_voxel_um`` recovers the voxel
    resolution (name token, else OME metadata) so ``voxel_size_um`` / ``level_shapes`` match the S3 path exactly.
    Only L0 need be downloaded."""
    if local_vol:
        from ..data import ZarrSegment, parse_voxel_um
        return ZarrSegment(local_vol, parse_voxel_um(local_vol))
    from ..config import Config
    from ..data import get_backend
    be = get_backend(Config.from_env())
    return be.open_scroll_volume(_find_scroll(be, name))


def create_surface_zarr(path, shape, chunk, voxel_um, scroll, name, overwrite=True):
    """Create a VC3D-friendly OME-Zarr v2 surface volume: resolution group "0" (uint8, fill 0) + root meta.json
    + multiscales attrs. Returns the level-0 array handle. ``overwrite=False`` with an existing level "0" OPENS
    it in place (resume / multi-instance) rather than clobbering — hardcoding ``mode="w"`` would wipe it, and on
    a shared/network volume the resulting rmdir of a partial pyramid level races (OSError 39)."""
    import zarr
    if not overwrite and os.path.exists(os.path.join(path, "0")):
        return zarr.open_group(path, mode="r+")["0"]                    # resume: reuse the existing surface, no wipe
    if os.path.exists(path):
        import shutil
        shutil.rmtree(path)
    Dz, Dy, Dx = (int(s) for s in shape)
    ch = int(chunk)
    root = zarr.open_group(path, mode="w")                              # zarr 2.x writes v2 natively (VC3D wants v2)
    root.create_dataset("0", shape=(Dz, Dy, Dx), chunks=(ch, ch, ch), dtype="uint8", fill_value=0)
    root.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [{"name": a, "type": "space"} for a in ("z", "y", "x")],
        "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0]}]}],
        "type": "local max"}]
    meta = {"height": Dy, "width": Dx, "slices": Dz, "min": 0.0, "max": 255.0,
            "name": name, "type": "vol", "uuid": f"{scroll}-{name}", "voxelsize": float(voxel_um),
            "format": "zarr"}
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f)
    return root["0"]


def finalize_pyramid(path, min_dim=128, max_levels=5, region=None, read_workers=24):
    """Build OME-Zarr pyramid levels 1..N from level "0" via streaming 2x max-pool (max preserves the thin
    surface at low res) and update the multiscales attrs. ``region``=(z0,z1,y0,y1,x0,x1) in L0 voxels restricts
    the iteration to that box's chunks at each level — ESSENTIAL when the canvas is the full volume but only a
    sub-region has data (else it scans millions of empty full-canvas chunks; ~15 min/level). None = full canvas."""
    import time
    import zarr
    root = zarr.open_group(path, mode="r+")
    base = root["0"]
    Z, Y, X = base.shape
    ch = base.chunks[0]
    levels = [("0", (1.0, 1.0, 1.0))]
    prev, lvl = "0", 1
    cz, cy, cx = Z, Y, X
    print(f"[finalize] {path.split('/')[-1]} — building pyramid from L0 ({Z},{Y},{X})", flush=True)
    while max(cz, cy, cx) > min_dim and lvl <= max_levels:
        src = root[prev]
        nz, ny, nx = (cz + 1) // 2, (cy + 1) // 2, (cx + 1) // 2
        if str(lvl) in root and tuple(root[str(lvl)].shape) == (nz, ny, nx):   # reuse a valid existing level IN PLACE
            dst = root[str(lvl)]                                              # — no rmdir (network volumes fail it
        else:                                                                 #   under load); a re-run overwrites its
            dst = root.create_dataset(str(lvl), shape=(nz, ny, nx),           #   chunks (deterministic, same data)
                                      chunks=(min(ch, nz), min(ch, ny), min(ch, nx)), dtype="uint8", fill_value=0)
        blk = ch
        t_lvl = time.time(); ktile = 0

        def _rng(r0, r1, n):                                            # dst-level, blk-aligned range over region
            if region is None:
                return range(0, n, blk)
            a = max(0, ((r0 >> lvl) // blk) * blk)
            b = min(n, (((r1 >> lvl) // blk) + 1) * blk)
            return range(a, b, blk)

        rz = _rng(region[0], region[1], nz) if region else range(0, nz, blk)
        ry = _rng(region[2], region[3], ny) if region else range(0, ny, blk)
        rx = _rng(region[4], region[5], nx) if region else range(0, nx, blk)
        tiles = [(oz, oy, ox) for oz in rz for oy in ry for ox in rx]
        ntiles = len(tiles)

        import threading
        _prog = {"k": 0}
        _lock = threading.Lock()

        def _pool_and_write(t):
            """Read one dst-tile's 2x source region, max-pool it, and WRITE it — all inside the worker thread.
            Reads AND writes are latency-bound network-volume round-trips; tiles are output-disjoint and
            chunk-aligned (each tile == one dst chunk), so parallel writes never race (same guarantee the
            multi-instance inference relies on). A huge CPU box just needs a high ``read_workers``."""
            oz, oy, ox = t
            iz0, iy0, ix0 = oz * 2, oy * 2, ox * 2
            iz1, iy1, ix1 = min((oz + blk) * 2, cz), min((oy + blk) * 2, cy), min((ox + blk) * 2, cx)
            sub = np.asarray(src[iz0:iz1, iy0:iy1, ix0:ix1])
            if int(sub.max()) != 0:                                  # skip all-air tiles (dst pre-filled 0)
                pz, py, px = sub.shape[0] % 2, sub.shape[1] % 2, sub.shape[2] % 2
                if pz or py or px:
                    sub = np.pad(sub, ((0, pz), (0, py), (0, px)), mode="edge")
                pooled = sub.reshape(sub.shape[0] // 2, 2, sub.shape[1] // 2, 2,
                                     sub.shape[2] // 2, 2).max(axis=(1, 3, 5)).astype(np.uint8)
                dst[oz:oz + pooled.shape[0], oy:oy + pooled.shape[1], ox:ox + pooled.shape[2]] = pooled
            with _lock:                                              # progress only (cheap); the work is lock-free
                _prog["k"] += 1
                k = _prog["k"]
                if k % 2000 == 0:
                    el = time.time() - t_lvl
                    eta = (ntiles - k) / max(k / max(el, 1e-9), 1e-9) / 60.0
                    print(f"[finalize] level {lvl}: {k}/{ntiles} tiles {el:.0f}s ETA {eta:.0f}m", flush=True)

        from concurrent.futures import ThreadPoolExecutor            # FULLY parallel: read + max-pool + write per tile
        with ThreadPoolExecutor(max_workers=read_workers) as ex:     # (numpy max-pool releases the GIL -> real cores)
            list(ex.map(_pool_and_write, tiles))                     # in-flight bounded to read_workers; nothing buffered
        ktile = _prog["k"]
        levels.append((str(lvl), (float(2 ** lvl),) * 3))
        print(f"[finalize] level {lvl} DONE shape=({nz},{ny},{nx}) {ntiles} tiles {time.time()-t_lvl:.0f}s", flush=True)
        prev, cz, cy, cx, lvl = str(lvl), nz, ny, nx, lvl + 1
    root.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [{"name": a, "type": "space"} for a in ("z", "y", "x")],
        "datasets": [{"path": p, "coordinateTransformations": [{"type": "scale", "scale": list(s)}]}
                     for p, s in levels],
        "type": "local max"}]
    print(f"[finalize] {len(levels)} levels; multiscales updated", flush=True)

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


def load_affinity_net_with_head(model_folder, ckpt_name="checkpoint_best.pth", device="cuda",
                                configuration="3d_fullres", n_orient=6):
    """Load the FULL ``AffinityHeadNet`` (base nnU-Net + the affinity head) so inference can emit BOTH the surface
    ``seg`` and the per-voxel ``aff`` field — unlike :func:`load_affinity_net`, which drops the head. Used only when
    ``hercunet infer --keep-affinity`` is set (to materialise the affinity for the band-refinement probe).

    The head is a parallel 1×1×1 branch off the last decoder feature map, so ``seg = base(x)`` is byte-identical to
    the head-less load — the surface deliverable is unchanged. ``n_aff`` (the number of affinity offsets, 9 for
    HercUNet v0) is read from the checkpoint's ``aff_head.weight`` so it can never drift from the trained head.
    Returns the same tuple as :func:`load_affinity_net` plus ``n_aff``."""
    import torch
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from ..training.affinity import AffinityHeadNet

    plans = json.load(open(os.path.join(model_folder, "plans.json")))
    dataset_json = json.load(open(os.path.join(model_folder, "dataset.json")))
    pm = PlansManager(plans)
    cfg = pm.get_configuration(configuration)
    lm = pm.get_label_manager(dataset_json)
    num_in = 2 + int(n_orient)
    base = get_network_from_plans(
        cfg.network_arch_class_name, cfg.network_arch_init_kwargs,
        cfg.network_arch_init_kwargs_req_import, num_in, lm.num_segmentation_heads,
        allow_init=True, deep_supervision=False)
    ckpt = torch.load(os.path.join(model_folder, "fold_0", ckpt_name), map_location="cpu", weights_only=False)
    sd = ckpt["network_weights"]
    if "aff_head.weight" not in sd:
        raise SystemExit("hercunet infer --keep-affinity: this checkpoint has no affinity head "
                         "(aff_head.* missing) — it is not an AffinityMalis/HercUNet model.")
    n_aff = int(sd["aff_head.weight"].shape[0])
    net = AffinityHeadNet(base, n_aff=n_aff)
    net.return_aff = True
    missing, unexpected = net.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, f"affinity full-net mismatch: missing={missing[:4]} unexpected={unexpected[:4]}"
    net.eval().to(device)
    print(f"[load-affinity+head] AffinityHeadNet {num_in}ch, n_aff={n_aff}, ep={ckpt.get('current_epoch')}", flush=True)
    return net, cfg, lm, pm, dataset_json, num_in, n_aff


def create_affinity_zarr(path, n_aff, shape, chunk, voxel_um, offsets, pass_idx, overwrite=True):
    """Create a 4-D ``(n_aff, Z, Y, X)`` uint8 zarr for the affinity field (``aff = v/255``; v=0 where no window
    covered). Chunked ``(n_aff, chunk, chunk, chunk)`` so each blend tile (chunk-aligned, full-channel) writes
    disjoint chunks — multi-instance stays race-free, same guarantee as the surface buffer. The offset table
    (channel → ``(dz,dy,dx)`` and its µm span) and voxel size are recorded in the attrs so the band solve knows
    exactly what each channel means. ``overwrite=False`` opens an existing one in place (resume/multi follower)."""
    import zarr
    if not overwrite and os.path.exists(os.path.join(path, "0")):
        return zarr.open_group(path, mode="r+")["0"]
    if os.path.exists(path):
        import shutil
        shutil.rmtree(path)
    Dz, Dy, Dx = (int(s) for s in shape)
    ch, C = int(chunk), int(n_aff)
    root = zarr.open_group(path, mode="w")
    arr = root.create_dataset("0", shape=(C, Dz, Dy, Dx), chunks=(C, ch, ch, ch), dtype="uint8", fill_value=0)
    attrs = {
        "content": "hercunet_affinity",
        "encoding": "uint8; aff = v/255 = predicted same-sheet probability of the ordered pair (u, u+offset)",
        "offsets": [[int(d) for d in o] for o in offsets],
        "offsets_um": [[round(float(d) * float(voxel_um), 3) for d in o] for o in offsets],
        "voxelsize": float(voxel_um), "pass": int(pass_idx), "n_aff": C,
    }
    root.attrs.update(attrs)
    arr.attrs.update(attrs)
    return root["0"]


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


def s3_uri_to_https(s3_uri):
    """s3://bucket/key -> https://bucket.s3.amazonaws.com/key (region-less global endpoint; S3 redirects to the
    bucket's region). Used to OPEN/stream an ``s3://`` source over plain HTTPS range reads."""
    assert s3_uri.startswith("s3://"), s3_uri
    bucket, _, key = s3_uri[len("s3://"):].partition("/")
    return f"https://{bucket}.s3.amazonaws.com/{key.rstrip('/')}"


def https_to_s3(url):
    """Virtual-hosted S3 https URL -> s3://bucket/key (inverse of :func:`s3_uri_to_https`, for feeding s5cmd).
    Handles ``bucket.s3.amazonaws.com``, ``bucket.s3.<region>.amazonaws.com`` and ``bucket.s3-<region>.…``."""
    import re
    m = re.match(r"https?://([^./]+)\.s3(?:[.-][a-z0-9-]+)?\.amazonaws\.com/(.+)", url)
    if not m:
        raise ValueError(f"not a virtual-hosted S3 https URL: {url}")
    return f"s3://{m.group(1)}/{m.group(2).rstrip('/')}"


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


def _source_kind(source):
    """Classify an infer ``source`` positional → ``('local', path)`` | ``('http', url)`` | ``('scroll', id)``.
    ``file://`` and local paths (existing, or ending ``.zarr``) are local; ``s3://`` and ``http(s)://`` are
    remote (``s3://`` mapped to its https endpoint); anything else is treated as a data-layer scroll id."""
    s = str(source)
    if s.startswith("file://"):
        return "local", s[len("file://"):]
    if s.startswith("s3://"):
        return "http", s3_uri_to_https(s)
    if s.startswith(("http://", "https://")):
        return "http", s
    if os.path.exists(s) or s.rstrip("/").endswith(".zarr"):
        return "local", s
    return "scroll", s


def source_label(source):
    """A short volume name for VC3D metadata / per-pass naming: the zarr basename (minus ``.zarr``), or the
    scroll id for the catalog form."""
    s = str(source).rstrip("/")
    if s.startswith("file://"):
        s = s[len("file://"):]
    base = s.split("/")[-1]
    return base[:-5] if base.endswith(".zarr") else base


def open_source(source):
    """Open an inference CT ``source`` → a volume exposing ``.meta.level_shapes`` / ``.meta.voxel_size_um`` and
    ``read_window(level, z0,z1,y0,y1, x0,x1) -> (block, origin)`` — the primitive :func:`hercunet.data.iter_windows`
    drives.

    ``source`` is the single positional argument of ``hercunet infer``:
      * ``file:///path/x.zarr`` or a local path / ``*.zarr`` → opened **locally** (fast NVMe/tmpfs; GPU-bound);
      * ``s3://bucket/key.zarr`` or ``https://…/x.zarr`` → opened for **streaming** HTTPS range reads (no bucket
        listing);
      * a bare scroll id (e.g. ``PHerc1447``) → resolved through the data-layer **catalog** (this lists the bucket;
        the URL forms above skip that).
    ``parse_voxel_um`` recovers the voxel size from the name token (else the OME metadata), so ``voxel_size_um`` /
    ``level_shapes`` match regardless of source — a region-synced local copy keeps the source basename so its token
    survives."""
    kind, loc = _source_kind(source)
    if kind in ("local", "http"):
        from ..data import ZarrSegment, parse_voxel_um
        return ZarrSegment(loc, parse_voxel_um(loc))
    from ..config import Config
    from ..data import get_backend
    be = get_backend(Config.from_env())
    return be.open_scroll_volume(_find_scroll(be, loc))


def _s5cmd_presync_run(lines, anon=False, workers=256, poll=5.0):
    """Feed per-chunk ``cp`` commands to ``s5cmd --json … run`` (parallel), printing a throttled progress line and
    a final summary — the same clean, continuous style as the upload logs. A missing object (404) means an all-air
    chunk was never stored; that is EXPECTED (zarr serves it as ``fill_value`` at read time), so it is counted as
    ``air-miss`` and never aborts. Needs the s5cmd binary (the ``[infer]`` extra installs it)."""
    import json as _json
    import subprocess
    import tempfile
    import time
    s5 = _s5cmd_bin()
    if s5 is None:
        raise SystemExit("--pre-sync-source needs the s5cmd binary (install the [infer] extra).")
    # Pass the command list as a FILE, not via stdin: piping tens of thousands of commands into stdin while
    # concurrently draining stdout deadlocks once both OS pipe buffers fill (s5cmd blocks writing JSON, we block
    # writing commands). ``s5cmd run <file>`` avoids the stdin pipe entirely.
    cmdf = tempfile.NamedTemporaryFile("w", suffix=".s5cmds", delete=False)
    cmdf.write("\n".join(lines) + "\n")
    cmdf.close()
    argv = [s5, "--json"] + (["--no-sign-request"] if anon else []) + ["--numworkers", str(workers), "run", cmdf.name]
    # Strip AWS_REGION so s5cmd AUTO-DETECTS the SOURCE bucket's region: it may differ from the OUTPUT bucket's
    # (e.g. the open-data bucket is us-east-1 while results upload to an eu-west-1 bucket), and a forced wrong
    # region fails every read with a 301 BucketRegionError. The upload/preflight paths set their own region.
    env = os.environ.copy()
    env.pop("AWS_REGION", None)
    env.pop("AWS_DEFAULT_REGION", None)
    proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    total, done, miss = len(lines), 0, 0
    t0 = last = time.time()
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            ev = _json.loads(line)
        except ValueError:
            continue
        if ev.get("error"):
            miss += 1
            continue
        if ev.get("operation") == "cp":
            done += 1
        now = time.time()
        if now - last >= poll:
            seen = done + miss
            print(f"[pre-sync] {seen}/{total} ({100 * seen // max(total, 1)}%) got={done} air-miss={miss} "
                  f"{seen / max(now - t0, 1e-9):.0f}/s", flush=True)
            last = now
    proc.wait()
    try:
        os.unlink(cmdf.name)
    except OSError:
        pass
    print(f"[pre-sync] fetched {done} chunks, {miss} absent (air) of {total} in {time.time() - t0:.0f}s", flush=True)


def presync_region(source, region, dst_dir, halo_chunks=2, anon=None):
    """Download ONLY the L0 chunks covering ``region`` (+ a ``halo_chunks`` halo for the patch overshoot past the
    region edges) of a remote OME-Zarr to a local copy, and return the local ``.zarr`` path. This is what
    ``--pre-sync-source`` does for an ``s3://`` / ``https://`` source: it turns a per-chunk S3-streaming run into a
    GPU-bound local-read run without fetching the whole (hundreds-of-GB) volume. zarr serves any not-downloaded
    (air) chunk as ``fill_value``, so a region-only copy is exact as long as only that region is computed.

    ``region`` = ``(z0,z1,y0,y1,x0,x1)`` in L0 voxels, or ``None`` = the whole L0 (the full-volume case — big).
    ``dst_dir`` = parent dir for the copy; the copy keeps the source zarr's **basename** so the voxel-size name
    token survives (``parse_voxel_um``). ``anon`` forces ``--no-sign-request``; ``None`` = auto (anon for the public
    open-data bucket, signed — env creds — otherwise)."""
    import json as _json
    import math
    import requests

    if source.startswith("s3://"):
        s3_uri, https = source.rstrip("/"), s3_uri_to_https(source).rstrip("/")
    elif source.startswith(("http://", "https://")):
        https, s3_uri = source.rstrip("/"), https_to_s3(source).rstrip("/")
    else:
        raise SystemExit(f"--pre-sync-source: source must be s3:// or https://, got {source!r} "
                         "(a file:// / local source is already local — nothing to sync).")
    bucket = s3_uri[len("s3://"):].split("/", 1)[0]
    if anon is None:
        anon = (bucket == "vesuvius-challenge-open-data")

    za = requests.get(f"{https}/.zattrs", timeout=30)
    datasets = _json.loads(za.text)["multiscales"][0]["datasets"] if za.ok else [{"path": "0"}]
    l0 = datasets[0]["path"]
    zarray = _json.loads(requests.get(f"{https}/{l0}/.zarray", timeout=30).text)
    cz, cy, cx = zarray["chunks"]
    Dz, Dy, Dx = zarray["shape"]
    sep = zarray.get("dimension_separator", ".")

    name = s3_uri.split("/")[-1]                                       # keep the zarr basename (voxel token!)
    dst = os.path.join(dst_dir, name)
    os.makedirs(os.path.join(dst, l0), exist_ok=True)

    def _getfile(rel):                                                # copy a small metadata file over HTTP
        r = requests.get(f"{https}/{rel}", timeout=30)
        if r.ok:
            p = os.path.join(dst, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(r.content)
    for rel in (".zattrs", ".zgroup"):
        _getfile(rel)
    for d in datasets:                                                # every level's metadata (only L0 gets chunks)
        _getfile(f"{d['path']}/.zarray")
        _getfile(f"{d['path']}/.zattrs")

    def _rng(a, b, c, n):
        n_ch = (n + c - 1) // c
        return range(max(0, a // c - halo_chunks), min(n_ch - 1, (b - 1) // c + halo_chunks) + 1)
    if region is None:
        rz, ry, rx = (range((Dz + cz - 1) // cz), range((Dy + cy - 1) // cy), range((Dx + cx - 1) // cx))
    else:
        z0, z1, y0, y1, x0, x1 = region
        rz, ry, rx = _rng(z0, z1, cz, Dz), _rng(y0, y1, cy, Dy), _rng(x0, x1, cx, Dx)
    key = (lambda i, j, k: f"{i}/{j}/{k}") if sep == "/" else (lambda i, j, k: f"{i}{sep}{j}{sep}{k}")
    lines = [f"cp {s3_uri}/{l0}/{key(i, j, k)} {os.path.join(dst, l0, key(i, j, k))}"
             for i in rz for j in ry for k in rx]
    gb = len(lines) * cz * cy * cx / 1e9                              # uint8 upper bound (air chunks won't exist)
    print(f"[pre-sync] {len(lines)} L0 chunks (±{halo_chunks} halo, ≤~{gb:.1f} GB) "
          f"{'[anon]' if anon else '[signed]'} {s3_uri} -> {dst}", flush=True)
    # Disk-based progress watcher: s5cmd buffers its --json stream for large runs, so the event-driven counter in
    # _s5cmd_presync_run can stay silent for minutes; this ticks off the files actually on disk, independently.
    import threading
    import time as _time
    l0dir = os.path.join(dst, l0)
    stop = threading.Event()
    t0 = _time.time()

    def _watch():
        while not stop.wait(10.0):
            try:
                n = sum(len(fs) for _, _, fs in os.walk(l0dir))
            except OSError:
                n = 0
            print(f"[pre-sync] ~{n}/{len(lines)} chunks on disk ({100 * n // max(len(lines), 1)}%) "
                  f"{int(_time.time() - t0)}s", flush=True)

    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()
    try:
        _s5cmd_presync_run(lines, anon=anon)
    finally:
        stop.set()
    return dst


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

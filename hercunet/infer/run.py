"""Inference orchestration — resolve the model, pick GPUs, and drive the Jacobi refiner.

Two entry points, both thin wrappers over the ONE engine in :func:`hercunet.infer.jacobi_refine.jacobi_refine`
(``multi=True`` + the shared ``claim_queue``). The only real differences are *where the donedir lives* and *who is
the global leader*:

* :func:`single_instance` — one box. Fans across its **local GPUs**: one worker process per GPU (each pinned via
  ``CUDA_VISIBLE_DEVICES``), all sharing a **local-FS** donedir, this node's rank-0 worker is the leader. With a
  single GPU it runs the engine in-process (no claim-queue overhead). You run ONE command.

* :func:`multi_instance` — many pods. You run the SAME command on each pod (unavoidable across machines), with
  ``--out`` on **shared network storage**; one pod passes ``--leader``. Each pod ALSO fans across its own local
  GPUs (same per-GPU worker spawn), so exactly one worker across all pods × GPUs is the global leader.

Per-GPU workers are separate processes (a fresh ``python -m hercunet.infer._worker`` each) so every GPU gets its
own clean CUDA context — the robust way to use multiple GPUs for inference. The parent only spawns + waits (it
never imports torch or touches a GPU), and propagates any worker's non-zero exit.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile


# --------------------------------------------------------------------------- model / gpu / region resolution
def resolve_model(model=None, model_hf=None):
    """Return a LOCAL model folder (containing ``plans.json``, ``dataset.json``, ``dataset_fingerprint.json`` and
    ``fold_0/checkpoint_*.pth``). ``model`` is a local folder; ``model_hf`` is a HuggingFace repo id to
    ``snapshot_download`` the whole model from (public — no token). Exactly one must be given."""
    if model and model_hf:
        raise SystemExit("hercunet infer: pass --model (a local folder) OR --model-hf (download), not both.")
    if model:
        if not os.path.isdir(model):
            raise SystemExit(f"hercunet infer: --model {model} is not a directory.")
        return model
    if not model_hf:
        raise SystemExit("hercunet infer: give a model — --model <folder> or --model-hf <repo> "
                         "(e.g. --model-hf jimmylomro/hercunet-v0).")
    from huggingface_hub import snapshot_download
    print(f"[infer] downloading model from HF {model_hf} …", flush=True)
    path = snapshot_download(model_hf)
    print(f"[infer] model at {path}", flush=True)
    return path


def resolve_gpus(spec):
    """Turn a ``--gpus`` spec into a list of device indices. 'all'/None → every visible GPU; '0,1,3' → those
    indices; a single int → [int]. Fails if no CUDA GPU is visible."""
    if spec is None or str(spec).lower() == "all":
        import torch
        n = torch.cuda.device_count()
        if n == 0:
            raise SystemExit("hercunet infer: no CUDA GPU visible (torch.cuda.device_count()==0).")
        return list(range(n))
    gpus = [int(g) for g in str(spec).replace(" ", "").split(",") if g != ""]
    if not gpus:
        raise SystemExit(f"hercunet infer: could not parse --gpus {spec!r}.")
    return gpus


def parse_region(spec):
    """'z0:z1,y0:y1,x0:x1' → (z0,z1,y0,y1,x0,x1); None/'' → None (whole volume)."""
    if not spec:
        return None
    try:
        parts = [p.split(":") for p in str(spec).split(",")]
        (z0, z1), (y0, y1), (x0, x1) = ((int(a), int(b)) for a, b in parts)
    except Exception:
        raise SystemExit(f"hercunet infer: --region must be 'z0:z1,y0:y1,x0:x1', got {spec!r}.")
    if not (z0 < z1 and y0 < y1 and x0 < x1):
        raise SystemExit(f"hercunet infer: --region has an empty axis: {spec!r}.")
    return (z0, z1, y0, y1, x0, x1)


# --------------------------------------------------------------------------- the shared engine driver
def _run(model_dir, gpus, multi, is_leader_node, jargs):
    """Drive the refiner. ``multi`` False → run the engine in-process (single GPU, no claim-queue). ``multi`` True →
    spawn one ``_worker`` process per GPU on the shared donedir; rank-0 is the leader iff ``is_leader_node``."""
    # Fail fast: if we'll upload, prove the creds can WRITE to the prefix NOW — never after a multi-hour pass.
    # Only the leader node uploads, so only it needs (and checks) write access.
    if jargs.get("s3_prefix") and is_leader_node:
        from .nnunet_infer import preflight_s3
        preflight_s3(jargs["s3_prefix"])
    if not multi:
        from .engine import run_inference
        return run_inference(model=model_dir, device="cuda", multi=False, leader=True, **jargs)

    # multi: one worker subprocess per GPU, each pinned to its device, all sharing the donedir under out_prefix.
    import shutil
    procs = []
    fail = []
    tmp = tempfile.mkdtemp(prefix="hercunet-infer-")
    try:
        for rank, gpu in enumerate(gpus):
            wargs = dict(jargs, model=model_dir, device="cuda", multi=True,
                         leader=bool(is_leader_node and rank == 0), rank=rank, gpu=gpu)
            argfile = os.path.join(tmp, f"worker_{rank}.json")
            with open(argfile, "w") as f:
                json.dump(wargs, f)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)                   # pin this worker to one GPU (device="cuda" → it)
            role = "LEADER" if wargs["leader"] else "worker"
            print(f"[infer] spawning {role} rank {rank} on GPU {gpu}", flush=True)
            procs.append((rank, gpu,
                          subprocess.Popen([sys.executable, "-m", "hercunet.infer._worker", argfile], env=env)))

        for rank, gpu, pr in procs:                                  # wait for all; collect failures
            rc = pr.wait()
            if rc != 0:
                fail.append((rank, gpu, rc))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if fail:
        raise SystemExit("hercunet infer: worker(s) failed: "
                         + ", ".join(f"rank {r} (GPU {g}) exit {c}" for r, g, c in fail))
    print(f"[infer] all {len(gpus)} workers finished", flush=True)


def _maybe_presync(a, region):
    """Resolve the final CT source. With ``--pre-sync-source`` on an ``s3://`` / ``https://`` source, region-download
    only the covering L0 chunks to a local copy (s5cmd) and return that local path, so the engine reads locally
    (GPU-bound). A ``file://`` / local source is already local — nothing to sync. The default sync dir sits next to
    ``--out``; for multi-instance across pods pass an explicit POD-LOCAL dir so pods don't race on one copy."""
    src = a.source
    remote = src.startswith(("s3://", "http://", "https://"))
    if a.pre_sync_source is None:
        if remote:
            print("[infer] WARNING: streaming the CT directly from a remote store. Per-chunk HTTPS reads are "
                  "latency-bound and typically dominate runtime — a single slab can take hours. For a fast, "
                  "GPU-bound run, re-invoke with --pre-sync-source to region-download the CT locally first.",
                  flush=True)
        return src
    if not remote:
        print(f"[infer] --pre-sync-source ignored: source {src!r} is already local", flush=True)
        return src
    from .nnunet_infer import presync_region
    dst_dir = (a.pre_sync_source if a.pre_sync_source != "<auto>"
               else os.path.join(os.path.dirname(os.path.abspath(a.out)) or ".", "_ctsync"))
    return presync_region(src, region, dst_dir, level=int(getattr(a, "level", 0)))


def engine_jargs(a, region, source):
    """The MODEL-AGNOSTIC ``engine.run_inference`` kwargs read off the parsed CLI args ``a`` (``source`` already
    resolved by :func:`_maybe_presync`). The caller adds the model-specific ``model_type`` / ``model_opts`` /
    ``model_import`` (all JSON-serialisable, so the per-GPU worker subprocess can carry them). Reusable by any model's
    CLI (nnU-Net, HercuViT, …) — this is what lets a new model get the full fan-out without duplicating it."""
    return dict(
        source=source, out_prefix=a.out, ckpt=a.ckpt, air=a.air, level=int(getattr(a, "level", 0)),
        region=region, keep_buffers=getattr(a, "keep_buffers", False), finalise=a.finalise,
        resume=getattr(a, "resume", False), batch=getattr(a, "batch", 4), nb=getattr(a, "nb", 4),
        s3_prefix=getattr(a, "s3_prefix", None), upload=getattr(a, "upload", "last"),
        reclaim=getattr(a, "resume", False), claim_chunk=getattr(a, "claim_chunk", 6),
        readahead=getattr(a, "readahead", 2), prefetch_workers=getattr(a, "prefetch_workers", 3),
        tta=getattr(a, "tta", "none"), tta_passes=getattr(a, "tta_passes", "all"),
    )


def _common_jargs(a, region, source):
    """nnU-Net/Jacobi jargs = the engine kwargs + the nnU-Net model (``model_type`` + ``model_opts``)."""
    return dict(engine_jargs(a, region, source),
                model_type="nnunet-jacobi",
                model_opts=dict(affinity=not a.plain, keep_affinity=a.keep_affinity, n_orient=a.n_orient,
                                passes=a.passes, overlap=a.overlap, compile=getattr(a, "compile", False)))


def run_fanout(a, model_dir, jargs, *, across_pods):
    """MODEL-AGNOSTIC fan-out. Resolve GPUs for THIS node and drive the engine: single GPU → in-process; multiple GPUs
    (or ``across_pods``) → one ``_worker`` subprocess per GPU on the shared claim-queue. ``jargs`` is the full
    ``run_inference`` kwargs (incl ``region``/``source``/``model_type``/``model_opts``/``model_import``). ``model_dir``
    is the resolved LOCAL model path passed to each worker. Used by both ``hercunet infer`` and ``hercuvit infer``."""
    gpus = resolve_gpus(a.gpus)
    is_leader = a.leader if across_pods else True
    print(f"[infer] {'multi' if across_pods else 'single'}-instance: {len(gpus)} GPU(s) {gpus}, "
          f"model={model_dir}, source={jargs.get('source')}"
          + (f", leader-node={is_leader}" if across_pods else ""), flush=True)
    _run(model_dir, gpus, multi=(across_pods or len(gpus) > 1), is_leader_node=is_leader, jargs=jargs)


def single_instance(a):
    """One box, all local GPUs (``--gpus`` to restrict). Multi-GPU → per-GPU workers on a local-FS claim-queue,
    rank-0 the leader; single GPU → the engine in-process."""
    model_dir = resolve_model(a.model, a.model_hf)
    region = parse_region(a.region)
    source = _maybe_presync(a, region)
    run_fanout(a, model_dir, _common_jargs(a, region, source), across_pods=False)


def multi_instance(a):
    """Many pods. Run this SAME command on each pod (``--out`` on shared storage); pass ``--leader`` on exactly
    one pod. Each pod fans across its own local GPUs; the leader pod's rank-0 worker is the single global leader."""
    model_dir = resolve_model(a.model, a.model_hf)
    region = parse_region(a.region)
    source = _maybe_presync(a, region)
    run_fanout(a, model_dir, _common_jargs(a, region, source), across_pods=True)

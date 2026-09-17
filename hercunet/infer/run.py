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
        from .jacobi_refine import jacobi_refine
        return jacobi_refine(model=model_dir, device="cuda", multi=False, leader=True, **jargs)

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


def _common_jargs(a, region):
    """The jacobi_refine kwargs shared by both modes, read off the parsed CLI args ``a``."""
    return dict(
        scroll=a.scroll, out_prefix=a.out, passes=a.passes, ckpt=a.ckpt, air=a.air,
        region=region, keep_buffers=a.keep_buffers, finalise=a.finalise, resume=a.resume,
        batch=a.batch, nb=a.nb, s3_prefix=a.s3_prefix, upload=a.upload, reclaim=a.reclaim, claim_chunk=a.claim_chunk,
        affinity=not a.plain, n_orient=a.n_orient, overlap=a.overlap,
        readahead=a.readahead, prefetch_workers=a.prefetch_workers, local_vol=a.local_vol,
        tta=a.tta, tta_passes=a.tta_passes,
    )


def single_instance(a):
    """One box, all local GPUs (``--gpus`` to restrict). Multi-GPU → per-GPU workers on a local-FS claim-queue,
    rank-0 the leader; single GPU → the engine in-process."""
    model_dir = resolve_model(a.model, a.model_hf)
    gpus = resolve_gpus(a.gpus)
    region = parse_region(a.region)
    print(f"[infer] single-instance: {len(gpus)} GPU(s) {gpus}, model={model_dir}", flush=True)
    _run(model_dir, gpus, multi=(len(gpus) > 1), is_leader_node=True, jargs=_common_jargs(a, region))


def multi_instance(a):
    """Many pods. Run this SAME command on each pod (``--out`` on shared storage); pass ``--leader`` on exactly
    one pod. Each pod fans across its own local GPUs; the leader pod's rank-0 worker is the single global leader."""
    model_dir = resolve_model(a.model, a.model_hf)
    gpus = resolve_gpus(a.gpus)
    region = parse_region(a.region)
    print(f"[infer] multi-instance: {len(gpus)} local GPU(s) {gpus}, leader-node={a.leader}, model={model_dir}",
          flush=True)
    _run(model_dir, gpus, multi=True, is_leader_node=a.leader, jargs=_common_jargs(a, region))

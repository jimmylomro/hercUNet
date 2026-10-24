"""Training pre-extraction — the supervised-training counterpart of ``distill.py``'s CT pre-extraction.

Concurrent in-process ``s3fs`` reads DEADLOCK on this environment (measured: sockets ESTABLISHED, 0 B/s, low CPU),
independent of who issues them — a worker DataLoader fork/spawn, hand-rolled threads, or ``iter_windows``' own pool.
So, exactly like ``hercunet train export-labels`` (and the distillation path), the training set is pre-extracted to
a LOCAL cache by **shard PROCESSES** (each its own ``s3fs``, SINGLE-THREADED reads = ``iter_windows`` workers=1),
and the trainer then reads the local cache — where a ``num_workers > 0`` DataLoader is finally safe.

Unlike distillation (CT only), training needs the labels, so each cached window stores the **native** (pre-resample)
CT (Dinovol-normalised) **+ surf ``{0,1,2}`` + instance owner**, built with the SHARED :func:`build_surface_owner` (identical
to ``export-labels``). Caching the *native* window — not the final 128³ crop — keeps the random-crop-offset + flip
augmentation fully alive: :class:`LocalHercuvitDataset` does the (cheap) resample + augment per epoch at load.

The same window quality gate as :class:`HercuvitDataset` is applied here, so the cache holds exactly the gated
training set. Entry points: :func:`_ensure_train_cache` (orchestrator, idempotent) and :func:`prep_run`
(``hercuvit train prep-shard`` — one shard process).
"""
from __future__ import annotations


def prep_train_shard(*, corpus_dir, cache_dir, nshards, shard, record="base", identity_only=True,
                     surface_tau=0.91, ignore_score=0.80, confidence_ignore=False,
                     gate_score=0.85, gate_count=2, readahead=2):
    """ONE shard of the training pre-extraction (single-threaded reads → no s3fs-concurrency deadlock). Groups this
    shard's windows BY SCROLL, opens each volume ONCE via the shared backend, builds centre-aligned read requests
    with ``_window_req``, prefetches the CT with ``iter_windows(workers=1)`` overlapped with the label build, and
    writes one native ``win_*.npz`` (``ct`` f16, ``surf`` u8, ``owner`` i32, ``voxel_um``, ``has_identity``) per
    GATED window. Reused machinery throughout — nothing bespoke."""
    import collections
    import glob
    import os

    import numpy as np

    from ...config import Config
    from ...data import get_backend, iter_windows
    from ...labels.io.bundle import load_sample_bundle
    from ...data.dataset import _find_scroll, _scroll_of, _window_passes, _window_req, build_surface_owner
    from ..data.dataset import HercuvitDataset
    from ..data.normalize import dinovol_normalise

    os.makedirs(cache_dir, exist_ok=True)
    be = get_backend(Config.from_env())
    paths = sorted(glob.glob(os.path.join(corpus_dir, "sample_*.npz")))
    if not paths:
        raise SystemExit(f"no sample_*.npz under {corpus_dir!r} — pull jimmylomro/hercunet-corpus first")
    paths = [p for i, p in enumerate(paths) if i % nshards == shard]         # this shard's windows
    by_scroll = collections.OrderedDict()
    for p in paths:
        by_scroll.setdefault(_scroll_of(p), []).append(p)

    saved = 0
    skipped = 0
    for sid, sp in by_scroll.items():
        try:
            vol = be.open_scroll_volume(_find_scroll(be, sid))
        except Exception as e:                                               # noqa: BLE001 — a dead scroll never aborts
            print(f"[trainprep {shard}] skip scroll {sid}: {type(e).__name__} {str(e)[:60]}", flush=True)
            continue
        metas = {}; reqs = []; use = []
        for p in sp:
            try:
                meta = HercuvitDataset._peek_meta(p)                         # cheap meta-only gate (no mesh decode)
            except Exception:                                                # noqa: BLE001
                continue
            per_sheet = meta.get("quality", {}).get("per_sheet", {})
            if not _window_passes(per_sheet, gate_score, gate_count):        # same window quality gate as the dataset
                skipped += 1; continue
            if identity_only and not meta.get("records", {}).get(record, {}).get("cluster_ids"):
                skipped += 1; continue
            metas[p] = meta; reqs.append(_window_req(meta, vol)); use.append(p)
        for p, (_r, blk, _o) in zip(use, iter_windows(vol, reqs, readahead=readahead, workers=1)):  # SINGLE-THREADED
            try:
                full_meta, recs = load_sample_bundle(p)                      # full bundle (meshes) for the labels
                rec = recs[record]
                surf, inst, _mag = build_surface_owner(rec, full_meta, surface_tau=surface_tau,
                                                       ignore_score=ignore_score, confidence_ignore=confidence_ignore)
                ct = np.asarray(blk)
                want = tuple(int(s) for s in full_meta["shape"])
                if tuple(ct.shape) != want:
                    print(f"[trainprep {shard}] shape {ct.shape} != {want} for {os.path.basename(p)} — skip", flush=True)
                    continue
                np.savez_compressed(
                    os.path.join(cache_dir, "win_" + os.path.basename(p)[:-4] + ".npz"),
                    ct=dinovol_normalise(ct).astype(np.float16),
                    surf=surf.astype(np.uint8),                              # {0 bg, 1 surface, 2 ignore}
                    owner=inst.astype(np.int32),                             # instance ids (0 = bg/ignore)
                    voxel_um=np.float32(full_meta["voxel_um"]),
                    has_identity=np.bool_(bool(rec.get("meshes"))),
                )
                saved += 1
                if saved <= 3 or saved % 10 == 0:                            # fine-grained liveness (slow S3 reads)
                    print(f"[trainprep {shard}/{nshards}] {saved} windows ({sid})", flush=True)
            except Exception as e:                                           # noqa: BLE001 — one bad window never aborts
                print(f"[trainprep {shard}] error on {os.path.basename(p)}: {type(e).__name__} {str(e)[:80]}", flush=True)
                continue
    print(f"[trainprep {shard}/{nshards}] DONE {saved} windows ({skipped} gated out) -> {cache_dir}", flush=True)
    return saved


def _ensure_train_cache(*, corpus_dir, cache_dir, nshards, identity_only=True):
    """Idempotently build the local training cache: if not already done, launch ``nshards`` ``train prep-shard``
    SHARD PROCESSES (each its own s3fs, single-threaded → no deadlock), wait, mark ``.done``. Reused next time.

    The cache encodes the gate + ``identity_only`` used to build it — change those and use a fresh ``--train-cache``
    dir (or delete its ``.done``), as a stale cache would be silently reused."""
    import glob
    import os
    import subprocess
    import sys

    os.makedirs(cache_dir, exist_ok=True)
    done = os.path.join(cache_dir, ".done")
    have = len(glob.glob(os.path.join(cache_dir, "win_*.npz")))
    if os.path.exists(done) and have:
        print(f"[hercuvit] train cache ready: {have} windows in {cache_dir} (cached)", flush=True)
        return cache_dir
    print(f"[hercuvit] pre-extracting train windows (CT + labels) → {cache_dir} via {nshards} shard processes "
          f"(single-threaded S3) …", flush=True)
    procs = []
    for s in range(nshards):
        cmd = [sys.executable, "-m", "hercunet.hercuvit.cli", "train", "prep-shard",
               "--corpus", corpus_dir, "--cache-dir", cache_dir, "--nshards", str(nshards), "--shard", str(s)]
        if not identity_only:
            cmd.append("--no-identity-only")
        procs.append(subprocess.Popen(cmd))
    rc = [p.wait() for p in procs]
    have = len(glob.glob(os.path.join(cache_dir, "win_*.npz")))
    if not have:
        raise SystemExit(f"train pre-extraction produced no windows (shard rc={rc})")
    open(done, "w").close()
    print(f"[hercuvit] train pre-extraction done: {have} windows (shard rc={rc})", flush=True)
    return cache_dir


def prep_run(args) -> int:
    """CLI entry for ``hercuvit train prep-shard`` — ONE shard of the training pre-extraction (the orchestrator
    launches N of these as separate processes; ``--corpus`` is the already-resolved LOCAL corpus dir)."""
    import os

    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    try:                                                     # silence the benign unauthenticated-request spam
        from huggingface_hub.utils import logging as _hflog
        _hflog.set_verbosity_error()
    except Exception:
        pass
    return prep_train_shard(corpus_dir=args.corpus, cache_dir=args.cache_dir,
                            nshards=int(args.nshards), shard=int(args.shard),
                            identity_only=not getattr(args, "no_identity_only", False))

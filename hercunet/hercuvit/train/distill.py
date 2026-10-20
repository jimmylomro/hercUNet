"""Distillation runner for the slim variants — makes ``hercuvit-tiny`` / ``-nano`` from the frozen Dinovol
teacher (design ``docs/hercuvit.md`` §8.1). Entry point ``run(args)`` (wired to ``hercuvit distill``).

The teacher is the frozen Dinovol encoder (full global attention); the student is a slim encoder (stem copied,
slim transformer in its final windowed + spatial-reduction pattern). We regress student features to the teacher
on unlabelled 128³ crops with token-cosine + Gram + cls-cosine losses (``model/distill.py``). No labels, no
trained hercuvit-small needed — so this is testable today against the released Dinovol checkpoint.

The distilled student encoder is saved so supervised training (``hercuvit train``) can warm-start from it instead
of the random fresh blocks.

``--smoke`` runs a few steps on synthetic volumes to exercise the whole machinery with no data dependency.
Real unlabelled-crop streaming from a scroll/corpus is a thin wrapper over the data layer — see the TODO in
``docs/hercuvit.md`` §13; the smoke path and the loss/optimiser machinery are complete.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from ..model.hercuvit_net import build_hercuvit
from ..model.encoder import build_and_load_backbone, DINOVOL_HF_REPO, DINOVOL_CKPT_FILE
from ..model.distill import (DistillProjection, HintProjections, distill_step_losses, hint_block_pairs,
                             hint_token_loss)


def _auto_attention_backend(module: torch.nn.Module) -> None:
    """Let torch choose the SDPA kernel (the dinovol config forces cuDNN, which lacks plans for some shapes)."""
    for m in module.modules():
        if hasattr(m, "attention_priority"):
            m.attention_priority = None


def resolve_dinovol_ckpt(spec: str) -> str:
    """A local path, or a Hugging Face repo id → cached ``teacher_305k.pt`` (reusing hercunet's HF helper)."""
    if spec and Path(spec).exists():
        return spec
    from ...train.launch import resolve_pretrained  # reuse hercunet's hf_hub_download wrapper
    return resolve_pretrained(pretrained_hf=spec or DINOVOL_HF_REPO, hf_file=DINOVOL_CKPT_FILE)


def _synthetic_crops(n: int, batch: int, device, generator):
    for _ in range(n):
        yield torch.randn(batch, 1, 128, 128, 128, device=device, generator=generator)


# --------------------------------------------------------------------------- unlabelled-crop corpus (3 modes)
def _crops_from_vol(vol, n, batch, device, *, rng, level=0, crop=128, min_material=0.25, readahead=4, workers=4):
    """Yield up to ``n`` Dinovol-normalised ``crop``³ batches of **material** tiles from an already-open volume.
    Data-layer-native (``locate_material`` + ``iter_material_tiles`` — air-skipped, prefetched). May yield fewer
    than ``n`` if the volume runs dry (the caller loops / moves on). Reused by every corpus mode."""
    import numpy as np
    from ...data.tiling import locate_material, iter_material_tiles
    from ..data.normalize import dinovol_normalise
    Z, Y, X = vol.meta.level_shapes[level]
    buf: list = []
    produced = 0
    guard = 0
    while produced < n and guard < 10 * n + 50:
        guard += 1
        z0 = int(rng.integers(0, max(1, Z - crop)))
        region = locate_material(vol, level, z0 + crop // 2)
        if region is None:
            continue
        tiles = list(iter_material_tiles(vol, level, z0, z0 + crop, region, tile=crop, halo=0,
                                         min_material=min_material, readahead=readahead, workers=workers))
        rng.shuffle(tiles)
        for t in tiles:
            blk = np.asarray(t.block)
            if blk.shape != (crop, crop, crop):          # bbox-edge tiles can be clipped — skip
                continue
            buf.append(dinovol_normalise(blk))
            if len(buf) == batch:
                yield torch.from_numpy(np.stack(buf)[:, None]).float().to(device)
                buf = []
                produced += 1
                if produced >= n:
                    return


def _scroll_crops(source, n, batch, device, *, seed=0, **kw):
    """``n`` material crops from ONE scroll/zarr (back-compat ``--source``). Opens via ``open_source``, streams
    with :func:`_crops_from_vol`."""
    import numpy as np
    from ...infer.nnunet_infer import open_source
    vol = open_source(source)
    rng = np.random.default_rng(seed)
    produced = 0
    for x in _crops_from_vol(vol, n, batch, device, rng=rng, **kw):
        yield x
        produced += 1
    if produced < n:
        raise RuntimeError(f"could not sample {n} tissue crops from {source!r} (got {produced})")


def _resolve_scrolls(spec):
    """``--scrolls`` → scroll-id list. ``None``/``all`` = the full mined-scroll set (``m7mine.SCROLLS``); else a
    comma list."""
    from ...labels.m7mine import SCROLLS
    if not spec or str(spec).strip().lower() == "all":
        return list(SCROLLS)
    return [s.strip() for s in str(spec).split(",") if s.strip()]


def _random_crops(scrolls, n, batch, device, *, seed=0, chunk=64, **kw):
    """RANDOM mode: material crops spread across MANY scrolls (round-robin, each scroll opened once + cached).
    The scalable default — the whole corpus of scrolls, unlabelled."""
    import numpy as np
    from ...infer.nnunet_infer import open_source
    rng = np.random.default_rng(seed)
    cache: dict = {}
    produced = 0
    guard = 0
    while produced < n and guard < 50 + 30 * len(scrolls):
        scr = scrolls[int(rng.integers(0, len(scrolls)))]
        if scr not in cache:
            try:
                cache[scr] = open_source(scr)
            except Exception as e:                        # a dead/missing scroll never aborts the run
                print(f"[distil] skip scroll {scr}: {type(e).__name__} {str(e)[:80]}", flush=True)
                cache[scr] = None
        if cache[scr] is None:
            guard += 1
            continue
        got = 0
        for x in _crops_from_vol(cache[scr], min(chunk, n - produced), batch, device, rng=rng, **kw):
            yield x
            produced += 1
            got += 1
            if produced >= n:
                return
        if got == 0:
            guard += 1
    if produced < n:
        raise RuntimeError(f"could not sample {n} crops across {len(scrolls)} scrolls (got {produced})")


def _herculabels_crops(corpus_dir, n, batch, device, *, seed=0, readahead=6, workers=6):
    """HERCULABELS mode — CT streamed with hercunet's PROVEN pattern (``train/dataset.py`` export-labels): group
    the windows BY SCROLL, open each volume ONCE via the shared backend (catalog built once), build centre-aligned
    read requests with ``_window_req``, and ``iter_windows(vol, reqs)`` to prefetch the reads overlapped with the
    distill compute. No per-window ``open_source``, no hand-rolled threads — all reused machinery. CT-only."""
    import collections, glob, os
    import numpy as np
    from ...config import Config
    from ...data import get_backend, iter_windows
    from ...train.dataset import _find_scroll, _window_req, _scroll_of
    from ..data.dataset import HercuvitDataset, resample_to_crop, TARGET_VOXEL_UM, CROP
    from ..data.normalize import dinovol_normalise
    if not corpus_dir:
        raise SystemExit("hercuvit distil --corpus-mode herculabels needs --corpus <dir> (or --corpus-hf).")
    be = get_backend(Config.from_env())
    paths = sorted(glob.glob(os.path.join(corpus_dir, "sample_*.npz")))
    if not paths:
        raise SystemExit(f"no sample_*.npz under {corpus_dir!r} — pull jimmylomro/hercunet-corpus first")
    by_scroll = collections.OrderedDict()
    for p in paths:
        by_scroll.setdefault(_scroll_of(p), []).append(p)
    rng = np.random.default_rng(seed)
    scrolls = list(by_scroll)
    print(f"[distil] herculabels: {len(paths)} windows / {len(scrolls)} scrolls — hercunet iter_windows prefetch "
          f"(readahead {readahead})", flush=True)
    produced = 0
    buf: list = []
    while produced < n:
        rng.shuffle(scrolls)
        for sid in scrolls:
            try:
                vol = be.open_scroll_volume(_find_scroll(be, sid))
            except Exception as e:                               # a missing/dead scroll never aborts the run
                print(f"[distil] skip scroll {sid}: {type(e).__name__} {str(e)[:60]}", flush=True)
                continue
            sp = list(by_scroll[sid]); rng.shuffle(sp)
            metas = {}; reqs = []; use = []
            for p in sp:
                try:
                    meta = HercuvitDataset._peek_meta(p)         # cheap meta-only read (no mesh decode)
                except Exception:
                    continue
                metas[p] = meta; reqs.append(_window_req(meta, vol)); use.append(p)
            for p, (_r, blk, _o) in zip(use, iter_windows(vol, reqs, readahead=readahead, workers=workers)):
                meta = metas[p]
                ct, _ = resample_to_crop(dinovol_normalise(np.asarray(blk)), {}, float(meta["voxel_um"]),
                                         target_voxel_um=TARGET_VOXEL_UM, crop=CROP, rng=rng)
                buf.append(torch.from_numpy(np.ascontiguousarray(ct)).float()[None])
                if len(buf) == batch:
                    yield torch.stack(buf).to(device)
                    buf = []
                    produced += 1
                    if produced >= n:
                        return


def _m7_crops(manifest, n, batch, device, *, seed=0, crop=128):
    """M7 mode: 128³ CT crops centred on the approved m7 compressed-sheet regions (reuses the scout manifest —
    no re-mining). Hard regions where m7 beats the mesh pipeline."""
    import json
    import numpy as np
    from ..data.normalize import dinovol_normalise
    from ...data import ZarrSegment
    if not manifest:
        raise SystemExit("hercuvit distil --corpus-mode m7 needs --manifest <scout manifest.json>.")
    with open(manifest) as f:
        regions = json.load(f)
    rng = np.random.default_rng(seed)
    segs: dict = {}
    h = crop // 2
    buf: list = []
    produced = 0
    order = rng.permutation(len(regions)).tolist()
    i = 0
    guard = 0
    while produced < n and guard < 10 * n + 50:
        if i >= len(order):
            order = rng.permutation(len(regions)).tolist()
            i = 0
        r = regions[order[i]]
        i += 1
        guard += 1
        z, y, x = r.get("center_l0") or [o + h for o in r["origin_l0"]]
        url = r["vol"]
        if url not in segs:
            try:
                segs[url] = ZarrSegment(url, None)
            except Exception as e:
                print(f"[distil] skip m7 region {r.get('scroll')}: {type(e).__name__} {str(e)[:80]}", flush=True)
                segs[url] = None
        if segs[url] is None:
            continue
        try:
            blk = np.asarray(segs[url].read_window(0, z - h, z + h, y - h, y + h, x - h, x + h)[0])
        except Exception:
            continue
        if blk.shape != (crop, crop, crop):
            continue
        buf.append(torch.from_numpy(dinovol_normalise(blk))[None])    # (1,128,128,128)
        if len(buf) == batch:
            yield torch.stack(buf).float().to(device)
            buf = []
            produced += 1
    if produced < n:
        raise RuntimeError(f"could not sample {n} crops from m7 manifest {manifest!r} (got {produced})")


def _labels_crops(corpus_dir, n, batch, device, *, manifest=None, seed=0):
    """Combined LABELS mode: round-robin CT crops from the herculabels windows AND the m7-mined regions — the two
    label-derived sources of the ``jimmylomro/hercunet-corpus`` (``sample_*.npz`` + the m7 manifest under
    ``corpus_index/``). Both are raw CT → ``dinovol_normalise`` (consistent). Falls back to herculabels-only when
    no m7 manifest is found."""
    import glob, os
    if not corpus_dir:
        raise SystemExit("hercuvit distil --corpus-mode labels needs --corpus <dir> (hercunet-corpus) or --corpus-hf.")
    man = manifest
    if man is None:                                   # auto-find the m7 manifest in the corpus layout
        hits = (glob.glob(os.path.join(corpus_dir, "corpus_index", "*.json"))
                + glob.glob(os.path.join(corpus_dir, "**", "*manifest*.json"), recursive=True))
        man = hits[0] if hits else None
    gens = [_herculabels_crops(corpus_dir, n, batch, device, seed=seed)]
    if man:
        print(f"[distil] labels mode: herculabels + m7 ({man})", flush=True)
        gens.append(_m7_crops(man, n, batch, device, seed=seed + 1))
    else:
        print("[distil] labels mode: no m7 manifest found — herculabels only", flush=True)
    produced = 0
    i = 0
    while produced < n and gens:
        g = gens[i % len(gens)]
        try:
            yield next(g)
            produced += 1
            i += 1
        except StopIteration:
            gens.pop(i % len(gens))


# --------------------------------------------------------------------------- sharded CT pre-extraction (the fix)
# Concurrent s3fs reads DEADLOCK in-process on this environment (measured: sockets open, 0 B/s), so — exactly like
# hercunet's export-labels — we parallelise S3 across SHARD PROCESSES (each its own s3fs, SINGLE-THREADED reads =
# iter_windows workers=1), write Dinovol-normalised 128³ crops to a LOCAL cache, then distil on local (fast, safe).

def prep_shard(*, corpus_dir, manifest, mode, crops_out, nshards, shard, seed=0):
    """ONE shard of the CT pre-extraction (single-threaded reads → no s3fs-concurrency deadlock). Writes this
    shard's Dinovol-normalised 128³ crops as .npy into ``crops_out``. Reused machinery: the backend, ``_window_req``,
    ``iter_windows`` (workers=1), ``resample_to_crop``, ``dinovol_normalise``; m7 via the manifest's direct URLs."""
    import os, glob, collections, json
    import numpy as np
    from ..data.dataset import HercuvitDataset, resample_to_crop, TARGET_VOXEL_UM, CROP
    from ..data.normalize import dinovol_normalise
    os.makedirs(crops_out, exist_ok=True)
    saved = 0

    def _save(ct, tag):
        np.save(os.path.join(crops_out, f"{tag}.npy"), np.ascontiguousarray(ct).astype(np.float16))

    if mode in ("herculabels", "labels") and corpus_dir:
        from ...config import Config
        from ...data import get_backend, iter_windows
        from ...train.dataset import _find_scroll, _window_req, _scroll_of
        be = get_backend(Config.from_env())
        paths = sorted(glob.glob(os.path.join(corpus_dir, "sample_*.npz")))
        paths = [p for i, p in enumerate(paths) if i % nshards == shard]      # this shard's windows
        by_scroll = collections.OrderedDict()
        for p in paths:
            by_scroll.setdefault(_scroll_of(p), []).append(p)
        for sid, sp in by_scroll.items():
            try:
                vol = be.open_scroll_volume(_find_scroll(be, sid))
            except Exception as e:                                           # noqa: BLE001
                print(f"[prep {shard}] skip scroll {sid}: {type(e).__name__} {str(e)[:60]}", flush=True); continue
            metas = {}; reqs = []; use = []
            for p in sp:
                try:
                    meta = HercuvitDataset._peek_meta(p)
                except Exception:
                    continue
                metas[p] = meta; reqs.append(_window_req(meta, vol)); use.append(p)
            for p, (_r, blk, _o) in zip(use, iter_windows(vol, reqs, readahead=1, workers=1)):   # SINGLE-THREADED
                meta = metas[p]
                ct, _ = resample_to_crop(dinovol_normalise(np.asarray(blk)), {}, float(meta["voxel_um"]),
                                         target_voxel_um=TARGET_VOXEL_UM, crop=CROP, rng=None)
                _save(ct, "h_" + os.path.basename(p)[:-4]); saved += 1
                if saved <= 3 or saved % 10 == 0:                    # fine-grained liveness (slow cross-region reads)
                    print(f"[prep {shard}/{nshards}] {saved} crops ({sid})", flush=True)

    if mode in ("m7", "labels") and manifest:
        from ...data import ZarrSegment
        regions = json.load(open(manifest))
        regions = [r for i, r in enumerate(regions) if i % nshards == shard]
        segs = {}; h = CROP // 2
        for r in regions:
            z, y, x = r.get("center_l0") or [o + h for o in r["origin_l0"]]
            url = r["vol"]
            if url not in segs:
                try:
                    segs[url] = ZarrSegment(url, None)
                except Exception:
                    segs[url] = None
            if segs[url] is None:
                continue
            try:
                blk = np.asarray(segs[url].read_window(0, z - h, z + h, y - h, y + h, x - h, x + h)[0])
            except Exception:
                continue
            if blk.shape != (CROP, CROP, CROP):
                continue
            _save(dinovol_normalise(blk), f"m7_{r.get('scroll')}_{z}_{y}_{x}"); saved += 1
    print(f"[prep {shard}/{nshards}] DONE {saved} crops -> {crops_out}", flush=True)
    return saved


def _ensure_crop_cache(*, corpus_dir, manifest, mode, crops_dir, nshards):
    """Idempotently build the local crop cache: if not already done, launch ``nshards`` ``distill-prep`` SHARD
    PROCESSES (each its own s3fs, single-threaded → no deadlock), wait, mark ``.done``. Reused next time."""
    import os, glob, sys, subprocess
    os.makedirs(crops_dir, exist_ok=True)
    done = os.path.join(crops_dir, ".done")
    have = len(glob.glob(os.path.join(crops_dir, "*.npy")))
    if os.path.exists(done) and have:
        print(f"[distil] crop cache ready: {have} crops in {crops_dir} (cached)", flush=True)
        return crops_dir
    print(f"[distil] pre-extracting CT crops → {crops_dir} via {nshards} shard processes (single-threaded S3) …", flush=True)
    procs = []
    for s in range(nshards):
        cmd = [sys.executable, "-m", "hercunet.hercuvit.cli", "distill-prep", "--crops-out", crops_dir,
               "--corpus-mode", mode, "--nshards", str(nshards), "--shard", str(s)]
        if corpus_dir:
            cmd += ["--corpus", corpus_dir]
        if manifest:
            cmd += ["--manifest", manifest]
        procs.append(subprocess.Popen(cmd))
    rc = [p.wait() for p in procs]
    have = len(glob.glob(os.path.join(crops_dir, "*.npy")))
    if not have:
        raise SystemExit(f"pre-extraction produced no crops (shard rc={rc})")
    open(done, "w").close()
    print(f"[distil] pre-extraction done: {have} crops (shard rc={rc})", flush=True)
    return crops_dir


def _local_crops(crops_dir, n, batch, device, *, seed=0, augment=True):
    """Stream Dinovol-normalised 128³ crops from the LOCAL cache (fast, concurrent-safe — no s3fs). Cycles the
    cache with a shuffle + geometric augmentation for variety across the long run."""
    import glob, os
    import numpy as np
    from ..data.dataset import geometric_augment
    files = sorted(glob.glob(os.path.join(crops_dir, "*.npy")))
    if not files:
        raise SystemExit(f"no crops in {crops_dir}")
    print(f"[distil] local crop cache: {len(files)} crops (augment={augment})", flush=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files)).tolist(); i = 0
    produced = 0; buf: list = []
    while produced < n:
        if i >= len(order):
            order = rng.permutation(len(files)).tolist(); i = 0
        ct = np.load(files[order[i]]).astype(np.float32); i += 1
        if augment:
            ct, _ = geometric_augment(ct, {}, rng)
        buf.append(torch.from_numpy(np.ascontiguousarray(ct)).float()[None])
        if len(buf) == batch:
            yield torch.stack(buf).to(device)
            buf = []
            produced += 1


def _corpus_crops(mode, n, batch, device, *, scrolls=None, corpus=None, manifest=None, source=None, seed=0, **kw):
    """Mode-switched unlabelled-crop stream for distillation. ``random`` (many scrolls) | ``herculabels`` (corpus
    CT) | ``m7`` (manifest regions) | ``labels`` (herculabels + m7) | ``scroll`` (one ``--source``). All yield
    Dinovol-normalised 128³ batches."""
    if mode == "random":
        yield from _random_crops(_resolve_scrolls(scrolls), n, batch, device, seed=seed, **kw)
    elif mode == "herculabels":
        yield from _herculabels_crops(corpus, n, batch, device, seed=seed)
    elif mode == "m7":
        yield from _m7_crops(manifest, n, batch, device, seed=seed)
    elif mode == "labels":
        yield from _labels_crops(corpus, n, batch, device, manifest=manifest, seed=seed)
    elif mode == "scroll":
        yield from _scroll_crops(source, n, batch, device, seed=seed, **kw)
    else:
        raise SystemExit(f"hercuvit distil: unknown --corpus-mode {mode!r} (random|herculabels|m7|labels|scroll).")


def _save_distilled(out, student, projection, variant, student_net, teacher_ckpt, model_cfg, arch=None):
    """Save the SELF-CONTAINED distilled encoder: enough to rebuild the exact windowed variant and load the
    trained weights, with NO raw Dinovol needed. ``build_hercuvit`` detects the ``student_backbone`` key, rebuilds
    from ``model_cfg`` + ``variant_cfg`` (the retrofit keys), and loads — so distil / train / infer share ONE
    encoder, selected by pointing ``--encoder`` at this file (the variant is read back from it). ``arch`` is the
    recipe-derived encoder arch, stored for reference (the rebuild uses ``model_cfg`` + ``variant_cfg``)."""
    from pathlib import Path as _P
    _P(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"student_backbone": student.state_dict(), "projection": projection.state_dict(),
         "variant": variant, "model_cfg": dict(model_cfg),
         "variant_cfg": getattr(student_net, "_hercuvit_variant_cfg", {}),
         "arch": dict(arch or {}), "teacher_ckpt": teacher_ckpt},
        out,
    )


def distill(
    *,
    teacher_ckpt: str,
    student_variant: str = "small",
    out: str | None = None,
    steps: int = 2000,
    lr: float = 2e-4,
    token_weight: float = 1.0,
    gram_weight: float = 1.0,
    cls_weight: float = 0.5,
    hint_weight: float = 0.5,
    hint_fractions: tuple = (0.33, 0.66),
    gram_n_sample: int = 512,
    batch: int = 1,
    device: str = "cuda",
    smoke: bool = False,
    corpus_mode: str = "random",
    scrolls: str | None = None,
    corpus: str | None = None,
    manifest: str | None = None,
    source: str | None = None,
    checkpoint_every: int = 0,
    warmup: int = 0,
    lr_schedule: str = "cosine",
    weight_decay: float = 0.05,
    prep_shards: int = 8,
    crops_dir: str | None = None,
    log_every: int = 50,
    seed: int = 0,
) -> dict:
    """Distil the frozen global Dinovol teacher into a windowed **variant** (``small`` | ``tiny`` | ``nano``) on
    unlabelled CT crops, so the windowed encoder recovers the teacher's token identity. Saves the student encoder
    (reloadable by pointing ``--encoder`` at the saved file — SAME encoder for train + infer). Returns a summary."""
    import time, math
    from .recipe import VARIANTS, variant_recipe
    if student_variant not in VARIANTS:
        raise ValueError(f"unknown variant {student_variant!r}; expected one of {sorted(VARIANTS)}")
    # the variant's arch comes from its shipped recipe (recipes/<variant>.yml) — the yaml IS the definition.
    student_arch = variant_recipe(student_variant).encoder_arch()
    ckpt = resolve_dinovol_ckpt(teacher_ckpt)

    teacher, t_cfg, _, miss, unexp = build_and_load_backbone(ckpt, device=device, return_embed_skips=False)
    if miss or unexp:
        raise RuntimeError(f"teacher load not clean: missing={miss[:5]} unexpected={unexp[:5]}")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    _auto_attention_backend(teacher)

    # build the slim student FROM SCRATCH off the teacher, per the recipe's encoder arch (load_matching transfers the
    # shape-compatible teacher weights, the rest is distilled).
    student_net = build_hercuvit(ckpt, arch_override=(student_arch or None), device=device,
                                 with_instance=False, deep_supervision=False)
    student = student_net.backbone
    _auto_attention_backend(student)
    student.train()
    t_dim = int(t_cfg["embed_dim"])
    s_dim = int(student_arch.get("embed_dim", t_dim))   # small = no embed_dim override → teacher width (identity proj)
    projection = DistillProjection(s_dim, t_dim).to(device)

    # full self-contained cfg saved in the encoder (teacher cfg + the recipe's Eva overrides) so the ckpt rebuilds
    # the exact windowed variant with no raw Dinovol needed.
    model_cfg = dict(t_cfg)
    for k in ("embed_dim", "depth", "num_heads", "mlp_ratio"):
        if k in student_arch:
            model_cfg[k] = student_arch[k]

    # intermediate-block (hint) distillation: depth-aligned teacher↔student blocks (single-scale ViT → by depth, not
    # resolution; docs §13.7). Forward hooks capture each hinted block's output; a per-hint projection matches width.
    hint_pairs = hint_block_pairs(len(teacher.blocks), len(student.blocks), hint_fractions) if hint_weight > 0 else []
    hint_proj = HintProjections(len(hint_pairs), s_dim, t_dim).to(device)
    _t_store, _s_store, _hooks = {}, {}, []

    def _mk_hook(store, rank):
        def _h(_m, _i, out):
            store[rank] = out[0] if isinstance(out, tuple) else out
        return _h
    for rank, (ti, si) in enumerate(hint_pairs):
        _hooks.append(teacher.blocks[ti].register_forward_hook(_mk_hook(_t_store, rank)))
        _hooks.append(student.blocks[si].register_forward_hook(_mk_hook(_s_store, rank)))
    if hint_pairs:
        print(f"[distil] hint layers (teacher→student block): {hint_pairs}  w={hint_weight}", flush=True)

    params = ([p for p in student.parameters() if p.requires_grad]
              + list(projection.parameters()) + list(hint_proj.parameters()))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def _lr_at(step):                                   # linear warmup → cosine decay to 0 (serious runs), or constant
        if warmup and step <= warmup:
            return lr * step / max(1, warmup)
        if lr_schedule == "cosine":
            prog = min(1.0, max(0.0, (step - warmup) / max(1, steps - warmup)))
            return 0.5 * lr * (1.0 + math.cos(math.pi * prog))
        return lr

    gen = torch.Generator(device=device).manual_seed(seed)
    amp = device.startswith("cuda") if isinstance(device, str) else False

    import os
    use_synthetic = smoke and not (source or corpus or manifest or scrolls)
    if use_synthetic:
        print("[distil] smoke: SYNTHETIC crops (no data source) — exercises the machinery only.", flush=True)
        src = _synthetic_crops(steps, batch, device, gen)
    else:
        mode = "scroll" if source else corpus_mode
        print(f"[distil] {student_variant} <- teacher | mode={mode} | steps={steps} batch={batch} lr={lr}", flush=True)
        if mode in ("herculabels", "m7", "labels"):
            # pre-extract to a LOCAL crop cache via shard processes (single-threaded S3 reads), then distil on it
            man = manifest
            if man is None and mode in ("m7", "labels") and corpus:          # auto-find the m7 manifest in the corpus
                import glob
                hits = (glob.glob(os.path.join(corpus, "corpus_index", "*.json"))
                        + glob.glob(os.path.join(corpus, "**", "*manifest*.json"), recursive=True))
                man = hits[0] if hits else None
            cdir = crops_dir or os.path.join(os.path.dirname(out) if out else ".", f"distil_crops_{mode}")
            _ensure_crop_cache(corpus_dir=corpus, manifest=man, mode=mode, crops_dir=cdir, nshards=prep_shards)
            src = _local_crops(cdir, steps, batch, device, seed=seed)
        else:
            src = _corpus_crops(mode, steps, batch, device, scrolls=scrolls, corpus=corpus, manifest=manifest,
                                source=source, seed=seed)

    last: dict = {}
    best_cos = -1.0
    t0 = time.time()
    seen = 0
    for step, x in enumerate(src, start=1):
        cur_lr = _lr_at(step)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            t_feats = teacher.forward_features(x, view_kind="global")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            s_feats = student.forward_features(x, view_kind="global")
            loss, terms = distill_step_losses(
                s_feats, t_feats, projection,
                token_weight=token_weight, gram_weight=gram_weight, cls_weight=cls_weight,
                gram_n_sample=gram_n_sample,
            )
            if hint_pairs:                                              # intermediate-block token-cosine hints
                l_hint = hint_token_loss([_s_store[r] for r in range(len(hint_pairs))],
                                         [_t_store[r] for r in range(len(hint_pairs))],
                                         hint_proj, int(teacher.num_prefix_tokens))
                loss = loss + hint_weight * l_hint
                terms["hint"] = float(l_hint.detach())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        last = terms
        seen += batch
        cos = 1.0 - float(terms.get("token", 1.0))              # token-cosine SIMILARITY (↑ = closer to teacher)
        best_cos = max(best_cos, cos)
        if step % log_every == 0 or step == 1 or step == steps:
            dt = time.time() - t0
            cps = seen / max(dt, 1e-6)
            eta_m = (steps - step) * (dt / max(step, 1)) / 60.0
            msg = " ".join(f"{k}={v:.4f}" for k, v in terms.items())
            print(f"[distil {student_variant} {step}/{steps}] {msg} cos={cos:.4f} "
                  f"| lr {cur_lr:.1e} {cps:.1f} crop/s ETA {eta_m:.0f}m", flush=True)
        if checkpoint_every and out and step % checkpoint_every == 0:
            _save_distilled(out, student, projection, student_variant, student_net, ckpt, model_cfg, arch=student_arch)
            print(f"[distil] checkpoint @ step {step} (cos {cos:.4f}, best {best_cos:.4f}) -> {out}", flush=True)

    for h in _hooks:                                            # drop the hint forward hooks
        h.remove()
    if out:
        _save_distilled(out, student, projection, student_variant, student_net, ckpt, model_cfg, arch=student_arch)
    return {"variant": student_variant, "steps": steps, "final_terms": last, "best_cos": best_cos,
            "final_cos": 1.0 - float(last.get("token", 1.0)), "out": out}


def run(args) -> dict:
    """CLI entry (``hercuvit distill``). Distils one variant, or ``--student all`` → small + tiny + nano (each
    saved under ``--out`` as ``<variant>.pt``). Missing/None attrs fall back to defaults (the recipe values)."""
    import os
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)       # deprecated → drop it (silences the FutureWarning)
    try:                                                     # silence the benign unauthenticated-request spam
        from huggingface_hub.utils import logging as _hflog
        _hflog.set_verbosity_error()
    except Exception:
        pass
    from .chain import resolve_corpus
    def g(name, default):
        v = getattr(args, name.replace("-", "_"), default)
        return default if v is None else v
    smoke = bool(g("smoke", False))
    student = g("student", "small")
    mode = g("corpus_mode", "random")
    corpus_dir = None
    if getattr(args, "corpus", None) or getattr(args, "corpus_hf", None):       # pull (or reuse) the corpus IN the CLI
        # distillation needs only the CT-bearing bits of the corpus — NOT the heavy m7_corpus/ tifs.
        pats = {"herculabels": ["sample_*.npz"], "labels": ["sample_*.npz", "corpus_index/*"],
                "m7": ["corpus_index/*"]}.get(mode)
        corpus_dir = resolve_corpus(getattr(args, "corpus", None), getattr(args, "corpus_hf", None), allow_patterns=pats)
    out = g("out", None)
    common = dict(
        teacher_ckpt=g("teacher", DINOVOL_HF_REPO),
        steps=int(g("steps", 8 if smoke else 2000)),   # --smoke → a few steps unless an explicit --steps
        lr=float(g("distill_lr", 2e-4)),
        token_weight=float(g("token_weight", 1.0)),
        gram_weight=float(g("gram_weight", 1.0)),
        cls_weight=float(g("cls_weight", 0.5)),
        hint_weight=float(g("hint_weight", 0.5)),
        hint_fractions=(tuple(float(f) for f in str(g("hint_fractions", "0.33,0.66")).split(",") if f.strip())),
        batch=int(g("batch", 1)),
        device=g("device", "cuda"),
        smoke=smoke,
        corpus_mode=g("corpus_mode", "random"),
        scrolls=getattr(args, "scrolls", None),
        corpus=corpus_dir,
        manifest=getattr(args, "manifest", None),
        source=g("source", None),
        checkpoint_every=int(g("checkpoint_every", 0)),
        warmup=int(g("warmup", 0)),
        lr_schedule=g("lr_schedule", "cosine"),
        prep_shards=int(g("prep_shards", 8)),
    )
    variants = ["small", "tiny", "nano"] if student == "all" else [student]
    results = []
    for v in variants:
        vout = out
        if out and len(variants) > 1:                 # --student all: out is a dir; one encoder file per variant
            os.makedirs(out, exist_ok=True)
            vout = os.path.join(out, f"{v}.pt")
        print(f"[distil] ===== variant {v} -> {vout} =====", flush=True)
        results.append(distill(student_variant=v, out=vout, **common))
    return results[0] if len(results) == 1 else {"variants": results}


def prep_run(args) -> int:
    """CLI entry for ``hercuvit distill-prep`` — ONE shard of the CT pre-extraction (the orchestrator launches N of
    these as separate processes; ``--corpus`` is the already-resolved LOCAL corpus dir)."""
    import os
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    try:
        from huggingface_hub.utils import logging as _hflog
        _hflog.set_verbosity_error()
    except Exception:
        pass
    return prep_shard(corpus_dir=getattr(args, "corpus", None), manifest=getattr(args, "manifest", None),
                      mode=getattr(args, "corpus_mode", "labels"), crops_out=args.crops_out,
                      nshards=int(args.nshards), shard=int(args.shard))

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

from ..model.hercuvit_net import build_variant, VARIANT_ARCH
from ..model.encoder import build_and_load_backbone, DINOVOL_HF_REPO, DINOVOL_CKPT_FILE
from ..model.distill import DistillProjection, distill_step_losses


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


def _herculabels_crops(corpus_dir, n, batch, device, *, seed=0):
    """HERCULABELS mode: stream the CT of the herculabels windows (reuses :class:`HercuvitDataset`; CT only,
    labels dropped). Curated sheet-bearing crops, already Dinovol-normalised by the dataset."""
    import numpy as np
    from ..data.dataset import HercuvitDataset
    if not corpus_dir:
        raise SystemExit("hercuvit distil --corpus-mode herculabels needs --corpus <dir> (or --corpus-hf).")
    ds = HercuvitDataset(corpus_dir, augment=True, identity_only=False)   # all windows; augment = free variety
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ds)).tolist()
    buf: list = []
    produced = 0
    i = 0
    while produced < n:
        if i >= len(order):
            order = rng.permutation(len(ds)).tolist()
            i = 0
        buf.append(ds[order[i]]["ct"])                   # (1,128,128,128) Dinovol-normalised
        i += 1
        if len(buf) == batch:
            yield torch.stack(buf).to(device)
            buf = []
            produced += 1


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


def _corpus_crops(mode, n, batch, device, *, scrolls=None, corpus=None, manifest=None, source=None, seed=0, **kw):
    """Mode-switched unlabelled-crop stream for distillation. ``random`` (many scrolls) | ``herculabels`` (corpus
    CT) | ``m7`` (manifest regions) | ``scroll`` (one ``--source``). All yield Dinovol-normalised 128³ batches."""
    if mode == "random":
        yield from _random_crops(_resolve_scrolls(scrolls), n, batch, device, seed=seed, **kw)
    elif mode == "herculabels":
        yield from _herculabels_crops(corpus, n, batch, device, seed=seed)
    elif mode == "m7":
        yield from _m7_crops(manifest, n, batch, device, seed=seed)
    elif mode == "scroll":
        yield from _scroll_crops(source, n, batch, device, seed=seed, **kw)
    else:
        raise SystemExit(f"hercuvit distil: unknown --corpus-mode {mode!r} (random|herculabels|m7|scroll).")


def _save_distilled(out, student, projection, variant, student_net, teacher_ckpt):
    """Save the distilled ENCODER in the form ``build_hercuvit(distilled_encoder=…)`` reloads — so distil, train
    and infer all share one encoder (keys = the retrofitted backbone's ``state_dict``)."""
    from pathlib import Path as _P
    _P(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"student_backbone": student.state_dict(), "projection": projection.state_dict(),
         "variant": variant, "variant_cfg": getattr(student_net, "_hercuvit_variant_cfg", {}),
         "arch": VARIANT_ARCH.get(variant, {}), "teacher_ckpt": teacher_ckpt},
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
    log_every: int = 50,
    seed: int = 0,
) -> dict:
    """Distil the frozen global Dinovol teacher into a windowed **variant** (``small`` | ``tiny`` | ``nano``) on
    unlabelled CT crops, so the windowed encoder recovers the teacher's token identity. Saves the student encoder
    (reloadable by ``build_hercuvit(distilled_encoder=…)`` — SAME encoder for train + infer). Returns a summary."""
    import time
    if student_variant not in VARIANT_ARCH:
        raise ValueError(f"unknown variant {student_variant!r}; expected one of {sorted(VARIANT_ARCH)}")
    ckpt = resolve_dinovol_ckpt(teacher_ckpt)

    teacher, t_cfg, _, miss, unexp = build_and_load_backbone(ckpt, device=device, return_embed_skips=False)
    if miss or unexp:
        raise RuntimeError(f"teacher load not clean: missing={miss[:5]} unexpected={unexp[:5]}")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    _auto_attention_backend(teacher)

    student_net = build_variant(ckpt, student_variant, device=device, with_instance=False, deep_supervision=False)
    student = student_net.backbone
    _auto_attention_backend(student)
    student.train()
    t_dim = int(t_cfg["embed_dim"])
    s_dim = int(VARIANT_ARCH[student_variant].get("embed_dim", t_dim))   # small = {} → teacher width (identity proj)
    projection = DistillProjection(s_dim, t_dim).to(device)

    params = [p for p in student.parameters() if p.requires_grad] + list(projection.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    gen = torch.Generator(device=device).manual_seed(seed)
    amp = device.startswith("cuda") if isinstance(device, str) else False

    use_synthetic = smoke and not (source or corpus or manifest or scrolls)
    if use_synthetic:
        print("[distil] smoke: SYNTHETIC crops (no data source) — exercises the machinery only.", flush=True)
        src = _synthetic_crops(steps, batch, device, gen)
    else:
        mode = "scroll" if source else corpus_mode
        print(f"[distil] {student_variant} <- teacher | mode={mode} | steps={steps} batch={batch} lr={lr}", flush=True)
        src = _corpus_crops(mode, steps, batch, device, scrolls=scrolls, corpus=corpus, manifest=manifest,
                            source=source, seed=seed)

    last: dict = {}
    t0 = time.time()
    seen = 0
    for step, x in enumerate(src, start=1):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            t_feats = teacher.forward_features(x, view_kind="global")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            s_feats = student.forward_features(x, view_kind="global")
            loss, terms = distill_step_losses(
                s_feats, t_feats, projection,
                token_weight=token_weight, gram_weight=gram_weight, cls_weight=cls_weight,
                gram_n_sample=gram_n_sample,
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        last = terms
        seen += batch
        if step % log_every == 0 or step == 1 or step == steps:
            dt = time.time() - t0
            cps = seen / max(dt, 1e-6)
            eta_m = (steps - step) * (dt / max(step, 1)) / 60.0
            cos = 1.0 - float(terms.get("token", 1.0))          # token-cosine SIMILARITY (↑ = closer to teacher)
            msg = " ".join(f"{k}={v:.4f}" for k, v in terms.items())
            print(f"[distil {student_variant} {step}/{steps}] {msg} cos={cos:.4f} "
                  f"| {cps:.1f} crop/s ETA {eta_m:.0f}m", flush=True)
        if checkpoint_every and out and step % checkpoint_every == 0:
            _save_distilled(out, student, projection, student_variant, student_net, ckpt)
            print(f"[distil] checkpoint @ step {step} -> {out}", flush=True)

    if out:
        _save_distilled(out, student, projection, student_variant, student_net, ckpt)
    return {"variant": student_variant, "steps": steps, "final_terms": last,
            "final_cos": 1.0 - float(last.get("token", 1.0)), "out": out}


def run(args) -> dict:
    """CLI entry (``hercuvit distill``). Distils one variant, or ``--student all`` → small + tiny + nano (each
    saved under ``--out`` as ``<variant>.pt``). Missing/None attrs fall back to defaults (the recipe values)."""
    import os
    from .chain import resolve_corpus
    def g(name, default):
        v = getattr(args, name.replace("-", "_"), default)
        return default if v is None else v
    smoke = bool(g("smoke", False))
    student = g("student", "small")
    corpus_dir = None
    if getattr(args, "corpus", None) or getattr(args, "corpus_hf", None):       # herculabels mode source
        corpus_dir = resolve_corpus(getattr(args, "corpus", None), getattr(args, "corpus_hf", None))
    out = g("out", None)
    common = dict(
        teacher_ckpt=g("teacher", DINOVOL_HF_REPO),
        steps=int(g("steps", 8 if smoke else 2000)),   # --smoke → a few steps unless an explicit --steps
        lr=float(g("distill_lr", 2e-4)),
        token_weight=float(g("token_weight", 1.0)),
        gram_weight=float(g("gram_weight", 1.0)),
        cls_weight=float(g("cls_weight", 0.5)),
        batch=int(g("batch", 1)),
        device=g("device", "cuda"),
        smoke=smoke,
        corpus_mode=g("corpus_mode", "random"),
        scrolls=getattr(args, "scrolls", None),
        corpus=corpus_dir,
        manifest=getattr(args, "manifest", None),
        source=g("source", None),
        checkpoint_every=int(g("checkpoint_every", 0)),
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

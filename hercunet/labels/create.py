"""STAGE 1 entry point — pseudo-label generation into a ``.herculabels`` corpus.

``create`` is the single function ``hercunet labels create`` lands behind. It orchestrates, per window,
the pipeline documented in ``submission/writeup/herculabels.md``:

    cleaned substrate + frame field → 2.5-D meshlets → slab selection (positive + gap-gated negative)
    → 8-D contrastive embedding → probeom clustering → medial-mesh fit → quality

and writes each window (base sheet meshes + the meshlet cloud) into a corpus container. With
``--interactive`` it opens the Qt viewer and grinds windows open-endedly; otherwise it generates
``--count`` windows headless. Augmentations are NOT produced here — they are a re-runnable ``export``
(see docs/herculabels.md).
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np

_MIN_MATERIAL_FRAC = 0.15         # skip windows with less material than this
_MAX_ATTEMPTS_PER_WINDOW = 40     # give up drawing a non-air window after this many tries (default screen)
_MAX_HIMAT_SCAN = 20000           # coarse-mask himat screen is instant → allow a big scan before falling back


def _require_gpu(gpu: bool) -> None:
    """GPU gate — the pipeline is impractically slow without CUDA, so refuse rather than crawl."""
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"hercunet labels create: PyTorch is required ({type(e).__name__}: {e})")
    if gpu and not torch.cuda.is_available():
        raise SystemExit(
            "hercunet labels create: no CUDA GPU detected — label generation requires a GPU. "
            "(Pass --no-gpu only for CPU testing; it is impractically slow.)"
        )


def create(
    corpus_path: str,
    *,
    count: int | None,
    interactive: bool = False,
    scroll: str | None = None,
    coords: str | None = None,
    coords_file: str | None = None,
    himat: float | None = None,
    voxel_min: float = 7.5,
    voxel_max: float = 9.5,
    seed_stride_um: float = 40.0,
    sample_um: float = 20.0,
    min_cluster_size: int = 250,
    old_negatives: bool = False,
    seed: int = 0,
    gpu: bool = True,
    deterministic: bool = True,
) -> None:
    """Create the corpus at ``corpus_path`` (a ``.herculabels`` dir; fails if it exists). ``interactive``
    opens the viewer for open-ended grinding (no ``count``); otherwise ``count`` windows are generated
    headless. ``coords`` (``"z,y,x"``, requires ``scroll``) targets one exact window; ``coords_file`` grinds
    an ordered list; ``himat`` mines high-material windows via the coarse mask. ``old_negatives`` selects the
    legacy gap-gated negatives (§5.2) that generated the published corpus; the default is the current
    slab-field negatives (§5.3)."""
    _require_gpu(gpu)
    if interactive:                                             # fail fast (before making the corpus dir) if [viz] is missing
        from ..viz.interactive import require_viewer
        require_viewer()
    mat_thr = float(himat) if himat is not None else _MIN_MATERIAL_FRAC
    # The slab-field negatives (§5.3) are now the DEFAULT; --old-negatives selects the gap-gated method
    # (§5.2) that made the published corpus. Internally the pipeline still keys on ``slab_negatives``.
    slab_negatives = not old_negatives

    from ..config import Config
    from ..data import get_backend
    from ._brick import build_brick, find_scroll, in_band_scrolls, sample_windows, window_material_frac
    from ._generate import generate
    from .corpus import Corpus

    be = get_backend(Config.from_env())
    run_id = f"run{seed}"
    gen_params = dict(voxel_min=voxel_min, voxel_max=voxel_max, seed_stride_um=seed_stride_um,
                      sample_um=sample_um, min_cluster_size=min_cluster_size, slab_negatives=slab_negatives,
                      sigma_tensor=4.0, merge_size=3, n_merges=2, n_splits=2, conf_ds=4, level=0)
    source = "coords-file" if coords_file else ("coords" if coords is not None else "random")
    params_hash = hashlib.md5(json.dumps(gen_params, sort_keys=True).encode()).hexdigest()[:12]
    create_params = dict(mode="interactive" if interactive else "headless", scroll=scroll, seed=seed,
                         source=source, coords=coords, coords_file=coords_file, himat=himat,
                         negatives=("gap-gated" if old_negatives else "slab-field"),
                         count=count, gen_params=gen_params, params_hash=params_hash)

    # Build the container FIRST so an existing-corpus clash fails before any heavy compute.
    corpus = Corpus.create(corpus_path, create_params=create_params)
    print(f"[create] new corpus {corpus.root}  (mode={create_params['mode']}, source={source})", flush=True)

    def _args_for(sid, cz, cy, cx):
        return SimpleNamespace(
            scroll=sid, coords=f"{cz},{cy},{cx}", level=0, gpu=gpu,
            deterministic=deterministic, run_id=run_id, sigma_tensor=4.0,
            seed_stride_um=seed_stride_um, sample_um=sample_um, min_cluster_size=min_cluster_size,
            merge_size=3, n_merges=2, n_splits=2, conf_ds=4, slab_negatives=slab_negatives, source=source,
        )

    if interactive:                                              # launch the Qt viewer; it grinds into the corpus
        from ..viz.interactive import launch_interactive
        from .grind import GrindSession, parse_coords_file

        scrolls = _resolve_scrolls(be, scroll, voxel_min, voxel_max, find_scroll, in_band_scrolls)
        coords_list = first = None                               # None,None = open-ended random grinding
        if coords_file:
            coords_list = parse_coords_file(coords_file, scroll)
        elif coords is not None:
            cz, cy, cx = (int(v) for v in coords.split(","))
            first = (find_scroll(be, scroll).scroll_id, cz, cy, cx)
        session = GrindSession(be, scrolls, _args_for, gpu, seed, corpus=corpus,
                               scroll=scroll, coords_list=coords_list, first=first, himat=himat)
        launch_interactive(session)
        return

    # ---- headless generation ----
    def _emit(sid, cz, cy, cx, brick):
        res = generate(_args_for(sid, cz, cy, cx), brick)
        if res is None:
            return False
        corpus.write_window(res.meta, res.meshes, res.meshlet)
        return True

    if coords_file:                                              # regenerate an EXACT listed set of windows
        from .grind import parse_coords_file
        entries = parse_coords_file(coords_file, scroll)
        print(f"[create] generating {len(entries)} window(s) from {coords_file}", flush=True)
        made = 0
        for i, (sid, cz, cy, cx) in enumerate(entries, 1):
            target = find_scroll(be, sid)
            brick = build_brick(be, target, (cz, cy, cx), level=0, gpu=gpu)
            if _emit(target.scroll_id, cz, cy, cx, brick):
                made += 1
            else:
                print(f"[skip] {sid} z{cz} y{cy} x{cx} — no fittable sheet", flush=True)
            if i % 50 == 0:
                print(f"[create]   {i}/{len(entries)} ({made} written)…", flush=True)
        print(f"[create] wrote {made}/{len(entries)} window(s) into {corpus.root}", flush=True)
        return

    if coords is not None:                                        # one exact, caller-specified window
        cz, cy, cx = (int(v) for v in coords.split(","))
        target = find_scroll(be, scroll)
        print(f"[create] exact window {target.scroll_id} z{cz} y{cy} x{cx}", flush=True)
        brick = build_brick(be, target, (cz, cy, cx), level=0, gpu=gpu)
        frac = float((brick["bced"] > np.percentile(brick["bced"], 55)).mean())
        if frac < mat_thr:
            print(f"[warn] material {frac:.2f} < {mat_thr} — window looks mostly air", flush=True)
        made = int(_emit(target.scroll_id, cz, cy, cx, brick))
        print(f"[create] wrote {made}/1 window into {corpus.root}", flush=True)
        return

    scrolls = _resolve_scrolls(be, scroll, voxel_min, voxel_max, find_scroll, in_band_scrolls)
    print(f"[create] sampling {count} window(s) from: {', '.join(scrolls)}"
          + (f" (himat ≥{mat_thr:.2f})" if himat is not None else ""), flush=True)

    made = attempts = 0
    stream = sample_windows(be, scrolls, level=0, seed=seed)
    cap = _MAX_HIMAT_SCAN if himat is not None else count * _MAX_ATTEMPTS_PER_WINDOW
    best = None                                                  # (frac, sid, coords) fallback when himat unmet
    for sid, wcoords in stream:
        if made >= count:
            break
        attempts += 1
        if attempts > cap:
            print(f"[create] gave up after {attempts} draws with {made}/{count} windows", flush=True)
            break
        target = find_scroll(be, sid)
        try:
            if himat is not None:                                # instant coarse-mask screen — read only winners
                frac = window_material_frac(be, target, wcoords)
                if best is None or frac > best[0]:
                    best = (frac, sid, wcoords)
                if frac < mat_thr:
                    continue
            brick = build_brick(be, target, wcoords, level=0, gpu=gpu)
        except Exception as exc:                                 # a scroll/window that won't read (metadata already
            print(f"[skip] {sid} {wcoords} — read failed after retries "  # retried) must not kill the whole sample
                  f"({type(exc).__name__}: {str(exc)[:100]}); drawing another", flush=True)
            continue
        if himat is None:                                        # default: read + screen each candidate
            frac = float((brick["bced"] > np.percentile(brick["bced"], 55)).mean())
            if frac < mat_thr:
                print(f"[skip-air] {sid} {wcoords} material {frac:.2f} < {mat_thr}", flush=True)
                continue
        if _emit(sid, wcoords[0], wcoords[1], wcoords[2], brick):
            made += 1

    print(f"[create] wrote {made}/{count} window(s) into {corpus.root}", flush=True)


def _resolve_scrolls(be, scroll, voxel_min, voxel_max, find_scroll, in_band_scrolls) -> dict:
    """{scroll_id: voxel_um} to sample from — one explicit ``--scroll`` or the in-band corpus set."""
    if scroll:
        t0 = find_scroll(be, scroll)
        return {t0.scroll_id: float(t0.resolution_um or 8.64)}
    scrolls = in_band_scrolls(be, voxel_min, voxel_max)
    if not scrolls:
        raise SystemExit(f"no scrolls with voxel size in [{voxel_min}, {voxel_max}] µm")
    return scrolls

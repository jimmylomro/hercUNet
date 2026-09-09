"""STAGE 1 entry point — pseudo-label generation.

``create`` is the single function the CLI (``hercunet labels create``) and any programmatic caller
land behind. It orchestrates, per window, the pipeline documented in ``docs/pseudo-labels.md``:

    cleaned substrate + frame field → 2.5-D meshlets → slab selection (positive + gap-gated negative)
    → 8-D contrastive embedding → probeom clustering → medial-mesh fit → ∇φ / owner_full
    → confidence + quality → .npz
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

_MIN_MATERIAL_FRAC = 0.15         # skip windows with less material than this
_MAX_ATTEMPTS_PER_WINDOW = 40     # give up drawing a non-air window after this many tries


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
    *,
    count: int,
    output_dir: str,
    scroll: str | None = None,
    visualise: str | None = None,
    visualise_output_dir: str | None = None,
    voxel_min: float = 7.5,
    voxel_max: float = 9.5,
    seed_stride_um: float = 40.0,
    sample_um: float = 20.0,
    min_cluster_size: int = 250,
    slab_negatives: bool = False,
    seed: int = 0,
    gpu: bool = True,
    deterministic: bool = True,
) -> None:
    """Generate ``count`` sheet-membership pseudo-label windows into ``output_dir``.

    Parameters mirror the CLI flags one-to-one. ``visualise`` (only valid with ``count == 1``) renders
    the window: ``"image"`` writes a cross-section montage of the streamlets coloured by cluster to
    ``visualise_output_dir`` (default: ``output_dir``); ``"interactive"`` is not implemented yet.
    """
    if visualise and count != 1:
        raise ValueError("visualise requires count == 1")
    if visualise not in (None, "image", "interactive"):
        raise ValueError(f"visualise must be None, 'image' or 'interactive' (got {visualise!r})")
    if visualise == "interactive":
        raise SystemExit(
            "hercunet labels create: --visualise interactive (the live viewer) is not implemented yet. "
            "Use --visualise image for the cross-section montage."
        )
    _require_gpu(gpu)
    viz_out = visualise_output_dir or output_dir

    from ..config import Config
    from ..data import get_backend
    from ._brick import build_brick, find_scroll, in_band_scrolls, sample_windows
    from ._generate import generate

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    be = get_backend(Config.from_env())

    if scroll:
        target0 = find_scroll(be, scroll)
        scrolls = {target0.scroll_id: float(target0.resolution_um or 8.64)}
    else:
        scrolls = in_band_scrolls(be, voxel_min, voxel_max)
        if not scrolls:
            raise SystemExit(f"no scrolls with voxel size in [{voxel_min}, {voxel_max}] µm")
    print(f"[create] sampling {count} window(s) from: {', '.join(scrolls)}", flush=True)

    run_id = f"run{seed}"
    made = attempts = 0
    stream = sample_windows(be, scrolls, level=0, seed=seed)
    for sid, coords in stream:
        if made >= count:
            break
        attempts += 1
        if attempts > count * _MAX_ATTEMPTS_PER_WINDOW:
            print(f"[create] gave up after {attempts} draws with only {made}/{count} non-air windows", flush=True)
            break
        target = find_scroll(be, sid)
        brick = build_brick(be, target, coords, level=0, gpu=gpu)
        frac = float((brick["bced"] > np.percentile(brick["bced"], 55)).mean())
        if frac < _MIN_MATERIAL_FRAC:
            print(f"[skip-air] {sid} {coords} material {frac:.2f}", flush=True)
            continue
        args = SimpleNamespace(
            scroll=sid, coords=f"{coords[0]},{coords[1]},{coords[2]}", level=0, gpu=gpu,
            deterministic=deterministic, run_id=run_id, out=str(out), sigma_tensor=4.0,
            seed_stride_um=seed_stride_um, sample_um=sample_um, min_cluster_size=min_cluster_size,
            merge_size=3, n_merges=2, n_splits=2, conf_ds=4,
            visualise=visualise, visualise_out=str(viz_out), slab_negatives=slab_negatives,
        )
        if generate(args, brick) is not None:
            made += 1

    print(f"[create] generated {made}/{count} pseudo-label window(s) into {out}", flush=True)

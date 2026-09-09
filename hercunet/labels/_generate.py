"""Per-window generation: extraction → medial-mesh fit → ∇φ / owner → confidence + quality →
merge/split augmentations → one ``.npz`` bundle.

Ported from the research ``generate_sample.generate`` with the debug-grid / render paths removed.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time

import numpy as np

from hercunet.labels.augment import augment as aug
import hercunet.labels.io.bundle as sample_io
from ._window import extract_window
from hercunet.labels.mesh.medial import (
    build_gradphi,
    confidence_volume,
    fit_sheet_meshes,
    intersection_confidence,
    mesh_field_congruence,
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "nogit"


def enable_determinism(strict: bool = True) -> None:
    """Bit-reproducible CUDA so a re-run of the same window reproduces the sample byte-for-byte.
    ``CUBLAS_WORKSPACE_CONFIG`` must be set before the first cuBLAS handle; the launcher exports it,
    the ``setdefault`` here only covers a fresh single-process run."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _refit(C, plab_new, labels, dev):
    pl = np.where(np.isin(plab_new, list(labels)), plab_new, -1)
    return fit_sheet_meshes(C["pts"], pl, C["bced"], None, C["vu"], normals=C["normals"], jac=C["jac"],
                            step_um=30.0, device=dev, verbose=False)


def generate(args, brick: dict):
    """Run the full per-window pipeline on a prefetched ``brick`` and write the ``.npz``. Returns the
    output path (or None if the window is empty)."""
    dev = "cuda" if args.gpu else "cpu"
    if getattr(args, "deterministic", False):
        enable_determinism(strict=True)
    master_seed = int(hashlib.md5(f"{args.run_id}|{args.scroll}|{args.level}|{args.coords}".encode()).hexdigest()[:8], 16)
    np.random.seed(master_seed % (2**32))
    try:
        import torch
        torch.manual_seed(master_seed % (2**31))
    except Exception:
        pass
    t0 = time.time()
    C = extract_window(args, brick)
    vu, shape = C["vu"], C["bced"].shape
    pts, plab, sid, E = C["pts"], C["plab"], C["sid"], C["E"]
    ulab = aug.per_unit_labels(sid, plab, n_units=len(E))

    if getattr(args, "visualise", None) == "image":                  # cross-section montage, coloured by cluster
        from ._viz import render_window_montage
        render_window_montage(C["bced"], pts, plab, C["org"], vu, args.visualise_out,
                              scroll=args.scroll, level=args.level, coords=args.coords)

    base_meshes = fit_sheet_meshes(pts, plab, C["bced"], None, vu, normals=C["normals"], jac=C["jac"],
                                   step_um=30.0, device=dev, verbose=False)
    # quality: mesh-vs-fibre-field normal congruence (stamped in meta, never used to drop)
    q_win, q_sheet = mesh_field_congruence(base_meshes, C["ff_normal"], C["ff_coherence"])
    mag, lab3, nrm3, overlap = build_gradphi(base_meshes, shape, vu, use_gpu=args.gpu)
    conf_geom, cover = intersection_confidence(overlap, lab3)
    # embedding confidence from prob-eom propagation distance (channel key stays "sharp2")
    pd = np.asarray(C["prop_dist"], np.float32)
    _sc = float(np.percentile(pd[pd > 0], 90)) if np.any(pd > 0) else 1.0
    conf_sharp2, _ = confidence_volume(pts, np.clip(pd / (_sc + 1e-9), 0.0, 1.0), shape, 6.0)
    conf_dropped, _ = aug.dropped_lowconf(pts, plab, shape)
    records = {"base": dict(kind="base", source_ids=[], meshes=base_meshes,
                            conf=dict(intersection=conf_geom, sharp2=conf_sharp2, dropped=conf_dropped))}
    print(f"  base: {len(base_meshes)} sheets ({time.time()-t0:.0f}s)", flush=True)

    # ---- augmentations (same run — reuse the clusters/streamlets) ----
    i = 0
    merge_sets, _msizes = aug.ordinal_merge_sets(E, ulab, n_merge=args.merge_size, max_sets=args.n_merges)
    for grp in merge_sets:
        plab_m, mid = aug.relabel_merge_set(plab, grp)
        rf = _refit(C, plab_m, [mid], dev)
        mm = {k: v for k, v in base_meshes.items() if k not in grp}; mm.update(rf)
        amag, alab, _n, aovl = build_gradphi(mm, shape, vu, use_gpu=args.gpu)
        ci, acov = intersection_confidence(aovl, alab)
        mask = np.zeros(shape, bool)
        for x, y in zip(grp[:-1], grp[1:]):
            mask |= aug.interface_band(lab3, x, y, band_px=3)
        records[f"aug{i}"] = dict(kind="merge", source_ids=list(grp), meshes=mm,
                                  conf=dict(intersection=ci, sharp2=conf_sharp2, dropped=conf_dropped,
                                            perturb=aug.low_confidence_from_mask(mask, cover)))
        i += 1
    labs = sorted(int(x) for x in np.unique(ulab) if x >= 0)
    sizes = {c: int((ulab == c).sum()) for c in labs}
    done = 0
    for c in sorted(labs, key=lambda k: -sizes[k]):
        if done >= args.n_splits:
            break
        res = aug.borderline_split(E, ulab, c)
        if res is None:
            continue
        unit_side, _gap = res
        plab_s, (ia, ib) = aug.relabel_split(plab, sid, c, unit_side)
        rf = _refit(C, plab_s, [ia, ib], dev)
        mm = {k: v for k, v in base_meshes.items() if k != c}; mm.update(rf)
        amag, alab, _n, aovl = build_gradphi(mm, shape, vu, use_gpu=args.gpu)
        ci, _ac = intersection_confidence(aovl, alab)
        mask = aug.interface_band(alab, ia, ib, band_px=3)
        records[f"aug{i}"] = dict(kind="split", source_ids=[c], meshes=mm,
                                  conf=dict(intersection=ci, sharp2=conf_sharp2, dropped=conf_dropped,
                                            perturb=aug.low_confidence_from_mask(mask, cover)))
        i += 1; done += 1

    meta = dict(code_git_sha=_git_sha(),
                params_hash=hashlib.md5(json.dumps(vars(args), sort_keys=True).encode()).hexdigest()[:12],
                scroll=args.scroll, level=args.level, voxel_um=float(vu),
                coords=[int(x) for x in args.coords.split(",")], org=[int(x) for x in C["org"]],
                shape=[int(s) for s in shape], master_seed=master_seed, run_id=args.run_id,
                conf_ds=args.conf_ds, n_sheets=len(base_meshes),
                quality=dict(window=round(q_win, 4), method="mesh_field_congruence_v1", threshold_ref=0.80,
                             per_sheet={str(c): round(d["score"], 4) for c, d in q_sheet.items()}))
    print(f"  quality: window congruence {q_win:.3f} (ref threshold 0.80)", flush=True)
    os.makedirs(args.out, exist_ok=True)
    zc, yc, xc = args.coords.split(",")
    path = os.path.join(args.out, f"sample_s{args.scroll}_L{args.level}_z{zc}_y{yc}_x{xc}.npz")
    sample_io.save_sample_bundle(path, meta, records)
    mb = os.path.getsize(path) / 1e6
    print(f"→ {path}  ({len(records)} records: 1 base + {len(records)-1} augs, {mb:.2f} MB, "
          f"{time.time()-t0:.0f}s total)", flush=True)
    return path

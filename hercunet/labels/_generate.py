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
from types import SimpleNamespace

import numpy as np

from hercunet.labels.augment import augment as aug
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


def _refit(ml, plab_new, labels, dev):
    pl = np.where(np.isin(plab_new, list(labels)), plab_new, -1)
    return fit_sheet_meshes(ml["pts"], pl, ml.get("bced"), None, ml["vu"], normals=ml["normals"],
                            jac=ml["jac"], step_um=30.0, device=dev, verbose=False)


def meshlet_from_context(C, ulab):
    """Extract the sparse MESHLET cloud from a pipeline context ``C`` — everything an edit / a mesh refit /
    a confidence recompute needs, and NOTHING dense (no ``bced``, no ``ff_*``). This is what a corpus window
    persists (see docs/herculabels.md); frames are downcast to f16 at save time by the bundle writer."""
    shape = np.asarray(C["bced"]).shape
    return dict(pts=np.asarray(C["pts"], np.float32), sid=np.asarray(C["sid"], np.int64),
                plab=np.asarray(C["plab"], np.int32), normals=np.asarray(C["normals"], np.float32),
                jac=np.asarray(C["jac"], np.float32), prop_dist=np.asarray(C["prop_dist"], np.float32),
                E=np.asarray(C["E"], np.float32), ulab=np.asarray(ulab, np.int32),
                vu=float(C["vu"]), shape=tuple(int(s) for s in shape))


def base_confidence(meshes, ml, shape, vu, *, gpu):
    """The base window's three confidence channels — recomputed purely from the sheet ``meshes`` + the
    meshlet cloud ``ml`` (no dense fields): intersection (∇φ envelope overlap), sharp2 (embedding
    propagation distance), dropped (culled-cluster footprint). Returns ``(conf_dict, lab3, cover)`` — the
    latter two feed the augmentation records. Not stored in the corpus; emitted live + written on export."""
    _mag, lab3, _nrm, overlap = build_gradphi(meshes, shape, vu, use_gpu=gpu)
    conf_geom, cover = intersection_confidence(overlap, lab3)
    pts, plab, pd = ml["pts"], np.asarray(ml["plab"]), np.asarray(ml["prop_dist"], np.float32)
    sc = float(np.percentile(pd[pd > 0], 90)) if np.any(pd > 0) else 1.0
    conf_sharp2, _ = confidence_volume(pts, np.clip(pd / (sc + 1e-9), 0.0, 1.0), shape, 6.0)
    conf_dropped, _ = aug.dropped_lowconf(pts, plab, shape)
    return dict(intersection=conf_geom, sharp2=conf_sharp2, dropped=conf_dropped), lab3, cover


def build_records(meshes, ml, shape, vu, *, gpu, augment, merge_size=3, n_merges=2, n_splits=2, dev=None):
    """Build the TRAINING records for one window from its base ``meshes`` + meshlet cloud ``ml`` — the base
    record and, when ``augment``, the merge/split augmentation records. Everything is regenerated from the
    stored cloud (E/ulab/plab/sid/pts/normals/jac/prop_dist), so ``export`` can produce augmentations without
    re-running the pipeline and without any dense field. Returns ``{key: {kind, source_ids, meshes, conf}}``."""
    dev = dev or ("cuda" if gpu else "cpu")
    conf, lab3, cover = base_confidence(meshes, ml, shape, vu, gpu=gpu)
    records = {"base": dict(kind="base", source_ids=[], meshes=meshes, conf=conf)}
    if not augment:
        return records
    E, ulab, plab, sid = ml["E"], np.asarray(ml["ulab"]), np.asarray(ml["plab"]), np.asarray(ml["sid"])
    i = 0
    for grp in aug.ordinal_merge_sets(E, ulab, n_merge=merge_size, max_sets=n_merges)[0]:
        plab_m, mid = aug.relabel_merge_set(plab, grp)
        mm = {k: v for k, v in meshes.items() if k not in grp}; mm.update(_refit(ml, plab_m, [mid], dev))
        _am, alab, _n, aovl = build_gradphi(mm, shape, vu, use_gpu=gpu)
        ci, _ac = intersection_confidence(aovl, alab)
        mask = np.zeros(shape, bool)
        for x, y in zip(grp[:-1], grp[1:]):
            mask |= aug.interface_band(lab3, x, y, band_px=3)
        records[f"aug{i}"] = dict(kind="merge", source_ids=list(grp), meshes=mm,
                                  conf=dict(intersection=ci, sharp2=conf["sharp2"], dropped=conf["dropped"],
                                            perturb=aug.low_confidence_from_mask(mask, cover)))
        i += 1
    labs = sorted(int(x) for x in np.unique(ulab) if x >= 0)
    sizes = {c: int((ulab == c).sum()) for c in labs}
    done = 0
    for c in sorted(labs, key=lambda k: -sizes[k]):
        if done >= n_splits:
            break
        res = aug.borderline_split(E, ulab, c)
        if res is None:
            continue
        unit_side, _gap = res
        plab_s, (ia, ib) = aug.relabel_split(plab, sid, c, unit_side)
        mm = {k: v for k, v in meshes.items() if k != c}; mm.update(_refit(ml, plab_s, [ia, ib], dev))
        _am, alab, _n, aovl = build_gradphi(mm, shape, vu, use_gpu=gpu)
        ci, _ac = intersection_confidence(aovl, alab)
        mask = aug.interface_band(alab, ia, ib, band_px=3)
        records[f"aug{i}"] = dict(kind="split", source_ids=[c], meshes=mm,
                                  conf=dict(intersection=ci, sharp2=conf["sharp2"], dropped=conf["dropped"],
                                            perturb=aug.low_confidence_from_mask(mask, cover)))
        i += 1; done += 1
    return records


def generate(args, brick: dict, progress=None):
    """Run the per-window pipeline on a prefetched ``brick`` and return the window's data — the base sheet
    ``meshes``, the ``meshlet`` cloud, and ``meta`` (provenance + quality). Does NOT write and does NOT
    augment: a corpus stores base + meshlets, and augmentations are regenerated at ``export`` time
    (:func:`build_records`). Returns a ``SimpleNamespace(meta, meshes, meshlet)`` or ``None`` if the window
    has no fittable sheet.

    ``progress`` (optional ``callable(stage, payload)``) is an OBSERVER hook for the interactive viewer — it
    never alters the computation. It threads into :func:`extract_window` (``"streamlets"``) and emits
    ``"clusters"`` (labelled cloud), ``"editctx"`` (in-process edit context), ``"meshes"`` (fitted sheets)
    and ``"confidence"`` (the three base channels, for the overlay). The confidence channels are computed
    for the overlay only when a viewer is watching; a headless run skips them (recomputed at export)."""
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
    C = extract_window(args, brick, progress=progress)
    vu, shape = C["vu"], C["bced"].shape
    pts, plab, sid, E = C["pts"], C["plab"], C["sid"], C["E"]
    ulab = aug.per_unit_labels(sid, plab, n_units=len(E))
    if progress is not None:                                          # observer hook (viewer) — labelled cloud
        progress("clusters", {"pts": pts.astype(np.float32), "plab": plab.astype(np.int32),
                              "corner": brick.get("corner")})
        from hercunet.labels.edit import build_edit_context             # hand the viewer the edit context
        progress("editctx", build_edit_context(C, args, dev))          # (in-process reference; edits + save)
        if C.get("pos_nbr") is not None:                                # per-point positive/negative sample tables
            progress("samples", {"pos_nbr": C["pos_nbr"], "neg_nbr": C["neg_nbr"]})  # viewer 'samples' overlay (live only)

    base_meshes = fit_sheet_meshes(pts, plab, C["bced"], None, vu, normals=C["normals"], jac=C["jac"],
                                   step_um=30.0, device=dev, verbose=False)
    if not base_meshes:
        print(f"  no fittable sheet in this window ({time.time()-t0:.0f}s) — skipping", flush=True)
        return None
    print(f"  base: {len(base_meshes)} sheets ({time.time()-t0:.0f}s)", flush=True)
    meshlet = meshlet_from_context(C, ulab)
    if progress is not None:                                          # observer hooks (viewer) — meshes + overlay
        progress("meshes", {"meshes": base_meshes})
        conf, _lab3, _cover = base_confidence(base_meshes, meshlet, shape, vu, gpu=args.gpu)
        progress("confidence", conf)
    # quality: mesh-vs-fibre-field normal congruence (stamped in meta, never used to drop)
    q_win, q_sheet = mesh_field_congruence(base_meshes, C["ff_normal"], C["ff_coherence"])
    print(f"  quality: window congruence {q_win:.3f} (ref threshold 0.80)", flush=True)
    meta = dict(code_git_sha=_git_sha(),
                params_hash=hashlib.md5(json.dumps(_arg_dict(args), sort_keys=True).encode()).hexdigest()[:12],
                scroll=args.scroll, level=args.level, voxel_um=float(vu),
                coords=[int(x) for x in args.coords.split(",")], org=[int(x) for x in C["org"]],
                shape=[int(s) for s in shape], master_seed=master_seed, run_id=args.run_id,
                conf_ds=args.conf_ds, n_sheets=len(base_meshes), source=getattr(args, "source", None),
                edited=False, kind="base",
                quality=dict(window=round(q_win, 4), method="mesh_field_congruence_v1", threshold_ref=0.80,
                             per_sheet={str(c): round(d["score"], 4) for c, d in q_sheet.items()}))
    return SimpleNamespace(meta=meta, meshes=base_meshes, meshlet=meshlet)


def _arg_dict(args):
    """JSON-able view of an args namespace for the params hash (drops non-serialisable / volatile fields)."""
    drop = {"corpus", "source"}
    return {k: v for k, v in vars(args).items()
            if k not in drop and isinstance(v, (str, int, float, bool, type(None)))}

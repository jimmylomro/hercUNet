"""Herculabels → sheet-identity targets for the decoder (the label-prep extraction step).

Herculabels samples (``/media/jaime/backup/corpus/sample_s*_L0_*.npz``) store each sheet as a **mesh**, not a
dense ``owner_full`` volume:

* ``<rec>/cluster_ids``        — (K,) int32, the sheet instance ids present (0-based in the file);
* ``<rec>/V_k``               — (Hk, Wk, 3) float16, sheet *k*'s surface grid in **LOCAL window voxel
                                 coordinates, axis order (z, y, x)** (values may spill a little outside the
                                 window — clip);
* ``<rec>/foot_k``, ``width_k`` — per-sheet foot points / widths (not needed here);
* ``<rec>/conf_*``            — dense confidence volumes at ``conf_ds`` downsampling;
* ``meta_json``               — JSON with ``shape`` (z,y,x), ``voxel_um``, ``conf_ds``, ``records`` (per
                                 record ``kind``/``cluster_ids``/``cluster_shapes``), ``quality.per_sheet``.

``<rec>`` is one of ``base``/``aug0``/``aug1``/... (an augmentation view).

We turn the ``V_k`` meshes into:

1. a dense ``owner_full`` volume (int, ``0`` = ignore, ``cluster_id + 1`` = sheet) — the ``malis`` convention
   (``>0`` labelled, ``<=0`` ignore), reusable by the affinity/MALIS head too;
2. a coarse **node-ownership grid** (via :mod:`hercunet.tracer.grid`) — the decoder's per-node target.

**m7 gate.** m7-derived samples have surface/affinity labels but **no** ``cluster_ids`` / ``V_k`` → no sheet
identity → the decoder must not be supervised on them. :func:`has_identity` is that per-sample gate; it is the
single source of truth the dataset/trainer should consult before adding the decoder loss.

Pure numpy; no torch/nnunet. Safe to import on any host.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

# Axis order of the V_k coordinate channels, verified against window ``shape`` on the corpus.
VZ, VY, VX = 0, 1, 2
IGNORE = 0  # owner_full ignore label (malis treats <=0 as ignore; we use 0 and positive ids)


@dataclass(frozen=True)
class HercuSample:
    """One herculabels record, decoded enough to build targets."""

    record: str                 # "base" / "aug0" / ...
    shape: tuple                 # (D, H, W) window voxel shape, (z, y, x)
    voxel_um: float
    cluster_ids: np.ndarray      # (K,) int, file-native (0-based) sheet ids present
    meshes: dict                 # cluster_id (file-native) -> (Hk, Wk, 3) float32 local (z,y,x) voxel coords
    per_sheet_quality: dict      # cluster_id (file-native, as int) -> float quality in [0,1] (may be empty)

    @property
    def n_sheets(self) -> int:
        return len(self.cluster_ids)


def _decode_meta(npz) -> dict:
    raw = npz["meta_json"]
    return json.loads(bytes(np.asarray(raw).tolist()).decode("utf-8", "replace"))


def has_identity(npz_or_meta, record: str = "base") -> bool:
    """True iff this sample/record carries sheet identity (herculabels), False for m7-derived samples.

    Accepts an open ``np.load`` mapping (checks for ``<record>/cluster_ids`` + at least one ``<record>/V_*``)
    or a decoded ``meta_json`` dict (checks ``records[record].cluster_ids`` non-empty). This is the ONLY gate
    the trainer should use to decide whether to add the decoder loss for a sample.
    """
    if isinstance(npz_or_meta, dict):
        rec = npz_or_meta.get("records", {}).get(record, {})
        return bool(rec.get("cluster_ids"))
    files = set(getattr(npz_or_meta, "files", []))
    has_ids = f"{record}/cluster_ids" in files
    has_mesh = any(k.startswith(f"{record}/V_") for k in files)
    if not (has_ids and has_mesh):
        return False
    return int(np.asarray(npz_or_meta[f"{record}/cluster_ids"]).size) > 0


def load_sample(npz, record: str = "base") -> HercuSample:
    """Decode one record of an herculabels npz into a :class:`HercuSample`.

    Raises ``ValueError`` if the record has no identity (call :func:`has_identity` first for m7 samples).
    """
    if not has_identity(npz, record):
        raise ValueError(f"record {record!r} has no sheet identity (m7-derived or empty) — gate with has_identity()")
    meta = _decode_meta(npz)
    shape = tuple(int(s) for s in meta["shape"])
    voxel_um = float(meta.get("voxel_um", 0.0))
    cluster_ids = np.asarray(npz[f"{record}/cluster_ids"]).astype(np.int64).ravel()
    meshes = {}
    for cid in cluster_ids:
        key = f"{record}/V_{cid}"
        if key in npz.files:
            meshes[int(cid)] = np.asarray(npz[key]).astype(np.float32).reshape(-1, 3)
    rec_meta = meta.get("records", {}).get(record, {})
    q = {int(k): float(v) for k, v in rec_meta.get("per_sheet", meta.get("quality", {}).get("per_sheet", {}) or {}).items()}
    return HercuSample(record=record, shape=shape, voxel_um=voxel_um,
                       cluster_ids=cluster_ids, meshes=meshes, per_sheet_quality=q)


def rasterise_owner_full(sample: HercuSample, *, min_quality: float = 0.0) -> np.ndarray:
    """Paint the ``V_k`` meshes into a dense ``owner_full`` volume (D,H,W) int32.

    ``0`` = ignore (no sheet); a sheet voxel carries ``cluster_id + 1`` (so ``0`` stays ignore, matching the
    ``malis`` ``>0``-labelled convention). Sheets below ``min_quality`` (from ``meta.quality.per_sheet``) are
    skipped — they are the untrustworthy labels (e.g. the congruence-filtered ones), which we do NOT want to
    supervise identity on.

    Where two sheets rasterise into the same voxel (a touch/merge) the **later** sheet wins the voxel; such
    contested voxels are rare and are handled downstream by node-majority + the hard manifold gate, not here.
    """
    D, H, W = sample.shape
    owner = np.zeros((D, H, W), np.int32)
    for cid in sample.cluster_ids:
        cid = int(cid)
        if sample.per_sheet_quality.get(cid, 1.0) < min_quality:
            continue
        V = sample.meshes.get(cid)
        if V is None or V.size == 0:
            continue
        z = np.rint(V[:, VZ]).astype(np.int64)
        y = np.rint(V[:, VY]).astype(np.int64)
        x = np.rint(V[:, VX]).astype(np.int64)
        ok = (z >= 0) & (z < D) & (y >= 0) & (y < H) & (x >= 0) & (x < W)
        owner[z[ok], y[ok], x[ok]] = cid + 1
    return owner


def contested_mask(sample: HercuSample, *, min_quality: float = 0.0) -> np.ndarray:
    """(D,H,W) bool: voxels hit by >1 distinct sheet (merge/touch footprints).

    Useful as a diagnostic and to *down-weight* those voxels in supervision — they are exactly where the GT
    itself is ambiguous. (Does not alter :func:`rasterise_owner_full`'s last-writer-wins owner.)"""
    D, H, W = sample.shape
    count = np.zeros((D, H, W), np.uint8)
    for cid in sample.cluster_ids:
        cid = int(cid)
        if sample.per_sheet_quality.get(cid, 1.0) < min_quality:
            continue
        V = sample.meshes.get(cid)
        if V is None or V.size == 0:
            continue
        z = np.rint(V[:, VZ]).astype(np.int64)
        y = np.rint(V[:, VY]).astype(np.int64)
        x = np.rint(V[:, VX]).astype(np.int64)
        ok = (z >= 0) & (z < D) & (y >= 0) & (y < H) & (x >= 0) & (x < W)
        hit = np.zeros((D, H, W), bool)
        hit[z[ok], y[ok], x[ok]] = True
        count += hit.astype(np.uint8)
    return count > 1

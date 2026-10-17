"""HercuViT training dataset over the herculabels corpus.

One sample = one herculabels window (``sample_*.npz``; pulled from ``jimmylomro/hercunet-corpus``). The npz holds
sheet **meshes** and provenance (scroll / level / origin / shape / voxel size) but **no CT volume** (verified:
herculabels never stores dense CT — see ``docs/herculabels.md`` "Outputs"), so the CT cube is **streamed from the
scroll** at the window's stored coordinates through the reused data layer
(:func:`hercunet.infer.nnunet_infer.open_source` → ``read_window``).

Per sample this yields, at Dinovol's ~9.6 µm / 128³ working scale:
  * ``ct``          — ``(1, 128, 128, 128)`` float32, single-channel, Dinovol **robust**-normalised CT;
  * ``sem``         — ``(128, 128, 128)`` int64 semantic target (1 = sheet-surface, 0 = background);
  * ``owner``       — ``(128, 128, 128)`` int64 instance ids (0 = none, ``cluster_id + 1`` = sheet) for the
                      Mask2Former head;
  * ``has_identity``— bool; whether this sample carries sheet identity (gates the instance loss, §7).

**Semantic target derivation** (``docs/hercuvit.md`` §13 TODO, confirmed against ``docs/herculabels.md``): the
herculabels ``V_k`` meshes are the per-sheet **medial surfaces**; :func:`rasterise_owner_full` paints them into a
thin ``owner`` volume (``>0`` on a sheet-surface voxel). The semantic *sheet-surface* class is therefore exactly
``owner > 0``. This is the thin medial surface, matching the pixel decoder's full-resolution surface head.

**Resampling choice** (§12 watch-point #3): each window is native 7–10 µm. We centre-crop the native cube to the
physical field of view Dinovol expects (``128 · target_voxel_um`` µm per axis), then trilinear-resize that crop to
128³ (nearest for ``owner``). This keeps the encoder at its trained ~9.6 µm scale rather than squashing a whole
~1500 µm window into 128³ (which would coarsen to ~11–14 µm and risk merging 1–2-voxel-separated wraps). Axes
shorter than the target FOV use the whole axis (slightly finer than 9.6 µm). Normalisation is applied to the
native crop *before* the resize, matching villa ``_finalize_crop``.

Augmentation (§8): Dinovol-style geometry only — flips, 90° rotations, axis transposes; **no blur**; CT and both
targets follow the *same* spatial transform.
"""
from __future__ import annotations

import glob
import os
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ...tracer.labels import has_identity, load_sample, rasterise_owner_full
from .normalize import dinovol_normalise

#: Dinovol working scale (``docs/hercuvit.md`` §2) — the CT cube is resampled to this voxel size and side.
TARGET_VOXEL_UM = 9.6
CROP = 128


# --------------------------------------------------------------------------- CT streaming (reused data layer)
class ScrollCtReader:
    """Stream a window's CT cube from its scroll through the reused data layer, caching the open volume per scroll.

    ``open_source`` resolves a scroll id (e.g. ``"PHerc1447"`` or the bare numeric id herculabels stores) through
    the data-layer catalog; the same id the herculabels generator used, so it resolves identically. ``read_window``
    returns the native-resolution cube ``[org : org + shape]`` at the window's pyramid ``level``.
    """

    def __init__(self) -> None:
        self._vols: dict = {}

    def _vol(self, scroll):
        key = str(scroll)
        if key not in self._vols:
            from ...infer.nnunet_infer import open_source
            self._vols[key] = open_source(key)
        return self._vols[key]

    def __call__(self, scroll, level, org, shape) -> np.ndarray:
        z0, y0, x0 = (int(o) for o in org)
        dz, dy, dx = (int(s) for s in shape)
        blk, _origin = self._vol(scroll).read_window(int(level), z0, z0 + dz, y0, y0 + dy, x0, x0 + dx)
        return np.asarray(blk)


# --------------------------------------------------------------------------- pure helpers (unit-testable)
def resample_to_crop(
    ct: np.ndarray,
    owner: np.ndarray,
    voxel_um: float,
    *,
    target_voxel_um: float = TARGET_VOXEL_UM,
    crop: int = CROP,
) -> tuple[np.ndarray, np.ndarray]:
    """Centre-crop the native cube to Dinovol's physical FOV, then resize to ``crop``³ (CT trilinear, owner nearest).

    ``ct`` / ``owner`` are ``(D, H, W)`` at ``voxel_um``. The target FOV is ``crop · target_voxel_um`` µm per axis;
    an axis shorter than that uses the whole axis. Returns ``(ct_crop[D'..], owner_crop)`` both ``(crop,)*3``,
    float32 / int64. Normalisation is NOT applied here (the caller normalises the native crop first).
    """
    ct = np.asarray(ct, dtype=np.float32)
    owner = np.asarray(owner)
    fov_vox = max(1, int(round(crop * target_voxel_um / float(voxel_um))))  # native voxels spanning the FOV

    def _centre_slice(n: int, want: int) -> slice:
        take = min(n, want)
        lo = (n - take) // 2
        return slice(lo, lo + take)

    sz = _centre_slice(ct.shape[0], fov_vox)
    sy = _centre_slice(ct.shape[1], fov_vox)
    sx = _centre_slice(ct.shape[2], fov_vox)
    ct_c = ct[sz, sy, sx]
    own_c = owner[sz, sy, sx]

    ct_t = torch.from_numpy(np.ascontiguousarray(ct_c)).float()[None, None]
    ct_r = F.interpolate(ct_t, size=(crop, crop, crop), mode="trilinear", align_corners=False)[0, 0]
    own_t = torch.from_numpy(np.ascontiguousarray(own_c)).float()[None, None]
    own_r = F.interpolate(own_t, size=(crop, crop, crop), mode="nearest")[0, 0]
    return ct_r.numpy().astype(np.float32), own_r.numpy().astype(np.int64)


def geometric_augment(ct: np.ndarray, owner: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Dinovol-style geometry on a cubic volume: independent per-axis flips + a random axis permutation (90°
    rotations / transposes). CT and ``owner`` get the identical transform. No blur, no intensity change here."""
    flips = tuple(ax for ax in range(3) if rng.random() < 0.5)
    if flips:
        ct = np.flip(ct, flips)
        owner = np.flip(owner, flips)
    perm = rng.permutation(3)
    ct = np.ascontiguousarray(np.transpose(ct, perm))
    owner = np.ascontiguousarray(np.transpose(owner, perm))
    return ct, owner


# --------------------------------------------------------------------------- dataset
class HercuvitDataset(Dataset):
    """herculabels corpus → Dinovol-normalised CT + semantic + owner targets at 128³ / ~9.6 µm.

    Args:
        corpus_dir: directory of ``sample_*.npz`` (the pulled ``jimmylomro/hercunet-corpus``).
        record: which herculabels record to read (``"base"`` by default; aug records are label-space variants).
        identity_only: keep only samples carrying sheet identity (default True — v1 supervises both heads on
            herculabels samples; the instance loss is gated by ``has_identity`` regardless). m7-derived
            semantic-only samples are a future extension (§13 TODO).
        target_voxel_um / crop: Dinovol working scale.
        augment: apply :func:`geometric_augment`.
        ct_reader: callable ``(scroll, level, org, shape) -> (D,H,W) CT``; defaults to :class:`ScrollCtReader`
            (streams from the scroll). Tests inject a synthetic reader to stay offline.
        seed: base RNG seed (per-item stream derived from it + index).
    """

    def __init__(
        self,
        corpus_dir: str,
        *,
        record: str = "base",
        identity_only: bool = True,
        target_voxel_um: float = TARGET_VOXEL_UM,
        crop: int = CROP,
        augment: bool = True,
        ct_reader: Optional[Callable] = None,
        seed: int = 0,
    ) -> None:
        self.corpus_dir = corpus_dir
        self.record = record
        self.target_voxel_um = float(target_voxel_um)
        self.crop = int(crop)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.ct_reader = ct_reader if ct_reader is not None else ScrollCtReader()

        paths = sorted(glob.glob(os.path.join(corpus_dir, "sample_*.npz")))
        if not paths:
            raise FileNotFoundError(f"no sample_*.npz under {corpus_dir!r} — pull jimmylomro/hercunet-corpus first")
        self.samples: list[str] = []
        for p in paths:
            if not identity_only:
                self.samples.append(p)
                continue
            with np.load(p, allow_pickle=True) as z:
                if has_identity(z, record):
                    self.samples.append(p)
        if not self.samples:
            raise FileNotFoundError(f"no samples with identity (record {record!r}) under {corpus_dir!r}")

    def __len__(self) -> int:
        return len(self.samples)

    def _meta(self, z) -> dict:
        import json
        raw = z["meta_json"]
        return json.loads(bytes(np.asarray(raw).tolist()).decode("utf-8", "replace"))

    def __getitem__(self, idx: int) -> dict:
        path = self.samples[idx]
        rng = np.random.default_rng(self.seed + idx)
        with np.load(path, allow_pickle=True) as z:
            ident = has_identity(z, self.record)
            meta = self._meta(z)
            sample = load_sample(z, self.record)
            owner_native = rasterise_owner_full(sample)                      # (D,H,W) int32, thin medial surfaces

        scroll = meta.get("scroll")
        level = int(meta.get("level", 0))
        org = meta.get("org") or meta.get("coords")
        shape = meta.get("shape")
        voxel_um = float(meta.get("voxel_um", self.target_voxel_um))
        ct_native = self.ct_reader(scroll, level, org, shape)
        if tuple(ct_native.shape) != tuple(int(s) for s in shape):
            raise ValueError(f"{path}: CT read shape {ct_native.shape} != meta shape {tuple(shape)}")

        # robust-normalise the NATIVE crop (villa order: normalise then resize), then resample to 128³ / 9.6 µm.
        ct_norm = dinovol_normalise(ct_native)
        ct, owner = resample_to_crop(ct_norm, owner_native, voxel_um,
                                     target_voxel_um=self.target_voxel_um, crop=self.crop)
        if self.augment:
            ct, owner = geometric_augment(ct, owner, rng)

        sem = (owner > 0).astype(np.int64)
        return {
            "ct": torch.from_numpy(np.ascontiguousarray(ct)).float().unsqueeze(0),   # (1,128,128,128)
            "sem": torch.from_numpy(np.ascontiguousarray(sem)).long(),               # (128,128,128)
            "owner": torch.from_numpy(np.ascontiguousarray(owner)).long(),           # (128,128,128)
            "has_identity": bool(ident),
            "path": path,
        }


def hercuvit_collate(batch: Sequence[dict]) -> dict:
    """Stack a batch; keep ``has_identity`` / ``path`` as per-item lists (the loss gates per item)."""
    return {
        "ct": torch.stack([b["ct"] for b in batch], 0),
        "sem": torch.stack([b["sem"] for b in batch], 0),
        "owner": torch.stack([b["owner"] for b in batch], 0),
        "has_identity": [bool(b["has_identity"]) for b in batch],
        "path": [b["path"] for b in batch],
    }

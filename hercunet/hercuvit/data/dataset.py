"""HercuViT training dataset over the herculabels corpus — **reusing the hercunet export-labels pipeline**.

One sample = one herculabels window (``sample_*.npz``; pulled from ``jimmylomro/hercunet-corpus``). The npz holds
sheet **meshes** + confidence + quality (no dense CT), so the CT cube is streamed from the scroll via the data
layer (:class:`ScrollCtReader` → ``open_source``/``read_window``).

The **labels are built by the exact same functions ``hercunet train export-labels`` uses** — no bespoke
reimplementation (that was the earlier mistake):

  * :func:`hercunet.labels.mesh.medial.build_gradphi_edt` → ``mag`` (|∇φ|), ``lab3`` (band-limited sheet id),
    ``owner_full`` (dense owner, 0 = ignore);
  * :func:`hercunet.train.dataset._weight` → per-voxel confidence weight from the stored ``conf`` channels;
  * :func:`hercunet.train.dataset._build_surface_label` → the 3-class surface ``{0 bg, 1 surface, 2 ignore}``
    (surface = ``|∇φ| >= surface_tau``; **ignore = collapsed-sheet ∪ confidence (weight < 0.5)** — the
    confidence masking; ``confidence_ignore=False`` to MATCH the released (uncleaned) HercUNet training —
    the confidence field supervises a confidence head, it does not mask the loss);
  * :func:`hercunet.train.dataset._window_passes` → the window quality gate (``>= gate_count`` sheets ``>=
    gate_score``), applied at load time;
  * the instance owner = ``lab3 + 1`` with ``surf == 2 -> 0`` (ignore excluded from MALIS/M2F), exactly as
    ``export-labels`` writes ``ownersTr``.

HercuViT-specific (not a reimplementation of hercunet machinery, a genuine new requirement):
  * CT is **Dinovol robust-normalised** (:func:`dinovol_normalise`) — the frozen encoder's input distribution;
  * everything is **resampled to Dinovol's ~9.6 µm / 128³** working scale (the corpus is native 7-10 µm).

Augmentation (§8): Dinovol-style geometry (flips + axis permutation; no blur); CT + all label volumes share the
same transform.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# --- reused hercunet label pipeline: the SHARED per-window label builder `export_labels` also uses (one source,
#     no duplicated orchestration) + the window quality gate ---
from ...labels.io.bundle import load_sample_bundle
from ...train.dataset import _window_passes, build_surface_owner
from .normalize import dinovol_normalise

#: Dinovol working scale (``docs/hercuvit.md`` §2) — CT + labels are resampled to this voxel size and side.
TARGET_VOXEL_UM = 9.6
CROP = 128
SURFACE = 1
IGNORE = 2


# --------------------------------------------------------------------------- CT streaming (reused data layer)
class ScrollCtReader:
    """Stream a window's CT cube from its scroll through the reused data layer, caching the open volume per scroll."""

    def __init__(self) -> None:
        self._vols: dict = {}

    def _vol(self, scroll):
        key = str(scroll)
        if key not in self._vols:
            from ...infer.nnunet_infer import open_source
            self._vols[key] = open_source(key)
        return self._vols[key]

    def __call__(self, scroll, level, org, shape) -> np.ndarray:
        # ``org`` is the window CENTRE (z,y,x), as written by the label export — mirror hercunet's origin math
        # (train/dataset.py:_window_req, "same origin math as label generation's open_ct"): the read origin is
        # ``centre - shape//2`` clipped to the level's bounds. Treating ``org`` as the origin (no -shape//2) reads a
        # window offset by half a window → CT misaligned with the label on every axis.
        zc, yc, xc = (int(o) for o in org)
        dz, dy, dx = (int(s) for s in shape)
        vol = self._vol(scroll)
        Zf, Yf, Xf = vol.meta.level_shapes[int(level)]
        z0 = int(np.clip(zc - dz // 2, 0, max(0, Zf - dz)))
        y0 = int(np.clip(yc - dy // 2, 0, max(0, Yf - dy)))
        x0 = int(np.clip(xc - dx // 2, 0, max(0, Xf - dx)))
        blk, _origin = vol.read_window(int(level), z0, z0 + dz, y0, y0 + dy, x0, x0 + dx)
        return np.asarray(blk)


# --------------------------------------------------------------------------- pure helpers (unit-testable)
def resample_to_crop(ct: np.ndarray, labels: dict, voxel_um: float, *,
                     target_voxel_um: float = TARGET_VOXEL_UM, crop: int = CROP,
                     rng: np.random.Generator | None = None) -> tuple[np.ndarray, dict]:
    """Crop the native cube to Dinovol's physical FOV, then resize to ``crop``³ (CT trilinear, every label nearest).
    ``labels`` maps name -> ``(D,H,W)`` int volume. Returns ``(ct_crop, {name: label_crop})``.

    The FOV (``crop·target_voxel_um/voxel_um`` ≈ 155³ for a 128³@9.6µm crop of a 7.91µm window) is smaller than the
    native herculabels window (~220³), so a crop is needed to hold the 9.6µm scale. With ``rng`` the crop offset is
    RANDOM (whole native window is seen across epochs + a free spatial augmentation); without it the crop is CENTRED
    (deterministic — used for validation). Centre-only would keep ~35% of each window and never see the outer sheets."""
    ct = np.asarray(ct, dtype=np.float32)
    fov_vox = max(1, int(round(crop * target_voxel_um / float(voxel_um))))

    def _c(n, want):
        take = min(n, want)
        if rng is not None and n > take:
            lo = int(rng.integers(0, n - take + 1))
        else:
            lo = (n - take) // 2
        return slice(lo, lo + take)

    sz, sy, sx = _c(ct.shape[0], fov_vox), _c(ct.shape[1], fov_vox), _c(ct.shape[2], fov_vox)
    ct_t = torch.from_numpy(np.ascontiguousarray(ct[sz, sy, sx])).float()[None, None]
    ct_r = F.interpolate(ct_t, size=(crop, crop, crop), mode="trilinear", align_corners=False)[0, 0].numpy().astype(np.float32)
    out = {}
    for name, lab in labels.items():
        lt = torch.from_numpy(np.ascontiguousarray(np.asarray(lab)[sz, sy, sx]).astype(np.float32))[None, None]
        out[name] = F.interpolate(lt, size=(crop, crop, crop), mode="nearest")[0, 0].numpy().astype(np.int64)
    return ct_r, out


def geometric_augment(ct: np.ndarray, labels: dict, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Dinovol-style geometry on a cubic volume: per-axis flips + a random axis permutation. CT and every label
    get the identical transform. No blur, no intensity change."""
    flips = tuple(ax for ax in range(3) if rng.random() < 0.5)
    perm = rng.permutation(3)

    def _t(a):
        if flips:
            a = np.flip(a, flips)
        return np.ascontiguousarray(np.transpose(a, perm))

    return _t(ct), {k: _t(v) for k, v in labels.items()}


# --------------------------------------------------------------------------- dataset
class HercuvitDataset(Dataset):
    """herculabels corpus → Dinovol-normalised CT + confidence-masked surface ``{0,1,2}`` + instance owner at
    128³ / ~9.6 µm, using the hercunet export-labels pipeline.

    Args mirror ``export-labels``' knobs: ``surface_tau`` (|∇φ| threshold), ``ignore_score`` (collapsed-sheet
    quality cut), ``confidence_ignore`` (mask low-confidence surface voxels to ignore — False, matching hercunet), and the
    window gate ``gate_score`` / ``gate_count``. ``record`` = ``"base"`` (clean) by default.
    """

    def __init__(
        self,
        corpus_dir: str,
        *,
        record: str = "base",
        identity_only: bool = True,
        surface_tau: float = 0.91,
        ignore_score: float = 0.80,
        confidence_ignore: bool = False,
        gate_score: float = 0.85,
        gate_count: int = 2,
        target_voxel_um: float = TARGET_VOXEL_UM,
        crop: int = CROP,
        augment: bool = True,
        ct_reader: Optional[Callable] = None,
        seed: int = 0,
        ct_only: bool = False,
    ) -> None:
        self.ct_only = bool(ct_only)          # distillation: read only the CT (skip the expensive label build)
        self.corpus_dir = corpus_dir
        self.record = record
        self.surface_tau = float(surface_tau)
        self.ignore_score = float(ignore_score)
        self.confidence_ignore = bool(confidence_ignore)
        self.gate_score = float(gate_score)
        self.gate_count = int(gate_count)
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
            meta = self._peek_meta(p)
            per_sheet = meta.get("quality", {}).get("per_sheet", {})
            if not _window_passes(per_sheet, self.gate_score, self.gate_count):     # reused window quality gate
                continue
            if identity_only and not meta.get("records", {}).get(record, {}).get("cluster_ids"):
                continue
            self.samples.append(p)
        if not self.samples:
            raise FileNotFoundError(f"no gated samples (record {record!r}) under {corpus_dir!r}")

    @staticmethod
    def _peek_meta(path: str) -> dict:
        """Cheap meta-only read (no mesh decode) for the load-time gate."""
        with np.load(path, allow_pickle=True) as z:
            raw = z["meta_json"]
        return json.loads(bytes(np.asarray(raw).tolist()).decode("utf-8", "replace"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path = self.samples[idx]
        rng = np.random.default_rng(self.seed + idx)
        meta, recs = load_sample_bundle(path)                            # reused loader
        rec = recs[self.record]
        shape = tuple(meta["shape"]); vu = float(meta["voxel_um"])
        ident = bool(rec.get("meshes"))

        if self.ct_only:                                                 # distillation: CT crop only, no labels
            ct_native = self.ct_reader(meta.get("scroll"), int(meta.get("level", 0)),
                                       meta.get("org") or meta.get("coords"), shape)
            ct, _ = resample_to_crop(dinovol_normalise(ct_native), {}, vu, target_voxel_um=self.target_voxel_um,
                                     crop=self.crop, rng=(rng if self.augment else None))
            if self.augment:
                ct, _ = geometric_augment(ct, {}, rng)
            return {"ct": torch.from_numpy(np.ascontiguousarray(ct)).float().unsqueeze(0), "path": path}

        # labels — the SHARED per-window builder `export_labels` also uses (one source, no duplication)
        surf, inst, _mag = build_surface_owner(rec, meta, surface_tau=self.surface_tau, ignore_score=self.ignore_score,
                                               confidence_ignore=self.confidence_ignore)

        # CT (Dinovol robust norm) then resample CT + labels to 128³ / 9.6 µm
        ct_native = self.ct_reader(meta.get("scroll"), int(meta.get("level", 0)),
                                   meta.get("org") or meta.get("coords"), shape)
        if tuple(ct_native.shape) != tuple(int(s) for s in shape):
            raise ValueError(f"{path}: CT read shape {ct_native.shape} != meta shape {tuple(shape)}")
        ct_norm = dinovol_normalise(ct_native)
        # random crop offset when augmenting (whole ~220³ window seen over epochs); centre-cropped for val.
        ct, labs = resample_to_crop(ct_norm, {"surf": surf, "owner": inst}, vu,
                                    target_voxel_um=self.target_voxel_um, crop=self.crop,
                                    rng=(rng if self.augment else None))
        if self.augment:
            ct, labs = geometric_augment(ct, labs, rng)

        return {
            "ct": torch.from_numpy(np.ascontiguousarray(ct)).float().unsqueeze(0),       # (1,128,128,128)
            "sem": torch.from_numpy(np.ascontiguousarray(labs["surf"])).long(),          # {0 bg,1 surface,2 ignore}
            "owner": torch.from_numpy(np.ascontiguousarray(labs["owner"])).long(),       # instance ids (0 = bg/ignore)
            "has_identity": bool(ident),
            "path": path,
        }


def split_cache(cache_dir: str, *, fold: int = 0, n_splits: int = 5, seed: int = 12345):
    """Train/val split of a window cache, **consistent with hercunet/nnU-Net** — ``KFold(n_splits=5, shuffle=True,
    random_state=12345)`` over the SORTED ``win_*.npz`` keys, returning the ``fold``-th (train_files, val_files).
    A plain per-window split (every scroll stays in training; validation is held-out windows, not held-out scrolls),
    the same convention nnU-Net uses for its 5-fold ``splits_final.json``. Deterministic: same split every run."""
    from sklearn.model_selection import KFold
    files = sorted(glob.glob(os.path.join(cache_dir, "win_*.npz")))
    if not files:
        raise FileNotFoundError(f"no win_*.npz under {cache_dir!r} — run the training pre-extraction first")
    splits = list(KFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed)).split(files))
    tr_idx, va_idx = splits[int(fold) % len(splits)]
    return [files[i] for i in tr_idx], [files[i] for i in va_idx]


class LocalHercuvitDataset(Dataset):
    """Training dataset backed by a LOCAL pre-extracted cache (``train prep-shard`` → native ``.npz`` windows).

    Each cached file holds one window's **native** CT (Dinovol-normalised float16) + native ``surf`` ``{0,1,2}`` +
    native instance ``owner`` — everything expensive (the S3 CT read + the ``build_gradphi_edt`` label build) done
    once. ``__getitem__`` only does the cheap per-epoch work: :func:`resample_to_crop` (random crop offset when
    augmenting) + :func:`geometric_augment`, returning the SAME dict as :class:`HercuvitDataset`. Reading from local
    ``.npz`` is fork/concurrency-safe, so a ``num_workers > 0`` DataLoader is finally safe here (the live S3 path
    deadlocks under worker concurrency — see ``train/preextract.py``).

    Augmentation is fresh per call (``default_rng()``), so every epoch sees a different crop offset + flip of each
    window — a genuine improvement over the live dataset's ``seed + idx`` seeding, which froze one view per sample."""

    def __init__(self, cache_dir: str, *, augment: bool = True, target_voxel_um: float = TARGET_VOXEL_UM,
                 crop: int = CROP, files: "list[str] | None" = None) -> None:
        self.cache_dir = cache_dir
        self.augment = bool(augment)
        self.target_voxel_um = float(target_voxel_um)
        self.crop = int(crop)
        # ``files`` lets the caller pass a train/val SUBSET (see :func:`split_cache`); else use the whole cache.
        self.files = list(files) if files is not None else sorted(glob.glob(os.path.join(cache_dir, "win_*.npz")))
        if not self.files:
            raise FileNotFoundError(f"no win_*.npz under {cache_dir!r} — run the training pre-extraction first")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx]
        with np.load(path, allow_pickle=False) as z:
            ct_norm = np.asarray(z["ct"], dtype=np.float32)
            surf = np.asarray(z["surf"])
            owner = np.asarray(z["owner"])
            vu = float(z["voxel_um"])
            ident = bool(z["has_identity"])
        rng = np.random.default_rng() if self.augment else None           # fresh entropy → per-epoch variety
        ct, labs = resample_to_crop(ct_norm, {"surf": surf, "owner": owner}, vu,
                                    target_voxel_um=self.target_voxel_um, crop=self.crop, rng=rng)
        if self.augment:
            ct, labs = geometric_augment(ct, labs, rng)
        return {
            "ct": torch.from_numpy(np.ascontiguousarray(ct)).float().unsqueeze(0),       # (1,128,128,128)
            "sem": torch.from_numpy(np.ascontiguousarray(labs["surf"])).long(),          # {0 bg,1 surface,2 ignore}
            "owner": torch.from_numpy(np.ascontiguousarray(labs["owner"])).long(),       # instance ids (0 = bg/ignore)
            "has_identity": ident,
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

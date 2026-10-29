"""Stitcher training data — the per-corner samples that ``hercuvit labels create`` writes (docs §5/§7).

Each ``stitch_*.npz`` is one corner: overlap-kept per-window normalised-PCA spatial tokens (``Xs/Ps/slot_s``),
fragment median-embedding queries (``Xq/slot_q/conf_q``), and the same-object label matrix ``Y`` (1/0/-1 ignore)
from overlap-voting against the centred pass. :class:`StitchSampleDataset` returns one corner's variable-length
tensors; :func:`stitch_collate` pads a batch to ``S_max`` / ``M_max`` with boolean validity masks.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def split_samples(files, *, fold: int = 0, n_splits: int = 5, seed: int = 12345):
    """Train/val split of the sample files — same convention as the HercuViT window cache
    (``KFold(5, shuffle=True, random_state=12345)`` over the SORTED files): returns the ``fold``-th
    ``(train_files, val_files)``. Deterministic; every corner stays assigned to the same side across runs."""
    from sklearn.model_selection import KFold
    files = sorted(files)
    if len(files) < n_splits:                      # too few to fold — everything trains, nothing validates
        return list(files), []
    splits = list(KFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed)).split(files))
    tr, va = splits[int(fold) % len(splits)]
    return [files[i] for i in tr], [files[i] for i in va]


def find_samples(samples_dir: str):
    """Sorted list of ``stitch_*.npz`` under a ``labels create`` output dir."""
    files = sorted(glob.glob(os.path.join(samples_dir, "stitch_*.npz")))
    if not files:
        raise FileNotFoundError(f"no stitch_*.npz under {samples_dir!r} — run `hercuvit labels create` first")
    return files


class StitchSampleDataset(Dataset):
    """One ``stitch_*.npz`` corner per item → a dict of tensors (variable ``S``, ``M``). ``pca_dim`` is asserted
    against each file so a config/sample mismatch fails loudly rather than silently mis-shaping the decoder."""

    def __init__(self, files, *, pca_dim: int) -> None:
        self.files = list(files)
        self.pca_dim = int(pca_dim)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx]
        with np.load(path, allow_pickle=False) as z:
            Xs = np.asarray(z["Xs"], dtype=np.float32)            # (S, D)
            Ps = np.asarray(z["Ps"], dtype=np.float32)            # (S, 3)
            slot_s = np.asarray(z["slot_s"], dtype=np.int64)      # (S,)
            Xq = np.asarray(z["Xq"], dtype=np.float32)            # (M, D)
            slot_q = np.asarray(z["slot_q"], dtype=np.int64)      # (M,)
            conf_q = np.asarray(z["conf_q"], dtype=np.float32)    # (M,)
            Y = np.asarray(z["Y"], dtype=np.int64)                # (M, M) in {1, 0, -1}
        if Xs.shape[1] != self.pca_dim or Xq.shape[1] != self.pca_dim:
            raise ValueError(f"{os.path.basename(path)}: embedding dim {Xs.shape[1]} != config pca_dim "
                             f"{self.pca_dim} — re-extract with `labels create --pca-dim {self.pca_dim}` "
                             f"or train with --pca-dim {Xs.shape[1]}")
        return {"Xs": torch.from_numpy(Xs), "Ps": torch.from_numpy(Ps), "slot_s": torch.from_numpy(slot_s),
                "Xq": torch.from_numpy(Xq), "slot_q": torch.from_numpy(slot_q), "conf_q": torch.from_numpy(conf_q),
                "Y": torch.from_numpy(Y), "path": path}


def stitch_collate(batch) -> dict:
    """Pad a batch to ``S_max`` spatial tokens and ``M_max`` queries; build boolean validity masks. Padded spatial
    tokens / queries are masked out as attention keys (model) and as loss pairs (loss). ``Y`` pads with -1 (ignore)."""
    B = len(batch)
    D = batch[0]["Xs"].shape[1]
    S = max(b["Xs"].shape[0] for b in batch)
    M = max(b["Xq"].shape[0] for b in batch)

    Xs = torch.zeros(B, S, D)
    Ps = torch.zeros(B, S, 3)
    slot_s = torch.zeros(B, S, dtype=torch.long)
    valid_s = torch.zeros(B, S, dtype=torch.bool)
    Xq = torch.zeros(B, M, D)
    slot_q = torch.zeros(B, M, dtype=torch.long)
    conf_q = torch.zeros(B, M)
    valid_q = torch.zeros(B, M, dtype=torch.bool)
    Y = torch.full((B, M, M), -1, dtype=torch.long)

    for i, b in enumerate(batch):
        s, m = b["Xs"].shape[0], b["Xq"].shape[0]
        Xs[i, :s] = b["Xs"]; Ps[i, :s] = b["Ps"]; slot_s[i, :s] = b["slot_s"]; valid_s[i, :s] = True
        Xq[i, :m] = b["Xq"]; slot_q[i, :m] = b["slot_q"]; conf_q[i, :m] = b["conf_q"]; valid_q[i, :m] = True
        Y[i, :m, :m] = b["Y"]

    return {"Xs": Xs, "Ps": Ps, "slot_s": slot_s, "valid_s": valid_s,
            "Xq": Xq, "slot_q": slot_q, "conf_q": conf_q, "valid_q": valid_q,
            "Y": Y, "path": [b["path"] for b in batch]}

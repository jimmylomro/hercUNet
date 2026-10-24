"""``HercuvitSurfaceModel`` — the v2 decoupled HercuViT surface net as an :class:`~hercunet.infer.model.InferenceModel`,
so it runs on the SAME engine as the nnU-Net/Jacobi model (streaming, tiling, Gaussian blend, claim-queue, leader
finalise + S3). Single-pass (no Jacobi prev feedback), 1-channel Dinovol-normalised CT, surface-only output.

The blend is on the SURFACE (``sem_logits[:,1]-sem_logits[:,0]`` → Gaussian-blend → sigmoid); per-sheet masks are NOT
emitted here (the decoupled instance head is built off at load, ``with_instance=False``).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ...infer.model import InferenceModel, OutputSpec


class HercuvitSurfaceModel(InferenceModel):
    name = "hercuvit-surface"

    def __init__(self, net, p_size, *, overlap=0.25):
        self.net = net
        self._p = int(p_size)
        self._overlap = float(overlap)

    @classmethod
    def load(cls, *, model=None, model_hf=None, ckpt="checkpoint_best.pth", device="cuda",
             attention_backend=None, overlap=0.25, **_):
        from .loader import load_hercuvit
        net, recipe, _cfg = load_hercuvit(model, model_hf=model_hf, ckpt=ckpt, device=device,
                                          attention_backend=attention_backend, with_instance=False)
        return cls(net, int(recipe.window), overlap=overlap)

    @property
    def patch_size(self) -> int:
        return self._p

    @property
    def passes(self) -> int:
        return 1                              # single-pass: no Jacobi prev feedback

    @property
    def overlap(self) -> float:
        return self._overlap

    @property
    def output(self) -> OutputSpec:
        return OutputSpec(name="hercuvit-sem")

    @property
    def train_voxel_band(self):
        return (7.5, 10.0)                    # Dinovol ~9.6 µm working scale

    def norm_ct(self, sub: np.ndarray) -> np.ndarray:
        from ..data.normalize import dinovol_normalise
        return dinovol_normalise(sub)

    def base_input(self, ct_norm: np.ndarray, prev: Optional[np.ndarray]) -> np.ndarray:
        return ct_norm[None].astype(np.float32)   # [1,P,P,P] — single-channel; prev ignored (single-pass)

    def forward(self, xb_base):
        """``[B,1,P,P,P]`` → ``(sem_logits[B,2,P,P,P], None)`` — the decoupled surface head; no aux."""
        out = self.net(xb_base, run_instance=False)
        return out["sem_logits"][0], None

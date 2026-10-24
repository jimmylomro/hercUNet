"""Model-agnostic inference contract — the plug-in that lets ONE engine (:mod:`hercunet.infer.engine`) serve many
models (nnU-Net/Jacobi, HercuViT, future nets) without duplicating the streaming / tiling / Gaussian-blend /
claim-queue / leader-finalise / S3 machinery.

The engine owns the whole lifecycle (sub-volume or full-volume region, streaming + optional pre-sync, overlapping
tiles, prefetch, multi-pass Jacobi, work-stealing ``.done`` claim-queue across GPUs/pods, the per-pass barrier,
leader-automatic pyramid finalise + S3 upload, buffer pruning, resume). A model supplies ONLY the five things that
actually differ between nets, through this ABC:

  * ``load`` — build the net + read its patch size / output semantics from a checkpoint.
  * ``norm_ct`` — per-window CT normalisation (nnU-Net percentile fingerprint vs Dinovol robust).
  * ``base_input`` — the per-window net input BEFORE any TTA/orientation (``[ct, prev]`` vs ``[ct]``).
  * ``forward`` — one forward on a (possibly TTA-transformed) base input → the 2-class **surface logits** the engine
    blends (``logit[:,1]-logit[:,0]`` → Gaussian-blend → sigmoid) + optional **aux** channels (e.g. affinity).
  * the static config properties (``patch_size``, ``passes``, ``uses_prev``, ``overlap``, output specs, train band).

The blend is deliberately expressed on the surface **logit margin**, which is model-independent, so TTA (flip/
transpose the base input, invert the output) and the Jacobi prev-channel are handled generically — the model's
``forward`` just recomputes anything input-derived (e.g. structure-tensor orientation) from the transformed CT.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class OutputSpec:
    """How the engine should create + name the primary surface OME-Zarr. ``channels`` is 1 (a uint8 surface prob)."""
    name: str = "surface"            # VC3D volume-name suffix / log label
    dtype: str = "uint8"


@dataclass(frozen=True)
class AuxSpec:
    """An optional extra output (e.g. the affinity head): a 4-D ``(channels, Z, Y, X)`` uint8 zarr, emitted ONLY on
    the final pass and blended in PROBABILITY space with the same Gaussian weights as the surface."""
    name: str                        # file suffix, e.g. "aff"
    channels: int
    offsets: Optional[list] = None   # recorded in the zarr attrs (affinity offset table), or None
    final_pass_only: bool = True


# --------------------------------------------------------------------------- registry ----------------------------
_REGISTRY: dict[str, type["InferenceModel"]] = {}


class InferenceModel(ABC):
    """Contract between a model and :class:`hercunet.infer.engine.InferenceEngine`. Subclasses set ``name`` and are
    auto-registered (``InferenceModel.get(name)``) so a CLI can pick a model by string. An instance holds the loaded
    net + device and is used by every worker process (one per GPU)."""

    #: unique model key for the registry / CLI (e.g. "nnunet-jacobi", "hercuvit-surface"); set on the subclass.
    name: str = ""

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if getattr(cls, "name", ""):
            _REGISTRY[cls.name] = cls

    @classmethod
    def get(cls, name: str) -> type["InferenceModel"]:
        if name not in _REGISTRY:
            raise SystemExit(f"infer: unknown model {name!r}; known: {sorted(_REGISTRY)}")
        return _REGISTRY[name]

    # --- construction --------------------------------------------------------------------------------------------
    @classmethod
    @abstractmethod
    def load(cls, *, model=None, model_hf=None, ckpt: str, device: str = "cuda", **opts) -> "InferenceModel":
        """Resolve + build the trained net (local dir/file or HF repo) on ``device`` and return a ready instance."""

    # --- static config (read off the loaded checkpoint) ----------------------------------------------------------
    @property
    @abstractmethod
    def patch_size(self) -> int:
        """Cubic window side P (voxels)."""

    @property
    def passes(self) -> int:
        """Jacobi refinement passes (>=1). 1 = single-pass (no prev feedback), the default for most models."""
        return 1

    @property
    def uses_prev(self) -> bool:
        """Whether ``base_input`` consumes the previous pass's surface (Jacobi). False → the engine never reads it."""
        return self.passes > 1

    @property
    def overlap(self) -> float:
        """Window overlap for the Gaussian blend (0 → disjoint tiles). Default 0.25 (the HercUNet v0 setting)."""
        return 0.25

    @property
    def output(self) -> OutputSpec:
        return OutputSpec()

    @property
    def aux(self) -> Optional[AuxSpec]:
        """An extra output (affinity), or None. Emitted only on the final pass."""
        return None

    @property
    def train_voxel_band(self) -> Optional[tuple]:
        """``(lo_um, hi_um)`` the net was trained on, for the out-of-domain resolution warning; None to skip it."""
        return None

    # --- per-window hooks (hot path) -----------------------------------------------------------------------------
    @abstractmethod
    def norm_ct(self, sub: np.ndarray) -> np.ndarray:
        """Normalise one raw ``(P,P,P)`` CT window to the net's expected input scale (float32)."""

    @abstractmethod
    def base_input(self, ct_norm: np.ndarray, prev: Optional[np.ndarray]) -> np.ndarray:
        """Stack the per-window net input BEFORE TTA/orientation: ``(Cbase, P, P, P)`` float32. ``prev`` is the
        previous pass's surface prob in [0,1] (``None`` on pass 0 / when ``uses_prev`` is False)."""

    @abstractmethod
    def forward(self, xb_base):
        """One forward on a ``(B, Cbase, P, P, P)`` tensor (already on device, under the engine's autocast) →
        ``(surface_logits, aux)`` where ``surface_logits`` is ``(B, 2, P, P, P)`` (the engine blends
        ``[:,1]-[:,0]``) and ``aux`` is ``(B, Caux, P, P, P)`` in [0,1] or ``None``. Any input-derived feature
        (e.g. structure-tensor orientation) is recomputed HERE from ``xb_base`` so TTA stays correct."""

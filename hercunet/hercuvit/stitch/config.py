"""The stitcher's config — the single source of truth for its hyperparameters (docs/hercuvit-stitching.md §2/§7).

The stitcher is a ~1 M-param standalone decoder, so (unlike the big HercuViT recipe) one lean frozen dataclass
carries BOTH its architecture and its training schedule. The field defaults ARE the spec's §2 table and §7.8
optimiser; every knob is overridable from ``hercuvit stitcher train`` flags, and the resolved config is saved in the
checkpoint so inference rebuilds the exact decoder. No YAML, no env vars — British spelling, long ``--flags``.
"""
from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class StitchConfig:
    # --- architecture (docs §2) ---
    pca_dim: int = 3            # D — stored embedding dim; MUST match the samples' `labels create --pca-dim`
    d_model: int = 128         # d — decoder width
    heads: int = 4             # h — attention heads (d_h = d/h = 32)
    layers: int = 4            # L — decoder layers (deep-supervised)
    fourier_bands: int = 8     # F — position Fourier bands (→ 6F position features)

    # --- loss (focal BCE + SupCon contrastive, applied TOGETHER — experimental, not flag-gated) ---
    topk: int = 10             # k — edges kept per fragment at inference (unused in training)
    pos_weight: float = 0.0    # focal-BCE positive-class weight; <=0 = auto (neg/pos ratio measured per batch)
    focal_gamma: float = 2.0   # focal focusing exponent on hard pairs (0 = plain pos-weighted BCE)
    contrast_tau: float = 0.1  # SupCon temperature on the L2-normalised query embeddings
    contrast_weight: float = 1.0  # weight of the SupCon term added to the focal BCE (0 = off)

    # --- optimiser / schedule (docs §7.8) ---
    epochs: int = 100
    batch_size: int = 16       # corners per batch (B)
    base_lr: float = 1.0e-4
    weight_decay: float = 0.05  # not applied to LayerNorm, biases, s, b, E_slot
    grad_clip: float = 1.0
    warmup_steps: int = 200
    lr_schedule: str = "cosine"  # "cosine" (default, §7.8) | "constant"

    # --- data / validation ---
    val_fold: int = 0          # KFold(5, seed 12345) over the sorted sample files — a held-out fold
    val_interval: int = 5      # validate edge AP every N epochs (0 = no validation)

    def summary(self) -> str:
        """One-line ``field=value`` dump for the per-run audit log."""
        return "  ".join(f"{f.name}={getattr(self, f.name)}" for f in fields(self))


#: Config fields settable from ``hercuvit stitcher train`` flags (all of them; unset flags keep the defaults above).
CONFIG_FIELDS = tuple(f.name for f in fields(StitchConfig))


def config_from_dict(d: dict) -> StitchConfig:
    """Rebuild a :class:`StitchConfig` from a saved ``asdict`` (ignoring unknown keys from a newer/older schema)."""
    valid = {f.name for f in fields(StitchConfig)}
    return StitchConfig(**{k: v for k, v in (d or {}).items() if k in valid})

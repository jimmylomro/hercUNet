"""The HercuViT training recipe — the single auditable source of truth for its hyperparameters.

Mirrors ``hercunet/training/recipe.py``: a frozen dataclass schema, the values shipped in a YAML
(``hercunet/hercuvit/recipes/hercuvit.yml``, installed as package data), and the
``active() / set_active() / load()`` plumbing the trainer reads. No environment variables — every knob is a
field, overridable with ``hercuvit train fit --recipe <your.yml>`` (omitted fields fall back to the shipped
value). The trainer logs the resolved recipe at the start of each run (``[recipe] hercuvit: …``).
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path

#: Attention SDPA backends the retrofit may run under (see ``netbuild.apply_attention_backend``). ``auto`` lets
#: torch pick (flash/efficient/math) — REQUIRED on GPUs with no cuDNN plan for the windowed shapes (e.g. RTX
#: 4060/8 GB): the Dinovol config pins ``cudnn``, which raises "No execution plans support the graph" there.
ATTENTION_BACKENDS = ("auto", "flash", "cudnn", "efficient", "math")

#: The three HercuViT architecture variants (``docs/hercuvit.md`` §0/§3): ``small`` = the BUILT ~27 M full-weight
#: windowed retrofit of Dinovol; ``tiny`` (~13-14 M) / ``nano`` (~6-8 M) = distilled slim transformers (built via
#: ``hercuvit distill``). The per-variant architecture NUMBERS live in the single source of truth
#: :data:`hercunet.hercuvit.model.hercuvit_net.VARIANT_ARCH` — this recipe only selects the ``variant`` name and
#: may carry an ``arch_override`` dict to tweak individual fields over that canonical preset.
VARIANTS = ("small", "tiny", "nano")


@dataclass(frozen=True)
class HercuvitRecipe:
    """A HercuViT training recipe. Field values below are the shipped ``hercuvit.yml`` defaults."""

    name: str = "hercuvit"

    #: architecture variant name (small | tiny | nano); the arch numbers come from
    #: ``hercuvit_net.VARIANT_ARCH[variant]`` (single source of truth).
    variant: str = "small"
    #: optional per-field tweaks merged OVER ``VARIANT_ARCH[variant]`` (e.g. {"depth": 9}); empty by default.
    arch_override: dict = field(default_factory=dict)

    # --- architecture (must match the pillars / Dinovol config; passed to build_hercuvit) ---
    window: int = 128                               # input cube side (voxels) = Dinovol crop
    global_blocks: tuple = (4, 7, 10)               # 1-indexed blocks that keep GLOBAL attention (pillar 1)
    window_size: tuple = (8, 8, 8)                  # windowed-attention window in tokens (8³ → 2×2×2 windows)
    n_sem_classes: int = 2                          # background / sheet-surface
    mask_dim: int = 32                              # shared mask-feature width (Dinovol embedding width)
    n_queries: int = 64                             # instance queries (top-K objectness tokens)
    query_dim: int = 64                             # query decoder width
    instance_layers: int = 3                        # query-decoder layers
    deep_supervision: bool = True

    # --- attention backend (plumbing; NOT a pillar change — raised via flag, docs §12) ---
    attention_backend: str = "auto"

    # --- data ---
    target_voxel_um: float = 9.6                    # Dinovol working scale
    augment: bool = True
    batch_size: int = 1

    # --- optimiser / schedule ---
    base_lr: float = 1.0e-4
    transformer_lr_mult: float = 0.1                # transformer blocks train at 0.1× (after unfreeze)
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    warmup_steps: int = 200
    lr_schedule: str = "constant"                   # "constant" (Dinovol convention) | "cosine"

    # --- staged schedule (§7): semantic → unfreeze transformer (0.1×) → add instance ---
    semantic_epochs: int = 50                       # stage 1: transformer frozen, stem+decoder+sem head
    unfreeze_epochs: int = 25                       # stage 2: all trainable, transformer 0.1×, windows ON
    instance_epochs: int = 100                      # stage 3: + instance head, joint

    # --- semantic loss (thin medial-surface stack, ported from HercUNet — see train/surface_loss.py) ---
    sem_fg_weight: float = 30.0   # foreground CE weight — medial surfaces are ~0.1-0.3% of voxels; unweighted CE
                                  # collapses to all-background, so up-weight the surface class.
    w_sem_ce: float = 1.0
    w_sem_tversky: float = 1.0    # Focal-Tversky (precision/imbalance), replaces plain Dice
    w_sem_skel: float = 1.0       # clDice skeleton-recall (cover the GT medial skeleton)
    w_sem_sep: float = 0.5        # separation penalty (anti-merge: P(surface) in inter-sheet gaps)
    tversky_alpha: float = 0.5    # champion = symmetric (α=β=0.5) so Tversky = precision; skel-recall does recall
    tversky_beta: float = 0.5
    tversky_gamma: float = 1.0    # >1 = focal-Tversky
    skel_iters: int = 3
    sep_radius: int = 2
    deep_sup_weights: tuple = (1.0, 0.5, 0.25)      # full, 1/2, 1/4

    # --- instance loss (Mask2Former set prediction) ---
    w_class: float = 2.0                            # sheet / no-object classification
    w_mask_bce: float = 5.0
    w_mask_dice: float = 5.0
    w_objectness: float = 1.0                       # token two-stage objectness
    cost_class: float = 2.0                         # Hungarian matching costs
    cost_mask_bce: float = 5.0
    cost_mask_dice: float = 5.0
    no_object_weight: float = 0.1                   # down-weight the no-object class (DETR)
    aux_weight: float = 1.0                         # weight on each earlier decoder layer's aux loss
    point_sample: int = 2048                        # points for point-sampled mask loss (PointRend/M2F)
    oversample_ratio: float = 3.0
    importance_ratio: float = 0.75

    # --- distillation (config only; the logic is lead-owned — model/variants.py, model/distill.py,
    #     train/distill.py). Reserved surface for building hercuvit-tiny/nano from a trained small teacher (§8.1). ---
    distill_teacher: str = "scrollprize/dinovol_ps8_9.6um_101626"   # HF repo or ckpt path of the teacher
    distill_student_variant: str = "tiny"           # tiny | nano — the slim student to distil
    distill_gram_weight: float = 1.0                # Gram / feature-correlation distillation weight
    distill_token_weight: float = 1.0               # per-token feature distillation weight
    distill_cls_weight: float = 1.0                 # cls / register-token distillation weight
    distill_steps: int = 100000
    distill_lr: float = 1.0e-4

    def n_stages(self) -> int:
        return 3

    def summary(self) -> str:
        """One-line ``field=value`` dump (minus ``name``) for the per-run audit log."""
        return "  ".join(f"{f.name}={getattr(self, f.name)}" for f in fields(self) if f.name != "name")


_BUILTIN = "hercuvit"
_TUPLE_FIELDS = {"global_blocks", "window_size", "deep_sup_weights"}


def _from_yaml(text: str, source: str = "<recipe>") -> HercuvitRecipe:
    """Build a :class:`HercuvitRecipe` from YAML (a ``field: value`` mapping on top of the schema defaults)."""
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("hercuvit: PyYAML is required for training recipes — install the hercuvit extra "
                         "(pip install -e '.[hercuvit]').") from e
    over = yaml.safe_load(text) or {}
    if not isinstance(over, dict):
        raise SystemExit(f"hercuvit recipe {source}: expected a YAML mapping of field: value.")
    valid = {f.name for f in fields(HercuvitRecipe)}
    bad = sorted(k for k in over if k not in valid)
    if bad:
        raise SystemExit(f"hercuvit recipe {source}: unknown field(s) {bad}. Valid fields: {sorted(valid)}.")
    for k in _TUPLE_FIELDS:
        if over.get(k) is not None:
            over[k] = tuple(over[k])
    if over.get("attention_backend") is not None and over["attention_backend"] not in ATTENTION_BACKENDS:
        raise SystemExit(f"hercuvit recipe {source}: attention_backend must be one of {ATTENTION_BACKENDS}.")
    if over.get("variant") is not None and over["variant"] not in VARIANTS:
        raise SystemExit(f"hercuvit recipe {source}: variant must be one of {VARIANTS}.")
    return replace(HercuvitRecipe(), **over)


def _load_builtin(name: str = _BUILTIN) -> HercuvitRecipe:
    from importlib.resources import files
    res = files("hercunet.hercuvit.recipes").joinpath(f"{name}.yml")
    return _from_yaml(res.read_text(), source=f"hercunet/hercuvit/recipes/{name}.yml")


def load(path=None) -> HercuvitRecipe:
    """The recipe to train with: the shipped ``hercuvit.yml`` when ``path`` is None, else the variant at ``path``."""
    if not path:
        return HERCUVIT
    return _from_yaml(Path(path).read_text(), source=str(path))


#: The default recipe, loaded from the shipped YAML.
HERCUVIT = _load_builtin()

_ACTIVE = HERCUVIT


def active() -> HercuvitRecipe:
    """The recipe the trainer should use right now (the default unless a variant was made active)."""
    return _ACTIVE


def set_active(recipe: HercuvitRecipe) -> None:
    """Install ``recipe`` as active (called by ``fit`` before constructing the trainer)."""
    global _ACTIVE
    _ACTIVE = recipe

"""The HercuViT training recipe — the single auditable source of truth for its hyperparameters.

Mirrors ``hercunet/training/recipe.py``: a frozen dataclass schema, the values shipped as per-variant YAMLs
(``hercunet/hercuvit/recipes/{small,tiny,nano}.yml``, installed as package data; ``small`` is the default), and the
``active() / set_active() / load()`` plumbing the trainer reads. No environment variables — every knob is a
field, overridable with ``hercuvit train fit --recipe <your.yml>`` (omitted fields fall back to the shipped
value). The trainer logs the resolved recipe at the start of each run (``[recipe] hercuvit: …``).
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Optional

#: Attention SDPA backends the retrofit may run under (see ``netbuild.apply_attention_backend``). ``auto`` lets
#: torch pick (flash/efficient/math) — REQUIRED on GPUs with no cuDNN plan for the windowed shapes (e.g. RTX
#: 4060/8 GB): the Dinovol config pins ``cudnn``, which raises "No execution plans support the graph" there.
ATTENTION_BACKENDS = ("auto", "flash", "cudnn", "efficient", "math")

#: The HercuViT architecture variants (``docs/hercuvit.md`` §0/§3), each a SHIPPED MODEL YAML
#: (``recipes/<variant>.yml``): ``full`` = raw Dinovol loaded AS IS (all-global attention, no windowing, teacher
#: size); ``small`` = the full-weight windowed retrofit; ``tiny`` / ``nano`` = distilled slim transformers (built
#: via ``hercuvit distill``). Reading the yaml tells you the whole network; there is no hidden preset dict.
VARIANTS = ("full", "small", "tiny", "nano")

#: The recipe fields that make up the ENCODER half of the architecture (assembled by :meth:`HercuvitRecipe.encoder_arch`
#: into the ``arch_override`` dict that ``build_hercuvit`` consumes). ``None`` fields are omitted (taken from the .pt).
_ENCODER_ARCH_FIELDS = ("embed_dim", "depth", "num_heads", "mlp_ratio",
                        "global_blocks", "window_size", "global_attention", "sr_factor")


@dataclass(frozen=True)
class HercuvitRecipe:
    """A HercuViT training recipe. Field defaults below are the ``small`` variant; ``recipes/{small,tiny,nano}.yml``
    each fully spell out their network."""

    name: str = "hercuvit"

    #: architecture variant NAME — a label for this recipe (small | tiny | nano). It no longer *points* at a hidden
    #: preset: **this recipe IS the network definition** (the ``# --- encoder arch ---`` + decoder/head fields
    #: below fully specify it). The rule: training FROM SCRATCH, the net comes from these yaml fields; training
    #: from an ``--encoder`` .pt, the ENCODER comes from the .pt and the yaml's encoder fields are validated
    #: against it (mismatch = error), while the decoder + heads are still built fresh from the yaml.
    variant: str = "small"
    #: optional ad-hoc per-field tweaks merged OVER the assembled encoder arch (e.g. {"depth": 9}); empty by default.
    arch_override: dict = field(default_factory=dict)

    # --- encoder arch (the ENCODER half of the network definition; None = "take it from the encoder .pt / teacher",
    #     which is how `small` works — it is the Dinovol teacher, windowed. nano/tiny spell these out because they
    #     are SLIMMER than the teacher and are built by `hercuvit distill` from exactly these numbers). ---
    embed_dim: Optional[int] = None                 # token width (teacher 288; nano 192, tiny 288)
    depth: Optional[int] = None                     # transformer blocks (teacher 10; nano 6, tiny 8)
    num_heads: Optional[int] = None                 # attention heads (teacher 6; nano 4, tiny 6)
    mlp_ratio: Optional[float] = None               # MLP hidden / embed ratio (teacher's; nano 512/192, tiny 768/288)
    global_attention: str = "full"                  # "full" (small) | "spatial_reduction" (nano/tiny windowed blocks)
    sr_factor: tuple = (2, 2, 2)                    # spatial-reduction factor for the global blocks (nano/tiny)

    # --- architecture (shared; must match the pillars / Dinovol config; passed to build_hercuvit) ---
    window: int = 128                               # input cube side (voxels) = Dinovol crop
    global_blocks: tuple = (4, 7, 10)               # 1-indexed blocks that keep GLOBAL attention (pillar 1)
    window_size: tuple = (8, 8, 8)                  # windowed-attention window in tokens (8³ → 2×2×2 windows)
    decoder_channels: Optional[tuple] = None        # pixel-decoder widths (1/8,1/4,1/2,1/1); None = (192,128,48,16).
                                                    # Only the 1/8 width is free; (1/4,1/2,1/1) are LOCKED to the stem
                                                    # skips (128,48,16). nano slims the 1/8 to 128.
    n_sem_classes: int = 2                          # background / sheet-surface
    mask_dim: int = 32                              # shared mask-feature width (Dinovol embedding width)
    n_queries: int = 64                             # instance queries
    query_dim: int = 64                             # query decoder width
    instance_layers: int = 3                        # query-decoder layers
    deep_supervision: bool = True
    learned_queries: bool = True                    # persistent learned queries (DETR-style) — specialise, stable match
    fps_queries: bool = False                       # embedding-FPS anchored query init (§13.3); excludes learned_queries
    cross_attn: str = "deformable"                  # query-decoder cross-attn: "deformable" (anchor) | "masked" (§13.6)
    mask_mode: str = "sigmoid"                       # mask readout: "sigmoid" (independent M2F masks) | "softmax"
                                                    # (per-voxel ownership over queries, null=background — exclusive)
    add_null_query: bool = True                     # reserved null/no-sheet query (index 0) for the softmax ownership

    # --- pixel decoder (v2 "decoupled", docs/hercuvit-mask-architecture.md §10) ---
    decoder: str = "decoupled"                      # "decoupled": CNN-UNET pixel encoder (concat stem skips → U1@1/1 =
                                                    #   mask_features + a 1×1 surface head = objectness). The only kind.
    neck: str = "none"                              # patchdecode only: "deformable" (6× MS-deformable-attn over
                                                    #   {1/8,1/4}, part A) | "none" (author-exact: tokens → upsample)
    neck_layers: int = 6                            # deformable-neck layers
    neck_heads: int = 4                             # deformable-neck attention heads
    neck_points: int = 4                            # deformable-neck sampling points per (head, scale)
    surface_gate: bool = True                       # patchdecode: gate mask features by σ(surface) → the instance head
                                                    #   reasons on surface-aware features (K/V + FPS seeds)

    # --- attention backend (plumbing; NOT a pillar change — raised via flag, docs §12) ---
    attention_backend: str = "auto"

    # --- data ---
    target_voxel_um: float = 9.6                    # Dinovol working scale
    augment: bool = True
    batch_size: int = 1

    # --- optimiser / schedule ---
    base_lr: float = 1.0e-4
    encoder_lr_mult: float = 0.1                    # the whole encoder (stem+transformer) trains at 0.1× (after unfreeze)
    decoder_lr_mult: float = 1.0                    # pixel-decoder LR multiplier (1.0 = base; <1 fine-tunes a
                                                    # warm-started decoder gently while the fresh instance head is at base)
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    warmup_steps: int = 200
    lr_schedule: str = "constant"                   # "constant" (Dinovol convention) | "cosine"

    # --- schedule: LP-FT (Kumar et al., ICLR 2022) — the transformer is FROZEN for a head warmup (train the
    #     head on the distilled encoder's features so the random head's early gradients don't distort them),
    #     then fine-tuned at ``encoder_lr_mult`` × LR for the remainder. ---
    epochs: int = 175                               # total training epochs
    freeze_warmup_epochs: int = 50                  # epochs with the transformer frozen (0 = fine-tune from step 0)
    freeze_decoder_epochs: int = 0                  # epochs with the warm-started decoder TRUNK + surface head frozen
                                                    # (mask head + instance head still train) — instance-head warmup
                                                    # against a fixed surface gate after a surface-only warm-start
    softmax_temp: float = 1.0                        # temperature for the softmax-ownership readout (mask_mode=softmax)
    mask_loss_chunks: int = 1                        # split the dense mask loss into N spatial chunks (checkpointed)
                                                    # so the full (Q,128³) logits are never materialised (1 = dense)
    val_fold: int = 0                               # nnU-Net-consistent 5-fold split (KFold seed 12345): which fold
    val_interval: int = 5                           # validate every N epochs (0 = no validation)
    val_windows: int = 64                           # held-out windows per validation (cap for speed; 0 = all)

    # --- surface (semantic) supervision. w_surface>0 wires the thin medial-surface BCE stack onto the pixel
    #     decoder's surface head (author-style direct supervision). surface_only trains ONLY the surface head
    #     (instance head skipped) — the design-B staging: replicate the author's surface predictor first. ---
    w_surface: float = 0.0       # weight on the surface/semantic loss (0 = off; the instance-only legacy default)
    surface_only: bool = False   # train ONLY the surface head (skip the instance head forward + loss)

    # --- semantic loss (thin medial-surface stack, ported from HercUNet — see train/surface_loss.py) ---
    sem_fg_weight: float = 5.0   # foreground CE weight — medial surfaces are ~0.1-0.3% of voxels; unweighted CE
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

    # --- OPTIONAL SymSkel add-on (per-query thinness; train/losses.py:m2f_symskel_mask_loss) ---
    # Off by default (w_symskel=0 → pure plain-M2F). >0 adds the per-query SymSkel thin-surface stack (fg-CE +
    # Focal-Tversky + clDice skel-recall) on each matched sigmoid mask, for A/B against the plain-M2F baseline.
    w_symskel: float = 0.0                            # weight of the per-query SymSkel term (0 = off)

    # --- OPTIONAL AffMALIS add-on (membership-derived affinity + constrained MALIS; hercunet.training.malis) ---
    # Off by default (w_malis=0). aff_δ = Σ_q p(q|v)·p(q|v+δ) from the softmax membership, MALIS against owner — the
    # maximin attractive/repulsive passes for separation. See train/losses.py:affinity_malis_loss. Needs numba for a
    # usable speed (the njit maximin/union-find); crop bounds the O(E log E) cost (full-window is minutes/step).
    w_malis: float = 0.0                              # weight of the MALIS term (0 = off; try 1.0 to A/B separation)
    malis_crop: int = 48                              # random cube side (voxels) MALIS runs on per item (hercunet=48)
    malis_offsets: tuple = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (3, 0, 0), (0, 3, 0), (0, 0, 3),
                            (9, 0, 0), (0, 9, 0), (0, 0, 9))   # 3 axes × {1,3,9} voxels (hercunet default)

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

    # --- mask denoising (§13.2 / docs/dinovol_sheet_architecture.md): a train-only fixed-assignment query group
    #     that reconstructs each GT sheet, bypassing Hungarian. The standard DETR fix for query under-specialisation. ---
    denoise: bool = True                            # add the denoise query group during training (default ON; §6)
    dn_weight: float = 1.0                          # weight of the denoise-group loss in the total

    # NOTE: distillation (teacher, steps, lr, loss weights) is configured on the ``hercuvit distill`` CLI, NOT in
    # this recipe — a training recipe defines the SUPERVISED net + schedule only.

    def encoder_arch(self) -> dict:
        """Assemble the encoder-arch ``dict`` that ``build_hercuvit`` consumes (as ``arch_override``) from this
        recipe's own fields — the yaml IS the definition, no hidden preset. ``None`` encoder dims are omitted (so
        ``small``, which leaves them None, resolves to the Dinovol teacher's arch on load). ``decoder_channels`` is
        included when set so the slim decoder rides the same build → checkpoint → inference path. Any explicit
        ``arch_override`` tweak is merged last."""
        arch: dict = {}
        for name in _ENCODER_ARCH_FIELDS:
            v = getattr(self, name)
            if v is not None:
                arch[name] = v
        if self.decoder_channels is not None:
            arch["decoder_channels"] = tuple(self.decoder_channels)
        arch.update(self.arch_override or {})
        return arch

    def summary(self) -> str:
        """One-line ``field=value`` dump (minus ``name``) for the per-run audit log."""
        return "  ".join(f"{f.name}={getattr(self, f.name)}" for f in fields(self) if f.name != "name")


_BUILTIN = "small"
_TUPLE_FIELDS = {"global_blocks", "window_size", "deep_sup_weights", "sr_factor", "decoder_channels"}

#: TRAINING fields — set on the ``hercuvit train`` CLI (flags), NOT in the model yaml. The yaml defines THE MODEL;
#: a yaml that tries to set any of these is rejected. (``HercuvitRecipe`` holds both because the resolved run config
#: is model-from-yaml + training-from-CLI; the split is enforced at the yaml boundary, below.)
_TRAIN_FIELDS = frozenset({
    "attention_backend", "target_voxel_um", "augment", "batch_size",
    "base_lr", "encoder_lr_mult", "decoder_lr_mult", "weight_decay", "grad_clip", "warmup_steps", "lr_schedule",
    "epochs", "freeze_warmup_epochs", "freeze_decoder_epochs", "softmax_temp", "mask_loss_chunks",
    "val_fold", "val_interval", "val_windows",
    "w_surface", "surface_only",
    "sem_fg_weight", "w_sem_ce", "w_sem_tversky", "w_sem_skel", "w_sem_sep",
    "tversky_alpha", "tversky_beta", "tversky_gamma", "skel_iters", "sep_radius", "deep_sup_weights",
    "w_symskel", "w_malis", "malis_crop", "malis_offsets",
    "w_class", "w_mask_bce", "w_mask_dice", "w_objectness",
    "cost_class", "cost_mask_bce", "cost_mask_dice", "no_object_weight", "aux_weight",
    "point_sample", "oversample_ratio", "importance_ratio",
    "denoise", "dn_weight",
})
#: MODEL fields — the only keys a recipe yaml may set (everything that is not training).
_MODEL_FIELDS = frozenset(f.name for f in fields(HercuvitRecipe)) - _TRAIN_FIELDS


def _from_yaml(text: str, source: str = "<recipe>") -> HercuvitRecipe:
    """Build a :class:`HercuvitRecipe` from a MODEL yaml (a ``field: value`` mapping of MODEL fields only). Training
    config (epochs, batch_size, lr, schedule, loss weights, attention backend) is set on the CLI, not here, and is
    rejected if it appears in the yaml."""
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
    train_in_yaml = sorted(k for k in over if k in _TRAIN_FIELDS)
    if train_in_yaml:
        raise SystemExit(f"hercuvit recipe {source}: the model yaml must not set TRAINING config {train_in_yaml} — "
                         f"these are CLI flags (e.g. --epochs, --batch-size, --base-lr). The yaml defines the MODEL only.")
    for k in _TUPLE_FIELDS:
        v = over.get(k)
        if v is not None and not (k == "global_blocks" and v == "all"):   # keep the `full` sentinel "all" a string
            over[k] = tuple(v)
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
    """The recipe to train with: the shipped ``small.yml`` default when ``path`` is None, else the recipe at ``path``."""
    if not path:
        return HERCUVIT
    return _from_yaml(Path(path).read_text(), source=str(path))


def variant_recipe(name: str) -> HercuvitRecipe:
    """The shipped full-architecture recipe for a variant (``full`` | ``small`` | ``tiny`` | ``nano``). The yaml IS
    the net definition; used when training/distilling a variant without an explicit ``--recipe``. The ``full`` variant
    (raw all-global Dinovol) ships as ``recipes/full_decoupled.yml``; the windowed variants as ``recipes/<name>.yml``."""
    if name not in VARIANTS:
        raise SystemExit(f"hercuvit: unknown variant {name!r}; expected one of {VARIANTS}.")
    return _load_builtin("full_decoupled" if name == "full" else name)


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

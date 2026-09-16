"""The HercUNet **run301 / v0** training recipe — the single, auditable source of truth for the refiner's
hyperparameters.

The values live in a shipped YAML file, ``hercunet/recipes/hercunet.yml`` (installed with the package), and
:data:`RUN301` is loaded from it. It holds exactly the values the released HercUNet v0 (run301) trained with,
so a plain ``hercunet train fit`` reproduces the recipe with **no environment variables** (the old ``CK_*`` env
vars are gone). The trainer **logs the resolved recipe** at the start of every run (``[recipe] run301: …`` in
the training log), so each run carries its own provenance.

To run a variant: copy ``hercunet.yml``, edit the fields, and pass ``hercunet train fit --recipe <your.yml>``
(omitted fields fall back to the run301 value). :class:`Recipe` below is the schema (types + docstrings); the
YAML file is the data. In code, ``dataclasses.replace(RUN301, prev_fragprob=0.3)`` builds a variant directly.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path


@dataclass(frozen=True)
class Recipe:
    """A HercUNet refiner training recipe. Field values below are **run301 / v0** exactly."""

    name: str = "run301"

    # --- constrained MALIS (per-sheet instance separation; v3 §4.1) ---
    malis_w: float = 1.0             # max MALIS weight
    malis_warm_epochs: int = 50      # ramp 0 -> malis_w over these epochs
    malis_crop: int = 48             # per-item MALIS cube (bounds the O(E log E) cost)
    aff_offsets: tuple = (1, 3, 9)   # axis offsets used for the affinity edges (x, y, z each)

    # --- CT-material growth / air-suppression (v3 §4.3; normalized-CT space) ---
    mat_w: float = 1.0
    mat_warm_epochs: int = 50
    mat_tau: float = 0.0             # material threshold (owner>0 p50 ≈ +1.48 vs air p50 < -0.7 → 0 splits)
    mat_scale: float = 0.4

    # --- online DAgger (self-conditioning; teaches self-repair) ---
    dagger_prob: float = 0.5         # P(a step runs the self-conditioning warm-up)
    dagger_maxk: int = 2             # max warm-up passes

    # --- prev-channel composition (octant "+"-seam + fragmentation MAE) ---
    prev_pboot: float = 0.3          # P(whole-window bootstrap / cold prev)
    prev_tiling: str = "octant"      # "octant" (2x2x2 independent seam) | "whole"
    prev_augprob: float = 0.5        # per-region P(draw an aug variant vs the clean base)
    prev_zeroprob: float = 0.15      # per-region P(zero fill — a neighbour that missed)
    prev_noise: float = 0.0          # additive field noise std ([0,1] units)
    prev_fragprob: float = 0.5       # sheet-section dropout probability (gap-bridging signal) — RUN301 = 0.5
    prev_fragslabmax: int = 60       # max fragmentation slab thickness (voxels)
    prev_fraghalfprob: float = 0.4   # edge/half-space cut share when fragmentation fires

    # --- validation cross-section viz (debug safety check; OFF for run301) ---
    val_viz: int = 0
    val_viz_images: int = 10
    val_viz_slices: int = 10

    def offsets(self):
        """The ``(dz, dy, dx)`` affinity offset list built from ``aff_offsets`` (one per axis per value)."""
        ax = list(self.aff_offsets)
        return [(a, 0, 0) for a in ax] + [(0, a, 0) for a in ax] + [(0, 0, a) for a in ax]

    def summary(self) -> str:
        """One-line ``field=value`` dump (minus ``name``) for the per-run audit log."""
        return "  ".join(f"{f.name}={getattr(self, f.name)}" for f in fields(self) if f.name != "name")


_BUILTIN = "hercunet"    # hercunet/recipes/hercunet.yml — the run301 default


def _from_yaml(text: str, source: str = "<recipe>") -> Recipe:
    """Build a :class:`Recipe` from YAML text: a mapping of ``field: value``, applied on top of the schema
    defaults (so a partial file overrides only those fields). Unknown fields raise."""
    try:
        import yaml
    except ImportError as e:                                      # PyYAML ships with the [train] extra
        raise SystemExit("hercunet: PyYAML is required for training recipes — install the training extra "
                         "(pip install -e '.[train]').") from e
    over = yaml.safe_load(text) or {}
    if not isinstance(over, dict):
        raise SystemExit(f"hercunet recipe {source}: expected a YAML mapping of field: value.")
    valid = {f.name for f in fields(Recipe)}
    bad = sorted(k for k in over if k not in valid)
    if bad:
        raise SystemExit(f"hercunet recipe {source}: unknown field(s) {bad}. Valid fields: {sorted(valid)}.")
    if over.get("aff_offsets") is not None:
        over["aff_offsets"] = tuple(over["aff_offsets"])
    return replace(Recipe(), **over)


def _load_builtin(name: str = _BUILTIN) -> Recipe:
    """Load a shipped recipe (``hercunet/recipes/<name>.yml``) — install-safe via importlib.resources."""
    from importlib.resources import files
    res = files("hercunet.recipes").joinpath(f"{name}.yml")
    return _from_yaml(res.read_text(), source=f"hercunet/recipes/{name}.yml")


def load(path=None) -> Recipe:
    """The recipe to train with: :data:`RUN301` (the shipped ``hercunet.yml``) when ``path`` is None, else the
    variant at ``path`` (a YAML file; omitted fields fall back to the run301 values)."""
    if not path:
        return RUN301
    return _from_yaml(Path(path).read_text(), source=str(path))


# The default recipe = HercUNet v0 (run301), loaded from the shipped YAML. The trainer reads its knobs from it.
RUN301 = _load_builtin()


# --- active-recipe plumbing (the DEFAULT is always exactly run301) -----------------------------------------
# The trainer reads the ACTIVE recipe at construction. `hercunet train fit/chain --recipe <file.yml>` loads a
# variant and installs it active before the trainer is built; everything else uses run301.
_ACTIVE = RUN301


def active() -> Recipe:
    """The recipe the trainer should use right now (run301 unless a variant was made active)."""
    return _ACTIVE


def set_active(recipe: Recipe) -> None:
    """Install ``recipe`` as active (called by ``fit`` before constructing the trainer; each DDP worker
    re-installs it in its own process)."""
    global _ACTIVE
    _ACTIVE = recipe

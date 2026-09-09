"""STAGE 1 — pseudo-label generation.

The per-window pipeline (cleaned substrate + frame field → 2.5-D meshlets → slab selection →
8-D contrastive embedding → probeom clustering → medial-mesh fit → ∇φ / owner_full →
confidence + quality → .npz) is documented in ``docs/pseudo-labels.md`` and is being moved into
this package stage by stage. :func:`create` is the stable entry point the CLI lands behind.
"""

from .create import create

__all__ = ["create"]

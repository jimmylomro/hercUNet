"""STAGE 1 — pseudo-label generation.

The per-window pipeline (cleaned substrate + frame field → 2.5-D meshlets → slab selection →
8-D contrastive embedding → probeom clustering → medial-mesh fit → ∇φ / owner_full →
confidence + quality → .npz) is documented in ``submission/writeup/herculabels.md`` and is being moved into
this package stage by stage. :func:`create` is the stable entry point the CLI lands behind.
"""

from .corpus import Corpus
from .create import create
from .edit_cmd import edit_corpus
from .export import export
from .merge import merge_corpora

__all__ = ["create", "edit_corpus", "export", "merge_corpora", "Corpus"]

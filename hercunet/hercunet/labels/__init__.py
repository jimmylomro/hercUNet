"""STAGE 1 — pseudo-label generation (nnU-Net pipeline).

The shared geometry primitives it builds on (corpus, io, mesh, meshlets, embedding, fields, selection)
live at :mod:`hercunet.labels`; this subpackage holds the generation/edit/export entry points.
"""
from .create import create
from .edit_cmd import edit_corpus
from .export import export
from .merge import merge_corpora

__all__ = ["create", "edit_corpus", "export", "merge_corpora"]

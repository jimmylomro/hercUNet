"""Shared label geometry primitives used by both the nnU-Net refiner and HercuViT: the corpus reader,
I/O bundles, medial-mesh fit (``mesh.medial``), meshlets, contrastive embedding, structure-tensor fields,
and membership selection. The nnU-Net stage-1 GENERATION entry points live in
:mod:`hercunet.hercunet.labels`.
"""
from .corpus import Corpus

__all__ = ["Corpus"]

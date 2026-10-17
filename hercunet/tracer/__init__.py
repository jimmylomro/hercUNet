"""Sheet-segmentation decoder head for HercUNet (the ``docs/hercunet-tracer.md`` lead path).

A **new output head** on HercUNet that emits a *coarse sheet segmentation* as a node/edge manifold,
supervised by the **herculabels** ``owner_full`` instances HercUNet already trains on. Its output feeds the
3-D cylindrical winding solve as certified same-sheet ties (replacing qualz's fibre ``.afv`` ties).

Design (see the vesuvius-repo doc for the full rationale):

* A stock nnU-Net encoder/decoder (unchanged) produces dense features. A small **transformer decoder with
  instance queries** (DETR / MaskFormer / Mask2Former family, deformable-style cross-attention) reads those
  features and, per **coarse grid node**, emits a **softmax over queries = ownership**. An **edge is kept iff
  its two endpoints share an owner**; different owners ⇒ **cut** (the merge cut, for free). A node cannot
  bisect — a would-be split *is* a merge, and the head cuts it.
* **Soft** topology priors (degree cap, non-crossing, smoothness) *steer*; a **hard gate** *guarantees*
  correctness by dropping ambiguous fragments (purity > coverage).

Hard constraints honoured in this package:

* The decoder hooks the **UNet decoder feature map directly** (its own forward-hook on
  ``decoder.stages[-1]``), independent of the affinity head — the **affinity head stays MALIS-only** and
  nothing here reads from it (:mod:`hercunet.tracer.decoder`).
* The decoder is **only supervised where sheet identity exists** (herculabels ``cluster_ids`` / ``V_k``
  meshes). On m7-derived samples there is no identity, so the loss is **gated off** per sample
  (:func:`hercunet.tracer.labels.has_identity`, :func:`hercunet.tracer.losses.sheet_seg_loss`).
* **No graph / multicut** — every topology constraint is a soft pointwise penalty or a local manifold check,
  never a combinatorial solve (:mod:`hercunet.tracer.topology`).

Heavy deps (``torch``, ``scipy``) are imported lazily inside the modules that need them, so the label-prep /
extraction / metric utilities import cleanly on a non-training host (same convention as
``hercunet.training``).
"""
from __future__ import annotations

__all__ = [
    "labels",
    "grid",
    "topology",
    "decoder",
    "losses",
    "extract",
]

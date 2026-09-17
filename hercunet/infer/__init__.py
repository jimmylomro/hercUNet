"""STAGE 3 — iterative full-volume inference (the HercUNet v0 refiner).

Runs the trained AffinityMalis refiner as double-buffered **Jacobi passes** over a scroll: each pass reads the
previous pass's whole surface-probability volume as the ``prev`` input channel and writes a fresh one, healing
window seams with the villa Gaussian blend (``overlap`` default 0.25). Output is a per-pass VC3D-compatible
OME-Zarr surface-probability pyramid; the final pass is the deliverable.

The network is built stock from ``plans.json`` (``get_network_from_plans``) and loaded from ``network_weights`` —
the custom trainer class need NOT be importable, only a pinned ``nnunetv2==2.8.1``. Stock ``nnUNetv2_predict``
will NOT work (the 8-ch ``[CT, prev, orient(6)]`` stem + iteration are HercUNet-specific).

Modules:

* :mod:`~hercunet.infer.nnunet_infer` — net load (:func:`~hercunet.infer.nnunet_infer.load_affinity_net`),
  CT normalisation, scroll open (via the hercunet data layer), OME-Zarr buffer + pyramid + S3 upload.
* :mod:`~hercunet.infer.jacobi_refine` — the iterative engine (blend / disjoint passes, single- or multi-worker).
* :mod:`~hercunet.infer.claim_queue` — the elastic multi-worker claim/done protocol.
* :mod:`~hercunet.infer.run` — orchestration: ``single_instance`` (local multi-GPU) / ``multi_instance`` (pods).

CLI: ``hercunet infer single-instance …`` / ``hercunet infer multi-instance …`` (see ``docs/infer.md``).
"""

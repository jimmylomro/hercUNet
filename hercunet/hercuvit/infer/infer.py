"""HercuViT full-volume inference — drives the shared, model-agnostic engine (:mod:`hercunet.infer.engine`) with the
``hercuvit-surface`` model (:class:`hercunet.hercuvit.infer.model.HercuvitSurfaceModel`).

HercuViT is single-pass (no Jacobi ``prev`` feedback) and 1-channel; the engine streams the CT, tiles it, runs the
overlapping windows, Gaussian-blends the SURFACE (``sem_logits[:,1]-sem_logits[:,0]`` → sigmoid), and writes a
VC3D-friendly OME-Zarr ``{out}_sem.zarr`` + pyramid. Per-sheet masks are NOT emitted (the decoupled instance head is
built off at load). The engine supplies region/full-volume, empty-window skipping, streaming + prefetch, and — when
driven multi-process — the claim-queue ``.done`` work-stealing + leader finalise + S3 upload, same as hercunet.
"""
from __future__ import annotations


def single_instance(args) -> None:
    """``hercuvit infer single-instance`` handler — one box, semantic surface output via the shared engine."""
    import torch
    from ...infer.engine import run_inference
    from ...infer.run import parse_region

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[hercuvit] no CUDA visible — falling back to CPU (slow)", flush=True)
        device = "cpu"

    run_inference(
        source=args.source,
        out_prefix=args.out,                                  # engine writes {out}_pass0.zarr (single pass)
        model_type="hercuvit-surface",
        model_import="hercunet.hercuvit.infer.model",         # registers HercuvitSurfaceModel in this/worker process
        model=args.model, model_hf=args.model_hf, ckpt=args.ckpt, device=device,
        model_opts=dict(attention_backend=args.attention_backend, overlap=args.overlap),
        region=parse_region(args.region), level=args.level, air=args.air,
        finalise=args.finalise,                              # "last" | "no"
        multi=False,                                          # single-GPU in-process (multi-GPU/pod + S3 = future CLI flags)
    )

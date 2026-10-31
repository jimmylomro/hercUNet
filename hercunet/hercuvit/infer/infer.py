"""HercuViT full-volume inference — drives the shared, model-agnostic engine (:mod:`hercunet.infer.engine`) with the
``hercuvit-surface`` model (:class:`hercunet.hercuvit.infer.model.HercuvitSurfaceModel`) through the SAME fan-out as
``hercunet infer`` (:func:`hercunet.infer.run.run_fanout`): one worker process per local GPU on a shared ``.done``
claim-queue, a per-pass barrier, and leader-automatic pyramid finalise + S3 upload — across pods too (multi-instance).

HercuViT is single-pass (no Jacobi ``prev`` feedback) and 1-channel; the engine streams the CT, tiles it, Gaussian-
blends the SURFACE (``sem_logits[:,1]-sem_logits[:,0]`` → sigmoid) and writes ``{out}_pass0.zarr`` + pyramid. Per-sheet
masks are NOT emitted (the decoupled instance head is built off at load).
"""
from __future__ import annotations


def _resolve_model(a) -> str:
    """A LOCAL HercuViT model path (dir or .pth; ``model.load`` handles both). ``--model`` local, or ``--model-hf``
    snapshot-downloaded once in the parent so every worker reads the same local copy."""
    if a.model and a.model_hf:
        raise SystemExit("hercuvit infer: pass --model OR --model-hf, not both.")
    if a.model:
        return a.model
    if a.model_hf:
        from huggingface_hub import snapshot_download
        print(f"[hercuvit] downloading model from HF {a.model_hf} …", flush=True)
        return snapshot_download(a.model_hf)
    raise SystemExit("hercuvit infer: give a model — --model <dir|.pth> or --model-hf <repo>.")


def _drive(a, *, across_pods: bool) -> None:
    import torch
    from ...infer.run import run_fanout, parse_region, _maybe_presync, engine_jargs
    from ...infer.engine import run_inference

    # --- output selection (--surf / --mesh ∈ off|local|upload) -----------------------------------------------------
    if a.surf == "off" and a.mesh == "off":
        raise SystemExit("hercuvit infer: nothing to produce — set --surf and/or --mesh to 'local' or 'upload'.")
    if (a.surf == "upload" or a.mesh == "upload") and not a.s3_prefix:
        raise SystemExit("hercuvit infer: --surf/--mesh 'upload' needs --s3-prefix <s3://…>.")
    if a.s3_prefix and a.surf != "upload" and a.mesh != "upload":
        raise SystemExit("hercuvit infer: --s3-prefix given but neither --surf nor --mesh is 'upload' — "
                         "specify what to upload (e.g. --mesh upload).")
    # --stitcher implies --store-embeddings (the stitcher needs the emb zarr + the per-sheet medians) and runs the
    # cross-window stitch in the LEADER process once the sheets + embeddings are merged (fast enough not to shard).
    stitcher_uri = getattr(a, "stitcher", None)
    stitcher_local = None
    emb_mode = getattr(a, "store_embeddings", None)                    # None | "local" | "upload"
    if stitcher_uri and emb_mode is None:
        emb_mode = "local"                                            # --stitcher implies at least storing them
    store_emb = emb_mode is not None
    # upload the emb zarr when it was explicitly requested (--store-embeddings upload) OR when the run is already
    # uploading its sheet deliverable (--mesh upload) — i.e. "ship everything". Never when embeddings weren't stored.
    emb_upload = store_emb and (emb_mode == "upload" or a.mesh == "upload")
    if store_emb and a.mesh == "off":
        raise SystemExit("hercuvit infer: --store-embeddings/--stitcher needs the sheets — set --mesh local (or upload).")
    if emb_upload and not a.s3_prefix:
        raise SystemExit("hercuvit infer: uploading embeddings needs --s3-prefix <s3://…> "
                         "(--store-embeddings upload, or --mesh upload alongside stored embeddings).")
    if stitcher_uri:
        from ..stitch.infer import resolve_stitcher_uri                # resolve ONCE in the parent (cached download);
        stitcher_local = resolve_stitcher_uri(stitcher_uri)           #   the leader worker reuses this local path

    model_dir = _resolve_model(a)
    region = parse_region(a.region)
    source = _maybe_presync(a, region)
    jargs = dict(
        engine_jargs(a, region, source),
        model_type="hercuvit-surface",
        model_import="hercunet.hercuvit.infer.model",          # so a worker subprocess registers the model
        surf_store=a.surf, mesh_store=a.mesh,
        store_embeddings=store_emb, emb_dim=int(getattr(a, "emb_dim", 3)),
        emb_upload=emb_upload,                                # ship {out}_emb.zarr to --s3-prefix (leader, after merge)
        stitcher=stitcher_local,                              # None → no stitch; set → leader stitches after the merge
        finalise=(a.finalise if a.surf != "off" else "no"),   # surface pyramid only when the surface is stored
        upload=("last" if a.surf == "upload" else "no"),      # surface S3 upload gated by --surf (meshes by --mesh)
        model_opts=dict(attention_backend=a.attention_backend, overlap=a.overlap,
                        with_meshes=(a.mesh != "off"), mesh_fit_iters=a.mesh_fit_iters,
                        store_embeddings=store_emb, emb_dim=int(getattr(a, "emb_dim", 3))),
    )
    if a.device == "cpu" or not torch.cuda.is_available():
        if across_pods:
            raise SystemExit("hercuvit infer multi-instance requires CUDA GPUs.")
        print("[hercuvit] no CUDA (or --device cpu) — running the engine in-process on CPU (slow)", flush=True)
        run_inference(model=model_dir, device="cpu", multi=False, leader=True, **jargs)
    else:
        run_fanout(a, model_dir, jargs, across_pods=across_pods)


def single_instance(a) -> None:
    """``hercuvit infer single-instance`` — one box, fanning across its local GPUs (CPU fallback in-process)."""
    _drive(a, across_pods=False)


def multi_instance(a) -> None:
    """``hercuvit infer multi-instance`` — run the SAME command on each pod (``--out`` on shared storage, ``--leader``
    on one); each pod fans across its own GPUs on the shared claim-queue."""
    _drive(a, across_pods=True)

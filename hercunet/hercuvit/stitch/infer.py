"""``hercuvit stitcher infer`` — apply a trained stitcher to a volume's stored HercuViT artifacts (docs §8/§9).

NOT YET WIRED. The design (deliberately) reuses the production ``hercuvit infer`` path rather than re-running the
network here: a later ``hercuvit infer`` run emits the two stitching artifacts this command consumes —

  1. a **pooled-window-embedding zarr** — each window's per-window normalised-PCA tokens (``stitch.embed``), the
     stitcher's spatial memory ``Z`` source; and
  2. a **per-sheet median-embedding column in the ``.asv``** — each sheet's 3-D after-PCA ``median_after_pca``
     descriptor (same per-window basis), the stitcher's queries.

``stitcher infer`` then, per corner: gathers the overlapping windows' tokens from the zarr + the touching sheets'
median columns from the ``.asv``, runs :class:`StitchDecoder`, keeps the top-``k`` edges, and agglomerates them into
a fragment→object lookup. Building the zarr/``.asv`` emitters in the infer engine is the next increment (it also
adds ``hercuvit infer --stitcher <hf|file>://…`` to run the whole thing inline). The reusable pieces are already in
place: :func:`hercunet.hercuvit.stitch.embed.reduce_normalised_pca` / ``median_after_pca`` (one implementation) and
the trained-decoder loader below.
"""
from __future__ import annotations


def load_stitcher(path_or_repo: str, *, device="cuda"):
    """Rebuild a trained :class:`StitchDecoder` from a checkpoint (local ``.pth``/dir or an HF repo) → ``(model, cfg)``.
    Mirrors :func:`hercunet.hercuvit.infer.loader.load_hercuvit`: the config is stored in the checkpoint, so the
    decoder is rebuilt exactly and the weights loaded over it. Ready for the infer wiring + ``hercuvit infer
    --stitcher``."""
    import os

    import torch

    from .config import config_from_dict
    from .model import StitchDecoder

    root = path_or_repo
    if not os.path.exists(root):                                   # treat a non-path as an HF repo id
        from huggingface_hub import snapshot_download
        print(f"[stitcher] downloading stitcher from HF {root} …", flush=True)
        root = snapshot_download(root)
    if os.path.isdir(root):
        for name in ("checkpoint_best.pth", "checkpoint_final.pth", "checkpoint_latest.pth"):
            if os.path.isfile(os.path.join(root, name)):
                root = os.path.join(root, name); break
        else:
            raise SystemExit(f"hercuvit stitcher: no checkpoint_*.pth under {root}.")
    ck = torch.load(root, map_location="cpu", weights_only=False)
    cfg = config_from_dict(ck.get("config", {}))
    model = StitchDecoder(cfg)
    model.load_state_dict(ck["model"])
    model.eval().to(device)
    print(f"[stitcher] loaded {root} ({cfg.summary()})", flush=True)
    return model, cfg


def infer_run(a) -> None:
    raise NotImplementedError(
        "hercuvit stitcher infer is not wired yet. It will consume the pooled-window-embedding zarr + the per-sheet "
        "median column in the .asv that a future `hercuvit infer` run emits (see docs/hercuvit-stitching.md §8/§9). "
        "The trained-decoder loader (hercunet.hercuvit.stitch.infer.load_stitcher) and the shared embedding reduction "
        "(hercunet.hercuvit.stitch.embed) are already in place for that wiring. Train first: `hercuvit stitcher train`.")

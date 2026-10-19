"""``hercuvit train chain`` / ``fit`` — resolve the corpus + the Dinovol encoder, build HercuViT, and fit.

One command end to end (§3/§7/§10). Reuses hercunet's HF plumbing: the corpus is pulled from the HF **dataset**
(``jimmylomro/hercunet-corpus``) and the Dinovol encoder checkpoint from the HF **model** repo via
``hercunet.train.launch.resolve_pretrained`` (same ``hf_hub_download`` cache hercunet uses).
"""
from __future__ import annotations

from dataclasses import replace


def resolve_corpus(corpus=None, corpus_hf=None, *, allow_patterns=None) -> str:
    """Return a LOCAL corpus directory. ``corpus`` is a local dir; ``corpus_hf`` is an HF dataset to download
    (``snapshot_download`` — **cached and idempotent**: a second run re-uses the cache, never re-downloads).
    ``allow_patterns`` restricts the pull (e.g. distillation needs only ``sample_*.npz`` + ``corpus_index/*``, not
    the heavy ``m7_corpus/`` tifs)."""
    if corpus and corpus_hf:
        raise SystemExit("hercuvit train: pass --corpus (a local dir) OR --corpus-hf (download), not both.")
    if corpus:
        return corpus
    if not corpus_hf:
        raise SystemExit("hercuvit train: give a corpus — --corpus <dir> or --corpus-hf <repo> "
                         "(e.g. --corpus-hf jimmylomro/hercunet-corpus).")
    import os
    from huggingface_hub import snapshot_download
    try:                                      # hf_transfer = Rust chunked parallel downloader (big files); opt-in if present
        import hf_transfer  # noqa: F401
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    except ImportError:
        pass
    what = f" ({', '.join(allow_patterns)})" if allow_patterns else ""
    print(f"[hercuvit] resolving corpus from HF dataset {corpus_hf}{what} (cached if already pulled) …", flush=True)
    # max_workers = parallel file connections — the win for the many small sample_*.npz (default is 8).
    path = snapshot_download(corpus_hf, repo_type="dataset", allow_patterns=allow_patterns, max_workers=16)
    print(f"[hercuvit] corpus at {path}", flush=True)
    return path


def resolve_encoder(encoder=None, encoder_hf=None, hf_file=None) -> str:
    """Return a LOCAL encoder checkpoint path — a distilled ``hercuvit distill`` output OR a raw Dinovol teacher
    (reuses hercunet's ``resolve_pretrained`` for the HF pull)."""
    from ...train.launch import resolve_pretrained
    from ..model.encoder import DINOVOL_CKPT_FILE
    if encoder:
        return encoder
    if not encoder_hf:
        raise SystemExit("hercuvit train: give the encoder — --encoder <ckpt> or --encoder-hf <repo>.")
    return resolve_pretrained(None, encoder_hf, hf_file or DINOVOL_CKPT_FILE)


def resolve_encoder_and_variant(args):
    """Resolve the encoder + its variant for the train handlers. ``--variant`` is FROM-SCRATCH only (mutually
    exclusive with ``--encoder``; not supported yet). Normal training reads the variant FROM the encoder file."""
    from ..model.hercuvit_net import encoder_variant
    has_enc = bool(getattr(args, "encoder", None) or getattr(args, "encoder_hf", None))
    if getattr(args, "variant", None) and has_enc:
        raise SystemExit("hercuvit train: --variant is for FROM-SCRATCH training; it is mutually exclusive with "
                         "--encoder/--encoder-hf (which deduce the variant from the file).")
    if getattr(args, "variant", None) and not has_enc:
        raise NotImplementedError("from-scratch training (--variant without an encoder) is not supported yet — "
                                  "pass --encoder <distilled .pt> or --encoder-hf (raw Dinovol → 'full').")
    enc = resolve_encoder(getattr(args, "encoder", None), getattr(args, "encoder_hf", None),
                          getattr(args, "encoder_file", None))
    variant = encoder_variant(enc)
    print(f"[hercuvit] encoder {enc} → variant '{variant}'", flush=True)
    return enc, variant


def run_training(
    *,
    corpus_dir: str,
    encoder_ckpt: str,
    out: str,
    recipe_path: str | None = None,
    epochs: int | None = None,
    device: str = "cuda",
    attention_backend: str | None = None,
    variant: str | None = None,
    num_workers: int = 4,
    continue_training: bool = False,
    identity_only: bool = True,
) -> None:
    """Build HercuViT from an encoder checkpoint (``encoder_ckpt`` — a distilled encoder or a raw Dinovol teacher;
    ``build_hercuvit`` detects which + the windowed retrofit) and fit on the herculabels corpus. ``variant`` is the
    one deduced from the encoder (overrides the recipe)."""
    import os

    import torch
    from torch.utils.data import DataLoader

    from .. import netbuild
    from ..data.dataset import HercuvitDataset, hercuvit_collate
    from . import recipe as _recipe
    from .trainer import HercuvitTrainer

    rec = _recipe.load(recipe_path)
    if attention_backend:                                            # CLI override of the recipe field
        rec = replace(rec, attention_backend=attention_backend)
    if variant:                                                      # CLI override of the recipe field
        rec = replace(rec, variant=variant)
    _recipe.set_active(rec)

    blob = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)      # model_cfg for the saved checkpoints
    model_cfg = blob["model_cfg"] if (isinstance(blob, dict) and "student_backbone" in blob) else blob["config"]["model"]
    del blob
    net = netbuild.build(
        encoder_ckpt, device=device, attention_backend=rec.attention_backend,
        variant=rec.variant, arch_override=rec.arch_override,
        global_blocks=rec.global_blocks, window_size=rec.window_size,
        n_sem_classes=rec.n_sem_classes, mask_dim=rec.mask_dim, n_queries=rec.n_queries,
        query_dim=rec.query_dim, instance_layers=rec.instance_layers,
        deep_supervision=rec.deep_supervision, with_instance=True,
        add_null_query=rec.add_null_query, learned_queries=rec.learned_queries,
    )

    ds = HercuvitDataset(corpus_dir, augment=rec.augment, target_voxel_um=rec.target_voxel_um,
                         crop=rec.window, identity_only=identity_only)
    loader = DataLoader(ds, batch_size=rec.batch_size, shuffle=True, num_workers=num_workers,
                        collate_fn=hercuvit_collate, drop_last=True, pin_memory=(device != "cpu"))
    print(f"[hercuvit] corpus {corpus_dir}: {len(ds)} samples; batch {rec.batch_size}; device {device}", flush=True)

    trainer = HercuvitTrainer(net, rec, out, model_cfg=model_cfg, device=device)
    if epochs:
        trainer.num_epochs = int(epochs)
    if continue_training:
        latest = os.path.join(out, "checkpoint_latest.pth")
        if os.path.isfile(latest):
            trainer.load_checkpoint(latest)
        else:
            print("[hercuvit] --continue: no checkpoint_latest.pth found, starting fresh", flush=True)
    trainer.run(loader)


def fit(args) -> None:
    """``hercuvit train fit`` handler: corpus + encoder already local (or pulled by flag)."""
    corpus_dir = resolve_corpus(args.corpus, args.corpus_hf)
    encoder_ckpt, variant = resolve_encoder_and_variant(args)
    run_training(corpus_dir=corpus_dir, encoder_ckpt=encoder_ckpt, variant=variant, out=args.out,
                 recipe_path=args.recipe, epochs=args.epochs, device=args.device,
                 attention_backend=args.attention_backend,
                 num_workers=args.num_workers, continue_training=args.continue_)


def chain(args) -> None:
    """``hercuvit train chain`` handler: pull the corpus + encoder, build, fit — one command."""
    corpus_dir = resolve_corpus(args.corpus, args.corpus_hf)
    encoder_ckpt, variant = resolve_encoder_and_variant(args)
    run_training(corpus_dir=corpus_dir, encoder_ckpt=encoder_ckpt, variant=variant, out=args.out,
                 recipe_path=args.recipe, epochs=args.epochs, device=args.device,
                 attention_backend=args.attention_backend,
                 num_workers=args.num_workers, continue_training=args.continue_)

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
    from huggingface_hub import snapshot_download
    try:                                      # quieten the benign "unauthenticated request" spam (public dataset)
        from huggingface_hub.utils import logging as _hf_logging
        _hf_logging.set_verbosity_error()
    except Exception:
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
    device: str = "cuda",
    variant: str | None = None,
    num_workers: int = 4,
    continue_training: bool = False,
    identity_only: bool = True,
    pre_extract: bool = True,
    train_cache: str | None = None,
    prep_shards: int = 8,
    train_overrides: dict | None = None,
) -> None:
    """Build HercuViT from an encoder checkpoint (``encoder_ckpt`` — a distilled encoder or a raw Dinovol teacher;
    ``build_hercuvit`` detects which + the windowed retrofit) and fit on the herculabels corpus. ``variant`` is the
    one deduced from the encoder (overrides the recipe).

    ``pre_extract`` (default): pre-extract the gated training set (CT + labels) to a LOCAL cache via shard processes
    (``preextract._ensure_train_cache``), then train from :class:`LocalHercuvitDataset` with a safe worker DataLoader
    — the live S3 path deadlocks under worker concurrency and re-streams every epoch. ``--no-pre-extract`` falls back
    to the live :class:`HercuvitDataset` (use ``--num-workers 0`` there on an S3 corpus)."""
    import os

    import torch
    from torch.utils.data import DataLoader

    from .. import netbuild
    from ..data.dataset import HercuvitDataset, LocalHercuvitDataset, hercuvit_collate
    from . import recipe as _recipe
    from .trainer import HercuvitTrainer

    # The recipe IS the network definition. With no explicit --recipe, use the shipped recipe for the encoder's
    # deduced variant (e.g. a distilled nano .pt → recipes/nano.yml: slim decoder + nano heads). An explicit
    # --recipe wins. The encoder's arch then comes from the .pt and build_hercuvit validates the recipe against it.
    rec = _recipe.load(recipe_path) if recipe_path else (_recipe.variant_recipe(variant) if variant else _recipe.load(None))
    if train_overrides:                                              # TRAINING config from the CLI flags (yaml is model-only)
        rec = replace(rec, **train_overrides)
    _recipe.set_active(rec)

    blob = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)      # model_cfg for the saved checkpoints
    model_cfg = blob["model_cfg"] if (isinstance(blob, dict) and "student_backbone" in blob) else blob["config"]["model"]
    del blob
    net = netbuild.build(
        encoder_ckpt, device=device, attention_backend=rec.attention_backend,
        arch_override=rec.encoder_arch(),                            # the arch assembled FROM the recipe (yaml = net def)
        global_blocks=rec.global_blocks, window_size=rec.window_size,
        n_sem_classes=rec.n_sem_classes, mask_dim=rec.mask_dim, n_queries=rec.n_queries,
        query_dim=rec.query_dim, instance_layers=rec.instance_layers,
        deep_supervision=rec.deep_supervision, with_instance=True,
        add_null_query=rec.add_null_query, learned_queries=rec.learned_queries,
        fps_queries=rec.fps_queries, cross_attn=rec.cross_attn,
    )

    if pre_extract:
        from .preextract import _ensure_train_cache
        cache_dir = train_cache or os.path.join(out, "train_cache")
        _ensure_train_cache(corpus_dir=corpus_dir, cache_dir=cache_dir, nshards=prep_shards,
                            identity_only=identity_only)
        ds = LocalHercuvitDataset(cache_dir, augment=rec.augment, target_voxel_um=rec.target_voxel_um,
                                  crop=rec.window)
        src = cache_dir
    else:                                                            # live S3 (re-streams every epoch; workers unsafe)
        ds = HercuvitDataset(corpus_dir, augment=rec.augment, target_voxel_um=rec.target_voxel_um,
                             crop=rec.window, identity_only=identity_only)
        src = corpus_dir
    loader = DataLoader(ds, batch_size=rec.batch_size, shuffle=True, num_workers=num_workers,
                        collate_fn=hercuvit_collate, drop_last=True, pin_memory=(device != "cpu"))
    print(f"[hercuvit] {src}: {len(ds)} samples; batch {rec.batch_size}; workers {num_workers}; device {device}",
          flush=True)

    trainer = HercuvitTrainer(net, rec, out, model_cfg=model_cfg, device=device)
    if continue_training:
        latest = os.path.join(out, "checkpoint_latest.pth")
        if os.path.isfile(latest):
            trainer.load_checkpoint(latest)
        else:
            print("[hercuvit] --continue: no checkpoint_latest.pth found, starting fresh", flush=True)
    trainer.run(loader)


#: train fields settable on the CLI (the yaml is model-only); unset flags keep the recipe defaults.
_TRAIN_FLAGS = ("epochs", "freeze_warmup_epochs", "batch_size", "base_lr", "encoder_lr_mult",
                "weight_decay", "grad_clip", "warmup_steps", "lr_schedule", "augment",
                "target_voxel_um", "attention_backend", "w_symskel", "w_malis", "denoise", "dn_weight")


def _run_from_args(args) -> None:
    """Shared body for ``fit`` / ``chain`` — resolve corpus + encoder, then fit (optionally pre-extracting first)."""
    corpus_dir = resolve_corpus(args.corpus, args.corpus_hf)
    encoder_ckpt, variant = resolve_encoder_and_variant(args)
    train_overrides = {k: getattr(args, k) for k in _TRAIN_FLAGS if getattr(args, k, None) is not None}
    run_training(corpus_dir=corpus_dir, encoder_ckpt=encoder_ckpt, variant=variant, out=args.out,
                 recipe_path=args.recipe, device=args.device,
                 num_workers=args.num_workers, continue_training=args.continue_,
                 pre_extract=not getattr(args, "no_pre_extract", False),
                 train_cache=getattr(args, "train_cache", None),
                 prep_shards=int(getattr(args, "prep_shards", 8)),
                 train_overrides=train_overrides)


def fit(args) -> None:
    """``hercuvit train fit`` handler: corpus + encoder already local (or pulled by flag)."""
    _run_from_args(args)


def chain(args) -> None:
    """``hercuvit train chain`` handler: pull the corpus + encoder, build, fit — one command."""
    _run_from_args(args)

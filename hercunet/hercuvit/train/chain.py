"""``hercuvit train chain`` / ``fit`` — resolve the corpus + the Dinovol encoder, build HercuViT, and fit.

One command end to end (§3/§7/§10). Reuses hercunet's HF plumbing: the corpus is pulled from the HF **dataset**
(``jimmylomro/hercunet-corpus``) and the Dinovol encoder checkpoint from the HF **model** repo via
``hercunet.train.launch.resolve_pretrained`` (same ``hf_hub_download`` cache hercunet uses).
"""
from __future__ import annotations

from dataclasses import replace


def resolve_corpus(corpus=None, corpus_hf=None) -> str:
    """Return a LOCAL corpus directory. ``corpus`` is a local dir; ``corpus_hf`` is an HF dataset to download."""
    if corpus and corpus_hf:
        raise SystemExit("hercuvit train: pass --corpus (a local dir) OR --corpus-hf (download), not both.")
    if corpus:
        return corpus
    if not corpus_hf:
        raise SystemExit("hercuvit train: give a corpus — --corpus <dir> or --corpus-hf <repo> "
                         "(e.g. --corpus-hf jimmylomro/hercunet-corpus).")
    from huggingface_hub import snapshot_download
    print(f"[hercuvit] downloading corpus from HF dataset {corpus_hf} …", flush=True)
    path = snapshot_download(corpus_hf, repo_type="dataset")
    print(f"[hercuvit] corpus at {path}", flush=True)
    return path


def resolve_dinovol(dinovol=None, dinovol_hf=None, hf_file=None) -> str:
    """Return a LOCAL Dinovol checkpoint path (reuses hercunet's ``resolve_pretrained``)."""
    from ...train.launch import resolve_pretrained
    from ..model.encoder import DINOVOL_CKPT_FILE
    if dinovol:
        return dinovol
    if not dinovol_hf:
        raise SystemExit("hercuvit train: give the Dinovol encoder — --dinovol <ckpt> or --dinovol-hf <repo>.")
    return resolve_pretrained(None, dinovol_hf, hf_file or DINOVOL_CKPT_FILE)


def run_training(
    *,
    corpus_dir: str,
    dinovol_ckpt: str,
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
    """Build HercuViT (Dinovol warm-start + windowed retrofit + backend fix) and fit on the herculabels corpus."""
    import os

    import torch
    from torch.utils.data import DataLoader

    from .. import netbuild
    from ..data.dataset import HercuvitDataset, hercuvit_collate
    from ..model.encoder import load_dinovol_checkpoint
    from . import recipe as _recipe
    from .trainer import HercuvitTrainer

    rec = _recipe.load(recipe_path)
    if attention_backend:                                            # CLI override of the recipe field
        rec = replace(rec, attention_backend=attention_backend)
    if variant:                                                      # CLI override of the recipe field
        rec = replace(rec, variant=variant)
    _recipe.set_active(rec)

    _teacher, model_cfg, _full = load_dinovol_checkpoint(dinovol_ckpt, map_location="cpu")
    net = netbuild.build(
        dinovol_ckpt, device=device, attention_backend=rec.attention_backend,
        variant=rec.variant, arch_override=rec.arch_override,
        global_blocks=rec.global_blocks, window_size=rec.window_size,
        n_sem_classes=rec.n_sem_classes, mask_dim=rec.mask_dim, n_queries=rec.n_queries,
        query_dim=rec.query_dim, instance_layers=rec.instance_layers,
        deep_supervision=rec.deep_supervision, with_instance=True,
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
    """``hercuvit train fit`` handler: corpus + Dinovol already local (or pulled by flag)."""
    corpus_dir = resolve_corpus(args.corpus, args.corpus_hf)
    dinovol_ckpt = resolve_dinovol(args.dinovol, args.dinovol_hf, args.dinovol_file)
    run_training(corpus_dir=corpus_dir, dinovol_ckpt=dinovol_ckpt, out=args.out,
                 recipe_path=args.recipe, epochs=args.epochs, device=args.device,
                 attention_backend=args.attention_backend, variant=args.variant,
                 num_workers=args.num_workers, continue_training=args.continue_)


def chain(args) -> None:
    """``hercuvit train chain`` handler: pull the corpus + Dinovol encoder, build, fit — one command."""
    corpus_dir = resolve_corpus(args.corpus, args.corpus_hf)
    dinovol_ckpt = resolve_dinovol(args.dinovol, args.dinovol_hf, args.dinovol_file)
    run_training(corpus_dir=corpus_dir, dinovol_ckpt=dinovol_ckpt, out=args.out,
                 recipe_path=args.recipe, epochs=args.epochs, device=args.device,
                 attention_backend=args.attention_backend, variant=args.variant,
                 num_workers=args.num_workers, continue_training=args.continue_)

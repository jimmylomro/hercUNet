"""hercuvit — command-line interface for the HercuViT pipeline (parallel to ``hercunet``).

Subcommands, one per stage; the CLI only parses + validates and dispatches via ``set_defaults(func=…)`` with
**lazy imports** inside each handler (so ``--help`` is instant and heavy deps load only when used). Conventions
(as hercunet): every option is a long ``--flag`` with hyphens (no underscores, no single-dash forms), no
environment-variable configuration, British spelling.

    hercuvit train chain --corpus-hf jimmylomro/hercunet-corpus --dinovol-hf scrollprize/dinovol_ps8_9.6um_101626 --out <dir>
    hercuvit train fit   --corpus <dir> --dinovol <ckpt|--dinovol-hf> --out <dir> [--continue]
    hercuvit infer single-instance <scroll|zarr-url> --model <dir>|--model-hf <repo> --out <prefix> [--region …]

Installed as the ``hercuvit`` console script; also ``python -m hercunet.hercuvit``.
"""
from __future__ import annotations

import argparse

_CORPUS_HF = "jimmylomro/hercunet-corpus"
_DINOVOL_HF = "scrollprize/dinovol_ps8_9.6um_101626"
_MODEL_HF = "jimmylomro/hercuvit-v0"
_BACKENDS = ("auto", "flash", "cudnn", "efficient", "math")
_VARIANTS = ("small", "tiny", "nano")


# --------------------------------------------------------------------------- train (chain / fit) --
def _add_train_common(p: argparse.ArgumentParser) -> None:
    # corpus source (local dir or HF dataset)
    p.add_argument("--corpus", default=None, metavar="DIR",
                   help="local herculabels corpus dir (sample_*.npz)")
    p.add_argument("--corpus-hf", dest="corpus_hf", nargs="?", const=_CORPUS_HF, default=None, metavar="REPO",
                   help=f"download the corpus from an HF dataset (bare flag = {_CORPUS_HF})")
    # encoder source — the variant is DEDUCED from the .pt (distilled → small/tiny/nano; raw Dinovol → full)
    p.add_argument("--encoder", default=None, metavar="CKPT",
                   help="encoder checkpoint to start from: a `hercuvit distill` output (→ its windowed variant) OR "
                        "a raw Dinovol teacher (→ 'full', all-global). The variant is read from the file.")
    p.add_argument("--encoder-hf", dest="encoder_hf", nargs="?", const=_DINOVOL_HF, default=None, metavar="REPO",
                   help=f"download the encoder from an HF model repo (bare flag = the raw Dinovol {_DINOVOL_HF})")
    p.add_argument("--encoder-file", dest="encoder_file", default=None, metavar="NAME",
                   help="checkpoint filename within the HF encoder repo (default: teacher_305k.pt)")
    p.add_argument("--out", required=True, metavar="DIR", help="output dir for checkpoints + logs")
    p.add_argument("--recipe", default=None, metavar="FILE.yml",
                   help="YAML recipe defining the net (copy recipes/{small,tiny,nano}.yml and edit); omit to use "
                        "the encoder's variant recipe, or small.yml from scratch")
    p.add_argument("--epochs", type=int, default=None, metavar="N",
                   help="override the total training length (e.g. --epochs 1 for a smoke run)")
    p.add_argument("--device", default="cuda", help="cuda | cpu (default cuda)")
    p.add_argument("--attention-backend", dest="attention_backend", default=None, choices=_BACKENDS,
                   help="SDPA kernel: auto|flash|cudnn|efficient|math (default: the recipe's 'auto'). Use 'auto' "
                        "on GPUs with no cuDNN plan for the windowed shapes (e.g. RTX 4060)")
    p.add_argument("--variant", default=None, choices=_VARIANTS,
                   help="FROM-SCRATCH training of this variant (no encoder) — mutually exclusive with --encoder; "
                        "NOT supported yet (raises). Normal training deduces the variant from --encoder.")
    p.add_argument("--num-workers", dest="num_workers", type=int, default=4, help="DataLoader workers (default 4)")
    p.add_argument("--continue", dest="continue_", action="store_true",
                   help="resume from {out}/checkpoint_latest.pth instead of warm-starting fresh")
    # pre-extraction (default ON): pull the gated training set (CT+labels) to a LOCAL cache via shard processes,
    # then train from local — the live S3 path deadlocks under worker concurrency + re-streams every epoch.
    p.add_argument("--train-cache", dest="train_cache", default=None, metavar="DIR",
                   help="local pre-extracted training cache dir (default: {out}/train_cache)")
    p.add_argument("--prep-shards", dest="prep_shards", type=int, default=8, metavar="N",
                   help="processes for the local training pre-extraction (each single-threaded S3; default 8)")
    p.add_argument("--no-pre-extract", dest="no_pre_extract", action="store_true",
                   help="train directly from the live S3 corpus (slow, re-streams each epoch; use --num-workers 0)")


def _train_chain(args: argparse.Namespace) -> None:
    from .train.chain import chain
    chain(args)


def _train_fit(args: argparse.Namespace) -> None:
    from .train.chain import fit
    fit(args)


def _add_train_chain(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("chain", help="pull corpus + Dinovol, build, and fit (one command)",
                       description="Resolve the corpus (--corpus-hf) and Dinovol encoder (--dinovol-hf), build "
                                   "HercuViT (windowed retrofit + attention-backend fix), and run the staged fit.")
    _add_train_common(p)
    p.set_defaults(func=_train_chain)


def _add_train_fit(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("fit", help="train HercuViT on a (local or pulled) corpus + encoder",
                       description="Build HercuViT from the Dinovol encoder and fit on the herculabels corpus "
                                   "(staged: semantic → unfreeze transformer → add instance head).")
    _add_train_common(p)
    p.set_defaults(func=_train_fit)


def _add_train_prep(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("prep-shard", help="(internal) one shard of the training pre-extraction (CT + labels)",
                       description="Pre-extract one shard of the gated training set (native CT + surf + owner) to a "
                                   "local cache. The trainer launches N of these as separate processes.")
    p.add_argument("--corpus", required=True, metavar="DIR", help="local corpus dir (resolved by the orchestrator)")
    p.add_argument("--cache-dir", dest="cache_dir", required=True, metavar="DIR", help="local training-cache dir")
    p.add_argument("--nshards", type=int, default=8)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--no-identity-only", dest="no_identity_only", action="store_true",
                   help="keep windows with no identity record (default: identity-only, matching the dataset)")
    p.set_defaults(func=lambda a: __import__("hercunet.hercuvit.train.preextract",
                                             fromlist=["prep_run"]).prep_run(a))


# --------------------------------------------------------------------------------- distill --
def _distill(args: argparse.Namespace) -> None:
    # Lazy import so the lead's train/distill.py landing (or not yet) never affects the small path or the tests.
    from .train import distill
    distill.run(args)


def _add_distill(p: argparse.ArgumentParser) -> None:
    """Configure the ``hercuvit distill`` parser (a direct command — architecture/logic is lead-owned)."""
    p.add_argument("--teacher", default=_DINOVOL_HF, metavar="REPO|CKPT",
                   help=f"teacher model (HF repo or local ckpt; default {_DINOVOL_HF})")
    p.add_argument("--student", default="small", choices=("small", "tiny", "nano", "all"),
                   help="windowed variant to distil (small|tiny|nano, or 'all' → <variant>.pt per variant in --out)")
    p.add_argument("--out", required=True, metavar="PATH",
                   help="output encoder (.pt file; a DIR for --student all). Reloaded via --distilled-encoder")
    p.add_argument("--steps", type=int, default=None, help="distillation steps (default: the recipe's)")
    p.add_argument("--corpus-mode", dest="corpus_mode", default="random",
                   choices=("random", "herculabels", "m7", "labels"),
                   help="unlabelled-crop source: random (many scrolls, default) | herculabels (--corpus) | "
                        "m7 (--manifest) | labels (herculabels + m7 together, the hercunet-corpus)")
    p.add_argument("--scrolls", default=None, metavar="IDS",
                   help="comma scroll ids for --corpus-mode random (default: all mined scrolls)")
    p.add_argument("--manifest", default=None, metavar="FILE.json",
                   help="m7 scout manifest for --corpus-mode m7 (approved compressed-sheet regions)")
    p.add_argument("--checkpoint-every", dest="checkpoint_every", type=int, default=0, metavar="N",
                   help="save the encoder every N steps (resumable long runs; default: only at the end)")
    p.add_argument("--prep-shards", dest="prep_shards", type=int, default=8, metavar="N",
                   help="processes for the local CT pre-extraction (each single-threaded S3; default 8)")
    p.add_argument("--corpus", default=None, metavar="DIR", help="local herculabels corpus (for --corpus-mode herculabels)")
    p.add_argument("--corpus-hf", dest="corpus_hf", nargs="?", const=_CORPUS_HF, default=None, metavar="REPO",
                   help=f"download the herculabels corpus from an HF dataset (bare flag = {_CORPUS_HF})")
    p.add_argument("--gram-weight", dest="gram_weight", type=float, default=None, help="Gram distillation weight")
    p.add_argument("--token-weight", dest="token_weight", type=float, default=None, help="per-token distil weight")
    p.add_argument("--cls-weight", dest="cls_weight", type=float, default=None, help="cls/register distil weight")
    p.add_argument("--distill-lr", dest="distill_lr", type=float, default=None, help="peak distillation LR")
    p.add_argument("--warmup", type=int, default=0, metavar="N", help="linear LR warmup steps (serious runs: ~5%)")
    p.add_argument("--lr-schedule", dest="lr_schedule", default="cosine", choices=("cosine", "constant"),
                   help="LR schedule after warmup: cosine decay to ~0 (default) | constant")
    p.add_argument("--batch", type=int, default=1, help="crops per distillation step (default 1; serious: 8)")
    p.add_argument("--recipe", default=None, metavar="FILE.yml", help="YAML recipe (for the distill_* fields)")
    p.add_argument("--device", default="cuda", help="cuda | cpu (default cuda)")
    p.add_argument("--source", default=None, metavar="SCROLL|ZARR",
                   help="scroll id or zarr for real unlabelled CT crops (distil on real data, not synthetic)")
    p.add_argument("--smoke", action="store_true", help="tiny smoke run (few steps, for a quick sanity check)")
    p.set_defaults(func=_distill)


# ----------------------------------------------------------------------------------- infer --
def _infer_single(args: argparse.Namespace) -> None:
    from .infer.infer import single_instance
    single_instance(args)


def _add_infer_single(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("single-instance", help="full-volume semantic sheet-surface inference on one box",
                       description="Stream a scroll / zarr, run HercuViT over Gaussian-blended 128³ windows, and "
                                   "write {out}_sem.zarr (uint8 sheet-surface probability). v1: no cross-tile "
                                   "sheet-ID stitching.")
    p.add_argument("source", metavar="SOURCE",
                   help="CT source: a scroll id (e.g. PHerc1447), 's3://…/x.zarr', 'https://…/x.zarr', "
                        "'file:///abs/x.zarr' or a local path")
    p.add_argument("--model", default=None, metavar="DIR|PTH",
                   help="trained HercuViT checkpoint dir or .pth")
    p.add_argument("--model-hf", dest="model_hf", nargs="?", const=_MODEL_HF, default=None, metavar="REPO",
                   help=f"download the trained model from an HF repo (bare flag = {_MODEL_HF})")
    p.add_argument("--ckpt", default="checkpoint_best.pth", help="checkpoint filename within --model dir")
    p.add_argument("--out", required=True, metavar="PREFIX", help="output prefix — writes {PREFIX}_sem.zarr")
    p.add_argument("--region", default=None, metavar="z0:z1,y0:y1,x0:x1",
                   help="restrict to a sub-cube (level voxels); whole volume if omitted")
    p.add_argument("--level", type=int, default=0, metavar="L",
                   help="OME-Zarr pyramid level to run on (default 0; pick the level nearest ~9.6 µm)")
    p.add_argument("--overlap", type=float, default=0.25, help="window overlap for the Gaussian blend (0 = disjoint)")
    p.add_argument("--air", type=int, default=25, help="skip windows whose CT max < this (all-air; default 25)")
    p.add_argument("--finalise", default="last", metavar="last|no",
                   help="build the OME-Zarr pyramid for the output ('last' = yes, 'no' = skip)")
    p.add_argument("--device", default="cuda", help="cuda | cpu (default cuda)")
    p.add_argument("--attention-backend", dest="attention_backend", default=None, choices=_BACKENDS,
                   help="SDPA kernel override (default: the recipe's; 'auto' is safe on any GPU)")
    p.set_defaults(func=_infer_single)


# --------------------------------------------------------------------------------- dispatch --
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hercuvit",
        description="HercuViT — Dinovol-based full-resolution sheet segmentation (parallel to hercunet).",
    )
    groups = parser.add_subparsers(dest="group", metavar="<stage>", required=True)

    train = groups.add_parser("train", help="training (chain / fit)")
    train_cmds = train.add_subparsers(dest="command", metavar="<step>", required=True)
    _add_train_chain(train_cmds)
    _add_train_fit(train_cmds)
    _add_train_prep(train_cmds)

    distill = groups.add_parser("distill", help="distil a windowed variant (small/tiny/nano) from the Dinovol teacher")
    _add_distill(distill)

    prep = groups.add_parser("distill-prep", help="(internal) one shard of the distillation CT pre-extraction")
    prep.add_argument("--corpus", default=None, metavar="DIR", help="local corpus dir (resolved by the orchestrator)")
    prep.add_argument("--manifest", default=None, metavar="FILE.json", help="m7 scout manifest")
    prep.add_argument("--corpus-mode", dest="corpus_mode", default="labels", choices=("herculabels", "m7", "labels"))
    prep.add_argument("--crops-out", dest="crops_out", required=True, metavar="DIR", help="local crop-cache dir")
    prep.add_argument("--nshards", type=int, default=8)
    prep.add_argument("--shard", type=int, default=0)
    prep.set_defaults(func=lambda a: __import__("hercunet.hercuvit.train.distill", fromlist=["prep_run"]).prep_run(a))

    infer = groups.add_parser("infer", help="full-volume inference")
    infer_cmds = infer.add_subparsers(dest="command", metavar="<mode>", required=True)
    _add_infer_single(infer_cmds)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        import sys
        print("\nhercuvit: interrupted", file=sys.stderr, flush=True)
        raise SystemExit(130)


if __name__ == "__main__":
    main()

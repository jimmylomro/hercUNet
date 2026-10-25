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
    p.add_argument("--device", default="cuda", help="cuda | cpu (default cuda)")
    # --- training config (CLI FLAGS — the yaml is model-only). Unset flags keep the code defaults. ---
    p.add_argument("--epochs", type=int, default=None, metavar="N", help="total training epochs (default 175)")
    p.add_argument("--freeze-warmup-epochs", dest="freeze_warmup_epochs", type=int, default=None, metavar="N",
                   help="LP-FT: epochs with the transformer frozen before fine-tuning (default 50; 0 = from step 0)")
    p.add_argument("--freeze-decoder-epochs", dest="freeze_decoder_epochs", type=int, default=None, metavar="N",
                   help="instance-head warmup: epochs with the warm-started decoder trunk + surface head frozen "
                        "(mask head + instance head still train), then unfreeze for joint fine-tuning (default 0)")
    p.add_argument("--mask-loss-chunks", dest="mask_loss_chunks", type=int, default=None, metavar="N",
                   help="stream the dense mask loss in N checkpointed spatial chunks (never the full Q×128³; 1=dense)")
    p.add_argument("--val-fold", dest="val_fold", type=int, default=None, metavar="N",
                   help="nnU-Net-consistent 5-fold split (KFold seed 12345): which fold to hold out (default 0)")
    p.add_argument("--val-interval", dest="val_interval", type=int, default=None, metavar="N",
                   help="validate per-sheet coverage every N epochs (default 5; 0 = no validation)")
    p.add_argument("--val-windows", dest="val_windows", type=int, default=None, metavar="N",
                   help="held-out windows per validation, capped for speed (default 64; 0 = all)")
    p.add_argument("--batch-size", dest="batch_size", type=int, default=None, metavar="N", help="batch size (default 1)")
    p.add_argument("--base-lr", dest="base_lr", type=float, default=None, metavar="LR", help="base LR (default 1e-4)")
    p.add_argument("--encoder-lr-mult", dest="encoder_lr_mult", type=float, default=None, metavar="M",
                   help="LR multiplier CAP for the encoder — stem + transformer (default 0.1); the ceiling the "
                        "sigmoid ramp saturates at in --encoder-lr-schedule sigmoid")
    p.add_argument("--encoder-lr-schedule", dest="encoder_lr_schedule", default=None, choices=("step", "sigmoid"),
                   help="encoder fine-tune schedule: 'step' (legacy frozen→hard-unfreeze LP-FT) | 'sigmoid' (smooth "
                        "logistic ramp of the encoder LR mult over epochs, cap=encoder-lr-mult; default step)")
    p.add_argument("--encoder-sigmoid-midpoint", dest="encoder_sigmoid_midpoint", type=float, default=None, metavar="EP",
                   help="sigmoid mode: epoch where the encoder LR ramp reaches half the cap (default 120)")
    p.add_argument("--encoder-sigmoid-width", dest="encoder_sigmoid_width", type=float, default=None, metavar="EP",
                   help="sigmoid mode: ramp steepness in epochs (larger = gentler; default 25)")
    p.add_argument("--encoder-grad-enable-frac", dest="encoder_grad_enable_frac", type=float, default=None, metavar="F",
                   help="sigmoid mode: enable encoder grad once m(ep) ≥ F×cap (default 1e-3; deep-toe compute saving)")
    p.add_argument("--decoder-lr-mult", dest="decoder_lr_mult", type=float, default=None, metavar="M",
                   help="LR multiplier for the pixel decoder (default 1.0; <1 fine-tunes a warm-started decoder "
                        "gently while the fresh instance head trains at base LR)")
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=None, help="AdamW weight decay (default 0.05)")
    p.add_argument("--grad-clip", dest="grad_clip", type=float, default=None, help="grad-norm clip (default 1.0)")
    p.add_argument("--warmup-steps", dest="warmup_steps", type=int, default=None, help="LR warmup steps (default 200)")
    p.add_argument("--lr-schedule", dest="lr_schedule", default=None, choices=("constant", "cosine"),
                   help="LR schedule after warmup (default constant)")
    p.add_argument("--augment", dest="augment", default=None, action=argparse.BooleanOptionalAction,
                   help="geometric augmentation (--augment / --no-augment; default on)")
    p.add_argument("--target-voxel-um", dest="target_voxel_um", type=float, default=None,
                   help="resample working scale µm (default 9.6)")
    p.add_argument("--attention-backend", dest="attention_backend", default=None, choices=_BACKENDS,
                   help="SDPA kernel auto|flash|cudnn|efficient|math (default auto; hardware, not model)")
    p.add_argument("--w-surface", dest="w_surface", type=float, default=None,
                   help="weight of the surface/semantic BCE stack on the pixel-decoder surface head (default 0 = off; "
                        "author-style direct thin-surface supervision)")
    p.add_argument("--surface-only", dest="surface_only", default=None, action=argparse.BooleanOptionalAction,
                   help="design-B stage 1: train ONLY the surface head (skip the instance head) — replicate the "
                        "author's surface predictor first (default off)")
    p.add_argument("--w-malis", dest="w_malis", type=float, default=None,
                   help="weight of the optional AffMALIS separation loss (default 0 = off)")
    p.add_argument("--denoise", dest="denoise", default=None, action=argparse.BooleanOptionalAction,
                   help="mask denoising: a train-only fixed-assignment query group per GT sheet (default ON; "
                        "--no-denoise to disable)")
    p.add_argument("--dn-weight", dest="dn_weight", type=float, default=None,
                   help="weight of the denoise-group loss (default 1.0)")
    p.add_argument("--w-mask-bce", dest="w_mask_bce", type=float, default=None,
                   help="weight of the ownership-CE instance term (default 5.0)")
    p.add_argument("--variant", default=None, choices=_VARIANTS,
                   help="FROM-SCRATCH training of this variant (no encoder) — mutually exclusive with --encoder; "
                        "NOT supported yet (raises). Normal training deduces the variant from --encoder.")
    p.add_argument("--num-workers", dest="num_workers", type=int, default=4, help="DataLoader workers (default 4)")
    p.add_argument("--continue", dest="continue_", action="store_true",
                   help="resume from {out}/checkpoint_latest.pth instead of warm-starting fresh")
    p.add_argument("--init-from", dest="init_from", default=None, metavar="CKPT",
                   help="WARM-START: load model weights only (not optimiser/epoch) from this checkpoint before "
                        "training — e.g. carry a surface-only run's decoder into an instance-active run")
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
    p.add_argument("--hint-weight", dest="hint_weight", type=float, default=None,
                   help="intermediate-block (depth-aligned) distillation weight (default 0.5; 0 = final-layer only)")
    p.add_argument("--hint-fractions", dest="hint_fractions", default=None,
                   help="comma-separated relative depths for hint layers (default 0.33,0.66)")
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


def _infer_multi(args: argparse.Namespace) -> None:
    from .infer.infer import multi_instance
    multi_instance(args)


def _add_infer_common(p: argparse.ArgumentParser) -> None:
    """The full HercuViT infer flag set — shared by single/multi, driving the same engine fan-out as ``hercunet infer``
    (streaming + pre-sync, per-GPU/pod ``.done`` claim-queue, leader finalise + S3 upload). HercuViT is single-pass
    and 1-channel, so there is no --passes / --plain / affinity knob."""
    p.add_argument("source", metavar="SOURCE",
                   help="CT source: a scroll id (e.g. PHerc1447), 's3://…/x.zarr', 'https://…/x.zarr', "
                        "'file:///abs/x.zarr' or a local path")
    p.add_argument("--model", default=None, metavar="DIR|PTH", help="trained HercuViT checkpoint dir or .pth")
    p.add_argument("--model-hf", dest="model_hf", nargs="?", const=_MODEL_HF, default=None, metavar="REPO",
                   help=f"download the trained model from an HF repo (bare flag = {_MODEL_HF})")
    p.add_argument("--ckpt", default="checkpoint_best.pth", help="checkpoint filename within --model dir")
    p.add_argument("--out", required=True, metavar="PREFIX", help="output prefix — writes {PREFIX}_pass0.zarr")
    p.add_argument("--region", default=None, metavar="z0:z1,y0:y1,x0:x1",
                   help="restrict to a sub-cube (working-level voxels); whole volume if omitted")
    p.add_argument("--level", type=int, default=0, metavar="L",
                   help="OME-Zarr pyramid level to run on (default 0; pick the level nearest ~9.6 µm)")
    p.add_argument("--pre-sync-source", dest="pre_sync_source", nargs="?", const="<auto>", default=None, metavar="DIR",
                   help="for an s3://|https:// SOURCE: region-download the covering L0 chunks locally (s5cmd) and read "
                        "from there (GPU-bound). Bare flag syncs next to --out; pass a POD-LOCAL DIR for multi-instance")
    p.add_argument("--overlap", type=float, default=0.25, help="window overlap for the Gaussian blend (0 = disjoint)")
    p.add_argument("--tta", default="none", metavar="none|all|mirror|rotate|z,y,x",
                   help="per-window test-time augmentation (villa-style, averages logits); default none")
    p.add_argument("--tta-passes", default="all", dest="tta_passes", metavar="all|last|no",
                   help="which passes to apply --tta on (single-pass model → effectively pass 0); default all")
    p.add_argument("--gpus", default="all", metavar="all|0,1,3",
                   help="GPUs to use on THIS node (default all visible; one worker per GPU)")
    p.add_argument("--batch", type=int, default=4, help="windows per forward batch (default 4)")
    p.add_argument("--nb", type=int, default=4, help="windows per prefetch block/tile per axis (default 4)")
    p.add_argument("--air", type=int, default=25, help="skip windows whose CT max < this (all-air; default 25)")
    p.add_argument("--readahead", type=int, default=2, help="CT block-prefetch depth (default 2)")
    p.add_argument("--prefetch-workers", type=int, default=3, dest="prefetch_workers", help="CT prefetch threads (default 3)")
    p.add_argument("--finalise", default="last", metavar="last|all|no|0",
                   help="which passes get an OME-Zarr pyramid built ('last' default, 'no' = skip, or an index list)")
    p.add_argument("--s3-prefix", default=None, dest="s3_prefix", metavar="s3://…",
                   help="upload finished passes to this S3 prefix with s5cmd (AWS creds in env; write-preflighted)")
    p.add_argument("--upload", default="last", metavar="last|all|no|0",
                   help="which passes to upload to --s3-prefix (default last; ignored without --s3-prefix)")
    p.add_argument("--resume", action="store_true",
                   help="continue an interrupted run: skip _complete passes + re-queue tiles orphaned by killed workers")
    p.add_argument("--keep-buffers", action="store_true", dest="keep_buffers",
                   help="keep every pass buffer (default: prune to the last two)")
    p.add_argument("--claim-chunk", type=int, default=6, dest="claim_chunk", help="claim-queue chunk size per steal (default 6)")
    p.add_argument("--device", default="cuda", help="cuda | cpu (single-instance only; default cuda)")
    p.add_argument("--attention-backend", dest="attention_backend", default=None, choices=_BACKENDS,
                   help="SDPA kernel override (default: the recipe's; 'auto' is safe on any GPU)")


def _add_infer_single(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("single-instance", help="full-volume sheet-surface inference on ONE box (fans across its GPUs)",
                       description="Stream a scroll / zarr and run HercuViT over Gaussian-blended windows via the "
                                   "shared engine; one worker per local GPU (single GPU → in-process; CPU fallback). "
                                   "Writes {out}_pass0.zarr (uint8 surface prob). No cross-tile sheet-ID stitching.")
    _add_infer_common(p)
    p.set_defaults(func=_infer_single)


def _add_infer_multi(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("multi-instance", help="full-volume inference across MANY pods (shared network volume)",
                       description="Run the SAME command on each pod with --out on shared storage; pass --leader on "
                                   "exactly one pod. Each pod fans across its own GPUs on the shared .done claim-queue; "
                                   "the leader finalises the pyramid + uploads.")
    _add_infer_common(p)
    p.add_argument("--leader", action="store_true",
                   help="THIS pod is the leader (creates each pass zarr, finalises + uploads); pass on exactly one pod")
    p.set_defaults(func=_infer_multi)


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
    _add_infer_multi(infer_cmds)

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

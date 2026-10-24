"""hercunet — command-line interface for the HercUNet pipeline.

A single command with subcommands, one per pipeline stage. The CLI only parses and validates
arguments; the actual work lives in the ``hercunet`` package. Conventions: every option is a long
``--flag`` (no single-dash short forms), and flag names use hyphens, never underscores.

Installed as the ``hercunet`` console script (see ``pyproject.toml`` ``[project.scripts]``); also
runnable via ``python -m hercunet``.

Stage 1 — pseudo-label corpus generation — moved to the separate **herculabels** CLI
(``pip install -e ".[herculabels]"``; ``herculabels create|edit|merge|export|m7-mine``; see docs/herculabels.md
and docs/hercuvit-migration.md).

Stage 2 — refiner training chain (needs ``pip install -e ".[train]"``; see docs/stage2-port.md):

    hercunet train export-labels --corpus <c> --out $nnUNet_raw/Dataset301_… [--m7-corpus <m7>] [--max-candidates 4]
    hercunet train preprocess    --dataset 301 [--max-candidates 4] [--config 3d_fullres]
    hercunet train export-owner  --dataset 301
    hercunet train fit           --dataset 301 --pretrained <m7 ckpt> [--num-gpus N]
    hercunet train chain         --corpus <c> [--m7-corpus <m7>] --out … --dataset 301 --pretrained <m7> [--from … --until …]

This is the **HercUNet** (run301 / v0) recipe — the iterative AffinityMalis refiner. ``export-labels`` writes our
gate-passing corpus as CT + K candidate/prev crests + ownersTr (MALIS instance GT), uncleaned, and with
``--m7-corpus`` COPIES a pre-mined m7 corpus (from ``labels m7-mine``) in as ``m7_*`` cases — the rehearsal signal.
``preprocess`` transplants m7's ResEncUNetL plans and patches them to 1+K channels; ``fit`` launches the trainer by
importing the class and calling ``run_training()`` — no ``-tr`` discovery, no copy-paste into the nnunetv2 package
(knobs via the ``CK_*`` env vars). ``chain`` runs every step in order (same code), resumable with ``--from`` /
``--until`` / ``--skip``, over a pinned/unmodified ``nnunetv2==2.8.1``.
"""

from __future__ import annotations

import argparse


# ============================================================================ train (stage 2) ==
# `hercunet train <step>` — the refiner chain as separate, resumable steps over a pinned,
# unmodified nnunetv2==2.8.1 (no fork, no clone, no trainer copy-paste). See docs/training.md.

_M7_HF_REPO = "scrollprize/surface_m7_nnunet"   # the published m7 model (matches train.preprocess.M7_REPO)


def _add_nnunet_path(p: argparse.ArgumentParser) -> None:
    """Shared ``--nnunet-path`` flag: a dir holding nnUNet_raw/ nnUNet_preprocessed/ nnUNet_results/, applied
    in-process so no env export is needed (env vars remain the fallback for split layouts)."""
    p.add_argument("--nnunet-path", "--nnUNet-path", dest="nnunet_path", default=None, metavar="DIR",
                   help="dir containing nnUNet_raw/ nnUNet_preprocessed/ nnUNet_results/ (sets the nnU-Net "
                        "roots for this run; no env export needed). Omit to use the nnUNet_* env vars.")


def _train_export_labels(args: argparse.Namespace) -> None:
    from .train import dataset
    dataset.export_labels(corpus=args.corpus, out=args.out, m7_corpus=args.m7_corpus,
                          surface_tau=args.surface_tau, ignore_score=args.ignore_score,
                          gate_score=args.gate_score, gate_count=args.gate_count,
                          max_candidates=args.max_candidates, confidence_ignore=args.confidence_ignore,
                          prefix=args.prefix, nshards=args.nshards, shard=args.shard)


def _add_train_export_labels(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("export-labels", help="decode our corpus (+ append a pre-mined m7 corpus) into an nnU-Net dataset",
                       description="Export our gate-passing ∇φ pseudo-labels into $nnUNet_raw/DatasetNNN as CT + K "
                                   "candidate/prev crests + ownersTr (MALIS GT); with --m7-corpus, also COPY the "
                                   "pre-mined m7 cases in (no symlinks).")
    p.add_argument("--corpus", required=True, help="our source corpus (dir or .herculabels)")
    p.add_argument("--out", required=True, metavar="DATASET_DIR", help="$nnUNet_raw/DatasetNNN_… to write")
    p.add_argument("--m7-corpus", default=None, metavar="M7_CORPUS",
                   help="a pre-mined m7 corpus (from `hercunet labels m7-mine`) — its cases are copied in as m7_*")
    p.add_argument("--surface-tau", type=float, default=0.91, help="|∇φ| crest threshold (default 0.91 ≈ 2.6 vx)")
    p.add_argument("--ignore-score", type=float, default=0.80, help="collapsed-sheet ignore cutoff")
    p.add_argument("--gate-score", type=float, default=0.85, help="per-sheet gate score")
    p.add_argument("--gate-count", type=int, default=2, help="min sheets passing the gate")
    p.add_argument("--max-candidates", type=int, default=4, help="K candidate/prev channels (default 4)")
    p.add_argument("--confidence-ignore", action=argparse.BooleanOptionalAction, default=False,
                   help="apply the confidence ignore; HercUNet trains UNCLEANED, so default is --no-confidence-ignore")
    p.add_argument("--prefix", default="our", help="case-name prefix (default 'our')")
    p.add_argument("--nshards", type=int, default=1, help="shard the export across N workers")
    p.add_argument("--shard", type=int, default=0, help="this worker's shard index")
    p.set_defaults(func=_train_export_labels)


def _train_preprocess(args: argparse.Namespace) -> None:
    from .train.paths import set_nnunet_roots
    set_nnunet_roots(args.nnunet_path, needs=("nnUNet_raw", "nnUNet_preprocessed"))
    from .train.preprocess import preprocess_dataset
    for i, ds in enumerate(args.dataset):
        preprocess_dataset(ds, configuration=args.config, plans_id=args.plans_id,
                           num_processes=args.np, max_candidates=args.max_candidates,
                           setup_plans=(args.setup_plans and i == 0), m7_repo=args.m7_repo)


def _add_train_preprocess(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("preprocess", help="fingerprint → transplant m7 plans → patch channels → preprocess",
                       description="Wrap the stock nnU-Net funcs to preprocess in m7's ResEncUNetL geometry, "
                                   "patched to 1+K input channels for the iterative refiner.")
    p.add_argument("--dataset", required=True, type=int, nargs="+", metavar="ID",
                   help="dataset id(s) to preprocess, e.g. 301")
    p.add_argument("--max-candidates", type=int, default=4,
                   help="K candidate/prev channels → plans patched to 1+K (0 = single-channel, no patch)")
    p.add_argument("--config", default="3d_fullres", help="nnU-Net configuration (default 3d_fullres)")
    p.add_argument("--plans-id", default="nnUNetResEncUNetLPlans", help="plans identifier to transplant")
    p.add_argument("--np", type=int, default=8, help="worker processes (default 8)")
    p.add_argument("--setup-plans", action=argparse.BooleanOptionalAction, default=True,
                   help="seed Dataset100 from the m7 checkpoint first (on by default)")
    p.add_argument("--m7-repo", default="scrollprize/surface_m7_nnunet", help="HF repo for the m7 plans source")
    _add_nnunet_path(p)
    p.set_defaults(func=_train_preprocess)


def _train_export_owner(args: argparse.Namespace) -> None:
    from .train.paths import set_nnunet_roots
    set_nnunet_roots(args.nnunet_path, needs=("nnUNet_raw", "nnUNet_preprocessed"))
    from .train import dataset
    dataset.export_owner(dataset=args.dataset, out=args.out)


def _add_train_export_owner(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("export-owner", help="write per-sheet owner-id MALIS sidecars",
                       description="Write <case>_owner.b2nd (instance GT at preprocessed resolution) for MALIS.")
    p.add_argument("--dataset", required=True, type=int, metavar="ID", help="preprocessed dataset id")
    p.add_argument("--out", default=None, metavar="DIR", help="owner_b2nd output dir (default: alongside the dataset)")
    _add_nnunet_path(p)
    p.set_defaults(func=_train_export_owner)


def _train_fit(args: argparse.Namespace) -> None:
    from .train.paths import set_nnunet_roots
    set_nnunet_roots(args.nnunet_path, needs=("nnUNet_preprocessed", "nnUNet_results"))
    from .train.launch import fit, resolve_pretrained
    pretrained = resolve_pretrained(args.pretrained, args.pretrained_hf, args.pretrained_hf_file)
    fit(args.trainer, args.dataset, configuration=args.config, fold=args.fold,
        plans_identifier=args.plans_id, pretrained=pretrained, num_gpus=args.num_gpus,
        device=args.device, continue_training=args.continue_,
        recipe_path=args.recipe, epochs=args.epochs)


def _add_train_fit(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("fit", help="train the refiner (direct instantiation; no -tr discovery)",
                       description="Launch the HercUNet trainer by importing the class and calling run_training(). "
                                   "Trainer/DAgger/MALIS knobs are read from the CK_* env vars, as in the run301 chain.")
    p.add_argument("--trainer", default="hercunet",
                   help="alias 'hercunet' (the AffinityMalis iterative-DAgger refiner) or an explicit module:ClassName")
    p.add_argument("--dataset", required=True, help="preprocessed dataset id or DatasetNNN_… name")
    p.add_argument("--config", default="3d_fullres", help="nnU-Net configuration (default 3d_fullres)")
    p.add_argument("--fold", default=0, help="cross-val fold (int or 'all'; default 0)")
    p.add_argument("--plans-id", default="nnUNetResEncUNetLPlans", help="plans identifier")
    p.add_argument("--pretrained", default=None, metavar="CKPT", help="warm-start from a local .pth (e.g. the m7 checkpoint)")
    p.add_argument("--pretrained-hf", "--pretrained-huggingface", dest="pretrained_hf", nargs="?",
                   const=_M7_HF_REPO, default=None, metavar="REPO",
                   help=f"warm-start from a HuggingFace repo (downloads the checkpoint); bare flag pulls the m7 "
                        f"model ({_M7_HF_REPO})")
    p.add_argument("--pretrained-hf-file", default=None, metavar="NAME",
                   help="checkpoint filename within the HF repo (default: auto-resolve checkpoint_best/final.pth)")
    p.add_argument("--num-gpus", type=int, default=1, help="DDP world size (>1 spawns workers)")
    p.add_argument("--device", default="cuda", help="cuda | cpu (default cuda)")
    p.add_argument("--continue", dest="continue_", action="store_true",
                   help="resume from the last checkpoint instead of warm-starting")
    p.add_argument("--recipe", default=None, metavar="FILE.yml",
                   help="train with a YAML recipe file (copy hercunet/recipes/hercunet.yml and edit); omit for "
                        "the exact run301 recipe")
    p.add_argument("--epochs", type=int, default=None, metavar="N",
                   help="override the training length (e.g. --epochs 1 for a smoke run); default is the recipe's")
    _add_nnunet_path(p)
    p.set_defaults(func=_train_fit)


# ------------------------------------------------------------------------------- train chain --
_CHAIN_STEPS = ("export-labels", "preprocess", "export-owner", "fit")


def _selected_steps(args: argparse.Namespace) -> list[str]:
    """The steps to run, honouring --from / --until / --skip (defaults to all, in order)."""
    lo = _CHAIN_STEPS.index(args.from_) if args.from_ else 0
    hi = _CHAIN_STEPS.index(args.until) if args.until else len(_CHAIN_STEPS) - 1
    if lo > hi:
        raise SystemExit(f"hercunet train chain: --from {args.from_} is after --until {args.until}")
    skip = set(args.skip or ())
    return [s for s in _CHAIN_STEPS[lo:hi + 1] if s not in skip]


def _need(args: argparse.Namespace, *names: str) -> None:
    missing = [f"--{n.replace('_', '-')}" for n in names if getattr(args, n) is None]
    if missing:
        raise SystemExit(f"hercunet train chain: this step needs {', '.join(missing)}")


def _train_chain(args: argparse.Namespace) -> None:
    from .train import dataset as ds
    from .train.preprocess import preprocess_dataset
    from .train.launch import fit as _fit, resolve_pretrained

    steps = _selected_steps(args)
    # set the nnU-Net roots (--nnunet-path or env) for exactly the steps that touch them, before any nnU-Net call
    needs = set()
    if {"preprocess", "export-owner"} & set(steps):
        needs |= {"nnUNet_raw", "nnUNet_preprocessed"}
    if "fit" in steps:
        needs |= {"nnUNet_preprocessed", "nnUNet_results"}
    if needs:
        from .train.paths import set_nnunet_roots
        set_nnunet_roots(args.nnunet_path, needs=tuple(needs))
    print(f"[chain] running: {' -> '.join(steps)}", flush=True)
    for step in steps:
        print(f"[chain] === {step} ===", flush=True)
        if step == "export-labels":
            _need(args, "corpus", "out")
            ds.export_labels(corpus=args.corpus, out=args.out, m7_corpus=args.m7_corpus,
                             surface_tau=args.surface_tau, ignore_score=args.ignore_score,
                             gate_score=args.gate_score, gate_count=args.gate_count,
                             max_candidates=args.max_candidates, confidence_ignore=False,
                             prefix="our", nshards=1, shard=0)
        elif step == "preprocess":
            _need(args, "dataset")
            preprocess_dataset(int(args.dataset), configuration=args.config, plans_id=args.plans_id,
                               num_processes=args.np, max_candidates=args.max_candidates)
        elif step == "export-owner":
            _need(args, "dataset")
            ds.export_owner(dataset=int(args.dataset), out=None)
        elif step == "fit":
            _need(args, "dataset", "trainer")
            pretrained = resolve_pretrained(args.pretrained, args.pretrained_hf, args.pretrained_hf_file)
            _fit(args.trainer, args.dataset, configuration=args.config, fold=args.fold,
                 plans_identifier=args.plans_id, pretrained=pretrained,
                 num_gpus=args.num_gpus, device=args.device,
                 recipe_path=args.recipe, epochs=args.epochs)
    print("[chain] done", flush=True)


def _add_train_chain(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("chain", help="run the whole chain (export → … → fit) in order",
                       description="Run every train step in sequence. Resume with --from / --until / --skip "
                                   "(each step is the same code as its standalone subcommand).")
    # step selection (resumability)
    p.add_argument("--from", dest="from_", choices=_CHAIN_STEPS, default=None, help="start at this step")
    p.add_argument("--until", choices=_CHAIN_STEPS, default=None, help="stop after this step")
    p.add_argument("--skip", nargs="+", choices=_CHAIN_STEPS, default=None, help="skip these steps")
    # dataset assembly
    p.add_argument("--corpus", default=None, help="our source corpus (export-labels)")
    p.add_argument("--out", default=None, metavar="DATASET_DIR", help="$nnUNet_raw/DatasetNNN_… (our + copied m7 cases)")
    p.add_argument("--m7-corpus", default=None, metavar="M7_CORPUS",
                   help="a pre-mined m7 corpus (from `hercunet labels m7-mine`) to copy in as m7_* cases")
    p.add_argument("--max-candidates", type=int, default=4, help="K candidate/prev channels (export / preprocess)")
    p.add_argument("--surface-tau", type=float, default=0.91, help="export |∇φ| crest threshold")
    p.add_argument("--ignore-score", type=float, default=0.80, help="export collapsed-sheet ignore cutoff")
    p.add_argument("--gate-score", type=float, default=0.85, help="export per-sheet gate score")
    p.add_argument("--gate-count", type=int, default=2, help="export min sheets passing the gate")
    # preprocess + fit
    p.add_argument("--dataset", default=None, help="dataset id for preprocess / export-owner / fit")
    p.add_argument("--config", default="3d_fullres", help="nnU-Net configuration")
    p.add_argument("--plans-id", default="nnUNetResEncUNetLPlans", help="plans identifier")
    p.add_argument("--np", type=int, default=8, help="preprocess worker processes")
    p.add_argument("--trainer", default="hercunet", help="fit: alias 'hercunet' or module:ClassName")
    p.add_argument("--pretrained", default=None, metavar="CKPT", help="fit: warm-start from a local .pth")
    p.add_argument("--pretrained-hf", "--pretrained-huggingface", dest="pretrained_hf", nargs="?",
                   const=_M7_HF_REPO, default=None, metavar="REPO",
                   help=f"fit: warm-start from a HuggingFace repo (bare flag = the m7 model, {_M7_HF_REPO})")
    p.add_argument("--pretrained-hf-file", default=None, metavar="NAME", help="fit: checkpoint filename in the HF repo")
    p.add_argument("--fold", default=0, help="fit: cross-val fold")
    p.add_argument("--num-gpus", type=int, default=1, help="fit: DDP world size")
    p.add_argument("--device", default="cuda", help="fit: cuda | cpu")
    p.add_argument("--recipe", default=None, metavar="FILE.yml", help="fit: YAML recipe file (else run301)")
    p.add_argument("--epochs", type=int, default=None, metavar="N", help="fit: override training length")
    _add_nnunet_path(p)
    p.set_defaults(func=_train_chain)


# ------------------------------------------------------------------------------- infer (stage 3) --
_MODEL_HF_REPO = "jimmylomro/hercunet-v0"


def _add_infer_common(p: argparse.ArgumentParser) -> None:
    # model source (exactly one of --model / --model-hf)
    p.add_argument("--model", default=None, metavar="DIR",
                   help="local nnU-Net model folder (plans.json + dataset.json + dataset_fingerprint.json + fold_0/)")
    p.add_argument("--model-hf", "--model-huggingface", dest="model_hf", nargs="?",
                   const=_MODEL_HF_REPO, default=None, metavar="REPO",
                   help=f"download the model from a HuggingFace repo (public, no token); bare flag = HercUNet v0 "
                        f"({_MODEL_HF_REPO})")
    p.add_argument("--ckpt", default="checkpoint_best.pth", help="checkpoint filename within fold_0 (default best)")
    # what to infer — a SINGLE positional CT source
    p.add_argument("source", metavar="SOURCE",
                   help="CT OME-Zarr source (positional): 's3://bucket/x.zarr' or 'https://…/x.zarr' (streamed), "
                        "'file:///abs/x.zarr' or a local path (read locally — the fast NVMe/tmpfs path), or a bare "
                        "scroll id like 'PHerc1447' (resolved via the data-layer catalog; the URL forms skip that "
                        "bucket listing)")
    p.add_argument("--out", required=True, metavar="PREFIX",
                   help="output prefix — writes {PREFIX}_pass{p}.zarr (per-pass OME-Zarr surface-prob pyramids)")
    p.add_argument("--region", default=None, metavar="z0:z1,y0:y1,x0:x1",
                   help="restrict to a sub-cube — the WHOLE volume is inferred if omitted; use for a small smoke "
                        "test. Coordinates are in the WORKING level's voxels (see --level; = L0 voxels at the "
                        "default --level 0)")
    p.add_argument("--level", type=int, default=0, metavar="L",
                   help="OME-Zarr pyramid level to run on (default 0 = full res). The model was trained at the "
                        "NATIVE-COARSE ~7.5-9.5 µm band, so for a finer rescan (e.g. a 2.4 µm scroll) pick the level "
                        "nearest ~8.6 µm — usually --level 2 (2.4 µm -> 9.6 µm). A BIG warning prints if the chosen "
                        "level's voxel size falls outside ~[7.5, 10] µm (the model will be unreliable there)")
    p.add_argument("--pre-sync-source", dest="pre_sync_source", nargs="?", const="<auto>", default=None,
                   metavar="DIR",
                   help="for an s3://|https:// SOURCE: region-download only the covering L0 chunks locally (s5cmd) "
                        "and read from there — streaming becomes GPU-bound, without fetching the whole volume. Bare "
                        "flag syncs next to --out; pass a DIR to choose where (use a POD-LOCAL dir for multi-instance). "
                        "Ignored for a file://|local source")
    p.add_argument("--passes", type=int, default=4, help="Jacobi refinement passes (default 4; final pass = deliverable)")
    p.add_argument("--overlap", type=float, default=0.25,
                   help="window overlap for the Gaussian blend (default 0.25 = HercUNet v0); 0 = disjoint mode")
    p.add_argument("--tta", default="none", metavar="none|all|mirror|rotate|z,y,x",
                   help="test-time augmentation, applied per window (villa-style, averages logits): 'none' "
                        "(default), 'all' (mirrors z,y,x + rotations), 'mirror', 'rotate', or a comma list incl. "
                        "axes z,y,x (e.g. 'mirror,rotate' or 'z,y'). N variants → N× forwards on the chosen passes")
    p.add_argument("--tta-passes", default="all", dest="tta_passes", metavar="all|last|no|2,3",
                   help="which passes to apply --tta on (same grammar as --finalise): default 'all'; e.g. '2,3'. "
                        "Ignored when --tta none")
    p.add_argument("--gpus", default="all", metavar="all|0,1,3",
                   help="GPUs to use on THIS node (default all visible; one worker process per GPU)")
    # model shape
    p.add_argument("--plain", action="store_true",
                   help="2-channel [CT, prev] model instead of the 8-channel affinity model (default: affinity)")
    p.add_argument("--n-orient", type=int, default=6, dest="n_orient",
                   help="orientation channels for the affinity model (default 6)")
    # throughput / batching
    p.add_argument("--compile", action="store_true", dest="compile",
                   help="torch.compile the network (lossless ~1.37x on a datacenter GPU; one-time triton autotune "
                        "warmup of a few minutes, amortised over a full-scroll run). Bandwidth-bound, so batch size "
                        "does not matter — keep --batch 4")
    p.add_argument("--batch", type=int, default=4, help="windows per forward batch (default 4; bandwidth-bound → "
                   "batch size barely changes throughput, so no need to raise it)")
    p.add_argument("--nb", type=int, default=4, help="windows per prefetch block/tile per axis (default 4)")
    p.add_argument("--air", type=int, default=25, help="skip windows whose CT max < this (all-air; default 25)")
    p.add_argument("--readahead", type=int, default=2, help="CT block-prefetch depth (blocks in flight; default 2)")
    p.add_argument("--prefetch-workers", type=int, default=3, dest="prefetch_workers",
                   help="CT block-prefetch threads (default 3)")
    # output / resume
    p.add_argument("--finalise", default="last", metavar="last|all|no|0,2,3",
                   help="which passes get an OME-Zarr pyramid built (levels 1..N): 'last' (default, only the "
                        "deliverable), 'all', 'no' (none — build later on a CPU box), or a pass-index list like "
                        "'0,2,3' (negatives count from the end, so '-1' == last)")
    p.add_argument("--s3-prefix", default=None, dest="s3_prefix", metavar="s3://…",
                   help="upload finished passes to this S3 prefix with s5cmd (AWS creds in env). A write preflight "
                        "runs before inference starts, so bad creds fail fast; uploads log throttled progress")
    p.add_argument("--upload", default="last", metavar="last|all|no|0,2,3",
                   help="which passes to upload to --s3-prefix (same grammar as --finalise): 'last' (default), "
                        "'all', 'no', or a pass-index list like '0,2,3'. Ignored without --s3-prefix")
    p.add_argument("--resume", action="store_true",
                   help="continue an interrupted run: skip passes already marked _complete, and re-queue tiles "
                        "orphaned by killed workers (a stale .claim with no .done) so they aren't skipped. (A fresh "
                        "worker joining a LIVE multi-instance job omits --resume, so it won't disturb others' claims.)")
    p.add_argument("--keep-buffers", action="store_true", dest="keep_buffers",
                   help="keep every pass buffer (default: prune to the last two)")
    p.add_argument("--keep-affinity", action="store_true", dest="keep_affinity",
                   help="also save the affinity-head output of the FINAL pass to {PREFIX}_pass{last}_aff.zarr "
                        "(4-D (n_aff,Z,Y,X) uint8, aff=v/255 = same-sheet prob per offset). Requires the affinity "
                        "model (not --plain) and overlap>0. The shipped surface path skips the head; this re-enables "
                        "it on the deliverable pass for the band-refinement probe. With --s3-prefix it uploads "
                        "alongside the final pass as pass{last}_aff.zarr")
    p.add_argument("--claim-chunk", type=int, default=6, dest="claim_chunk",
                   help="claim-queue chunk size per steal (default 6)")


def _infer_preflight(args: argparse.Namespace) -> None:
    """Cross-flag warnings before dispatch. TTA + --keep-affinity: a mirror/rotate variant permutes the affinity
    offset channels (channel c = same-sheet-ness of the pair (u, u+δ_c)), so averaging augmented forwards would
    scramble the field. The engine therefore captures affinity TTA-FREE (identity forward only) — we warn so the
    user knows the saved affinity does NOT reflect the TTA'd surface. (We recommend not using TTA with HercUNet
    anyway.)"""
    import sys
    tta_on = str(getattr(args, "tta", "none")).strip().lower() not in ("", "none", "no", "off")
    if tta_on and getattr(args, "keep_affinity", False):
        print("hercunet infer: WARNING — --tta is set with --keep-affinity. TTA flips/rotations permute the "
              "affinity offset channels, so the affinity head is captured TTA-FREE (identity forward only) and will "
              "NOT reflect the TTA'd surface. (HercUNet is not recommended with TTA regardless.)",
              file=sys.stderr, flush=True)


def _infer_single(args: argparse.Namespace) -> None:
    _infer_preflight(args)
    from .infer.run import single_instance
    single_instance(args)


def _infer_multi(args: argparse.Namespace) -> None:
    _infer_preflight(args)
    from .infer.run import multi_instance
    multi_instance(args)


def _add_infer_single(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("single-instance", help="run inference on ONE box, fanning across its local GPUs",
                       description="Iterative Jacobi refinement on one machine. Uses all visible GPUs (one worker "
                                   "process per GPU, --gpus to restrict); a single GPU runs in-process. One command.")
    _add_infer_common(p)
    p.set_defaults(func=_infer_single)


def _add_infer_multi(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("multi-instance", help="run inference across MANY pods (shared network volume)",
                       description="Iterative Jacobi refinement across pods. Run the SAME command on each pod with "
                                   "--out on shared storage; pass --leader on exactly one pod. Each pod also fans "
                                   "across its own local GPUs.")
    _add_infer_common(p)
    p.add_argument("--leader", action="store_true",
                   help="THIS pod is the leader (creates each pass zarr, finalizes + uploads); pass on exactly one pod")
    p.set_defaults(func=_infer_multi)


# --------------------------------------------------------------------------------- dispatch --
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hercunet",
        description="HercUNet — cross-volume surface (sheet) detection pipeline.",
    )
    groups = parser.add_subparsers(dest="group", metavar="<stage>", required=True)

    train = groups.add_parser("train", help="refiner training chain (stage 2)")
    train_cmds = train.add_subparsers(dest="command", metavar="<step>", required=True)
    _add_train_export_labels(train_cmds)
    _add_train_preprocess(train_cmds)
    _add_train_export_owner(train_cmds)
    _add_train_fit(train_cmds)
    _add_train_chain(train_cmds)

    infer = groups.add_parser("infer", help="full-volume iterative inference (stage 3)")
    infer_cmds = infer.add_subparsers(dest="command", metavar="<mode>", required=True)
    _add_infer_single(infer_cmds)
    _add_infer_multi(infer_cmds)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:                                    # clean Ctrl-C — no traceback
        import sys
        print("\nhercunet: interrupted", file=sys.stderr, flush=True)
        raise SystemExit(130)


if __name__ == "__main__":
    main()

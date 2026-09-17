"""hercunet — command-line interface for the HercUNet pipeline.

A single command with subcommands, one per pipeline stage. The CLI only parses and validates
arguments; the actual work lives in the ``hercunet`` package. Conventions: every option is a long
``--flag`` (no single-dash short forms), and flag names use hyphens, never underscores.

Installed as the ``hercunet`` console script (see ``pyproject.toml`` ``[project.scripts]``); also
runnable via ``python -m hercunet``.

Stage 1 — pseudo-label corpora (``.herculabels``; see docs/herculabels.md):

    hercunet labels create <corpus.herculabels> --interactive [--scroll ID] [--coords-file F] [--himat F]
    hercunet labels create <corpus.herculabels> --count <N> [--scroll ID] [--coords Z,Y,X] [--himat F]
    hercunet labels edit   <corpus.herculabels>
    hercunet labels merge  <out.herculabels> <in1.herculabels> <in2.herculabels> [...]
    hercunet labels export <corpus.herculabels> <train_out> [--no-augment]
    hercunet labels m7-mine <m7_corpus> [--scroll s1,s5 | --manifest scout.json] [--max-candidates 4]

``create`` builds a new corpus (fails if it exists): ``--interactive`` opens the viewer and grinds windows
open-endedly (no ``--count``); otherwise it generates ``--count`` windows headless. ``edit`` re-opens the
corpus in the viewer for correction. ``merge`` unions several corpora into a new one (so you extend a corpus
by creating a fresh one and merging). ``export`` derives the training corpus (augmentations on by default).
``m7-mine`` pre-extracts an m7 pseudo-label corpus from compressed regions (consumed by
``train export-labels --m7-corpus``).

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


# --------------------------------------------------------------------------- labels create --
def _labels_create(args: argparse.Namespace) -> None:
    if args.interactive:
        if args.count is not None:
            raise SystemExit("hercunet labels create: cannot use --count with --interactive — "
                             "interactive mode grinds windows open-endedly (close the viewer to stop)")
    elif args.coords_file:
        if args.count is not None:
            raise SystemExit("hercunet labels create: cannot use --count with --coords-file — "
                             "the file defines exactly which windows to generate")
    elif args.count is None:
        raise SystemExit("hercunet labels create: --count is required (unless --interactive or --coords-file)")
    if args.coords is not None:
        if not args.scroll:
            raise SystemExit("hercunet labels create: --coords requires an explicit --scroll")
        if not args.interactive and not args.coords_file and args.count != 1:
            raise SystemExit("hercunet labels create: --coords targets one exact window, so requires --count 1")
    if args.himat is not None and not (0.0 <= args.himat <= 1.0):
        raise SystemExit("hercunet labels create: --himat must be a fraction in [0, 1]")

    from .labels import create

    create(
        args.corpus,
        count=args.count,
        interactive=args.interactive,
        scroll=args.scroll,
        coords=args.coords,
        coords_file=args.coords_file,
        himat=args.himat,
        voxel_min=args.voxel_min,
        voxel_max=args.voxel_max,
        seed_stride_um=args.seed_stride_um,
        sample_um=args.sample_um,
        min_cluster_size=args.min_cluster_size,
        old_negatives=args.old_negatives,
        seed=args.seed,
        gpu=args.gpu,
        deterministic=args.deterministic,
    )


def _add_labels_create(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "create",
        help="build a new .herculabels pseudo-label corpus",
        description="Generate pseudo-label windows into a new .herculabels corpus (fails if it exists).",
    )
    p.add_argument("corpus", metavar="CORPUS.herculabels",
                   help="path of the corpus container to create (must end in .herculabels; must not exist)")
    p.add_argument("--interactive", action="store_true",
                   help="open the viewer and grind windows open-endedly (no --count — close the viewer to "
                        "stop); each visited/edited window is written to the corpus")
    p.add_argument("--count", type=int, default=None,
                   help="number of windows to generate headless (required unless --interactive)")
    p.add_argument("--scroll", default=None,
                   help="scroll id to sample from, e.g. PHerc1447 (default: the standard corpus set)")
    p.add_argument("--coords", default=None, metavar="Z,Y,X",
                   help="one exact window centred at these voxel coords (requires --scroll; headless requires "
                        "--count 1; interactive starts here then grinds open-endedly)")
    p.add_argument("--coords-file", default=None, metavar="FILE",
                   help="generate exactly the windows listed in this file (one per line as 'Z,Y,X' with "
                        "--scroll, or 'SCROLL,Z,Y,X'; '#'/blank lines ignored). Headless: generates them all "
                        "into the corpus (no --count). With --interactive: grinds through them in order")
    p.add_argument("--himat", type=float, default=None, metavar="FRAC",
                   help="only accept windows whose material fill is at least FRAC (0-1) — mine HIGH-material "
                        "sheets via the coarse mask. Default: the standard 0.15 air-skip threshold")
    # generation knobs (defaults match the validated corpus operating point)
    p.add_argument("--voxel-min", type=float, default=7.5,
                   help="lower bound of the voxel-size band (µm) to sample scans from")
    p.add_argument("--voxel-max", type=float, default=9.5,
                   help="upper bound of the voxel-size band (µm) to sample scans from")
    p.add_argument("--seed-stride-um", type=float, default=40.0,
                   help="spine seed spacing in µm (larger = fewer streamlets, faster)")
    p.add_argument("--sample-um", type=float, default=20.0,
                   help="streamlet point sampling spacing in µm")
    p.add_argument("--min-cluster-size", type=int, default=250,
                   help="minimum cluster size (in units) for the probeom clustering")
    p.add_argument("--old-negatives", action="store_true",
                   help="use the OLD gap-gated negatives (submission/writeup/herculabels.md §5.2) that "
                        "generated the PUBLISHED corpus, instead of the default slab-field negatives (§5.3, the "
                        "current method). REQUIRED to reproduce the published corpus (run20260808)")
    p.add_argument("--seed", type=int, default=None,
                   help="base seed for the window-sampling stream; omit for a fresh random draw each run "
                        "(the seed used is printed and recorded in the manifest), or pass a value to reproduce a run")
    p.add_argument("--no-gpu", dest="gpu", action="store_false", default=True,
                   help="force CPU (GPU is used by default; label generation is impractically slow on CPU)")
    p.add_argument("--no-deterministic", dest="deterministic", action="store_false", default=True,
                   help="disable bit-reproducible generation (faster; determinism is on by default)")
    p.set_defaults(func=_labels_create)


# ----------------------------------------------------------------------------- labels edit --
def _labels_edit(args: argparse.Namespace) -> None:
    from .labels import edit_corpus
    edit_corpus(args.corpus, gpu=args.gpu)


def _add_labels_edit(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "edit",
        help="open an editable .herculabels corpus in the viewer",
        description="Re-open a .herculabels corpus in the viewer to correct its windows (writes in place).",
    )
    p.add_argument("corpus", metavar="CORPUS.herculabels", help="path of the corpus to edit")
    p.add_argument("--no-gpu", dest="gpu", action="store_false", default=True,
                   help="force CPU (GPU is used by default for the per-window CT re-read)")
    p.set_defaults(func=_labels_edit)


# ---------------------------------------------------------------------------- labels merge --
def _labels_merge(args: argparse.Namespace) -> None:
    from .labels import merge_corpora
    merge_corpora(args.out, args.inputs)


def _add_labels_merge(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "merge",
        help="merge several .herculabels corpora into a new one",
        description="Union the windows of INPUTS into a new corpus OUT (fails if OUT exists; inputs "
                    "untouched). On a duplicate window the edited copy wins, else the later input.",
    )
    p.add_argument("out", metavar="OUT.herculabels", help="path of the merged corpus to create (must not exist)")
    p.add_argument("inputs", metavar="IN.herculabels", nargs="+", help="corpora to merge (one or more)")
    p.set_defaults(func=_labels_merge)


# --------------------------------------------------------------------------- labels export --
def _labels_export(args: argparse.Namespace) -> None:
    from .labels import export
    export(args.corpus, args.train_out, augment=args.augment, gpu=args.gpu)


def _add_labels_export(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "export",
        help="derive a training corpus from a .herculabels corpus",
        description="Write the per-window training samples (base + augmentations by default) to TRAIN_OUT.",
    )
    p.add_argument("corpus", metavar="CORPUS.herculabels", help="path of the corpus to export from")
    p.add_argument("train_out", metavar="TRAIN_OUT", help="directory to write the training .npz samples into")
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                   help="generate the merge/split augmentation records (on by default; --no-augment for a "
                        "base-only export, e.g. a validation set)")
    p.add_argument("--no-gpu", dest="gpu", action="store_false", default=True,
                   help="force CPU (GPU is used by default for the ∇φ / mesh recompute)")
    p.set_defaults(func=_labels_export)


# ----------------------------------------------------------------------------- labels m7-mine --
def _labels_m7mine(args: argparse.Namespace) -> None:
    from .labels import m7mine
    m7mine.mine(out=args.out, scroll=args.scroll, manifest=args.manifest,
                max_candidates=args.max_candidates, tau=args.tau, workers=args.workers, images=args.images)


def _add_labels_m7mine(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("m7-mine", help="mine m7 pseudo-labels from compressed regions into an m7 corpus",
                       description="Scout coherent m7 crest in compressed regions (from the PUBLISHED m7 surface "
                                   "predictions) and write a pre-extracted m7 corpus of {0,1,2} labels. Consumed by "
                                   "`hercunet train export-labels --m7-corpus`. --manifest rebuilds an approved set.")
    p.add_argument("out", metavar="M7_CORPUS", help="m7 corpus directory to write")
    p.add_argument("--scroll", default=None, help="scroll id(s) to scout, e.g. s1,s5 (fresh scout)")
    p.add_argument("--manifest", default=None, help="rebuild the approved regions from this scout manifest.json (no re-scout)")
    p.add_argument("--max-candidates", type=int, default=4, help="K candidate slots (MUST match train export-labels)")
    p.add_argument("--tau", type=float, default=None, help="m7 normal-coherence gate (default: the validated value)")
    p.add_argument("--workers", type=int, default=8, help="parallel region readers (overlap anon-S3 latency)")
    p.add_argument("--images", action="store_true",
                   help="scout: also write the 3-panel QC render per approved region (raw m7 | coherence | "
                        "grabbed-coherent) and record its filename in the manifest")
    p.set_defaults(func=_labels_m7mine)


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
    # what to infer
    p.add_argument("--scroll", required=True, help="scroll id / name (resolved against the data-layer catalog)")
    p.add_argument("--out", required=True, metavar="PREFIX",
                   help="output prefix — writes {PREFIX}_pass{p}.zarr (per-pass OME-Zarr surface-prob pyramids)")
    p.add_argument("--region", default=None, metavar="z0:z1,y0:y1,x0:x1",
                   help="restrict to a sub-cube — the WHOLE scroll is inferred if omitted; use for a small smoke test")
    p.add_argument("--local-vol", default=None, dest="local_vol", metavar="PATH",
                   help="read a locally-downloaded copy of the scroll's OME-Zarr (fast NVMe/tmpfs) instead of "
                        "streaming from S3 — I/O-bound → GPU-bound; results are byte-identical. Only L0 is needed")
    p.add_argument("--passes", type=int, default=4, help="Jacobi refinement passes (default 4; final pass = deliverable)")
    p.add_argument("--overlap", type=float, default=0.25,
                   help="window overlap for the Gaussian blend (default 0.25 = HercUNet v0); 0 = disjoint mode")
    p.add_argument("--gpus", default="all", metavar="all|0,1,3",
                   help="GPUs to use on THIS node (default all visible; one worker process per GPU)")
    # model shape
    p.add_argument("--plain", action="store_true",
                   help="2-channel [CT, prev] model instead of the 8-channel affinity model (default: affinity)")
    p.add_argument("--n-orient", type=int, default=6, dest="n_orient",
                   help="orientation channels for the affinity model (default 6)")
    # throughput / batching
    p.add_argument("--batch", type=int, default=4, help="windows per forward batch (default 4)")
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
    p.add_argument("--resume", action="store_true", help="skip passes already marked _complete")
    p.add_argument("--keep-buffers", action="store_true", dest="keep_buffers",
                   help="keep every pass buffer (default: prune to the last two)")
    p.add_argument("--reclaim", action="store_true", help="re-queue blocks/tiles orphaned by a crashed worker")
    p.add_argument("--claim-chunk", type=int, default=6, dest="claim_chunk",
                   help="claim-queue chunk size per steal (default 6)")


def _infer_single(args: argparse.Namespace) -> None:
    from .infer.run import single_instance
    single_instance(args)


def _infer_multi(args: argparse.Namespace) -> None:
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

    labels = groups.add_parser("labels", help="pseudo-label generation (stage 1)")
    labels_cmds = labels.add_subparsers(dest="command", metavar="<command>", required=True)
    _add_labels_create(labels_cmds)
    _add_labels_edit(labels_cmds)
    _add_labels_merge(labels_cmds)
    _add_labels_export(labels_cmds)
    _add_labels_m7mine(labels_cmds)

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

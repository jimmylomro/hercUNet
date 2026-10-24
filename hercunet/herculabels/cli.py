"""herculabels — command-line interface for HercuLabels pseudo-label generation (stage 1).

Installed as the ``herculabels`` console script (see ``pyproject.toml`` ``[project.scripts]``); also runnable via
``python -m hercunet.herculabels.cli``. Conventions: every option is a long ``--flag`` (hyphens, no underscores).

    herculabels create <corpus.herculabels> --interactive [--scroll ID] [--coords-file F] [--himat F]
    herculabels create <corpus.herculabels> --count <N> [--scroll ID] [--coords Z,Y,X] [--himat F]
    herculabels edit   <corpus.herculabels>
    herculabels merge  <out.herculabels> <in1.herculabels> <in2.herculabels> [...]
    herculabels export <corpus.herculabels> <train_out> [--no-augment]
    herculabels m7-mine <m7_corpus> [--scroll s1,s5 | --manifest scout.json] [--max-candidates 4]

``create`` builds a new corpus (fails if it exists); ``edit`` re-opens it in the viewer; ``merge`` unions corpora;
``export`` derives the training corpus (augmentations on by default); ``m7-mine`` pre-extracts an m7 pseudo-label
corpus (consumed by ``hercunet train export-labels --m7-corpus``). See docs/herculabels.md.
"""

from __future__ import annotations

import argparse


# --------------------------------------------------------------------------- labels create --
def _labels_create(args: argparse.Namespace) -> None:
    if args.interactive:
        if args.count is not None:
            raise SystemExit("herculabels create: cannot use --count with --interactive — "
                             "interactive mode grinds windows open-endedly (close the viewer to stop)")
    elif args.coords_file:
        if args.count is not None:
            raise SystemExit("herculabels create: cannot use --count with --coords-file — "
                             "the file defines exactly which windows to generate")
    elif args.count is None:
        raise SystemExit("herculabels create: --count is required (unless --interactive or --coords-file)")
    if args.coords is not None:
        if not args.scroll:
            raise SystemExit("herculabels create: --coords requires an explicit --scroll")
        if not args.interactive and not args.coords_file and args.count != 1:
            raise SystemExit("herculabels create: --coords targets one exact window, so requires --count 1")
    if args.himat is not None and not (0.0 <= args.himat <= 1.0):
        raise SystemExit("herculabels create: --himat must be a fraction in [0, 1]")

    from .create import create

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
    from .edit_cmd import edit_corpus
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
    from .merge import merge_corpora
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
    from .export import export
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
    from . import m7mine
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="herculabels",
        description="HercuLabels — pseudo-label corpus generation + editing (stage 1).",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>", required=True)
    _add_labels_create(sub)
    _add_labels_edit(sub)
    _add_labels_merge(sub)
    _add_labels_export(sub)
    _add_labels_m7mine(sub)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:                                    # clean Ctrl-C — no traceback
        import sys
        print("\nherculabels: interrupted", file=sys.stderr, flush=True)
        raise SystemExit(130)


if __name__ == "__main__":
    main()

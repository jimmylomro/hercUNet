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
    hercunet labels export <corpus.herculabels> <train_out> [--no-augment]

``create`` builds a new corpus (fails if it exists): ``--interactive`` opens the viewer and grinds windows
open-endedly (no ``--count``); otherwise it generates ``--count`` windows headless. ``edit`` re-opens the
corpus in the viewer for correction. ``export`` derives the training corpus (augmentations on by default).
"""

from __future__ import annotations

import argparse


# --------------------------------------------------------------------------- labels create --
def _labels_create(args: argparse.Namespace) -> None:
    if args.interactive:
        if args.count is not None:
            raise SystemExit("hercunet labels create: --interactive grinds windows open-endedly — "
                             "do not pass --count (close the viewer to stop)")
    elif args.count is None:
        raise SystemExit("hercunet labels create: --count is required (unless --interactive)")
    if args.coords is not None:
        if not args.scroll:
            raise SystemExit("hercunet labels create: --coords requires an explicit --scroll")
        if not args.interactive and args.count != 1:
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
        slab_negatives=args.slab_negatives,
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
                   help="(--interactive) grind through the windows listed in this file, one per line as "
                        "'Z,Y,X' (uses --scroll) or 'SCROLL,Z,Y,X'; without it the viewer offers a "
                        "random-or-specify chooser")
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
    p.add_argument("--slab-negatives", action="store_true",
                   help="use the slab-field negatives (submission/md/herculabels.md §5.3, the newer method) "
                        "instead of the default gap-gated negatives (§5.2) that generated the corpus")
    p.add_argument("--seed", type=int, default=0,
                   help="base random seed for the deterministic sampling stream")
    p.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=True,
                   help="use the GPU where available (--no-gpu to force CPU)")
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True,
                   help="bit-reproducible generation (--no-deterministic for fastest)")
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
    _add_labels_export(labels_cmds)

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

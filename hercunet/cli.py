"""hercunet — command-line interface for the HercUNet pipeline.

A single command with subcommands, one per pipeline stage. The CLI only parses and validates
arguments; the actual work lives in the ``hercunet`` package. Conventions: every option is a long
``--flag`` (no single-dash short forms), and flag names use hyphens, never underscores.

Installed as the ``hercunet`` console script (see ``pyproject.toml`` ``[project.scripts]``); also
runnable via ``python -m hercunet``.

Currently implemented:

    hercunet labels create --count <N> --output-dir <DIR> [--scroll <ID>] [--visualise image] [...]

``--visualise`` renders the generated window (``image`` = a cross-section montage of the streamlets
coloured by cluster; ``interactive`` = a live viewer, not yet implemented) and is only valid with
``--count 1``. ``--visualise-output-dir`` sets where the image is written (default: --output-dir).
"""

from __future__ import annotations

import argparse


# --------------------------------------------------------------------------- labels create --
def _labels_create(args: argparse.Namespace) -> None:
    if args.visualise and args.count != 1:
        raise SystemExit("hercunet labels create: --visualise can only be used with --count 1")

    from .labels import create

    create(
        count=args.count,
        output_dir=args.output_dir,
        scroll=args.scroll,
        visualise=args.visualise,
        visualise_output_dir=args.visualise_output_dir,
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
        help="generate sheet-membership pseudo-label windows",
        description="Generate --count pseudo-label windows and write them to --output-dir.",
    )
    p.add_argument("--count", type=int, required=True,
                   help="number of pseudo-label windows to generate")
    p.add_argument("--output-dir", required=True,
                   help="directory to write the .npz pseudo-labels into (created if absent)")
    p.add_argument("--scroll", default=None,
                   help="scroll id to sample from, e.g. PHerc1447 (default: the standard corpus set)")
    p.add_argument("--visualise", choices=["image", "interactive"], default=None, metavar="MODE",
                   help="render the generated window (requires --count 1): 'image' = a cross-section "
                        "montage of the streamlets coloured by cluster; 'interactive' = live viewer "
                        "(not yet implemented)")
    p.add_argument("--visualise-output-dir", default=None,
                   help="directory for the --visualise image (default: --output-dir)")
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
                   help="use the slab-field negatives (docs/pseudo-labels.md §5.3, the newer method) instead "
                        "of the default gap-gated negatives (§5.2) that generated the corpus")
    p.add_argument("--seed", type=int, default=0,
                   help="base random seed for the deterministic sampling stream")
    p.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=True,
                   help="use the GPU where available (--no-gpu to force CPU)")
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True,
                   help="bit-reproducible generation (--no-deterministic for fastest)")
    p.set_defaults(func=_labels_create)


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

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

# Package reorganisation — migration guide (2026-10)

The repository was split so the three products no longer share one flat namespace: the nnU-Net refiner
(**HercUNet**), the Dinovol-based segmenter (**HercuViT**), and the pseudo-label generator + editor
(**HercuLabels**) are each their own package, over a set of shared top-level modules:

```
hercunet/
├── herculabels/  ← HercuLabels: stage-1 label GENERATION + the Qt editing UI (viz/) + CLI
├── hercunet/     ← the nnU-Net refiner: training, recipes, refine, CLI
├── hercuvit/     ← the Dinovol sheet segmenter (unchanged)
├── data/         ┐
├── infer/        │ SHARED (streaming data layer, model-agnostic inference engine,
├── losses/       │ constrained-MALIS + affinity losses, warm-start resolution, config, and the
├── labels/       │ label GEOMETRY primitives: corpus, io, mesh, meshlets, embedding, fields, selection)
├── pretrained.py │
└── config.py     ┘
```

## CLI changes — reinstall, and one BREAKING rename

**Reinstall once** after pulling, to refresh the console scripts (their internal targets moved):

```bash
pip install -e .            # or: pipenv install -e .
```

- `hercunet …` and `hercuvit …` are **unchanged** (`hercunet train fit …`, `hercunet infer …`,
  `hercuvit train chain …`, `hercuvit infer …`). `python -m hercunet …` / `python -m hercunet.hercuvit.cli …` too.
- **BREAKING:** stage-1 label generation moved to its own **`herculabels`** console script. The old
  `hercunet labels <cmd>` subcommands are **removed**; use:

  | Old | New |
  | --- | --- |
  | `hercunet labels create …` | `herculabels create …` |
  | `hercunet labels edit …`   | `herculabels edit …` |
  | `hercunet labels merge …`  | `herculabels merge …` |
  | `hercunet labels export …` | `herculabels export …` |
  | `hercunet labels m7-mine …`| `herculabels m7-mine …` |

## Installation extras (leaner core)

`pip install hercunet` now installs only the **streaming data layer** (numpy/zarr/fsspec/s3fs) — no torch, sklearn,
hdbscan or matplotlib. Install the product you need:

| Extra | Installs |
| --- | --- |
| `.[herculabels]` | label generation + the Qt editing UI (torch, scipy, sklearn, hdbscan, matplotlib, PySide6, …) |
| `.[train]` | the nnU-Net refiner training stack (nnunetv2, acvl-utils, …) |
| `.[infer]` | nnU-Net full-volume inference |
| `.[hercuvit]` | the Dinovol segmenter (torch, timm, einops, …) |
| `.[full]` | everything |

The old standalone `.[viz]` extra is folded into `.[herculabels]` (the UI is pure to HercuLabels).

## If you import the library in Python

Only **import paths** changed. Update any `import`/`from` statements per this table:

| Old import path | New import path |
| --- | --- |
| `hercunet.cli` | `hercunet.hercunet.cli` |
| `hercunet.training.*` (losses, recipe, trainers, prev_transform, …) | `hercunet.hercunet.training.*` |
| `hercunet.train.launch` (trainer launch/`fit`) | `hercunet.hercunet.training.launch` |
| `hercunet.train.preprocess`, `hercunet.train.paths` | `hercunet.hercunet.training.preprocess`, `…training.paths` |
| `hercunet.viz.*` | `hercunet.herculabels.viz.*` |
| `hercunet.recipes` (the nnU-Net `hercunet.yml`) | `hercunet.hercunet.recipes` |
| `hercunet.labels.create` / `edit` / `edit_cmd` / `merge` / `grind` / `m7mine` / `export` / `augment` / `common` / `_brick` / `_generate` / `_window` | `hercunet.herculabels.<same>` |
| `hercunet.train.dataset` | **`hercunet.data.dataset`** (shared) |
| `hercunet.training.malis`, `hercunet.training.affinity` | **`hercunet.losses.malis`**, **`hercunet.losses.affinity`** (shared) |
| `hercunet.train.launch.resolve_pretrained` | **`hercunet.pretrained.resolve_pretrained`** (shared) |

**Unchanged (shared, already at the top level):** `hercunet.data.*`, `hercunet.infer.*`, `hercunet.config`, and
the label geometry primitives `hercunet.labels.{corpus, io, mesh, meshlets, embedding, fields, selection}`. The
entire `hercunet.hercuvit.*` tree is unchanged.

## Why

- `train/` and `training/` were two halves of the same nnU-Net training and are now one package
  (`hercunet.hercunet.training`).
- Code genuinely shared with HercuViT (the streaming data layer, the inference engine, MALIS/affinity losses,
  warm-start resolution, and the label *geometry* primitives the training dataset builds on) stays at the top
  level as its own modules, so neither model reaches into the other's package.
- `hercunet.labels` keeps only the shared geometry primitives; the nnU-Net stage-1 *generation* entry points moved
  to `hercunet.herculabels`.

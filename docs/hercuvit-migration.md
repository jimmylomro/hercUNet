# Package reorganisation — migration guide (2026-10)

The repository was split into three clear layers so the nnU-Net refiner (**HercUNet**) and the Dinovol-based
segmenter (**HercuViT**) no longer share one flat namespace:

```
hercunet/
├── hercunet/     ← the nnU-Net refiner: stage-1 label GENERATION, training, viz, recipes, CLI
├── hercuvit/     ← the Dinovol sheet segmenter (unchanged)
├── data/         ┐
├── infer/        │ SHARED between both (streaming data layer, model-agnostic inference engine,
├── losses/       │ constrained-MALIS + affinity losses, warm-start resolution, config, and the
├── labels/       │ label GEOMETRY primitives: corpus, io, mesh, meshlets, embedding, fields, selection)
├── pretrained.py │
└── config.py     ┘
```

## The CLIs are unchanged

The console commands are **exactly the same** — `hercunet …` and `hercuvit …` still work verbatim
(`hercunet train fit …`, `hercunet labels create …`, `hercunet infer …`, `hercuvit train chain …`,
`hercuvit infer …`). Only the entry points' internal targets moved, so after pulling this change **reinstall the
package once** to refresh the console scripts:

```bash
pip install -e .            # or: pipenv install -e .
```

`python -m hercunet …` and `python -m hercunet.hercuvit.cli …` also work unchanged.

## If you import the library in Python

Only **import paths** changed. Update any `import`/`from` statements per this table:

| Old import path | New import path |
| --- | --- |
| `hercunet.cli` | `hercunet.hercunet.cli` |
| `hercunet.training.*` (losses, recipe, trainers, prev_transform, …) | `hercunet.hercunet.training.*` |
| `hercunet.train.launch` (trainer launch/`fit`) | `hercunet.hercunet.training.launch` |
| `hercunet.train.preprocess`, `hercunet.train.paths` | `hercunet.hercunet.training.preprocess`, `…training.paths` |
| `hercunet.viz.*` | `hercunet.hercunet.viz.*` |
| `hercunet.recipes` (the nnU-Net `hercunet.yml`) | `hercunet.hercunet.recipes` |
| `hercunet.labels.create` / `edit` / `edit_cmd` / `merge` / `grind` / `m7mine` / `export` / `augment` / `common` / `_brick` / `_generate` / `_window` | `hercunet.hercunet.labels.<same>` |
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
  to `hercunet.hercunet.labels`.

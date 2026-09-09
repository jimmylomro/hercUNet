# HercUNet documentation

Index of the per-part guides. Each stage of the project gets its own document here; they are written
as the corresponding code is ported in. See the [project README](../README.md) for the overview.

## Guides

| Document | Covers | Status |
|---|---|---|
| **[data-layer.md](data-layer.md)** | The streaming data layer — backends, `ZarrSegment`, chunk-aligned parallel reads, the L2 disk cache, prefetch (`iter_windows` / `VolumePrefetcher`), material tiling, configuration, and performance notes. | ✅ available |
| **[pseudo-labels.md](pseudo-labels.md)** | Stage 1 — pseudo-label generation: meshlets → pos/neg selection (corridor → gap-gated → slab-field gate) → 8-D triplet embedding → **probeom** → mesh/∇φ/`owner_full` → confidence/quality. | 📄 methodology (pre-port) |
| _refine.md_ | Stage 2 — the iterative HercUNet refiner (8-ch affinity/MALIS trainer, N-pass wrapper). | 🚧 planned |
| _infer.md_ | Stage 3 — single-instance full-volume inference (tile+halo, OME-Zarr out). | 🚧 planned |

## Reading order

1. **[data-layer.md](data-layer.md)** — everything else reads voxels through it, so start here.
2. _labels.md_, _refine.md_, _infer.md_ — added as each stage lands.

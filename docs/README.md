# HercUNet documentation

Index of the per-part guides. `docs/` holds **code-usage** guides (how to run things); the research
**methodology** write-ups live under [`../submission/md/`](../submission/md/). See the
[project README](../README.md) for the overview.

## Guides

| Document | Covers | Status |
|---|---|---|
| **[data-layer.md](data-layer.md)** | The streaming data layer — backends, `ZarrSegment`, chunk-aligned parallel reads, the L2 disk cache, prefetch (`iter_windows` / `VolumePrefetcher`), material tiling, configuration, and performance notes. | ✅ available |
| **[herculabels.md](herculabels.md)** | **Stage 1 usage** — the `hercunet labels create` / `edit` / `export` CLI over `.herculabels` corpora, the interactive **grinding viewer** (navigation, overlays, split/merge/delete, undo, save + next), and the corpus + export outputs (incl. the `meta.json` manifest). | ✅ available |
| _refine.md_ | Stage 2 — the iterative HercUNet refiner (8-ch affinity/MALIS trainer, N-pass wrapper). | 🚧 planned |
| _infer.md_ | Stage 3 — single-instance full-volume inference (tile+halo, OME-Zarr out). | 🚧 planned |

The **methodology** for stage 1 (how the labels are derived — meshlets, slab, contrastive embedding,
probeom, medial fit, confidence) is the research write-up
[`../submission/md/herculabels.md`](../submission/md/herculabels.md).

## Reading order

1. **[data-layer.md](data-layer.md)** — everything else reads voxels through it, so start here.
2. **[herculabels.md](herculabels.md)** — running stage 1 and the viewer; the methodology behind it is
   in [`../submission/md/herculabels.md`](../submission/md/herculabels.md).
3. _refine.md_, _infer.md_ — added as each stage lands.

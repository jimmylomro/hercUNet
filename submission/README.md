# submission/

Source material for the HercUNet submission write-up (the LaTeX PDF is generated from the Markdown here).

## Methodology write-ups — [`md/`](md/)

| Document | Covers |
|---|---|
| **[md/herculabels.md](md/herculabels.md)** | **Stage 1 — HercuLabels.** How the pseudo-labels are made: cleaned substrate + frame field, 2.5-D meshlets, positive slab + gap-gated negatives, the 8-D contrastive embedding, probabilistic excess-of-mass clustering, the medial-mesh fit, per-voxel confidence, and the merge/split augmentations. |

## Figures

- **[`images/`](images/)** — the JPEGs referenced by the `md/` documents (the ones that go into the PDF).
- **[`all_images/`](all_images/)** — the full-resolution source renders (PNG); the `images/` JPEGs are
  compressed copies of the ones actually used.

## See also

- **[../docs/](../docs/README.md)** — how to *run* the code (CLI, viewer, data layer), separate from the
  methodology here.

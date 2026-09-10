# submission/

Source material for the HercUNet submission write-up (the LaTeX PDF is generated from the Markdown here).

## Methodology write-ups — [`writeup/`](writeup/)

| Document | Covers |
|---|---|
| **[writeup/herculabels.md](writeup/herculabels.md)** | **Stage 1 — HercuLabels.** How the pseudo-labels are made: cleaned substrate + frame field, 2.5-D meshlets, positive slab + gap-gated negatives, the 8-D contrastive embedding, probabilistic excess-of-mass clustering, the medial-mesh fit, per-voxel confidence, the merge/split augmentations, and reproducibility. |
| **[writeup/corpus.md](writeup/corpus.md)** | **Corpus provenance** — the published stage-1 corpus (4031 windows across 22 scrolls): the exact extraction parameters, the reproduction command, and the full per-window index (scroll, coordinates, sheet count, quality). The window list is also provided as [`writeup/corpus_windows.txt`](writeup/corpus_windows.txt), a coordinate file `hercunet labels create --coords-file` consumes to regenerate the exact corpus. |

## Figures

- **[`images/`](images/)** — the JPEGs referenced by the `writeup/` documents (the ones that go into the PDF).
- **[`all_images/`](all_images/)** — the full-resolution source renders (PNG); the `images/` JPEGs are
  compressed copies of the ones actually used.

## See also

- **[../docs/](../docs/README.md)** — how to *run* the code (CLI, viewer, data layer), separate from the
  methodology here.

# submission/

Source material for the HercUNet submission write-up. The Markdown methodology documents in [`writeup/`](writeup/)
are the source of truth; [`writeup/paper.tex`](writeup/paper.tex) folds both stages into one self-contained LaTeX
paper for the PDF submission.

## Methodology write-ups — [`writeup/`](writeup/)

| Document | Covers |
|---|---|
| **[writeup/herculabels.md](writeup/herculabels.md)** | **Stage 1 — HercuLabels.** How the pseudo-labels are made: cleaned substrate + frame field, 2.5-D meshlets, positive slab + gap-gated negatives, the 8-D contrastive embedding, probabilistic excess-of-mass clustering, the medial-mesh fit, per-voxel confidence, the merge/split augmentations, and reproducibility. |
| **[writeup/hercunet.md](writeup/hercunet.md)** | **Stage 2 — HercUNet.** The learned iterative refiner: `D(CT, prev) → surface` as an iterated map, the carried CT-orientation field, the affinity head + constrained-MALIS instance separation, the loss stack (symmetric Focal–Tversky, separation, skeleton-recall, MALIS, CT-material growth), taught iteration (octant composition, fragmentation, online DAgger), the m7 rehearsal blend, the corpus-cleanup negative finding, seam-free iterative inference, and why we report no Dice (faces vs medials). |
| **[writeup/corpus.md](writeup/corpus.md)** | **Corpus provenance** — the published stage-1 corpus (4031 windows across 22 scrolls): the exact extraction parameters, the reproduction command, and the full per-window index (scroll, coordinates, sheet count, quality). The window lists themselves live in [`corpus/`](corpus/README.md). |

## Reproduction inputs — [`corpus/`](corpus/README.md)

The window lists that regenerate the training data from scratch: [`corpus/corpus_windows.txt`](corpus/corpus_windows.txt)
(the stage-1 ∇φ corpus, for `hercunet labels create --coords-file`), and the mined m7 rehearsal corpus as both a
human index [`corpus/m7_windows.txt`](corpus/m7_windows.txt) and the byte-exact rebuild manifest
[`corpus/m7_scout_himat_manifest.json`](corpus/m7_scout_himat_manifest.json) (for `hercunet labels m7-mine --manifest`). See
[`corpus/README.md`](corpus/README.md) for the exact commands.

## Figures

- **[`images/`](images/)** — the JPEGs referenced by the `writeup/` documents (the ones that go into the PDF).
- **[`all_images/`](all_images/)** — the full-resolution source renders (PNG); the `images/` JPEGs are
  compressed copies of the ones actually used.

## See also

- **[../docs/](../docs/README.md)** — how to *run* the code (CLI, viewer, data layer), separate from the
  methodology here.

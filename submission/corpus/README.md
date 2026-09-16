# submission/corpus/ — reproduction inputs

The exact window lists that regenerate the two published pseudo-label corpora, so anyone can rebuild the
training data from scratch. The methodology behind them is in [`../writeup/corpus.md`](../writeup/corpus.md)
(stage-1) and [`../writeup/hercunet.md`](../writeup/hercunet.md) (the m7 rehearsal blend).

| File | Corpus | Consumed by |
|---|---|---|
| [`corpus_windows.txt`](corpus_windows.txt) | **Stage-1 ∇φ pseudo-labels** — 4031 windows across 22 scrolls (`run20260808`), one `SCROLL,Z,Y,X` per line. | `hercunet labels create --coords-file` |
| [`m7_windows.txt`](m7_windows.txt) | **m7 rehearsal** — 100 coherent m7 windows scouted from compressed regions across 20 scrolls, one `SCROLL,Z,Y,X,SIZE`. | (human index) |
| [`m7_scout_himat_manifest.json`](m7_scout_himat_manifest.json) | **m7 rehearsal** — the full per-region record (m7/CT zarr paths, tau, coherence) for a byte-exact rebuild. | `hercunet labels m7-mine --manifest` |

## Stage-1 corpus (our ∇φ pseudo-labels)

`corpus_windows.txt` is the coordinate file `create` consumes to regenerate the exact windows. **Note
`--old-negatives`** — this corpus used the old gap-gated negatives; the current `create` default is the newer
slab-field negatives (see [`../writeup/corpus.md`](../writeup/corpus.md) §5.3), which gives an *equivalent* but
not identical corpus.

```bash
hercunet labels create scroll_corpus.herculabels --old-negatives --coords-file corpus_windows.txt
hercunet labels export  scroll_corpus.herculabels ./train_out
```

## m7 rehearsal corpus (mined from the public m7 detector)

The m7 corpus is coherent m7 surface predictions mined from **compressed** regions of the *published* m7
detector (`surf = m7 & (m7_normal_coherence > 0.88)`; incoherent m7 → ignore; `~m7` → background). MALIS is
off on m7 (no per-sheet membership). The scout is **deterministic** — no seeds; it is a pure function of the
public S3 zarrs + the fixed constants + the coherence judge — so the manifest reproduces exactly.

```bash
# byte-exact rebuild of the approved windows (reads the published m7 + CT from open-data S3):
hercunet labels m7-mine ./m7_corpus --manifest m7_scout_himat_manifest.json --max-candidates 4

# …or re-scout from scratch (same recipe; --images also writes the 3-panel QC renders):
hercunet labels m7-mine ./m7_corpus --scroll all --images --max-candidates 4
```

`--max-candidates` (K) must match the value used for `train export-labels` / `train preprocess`. The training
chain then folds the m7 corpus in with `hercunet train export-labels --m7-corpus ./m7_corpus …` (a copy, no
symlinks). See [`../../docs/training.md`](../../docs/training.md) and
[`../../docs/herculabels.md`](../../docs/herculabels.md).

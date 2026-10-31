# HercuViT — cross-window sheet stitcher (`hercuvit labels create` · `hercuvit stitcher …`)

`hercuvit infer` segments a volume **window by window**, so a sheet that spans several windows comes out as several
disjoint meshes with unrelated per-window ids (see [hercuvit.md](hercuvit.md) §"disjoint windows"). The **stitcher** is
a small learned decoder that reconnects them: it reads the stored HercuViT embeddings, predicts a signed **affinity**
between every pair of sheets that meet at a window corner, and writes those affinities into the `.asv` as an edge
table — so one file carries both the sheet meshes **and** the graph that links them across windows. A graph
agglomeration over that table (batch, or interactively in VC3D) then assembles whole sheets.

The research methodology — the primitive-agnostic method, the network, label generation, and the agglomeration — is
the write-up [`../submission/writeup/hercuvit-stitcher.md`](../submission/writeup/hercuvit-stitcher.md). This page is
how to **run** it.

> **Needs the `hercuvit` extra and (for extraction/training/inference) a CUDA GPU.** Install with
> `pip install ".[hercuvit]"` (or `pipenv install ".[hercuvit]"`); see [README.md](README.md) for GPU/CUDA.

---

## What it does (in one paragraph)

The stitcher trains and runs on **stored HercuViT artifacts**, never on a second CT pass. A
`hercuvit infer --store-embeddings` run emits two things per volume: a **pooled per-window embedding zarr**
(`{out}_emb.zarr`, each window's per-window normalised-PCA tokens) and, in each sheet's `.asv` record, that sheet's
**median embedding** (the per-sheet descriptor). Training samples come from `hercuvit labels create`, which at each
window corner runs HercuViT on the 8 octant windows **plus a centred pass** and overlap-votes a same-object label
matrix — fully **self-supervised, no ground truth**. `hercuvit stitcher train` fits the decoder on those samples, and
`hercuvit stitcher infer` applies it to the stored artifacts and writes the `affinities` table into the `.asv`. VC3D
then reads that table to colour sheets by affinity and to auto-stitch from a selected sheet.

The pipeline, end to end:

```
hercuvit infer --store-embeddings   →  {out}_meshes.asv  +  {out}_emb.zarr     (stored artifacts)
hercuvit labels create              →  stitch_*.npz                            (self-supervised training samples)
hercuvit stitcher train             →  checkpoint_{best,final,latest}.pth      (the trained decoder)
hercuvit stitcher infer             →  affinities table written into the .asv  (sheets + cross-window graph)
                                        ↳ open the .asv in VC3D: affinity colouring + auto-stitch
```

**Already have a trained stitcher? Do it all in one command.** `hercuvit infer --stitcher <hf|file>://PATH` folds
steps 1 + 4 together: it stores the embeddings **and** runs the stitcher in the **leader** process once the per-window
sheets are merged, writing a single **`{out}_stitched-meshes.asv`** that carries both the sheets and the affinities.
The stitcher defaults to `hf://jimmylomro/hercuvit-stitcher`, so a bare `--stitcher` pulls the published model:

```bash
pipenv run hercuvit infer single-instance PHerc1447 --model-hf jimmylomro/hercuvit-v0 \
  --region 11618:11810,3197:3389,3552:3744 --mesh local --overlap 0.1 \
  --stitcher \
  --out ./pherc1447_demo
# bare --stitcher = hf://jimmylomro/hercuvit-stitcher  (or pass --stitcher file://./stitch-run)
# -> ./pherc1447_demo_stitched-meshes.asv   (sheets + affinities, ready for VC3D)
```

Stitching is single-leader (the decoder is tiny and corner-batched, so a whole volume stitches in seconds — not worth
sharding across GPUs). The sections below cover the separate commands, needed when you are **training** a stitcher or
want the artifacts as distinct files.

---

## 1. Stored artifacts — `hercuvit infer --store-embeddings`

The stitcher consumes a normal `hercuvit infer` run that was asked to keep its embeddings. Add
`--store-embeddings` (and `--emb-dim D`, which **must** match the stitcher's `--pca-dim`, default `3`) to any infer
command that writes the sheet `.asv`:

```bash
pipenv run hercuvit infer single-instance PHerc1447 \
  --model-hf jimmylomro/hercuvit-v0 \
  --region 11618:11810,3197:3389,3552:3744 \
  --mesh local --overlap 0.1 \
  --store-embeddings --emb-dim 3 \
  --out ./pherc1447_demo
# -> ./pherc1447_demo_meshes.asv   (sheet meshes, each with its median embedding)
# -> ./pherc1447_demo_emb.zarr     (pooled per-window embeddings: emb (N,G,G,G,D) + origin (N,3))
```

- Use an **overlap** (`--overlap 0.1` is the stitcher's assumption) so window cores tile and corners are well formed.
- `_emb.zarr` holds one pooled normalised-PCA grid per window plus its global-L0 origin; it is small (loads in one go
  at stitch time).
- Each sheet's median descriptor is stored in its `.asv` record, so a sheet that has no valid `D`-dim embedding is
  simply skipped as a query at stitch time.
- `--store-embeddings` is tri-state like `--surf`/`--mesh`: bare = **local**, `--store-embeddings upload` also ships
  `{out}_emb.zarr` to `--s3-prefix` (as does any `--mesh upload` run with embeddings stored). It is only needed for the
  deferred/repeat two-step workflow below — `--stitcher` implies it and stitches in one go (see the top of this page).

---

## 2. Training samples — `hercuvit labels create`

Self-supervised stitching samples, one `.npz` per corner, with **no ground truth** (annotation coordinates, if used,
only supply corner *locations*). At each corner it runs HercuViT on the 8 octant windows + a centred pass and
overlap-votes a same-object label matrix.

```bash
# a few hundred random material corners on PHerc1447
pipenv run hercuvit labels create ./stitch-samples \
  --model-hf jimmylomro/hercuvit-v0 \
  --scroll PHerc1447 --count 500 --pca-dim 3

# or exactly the corpus windows (reuses the herculabel corpus coordinates as corners)
pipenv run hercuvit labels create ./stitch-samples \
  --model-hf jimmylomro/hercuvit-v0 \
  --coords-file ../submission/corpus/corpus_windows.txt --pca-dim 3
```

| Flag | Default | Meaning |
|---|---|---|
| `OUT_DIR` (positional) | (required) | where to write the per-corner `stitch_*.npz` |
| `--count N` | — | number of **random** material corners (required unless `--coords-file`/`--coords`) |
| `--scroll ID` | — | scroll id(s), e.g. `PHerc1447` or `5,PHerc1447` |
| `--coords-file FILE` | — | extract exactly the listed corners (`Z,Y,X` with `--scroll`, or `SCROLL,Z,Y,X`) |
| `--coords Z,Y,X` | — | one exact corner (needs `--scroll` and `--count 1`) |
| `--pca-dim D` | `3` | per-window normalised-PCA dimension (must match `stitcher train --pca-dim`) |
| `--overlap f` | `0.1` | window overlap fraction |
| `--min-vote N` | `25` | min overlap voxels to label a fragment |
| `--min-pts N` | `20` | min 1/8 surface cells for a fragment to become a query |
| `--himat FRAC` | light air-skip | `--count` only: keep corners with at least this material fill |
| `--infer-batch N` | `1` | HercuViT windows per forward (raise to ~9 on a big GPU for the batched speedup) |
| `--stream-workers N` | `8` | concurrent CT reads per corner (hides streaming latency) |
| `--nshards` / `--shard` | `1` / `0` | split work across processes (one per GPU) |
| `--regions` | off | REGION mode: download a few material regions per volume and extract non-overlapping corners from RAM; `--exclude-dir` avoids a prior corpus's corners |

For a large corpus, prefer `--regions` (amortises the download) and shard one process per GPU. `--exclude-dir` lets a
second extraction add **fresh, non-overlapping** corners to an existing corpus.

---

## 3. Train the decoder — `hercuvit stitcher train`

Fit the ~1 M-param decoder on the samples. The resolved config is saved in the checkpoint, so inference rebuilds the
exact model.

```bash
pipenv run hercuvit stitcher train ./stitch-run \
  --samples ./stitch-samples \
  --pca-dim 3 --epochs 100 --label-mode v2
# -> ./stitch-run/checkpoint_{best,final,latest}.pth
```

| Flag | Default | Meaning |
|---|---|---|
| `OUT_DIR` (positional) | (required) | checkpoint output dir (`checkpoint_{best,final,latest}.pth`) |
| `--samples DIR` | — | local dir of `stitch_*.npz` (from `labels create`) |
| `--samples-hf REPO` | — | download the samples from an HF dataset repo instead |
| `--pca-dim D` | `3` | stored embedding dim — **must** match the samples |
| `--d-model` / `--heads` / `--layers` / `--fourier-bands` | `128` / `4` / `4` / `8` | decoder architecture |
| `--epochs N` | `100` | total epochs |
| `--batch-size N` | `16` | corners per batch |
| `--base-lr LR` | `1e-4` | base LR (linear warmup → cosine) |
| `--label-mode {raw,v1,v2}` | **`v2`** | load-time label-quality fix on **train** (val stays raw so edge-AP stays comparable): `raw` = baseline (~25% of pairs labelled); `v1` = same-window pairs → negative + drop off-seam queries (100% labelled, seam-scoped); **`v2`** = same-window pairs → negative + off-seam pairs → hard negatives (100% labelled) |
| `--val-fold N` / `--val-interval N` | `0` / `5` | `KFold(5, seed 12345)` held-out fold; validate edge AP every N epochs (`0` = train on all) |
| `--pos-weight f` | auto | focal-BCE positive weight (`0` = auto neg/pos ratio per batch) |
| `--continue` | off | resume from `OUT_DIR/checkpoint_latest.pth` |
| `--device` | `cuda` | `cuda` \| `cuda:N` \| `cpu` |

The loss is a deep-supervised **focal BCE + supervised-contrastive** over the decoder layers (see the write-up §6).
`--label-mode v2` is the default because the raw pseudo-label matrix leaves ~75% of pairs ignored; `v2`/`v1` turn the
certain same-window and off-seam pairs into negatives, taking the matrix to 100% labelled.

---

## 4. Write the affinities — `hercuvit stitcher infer`

Apply a trained stitcher to a volume's stored artifacts and write the `affinities` edge table into the `.asv`:

```bash
pipenv run hercuvit stitcher infer \
  --embeddings-zarr ./pherc1447_demo_emb.zarr \
  --asv ./pherc1447_demo_meshes.asv \
  --stitcher ./stitch-run \
  --top-k 0
# edits the .asv in place: adds `affinities(src,dst,logit,prob,corner)` (+ an index on src) and `affinity_corners`
```

| Flag | Default | Meaning |
|---|---|---|
| `--embeddings-zarr ZARR` | (required) | the `{out}_emb.zarr` from `infer --store-embeddings` |
| `--asv ASV` | (required) | the sheet `.asv` (with per-sheet median embeddings); affinities written here unless `--out` |
| `--stitcher <hf\|file>://PATH` | `hf://jimmylomro/hercuvit-stitcher` | trained stitcher, resolved like every other inference model: `hf://<repo>` \| `file://<path>` \| a local `.pth`/dir \| a bare HF repo id |
| `--out ASV` | — | write the augmented `.asv` here instead of editing `--asv` in place (sheets copied, then the table added) |
| `--top-k K` | `0` | per sheet, keep only its K highest-affinity neighbours per corner (`0` = all in-range pairs; `10` is sensible for a large graph) |
| `--batch N` | `16` | corners per decoder forward |
| `--device` | `cuda` | `cuda` \| `cuda:N` \| `cpu` |

**Both directions** `(A→B)` and `(B→A)` are stored (the head is symmetric), so "every neighbour of sheet X" is a
single indexed lookup `WHERE src = X`. The pair's `corner` column records which window corner produced it (a pair can
recur at several corners; aggregation to a mean logit is the agglomeration step, done by the consumer). Use
`--stitcher file://PATH` or an HF repo id; a dir resolves `checkpoint_best.pth` → `final` → `latest`.

The stitcher's `--pca-dim` must match the zarr's emb dim and the sheets' median-embedding dim — i.e. the same `D` you
passed to `infer --store-embeddings --emb-dim` and to `labels create --pca-dim`. A mismatch is refused with a clear
message.

---

## 5. Use it — VC3D affinity colouring & auto-stitch

An `.asv` that carries the `affinities` table unlocks two things in the VC3D sheet-collection viewer (the
`vc3d-sheet-collection` branch), **in stitch mode only**:

- **Affinity colouring** — pick the latest selected sheet and colour every other sheet by its softmaxed affinity to
  it, blue (0) → red (1). The option is disabled unless the `.asv` has the table and a stitch is active.
- **Auto-stitch** — from the latest selected sheet, agglomerate the affinity graph over the `(2r±1)³` block of
  windows around it (GASP average-linkage over the signed logits) and add the sheet's resulting cluster to the
  stitch. The `±N win` radius controls the neighbourhood.

Both read the same edge table through the single-`src`-lookup above. The batch equivalent — solving the whole volume
graph into a fragment→object lookup — is the agglomeration in the write-up (§8).

---

## Reading order

1. **[hercuvit.md](hercuvit.md)** — segment a volume into per-window sheet meshes (`infer`), and the `.asv` store.
2. **This page** — store embeddings, extract samples, train the stitcher, write affinities.
3. **[`../submission/writeup/hercuvit-stitcher.md`](../submission/writeup/hercuvit-stitcher.md)** — the method
   (primitive-agnostic: it stitches any instances with an embedding field, not just sheets).

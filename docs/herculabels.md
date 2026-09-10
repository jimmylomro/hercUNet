# HercuLabels — stage 1 usage guide

**HercuLabels** is the pseudo-label generator (stage 1 of the pipeline): from raw micro-CT it produces,
per window, an unsupervised **sheet-instance segmentation** plus honest per-voxel confidence, collected
into an editable **`.herculabels` corpus** that you can grind, correct, combine, and then **export** to the
training format for the downstream refiner. This document covers **how to run it** — the four CLI verbs
(`create` / `edit` / `merge` / `export`), the interactive viewer, the corpus + export outputs, and the
container layout.

For **how it works** (the method — meshlets, slab, contrastive embedding, probeom clustering,
medial-mesh fit, confidence), see the methodology write-up: [`../submission/writeup/herculabels.md`](../submission/writeup/herculabels.md).

---

## Requirements

- **A CUDA GPU.** Generation is impractically slow on CPU, so it is refused by default (`--no-gpu`
  exists only for CPU testing).
- **Python env via pipenv.** Always run through the project env:

  ```bash
  cd ~/Desktop/hercunet
  pipenv install                 # core deps (torch, scipy, scikit-learn, hdbscan, matplotlib, …)
  pipenv install -e ".[viz]"     # + the interactive viewer (PySide6, pyqtgraph, PyOpenGL)
  ```

  Label generation needs only the core deps; **the interactive viewer (`--interactive` and `edit`) needs
  the `[viz]` extra**. (`.[full]` additionally installs `obstore`, an optional native read backend — not
  needed for the viewer.)

- **Data access.** Voxels are read through the streaming data layer (see
  [data-layer.md](data-layer.md)); configure the backend via `HERCUNET_*` env vars / `Config`.

Run `pipenv run hercunet labels <create|edit|export> --help` for the raw flag list of each verb.

---

## Quick start

```bash
# 1) Grind a new corpus interactively (edit + save each window; close the viewer to stop)
pipenv run hercunet labels create demo.herculabels --interactive --scroll PHerc1447

# 2) …or build one headless: 50 random material windows from the standard corpus
pipenv run hercunet labels create demo.herculabels --count 50

# 3) Re-open the corpus later to keep correcting windows
pipenv run hercunet labels edit demo.herculabels

# 4) Extend a corpus: grind a fresh one, then merge into a combined corpus
pipenv run hercunet labels merge combined.herculabels demo.herculabels more.herculabels

# 5) Export the training corpus (base + augmentations) for the refiner
pipenv run hercunet labels export combined.herculabels ./train_out
```

---

## Commands

Three verbs operate over a `.herculabels` corpus (a directory; always editable — there is no "finalise"):

| Verb | Invocation | What it does |
|---|---|---|
| **create** | `create <corpus.herculabels> …` | Build a **new** corpus (fails if it exists). `--interactive` grinds windows in the viewer (open-ended, no `--count`); headless requires `--count N`. |
| **edit** | `edit <corpus.herculabels>` | Re-open an existing corpus in the viewer and correct its windows in place (implicitly interactive; no pipeline re-run). |
| **merge** | `merge <out.herculabels> <in…>` | Union several corpora into a **new** one (fails if `out` exists; inputs untouched). This is how you extend a corpus — grind a fresh one and merge. |
| **export** | `export <corpus.herculabels> <train_out> [--no-augment]` | Derive the training corpus (per-window `.npz`, **no meshlets**). Augmentations on by default; `--no-augment` for a base-only export. Non-destructive + re-runnable. |

### create — headless

```bash
# Random windows from the whole corpus (scroll auto-picked in the voxel band)
pipenv run hercunet labels create demo.herculabels --count 100

# Random windows from one scroll
pipenv run hercunet labels create demo.herculabels --count 100 --scroll PHerc1447

# One exact window
pipenv run hercunet labels create demo.herculabels --count 1 --scroll PHerc1447 --coords 11714,3293,3648

# An exact SET of windows from a coordinate file (regenerate a known corpus) — no --count
pipenv run hercunet labels create demo.herculabels --coords-file windows.txt
```

Without `--coords`, windows are drawn randomly and windows that are mostly air are skipped. `--coords Z,Y,X`
targets one exact window (needs `--scroll`, and `--count 1` in headless mode). **`--coords-file FILE`**
generates exactly the windows listed in `FILE` (one per line, `Z,Y,X` with `--scroll` or `SCROLL,Z,Y,X`;
`#`/blank lines ignored) — the file defines the set, so no `--count`. This is how a published corpus is
regenerated from its window list (see the corpus provenance record, `submission/writeup/corpus.md`). Each window
writes its base sheet meshes + meshlet cloud into the corpus and appends a manifest entry.

### create — interactive (the grinding viewer)

```bash
# Open-ended grind of random windows from a scroll
pipenv run hercunet labels create demo.herculabels --interactive --scroll PHerc1447

# Grind only HIGH-material windows (≥ 55% fill) — the overlay reads "searching for a window with 55% material fill…"
pipenv run hercunet labels create demo.herculabels --interactive --scroll PHerc1447 --himat 0.55

# Grind an ordered list of specific windows
pipenv run hercunet labels create demo.herculabels --interactive --coords-file windows.txt

# Start at one exact window (then Next offers random / specify)
pipenv run hercunet labels create demo.herculabels --interactive --scroll PHerc1447 --coords 11714,3293,3648
```

- **No `--count`** — grinding is open-ended; you advance window-to-window and **close the viewer to
  stop**. Every window you visit is written to the corpus (its base labels on load, your corrections on
  save+next).
- **`--coords-file FILE`** — grind the windows listed in `FILE`, one per line, in order. Each line is
  `Z,Y,X` (uses `--scroll`) or `SCROLL,Z,Y,X`. `#` comments and blank lines are ignored. Example:

  ```
  # windows.txt
  PHerc1447,11714,3293,3648
  11714,3400,3700          # uses --scroll
  PHerc0800,9000,4000,4000
  ```

### edit — correct an existing corpus

```bash
pipenv run hercunet labels edit demo.herculabels
```

Opens the corpus's windows in the same viewer, one after another (list-mode Next), loading each from disk
(no pipeline re-run). The CT slice-pane background is re-read from the scroll on open; if the scroll isn't
available it falls back to a blank background and points-only editing still works. Corrections are written
back in place and the window is flagged `edited` in the manifest.

### merge — combine corpora

```bash
# Union two (or more) corpora into a new combined one
pipenv run hercunet labels merge combined.herculabels session1.herculabels session2.herculabels
```

`merge` builds a **new** output corpus (fails if it exists) holding the union of the inputs' windows; the
inputs are never modified. It's the way to **extend** a corpus — rather than re-opening one to add windows,
grind a fresh corpus and merge them — so `create`/`edit` stay simple.

- **Duplicate windows** (the same scroll + coords ground in more than one input) are resolved automatically:
  the **`edited` copy wins** over an unedited one, otherwise the later input wins (both logged). So a
  hand-corrected window is never clobbered by a stale duplicate — grind-then-merge is safe to repeat.
- Pure file + manifest operation (no GPU / data layer): windows are byte-copied. The merged `meta.json`
  records each source under `create.merged_from` for provenance.

### export — derive the training corpus

```bash
# Base + augmentations (the training set the refiner reads)
pipenv run hercunet labels export demo.herculabels ./train_out

# Base only (e.g. a validation set)
pipenv run hercunet labels export demo.herculabels ./val_out --no-augment
```

Export never modifies the corpus, so you can edit and re-export freely.

Both the viewer and `export` run **the same pipeline / augmentation code** as headless generation (the
viewer via an observer hook), so any change to the label code is reflected everywhere.

---

## Using the interactive viewer

The window is a 2×2 grid — three orthogonal cross-section panes (**z**, **y**, **x**) and a **3-D view**
(sheet meshes, or the embedding space) — over a log pane, a sheet read-out, and the sheet-edit controls.

### Navigating a cross-section pane

| Action | Control |
|---|---|
| Scroll through slices | mouse **wheel** |
| Zoom (about the view centre) | **Ctrl + wheel**, or **right-drag** |
| Pan | **left-drag** (bounded to the image) |
| Slices per wheel notch | the **scroll step** selector (1/2/3/5/10/20) |

Hovering a pane shows the **global `z y x` coords** (status bar) and, over a streamlet, its **sheet id**
(the big colour swatch, bottom right). The 3-D view orbits with drag and reports the sheet under the
cursor.

### The overlay selector (streamlets · none · confidence · samples)

A switch in the top controls bar chooses what the 2-D panes overlay on the CT:

- **streamlets** — the streamlet point cloud, coloured per sheet (grey before clustering finishes).
- **none** — CT only.
- **confidence** — a translucent **red low-confidence field**: the pipeline's combined uncertainty
  (intersection ⊕ sharp2 ⊕ dropped) plus any regions you deleted (see below), graded by uncertainty.
- **samples** — the **contrastive sampling** behind the embedding: **hover a streamlet point** and its
  training **positives light green** (the mutual surface slab) and its **negatives light red** (the
  gap-gated negatives), with the hovered anchor in white — the points near each pane's current slice.
  This lets you *see* whether the slab picks positives across a real between-sheet gap (a merge cause) or
  negatives on the same sheet. **Live only:** the sample tables exist only while a window is being
  generated (`create --interactive`); windows opened from a stored corpus via `edit` have no sampling to
  show, so the option is inert there.

The **3D section planes** toggle draws faint red planes in the 3-D view at each pane's current slice.

### The 3-D view (sheets · embeddings)

The bottom-right pane orbits the window in 3-D — **drag** to rotate, **wheel** to zoom, hover to read the
sheet under the cursor. A toggle in its black bottom band switches what it shows:

- **sheets** — the fitted sheet meshes, brightly shaded (two-sided lighting), one colour per sheet.
- **embeddings** — the **3-D PCA of the contrastive embedding**: one point per streamlet unit, coloured
  by its sheet. The same clusters as the sheets, seen in embedding space — tight cores per sheet with the
  adjacent wraps laid out as an ordered ladder.

Selection behaves the same in both: clicking a sheet/cluster selects it and dims the rest (here and in
the 2-D panes). Edits recolour both views live and **keep the current camera** so the change is easy to
see. The **3D section planes** apply to the sheets view only.

### Selecting sheets

- **Click** a sheet → selects it (others dim); the swatch shows it.
- **Click another sheet** → moves the selection; **click an air gap** → clears it.
- **Ctrl+click** a second sheet → both selected (the swatch splits to show both ids).

### Editing sheets

The corrections operate on the current selection and run off-thread behind a loading overlay. Noise
points are never shown, so a corrected window reads cleanly.

| Edit | When enabled | Effect | Key |
|---|---|---|---|
| **split** | exactly 1 selected | Force-splits the sheet at its weakest embedding seam; **keeps the sheet's id + colour** for one half, the other becomes one new sheet (auto-selected). | **s** |
| **merge** | exactly 2 selected | Force-merges the two into the **larger** sheet (keeps its id + colour). | **m** |
| **delete** | ≥1 selected | Marks the sheet(s) as noise **and** flags their footprint as a **low-confidence region** (visible under the *confidence* overlay). | **d** |
| **undo** | after any edit | Reverts the last edit (a stack of the **last 5**); restores labels, meshes, colours, and confidence. | **Ctrl+Z** |

### Grinding to the next window

The **save + next →** control (top-right; keys **N** / **→**) advances to the next window through a
sequence of centred **popovers**. It first asks for **confirmation** ("save this window's edits and move
to the next?"), then:

1. It **saves the current window's edited labels** into the corpus (overwriting that window, marked
   `edited`) — only if you made edits; an unedited window keeps the base entry already written when it
   loaded.
2. It loads the next window (resetting the whole viewer) — grinding a fresh one, or (in `edit`) loading
   the next stored window.

A counter shows **`window k / N`** (coords-file mode) or **`window k / ?`** (open-ended — total unknown).
In open-ended mode a second popover chooses the next window: **Random window**, or **Specify coords…**
(type `Z,Y,X` or `SCROLL,Z,Y,X`). Re-entering the coordinates of the window you're already on is rejected.
With a coords file it simply steps to the next entry; when the list is exhausted it stops. A window whose
pipeline fails is not a dead end — **save + next** stays enabled so you can skip it.

---

## Outputs

### The corpus (`<name>.herculabels/`)

`create`/`edit` write a corpus **directory**:

```
<name>.herculabels/
  meta.json                                # manifest (version, reproduction params, per-window entries)
  windows/
    s<scroll>_L<level>_z<z>_y<y>_x<x>.npz  # one editable window each
```

Each window `.npz` holds the fitted **base sheet meshes** (`base/V_<c>` centre-surface grid, packed
`base/foot_<c>`, `base/width_<c>`) and the **meshlet cloud** (`meshlet/pts`, `sid`, `plab`, `normals`,
`jac`, `prop_dist`, `E`, `ulab`, `vu`, `shape`) — **no dense CT/field volumes and no confidence**.
Confidence is fully derivable (intersection from the meshes, sharp2 from `prop_dist`, dropped from the
`plab == -1` footprint), so it's recomputed on export and shown live in the viewer, never stored.

`meta.json` is the manifest — everything needed to reproduce the corpus, plus per-window provenance:

```jsonc
{
  "herculabels_version": "v1",
  "created": "…", "code_git_sha": "…",
  "create": {                        // how to reproduce the corpus
    "mode": "interactive" | "headless", "scroll": "PHerc1447" | null, "seed": 0,
    "source": "random" | "coords" | "coords-file", "coords_file": null, "himat": null,
    "count": 128 | null, "params_hash": "…", "gen_params": { … }
  },
  "windows": [                       // one entry per window, in creation order
    { "id": "s1447_L0_z11714_y3293_x3648", "scroll": "PHerc1447", "level": 0,
      "coords": [11714, 3293, 3648], "source": "random", "seed": 0, "n_sheets": 17,
      "edited": false,               // true once you split/merge/delete here
      "quality": { "window": 0.83, "per_sheet": { "3": 0.91, "7": "manual", … } } }
  ]
}
```

Window content is reproducible via the pipeline's determinism (`master_seed = md5(run_id|scroll|level|coords)`);
augmentations are regenerated at export and are not required to be bit-reproducible.

### The training export (`export`)

`export` writes the per-window training samples the refiner consumes (a plain `.npz` dir, **no meshlets**):

```
<train_out>/sample_s<scroll>_L<level>_z<z>_y<y>_x<x>.npz   +  export_meta.json
```

Each sample holds, per record (the `base` record, plus merge/split augmentations unless `--no-augment`):

- **`meshes`** — the fitted per-sheet medial surfaces, one per cluster id.
- **`conf`** — regional confidence channels, downsampled + quantised: `intersection` (sheet-envelope
  overlap), `sharp2` (embedding propagation-distance), `dropped` (dropped/deleted-point — a deleted sheet's
  low-confidence region is captured here, being marked noise), plus `perturb` on augmentation records.
- **`meta_json`** — provenance: scroll, level, `voxel_um`, `coords`, `org`, `shape`, `run_id`, per-sheet
  quality, and `schema_version`.

---

## Parameters (operating point)

The generation knobs default to the validated corpus operating point; change them only deliberately.

| Flag | Default | Meaning |
|---|---|---|
| `--voxel-min` / `--voxel-max` | `7.5` / `9.5` | Voxel-size band (µm) to sample scans from (corpus native-coarse domain). |
| `--himat` | off | Only accept windows whose **material fill** is ≥ this fraction (0–1) — for mining HIGH-material sheets (as with the m7 himat pseudo-labels). Applies to random draws (headless and interactive); the interactive overlay then reads "searching for a window with N% material fill…". Default: the standard 0.15 air-skip. |
| `--seed-stride-um` | `40` | Spine seed spacing (larger ⇒ fewer streamlets, faster). |
| `--sample-um` | `20` | Streamlet point spacing. |
| `--min-cluster-size` | `250` | Minimum probeom cluster size (in units). |
| `--old-negatives` | off | Use the **old gap-gated** negatives (methodology §5.2) instead of the default **slab-field** negatives (§5.3). ⚠️ The default is now the slab-field method; **the published corpus was generated with `--old-negatives`** — pass it to reproduce that corpus (see [Reproducing the published corpus](#reproducing-the-published-corpus)). |
| `--seed` | `0` | Base seed for the deterministic sampling stream + `run_id`. |
| `--no-gpu` | (GPU on) | Force CPU. GPU is used by default; label generation is impractically slow on CPU. |
| `--no-deterministic` | (determinism on) | Disable bit-reproducible generation (faster). Determinism is on by default (same window ⇒ byte-identical base labels). |

---

## Reproducing the published corpus

> ⚠️ **The published stage-1 corpus was generated with the OLD gap-gated negatives — you MUST pass
> `--old-negatives` to regenerate it.** `create` now defaults to the newer slab-field negatives
> (methodology §5.3); without `--old-negatives` you get an *equivalent* corpus under the current method,
> **not** the published one.

The published corpus is **4031 windows across 22 scrolls** (`run20260808`). Its full provenance — every
window's coordinates and the exact extraction parameters — is in
[`submission/writeup/corpus.md`](../submission/writeup/corpus.md), and the window list is the coordinate file
[`submission/writeup/corpus_windows.txt`](../submission/writeup/corpus_windows.txt):

```bash
# regenerate exactly the published windows (note --old-negatives), then derive the training bundles:
pipenv run hercunet labels create scroll_corpus.herculabels \
    --old-negatives --coords-file submission/writeup/corpus_windows.txt
pipenv run hercunet labels export scroll_corpus.herculabels ./train_out

# …or a fresh, equivalent draw from the same operating point:
pipenv run hercunet labels create scroll_corpus.herculabels --old-negatives --count 4031 --seed 20260808
```

Per-window content is deterministic from `(run_id, scroll, coords)`; the original run spanned several code
revisions, so the result is equivalent by construction (same method, params, and `--old-negatives`) rather
than byte-identical to any single commit.

---

## Notes & gotchas

- **British spelling** throughout — flags and outputs use `colour`, `visualise`, `finalise`.
- **Determinism** sets `CUBLAS_WORKSPACE_CONFIG` and torch deterministic algorithms; a re-run of the
  same window reproduces the base labels exactly. Augmentations are regenerated at `export` and are not
  required to be bit-reproducible.
- **The corpus never stores augmentations** — they are a training-only artifact, generated by `export`
  (on by default). This keeps the corpus lean and always editable; edit and re-export as often as you like.
- **A corpus is always editable** — there is no destructive "finalise" step. The non-editable form is
  simply the `export` output (it has no meshlets, so nothing to edit).
- **Edits keep the 3-D camera** and never re-frame it, so a split/merge/delete is easy to see; the camera
  only re-frames on a new window or when you switch the sheets/embeddings view.
- **Robust to sparse windows** — a high `--himat` search is bounded (it falls back to the best window
  seen rather than hunting forever), and empty/degenerate or failed windows are handled without crashing
  (skip them with **save + next**).
- **`Ctrl-C`** exits cleanly from both the CLI and the viewer.

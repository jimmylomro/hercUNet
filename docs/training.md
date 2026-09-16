# Stage 2 — training the HercUNet refiner (`hercunet train …`)

Train the iterative surface refiner over the stage-1 pseudo-labels. The pipeline mirrors the
research-pod chain as **separate, resumable steps** under one command group, `hercunet train`,
and runs over a **pinned, unmodified** nnU-Net.

- **No fork, no clone, no copy-paste.** nnU-Net is a normal pip dependency
  (`nnunetv2==2.8.1`). Trainers live in `hercunet`; nothing is ever copied into the `nnunetv2`
  package (see [How it works](#how-it-works)).
- **All library calls — no subprocess.** Every step calls nnU-Net functions in-process; nothing
  shells out to `nnUNetv2_*` commands.

> ⚠️ **Pod-only + heavy.** Training needs a CUDA GPU, the preprocessed corpus, and the m7
> checkpoint. Do it on a training box, not a laptop.

## Install

```bash
pip install -e ".[train]"      # adds nnunetv2==2.8.1 + huggingface_hub, acvl-utils, connected-components-3d, numba, blosc2
```

The `train` extra is **heavy and opt-in** — the data layer, the viewer, and stage-1 label
generation never import it. Pin stays exact: the trainers reproduce nnU-Net's `train_step`
internals, so the version matters (`2.8.1` is what the released models were trained on).

nnU-Net keeps its data under three roots — `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`. You do
**not** have to export them: every `hercunet train` step that touches them takes **`--nnunet-path <dir>`**, a
single directory holding those three subdirs, and sets the roots in-process (nnU-Net reads them lazily, so
this is equivalent to exporting — no env management on your part):

```bash
# one dir with nnUNet_raw/  nnUNet_preprocessed/  nnUNet_results/  underneath it
hercunet train chain --nnunet-path /workspace/nnunet …
```

Exporting the env vars still works and is the way to use a **split layout** (roots in different places); an
explicit `--nnunet-path` wins over them. If a root a step needs is set by neither, the command fails fast
naming which one is missing (rather than raising deep inside nnU-Net):

```bash
export nnUNet_raw=/path/nnUNet_raw
export nnUNet_preprocessed=/path/nnUNet_preprocessed
export nnUNet_results=/path/nnUNet_results
```

## The chain

Four steps, run in order — the same code whether you run them one at a time or via `chain`:

```
export-labels  →  preprocess  →  export-owner  →  fit
```

This is the **HercUNet (run301 / v0) recipe** — the iterative AffinityMalis refiner, **not** the
single-pass ablation.

> **The m7 rehearsal labels are mined once, in the labels stage** — not during training prep.
> `hercunet labels m7-mine` scouts coherent m7 crest in compressed regions (from the published m7
> predictions) and writes a **pre-extracted m7 corpus**; `train export-labels --m7-corpus` then
> **copies** those cases into the dataset (no re-mining, no symlinks). See
> [herculabels.md](herculabels.md) for `m7-mine`.

| Step | `hercunet train …` | What it does | Status |
|---|---|---|---|
| 1 | `export-labels` | our gate-passing sample corpus → nnU-Net cases: CT + K candidate/prev crests + `ownersTr` (MALIS instance GT), **uncleaned**; with `--m7-corpus`, also **copy** the pre-mined m7 cases in as `m7_*` | ✅ live |
| 2 | `preprocess` | fingerprint → transplant m7's ResEncUNetL plans → **patch to 1+K input channels** → preprocess | ✅ live |
| 3 | `export-owner` | write `<case>_owner.b2nd` per-sheet instance GT for the MALIS objective | ✅ live |
| 4 | `fit` | train the refiner by **direct instantiation** of the trainer class (single-GPU or DDP); knobs via `CK_*` env | ✅ live |

All four steps are our own in-process code (no nnU-Net internals in 1/3, stock library calls in 2/4). Step 1
consumes a directory of `sample_s*.npz` bundles written by `hercunet labels export` (the light corpus→samples
step) and decodes them with the library's own ∇φ EDT decoder (`hercunet.labels.mesh.medial.build_gradphi_edt`)
— the two commands stay separate on purpose (`labels export` is cheap; `export-labels` is the heavy CT-fetch +
decode + emit).

### Run the whole thing

```bash
# 0) once, in the labels stage: pre-extract the m7 rehearsal corpus
hercunet labels m7-mine /path/m7_corpus --scroll s1,s5 --max-candidates 4

# 1) the training chain (copies the m7 corpus in during export-labels).
#    --nnunet-path sets the three nnU-Net roots; --out is <that dir>/nnUNet_raw/DatasetNNN_…
hercunet train chain \
  --nnunet-path /workspace/nnunet \
  --corpus    /path/corpus \
  --m7-corpus /path/m7_corpus \
  --out       /workspace/nnunet/nnUNet_raw/Dataset301_AffMalisFull \
  --dataset 301 --max-candidates 4 \
  --pretrained /path/m7/checkpoint_best.pth --num-gpus 4
```

`export-labels` writes our cases and copies the m7 corpus's cases into the **same** `--out` dataset;
`--max-candidates` (K, the prev/candidate channels) must be identical across `labels m7-mine`,
`export-labels`, and `preprocess`.

`chain` calls each step's function in order — it is a thin orchestrator, not a copy of the step
logic. It is **resumable**:

- `--from <step>` — start here (skip earlier steps)
- `--until <step>` — stop after here
- `--skip <step> …` — drop specific steps

```bash
# resume after data prep: just preprocess → export-owner → fit
hercunet train chain --from preprocess --dataset 301 \
  --pretrained /path/m7/checkpoint_best.pth --num-gpus 4
```

If a selected step is missing a required argument (e.g. `--from fit` without `--pretrained`), the
chain fails fast with a clear message before doing any work.

## The steps individually

### `preprocess` — fingerprint → m7 plans → patch channels → preprocess

```bash
hercunet train preprocess --dataset 301 --max-candidates 4
```

Wraps three stock nnU-Net 2.8.1 functions in-process:
`extract_fingerprints` → `move_plans_between_datasets` → `preprocess`. It first seeds a
`Dataset100` "plans source" from the published m7 checkpoint (downloaded via `huggingface_hub`)
and transplants m7's `nnUNetResEncUNetLPlans` onto each target dataset — **so we train in m7's
exact ResEncUNetL geometry** — then **patches the plans to `1+K` input channels** (CT + K prev/
candidate crests) for the iterative refiner. Flags: `--max-candidates K` (default 4; `0` = no
patch), `--config` (default `3d_fullres`), `--plans-id`, `--np`, `--no-setup-plans`, `--m7-repo`.

### `fit` — train the refiner

```bash
hercunet train fit --dataset 301 --pretrained /path/m7/checkpoint_best.pth --num-gpus 4
```

`fit` **imports the trainer class and calls `trainer.run_training()`** — it never uses nnU-Net's
`-tr` string discovery, so no trainer file is copied into the `nnunetv2` package. Flags:

- `--trainer` — defaults to the alias `hercunet` (the AffinityMalis iterative-DAgger refiner); or
  an explicit `module:ClassName` (e.g. `hercunet.training.trainers.affinity_malis_iter_dagger:nnUNetTrainer_AffinityMalis_IterDagger_500epochs`).
- `--dataset` (id or `DatasetNNN_…`), `--config`, `--fold`, `--plans-id`.
- `--pretrained CKPT` — **warm-start** from a checkpoint. This is the single warm-start path (no env
  var): the trainer's `load_pretrained` inspects the checkpoint and does the right thing per parameter —
  an **m7-like** checkpoint (single-channel input stem) has its stem **expanded 1→8 channels** (channel 0 =
  m7's CT weights, the `prev` + orientation channels zero-init, so the first forward matches m7) and the
  affinity head initialises fresh; a **HercUNet-shaped** checkpoint loads directly. The log line reports
  `matched / stem-expanded / fresh` so a wrong checkpoint is obvious.
- `--num-gpus` — `>1` runs DDP via `torch.multiprocessing.spawn` (nnU-Net's own mechanism; each
  worker re-imports the class and runs the same in-process path — no shell).
- `--device` (`cuda`|`cpu`), `--continue` (**resume** from the last checkpoint — same-architecture load —
  instead of warm-starting).
- `--recipe FILE.yml` — train with a YAML recipe (see below); omit for the exact run301 recipe.
- `--epochs N` — override the training length (e.g. `--epochs 1` for a smoke run).

Checkpoints land in the standard `$nnUNet_results/…/fold_N/` tree, so inference (`hercunet infer`,
`nnUNetv2_predict`) finds them unchanged.

### The training recipe (hyperparameters — a YAML file, no env vars)

Every refiner hyperparameter lives in **one auditable, shipped file**,
[`hercunet/recipes/hercunet.yml`](../hercunet/recipes/hercunet.yml) — installed with the package. It holds
**exactly** the values HercUNet v0 (run301) trained with, and it *is* what the trainer loads by default — so a
bare `hercunet train fit` reproduces the recipe with **no environment variables** (the old `CK_*` env vars are
gone). ([`hercunet/training/recipe.py`](../hercunet/training/recipe.py) is just the schema — types + docs.) At
the start of every run the trainer logs the resolved recipe (`[recipe] run301: malis_w=1.0  …`) into the
nnU-Net training log, so each run carries its own provenance.

| Group | Fields (run301 default) |
|---|---|
| Constrained MALIS | `malis_w=1.0`, `malis_warm_epochs=50`, `malis_crop=48`, `aff_offsets=[1,3,9]` |
| CT-material growth | `mat_w=1.0`, `mat_warm_epochs=50`, `mat_tau=0.0`, `mat_scale=0.4` |
| Online DAgger | `dagger_prob=0.5`, `dagger_maxk=2` |
| Prev composition | `prev_pboot=0.3`, `prev_tiling=octant`, `prev_augprob=0.5`, `prev_zeroprob=0.15`, `prev_noise=0.0`, `prev_fragprob=0.5`, `prev_fragslabmax=60`, `prev_fraghalfprob=0.4` |
| Val viz (debug, off) | `val_viz=0`, `val_viz_images=10`, `val_viz_slices=10` |

**Override — copy the YAML, edit, pass it (auditable, no env).** Copy the shipped recipe, change the fields you
want, and run with `--recipe`. Any field you omit falls back to its run301 value; unknown field names are
rejected (a typo can't silently no-op). The YAML file is a keepable record of the variant, and the resolved
recipe is logged like any other run:

```bash
cp "$(python -c 'import hercunet.recipes,os;print(os.path.dirname(hercunet.recipes.__file__))')/hercunet.yml" my_recipe.yml
# edit my_recipe.yml (e.g. prev_fragprob: 0.3), then:
hercunet train fit --dataset 301 --pretrained /path/m7/checkpoint_best.pth --recipe my_recipe.yml
```

Training **length** is separate from the recipe: `--epochs N` overrides it (the trainer class is 500 epochs;
`--epochs 1` is the usual smoke). In code, a variant is `dataclasses.replace(RUN301, prev_fragprob=0.3)`.
`--recipe` / `--epochs` also work on `hercunet train chain`.

### `export-labels` / `export-owner` (+ `labels m7-mine`)

Dataset assembly — our own in-process code (no nnU-Net internals), the **HercUNet** recipe (not the ablation):

- `export-labels` (`hercunet/train/dataset.py`) — reads the `sample_s*.npz` bundles from `hercunet labels
  export`, decodes each window with `hercunet.labels.mesh.medial.build_gradphi_edt` (the library's CPU/EDT ∇φ
  decoder, returning the unbounded-Voronoi `owner_full` for the collapsed-sheet ignore), and emits `imagesTr`
  (CT + K candidate crests), `labelsTr` ({0,1,2}), `ownersTr` (band-limited instance GT for MALIS), and
  `dataset.json`. `--m7-corpus` then **copies** the pre-mined m7 cases in as `m7_*` (real writes, no symlinks;
  m7 cases carry no `ownersTr`, so MALIS stays silent on them).
- `hercunet labels m7-mine` (`hercunet/labels/m7mine.py`) — scout (`--scroll`) or deterministic rebuild
  (`--manifest`) of coherent m7 crest from the **published** m7 predictions + CT on the open-data bucket
  (`surf = m7 & (m7_normal_coherence > tau)`), written as a standalone **m7 corpus**. Lives in the **labels**
  stage — the mining is pseudo-label generation, done once, not during training prep.
- `export-owner` (`hercunet/train/dataset.py`) — self-contained: crops each `ownersTr/<case>.tif` with the
  preprocessing's `bbox_used_for_cropping` and writes `<case>_owner.b2nd` at preprocessed resolution.

## How it works

**Why there was ever a copy-paste.** nnU-Net's `nnUNetv2_train -tr <Name>` resolves the trainer by
name with a search scoped to the installed package directory, so a custom trainer historically had
to be copied into `nnunetv2/training/nnUNetTrainer/`. `hercunet train fit` sidesteps this entirely
by **passing the class it imported** instead of a name string — the discovery is never invoked. It
is a faithful reduction of nnU-Net's own `run_training` (build plans/dataset.json, construct the
trainer, optional pretrained load, `run_training()`), minus the name lookup; DDP mirrors nnU-Net's
`run_ddp`/`mp.spawn`.

**Package layout.** Trainers live under `hercunet.training.trainers.*` (subclass `nnUNetTrainer`,
import their siblings by `hercunet.*` paths); the launch driver is `hercunet/train/launch.py`; the
preprocessing wrapper is `hercunet/train/preprocess.py`. The `--trainer` aliases resolve to those
modules.

**Design invariants** (do not break): pin `nnunetv2==2.8.1`; `build_network_architecture` is a
staticmethod the trainer overrides to build the 8-channel `[CT, prev, orientation×6]` stem +
affinity head; the owner-id sidecar feeds the constrained-MALIS objective; the m7-plans transplant
is what keeps our geometry identical to m7.

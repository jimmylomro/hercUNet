# HercUNet — installation & usage

The entry point for **running** HercUNet: requirements, install, the CLI, and pointers to the per-part
guides. Deep detail lives in the specific guides — e.g. [herculabels.md](herculabels.md) for stage-1 label
generation and the viewer; the research **methodology** is under
[`../submission/writeup/`](../submission/writeup/). Project overview: the [project README](../README.md).

> ⚠️ **Compute warning — stage-1 label generation is heavy.** Generating one window is **very RAM-hungry
> (often > 20 GB)** and slow: on a laptop (Intel i9, 32 GB RAM, 8 GB RTX 4060 GPU) **a single window takes
> ~4–5 minutes**. Because of this, the recommended workflow is to **build corpora non-interactively**
> (`create --count …`, or `--coords-file` for a known set — headless) and then **`edit` only the windows you
> want to correct**. The interactive viewer runs the *full* pipeline per window, so it is for spot-checking
> and correction, **not** bulk generation.

## Requirements

- **Linux**, **Python ≥ 3.9** — the package's real minimum (the code is 3.9-compatible). The shipped
  `Pipfile` pins **3.12** as the dev interpreter, so `pipenv install` wants a 3.12 Python; plain `pip install`
  works on any ≥ 3.9.
- **[pipenv](https://pipenv.pypa.io)** — the repo ships a `Pipfile`; run everything through it (`pipenv run …`).
- **An NVIDIA GPU + CUDA** for the heavy compute — stage-1 label generation, stage-2 refiner **training**, and
  inference. The data layer, the CLI, and reading/merging/exporting corpora work **CPU-only**; anything that
  actually looks at voxels wants the GPU — see [GPU / CUDA](#gpu--cuda).

## Install

```bash
git clone https://github.com/jimmylomro/hercUNet.git
cd hercUNet
pipenv install                 # core: streaming data layer + stage-1 label generation
pipenv install -e ".[viz]"     # + the interactive label viewer / editor (PySide6, pyqtgraph, PyOpenGL)
```

`pipenv install` installs the package (editable, via the `Pipfile`) with its full runtime. The viewer is an
**opt-in extra** (`[viz]`) — core label generation and the CLI never import it. Plain pip works too:
`pip install .` (or `pip install ".[viz]"`); add `-e` only if you're editing the code.

**Stage-2 training** (`hercunet train …`) needs an **NVIDIA GPU** — a local workstation or a pod. Clone the repo
and install the `train` extra:

```bash
pip install ".[train]"             # contributors editing the code: add -e for an editable install
```

> **Note — reuse an existing CUDA `torch`.** The `train` extra pulls a default `torch` wheel. If your machine
> already has a working CUDA `torch` you want to keep (common on GPU pods, or a curated local env), install into
> a venv created with `--system-site-packages` so it is inherited rather than re-downloaded — this also
> sidesteps a PEP-668 "externally managed" system `pip`:
> ```bash
> python3 -m venv --system-site-packages .venv && .venv/bin/pip install ".[train]"
> ```

See **[training.md](training.md)** for the full walk-through — install notes, getting the data (download vs
build), and the train/resume commands.

### Extras

| Extra | Adds | For |
|---|---|---|
| _(none)_ | data layer + stage-1 label gen | the default — everything runs |
| `viz` | PySide6, pyqtgraph, PyOpenGL | the interactive viewer (`create --interactive`, `edit`) |
| `accel` | obstore | a faster native (non-boto) S3 reader |
| `train` | nnunetv2==2.8.1 + huggingface_hub, acvl-utils, cc3d, numba, blosc2, tifffile, pyyaml | stage-2 refiner training (`hercunet train …`) — needs a CUDA GPU |
| `infer` | nnunetv2==2.8.1 + huggingface_hub + s5cmd | stage-3 full-volume inference (`hercunet infer …`) — needs a CUDA GPU |
| `full` | viz + accel + train + infer | everything |
| `dev` | pytest | the test suite |

## GPU / CUDA

The compute-heavy stages need a **CUDA GPU**: stage-1 label generation (refused on CPU by default — `--no-gpu`
exists only for slow CPU testing), stage-2 refiner **training**, and inference. Everything else — the data
layer, the CLI, reading/merging/exporting corpora — runs without a GPU. The `torch` guidance below applies to
all of them.

`torch>=2.0` is left **unpinned** in `pyproject.toml` on purpose, so it works with whatever PyTorch / CUDA
build you install — the CUDA version is chosen by *which wheel* you install, not by a version number:

- **Default.** `pipenv install` pulls the default PyPI `torch` wheel, which bundles a **CUDA 12.x** runtime.
  You only need a reasonably recent NVIDIA driver (CUDA-12-capable — Linux driver ≳ 525); the wheel ships
  its own CUDA runtime, so nothing CUDA-related has to be installed system-wide.

- **A different CUDA build** (older CUDA 11.8, a specific 12.x, a future CUDA 13, or CPU-only): install the
  matching `torch` wheel from the PyTorch index **first**, then install the package:

  ```bash
  # pick the tag that matches your driver from https://pytorch.org/get-started/locally/
  pipenv run pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA 12.4
  pipenv run pip install torch --index-url https://download.pytorch.org/whl/cu118   # CUDA 11.8
  pipenv run pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU only
  pipenv install
  ```

  The `cuNNN` tag is the CUDA version (`cu118`, `cu121`, `cu124`, `cu128`, …); when PyTorch publishes
  CUDA-13 wheels, use their `cu13x` tag. The live matrix is at <https://pytorch.org/get-started/locally/>.

- **Pinning a build in the repo (reproducible env).** Because CUDA is selected by the index and not by a
  version spec, you do **not** edit `torch>=2.0` in `pyproject.toml`. Instead add the PyTorch index as a
  source in the **`Pipfile`** and require `torch` from it:

  ```toml
  [[source]]
  name = "pytorch"
  url = "https://download.pytorch.org/whl/cu124"     # ← change the cuNNN tag here (e.g. cu118, cu128, cu130)
  verify_ssl = true

  [packages]
  torch = { version = ">=2.0", index = "pytorch" }
  hercunet = { editable = true, path = "." }
  ```

  then `pipenv lock && pipenv install`.

Verify the install:

```bash
pipenv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## CLI

Everything is one `hercunet` console script. Run `pipenv run hercunet --help`, or
`pipenv run hercunet <group> <command> --help` for any subcommand.

### Stage 1 — labels (`hercunet labels …`)

Build, edit, combine, and export `.herculabels` pseudo-label corpora:

| Command | Purpose |
|---|---|
| `create <corpus.herculabels> …` | generate a new corpus — `--interactive` (viewer grind) or headless (`--count N`, or `--coords-file FILE` for an exact set) |
| `edit <corpus.herculabels>` | re-open a corpus in the viewer to correct its windows in place |
| `merge <out.herculabels> <in…>` | union several corpora into a new one |
| `export <corpus.herculabels> <train_out>` | derive the training corpus (base + augmentations; `--no-augment` for base only) |
| `m7-mine <m7_corpus> …` | pre-extract an m7 pseudo-label corpus from compressed regions (the rehearsal signal; consumed by `train export-labels --m7-corpus`) |

The full flag list, the interactive grinding viewer (navigation, overlays, split/merge/delete, undo,
save + next), the outputs, and the `.herculabels` container format are in **[herculabels.md](herculabels.md)**.

### Stage 2 — training (`hercunet train …`)

Train the iterative refiner over a pinned, unmodified `nnunetv2==2.8.1` (no fork, no clone, no
trainer copy-paste; all in-process, no subprocess). The chain is separate, resumable steps:

| Command | Purpose |
|---|---|
| `export-labels … [--m7-corpus M7]` | our gate-passing corpus → nnU-Net cases (CT + K candidate/prev crests + owners, uncleaned); copies a pre-mined m7 corpus in |
| `preprocess --dataset NNN …` | fingerprint → transplant m7's ResEncUNetL plans → patch to 1+K channels → preprocess |
| `export-owner --dataset NNN` | write the per-sheet owner-id MALIS sidecars |
| `fit --dataset NNN …` | train the refiner by direct instantiation (single-GPU or DDP) |
| `chain …` | run all of the above in order — resumable with `--from` / `--until` / `--skip` |

Needs the `train` extra and the nnU-Net env roots. Full details, flags, and the design (why there is
no copy-paste) are in **[training.md](training.md)**.

### Stage 3 — inference (`hercunet infer …`)

Run the trained refiner over a whole scroll (or a sub-cube) as iterative Jacobi passes and write a per-pass
surface-probability OME-Zarr. The network is built stock from `plans.json` and loaded from `network_weights`, so
inference needs only the pinned `nnunetv2==2.8.1` (the `infer` extra), not the training code:

| Command | Purpose |
|---|---|
| `single-instance …` | one box — fans across its local GPUs automatically (one worker per GPU); you run one command |
| `multi-instance …` | many pods — run the same command per pod over a shared network volume, `--leader` on one |

The model is public on Hugging Face: `--model-hf jimmylomro/hercunet-v0` (no token), or `--model <folder>` for a
local run. Defaults reproduce HercUNet v0 (overlap 0.25, 4 passes, final pass = deliverable).

**Simplest run** — nothing local, no S3 (pulls the model from HF, streams a ~5 mm PHerc1447 window, writes the
result beside you):

```bash
hercunet infer single-instance --model-hf jimmylomro/hercunet-v0 --scroll PHerc1447 \
  --region 10889:11401,2848:3360,3915:4427 --out ./infer-demo --passes 3
# -> ./infer-demo_pass2.zarr  (open in VC3D). Uses every visible GPU; add --keep-buffers to compare passes.
```

Full walk-through, the smoke-test recipe, and all flags are in **[infer.md](infer.md)**.

## Library — the data layer

Voxels are read through a streaming multiscale OME-Zarr **data layer** (backends, chunk-aligned parallel
reads, an on-disk cache, prefetch, material tiling), configured via `HERCUNET_*` env vars / `Config`. See
**[data-layer.md](data-layer.md)** for the API.

## Guides

| Document | Covers | Status |
|---|---|---|
| **[data-layer.md](data-layer.md)** | The streaming data layer — backends, `ZarrSegment`, chunk-aligned parallel reads, the on-disk cache, prefetch (`iter_windows` / `VolumePrefetcher`), material tiling, configuration, performance notes. | ✅ available |
| **[herculabels.md](herculabels.md)** | **Stage 1 usage** — the `create` / `edit` / `merge` / `export` CLI over `.herculabels` corpora, the interactive grinding viewer, and the corpus + export outputs (incl. the `meta.json` manifest). | ✅ available |
| **[training.md](training.md)** | **Stage 2 training** — the `hercunet train` chain (export → build → preprocess → export-owner → fit) over pinned `nnunetv2==2.8.1`, direct-instantiation launch (no copy-paste), DDP, and the design. | ✅ available |
| **[infer.md](infer.md)** | **Stage 3 inference** — the `hercunet infer` single-instance (local multi-GPU) / multi-instance (pods) commands, iterative Jacobi refinement with the Gaussian blend, OME-Zarr output, and the small-cube smoke test. | ✅ available |

The **methodology** for stage 1 (how the labels are derived — meshlets, slab, contrastive embedding,
probeom, medial fit, confidence, and the ablation validating it) is the research write-up
[`../submission/writeup/herculabels.md`](../submission/writeup/herculabels.md), with the published-corpus
provenance in [`../submission/writeup/corpus.md`](../submission/writeup/corpus.md).

## Reading order

1. **This page** — install, GPU/CUDA, and the CLI.
2. **[data-layer.md](data-layer.md)** — everything reads voxels through it.
3. **[herculabels.md](herculabels.md)** — running stage 1 and the viewer.
4. **[training.md](training.md)** — stage 2, training the refiner (`hercunet train`).
5. **[infer.md](infer.md)** — stage 3, full-volume inference (`hercunet infer`).

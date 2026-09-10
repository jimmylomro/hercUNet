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
- **An NVIDIA GPU + CUDA** for stage-1 label generation. The data layer, the CLI, and reading corpora work
  CPU-only; label generation is GPU-only in practice — see [GPU / CUDA](#gpu--cuda).

## Install

```bash
git clone https://github.com/jimmylomro/hercUNet.git
cd hercUNet
pipenv install                 # core: streaming data layer + stage-1 label generation
pipenv install -e ".[viz]"     # + the interactive label viewer / editor (PySide6, pyqtgraph, PyOpenGL)
```

`pipenv install` installs the package editable with its full runtime. The viewer is an **opt-in extra**
(`[viz]`) — core label generation and the CLI never import it. Plain pip works too (`pip install -e .`,
`pip install -e ".[viz]"`).

### Extras

| Extra | Adds | For |
|---|---|---|
| _(none)_ | data layer + stage-1 label gen | the default — everything runs |
| `viz` | PySide6, pyqtgraph, PyOpenGL | the interactive viewer (`create --interactive`, `edit`) |
| `accel` | obstore | a faster native (non-boto) S3 reader |
| `full` | viz + accel | both |
| `dev` | pytest | the test suite |

## GPU / CUDA

Stage-1 label generation requires a **CUDA GPU** — it is refused on CPU by default (`--no-gpu` exists only
for slow CPU testing). Everything else (the data layer, the CLI, reading/merging/exporting corpora) runs
without a GPU.

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

The full flag list, the interactive grinding viewer (navigation, overlays, split/merge/delete, undo,
save + next), the outputs, and the `.herculabels` container format are in **[herculabels.md](herculabels.md)**.
Stages 2 (refine) and 3 (infer) land as they are ported.

## Library — the data layer

Voxels are read through a streaming multiscale OME-Zarr **data layer** (backends, chunk-aligned parallel
reads, an on-disk cache, prefetch, material tiling), configured via `HERCUNET_*` env vars / `Config`. See
**[data-layer.md](data-layer.md)** for the API.

## Guides

| Document | Covers | Status |
|---|---|---|
| **[data-layer.md](data-layer.md)** | The streaming data layer — backends, `ZarrSegment`, chunk-aligned parallel reads, the on-disk cache, prefetch (`iter_windows` / `VolumePrefetcher`), material tiling, configuration, performance notes. | ✅ available |
| **[herculabels.md](herculabels.md)** | **Stage 1 usage** — the `create` / `edit` / `merge` / `export` CLI over `.herculabels` corpora, the interactive grinding viewer, and the corpus + export outputs (incl. the `meta.json` manifest). | ✅ available |
| _refine.md_ | Stage 2 — the iterative HercUNet refiner (8-ch affinity/MALIS trainer, N-pass wrapper). | 🚧 planned |
| _infer.md_ | Stage 3 — single-instance full-volume inference (tile+halo, OME-Zarr out). | 🚧 planned |

The **methodology** for stage 1 (how the labels are derived — meshlets, slab, contrastive embedding,
probeom, medial fit, confidence, and the ablation validating it) is the research write-up
[`../submission/writeup/herculabels.md`](../submission/writeup/herculabels.md), with the published-corpus
provenance in [`../submission/writeup/corpus.md`](../submission/writeup/corpus.md).

## Reading order

1. **This page** — install, GPU/CUDA, and the CLI.
2. **[data-layer.md](data-layer.md)** — everything reads voxels through it.
3. **[herculabels.md](herculabels.md)** — running stage 1 and the viewer.
4. _refine.md_, _infer.md_ — added as each stage lands.

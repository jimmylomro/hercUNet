# CLAUDE.md — working in the HercUNet repo

Guidance for AI agents (and humans) developing in this repo. Read this first, then the relevant `docs/` guide
**before** writing code. This is a clean, self-contained package with a real CLI and a documented pipeline —
treat it as such.

> This branch (`hercuvit`) also hosts **HercuViT** — a Dinovol-based sheet-segmentation net built PARALLEL to
> HercUNet (`hercunet/hercuvit/`, its own `hercuvit` CLI, spec in `docs/hercuvit.md`). It deliberately **reuses**
> HercUNet's machinery (data layer, label pipeline, losses). The rule below applies to it with full force.

## ⛔ STOP — do NOT rewrite code that already exists here. READ IT AND REUSE IT.

Reimplementing logic the repo already has is the **#1 cause of silent, expensive bugs** in this codebase. If a
function, reader, loss, dataset step, or CLI path already does the task, **call it** — do not reimplement it, do
not "write a cleaner version", do not paraphrase its math into new code. The existing code encodes **hard-won,
non-obvious conventions** (coordinate origins, voxel scales, normalisation, confidence masking, label semantics)
that a function signature does NOT reveal and that you *will* get wrong by rewriting.

**This is not hypothetical — here is what it already cost us.** HercuViT's CT reader was rewritten from scratch
and treated `meta["org"]` as the window **ORIGIN**. But the entire label pipeline writes `meta["org"]` as the
window **CENTRE** — the real reader, `hercunet/train/dataset.py:_window_req`, does `z0 = centre - shape//2`
(clipped to the level shape; "same origin math as label generation's open_ct"). The rewrite therefore read every
CT crop **offset by half a window on every axis**, so the CT was misaligned with its label, the net "wouldn't
learn", and it burned a long multi-session debugging arc chasing the head, the attention, and the losses —
**all of which simply reusing `_window_req` would have avoided for free.**

**The rule:**
1. Before writing any data / IO / coordinate / label / loss / pipeline code, **search the repo for the capability
   first** (`grep -r`, read the relevant `docs/` guide) and **reuse it verbatim**.
2. If something genuinely doesn't exist, write ONE **new reusable function in the correct module** and say so —
   never a one-off script that re-derives existing logic.
3. When a value's meaning is non-obvious (is `org` a centre or an origin? what voxel size is this? is `owner==0`
   background or ignore?), **read the code that produces it**, do not guess. Verify against code before you act.

## What this repo is
HercUNet: a sheet-separation **refiner** for carbonised-papyrus µCT — a stock `nnunetv2==2.8.1` (pinned,
unmodified) wrapped by the `hercunet` CLI. It has three stages, each with a guide under `docs/`:
- **labels** — stage-1 pseudo-label generation (`hercunet labels …`; `docs/herculabels.md`),
- **train** — the stage-2 refiner chain (`hercunet train …`; `docs/training.md`),
- **infer** — stage-3 full-volume inference (`hercunet infer …`; `docs/infer.md`),
plus the streaming **data layer** (`docs/data-layer.md`) and the sheet-segmentation decoder
(`--meshes`; `docs/tracer-sheet-decoder.md`). Install & CLI overview: `docs/README.md`.

## Hard constraints (non-negotiable)
1. **No environment variables as user-facing configuration.** Every option a user controls is a `--flag`.
   Thread/worker caps, paths, and knobs are flags (or derived automatically), never env vars. (Internal,
   non-UX env like the launcher setting `nnUNet_compile=0` is fine; never expose config through the environment.)
2. **Hyphens, never underscores, for file names and CLI flags.** Docs are `some-topic.md` (not `some_topic.md`);
   flags are `--pretrained-hf`, `--no-meshes`, `--sheet-preds` (not `--pretrained_hf`). Python identifiers use
   underscores as normal — this rule is about *user-facing names*.
3. **British spelling** in prose, comments, and docs (`parallelise`, `initialise`, `colour`).

## Everything is already on Hugging Face — do NOT recreate it
Public, no token. Rebuilding/mining/re-uploading any of this by hand is wasted work — pull it:
- **`jimmylomro/hercunet-corpus`** (dataset) — the full training corpus: the 4031 herculabels `sample_*.npz`,
  the **pre-built `m7_corpus/`** (nnU-Net tif cases), and `corpus_index/` (window lists + m7 manifest). Get it
  with `hf download jimmylomro/hercunet-corpus --repo-type dataset --local-dir <dir>`, then feed
  `--corpus <dir> --m7-corpus <dir>/m7_corpus`. **Never scp the corpus or re-run `labels m7-mine` — it's here.**
- **`jimmylomro/hercunet-v0`** (model) — the HercUNet v0 refiner checkpoint + plans. Auto-pulls via
  `--pretrained-hf jimmylomro/hercunet-v0` (train warm-start) and `--model-hf jimmylomro/hercunet-v0` (infer).
- **`scrollprize/surface_m7_nnunet`** (model) — the m7 plans the `preprocess` step transplants (pulled
  automatically) and the default m7 warm-start source.

Only build a corpus from scratch (`labels create` / `labels m7-mine`) if you deliberately need a *new* one;
to reproduce HercUNet v0, download the published corpus (`docs/training.md`, Option A).

## How to use the CLI (the happy path)
- **Install:** a venv (or `pipenv`) + the right extra — `pip install -e ".[train]"` (training),
  `".[infer]"` (inference), `".[viz]"` (label viewer). On a GPU pod, reuse the preinstalled CUDA torch with a
  `python -m venv --system-site-packages` venv so the install doesn't pull a mismatched torch.
- **Training is ONE command — the chain does everything** (export-labels → preprocess → export-owner → fit):
  ```
  hercunet train chain --nnunet-path <root> --corpus <dir> --m7-corpus <dir>/m7_corpus \
    --out <root>/nnUNet_raw/DatasetNNN_… --dataset NNN --pretrained-hf jimmylomro/hercunet-v0 [--meshes]
  ```
  `--meshes` adds the sheet-segmentation decoder (supervised on herculabels `owner` only; m7 cases train the
  surface/affinity heads). Batch size lives in the nnU-Net **plan** (the standard value); don't hardcode a lower
  one — size the GPU to the model.
- **Inference:** `hercunet infer single-instance <scroll|zarr-url> --model-hf … --out … [--region …]`.

## Development — fully integrated, never bolted-on
- **Read the relevant `docs/` guide before building.** The pipeline, formats, and conventions are documented;
  align to them. (And re-read the ⛔ warning at the top — reuse before you rewrite.)
- **Extend the existing machinery; do not write standalone/janky scripts or patch on top.** Concretely:
  - a new trainer **subclasses** the existing one (inherit the owner-aware dataloader, MALIS, DAgger,
    prev-channel, warm-start, and the per-term logging); register it as an **alias** in `train/launch.py`.
  - new pipeline work goes **into the chain** (`train/dataset.py`, the chain steps), not around it.
  - hyperparameters go in the **recipe** (`training/recipe.py`); new loss terms fold into the existing
    `_acc` / per-term logging, in the same single-line style.
  - reuse the **data layer** (`data/`, streaming/caching/prefetch) and the existing **parallelism**
    (e.g. `nshards` process-sharding for export; `ThreadPoolExecutor` where the heavy call releases the GIL).
    Size workers to the **cgroup-allocated vCPUs**, not `nproc` (on cloud pods `nproc` reports the host, not the
    allocation).
- **Warm-start is per-parameter and smart** (`load_pretrained`): it loads matching weights directly and
  stem-expands a narrower input stem — so a new head initialises fresh and a HercUNet checkpoint warm-starts
  cleanly. Build new heads to be additive (keep the existing outputs unchanged).
- **Tests:** add unit / byte-equality tests under `tests/`; keep them CPU-runnable where possible.
- **Running jobs:** launch long jobs detached with an unbuffered log file (`python -u … > …log 2>&1`), don't
  block-polling; the `train chain` logs every stage (export → preprocess → fit) to one stream.

## Layout
`hercunet/{labels,train,training,infer,data,tracer,unwrap,viz,hercuvit}` · `docs/` (per-stage guides) · `tests/` ·
`submission/` (writeup). The CLI is `hercunet` (`python -m hercunet …`), dispatched in `hercunet/cli.py`;
HercuViT adds the parallel `hercuvit` CLI (`python -m hercunet.hercuvit …`).

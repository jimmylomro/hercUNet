# HercUNet — Stage 3: full-volume inference (`hercunet infer`)

Run the trained HercUNet refiner over a whole scroll (or a sub-cube) and write a per-pass **surface-probability
OME-Zarr** you can open in VC3D. This is the stage-2 model (`hercunet train`) put to work; see
[training.md](training.md) for how the model is made and [README.md](README.md) for install / GPU / CUDA.

> **Needs a CUDA GPU.** Inference builds the 8-channel network and runs it over millions of windows — GPU-only in
> practice. Everything scales with how many GPUs you give it.

> ⚠️ **Storage — each pass buffer is large.** Every pass writes a **full-resolution uint8 surface-probability
> OME-Zarr**. For a whole scroll that is **hundreds of GB — up to ~500 GB per pass** (it scales with the scroll's
> material volume; only non-air chunks are written). By default buffers are pruned to the last two, so budget for
> **~2 buffers on disk at once** (a third may briefly exist while one finalises) — i.e. **~1 TB free for a big
> scroll**. `--keep-buffers` keeps **every** pass (≈ `passes ×` a buffer). **Point `--out` at a disk with that
> much free space** — otherwise a pass dies partway and the time is lost. Each run prints its own upper-bound
> estimate at startup (`[jacobi] STORAGE: up to ~NNN GB per pass buffer …`); a `--region` smoke test is tiny.

---

## What it does (in one paragraph)

HercUNet inference is **iterative**, not one-shot. It runs *N* double-buffered **Jacobi passes** over the volume:
pass 0 bootstraps from CT alone (`prev = 0`); every later pass reads the **previous pass's whole
surface-probability volume** back in as an extra input channel (`prev`) and refines it. Within a pass, windows are
merged with the **villa Gaussian blend** (overlap `0.25`) so there are no grid seams, and the window grid shifts
by half a stride between passes so each pass heals the last one's seams. Information propagates ~one window per
pass, so a few passes (default **4**) let structure knit across window boundaries. The **final pass is the
deliverable** (earlier passes are pruned unless you ask to keep them).

The network is built **stock** from the model's `plans.json` (`get_network_from_plans`) and loaded from
`network_weights` — so **only a pinned `nnunetv2==2.8.1` is required**, not the training code. Note that stock
`nnUNetv2_predict` will **not** work: the 8-channel `[CT, prev, orientation×6]` stem and the iteration are
HercUNet-specific.

---

## Install

```bash
pip install ".[infer]"      # nnunetv2==2.8.1 + huggingface_hub, on top of the core data layer
```

The `[infer]` extra is much lighter than `[train]` (no MALIS / owner / DAgger machinery). `torch`, `zarr`,
`scipy`, `numpy` are already core deps. Notes:

- **Reuse an existing CUDA `torch`** (common on GPU pods): install into a venv made with `--system-site-packages`
  so the pod's torch is inherited rather than re-downloaded (this also sidesteps a PEP-668 "externally managed"
  system pip):
  ```bash
  python3 -m venv --system-site-packages .venv && .venv/bin/pip install ".[infer]"
  ```
- **`-e`** (editable) is only for contributors editing the code; a plain `pip install ".[infer]"` is the norm.
- **S3 upload** (`--s3-prefix`) uses `s5cmd` for fast parallel uploads (a zarr is 100k+ tiny objects — s5cmd's Go
  concurrency beats boto/s3fs by a wide margin). The `[infer]` extra installs it for you (a PyPI wheel that bundles
  the `s5cmd` binary on `PATH` — no Go toolchain), so you only add `AWS_ACCESS_KEY_ID`/`SECRET` (and `AWS_REGION`)
  to the env. It's optional — omit `--s3-prefix` to keep everything local. When set, the run does a **write
  preflight** (a tiny write+delete round-trip to the prefix) **before any inference starts** and aborts
  immediately with a clear message if the creds can't write there — so a multi-hour pass is never lost to a
  permission error at upload time. Uploads print a **throttled progress line** every few seconds
  (`[upload] cp: 12000/470000 files (2%) 850 files/s 14s`), not one line per chunk file.

## Getting the model

The published HercUNet v0 model is **public on Hugging Face** (no token):

- **`--model-hf jimmylomro/hercunet-v0`** — downloads the model folder (`plans.json`, `dataset.json`,
  `dataset_fingerprint.json`, `fold_0/checkpoint_best.pth`) via `snapshot_download` and uses it. The bare flag
  `--model-hf` defaults to this repo.
- **`--model <folder>`** — point at a local model folder instead (e.g. a run you trained yourself; the folder
  `hercunet train fit` writes under `$nnUNet_results/DatasetNNN_…/`).

---

## The two commands

Inference is split by *where it runs*, because that's the only thing that really differs. Both drive the **same
engine** (the iterative Jacobi refiner + an elastic claim-queue); the split is about GPUs vs pods.

### `hercunet infer single-instance` — one box

Runs on one machine and **fans across its local GPUs automatically** — one worker process per GPU, coordinated on
the local filesystem. You run **one command**; you don't launch a process per GPU by hand.

```bash
hercunet infer single-instance \
  --model-hf jimmylomro/hercunet-v0 \
  --scroll PHerc1447 \
  --out /workspace/preds/1447_run301 \
  --passes 4                      # → /workspace/preds/1447_run301_pass{0..3}.zarr
```

- `--gpus all` (default) uses every visible GPU; `--gpus 0,1` restricts. With a single GPU it runs in-process
  (no claim-queue overhead).

### `hercunet infer multi-instance` — many pods

For scaling across **separate machines** sharing a **network volume**. You run the **same command on each pod**
(that part is unavoidable across machines), with `--out` on the shared storage and `--leader` on **exactly one**
pod. Each pod *also* fans across its own local GPUs, so exactly one worker across all pods × GPUs is the global
leader (it creates each pass's zarr, builds the pyramid, uploads).

```bash
# on the LEADER pod:
hercunet infer multi-instance --leader --model-hf jimmylomro/hercunet-v0 \
  --scroll PHerc1447 --out /mnt/shared/preds/1447_run301 --passes 4 --s3-prefix s3://…/1447

# on every FOLLOWER pod (same command, no --leader):
hercunet infer multi-instance          --model-hf jimmylomro/hercunet-v0 \
  --scroll PHerc1447 --out /mnt/shared/preds/1447_run301 --passes 4
```

Workers steal work dynamically (a faster GPU does more); a hard barrier between passes guarantees pass *p* is
fully written before pass *p+1* reads it. `--reclaim` re-queues work orphaned by a crashed worker. Extra pods can
join mid-run.

---

## Smoke test on a small cube

Use `--region z0:z1,y0:y1,x0:x1` to run just a sub-cube (the buffer canvas stays full-volume; only that box is
computed). Good for a first check on a fresh pod:

```bash
hercunet infer single-instance --model-hf jimmylomro/hercunet-v0 \
  --scroll PHerc1447 --region 6000:6384,3000:3384,3000:3384 \
  --out /workspace/preds/smoke --passes 2 --keep-buffers --finalise all
```

Expect: pass 0 already detects sheets from CT alone; pass 1 visibly improves (iteration is the whole point). Open
`smoke_pass0.zarr` / `smoke_pass1.zarr` in VC3D and compare. (`--finalise all` gives both passes a pyramid here;
the default `last` would pyramid only pass 1 — fine for a full run where only the deliverable matters.)

---

## Full-scroll run

The published full-scroll runs used **overlap 0.25, 4 passes**, with **pass 3 as the deliverable** (a full
PHerc1447 run is ~12–14 GPU-hours on one GPU; multi-GPU / multi-pod scales it down near-linearly). Defaults match
that, so a plain `--passes 4` reproduces the recipe. Add `--s3-prefix s3://…` to stream each finished pass to S3
for VC3D.

> ⚠️ **Check free space first.** A full-scroll pass buffer is **hundreds of GB (up to ~500 GB)**; with the default
> prune-to-last-two you need **~2 buffers (~1 TB) free** where `--out` lives. `--keep-buffers` needs `passes ×` a
> buffer. The run prints an upper-bound estimate at startup — read it before walking away.

---

## Output

Each pass writes `{out}_pass{p}.zarr` — a **VC3D-friendly OME-Zarr v2** surface-probability volume (uint8, level
"0" + a max-pooled pyramid, plus `meta.json`). Max-pooling (not averaging) preserves the thin surface at low
resolution. By default the pyramid is built **only for the last pass** (the deliverable) — earlier passes are
written at L0 only, which VC3D can still view. `--finalise all` builds it for every pass, `--finalise no` for none
(build later on a CPU box), or pass an index list like `--finalise 0,3`.

**Which passes upload** is controlled separately by `--upload`, with the **same grammar and default** as
`--finalise` (`last` / `all` / `no` / index list). So `--s3-prefix …` alone uploads only the final pass; add
`--upload all` to stream every pass to S3 (e.g. to watch each one in VC3D as it lands). For an uploaded pass, L0
goes up first (directly viewable) and then, if that pass was also finalised, the pyramid levels.

## Resume

`--resume` skips any pass already marked `_complete` (a `{out}_pass{p}.zarr.done/_complete` marker), so an
interrupted multi-pass run continues from where it stopped without recomputing finished passes.

---

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--model-hf [REPO]` | `jimmylomro/hercunet-v0` | download the model from HF (public); bare flag = HercUNet v0 |
| `--model DIR` | — | use a local model folder instead |
| `--scroll` | (required) | scroll id / name (resolved against the data-layer catalog) |
| `--out PREFIX` | (required) | writes `{PREFIX}_pass{p}.zarr` |
| `--passes N` | `4` | Jacobi passes; final pass = deliverable |
| `--overlap f` | `0.25` | Gaussian-blend window overlap (HercUNet v0); `0` = disjoint raw-write mode |
| `--region z0:z1,y0:y1,x0:x1` | **whole scroll** | restrict to a sub-cube (smoke tests); omit to infer the entire volume |
| `--local-vol PATH` | — | read a local OME-Zarr copy instead of streaming S3 (I/O-bound → GPU-bound; byte-identical) |
| `--gpus all\|0,1,3` | `all` | local GPUs to use (one worker process each) |
| `--leader` | (multi only) | this pod creates/finalises/uploads — exactly one pod |
| `--plain` | off | 2-channel `[CT, prev]` model instead of the 8-channel affinity model |
| `--batch` / `--nb` | `4` / `4` | forward batch size / windows-per-block per axis |
| `--air` | `25` | skip windows whose CT max is below this (all-air) |
| `--s3-prefix s3://…` | — | upload finished passes with s5cmd (AWS creds in env; write-preflighted; throttled progress logs) |
| `--upload last\|all\|no\|0,2,3` | `last` | which passes upload to `--s3-prefix` (same grammar as `--finalise`); inert without `--s3-prefix` |
| `--finalise last\|all\|no\|0,2,3` | `last` | which passes get an OME-Zarr pyramid (default: only the last/deliverable) |
| `--resume` | off | skip passes already `_complete` |
| `--keep-buffers` | off | keep every pass buffer (default prunes to the last two) — ⚠️ `passes ×` a ~500 GB buffer |
| `--reclaim` | off | re-queue work orphaned by a crashed worker |

**Local volume shortcut:** pass `--local-vol /path/to/scroll.zarr` to read a locally-downloaded copy of the scroll
instead of streaming from S3 — turns an I/O-bound run into a GPU-bound one (decompressed chunks are byte-identical,
so results match the S3 path exactly). Only L0 need be downloaded.

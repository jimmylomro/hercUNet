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

## The CT source (positional `SOURCE`)

Inference takes the CT volume as a **single positional argument** — there is no `--scroll` flag and no implicit
bucket listing unless you ask for one:

| `SOURCE` | How it is read |
|---|---|
| `s3://bucket/key.zarr` | **streamed** over HTTPS range reads — no bucket listing |
| `https://…/x.zarr` | **streamed** (already an endpoint) |
| `file:///abs/x.zarr` or a local path / `*.zarr` | **read locally** (fast NVMe/tmpfs → GPU-bound) |
| `PHerc1447` (a bare scroll id) | resolved through the data-layer **catalog** (this lists the open-data bucket to find the finest aligned volume; the URL forms above skip that) |

**Remote → local, automatically — `--pre-sync-source`.** Streaming a scroll chunk-by-chunk from S3 is latency-bound
and slow (a single slab can take hours). For an `s3://` / `https://` source, add **`--pre-sync-source`** and the run
**region-downloads only the L0 chunks covering `--region`** (plus a small halo for the patch overshoot) to a local
copy with `s5cmd`, then reads from there — turning the run GPU-bound **without fetching the whole (hundreds-of-GB)
volume**. zarr serves any not-downloaded (air) chunk as its fill value, so a region-only copy is exact for the
region you compute.

- **Streaming is anonymous HTTPS** (the reader does not sign requests) — fine for the public open-data bucket; a
  streamed run prints a **warning** recommending `--pre-sync-source`, because per-chunk reads are latency-bound and
  usually dominate runtime. (A private bucket cannot be streamed; localise it with `--pre-sync-source`, which uses
  your AWS creds.)
- Bare `--pre-sync-source` syncs next to `--out`; `--pre-sync-source /mnt/nvme/ct` chooses the dir (the copy keeps
  the source zarr's basename so its voxel-size name token survives).
- The public open-data bucket is fetched **anonymously**; a private bucket uses your AWS env creds.
- It is **ignored** for a `file://` / local source (already local), and needs `s5cmd` (bundled in the `[infer]`
  extra). With no `--region` it syncs the **whole** L0.
- For **multi-instance** across pods, pass an explicit **pod-local** dir so each pod syncs its own copy — do not
  point it at the shared `--out` volume.

## Resolution — run at the model's scale (`--level`)

HercUNet v0 was trained **only** on the native-coarse **~7.5–9.5 µm** grand-prize domain (see the corpus). Most
prize scrolls ship an ~8.6/9.4 µm masked volume, so the default **`--level 0`** is correct for them. But some
scrolls were **rescanned finer** (2.4 µm, 1.1 µm). Running the model at a fine level is **out of distribution** —
a 1.5 mm window would cover a fraction of what the model expects — and the predictions are unreliable.

`--level L` picks the **OME-Zarr pyramid level** to run on. Each level is ~2× coarser, so for a fine rescan choose
the level nearest ~8.6 µm:

| source voxel | run with | effective voxel |
|---|---|---|
| 8.64 / 9.36 µm (most scrolls) | `--level 0` (default) | 8.6 / 9.4 µm |
| 2.4 µm rescan | `--level 2` | ~9.6 µm |
| 1.1 µm rescan | `--level 3` | ~9.0 µm |

If the chosen level's voxel size lands **outside ~[7.5, 10] µm**, inference prints a big, unmissable warning and
names the closest in-band level — the run still proceeds (you may intend it), but heed it. The whole engine then
operates on that level's grid: the output surface zarr is written at the level's resolution (correct `voxelsize`
for VC3D), `--region` coordinates are in the level's voxels, and `--pre-sync-source` fetches that level's chunks.

## Simplest run (local, nothing else needed)

No local data, no S3, no config — pull the model from Hugging Face, stream a small PHerc1447 window straight from
the open-data bucket (the `PHerc1447` catalog form), and write the result beside you:

```bash
# install (needs an NVIDIA GPU) — pick one:
python -m venv .venv && . .venv/bin/activate && pip install ".[infer]"   # venv
# or:  pipenv install ".[infer]"   → then prefix the command below with `pipenv run`

hercunet infer single-instance \
  --model-hf jimmylomro/hercunet-v0 \
  PHerc1447 \
  --region 10889:11401,2848:3360,3915:4427 \
  --out ./infer-demo --passes 3
```

- **`SOURCE`** here is the bare scroll id `PHerc1447`; for a fast slab prefer the direct zarr URL + `--pre-sync-source`
  (see above and the full-scroll section).
- **`--region`** is a ~5 mm cube (fast); drop it to run the whole volume (see the storage warning above).
- Writes **`./infer-demo_pass2.zarr`** — the last pass is the deliverable; open it in VC3D.
- Uses **every visible GPU** automatically. Add **`--keep-buffers`** to keep all three passes and watch the
  iteration improve pass-to-pass, or **`--s3-prefix s3://…`** to upload instead of keeping it local.

## The two commands

Inference is split by *where it runs*, because that's the only thing that really differs. Both drive the **same
engine** (the iterative Jacobi refiner + an elastic claim-queue); the split is about GPUs vs pods.

### `hercunet infer single-instance` — one box

Runs on one machine and **fans across its local GPUs automatically** — one worker process per GPU, coordinated on
the local filesystem. You run **one command**; you don't launch a process per GPU by hand.

```bash
hercunet infer single-instance \
  --model-hf jimmylomro/hercunet-v0 \
  s3://vesuvius-challenge-open-data/PHerc1447/volumes/20250521151220-8.640um-1.2m-116keV-masked.zarr \
  --pre-sync-source \             # region-download the L0 chunks locally first → GPU-bound (not S3-bound)
  --region 5300:5600,2820:5508,1920:4448 \
  --out /workspace/preds/1447_run301 \
  --passes 4                      # → /workspace/preds/1447_run301_pass{0..3}.zarr
```

- `--gpus all` (default) uses every visible GPU; `--gpus 0,1` restricts. With a single GPU it runs in-process
  (no claim-queue overhead).
- The `SOURCE` here is the direct S3 zarr URL (no bucket listing); `--pre-sync-source` makes the slab read from a
  local copy. Drop `--pre-sync-source` to stream (slow), or pass `PHerc1447` to let the catalog pick the volume.

### `hercunet infer multi-instance` — many pods

For scaling across **separate machines** sharing a **network volume**. You run the **same command on each pod**
(that part is unavoidable across machines), with `--out` on the shared storage and `--leader` on **exactly one**
pod. Each pod *also* fans across its own local GPUs, so exactly one worker across all pods × GPUs is the global
leader (it creates each pass's zarr, builds the pyramid, uploads).

```bash
# on the LEADER pod:
hercunet infer multi-instance --leader --model-hf jimmylomro/hercunet-v0 \
  s3://vesuvius-challenge-open-data/PHerc1447/volumes/20250521151220-8.640um-1.2m-116keV-masked.zarr \
  --out /mnt/shared/preds/1447_run301 --passes 4 --s3-prefix s3://…/1447

# on every FOLLOWER pod (same command, no --leader):
hercunet infer multi-instance          --model-hf jimmylomro/hercunet-v0 \
  s3://vesuvius-challenge-open-data/PHerc1447/volumes/20250521151220-8.640um-1.2m-116keV-masked.zarr \
  --out /mnt/shared/preds/1447_run301 --passes 4
```

Workers steal work dynamically (a faster GPU does more); a hard barrier between passes guarantees pass *p* is
fully written before pass *p+1* reads it. `--reclaim` re-queues work orphaned by a crashed worker. Extra pods can
join mid-run. The `SOURCE` is positional (same on every pod); to go local, give each pod a **pod-local**
`--pre-sync-source /local/dir` (each pod syncs its own copy — never the shared `--out` volume).

---

## Smoke test on a small cube

Use `--region z0:z1,y0:y1,x0:x1` to run just a sub-cube (the buffer canvas stays full-volume; only that box is
computed). Good for a first check on a fresh pod:

```bash
hercunet infer single-instance --model-hf jimmylomro/hercunet-v0 \
  PHerc1447 --region 6000:6384,3000:3384,3000:3384 \
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

## Test-time augmentation (`--tta`)

Off by default. When on, each window is run under several flips/rotations, the results un-augmented, and the
**logits averaged** (villa's `tta.py` scheme) — trading compute for a smoother, less orientation-biased prediction.
It applies **per window**, independently. Two families, combinable:

- **mirroring** — `torch.flip` over the chosen axes: `--tta mirror` (all of z,y,x → 8 variants) or a subset like
  `--tta z,y`.
- **rotation** — axis-swap transposes: `--tta rotate` (2 extra variants).
- `--tta all` = both (10 variants); `--tta none` = off.

`--tta-passes` picks which Jacobi passes get it (same `all`/`last`/`no`/index-list grammar as `--finalise`), e.g.
`--tta all --tta-passes 2,3` to augment only the last two passes. **Cost:** N variants ⇒ **N× forwards** on each
augmented pass, so `--tta all --tta-passes last` (10× on just the deliverable) is usually the sensible knob. The
8-channel orientation stays correct automatically — it's recomputed in-forward from the already-flipped CT.

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
| `SOURCE` (positional) | (required) | CT OME-Zarr: `s3://…zarr` / `https://…zarr` (streamed), `file://…zarr` or a local path (read locally), or a bare scroll id (catalog lookup) |
| `--model-hf [REPO]` | `jimmylomro/hercunet-v0` | download the model from HF (public); bare flag = HercUNet v0 |
| `--model DIR` | — | use a local model folder instead |
| `--out PREFIX` | (required) | writes `{PREFIX}_pass{p}.zarr` |
| `--pre-sync-source [DIR]` | — | for an `s3://`/`https://` SOURCE: region-download the covering L0 chunks locally (s5cmd) and read from there (→ GPU-bound). Bare = sync next to `--out`; `DIR` = choose where. Ignored for a local source |
| `--passes N` | `4` | Jacobi passes; final pass = deliverable |
| `--overlap f` | `0.25` | Gaussian-blend window overlap (HercUNet v0); `0` = disjoint raw-write mode |
| `--tta none\|all\|mirror\|rotate\|z,y,x` | `none` | test-time augmentation per window (villa-style, logit average); N variants → N× forwards |
| `--tta-passes all\|last\|no\|2,3` | `all` | which passes get TTA (same grammar as `--finalise`); ignored when `--tta none` |
| `--level L` | `0` | OME-Zarr pyramid level to run on (0 = full res). Pick the level nearest **~8.6 µm** for a finer rescan (a 2.4 µm scroll → `--level 2` = 9.6 µm). A **big warning** prints if the level's voxel size is outside **~[7.5, 10] µm** (the model's trained band). `--region` and the output zarr are then on this level's grid |
| `--region z0:z1,y0:y1,x0:x1` | **whole volume** | restrict to a sub-cube (smoke tests); omit to infer the entire volume. Coordinates are in the **working level's** voxels (= L0 at `--level 0`) |
| `--gpus all\|0,1,3` | `all` | local GPUs to use (one worker process each) |
| `--leader` | (multi only) | this pod creates/finalises/uploads — exactly one pod |
| `--plain` | off | 2-channel `[CT, prev]` model instead of the 8-channel affinity model |
| `--batch` / `--nb` | `4` / `4` | forward batch size / windows-per-block per axis |
| `--prefetch-workers N` | `3` | threads that assemble upcoming batches (CT normalise + prev-gather + stack) and read CT blocks **while the GPU runs the current one** — overlaps CPU/IO with compute so the cards don't stall between tiles. Raise on many-core boxes; `8` is a good start |
| `--readahead N` | `2` | how many batches/blocks to keep assembled in flight (queue depth). Deeper absorbs IO jitter at the cost of RAM; pair with `--prefetch-workers` (e.g. `6`) |
| `--compile` | off | `torch.compile` the network (lossless, ~1.3–1.4× on real GPUs). May stall on first-run autotuning on some arches; keep `--batch 4` |
| `--air` | `25` | skip windows whose CT max is below this (all-air) |
| `--s3-prefix s3://…` | — | upload finished passes with s5cmd (AWS creds in env; write-preflighted; throttled progress logs) |
| `--upload last\|all\|no\|0,2,3` | `last` | which passes upload to `--s3-prefix` (same grammar as `--finalise`); inert without `--s3-prefix` |
| `--finalise last\|all\|no\|0,2,3` | `last` | which passes get an OME-Zarr pyramid (default: only the last/deliverable) |
| `--resume` | off | skip passes already `_complete` |
| `--keep-buffers` | off | keep every pass buffer (default prunes to the last two) — ⚠️ `passes ×` a ~500 GB buffer |
| `--keep-affinity` | off | also save (and, with `--s3-prefix`, upload) the affinity head of the final pass as `{PREFIX}_pass{last}_aff.zarr` (4-D `(n_aff,Z,Y,X)` uint8); needs the affinity model + `overlap>0` |
| `--reclaim` | off | re-queue work orphaned by a crashed worker |

**Going local (fast):** the old `--local-vol` flag is gone — pass the local copy **as the positional `SOURCE`**
(`file:///path/x.zarr` or just the path). To localise an `s3://` run without pre-downloading by hand, add
`--pre-sync-source`: it region-downloads the covering L0 chunks with `s5cmd` and reads from the copy (decompressed
chunks are byte-identical, so results match the streamed path exactly). Only L0 is fetched, and only the chunks the
`--region` touches. See **[The CT source](#the-ct-source-positional-source)** above.

**Keeping the GPUs fed.** Each pass assembles its batches (CT normalise + prev-pass gather + stack) and reads the
CT/prev slabs on a small thread pool that runs *ahead* of the GPU, so compute overlaps the CPU/IO instead of
stalling on it between tiles. If `nvidia-smi` shows utilisation sawtoothing 100%→0% — typically on a many-core box
where the defaults (`--prefetch-workers 3 --readahead 2`) can't keep the queue full — raise them; `--prefetch-workers 8
--readahead 6` is a good fat-box setting (measured +~26% on pass 0 and more on later passes, where the dense
prev-slab read is also hidden). Within a batch the work is memory-bandwidth-bound, so `--batch` barely changes
throughput — leave it at `4`. These knobs only reorder work: results are byte-identical.

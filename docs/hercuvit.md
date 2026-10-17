# HercuViT — implementation spec

*A full-resolution sheet-segmentation network built on the frozen Dinovol encoder, with a windowed-attention
retrofit, a skip-connection pixel decoder, and Mask2Former-style instance heads. Parallel tooling to
`hercunet`, sharing its data layer, corpus, Hugging Face plumbing and conventions.*

**Status:** spec (Part I authoritative). Part II is filled by the implementation; Part III tracks TODOs and the
conflict log. Branch `hercuvit`, cut from `main`.

Design source: the vesuvius-repo design doc `docs/dinovol_sheet_architecture.md` (Variant A / Variant B). This
document is the hercunet-private **implementation** of it, and overrides the design doc wherever Dinovol's real
config differs — **except** where that would change one of the three architectural pillars (see §1).

---

## 0. Scope

- **v1 = supervised SHEET segmentation only**: a semantic sheet-surface head *and* a Mask2Former-style instance
  (sheet-ID) head, trained on the herculabels corpus.
- **Variant A first** (retrofit the frozen Dinovol encoder with windowed attention; no distillation). Variant B
  (distilled slim transformer) is documented but deferred.
- **Variant naming (first-class across recipe / CLI / docs):** `hercuvit-small` = Variant A (the BUILT ~27 M
  full-weight windowed retrofit), `hercuvit-tiny` = Variant B (~13–14 M, distilled), `hercuvit-nano` = Variant C
  (~6–8 M, distilled). All three build through the **one** `build_hercuvit(ckpt, arch_override=…)` entry; the
  per-variant arch numbers are the single source of truth in `model/hercuvit_net.py:VARIANT_ARCH`. `small` is the
  trained v1 deliverable; `tiny`/`nano` are distilled from a trained `small` teacher via `hercuvit distill`
  (`model/distill.py`, `train/distill.py`, lead-owned) before supervised training.
- **Fibre and ink are future**: Dinovol was SSL-supervised for them too, but we segment sheets now. The point /
  ink / region heads are **wired but inactive** (stubbed with Dinovol's `dense_head` dims) so they can be
  supervised later without re-architecting.
- Out of scope v1: cross-tile sheet-ID reconciliation (the stitcher, design §10/§12); ResEnc-L logit
  distillation; Variant B.

---

## 1. Architectural pillars (TOP PRIORITY — never silently changed)

HercuViT's architecture is the product; Dinovol is the initialisation. These three pillars are fixed. If
resolving a Dinovol conflict would require changing any of them, **STOP and raise it with the owner before
writing code** — log it in §12 (Conflict log).

1. **Windowed attention retrofit** (design §5.1). 10 Dinovol blocks; a subset keep global attention, the rest
   use non-overlapping windows of 8³ tokens (= 2×2×2 = 8 windows over the 16³ grid). No shifted windows;
   interleaved global blocks propagate across windows (ViTDet).
2. **Pixel decoder with skip connections** (design §6). A lean UNet decoder consuming the Dinovol stem's real
   multi-resolution skips (s0 1/1, s1 1/2, s2 1/4, s3 1/8), additive skips, one conv per level, producing
   full-resolution semantic logits **and** 32-d mask features.
3. **Mask2Former-style heads** (design §7). Instance masks as the dot product between per-query 32-d mask
   embeddings and the shared full-resolution 32-d mask features; anchored two-stage query selection; DETR set
   prediction with Hungarian matching.

---

## 2. Dinovol config overrides (authoritative for encoder internals)

Read verbatim from the released checkpoint's own `config` (`scrollprize/dinovol_ps8_9.6um_101626`,
`teacher_305k.pt`, step 305000). These override the design doc **because they do not touch the pillars** — they
are how the pretrained weights must be instantiated to load and behave correctly.

| Property | Design doc said | Dinovol checkpoint (authoritative) |
|---|---|---|
| RoPE | axial | **mixed** (`rope_type: mixed`, learned `mix_frequencies (6,24,3)`, `base 100.0`, `normalize_coords: separate`) |
| Special tokens | `cls_token` only | **4 register tokens + 1 cls** (`num_reg_tokens: 4`) → carry all 5 through windowed blocks, mix only in global |
| embed_dim / depth / heads | 288 / 10 / 6 | 288 / 10 / 6 (head dim 48) ✓ |
| patch / crop | 8 / 128³ | 8 / 128³ ✓ |
| FFN | SwiGLU 2048 | SwiGLU hidden 2048 (`mlp_ratio 7.111…`, `fc1 4096×288` gate+value packed), `swiglu_mlp`, `scale_mlp: false` |
| Attn | — | `qkv_bias: false`, `qkv_fused: true`, `attention_type: full` (retrofit changes grouping only), `attention_backend: cudnn` |
| Regularisation | — | `drop_path_rate: 0.3` (not uniform), LayerScale `init_values: 1e-5` |
| Norms | — | `untie_cls_and_patch_norms: false`, `untie_global_and_local_cls_norm: true` |
| Stem | 16→48→128→192→288 | `deeper_embed_stage_channels [16,48,128,192]`, `strides [1,2,2,2]`, `depth_per_level [1,2,2,1]`, `stem_channels 16`, `stem_type conv`, `final_projection true`, `channels_last true` |
| Input | — | **1 channel** (`input_channels: 1`) — HercuViT feeds single-channel Dinovol-normalised CT, NOT hercunet's `[ct, prev, orient(6)]` stack |
| Discarded SSL head | — | `dense_head`: hidden 256, out 64, ink 32, region 128 — **the fibre/ink/region hook** (§11), stubbed now |

**Conflict scan result (see §1):** mixed RoPE, register tokens, SwiGLU/LayerScale, qkv-fused all **coexist**
with the three pillars — windowing changes only which tokens attend together (RoPE applied at absolute 16³
coords; the 5 special tokens skip windowed blocks). No pillar change required. Watch-points that could still
force one are listed in §12.

---

## 3. Package layout (new, parallel to `hercunet` — nothing in `hercunet` is modified except additive hooks)

```
hercunet/hercuvit/
  __init__.py
  cli.py                  # parallel CLI: build_parser()/main(); subparsers + set_defaults(func=) + lazy imports
  __main__.py             # `python -m hercunet.hercuvit`
  dinovol/                # VENDORED (MIT) Dinovol backbone, imports rewritten dinovol_2.* -> hercunet.hercuvit.dinovol.*
    __init__.py
    dinov2_eva.py         # Eva, EvaWithChunking, EvaAttention, EvaBlock, EvaAxialAttention
    patch_encode_decode.py# PatchEmbedDeeper (stem + skips), PatchDecode, conv blocks
    rope.py               # MixedRopePositionEmbedding, RopePositionEmbedding
    point_embeddings.py   # sample_grid_points (for future point/ink heads)
    THIRD_PARTY.md        # MIT attribution to ScrollPrize/villa dinovol
  model/
    encoder.py            # build Eva from the checkpoint config; load teacher_305k.pt backbone.*; expose stem skips + tokens + cls/reg
    windowed.py           # PILLAR 1: windowed-attention retrofit over EvaBlock (absolute mixed-RoPE; 5 special tokens carried; 3 global blocks)
    pixel_decoder.py      # PILLAR 2: lean additive-skip UNet decoder; semantic head + 32-d mask features; deep supervision
    instance_head.py      # PILLAR 3: two-stage query selection + query decoder (deformable cross-attn) + mask dot-product
    heads_future.py       # STUB point/ink/region heads (Dinovol dense_head dims) — inactive, for fibre/ink later
    hercuvit_net.py       # top-level nn.Module: encoder -> windowed transformer -> pixel decoder -> heads; forward() -> dict
  data/
    normalize.py          # Dinovol normalisation (replicated from villa ssl_zarr_dataset) — see TODO
    dataset.py            # herculabels corpus -> 128³ CT @ ~9.6µm (1ch) + semantic target + owner volume; reuses the data layer
  train/
    recipe.py             # @dataclass(frozen=True) HercuvitRecipe + shipped hercuvit.yml; active()/set_active()/load() pattern
    losses.py             # semantic Dice+CE (deep sup); instance Hungarian + mask BCE/Dice (point-sampled) + objectness + aux
    trainer.py            # STANDALONE training loop (NOT nnU-Net); reuses HF pull, warm-start pattern, single-line logging
    chain.py              # one-command: pull-corpus -> pull-dinovol -> build -> fit
  infer/
    loader.py             # load a trained HercuViT for inference (returns the same shape of tuple as load_affinity_net)
    infer.py              # full-volume tiling/blend over 128³ windows; reuses the hercunet infer engine where possible
```

Shared/brought-in modules (pure torch/numpy, copied from branch `sheet-seg-decoder` into `hercunet/tracer/`
on this branch via `git checkout sheet-seg-decoder -- <path>`; additive, does not touch hercunet):
- `tracer/labels.py` — `load_sample`, `rasterise_owner_full`, `has_identity` (herculabels npz -> dense owner + m7 gate).
- `tracer/decoder.py` — reuse **`DeformableCrossAttn`** only (the 3-D `grid_sample` deformable attention); the
  node-ownership `InstanceDecoder` is NOT used (different formulation from the Mask2Former head).

---

## 4. Reuse map (verbatim hercunet entry points — architecture-agnostic, used as-is)

| Need | Reused from | Signature / entry |
|---|---|---|
| Stream CT from scroll/zarr | `hercunet/infer/nnunet_infer.py` | `open_source(source)` → vol with `.meta`, `read_window(level,z0,z1,y0,y1,x0,x1)` |
| Prefetch/iterate windows | `hercunet/data/prefetch.py` | `iter_windows(vol, requests, readahead, workers)`, `VolumePrefetcher` |
| HF model pull (encoder/ckpt) | `hercunet/train/launch.py` / `hercunet/infer/run.py` | `resolve_pretrained(pretrained, pretrained_hf, hf_file)` (hf_hub_download), `resolve_model(model, model_hf)` (snapshot_download) |
| HF corpus pull | CLAUDE.md | `hf download jimmylomro/hercunet-corpus --repo-type dataset --local-dir <dir>` → `--corpus <dir>` |
| Per-param warm-start | pattern from `affinity_malis_iter_dagger.py:load_pretrained` | match-by-name + shape, skip/fresh-init non-matching (here: load `backbone.*` into the vendored Eva) |
| Infer tiling/halo/blend/claim-queue | `hercunet/infer/jacobi_refine.py` | `pass_blend_tiles`, `make_gaussian`, claim-queue; swap the net loader + per-window forward |
| Logging style | `affinity_malis_iter_dagger.py` | `_reset_terms`/`_acc`/single-line `[ep N] loss …` — mirror in `trainer.py` |
| Test patterns | `tests/` | CPU-runnable, `torch.manual_seed(0)`, tiny synthetic tensors, permutation/byte-equality/zero-grad invariants |

**NOT reused:** `ct_orientation` (hercuvit is 1-channel, no orientation stack); the nnU-Net trainer chain
(dinovol is not an nnU-Net — see §7); nnU-Net preprocessing (hercuvit reads the corpus directly).

---

## 5. Encoder (Dinovol, reused) — `model/encoder.py`

- Instantiate the vendored `Eva` with the exact §2 config; load `teacher_305k.pt["teacher"]` → strip the
  `backbone.` prefix → `load_state_dict` into the Eva (strict on backbone; ignore `dense_head.*`, DINO/iBOT
  heads). Verify zero missing/unexpected on the backbone subset in a test.
- Expose: the four stem skips `s0 (16@128³), s1 (48@64³), s2 (128@32³), s3 (192@16³)`; the final 288-d token
  grid `(B,288,16,16,16)`; and the `cls + 4 reg` tokens.
- **Stem trainable at full LR** in supervised training (design §4 caveat: it only ever saw gradient through the
  1/8 tokens); transformer frozen in stage 1, 0.1× LR after unfreeze.

## 6. Pillars — module responsibilities

### 6.1 Windowed attention (`model/windowed.py`) — PILLAR 1
- Wrap the vendored `EvaBlock` so blocks `{4,7,10}` (1-indexed) run **global** attention and the other 7 run
  **windowed** (8³ token windows). Default; the global-block indices are a recipe field.
- Mixed RoPE applied at **absolute** 16³ coordinates in all blocks (no window-local coords).
- The 5 special tokens (cls + 4 reg) **pass through windowed blocks unchanged** and attend in global blocks only.
- Must not alter the qkv-fused / SwiGLU / LayerScale structure (weights must load). If it would → §12 conflict.

### 6.2 Pixel decoder (`model/pixel_decoder.py`) — PILLAR 2
- Levels (design §6): 1/8 project tokens 288→192 + add s3 + conv; 1/4 transposed-conv→128 + add s2 + conv;
  1/2 →48 + add s1 + conv; 1/1 →16 + add s0 + conv. Additive skips, one 3×3×3 conv block per level.
- **Semantic head**: 1×1×1 conv 16 → n_sem_classes (sheet-surface vs background, extensible). Deep supervision
  at 1/2 and 1/4.
- **Mask features**: 1×1×1 conv 16 → 32 at full resolution (shared with the instance head).
- Fallback if semantic parity fails: standard nnU-Net decoder (concat skips, two convs/level) — documented, not default.

### 6.3 Instance head (`model/instance_head.py`) — PILLAR 3
- **Query selection (two-stage):** linear objectness over the 4096 tokens; top-K=64 become queries, each with
  an anchor (token-cell centre, refined per layer) and a content vector (token proj 288→64).
- **Query decoder:** 3 layers, width 64, each = self-attn among queries + **3D multi-scale deformable
  cross-attn** (reuse `tracer/decoder.py:DeformableCrossAttn`) around the anchor over the 1/8 tokens and 1/4
  decoder features + FFN.
- **Per query:** sheet / no-object score, refined anchor, 32-d mask embedding. Mask = `sigmoid(mask_emb ·
  mask_features)` at full resolution.
- **Fallback:** Mask2Former masked attention if 3D deformable kernels are slow/unstable — **raise before
  switching** (§12), as it touches the pillar.

---

## 7. Training (`train/trainer.py`) — standalone, parallel to hercunet

HercuViT is **not** an nnU-Net, so it does not subclass `nnUNetTrainer` (which would force a Dice+CE
deep-supervision loop and nnU-Net preprocessing). Per the owner's "whole new tooling, in parallel" directive it
has its **own** PyTorch loop that reuses hercunet modules (data layer, HF pull, warm-start pattern, logging,
tests). bf16 autocast, AdamW, grad-clip; constant LR with warmup (Dinovol convention) or cosine.

Staged recipe (design §8.2), runnable as one `train chain`:
1. **Semantic.** Freeze transformer; **stem at full LR** + pixel decoder + semantic head. Dice + CE, deep
   supervision. (ResEnc-L logit distillation = future, §13.)
2. **Unfreeze.** All trainable, transformer 0.1× LR, windows ON. Optional feature-consistency loss vs the
   frozen global-attention encoder to limit drift.
3. **Instance.** Add the instance head; joint train, encoder low LR. Losses: Hungarian (mask + class cost);
   mask BCE + Dice on **importance-sampled points** (PointRend / Mask2Former) not all 128³; sheet/no-object;
   token objectness; auxiliary losses after each decoder layer.

Warm-start: encoder from Dinovol (§5). Logging mirrors hercunet's single `[ep N] loss …` line with per-term
accumulation. Long runs launched detached, unbuffered, logged to `.logs/` (CLAUDE.md).

Per-sample gating: train instance losses only on herculabels samples (`has_identity` True); semantic can use
all available sheet-surface labels.

---

## 8. Data (`data/dataset.py`, `data/normalize.py`)

- Corpus: herculabels `sample_*.npz` (pulled from `jimmylomro/hercunet-corpus`). Per sample:
  `load_sample` → CT cube; `rasterise_owner_full` → dense owner `(D,H,W) int32` (0=ignore, id+1=sheet);
  semantic sheet-surface target (derivation to confirm vs `docs/herculabels.md` — §13 TODO).
- Resample each sample to Dinovol's working scale **~9.6 µm, 128³** (corpus is native 7–10 µm; keep the 1–2
  voxel separation intact — §12 watch-point).
- **Single-channel** CT, **Dinovol normalisation** (replicated from villa `ssl_zarr_dataset.py` — §13 TODO),
  NOT the nnU-Net fingerprint percentile norm and NOT orientation channels.
- Augmentation: Dinovol-style geometry (flips, 90° rotations, axis transposes); **no blur** (preserve 1-voxel
  structure); light intensity jitter. Owner + semantic targets follow the same spatial transforms.

---

### 8.1 Distillation (`hercuvit-tiny` / `-nano`) — lead-owned

The slim variants are **distilled** from a trained `hercuvit-small` (or the Dinovol teacher) rather than trained
from scratch: `model/distill.py` + `train/distill.py` (lead-owned) match the student's token / Gram / cls features
to the teacher's on unlabeled crops, then the student is trained supervised through the normal chain. Arch numbers
come from `VARIANT_ARCH` (single source of truth); the recipe carries the distillation knobs (`distill_teacher`,
`distill_student_variant`, `distill_{gram,token,cls}_weight`, `distill_steps`, `distill_lr`) and the CLI exposes
`hercuvit distill --teacher … --student {tiny,nano} --out … [--steps N] [--smoke]` (handler lazy-imports
`train.distill:run`). Smoke-verified by the lead (tiny 0.96→0.43, nano 1.45→1.05, both < 4.2 GB).

## 9. Inference (`infer/loader.py`, `infer/infer.py`)

- Reuse the hercunet tiling/halo/Gaussian-blend/claim-queue engine; add a HercuViT loader returning the same
  tuple shape as `load_affinity_net` and a per-window forward.
- v1 output: full-volume **semantic sheet-surface** (+ optional per-tile instance masks). **No cross-tile
  sheet-ID stitching** in v1 (design §10/§12 → future).

---

## 10. CLI (`hercunet/hercuvit/cli.py`) — parallel to `hercunet`

Console script `hercuvit = "hercunet.hercuvit.cli:main"`; also `python -m hercunet.hercuvit`. Hyphenated flags,
no env-var config, British spelling. Subparsers mirror hercunet's `set_defaults(func=…)` + lazy-import pattern.

- `hercuvit train chain --corpus <dir> | --corpus-hf jimmylomro/hercunet-corpus --dinovol-hf scrollprize/dinovol_ps8_9.6um_101626 --out <dir> [--recipe hercuvit.yml] [--epochs N]`
- `hercuvit train fit … [--continue]`
- `hercuvit infer single-instance <scroll|zarr-url> --model <dir> | --model-hf <repo> --out <prefix> [--region …]`

`--dinovol-hf` default const = `scrollprize/dinovol_ps8_9.6um_101626`.

---

## 11. Future: fibre & ink (wired now, inactive)

Dinovol's discarded `dense_head` (hidden 256, out 64, ink 32, region 128) is the template. `heads_future.py`
stubs: a dense **point** head (out 64) for fibre/sheet identity by embedding similarity; an **ink** prototype
head (dim 32, cross-view/cross-batch cosine to one "ink" prototype); a DetCon-style **region** head (128).
Wired into `hercuvit_net.py` behind recipe flags, trained off in v1. Supervise later with point/InfoNCE/DetCon
on fibre and ink labels (the author's SSL recipe) without re-architecting.

---

## 12. Conflict log (pillars vs Dinovol) — raise IMMEDIATELY, resolve with owner

| # | Potential conflict | Status |
|---|---|---|
| 1 | Windowing `EvaBlock` can't be wrapped without altering qkv-fused/SwiGLU/LayerScale (weights won't load) | OPEN watch-point — not triggered |
| 2 | 3D deformable cross-attn unstable → masked-attention fallback changes the instance-head design | OPEN watch-point — raise before switching |
| 3 | Dinovol 9.6µm scale vs corpus native scale → resampling degrades 1–2-voxel separation (decoder pillar) | OPEN watch-point |
| — | mixed RoPE vs windowing | RESOLVED — compatible (absolute-coord RoPE) |
| — | 4 reg + cls vs windowing | RESOLVED — 5 special tokens carried through windowed blocks |
| 4 | Dinovol config pins the **cuDNN** SDPA backend (`attention_priority`), but cuDNN has **no execution plan** for the windowed-attention shapes on some GPUs (verified failing on an RTX 4060: *"cuDNN Frontend error: No execution plans support the graph"*) | **RESOLVED-via-flag** — `--attention-backend` (recipe `attention_backend`, default `auto`); the plumbing (`netbuild.apply_attention_backend`) sets each attention module's `attention_priority` after build (`None` for auto → torch picks flash/efficient/math). No pillar edit. This is what makes training fit 8 GB (infer ~0.9 GB, train ~5.0 GB bf16 batch 1). |

---

## 13. TODOs (confirm / implement; written here if not yet possible)

- [x] **Exact Dinovol normalisation** — extracted from villa `ssl_zarr_dataset.py` (`_finalize_crop` →
  `self.normalizer.run(crop)` then trilinear resize) + `dataset/normalization.py`. The released checkpoint's
  `config.dataset` has **no** `normalization_scheme` key → villa default **`robust`** (percentile 1–99 clip, then
  median / 1.4826·MAD). Replicated faithfully in `data/normalize.py` (`normalize_robust`); villa's `gaussian_blur`
  is deliberately NOT applied (§8). *Verify:* if a future checkpoint sets `normalization_scheme` explicitly, match
  it there.
- [x] **Semantic sheet-surface target derivation** — confirmed against `docs/herculabels.md`: `V_k` meshes are the
  per-sheet medial surfaces; `rasterise_owner_full` paints them into a thin `owner` volume, so the semantic
  sheet-surface class is exactly `owner > 0` (`data/dataset.py`).
- [x] **Corpus native voxel size + resampling** — corpus npz carries `voxel_um` (7–10 µm) + `shape`/`org`/`scroll`
  per window; **no CT is stored**, so CT is streamed from the scroll via `open_source`. Resampling choice: centre-
  crop the native cube to the `128·9.6 µm` physical FOV, then trilinear-resize to 128³ (owner nearest) — keeps the
  encoder at ~9.6 µm and the 1–2-voxel separation (§12 #3) rather than squashing a whole window to 128³.
- [x] **Bring `tracer/labels.py` + `DeformableCrossAttn`** — present on this branch (`hercunet/tracer/`); used by
  the dataset and the instance head.
- [x] **Attention backend** — `--attention-backend` flag + recipe field + `netbuild.apply_attention_backend`
  (§12 #4). Default `auto`.
- [ ] **Point-sampling approximation** — the instance mask loss uses a cheap importance mix (half GT-sheet voxels,
  half uniform) instead of full PointRend/Mask2Former uncertainty sampling (needs a first forward). Revisit if
  mask quality plateaus.
- [ ] **hercuvit-tiny / -nano** — arch presets live in `VARIANT_ARCH`; the slim nets are **distilled** from a
  trained `hercuvit-small` teacher via `hercuvit distill` (`model/distill.py`, `train/distill.py`, lead-owned),
  then trained supervised through the same chain. Deferred, depends on a validated `small`.
  - Inference note: the self-contained `netbuild.build_from_cfg` (used by `infer/loader.py`, no Dinovol ckpt)
    reconstructs the Eva-key overrides but **not** `global_attention="spatial_reduction"`; rebuild tiny/nano
    inference via the Dinovol ckpt + `netbuild.build` until that path is extended.
- [ ] ResEnc-L logit distillation (semantic stage) — FUTURE (needs a ResEnc-L teacher).
- [ ] Cross-tile sheet-ID stitching — FUTURE (design §10/§12); v1 infer emits the semantic surface only.
- [ ] Fibre/ink head activation — FUTURE (§11).

---

## Part II — Implementation description

Faithful, file-by-file account of what was built around the three pillars. The pillar / `model/` modules and the
vendored `dinovol/` were **read only** (interfaces); the plumbing below is everything else. British spelling,
hyphenated flags, no env-var config throughout. Validated on the vesuvius pipenv (torch 2.6, timm 1.0.22) against
the cached Dinovol checkpoint `/tmp/dino_ck/teacher_305k.pt`.

### Pillar / model modules (read-only — described for completeness)

- `model/encoder.py` — builds the vendored Dinovol `Eva` from the checkpoint's own `config['model']`
  (`build_dinovol_backbone`) and loads the `teacher.*` → `backbone.*` weights (`build_and_load_backbone`, clean
  missing/unexpected). Exposes `DINOVOL_HF_REPO` / `DINOVOL_CKPT_FILE`, `load_dinovol_checkpoint` (→ teacher sd +
  `model_cfg`), the stem skips + 288-d token grid + cls/reg tokens via `forward_features(x, view_kind="global")`.
- `model/windowed.py` — Pillar 1. `WindowedEvaAttention` (reassigns `attn.__class__`, parameter-preserving) runs
  non-overlapping 8³-token windowed SDPA on the patch tokens, RoPE at absolute 16³ coords, the 5 special tokens
  passed through unchanged. `retrofit_windowed(backbone, global_blocks, window_size)`.
- `model/pixel_decoder.py` — Pillar 2. Additive-skip UNet decoder (token grid + `[s0,s1,s2]` → full-res); semantic
  head (+ deep sup at 1/2, 1/4), 32-d mask features, and `feat_14` (1/4) for the instance head.
- `model/instance_head.py` — Pillar 3. Two-stage query selection (objectness over 4096 tokens → top-64 anchored
  queries) + query decoder (self-attn + `DeformableCrossAttn` over 1/8 tokens & 1/4 feats) → per-query
  class/mask-embed/anchor. Mask helpers `mask_logits_from_embed` (full) / `mask_logits_at_points` (training).
- `model/hercuvit_net.py` — assembles encoder → windowed retrofit → pixel decoder → heads; `forward(x)` → the
  output dict. `build_hercuvit(ckpt, arch_override=…, …)` is the **single** build entry; `VARIANT_ARCH` +
  `build_variant` are the canonical per-variant arch source of truth. `param_groups` / `set_transformer_trainable`
  drive staged training. `model/heads_future.py` stubs the inactive fibre/ink/region heads (§11).

### Plumbing built (this task)

- `data/normalize.py` — **Dinovol robust normalisation**, ported faithfully from villa `dataset/normalization.py`
  (`normalize_robust`: percentile 1–99 clip, then `(x−median)/(1.4826·MAD)` with villa's fallback ladder +
  `nan_to_num`). The released checkpoint's `config.dataset` has no `normalization_scheme` key → villa default
  `robust` (recorded as `DINOVOL_NORMALIZATION_SCHEME`). `gaussian_blur` intentionally omitted (§8). Exposes
  `dinovol_normalise` + `DinovolNormalizer.run`.
- `data/dataset.py` — `HercuvitDataset` over herculabels `sample_*.npz`. The npz stores meshes + provenance, **no
  CT**, so CT is streamed from the scroll (`ScrollCtReader` → reused `open_source`/`read_window`, per-scroll
  cache). `load_sample`/`rasterise_owner_full`/`has_identity` → owner (thin medial surfaces); semantic target =
  `owner>0`. `resample_to_crop` centre-crops the native cube to the `128·9.6 µm` FOV then trilinear-resizes CT /
  nearest-resizes owner to 128³ (normalisation applied to the native crop first, villa order). `geometric_augment`
  = flips + axis permutation (no blur), same transform to CT + targets. `identity_only` filters to herculabels
  samples (instance loss gated by `has_identity` regardless). `hercuvit_collate` keeps per-item flags as lists. CT
  reader is injectable for offline tests.
- `train/recipe.py` — `@dataclass(frozen=True) HercuvitRecipe` + shipped `recipes/hercuvit.yml` (package data) +
  `active()/set_active()/load()`, mirroring `hercunet/training/recipe.py`. Fields: window/arch, `attention_backend`
  (default `auto`), `variant` + `arch_override` tweak (arch numbers live in `VARIANT_ARCH`), data, optimiser/stage
  schedule, semantic + instance loss weights + Hungarian costs + point-sample, and the `distill_*` knobs. `VARIANTS`
  = names only (no duplicated arch numbers).
- `netbuild.py` — build plumbing OUTSIDE `model/`. `apply_attention_backend(net, backend)` sets each attention
  module's `attention_priority` (§12 #4 fix). `build(ckpt, variant, arch_override, attention_backend, …)` = single
  entry (`resolve_arch` merges the recipe tweak over `VARIANT_ARCH[variant]` → `build_hercuvit` → backend fix;
  stashes `_hercuvit_arch`). `build_from_cfg(model_cfg, arch_override, …)` rebuilds the architecture from the stored
  cfg with **no** Dinovol weights (inference), patching the Eva keys for slim variants (spatial-reduction not
  reconstructed here — §13).
- `train/losses.py` — `semantic_loss` (Dice+CE, deep sup; targets max-pooled per scale to keep thin surfaces) and
  `instance_loss` (per item, per decoder layer: Hungarian `linear_sum_assignment` on class + point-mask BCE/Dice
  cost → matched mask BCE+Dice at importance-sampled points via `mask_logits_at_points`, sheet/no-object CE, token
  objectness BCE, aux per layer). `has_identity=False` → a grad-connected zero over every instance output.
- `train/trainer.py` — `HercuvitTrainer`: standalone bf16-autocast AdamW loop on `net.param_groups`, grad-clip,
  warmup + constant/cosine LR; 3-stage schedule (semantic → unfreeze transformer 0.1× → +instance), optimiser
  rebuilt per stage; single-line `[ep N] loss …` logging; checkpoint save (model + `model_cfg` + resolved
  `arch_override`/`variant` + optim) and resume.
- `train/chain.py` — `resolve_corpus` (local / HF dataset `snapshot_download`), `resolve_dinovol` (reuses
  `hercunet.train.launch.resolve_pretrained`), `run_training` (build + dataset + loader + trainer), and the `fit` /
  `chain` CLI handlers.
- `infer/loader.py` — `load_hercuvit` resolves a trained checkpoint (local / HF), rebuilds via `build_from_cfg`
  (no Dinovol ckpt needed) and loads the trained weights; returns `(net, recipe, model_cfg)`. Instance-head keys
  are allowed to be "unexpected" when loading a full checkpoint into a semantic-only net.
- `infer/infer.py` — `infer_volume` streams `open_source`, tiles with the reused `pass_blend_tiles` +
  `make_gaussian`, forwards 128³ robust-normalised windows, Gaussian-blends per tile, writes `{out}_sem.zarr`
  (uint8 surface prob) via `create_surface_zarr` + `finalize_pyramid`; single pass, no prev channel. `single_instance`
  is the CLI handler (CPU fallback if no CUDA). Per-tile instance masks / cross-tile stitching = future (§13).
- `cli.py` + `__main__.py` — `hercuvit` console script mirroring `hercunet/cli.py` (argparse subparsers +
  `set_defaults(func=…)` + lazy imports): `train chain`, `train fit [--continue]`, `distill`, `infer
  single-instance`. Hyphenated flags; `--variant`, `--attention-backend`, `--corpus/--corpus-hf`,
  `--dinovol/--dinovol-hf` (default const `scrollprize/dinovol_ps8_9.6um_101626`). `distill` lazy-imports
  `train.distill:run`.
- `pyproject.toml` — `[project.scripts] hercuvit = "hercunet.hercuvit.cli:main"`, a `hercuvit` optional-deps extra
  (torch/timm/einops/scipy/huggingface_hub/pyyaml/zarr/blosc2), `full` extended, and package-data for
  `hercunet.hercuvit.recipes = ["*.yml"]`. Existing `hercunet` entry/extras untouched.
- `tests/hercuvit/` — CPU, `torch.manual_seed(0)`, tiny synthetic tensors: normalisation shape/stats; dataset
  resample + augment-label-consistency + a full offline `__getitem__` (synthetic npz + injected CT reader);
  semantic deep-sup downsampling; Hungarian identity recovery; `has_identity=False` zero + zero-grad; a tiny
  end-to-end forward+backward through the real pillar heads + both losses; attention-backend helper; variant
  source-of-truth / `resolve_arch`; recipe YAML round-trip; distill fields; CLI parse of every subcommand; and a
  real-Dinovol build + 128³ forward (skipped unless the checkpoint is cached). 19 tests, all passing.

### Deviations / notes

- **No `--epochs` per-stage control** — `--epochs` overrides the *total* `num_epochs` (as hercunet's does); the
  per-stage split stays in the recipe (`semantic_epochs`/`unfreeze_epochs`/`instance_epochs`).
- **`variants.py` not created** — an earlier instruction mentioned a `build_variant` delegation seam; superseded by
  the lead putting `VARIANT_ARCH`/`build_variant` in `hercuvit_net.py`. All variants go through `build_hercuvit`.
- **Inference tuple** diverges from `load_affinity_net`'s nnU-Net tuple (`(net, recipe, model_cfg)`) — HercuViT is
  not an nnU-Net, so there is no plans/label manager; the recipe carries the working scale + class count.
- Everything flagged best-effort (Dinovol norm, semantic target, resampling, point-sampling) is recorded in §13.

---

## Part III — Pod PoC findings (5090, scroll-5 corpus + PHerc1447)

First end-to-end validation on real data (5090, dinovol teacher + 40-sample scroll-5 herculabels subset):

- **Data path validated.** `open_source` streams scroll-5 CT (the flagged scroll-id resolution works: meta
  `scroll=5` → data-layer catalog); `HercuvitDataset` yields robust-normalised 128³ CT + `owner>0` surface + owner.
  Real unlabelled-crop streaming for distillation is now **data-layer-native** (`locate_material` +
  `iter_material_tiles`, air-skipped + prefetched) — not a hand-rolled reader.
- **Distillation works on real crops.** hercuvit-tiny distilled from the frozen teacher: held-out token cosine to
  the teacher **0.39 → 0.71** (final loss 0.20). Peak ~4 GB.
- **Window-seam measured + fixed.** Full-attention teacher boundary |Δ|-ratio **1.01** (no seam); windowed student
  **1.34–1.47** (a real window-boundary discontinuity — the vertical/horizontal lines in the token PCA).
  **Swin shifted windows** (alternating `window//2` cyclic shift + masked attention, RoPE kept absolute) cut it to
  **1.11** — near the teacher. This **supersedes the design doc's §5.1 no-shift (ViTDet) choice** (2 global blocks +
  short distillation did not smooth it); `retrofit_windowed(..., shifted=True)` is the default.
- **Thin-surface loss.** The sheet-surface target is a ~0.2%-positive 1-voxel medial surface; a plain Dice+CE head
  collapses to all-background. Ported HercUNet's thin-structure stack into a self-contained module
  (`train/surface_loss.py`, **no nnU-Net dep**): foreground-weighted CE + Focal-Tversky + clDice skeleton-recall +
  separation penalty. (Note: `soft_skel` erodes an already-1-voxel GT to near-empty, so skeleton-recall adds little
  gradient on this target — the author's point-contrastive + linear-probe surface head is the robust complement.)
- **Supervised stage-1 (semantic, transformer frozen), 2000 steps / 34 samples:** train surfDice@2 **0.245**
  (fitting), val surfDice@2 **0.066** (undertrained at PoC scale). Real training step fits **~5 GB** (small, bf16).
- **Decision (plan §1):** surface = author's point-contrastive + linear probe (robust to the thin-surface
  imbalance) **and** hercunet thin-surface losses; Mask2Former head for per-sheet IDs. Dense-only semantic is a
  baseline, not the final surface path.

TODO (updated §13): point-contrastive + linear-probe surface head (author's recipe); scale up corpus + steps for a
real semantic number; `shifted` as a recipe/CLI flag threaded through `build_from_cfg` for inference parity.

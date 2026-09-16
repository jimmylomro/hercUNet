"""``hercunet train {export-labels, export-owner}`` — dataset assembly (HercUNet run301 / v0 recipe).

Our own code (no nnU-Net internals), the iterative AffinityMalis refiner recipe — NOT the ablation:

- export-labels : decode our exported ∇φ sample corpus into nnU-Net cases — CT + K candidate/prev crests +
                  ``ownersTr`` (MALIS instance GT), **uncleaned**. With ``m7_corpus`` (pre-mined by
                  ``hercunet labels m7-mine``), also **copy** those m7 cases in as ``m7_*`` (real writes, no
                  symlinks) — the rehearsal signal.  [✅ live]
- export-owner  : write ``<case>_owner.b2nd`` at preprocessed resolution for constrained MALIS.  [✅ live]

The ``--corpus`` for export-labels is a directory of ``sample_s*.npz`` bundles produced by
``hercunet labels export`` (base record + augmentations). Keeping the two commands separate — ``labels
export`` is the light corpus→samples step, ``train export-labels`` the heavy samples→nnU-Net (CT fetch +
∇φ decode + emit) step — is deliberate (they do not overlap in compute).
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re

import numpy as np

_SCROLL_RE = re.compile(r"sample_s(.+?)_L\d")


# ============================================================================ export-labels (✅) ==
def export_labels(*, corpus, out, m7_corpus=None, surface_tau=0.91, ignore_score=0.80,
                  gate_score=0.85, gate_count=2, max_candidates=4, confidence_ignore=False,
                  prefix="our", nshards=1, shard=0, write_dataset_json=True, limit=None,
                  log_every=25, be=None):
    """Export our ∇φ pseudo-label samples as an nnU-Net **raw** iterative dataset under ``out``.

    Input channels ``[CT, cand0(base), cand1(aug), …]``, target = the clean BASE surface {0,1,2}. The
    ``prev`` channel is NOT baked — the AffinityMalis trainer's ``ComposePrevTransform`` composes a fresh
    ``prev`` from the candidate channels every epoch. Per case, under ``out``:
      • ``imagesTr/<case>_0000.tif`` — CT uint8 (percentile-normed, matches m7).
      • ``imagesTr/<case>_000{k+1}.tif`` — candidate k ∇φ mag float32 [0,1]; ABSENT slots = -1 sentinel.
      • ``labelsTr/<case>.tif`` — base surface {0 bg, 1 surface, 2 ignore} uint8.
      • ``ownersTr/<case>.tif`` — per-sheet instance GT (int; 0 = bg/ignore) for constrained MALIS.
      • ``<case>.json`` spacing sidecars.
    ``max_candidates`` (K) MUST match ``preprocess`` (plans patched to 1+K) and ``labels m7-mine``. HercUNet
    trains UNCLEANED, so ``confidence_ignore`` defaults False (keep only the collapsed-sheet geometric ignore).
    """
    import tifffile

    from hercunet.labels.io.bundle import load_sample_bundle
    from hercunet.labels.mesh.medial import build_gradphi_edt

    if be is None:
        from hercunet.config import Config
        from hercunet.data import get_backend
        be = get_backend(Config.from_env())
    from hercunet.data.prefetch import iter_windows

    K = int(max_candidates)
    paths = _resolve_sample_paths(corpus)
    if nshards > 1:
        paths = [p for i, p in enumerate(paths) if i % nshards == shard]
    print(f"[export-labels] shard {shard}/{nshards}: {len(paths)} sample bundle(s) from {corpus}", flush=True)

    images_dir = os.path.join(out, "imagesTr")
    labels_dir = os.path.join(out, "labelsTr")
    owners_dir = os.path.join(out, "ownersTr")
    for d in (images_dir, labels_dir, owners_dir):
        os.makedirs(d, exist_ok=True)

    by_scroll = collections.OrderedDict()
    for p in paths:
        by_scroll.setdefault(_scroll_of(p), []).append(p)

    cases = []
    for sid, spaths in by_scroll.items():
        try:
            vol = be.open_scroll_volume(_find_scroll(be, sid))
        except Exception as e:                                       # noqa: BLE001
            print(f"[export-labels] scroll {sid} unopenable ({e}) — skipping its samples", flush=True)
            continue
        metas, reqs, use_paths = {}, [], []
        for p in spaths:
            meta, recs = load_sample_bundle(p)
            per_sheet = meta.get("quality", {}).get("per_sheet", {})
            if not _window_passes(per_sheet, gate_score, gate_count):    # >=gate_count sheets >= gate_score
                continue
            metas[p] = (meta, recs)
            reqs.append(_window_req(meta, vol))
            use_paths.append(p)
        for p, (_req, blk, _org) in zip(use_paths, iter_windows(vol, reqs, readahead=3)):
            meta, recs = metas[p]
            shape = tuple(meta["shape"]); vu = meta["voxel_um"]; ds = int(meta.get("conf_ds", 4))
            per_sheet = meta.get("quality", {}).get("per_sheet", {})
            if "base" not in recs:
                continue
            try:                                                     # base decode → mag + owner_full for the LABEL
                mag_b, lab3, _nrm, _w3, owner_full = build_gradphi_edt(
                    recs["base"]["meshes"], shape, vu, return_owner_full=True)
            except Exception as e:                                   # noqa: BLE001  (skip a bad record, keep going)
                print(f"[export-labels] SKIP {p} [base]: {e}", flush=True)
                continue
            weight_b = _weight(recs["base"].get("conf", {}), shape, ds)
            surf = _build_surface_label(mag_b, owner_full, weight_b, per_sheet, surface_tau=surface_tau,
                                        ignore_score=ignore_score, confidence_ignore=confidence_ignore).astype(np.uint8)

            candidates = [np.clip(mag_b, 0.0, 1.0).astype(np.float32)]   # slot 0 = clean base; rest = aug variants
            for name, r in recs.items():
                if name == "base" or len(candidates) >= K:
                    continue
                try:
                    mag_a = build_gradphi_edt(r["meshes"], shape, vu, return_owner_full=False)[0]
                    candidates.append(np.clip(mag_a, 0.0, 1.0).astype(np.float32))
                except Exception as e:                               # noqa: BLE001
                    print(f"[export-labels] skip aug {p}[{name}]: {e}", flush=True)
            while len(candidates) < K:                                # pad absent slots with the -1 sentinel
                candidates.append(np.full(shape, -1.0, np.float32))

            cid = _case_id(p, prefix)
            ct8 = (np.clip(_norm_ct(blk), 0.0, 1.0) * 255.0).round().astype(np.uint8)
            tifffile.imwrite(os.path.join(images_dir, f"{cid}_0000.tif"), ct8)
            for k in range(K):
                tifffile.imwrite(os.path.join(images_dir, f"{cid}_{k + 1:04d}.tif"), candidates[k])
            tifffile.imwrite(os.path.join(labels_dir, f"{cid}.tif"), surf)
            # MALIS instance GT: band-limited owner id (lab3: sheet id in-band / -1 outside) shifted so 0 = bg;
            # collapsed-sheet ignore(2) → 0 (excluded from MALIS pairs).
            inst = (lab3.astype(np.int32) + 1)                        # -1 → 0 (bg), sheet ids → >=1
            inst[surf == 2] = 0
            tifffile.imwrite(os.path.join(owners_dir, f"{cid}.tif"), inst)
            for jp in (os.path.join(images_dir, f"{cid}.json"), os.path.join(labels_dir, f"{cid}.json")):
                with open(jp, "w") as f:
                    json.dump({"spacing": [1.0, 1.0, 1.0]}, f)
            cases.append(cid)
            if log_every and len(cases) % log_every == 0:
                print(f"[export-labels] {len(cases)} cases -> {out}", flush=True)
            if limit and len(cases) >= limit:
                break
        if limit and len(cases) >= limit:
            break

    n_m7 = _copy_m7_corpus(m7_corpus, out, K) if m7_corpus else 0

    if write_dataset_json:
        with open(os.path.join(out, "dataset.json"), "w") as f:
            json.dump({"channel_names": _iter_channel_names(K),
                       "labels": {"background": 0, "surface": 1, "ignore": 2},
                       "numTraining": len(cases) + n_m7,
                       "file_ending": ".tif",
                       "overwrite_image_reader_writer": "Tiff3DIO"}, f, indent=4)
    print(f"[export-labels] DONE {len(cases)} our cases + {n_m7} m7 cases -> {out}", flush=True)
    return cases


def _copy_m7_corpus(m7_corpus, out, K):
    """COPY a pre-mined m7 corpus's cases (from ``hercunet labels m7-mine``) into ``out`` as ``m7_*`` — real
    file writes, NO symlinks. m7 cases are surface-only (no ``ownersTr`` → MALIS stays silent on them). The m7
    corpus is itself an nnU-Net raw dataset (``imagesTr/`` with 1+K channels, ``labelsTr/``); we copy every
    image/label tif + its spacing sidecar. Returns the number of m7 label cases copied."""
    import shutil

    src_img = os.path.join(m7_corpus, "imagesTr")
    src_lab = os.path.join(m7_corpus, "labelsTr")
    if not (os.path.isdir(src_img) and os.path.isdir(src_lab)):
        raise SystemExit(f"hercunet train export-labels: --m7-corpus {m7_corpus} is not an nnU-Net corpus "
                         f"(missing imagesTr/ or labelsTr/). Run `hercunet labels m7-mine` first.")
    dst_img = os.path.join(out, "imagesTr")
    dst_lab = os.path.join(out, "labelsTr")
    os.makedirs(dst_img, exist_ok=True); os.makedirs(dst_lab, exist_ok=True)

    labels = sorted(f for f in os.listdir(src_lab) if f.endswith(".tif"))
    n = 0
    for lab in labels:
        case = lab[:-4]
        chans = sorted(glob.glob(os.path.join(src_img, f"{case}_[0-9][0-9][0-9][0-9].tif")))
        if len(chans) != 1 + K:                                      # channel count must match our cases
            print(f"[export-labels] m7 case {case}: {len(chans)} channels != 1+K={1 + K}; skipping", flush=True)
            continue
        for c in chans:
            shutil.copy2(c, os.path.join(dst_img, os.path.basename(c)))
        shutil.copy2(os.path.join(src_lab, lab), os.path.join(dst_lab, lab))
        for side in (os.path.join(src_img, f"{case}.json"), os.path.join(src_lab, f"{case}.json")):
            if os.path.isfile(side):
                shutil.copy2(side, os.path.join(dst_img if side == os.path.join(src_img, f"{case}.json")
                                                else dst_lab, f"{case}.json"))
        n += 1
    print(f"[export-labels] copied {n} m7 case(s) from {m7_corpus}", flush=True)
    return n


# ============================================================================ export-owner (✅) ==


def export_owner(*, dataset, out=None):
    """Write the MALIS instance-GT sidecar ``<case>_owner.b2nd`` at PREPROCESSED resolution.

    For a dataset that has ``ownersTr/<case>.tif`` (from :func:`export_labels`), this crops each raw owner tif
    with the SAME crop nnU-Net used (``props['bbox_used_for_cropping']`` in ``<case>.pkl``) and saves
    ``<case>_owner.b2nd`` (1,Z',Y',X') int16, byte-aligned to ``<case>_seg.b2nd``. The AffinityMalis trainer's
    ``_OwnerDataLoader`` loads it and threads it as a second segmentation channel (nearest interp).
    Self-contained (numpy / tifffile / blosc2 / pickle only).

    dataset : nnU-Net dataset id (int) or ``DatasetNNN_…`` name.
    out     : owner sidecar dir; default ``<preprocessed>/<config>/owner_b2nd`` (a SUBDIR so nnU-Net's
              case-glob ignores it).
    """
    import pickle

    import tifffile
    import blosc2

    from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

    name = maybe_convert_to_dataset_name(dataset)
    raw = _dataset_dir(nnUNet_raw, name)
    pp = _resolve_pp_config(os.path.join(nnUNet_preprocessed, name))
    owners_raw = os.path.join(raw, "ownersTr")
    if not os.path.isdir(owners_raw):
        raise SystemExit(f"hercunet train export-owner: no ownersTr/ under {raw} (run export-labels first).")
    owner_dir = out or os.path.join(pp, "owner_b2nd")
    os.makedirs(owner_dir, exist_ok=True)

    def blosc_save(arr, path):
        if os.path.lexists(path):
            try:
                os.remove(path)
            except IsADirectoryError:
                import shutil
                shutil.rmtree(path)
        blosc2.asarray(np.ascontiguousarray(arr), urlpath=path, mode="w")

    pkls = sorted(glob.glob(os.path.join(pp, "*.pkl")))
    if not pkls:
        raise SystemExit(f"hercunet train export-owner: no preprocessed cases in {pp} (run preprocess first).")
    ok = bad = 0
    for pk in pkls:
        case = os.path.basename(pk)[:-4]
        with open(pk, "rb") as f:
            props = pickle.load(f)
        seg_shape = tuple(blosc2.open(os.path.join(pp, case + "_seg.b2nd"), mode="r").shape)  # (1,Z',Y',X')
        tif = os.path.join(owners_raw, case + ".tif")
        if not os.path.isfile(tif):                                   # m7 (MALIS-off) cases have no owner
            print(f"  MISS owner tif {case}", flush=True); bad += 1; continue
        owner = tifffile.imread(tif).astype(np.int16)                 # (Z,Y,X) raw window shape
        bbox = props.get("bbox_used_for_cropping")
        if bbox is not None:
            sl = tuple(slice(int(b[0]), int(b[1])) for b in bbox)
            owner = owner[sl]
        if owner.shape != seg_shape[1:]:
            print(f"  SHAPE-MISMATCH {case}: owner{owner.shape} vs seg{seg_shape[1:]}", flush=True)
            bad += 1; continue
        blosc_save(owner[None], os.path.join(owner_dir, case + ".b2nd"))
        ok += 1
        if ok % 50 == 0:
            print(f"  {ok} owner sidecars written", flush=True)
    print(f"[export-owner] {ok} written, {bad} skipped -> {owner_dir}", flush=True)


# ================================================================================= helpers ==
def _resolve_sample_paths(corpus):
    """The list of ``sample_s*.npz`` bundles under ``corpus`` (a dir written by ``hercunet labels export``)."""
    from hercunet.labels.corpus import Corpus
    if Corpus.is_corpus(corpus):
        raise SystemExit(
            f"hercunet train export-labels: --corpus {corpus} is a .herculabels corpus, not a sample dir. "
            f"Run `hercunet labels export {corpus} <sample_dir>` first, then point --corpus at <sample_dir>.")
    paths = sorted(glob.glob(os.path.join(corpus, "*.npz")))
    if not paths:
        raise SystemExit(f"hercunet train export-labels: no sample_*.npz bundles under {corpus} "
                         f"(produce them with `hercunet labels export`).")
    return paths


def _scroll_of(path):
    """Scroll id straight from the sample filename (no load) — for cheap by-scroll grouping."""
    m = _SCROLL_RE.search(path.rsplit("/", 1)[-1])
    return m.group(1) if m else None


def _find_scroll(be, sid):
    """The ScrollInfo whose id matches ``sid`` (from the sample filename). Exact ``scroll_id`` match, then a
    tolerant substring over id / name / PHerc handle."""
    key = str(sid).lower().replace(" ", "")
    scrolls = list(be.list_scrolls())
    for s in scrolls:                                                # exact id first
        if str(s.scroll_id).lower().replace(" ", "") == key:
            return s
    for s in scrolls:                                                # then tolerant substring
        for c in (s.scroll_id, s.name, s.extra.get("pherc", "")):
            if key and key in str(c).lower().replace(" ", ""):
                return s
    raise KeyError(f"scroll {sid} not found")


def _window_req(meta, vol):
    """(level, z0,z1, y0,y1, x0,x1) for THIS window — same origin math as label generation's open_ct."""
    L = int(meta["level"]); Zf, Yf, Xf = vol.meta.level_shapes[L]
    Z, Y, X = (int(s) for s in meta["shape"]); zc, yc, xc = (int(v) for v in meta["org"])
    z0 = int(np.clip(zc - Z // 2, 0, max(0, Zf - Z)))
    y0 = int(np.clip(yc - Y // 2, 0, max(0, Yf - Y)))
    x0 = int(np.clip(xc - X // 2, 0, max(0, Xf - X)))
    return (L, z0, z0 + Z, y0, y0 + Y, x0, x0 + X)


def _norm_ct(blk):
    g = np.asarray(blk, np.float32)
    lo, hi = np.percentile(g, 1), np.percentile(g, 99)
    return np.clip((g - lo) / max(hi - lo, 1e-3), 0.0, 1.0)


def _weight(conf, shape, ds):
    """Per-voxel confidence weight ∈ [0,1] = product of the stored per-channel confidences (any one doubt drops
    it): intersection (crossings), sharp2 (co-membership / dip-seam), dropped (culled fragments), and — on aug
    records — perturb (the synthetic merge/split break zone)."""
    from hercunet.labels.io.bundle import upsample_conf
    w = np.ones(shape, np.float32)
    for chan in (conf or {}):
        w = w * upsample_conf(conf[chan], shape, ds).astype(np.float32)
    return np.clip(w, 0.0, 1.0)


def _window_passes(per_sheet, gate_score=0.85, gate_count=2):
    """Per-sheet GATE — keep a window only if it has at least ``gate_count`` sheets with fiber-field congruence
    >= ``gate_score`` (>=2 confidently-clean sheets corroborating real, well-segmented material)."""
    return sum(1 for v in (per_sheet or {}).values() if float(v) >= gate_score) >= gate_count


def _build_surface_label(mag, owner_full, weight, per_sheet, *, surface_tau=0.91, ignore_score=0.80,
                         confidence_ignore=True):
    """The 3-class surface label {0 bg, 1 surface, 2 ignore}. surface = ``|∇φ| >= surface_tau``. IGNORE(2) is the
    union of: COLLAPSED-SHEET (always) — a sheet whose congruence < ``ignore_score`` is geometrically wrong, so
    its FULL unbounded Voronoi territory (``owner_full`` == that sheet) is ignored; and CONFIDENCE (only if
    ``confidence_ignore``) — stored ``weight`` < 0.5 on surface voxels. HercUNet trains uncleaned
    (``confidence_ignore=False``): the confidence field was meant to SUPERVISE a confidence head, not mask the
    loss on the merge/seam regions the iterative refiner must learn to resolve."""
    core = mag >= surface_tau
    surf = core.astype(np.uint8)
    bad = [int(c) for c, s in (per_sheet or {}).items() if float(s) < ignore_score]
    cong = np.isin(owner_full, bad) if bad else np.zeros(core.shape, bool)
    conf = (core & (weight < 0.5)) if confidence_ignore else np.zeros(core.shape, bool)
    surf[cong] = 2
    surf[conf] = 2
    return surf


def _case_id(path, prefix):
    """Stable nnU-Net case id from a sample path. e.g. our_s5_z1008_y5826_x3738 (base record)."""
    base = os.path.splitext(os.path.basename(path))[0].replace("sample_", "")
    return f"{prefix}_{base}"


def _iter_channel_names(max_candidates):
    """channel_names for the per-epoch iterative dataset: CT + ``max_candidates`` candidate ∇φ slots."""
    names = {"0": "CT"}
    for k in range(max_candidates):
        names[str(k + 1)] = f"cand{k}"
    return names


def _dataset_dir(root, name):
    """Resolve ``DatasetNNN_…`` under ``root`` (tolerate an id-only name)."""
    exact = os.path.join(root, name)
    if os.path.isdir(exact):
        return exact
    hits = sorted(glob.glob(os.path.join(root, name.split("_")[0] + "*")))
    if not hits:
        raise FileNotFoundError(f"no dataset folder for {name!r} under {root}")
    return hits[0]


def _resolve_pp_config(pp_root):
    """The preprocessed data-config subdir under ``pp_root`` (the one holding ``*.pkl`` + ``*_seg.b2nd``).
    Detected, not guessed: nnU-Net names it by the config's ``data_identifier`` (e.g. ``nnUNetPlans_3d_fullres``
    when m7's plans are transplanted), which is NOT the plans id — so we find the single subdir that actually
    has cases (``gt_segmentations`` has none). Exactly one config is preprocessed, so this is unambiguous."""
    hits = [d for d in sorted(glob.glob(os.path.join(pp_root, "*"))) if glob.glob(os.path.join(d, "*.pkl"))]
    if not hits:
        raise SystemExit(f"hercunet train export-owner: no preprocessed config with cases under {pp_root} "
                         f"(run preprocess first).")
    if len(hits) > 1:
        raise SystemExit(f"hercunet train export-owner: multiple preprocessed configs with cases under "
                         f"{pp_root} ({', '.join(os.path.basename(h) for h in hits)}) — expected exactly one.")
    return hits[0]

"""``hercunet labels m7-mine`` — mine m7 pseudo-labels into a pre-extracted **m7 corpus**.

m7 (the published ScrollPrize surface detector) beats our mesh pipeline in many *compressed* regions, so we
mine its coherent crest there as extra detection signal (writeup Part II — the rehearsal signal). This reads
the **published m7 surface predictions** + CT from the open-data bucket (NOT m7 inference) and turns each
window into a ``{0 bg, 1 surface, 2 ignore}`` label via ``surf = m7 & (m7_normal_coherence > tau)`` — the
SAME recipe used for corpus cleanup, so mined labels are one construction.

Two paths:
  * fresh scout (``--scroll``): scan each scroll's L5 for the real-papyrus compressed band, pick coherent
    regions with the coherence judge, write the m7 corpus + a ``manifest.json`` (for reproducibility).
  * rebuild (``--manifest``): re-open the approved regions and rebuild the EXACT same labels
    deterministically (same regions + same tau ⇒ identical corpus).

The result is a standalone **m7 corpus** — an nnU-Net raw dataset (``imagesTr/`` CT + K all-(-1) candidate
slots, ``labelsTr/`` {0,1,2}, spacing sidecars, iterative ``dataset.json``). MALIS is OFF on m7 (no per-sheet
membership) so **no ``ownersTr``** is written (the loader treats a missing owner as all-zero → MALIS silent;
the CT-material grow/air-suppress term handles separation geometrically). Candidate slots are all -1 → the
prev transform bootstraps prev to 0 → these cases train COLD detection of compressed-region sheets. Cases use
the ``m7`` prefix so the trainer's per-source logging splits them. Consumed by
``hercunet train export-labels --m7-corpus`` which COPIES the cases in (no re-mining, no symlinks).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from scipy import ndimage as ndi

BUCKET = "vesuvius-challenge-open-data"
CUBE = 192
COH_TAU = 0.88          # m7 normal-coherence gate (validated mine_prod5 default)
# scout defaults (mine_prod5): real papyrus > MAT_THR; compressed material band [MATLO, MATHI]; per-scroll cap.
MAT_THR, MATLO, MATHI, PER_SCROLL, TARGET = 95, 0.75, 0.95, 18, 100
NC5 = 6                 # L5 super-voxel = CUBE / 2^5, rounded (mine_prod5)
SCROLLS = ["PHerc0125", "PHerc0175A", "PHerc0175B", "PHerc0191", "PHerc0211", "PHerc0257", "PHerc0268",
           "PHerc0306B", "PHerc0343", "PHerc0483A", "PHerc0483B", "PHerc0490A", "PHerc0490B", "PHerc0800",
           "PHerc0813", "PHerc0826", "PHerc0846B", "PHerc1218", "PHerc1447", "PHerc1545"]


# ------------------------------------------------------------------ the shared m7-label recipe --
def m7_coherence(m):
    """m7's OWN normal-coherence field: structure tensor on the Gaussian-smoothed m7 binary. High where m7's
    surface normals vary smoothly (a clean sheet), low at ragged/merged zones. ``m``: bool (Z,Y,X). Returns
    coherence float (Z,Y,X) in [0,1]."""
    from hercunet.labels.fields.structure_tensor_torch import structure_tensor_frame_torch as ST
    f = ndi.gaussian_filter((m * 255).astype(np.float32), 1.0)
    return ST(f, sigma_grad=1.0, sigma_tensor=4.0, want_fibre=False)["coherence"]


def label_from_window(m, tau=COH_TAU):
    """3-class label {0 bg, 1 surface, 2 ignore} from an m7 binary window:
        surface(1) = m & coh>tau     (coherent m7 crest — trusted positive)
        ignore(2)  = m & ~(coh>tau)  (incoherent m7 — uncertain, masked)
        bg(0)      = ~m               (m7 says no sheet — negative)
    Returns (label uint8, coh float, surf bool)."""
    coh = m7_coherence(m)
    surf = m & (coh > tau)
    lab = np.zeros(m.shape, np.uint8)
    lab[m & ~surf] = 2
    lab[surf] = 1
    return lab, coh, surf


# --------------------------------------------------------------------------------- entry point --
def mine(*, out, scroll=None, manifest=None, max_candidates=4, tau=None, workers=8, images=False):
    """Write an m7 corpus into ``out``. ``--manifest`` rebuilds an approved region set deterministically;
    ``--scroll`` scouts fresh. ``images`` (scout only) also writes the 3-panel QC render per approved region
    (raw m7 | coherence | grabbed-coherent) and records its filename in the manifest. ``max_candidates`` (K)
    MUST match ``train export-labels``."""
    K = int(max_candidates)
    tau = COH_TAU if tau is None else float(tau)
    if manifest:
        return _export_from_manifest(manifest, out, K, tau=tau, workers=workers)
    if scroll:
        return _scout(out, _select_scrolls(scroll), K=K, tau=tau, workers=workers, images=images)
    raise SystemExit("hercunet labels m7-mine: provide --scroll (fresh scout) or --manifest (rebuild).")


def _select_scrolls(scroll):
    """Filter the known open-data scroll list by the ``--scroll`` tokens (case-insensitive substring, so
    ``1447,0800`` or ``PHerc1447`` both work); ``all`` = every scroll."""
    toks = [t.strip().lower() for t in str(scroll).split(",") if t.strip()]
    if "all" in toks:
        return list(SCROLLS)
    hits = [s for s in SCROLLS if any(t in s.lower() for t in toks)]
    if not hits:
        raise SystemExit(f"hercunet labels m7-mine: --scroll {scroll!r} matched none of the open-data scrolls "
                         f"({', '.join(SCROLLS)}). Use PHerc ids or a substring, or 'all'.")
    return hits


# ---------------------------------------------------------------------------- case + json I/O --
def _norm_ct(blk):
    g = np.asarray(blk, np.float32)
    lo, hi = np.percentile(g, 1), np.percentile(g, 99)
    return np.clip((g - lo) / max(hi - lo, 1e-3), 0.0, 1.0)


def _iter_channel_names(K):
    names = {"0": "CT"}
    for k in range(K):
        names[str(k + 1)] = f"cand{k}"
    return names


def _write_case(out_dir, cid, ct_u8, lab_u8, K):
    """One nnU-Net case in the iterative format: CT + K all-(-1) candidate slots (prev bootstraps to 0) +
    {0,1,2} label + spacing sidecars. No ownersTr (MALIS silent on m7)."""
    import tifffile
    images = os.path.join(out_dir, "imagesTr")
    labels = os.path.join(out_dir, "labelsTr")
    os.makedirs(images, exist_ok=True)
    os.makedirs(labels, exist_ok=True)
    tifffile.imwrite(os.path.join(images, f"{cid}_0000.tif"), ct_u8)
    sent = np.full(ct_u8.shape, -1.0, np.float32)                # candidate slots all -1 → prev bootstraps to 0
    for k in range(K):
        tifffile.imwrite(os.path.join(images, f"{cid}_{k + 1:04d}.tif"), sent)
    tifffile.imwrite(os.path.join(labels, f"{cid}.tif"), lab_u8)
    for jp in (os.path.join(images, f"{cid}.json"), os.path.join(labels, f"{cid}.json")):
        with open(jp, "w") as f:
            json.dump({"spacing": [1.0, 1.0, 1.0]}, f)


def _rewrite_dataset_json(out_dir, K):
    """(Re)write dataset.json with numTraining = ALL labelsTr cases present. Idempotent."""
    n = len(glob.glob(os.path.join(out_dir, "labelsTr", "*.tif")))
    with open(os.path.join(out_dir, "dataset.json"), "w") as f:
        json.dump({"channel_names": _iter_channel_names(K),
                   "labels": {"background": 0, "surface": 1, "ignore": 2},
                   "numTraining": n, "file_ending": ".tif",
                   "overwrite_image_reader_writer": "Tiff3DIO"}, f, indent=4)
    return n


def _https(path):
    """manifest 's3://bucket/key' or 'bucket/key' → anon https URL (what ZarrSegment opens)."""
    p = path.replace("s3://", "")
    bucket, _, key = p.partition("/")
    return f"https://{bucket}.s3.amazonaws.com/{key}"


# ---------------------------------------------------------------------------- rebuild (manifest) --
def _export_from_manifest(manifest_path, out_dir, K, *, tau=COH_TAU, prefix="m7", workers=8):
    """Rebuild the approved labels from a scout manifest → m7 cases in ``out_dir``. Reads each m7 + CT window
    through the data layer (``ZarrSegment.read_window``) on a THREAD POOL so anon-S3 latency of many windows
    OVERLAPS (sequential reads on the throttled open-data bucket are ~1 region/40s — unworkable). Returns the
    number of cases written."""
    from concurrent.futures import ThreadPoolExecutor
    import threading

    from hercunet.data.zarr_reader import ZarrSegment

    with open(manifest_path) as f:
        regions = json.load(f)
    print(f"[m7-mine] rebuild: {len(regions)} regions from {manifest_path} -> {out_dir} "
          f"(K={K}, workers={workers})", flush=True)
    segs, lock, cnt = {}, threading.Lock(), {"done": 0, "skip": 0}

    def seg(path):                                              # open each zarr once, reuse across regions/threads
        u = _https(path)
        with lock:
            if u not in segs:
                segs[u] = ZarrSegment(u, None)
            return segs[u]

    def do(r):
        z0, y0, x0 = r["origin_l0"]
        cube = int(r.get("size", CUBE))
        t = float(r.get("tau", tau))
        cid = f"{prefix}_{r['scroll']}_z{z0}_y{y0}_x{x0}"
        try:
            m = seg(r["m7"]).read_window(0, z0, z0 + cube, y0, y0 + cube, x0, x0 + cube)[0] > 0
            ct = seg(r["vol"]).read_window(0, z0, z0 + cube, y0, y0 + cube, x0, x0 + cube)[0]
        except Exception as e:                                  # noqa: BLE001
            with lock:
                cnt["skip"] += 1
            print(f"[m7-mine] SKIP {cid}: read {e}", flush=True)
            return
        if m.sum() < 1200:                                     # same floor mine_prod5 used
            with lock:
                cnt["skip"] += 1
            print(f"[m7-mine] SKIP {cid}: m7 empty ({int(m.sum())})", flush=True)
            return
        lab, _coh, surf = label_from_window(m, t)
        ct_u8 = (np.clip(_norm_ct(ct), 0.0, 1.0) * 255.0).round().astype(np.uint8)
        _write_case(out_dir, cid, ct_u8, lab.astype(np.uint8), K)
        with lock:
            cnt["done"] += 1
            if cnt["done"] == 1 or cnt["done"] % 10 == 0:
                print(f"[m7-mine] {cnt['done']}/{len(regions)} {cid} "
                      f"surf%={surf.sum() / max(m.sum(), 1) * 100:.0f}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(do, regions))
    n = _rewrite_dataset_json(out_dir, K)
    print(f"[m7-mine] DONE rebuild: {cnt['done']} written, {cnt['skip']} skipped; corpus now {n} cases -> "
          f"{out_dir}", flush=True)
    return cnt["done"]


# ------------------------------------------------------------------------------ scout (--scroll) --
def _openz(fs, p):
    import zarr
    import s3fs
    return zarr.open(s3fs.S3Map(p, s3=fs), mode="r")


def _render(out_dir, idx, r, ctS, m2d, surf2d, coh2d):
    """3-panel QC render (raw m7 red | m7-normal-coherence | grabbed-coherent green), identical to mine_prod5.
    Returns the JPEG filename (also stored in the manifest's ``image`` field)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = np.clip(ctS / 255.0, 0, 1)
    raw = np.stack([g, g, g], -1)
    raw[m2d] = 0.55 * np.array([1, .15, .15]) + .45 * raw[m2d]
    grab = np.stack([g, g, g], -1)
    grab[surf2d] = [.1, 1, .1]
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.6))
    ax[0].imshow(np.clip(raw, 0, 1)); ax[0].set_title("raw m7 (red)", fontsize=9); ax[0].axis("off")
    ax[1].imshow(coh2d, cmap="magma", vmin=0, vmax=1)
    ax[1].set_title("m7-normal-coherence", fontsize=9); ax[1].axis("off")
    ax[2].imshow(np.clip(grab, 0, 1))
    ax[2].set_title(f"grabbed coherent ({r['coh_score'] * 100:.0f}% of m7); rest ignore", fontsize=9)
    ax[2].axis("off")
    c = r["center_l0"]
    fig.suptitle(f"#{idx:03d} {r['scroll']}  L0 z{c[0]} y{c[1]} x{c[2]}  papyrus {r['material']:.2f}  "
                 f"coh {r['coh_score']:.2f}", fontsize=10)
    fig.tight_layout()
    fn = f"region_{idx:03d}_{r['scroll']}.jpg"
    fig.savefig(os.path.join(out_dir, fn), dpi=140, bbox_inches="tight", pil_kwargs={"quality": 88})
    plt.close(fig)
    return fn


def _scout(out_dir, scrolls, *, K=4, tau=COH_TAU, workers=8, images=False, target=TARGET,
           per_scroll=PER_SCROLL, mat_thr=MAT_THR, matlo=MATLO, mathi=MATHI):
    """Faithful port of the mine_prod5 scout: find compressed high-real-papyrus regions where m7 is coherent,
    write a ``manifest.json`` AND the m7 label cases. Fully automatic — the coherence judge selects the regions;
    the manifest gives reproducibility. ``images=True`` also writes the per-region 3-panel QC render and records
    its filename in each region's ``image`` field. Returns the manifest list."""
    import s3fs

    os.makedirs(out_dir, exist_ok=True)
    fs = s3fs.S3FileSystem(anon=True)
    passers = []
    for sc in scrolls:
        try:
            b = f"{BUCKET}/{sc}"
            m7g = fs.glob(b + "/representations/predictions/surfaces/*surface-m7*.zarr")
            if not m7g:
                print(f"{sc}: no m7", flush=True)
                continue
            m7p = m7g[0]
            ts = m7p.split("/")[-1].split("-")[0]
            volg = [v for v in fs.glob(b + "/volumes/*masked.zarr") if v.split("/")[-1].startswith(ts)] \
                or fs.glob(b + "/volumes/*masked.zarr")
            mz = _openz(fs, m7p)
            vz = _openz(fs, volg[0])
            vp = volg[0]
            ct5 = np.asarray(vz["5"])
            Z, Y, X = ct5.shape
            nz, ny, nx = Z // NC5, Y // NC5, X // NC5
            mm = (ct5[:nz * NC5, :ny * NC5, :nx * NC5] > mat_thr).reshape(
                nz, NC5, ny, NC5, nx, NC5).mean((1, 3, 5))
            del ct5
            sel = (mm >= matlo) & (mm <= mathi)
            band = np.argwhere(sel)
            band = band[np.argsort(-mm[sel])]
            picks, nk = [], 0
            for bz, by, bx in band:
                if len(picks) >= per_scroll:
                    break
                if any(abs(bz - p[0]) < 3 and abs(by - p[1]) < 3 and abs(bx - p[2]) < 3 for p in picks):
                    continue
                z0, y0, x0 = int(bz * CUBE), int(by * CUBE), int(bx * CUBE)
                if z0 + CUBE > mz["0"].shape[0] or y0 + CUBE > mz["0"].shape[1] or x0 + CUBE > mz["0"].shape[2]:
                    continue
                picks.append((bz, by, bx))
                m = np.asarray(mz["0"][z0:z0 + CUBE, y0:y0 + CUBE, x0:x0 + CUBE]) > 0
                if m.sum() < 1200:
                    continue
                coh = m7_coherence(m)
                surf = m & (coh > tau)
                score = float(surf.sum() / max(m.sum(), 1))
                if score < 0.4:
                    continue
                zc = CUBE // 2
                rec = dict(scroll=sc, m7=m7p, vol=vp, origin_l0=[z0, y0, x0],
                           center_l0=[z0 + zc, y0 + 96, x0 + 96], size=CUBE, material=float(mm[bz, by, bx]),
                           material_thr=mat_thr, coh_score=score, tau=tau, mean_coh=float(coh[m].mean()),
                           judge="m7_normal_coherence>tau -> surface(1); rest -> ignore(2)")
                if images:                                     # capture the centre cross-section for the QC render
                    ctS = np.asarray(vz["0"][z0 + zc, y0:y0 + CUBE, x0:x0 + CUBE]).astype(np.uint8)
                    rec["_r"] = (ctS, m[zc].copy(), surf[zc].copy(), coh[zc].copy())
                passers.append(rec)
                nk += 1
            print(f"{sc}: {nk} pass", flush=True)
        except Exception as e:                                  # noqa: BLE001
            print(f"{sc}: ERR {e}", flush=True)
    passers.sort(key=lambda r: -r["coh_score"])
    sel = passers[:target]
    print(f"\n[m7-mine] TOTAL {len(passers)} passers; approving top {len(sel)}", flush=True)
    for i, r in enumerate(sel):                                # render (optional) + drop the transient slices
        rr = r.pop("_r", None)
        if images and rr is not None:
            r["image"] = _render(out_dir, i, r, *rr)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(sel, f, indent=1)
    print(f"[m7-mine] wrote manifest.json ({len(sel)} regions{', +renders' if images else ''}) -> {out_dir}",
          flush=True)
    if sel:
        _export_from_manifest(os.path.join(out_dir, "manifest.json"), out_dir, K, tau=tau, workers=workers)
    return sel

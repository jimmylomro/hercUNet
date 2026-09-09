"""Per-window ∇φ pseudo-label SAMPLE format — writer/reader for the cloud corpus.

One ``.npz`` per WINDOW holds the base sample + all its augmentations, each a self-contained, marked record.
From a record ``build_gradphi`` regenerates ∇φ + normals; the confidence is stored as 3 (base) / 4 (aug)
SEPARATE channels, each downsampled 4× and quantised to uint8 (a smooth regional weight — see
docs/sheet_membership_embeddings_v3.md §12), so channels can be re-blended later without regenerating.

Layout (npz keys):
  meta_json                       : json blob — schema_version, code_git_sha, params_hash, scroll, level,
                                     voxel_um, coords, org, shape, master_seed, run_id, conf_ds, records[...]
  <k>/cluster_ids                 : int32                    (k = "base" | "aug0" | ...)
  <k>/V_<c>  <k>/foot_<c>  <k>/width_<c>   : mesh grid f16 / bit-packed footprint / per-vertex width f16
  <k>/conf_<chan>                 : uint8 downsampled [zd,yd,xd]   (chan: intersection|sharp2|dropped|perturb)
Each record's meta (kind, source_ids, shape) lives in meta_json["records"][k].
"""
from __future__ import annotations

import json
import numpy as np
from scipy import ndimage as ndi

SCHEMA_VERSION = 1


def downsample_conf(field, ds=4):
    """Confidence field [Z,H,W] float∈[0,1] → uint8 [ceil(Z/ds),...] via order-1 zoom + 0..255 quantise."""
    small = ndi.zoom(np.asarray(field, np.float32), 1.0 / ds, order=1)
    return np.clip(np.rint(small * 255.0), 0, 255).astype(np.uint8)


def save_sample_bundle(path, meta, records):
    """Write a window bundle. ``meta`` dict (provenance); ``records`` = {key: rec} where each rec has
    ``kind``, ``source_ids``, ``meshes`` {c: {V,foot,nu,nv,width}}, and ``conf`` {chan: full-res field}.
    Confidence is downsampled+quantised here. Returns path."""
    ds = int(meta.get("conf_ds", 4))
    out = {}
    rec_meta = {}
    for k, rec in records.items():
        cids = sorted(int(c) for c in rec["meshes"])
        rec_meta[k] = {"kind": rec["kind"], "source_ids": [int(s) for s in rec.get("source_ids", [])],
                       "cluster_ids": cids,
                       "cluster_shapes": {str(c): [int(rec["meshes"][c]["nu"]), int(rec["meshes"][c]["nv"])] for c in cids},
                       "conf_channels": sorted(rec["conf"])}
        out[f"{k}/cluster_ids"] = np.asarray(cids, np.int32)
        for c in cids:
            m = rec["meshes"][c]
            out[f"{k}/V_{c}"] = np.asarray(m["V"], np.float16)
            out[f"{k}/foot_{c}"] = np.packbits(np.asarray(m["foot"]).reshape(-1))
            if m.get("width") is not None:
                out[f"{k}/width_{c}"] = np.asarray(m["width"], np.float16)
        for chan, fld in rec["conf"].items():
            out[f"{k}/conf_{chan}"] = downsample_conf(fld, ds)
    meta = {**meta, "schema_version": SCHEMA_VERSION, "records": rec_meta}
    out["meta_json"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    np.savez_compressed(path, **out)
    return path



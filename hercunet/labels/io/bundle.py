"""Per-window ∇φ pseudo-label SAMPLE format — writers/readers for the corpus + the training export.

Two related on-disk shapes, both one ``.npz`` per window:

* **Training sample** (``save_sample_bundle`` / ``load_sample_bundle``) — the EXPORT format the training
  pipeline consumes. Holds the base record + all its augmentations, each self-contained. From a record
  ``build_gradphi`` regenerates ∇φ + normals; confidence is stored as 3 (base) / 4 (aug) SEPARATE channels,
  each downsampled 4× and quantised to uint8, so channels can be re-blended later without regenerating.

* **Corpus window** (``save_window_bundle`` / ``load_window_bundle``) — the ``.herculabels`` editable unit.
  Holds the base sheet meshes + the ``meshlet`` cloud (resampled points, cluster ids, per-point frames,
  per-unit embedding) and NOTHING dense: no ``bced``/``ff`` (editing never reads them) and no confidence
  (fully derivable from meshes + the meshlet cloud, so it is recomputed on export / shown live, never stored
  and never double-quantised). See docs/herculabels.md.

Training-sample npz keys:
  meta_json                       : json blob — schema_version, code_git_sha, params_hash, scroll, level,
                                     voxel_um, coords, org, shape, master_seed, run_id, conf_ds, records[...]
  <k>/cluster_ids                 : int32                    (k = "base" | "aug0" | ...)
  <k>/V_<c>  <k>/foot_<c>  <k>/width_<c>   : mesh grid f16 / bit-packed footprint / per-vertex width f16
  <k>/conf_<chan>                 : uint8 downsampled [zd,yd,xd]   (chan: intersection|sharp2|dropped|perturb)
Corpus-window npz keys:
  meta_json                       : json blob — as above (+ edited flag, quality), records = {"base": ...}
  base/cluster_ids  base/V_<c>  base/foot_<c>  base/width_<c>   : the base sheet meshes (no conf)
  meshlet/<name>                  : pts f32[P,3] · sid i64[P] · plab i32[P] · normals f16[P,3] · jac f16[P,3,3]
                                    · prop_dist f32[P] · E f32[N,8] · ulab i32[N] · vu f32[] · shape i32[3]
"""
from __future__ import annotations

import json
import numpy as np
from scipy import ndimage as ndi

SCHEMA_VERSION = 1

# meshlet cloud arrays + the dtype each is stored as (frames f16 to halve the cloud; precision ample for refit)
_MESHLET_DTYPES = {"pts": np.float32, "sid": np.int64, "plab": np.int32, "normals": np.float16,
                   "jac": np.float16, "prop_dist": np.float32, "E": np.float32, "ulab": np.int32,
                   "vu": np.float32, "shape": np.int32}


def downsample_conf(field, ds=4):
    """Confidence field [Z,H,W] float∈[0,1] → uint8 [ceil(Z/ds),...] via order-1 zoom + 0..255 quantise."""
    small = ndi.zoom(np.asarray(field, np.float32), 1.0 / ds, order=1)
    return np.clip(np.rint(small * 255.0), 0, 255).astype(np.uint8)


def _pack_meshes(out, prefix, meshes):
    """Write ``meshes`` {c: {V,foot,nu,nv,width}} into ``out`` under ``prefix`` (e.g. "base"). Returns the
    per-mesh shape map {str(c): [nu, nv]} for the record meta."""
    cids = sorted(int(c) for c in meshes)
    out[f"{prefix}/cluster_ids"] = np.asarray(cids, np.int32)
    shapes = {}
    for c in cids:
        m = meshes[c]
        shapes[str(c)] = [int(m["nu"]), int(m["nv"])]
        out[f"{prefix}/V_{c}"] = np.asarray(m["V"], np.float16)
        out[f"{prefix}/foot_{c}"] = np.packbits(np.asarray(m["foot"]).reshape(-1))
        if m.get("width") is not None:
            out[f"{prefix}/width_{c}"] = np.asarray(m["width"], np.float16)
    return shapes


def _unpack_meshes(z, prefix, rec_meta):
    """Reconstruct {c: {V,foot,nu,nv,width}} from an npz ``z`` under ``prefix`` using the record meta
    (``cluster_ids`` + ``cluster_shapes``). Inverse of :func:`_pack_meshes`."""
    from hercunet.labels.mesh.medial import grid_triangles         # verts + tris for the 3-D view / QC
    meshes = {}
    for c in rec_meta["cluster_ids"]:
        c = int(c)
        nu, nv = rec_meta["cluster_shapes"][str(c)]
        nu, nv = int(nu), int(nv)
        V = np.asarray(z[f"{prefix}/V_{c}"], np.float32).reshape(nu, nv, 3)
        foot = np.unpackbits(z[f"{prefix}/foot_{c}"])[:nu * nv].reshape(nu, nv).astype(bool)
        m = dict(V=V, foot=foot, nu=nu, nv=nv, verts=V.reshape(-1, 3)[foot.reshape(-1)],
                 tris=grid_triangles(foot))
        wk = f"{prefix}/width_{c}"
        m["width"] = np.asarray(z[wk], np.float32) if wk in z.files else None
        meshes[c] = m
    return meshes


def save_sample_bundle(path, meta, records):
    """Write a TRAINING sample bundle (the export format). ``meta`` dict (provenance); ``records`` =
    {key: rec} where each rec has ``kind``, ``source_ids``, ``meshes`` {c: {V,foot,nu,nv,width}}, and
    ``conf`` {chan: full-res field}. Confidence is downsampled+quantised here. Returns path."""
    ds = int(meta.get("conf_ds", 4))
    out = {}
    rec_meta = {}
    for k, rec in records.items():
        shapes = _pack_meshes(out, k, rec["meshes"])
        rec_meta[k] = {"kind": rec["kind"], "source_ids": [int(s) for s in rec.get("source_ids", [])],
                       "cluster_ids": sorted(int(c) for c in rec["meshes"]),
                       "cluster_shapes": shapes, "conf_channels": sorted(rec["conf"])}
        for chan, fld in rec["conf"].items():
            out[f"{k}/conf_{chan}"] = downsample_conf(fld, ds)
    meta = {**meta, "schema_version": SCHEMA_VERSION, "records": rec_meta}
    out["meta_json"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    np.savez_compressed(path, **out)
    return path


def load_sample_bundle(path):
    """Read a TRAINING sample bundle → (meta, records) with records = {key: {kind, source_ids, meshes,
    conf}}; ``conf`` channels come back as the stored downsampled uint8 arrays."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(z["meta_json"]).decode())
    records = {}
    for k, rm in meta["records"].items():
        conf = {chan: np.asarray(z[f"{k}/conf_{chan}"]) for chan in rm.get("conf_channels", [])}
        records[k] = dict(kind=rm["kind"], source_ids=rm.get("source_ids", []),
                          meshes=_unpack_meshes(z, k, rm), conf=conf)
    return meta, records


def save_window_bundle(path, meta, meshes, meshlet):
    """Write a CORPUS window (the ``.herculabels`` editable unit): base ``meshes`` + the ``meshlet`` cloud,
    no dense fields, no confidence. ``meta`` carries provenance + ``edited`` + ``quality``. Returns path."""
    out = {}
    shapes = _pack_meshes(out, "base", meshes)
    for name, dt in _MESHLET_DTYPES.items():
        if name in meshlet and meshlet[name] is not None:
            out[f"meshlet/{name}"] = np.asarray(meshlet[name], dt)
    rec_meta = {"base": {"kind": meta.get("kind", "base"), "source_ids": [],
                         "cluster_ids": sorted(int(c) for c in meshes), "cluster_shapes": shapes,
                         "conf_channels": []}}
    meta = {**meta, "schema_version": SCHEMA_VERSION, "records": rec_meta}
    out["meta_json"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    np.savez_compressed(path, **out)
    return path


def load_window_bundle(path):
    """Read a CORPUS window → (meta, meshes, meshlet). ``meshlet`` is a dict of the stored arrays
    (``vu`` returned as a python float, ``shape`` as a 3-tuple)."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(z["meta_json"]).decode())
    meshes = _unpack_meshes(z, "base", meta["records"]["base"])
    meshlet = {}
    for name in _MESHLET_DTYPES:
        key = f"meshlet/{name}"
        if key in z.files:
            meshlet[name] = np.asarray(z[key])
    if "vu" in meshlet:
        meshlet["vu"] = float(np.asarray(meshlet["vu"]).reshape(-1)[0])
    if "shape" in meshlet:
        meshlet["shape"] = tuple(int(s) for s in meshlet["shape"])
    return meta, meshes, meshlet



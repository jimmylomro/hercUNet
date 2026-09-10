"""Per-window extraction: cleaned brick → frame field → 2.5-D meshlets → slab selection →
embedding → probeom clustering. Returns the superset the mesh fit + confidences need.

Ported from the research ``gradphi_confidence.extract_window`` with the viz/debug paths removed;
``brick`` (the CED'd density + voxel size + origin) is always supplied by the caller.
"""

from __future__ import annotations


import numpy as np

from hercunet.labels.meshlets.streamlets3d import (
    _frame_field,
    cluster_streamlets,
    grow_2p5d_streamlets,
    grow_ribs_2p5d,
    sample_streamlet_points,
)
from hercunet.labels.selection.membership import normal_jacobian_batch


def extract_window(args, brick: dict, progress=None) -> dict:
    """Frame → spines+ribs → embedding → probeom for one window.

    ``args`` carries: ``sigma_tensor``, ``gpu``, ``seed_stride_um``, ``sample_um``,
    ``min_cluster_size``. ``brick`` = ``{bced, voxel_um, org}``. Returns the per-point superset
    (points, labels, normals/jac, embedding, propagation distance, full-res fibre field).

    ``progress`` (optional ``callable(stage, payload)``) is an OBSERVER hook for the interactive
    viewer — it never changes the computation. Emits ``"streamlets"`` with the pre-cluster point
    cloud (local z,y,x + spine/rib kind) once growth+sampling finishes, before clustering starts.
    """
    bced = np.ascontiguousarray(brick["bced"], np.float32)
    vu = brick["voxel_um"]
    org = brick["org"]

    ff = _frame_field(bced, args.sigma_tensor, args.gpu)
    material = bced > np.percentile(bced, 55)
    gate = np.asarray(ff["coherence"], np.float32)

    spines = grow_2p5d_streamlets(material, ff["normal"], ff["fibre"], ff["coherence"], vu,
                                  seed_stride_um=args.seed_stride_um, max_len_um=250.0, min_coh=0.05,
                                  seed_coh=0.10, use_gpu=args.gpu)
    ribs, owner = grow_ribs_2p5d(spines, ff["normal"], ff["fibre"], gate, material, vu,
                                 rib_spacing_um=20.0, max_len_um=100.0, min_gate=0.05, use_gpu=args.gpu)
    S = len(spines)
    spts, ssid, _ = sample_streamlet_points(spines, vu, args.sample_um)
    rpts, rsid, _ = sample_streamlet_points(ribs, vu, args.sample_um)
    rowner = np.asarray([owner[i] for i in rsid]) if len(ribs) else np.zeros(0, int)
    pts = np.concatenate([spts, rpts], 0) if len(ribs) else spts
    sid = np.concatenate([ssid, rowner], 0) if len(ribs) else ssid
    kind = (np.concatenate([np.zeros(len(spts), int), np.ones(len(rpts), int)]) if len(ribs)
            else np.zeros(len(spts), int))
    cent = np.zeros((S, 3)); cnt = np.zeros(S)
    np.add.at(cent, sid, pts); np.add.at(cnt, sid, 1.0); cent /= np.maximum(cnt[:, None], 1.0)
    print(f"grew {S} spines + {len(ribs)} ribs → {len(pts)} pts", flush=True)
    if progress is not None:                                           # observer hook (viewer) — pre-cluster cloud
        progress("streamlets", {"pts": pts.astype(np.float32), "kind": kind.astype(np.int8)})

    lab, nB, info = cluster_streamlets(spines, bced, ff, vu, sample_um=args.sample_um, sigma_n_um=25.0,
                                       pos_pull=1.0, seed=0, presampled=(pts, sid, cent), n_units=S,
                                       max_sheet_um=220.0, sigma_s_um=180.0, ptype=kind, loss_mode="triplet",
                                       vic_inv=25.0, vic_var=0.1, vic_cov=0.01, width_lambda_um=30.0,
                                       width_steep_um=12.0, samples_per_point=80, dim=8, neg_tau=0.32,
                                       cluster_selection_method="probeom",
                                       min_cluster_size=getattr(args, "min_cluster_size", 250),
                                       slab_negatives=getattr(args, "slab_negatives", False),
                                       use_gpu=args.gpu, verbose=True)
    prop_dist = np.asarray(info["prop_dist"], np.float32)              # per-UNIT E-dist to core (0 = core)
    ipts, isid = np.asarray(info["pts"]), np.asarray(info["sid"])      # canonical point cloud + unit ids
    plab = lab[isid]                                                   # per-point cluster label (de-noised)

    nf = ff["normal"] / (np.linalg.norm(ff["normal"], axis=-1, keepdims=True) + 1e-9)
    Z, Y, X = nf.shape[:3]
    pi = np.round(ipts).astype(int)
    pi[:, 0] = pi[:, 0].clip(0, Z - 1); pi[:, 1] = pi[:, 1].clip(0, Y - 1); pi[:, 2] = pi[:, 2].clip(0, X - 1)
    normals = nf[pi[:, 0], pi[:, 1], pi[:, 2]]
    jac = normal_jacobian_batch(nf, ipts, normals)
    return dict(bced=bced, vu=vu, org=org, pts=ipts.astype(np.float32), sid=isid.astype(np.int64),
                plab=plab.astype(np.int32), normals=normals.astype(np.float32), jac=jac.astype(np.float32),
                E=np.asarray(info["E"], np.float32), prop_dist=prop_dist[isid].astype(np.float32),
                ff_normal=np.asarray(ff["normal"], np.float32), ff_coherence=np.asarray(ff["coherence"], np.float32),
                nB=int(nB))

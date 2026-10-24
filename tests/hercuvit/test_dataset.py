"""HercuViT dataset — resample FOV/shape, geometric augmentation, and a full __getitem__ on a synthetic corpus
(the real label pipeline — load_sample_bundle / build_gradphi_edt — is monkeypatched to stay offline)."""
import json
import os

import numpy as np
import torch

import hercunet.hercuvit.data.dataset as DS
from hercunet.hercuvit.data.dataset import (HercuvitDataset, geometric_augment, hercuvit_collate,
                                            resample_to_crop)


def test_resample_to_crop_shape_and_labels():
    ct = np.random.rand(20, 20, 20).astype(np.float32)
    owner = np.zeros((20, 20, 20), np.int32)
    owner[8:12, 8:12, 8:12] = 5
    ct_r, labs = resample_to_crop(ct, {"owner": owner, "surf": (owner > 0).astype(np.int32)},
                                  voxel_um=8.0, target_voxel_um=9.6, crop=16)
    assert ct_r.shape == (16, 16, 16) and labs["owner"].shape == (16, 16, 16)
    assert ct_r.dtype == np.float32 and labs["owner"].dtype == np.int64
    assert set(np.unique(labs["owner"]).tolist()) <= {0, 5} and (labs["owner"] == 5).any()


def test_geometric_augment_is_label_consistent():
    rng = np.random.default_rng(3)
    ct = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
    owner = ct.astype(np.int64) % 3
    ct2, labs = geometric_augment(ct.copy(), {"owner": owner.copy()}, rng)
    assert ct2.shape == ct.shape and labs["owner"].shape == owner.shape
    # the SAME spatial transform on both: owner id at each voxel still matches ct%3 after the transform.
    assert np.array_equal(labs["owner"], ct2.astype(np.int64) % 3)


def test_dataset_getitem_offline(tmp_path, monkeypatch):
    shape = (20, 20, 20)
    H = shape[1]
    meta = {
        "scroll": "demo", "level": 0, "voxel_um": 8.0, "org": [0, 0, 0], "coords": [0, 0, 0],
        "shape": list(shape), "conf_ds": 4,
        "records": {"base": {"kind": "base", "cluster_ids": [0, 1]}},
        "quality": {"per_sheet": {"0": 0.9, "1": 0.9}},
    }
    mb = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    np.savez(os.path.join(str(tmp_path), "sample_sdemo_L0_z0_y0_x0.npz"), meta_json=mb)

    def fake_bundle(path):
        return meta, {"base": {"meshes": {0: None, 1: None}, "conf": {}}}

    def fake_edt(meshes, shp, vu, return_owner_full=False):
        mag = np.zeros(shp, np.float32); mag[:, H // 2, :] = 1.0           # a surface plane (|∇φ| >= tau)
        lab3 = np.full(shp, -1, np.int32); lab3[:, H // 2, :] = 0
        owner_full = np.zeros(shp, np.int32); owner_full[:, H // 2, :] = 1
        return (mag, lab3, None, None, owner_full) if return_owner_full else (mag, lab3, None, None)

    monkeypatch.setattr(DS, "load_sample_bundle", fake_bundle)
    # labels are now built by the SHARED hercunet.data.dataset.build_surface_owner, which imports build_gradphi_edt
    # locally from hercunet.labels.mesh.medial — patch it at the source so the real surf/owner construction runs.
    monkeypatch.setattr("hercunet.labels.mesh.medial.build_gradphi_edt", fake_edt)

    def fake_reader(scroll, level, org, shape):
        return np.random.default_rng(7).normal(100.0, 20.0, tuple(int(s) for s in shape)).astype(np.float32)

    ds = HercuvitDataset(str(tmp_path), target_voxel_um=9.6, crop=16, augment=True,
                         ct_reader=fake_reader, gate_count=0)
    assert len(ds) == 1
    it = ds[0]
    assert it["ct"].shape == (1, 16, 16, 16) and it["sem"].shape == (16, 16, 16) and it["owner"].shape == (16, 16, 16)
    assert it["has_identity"] is True
    assert set(np.unique(it["sem"].numpy()).tolist()) <= {0, 1, 2}        # 3-class {bg, surface, ignore}
    assert int((it["sem"] == 1).sum()) > 0                                # surface present

    batch = hercuvit_collate([ds[0], ds[0]])
    assert batch["ct"].shape == (2, 1, 16, 16, 16) and batch["has_identity"] == [True, True]

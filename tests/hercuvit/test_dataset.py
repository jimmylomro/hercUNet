"""HercuViT dataset — resample FOV/shape, geometric augmentation, and a full __getitem__ on a synthetic corpus."""
import json
import os

import numpy as np
import torch

from hercunet.hercuvit.data.dataset import (HercuvitDataset, geometric_augment, hercuvit_collate,
                                            resample_to_crop)


def test_resample_to_crop_shape_and_labels():
    torch.manual_seed(0)
    ct = np.random.rand(20, 20, 20).astype(np.float32)
    owner = np.zeros((20, 20, 20), np.int32)
    owner[8:12, 8:12, 8:12] = 5                      # a labelled block
    ct_r, own_r = resample_to_crop(ct, owner, voxel_um=8.0, target_voxel_um=9.6, crop=16)
    assert ct_r.shape == (16, 16, 16) and own_r.shape == (16, 16, 16)
    assert ct_r.dtype == np.float32 and own_r.dtype == np.int64
    # nearest resampling preserves labels as a subset of the originals (no interpolation-invented ids).
    assert set(np.unique(own_r).tolist()) <= {0, 5}
    assert (own_r == 5).any()


def test_geometric_augment_is_label_consistent():
    rng = np.random.default_rng(3)
    ct = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
    owner = (ct.astype(np.int64) % 3)
    ct2, owner2 = geometric_augment(ct.copy(), owner.copy(), rng)
    assert ct2.shape == ct.shape and owner2.shape == owner.shape
    # the SAME spatial transform on both: the owner id at each voxel still matches ct%3 after the transform.
    assert np.array_equal(owner2, ct2.astype(np.int64) % 3)


def _write_synthetic_sample(path, shape=(20, 20, 20), voxel_um=8.0):
    """Minimal herculabels npz (base record with one sheet) — matches hercunet.tracer.labels.load_sample."""
    D, H, W = shape
    # a thin diagonal sheet of surface points in local (z,y,x) voxel coords
    zz = np.arange(4, D - 4)
    V0 = np.stack([zz, zz, np.full_like(zz, W // 2)], axis=1).astype(np.float16)
    meta = {
        "scroll": "demo", "level": 0, "voxel_um": voxel_um,
        "coords": [0, 0, 0], "org": [0, 0, 0], "shape": list(shape), "conf_ds": 4,
        "records": {"base": {"kind": "base", "cluster_ids": [0]}},
        "quality": {"per_sheet": {}},
    }
    meta_bytes = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    np.savez(path, **{
        "base/cluster_ids": np.array([0], np.int32),
        "base/V_0": V0,
        "meta_json": meta_bytes,
    })


def test_dataset_getitem_offline(tmp_path):
    torch.manual_seed(0)
    _write_synthetic_sample(os.path.join(str(tmp_path), "sample_sdemo_L0_z0_y0_x0.npz"))

    def fake_reader(scroll, level, org, shape):      # inject synthetic CT (no S3)
        rng = np.random.default_rng(7)
        return rng.normal(100.0, 20.0, size=tuple(int(s) for s in shape)).astype(np.float32)

    ds = HercuvitDataset(str(tmp_path), target_voxel_um=9.6, crop=16, augment=True, ct_reader=fake_reader)
    assert len(ds) == 1
    item = ds[0]
    assert item["ct"].shape == (1, 16, 16, 16)
    assert item["sem"].shape == (16, 16, 16) and item["owner"].shape == (16, 16, 16)
    assert item["has_identity"] is True
    assert item["ct"].dtype == torch.float32 and item["sem"].dtype == torch.long
    # the sheet rasterised → some positive semantic voxels; sem is exactly owner>0.
    assert int(item["sem"].sum()) > 0
    assert torch.equal(item["sem"], (item["owner"] > 0).long())

    batch = hercuvit_collate([ds[0], ds[0]])
    assert batch["ct"].shape == (2, 1, 16, 16, 16)
    assert batch["has_identity"] == [True, True]

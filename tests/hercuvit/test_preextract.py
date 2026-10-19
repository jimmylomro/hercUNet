"""HercuViT training pre-extraction — CLI wiring, LocalHercuvitDataset round-trip, cache idempotency (CPU, no S3).

The sharded S3 read + ``build_gradphi_edt`` label build (``prep_train_shard``) needs the live bucket + the corpus
meshes, so it is validated on a pod, not in CI. Here we test the pure local pieces: the trainer reads a native
cache the same way the dataset writes it, and the cache orchestrator is idempotent."""
import numpy as np
import pytest


def _write_native_window(path, *, shape=(40, 44, 48), voxel_um=8.0, has_identity=True, seed=0):
    """Write one synthetic native ``win_*.npz`` exactly as ``prep_train_shard`` does (ct f16 / surf u8 / owner i32)."""
    rng = np.random.default_rng(seed)
    ct = rng.standard_normal(shape).astype(np.float16)
    surf = rng.integers(0, 3, size=shape, dtype=np.uint8)              # {0,1,2}
    owner = rng.integers(0, 5, size=shape, dtype=np.int32)             # instance ids
    np.savez_compressed(path, ct=ct, surf=surf, owner=owner,
                        voxel_um=np.float32(voxel_um), has_identity=np.bool_(has_identity))


def test_local_dataset_roundtrip_centred(tmp_path):
    from hercunet.hercuvit.data.dataset import LocalHercuvitDataset
    for i in range(3):
        _write_native_window(tmp_path / f"win_sample_{i}.npz", seed=i)
    ds = LocalHercuvitDataset(str(tmp_path), augment=False, target_voxel_um=9.6, crop=16)
    assert len(ds) == 3
    item = ds[0]
    assert item["ct"].shape == (1, 16, 16, 16)                        # (C, crop, crop, crop)
    assert item["sem"].shape == (16, 16, 16)
    assert item["owner"].shape == (16, 16, 16)
    assert item["ct"].dtype.is_floating_point
    assert item["sem"].dtype == __import__("torch").long
    assert isinstance(item["has_identity"], bool) and item["has_identity"] is True
    assert set(np.unique(item["sem"].numpy())).issubset({0, 1, 2})    # labels survive resample (nearest)


def test_local_dataset_augment_runs(tmp_path):
    from hercunet.hercuvit.data.dataset import LocalHercuvitDataset
    _write_native_window(tmp_path / "win_a.npz")
    ds = LocalHercuvitDataset(str(tmp_path), augment=True, target_voxel_um=9.6, crop=16)
    item = ds[0]                                                      # augment path (random crop offset + flip) runs
    assert item["ct"].shape == (1, 16, 16, 16) and item["owner"].shape == (16, 16, 16)


def test_local_dataset_empty_dir_raises(tmp_path):
    from hercunet.hercuvit.data.dataset import LocalHercuvitDataset
    with pytest.raises(FileNotFoundError):
        LocalHercuvitDataset(str(tmp_path))


def test_ensure_train_cache_idempotent_when_done(tmp_path, monkeypatch):
    """A ready cache (``.done`` + files) is reused WITHOUT spawning any shard process."""
    import subprocess
    from hercunet.hercuvit.train import preextract
    _write_native_window(tmp_path / "win_x.npz")
    (tmp_path / ".done").write_text("")
    called = {"popen": 0}
    monkeypatch.setattr(subprocess, "Popen",
                        lambda *a, **k: called.__setitem__("popen", called["popen"] + 1))
    out = preextract._ensure_train_cache(corpus_dir="unused", cache_dir=str(tmp_path), nshards=4)
    assert out == str(tmp_path)
    assert called["popen"] == 0                                       # cache hit → no subprocess launched


def test_train_prep_shard_cli_wiring():
    from hercunet.hercuvit.cli import build_parser
    p = build_parser()
    a = p.parse_args(["train", "prep-shard", "--corpus", "c", "--cache-dir", "d", "--nshards", "3", "--shard", "1"])
    assert a.corpus == "c" and a.cache_dir == "d" and a.nshards == 3 and a.shard == 1
    assert a.no_identity_only is False                                # identity-only by default (matches the dataset)
    assert callable(a.func)


def test_train_common_preextract_flags():
    from hercunet.hercuvit.cli import build_parser
    p = build_parser()
    b = p.parse_args(["train", "fit", "--corpus", "c", "--encoder", "e", "--out", "o"])
    assert b.no_pre_extract is False                                  # pre-extraction ON by default
    assert b.train_cache is None and b.prep_shards == 8
    c = p.parse_args(["train", "chain", "--corpus-hf", "--encoder-hf", "--out", "o",
                      "--no-pre-extract", "--train-cache", "/tmp/tc", "--prep-shards", "6"])
    assert c.no_pre_extract is True and c.train_cache == "/tmp/tc" and c.prep_shards == 6

"""Unit tests for the .asv sheet store (:mod:`hercunet.meshes.asv`).

Pins the format invariants: grid-implicit triangles reconstruct exactly from the footprint (nothing geometric is
lost there), the only loss is the uint16 vertex quantisation inside each sheet's bbox (bounded by the quant step),
blobs are self-describing across codecs, and the writer/reader/merge round-trip preserves footprint + geometry +
the rtree bbox query.
"""
import numpy as np
import pytest

from hercunet.meshes import asv
from hercunet.meshes.fit import grid_triangles


def _synthetic(nu=7, nv=5, seed=0, hole=False):
    rng = np.random.default_rng(seed)
    foot = np.ones((nu, nv), bool)
    if hole:
        foot[nu // 2, nv // 2] = False                                 # interior hole -> non-trivial remap
    nvert = int(foot.sum())
    verts = rng.uniform([10.0, 20.0, 30.0], [15.0, 28.0, 45.0], size=(nvert, 3)).astype(np.float32)
    return foot, verts


def _quant_tol(verts):
    rng = np.maximum(verts.max(0) - verts.min(0), 1e-6)
    return float((rng / 65535.0).max()) + 1e-5


def test_pack_unpack_roundtrip():
    foot, verts = _synthetic()
    blob, lo, hi = asv.pack_sheet(foot, verts)
    g = asv.unpack_sheet(blob)
    assert g["nu"] == foot.shape[0] and g["nv"] == foot.shape[1]
    assert (g["foot"] == foot).all()
    assert np.abs(g["verts"] - verts).max() <= _quant_tol(verts)       # only the quant step is lost
    assert g["tris"].min() >= 0 and g["tris"].max() < len(g["verts"])  # tris index the footprint verts
    assert len(g["tris"]) == len(grid_triangles(foot))                 # triangle count reconstructed exactly


def test_hole_footprint_remap():
    foot, verts = _synthetic(hole=True)
    g = asv.unpack_sheet(asv.pack_sheet(foot, verts)[0])
    assert (g["foot"] == foot).all()
    assert g["tris"].max() < int(foot.sum())                           # no triangle references the hole cell
    assert len(g["tris"]) == len(grid_triangles(foot))


def test_codec_self_describing(monkeypatch):
    foot, verts = _synthetic()
    monkeypatch.setattr(asv, "_WRITE_CODEC", asv._ZLIB)               # force zlib blob
    blob = asv.pack_sheet(foot, verts)[0]
    assert blob[0] == asv._ZLIB
    g = asv.unpack_sheet(blob)                                        # still readable (dispatch on the codec byte)
    assert (g["foot"] == foot).all()


def test_writer_reader_and_bbox(tmp_path):
    foot, verts = _synthetic()
    p = tmp_path / "x.asv"
    frame = {"vc_open_data_coordinate_space": "PHercX/scan"}
    with asv.AsvWriter(p, frame=frame, voxel_um=8.64, grid_step_um=30.0) as w:
        w.add_sheet(foot, verts, name="s1", window="w0")
        w.add_sheet(foot, verts + 100.0, name="s2", window="w0")      # shifted far away
    r = asv.AsvReader(p)
    try:
        md = r.metadata
        assert md["complete"] == "true" and md["format_version"] == "1"
        assert int(md["sheet_count"]) == 2 and int(md["vert_count"]) == 2 * int(foot.sum())
        assert len(r) == 2
        ids = r.ids_in_box(verts.min(0), verts.max(0))                # box around s1 only
        assert 1 in ids and 2 not in ids
        s = r.sheet(1)
        assert s["name"] == "s1" and s["window"] == "w0" and (s["foot"] == foot).all()
    finally:
        r.close()


def test_writer_refuses_bad_extension(tmp_path):
    with pytest.raises(ValueError):
        asv.AsvWriter(tmp_path / "x.afv", frame={"vc_open_data_coordinate_space": "s"}, voxel_um=8.0, grid_step_um=30.0)


def test_merge_shards(tmp_path):
    foot, verts = _synthetic()
    frame = {"vc_open_data_coordinate_space": "PHercX/scan"}
    shards = []
    for i in (1, 2, 3):
        sp = tmp_path / f"sh{i}.asv"
        with asv.AsvWriter(sp, frame=frame, voxel_um=8.64, grid_step_um=30.0) as w:
            w.add_sheet(foot, verts + 10.0 * i, name=f"s{i}")
        shards.append(sp)
    info = asv.merge_asv(shards, tmp_path / "merged.asv", frame=frame, voxel_um=8.64, grid_step_um=30.0)
    assert info["sheet_count"] == 3
    r = asv.AsvReader(tmp_path / "merged.asv")
    try:
        assert len(r) == 3
        assert sorted(s["name"] for s in r) == ["s1", "s2", "s3"]      # ids renumbered 1..3, names preserved
        assert len(r.ids_in_box((-1e9,) * 3, (1e9,) * 3)) == 3         # rtree rebuilt in the merged file
    finally:
        r.close()

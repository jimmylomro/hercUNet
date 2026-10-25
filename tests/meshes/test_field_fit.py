"""Unit tests for the KNN-free field sheet fit (:func:`hercunet.meshes.field_fit.fit_sheets_field`), on CPU.

Pins: it fits each gated sheet's grid onto the sheet (verts land on the plane), reconstructs valid triangles, and
applies the min-points / degenerate drops (the field-fit analogue of the slab fit's drops)."""
import numpy as np

from hercunet.meshes.field_fit import fit_sheets_field


def _two_planes(D=36, z1=12, z2=24, th=2):
    owner = np.zeros((D, D, D), np.int64)
    gate = np.zeros((D, D, D), bool)
    for zc, sid in [(z1, 1), (z2, 2)]:
        owner[zc - th:zc + th + 1] = sid
        gate[zc - th:zc + th + 1] = True
    return owner, gate


def test_fits_two_planes():
    owner, gate = _two_planes()
    m = fit_sheets_field(owner, gate, voxel_um=8.0, iters=25, device="cpu")
    assert set(m) == {1, 2}
    assert abs(m[1]["verts"][:, 0].mean() - 12) < 2.0                   # verts land on the z=12 plane
    assert abs(m[2]["verts"][:, 0].mean() - 24) < 2.0
    for mm in m.values():                                              # tris index the FULL grid V (nu*nv), not verts
        assert len(mm["tris"]) > 0 and mm["tris"].min() >= 0 and mm["tris"].max() < mm["nu"] * mm["nv"]


def test_min_points_drop():
    owner, gate = _two_planes()
    owner[0, 0, 0] = 3; gate[0, 0, 0] = True                           # a 1-voxel sheet -> dropped by min_pts
    m = fit_sheets_field(owner, gate, voxel_um=8.0, iters=10, device="cpu", min_pts=400)
    assert 3 not in m and set(m) == {1, 2}


def test_empty_returns_empty():
    owner = np.zeros((20, 20, 20), np.int64); gate = np.zeros((20, 20, 20), bool)
    assert fit_sheets_field(owner, gate, voxel_um=8.0, device="cpu") == {}

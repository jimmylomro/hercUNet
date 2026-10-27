"""Automated Sheet Volumes (.asv) — a compact, random-access store of fitted sheet meshes.

A sibling of VC3D's Automated **Fibre** Volumes (.afv): the same container idea — a single read-only SQLite file
with a ``metadata`` table, a spatially-indexed collection, and native-L0 XYZ coordinates — but the payload is our
sheet *meshes*, not fibre polylines. We own this format (there is no VC3D sheet standard yet) and use it as the
geometry store for the unwrapping pipeline, written one sheet at a time by inference and queried by bounding box.

Per-sheet geometry is stored **grid-implicit + quantised** (see :mod:`hercunet.meshes.fit`): a sheet is a regular
``nu×nv`` grid with a footprint mask, so the triangles need not be stored (they reconstruct exactly from the
footprint via :func:`hercunet.meshes.fit.grid_triangles`), and the footprint-only vertices are quantised to uint16
inside the sheet's own bounding box (lossless triangulation; sub-0.01 µm vertex error at a 30 µm grid). The packed
blob is then zstd-compressed. A mid-size sheet is ~6–13 kB packed vs ~70 kB as naive float32 verts + int32 tris.

Schema (format version 1)::

    metadata(key TEXT PRIMARY KEY, value TEXT)          -- uuid, frame, root, counts, voxel_um, grid_step_um, codec
    sheets(id, name, window, family, nu, nv, vert_count,
           min_x..max_z REAL, payload BLOB, annotation)  -- one row per sheet instance; bbox columns for cheap filter
    sheet_bounds USING rtree(id, min_x,max_x, ...)        -- spatial index for bbox queries

Coordinates are native L0 voxel XYZ (x, y, z — matching .afv), so sheets from every window live in one global space
and interoperate with fibre collections. The caller converts window-local verts to global L0 before adding them.
"""
from __future__ import annotations

import json
import os
import sqlite3
import struct
import tempfile
import uuid
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

from .fit import grid_triangles

FORMAT_VERSION = 1
_MAGIC = b"ASV1"

# Each blob is self-describing: its first byte is the codec id, so a file written with zstd on one box still reads on
# a zlib-only box (and vice versa) — the reader never depends on what is installed where it writes. zlib is always
# available and is the default; zstd is used when importable (marginally smaller on already-quantised verts).
_ZLIB, _ZSTD = 1, 2
_CODEC_NAME = {_ZLIB: "zlib", _ZSTD: "zstd"}

import zlib as _zlib
_COMP = {_ZLIB: lambda b: _zlib.compress(b, 9)}
_DECOMP = {_ZLIB: _zlib.decompress}
try:
    import zstandard as _zstd
    _COMP[_ZSTD] = lambda b: _zstd.ZstdCompressor(level=19).compress(b)
    _DECOMP[_ZSTD] = lambda b: _zstd.ZstdDecompressor().decompress(b)
    _WRITE_CODEC = _ZSTD
except Exception:                                                     # pragma: no cover - environment-dependent
    _WRITE_CODEC = _ZLIB

SCHEMA = """
CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE sheets(
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, window TEXT NOT NULL DEFAULT '', family TEXT NOT NULL DEFAULT '',
  nu INTEGER NOT NULL, nv INTEGER NOT NULL, vert_count INTEGER NOT NULL CHECK(vert_count >= 3),
  -- quality columns (nullable): area_mm2 = true surface area; n_voxels = gated voxel support; self_mass =
  -- sum own-membership (confidence-weighted size); confidence = self_mass/n_voxels (mean own-membership);
  -- mass_frac = self_mass / window max (the prune's size term, normalised); fit_rms_um = RMS distance from the
  -- gated mask voxels to the fitted mesh (mesh-vs-mask displacement); coverage = fraction of mask voxels within
  -- one grid step of the mesh (how much of the mask the mesh covers, 0-1).
  area_mm2 REAL, n_voxels INTEGER, self_mass REAL, confidence REAL, mass_frac REAL,
  fit_rms_um REAL, coverage REAL,
  min_x REAL NOT NULL, max_x REAL NOT NULL, min_y REAL NOT NULL, max_y REAL NOT NULL,
  min_z REAL NOT NULL, max_z REAL NOT NULL, payload BLOB NOT NULL, annotation TEXT NOT NULL DEFAULT '{}');
CREATE VIRTUAL TABLE sheet_bounds USING rtree(id, min_x, max_x, min_y, max_y, min_z, max_z);
"""


# --------------------------------------------------------------------------------------------------------------------
# packing: one sheet mesh  <->  one compressed blob (triangles implicit, verts uint16 in the sheet bbox)
# --------------------------------------------------------------------------------------------------------------------
def pack_sheet(foot: np.ndarray, verts: np.ndarray) -> tuple[bytes, np.ndarray, np.ndarray]:
    """Pack one sheet into its on-disk blob. ``foot`` (nu,nv) bool footprint; ``verts`` (nvert,3) float XYZ, the
    footprint-only vertices in ``foot.ravel()`` order (as :func:`fit_sheet_meshes` emits ``mesh['verts']``, but with
    columns ordered x,y,z). Returns ``(blob, lo, hi)`` — ``lo``/``hi`` are the float32 bbox corners (for the row's
    bbox columns). Lossy only in the uint16 vertex quantisation inside [lo,hi]; triangles are not stored."""
    foot = np.ascontiguousarray(foot, dtype=bool)
    verts = np.ascontiguousarray(verts, dtype=np.float32)
    nu, nv = foot.shape
    nvert = int(foot.sum())
    if verts.shape != (nvert, 3):
        raise ValueError(f"verts {verts.shape} does not match footprint vertex count {nvert}")
    lo = verts.min(0).astype("<f4"); hi = verts.max(0).astype("<f4")
    rng = np.maximum(hi - lo, 1e-6)
    vq = ((verts - lo) / rng * 65535.0).round().clip(0, 65535).astype("<u2")
    raw = (_MAGIC + struct.pack("<HH", nu, nv) + lo.tobytes() + hi.tobytes()
           + np.packbits(foot.ravel()).tobytes() + vq.tobytes())
    return bytes([_WRITE_CODEC]) + _COMP[_WRITE_CODEC](raw), lo, hi


def unpack_sheet(blob: bytes) -> dict[str, Any]:
    """Inverse of :func:`pack_sheet`. Returns ``{'nu','nv','verts' (nvert,3 f32 XYZ),'tris' (ntri,3 int) indexing
    verts,'foot' (nu,nv bool)}``. Triangles are reconstructed from the footprint, so they are exact."""
    codec = blob[0]
    if codec not in _DECOMP:
        raise ValueError(f"sheet blob needs codec {_CODEC_NAME.get(codec, codec)} which is not available here")
    raw = _DECOMP[codec](blob[1:])
    if raw[:4] != _MAGIC:
        raise ValueError("not an .asv sheet blob")
    nu, nv = struct.unpack_from("<HH", raw, 4)
    off = 8
    lo = np.frombuffer(raw, "<f4", 3, off); off += 12
    hi = np.frombuffer(raw, "<f4", 3, off); off += 12
    nbits = (nu * nv + 7) // 8
    foot = np.unpackbits(np.frombuffer(raw, np.uint8, nbits, off), count=nu * nv).astype(bool).reshape(nu, nv)
    off += nbits
    nvert = int(foot.sum())
    vq = np.frombuffer(raw, "<u2", nvert * 3, off).reshape(nvert, 3).astype(np.float32)
    rng = np.maximum(hi - lo, 1e-6)
    verts = vq / 65535.0 * rng + lo
    tris_flat = grid_triangles(foot)                                  # indices into the flat nu*nv grid
    remap = np.full(nu * nv, -1, np.int64); remap[foot.ravel()] = np.arange(nvert)   # flat-grid -> footprint index
    tris = remap[tris_flat]
    return dict(nu=nu, nv=nv, verts=verts.astype(np.float32), tris=tris, foot=foot)


def sheet_area_mm2(foot: np.ndarray, verts: np.ndarray, voxel_um: float) -> float:
    """True surface area of a sheet mesh in mm^2. ``verts`` are footprint-only L0 voxel XYZ; triangles come from the
    footprint, so the cross-product areas are in voxel^2, scaled by (voxel_um/1000)^2."""
    foot = np.ascontiguousarray(foot, dtype=bool)
    nvert = int(foot.sum())
    remap = np.full(foot.size, -1, np.int64)
    remap[foot.ravel()] = np.arange(nvert)
    tri = remap[grid_triangles(foot)]                                 # footprint-complete tris -> vertex indices
    if tri.size == 0:
        return 0.0
    v = np.asarray(verts, dtype=np.float64)[tri]                      # (ntri,3,3)
    area_vox2 = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1).sum()
    return float(area_vox2) * (float(voxel_um) / 1000.0) ** 2


# --------------------------------------------------------------------------------------------------------------------
# writer: stream sheets into a file, publish atomically on close (mirrors afv's temp-then-link)
# --------------------------------------------------------------------------------------------------------------------
class AsvWriter:
    """Incrementally write an .asv. Open, :meth:`add_sheet` per fitted sheet (one window at a time), then
    :meth:`close` to stamp ``complete`` and publish atomically. ``frame`` must carry the open-data coordinate space
    (e.g. ``{"vc_open_data_coordinate_space": "PHerc1447/20250521151220"}``); ``voxel_um`` is the L0 voxel size and
    ``grid_step_um`` the mesh grid spacing. Not safe for concurrent writers — in distributed inference each worker
    writes its own shard and the leader :func:`merge_asv` merges them."""

    def __init__(self, path: str | os.PathLike, *, frame: Mapping[str, Any], voxel_um: float,
                 grid_step_um: float, root: Mapping[str, Any] | None = None,
                 metadata: Mapping[str, Any] | None = None, overwrite: bool = False):
        self.destination = Path(path)
        if self.destination.suffix.lower() != ".asv":
            raise ValueError("Automated Sheet Volumes must use the .asv extension")
        if self.destination.exists() and not overwrite:
            raise FileExistsError(self.destination)
        if not frame.get("vc_open_data_coordinate_space"):
            raise ValueError("frame needs vc_open_data_coordinate_space")
        fd, name = tempfile.mkstemp(prefix=f".{self.destination.name}.", suffix=".partial", dir=self.destination.parent)
        os.close(fd)
        self._temporary = Path(name)
        self._db = sqlite3.connect(str(self._temporary))
        self._db.executescript(SCHEMA)
        self._n = 0
        self._vert_total = 0
        self._frame = dict(frame)
        self._root = dict(root or {"type": "hercuvit_sheet_collection", "version": FORMAT_VERSION})
        self._voxel_um = float(voxel_um)
        self._grid_step_um = float(grid_step_um)
        self._extra = dict(metadata or {})
        self._closed = False

    def add_sheet(self, foot: np.ndarray, verts: np.ndarray, *, name: str = "", window: str = "",
                  family: str = "", n_voxels: int | None = None, self_mass: float | None = None,
                  confidence: float | None = None, mass_frac: float | None = None,
                  fit_rms_um: float | None = None, coverage: float | None = None,
                  annotation: Mapping[str, Any] | None = None) -> int:
        """Add one sheet (``foot`` footprint, ``verts`` footprint-only global-L0 XYZ). ``area_mm2`` is computed from
        the mesh; the remaining quality stats are stored in their columns when given. Returns its row id."""
        blob, lo, hi = pack_sheet(foot, verts)
        self._n += 1
        sid = self._n
        nvert = int(foot.sum())
        self._vert_total += nvert
        area_mm2 = sheet_area_mm2(foot, verts, self._voxel_um)
        ann = json.dumps(dict(annotation or {}), separators=(",", ":"), allow_nan=False)
        f = lambda x: None if x is None else float(x)
        self._db.execute(
            "INSERT INTO sheets(id,name,window,family,nu,nv,vert_count,area_mm2,n_voxels,self_mass,confidence,"
            "mass_frac,fit_rms_um,coverage,min_x,max_x,min_y,max_y,min_z,max_z,payload,annotation) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (sid, name or f"sheet_{sid}", window, family, int(foot.shape[0]), int(foot.shape[1]), nvert,
             float(area_mm2), None if n_voxels is None else int(n_voxels), f(self_mass), f(confidence),
             f(mass_frac), f(fit_rms_um), f(coverage),
             float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), float(lo[2]), float(hi[2]), blob, ann))
        self._db.execute("INSERT INTO sheet_bounds VALUES(?,?,?,?,?,?,?)",
                         (sid, float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), float(lo[2]), float(hi[2])))
        return sid

    def close(self) -> dict[str, Any]:
        """Stamp metadata (``complete=true``), commit, and publish ``destination`` atomically."""
        if self._closed:
            return {"uuid": None}
        uid = str(uuid.uuid4())
        meta = {"uuid": uid, "format_version": str(FORMAT_VERSION), "complete": "true",
                "codec": _CODEC_NAME[_WRITE_CODEC],
                "frame": json.dumps(self._frame, separators=(",", ":")),
                "root": json.dumps(self._root, separators=(",", ":")),
                "voxel_um": repr(self._voxel_um), "grid_step_um": repr(self._grid_step_um),
                "sheet_count": str(self._n), "vert_count": str(self._vert_total),
                **{k: json.dumps(v) if not isinstance(v, str) else v for k, v in self._extra.items()}}
        self._db.executemany("INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)", list(meta.items()))
        self._db.commit()
        self._db.close()
        self._closed = True
        _publish(self._temporary, self.destination, overwrite=True)
        return {"uuid": uid, "sheet_count": self._n, "vert_count": self._vert_total}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.close()
        else:                                                         # leave the .partial for debugging on error
            self._db.close()


def _publish(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(destination)
        destination.unlink()
    os.replace(temporary, destination)


# --------------------------------------------------------------------------------------------------------------------
# reader + leader-side shard merge
# --------------------------------------------------------------------------------------------------------------------
class AsvReader:
    """Read an .asv: metadata, bounding-box queries, and per-sheet geometry (verts + reconstructed tris)."""

    def __init__(self, path: str | os.PathLike):
        self._db = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)

    @property
    def metadata(self) -> dict[str, str]:
        return dict(self._db.execute("SELECT key,value FROM metadata").fetchall())

    def __len__(self) -> int:
        return int(self._db.execute("SELECT COUNT(*) FROM sheets").fetchone()[0])

    def ids_in_box(self, lo_xyz, hi_xyz) -> list[int]:
        """Row ids whose bbox intersects the query box (rtree)."""
        x0, y0, z0 = (float(v) for v in lo_xyz); x1, y1, z1 = (float(v) for v in hi_xyz)   # sqlite won't match np scalars
        rows = self._db.execute(
            "SELECT id FROM sheet_bounds WHERE max_x>=? AND min_x<=? AND max_y>=? AND min_y<=? AND max_z>=? AND min_z<=?",
            (x0, x1, y0, y1, z0, z1)).fetchall()
        return [r[0] for r in rows]

    def sheet(self, sid: int) -> dict[str, Any]:
        """Full geometry for one sheet id — ``name, window, family, verts, tris, foot, nu, nv, annotation``."""
        row = self._db.execute(
            "SELECT name,window,family,payload,annotation FROM sheets WHERE id=?", (sid,)).fetchone()
        if row is None:
            raise KeyError(sid)
        name, window, family, blob, ann = row
        g = unpack_sheet(blob)
        g.update(id=sid, name=name, window=window, family=family, annotation=json.loads(ann))
        return g

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for (sid,) in self._db.execute("SELECT id FROM sheets ORDER BY id").fetchall():
            yield self.sheet(sid)

    def close(self) -> None:
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def merge_asv(shards: Iterable[str | os.PathLike], out: str | os.PathLike, *, frame: Mapping[str, Any],
              voxel_um: float, grid_step_um: float, root: Mapping[str, Any] | None = None,
              metadata: Mapping[str, Any] | None = None, overwrite: bool = False) -> dict[str, Any]:
    """Merge per-worker shard .asv files into one published .asv (leader step). Row ids are renumbered 1..N in shard
    order; blobs are copied verbatim (no re-pack), bbox rows re-inserted into a fresh rtree."""
    w = AsvWriter(out, frame=frame, voxel_um=voxel_um, grid_step_um=grid_step_um, root=root,
                  metadata=metadata, overwrite=overwrite)
    for shard in shards:
        src = sqlite3.connect(f"file:{Path(shard)}?mode=ro", uri=True)
        for row in src.execute(
                "SELECT name,window,family,nu,nv,vert_count,area_mm2,n_voxels,self_mass,confidence,mass_frac,"
                "fit_rms_um,coverage,min_x,max_x,min_y,max_y,min_z,max_z,payload,annotation "
                "FROM sheets ORDER BY id").fetchall():
            (name, window, family, nu, nv, vc, area, nvox, sm, conf, mf, rms, cov,
             x0, x1, y0, y1, z0, z1, blob, ann) = row
            w._n += 1; sid = w._n; w._vert_total += int(vc)                     # reuse writer counters
            w._db.execute("INSERT INTO sheets(id,name,window,family,nu,nv,vert_count,area_mm2,n_voxels,self_mass,"
                          "confidence,mass_frac,fit_rms_um,coverage,min_x,max_x,min_y,max_y,min_z,max_z,payload,"
                          "annotation) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                          (sid, name, window, family, nu, nv, vc, area, nvox, sm, conf, mf, rms, cov,
                           x0, x1, y0, y1, z0, z1, blob, ann))
            w._db.execute("INSERT INTO sheet_bounds VALUES(?,?,?,?,?,?,?)", (sid, x0, x1, y0, y1, z0, z1))
        src.close()
    return w.close()

# The Data Layer

`hercunet.data` is a transparent, reusable layer for **streaming teravoxel OME-Zarr scroll volumes**
off the Vesuvius open-data servers. It gives every consumer — the viewer, offline analysis scripts,
the inference pipeline — the same fast, cached, prefetched, material-aware access, so nobody
re-implements read logic and nobody re-downloads what's already local.

> **Why it exists.** The `vesuvius` library's `Volume` has **no read caching and no prefetch**, and
> its opener fails on the masked HTTP zarr *groups* published for these scrolls (it can't list group
> members over plain HTTP, so it reports "no arrays found" for perfectly good stores). We read the
> pyramid arrays directly by their known sub-paths and add the caching/prefetch/tiling on top.

---

## Contents
- [Concepts](#concepts)
- [Install](#install)
- [Quickstart](#quickstart)
- [Opening volumes](#opening-volumes)
- [Reading data](#reading-data)
- [Prefetching](#prefetching)
- [Material-aware tiling](#material-aware-tiling)
- [The on-disk cache](#the-on-disk-cache)
- [Configuration](#configuration-environment-variables)
- [Performance notes](#performance-notes)
- [API reference](#api-reference)
- [Extending: a custom backend](#extending-a-custom-backend)

---

## Concepts

The layer is built from a few small, composable pieces:

| Piece | What it is |
|---|---|
| **`SegmentSource`** (backend) | Enumerates scrolls/segments and opens them. Two ship: `vesuvius` (streaming, real data) and `demo` (procedural, offline). Selected by env; the rest of the code depends only on this interface. |
| **`SegmentVolume`** | One opened multiscale volume. All reads go through it; coordinates are per-pyramid-level (level 0 = full res). |
| **`ZarrSegment`** | The concrete `SegmentVolume` for an OME-Zarr store: **chunk-aligned, parallel, disk-cached** reads. |
| **`SlabDiskCache`** | The L2 on-disk cache. Whole chunks are the cache unit, so any overlapping request reuses them. |
| **`iter_windows` / `VolumePrefetcher`** | Background prefetch — warm the chunk cache ahead of the compute so the GPU never idles on downloads. |
| **`locate_material` / `iter_material_tiles`** | Find where the papyrus is from one cheap coarse read, skip the air, and stream only the material tiles. |

Axis convention everywhere: **axis 0 = z / depth**, axis 1 = y, axis 2 = x.

---

## Install

```bash
pip install -e .            # core = the data layer (numpy, zarr<3, fsspec, s3fs, requests, Pillow)
pip install -e ".[accel]"   # + obstore, the native (non-boto) Rust S3 reader
```

Requires Python ≥ 3.9.

---

## Quickstart

**Open a scroll volume straight from a URL and read a sub-block:**

```python
from hercunet.data import ZarrSegment

url = ("https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1447/volumes/"
       "20250521151220-8.640um-1.2m-116keV-masked.zarr")
vol = ZarrSegment(url, voxel_size_um=8.64)

print(vol.meta.num_levels)        # 6
print(vol.meta.level_shapes[0])   # (24297, 8343, 8343)

# read a 16 x 64 x 64 block at full res (level 0), level-native coords
block, origin = vol.read_window(level=0, z0=12160, z1=12176,
                                y0=4096, y1=4160, x0=4096, x1=4160)
print(block.shape, block.dtype, origin)   # (16, 64, 64) uint8 (12160, 4096, 4096)
```

The first read fetches from S3; a subsequent read of any overlapping region is served from the
on-disk cache in milliseconds.

---

## Opening volumes

### Directly by URL — `ZarrSegment`

```python
from hercunet.data import ZarrSegment

vol = ZarrSegment(base_url, voxel_size_um=None, voxel_override=None)
```

- `base_url` — the multiscale store root (the directory containing `0/`, `1/`, … or a bare array).
  `file://` and local filesystem paths work too.
- `voxel_size_um` — fallback voxel size; the OME-Zarr scale metadata is authoritative when present.
- `voxel_override` — force a voxel size (wins outright; use when the store's scale metadata is the
  trivial `1.0 µm` placeholder some standardised volumes carry).

### Through a backend — `get_backend`

The backend enumerates the whole open-data bucket, so you can browse by scroll/segment:

```python
from hercunet.config import Config
from hercunet.data import get_backend

be = get_backend(Config.from_env())          # HERCUNET_BACKEND=vesuvius (default) | demo
scrolls = be.list_scrolls()
vol = be.open_scroll_volume(scrolls[0])       # raw µCT volume of a scroll

segs = be.list_segments(scrolls[0])           # published surface segments
surf = be.open_segment(segs[0])               # a flattened surface volume
```

The **`demo` backend** (`HERCUNET_BACKEND=demo`) synthesises a procedural volume with the same API
and **no network** — ideal for tests and offline development.

---

## Reading data

All reads are methods on `SegmentVolume`. Coordinates passed to the `*_window` / `*_column` methods
are **level-native**; the `read_region` / `read_tile` helpers take **full-resolution** coordinates
and map them into the chosen level for you.

| Method | Returns | Use for |
|---|---|---|
| `read_window(level, z0,z1, y0,y1, x0,x1)` | `(block[dz,dy,dx], (z0c,y0c,x0c))` | An arbitrary sub-block (a z-slab around a depth). The workhorse. |
| `read_depth_column(level, y0,y1, x0,x1)` | `(block[dz,dy,dx], full_res_bbox)` | The **full** depth stack over a (y,x) window — the material to measure sheet thickness from. |
| `read_region(level, z, y0,y1, x0,x1)` | `(plane[dy,dx], full_res_bbox)` | One depth layer over a full-res window (viewport reads). |
| `read_tile(level, z, ty, tx, tile)` | `(plane, full_res_bbox)` | A fixed-grid tile (the viewer's cache unit). |
| `read_tile_slab(level, lz0,lz1, ty,tx, tile)` | `(block, full_res_bbox)` | A depth slab of one tile. |
| `sample_line(level, zs, ys, xs)` | `values[N]` | Vectorised point sampling along a path (profiles/spectra) — fetches only the chunks on the line. |

```python
# full through-sheet column over a small window (depth axis = the layers through the papyrus)
col, bbox = vol.read_depth_column(level=0, y0=4096, y1=4160, x0=4096, x1=4160)
print(col.shape)   # (Z, 64, 64)

# sample intensity along an arbitrary 3-D polyline
import numpy as np
zs = np.linspace(12160, 12200, 50); ys = np.full(50, 4100.0); xs = np.linspace(4096, 4200, 50)
profile = vol.sample_line(0, zs, ys, xs)
```

### Re-orienting the depth axis — `OrientedVolume`

To scrub *across* the winding instead of through the surface, wrap a volume so a different base axis
becomes depth (axis 0):

```python
from hercunet.data.segment import OrientedVolume
across = OrientedVolume(vol, axis="y")   # "z" (identity) | "y" | "x"
```

Reads are transposed transparently and cached separately per axis.

---

## Prefetching

Reads already cache their chunks, so **prefetching = issuing the anticipated reads early on
background threads**; the foreground read that follows is then a cache hit. This overlaps S3 I/O
with compute so the GPU doesn't idle on downloads.

### `iter_windows` — the streaming primitive

Iterate a list of read requests **in order**, reading up to `readahead` ahead on `workers` threads:

```python
from hercunet.data import iter_windows

# each request is a read_window arg tuple: (level, z0, z1, y0, y1, x0, x1)
reqs = [(0, z, z+16, y, y+512, x, x+512) for (y, x) in tile_origins]

for request, block, origin in iter_windows(vol, reqs, readahead=4, workers=4):
    process(block)          # download of the next blocks overlaps this compute
```

### `VolumePrefetcher` — strategy-based warming

For interactive / viewport use, warm specific shapes ahead of demand:

```python
from hercunet.data import VolumePrefetcher

pf = VolumePrefetcher(vol, max_workers=6)
pf.window(0, z0, z1, y0, y1, x0, x1)             # one exact block
pf.slice(0, z, y0, y1, x0, x1, halfdepth=1)      # a whole (y,x) region at depth z (± halfdepth)
pf.neighbours(0, z0, z1, y0, y1, x0, x1)         # neighbouring depths AND resolutions (zoom/scroll)
pf.drain()                                        # block until warmed (e.g. before timing)
pf.shutdown()
```

Prefetches are de-duplicated (the same block is never enqueued twice).

---

## Material-aware tiling

The pyramid exists so you can find the papyrus cheaply. `locate_material` reads **one coarse slice**
to get the material bounding box + a coarse mask; `iter_material_tiles` then streams only the
full-resolution tiles that overlap material, skipping air with no download at all.

```python
from hercunet.data import locate_material, iter_material_tiles, n_material_tiles

region = locate_material(vol, level=0, z=12160, bbox_levels=2)   # None if the slice is empty
if region is not None:
    total = n_material_tiles(vol, 0, region, tile=512)           # for a progress bar
    for tile in iter_material_tiles(vol, level=0, z0=12160, z1=12176, region=region,
                                    tile=512, halo=8, readahead=4, workers=4):
        # tile.block is the haloed full-res data; tile.wy/wx place it; tile.oz is the depth offset
        compute(tile.block)
```

> **Lesson baked in:** a full-slice read at the working level pulls the whole 128-deep chunk band
> (many GB). *Always* locate material at a coarse level, then compute at the full level.

---

## The on-disk cache

`SlabDiskCache` is the **L2** cache: chunks evicted from RAM are reloaded from local disk (ms)
instead of refetched from S3 (seconds). It is shared across the viewer and every offline script, so
a re-run or a parameter sweep over the same region skips S3 entirely.

- **Unit:** whole zarr chunks (the store fetches whole chunks anyway → zero extra download, full
  reuse across tiles / depths / re-runs).
- **Index:** a small SQLite `index.db` (WAL) tracks each blob's size + access time and a running
  byte total, so a `put` is one file write + a couple of indexed statements — **no directory scans**.
- **Eviction:** O(log N), oldest-first (`ORDER BY atime`), down to 90 % of budget then quiet.
- **Identity:** keyed by `source_id` (store URL + depth orientation), so scrolls / segments /
  orientations never collide.
- **Safety:** best-effort and thread-safe; a throttled/absent chunk is zero-filled for *that* read
  but **not** cached, so a transient 403 never poisons the cache with permanent air.

Enabled by default; controlled entirely by env (below). Set `HERCUNET_BLOCK_CACHE=0` to bypass it
(e.g. a one-shot full-volume pass that will never re-read the same voxels).

---

## Configuration (environment variables)

Everything is overridable by env, so the same code points at a different cache or backend with no
edits. `Config.from_env()` reads them.

| Variable | Default | Meaning |
|---|---|---|
| `HERCUNET_BACKEND` | `vesuvius` | `vesuvius` (streaming real data) or `demo` (procedural offline). |
| `HERCUNET_BLOCK_CACHE` | `1` | `0` disables the L2 disk cache for reads. |
| `HERCUNET_DISK_CACHE` | `~/.vesuvius/slabcache` | L2 cache directory (persists across reboots). |
| `HERCUNET_DISK_CACHE_MB` | `65536` | L2 byte budget (MB). |
| `HERCUNET_READ_WORKERS` | `12` | Threads fetching a block's chunks in parallel. **The main read-throughput knob.** |
| `HERCUNET_CACHE` | `~/.cache/hercunet` | Root for the catalog/RAM cache. |
| `HERCUNET_CACHE_MB` | scaled to RAM (512–2048) | RAM slab-cache budget (MB). |
| `HERCUNET_MEM_FLOOR_MB` | `2048` | Keep at least this much system RAM free (shed cache under pressure). |
| `HERCUNET_PREFETCH_MB` | `16384` | Cap on how much a single pre-load action downloads. |
| `HERCUNET_REFRESH_CATALOG` | `1` | On startup, list each scroll's segments dir and merge new ones. |
| `HERCUNET_SEGMENTS_ROOT` | `dl.ash2txt.org/other/dev/scrolls/` | Base URL for the targeted segment-catalog refresh. |
| `HERCUNET_GPU` | `1` | Allow CuPy/CUDA where used. |

---

## Performance notes

Measured against the PHerc1447 L0 volume (`…-8.640um…-masked.zarr`):

- **The scroll volumes are uncompressed** (`compressor: None`, 128³ uint8 chunks = 2 MB each), so a
  read is **pure network IO** — there is no decompression to parallelise, only transfers.
- **A single HTTP stream is throttled** (~1 MB/s cross-region). Throughput is therefore entirely a
  function of **concurrency**: serial → parallel was ~13× in testing. `HERCUNET_READ_WORKERS` (per
  block) and the prefetch `workers`/`readahead` (across blocks) are the levers.
- **`s3://` beats `https://`** — the s3 protocol path measured ~2× the https path. For the very
  fastest transfers, the optional native reader (`obstore`, `pip install -e ".[accel]"`) is worth
  benchmarking on your machine.
- **The cache turns re-reads free:** a cold read of a core block was ~2.4 s; the warm cache hit was
  ~11 ms (identical bytes) — ~200× faster.

---

## API reference

Import surface (`from hercunet.data import …`):

```
# abstractions
SegmentSource, get_backend            # backend interface + selection
SegmentVolume                         # opened multiscale volume (read_* methods)
ScrollInfo, SegmentInfo, SegmentMeta  # metadata records

# streaming primitives
ZarrSegment, parse_voxel_um           # the OME-Zarr reader
SlabDiskCache                         # the L2 disk cache
iter_windows, VolumePrefetcher        # prefetch
MaterialRegion, locate_material, iter_material_tiles, n_material_tiles   # tiling
```

`SegmentMeta` fields/props: `level_shapes`, `dtype`, `voxel_size_um`, `shape`, `num_levels`,
`num_layers`, `downsample_x(level)`, `window_cap_px(mm)`.

---

## Extending: a custom backend

Implement `SegmentSource` (enumerate + open) and return `SegmentVolume`s; the rest of `hercunet`
works unchanged. To back a new store format, subclass `SegmentVolume` and provide the two abstract
reads:

```python
from hercunet.data import SegmentVolume, SegmentMeta

class MyVolume(SegmentVolume):
    def __init__(self, ...):
        super().__init__(SegmentMeta(level_shapes=..., dtype="uint8", voxel_size_um=...))
        self.source_id = "my-store://..."   # stable id for the disk cache; None disables it

    def _read_block(self, level, z0, z1, y0, y1, x0, x1):
        ...   # return a (dz, dy, dx) ndarray

    def _read_points(self, level, zs, ys, xs):
        ...   # return a 1-D ndarray of values at the points
```

All the higher-level reads (`read_window`, `read_region`, tiling, prefetch, caching) are built on
those two primitives, so you get them for free.

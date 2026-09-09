# HercUNet

**Cross-volume surface (sheet) detection for carbonised Herculaneum scrolls.**

Ink and text cannot be read until the papyrus sheets inside a scroll are correctly detected and
separated. Even if ink is detected on the wrapped volume, we need to unroll to read.
Existing surface detectors are trained per-corpus and don't generalise across volumes —
they **merge touching sheets** and **break thin ones**, topological errors that voxel-accuracy alone
never fixes. HercUNet attacks that with an **iterative refiner** trained on multi-scroll,
native-resolution pseudo-labels, running over a fast **streaming data layer** for teravoxel OME-Zarr
scroll volumes.

The project is built as a three-stage pipeline over one shared data layer, **piece by piece**, each
stage shipping with a visual tool that shows what it does.

## 📚 Documentation

**All documentation lives in [`docs/`](docs/) — start with the [docs index](docs/README.md).** Each
part of the project gets its own comprehensive guide there.

- **[Data layer](docs/data-layer.md)** — streaming OME-Zarr reads, caching, prefetch, tiling. ✅ **Available now.**

## Pipeline stages

| Stage | Module | Status |
|---|---|---|
| **Data layer** — streaming multiscale OME-Zarr reads (chunk-cache + parallel prefetch + material tiling) | `hercunet.data` | ✅ implemented |
| **Stage 1 — Pseudo-label generation** | `hercunet.labels` | 🚧 skeleton |
| **Stage 2 — Iterative HercUNet refiner** | `hercunet.refine` | 🚧 skeleton |
| **Stage 3 — Full-volume inference** | `hercunet.infer` | 🚧 skeleton |
| **Per-stage visual tools** | `hercunet.viz` | 🚧 skeleton |

## Install

```bash
pip install -e .            # core = the data layer
pip install -e ".[accel]"   # + obstore (native, non-boto S3 reader)
pip install -e ".[dev]"     # + pytest
```

Requires Python ≥ 3.9.

## Quickstart

```python
from hercunet.data import ZarrSegment

# open a scroll volume straight from the Vesuvius open-data bucket
url = ("https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1447/volumes/"
       "20250521151220-8.640um-1.2m-116keV-masked.zarr")
vol = ZarrSegment(url, voxel_size_um=8.64)

block, origin = vol.read_window(level=0, z0=12160, z1=12176,
                                y0=4096, y1=4160, x0=4096, x1=4160)
```

See the **[data-layer guide](docs/data-layer.md)** for the full API — backends, prefetching,
material tiling, the on-disk cache, and configuration.

## Status

Early / in-progress. The data layer is complete and tested; the pipeline stages are being ported in
one at a time.

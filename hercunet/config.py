"""Runtime configuration: cache location, the rule constant, and backend selection.

All values are overridable by environment variable so the app can be pointed at a
different cache or forced onto the demo backend without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _default_disk_cache() -> str:
    """Default L2 slab-cache dir: ``~/.vesuvius/slabcache`` — persists across reboots (unlike
    /tmp), so the cache and its SQLite index are not lost. Override with HERCUNET_DISK_CACHE."""
    return str(Path.home() / ".vesuvius" / "slabcache")

# The Vesuvius rulebook caps the analysis window at ~0.5 x 0.5 mm so a whole letter
# cannot fit (anti-hallucination, see project brief section 3.2). The viewer draws this
# as a reference square; it is a physical size, converted to pixels per-segment using
# the segment's voxel size in micrometres.
WINDOW_CAP_MM: float = 0.5


def _cache_root() -> Path:
    env = os.environ.get("HERCUNET_CACHE")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "hercureader"


# Conventional root under which per-scroll segment directories live, e.g.
# {root}/1/segments/54keV_7.91um/<id>.zarr/. Overridable by env. Used only by the cheap
# targeted catalog refresh (list a scroll's segments dir), never a full recursive crawl.
_DEFAULT_SEGMENTS_ROOT = "https://dl.ash2txt.org/other/dev/scrolls/"


def _is_truthy(value: str) -> bool:
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _total_ram_mb() -> int:
    """Total physical RAM in MB (Linux /proc/meminfo), or 8192 if unknown."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return 8192


def _default_cache_mb() -> int:
    """RAM slab-cache budget scaled to the machine, clamped to [512, 2048] MB — deliberately
    CONSERVATIVE.

    The RAM cache only needs to hold the working set (visible tiles + immediate depth/pan
    neighbours) for smooth navigation; the *aggressive* buffering lives in the on-disk L2
    (SQLite + .npy, tens of GB), which re-reads in milliseconds — far cheaper than the network
    it replaces. The old fixed 8 GB RAM default OOM-killed small laptops: real RSS runs ~3x the
    slab byte budget (pyqtgraph RGBA copies, CuPy/torch host buffers, numpy temporaries, malloc
    fragmentation), so 8 GB configured meant ~25 GB resident. Scale to ~1/16 of RAM and let the
    disk L2 + the runtime memory-pressure guard (see VizPane) absorb the rest."""
    return int(min(2048, max(512, _total_ram_mb() // 16)))


@dataclass(frozen=True)
class Config:
    cache_dir: Path
    backend: str          # "vesuvius" | "demo"
    prefer_gpu: bool      # try CuPy/CUDA when available
    tile_margin: float    # extra fraction of viewport to prefetch around the visible area
    debounce_ms: int      # coalesce rapid pan/zoom before issuing a read
    refresh_catalog: bool # on startup, list each scroll's segments dir and merge in new ones
    segments_root: str    # base URL for the targeted segment-directory refresh
    slab_cache_mb: int    # RAM budget for the depth-slab tile cache (MB)
    mem_floor_mb: int     # keep at least this much SYSTEM RAM available (shed cache under pressure)
    disk_cache_dir: str   # L2 on-disk slab cache location (empty string disables it)
    disk_cache_mb: int    # byte budget for the on-disk slab cache (MB)
    prefetch_mb: int      # cap on how much a single "pre-load into cache" action downloads

    @staticmethod
    def from_env() -> "Config":
        backend = os.environ.get("HERCUNET_BACKEND", "vesuvius").strip().lower()
        cache = _cache_root()
        cache.mkdir(parents=True, exist_ok=True)
        return Config(
            cache_dir=cache,
            backend=backend,
            prefer_gpu=_is_truthy(os.environ.get("HERCUNET_GPU", "1")),
            tile_margin=0.25,
            debounce_ms=80,
            refresh_catalog=_is_truthy(os.environ.get("HERCUNET_REFRESH_CATALOG", "1")),
            segments_root=os.environ.get("HERCUNET_SEGMENTS_ROOT", _DEFAULT_SEGMENTS_ROOT),
            slab_cache_mb=int(os.environ.get("HERCUNET_CACHE_MB", str(_default_cache_mb()))),
            mem_floor_mb=int(os.environ.get("HERCUNET_MEM_FLOOR_MB", "2048")),
            disk_cache_dir=os.environ.get("HERCUNET_DISK_CACHE", _default_disk_cache()),
            disk_cache_mb=int(os.environ.get("HERCUNET_DISK_CACHE_MB", "65536")),
            prefetch_mb=int(os.environ.get("HERCUNET_PREFETCH_MB", "16384")),
        )

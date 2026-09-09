"""Smoke tests for the data layer.

Offline test (default, no network): the procedural ``demo`` backend exercises the full read
path — catalog -> open volume -> ``read_window`` -> ``iter_windows`` prefetch — with zero
external dependencies, so CI stays hermetic.

Network test (opt-in, ``HERCUNET_NET_TEST=1``): opens the live PHerc1447 volume over the
Vesuvius S3 bucket and checks a cold read + a warm L2-cache hit return identical bytes.
"""
import os

import numpy as np
import pytest


def test_demo_read_path(monkeypatch):
    monkeypatch.setenv("HERCUNET_BACKEND", "demo")
    from hercunet.config import Config
    from hercunet.data import get_backend, iter_windows

    be = get_backend(Config.from_env())
    scrolls = be.list_scrolls()
    assert scrolls, "demo backend should list at least one scroll"

    vol = be.open_scroll_volume(scrolls[0])
    Z, Y, X = vol.meta.level_shapes[0]
    assert Z > 0 and Y > 0 and X > 0

    blk, origin = vol.read_window(0, 0, 4, 0, 64, 0, 64)
    assert blk.shape == (4, 64, 64)
    assert origin == (0, 0, 0)

    reqs = [(0, 0, 4, 0, 64, 0, 64), (0, 0, 4, 64, 128, 0, 64)]
    blocks = [b for _, b, _ in iter_windows(vol, reqs, readahead=2, workers=2)]
    assert [b.shape for b in blocks] == [(4, 64, 64), (4, 64, 64)]


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("HERCUNET_NET_TEST") != "1",
                    reason="set HERCUNET_NET_TEST=1 to hit the live Vesuvius S3 bucket")
def test_live_read_and_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HERCUNET_DISK_CACHE", str(tmp_path / "slabcache"))
    from hercunet.data import ZarrSegment

    url = ("https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1447/volumes/"
           "20250521151220-8.640um-1.2m-116keV-masked.zarr")
    vol = ZarrSegment(url, 8.64)
    assert vol.meta.level_shapes[0] == (24297, 8343, 8343)

    win = (0, 12160, 12176, 4096, 4160, 4096, 4160)  # 16x64x64 in the scroll core
    cold, _ = vol.read_window(*win)
    warm, _ = vol.read_window(*win)                    # served from the L2 disk cache
    assert cold.shape == (16, 64, 64)
    assert np.array_equal(cold, warm)

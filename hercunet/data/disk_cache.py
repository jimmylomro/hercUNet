"""On-disk L2 cache for depth slabs, indexed by SQLite.

RAM is the fast L1 (``VizPane._slabs``); this is the L2. A slab evicted from RAM is reloaded
from local disk (ms) instead of refetched from S3 (seconds). Slabs live as ``.npy`` blobs
(cheap ``np.load``); a small SQLite index (``index.db``) holds each blob's size and insert
time plus a running byte total. That buys three things the old directory-scan approach could
not:

  * **No per-write scans.** A ``put`` is one file write + a couple of indexed SQL statements;
    it never lists the directory. (The scan-on-every-put under a shared lock is what made one
    stalled worker stall them all when the cache was full.)
  * **O(log N) eviction.** Over budget → ``ORDER BY atime`` picks the oldest to drop.
  * **Shared accounting.** The viewer's cache and the background prefetcher open the SAME
    ``index.db``, so their byte totals agree instead of drifting.

Keys are ``(source_id, level, slab_z0, ty, tx)``; ``source_id`` folds in the store URL and
depth orientation, so scrolls/segments/orientations never collide. Everything is best-effort
and thread-safe; all access is off the GUI thread.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np


class SlabDiskCache:
    """A thread-safe, size-bounded on-disk slab store with a SQLite index."""

    def __init__(self, root: str | os.PathLike | None = None, budget_bytes: int = 20 * 2**30):
        if root is None:
            root = Path.home() / ".vesuvius" / "slabcache"  # persists across reboots
        self.root = Path(root)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        self.budget = int(budget_bytes)
        self._low_water = int(self.budget * 0.9)  # evict down to here, then stay quiet
        self._lock = threading.Lock()             # guards this instance's connection
        self._db = sqlite3.connect(
            str(self.root / "index.db"), check_same_thread=False, isolation_level=None
        )
        for pragma in (
            "PRAGMA journal_mode=WAL",       # concurrent readers; single writer
            "PRAGMA synchronous=NORMAL",     # a cache — durability is not critical
            "PRAGMA busy_timeout=5000",      # wait, don't fail, on cross-instance writers
        ):
            try:
                self._db.execute(pragma)
            except sqlite3.Error:
                pass
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS slabs (name TEXT PRIMARY KEY, size INTEGER, atime REAL)"
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_atime ON slabs(atime)")
        self._db.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v INTEGER)")
        self._reconcile()

    def _reconcile(self) -> None:
        """Once per store, adopt any pre-existing ``.npy`` files (e.g. written by an older
        version, or left by a previous run) into the index so they are counted and evictable."""
        with self._lock:
            row = self._db.execute("SELECT v FROM meta WHERE k='total'").fetchone()
            if row is not None:
                return  # already indexed
            total = 0
            for f in self.root.glob("*.npy"):
                try:
                    st = f.stat()
                except OSError:
                    continue
                total += st.st_size
                self._db.execute(
                    "INSERT OR REPLACE INTO slabs VALUES (?,?,?)", (f.name, st.st_size, st.st_mtime)
                )
            self._db.execute("INSERT OR REPLACE INTO meta VALUES ('total', ?)", (total,))

    def _path(self, source_id: str, key: tuple) -> Path:
        h = hashlib.sha1(f"{source_id}|{key!r}".encode()).hexdigest()
        return self.root / f"{h}.npy"

    def has(self, source_id: str, key: tuple) -> bool:
        """Indexed presence check (no read, no directory scan) — lets prefetch skip hits."""
        name = self._path(source_id, key).name
        with self._lock:
            return self._db.execute(
                "SELECT 1 FROM slabs WHERE name=?", (name,)
            ).fetchone() is not None

    def get(self, source_id: str, key: tuple) -> np.ndarray | None:
        """Return the cached slab array, or None on a miss/error. Reads the blob off-lock."""
        p = self._path(source_id, key)
        try:
            return np.load(p, allow_pickle=False)
        except (OSError, ValueError, EOFError):
            return None

    def put(self, source_id: str, key: tuple, block: np.ndarray) -> None:
        """Write a slab (atomic replace) and index it. The write is lock-free (workers write
        in parallel); the lock is held only for the fast indexed bookkeeping."""
        p = self._path(source_id, key)
        tmp = p.with_name(f"{p.name}.tmp{os.getpid()}.{threading.get_ident()}")
        try:
            with open(tmp, "wb") as fh:  # a file handle avoids np.save's ".npy" munging
                np.save(fh, np.ascontiguousarray(block), allow_pickle=False)
            os.replace(tmp, p)
            size = p.stat().st_size
        except OSError:
            try:
                tmp.unlink()
            except OSError:
                pass
            return
        with self._lock:
            try:
                old = self._db.execute(
                    "SELECT size FROM slabs WHERE name=?", (p.name,)
                ).fetchone()
                self._db.execute(
                    "INSERT OR REPLACE INTO slabs VALUES (?,?,?)", (p.name, size, time.time())
                )
                self._db.execute(
                    "UPDATE meta SET v = v + ? WHERE k='total'", (size - (old[0] if old else 0),)
                )
                total = self._db.execute("SELECT v FROM meta WHERE k='total'").fetchone()[0]
                if total > self.budget:
                    self._evict(total)
            except sqlite3.Error:
                pass

    def _evict(self, total: int) -> None:
        """Drop oldest slabs until under the low-water mark. Called under the lock, only when
        over budget; evicting to 90% means it then stays quiet for a long stretch of writes."""
        while total > self._low_water:
            rows = self._db.execute(
                "SELECT name, size FROM slabs ORDER BY atime ASC LIMIT 128"
            ).fetchall()
            if len(rows) <= 1:  # never evict the last (freshest) slab
                break
            for name, size in rows:
                if total <= self._low_water:
                    break
                try:
                    (self.root / name).unlink()
                except OSError:
                    pass
                self._db.execute("DELETE FROM slabs WHERE name=?", (name,))
                total -= size
        self._db.execute("UPDATE meta SET v=? WHERE k='total'", (total,))

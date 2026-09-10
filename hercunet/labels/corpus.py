"""The ``.herculabels`` corpus container — a directory of editable pseudo-label windows + a manifest.

A corpus is a directory ``<name>.herculabels/`` holding ``meta.json`` (the manifest) and ``windows/<id>.npz``
(one editable window each: base sheet meshes + the meshlet cloud, no dense fields — see
docs/herculabels.md). It is the human-editable source of truth; the training set is a separate,
re-runnable ``export`` derived from it. Every corpus is editable for its whole life — there is no
"finalise"/frozen state.

The manifest records everything needed to reproduce the corpus (seeds, scroll, coords/source, gen params)
and, per window, whether a human has edited it. ``herculabels_version`` tags the format.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

HERCULABELS_VERSION = "v1"
_EXT = ".herculabels"


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "nogit"


def window_id(scroll, level, coords) -> str:
    """Stable per-window id / filename stem: ``s<scroll>_L<level>_z<z>_y<y>_x<x>``."""
    z, y, x = (int(c) for c in coords)
    return f"s{scroll}_L{int(level)}_z{z}_y{y}_x{x}"


class Corpus:
    """Read/write a ``.herculabels`` corpus directory + its manifest. Use :meth:`create` to start a new
    corpus (fails if it exists) or :meth:`open` to load one."""

    def __init__(self, root: Path, meta: dict):
        self.root = Path(root)
        self.meta = meta

    # -- lifecycle --------------------------------------------------------------------------------
    @classmethod
    def create(cls, path, *, create_params: dict) -> "Corpus":
        """Create a new corpus at ``path`` (a ``.herculabels`` dir). **Fails if it already exists.**
        ``create_params`` records how to reproduce the corpus (mode, scroll, seed, source, coords, himat,
        count, gen_params) and is stored under the manifest's ``create`` key."""
        root = Path(path)
        if root.suffix != _EXT:
            raise SystemExit(f"corpus path must end in '{_EXT}' (got {root.name!r})")
        if root.exists():
            raise SystemExit(f"corpus already exists: {root} — refusing to overwrite (choose a new path)")
        (root / "windows").mkdir(parents=True, exist_ok=False)
        from datetime import datetime, timezone
        meta = {"herculabels_version": HERCULABELS_VERSION,
                "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "code_git_sha": _git_sha(), "create": dict(create_params), "windows": []}
        c = cls(root, meta)
        c._save_manifest()
        return c

    @classmethod
    def open(cls, path) -> "Corpus":
        """Open an existing corpus directory and load its manifest."""
        root = Path(path)
        mf = root / "meta.json"
        if not mf.exists():
            raise SystemExit(f"not a herculabels corpus (no meta.json): {root}")
        meta = json.loads(mf.read_text())
        ver = meta.get("herculabels_version")
        if ver != HERCULABELS_VERSION:
            raise SystemExit(f"corpus {root} is herculabels {ver!r}, this build writes {HERCULABELS_VERSION!r}")
        return cls(root, meta)

    @staticmethod
    def is_corpus(path) -> bool:
        p = Path(path)
        return p.suffix == _EXT and (p / "meta.json").exists()

    # -- windows ----------------------------------------------------------------------------------
    def window_path(self, wid: str) -> Path:
        return self.root / "windows" / f"{wid}.npz"

    def write_window(self, meta: dict, meshes: dict, meshlet: dict) -> str:
        """Write (or overwrite) one window from its ``meta`` + base ``meshes`` + ``meshlet`` cloud, and
        upsert its manifest entry. ``meta`` carries scroll/level/coords/source/edited/quality. Returns the
        window id."""
        import hercunet.labels.io.bundle as bundle
        wid = window_id(meta["scroll"], meta["level"], meta["coords"])
        bundle.save_window_bundle(self.window_path(wid), meta, meshes, meshlet)
        entry = {"id": wid, "scroll": meta["scroll"], "level": int(meta["level"]),
                 "coords": [int(c) for c in meta["coords"]], "source": meta.get("source"),
                 "seed": meta.get("master_seed"), "n_sheets": int(meta.get("n_sheets", len(meshes))),
                 "edited": bool(meta.get("edited", False)), "quality": meta.get("quality")}
        self._upsert_window(entry)
        self._save_manifest()
        return wid

    def load_window(self, wid_or_entry):
        """Load one window → (meta, meshes, meshlet). Accepts a window id or a manifest entry dict."""
        import hercunet.labels.io.bundle as bundle
        wid = wid_or_entry["id"] if isinstance(wid_or_entry, dict) else wid_or_entry
        return bundle.load_window_bundle(self.window_path(wid))

    def iter_windows(self):
        """Yield manifest window entries in creation order."""
        yield from self.meta.get("windows", [])

    def __len__(self) -> int:
        return len(self.meta.get("windows", []))

    def has_window(self, scroll, level, coords) -> bool:
        wid = window_id(scroll, level, coords)
        return any(w["id"] == wid for w in self.meta.get("windows", []))

    # -- internals --------------------------------------------------------------------------------
    def _upsert_window(self, entry: dict) -> None:
        wins = self.meta.setdefault("windows", [])
        for i, w in enumerate(wins):
            if w["id"] == entry["id"]:
                wins[i] = entry
                return
        wins.append(entry)

    def _save_manifest(self) -> None:
        (self.root / "meta.json").write_text(json.dumps(self.meta, indent=2))

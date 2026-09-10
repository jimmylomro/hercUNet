"""Interactive 'grind' session: feed the Qt viewer one window at a time, open-endedly.

The viewer (``hercunet/viz``) asks this session for each next window's run closure — a random material
draw, an exact ``Z,Y,X``, or the next entry of a coords file — and, on advance, hands the edited state
back to :meth:`GrindSession.save`. All pipeline/label logic stays here in the labels package; the viewer
only drives it.
"""

from __future__ import annotations

import numpy as np

_MIN_MATERIAL_FRAC = 0.15         # skip windows with less material than this
_MAX_SEARCH = 200                 # give up the (read-per-window) default search after this many draws
_MAX_SCAN = 20000                 # coarse-mask himat screen is instant → allow a big scan before falling back


def parse_coords_file(path, default_scroll):
    """Parse a grind coords file. Each non-empty, non-``#`` line is ``Z,Y,X`` (uses ``default_scroll``)
    or ``SCROLL,Z,Y,X``. Returns ``[(scroll, z, y, x), ...]`` in file order."""
    out = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.replace(",", " ").split()
            if len(parts) == 3:
                if not default_scroll:
                    raise SystemExit(f"coords-file line '{ln}' has no scroll and no --scroll was given")
                out.append((default_scroll, int(parts[0]), int(parts[1]), int(parts[2])))
            elif len(parts) == 4:
                out.append((parts[0], int(parts[1]), int(parts[2]), int(parts[3])))
            else:
                raise SystemExit(f"coords-file line '{ln}' must be 'Z,Y,X' or 'SCROLL,Z,Y,X'")
    return out


class GrindSession:
    """Supplies the viewer with each window's pipeline closure and saves edited windows.

    ``coords_list`` = an ordered ``[(scroll, z, y, x), ...]`` to grind (from ``--coords-file`` / ``--coords``),
    or ``None`` for open-ended random drawing (the viewer's Next control then chooses random-or-specify)."""

    def __init__(self, be, scrolls, make_args, gpu, seed, *, corpus=None, scroll=None, coords_list=None,
                 first=None, himat=None):
        self.be = be
        self.scrolls = scrolls
        self._make_args = make_args
        self.gpu = gpu
        self.corpus = corpus                                    # the .herculabels corpus each window is written to
        self.scroll = scroll                                     # default scroll for the "specify coords" popover
        self.coords_list = coords_list                          # [(scroll,z,y,x), ...] or None (open-ended)
        self.first = first                                      # open-ended: the starting window (or None = random)
        self.himat = himat                                     # min material-fill fraction (None = default 0.15)
        self._seed = seed
        self._stream = None                                     # lazy random sample_windows iterator
        self._targets = {}                                     # sid -> target descriptor (cache)
        self._current_meta = None                              # meta of the loaded window (base for an edited save)

    def _target(self, sid):
        from ._brick import find_scroll
        if sid not in self._targets:
            self._targets[sid] = find_scroll(self.be, sid)
        return self._targets[sid]

    def _search_himat(self, progress, thr):
        """FAST high-material search (the m7 himat way): screen candidate windows against the scroll's
        COARSE material mask — an instant numpy slice-mean, no per-window full-res read — and take the
        first window with material ≥ ``thr`` (or the best seen after a big scan). Returns (target,z,y,x)."""
        from ._brick import window_material_frac
        progress("status", f"searching for a window with ≥{int(round(thr * 100))}% material fill…")
        best = None
        for tries in range(1, _MAX_SCAN + 1):
            try:
                sid, (cz, cy, cx) = next(self._stream)
            except StopIteration:
                break
            t = self._target(sid)
            frac = window_material_frac(self.be, t, (cz, cy, cx))
            if best is None or frac > best[0]:
                best = (frac, t, (cz, cy, cx))
            if frac >= thr:
                return t, cz, cy, cx
            if tries % 500 == 0:
                progress("status", f"  scanned {tries} windows (best {best[0]:.2f})…")
        if best is None:
            raise RuntimeError("no sampleable windows (all scrolls failed geometry)")
        _f, t, (cz, cy, cx) = best
        progress("status", f"  no window ≥{thr:.2f} in {tries} scans — using best ({_f:.2f})")
        return t, cz, cy, cx

    def _search_default(self, progress):
        """Default behaviour: read + CED each candidate and keep the first with ≥ the 0.15 air-skip material
        (bounded, using the best seen if none qualifies). Returns (target, brick, z, y, x)."""
        from ._brick import build_brick
        best = None
        for tries in range(1, _MAX_SEARCH + 1):
            try:
                sid, (cz, cy, cx) = next(self._stream)
            except StopIteration:
                break
            t = self._target(sid)
            progress("status", f"trying {sid} z{cz} y{cy} x{cx} …")
            b = build_brick(self.be, t, (cz, cy, cx), level=0, gpu=self.gpu)
            frac = float((b["bced"] > np.percentile(b["bced"], 55)).mean())
            if best is None or frac > best[0]:
                best = (frac, t, (cz, cy, cx), b)
            if frac >= _MIN_MATERIAL_FRAC:
                return t, b, cz, cy, cx
            progress("status", "  window is mostly air — resampling")
        if best is None:
            raise RuntimeError("no sampleable windows (all scrolls failed geometry)")
        _f, t, (cz, cy, cx), b = best
        progress("status", f"  no material window in {tries} draws — using best ({_f:.2f})")
        return t, b, cz, cy, cx

    def run_for(self, request):
        """``request``: ``None`` → draw a random material window (or a high-``--himat`` one); ``(scroll,
        z, y, x)`` → that exact one. Returns ``run_fn(progress)`` for the viewer's worker thread."""
        from ._brick import build_brick, sample_windows
        from ._generate import generate

        def run(progress):
            if request is None:
                if self._stream is None:
                    self._stream = sample_windows(self.be, self.scrolls, level=0, seed=self._seed)
                if self.himat is not None:                       # coarse-mask screen → read only the winner
                    target, cz, cy, cx = self._search_himat(progress, self.himat)
                    brick = build_brick(self.be, target, (cz, cy, cx), level=0, gpu=self.gpu)
                else:                                            # current per-window read+screen behaviour
                    target, brick, cz, cy, cx = self._search_default(progress)
            else:
                sid, cz, cy, cx = request
                target = self._target(sid)
                progress("status", f"reading {sid} z{cz} y{cy} x{cx}")
                brick = build_brick(self.be, target, (cz, cy, cx), level=0, gpu=self.gpu)
            progress("brick", {"bced": brick["bced"], "corner": brick["corner"],
                               "voxel_um": brick["voxel_um"], "scroll": target.scroll_id,
                               "coords": (cz, cy, cx)})
            res = generate(self._make_args(target.scroll_id, cz, cy, cx), brick, progress=progress)
            if res is None:
                return None
            self._current_meta = res.meta                        # base for a later edited save
            if self.corpus is not None:                          # persist the (unedited) base window immediately
                self.corpus.write_window(res.meta, res.meshes, res.meshlet)
            return res

        return run

    def save(self, ctx, meshes, conf_region=None):
        """Persist the edited window to the corpus — assemble it label-side (meta + meshes + meshlet) and
        overwrite the base entry (``edited: true``). ``conf_region`` is unused (the delete low-confidence is
        implicit in ``plab == -1`` and recomputed at export). Returns the window id (or None)."""
        from .edit import save_edited_window
        meta, meshes, meshlet = save_edited_window(ctx, meshes, self._current_meta)
        if self.corpus is not None:
            return self.corpus.write_window(meta, meshes, meshlet)
        return None

    def delete(self, scroll, coords, level=0):
        """Remove the window at ``scroll`` + ``coords`` (z,y,x) from the corpus (skip a bad one). Takes
        explicit coords — not ``_current_meta`` — so a failed window can't delete the previous good one.
        Returns the window id if it was present, else None."""
        from .corpus import window_id
        if self.corpus is None or scroll is None or coords is None:
            return None
        return self.corpus.delete_window(window_id(scroll, level, coords))


class EditSession:
    """Drive the viewer over an EXISTING ``.herculabels`` corpus (the ``edit`` command) — each window is
    LOADED off disk (base meshes + meshlet cloud), never re-generated. Presents the corpus windows as a
    fixed ordered list so the viewer navigates them with its list-mode Next control, and writes edits back
    in place. The CT slice-pane background is re-read from the scroll on open (a cheap per-window read); if
    the scroll is unavailable it falls back to a blank background (points-only editing still works)."""

    def __init__(self, be, corpus, gpu, *, scroll=None):
        self.be = be
        self.corpus = corpus
        self.gpu = gpu
        self.scroll = scroll                                    # default scroll (only for parity with the viewer)
        self._entries = list(corpus.iter_windows())
        self.coords_list = [(e["scroll"], *e["coords"]) for e in self._entries]   # viewer list-mode navigation
        self.first = None
        self.himat = None
        self._targets = {}
        self._current_meta = None

    def _target(self, sid):
        from ._brick import find_scroll
        if sid not in self._targets:
            self._targets[sid] = find_scroll(self.be, sid)
        return self._targets[sid]

    def _entry_for(self, scroll, cz, cy, cx):
        for e in self._entries:
            if e["scroll"] == scroll and list(e["coords"]) == [cz, cy, cx]:
                return e
        raise KeyError(f"window {scroll} z{cz} y{cy} x{cx} not in corpus")

    def run_for(self, request):
        """``request`` = ``(scroll, z, y, x)`` (a coords-list entry). Returns ``run(progress)`` that loads
        that window from the corpus and replays it to the viewer (no pipeline)."""
        from types import SimpleNamespace
        from ._brick import build_brick
        from ._generate import base_confidence
        from .edit import edit_context_from_window

        def run(progress):
            scroll, cz, cy, cx = request
            meta, meshes, meshlet = self.corpus.load_window(self._entry_for(scroll, cz, cy, cx))
            self._current_meta = meta
            vu, shape = float(meshlet["vu"]), tuple(int(s) for s in meshlet["shape"])
            try:                                                 # re-read the CT window for the slice-pane background
                brick = build_brick(self.be, self._target(scroll), (cz, cy, cx),
                                    level=int(meta["level"]), gpu=self.gpu)
                bced, corner = brick["bced"], brick["corner"]
            except Exception as e:                               # scroll not available → blank background
                progress("status", f"CT re-read failed ({type(e).__name__}) — editing on a blank background")
                bced, corner = np.zeros(shape, np.float32), (0, 0, 0)
            progress("brick", {"bced": bced, "corner": corner, "voxel_um": vu,
                               "scroll": scroll, "coords": (cz, cy, cx)})
            pts, plab = np.asarray(meshlet["pts"], np.float32), np.asarray(meshlet["plab"], np.int32)
            progress("streamlets", {"pts": pts, "kind": np.zeros(len(pts), np.int8)})
            progress("clusters", {"pts": pts, "plab": plab, "corner": corner})
            progress("editctx", edit_context_from_window(meta, meshlet, "cuda" if self.gpu else "cpu"))
            progress("meshes", {"meshes": meshes})
            conf, _lab3, _cover = base_confidence(meshes, meshlet, shape, vu, gpu=self.gpu)
            progress("confidence", conf)
            return SimpleNamespace(meta=meta, meshes=meshes, meshlet=meshlet)

        return run

    def save(self, ctx, meshes, conf_region=None):
        """Overwrite the edited window in the corpus (same as :meth:`GrindSession.save`)."""
        from .edit import save_edited_window
        meta, meshes, meshlet = save_edited_window(ctx, meshes, self._current_meta)
        return self.corpus.write_window(meta, meshes, meshlet)

    def delete(self, scroll, coords, level=0):
        """Remove the window at ``scroll`` + ``coords`` from the corpus (same as :meth:`GrindSession.delete`)."""
        from .corpus import window_id
        if scroll is None or coords is None:
            return None
        return self.corpus.delete_window(window_id(scroll, level, coords))

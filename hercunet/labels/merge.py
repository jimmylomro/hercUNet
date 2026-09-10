"""``hercunet labels merge`` — combine several ``.herculabels`` corpora into one.

Creates a **new** output corpus (fails if it exists) holding the union of the inputs' windows; the inputs
are never modified. When the same window id (same scroll + coords) appears in more than one input, the
**edited** copy is kept in preference to an unedited one, otherwise the later input wins (with a warning) —
so a hand-corrected window is never clobbered by a stale duplicate. The merged manifest records each source
under ``create.merged_from`` for provenance.
"""

from __future__ import annotations


def _prefer(new_entry: dict, old_entry: dict) -> bool:
    """Should ``new_entry`` replace the already-chosen ``old_entry`` for the same window id? Prefer an
    edited window over an unedited one; for an equal edited-status, the later input wins."""
    new_ed, old_ed = bool(new_entry.get("edited")), bool(old_entry.get("edited"))
    if new_ed != old_ed:
        return new_ed
    return True


def merge_corpora(out_path: str, input_paths: list[str]) -> None:
    """Merge the corpora at ``input_paths`` into a new corpus at ``out_path`` (a ``.herculabels`` dir;
    fails if it exists)."""
    from .corpus import Corpus

    inputs = [Corpus.open(p) for p in input_paths]
    merged_from = [{"root": str(c.root), "herculabels_version": c.meta.get("herculabels_version"),
                    "n_windows": len(c), "create": c.meta.get("create")} for c in inputs]
    out = Corpus.create(out_path, create_params={"mode": "merged", "merged_from": merged_from})
    print(f"[merge] {len(inputs)} corpus(es) → {out.root}", flush=True)

    chosen: dict = {}                                            # wid -> (entry, src corpus)
    collisions = 0
    for c in inputs:
        for entry in c.iter_windows():
            wid = entry["id"]
            prev = chosen.get(wid)
            if prev is None:
                chosen[wid] = (entry, c)
                continue
            collisions += 1
            if _prefer(entry, prev[0]):
                kept, dropped = c, prev[1]
                chosen[wid] = (entry, c)
            else:
                kept, dropped = prev[1], c
            print(f"[merge] duplicate {wid}: keeping {kept.root} over {dropped.root} "
                  f"(edited={bool(chosen[wid][0].get('edited'))})", flush=True)

    made = skipped = 0
    for wid, (entry, c) in chosen.items():
        if not c.window_path(wid).exists():                     # manifest entry with no file — skip, don't abort
            print(f"[merge] {wid}: source file missing in {c.root} — skipping", flush=True)
            skipped += 1
            continue
        out.import_window(c, entry)
        made += 1
    out.save()

    total_in = sum(len(c) for c in inputs)
    print(f"[merge] wrote {made} window(s) into {out.root} "
          f"({total_in} across inputs, {collisions} duplicate id(s) resolved"
          + (f", {skipped} skipped" if skipped else "") + ")", flush=True)

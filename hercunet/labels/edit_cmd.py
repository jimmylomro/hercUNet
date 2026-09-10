"""``hercunet labels edit`` — open an editable ``.herculabels`` corpus in the viewer.

Implicitly interactive: each window is loaded from the corpus (base meshes + meshlet cloud), shown in the
same viewer as ``create --interactive``, and written back on save+advance. No labels are re-generated. The
CT slice-pane background is re-read from the scroll on open (see :class:`hercunet.labels.grind.EditSession`).
"""

from __future__ import annotations


def edit_corpus(corpus_path: str, *, gpu: bool = True) -> None:
    """Launch the viewer over the corpus at ``corpus_path`` and grind through its windows for correction."""
    from ..config import Config
    from ..data import get_backend
    from ..viz.interactive import launch_interactive
    from .corpus import Corpus
    from .grind import EditSession

    corpus = Corpus.open(corpus_path)
    if len(corpus) == 0:
        raise SystemExit(f"corpus {corpus.root} has no windows to edit")
    print(f"[edit] {corpus.root}  ({len(corpus)} window(s))", flush=True)
    be = get_backend(Config.from_env())
    scroll = (corpus.meta.get("create") or {}).get("scroll")
    launch_interactive(EditSession(be, corpus, gpu, scroll=scroll))

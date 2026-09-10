"""``hercunet labels export`` — derive a TRAINING corpus from an editable ``.herculabels`` corpus.

Reads each corpus window (base sheet meshes + meshlet cloud) and writes the per-window training ``.npz``
the training pipeline consumes — the base record plus, by default, the merge/split augmentations
(:func:`hercunet.labels._generate.build_records`, regenerated from the stored meshlet cloud, no pipeline
re-run and no dense fields). Non-destructive and re-runnable: the corpus is never modified.

``--no-augment`` writes base-only samples (e.g. a validation set). The training consumer decides whether to
USE the augmentations that are present.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def export(corpus_path, train_out, *, augment=True, gpu=True,
           merge_size=3, n_merges=2, n_splits=2) -> None:
    """Export ``corpus_path`` (a ``.herculabels`` dir) into ``train_out`` as training samples. ``augment``
    (default True) also regenerates the augmentation records; ``gpu`` runs the ∇φ / refit on CUDA."""
    from .corpus import Corpus
    from ._generate import build_records
    import hercunet.labels.io.bundle as bundle

    corpus = Corpus.open(corpus_path)
    out = Path(train_out)
    out.mkdir(parents=True, exist_ok=True)
    n = len(corpus)
    print(f"[export] {corpus.root} → {out}  ({n} window(s), augment={augment})", flush=True)

    made = 0
    for k, entry in enumerate(corpus.iter_windows(), 1):
        wid = entry["id"]
        try:
            meta, meshes, ml = corpus.load_window(entry)
        except Exception as e:
            print(f"[export] {wid}: load failed ({type(e).__name__}: {e}) — skipping", flush=True)
            continue
        if not meshes:
            print(f"[export] {wid}: no sheets — skipping", flush=True)
            continue
        shape, vu = ml["shape"], ml["vu"]
        # seed from the window's master seed so a re-export is stable (augs need not be bit-exact, but a
        # deterministic draw keeps successive exports from drifting).
        seed = int(meta.get("master_seed", 0)) % (2**32)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed % (2**31))
        except Exception:
            pass
        records = build_records(meshes, ml, shape, vu, gpu=gpu, augment=augment,
                                merge_size=merge_size, n_merges=n_merges, n_splits=n_splits)
        z, y, x = meta["coords"]
        path = out / f"sample_s{meta['scroll']}_L{meta['level']}_z{z}_y{y}_x{x}.npz"
        bundle.save_sample_bundle(str(path), meta, records)
        made += 1
        print(f"[export] ({k}/{n}) {wid} → {path.name}  "
              f"({len(records)} records: 1 base + {len(records)-1} augs)", flush=True)

    manifest = {"herculabels_version": corpus.meta.get("herculabels_version"),
                "derived_from": str(corpus.root), "augmented": bool(augment),
                "params_hash": corpus.meta.get("create", {}).get("params_hash"),
                "exported": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "n_windows": made}
    (out / "export_meta.json").write_text(json.dumps(manifest, indent=2))
    print(f"[export] wrote {made}/{n} training sample(s) into {out}", flush=True)

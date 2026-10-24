"""Representation probes for HercuViT.

Quantitative analyses of the FROZEN Dinovol encoder's token embeddings (and, later, of the trained decoder),
used as the empirical evidence in the submission writeup (``submission/writeup/hercuvit.md``). Each probe is a
runnable module — e.g. ``python -m hercunet.hercuvit.probes.frozen_encoder --nwin 250 --out /root/enc_stats`` —
that writes a ``stats.json`` and paper figures. Pure-numpy statistics live in :mod:`hercunet.hercuvit.probes.stats`.
"""

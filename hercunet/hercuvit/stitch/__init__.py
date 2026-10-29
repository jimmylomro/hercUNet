"""HercuViT cross-window sheet-stitching: self-supervised pseudo-label extraction for the stitching decoder.

See ``docs/hercuvit-stitching.md`` for the design. The stitcher connects sheet fragments that the same physical
sheet produces in disjoint inference windows; this subpackage extracts its training samples (per-window
normalised-PCA spatial tokens + fragment median-embedding queries + a pairwise same-object label matrix) using
HercuViT's own predictions — no ground truth.
"""

# Vendored third-party code

`dinov2_eva.py`, `patch_encode_decode.py`, `rope.py`, `point_embeddings.py` are vendored (with import paths
rewritten to be package-relative) from **ScrollPrize / villa**, `dinovol/dinovol_2/` on branch
`dinovol-updates` (https://github.com/ScrollPrize/villa/tree/dinovol-updates), which is MIT-licensed.
They implement the Dinovol 3D DINO backbone (EVA ViT + 3D RoPE + CNN stem). HercuViT reuses them unmodified
except for the relative-import rewrite; all HercuViT-specific architecture (windowed retrofit, pixel decoder,
instance heads) lives outside this directory.

"""HercuViT full-volume inference — semantic sheet-surface over a scroll / zarr (``docs/hercuvit.md`` §9).

Reuses the hercunet inference engine's building blocks: :func:`hercunet.infer.nnunet_infer.open_source` (stream CT
from a scroll id / ``s3://`` / ``https://`` / local zarr), the Gaussian-blend tiling from
:mod:`hercunet.infer.jacobi_refine` (:func:`make_gaussian`, :func:`pass_blend_tiles` — fixed disjoint output tiles
each running the overlapping windows that cover them, so blending never races), and :func:`create_surface_zarr` /
:func:`finalize_pyramid` to write a VC3D-friendly OME-Zarr.

Single pass (no Jacobi ``prev`` channel — HercuViT is 1-channel), v1 output = the full-volume **semantic
sheet-surface probability** (uint8, ``p·255``). Per-tile instance masks and cross-tile sheet-ID stitching are
**future** (§9/§13); the loader can build the instance head, but v1 emits the surface only.
"""
from __future__ import annotations

import numpy as np
import torch


def _parse_region(spec, shape):
    """``z0:z1,y0:y1,x0:x1`` (level voxels) → ``(z0,z1,y0,y1,x0,x1)``; None → whole volume."""
    if not spec:
        return None
    try:
        zs, ys, xs = spec.split(",")
        z0, z1 = (int(v) for v in zs.split(":"))
        y0, y1 = (int(v) for v in ys.split(":"))
        x0, x1 = (int(v) for v in xs.split(":"))
    except ValueError as e:
        raise SystemExit(f"hercuvit infer: bad --region {spec!r}; want z0:z1,y0:y1,x0:x1") from e
    Z, Y, X = shape
    return (max(0, z0), min(Z, z1), max(0, y0), min(Y, y1), max(0, x0), min(X, x1))


def infer_volume(net, source, out_prefix, recipe, *, device="cuda", region=None, level=0,
                 overlap=0.25, air=25, chunk=128, finalise=True):
    """Stream ``source``, run HercuViT over Gaussian-blended 128³ windows, write ``{out_prefix}_sem.zarr`` (uint8).

    ``overlap`` = window overlap for the blend (0 → disjoint). ``air`` skips all-air windows (CT max < air). The
    model runs at ``recipe.window`` (128³); pick ``--level`` so the voxel size is ≈ ``recipe.target_voxel_um``.
    """
    from ...infer.nnunet_infer import (create_surface_zarr, finalize_pyramid, open_source,
                                       source_label)
    from ...infer.jacobi_refine import make_gaussian, pass_blend_tiles
    from ...data.prefetch import iter_windows
    from ..data.normalize import dinovol_normalise

    vol = open_source(source)
    shape = tuple(int(s) for s in vol.meta.level_shapes[level])
    factor = vol.meta.level_shapes[0][2] / vol.meta.level_shapes[level][2]
    voxel_um = float(vol.meta.voxel_size_um) * float(factor)
    if not (7.0 <= voxel_um <= 10.5):
        print(f"[hercuvit] WARNING: level {level} voxel ≈ {voxel_um:.2f} µm is outside the trained ~[7.5,10] µm "
              f"band — predictions will be unreliable. Pick a level nearer {recipe.target_voxel_um} µm.", flush=True)

    P = int(recipe.window)
    S = max(1, int(round(P * (1.0 - overlap)))) if overlap > 0 else P
    reg = _parse_region(region, shape) if isinstance(region, str) else region
    gauss = make_gaussian(P).astype(np.float32)

    label = source_label(source)
    out_path = f"{out_prefix}_sem.zarr"
    out = create_surface_zarr(out_path, shape, chunk, voxel_um, scroll=label, name=f"{label}-hercuvit-sem")

    tiles = pass_blend_tiles(shape, P, S, 0, reg, chunk, level=level)
    print(f"[hercuvit] {label}: level {level} shape {shape} voxel {voxel_um:.2f}µm — {len(tiles)} tiles, "
          f"window {P} stride {S}", flush=True)
    gauss_t = torch.from_numpy(gauss).to(device)

    done = 0
    for tile, (_req, block, _org) in zip(tiles, iter_windows(vol, [t["req"] for t in tiles], readahead=2, workers=3)):
        oz, oy, ox = tile["origin"]
        sz, sy, sx = tile["starts"]
        bz, by, bx = block.shape                                        # the tile's read-box extent
        acc = np.zeros((bz, by, bx), np.float32)
        wsum = np.zeros((bz, by, bx), np.float32)
        block = np.asarray(block)
        for z0 in sz:
            for y0 in sy:
                for x0 in sx:
                    lz, ly, lx = z0 - oz, y0 - oy, x0 - ox              # window offset within the read box
                    win = block[lz:lz + P, ly:ly + P, lx:lx + P]
                    if win.shape != (P, P, P) or int(win.max()) < air:
                        continue
                    ct = dinovol_normalise(win)
                    x = torch.from_numpy(ct).float()[None, None].to(device)
                    with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
                        prob = torch.softmax(net(x)["sem_logits"][0].float(), dim=1)[0, 1]  # (P,P,P) fg
                    pg = (prob * gauss_t).float().cpu().numpy()
                    acc[lz:lz + P, ly:ly + P, lx:lx + P] += pg
                    wsum[lz:lz + P, ly:ly + P, lx:lx + P] += gauss
        tz0, tz1, ty0, ty1, tx0, tx1 = tile["owned"]
        # owned box relative to the read-box origin
        rz0, ry0, rx0 = tz0 - oz, ty0 - oy, tx0 - ox
        rz1, ry1, rx1 = tz1 - oz, ty1 - oy, tx1 - ox
        sub_w = wsum[rz0:rz1, ry0:ry1, rx0:rx1]
        sub_a = acc[rz0:rz1, ry0:ry1, rx0:rx1]
        prob = np.where(sub_w > 1e-6, sub_a / np.maximum(sub_w, 1e-6), 0.0)
        out[tz0:tz1, ty0:ty1, tx0:tx1] = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
        done += 1
        if done % 50 == 0:
            print(f"[hercuvit] {done}/{len(tiles)} tiles", flush=True)

    if finalise:
        finalize_pyramid(out_path, region=reg)
    print(f"[hercuvit] wrote {out_path}", flush=True)
    return out_path


def single_instance(args) -> None:
    """``hercuvit infer single-instance`` handler — one box, semantic surface output."""
    import torch as _torch
    from .loader import load_hercuvit

    device = args.device
    if device == "cuda" and not _torch.cuda.is_available():
        print("[hercuvit] no CUDA visible — falling back to CPU (slow)", flush=True)
        device = "cpu"
    net, recipe, _cfg = load_hercuvit(args.model, model_hf=args.model_hf, ckpt=args.ckpt, device=device,
                                      attention_backend=args.attention_backend, with_instance=False)
    infer_volume(net, args.source, args.out, recipe, device=device, region=args.region, level=args.level,
                 overlap=args.overlap, air=args.air, finalise=(args.finalise != "no"))

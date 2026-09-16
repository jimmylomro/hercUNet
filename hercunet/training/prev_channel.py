"""Synthesise the ``prev`` input channel for the iterative ∇φ refiner (input = [CT, prev] → surface seg).

The refiner is one iteration-agnostic map ``D(CT, prev) -> surface``: where ``prev ≈ 0`` it must predict from CT
alone (bootstrap / air / a previous-pass miss — all the same correct output = the oracle truth); where ``prev``
is a nonzero crest it refines it against CT. Training teaches BOTH from a single window by feeding a corrupted
``prev`` while always supervising the clean base-record surface (docs/winding_field_segmentation_v2.md).

``prev`` is a continuous [0,1] crest field — at inference it is the model's own softmax surface-probability from
pass i-1; at training it is a decoded-record ``mag`` (the |∇φ| super-Gaussian bump, already ~[0,1]). The
corruption SOURCE is the precomputed aug records (merge/split variants stored per window at label-gen time), so
this module never invents geometry — it only *selects and stitches* already-valid candidate fields.

The signature behaviour ("these two sheets are clearly separate in my neighbour; here they're compressed and I
fused them → split them") is taught by the OCTANT tiling: fill each octant of ``prev`` from an independently
sampled candidate so a plausible-but-inconsistent "+"-seam meets at the window centre — exactly the inference
geometry where a half-window-shifted window's octants come from different previous-pass windows.

Pure numpy; no torch/nnunet. ``candidates[0]`` is conventionally the clean BASE decode.
"""
from __future__ import annotations

import numpy as np


def _octant_slices(shape, rng):
    """Yield (z_slice, y_slice, x_slice) for the 2x2x2 octants split at the window centre, with the split plane
    jittered by up to ±12.5% per axis so the seam is not always dead-centre (still interior at inference)."""
    cuts = []
    for n in shape:
        c = n // 2
        j = int(round(rng.uniform(-0.125, 0.125) * n))
        cuts.append(int(np.clip(c + j, n // 4, 3 * n // 4)))
    zc, yc, xc = cuts
    zs = [slice(0, zc), slice(zc, shape[0])]
    ys = [slice(0, yc), slice(yc, shape[1])]
    xs = [slice(0, xc), slice(xc, shape[2])]
    for a in zs:
        for b in ys:
            for c in xs:
                yield (a, b, c)


def _pick(candidates, rng, aug_prob, zero_prob):
    """Pick a candidate field for one region: zeros with prob ``zero_prob`` (a neighbour with no/lost prior),
    an AUG (index>=1) with prob ``aug_prob`` if any exist, else the clean BASE (index 0)."""
    r = rng.random()
    if r < zero_prob:
        return None                                                  # -> fill zeros
    if len(candidates) > 1 and r < zero_prob + aug_prob:
        return candidates[rng.integers(1, len(candidates))]
    return candidates[0]


def _oriented_slab_mask(shape, rng, thickness):
    """Boolean mask [Z,Y,X] of a flat band ("slab") of the given voxel ``thickness``, at a RANDOM isotropic
    orientation and a random position inside the window. A slab severs any sheet it crosses regardless of the
    sheet's own orientation — over many samples every sheet is cut at all angles, so no normal field is needed to
    produce ⟂-ish severances. Pure numpy (one Z*Y*X float field per call)."""
    Z, Y, X = shape
    u = rng.normal(size=3).astype(np.float32)
    n = float(np.linalg.norm(u))
    u = np.array([1.0, 0.0, 0.0], np.float32) if n < 1e-8 else (u / n)
    z = np.arange(Z, dtype=np.float32)[:, None, None]
    y = np.arange(Y, dtype=np.float32)[None, :, None]
    x = np.arange(X, dtype=np.float32)[None, None, :]
    d = u[0] * z + u[1] * y + u[2] * x                               # signed distance along the slab normal
    d0 = float(rng.uniform(float(d.min()), float(d.max())))
    return np.abs(d - d0) <= (0.5 * float(thickness))


def fragment_prev(prev, rng, *, frag_prob=0.5, half_prob=0.0, half_frac=(0.30, 0.70),
                  n_slabs=(0, 3), slab_thickness=(3, 60), n_boxes=(0, 3), box_frac=(0.05, 0.30)):
    """Cut/fragment a composed ``prev`` crest to teach GAP-BRIDGING — structured masked-autoencoding on the prev
    channel (docs/winding_field_segmentation_v2.md; the "next-retraining" fix for iterations that DENOISE/thin but
    don't re-connect). Zeros out oriented slabs (sever sheets into disconnected sections at any angle) and random
    boxes (holes / along-sheet breaks).

    PREV ONLY — the caller's target and loss mask are untouched, so the clean label still supervises the cut
    voxels as surface and the loss penalises not re-filling them (that penalty IS the bridge-the-gap signal). If a
    cut instead landed in the ignore label there'd be no signal, so we never touch the label here.

    TWO teaching modes, chosen per call:
    - ``half_prob``  : EDGE/HALF-SPACE cut — zero ALL prev on one side of a random plane, leaving the sheet
      entering from ONE edge only. This forces EXTRAPOLATION (extend the sheet from a single edge into unknown
      territory) = "complete from the info a neighbour window carried in", the multi-window / cross-window skill.
      ``half_frac`` is the fraction of the window removed (uniform range). This is the mode the plain slab lacks:
      a slab is a *band*, so it leaves BOTH ends visible (within-window INTERPOLATION), no matter how wide.
    - else (slabs + boxes): within-window completion. ``slab_thickness`` is the cut-size knob — thin bands teach
      single-shot within-window completion; wider bands a bigger both-ends gap (still interpolation).

    Applied with prob ``frag_prob`` (else prev is returned unchanged). Pure numpy.
    """
    if rng.random() >= frag_prob:
        return prev
    out = np.array(prev, np.float32)                                 # own copy; never mutate the caller's array
    shape = out.shape
    if half_prob > 0.0 and rng.random() < half_prob:                 # EDGE/HALF cut -> extend-from-one-edge
        u = rng.normal(size=3).astype(np.float32)
        n = float(np.linalg.norm(u)); u = np.array([1., 0., 0.], np.float32) if n < 1e-8 else u / n
        zz = np.arange(shape[0], dtype=np.float32)[:, None, None]
        yy = np.arange(shape[1], dtype=np.float32)[None, :, None]; xx = np.arange(shape[2], dtype=np.float32)[None, None, :]
        d = u[0] * zz + u[1] * yy + u[2] * xx
        d0 = float(np.quantile(d, float(rng.uniform(*half_frac))))   # remove this fraction of the window
        out[(d > d0) if rng.random() < 0.5 else (d < d0)] = 0.0
        return out
    for _ in range(int(rng.integers(n_slabs[0], n_slabs[1] + 1))):
        t = float(rng.uniform(*slab_thickness))
        out[_oriented_slab_mask(shape, rng, t)] = 0.0
    for _ in range(int(rng.integers(n_boxes[0], n_boxes[1] + 1))):
        f = float(rng.uniform(*box_frac))
        sz = [max(1, int(round(f * s))) for s in shape]
        o = [int(rng.integers(0, shape[i] - sz[i] + 1)) for i in range(3)]
        out[o[0]:o[0] + sz[0], o[1]:o[1] + sz[1], o[2]:o[2] + sz[2]] = 0.0
    return out


def compose_prev(candidates, rng, *, p_boot=0.3, tiling="octant",
                 aug_prob=0.5, zero_prob=0.15, noise_sigma=0.0,
                 frag_prob=0.3, frag_slab_max=60, frag_half_prob=0.4):
    """Build a ``prev`` field [Z,Y,X] float32 in [0,1] from decoded candidate mags.

    candidates : list of float32 [Z,Y,X] in [0,1]; candidates[0] = clean base decode, rest = aug variants.
    p_boot     : probability the WHOLE window is zeroed (bootstrap / iteration-1 behaviour).
    tiling     : "whole" (one candidate for the window) | "octant" (2x2x2, independent per octant, "+"-seam).
    aug_prob   : per-region probability of drawing an aug variant (vs the clean base).
    zero_prob  : per-region probability of a zero fill (partial-presence / a neighbour that missed).
    noise_sigma: optional additive Gaussian field noise (std, in [0,1] units), clipped back to [0,1].
    frag_prob  : probability of applying sheet-section dropout (:func:`fragment_prev`) to the composed prev —
                 the gap-bridging / masked-autoencoding signal. The run301 recipe sets this to 0.5 (see
                 ``hercunet.training.recipe``); set it to 0 for a no-fragmentation run.
    frag_slab_max: max slab thickness (voxels) for fragmentation — the "cut-size distribution" knob (higher =
                 more wide severances that only multi-pass iteration can bridge).

    The TARGET is always the clean base surface regardless of what this returns — "the output is always the
    truth". Returns zeros on bootstrap so the model learns cold prediction from CT alone; fragmentation is NOT
    applied on bootstrap windows (prev is already all-zero there).
    """
    base = np.asarray(candidates[0], np.float32)
    shape = base.shape
    if rng.random() < p_boot:
        return np.zeros(shape, np.float32)

    if tiling == "whole":
        pick = _pick(candidates, rng, aug_prob, zero_prob)
        prev = np.zeros(shape, np.float32) if pick is None else np.asarray(pick, np.float32).copy()
    elif tiling == "octant":
        prev = np.zeros(shape, np.float32)
        for sl in _octant_slices(shape, rng):
            pick = _pick(candidates, rng, aug_prob, zero_prob)
            if pick is not None:
                prev[sl] = np.asarray(pick, np.float32)[sl]
    else:
        raise ValueError(f"unknown tiling {tiling!r}")

    if frag_prob > 0.0:                                              # sheet-section dropout: teach gap-bridging
        prev = fragment_prev(prev, rng, frag_prob=frag_prob, half_prob=frag_half_prob,
                             slab_thickness=(3, int(frag_slab_max)))
    if noise_sigma > 0.0:
        prev = prev + rng.normal(0.0, noise_sigma, size=shape).astype(np.float32)
    return np.clip(prev, 0.0, 1.0)

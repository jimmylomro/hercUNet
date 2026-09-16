"""v3 building blocks: CT orientation field + affinity output head (winding_field_segmentation_v3 §2.3, §3).

Two pieces, both self-contained:

1. ``ct_orientation`` — the **carried orientation field**, computed from CT on the GPU as the smoothed
   **structure tensor** (6 unit-trace components). It is **sign-free by construction** (``J = Σ ∇g ∇gᵀ`` — the
   outer product carries orientation with no normal sign) and needs **no eigendecomposition** (just convs), so
   it is cheap enough to recompute every train step from the *augmented* CT (staying consistent under
   augmentation, which precomputed/stored tensor channels would not). Fed to the net as 6 extra input channels.
   Why CT-derived, not prediction-derived: at a merge the prediction fuses two wraps and its orientation
   averages into one, destroying the separation cue; the CT still shows both wrap orientations (v3 §2.3).

2. ``AffinityHeadNet`` — wraps a stock nnU-Net (the ``MultiTaskNet`` forward-hook pattern from
   ``nnUNetTrainer_MultiTaskSurface``) with a 1×1 conv **affinity head** off the last decoder feature map.
   Returns ``(seg, aff)`` — seg is the stock deep-supervision output (list in train), aff is the |offsets|-channel
   sigmoid affinity at full resolution, supervised by constrained MALIS (``malis.ConstrainedMalisLoss``).
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


# ------------------------- CT orientation field (sign-free structure tensor) -------------------------

def _gauss1d(sigma, order, device, dtype):
    r = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    if order == 1:
        g = g * (-x / (sigma ** 2))                       # derivative-of-Gaussian (unnormalised deriv kernel)
    return g


def _sepconv(v, kz, ky, kx):
    """Separable 3-axis conv of (B,1,D,H,W) with 1-D kernels along z,y,x (reflect pad)."""
    def ax(t, k, dim):
        r = k.numel() // 2
        w = k.view(1, 1, *([1] * (dim - 2)), -1, *([1] * (4 - dim)))
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (4 - dim)] = r; pad[2 * (4 - dim) + 1] = r
        t = F.pad(t, pad, mode="reflect")
        return F.conv3d(t, w)
    return ax(ax(ax(v, kz, 2), ky, 3), kx, 4)


def ct_orientation(ct, sigma_grad=1.0, sigma_tensor=3.0, eps=1e-6):
    """CT (B,1,D,H,W) -> orientation field (B,6,D,H,W): unit-trace structure tensor
    [Jzz,Jyy,Jxx,Jzy,Jzx,Jyx]. Sign-free, GPU, no eigendecomposition. sigma_* in voxels."""
    dev, dt = ct.device, ct.dtype
    ct = ct.float()
    g0 = _gauss1d(sigma_grad, 0, dev, torch.float32)
    g1 = _gauss1d(sigma_grad, 1, dev, torch.float32)
    gz = _sepconv(ct, g1, g0, g0)
    gy = _sepconv(ct, g0, g1, g0)
    gx = _sepconv(ct, g0, g0, g1)
    gt = _gauss1d(sigma_tensor, 0, dev, torch.float32)
    def integ(a):
        return _sepconv(a, gt, gt, gt)
    Jzz, Jyy, Jxx = integ(gz * gz), integ(gy * gy), integ(gx * gx)
    Jzy, Jzx, Jyx = integ(gz * gy), integ(gz * gx), integ(gy * gx)
    tr = Jzz + Jyy + Jxx + eps
    J = torch.cat([Jzz, Jyy, Jxx, Jzy, Jzx, Jyx], dim=1) / tr
    return J.to(dt)


# ------------------------- Affinity output head (MultiTaskNet forward-hook pattern) -------------------------

class AffinityHeadNet(nn.Module):
    """Wrap a stock nnU-Net with a full-res affinity head off the last decoder feature map.

    forward returns ``(seg, aff)``:
      * ``seg`` — the stock network output (deep-supervision list in train, tensor at eval);
      * ``aff`` — (B, n_aff, Z, Y, X) sigmoid affinities at full res.
    ``return_aff=False`` (eval default) returns seg only, so stock ``nnUNetv2_predict`` stays native on the
    surface head. Warm-start fills ``base.*`` from m7; the affinity head inits fresh.
    """

    def __init__(self, base: nn.Module, n_aff: int):
        super().__init__()
        self.base = base
        feat = base.decoder.stages[-1].output_channels if hasattr(base.decoder.stages[-1], "output_channels") \
            else self._infer_feat(base)
        self.aff_head = nn.Conv3d(feat, n_aff, kernel_size=1)
        self.return_aff = True
        self._feat = None
        base.decoder.stages[-1].register_forward_hook(self._grab)

    @staticmethod
    def _infer_feat(base):
        # last decoder stage's final conv out-channels
        m = base.decoder.stages[-1]
        for mod in reversed(list(m.modules())):
            if isinstance(mod, (nn.Conv3d, nn.Conv2d)):
                return mod.out_channels
        raise RuntimeError("could not infer decoder feature width for the affinity head")

    def _grab(self, module, inp, out):
        self._feat = out[0] if isinstance(out, (list, tuple)) else out

    def forward(self, x):
        seg = self.base(x)
        if not self.return_aff:
            return seg
        aff = torch.sigmoid(self.aff_head(self._feat))
        return seg, aff


# ------------------------- CT-material growth / air-suppression (winding_field_segmentation_v3 §4.3) -------------

def material_prob(ct, tau, scale):
    """Smooth, CT-DERIVED, label-free material scaffold: ``m(x) = σ((CT − τ)/s)`` ∈ [0,1]. High on papyrus,
    low in air. ``ct`` is the (augmented, normalized) CT the net sees; ``τ``/``s`` live in that normalized space
    (calibrate from a preprocessed case's CT histogram). This is the ONLY source of "where is material" — never
    the mesh-derived ``_lab`` band (that is a label artifact, absent at inference; see v3 §4.3)."""
    return torch.sigmoid((ct - tau) / scale)


def _across_quadform(orient, offset):
    """``across = δ̂ᵀ J δ̂`` ∈ [0,1] for the unit-trace structure tensor ``J`` (channels [Jzz,Jyy,Jxx,Jzy,Jzx,Jyx])
    and integer offset ``δ``. For a sheet ``J ≈ N Nᵀ`` so this ≈ ``(δ̂·N)²`` — 1 when ``δ`` is ACROSS the sheet
    (along the normal), 0 when ALONG the tangent plane. No eigendecomposition: a fixed constant-weighted sum of
    the 6 ST channels (for axis-aligned ``δ`` it is just the matching diagonal channel)."""
    dz, dy, dx = offset
    n2 = float(dz * dz + dy * dy + dx * dx)
    c = [dz * dz / n2, dy * dy / n2, dx * dx / n2, 2 * dz * dy / n2, 2 * dz * dx / n2, 2 * dy * dx / n2]
    across = (c[0] * orient[:, 0:1] + c[1] * orient[:, 1:2] + c[2] * orient[:, 2:3]
              + c[3] * orient[:, 3:4] + c[4] * orient[:, 4:5] + c[5] * orient[:, 5:6])
    return across.clamp(0.0, 1.0)


def material_grow_loss(aff, ct, orient, offsets, *, tau, scale):
    """Smooth-CT + ST-orientation gated growth / air-suppression on the affinity field (v3 §4.3).

    Two objectives, both driven by DENSE CT (not the sparse labels) so the net grows through material even
    where there is no label ("the data demands it"), and never through air:

      * GROW  (target 1, weight ``g·along``): where the edge is IN material (``g = m(u)·m(u+δ)``) AND ALONG the
        sheet tangent (``along = 1 − δ̂ᵀJδ̂``). Completes sheets through unlabeled material, in-plane only —
        across-normal growth (``along≈0``) is left to MALIS, so stacked sheets are not merged.
      * AIR   (target 0, weight ``1 − g``): where either endpoint is non-material. Hard no-grow through voids.

    ``aff`` (B,C,D,H,W) in [0,1]; ``ct`` (B,1,D,H,W) normalized; ``orient`` (B,6,D,H,W) unit-trace ST. MEMORY-LEAN:
    pass ``aff`` in its native (autocast fp16) dtype — no fp32 copy. Intermediates stay in aff.dtype (retained for
    backward); only the per-offset REDUCTIONS are cast to fp32 (fp16 .sum() over ~7M voxels would overflow).
    Weights are detached (data-derived, no grad). Returns (loss, grow, air)."""
    dt = aff.dtype
    m = material_prob(ct, tau, scale).detach().to(dt)                # (B,1,D,H,W) dense material scaffold
    orient = orient.to(dt)
    D, H, W = aff.shape[2:]
    f32 = torch.float32
    grow_num = torch.zeros((), device=aff.device, dtype=f32); grow_den = grow_num.clone()
    air_num = grow_num.clone(); air_den = grow_num.clone()
    for c, (dz, dy, dx) in enumerate(offsets):
        z0, z1 = max(0, -dz), D - max(0, dz)
        y0, y1 = max(0, -dy), H - max(0, dy)
        x0, x1 = max(0, -dx), W - max(0, dx)
        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            continue
        a = aff[:, c:c + 1, z0:z1, y0:y1, x0:x1]                     # predicted aff at u (view, no copy)
        m_u = m[:, :, z0:z1, y0:y1, x0:x1]
        m_v = m[:, :, z0 + dz:z1 + dz, y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        g = m_u * m_v                                               # edge material: both endpoints
        along = (1.0 - _across_quadform(orient, (dz, dy, dx))[:, :, z0:z1, y0:y1, x0:x1]).detach()
        w_grow = (g * along).detach()
        w_air = (1.0 - g).detach()
        grow_num = grow_num + (w_grow * (a - 1.0).pow(2)).float().sum()   # fp32 reduction (no fp16 overflow)
        grow_den = grow_den + w_grow.float().sum()
        air_num = air_num + (w_air * a.pow(2)).float().sum()
        air_den = air_den + w_air.float().sum()
    grow = grow_num / grow_den.clamp_min(1.0)
    air = air_num / air_den.clamp_min(1.0)
    return grow + air, grow.detach(), air.detach()

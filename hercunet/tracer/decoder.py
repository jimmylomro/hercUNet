"""The instance-query decoder head and its nnU-Net wrapper.

A small transformer decoder (DETR / MaskFormer / Mask2Former family, **deformable-style cross-attention**, 2
layers by default) reads the nnU-Net decoder feature map and emits, per coarse grid node, a distribution over
**instance queries** (the ownership softmax; computed in :mod:`hercunet.tracer.losses` / ``extract``). Each
query attends to the features (deformable sampling) and to the other queries (self-attention — the competition
that lets one query cut where another already owns a region).

``SheetSegHeadNet`` bolts it on **off the UNet decoder features directly** — its own forward-hook on
``decoder.stages[-1]``, independent of the affinity head. The **affinity head stays MALIS-only**; this head
never reads it. In the production multi-head net, surface/affinity/sheet heads each hook the shared feature map.

Inspiration: deformable-DETR sampling + MaskFormer query·feature mask logits, adapted to a 3-D coarse node
grid and a per-node (not per-pixel) softmax. torch only.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint


def _pool_to_nodes(feat, stride):
    """Average-pool a dense feature map (B,C,D,H,W) to the node grid (B,C,nd,nh,nw) at ``stride`` voxels."""
    return F.avg_pool3d(feat, kernel_size=stride, stride=stride, ceil_mode=True)


class DeformableCrossAttn(nn.Module):
    """Single-scale 3-D deformable cross-attention: each query samples ``n_points`` locations per head around
    its reference point on the node-feature grid and attends over them."""

    def __init__(self, dim, n_heads=4, n_points=8):
        super().__init__()
        assert dim % n_heads == 0
        self.dim, self.h, self.k = dim, n_heads, n_points
        self.hd = dim // n_heads
        self.sampling_offsets = nn.Linear(dim, n_heads * n_points * 3)
        self.attn_weights = nn.Linear(dim, n_heads * n_points)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.sampling_offsets.weight); nn.init.zeros_(self.sampling_offsets.bias)
        nn.init.zeros_(self.attn_weights.weight); nn.init.zeros_(self.attn_weights.bias)

    def forward(self, q, ref, node_feat):
        """q:(B,Q,dim) queries; ref:(B,Q,3) reference points in [-1,1] (z,y,x); node_feat:(B,dim,nd,nh,nw)."""
        B, Q, _ = q.shape
        nd, nh, nw = node_feat.shape[2:]
        val = self.value_proj(node_feat.flatten(2).transpose(1, 2))          # (B, N, dim)
        val = val.transpose(1, 2).reshape(B, self.dim, nd, nh, nw)
        off = self.sampling_offsets(q).view(B, Q, self.h, self.k, 3)
        aw = self.attn_weights(q).view(B, Q, self.h, self.k)
        aw = torch.softmax(aw, dim=-1)
        # sampling locations in [-1,1]; offsets are in normalised units (small, zero-init => start at ref)
        loc = ref.view(B, Q, 1, 1, 3) + 0.1 * off                            # (B,Q,h,k,3)
        out = q.new_zeros(B, Q, self.dim)
        for hh in range(self.h):
            vh = val[:, hh * self.hd:(hh + 1) * self.hd]                      # (B,hd,nd,nh,nw)
            # grid_sample wants grid last-dim order (x,y,z); our coords are (z,y,x) -> flip
            g = loc[:, :, hh].flip(-1).reshape(B, Q, self.k, 1, 3)            # (B,Q,k,1,3)
            samp = F.grid_sample(vh, g, mode="bilinear", padding_mode="border",
                                 align_corners=True)                          # (B,hd,Q,k,1)
            samp = samp.squeeze(-1).permute(0, 2, 3, 1)                       # (B,Q,k,hd)
            wsum = (samp * aw[:, :, hh].unsqueeze(-1)).sum(dim=2)             # (B,Q,hd)
            out[:, :, hh * self.hd:(hh + 1) * self.hd] = wsum
        return self.out_proj(out)


class DecoderLayer(nn.Module):
    def __init__(self, dim, n_heads=4, n_points=8, mlp_ratio=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.cross_attn = DeformableCrossAttn(dim, n_heads, n_points)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))
        self.n1 = nn.LayerNorm(dim); self.n2 = nn.LayerNorm(dim); self.n3 = nn.LayerNorm(dim)

    def forward(self, q, ref, node_feat):
        s = self.n1(q)
        q = q + self.self_attn(s, s, s, need_weights=False)[0]
        q = q + self.cross_attn(self.n2(q), ref, node_feat)
        q = q + self.mlp(self.n3(q))
        return q


class InstanceDecoder(nn.Module):
    """Queries → refined query embeddings → per-node ownership logits.

    forward(feat) returns ``ownership_logits`` (B, Q, nd, nh, nw): the (unnormalised) score of each query owning
    each node, as ``query · node_feature``. Query 0 is the reserved **null / no-sheet** slot. The per-node
    softmax over Q, the Hungarian matching and the losses live in :mod:`hercunet.tracer.losses`.

    **``node_stride`` is bounded by the sheet-representable resolution: L2 (stride 4 = 4× per axis, 64× volume)
    is the COARSEST that still represents sheets — coarser (L3, stride 8) merges adjacent wraps. Keep
    ``node_stride <= 4``; the owner-target pooling and grid-prev rasterise must use the SAME stride.**
    """

    def __init__(self, feat_ch, dim=128, n_queries=64, n_layers=2, n_heads=4, n_points=8, node_stride=4,
                 grad_checkpoint=False):
        super().__init__()
        self.dim, self.Q, self.node_stride = dim, n_queries, node_stride
        self.grad_checkpoint = bool(grad_checkpoint)
        self.node_proj = nn.Conv3d(feat_ch, dim, kernel_size=1)
        self.query_embed = nn.Embedding(n_queries, dim)
        self.ref_mlp = nn.Linear(dim, 3)                                     # reference point in [-1,1]
        self.layers = nn.ModuleList([DecoderLayer(dim, n_heads, n_points) for _ in range(n_layers)])
        self.mask_proj = nn.Linear(dim, dim)                                 # MaskFormer-style mask embedding

    def forward(self, feat, pre_pooled=False):
        # ``pre_pooled``: ``feat`` is already the stride-pooled node-feature grid (B,Cfeat,nd,nh,nw) — skip the
        # internal avg-pool. Used by decoder-only training on CACHED features (the frozen encoder is run once and
        # its decoder features pooled to the node grid and stored, so the big full-res map need not be kept).
        node_feat = self.node_proj(feat if pre_pooled else _pool_to_nodes(feat, self.node_stride))
        B = feat.shape[0]
        q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B,Q,dim)
        ref = torch.tanh(self.ref_mlp(q))                                    # (B,Q,3) in (-1,1)
        for layer in self.layers:
            if self.grad_checkpoint and self.training and q.requires_grad:
                q = torch.utils.checkpoint.checkpoint(layer, q, ref, node_feat, use_reentrant=False)
            else:
                q = layer(q, ref, node_feat)
        me = self.mask_proj(q)                                               # (B,Q,dim)
        # fp32 query·feature (disable autocast) — the fp16 einsum can overflow to inf when activations spike,
        # which becomes NaN in the ownership softmax and crashes the Hungarian match. Compute in fp32, then clamp
        # any residual non-finite value so ``own`` is ALWAYS finite (keeps the loss + grads finite for grad-clip).
        with torch.autocast(device_type=node_feat.device.type, enabled=False):
            own = torch.einsum("bqd,bdzyx->bqzyx", me.float(), node_feat.float()) / (self.dim ** 0.5)
        return torch.nan_to_num(own, nan=0.0, posinf=30.0, neginf=-30.0)


class SheetSegHeadNet(nn.Module):
    """Wrap a stock nnU-Net with the sheet-segmentation decoder head, hooked off the UNet decoder features.

    forward(x) returns ``(seg, own)``:
      * ``seg`` — the stock network output (deep-supervision list in train, tensor at eval), unchanged;
      * ``own`` — (B, Q, nd, nh, nw) ownership logits from the decoder.
    ``return_own=False`` returns ``seg`` only (keeps stock ``nnUNetv2_predict`` native on the surface head).
    The hook is this head's own (``decoder.stages[-1]``) and is independent of any affinity head.
    """

    def __init__(self, base: nn.Module, *, dim=128, n_queries=64, n_layers=2, n_heads=4, n_points=8,
                 node_stride=4):
        super().__init__()
        self.base = base
        feat = self._infer_feat(base)
        self.decoder = InstanceDecoder(feat, dim=dim, n_queries=n_queries, n_layers=n_layers,
                                       n_heads=n_heads, n_points=n_points, node_stride=node_stride)
        self.return_own = True
        self._feat = None
        base.decoder.stages[-1].register_forward_hook(self._grab)

    @staticmethod
    def _infer_feat(base):
        last = base.decoder.stages[-1]
        if hasattr(last, "output_channels"):
            return last.output_channels
        for mod in reversed(list(last.modules())):
            if isinstance(mod, (nn.Conv3d, nn.Conv2d)):
                return mod.out_channels
        raise RuntimeError("could not infer decoder feature width for the sheet-seg head")

    def _grab(self, module, inp, out):
        self._feat = out[0] if isinstance(out, (list, tuple)) else out

    def forward(self, x):
        seg = self.base(x)
        if not self.return_own:
            return seg
        own = self.decoder(self._feat)
        return seg, own

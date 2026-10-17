"""Pillar 1 — windowed-attention retrofit of the Dinovol transformer (ViTDet-style).

The 10 Dinovol blocks are retrofitted so a subset keep **global** attention and the rest use
**non-overlapping windows** of ``window_size`` tokens (default 8³ → 2×2×2 = 8 windows over the 16³ token grid).
Interleaved global blocks propagate information across windows; there are **no shifted windows**.

Design invariants (``docs/hercuvit.md`` §1, §6.1), all preserved here:

- **Weights are untouched.** We do not change any parameter or module shape — the retrofit only reassigns the
  attention instance's ``__class__`` to :class:`WindowedEvaAttention` and sets ``window_size``. The qkv / proj /
  sub-norm / LayerScale / RoPE weights are exactly Dinovol's, so the checkpoint loads unchanged (this is what
  keeps conflict #1 in the doc's §12 closed).
- **RoPE stays absolute.** Rotary embeddings are applied to q/k at their absolute 16³ coordinates (exactly as
  the global blocks do, via the block's own mixed-RoPE ``rope_embed``) *before* the tokens are partitioned into
  windows, so the pretrained frequencies keep their meaning.
- **The 5 special tokens (cls + 4 register) pass through windowed blocks unchanged.** In a windowed block the
  attention sub-layer produces a zero update for the prefix tokens (so the residual leaves them untouched) and
  the patch tokens attend only within their window and never to the prefix tokens. The per-token MLP sub-layer
  still runs on every token, exactly as in a global block. The prefix tokens therefore mix only in the global
  blocks.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from ..dinovol.dinov2_eva import EvaAttention, apply_rotary_embedding

#: Default 1-indexed blocks that keep global attention (design §5.1: blocks 4, 7, 10 of 10).
DEFAULT_GLOBAL_BLOCKS: tuple[int, ...] = (4, 7, 10)
#: Default window size in tokens per axis (8³ tokens = 64³ voxels at patch 8 → 2×2×2 windows over a 16³ grid).
DEFAULT_WINDOW_SIZE: tuple[int, int, int] = (8, 8, 8)


class WindowedEvaAttention(EvaAttention):
    """``EvaAttention`` whose patch tokens attend only within non-overlapping ``window_size`` windows.

    Shares all parameters with the base attention (this class only overrides ``forward``). The instance gains a
    ``window_size`` attribute; everything else — qkv projection, bias handling, RoPE, post-attention sub-norm,
    output projection — is the Dinovol attention, unchanged.
    """

    #: set by :func:`retrofit_windowed`; tokens-per-axis of each window.
    window_size: tuple[int, int, int] = DEFAULT_WINDOW_SIZE

    def forward(self, x, rope=None, attn_mask=None, rope_shape=None):  # noqa: D401 - matches base signature
        if attn_mask is not None:
            raise ValueError("windowed attention does not support an explicit attn_mask")
        if rope_shape is None:
            raise ValueError("windowed attention requires rope_shape (the token grid, e.g. (16,16,16))")
        B, N, C = x.shape
        P = self.num_prefix_tokens

        # --- qkv (identical to EvaAttention.forward) ---
        if self.qkv is not None:
            if self.q_bias is None:
                qkv = self.qkv(x)
            else:
                qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
                if self.qkv_bias_separate:
                    qkv = self.qkv(x)
                    qkv = qkv + qkv_bias
                else:
                    qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # each (B, num_heads, N, head_dim)
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

        # --- RoPE at absolute coordinates (prefix tokens skipped), exactly as the global path ---
        if rope is not None:
            q = apply_rotary_embedding(q, rope, prefix_tokens=P).type_as(v)
            k = apply_rotary_embedding(k, rope, prefix_tokens=P).type_as(v)

        # --- windowed attention on the patch tokens only ---
        attn_pat = self._windowed_sdpa(q[:, :, P:], k[:, :, P:], v[:, :, P:], tuple(rope_shape))  # (B,H,G,D)
        patch = attn_pat.transpose(1, 2).reshape(B, N - P, -1)  # (B, G, H*D)
        patch = self.norm(patch)          # per-token sub-norm (identical to attending all then slicing)
        patch = self.proj(patch)
        patch = self.proj_drop(patch)

        # prefix tokens: zero update → residual leaves cls + register tokens unchanged in windowed blocks
        out = patch.new_zeros(B, N, patch.shape[-1])
        out[:, P:] = patch
        return out

    def _windowed_sdpa(self, q, k, v, spatial_shape):
        """Per-window scaled-dot-product attention. q/k/v: ``(B, H, G, D)`` patch tokens in (z,y,x) raster order."""
        B, H, G, D = q.shape
        Dz, Dy, Dx = spatial_shape
        wz, wy, wx = self.window_size
        if Dz % wz or Dy % wy or Dx % wx:
            raise ValueError(f"token grid {spatial_shape} not divisible by window {self.window_size}")
        if Dz * Dy * Dx != G:
            raise ValueError(f"patch-token count {G} != prod(spatial_shape {spatial_shape})")
        nz, ny, nx = Dz // wz, Dy // wy, Dx // wx

        def to_windows(t):
            t = t.view(B, H, nz, wz, ny, wy, nx, wx, D)
            t = t.permute(0, 2, 4, 6, 1, 3, 5, 7, 8).contiguous()      # (B, nz, ny, nx, H, wz, wy, wx, D)
            return t.view(B * nz * ny * nx, H, wz * wy * wx, D)

        out = super()._attention(to_windows(q), to_windows(k), to_windows(v))  # (B*nwin, H, w³, D)
        out = out.view(B, nz, ny, nx, H, wz, wy, wx, D)
        out = out.permute(0, 4, 1, 5, 2, 6, 3, 7, 8).contiguous()              # (B, H, nz, wz, ny, wy, nx, wx, D)
        return out.view(B, H, G, D)


class SpatialReductionAttention(EvaAttention):
    """``EvaAttention`` for *global* blocks of the slim variants: queries stay full-resolution while the patch
    keys/values are spatially average-pooled by ``sr_factor`` (PVT-style), so global mixing costs a fraction.

    Parameter-preserving (only overrides ``forward``). RoPE is applied to q and k at absolute coordinates
    *before* the keys are pooled (an approximation the distilled student is trained to compensate for, design
    §5.2 / §8.1). The prefix tokens (cls + register) keep their full keys/values and are updated normally — this
    is a global block, so they mix here.
    """

    sr_factor: tuple[int, int, int] = (2, 2, 2)

    def forward(self, x, rope=None, attn_mask=None, rope_shape=None):  # noqa: D401 - matches base signature
        if attn_mask is not None:
            raise ValueError("spatial-reduction attention does not support an explicit attn_mask")
        if rope_shape is None:
            raise ValueError("spatial-reduction attention requires rope_shape (the token grid)")
        B, N, C = x.shape
        P = self.num_prefix_tokens

        if self.qkv is not None:
            if self.q_bias is None:
                qkv = self.qkv(x)
            else:
                qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
                if self.qkv_bias_separate:
                    qkv = self.qkv(x)
                    qkv = qkv + qkv_bias
                else:
                    qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

        if rope is not None:
            q = apply_rotary_embedding(q, rope, prefix_tokens=P).type_as(v)
            k = apply_rotary_embedding(k, rope, prefix_tokens=P).type_as(v)

        kp = self._pool(k[:, :, P:], tuple(rope_shape))
        vp = self._pool(v[:, :, P:], tuple(rope_shape))
        k2 = torch.cat([k[:, :, :P], kp], dim=2)   # prefix keys kept full + pooled patch keys
        v2 = torch.cat([v[:, :, :P], vp], dim=2)

        out = super()._attention(q, k2, v2)        # queries full-res, reduced kv
        out = out.transpose(1, 2).reshape(B, N, -1)
        out = self.norm(out)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def _pool(self, t, spatial_shape):
        B, H, G, D = t.shape
        Dz, Dy, Dx = spatial_shape
        fz, fy, fx = self.sr_factor
        if Dz % fz or Dy % fy or Dx % fx:
            raise ValueError(f"token grid {spatial_shape} not divisible by sr_factor {self.sr_factor}")
        t = t.view(B, H, Dz // fz, fz, Dy // fy, fy, Dx // fx, fx, D)
        t = t.mean(dim=(3, 5, 7))
        return t.reshape(B, H, (Dz // fz) * (Dy // fy) * (Dx // fx), D)


def retrofit_windowed(
    backbone,
    *,
    global_blocks: Iterable[int] = DEFAULT_GLOBAL_BLOCKS,
    window_size: Sequence[int] = DEFAULT_WINDOW_SIZE,
    global_attention: str = "full",
    sr_factor: Sequence[int] = (2, 2, 2),
):
    """In-place: every block except ``global_blocks`` (1-indexed) uses windowed attention; the global blocks use
    ``global_attention`` (``"full"`` for hercuvit-small, ``"spatial_reduction"`` for tiny/nano).

    Parameter-preserving — only reassigns each block's ``attn.__class__`` and sets ``window_size`` / ``sr_factor``.
    Returns ``backbone``. Requires ``attention_type == "full"`` (axial blocks unsupported).
    """
    if global_attention not in ("full", "spatial_reduction"):
        raise ValueError(f"global_attention must be 'full' or 'spatial_reduction', got {global_attention!r}")
    blocks = list(backbone.blocks)
    global_set = {int(i) for i in global_blocks}
    win = tuple(int(w) for w in window_size)
    srf = tuple(int(s) for s in sr_factor)
    n_windowed = n_sr = 0
    for idx1, block in enumerate(blocks, start=1):
        attn = getattr(block, "attn", None)
        if attn is None:
            raise AttributeError(f"block {idx1} has no .attn — unexpected backbone layout")
        if getattr(block, "axial_attns", None) and len(block.axial_attns) > 0:
            raise ValueError("windowed retrofit requires attention_type='full' (found axial sub-attentions)")
        if idx1 in global_set:
            if global_attention == "spatial_reduction":
                attn.__class__ = SpatialReductionAttention
                attn.sr_factor = srf
                n_sr += 1
            continue
        attn.__class__ = WindowedEvaAttention
        attn.window_size = win
        n_windowed += 1
    backbone._hercuvit_windowed = {
        "global_blocks": sorted(global_set), "window_size": win, "global_attention": global_attention,
        "sr_factor": srf, "n_windowed": n_windowed, "n_spatial_reduction": n_sr,
    }
    return backbone

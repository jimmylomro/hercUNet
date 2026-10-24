"""Unit tests for the QK-normed mask-decoder attention (``model.decoder_common.QKNormAttention``).

The point of QK-norm is that the pre-softmax logits stay bounded regardless of the magnitude of the projected
queries/keys — the structural fix for the mask-decoder attention-logit explosion (unbounded key norm → runaway
cross-attention logits). These tests pin that invariant plus the Mask2Former fully-masked-row guard.
"""
import math

import torch

from hercunet.hercuvit.model.decoder_common import QKNormAttention, MaskDecoderLayer


def _logits(attn: QKNormAttention, query, key):
    """Recompute the post-QK-norm pre-softmax logits the way SDPA would (for inspection only)."""
    q = attn.q_norm(attn._heads(attn.q_proj(query)))      # (B,H,Lq,hd)
    k = attn.k_norm(attn._heads(attn.k_proj(key)))        # (B,H,Lk,hd)
    return torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(attn.head_dim)


def test_shape_and_finiteness():
    torch.manual_seed(0)
    attn = QKNormAttention(32, 4)
    q = torch.randn(2, 5, 32); kv = torch.randn(2, 7, 32)
    out = attn(q, kv, kv)
    assert out.shape == (2, 5, 32)
    assert torch.isfinite(out).all()


def test_logits_bounded_under_huge_inputs():
    """Blow the key/query projection up by ×1000; QK-norm must keep the logits O(1) (unlike a raw dot)."""
    torch.manual_seed(0)
    attn = QKNormAttention(64, 8).eval()
    with torch.no_grad():
        attn.k_proj.weight.mul_(1000.0); attn.q_proj.weight.mul_(1000.0)
    q = torch.randn(2, 6, 64) * 50.0; kv = torch.randn(2, 9, 64) * 50.0
    lg = _logits(attn, q, kv)
    # head_dim=8 → |logit| <= sqrt(head_dim) ~ 2.83 for unit-RMS q,k; allow generous slack but FAR below the ~2e4 blow-up.
    assert lg.abs().max().item() < 20.0, lg.abs().max().item()
    assert torch.isfinite(attn(q, kv, kv)).all()


def test_fully_masked_row_is_finite():
    """A query row masked against every key (True=masked everywhere) must not return NaN (SDPA all--inf row guard)."""
    torch.manual_seed(0)
    attn = QKNormAttention(32, 4)
    B, H, Lq, Lk = 2, 4, 5, 7
    q = torch.randn(B, Lq, 32); kv = torch.randn(B, Lk, 32)
    mask = torch.zeros(B * H, Lq, Lk, dtype=torch.bool)
    mask[:, 0, :] = True                                   # row 0 fully masked for every (batch·head)
    out = attn(q, kv, kv, attn_mask=mask)
    assert torch.isfinite(out).all()


def test_mask_blocks_attention():
    """With all but one key masked, the output must equal that key's value projection (attention collapses to it)."""
    torch.manual_seed(0)
    attn = QKNormAttention(16, 2).eval()
    B, H, Lq, Lk = 1, 2, 3, 4
    q = torch.randn(B, Lq, 16); kv = torch.randn(B, Lk, 16)
    mask = torch.ones(B * H, Lq, Lk, dtype=torch.bool)    # mask everything...
    mask[:, :, 2] = False                                 # ...except key index 2
    out = attn(q, kv, kv, attn_mask=mask)
    # all attention mass lands on key 2 → attended value == v[...,2] for every query; compare through out_proj.
    v = attn._heads(attn.v_proj(kv))                      # (B,H,Lk,hd)
    attended = v[:, :, 2:3].expand(B, H, Lq, attn.head_dim)
    ref = attn.out_proj(attended.transpose(1, 2).reshape(B, Lq, 16))
    assert torch.allclose(out, ref, atol=1e-4), (out - ref).abs().max().item()


def test_decoder_layer_uses_qknorm():
    layer = MaskDecoderLayer(32, 4)
    assert isinstance(layer.cross, QKNormAttention)
    assert isinstance(layer.self_attn, QKNormAttention)

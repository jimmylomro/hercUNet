"""Design-B pixel decoder — deformable neck, PatchDecode tail + surface gate, and the gated-K/V instance head.

Shape + gradient contracts (CPU, small grids) for the modules in ``docs/dinovol_sheet_architecture.md`` §3.
"""
import torch

from hercunet.hercuvit.model.neck import DeformableNeck
from hercunet.hercuvit.model.patch_decoder import PatchDecoder
from hercunet.hercuvit.model.instance_head import InstanceHead, mask_logits_from_embed


def _grids(side8=4, token_dim=288, s2_dim=128):
    """A tiny 1/8 token grid + 1/4 s2 skip (side8=4 → 1/1 = 32³, cheap on CPU)."""
    t = torch.randn(1, token_dim, side8, side8, side8)
    s = torch.randn(1, s2_dim, side8 * 2, side8 * 2, side8 * 2)
    return t, s


def test_deformable_neck_is_identity_at_init():
    t, s = _grids()
    neck = DeformableNeck(288, 128, n_layers=2, n_heads=4, n_points=4)
    out = neck(t, s)
    assert out.shape == t.shape
    # output projections + MLP tails are zero-init and residual, so the neck starts as a pass-through.
    assert torch.allclose(out, t, atol=1e-5)


def test_patchdecoder_shapes_and_gate():
    for neck in ("deformable", "none"):
        t, s = _grids()
        pd = PatchDecoder(token_dim=288, s2_dim=128, mask_dim=16, n_sem_classes=2,
                          neck=neck, neck_layers=2, surface_gate=True, deep_supervision=True)
        out = pd(t, s)
        D = t.shape[-1] * 8                                   # 1/1 side
        assert out["sem_logits"][0].shape == (1, 2, D, D, D)
        assert len(out["sem_logits"]) == 3                    # full, 1/2, 1/4 deep supervision
        assert out["mask_features"].shape == (1, 16, D, D, D)
        assert out["gated_features"].shape == (1, 16, D, D, D)
        # the gate scales mask features by σ(surface) ∈ [0,1], so |gated| ≤ |mask features| everywhere.
        assert (out["gated_features"].abs() <= out["mask_features"].abs() + 1e-6).all()


def test_gated_instance_head_disconnects_raw_tokens():
    """Design B: the instance head reasons on the gated features only — raw tokens carry no gradient to it."""
    side8 = 4
    D = side8 * 8
    tokens = torch.randn(1, 288, side8, side8, side8, requires_grad=True)
    gated = torch.randn(1, 16, D, D, D, requires_grad=True)
    head = InstanceHead(token_dim=288, feat_14_dim=128, dim=32, n_queries=8, n_layers=2, mask_dim=16,
                        add_null=True, fps_queries=True, cross_attn="masked", gated_kv=True)
    head.train()
    owner = torch.zeros(1, D, D, D)
    owner[:, 2:6, 4:12, 4:12] = 1
    res = head(tokens, None, mask_features=gated, dn_owner=owner)
    lay = res["layers"][-1]
    assert lay["class_logits"].shape[-1] == 2 and lay["mask_embed"].shape[-1] == 16
    assert res["n_match"] == 9                                # 8 queries + 1 null
    assert "dn_sheet" in res                                  # a denoise group was appended (train + dn_owner)
    ml = mask_logits_from_embed(lay["mask_embed"], gated)
    assert ml.shape == (1, lay["mask_embed"].shape[1], D, D, D)
    (ml.float().mean() + lay["class_logits"].float().mean()).backward()
    assert gated.grad is not None
    assert tokens.grad is None                                # raw frozen tokens are disconnected from the queries

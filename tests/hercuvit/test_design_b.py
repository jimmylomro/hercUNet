"""Design-B pixel decoder — deformable neck, PatchDecode tail + surface concat, and the concat-K/V instance head.

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


def test_patchdecoder_shapes_and_concat():
    for neck in ("deformable", "none"):
        t, s = _grids()
        pd = PatchDecoder(token_dim=288, s2_dim=128, mask_dim=16, n_sem_classes=2,
                          neck=neck, neck_layers=2, surface_gate=True, deep_supervision=True)
        out = pd(t, s)
        D = t.shape[-1] * 8                                   # 1/1 side
        assert out["sem_logits"][0].shape == (1, 2, D, D, D)
        assert len(out["sem_logits"]) == 3                    # full, 1/2, 1/4 deep supervision
        assert out["mask_features"].shape == (1, 16, D, D, D)
        # instance features = concat([softmax(surface) (2 dims), mask_feat (16)]) → 18 dims; §3
        assert pd.instance_feat_dim == 18
        assert out["instance_features"].shape == (1, 18, D, D, D)
        # the first n_sem dims ARE the surface softmax: per-voxel they sum to 1 and lie in [0,1] (a direction, not a gate)
        surf = out["instance_features"][:, :2]
        assert torch.allclose(surf.sum(dim=1), torch.ones(1, D, D, D), atol=1e-4)
        assert (surf >= -1e-6).all() and (surf <= 1 + 1e-6).all()
        # the remaining dims are the raw mask features, unmodified (not magnitude-gated)
        assert torch.allclose(out["instance_features"][:, 2:], out["mask_features"], atol=1e-6)


def test_decoder_trunk_freeze_keeps_mask_head_trainable():
    """set_decoder_trunk_trainable(False) freezes neck + upsample tail + surface head, but the mask head (untrained
    after a surface-only warm-start) stays trainable, as does the instance head."""
    import torch.nn as nn
    from hercunet.hercuvit.model.hercuvit_net import HercuVitNet
    from hercunet.hercuvit.model.patch_decoder import PatchDecoder

    class _Stub(nn.Module):  # a tiny backbone stub so HercuVitNet builds without the real encoder
        def __init__(self): super().__init__(); self.p = nn.Parameter(torch.zeros(1)); self.return_embed_skips = True
    net = HercuVitNet(_Stub(), token_dim=288, decoder="patchdecode", neck="deformable", neck_layers=1,
                      mask_dim=16, surface_gate=True, with_instance=True, fps_queries=True, cross_attn="masked",
                      add_null_query=True)
    net.set_decoder_trunk_trainable(False)
    pd = net.pixel_decoder
    assert isinstance(pd, PatchDecoder)
    assert all(not p.requires_grad for p in pd.neck.parameters())          # neck frozen
    assert all(not p.requires_grad for p in pd.surface_head.parameters())  # surface head frozen
    assert all(not p.requires_grad for p in pd.up4.parameters())           # upsample tail frozen
    assert all(p.requires_grad for p in pd.mask_head.parameters())         # mask head STILL trainable
    assert all(p.requires_grad for p in net.instance_head.parameters())    # instance head trainable
    net.set_decoder_trunk_trainable(True)                                   # unfreeze → all back on
    assert all(p.requires_grad for p in pd.neck.parameters())
    assert all(p.requires_grad for p in pd.surface_head.parameters())


def test_concat_instance_head_disconnects_raw_tokens():
    """Design B: the instance head reasons on the surface-concat features only (feat_dim = n_sem+mask_dim) — raw
    tokens carry no gradient to it, and mask_embed writes into the full concat width."""
    side8 = 4
    D = side8 * 8
    tokens = torch.randn(1, 288, side8, side8, side8, requires_grad=True)
    inst_feat = torch.randn(1, 18, D, D, D, requires_grad=True)   # [2 surface | 16 mask-feat] concat width
    head = InstanceHead(token_dim=288, feat_14_dim=128, dim=32, n_queries=8, n_layers=2, mask_dim=16, feat_dim=18,
                        add_null=True, fps_queries=True, cross_attn="masked", gated_kv=True)
    head.train()
    assert head.n_sem == 2                                    # 2 surface concat dims
    owner = torch.zeros(1, D, D, D)
    owner[:, 2:6, 4:12, 4:12] = 1
    res = head(tokens, None, mask_features=inst_feat, dn_owner=owner)
    lay = res["layers"][-1]
    assert lay["class_logits"].shape[-1] == 2 and lay["mask_embed"].shape[-1] == 18   # writes full concat width
    assert res["n_match"] == 9                                # 8 queries + 1 null
    assert "dn_sheet" in res                                  # a denoise group was appended (train + dn_owner)
    ml = mask_logits_from_embed(lay["mask_embed"], inst_feat)
    assert ml.shape == (1, lay["mask_embed"].shape[1], D, D, D)
    (ml.float().mean() + lay["class_logits"].float().mean()).backward()
    assert inst_feat.grad is not None
    assert tokens.grad is None                                # raw frozen tokens are disconnected from the queries

"""HercuViT losses — semantic deep-sup downsampling, Hungarian identity recovery, has_identity gating, e2e."""
import numpy as np
import torch

from hercunet.hercuvit.train import losses as L
from hercunet.hercuvit.train.recipe import HERCUVIT


def test_semantic_target_downsample_preserves_surface():
    tgt = torch.zeros(1, 1, 8, 8, 8)
    tgt[0, 0, 3, 4, 5] = 1.0                      # a single surface voxel
    half = L._downsample_target(tgt, 4)
    quarter = L._downsample_target(tgt, 2)
    assert half.shape[-1] == 4 and quarter.shape[-1] == 2
    # max-pool keeps the surface alive at every scale (a mean/area pool could erase it).
    assert float(half.sum()) == 1.0
    assert float(quarter.sum()) == 1.0


def test_semantic_loss_runs_and_backprops():
    torch.manual_seed(0)
    logits = [torch.randn(1, 2, 8, 8, 8, requires_grad=True),
              torch.randn(1, 2, 4, 4, 4, requires_grad=True),
              torch.randn(1, 2, 2, 2, 2, requires_grad=True)]
    target = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    target[0, 2:5, 2:5, 2:5] = 1
    out = L.semantic_loss(logits, target, HERCUVIT)
    assert out["sem"].requires_grad and torch.isfinite(out["sem"])
    out["sem"].backward()
    assert logits[0].grad is not None and torch.isfinite(logits[0].grad).all()


def test_hungarian_recovers_identity():
    # 3 disjoint GT masks over P points; predictions are the same masks in a known permutation.
    torch.manual_seed(0)
    P = 30
    gt = torch.zeros(3, P)
    gt[0, 0:10] = 1.0
    gt[1, 10:20] = 1.0
    gt[2, 20:30] = 1.0
    perm = [2, 0, 1]                              # query i predicts GT perm[i]
    pred = torch.stack([(2 * gt[perm[i]] - 1) * 10.0 for i in range(3)], 0)   # confident logits
    cost = (HERCUVIT.cost_mask_bce * L._bce_cost(pred, gt)
            + HERCUVIT.cost_mask_dice * L._dice_cost(torch.sigmoid(pred), gt))
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(cost.numpy())
    recovered = {int(r): int(c) for r, c in zip(rows, cols)}
    assert recovered == {0: 2, 1: 0, 2: 1}


def _toy_instance(B=1, Q=4, C=3, side=8, mask_dim=3, requires_grad=False):
    g = side ** 3
    objectness = torch.randn(B, g, requires_grad=requires_grad)
    layers = [{
        "class_logits": torch.randn(B, Q, 2, requires_grad=requires_grad),
        "mask_embed": torch.randn(B, Q, mask_dim, requires_grad=requires_grad),
        "anchors": torch.zeros(B, Q, 3),
    } for _ in range(2)]
    return {"objectness": objectness, "query_idx": torch.zeros(B, Q, dtype=torch.long), "layers": layers}


def test_instance_loss_zero_and_zero_grad_when_no_identity():
    torch.manual_seed(0)
    side, mask_dim = 8, 3
    inst = _toy_instance(side=side, mask_dim=mask_dim, requires_grad=True)
    mask_features = torch.randn(1, mask_dim, side, side, side, requires_grad=True)
    owner = torch.zeros(1, side, side, side, dtype=torch.long)
    out = L.instance_loss(inst, mask_features, owner, has_identity=[False], recipe=HERCUVIT)
    # gated off → exactly zero, but STILL grad-connected (so the optimiser step is well-defined / DDP-safe).
    assert float(out["inst"]) == 0.0
    assert out["inst"].requires_grad
    out["inst"].backward()
    assert torch.count_nonzero(inst["objectness"].grad) == 0           # grad exists and is all-zero
    assert torch.count_nonzero(inst["layers"][-1]["mask_embed"].grad) == 0


def test_instance_loss_supervises_with_identity():
    torch.manual_seed(0)
    side, mask_dim = 8, 3
    inst = _toy_instance(side=side, mask_dim=mask_dim, requires_grad=True)
    mask_features = torch.randn(1, mask_dim, side, side, side, requires_grad=True)
    owner = torch.zeros(1, side, side, side, dtype=torch.long)
    owner[0, 1:4, 1:4, 1:4] = 1
    owner[0, 5:7, 5:7, 5:7] = 2
    out = L.instance_loss(inst, mask_features, owner, has_identity=[True], recipe=HERCUVIT)
    assert out["inst"].requires_grad and torch.isfinite(out["inst"])
    assert float(out["inst"]) > 0.0
    out["inst"].backward()
    assert mask_features.grad is not None and torch.isfinite(mask_features.grad).all()


def test_end_to_end_pillars_forward_backward():
    """Tiny end-to-end through the REAL pillar heads (pixel decoder + instance head) + both losses."""
    torch.manual_seed(0)
    from hercunet.hercuvit.model.pixel_decoder import PixelDecoder
    from hercunet.hercuvit.model.instance_head import InstanceHead

    token_dim, mask_dim = 8, 3
    dec = PixelDecoder(token_dim=token_dim, skip_channels=(2, 4, 6),
                       decoder_channels=(8, 6, 4, 2), n_sem_classes=2, mask_dim=mask_dim, deep_supervision=True)
    inst = InstanceHead(token_dim=token_dim, feat_14_dim=6, dim=4, n_queries=4, n_layers=2, mask_dim=mask_dim)

    B = 1
    tokens = torch.randn(B, token_dim, 2, 2, 2)                        # 1/8 token grid, side 2 → output 16³
    skips = [torch.randn(B, 2, 16, 16, 16), torch.randn(B, 4, 8, 8, 8), torch.randn(B, 6, 4, 4, 4)]
    dec_out = dec(tokens, skips)
    instance = inst(tokens, dec_out["feat_14"])
    assert dec_out["sem_logits"][0].shape == (B, 2, 16, 16, 16)
    assert dec_out["mask_features"].shape == (B, mask_dim, 16, 16, 16)

    sem_t = torch.zeros(B, 16, 16, 16, dtype=torch.long)
    sem_t[0, 4:9, 4:9, 4:9] = 1
    owner = torch.zeros(B, 16, 16, 16, dtype=torch.long)
    owner[0, 4:9, 4:9, 4:9] = 1

    sem = L.semantic_loss(dec_out["sem_logits"], sem_t, HERCUVIT)
    ins = L.instance_loss(instance, dec_out["mask_features"], owner, has_identity=[True], recipe=HERCUVIT)
    total = sem["sem"] + ins["inst"]
    assert torch.isfinite(total)
    total.backward()
    # gradients reach the decoder's semantic head and the instance head's mask embedding.
    assert dec.sem_head.weight.grad is not None
    assert inst.mask_embed.weight.grad is not None

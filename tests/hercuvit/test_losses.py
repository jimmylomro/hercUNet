"""HercuViT losses — semantic deep-sup downsampling, and the v2 DECOUPLED instance loss (no-null fg-gated ownership
CE): has_identity gating + grad-connected zero, supervision, and a tiny end-to-end through the real decoupled heads."""
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


def _toy_decoupled_instance(B=1, Q=4, C=3, L_layers=2, requires_grad=False):
    """A minimal decoupled instance dict: per-layer ``mask_embed`` (B,Q,C) + ``n_match`` (no null, no class)."""
    layers = [{"mask_embed": torch.randn(B, Q, C, requires_grad=requires_grad)} for _ in range(L_layers)]
    return {"layers": layers, "n_match": Q}


def test_decoupled_loss_zero_and_grad_connected_when_no_identity():
    torch.manual_seed(0)
    side, mask_dim = 8, 3
    inst = _toy_decoupled_instance(C=mask_dim, requires_grad=True)
    mask_features = torch.randn(1, mask_dim, side, side, side, requires_grad=True)
    owner = torch.zeros(1, side, side, side, dtype=torch.long)
    out = L.instance_loss_decoupled(inst, mask_features, owner, has_identity=[False], recipe=HERCUVIT)
    # gated off → exactly zero, but STILL grad-connected (so the optimiser step is well-defined / DDP-safe).
    assert float(out["inst"].detach()) == 0.0
    assert out["inst"].requires_grad
    out["inst"].backward()
    assert torch.count_nonzero(inst["layers"][-1]["mask_embed"].grad) == 0   # grad exists and is all-zero


def test_decoupled_loss_supervises_with_identity():
    torch.manual_seed(0)
    side, mask_dim = 8, 3
    inst = _toy_decoupled_instance(C=mask_dim, requires_grad=True)
    mask_features = torch.randn(1, mask_dim, side, side, side, requires_grad=True)
    owner = torch.zeros(1, side, side, side, dtype=torch.long)
    owner[0, 1:4, 1:4, 1:4] = 1                   # two thin "sheets" (ids 1,2)
    owner[0, 5:7, 5:7, 5:7] = 2
    out = L.instance_loss_decoupled(inst, mask_features, owner, has_identity=[True], recipe=HERCUVIT)
    assert out["inst"].requires_grad and torch.isfinite(out["inst"])
    assert float(out["inst"].detach()) > 0.0
    out["inst"].backward()
    assert mask_features.grad is not None and torch.isfinite(mask_features.grad).all()


def test_end_to_end_decoupled_forward_backward():
    """Tiny end-to-end through the REAL decoupled heads (pixel encoder + mask decoder) + semantic & instance losses."""
    torch.manual_seed(0)
    from hercunet.hercuvit.model.decoupled_pixel_encoder import DecoupledPixelEncoder
    from hercunet.hercuvit.model.decoupled_mask_decoder import DecoupledMaskDecoder

    token_dim, mask_dim = 8, 3
    dec = DecoupledPixelEncoder(token_dim=token_dim, skip_channels=(2, 4, 6),
                                n_sem_classes=2, mask_dim=mask_dim, deep_supervision=True)
    head = DecoupledMaskDecoder(dim=mask_dim, mask_dim=mask_dim, n_heads=1, n_queries=4, n_layers=3)

    B = 1
    tokens = torch.randn(B, token_dim, 4, 4, 4)                        # 1/8 token grid, side 4 → output 32³
    skips = [torch.randn(B, 2, 32, 32, 32), torch.randn(B, 4, 16, 16, 16), torch.randn(B, 6, 8, 8, 8)]
    d = dec(tokens, skips)
    assert d["sem_logits"][0].shape == (B, 2, 32, 32, 32)
    assert d["mask_features"].shape == (B, mask_dim, 32, 32, 32)
    assert len(d["memory"]) == 3                                       # [1/32,1/16,1/8]

    instance = head(d["memory"], d["mask_features"], dn_owner=None)

    sem_t = torch.zeros(B, 32, 32, 32, dtype=torch.long)
    sem_t[0, 8:18, 8:18, 8:18] = 1
    owner = torch.zeros(B, 32, 32, 32, dtype=torch.long)
    owner[0, 10, 10:14, 10:14] = 1                                     # a thin sheet
    owner[0, 20, 20:24, 20:24] = 2

    sem = L.semantic_loss(d["sem_logits"], sem_t, HERCUVIT)
    ins = L.instance_loss_decoupled(instance, d["mask_features"], owner, has_identity=[True], recipe=HERCUVIT)
    total = sem["sem"] + ins["inst"]
    assert torch.isfinite(total)
    total.backward()
    # gradients reach the decoder's surface head and the mask decoder's mask-embed MLP.
    assert dec.surf_head.weight.grad is not None
    assert head.mask_embed[-1].weight.grad is not None

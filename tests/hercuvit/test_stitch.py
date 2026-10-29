"""Stitcher train side — the shared embedding reduction, the decoder forward (shapes, padding-invariance, deep
supervision), the collate padding + masks, and the ω-weighted deep-supervised BCE loss. CPU-only, no network."""
import numpy as np
import torch

from hercunet.hercuvit.stitch.config import StitchConfig, config_from_dict
from hercunet.hercuvit.stitch.embed import reduce_normalised_pca, median_after_pca
from hercunet.hercuvit.stitch.model import StitchDecoder, fourier_features
from hercunet.hercuvit.stitch.dataset import stitch_collate
from hercunet.hercuvit.stitch.loss import stitch_loss


# ----------------------------------------------------------------------------- embedding reduction
def test_reduce_normalised_pca_shape_and_median_after_projection():
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(4, 4, 4, 12)).astype(np.float32)
    surf = rng.random((4, 4, 4)) > 0.3
    emb = reduce_normalised_pca(feats, surf, D=3)
    assert emb.shape == (4, 4, 4, 3) and emb.dtype == np.float32
    # median_after_pca IS the per-dim median of the fragment's already-projected cells (docs §3.4).
    med = median_after_pca(emb, surf)
    assert med.shape == (3,)
    assert np.allclose(med, np.median(emb[surf].reshape(-1, 3), 0))


def test_median_does_not_commute_with_linear_projection():
    """Why median must be taken AFTER the projection: the per-dim median is non-linear, so for a linear map A,
    median(A·x) != A·median(x) in general (median-then-project would land the query off the token manifold)."""
    A = np.array([[1.0, 1.0], [1.0, -1.0]])
    X = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]])            # 3 points in 2-D
    after = np.median(X @ A.T, 0)                                 # median AFTER the projection (correct)
    before = np.median(X, 0) @ A.T                                # project the raw median (wrong order)
    assert not np.allclose(after, before)


# ----------------------------------------------------------------------------- fourier features
def test_fourier_features_dim():
    P = torch.rand(2, 5, 3) * 2 - 1
    ff = fourier_features(P, bands=8)
    assert ff.shape == (2, 5, 6 * 8)


# ----------------------------------------------------------------------------- decoder forward
def _cfg(**kw):
    return config_from_dict({"pca_dim": 3, "d_model": 32, "heads": 4, "layers": 3, "fourier_bands": 4, **kw})


def _toy_batch(B=2, S=20, M=5, D=3):
    g = torch.Generator().manual_seed(1)
    return (torch.randn(B, S, D, generator=g), torch.rand(B, S, 3, generator=g) * 2 - 1,
            torch.randint(0, 8, (B, S), generator=g), torch.ones(B, S, dtype=torch.bool),
            torch.randn(B, M, D, generator=g), torch.randint(0, 8, (B, M), generator=g),
            torch.ones(B, M, dtype=torch.bool))


def test_decoder_forward_shapes_and_deep_supervision():
    cfg = _cfg()
    net = StitchDecoder(cfg).eval()
    Xs, Ps, ss, vs, Xq, sq, vq = _toy_batch()
    with torch.no_grad():
        out = net(Xs, Ps, ss, vs, Xq, sq, vq)
    assert len(out) == cfg.layers                                 # one {logit, emb} per layer (deep supervision)
    for lo in out:
        assert lo["logit"].shape == (2, 5, 5) and torch.isfinite(lo["logit"]).all()
        assert lo["emb"].shape == (2, 5, cfg.d_model) and torch.isfinite(lo["emb"]).all()


def test_decoder_padding_invariance():
    """Padding queries/spatial tokens (valid=False) must not change the real queries' logits."""
    cfg = _cfg()
    net = StitchDecoder(cfg).eval()
    Xs, Ps, ss, vs, Xq, sq, vq = _toy_batch(B=1, S=12, M=4)
    with torch.no_grad():
        base = net(Xs, Ps, ss, vs, Xq, sq, vq)[-1]["logit"]
        # append garbage padding, marked invalid
        Xs2 = torch.cat([Xs, torch.randn(1, 6, 3)], 1); Ps2 = torch.cat([Ps, torch.randn(1, 6, 3)], 1)
        ss2 = torch.cat([ss, torch.randint(0, 8, (1, 6))], 1)
        vs2 = torch.cat([vs, torch.zeros(1, 6, dtype=torch.bool)], 1)
        Xq2 = torch.cat([Xq, torch.randn(1, 3, 3)], 1); sq2 = torch.cat([sq, torch.randint(0, 8, (1, 3))], 1)
        vq2 = torch.cat([vq, torch.zeros(1, 3, dtype=torch.bool)], 1)
        padded = net(Xs2, Ps2, ss2, vs2, Xq2, sq2, vq2)[-1]["logit"]
    assert torch.allclose(base, padded[:, :4, :4], atol=1e-5)


# ----------------------------------------------------------------------------- collate
def _sample(S, M, D=3):
    g = torch.Generator().manual_seed(S + M)
    Y = torch.randint(-1, 2, (M, M), generator=g)
    return {"Xs": torch.randn(S, D, generator=g), "Ps": torch.rand(S, 3, generator=g) * 2 - 1,
            "slot_s": torch.randint(0, 8, (S,), generator=g), "Xq": torch.randn(M, D, generator=g),
            "slot_q": torch.randint(0, 8, (M,), generator=g), "conf_q": torch.rand(M, generator=g),
            "Y": Y, "path": f"s{S}_{M}.npz"}


def test_collate_pads_and_masks():
    batch = [_sample(10, 3), _sample(16, 5)]
    out = stitch_collate(batch)
    assert out["Xs"].shape == (2, 16, 3) and out["Xq"].shape == (2, 5, 3)
    assert out["valid_s"][0].sum() == 10 and out["valid_s"][1].sum() == 16
    assert out["valid_q"][0].sum() == 3 and out["valid_q"][1].sum() == 5
    assert (out["Y"][0, 3:, :] == -1).all()                       # padded query rows are ignore
    assert out["Y"][0, :3, :3].equal(batch[0]["Y"])               # real block preserved


# ----------------------------------------------------------------------------- loss
def test_loss_ignores_padding_diagonal_and_ignore_and_backprops():
    cfg = _cfg()
    net = StitchDecoder(cfg)
    bc = stitch_collate([_sample(10, 3), _sample(16, 5)])
    pred = net(bc["Xs"], bc["Ps"], bc["slot_s"], bc["valid_s"], bc["Xq"], bc["slot_q"], bc["valid_q"])
    loss, st = stitch_loss(pred, bc["Y"], bc["valid_q"], bc["conf_q"])
    assert torch.isfinite(loss) and loss.requires_grad
    # no valid pair involves a padded query, the diagonal, or an ignore(-1) label
    assert st["n_valid"] == st["npos"] + st["nneg"]
    assert "focal" in st and "contrast" in st                     # both terms reported
    loss.backward()
    assert net.log_s.grad is not None and torch.isfinite(net.log_s.grad)   # focal term feeds s
    assert net.head.weight.grad is not None and torch.isfinite(net.head.weight.grad).all()  # contrastive feeds the head


def test_loss_auto_pos_weight_matches_ratio():
    """pos_weight<=0 → measured neg/pos over valid labelled pairs."""
    M = 4
    Y = torch.tensor([[[-1, 1, 0, 0], [1, -1, 0, 0], [0, 0, -1, 1], [0, 0, 1, -1]]])
    vq = torch.ones(1, M, dtype=torch.bool)
    conf = torch.ones(1, M)
    pred = [{"logit": torch.zeros(1, M, M), "emb": torch.zeros(1, M, 8)}]
    _, st = stitch_loss(pred, Y, vq, conf, pos_weight=0.0)
    # off-diagonal: 4 positives, 8 negatives → ratio 2.0
    assert st["npos"] == 4 and st["nneg"] == 8 and abs(st["pos_weight"] - 2.0) < 1e-6

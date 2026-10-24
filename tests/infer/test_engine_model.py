"""Tests for the model-agnostic inference abstraction (``hercunet.infer.model`` + ``engine._tta_surface``).

These cover exactly what the engine migration introduced: the ``InferenceModel`` registry/contract, the built-in
models' plug-in config, and the generic TTA averaging (the blend/claim-queue/leader orchestration in
``engine.run_inference`` is unchanged from the validated Jacobi engine and is covered by the GPU byte-identical gate).
"""
import numpy as np
import pytest
import torch

from hercunet.infer.model import AuxSpec, InferenceModel, OutputSpec
import hercunet.infer.models  # noqa: F401 — registers nnunet-jacobi
import hercunet.hercuvit.infer.model  # noqa: F401 — registers hercuvit-surface


# --------------------------------------------------------------------------- a minimal plug-in model (no net) ----
class _FakeModel(InferenceModel):
    name = "fake-test-model"

    def __init__(self, p=8, passes=1, overlap=0.25):
        self._p, self._passes, self._overlap = p, passes, overlap

    @classmethod
    def load(cls, *, model=None, model_hf=None, ckpt="x", device="cpu", **opts):
        return cls(**opts)

    @property
    def patch_size(self):
        return self._p

    @property
    def passes(self):
        return self._passes

    @property
    def overlap(self):
        return self._overlap

    def norm_ct(self, sub):
        return sub.astype(np.float32) / 255.0

    def base_input(self, ct_norm, prev):
        return ct_norm[None].astype(np.float32)                      # 1-channel; prev ignored

    def forward(self, xb_base):
        # ELEMENTWISE (flip/transpose-equivariant) surface logits: class-1 = 2*ct, class-0 = 0.
        c1 = xb_base[:, 0] * 2.0
        c0 = torch.zeros_like(c1)
        return torch.stack([c0, c1], dim=1), None


def test_registry_get_and_unknown():
    assert InferenceModel.get("nnunet-jacobi").__name__ == "NnunetJacobiModel"
    assert InferenceModel.get("fake-test-model") is _FakeModel
    with pytest.raises(SystemExit):
        InferenceModel.get("does-not-exist")


def test_builtin_models_register_via_import():
    from hercunet.infer.model import _REGISTRY
    import hercunet.infer.models  # noqa: F401 — registers nnunet-jacobi
    import hercunet.hercuvit.infer.model  # noqa: F401 — registers hercuvit-surface
    assert {"nnunet-jacobi", "hercuvit-surface"} <= set(_REGISTRY)


def test_model_defaults():
    m = _FakeModel.load()
    assert m.patch_size == 8 and m.passes == 1 and m.overlap == 0.25
    assert m.uses_prev is False                       # passes==1 → engine never reads prev
    assert m.aux is None
    assert isinstance(m.output, OutputSpec)


def test_uses_prev_tracks_passes():
    assert _FakeModel.load(passes=3).uses_prev is True
    assert _FakeModel.load(passes=1).uses_prev is False


def test_base_input_shapes_nnunet_vs_hercuvit():
    from hercunet.infer.models.nnunet_jacobi import NnunetJacobiModel
    from hercunet.hercuvit.infer.model import HercuvitSurfaceModel
    P = 4
    ct = np.ones((P, P, P), np.float32)
    # nnU-Net: [CT, prev] (2 ch); prev=None → zeros slot
    nn = NnunetJacobiModel.__new__(NnunetJacobiModel); nn._p = P
    assert nn.base_input(ct, None).shape == (2, P, P, P)
    assert nn.base_input(ct, np.ones((P, P, P), np.float32)).shape == (2, P, P, P)
    # HercuViT: [CT] (1 ch); prev ignored
    hv = HercuvitSurfaceModel.__new__(HercuvitSurfaceModel); hv._p = P
    assert hv.base_input(ct, None).shape == (1, P, P, P)
    assert hv.base_input(ct, np.ones((P, P, P), np.float32)).shape == (1, P, P, P)


def test_tta_surface_identity():
    from hercunet.infer.engine import _tta_surface
    torch.manual_seed(0)
    m = _FakeModel.load()
    xb = torch.rand(2, 1, 8, 8, 8)
    out = _tta_surface(m, xb, [((), None)])                          # TTA off
    assert torch.allclose(out, m.forward(xb)[0])


def test_tta_surface_equivariant_model_is_invariant():
    """For an elementwise (flip/transpose-equivariant) forward, TTA (apply transform → forward → invert, average)
    must return the SAME logits as the identity forward — proving the invert-and-average is correct."""
    from hercunet.infer.engine import _tta_surface
    torch.manual_seed(1)
    m = _FakeModel.load()
    xb = torch.rand(1, 1, 8, 8, 8)
    variants = [((), None), ((2,), None), ((3,), None), ((2, 3), (2, 3))]   # flips + a transpose
    out_tta = _tta_surface(m, xb, variants)
    out_id = _tta_surface(m, xb, [((), None)])
    assert torch.allclose(out_tta, out_id, atol=1e-5)


def test_aux_spec_fields():
    a = AuxSpec(name="aff", channels=9, offsets=[(1, 0, 0)])
    assert a.channels == 9 and a.final_pass_only is True and a.name == "aff"


def test_hercuvit_infer_cli_fanout_flags():
    """The hercuvit infer CLI exposes the full fan-out flag set (single + multi) and maps to the engine jargs."""
    from hercunet.hercuvit.cli import build_parser
    from hercunet.infer.run import engine_jargs
    p = build_parser()
    a = p.parse_args(["infer", "single-instance", "PHerc1447", "--model", "m", "--out", "o",
                      "--gpus", "0,1", "--s3-prefix", "s3://b/x", "--pre-sync-source", "--resume"])
    assert a.func.__name__ == "_infer_single" and a.gpus == "0,1" and a.s3_prefix == "s3://b/x" and a.resume
    b = p.parse_args(["infer", "multi-instance", "PHerc1447", "--model-hf", "--out", "o", "--leader"])
    assert b.func.__name__ == "_infer_multi" and b.leader is True
    # engine_jargs (model-agnostic) builds cleanly from the parsed args — no model bits leak in
    j = engine_jargs(a, None, "src")
    assert "model_type" not in j and j["out_prefix"] == "o" and j["claim_chunk"] == 6 and j["tta"] == "none"

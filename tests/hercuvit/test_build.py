"""Plumbing: attention-backend helper, variant presets/arch_override, recipe round-trip, CLI parse, real build."""
import os

import pytest
import torch
from torch import nn

from hercunet.hercuvit import netbuild
from hercunet.hercuvit.train import recipe as R


def test_apply_attention_backend_sets_priority():
    # a module exposing the attribute, inside a net, must be retargeted.
    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention_priority = ["cudnn-placeholder"]

    net = nn.Sequential(Attn(), nn.Linear(2, 2), Attn())
    n = netbuild.apply_attention_backend(net, "auto")
    assert n == 2
    for m in net.modules():
        if hasattr(m, "attention_priority"):
            assert m.attention_priority is None          # auto → None (torch picks the kernel)
    netbuild.apply_attention_backend(net, "flash")
    prios = [m.attention_priority for m in net.modules() if hasattr(m, "attention_priority")]
    assert all(isinstance(p, list) and p for p in prios)


def test_recipe_default_and_yaml_roundtrip(tmp_path):
    rec = R.active()
    assert rec.name == "hercuvit" and rec.variant == "small"
    assert rec.attention_backend == "auto"
    # a partial YAML override only touches the named fields.
    f = os.path.join(str(tmp_path), "v.yml")
    with open(f, "w") as fh:
        fh.write("variant: small\nbase_lr: 2.0e-4\nattention_backend: math\n")
    r2 = R.load(f)
    assert r2.base_lr == 2.0e-4 and r2.attention_backend == "math" and r2.semantic_epochs == rec.semantic_epochs


def test_variant_source_of_truth_and_resolve():
    # arch numbers are the single source of truth in hercuvit_net.VARIANT_ARCH; the recipe only selects a name.
    from hercunet.hercuvit.model.hercuvit_net import VARIANT_ARCH
    assert set(R.VARIANTS) == set(VARIANT_ARCH) == {"small", "tiny", "nano"}
    assert VARIANT_ARCH["small"] == {}                         # small = Dinovol-matching defaults
    assert VARIANT_ARCH["tiny"]["depth"] == 8 and VARIANT_ARCH["nano"]["embed_dim"] == 192
    assert R.HERCUVIT.variant == "small" and R.HERCUVIT.arch_override == {}
    # netbuild.resolve_arch merges the recipe tweak OVER the canonical preset.
    assert netbuild.resolve_arch("small") == {}
    merged = netbuild.resolve_arch("tiny", {"depth": 9})
    assert merged["depth"] == 9 and merged["embed_dim"] == 288 and merged["global_attention"] == "spatial_reduction"


def test_distill_fields_present():
    rec = R.active()
    assert rec.distill_teacher.startswith("scrollprize/")
    assert rec.distill_student_variant in ("tiny", "nano")
    assert rec.distill_steps > 0


def test_cli_parses_all_subcommands():
    from hercunet.hercuvit.cli import build_parser
    p = build_parser()
    a = p.parse_args(["train", "chain", "--corpus-hf", "--dinovol-hf", "--out", "/tmp/o", "--variant", "tiny"])
    assert a.variant == "tiny" and a.out == "/tmp/o"
    a = p.parse_args(["train", "fit", "--corpus", "/c", "--dinovol", "/d.pt", "--out", "/o",
                      "--attention-backend", "flash"])
    assert a.attention_backend == "flash"
    a = p.parse_args(["distill", "--student", "nano", "--out", "/o", "--smoke", "--batch", "2"])
    assert a.student == "nano" and a.func.__name__ == "_distill" and a.smoke is True and a.batch == 2
    a = p.parse_args(["infer", "single-instance", "PHerc1447", "--model", "/m", "--out", "/o"])
    assert a.source == "PHerc1447" and a.command == "single-instance"


@pytest.mark.skipif(not os.path.exists("/tmp/dino_ck/teacher_305k.pt"),
                    reason="Dinovol checkpoint not cached at /tmp/dino_ck/teacher_305k.pt")
def test_real_build_forward_cpu():
    """Build the REAL small net from the cached Dinovol ckpt and run one 128³ forward (auto backend, CPU)."""
    ck = "/tmp/dino_ck/teacher_305k.pt"
    net = netbuild.build(ck, device="cpu", attention_backend="auto", arch_override=None,
                         with_instance=True)
    net.eval()
    x = torch.randn(1, 1, 128, 128, 128)
    with torch.no_grad():
        out = net(x)
    assert out["sem_logits"][0].shape == (1, 2, 128, 128, 128)
    assert out["mask_features"].shape[1] == 32
    assert out["instance"] is not None and len(out["instance"]["layers"]) == 3

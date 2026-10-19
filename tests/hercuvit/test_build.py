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
    assert rec.name == "small" and rec.variant == "small"       # default recipe is now the `small` variant yaml
    assert rec.attention_backend == "auto"
    # a partial YAML override only touches the named fields.
    f = os.path.join(str(tmp_path), "v.yml")
    with open(f, "w") as fh:
        fh.write("variant: small\nbase_lr: 2.0e-4\nattention_backend: math\n")
    r2 = R.load(f)
    assert r2.base_lr == 2.0e-4 and r2.attention_backend == "math" and r2.epochs == rec.epochs


def test_variant_recipes_define_arch():
    # Each variant is a FULL recipe YAML — the recipe IS the network definition (no hidden VARIANT_ARCH dict).
    assert set(R.VARIANTS) == {"small", "tiny", "nano"}
    nano = R.variant_recipe("nano").encoder_arch()
    assert nano["embed_dim"] == 192 and nano["depth"] == 6 and nano["num_heads"] == 4
    assert nano["global_attention"] == "spatial_reduction" and tuple(nano["global_blocks"]) == (3, 6)
    assert tuple(nano["decoder_channels"]) == (128, 128, 48, 16)        # nano slims the 1/8 decoder width
    tiny = R.variant_recipe("tiny").encoder_arch()
    assert tiny["depth"] == 8 and "decoder_channels" not in tiny        # tiny keeps the default decoder
    small = R.variant_recipe("small").encoder_arch()
    assert small.get("embed_dim") is None                              # small = teacher dims (come from the .pt)
    assert small["global_attention"] == "full"
    assert R.HERCUVIT.variant == "small"                               # small is the shipped default
    # arch_override still merges OVER the assembled arch
    from dataclasses import replace
    assert replace(R.variant_recipe("tiny"), arch_override={"depth": 9}).encoder_arch()["depth"] == 9


def test_schedule_fields():
    rec = R.active()
    assert rec.epochs > 0 and 0 <= rec.freeze_warmup_epochs <= rec.epochs   # LP-FT freeze-warmup schedule
    assert not hasattr(rec, "distill_teacher") and not hasattr(rec, "semantic_epochs")   # dead fields removed


def test_cli_parses_all_subcommands():
    from hercunet.hercuvit.cli import build_parser
    p = build_parser()
    a = p.parse_args(["train", "chain", "--corpus-hf", "--encoder-hf", "--out", "/tmp/o", "--variant", "tiny"])
    assert a.variant == "tiny" and a.out == "/tmp/o"
    a = p.parse_args(["train", "fit", "--corpus", "/c", "--encoder", "/d.pt", "--out", "/o",
                      "--attention-backend", "flash"])
    assert a.attention_backend == "flash" and a.encoder == "/d.pt"
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

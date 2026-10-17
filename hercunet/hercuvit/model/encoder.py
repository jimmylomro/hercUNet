"""HercuViT encoder — the Dinovol EVA backbone, built from its own checkpoint config and weights.

This is the *initialisation* for HercuViT. The build is deliberately delegated to the vendored Dinovol builder
(``SSLMetaArch._build_backbone``) so that RoPE implementation (mixed), register tokens (4 + cls), SwiGLU /
LayerScale, qkv-fusion and the CNN stem exactly match the released checkpoint — see ``docs/hercuvit.md`` §2.

The backbone is loaded from ``scrollprize/dinovol_ps8_9.6um_101626`` (``teacher_305k.pt``). With
``return_embed_skips = True`` the backbone's ``forward_features`` returns, alongside ``x_norm_patchtokens``
(the 16³ × 288 token grid), ``embed_skips`` — the CNN stem's multi-resolution stage outputs (s0 1/1, s1 1/2,
s2 1/4, s3 1/8) that the pixel decoder (pillar 2) consumes.

The windowed-attention retrofit (pillar 1) is applied *on top* of this backbone in ``model/windowed.py`` by
wrapping ``backbone.blocks``; it does not live here.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from ..dinovol.model import DinoVitStudentTeacher

#: Hugging Face model repo + checkpoint file for the Dinovol encoder (pulled like hercunet pulls its models).
DINOVOL_HF_REPO = "scrollprize/dinovol_ps8_9.6um_101626"
DINOVOL_CKPT_FILE = "teacher_305k.pt"

_BACKBONE_PREFIX = "backbone."


def load_dinovol_checkpoint(ckpt_path, *, map_location="cpu") -> tuple[dict, dict, dict]:
    """Return ``(teacher_state_dict, model_config, full_config)`` from a Dinovol checkpoint.

    The checkpoint stores the EMA ``teacher`` weights plus the resolved training ``config``; the backbone build
    reads ``config['model']``.
    """
    ck = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    teacher = ck["teacher"]
    full_cfg = ck["config"]
    model_cfg = full_cfg["model"]
    return teacher, model_cfg, full_cfg


def _strip_backbone_prefix(state_dict: Mapping[str, Any]) -> dict:
    """Keep only ``backbone.*`` tensors, stripping the prefix (drops DINO / iBOT / dense SSL heads)."""
    return {
        k[len(_BACKBONE_PREFIX):]: v
        for k, v in state_dict.items()
        if k.startswith(_BACKBONE_PREFIX)
    }


def build_dinovol_backbone(model_cfg: Mapping[str, Any]) -> nn.Module:
    """Instantiate the Dinovol EVA backbone exactly as the checkpoint config specifies (no weights loaded)."""
    return DinoVitStudentTeacher._build_backbone(model_cfg)


def build_and_load_backbone(
    ckpt_path,
    *,
    device: str | torch.device = "cpu",
    return_embed_skips: bool = True,
) -> tuple[nn.Module, dict, dict, list[str], list[str]]:
    """Build the backbone from the checkpoint config and load its ``backbone.*`` weights.

    Returns ``(backbone, model_cfg, full_cfg, missing, unexpected)``. ``missing``/``unexpected`` (with
    ``num_batches_tracked`` filtered out) should both be empty for a faithful load — the caller/tests assert so.
    """
    teacher, model_cfg, full_cfg = load_dinovol_checkpoint(ckpt_path, map_location=device)
    net = build_dinovol_backbone(model_cfg)
    backbone_sd = _strip_backbone_prefix(teacher)
    missing, unexpected = net.load_state_dict(backbone_sd, strict=False)
    missing = [m for m in missing if "num_batches_tracked" not in m]
    unexpected = [u for u in unexpected if "num_batches_tracked" not in u]
    net.return_embed_skips = bool(return_embed_skips)
    net.to(device).eval()
    return net, model_cfg, full_cfg, missing, unexpected


def load_matching_backbone_weights(net: nn.Module, backbone_state_dict: Mapping[str, Any]) -> list[str]:
    """Copy only the shape-matching parameters from ``backbone_state_dict`` (a stem/attention/etc. subset) into
    ``net``; leave the rest at their init. Returns the list of parameters left fresh (to be distilled).

    Used for the slim variants (hercuvit-tiny / -nano): the CNN stem always matches Dinovol and transfers, the
    attention transfers when ``embed_dim`` is unchanged, and the mismatched blocks (narrower / fewer / smaller
    FFN) stay randomly initialised for the distillation stage to fill.
    """
    target = net.state_dict()
    copy = {k: backbone_state_dict[k] for k, v in target.items()
            if k in backbone_state_dict and tuple(backbone_state_dict[k].shape) == tuple(v.shape)}
    net.load_state_dict({**target, **copy}, strict=True)
    return [k for k in target if k not in copy and "num_batches_tracked" not in k]


def encode(backbone: nn.Module, x: torch.Tensor) -> dict:
    """Run the (global-view) backbone and return its feature dict.

    ``x``: ``(B, 1, 128, 128, 128)`` single-channel, Dinovol-normalised CT. The returned dict contains
    ``x_norm_patchtokens`` ``(B, 16³, 288)`` and, when ``return_embed_skips`` is set, ``embed_skips`` — the list
    of CNN stem stage features [s0 1/1, s1 1/2, s2 1/4, s3 1/8].
    """
    return backbone.forward_features(x, view_kind="global")

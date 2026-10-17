"""Net-build plumbing shared by training and inference.

Two things, both deliberately OUTSIDE ``model/`` (the pillars are read-only):

1. :func:`apply_attention_backend` — retarget the SDPA kernel on every attention module after the net is built.
   The Dinovol checkpoint config pins ``attention_backend: cudnn`` (each attention sets its ``attention_priority``
   to the cuDNN-first list), but cuDNN has **no execution plan** for the windowed-attention shapes on some GPUs
   (verified failing on an RTX 4060: *"cuDNN Frontend error: No execution plans support the graph"*). Setting
   ``attention_priority = None`` lets torch auto-pick flash/efficient/math and fixes it (this is what lets
   training fit 8 GB — infer ~0.9 GB, train ~5.0 GB bf16 batch 1). We do it here, in plumbing, not by editing the
   pillar modules. Recorded RESOLVED-via-flag in ``docs/hercuvit.md`` §12.

2. :func:`build` — ``build_hercuvit`` + the backend fix in one call (what the trainer uses). Every architecture
   variant (small / tiny / nano) goes through the **single** ``build_hercuvit(ckpt_path, arch_override=…)`` entry:
   ``arch_override`` is ``None`` for ``small`` (the Dinovol-matching defaults, weights loaded strictly) and a
   per-variant config dict for ``tiny`` / ``nano`` (``build_hercuvit`` honours it and loads the Dinovol weights
   ``strict=False``, transferring only the shape-matching ones so the slim students' extra capacity is distilled).

:func:`build_from_cfg` rebuilds the architecture from a stored Dinovol ``model_cfg`` **without** the Dinovol
weights (used by the inference loader, which then loads the trained HercuViT weights over it) — so inference does
not need the Dinovol checkpoint, only the trained HercuViT checkpoint.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from .dinovol.dinov2_eva import attention_priority as _dinovol_priority
from .model.encoder import build_dinovol_backbone
from .model.hercuvit_net import VARIANT_ARCH, HercuVitNet, build_hercuvit  # noqa: F401 (re-export)
from .model.windowed import DEFAULT_GLOBAL_BLOCKS, DEFAULT_WINDOW_SIZE, retrofit_windowed


def apply_attention_backend(net, backend: str = "auto") -> int:
    """Set ``attention_priority`` on every attention module in ``net``; return how many were set.

    ``backend`` ∈ ``{auto, flash, cudnn, efficient, math}``. ``auto`` → ``None`` (torch picks the kernel, the
    safe default). Others use Dinovol's own ``attention_priority(backend)`` helper (that backend first, the rest
    as fallbacks). In-place; touches only the ``attention_priority`` attribute, never a parameter or module shape.
    """
    backend = (backend or "auto").lower()
    priority = None if backend == "auto" else _dinovol_priority(backend)
    n = 0
    for module in net.modules():
        if hasattr(module, "attention_priority"):
            module.attention_priority = priority
            n += 1
    return n


def resolve_arch(variant: str = "small", arch_override: Optional[dict] = None) -> dict:
    """Merge the recipe's ``arch_override`` tweak OVER the canonical ``VARIANT_ARCH[variant]`` (single source of
    truth in ``hercuvit_net``). Returns the merged arch dict ({} for plain small → Dinovol-matching defaults)."""
    if variant not in VARIANT_ARCH:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANT_ARCH)}")
    return {**VARIANT_ARCH[variant], **(arch_override or {})}


def build(
    ckpt_path,
    *,
    device: str | torch.device = "cpu",
    attention_backend: str = "auto",
    variant: str = "small",
    arch_override: Optional[dict] = None,
    global_blocks: Sequence[int] = DEFAULT_GLOBAL_BLOCKS,
    window_size: Sequence[int] = DEFAULT_WINDOW_SIZE,
    **kwargs,
) -> HercuVitNet:
    """``build_hercuvit(..., arch_override=…)`` + :func:`apply_attention_backend` — the single build entry for every
    variant. ``arch`` = ``VARIANT_ARCH[variant]`` + the recipe tweak; ``{}`` → small (Dinovol-matching). The
    returned net also carries ``_hercuvit_arch`` (the resolved arch) so the trainer can store it for inference."""
    arch = resolve_arch(variant, arch_override)
    net = build_hercuvit(ckpt_path, arch_override=(arch or None), device=device,
                         global_blocks=global_blocks, window_size=window_size, **kwargs)
    apply_attention_backend(net, attention_backend)
    net._hercuvit_arch = arch
    return net


_EVA_ARCH_KEYS = ("embed_dim", "depth", "num_heads", "mlp_ratio")


def build_from_cfg(
    model_cfg: dict,
    *,
    device: str | torch.device = "cpu",
    attention_backend: str = "auto",
    arch_override: Optional[dict] = None,
    global_blocks: Sequence[int] = DEFAULT_GLOBAL_BLOCKS,
    window_size: Sequence[int] = DEFAULT_WINDOW_SIZE,
    n_sem_classes: int = 2,
    mask_dim: int = 32,
    n_queries: int = 64,
    query_dim: int = 64,
    instance_layers: int = 3,
    deep_supervision: bool = True,
    with_instance: bool = True,
) -> HercuVitNet:
    """Rebuild the HercuViT architecture from a stored Dinovol ``model_cfg`` (NO Dinovol weights loaded).

    The inference loader uses this then loads the trained HercuViT ``state_dict`` over it, so inference needs only
    the trained checkpoint. Uses the same public pillar pieces as ``build_hercuvit`` — no ``model/`` edit. The
    stored ``arch_override`` (from the checkpoint) patches the Eva config keys + retrofit for the slim variants so
    the rebuilt architecture matches the trained one. NOTE: ``global_attention="spatial_reduction"`` (tiny/nano)
    is NOT reconstructed by this self-contained path — rebuild those via the Dinovol ckpt + ``build`` (§13).
    """
    arch = dict(arch_override or {})
    cfg = dict(model_cfg)
    for key in _EVA_ARCH_KEYS:
        if key in arch:
            cfg[key] = arch[key]
    gblocks = arch.get("global_blocks", global_blocks)
    win = arch.get("window_size", window_size)
    gattn = arch.get("global_attention", "full")      # tiny/nano train with spatial_reduction — reconstruct it
    srf = arch.get("sr_factor", (2, 2, 2))            # so inference runs the SAME attention pattern as training
    backbone = build_dinovol_backbone(cfg)
    backbone.return_embed_skips = True
    retrofit_windowed(backbone, global_blocks=gblocks, window_size=win, global_attention=gattn, sr_factor=srf)
    model_cfg = cfg
    net = HercuVitNet(
        backbone,
        token_dim=int(model_cfg["embed_dim"]),
        n_sem_classes=n_sem_classes,
        mask_dim=mask_dim,
        n_queries=n_queries,
        query_dim=query_dim,
        instance_layers=instance_layers,
        deep_supervision=deep_supervision,
        with_instance=with_instance,
    ).to(device)
    apply_attention_backend(net, attention_backend)
    return net

"""HercuViT network — assembles the three pillars on the Dinovol encoder.

``encoder (windowed-retrofit Dinovol) → pixel decoder (skips) → {semantic head, Mask2Former instance head}``.

Build it with :func:`build_hercuvit` (pulls / loads the Dinovol checkpoint, applies the windowed retrofit, and
attaches the decoder + heads). The fibre / ink / region heads (``heads_future``) are attached but inactive in
v1 (``docs/hercuvit.md`` §11).
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .encoder import (  # noqa: F401 (re-export)
    build_and_load_backbone,
    build_dinovol_backbone,
    load_dinovol_checkpoint,
    load_matching_backbone_weights,
    _strip_backbone_prefix,
    DINOVOL_HF_REPO,
    DINOVOL_CKPT_FILE,
)
from .windowed import retrofit_windowed, DEFAULT_GLOBAL_BLOCKS, DEFAULT_WINDOW_SIZE
from .pixel_decoder import PixelDecoder
from .instance_head import InstanceHead


class HercuVitNet(nn.Module):
    """Encoder + pixel decoder + semantic / instance heads.

    forward(x) with ``x`` = ``(B, 1, 128, 128, 128)`` Dinovol-normalised CT returns a dict:
      ``sem_logits``: list of semantic logits (full-res first, then 1/2, 1/4 if deep supervision);
      ``mask_features``: ``(B, mask_dim, 128³)`` shared mask features;
      ``instance``: the instance-head dict (``objectness``, ``query_idx``, ``layers``) or ``None``.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        token_dim: int = 288,
        skip_channels: Sequence[int] = (16, 48, 128),
        decoder_channels: Sequence[int] = (192, 128, 48, 16),
        n_sem_classes: int = 2,
        mask_dim: int = 32,
        n_queries: int = 64,
        query_dim: int = 64,
        instance_layers: int = 3,
        deep_supervision: bool = True,
        with_instance: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.backbone.return_embed_skips = True
        self.pixel_decoder = PixelDecoder(
            token_dim=token_dim,
            skip_channels=skip_channels,
            decoder_channels=decoder_channels,
            n_sem_classes=n_sem_classes,
            mask_dim=mask_dim,
            deep_supervision=deep_supervision,
        )
        self.with_instance = bool(with_instance)
        if self.with_instance:
            self.instance_head = InstanceHead(
                token_dim=token_dim,
                feat_14_dim=int(decoder_channels[1]),
                dim=query_dim,
                n_queries=n_queries,
                n_layers=instance_layers,
                mask_dim=mask_dim,
            )
        else:
            self.instance_head = None

    def forward(self, x: torch.Tensor) -> dict:
        feats = self.backbone.forward_features(x, view_kind="global")
        tokens_seq = feats["x_norm_patchtokens"]                     # (B, G, token_dim)
        skips = feats["embed_skips"]                                 # [s0@128³, s1@64³, s2@32³]
        B, G, C = tokens_seq.shape
        side = round(G ** (1 / 3))
        tokens_grid = tokens_seq.transpose(1, 2).reshape(B, C, side, side, side)  # (B, C, 16,16,16)

        dec = self.pixel_decoder(tokens_grid, skips)
        out = {"sem_logits": dec["sem_logits"], "mask_features": dec["mask_features"], "instance": None}
        if self.instance_head is not None:
            out["instance"] = self.instance_head(tokens_grid, dec["feat_14"])
        return out

    # --- staged training helpers (the trainer drives the schedule; see docs/hercuvit.md §7) ---
    def set_transformer_trainable(self, flag: bool) -> None:
        """Freeze/unfreeze the transformer blocks (the stem stays trainable for voxel detail)."""
        for p in self.backbone.blocks.parameters():
            p.requires_grad = bool(flag)

    def param_groups(self, *, base_lr: float, transformer_lr_mult: float = 0.1) -> list[dict]:
        """Two LR groups: the transformer blocks at ``transformer_lr_mult`` × ``base_lr``, everything else at ``base_lr``."""
        block_ids = {id(p) for p in self.backbone.blocks.parameters()}
        blocks, rest = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (blocks if id(p) in block_ids else rest).append(p)
        groups = [{"params": rest, "lr": base_lr}]
        if blocks:
            groups.append({"params": blocks, "lr": base_lr * transformer_lr_mult})
        return groups


#: Eva-backbone config keys an ``arch_override`` may set (the slim variants change these).
_EVA_ARCH_KEYS = ("embed_dim", "depth", "num_heads", "mlp_ratio")
#: attention-retrofit keys an ``arch_override`` may set.
_RETROFIT_KEYS = ("global_blocks", "window_size", "global_attention", "sr_factor")

#: Canonical architecture presets for the three variants — the SINGLE source of truth (recipe + distill import
#: this; the recipe may still override individual fields). ``small`` = {} means "dinovol-matching defaults".
#: ``mlp_ratio`` is the hidden/embed ratio (tiny: hidden 768, nano: hidden 512).
VARIANT_ARCH: dict[str, dict] = {
    "small": {},
    "tiny": {
        "embed_dim": 288, "depth": 8, "num_heads": 6, "mlp_ratio": 768 / 288,
        "global_blocks": (4, 8), "global_attention": "spatial_reduction", "sr_factor": (2, 2, 2),
    },
    "nano": {
        "embed_dim": 192, "depth": 6, "num_heads": 4, "mlp_ratio": 512 / 192,
        "global_blocks": (3, 6), "global_attention": "spatial_reduction", "sr_factor": (2, 2, 2),
    },
}


def build_variant(ckpt_path, variant: str = "small", **kwargs) -> HercuVitNet:
    """Convenience wrapper: build a named variant (``small`` / ``tiny`` / ``nano``) from :data:`VARIANT_ARCH`."""
    if variant not in VARIANT_ARCH:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANT_ARCH)}")
    arch = VARIANT_ARCH[variant] or None
    return build_hercuvit(ckpt_path, arch_override=arch, **kwargs)


def build_hercuvit(
    ckpt_path,
    *,
    arch_override: dict | None = None,
    device: str | torch.device = "cpu",
    global_blocks: Sequence[int] = DEFAULT_GLOBAL_BLOCKS,
    window_size: Sequence[int] = DEFAULT_WINDOW_SIZE,
    global_attention: str = "full",
    sr_factor: Sequence[int] = (2, 2, 2),
    n_sem_classes: int = 2,
    mask_dim: int = 32,
    n_queries: int = 64,
    query_dim: int = 64,
    instance_layers: int = 3,
    deep_supervision: bool = True,
    with_instance: bool = True,
    strict_load: bool = True,
) -> HercuVitNet:
    """Build **any** HercuViT variant — one config-driven path.

    ``arch_override`` (from a recipe preset) may set the Eva keys ``embed_dim / depth / num_heads / mlp_ratio``
    and the retrofit keys ``global_blocks / window_size / global_attention / sr_factor``. With no override it
    builds **hercuvit-small** (the dinovol-matching config, weights loaded strictly). When the Eva config differs
    from the checkpoint (hercuvit-tiny / -nano) only the shape-matching Dinovol weights are transferred (stem
    always; attention when ``embed_dim`` is unchanged) and the rest stay fresh for distillation
    (``train/distill.py``). ``global_attention="spatial_reduction"`` selects PVT-style global blocks for the slim
    variants. Returns the net on ``device``; ``net._hercuvit_fresh_params`` lists the un-transferred params.
    """
    arch = dict(arch_override or {})
    teacher, ckpt_cfg, _ = load_dinovol_checkpoint(ckpt_path, map_location=device)
    cfg = dict(ckpt_cfg)
    for key in _EVA_ARCH_KEYS:
        if key in arch:
            cfg[key] = arch[key]
    gblocks = arch.get("global_blocks", global_blocks)
    win = arch.get("window_size", window_size)
    gattn = arch.get("global_attention", global_attention)
    srf = arch.get("sr_factor", sr_factor)

    is_small = all(cfg.get(k) == ckpt_cfg.get(k) for k in _EVA_ARCH_KEYS)
    backbone = build_dinovol_backbone(cfg)
    bb_sd = _strip_backbone_prefix(teacher)
    fresh: list[str] = []
    if is_small:
        missing, unexpected = backbone.load_state_dict(bb_sd, strict=False)
        missing = [m for m in missing if "num_batches_tracked" not in m]
        unexpected = [u for u in unexpected if "num_batches_tracked" not in u]
        if strict_load and (missing or unexpected):
            raise RuntimeError(f"Dinovol backbone load not clean: missing={missing[:5]} unexpected={unexpected[:5]}")
    else:
        fresh = load_matching_backbone_weights(backbone, bb_sd)  # slim variant: transfer stem/attn, distil the rest
    backbone.return_embed_skips = True
    retrofit_windowed(backbone, global_blocks=gblocks, window_size=win, global_attention=gattn, sr_factor=srf)

    net = HercuVitNet(
        backbone,
        token_dim=int(cfg["embed_dim"]),
        n_sem_classes=n_sem_classes,
        mask_dim=mask_dim,
        n_queries=n_queries,
        query_dim=query_dim,
        instance_layers=instance_layers,
        deep_supervision=deep_supervision,
        with_instance=with_instance,
    )
    net = net.to(device)
    net._hercuvit_fresh_params = fresh      # params left for distillation (empty for small)
    net._hercuvit_variant_cfg = {"eva": {k: cfg[k] for k in _EVA_ARCH_KEYS}, **backbone._hercuvit_windowed}
    return net

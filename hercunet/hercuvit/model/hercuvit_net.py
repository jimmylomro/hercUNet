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
from .patch_decoder import PatchDecoder
from .instance_head import InstanceHead

#: Canonical pixel-decoder widths ``(1/8, 1/4, 1/2, 1/1)``. Only ``d8`` (the 1/8 width) is free — ``(d4,d2,d1)``
#: are LOCKED to the stem skip channels ``(128,48,16)`` by the additive skips. A slim variant sets a smaller d8 in
#: its recipe (``decoder_channels``, e.g. nano uses d8=128); it rides the ``arch_override`` dict through
#: build → checkpoint → inference.
DEFAULT_DECODER_CHANNELS = (192, 128, 48, 16)


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
        decoder_channels: Sequence[int] = DEFAULT_DECODER_CHANNELS,
        n_sem_classes: int = 2,
        mask_dim: int = 32,
        n_queries: int = 64,
        query_dim: int = 64,
        instance_layers: int = 3,
        deep_supervision: bool = True,
        with_instance: bool = True,
        add_null_query: bool = False,
        learned_queries: bool = False,
        fps_queries: bool = False,
        cross_attn: str = "deformable",
        decoder: str = "unet",
        neck: str = "none",
        neck_layers: int = 6,
        neck_heads: int = 4,
        neck_points: int = 4,
        surface_gate: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.backbone.return_embed_skips = True
        self.decoder_kind = str(decoder)
        if self.decoder_kind == "patchdecode":                              # design B: deformable neck + PatchDecode tail
            self.pixel_decoder = PatchDecoder(
                token_dim=token_dim,
                s2_dim=int(skip_channels[2]),
                mask_dim=mask_dim,
                n_sem_classes=n_sem_classes,
                neck=neck, neck_layers=neck_layers, neck_heads=neck_heads, neck_points=neck_points,
                surface_gate=surface_gate,
                deep_supervision=deep_supervision,
            )
        elif self.decoder_kind == "unet":
            self.pixel_decoder = PixelDecoder(
                token_dim=token_dim,
                skip_channels=skip_channels,
                decoder_channels=decoder_channels,
                n_sem_classes=n_sem_classes,
                mask_dim=mask_dim,
                deep_supervision=deep_supervision,
            )
        else:
            raise ValueError(f"decoder must be 'unet' or 'patchdecode', got {decoder!r}.")
        # design B gates the mask features by σ(surface) and feeds those gated features to the instance head's K/V +
        # FPS seeds (so the queries reason on surface-aware features, not raw frozen tokens).
        self._gated_kv = (self.decoder_kind == "patchdecode" and bool(surface_gate))
        self.with_instance = bool(with_instance)
        if self.with_instance:
            self.instance_head = InstanceHead(
                token_dim=token_dim,
                feat_14_dim=int(decoder_channels[1]),
                dim=query_dim,
                n_queries=n_queries,
                n_layers=instance_layers,
                mask_dim=mask_dim,
                add_null=add_null_query,
                learned_queries=learned_queries,
                fps_queries=fps_queries,
                cross_attn=cross_attn,
                gated_kv=self._gated_kv,
            )
        else:
            self.instance_head = None

    def forward(self, x: torch.Tensor, dn_owner: torch.Tensor | None = None, run_instance: bool = True) -> dict:
        """``run_instance=False`` skips the instance head entirely (surface-only training/validation) — the net then
        computes just the pixel decoder + surface head."""
        feats = self.backbone.forward_features(x, view_kind="global")
        tokens_seq = feats["x_norm_patchtokens"]                     # (B, G, token_dim)
        skips = feats["embed_skips"]                                 # [s0@128³, s1@64³, s2@32³]
        B, G, C = tokens_seq.shape
        side = round(G ** (1 / 3))
        tokens_grid = tokens_seq.transpose(1, 2).reshape(B, C, side, side, side)  # (B, C, 16,16,16)

        if self.decoder_kind == "patchdecode":
            dec = self.pixel_decoder(tokens_grid, skips[2])          # the neck consumes the 1/4 stem skip (s2)
            mf_inst = dec.get("gated_features", dec["mask_features"])  # instance head reasons on surface-gated features
            feat_14 = None                                           # design B: no 1/4 decoder feat (gated-K/V instead)
        else:
            dec = self.pixel_decoder(tokens_grid, skips)
            mf_inst = dec["mask_features"]
            feat_14 = dec["feat_14"]
        out = {"sem_logits": dec["sem_logits"], "mask_features": mf_inst, "instance": None}
        if self.instance_head is not None and run_instance:
            out["instance"] = self.instance_head(tokens_grid, feat_14, mask_features=mf_inst, dn_owner=dn_owner)
        return out

    # --- staged training helpers (the trainer drives the schedule; see docs/hercuvit.md §7) ---
    def set_encoder_trainable(self, flag: bool) -> None:
        """Freeze/unfreeze the WHOLE encoder — the stem AND the transformer blocks (and all backbone params).
        When frozen, only the fresh pixel decoder + heads train."""
        for p in self.backbone.parameters():
            p.requires_grad = bool(flag)

    def set_decoder_trunk_trainable(self, flag: bool) -> None:
        """Freeze/unfreeze the WARM-STARTED decoder trunk — the deformable neck + PatchDecode upsample tail + the
        **surface head** — while keeping the (fresh, untrained) **mask-feature head** trainable. For an instance-head
        warmup after a surface-only warm-start: the instance head then learns against a FIXED good surface gate
        (``mask_feat · σ(surface)``), and the big early instance gradients cannot drag the trunk or surface head off
        their trained optimum, while the mask features (never trained in the surface-only run) still adapt. The
        surface head and mask head are sibling 1×1 convs off the same tail features, so freezing one leaves the other
        free. patchdecode decoder only (no-op for the UNet decoder)."""
        pd = getattr(self, "pixel_decoder", None)
        if not isinstance(pd, PatchDecoder):
            return
        keep = {id(p) for p in pd.mask_head.parameters()}        # mask head stays trainable (it is untrained)
        for p in pd.parameters():
            if id(p) not in keep:
                p.requires_grad = bool(flag)
        for p in pd.mask_head.parameters():
            p.requires_grad = True

    def param_groups(self, *, base_lr: float, encoder_lr_mult: float = 0.1) -> list[dict]:
        """Two LR groups: the whole pretrained ENCODER (backbone: stem + transformer) at ``encoder_lr_mult`` ×
        ``base_lr``, and the fresh decoder + heads at ``base_lr``."""
        enc_ids = {id(p) for p in self.backbone.parameters()}
        enc, rest = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (enc if id(p) in enc_ids else rest).append(p)
        groups = [{"params": rest, "lr": base_lr}]
        if enc:
            groups.append({"params": enc, "lr": base_lr * encoder_lr_mult})
        return groups


#: Eva-backbone config keys the encoder arch (from the recipe) may set (the slim variants change these).
_EVA_ARCH_KEYS = ("embed_dim", "depth", "num_heads", "mlp_ratio")
#: attention-retrofit keys the encoder arch may set.
_RETROFIT_KEYS = ("global_blocks", "window_size", "global_attention", "sr_factor")

# NOTE: there is deliberately NO hidden VARIANT_ARCH preset dict any more. Each variant's full architecture lives
# in its shipped recipe YAML (``recipes/{small,tiny,nano}.yml``) — the recipe IS the network definition. The build
# takes the assembled ``arch_override`` dict (``HercuvitRecipe.encoder_arch()``); when loading an ``--encoder`` .pt
# the encoder comes from the file and that dict is VALIDATED against it (``_validate_encoder_arch``).


def _validate_encoder_arch(arch: dict, cfg: dict, windowed: dict, *, ckpt_path) -> None:
    """When loading a distilled encoder, the ``.pt`` IS the encoder definition — so any encoder-arch field the
    recipe states must MATCH it, or we raise (a nano recipe loaded onto a small .pt, say). Decoder/head fields are
    not checked here (they are built fresh from the recipe)."""
    mism = []
    for k in _EVA_ARCH_KEYS:
        if k in arch and cfg.get(k) is not None and _ne(arch[k], cfg[k]):
            mism.append(f"{k}: recipe {arch[k]} != encoder .pt {cfg[k]}")
    for k in ("global_blocks", "window_size", "global_attention", "sr_factor"):
        if k in arch and k in windowed and _ne(arch[k], windowed[k]):
            mism.append(f"{k}: recipe {arch[k]} != encoder .pt {windowed[k]}")
    if mism:
        raise SystemExit(f"hercuvit: recipe encoder arch disagrees with --encoder {ckpt_path} — the .pt is "
                         f"authoritative for a loaded encoder. Fix the recipe (or use the matching variant):\n  "
                         + "\n  ".join(mism))


def _ne(a, b) -> bool:
    """Tolerant inequality: compares sequences elementwise and floats with a small tolerance (mlp_ratio)."""
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return tuple(a) != tuple(b)
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) > 1e-6
    return a != b


def encoder_variant(ckpt_path) -> str:
    """Deduce the variant from an encoder ``.pt``: a distilled encoder reports its baked ``variant``
    (small/tiny/nano); a raw Dinovol teacher is ``full`` (all-global). Lets the CLI pick the variant from the file
    instead of a flag."""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "student_backbone" in blob:
        return blob.get("variant", "small")
    return "full"


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
    add_null_query: bool = False,
    learned_queries: bool = False,
    fps_queries: bool = False,
    cross_attn: str = "deformable",
    decoder: str = "unet",
    neck: str = "none",
    neck_layers: int = 6,
    neck_heads: int = 4,
    neck_points: int = 4,
    surface_gate: bool = True,
    strict_load: bool = True,
) -> HercuVitNet:
    """Build **any** HercuViT variant from an encoder ``.pt`` — one config-driven path that DETECTS the kind:

    * a **distilled encoder** (a ``hercuvit distill`` output, has ``student_backbone``) is self-contained — the
      exact windowed variant (small/tiny/nano) is rebuilt from its saved ``model_cfg`` + ``variant_cfg`` and the
      trained windowed weights loaded. ``arch_override`` is ignored (the file is authoritative).
    * a **raw Dinovol teacher** is built per ``arch_override``: ``{}`` → small (windowed retrofit, strict load);
      ``{"global_blocks": "all"}`` (the ``full`` variant) → ALL-global, no windowing; tiny/nano transfer the
      shape-matching weights and leave the rest fresh for distillation.

    Returns the net on ``device``; ``net._hercuvit_fresh_params`` lists un-transferred params (empty for a
    distilled encoder / small / full).
    """
    arch = dict(arch_override or {})
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(blob, dict) and "student_backbone" in blob:
        # DISTILLED encoder — self-contained: rebuild the exact variant from its saved cfg + load the trained weights.
        cfg = dict(blob["model_cfg"])
        vcfg = blob.get("variant_cfg", {})
        backbone = build_dinovol_backbone(cfg)
        backbone.return_embed_skips = True
        _validate_encoder_arch(arch, cfg, vcfg, ckpt_path=ckpt_path)   # loaded encoder: .pt wins; recipe must agree
        retrofit_windowed(backbone, global_blocks=vcfg.get("global_blocks", global_blocks),
                          window_size=vcfg.get("window_size", window_size),
                          global_attention=vcfg.get("global_attention", global_attention),
                          sr_factor=vcfg.get("sr_factor", sr_factor))
        dm, du = backbone.load_state_dict(blob["student_backbone"], strict=False)
        dm = [k for k in dm if "num_batches_tracked" not in k]
        du = [k for k in du if "num_batches_tracked" not in k]
        if strict_load and (dm or du):
            raise RuntimeError(f"distilled encoder load not clean: missing={dm[:5]} unexpected={du[:5]}")
        fresh: list[str] = []
    else:
        # RAW Dinovol teacher — build per the variant's arch_override.
        teacher = blob["teacher"]
        ckpt_cfg = blob["config"]["model"]
        cfg = dict(ckpt_cfg)
        for key in _EVA_ARCH_KEYS:
            if key in arch:
                cfg[key] = arch[key]
        gblocks = arch.get("global_blocks", global_blocks)
        if gblocks == "all":                                        # the 'full' variant: window nothing = all-global
            gblocks = tuple(range(1, int(cfg["depth"]) + 1))
        win = arch.get("window_size", window_size)
        gattn = arch.get("global_attention", global_attention)
        srf = arch.get("sr_factor", sr_factor)
        is_small = all(cfg.get(k) == ckpt_cfg.get(k) for k in _EVA_ARCH_KEYS)
        backbone = build_dinovol_backbone(cfg)
        bb_sd = _strip_backbone_prefix(teacher)
        fresh = []
        if is_small:
            missing, unexpected = backbone.load_state_dict(bb_sd, strict=False)
            missing = [m for m in missing if "num_batches_tracked" not in m]
            unexpected = [u for u in unexpected if "num_batches_tracked" not in u]
            if strict_load and (missing or unexpected):
                raise RuntimeError(f"Dinovol backbone load not clean: missing={missing[:5]} unexpected={unexpected[:5]}")
        else:
            fresh = load_matching_backbone_weights(backbone, bb_sd)  # slim: transfer stem/attn, distil the rest
        backbone.return_embed_skips = True
        retrofit_windowed(backbone, global_blocks=gblocks, window_size=win, global_attention=gattn, sr_factor=srf)

    net = HercuVitNet(
        backbone,
        token_dim=int(cfg["embed_dim"]),
        decoder_channels=tuple(arch.get("decoder_channels", DEFAULT_DECODER_CHANNELS)),
        n_sem_classes=n_sem_classes,
        mask_dim=mask_dim,
        n_queries=n_queries,
        query_dim=query_dim,
        instance_layers=instance_layers,
        deep_supervision=deep_supervision,
        with_instance=with_instance,
        add_null_query=add_null_query,
        learned_queries=learned_queries,
        fps_queries=fps_queries,
        cross_attn=cross_attn,
        decoder=decoder,
        neck=neck, neck_layers=neck_layers, neck_heads=neck_heads, neck_points=neck_points,
        surface_gate=surface_gate,
    )
    net = net.to(device)
    net._hercuvit_fresh_params = fresh      # params left for distillation (empty for small)
    net._hercuvit_variant_cfg = {"eva": {k: cfg[k] for k in _EVA_ARCH_KEYS}, **backbone._hercuvit_windowed}
    return net

"""Pillar 2 — the pixel decoder: a lean UNet decoder with additive skips over the Dinovol stem.

Recovers voxel-exact localisation (needed where 1–2 voxels separate two wraps) by consuming the Dinovol stem's
real multi-resolution features, which plain Dinovol computes and then discards. Design ``docs/hercuvit.md`` §6.

Layout (additive skips, one 3×3×3 conv block per level; channels mirror the stem):

| level | op | out |
|---|---|---|
| 1/8 | project tokens 288→192, conv block | 192 @ 16³ |
| 1/4 | transposed-conv 2× → 128, **+ s2 (128@32³)**, conv block | 128 @ 32³ |
| 1/2 | transposed-conv 2× → 48, **+ s1 (48@64³)**, conv block | 48 @ 64³ |
| 1/1 | transposed-conv 2× → 16, **+ s0 (16@128³)**, conv block | 16 @ 128³ |

Outputs: the full-resolution **semantic head** (sheet-surface logits; deep supervision at 1/2 and 1/4) and the
shared **32-d mask features** consumed by the instance head (pillar 3).

Note (verified against the checkpoint): the stem exposes **three** ``embed_skips`` — s0 (16@128³), s1 (48@64³),
s2 (128@32³). The 1/8 "s3" level is already folded into the 288-d tokens by the stem's ``final_proj``, so the
1/8 decoder level projects the tokens rather than fusing a separate s3 skip. This matches design §6 (s3 info is
present, just carried by the tokens) and is **not** a pillar change.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """One 3×3×3 conv + InstanceNorm + LeakyReLU (nnU-Net decoder convention)."""
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm3d(out_ch, affine=True),
        nn.LeakyReLU(inplace=True),
    )


class PixelDecoder(nn.Module):
    """Token grid + stem skips → full-resolution semantic logits + 32-d mask features.

    Args:
        token_dim: channels of the 1/8 token grid (288 for Dinovol).
        skip_channels: fine→coarse stem skip channels ``(s0, s1, s2)`` = ``(16, 48, 128)``.
        decoder_channels: per-level decoder width ``(1/8, 1/4, 1/2, 1/1)`` = ``(192, 128, 48, 16)``;
            the 1/4, 1/2, 1/1 widths must equal ``s2, s1, s0`` respectively (additive skips).
        n_sem_classes: semantic classes (2 = background / sheet-surface by default; extensible).
        mask_dim: width of the shared mask features (32, matching Dinovol's embedding width).
        deep_supervision: also emit semantic logits at 1/2 and 1/4 for deep supervision.
    """

    def __init__(
        self,
        *,
        token_dim: int = 288,
        skip_channels: Sequence[int] = (16, 48, 128),
        decoder_channels: Sequence[int] = (192, 128, 48, 16),
        n_sem_classes: int = 2,
        mask_dim: int = 32,
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        s0, s1, s2 = (int(c) for c in skip_channels)
        d8, d4, d2, d1 = (int(c) for c in decoder_channels)
        if (d4, d2, d1) != (s2, s1, s0):
            raise ValueError(
                f"additive skips require decoder widths (1/4,1/2,1/1)=({d4},{d2},{d1}) to equal "
                f"skip channels (s2,s1,s0)=({s2},{s1},{s0})"
            )
        self.deep_supervision = bool(deep_supervision)
        self.n_sem_classes = int(n_sem_classes)

        # 1/8: project tokens -> d8, then conv block
        self.proj8 = nn.Conv3d(int(token_dim), d8, kernel_size=1)
        self.block8 = _conv_block(d8, d8)
        # upsamplers (transposed conv 2×) + post-skip conv blocks
        self.up4 = nn.ConvTranspose3d(d8, d4, kernel_size=2, stride=2)
        self.block4 = _conv_block(d4, d4)
        self.up2 = nn.ConvTranspose3d(d4, d2, kernel_size=2, stride=2)
        self.block2 = _conv_block(d2, d2)
        self.up1 = nn.ConvTranspose3d(d2, d1, kernel_size=2, stride=2)
        self.block1 = _conv_block(d1, d1)

        # heads
        self.sem_head = nn.Conv3d(d1, self.n_sem_classes, kernel_size=1)
        self.mask_head = nn.Conv3d(d1, int(mask_dim), kernel_size=1)
        if self.deep_supervision:
            self.sem_head_2 = nn.Conv3d(d2, self.n_sem_classes, kernel_size=1)  # 1/2
            self.sem_head_4 = nn.Conv3d(d4, self.n_sem_classes, kernel_size=1)  # 1/4

    def forward(self, tokens_grid: torch.Tensor, skips: Sequence[torch.Tensor]) -> dict:
        """``tokens_grid``: ``(B, token_dim, 16,16,16)``. ``skips``: fine→coarse ``[s0@128³, s1@64³, s2@32³]``.

        Returns ``{"sem_logits": [full, (1/2, 1/4 if deep_supervision)], "mask_features": (B, mask_dim, 128³)}``.
        The deep-supervision logits are at their native resolution (the trainer downsamples targets to match).
        """
        s0, s1, s2 = skips

        x8 = self.block8(self.proj8(tokens_grid))       # 192 @ 16³
        x4 = self.block4(self.up4(x8) + s2)             # 128 @ 32³
        x2 = self.block2(self.up2(x4) + s1)             # 48  @ 64³
        x1 = self.block1(self.up1(x2) + s0)             # 16  @ 128³

        sem_logits = [self.sem_head(x1)]
        if self.deep_supervision:
            sem_logits.append(self.sem_head_2(x2))
            sem_logits.append(self.sem_head_4(x4))
        # ``feat_14`` (x4, 128 @ 32³) is exposed for the instance head's multi-scale deformable cross-attention.
        return {"sem_logits": sem_logits, "mask_features": self.mask_head(x1), "feat_14": x4}

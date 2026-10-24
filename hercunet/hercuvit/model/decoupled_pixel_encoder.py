"""Decoupled pixel encoder (architecture v2) — built PRECISE to the Dinovol author's surface decoder
(``third-party/villa`` ``dinovol/evals/kaggle_surfaces/model.py`` + the upstream Vesuvius ``Decoder``) and to
``docs/hercuvit-mask-architecture.md`` §10.

A CNN-UNET pixel encoder on the frozen Dinovol stem, with a DECOUPLED surface head that IS the objectness (no null
query, no no-object class downstream). Matches the author's decoder EXACTLY except for the final width:

  * **No bottleneck block.** The ViT 1/8 tokens are fed DIRECTLY into the first transpconv as the bottleneck — the
    author does ``decoder([*embed_skips, tokens])`` with nothing between the encoder and the deconvs. (The old
    design-A ``proj8``+``block8`` bottleneck is gone.)
  * Each decoder stage = ``ConvTranspose3d`` (upsample to the skip's native width) → ``torch.cat`` CONCAT skip →
    **``2× BasicBlockD`` residual blocks** (``n_conv_per_stage=2``, the author's setting; see ``resblocks.py``).
  * **U1 @1/1 IS ``mask_features``** (dim ``mask_dim``=32): the final stage's residual blocks output ``mask_dim``
    instead of the stem's 1/1 native width (16) — the ONLY deviation from the author, the required 32-d reduction.
    The 2-class **surface head** is a ``1×1`` conv straight off U1 (``[0 bg, 1 surface]`` = the objectness source),
    exactly the author's ``seg_layers[-1]`` (1×1 conv → 2 classes, ``deep_supervision=False``).
  * **Feature pyramid = bilinear-downsample of U1**, kept at ``mask_dim`` (NO up-projection) — the masked-cross-
    attention MEMORY only; the decoder runs natively at ``mask_dim`` and never attends finer than 1/8.

Stem skips (verified): s0 (16@128³, 1/1), s1 (48@64³, 1/2), s2 (128@32³, 1/4); the 1/8 level is the 288-d tokens.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .resblocks import StackedResidualBlocks


class DecoupledPixelEncoder(nn.Module):
    """Token grid + CONCAT stem skips → decoupled 1/1 surface logits + 1/1 mask features (= U1) + U1-pyramid memory.

    Args:
        token_dim: channels of the 1/8 token grid (288 for Dinovol) — fed directly as the bottleneck (no conv).
        skip_channels: fine→coarse stem skip channels ``(s0, s1, s2)`` = ``(16, 48, 128)``. The decoder's transpconv
            targets and the first two stage widths are the author's native skip widths ``(s2, s1)``; the final stage
            outputs ``mask_dim``.
        n_sem_classes: surface classes (2 = ``[bg, surface]``; the collapsed-sheet ignore is label 2 in the TARGET,
            handled by the loss, not a head class).
        mask_dim: shared mask-feature width (= U1's 1/1 width); the pyramid memory keeps this width too.
        n_conv_per_stage: residual blocks per decoder stage (2 = the author's ``blocks=2``).
        deep_supervision: also emit surface logits at 1/2 and 1/4 (OFF by default — matches the author).
    """

    def __init__(
        self,
        *,
        token_dim: int = 288,
        skip_channels: Sequence[int] = (16, 48, 128),
        n_sem_classes: int = 2,
        mask_dim: int = 32,
        n_conv_per_stage: int = 2,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        s0, s1, s2 = (int(c) for c in skip_channels)
        md = int(mask_dim)
        self.deep_supervision = bool(deep_supervision)
        self.mask_dim = md
        nb = int(n_conv_per_stage)

        # NO bottleneck conv — the ViT tokens ARE the bottleneck. Stage s: transpconv(in → skip-native) → cat skip →
        # 2× BasicBlockD. transpconv targets = native skip widths (s2,s1,s0); block outputs = (s2,s1,mask_dim).
        self.up0 = nn.ConvTranspose3d(int(token_dim), s2, kernel_size=2, stride=2, bias=True)   # 1/8→1/4
        self.block0 = StackedResidualBlocks(nb, s2 + s2, s2)                                     # cat s2 → s2
        self.up1 = nn.ConvTranspose3d(s2, s1, kernel_size=2, stride=2, bias=True)                # 1/4→1/2
        self.block1 = StackedResidualBlocks(nb, s1 + s1, s1)                                     # cat s1 → s1
        self.up2 = nn.ConvTranspose3d(s1, s0, kernel_size=2, stride=2, bias=True)                # 1/2→1/1
        self.block2 = StackedResidualBlocks(nb, s0 + s0, md)                                     # cat s0 → mask_dim (U1)

        # DECOUPLED surface head (= objectness): 1×1 conv straight off U1 (author's seg_layers[-1])
        self.surf_head = nn.Conv3d(md, int(n_sem_classes), kernel_size=1, bias=True)
        # NO pyramid projection: the memory keeps U1's mask_dim width (the decoder runs natively at mask_dim).
        if self.deep_supervision:
            self.surf_head_2 = nn.Conv3d(s1, int(n_sem_classes), kernel_size=1, bias=True)       # 1/2 (off block1)
            self.surf_head_4 = nn.Conv3d(s2, int(n_sem_classes), kernel_size=1, bias=True)       # 1/4 (off block0)

    def forward(self, tokens_grid: torch.Tensor, skips: Sequence[torch.Tensor]) -> dict:
        """``tokens_grid`` (B, token_dim, 16³) fed directly as the bottleneck; ``skips`` fine→coarse
        ``[s0@128³, s1@64³, s2@32³]``.

        Returns ``{"sem_logits": [1/1 (+1/2,1/4 if deep_sup)], "mask_features": U1@1/1 (B,mask_dim,128³),
        "memory": [1/32, 1/16, 1/8] (B,mask_dim,·)}``. The masks are read at 1/1 by a plain dot against
        ``mask_features`` in the decoder; the decoder ATTENDS only to ``memory`` (≤1/8), at ``mask_dim`` width.
        """
        s0, s1, s2 = skips
        x4 = self.block0(torch.cat([self.up0(tokens_grid), s2], dim=1))   # 128 @ 32³  (CONCAT s2)
        x2 = self.block1(torch.cat([self.up1(x4), s1], dim=1))           # 48  @ 64³  (CONCAT s1)
        u1 = self.block2(torch.cat([self.up2(x2), s0], dim=1))           # mask_dim @ 128³ = mask_features (CONCAT s0)

        sem_logits = [self.surf_head(u1)]                               # decoupled 2-class surface @1/1 (objectness)
        if self.deep_supervision:
            sem_logits.append(self.surf_head_2(x2))
            sem_logits.append(self.surf_head_4(x4))

        # feature pyramid = bilinear(trilinear) downsample of U1, kept at mask_dim (NO projection); coarse→fine
        S = u1.shape[-1]
        def _lvl(side: int) -> torch.Tensor:
            return F.interpolate(u1, size=(side, side, side), mode="trilinear", align_corners=False)
        memory = [_lvl(S // 32), _lvl(S // 16), _lvl(S // 8)]           # [1/32, 1/16, 1/8] — attention MEMORY only
        return {"sem_logits": sem_logits, "mask_features": u1, "memory": memory}

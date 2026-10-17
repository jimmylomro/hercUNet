"""Future fibre / ink / region heads — wired-ready, inactive in v1 (``docs/hercuvit.md`` §11).

Dinovol was SSL-supervised for fibres and ink too, through a dense head with these exact widths
(``dense_head``: hidden 256, out 64, ink 32, region 128). HercuViT v1 segments sheets only, so these heads are
**not** attached to :class:`hercunet.hercuvit.model.hercuvit_net.HercuVitNet` yet — they live here so fibre/ink
supervision can be added later (point / InfoNCE / DetCon on dense voxel embeddings) without re-architecting.

To activate later: instantiate :class:`FutureDenseHeads` with the pixel decoder's full-resolution feature width
and attach it in ``HercuVitNet.__init__`` behind a recipe flag; feed it the 1/1 decoder features, and add the
point / ink / region losses (the author's SSL recipe) to ``train/losses.py``.
"""
from __future__ import annotations

import torch
from torch import nn

#: Dinovol dense_head widths (from the checkpoint config) — kept identical for transfer.
POINT_DIM = 64
INK_DIM = 32
REGION_DIM = 128
HIDDEN_DIM = 256


class _MLPHead(nn.Module):
    def __init__(self, in_ch: int, hidden: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, out_ch, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FutureDenseHeads(nn.Module):
    """Dense per-voxel embedding heads for fibre identity (point), ink (prototype), and region (DetCon).

    Inactive placeholder: not attached in v1. ``forward`` returns the three per-voxel embedding maps so a future
    trainer can apply contrastive / prototype / region losses. Widths match Dinovol's ``dense_head`` for transfer.
    """

    def __init__(self, in_ch: int, *, hidden: int = HIDDEN_DIM, point_dim: int = POINT_DIM,
                 ink_dim: int = INK_DIM, region_dim: int = REGION_DIM) -> None:
        super().__init__()
        self.point = _MLPHead(in_ch, hidden, point_dim)
        self.ink = _MLPHead(in_ch, hidden, ink_dim)
        self.region = _MLPHead(in_ch, hidden, region_dim)

    def forward(self, feat_full: torch.Tensor) -> dict:
        """``feat_full`` (B, in_ch, 128³) → ``{"point","ink","region"}`` per-voxel embedding maps."""
        return {"point": self.point(feat_full), "ink": self.ink(feat_full), "region": self.region(feat_full)}

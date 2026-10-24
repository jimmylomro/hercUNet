"""ResNet-D residual blocks for the decoupled pixel encoder's deconv stages — ported VERBATIM (math-identical) from
the upstream Vesuvius ``Decoder`` the Dinovol author uses for his surface decoder
(``third-party/villa`` ``vesuvius/src/vesuvius/models/build/{resblocks,simple_conv_blocks}.py``), so our deconv
blocks match the author's exactly: each decoder stage is ``2× BasicBlockD`` (``n_conv_per_stage=2``).

This is a minimal **3D-only** subset of ``BasicBlockD`` / ``StackedResidualBlocks``: ``nn.Conv3d`` + ``InstanceNorm3d``
+ ``LeakyReLU(0.01)``, ``conv_bias=True`` (the author's ``SurfaceModel`` adapter). The upstream options we do NOT
use (BottleneckD, squeeze-excitation, stochastic depth, gated activations, dropout, strided residual avgpool for the
decoder's stride-1 stages) are omitted — none are active in the surface decoder path.
"""
from __future__ import annotations

from torch import nn


def _conv_norm_relu(in_ch: int, out_ch: int, k: int = 3, stride: int = 1, *, nonlin: bool = True,
                    bias: bool = True) -> nn.Sequential:
    """ConvDropoutNormReLU (no dropout): ``Conv3d(pad=(k-1)//2) → InstanceNorm3d → [LeakyReLU]``. ``bias`` on the conv
    matches the author: True for the feature convs, False for the 1×1 residual-skip projection."""
    ops: list[nn.Module] = [nn.Conv3d(in_ch, out_ch, k, stride, padding=(k - 1) // 2, bias=bias)]
    ops.append(nn.InstanceNorm3d(out_ch, eps=1e-5, affine=True))
    if nonlin:
        ops.append(nn.LeakyReLU(negative_slope=1e-2, inplace=True))
    return nn.Sequential(*ops)


class BasicBlockD(nn.Module):
    """ResNet-D basic block: ``out = conv2(conv1(x)) + skip(x)``, then LeakyReLU. ``conv1`` has norm+nonlin; ``conv2``
    has norm only (the nonlin is applied AFTER the residual add). ``skip`` is identity when ``in==out`` (decoder
    stages are stride-1), else a 1×1 conv+norm projection (no nonlin). Matches ``BasicBlockD.forward`` exactly."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _conv_norm_relu(in_ch, out_ch, k, stride, nonlin=True)
        self.conv2 = _conv_norm_relu(out_ch, out_ch, k, 1, nonlin=False)
        self.nonlin2 = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        if stride != 1 or in_ch != out_ch:
            ops: list[nn.Module] = []
            if stride != 1:
                ops.append(nn.AvgPool3d(stride, stride))
            if in_ch != out_ch:
                ops.append(_conv_norm_relu(in_ch, out_ch, 1, 1, nonlin=False, bias=False))  # author: skip conv has no bias
            self.skip: nn.Module = nn.Sequential(*ops)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        out = self.conv2(self.conv1(x)) + self.skip(x)
        return self.nonlin2(out)


class StackedResidualBlocks(nn.Module):
    """``n_blocks`` ``BasicBlockD`` in sequence: the first maps ``in_ch → out_ch`` (at ``initial_stride``), the rest
    ``out_ch → out_ch`` (stride 1). Matches the author's ``StackedResidualBlocks`` for ``block=BasicBlockD``."""

    def __init__(self, n_blocks: int, in_ch: int, out_ch: int, k: int = 3, initial_stride: int = 1) -> None:
        super().__init__()
        assert n_blocks > 0
        blocks = [BasicBlockD(in_ch, out_ch, k, initial_stride)]
        blocks += [BasicBlockD(out_ch, out_ch, k, 1) for _ in range(1, n_blocks)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

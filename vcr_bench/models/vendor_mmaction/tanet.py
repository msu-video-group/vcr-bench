from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from .resnet2d import Bottleneck2d, ResNet2d


class TAM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_segments: int,
        alpha: int = 2,
        adaptive_kernel_size: int = 3,
        beta: int = 4,
        conv1d_kernel_size: int = 3,
        adaptive_convolution_stride: int = 1,
        adaptive_convolution_padding: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_segments = int(num_segments)
        self.adaptive_convolution_stride = int(adaptive_convolution_stride)
        self.adaptive_convolution_padding = int(adaptive_convolution_padding)

        self.G = nn.Sequential(
            nn.Linear(num_segments, num_segments * alpha, bias=False),
            nn.BatchNorm1d(num_segments * alpha),
            nn.ReLU(inplace=True),
            nn.Linear(num_segments * alpha, adaptive_kernel_size, bias=False),
            nn.Softmax(-1),
        )
        self.L = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels // beta,
                conv1d_kernel_size,
                stride=1,
                padding=conv1d_kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(in_channels // beta),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // beta, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()
        num_batches = n // self.num_segments
        x = x.view(num_batches, self.num_segments, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        theta_out = torch.nn.functional.adaptive_avg_pool2d(x.view(-1, self.num_segments, h, w), (1, 1))
        conv_kernel = self.G(theta_out.view(-1, self.num_segments)).view(num_batches * c, 1, -1, 1)
        local_activation = self.L(theta_out.view(-1, c, self.num_segments)).view(num_batches, c, self.num_segments, 1, 1)
        new_x = x * local_activation
        y = torch.nn.functional.conv2d(
            new_x.view(1, num_batches * c, self.num_segments, h * w),
            conv_kernel,
            bias=None,
            stride=(self.adaptive_convolution_stride, 1),
            padding=(self.adaptive_convolution_padding, 0),
            groups=num_batches * c,
        )
        y = y.view(num_batches, c, self.num_segments, h, w)
        return y.permute(0, 2, 1, 3, 4).contiguous().view(n, c, h, w)


class TABlock(nn.Module):
    def __init__(self, block: nn.Module, num_segments: int, tam_cfg: dict) -> None:
        super().__init__()
        self.block = block
        self.num_segments = int(num_segments)
        self.tam = TAM(block.conv1.conv.out_channels, self.num_segments, **dict(tam_cfg))
        if not isinstance(self.block, Bottleneck2d):
            raise NotImplementedError("TANet currently supports Bottleneck2d blocks only")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.block.conv1(x)
        out = self.tam(out)
        out = self.block.conv2(out)
        out = self.block.conv3(out)
        if self.block.downsample is not None:
            identity = self.block.downsample(x)
        out = out + identity
        return self.block.relu(out)


class TANet(ResNet2d):
    def __init__(self, depth: int, num_segments: int, tam_cfg: dict | None = None, **kwargs) -> None:
        super().__init__(depth, **kwargs)
        self.num_segments = int(num_segments)
        self.tam_cfg = deepcopy(tam_cfg or {})
        self.make_tam_modeling()

    def make_tam_modeling(self) -> None:
        def make_tam_block(stage: nn.Sequential) -> nn.Sequential:
            blocks = list(stage.children())
            for i, block in enumerate(blocks):
                blocks[i] = TABlock(block, self.num_segments, deepcopy(self.tam_cfg))
            return nn.Sequential(*blocks)

        for i in range(self.num_stages):
            layer_name = f"layer{i + 1}"
            setattr(self, layer_name, make_tam_block(getattr(self, layer_name)))

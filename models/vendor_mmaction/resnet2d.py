from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .common import ConvModule, build_activation_layer


class BasicBlock2d(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
        style: str = "pytorch",
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        if style not in ("pytorch", "caffe"):
            raise ValueError(f"Unsupported style: {style}")
        self.conv1 = ConvModule(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.conv2 = ConvModule(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )
        self.downsample = downsample
        self.relu = build_activation_layer(act_cfg or {"type": "ReLU", "inplace": True})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class Bottleneck2d(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
        style: str = "pytorch",
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        if style not in ("pytorch", "caffe"):
            raise ValueError(f"Unsupported style: {style}")
        if style == "pytorch":
            conv1_stride = 1
            conv2_stride = stride
        else:
            conv1_stride = stride
            conv2_stride = 1
        self.conv1 = ConvModule(
            inplanes,
            planes,
            kernel_size=1,
            stride=conv1_stride,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.conv2 = ConvModule(
            planes,
            planes,
            kernel_size=3,
            stride=conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.conv3 = ConvModule(
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )
        self.downsample = downsample
        self.relu = build_activation_layer(act_cfg or {"type": "ReLU", "inplace": True})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class ResNet2d(nn.Module):
    arch_settings = {
        18: (BasicBlock2d, (2, 2, 2, 2)),
        34: (BasicBlock2d, (3, 4, 6, 3)),
        50: (Bottleneck2d, (3, 4, 6, 3)),
        101: (Bottleneck2d, (3, 4, 23, 3)),
    }

    def __init__(
        self,
        depth: int = 50,
        pretrained: str | None = None,
        in_channels: int = 3,
        num_stages: int = 4,
        base_channels: int = 64,
        out_indices: Sequence[int] = (3,),
        strides: Sequence[int] = (1, 2, 2, 2),
        dilations: Sequence[int] = (1, 1, 1, 1),
        style: str = "pytorch",
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
        norm_eval: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unsupported ResNet2d kwargs: {unknown}")
        if depth not in self.arch_settings:
            raise KeyError(f"invalid depth {depth} for resnet2d")
        self.depth = int(depth)
        self.pretrained = pretrained
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.num_stages = int(num_stages)
        self.out_indices = tuple(int(i) for i in out_indices)
        self.strides = tuple(int(x) for x in strides)
        self.dilations = tuple(int(x) for x in dilations)
        self.style = str(style)
        self.conv_cfg = dict(conv_cfg or {"type": "Conv2d"})
        self.norm_cfg = dict(norm_cfg or {"type": "BN2d", "requires_grad": True})
        self.act_cfg = dict(act_cfg or {"type": "ReLU", "inplace": True})
        self.norm_eval = bool(norm_eval)

        self.block, stage_blocks = self.arch_settings[self.depth]
        self.stage_blocks = stage_blocks[: self.num_stages]
        self.inplanes = self.base_channels

        self._make_stem_layer()

        self.res_layers: list[str] = []
        for i, num_blocks in enumerate(self.stage_blocks):
            planes = self.base_channels * 2**i
            res_layer = self.make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=self.strides[i],
                dilation=self.dilations[i],
                style=self.style,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )
            self.inplanes = planes * self.block.expansion
            layer_name = f"layer{i + 1}"
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * self.base_channels * 2 ** (len(self.stage_blocks) - 1)

    @staticmethod
    def make_res_layer(
        block: type[nn.Module],
        inplanes: int,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilation: int = 1,
        style: str = "pytorch",
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
    ) -> nn.Module:
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = ConvModule(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
            )
        layers = [
            block(
                inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                style=style,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
        ]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    stride=1,
                    dilation=dilation,
                    style=style,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )
        return nn.Sequential(*layers)

    def _make_stem_layer(self) -> None:
        self.conv1 = ConvModule(
            self.in_channels,
            self.base_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            x = getattr(self, layer_name)(x)
            if i in self.out_indices:
                outs.append(x)
        return outs[0] if len(outs) == 1 else tuple(outs)


class C2D(ResNet2d):
    def _make_stem_layer(self) -> None:
        self.conv1 = ConvModule(
            self.in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.maxpool3d_1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 0, 0))
        self.maxpool3d_2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batches = x.shape[0]

        def to_2d(y: torch.Tensor) -> torch.Tensor:
            y = y.permute(0, 2, 1, 3, 4)
            return y.reshape(-1, y.shape[2], y.shape[3], y.shape[4])

        def to_3d(y: torch.Tensor) -> torch.Tensor:
            y = y.reshape(batches, -1, y.shape[1], y.shape[2], y.shape[3])
            return y.permute(0, 2, 1, 3, 4)

        x = to_2d(x)
        x = self.conv1(x)
        x = to_3d(x)
        x = self.maxpool3d_1(x)
        x = to_2d(x)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            x = getattr(self, layer_name)(x)
            if i == 0:
                x = to_3d(x)
                x = self.maxpool3d_2(x)
                x = to_2d(x)
            if i in self.out_indices:
                outs.append(to_3d(x))
        return outs[0] if len(outs) == 1 else tuple(outs)

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .common import ConvModule


class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: float) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.bottleneck = self._round_width(channels, reduction)
        self.fc1 = nn.Conv3d(channels, self.bottleneck, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(self.bottleneck, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _round_width(width: int, multiplier: float, min_width: int = 8, divisor: int = 8) -> int:
        width *= multiplier
        width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
        if width_out < 0.9 * width:
            width_out += divisor
        return int(width_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return residual * x


class BlockX3D(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        outplanes: int,
        spatial_stride: int = 1,
        downsample: nn.Module | None = None,
        se_ratio: float | None = None,
        use_swish: bool = True,
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = ConvModule(
            inplanes,
            planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.conv2 = ConvModule(
            planes,
            planes,
            kernel_size=3,
            stride=(1, spatial_stride, spatial_stride),
            padding=1,
            groups=planes,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )
        self.se_module = SEModule(planes, se_ratio) if se_ratio is not None else None
        self.swish = nn.SiLU(inplace=True) if use_swish else nn.Identity()
        self.conv3 = ConvModule(
            planes,
            outplanes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se_module is not None:
            out = self.se_module(out)
        out = self.swish(out)
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class X3D(nn.Module):
    def __init__(
        self,
        gamma_w: float = 1.0,
        gamma_b: float = 2.25,
        gamma_d: float = 2.2,
        in_channels: int = 3,
        num_stages: int = 4,
        spatial_strides: tuple[int, ...] = (2, 2, 2, 2),
        se_style: str = "half",
        se_ratio: float | None = 1 / 16,
        use_swish: bool = True,
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.gamma_w = float(gamma_w)
        self.gamma_b = float(gamma_b)
        self.gamma_d = float(gamma_d)
        self.in_channels = int(in_channels)
        self.base_channels = self._round_width(24, self.gamma_w)
        self.stage_blocks = [self._round_repeats(x, self.gamma_d) for x in [1, 2, 5, 3]][:num_stages]
        self.spatial_strides = tuple(int(x) for x in spatial_strides[:num_stages])
        self.se_style = str(se_style)
        self.se_ratio = se_ratio
        self.use_swish = bool(use_swish)
        self.conv_cfg = dict(conv_cfg or {"type": "Conv3d"})
        self.norm_cfg = dict(norm_cfg or {"type": "BN3d", "requires_grad": True})
        self.act_cfg = dict(act_cfg or {"type": "ReLU", "inplace": True})

        self.layer_inplanes = self.base_channels
        self._make_stem_layer()

        self.res_layers: list[str] = []
        for i, num_blocks in enumerate(self.stage_blocks):
            inplanes = self.base_channels * 2**i
            planes = int(inplanes * self.gamma_b)
            res_layer = self.make_res_layer(
                self.layer_inplanes,
                inplanes,
                planes,
                num_blocks,
                spatial_stride=self.spatial_strides[i],
            )
            self.layer_inplanes = inplanes
            layer_name = f"layer{i + 1}"
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.base_channels * 2 ** (len(self.stage_blocks) - 1)
        self.conv5 = ConvModule(
            self.feat_dim,
            int(self.feat_dim * self.gamma_b),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.feat_dim = int(self.feat_dim * self.gamma_b)

    @staticmethod
    def _round_width(width: int, multiplier: float, min_depth: int = 8, divisor: int = 8) -> int:
        if not multiplier:
            return width
        width *= multiplier
        new_filters = max(min_depth, int(width + divisor / 2) // divisor * divisor)
        if new_filters < 0.9 * width:
            new_filters += divisor
        return int(new_filters)

    @staticmethod
    def _round_repeats(repeats: int, multiplier: float) -> int:
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))

    def make_res_layer(
        self,
        layer_inplanes: int,
        inplanes: int,
        planes: int,
        blocks: int,
        spatial_stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if spatial_stride != 1 or layer_inplanes != inplanes:
            downsample = ConvModule(
                layer_inplanes,
                inplanes,
                kernel_size=1,
                stride=(1, spatial_stride, spatial_stride),
                padding=0,
                bias=False,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=None,
            )

        use_se = [False] * blocks
        if self.se_style == "all":
            use_se = [True] * blocks
        elif self.se_style == "half":
            use_se = [i % 2 == 0 for i in range(blocks)]

        layers = [
            BlockX3D(
                layer_inplanes,
                planes,
                inplanes,
                spatial_stride=spatial_stride,
                downsample=downsample,
                se_ratio=self.se_ratio if use_se[0] else None,
                use_swish=self.use_swish,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )
        ]
        for i in range(1, blocks):
            layers.append(
                BlockX3D(
                    inplanes,
                    planes,
                    inplanes,
                    spatial_stride=1,
                    se_ratio=self.se_ratio if use_se[i] else None,
                    use_swish=self.use_swish,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
            )
        return nn.Sequential(*layers)

    def _make_stem_layer(self) -> None:
        self.conv1_s = ConvModule(
            self.in_channels,
            self.base_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None,
        )
        self.conv1_t = ConvModule(
            self.base_channels,
            self.base_channels,
            kernel_size=(5, 1, 1),
            stride=(1, 1, 1),
            padding=(2, 0, 0),
            groups=self.base_channels,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1_s(x)
        x = self.conv1_t(x)
        for layer_name in self.res_layers:
            x = getattr(self, layer_name)(x)
        return self.conv5(x)

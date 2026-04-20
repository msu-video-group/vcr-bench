from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .common import ConvModule


class DownSample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=(3, 1, 1),
        stride=(1, 1, 1),
        padding=(1, 0, 0),
        groups: int = 1,
        bias: bool = False,
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
        downsample_position: str = "after",
        downsample_scale=(1, 2, 2),
    ) -> None:
        super().__init__()
        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=bias,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.downsample_position = str(downsample_position)
        self.pool = nn.MaxPool3d(downsample_scale, downsample_scale, (0, 0, 0), ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample_position == "before":
            return self.conv(self.pool(x))
        return self.pool(self.conv(x))


class LevelFusion(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int],
        mid_channels: Sequence[int],
        out_channels: int,
        downsample_scales=((1, 1, 1), (1, 1, 1)),
    ) -> None:
        super().__init__()
        self.downsamples = nn.ModuleList(
            [
                DownSample(
                    int(in_channels[i]),
                    int(mid_channels[i]),
                    kernel_size=(1, 1, 1),
                    stride=(1, 1, 1),
                    padding=(0, 0, 0),
                    groups=32,
                    bias=False,
                    norm_cfg=dict(type="BN3d", requires_grad=True),
                    act_cfg=dict(type="ReLU", inplace=True),
                    downsample_position="before",
                    downsample_scale=downsample_scales[i],
                )
                for i in range(len(in_channels))
            ]
        )
        self.fusion_conv = ConvModule(
            int(sum(mid_channels)),
            int(out_channels),
            1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=dict(type="Conv3d"),
            norm_cfg=dict(type="BN3d", requires_grad=True),
            act_cfg=dict(type="ReLU", inplace=True),
        )

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        out = [self.downsamples[i](feature) for i, feature in enumerate(x)]
        return self.fusion_conv(torch.cat(out, dim=1))


class SpatialModulation(nn.Module):
    def __init__(self, in_channels: Sequence[int], out_channels: int) -> None:
        super().__init__()
        self.spatial_modulation = nn.ModuleList()
        for channel in in_channels:
            downsample_scale = int(out_channels) // int(channel)
            downsample_factor = 0
            while 2 ** downsample_factor < downsample_scale:
                downsample_factor += 1
            if downsample_factor < 1:
                self.spatial_modulation.append(nn.Identity())
                continue
            ops = []
            for factor in range(downsample_factor):
                in_factor = 2**factor
                out_factor = 2 ** (factor + 1)
                ops.append(
                    ConvModule(
                        int(channel) * in_factor,
                        int(channel) * out_factor,
                        (1, 3, 3),
                        stride=(1, 2, 2),
                        padding=(0, 1, 1),
                        bias=False,
                        conv_cfg=dict(type="Conv3d"),
                        norm_cfg=dict(type="BN3d", requires_grad=True),
                        act_cfg=dict(type="ReLU", inplace=True),
                    )
                )
            self.spatial_modulation.append(nn.Sequential(*ops))

    def forward(self, x: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        return [self.spatial_modulation[i](feat) for i, feat in enumerate(x)]


class TemporalModulation(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample_scale: int = 8) -> None:
        super().__init__()
        self.conv = ConvModule(
            int(in_channels),
            int(out_channels),
            (3, 1, 1),
            stride=(1, 1, 1),
            padding=(1, 0, 0),
            bias=False,
            groups=32,
            conv_cfg=dict(type="Conv3d"),
            act_cfg=None,
        )
        self.pool = nn.MaxPool3d((int(downsample_scale), 1, 1), (int(downsample_scale), 1, 1), (0, 0, 0), ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(x))


class AuxHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, loss_weight: float = 0.5) -> None:
        super().__init__()
        self.conv = ConvModule(
            int(in_channels),
            int(in_channels) * 2,
            (1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            bias=False,
            conv_cfg=dict(type="Conv3d"),
            norm_cfg=dict(type="BN3d", requires_grad=True),
            act_cfg=None,
        )
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(int(in_channels) * 2, int(out_channels))
        self.loss_weight = float(loss_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


class TPN(nn.Module):
    def __init__(
        self,
        in_channels: tuple[int, ...],
        out_channels: int,
        spatial_modulation_cfg: dict | None = None,
        temporal_modulation_cfg: dict | None = None,
        upsample_cfg: dict | None = None,
        downsample_cfg: dict | None = None,
        level_fusion_cfg: dict | None = None,
        aux_head_cfg: dict | None = None,
        flow_type: str = "cascade",
    ) -> None:
        super().__init__()
        self.in_channels = tuple(int(v) for v in in_channels)
        self.out_channels = int(out_channels)
        self.num_tpn_stages = len(self.in_channels)
        self.flow_type = str(flow_type)

        self.temporal_modulation_ops = nn.ModuleList()
        self.upsample_ops = nn.ModuleList()
        self.downsample_ops = nn.ModuleList()

        self.level_fusion_1 = LevelFusion(**dict(level_fusion_cfg or {}))
        self.spatial_modulation = SpatialModulation(**dict(spatial_modulation_cfg or {}))

        downsample_scales = tuple(dict(temporal_modulation_cfg or {}).get("downsample_scales", (8,) * self.num_tpn_stages))
        for i in range(self.num_tpn_stages):
            self.temporal_modulation_ops.append(
                TemporalModulation(self.in_channels[-1], self.out_channels, int(downsample_scales[i]))
            )
            if i < self.num_tpn_stages - 1:
                self.upsample_ops.append(nn.Upsample(**dict(upsample_cfg or {"scale_factor": (1, 1, 1), "mode": "nearest"})))
                self.downsample_ops.append(DownSample(self.out_channels, self.out_channels, **dict(downsample_cfg or {})))

        self.level_fusion_2 = LevelFusion(**dict(level_fusion_cfg or {}))
        out_dims = int(dict(level_fusion_cfg or {}).get("out_channels", 2048))
        self.pyramid_fusion = ConvModule(
            out_dims * 2,
            2048,
            1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=dict(type="Conv3d"),
            norm_cfg=dict(type="BN3d", requires_grad=True),
            act_cfg=None,
        )
        self.aux_head = AuxHead(self.in_channels[-2], **dict(aux_head_cfg or {})) if aux_head_cfg is not None else None

    def forward(self, x: tuple[torch.Tensor, ...]) -> torch.Tensor:
        spatial_modulation_outs = self.spatial_modulation(x)
        temporal_modulation_outs = [
            self.temporal_modulation_ops[i](spatial_modulation_outs[i]) for i in range(self.num_tpn_stages)
        ]

        outs = [out.clone() for out in temporal_modulation_outs]
        for i in range(self.num_tpn_stages - 1, 0, -1):
            outs[i - 1] = outs[i - 1] + self.upsample_ops[i - 1](outs[i])
        top_down_outs = self.level_fusion_1(outs)

        if self.flow_type == "parallel":
            outs = [out.clone() for out in temporal_modulation_outs]
        for i in range(self.num_tpn_stages - 1):
            outs[i + 1] = outs[i + 1] + self.downsample_ops[i](outs[i])
        bottom_up_outs = self.level_fusion_2(outs)

        return self.pyramid_fusion(torch.cat([top_down_outs, bottom_up_outs], dim=1))

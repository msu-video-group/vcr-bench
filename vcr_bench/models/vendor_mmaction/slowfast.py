from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvModule
from .resnet3d import ResNet3d


class ResNet3dPathway(nn.Module):
    def __init__(
        self,
        depth: int = 50,
        pretrained: str | None = None,
        lateral: bool = False,
        speed_ratio: int = 8,
        channel_ratio: int = 8,
        fusion_kernel: int = 5,
        lateral_infl: int = 2,
        lateral_activate: Sequence[int] = (1, 1, 1, 1),
        in_channels: int = 3,
        num_stages: int = 4,
        base_channels: int = 64,
        spatial_strides: Sequence[int] = (1, 2, 2, 2),
        temporal_strides: Sequence[int] = (1, 1, 1, 1),
        dilations: Sequence[int] = (1, 1, 1, 1),
        conv1_kernel: Sequence[int] = (3, 7, 7),
        conv1_stride_s: int = 2,
        conv1_stride_t: int = 1,
        pool1_stride_s: int = 2,
        pool1_stride_t: int = 1,
        with_pool1: bool = True,
        with_pool2: bool = True,
        style: str = "pytorch",
        inflate: Sequence[int] = (1, 1, 1, 1),
        inflate_style: str = "3x1x1",
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
        norm_eval: bool = False,
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        probe = ResNet3d(
            depth=depth,
            pretrained=pretrained,
            pretrained2d=False,
            in_channels=in_channels,
            num_stages=num_stages,
            base_channels=base_channels,
            out_indices=(num_stages - 1,),
            spatial_strides=spatial_strides,
            temporal_strides=temporal_strides,
            dilations=dilations,
            conv1_kernel=conv1_kernel,
            conv1_stride_s=conv1_stride_s,
            conv1_stride_t=conv1_stride_t,
            pool1_stride_s=pool1_stride_s,
            pool1_stride_t=pool1_stride_t,
            with_pool1=with_pool1,
            with_pool2=with_pool2,
            style=style,
            inflate=inflate,
            inflate_style=inflate_style,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            zero_init_residual=zero_init_residual,
        )

        self.depth = int(depth)
        self.pretrained = pretrained
        self.lateral = bool(lateral)
        self.speed_ratio = int(speed_ratio)
        self.channel_ratio = int(channel_ratio)
        self.fusion_kernel = int(fusion_kernel)
        self.lateral_infl = int(lateral_infl)
        self.lateral_activate = tuple(int(x) for x in lateral_activate)
        self.in_channels = int(in_channels)
        self.num_stages = int(num_stages)
        self.base_channels = int(base_channels)
        self.spatial_strides = tuple(int(x) for x in spatial_strides)
        self.temporal_strides = tuple(int(x) for x in temporal_strides)
        self.dilations = tuple(int(x) for x in dilations)
        self.conv1_kernel = tuple(conv1_kernel)
        self.conv1_stride_s = int(conv1_stride_s)
        self.conv1_stride_t = int(conv1_stride_t)
        self.pool1_stride_s = int(pool1_stride_s)
        self.pool1_stride_t = int(pool1_stride_t)
        self.with_pool1 = bool(with_pool1)
        self.with_pool2 = bool(with_pool2)
        self.style = str(style)
        self.stage_inflations = tuple(inflate)
        self.inflate_style = str(inflate_style)
        self.conv_cfg = dict(conv_cfg or {"type": "Conv3d"})
        self.norm_cfg = dict(norm_cfg or {"type": "BN3d", "requires_grad": True})
        self.act_cfg = dict(act_cfg or {"type": "ReLU", "inplace": True})
        self.norm_eval = bool(norm_eval)
        self.zero_init_residual = bool(zero_init_residual)

        self.block = probe.block
        self.stage_blocks = probe.stage_blocks
        self.lateral_inplanes = self._calculate_lateral_inplanes()

        self.conv1 = probe.conv1
        self.maxpool = probe.maxpool
        self.pool2 = probe.pool2

        if self.lateral and self.lateral_activate[0]:
            self.conv1_lateral = ConvModule(
                self.base_channels // self.channel_ratio,
                self.base_channels * self.lateral_infl // self.channel_ratio,
                kernel_size=(self.fusion_kernel, 1, 1),
                stride=(self.speed_ratio, 1, 1),
                padding=((self.fusion_kernel - 1) // 2, 0, 0),
                bias=False,
                conv_cfg=self.conv_cfg,
                norm_cfg=None,
                act_cfg=None,
            )

        self.res_layers: list[str] = []
        self.lateral_connections: list[str] = []
        current_inplanes = self.base_channels
        for i, num_blocks in enumerate(self.stage_blocks):
            planes = self.base_channels * 2**i
            stage_inplanes = current_inplanes + self.lateral_inplanes[i]
            res_layer = ResNet3d.make_res_layer(
                self.block,
                stage_inplanes,
                planes,
                num_blocks,
                spatial_stride=self.spatial_strides[i],
                temporal_stride=self.temporal_strides[i],
                dilation=self.dilations[i],
                style=self.style,
                inflate=self.stage_inflations[i],
                inflate_style=self.inflate_style,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                conv_cfg=self.conv_cfg,
            )
            current_inplanes = planes * self.block.expansion
            layer_name = f"layer{i + 1}"
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

            if self.lateral and i != self.num_stages - 1 and self.lateral_activate[i + 1]:
                lateral_name = f"layer{i + 1}_lateral"
                lateral_module = ConvModule(
                    current_inplanes // self.channel_ratio,
                    current_inplanes * self.lateral_infl // self.channel_ratio,
                    kernel_size=(self.fusion_kernel, 1, 1),
                    stride=(self.speed_ratio, 1, 1),
                    padding=((self.fusion_kernel - 1) // 2, 0, 0),
                    bias=False,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=None,
                    act_cfg=None,
                )
                self.add_module(lateral_name, lateral_module)
                self.lateral_connections.append(lateral_name)

        self.feat_dim = current_inplanes

    def _calculate_lateral_inplanes(self) -> list[int]:
        expansion = 1 if self.depth < 50 else 4
        lateral_inplanes: list[int] = []
        for i in range(self.num_stages):
            if expansion % 2 == 0:
                planes = self.base_channels * (2**i) * ((expansion // 2) ** (1 if i > 0 else 0))
            else:
                planes = self.base_channels * (2**i) // (2 ** (1 if i > 0 else 0))
            if self.lateral and self.lateral_activate[i]:
                lateral_inplane = planes * self.lateral_infl // self.channel_ratio
            else:
                lateral_inplane = 0
            lateral_inplanes.append(int(lateral_inplane))
        return lateral_inplanes


class ResNet3dSlowFast(nn.Module):
    def __init__(
        self,
        pretrained: str | None = None,
        resample_rate: int = 8,
        speed_ratio: int = 8,
        channel_ratio: int = 8,
        slow_pathway: dict | None = None,
        fast_pathway: dict | None = None,
    ) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.resample_rate = int(resample_rate)
        self.speed_ratio = int(speed_ratio)
        self.channel_ratio = int(channel_ratio)

        slow_cfg = dict(
            depth=50,
            pretrained=None,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            fusion_kernel=5,
        )
        fast_cfg = dict(
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
        )
        if slow_pathway:
            slow_cfg.update({k: v for k, v in slow_pathway.items() if k != "type"})
        if fast_pathway:
            fast_cfg.update({k: v for k, v in fast_pathway.items() if k != "type"})
        if slow_cfg.get("lateral", False):
            slow_cfg["speed_ratio"] = self.speed_ratio
            slow_cfg["channel_ratio"] = self.channel_ratio

        self.slow_path = ResNet3dPathway(**slow_cfg)
        self.fast_path = ResNet3dPathway(**fast_cfg)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_slow = F.interpolate(
            x,
            mode="nearest",
            scale_factor=(1.0 / self.resample_rate, 1.0, 1.0),
        )
        x_slow = self.slow_path.conv1(x_slow)
        if self.slow_path.with_pool1:
            x_slow = self.slow_path.maxpool(x_slow)

        fast_scale = 1.0 / max(1, self.resample_rate // self.speed_ratio)
        x_fast = F.interpolate(
            x,
            mode="nearest",
            scale_factor=(fast_scale, 1.0, 1.0),
        )
        x_fast = self.fast_path.conv1(x_fast)
        if self.fast_path.with_pool1:
            x_fast = self.fast_path.maxpool(x_fast)

        if self.slow_path.lateral and hasattr(self.slow_path, "conv1_lateral"):
            x_slow = torch.cat((x_slow, self.slow_path.conv1_lateral(x_fast)), dim=1)

        for i, layer_name in enumerate(self.slow_path.res_layers):
            x_slow = getattr(self.slow_path, layer_name)(x_slow)
            x_fast = getattr(self.fast_path, layer_name)(x_fast)
            if i == 0 and self.slow_path.with_pool2:
                x_slow = self.slow_path.pool2(x_slow)
            if i == 0 and self.fast_path.with_pool2:
                x_fast = self.fast_path.pool2(x_fast)
            if i != len(self.slow_path.res_layers) - 1 and self.slow_path.lateral:
                lateral_name = self.slow_path.lateral_connections[i]
                x_slow = torch.cat((x_slow, getattr(self.slow_path, lateral_name)(x_fast)), dim=1)

        return x_slow, x_fast

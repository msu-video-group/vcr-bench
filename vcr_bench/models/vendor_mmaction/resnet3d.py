from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _ntuple, _triple

from .common import ConvModule, build_activation_layer, build_norm_layer


class NonLocal3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        sub_sample: bool = True,
        use_scale: bool = False,
        mode: str = "embedded_gaussian",
        norm_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        if mode not in {"embedded_gaussian", "dot_product"}:
            raise ValueError(f"Unsupported non-local mode: {mode}")
        inter_channels = max(1, int(in_channels) // 2)
        self.inter_channels = inter_channels
        self.use_scale = bool(use_scale)
        self.mode = str(mode)
        self.g = nn.Conv3d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv3d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv3d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.conv_out = nn.Conv3d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0)
        _, self.bn = build_norm_layer(norm_cfg or {"type": "BN3d"}, in_channels)
        nn.init.constant_(self.bn.weight, 0.0)
        nn.init.constant_(self.bn.bias, 0.0)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2)) if sub_sample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        g_x = self.g(x)
        phi_x = self.phi(x)
        if self.pool is not None:
            g_x = self.pool(g_x)
            phi_x = self.pool(phi_x)
        theta_x = self.theta(x)
        batch_size = x.shape[0]
        theta_x = theta_x.view(batch_size, self.inter_channels, -1).transpose(1, 2)
        phi_x = phi_x.view(batch_size, self.inter_channels, -1)
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            pairwise_weight = pairwise_weight / float(self.inter_channels) ** 0.5
        if self.mode == "embedded_gaussian":
            pairwise_weight = F.softmax(pairwise_weight, dim=-1)
        else:
            pairwise_weight = pairwise_weight / float(pairwise_weight.shape[-1])
        g_x = g_x.view(batch_size, self.inter_channels, -1).transpose(1, 2)
        y = torch.matmul(pairwise_weight, g_x).transpose(1, 2).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.shape[2:])
        y = self.bn(self.conv_out(y))
        return residual + y


class BasicBlock3d(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        spatial_stride: int = 1,
        temporal_stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
        style: str = "pytorch",
        inflate: bool = True,
        non_local: bool = False,
        non_local_cfg: dict | None = None,
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if style not in ("pytorch", "caffe"):
            raise ValueError(f"Unsupported style: {style}")
        self.conv1 = ConvModule(
            inplanes,
            planes,
            kernel_size=(3, 3, 3) if inflate else (1, 3, 3),
            stride=(temporal_stride, spatial_stride, spatial_stride),
            padding=(1, dilation, dilation) if inflate else (0, dilation, dilation),
            dilation=(1, dilation, dilation),
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.conv2 = ConvModule(
            planes,
            planes,
            kernel_size=(3, 3, 3) if inflate else (1, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1) if inflate else (0, 1, 1),
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )
        self.downsample = downsample
        self.non_local_block = NonLocal3d(planes, **dict(non_local_cfg or {})) if non_local else None
        self.relu = build_activation_layer(act_cfg or {"type": "ReLU", "inplace": True})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        if self.non_local_block is not None:
            out = self.non_local_block(out)
        out = self.relu(out)
        return out


class Bottleneck3d(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        spatial_stride: int = 1,
        temporal_stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
        style: str = "pytorch",
        inflate: bool = True,
        inflate_style: str = "3x1x1",
        non_local: bool = False,
        non_local_cfg: dict | None = None,
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        if style not in ("pytorch", "caffe"):
            raise ValueError(f"Unsupported style: {style}")
        if inflate_style not in ("3x1x1", "3x3x3"):
            raise ValueError(f"Unsupported inflate_style: {inflate_style}")

        if style == "pytorch":
            conv1_stride_s = 1
            conv2_stride_s = spatial_stride
            conv1_stride_t = 1
            conv2_stride_t = temporal_stride
        else:
            conv1_stride_s = spatial_stride
            conv2_stride_s = 1
            conv1_stride_t = temporal_stride
            conv2_stride_t = 1

        if inflate:
            if inflate_style == "3x1x1":
                conv1_kernel_size = (3, 1, 1)
                conv1_padding = (1, 0, 0)
                conv2_kernel_size = (1, 3, 3)
                conv2_padding = (0, dilation, dilation)
            else:
                conv1_kernel_size = (1, 1, 1)
                conv1_padding = (0, 0, 0)
                conv2_kernel_size = (3, 3, 3)
                conv2_padding = (1, dilation, dilation)
        else:
            conv1_kernel_size = (1, 1, 1)
            conv1_padding = (0, 0, 0)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, dilation, dilation)

        self.conv1 = ConvModule(
            inplanes,
            planes,
            conv1_kernel_size,
            stride=(conv1_stride_t, conv1_stride_s, conv1_stride_s),
            padding=conv1_padding,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.conv2 = ConvModule(
            planes,
            planes,
            conv2_kernel_size,
            stride=(conv2_stride_t, conv2_stride_s, conv2_stride_s),
            padding=conv2_padding,
            dilation=(1, dilation, dilation),
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.conv3 = ConvModule(
            planes,
            planes * self.expansion,
            1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )
        self.downsample = downsample
        self.non_local_block = NonLocal3d(planes * self.expansion, **dict(non_local_cfg or {})) if non_local else None
        self.relu = build_activation_layer(act_cfg or {"type": "ReLU", "inplace": True})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        if self.non_local_block is not None:
            out = self.non_local_block(out)
        out = self.relu(out)
        return out


class ResNet3d(nn.Module):
    arch_settings = {
        18: (BasicBlock3d, (2, 2, 2, 2)),
        34: (BasicBlock3d, (3, 4, 6, 3)),
        50: (Bottleneck3d, (3, 4, 6, 3)),
        101: (Bottleneck3d, (3, 4, 23, 3)),
    }

    def __init__(
        self,
        depth: int = 50,
        pretrained: str | None = None,
        pretrained2d: bool = False,
        in_channels: int = 3,
        num_stages: int = 4,
        base_channels: int = 64,
        out_indices: Sequence[int] = (3,),
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
        non_local: Sequence[int] = (0, 0, 0, 0),
        non_local_cfg: dict | None = None,
        inflate_style: str = "3x1x1",
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
        norm_eval: bool = False,
        zero_init_residual: bool = True,
        lateral: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            # Keep constructor compatible with mmaction-style configs while the
            # local vendor port implements only the pieces we actually use.
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unsupported ResNet3d kwargs: {unknown}")
        if depth not in self.arch_settings:
            raise KeyError(f"invalid depth {depth} for resnet3d")
        self.depth = int(depth)
        self.pretrained = pretrained
        self.pretrained2d = bool(pretrained2d)
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.num_stages = int(num_stages)
        self.out_indices = tuple(int(i) for i in out_indices)
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
        self.stage_inflations = _ntuple(num_stages)(inflate)
        self.non_local_stages = _ntuple(num_stages)(non_local)
        self.non_local_cfg = dict(non_local_cfg or {})
        self.inflate_style = str(inflate_style)
        self.conv_cfg = dict(conv_cfg or {"type": "Conv3d"})
        self.norm_cfg = dict(norm_cfg or {"type": "BN3d", "requires_grad": True})
        self.act_cfg = dict(act_cfg or {"type": "ReLU", "inplace": True})
        self.norm_eval = bool(norm_eval)
        self.zero_init_residual = bool(zero_init_residual)
        self.lateral = bool(lateral)

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
                spatial_stride=self.spatial_strides[i],
                temporal_stride=self.temporal_strides[i],
                dilation=self.dilations[i],
                style=self.style,
                inflate=self.stage_inflations[i],
                non_local=self.non_local_stages[i],
                non_local_cfg=self.non_local_cfg,
                inflate_style=self.inflate_style,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg,
                act_cfg=self.act_cfg,
            )
            self.inplanes = planes * self.block.expansion
            layer_name = f"layer{i + 1}"
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * self.base_channels * 2 ** (len(self.stage_blocks) - 1)

    @staticmethod
    def make_res_layer(
        block: type[Bottleneck3d],
        inplanes: int,
        planes: int,
        blocks: int,
        spatial_stride: int = 1,
        temporal_stride: int = 1,
        dilation: int = 1,
        style: str = "pytorch",
        inflate: int | Sequence[int] = 1,
        non_local: int | Sequence[int] = 0,
        non_local_cfg: dict | None = None,
        inflate_style: str = "3x1x1",
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
        conv_cfg: dict | None = None,
    ) -> nn.Module:
        inflate = inflate if not isinstance(inflate, int) else (inflate,) * blocks
        non_local = non_local if not isinstance(non_local, int) else (non_local,) * blocks
        downsample = None
        if spatial_stride != 1 or inplanes != planes * block.expansion:
            downsample = ConvModule(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=(temporal_stride, spatial_stride, spatial_stride),
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
            )

        layers = [
            block(
                inplanes,
                planes,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                downsample=downsample,
                style=style,
                inflate=(inflate[0] == 1),
                non_local=(non_local[0] == 1),
                non_local_cfg=non_local_cfg,
                inflate_style=inflate_style,
                norm_cfg=norm_cfg,
                conv_cfg=conv_cfg,
                act_cfg=act_cfg,
            )
        ]
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    spatial_stride=1,
                    temporal_stride=1,
                    dilation=dilation,
                    style=style,
                    inflate=(inflate[i] == 1),
                    non_local=(non_local[i] == 1),
                    non_local_cfg=non_local_cfg,
                    inflate_style=inflate_style,
                    norm_cfg=norm_cfg,
                    conv_cfg=conv_cfg,
                    act_cfg=act_cfg,
                )
            )
        return nn.Sequential(*layers)

    def _make_stem_layer(self) -> None:
        self.conv1 = ConvModule(
            self.in_channels,
            self.base_channels,
            kernel_size=self.conv1_kernel,
            stride=(self.conv1_stride_t, self.conv1_stride_s, self.conv1_stride_s),
            padding=tuple((k - 1) // 2 for k in _triple(self.conv1_kernel)),
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(self.pool1_stride_t, self.pool1_stride_s, self.pool1_stride_s),
            padding=(0, 1, 1),
        )
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        if self.with_pool1:
            x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            x = getattr(self, layer_name)(x)
            if i == 0 and self.with_pool2:
                x = self.pool2(x)
            if i in self.out_indices:
                outs.append(x)
        return outs[0] if len(outs) == 1 else tuple(outs)


class ResNet3dSlowOnly(ResNet3d):
    def __init__(
        self,
        conv1_kernel: Sequence[int] = (1, 7, 7),
        conv1_stride_t: int = 1,
        pool1_stride_t: int = 1,
        inflate: Sequence[int] = (0, 0, 1, 1),
        with_pool2: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            conv1_kernel=conv1_kernel,
            conv1_stride_t=conv1_stride_t,
            pool1_stride_t=pool1_stride_t,
            inflate=inflate,
            with_pool2=with_pool2,
            **kwargs,
        )


class ResNet2Plus1d(ResNet3d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if str(self.conv_cfg.get("type")) != "Conv2plus1d":
            raise ValueError("ResNet2Plus1d requires conv_cfg.type='Conv2plus1d'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        for layer_name in self.res_layers:
            x = getattr(self, layer_name)(x)
        return x

from __future__ import annotations

import torch.nn as nn

from .common import ConvModule
from .resnet3d import Bottleneck3d, ResNet3d


class CSNBottleneck3d(Bottleneck3d):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        *args,
        bottleneck_mode: str = "ir",
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.bottleneck_mode = str(bottleneck_mode)
        if self.bottleneck_mode not in ("ip", "ir"):
            raise ValueError(f"Unsupported CSN bottleneck mode: {self.bottleneck_mode}")
        super().__init__(
            inplanes,
            planes,
            *args,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs,
        )

        conv2_layers: list[nn.Module] = []
        if self.bottleneck_mode == "ip":
            conv2_layers.append(
                ConvModule(
                    planes,
                    planes,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=None,
                )
            )

        conv2_kernel_size = self.conv2.conv.kernel_size
        conv2_stride = self.conv2.conv.stride
        conv2_padding = self.conv2.conv.padding
        conv2_dilation = self.conv2.conv.dilation
        conv2_bias = bool(self.conv2.conv.bias)
        conv2_layers.append(
            ConvModule(
                planes,
                planes,
                kernel_size=conv2_kernel_size,
                stride=conv2_stride,
                padding=conv2_padding,
                dilation=conv2_dilation,
                groups=planes,
                bias=conv2_bias,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )
        )
        self.conv2 = nn.Sequential(*conv2_layers)


class ResNet3dCSN(ResNet3d):
    arch_settings = {
        50: (CSNBottleneck3d, (3, 4, 6, 3)),
    }

    def __init__(
        self,
        depth: int = 50,
        pretrained: str | None = None,
        temporal_strides=(1, 2, 2, 2),
        conv1_kernel=(3, 7, 7),
        conv1_stride_t: int = 1,
        pool1_stride_t: int = 1,
        norm_cfg: dict | None = None,
        inflate_style: str = "3x3x3",
        bottleneck_mode: str = "ir",
        bn_frozen: bool = False,
        **kwargs,
    ) -> None:
        self.bn_frozen = bool(bn_frozen)
        self.bottleneck_mode = str(bottleneck_mode)
        super().__init__(
            depth=depth,
            pretrained=pretrained,
            temporal_strides=temporal_strides,
            conv1_kernel=conv1_kernel,
            conv1_stride_t=conv1_stride_t,
            pool1_stride_t=pool1_stride_t,
            norm_cfg=norm_cfg or dict(type="BN3d", requires_grad=True, eps=1e-3),
            inflate_style=inflate_style,
            **kwargs,
        )

    def make_res_layer(
        self,
        block,
        inplanes: int,
        planes: int,
        blocks: int,
        spatial_stride: int = 1,
        temporal_stride: int = 1,
        dilation: int = 1,
        style: str = "pytorch",
        inflate: int | tuple[int, ...] = 1,
        non_local: int | tuple[int, ...] = 0,
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
                bottleneck_mode=self.bottleneck_mode,
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
                    bottleneck_mode=self.bottleneck_mode,
                    norm_cfg=norm_cfg,
                    conv_cfg=conv_cfg,
                    act_cfg=act_cfg,
                )
            )
        return nn.Sequential(*layers)

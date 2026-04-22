from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple


def build_activation_layer(act_cfg: dict[str, Any] | None) -> nn.Module:
    if not act_cfg:
        return nn.Identity()
    act_type = str(act_cfg.get("type", "ReLU"))
    inplace = bool(act_cfg.get("inplace", True))
    if act_type == "ReLU":
        return nn.ReLU(inplace=inplace)
    if act_type == "GELU":
        return nn.GELU()
    if act_type in {"Swish", "SiLU"}:
        return nn.SiLU(inplace=inplace)
    raise ValueError(f"Unsupported activation type: {act_type}")


def build_norm_layer(norm_cfg: dict[str, Any] | None, num_features: int) -> tuple[str, nn.Module]:
    cfg = dict(norm_cfg or {"type": "BN3d"})
    norm_type = str(cfg.get("type", "BN3d"))
    if norm_type in ("BN3d", "SyncBN"):
        layer = nn.BatchNorm3d(num_features)
    elif norm_type == "BN2d":
        layer = nn.BatchNorm2d(num_features)
    elif norm_type == "LN":
        layer = nn.LayerNorm(num_features, eps=float(cfg.get("eps", 1e-5)))
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")
    return "bn", layer


class Conv2plus1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        norm_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.norm_cfg = dict(norm_cfg or {"type": "BN3d"})

        mid_channels = 3 * (in_channels * out_channels * kernel_size[1] * kernel_size[2])
        denom = in_channels * kernel_size[1] * kernel_size[2] + 3 * out_channels
        mid_channels = max(1, int(mid_channels / denom))

        self.conv_s = nn.Conv3d(
            in_channels,
            mid_channels,
            kernel_size=(1, kernel_size[1], kernel_size[2]),
            stride=(1, stride[1], stride[2]),
            padding=(0, padding[1], padding[2]),
            dilation=(1, dilation[1], dilation[2]),
            groups=groups,
            bias=bias,
        )
        _, self.bn_s = build_norm_layer(self.norm_cfg, mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv_t = nn.Conv3d(
            mid_channels,
            out_channels,
            kernel_size=(kernel_size[0], 1, 1),
            stride=(stride[0], 1, 1),
            padding=(padding[0], 0, 0),
            dilation=(dilation[0], 1, 1),
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_s(x)
        x = self.bn_s(x)
        x = self.relu(x)
        x = self.conv_t(x)
        return x


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = False,
        conv_cfg: dict[str, Any] | None = None,
        norm_cfg: dict[str, Any] | None = None,
        act_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        cfg = dict(conv_cfg or {"type": "Conv3d"})
        conv_type = str(cfg.get("type", "Conv3d"))
        if conv_type == "Conv3d":
            self.conv = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        elif conv_type == "Conv2d":
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        elif conv_type == "Conv2plus1d":
            self.conv = Conv2plus1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
                norm_cfg=norm_cfg,
            )
        else:
            raise ValueError(f"Unsupported conv type: {conv_type}")
        self.with_norm = norm_cfg is not None
        if self.with_norm:
            self.norm_name, self.bn = build_norm_layer(norm_cfg, out_channels)
        else:
            self.norm_name, self.bn = "bn", nn.Identity()
        self.with_activation = act_cfg is not None
        self.activate = build_activation_layer(act_cfg)

    @property
    def norm(self) -> nn.Module:
        return self.bn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        return x

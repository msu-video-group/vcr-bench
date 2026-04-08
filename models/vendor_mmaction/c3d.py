from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .common import ConvModule


class C3D(nn.Module):
    def __init__(
        self,
        conv_cfg: dict[str, Any] | None = None,
        norm_cfg: dict[str, Any] | None = None,
        act_cfg: dict[str, Any] | None = None,
        out_dim: int = 8192,
        dropout_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.conv_cfg = dict(conv_cfg or {"type": "Conv3d"})
        self.norm_cfg = dict(norm_cfg) if norm_cfg is not None else None
        self.act_cfg = dict(act_cfg or {"type": "ReLU"})
        self.out_dim = int(out_dim)
        self.dropout_ratio = float(dropout_ratio)

        c3d_conv_param = dict(
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

        self.conv1a = ConvModule(3, 64, **c3d_conv_param)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2a = ConvModule(64, 128, **c3d_conv_param)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = ConvModule(128, 256, **c3d_conv_param)
        self.conv3b = ConvModule(256, 256, **c3d_conv_param)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = ConvModule(256, 512, **c3d_conv_param)
        self.conv4b = ConvModule(512, 512, **c3d_conv_param)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = ConvModule(512, 512, **c3d_conv_param)
        self.conv5b = ConvModule(512, 512, **c3d_conv_param)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(self.out_dim, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=self.dropout_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1a(x)
        x = self.pool1(x)

        x = self.conv2a(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.pool4(x)

        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.pool5(x)

        x = x.flatten(start_dim=1)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        return x

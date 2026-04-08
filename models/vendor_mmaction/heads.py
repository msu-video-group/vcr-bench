from __future__ import annotations

import torch
import torch.nn as nn


class I3DHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        spatial_type: str = "avg",
        dropout_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.spatial_type = str(spatial_type)
        self.dropout_ratio = float(dropout_ratio)
        self.dropout = nn.Dropout(p=self.dropout_ratio) if self.dropout_ratio > 0 else None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1)) if self.spatial_type == "avg" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        return self.fc_cls(x)


class TSNHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        num_segs: int,
        dropout_ratio: float = 0.4,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.num_segs = int(num_segs)
        self.dropout_ratio = float(dropout_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=self.dropout_ratio) if self.dropout_ratio > 0 else None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc_cls(x)
        return x.view((-1, self.num_segs, self.num_classes)).mean(dim=1)


class SlowFastHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        spatial_type: str = "avg",
        dropout_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.spatial_type = str(spatial_type)
        self.dropout_ratio = float(dropout_ratio)
        self.dropout = nn.Dropout(p=self.dropout_ratio) if self.dropout_ratio > 0 else None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1)) if self.spatial_type == "avg" else None

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_slow, x_fast = x
        if self.avg_pool is not None:
            x_slow = self.avg_pool(x_slow)
            x_fast = self.avg_pool(x_fast)
        x = torch.cat((x_fast, x_slow), dim=1)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.fc_cls(x)


class UniFormerHead(nn.Module):
    def __init__(self, num_classes: int, in_channels: int, dropout_ratio: float = 0.0) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.dropout = nn.Dropout(p=float(dropout_ratio)) if dropout_ratio and dropout_ratio > 0 else None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc_cls(x)


class MViTHead(nn.Module):
    def __init__(self, num_classes: int, in_channels: int, dropout_ratio: float = 0.5) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=float(dropout_ratio)) if dropout_ratio and dropout_ratio > 0 else None
        self.fc_cls = nn.Linear(int(in_channels), int(num_classes))

    def forward(self, feats: tuple[list[torch.Tensor | None], ...]) -> torch.Tensor:
        _, cls_token = feats[-1]
        if cls_token is None:
            raise ValueError("MViTHead expects cls token output from backbone")
        if self.dropout is not None:
            cls_token = self.dropout(cls_token)
        return self.fc_cls(cls_token)


class X3DHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        spatial_type: str = "avg",
        dropout_ratio: float = 0.5,
        fc1_bias: bool = False,
    ) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1)) if spatial_type == "avg" else None
        self.fc1 = nn.Linear(int(in_channels), 2048, bias=bool(fc1_bias))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=float(dropout_ratio)) if dropout_ratio and dropout_ratio > 0 else None
        self.fc2 = nn.Linear(2048, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool is not None:
            x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc2(x)


class TPNHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        spatial_type: str = "avg",
        dropout_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1)) if spatial_type == "avg" else None
        self.dropout = nn.Dropout(p=float(dropout_ratio)) if dropout_ratio and dropout_ratio > 0 else None
        self.fc_cls = nn.Linear(int(in_channels), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool is not None:
            x = self.pool(x)
        x = x.view(x.shape[0], -1)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc_cls(x)

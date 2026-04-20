from __future__ import annotations

import torch
import torch.nn as nn


class Recognizer3D(nn.Module):
    def __init__(self, backbone: nn.Module, cls_head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.cls_head = cls_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.cls_head(feat)


class Recognizer2D(nn.Module):
    def __init__(self, backbone: nn.Module, cls_head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.cls_head = cls_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.cls_head(feat)


class Recognizer3DWithNeck(nn.Module):
    def __init__(self, backbone: nn.Module, neck: nn.Module, cls_head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.cls_head = cls_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = self.neck(feat)
        return self.cls_head(feat)

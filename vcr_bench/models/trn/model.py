from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..tsn.model import TSNClassifier


class _RelationModuleMultiScale(nn.Module):
    def __init__(self, hidden_dim: int, num_segments: int, num_classes: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_segments = int(num_segments)
        self.num_classes = int(num_classes)
        self.scales = list(range(self.num_segments, 1, -1))
        self.relations_scales = [
            list(itertools.combinations(range(self.num_segments), scale))
            for scale in self.scales
        ]
        self.subsample_scales = [min(3, len(relations)) for relations in self.relations_scales]

        self.fc_fusion_scales = nn.ModuleList(
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(scale * self.hidden_dim, 256),
                nn.ReLU(),
                nn.Linear(256, self.num_classes),
            )
            for scale in self.scales
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act_all = x[:, self.relations_scales[0][0], :]
        act_all = act_all.reshape(act_all.size(0), self.scales[0] * self.hidden_dim)
        act_all = self.fc_fusion_scales[0](act_all)

        for scale_id in range(1, len(self.scales)):
            sampled = np.random.choice(
                len(self.relations_scales[scale_id]),
                self.subsample_scales[scale_id],
                replace=False,
            )
            for idx in sampled:
                act_relation = x[:, self.relations_scales[scale_id][idx], :]
                act_relation = act_relation.reshape(
                    act_relation.size(0),
                    self.scales[scale_id] * self.hidden_dim,
                )
                act_all = act_all + self.fc_fusion_scales[scale_id](act_relation)
        return act_all


class _TRNHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        num_segments: int = 8,
        hidden_dim: int = 256,
        dropout_ratio: float = 0.8,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.num_segments = int(num_segments)
        self.hidden_dim = int(hidden_dim)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=float(dropout_ratio)) if dropout_ratio > 0 else None
        self.fc_cls = nn.Linear(self.in_channels, self.hidden_dim)
        self.consensus = _RelationModuleMultiScale(
            hidden_dim=self.hidden_dim,
            num_segments=self.num_segments,
            num_classes=self.num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc_cls(x)
        x = x.reshape((-1, self.num_segments) + x.shape[1:])
        return self.consensus(x)


class TRNClassifier(TSNClassifier):
    model_name = "trn"
    MODEL_CAPABILITIES: dict[str, Any] = {
        "backbones": {
            "r50": {
                "display_name": "TRN R50",
                "datasets": {
                    "sthv2": {
                        "num_classes": 174,
                        "checkpoint_url": "https://huggingface.co/maxv65/vcr-bench/resolve/main/trn_r50_sthv2.pth",
                        "checkpoint_filename": "trn_r50_sthv2.pth",
                    }
                },
            }
        }
    }

    def __init__(self, *args, **kwargs) -> None:
        checkpoint_path = kwargs.pop("checkpoint_path", None)
        load_weights = kwargs.pop("load_weights", None)
        auto_download = kwargs.pop("auto_download", True)
        backbone = kwargs.pop("backbone", None)
        weights_dataset = kwargs.pop("weights_dataset", None)
        num_classes = kwargs.pop("num_classes", None)
        super().__init__(
            *args,
            backbone="r50" if backbone is None else backbone,
            weights_dataset="sthv2" if weights_dataset is None else weights_dataset,
            num_classes=174 if num_classes is None else num_classes,
            checkpoint_path=checkpoint_path,
            load_weights=False,
            auto_download=auto_download,
            **kwargs,
        )
        self.model.cls_head = _TRNHead(
            num_classes=self.num_classes,
            in_channels=2048,
            num_segments=int(self.loading_configs["test"]["num_clips"]),
            hidden_dim=256,
            dropout_ratio=0.8,
        )

        should_load = True if load_weights is None else bool(load_weights)
        if should_load:
            self.checkpoint_path = self.resolve_checkpoint_path(
                self.backbone,
                self.weights_dataset,
                checkpoint_path,
                auto_download=auto_download,
            )
            self._load_checkpoint(self.checkpoint_path)
        else:
            self.checkpoint_path = checkpoint_path

        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False

    def _build_stage_config_dicts(self) -> tuple[dict[str, dict], dict[str, dict]]:
        common_loading = {
            "raw_input_format": "NTHWC",
            "sampled_format": "NTHWC",
            "clip_len": 1,
            "frame_interval": 1,
            "full_videos": False,
        }
        common_pre = {
            "preprocessed_format": "NTCHW",
            "resize_short": 256,
            "crop_size": 224,
            "input_range": 1.0,
            "mean": (123.675, 116.28, 103.53),
            "std": (58.395, 57.12, 57.375),
        }
        loading = {
            "train": {**common_loading, "num_clips": 8},
            "val": {**common_loading, "num_clips": 8},
            "test": {**common_loading, "num_clips": 8},
            "attack": {**common_loading, "num_clips": 8},
        }
        preprocessing = {
            "train": {**common_pre, "spatial_strategy": "center_crop"},
            "val": {**common_pre, "spatial_strategy": "center_crop"},
            "test": {**common_pre, "crop_size": 256, "spatial_strategy": "three_crop"},
            "attack": {**common_pre, "spatial_strategy": "center_crop"},
        }
        return loading, preprocessing


MODEL_CLASS = TRNClassifier


def create(**kwargs) -> TRNClassifier:
    return TRNClassifier(**kwargs)

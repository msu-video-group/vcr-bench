from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..base import BaseVideoClassifier
from ..pipeline_config import PipelineStage
from ..vendor_mmaction import ResNet2d, TSNHead


class _TSNRecognizer(nn.Module):
    def __init__(self, backbone: nn.Module, cls_head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.cls_head = cls_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"TSN expected 5D input, got {tuple(x.shape)}")
        if x.shape[1] == 3:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        elif x.shape[2] != 3:
            raise ValueError(f"TSN unsupported input layout for shape {tuple(x.shape)}")
        n, t, c, h, w = x.shape
        feat = self.backbone(x.reshape(n * t, c, h, w))
        self.cls_head.num_segs = t
        return self.cls_head(feat)


class TSNClassifier(BaseVideoClassifier):
    model_name = "tsn"
    raw_input_format = "NTHWC"
    preprocessed_format = "NTCHW"
    MODEL_CAPABILITIES: dict[str, Any] = {
        "backbones": {
            "r101": {
                "display_name": "ResNet-101",
                "datasets": {
                    "kinetics400": {
                        "num_classes": 400,
                        "checkpoint_url": (
                            "https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/"
                            "tsn_imagenet-pretrained-r101_8xb32-1x1x8-100e_kinetics400-rgb/"
                            "tsn_imagenet-pretrained-r101_8xb32-1x1x8-100e_kinetics400-rgb_20220906-23cff032.pth"
                        ),
                        "checkpoint_filename": "tsn_r101_kinetics400_mmaction.pth",
                    }
                },
            }
        }
    }

    def __init__(
        self,
        checkpoint_path: str | None = None,
        backbone: str | None = None,
        weights_dataset: str | None = None,
        device: str = "cuda",
        grad_forward_chunk_size: int | None = None,
        load_weights: bool | None = None,
        num_classes: int | None = None,
        auto_download: bool = True,
    ) -> None:
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.backbone = backbone or "r101"
        self.weights_dataset = weights_dataset or "kinetics400"
        if grad_forward_chunk_size is not None:
            self.grad_forward_chunk_size = int(grad_forward_chunk_size)
        weight_spec = self.get_weight_spec(self.backbone, self.weights_dataset)
        self.num_classes = int(num_classes if num_classes is not None else weight_spec.get("num_classes", 400))
        self.loading_configs, self.preprocessing_configs = self._build_stage_config_dicts()
        self._active_stage: PipelineStage = "test"  # type: ignore[assignment]
        self._bind_stage_functions("test")

        num_segments = int(self.loading_configs["test"]["num_clips"])
        backbone_module = ResNet2d(
            depth=101,
            pretrained="https://download.pytorch.org/models/resnet101-cd907fc2.pth",
            norm_eval=False,
            conv_cfg={"type": "Conv2d"},
            norm_cfg={"type": "BN2d", "requires_grad": True},
            act_cfg={"type": "ReLU", "inplace": True},
        )
        cls_head = TSNHead(
            num_classes=self.num_classes,
            in_channels=2048,
            num_segs=num_segments,
            dropout_ratio=0.4,
        )
        self.model = _TSNRecognizer(backbone=backbone_module, cls_head=cls_head)

        should_load = True if load_weights is None else bool(load_weights)
        self.checkpoint_path = (
            self.resolve_checkpoint_path(
                self.backbone,
                self.weights_dataset,
                checkpoint_path,
                auto_download=auto_download,
            )
            if should_load
            else checkpoint_path
        )
        if should_load and self.checkpoint_path is not None:
            self._load_checkpoint(self.checkpoint_path)

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
            "train": {**common_loading, "num_clips": 3},
            "val": {**common_loading, "num_clips": 3},
            "test": {**common_loading, "num_clips": 25},
            "attack": {**common_loading, "num_clips": 3},
        }
        preprocessing = {
            "train": {**common_pre, "spatial_strategy": "train_random_crop"},
            "val": {**common_pre, "spatial_strategy": "center_crop"},
            "test": {**common_pre, "spatial_strategy": "center_crop"},
            "attack": {**common_pre, "spatial_strategy": "center_crop"},
        }
        return loading, preprocessing

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        self.load_converted_checkpoint(
            checkpoint_path,
            strict=False,
            ignore_prefixes=("data_preprocessor.",),
            key_replacements=(
                ("model.backbone.", "backbone."),
                ("model.cls_head.", "cls_head."),
            ),
            required_keys=("cls_head.fc_cls.weight", "cls_head.fc_cls.bias"),
        )


MODEL_CLASS = TSNClassifier


def create(**kwargs) -> TSNClassifier:
    return TSNClassifier(**kwargs)

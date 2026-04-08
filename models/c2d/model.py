from __future__ import annotations

from typing import Any

import torch

from ..base import BaseVideoClassifier
from ..pipeline_config import PipelineStage
from ..vendor_mmaction import C2D, I3DHead, Recognizer3D


class C2DClassifier(BaseVideoClassifier):
    model_name = "c2d"
    raw_input_format = "NTHWC"
    preprocessed_format = "NCTHW"
    MODEL_CAPABILITIES: dict[str, Any] = {
        "backbones": {
            "r50": {
                "display_name": "ResNet-50 C2D",
                "datasets": {
                    "kinetics400": {
                        "num_classes": 400,
                        "checkpoint_url": "https://download.openmmlab.com/mmaction/v1.0/recognition/c2d/c2d_r50-in1k-pre_8xb32-8x8x1-100e_kinetics400-rgb/c2d_r50-in1k-pre_8xb32-8x8x1-100e_kinetics400-rgb_20221027-3ca304fa.pth",
                        "checkpoint_filename": "c2d_r50_kinetics400_mmaction.pth",
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
        self.backbone = backbone or "r50"
        self.weights_dataset = weights_dataset or "kinetics400"
        if grad_forward_chunk_size is not None:
            self.grad_forward_chunk_size = int(grad_forward_chunk_size)
        weight_spec = self.get_weight_spec(self.backbone, self.weights_dataset)
        self.num_classes = int(num_classes if num_classes is not None else weight_spec.get("num_classes", 400))
        self.loading_configs, self.preprocessing_configs = self._build_stage_config_dicts()
        self._active_stage: PipelineStage = "test"  # type: ignore[assignment]
        self._bind_stage_functions("test")

        should_load = True if load_weights is None else bool(load_weights)
        backbone_module = C2D(
            depth=50,
            pretrained="https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
            norm_eval=False,
            conv_cfg={"type": "Conv2d"},
            norm_cfg={"type": "BN2d", "requires_grad": True},
            act_cfg={"type": "ReLU", "inplace": True},
        )
        cls_head = I3DHead(
            num_classes=self.num_classes,
            in_channels=2048,
            spatial_type="avg",
            dropout_ratio=0.5,
        )
        self.model = Recognizer3D(backbone=backbone_module, cls_head=cls_head)

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
            "clip_len": 8,
            "frame_interval": 8,
            "full_videos": False,
        }
        common_pre = {
            "preprocessed_format": "NCTHW",
            "resize_short": 256,
            "crop_size": 224,
            "input_range": 1.0,
            "mean": (123.675, 116.28, 103.53),
            "std": (58.395, 57.12, 57.375),
        }
        loading = {
            "train": {**common_loading, "num_clips": 1},
            "val": {**common_loading, "num_clips": 1},
            "test": {**common_loading, "num_clips": 10},
            "attack": {**common_loading, "num_clips": 1},
        }
        preprocessing = {
            "train": {**common_pre, "spatial_strategy": "train_random_crop"},
            "val": {**common_pre, "spatial_strategy": "center_crop"},
            "test": {**common_pre, "crop_size": 256, "spatial_strategy": "three_crop"},
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


MODEL_CLASS = C2DClassifier


def create(**kwargs) -> C2DClassifier:
    return C2DClassifier(**kwargs)

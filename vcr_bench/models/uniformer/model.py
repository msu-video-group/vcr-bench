from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..base import BaseVideoClassifier
from ..pipeline_config import PipelineStage
from ..vendor_mmaction import I3DHead, Recognizer3D, UniFormer


class UniFormerClassifier(BaseVideoClassifier):
    model_name = "uniformer"
    raw_input_format = "NTHWC"
    preprocessed_format = "NCTHW"
    MODEL_CAPABILITIES: dict[str, Any] = {
        "backbones": {
            "base": {
                "display_name": "UniFormer Base",
                "datasets": {
                    "kinetics400": {
                        "num_classes": 400,
                        "checkpoint_url": (
                            "https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv1/"
                            "uniformer-base_imagenet1k-pre_32x4x1_kinetics400-rgb_20221219-b776322c.pth"
                        ),
                        "checkpoint_filename": "uniformer_base_kinetics400_mmaction.pth",
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
        self.backbone = backbone or "base"
        self.weights_dataset = weights_dataset or "kinetics400"
        if grad_forward_chunk_size is not None:
            self.grad_forward_chunk_size = int(grad_forward_chunk_size)
        weight_spec = self.get_weight_spec(self.backbone, self.weights_dataset)
        self.num_classes = int(num_classes if num_classes is not None else weight_spec.get("num_classes", 400))
        self.loading_configs, self.preprocessing_configs = self._build_stage_config_dicts()
        self._active_stage: PipelineStage = "test"  # type: ignore[assignment]
        self._bind_stage_functions("test")
        self.model = Recognizer3D(
            backbone=UniFormer(
                img_size=224,
                depth=(5, 8, 20, 7),
                embed_dim=(64, 128, 320, 512),
                head_dim=64,
                mlp_ratio=4,
                qkv_bias=True,
                drop_path_rate=0.3,
            ),
            cls_head=I3DHead(
                num_classes=self.num_classes,
                in_channels=512,
                spatial_type="avg",
                dropout_ratio=0.0,
            ),
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
            "clip_len": 32,
            "frame_interval": 4,
            "full_videos": False,
        }
        common_pre = {
            "preprocessed_format": "NCTHW",
            "resize_short": 224,
            "crop_size": 224,
            "input_range": 1.0,
            "mean": (114.75, 114.75, 114.75),
            "std": (57.375, 57.375, 57.375),
        }
        loading = {
            "train": {**common_loading, "num_clips": 1},
            "val": {**common_loading, "num_clips": 1},
            "test": {**common_loading, "num_clips": 4},
            "attack": {**common_loading, "num_clips": 1},
        }
        preprocessing = {
            "train": {**common_pre, "spatial_strategy": "center_crop"},
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


MODEL_CLASS = UniFormerClassifier


def create(**kwargs) -> UniFormerClassifier:
    return UniFormerClassifier(**kwargs)

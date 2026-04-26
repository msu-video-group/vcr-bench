from __future__ import annotations

from typing import Any

import torch

from ..base import BaseVideoClassifier
from ..pipeline_config import PipelineStage
from ..vendor_mmaction import Recognizer3D, ResNet3dSlowFast, SlowFastHead


class SlowFastClassifier(BaseVideoClassifier):
    model_name = "slowfast"
    raw_input_format = "NTHWC"
    preprocessed_format = "NCTHW"
    MODEL_CAPABILITIES: dict[str, Any] = {
        "backbones": {
            "r101": {
                "display_name": "SlowFast R101",
                "datasets": {
                    "kinetics400": {
                        "num_classes": 400,
                        "checkpoint_url": (
                            "https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/"
                            "slowfast_r101-r50_32xb8-4x16x1-256e_kinetics400-rgb/"
                            "slowfast_r101-r50_32xb8-4x16x1-256e_kinetics400-rgb_20220901-a77ac3ee.pth"
                        ),
                        "checkpoint_filename": "slowfast_r101_kinetics400_mmaction.pth",
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

        self.model = Recognizer3D(
            backbone=ResNet3dSlowFast(
                pretrained=None,
                resample_rate=8,
                speed_ratio=8,
                channel_ratio=8,
                slow_pathway=dict(
                    type="resnet3d",
                    depth=101,
                    pretrained=None,
                    lateral=True,
                    conv1_kernel=(1, 7, 7),
                    dilations=(1, 1, 1, 1),
                    conv1_stride_t=1,
                    pool1_stride_t=1,
                    inflate=(0, 0, 1, 1),
                    norm_eval=False,
                    fusion_kernel=5,
                ),
                fast_pathway=dict(
                    type="resnet3d",
                    depth=50,
                    pretrained=None,
                    lateral=False,
                    base_channels=8,
                    conv1_kernel=(5, 7, 7),
                    conv1_stride_t=1,
                    pool1_stride_t=1,
                    norm_eval=False,
                ),
            ),
            cls_head=SlowFastHead(
                in_channels=2304,
                num_classes=self.num_classes,
                spatial_type="avg",
                dropout_ratio=0.5,
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
            "frame_interval": 2,
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


MODEL_CLASS = SlowFastClassifier


def create(**kwargs) -> SlowFastClassifier:
    return SlowFastClassifier(**kwargs)

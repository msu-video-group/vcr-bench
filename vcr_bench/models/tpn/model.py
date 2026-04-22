from __future__ import annotations

from typing import Any

import torch

from ..base import BaseVideoClassifier
from ..pipeline_config import PipelineStage
from ..vendor_mmaction import Recognizer3DWithNeck, ResNet3dSlowOnly, TPN, TPNHead


class TPNClassifier(BaseVideoClassifier):
    model_name = "tpn"
    raw_input_format = "NTHWC"
    preprocessed_format = "NCTHW"
    MODEL_CAPABILITIES: dict[str, Any] = {
        "backbones": {
            "r50": {
                "display_name": "TPN SlowOnly R50",
                "datasets": {
                    "kinetics400": {
                        "num_classes": 400,
                        "checkpoint_url": (
                            "https://download.openmmlab.com/mmaction/v1.0/recognition/tpn/"
                            "tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb/"
                            "tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb_20220913-97d0835d.pth"
                        ),
                        "checkpoint_filename": "tpn_r50_kinetics400_mmaction.pth",
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

        self.model = Recognizer3DWithNeck(
            backbone=ResNet3dSlowOnly(
                depth=50,
                pretrained="torchvision://resnet50",
                lateral=False,
                out_indices=(2, 3),
                conv1_kernel=(1, 7, 7),
                conv1_stride_t=1,
                pool1_stride_t=1,
                inflate=(0, 0, 1, 1),
                norm_eval=False,
            ),
            neck=TPN(
                in_channels=(1024, 2048),
                out_channels=1024,
                spatial_modulation_cfg=dict(in_channels=(1024, 2048), out_channels=2048),
                temporal_modulation_cfg=dict(downsample_scales=(8, 8)),
                upsample_cfg=dict(scale_factor=(1, 1, 1), mode="nearest"),
                downsample_cfg=dict(downsample_scale=(1, 1, 1)),
                level_fusion_cfg=dict(
                    in_channels=(1024, 1024),
                    mid_channels=(1024, 1024),
                    out_channels=2048,
                    downsample_scales=((1, 1, 1), (1, 1, 1)),
                ),
                aux_head_cfg=dict(out_channels=self.num_classes, loss_weight=0.5),
            ),
            cls_head=TPNHead(
                num_classes=self.num_classes,
                in_channels=2048,
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
                ("model.neck.", "neck."),
                ("model.cls_head.", "cls_head."),
            ),
            required_keys=("cls_head.fc_cls.weight", "cls_head.fc_cls.bias"),
        )


MODEL_CLASS = TPNClassifier


def create(**kwargs) -> TPNClassifier:
    return TPNClassifier(**kwargs)

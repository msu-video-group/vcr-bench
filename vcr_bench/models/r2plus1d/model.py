from __future__ import annotations

from typing import Any

import torch

from ..base import BaseVideoClassifier
from ..pipeline_config import PipelineStage
from ..vendor_mmaction import I3DHead, Recognizer3D, ResNet2Plus1d


class R2Plus1DClassifier(BaseVideoClassifier):
    model_name = "r2plus1d"
    raw_input_format = "NTHWC"
    preprocessed_format = "NCTHW"
    MODEL_CAPABILITIES: dict[str, Any] = {
        "backbones": {
            "r34": {
                "display_name": "ResNet2Plus1d-34",
                "datasets": {
                    "kinetics400": {
                        "num_classes": 400,
                        "checkpoint_url": (
                            "https://download.openmmlab.com/mmaction/v1.0/recognition/r2plus1d/"
                            "r2plus1d_r34_8xb8-8x8x1-180e_kinetics400-rgb/"
                            "r2plus1d_r34_8xb8-8x8x1-180e_kinetics400-rgb_20220812-47cfe041.pth"
                        ),
                        "checkpoint_filename": "r2plus1d_r34_kinetics400_mmaction.pth",
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
        self.backbone = backbone or "r34"
        self.weights_dataset = weights_dataset or "kinetics400"
        if grad_forward_chunk_size is not None:
            self.grad_forward_chunk_size = int(grad_forward_chunk_size)
        weight_spec = self.get_weight_spec(self.backbone, self.weights_dataset)
        self.num_classes = int(num_classes if num_classes is not None else weight_spec.get("num_classes", 400))
        self.loading_configs, self.preprocessing_configs = self._build_stage_config_dicts()
        self._active_stage: PipelineStage = "test"  # type: ignore[assignment]
        self._bind_stage_functions("test")

        self.model = Recognizer3D(
            backbone=ResNet2Plus1d(
                depth=34,
                pretrained=None,
                pretrained2d=False,
                norm_eval=False,
                conv_cfg=dict(type="Conv2plus1d"),
                norm_cfg=dict(type="SyncBN", requires_grad=True, eps=1e-3),
                conv1_kernel=(3, 7, 7),
                conv1_stride_t=1,
                pool1_stride_t=1,
                inflate=(1, 1, 1, 1),
                spatial_strides=(1, 2, 2, 2),
                temporal_strides=(1, 2, 2, 2),
                zero_init_residual=False,
            ),
            cls_head=I3DHead(
                in_channels=512,
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

    def _aggregate_probs(self, probs: torch.Tensor) -> torch.Tensor:
        if probs.ndim == 1:
            return probs
        weights = probs.max(dim=-1).values
        weights = weights / weights.sum().clamp_min(1e-12)
        return torch.tensordot(probs, weights, dims=([0], [0]))

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        self.load_converted_checkpoint(
            checkpoint_path,
            strict=False,
            ignore_prefixes=("data_preprocessor.",),
            required_keys=("cls_head.fc_cls.weight", "cls_head.fc_cls.bias"),
        )


MODEL_CLASS = R2Plus1DClassifier


def create(**kwargs) -> R2Plus1DClassifier:
    return R2Plus1DClassifier(**kwargs)

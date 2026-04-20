from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..base import BaseVideoClassifier
from ..pipeline_config import PipelineStage
from ..vendor_mmaction import C3D, I3DHead


class _C3DRecognizer(nn.Module):
    def __init__(self, backbone: nn.Module, cls_head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.cls_head = cls_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls_head(self.backbone(x))


class C3DClassifier(BaseVideoClassifier):
    model_name = "c3d"
    raw_input_format = "NTHWC"
    preprocessed_format = "NCTHW"
    MODEL_CAPABILITIES: dict[str, Any] = {
        "backbones": {
            "c3d": {
                "display_name": "C3D",
                "datasets": {
                    "ucf101": {
                        "num_classes": 101,
                        "checkpoint_url": (
                            "https://download.openmmlab.com/mmaction/v1.0/recognition/c3d/"
                            "c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb/"
                            "c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb_20220811-31723200.pth"
                        ),
                        "checkpoint_filename": "c3d_ucf101_mmaction.pth",
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
        self.backbone = backbone or "c3d"
        self.weights_dataset = weights_dataset or "ucf101"
        if grad_forward_chunk_size is not None:
            self.grad_forward_chunk_size = int(grad_forward_chunk_size)
        weight_spec = self.get_weight_spec(self.backbone, self.weights_dataset)
        self.num_classes = int(num_classes if num_classes is not None else weight_spec.get("num_classes", 101))
        self.loading_configs, self.preprocessing_configs = self._build_stage_config_dicts()
        self._active_stage: PipelineStage = "test"  # type: ignore[assignment]
        self._bind_stage_functions("test")

        self.model = _C3DRecognizer(
            backbone=C3D(
                conv_cfg={"type": "Conv3d"},
                norm_cfg=None,
                act_cfg={"type": "ReLU"},
                out_dim=8192,
                dropout_ratio=0.5,
            ),
            cls_head=I3DHead(
                in_channels=4096,
                num_classes=self.num_classes,
                spatial_type="none",
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
            "clip_len": 16,
            "frame_interval": 1,
            "full_videos": False,
        }
        common_pre = {
            "preprocessed_format": "NCTHW",
            "resize_short": 128,
            "crop_size": 112,
            "input_range": 1.0,
            "mean": (104.0, 117.0, 128.0),
            "std": (1.0, 1.0, 1.0),
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


MODEL_CLASS = C3DClassifier


def create(**kwargs) -> C3DClassifier:
    return C3DClassifier(**kwargs)

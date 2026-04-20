from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..base import BaseVideoClassifier
from ..pipeline_config import PipelineStage
from ..vendor_mmaction import UniFormerHead, UniFormerV2


class _Recognizer(nn.Module):
    def __init__(self, backbone: nn.Module, cls_head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.cls_head = cls_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls_head(self.backbone(x))


class UniFormerV2Classifier(BaseVideoClassifier):
    model_name = "uniformerv2"
    raw_input_format = "NTHWC"
    preprocessed_format = "NCTHW"
    MODEL_CAPABILITIES: dict[str, Any] = {
        "backbones": {
            "base": {
                "display_name": "UniFormerV2 Base",
                "datasets": {
                    "kinetics400": {
                        "num_classes": 400,
                        "checkpoint_url": (
                            "https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv2/"
                            "uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_kinetics400-rgb/"
                            "uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_kinetics400-rgb_20230313-75be0806.pth"
                        ),
                        "checkpoint_filename": "uniformerv2_base_kinetics400_mmaction.pth",
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
        self.model = _Recognizer(
            backbone=UniFormerV2(
                input_resolution=224,
                patch_size=16,
                width=768,
                layers=12,
                heads=12,
                t_size=8,
                dw_reduction=1.5,
                backbone_drop_path_rate=0.0,
                temporal_downsample=False,
                no_lmhra=True,
                double_lmhra=True,
                return_list=[8, 9, 10, 11],
                n_layers=4,
                n_dim=768,
                n_head=12,
                mlp_factor=4.0,
                drop_path_rate=0.0,
                mlp_dropout=[0.5, 0.5, 0.5, 0.5],
            ),
            cls_head=UniFormerHead(num_classes=self.num_classes, in_channels=768, dropout_ratio=0.5),
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
            "frame_interval": 1,
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
            "test": {**common_pre, "spatial_strategy": "three_crop"},
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


MODEL_CLASS = UniFormerV2Classifier


def create(**kwargs) -> UniFormerV2Classifier:
    return UniFormerV2Classifier(**kwargs)

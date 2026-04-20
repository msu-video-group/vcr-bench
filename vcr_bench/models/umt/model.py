from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..base import BaseVideoClassifier
from ..legacy_imports import import_attr
from ..pipeline_config import PipelineStage


def _umt_root() -> Path:
    return Path(__file__).resolve().parents[3] / "Classifiers" / "UMT"


class UMTClassifier(BaseVideoClassifier):
    model_name = "umt"
    raw_input_format = "NTHWC"
    preprocessed_format = "NCTHW"
    MODEL_CAPABILITIES: dict[str, Any] = {
        "backbones": {
            "vit_large_p16": {
                "display_name": "UMT ViT-L/16",
                "datasets": {
                    "kinetics400": {
                        "num_classes": 400,
                        "checkpoint_relpath": "Classifiers/UMT/checkpoint/l16_ptk710_ftk710_ftk400_f8_res224.pth",
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
    ) -> None:
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.backbone = backbone or "vit_large_p16"
        self.weights_dataset = weights_dataset or "kinetics400"
        if grad_forward_chunk_size is not None:
            self.grad_forward_chunk_size = int(grad_forward_chunk_size)
        weight_spec = self.get_weight_spec(self.backbone, self.weights_dataset)
        self.num_classes = int(num_classes if num_classes is not None else weight_spec.get("num_classes", 400))
        self.loading_configs, self.preprocessing_configs = self._build_stage_config_dicts()
        self._active_stage: PipelineStage = "test"  # type: ignore[assignment]
        self._bind_stage_functions("test")
        self.model = self._build_model()
        self.checkpoint_path = checkpoint_path or self._default_checkpoint_path(self.backbone, self.weights_dataset)
        should_load = bool(self.checkpoint_path) if load_weights is None else bool(load_weights)
        if should_load:
            if not self.checkpoint_path:
                raise ValueError("checkpoint_path is required when load_weights=True")
            self._load_checkpoint(self.checkpoint_path)
        self.model.eval().to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

    def _build_model(self) -> torch.nn.Module:
        factory = import_attr("models", "umt_vit_large_patch16_224", _umt_root(), purge_modules=("models",))
        return factory(
            num_classes=self.num_classes,
            all_frames=8,
            tubelet_size=1,
            drop_path_rate=0.2,
            attn_drop_rate=0.0,
            use_checkpoint=False,
            use_mean_pooling=True,
            init_scale=0.001,
        )

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
            "input_range": 255.0,
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
        }
        loading = {
            "train": {**common_loading, "num_clips": 4},
            "val": {**common_loading, "num_clips": 4},
            "test": {**common_loading, "num_clips": 4},
            "attack": {**common_loading, "num_clips": 4},
        }
        preprocessing = {
            "train": {**common_pre, "spatial_strategy": "three_crop"},
            "val": {**common_pre, "spatial_strategy": "three_crop"},
            "test": {**common_pre, "spatial_strategy": "three_crop"},
            "attack": {**common_pre, "spatial_strategy": "center_crop"},
        }
        return loading, preprocessing

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict, strict=True)


MODEL_CLASS = UMTClassifier


def create(**kwargs) -> UMTClassifier:
    return UMTClassifier(**kwargs)

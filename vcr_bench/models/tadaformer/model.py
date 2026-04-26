from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

import torch

from ..base import BaseVideoClassifier
from ..legacy_imports import isolated_import_paths
from ..pipeline_config import PipelineStage


def _tada_root() -> Path:
    return Path(__file__).resolve().parent / "vendor"


class TAdaFormerClassifier(BaseVideoClassifier):
    model_name = "tadaformer"
    raw_input_format = "NTHWC"
    preprocessed_format = "NCTHW"
    MODEL_CAPABILITIES: dict[str, Any] = {
        "backbones": {
            "large_14": {
                "display_name": "TAdaFormer L/14",
                "datasets": {
                    "kinetics400": {
                        "num_classes": 400,
                        "checkpoint_filename": "l14_clip+k710_k400_64f_89.9.pyth",
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
        self.backbone = backbone or "large_14"
        self.weights_dataset = weights_dataset or "kinetics400"
        if grad_forward_chunk_size is not None:
            self.grad_forward_chunk_size = int(grad_forward_chunk_size)
        weight_spec = self.get_weight_spec(self.backbone, self.weights_dataset)
        self.num_classes = int(num_classes if num_classes is not None else weight_spec.get("num_classes", 400))
        self.loading_configs, self.preprocessing_configs = self._build_stage_config_dicts()
        self._active_stage: PipelineStage = "test"  # type: ignore[assignment]
        self._bind_stage_functions("test")
        self.model = self._build_model()
        should_load = bool(self._default_checkpoint_path(self.backbone, self.weights_dataset) or checkpoint_path) if load_weights is None else bool(load_weights)
        if should_load:
            self.checkpoint_path = self.resolve_checkpoint_path(
                self.backbone,
                self.weights_dataset,
                checkpoint_path,
                auto_download=auto_download,
            )
            if not self.checkpoint_path:
                raise ValueError("checkpoint_path is required when load_weights=True")
            self._load_checkpoint(self.checkpoint_path)
        else:
            self.checkpoint_path = checkpoint_path
        self.model.eval().to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

    def _build_model(self) -> torch.nn.Module:
        with isolated_import_paths(_tada_root(), purge_modules=("tadaconv",)):
            argv_backup = list(sys.argv)
            try:
                sys.argv = [argv_backup[0]]
                from tadaconv.models.base.builder import build_model
                from tadaconv.utils.config import Config

                cfg = Config(load=True)
                cfg.NUM_GPUS = 0 if self.device.type == "cpu" else max(int(getattr(cfg, "NUM_GPUS", 1)), 1)
                model, _ = build_model(cfg)
            finally:
                sys.argv = argv_backup
        return model

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
            "mean": (0.48145466, 0.4578275, 0.40821073),
            "std": (0.26862954, 0.26130258, 0.27577711),
        }
        loading = {
            "train": {**common_loading, "num_clips": 8},
            "val": {**common_loading, "num_clips": 8},
            "test": {**common_loading, "num_clips": 8},
            "attack": {**common_loading, "num_clips": 8},
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
        state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict, strict=True)


MODEL_CLASS = TAdaFormerClassifier


def create(**kwargs) -> TAdaFormerClassifier:
    return TAdaFormerClassifier(**kwargs)

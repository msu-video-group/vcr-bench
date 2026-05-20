from __future__ import annotations

from typing import Any

import torch

from ..base import BaseVideoClassifier
from ..pipeline_config import PipelineStage
from ._impl import internvideo2_1B_patch14_224


class InternVideo2Classifier(BaseVideoClassifier):
    model_name = "internvideo2"
    raw_input_format = "NTHWC"
    preprocessed_format = "NCTHW"
    MODEL_CAPABILITIES: dict[str, Any] = {
        "backbones": {
            "vit_1b_p14": {
                "display_name": "InternVideo2 1B",
                "datasets": {
                    "kinetics400": {
                        "num_classes": 400,
                        "checkpoint_url": "https://huggingface.co/maxv65/vcr-bench/resolve/main/internvideo2_1B_ft_k710_ft_k400_f16.pth",
                        "checkpoint_filename": "internvideo2_1B_ft_k710_ft_k400_f16.pth",
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
        self.backbone = backbone or "vit_1b_p14"
        self.weights_dataset = weights_dataset or "kinetics400"
        if grad_forward_chunk_size is not None:
            self.grad_forward_chunk_size = int(grad_forward_chunk_size)
        weight_spec = self.get_weight_spec(self.backbone, self.weights_dataset)
        self.num_classes = int(num_classes if num_classes is not None else weight_spec.get("num_classes", 400))
        self.loading_configs, self.preprocessing_configs = self._build_stage_config_dicts()
        self._active_stage: PipelineStage = "test"  # type: ignore[assignment]
        self._bind_stage_functions("test")
        self.model = self._build_model()
        self.checkpoint_path = self.resolve_checkpoint_path(
            self.backbone,
            self.weights_dataset,
            checkpoint_path,
            auto_download=auto_download,
        )
        should_load = bool(self.checkpoint_path) if load_weights is None else bool(load_weights)
        if should_load:
            if not self.checkpoint_path:
                raise ValueError("checkpoint_path is required when load_weights=True")
            self._load_checkpoint(self.checkpoint_path)
        self.model.eval().to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

    def _build_model(self) -> torch.nn.Module:
        return internvideo2_1B_patch14_224(
            num_classes=self.num_classes,
            use_flash_attn=False,
            use_fused_rmsnorm=False,
            use_fused_mlp=False,
            num_frames=16,
        )

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
            "resize_short": 224,
            "crop_size": 224,
            "input_range": 255.0,
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
        }
        loading = {
            "train": {**common_loading, "num_clips": 1},
            "val": {**common_loading, "num_clips": 1},
            "test": {**common_loading, "num_clips": 1},
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
        missing, unexpected = self.load_converted_checkpoint(
            checkpoint_path,
            strict=False,
            root_keys=("module", "state_dict", "model"),
            required_keys=("head.weight", "head.bias"),
        )
        if unexpected and len(unexpected) == len(missing):
            raise RuntimeError("InternVideo2 checkpoint keys did not match the model")


MODEL_CLASS = InternVideo2Classifier


def create(**kwargs) -> InternVideo2Classifier:
    return InternVideo2Classifier(**kwargs)

from __future__ import annotations

from typing import Any

import torch

from ..base import BaseVideoClassifier
from ..legacy_imports import import_attr
from ..pipeline_config import PipelineStage


def _ila_vendor_root():
    from pathlib import Path

    return Path(__file__).resolve().parent / "vendor"


def _load_labels(num_classes: int) -> list[str]:
    return [f"class_{idx}" for idx in range(num_classes)]


class ILAClassifier(BaseVideoClassifier):
    model_name = "ila"
    raw_input_format = "NTHWC"
    preprocessed_format = "NTCHW"
    MODEL_CAPABILITIES: dict[str, Any] = {
        "backbones": {
            "vit_b16": {
                "display_name": "X-CLIP ViT-B/16",
                "datasets": {
                    "kinetics400": {
                        "num_classes": 400,
                        "checkpoint_filename": "k400_B_16_8_rel.pth",
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
        self.backbone = backbone or "vit_b16"
        self.weights_dataset = weights_dataset or "kinetics400"
        if grad_forward_chunk_size is not None:
            self.grad_forward_chunk_size = int(grad_forward_chunk_size)
        weight_spec = self.get_weight_spec(self.backbone, self.weights_dataset)
        self.num_classes = int(num_classes if num_classes is not None else weight_spec.get("num_classes", 400))
        self.loading_configs, self.preprocessing_configs = self._build_stage_config_dicts()
        self._active_stage: PipelineStage = "test"  # type: ignore[assignment]
        self._bind_stage_functions("test")
        should_load = bool(self._default_checkpoint_path(self.backbone, self.weights_dataset) or checkpoint_path) if load_weights is None else bool(load_weights)
        if should_load:
            self.checkpoint_path = self.resolve_checkpoint_path(
                self.backbone,
                self.weights_dataset,
                checkpoint_path,
                auto_download=auto_download,
            )
        else:
            self.checkpoint_path = checkpoint_path
        self.text_tokens = self._build_text_tokens().to(self.device)
        self.model = self._build_model(load_weights=bool(load_weights) if load_weights is not None else False)
        if should_load:
            if not self.checkpoint_path:
                raise ValueError("checkpoint_path is required when load_weights=True")
            self._load_checkpoint(self.checkpoint_path)
        self.model.eval().to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

    def _build_text_tokens(self) -> torch.Tensor:
        tokenize = import_attr("clip", "tokenize", _ila_vendor_root(), purge_modules=("clip",))
        labels = _load_labels(self.num_classes)
        return tokenize(labels)

    def _build_model(self, *, load_weights: bool) -> torch.nn.Module:
        if load_weights and self.checkpoint_path:
            return self._load_runtime_model(self.checkpoint_path)
        xclip_cls = import_attr("models.xclip", "XCLIP", _ila_vendor_root(), purge_modules=("models", "clip"))
        return xclip_cls(
            512,
            224,
            12,
            768,
            16,
            77,
            49408,
            512,
            8,
            12,
            T=8,
            text_embeddings=self.text_tokens,
        )

    def _load_runtime_model(self, checkpoint_path: str) -> torch.nn.Module:
        xclip_load = import_attr("models.xclip", "load", _ila_vendor_root(), purge_modules=("models", "clip"))
        return xclip_load(
            model_path=checkpoint_path,
            name="ViT-B/16",
            device=self.device,
            T=8,
            text_features=self.text_tokens,
        )

    def _build_stage_config_dicts(self) -> tuple[dict[str, dict], dict[str, dict]]:
        common_loading = {
            "raw_input_format": "NTHWC",
            "sampled_format": "NTHWC",
            "clip_len": 1,
            "frame_interval": 1,
            "full_videos": False,
        }
        common_pre = {
            "preprocessed_format": "NTCHW",
            "resize_short": 224,
            "crop_size": 224,
            "input_range": 1.0,
            "mean": (123.675, 116.28, 103.53),
            "std": (58.395, 57.12, 57.375),
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
        self.model = self._load_runtime_model(checkpoint_path)


MODEL_CLASS = ILAClassifier


def create(**kwargs) -> ILAClassifier:
    return ILAClassifier(**kwargs)

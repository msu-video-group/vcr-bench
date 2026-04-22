from __future__ import annotations

from typing import Any

import torch

from ..base import BaseVideoClassifier
from ..pipeline_config import PipelineStage
from ..vendor_mmaction import Recognizer2D, TANet, TSNHead


class TANETClassifier(BaseVideoClassifier):
    model_name = "tanet"
    raw_input_format = "NTHWC"
    preprocessed_format = "NTCHW"
    MODEL_CAPABILITIES: dict[str, Any] = {
        "backbones": {
            "r50": {
                "display_name": "ResNet-50 TANet",
                "datasets": {
                    "kinetics400": {
                        "num_classes": 400,
                        "checkpoint_url": (
                            "https://download.openmmlab.com/mmaction/v1.0/recognition/tanet/"
                            "tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-100e_kinetics400-rgb/"
                            "tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-100e_kinetics400-rgb_20220919-a34346bc.pth"
                        ),
                        "checkpoint_filename": "tanet_r50_kinetics400_mmaction.pth",
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

        self.model = Recognizer2D(
            backbone=TANet(
                depth=50,
                num_segments=8,
                pretrained="torchvision://resnet50",
                norm_eval=False,
            ),
            cls_head=TSNHead(
                num_classes=self.num_classes,
                in_channels=2048,
                num_segs=8,
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
            "clip_len": 1,
            "frame_interval": 1,
            "full_videos": False,
            "temporal_strategy": "dense",
            "sample_range": 64,
            "num_sample_positions": 10,
        }
        common_pre = {
            "preprocessed_format": "NTCHW",
            "resize_short": 256,
            "crop_size": 224,
            "input_range": 1.0,
            "mean": (123.675, 116.28, 103.5),
            "std": (58.395, 57.12, 57.375),
        }
        loading = {
            "train": {**common_loading, "num_clips": 8},
            "val": {**common_loading, "num_clips": 8},
            "test": {**common_loading, "num_clips": 8},
            "attack": {**common_loading, "num_clips": 8},
        }
        preprocessing = {
            "train": {**common_pre, "spatial_strategy": "train_random_crop"},
            "val": {**common_pre, "spatial_strategy": "center_crop"},
            "test": {**common_pre, "spatial_strategy": "three_crop", "crop_size": 256},
            "attack": {**common_pre, "spatial_strategy": "center_crop"},
        }
        return loading, preprocessing

    def forward_preprocessed_with_grad(self, x: torch.Tensor, *, enable_grad: bool = True) -> torch.Tensor:
        # TANet uses a 2D backbone (ResNet2d) that expects [N, C, H, W].
        # The base class passes 5D [N, T, C, H, W] directly; flatten T into N first.
        if x.ndim == 5:
            n, t, c, h, w = x.shape
            x = x.reshape(n * t, c, h, w)
        device = getattr(self, "device", x.device)
        x = x.to(device, non_blocking=True)
        if enable_grad:
            chunk_size = getattr(self, "grad_forward_chunk_size", None)
            if isinstance(chunk_size, int) and chunk_size > 0 and x.shape[0] > chunk_size:
                logits_chunks = []
                for start in range(0, x.shape[0], chunk_size):
                    cur = self.model(x[start : start + chunk_size])
                    if isinstance(cur, tuple):
                        cur = cur[0]
                    if cur.ndim == 1:
                        cur = cur.unsqueeze(0)
                    logits_chunks.append(cur)
                logits = torch.cat(logits_chunks, dim=0)
            else:
                logits = self.model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
        else:
            with torch.no_grad():
                logits = self.model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        return logits

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
            key_replacements=(
                ("model.backbone.", "backbone."),
                ("model.cls_head.", "cls_head."),
            ),
            required_keys=("cls_head.fc_cls.weight", "cls_head.fc_cls.bias"),
        )


MODEL_CLASS = TANETClassifier


def create(**kwargs) -> TANETClassifier:
    return TANETClassifier(**kwargs)

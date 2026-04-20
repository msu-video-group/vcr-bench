from __future__ import annotations

from typing import Any

import pickle
import torch

from ..base import BaseVideoClassifier
from ..pipeline_config import PipelineStage
from ..vendor_mmaction import I3DHead, OnePeaceViT, Recognizer3D


class OnePeaceClassifier(BaseVideoClassifier):
    model_name = "onepeace"
    raw_input_format = "NTHWC"
    preprocessed_format = "NCTHW"
    MODEL_CAPABILITIES: dict[str, Any] = {
        "backbones": {
            "vit_l40": {
                "display_name": "ONE-PEACE Video ViT-L/40",
                "datasets": {
                    "kinetics400": {
                        "num_classes": 400,
                        "checkpoint_relpath": "Classifiers/ONE-PEACE/checkpoint/onepeace_video_k400.pth",
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
        self.backbone = backbone or "vit_l40"
        self.weights_dataset = weights_dataset or "kinetics400"
        if grad_forward_chunk_size is not None:
            self.grad_forward_chunk_size = int(grad_forward_chunk_size)
        weight_spec = self.get_weight_spec(self.backbone, self.weights_dataset)
        self.num_classes = int(num_classes if num_classes is not None else weight_spec.get("num_classes", 400))
        self.loading_configs, self.preprocessing_configs = self._build_stage_config_dicts()
        self._active_stage: PipelineStage = "test"  # type: ignore[assignment]
        self._bind_stage_functions("test")
        self.model = Recognizer3D(
            backbone=OnePeaceViT(
                attention_heads=24,
                bucket_size=16,
                num_frames=16,
                dropout=0.0,
                drop_path_rate=0.4,
                embed_dim=1536,
                ffn_embed_dim=6144,
                layers=40,
                rp_bias=False,
                shared_rp_bias=True,
                use_checkpoint=False,
            ),
            cls_head=I3DHead(
                in_channels=1536,
                num_classes=self.num_classes,
                spatial_type="avg",
                dropout_ratio=0.5,
            ),
        )
        self.checkpoint_path = checkpoint_path or self._default_checkpoint_path(self.backbone, self.weights_dataset)
        should_load = bool(self.checkpoint_path) if load_weights is None else bool(load_weights)
        if should_load:
            if not self.checkpoint_path:
                raise ValueError("checkpoint_path is required when load_weights=True")
            self._load_checkpoint(self.checkpoint_path)
        self.model.eval().to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

    def _build_stage_config_dicts(self) -> tuple[dict[str, dict], dict[str, dict]]:
        common_loading = {
            "raw_input_format": "NTHWC",
            "sampled_format": "NTHWC",
            "clip_len": 16,
            "frame_interval": 8,
            "full_videos": False,
        }
        common_pre = {
            "preprocessed_format": "NCTHW",
            "resize_short": 256,
            "crop_size": 256,
            "input_range": 1.0,
            "mean": (122.771, 116.746, 104.094),
            "std": (68.5, 66.632, 70.323),
        }
        loading = {
            "train": {**common_loading, "num_clips": 3},
            "val": {**common_loading, "num_clips": 3},
            "test": {**common_loading, "num_clips": 2},
            "attack": {**common_loading, "num_clips": 1},
        }
        preprocessing = {
            "train": {**common_pre, "spatial_strategy": "center_crop"},
            "val": {**common_pre, "spatial_strategy": "three_crop"},
            "test": {**common_pre, "spatial_strategy": "center_crop"},
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
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception:
            with open(checkpoint_path, "rb") as f:
                checkpoint = pickle.load(f, encoding="latin1")
        state_dict = self.extract_tensor_state_dict(
            checkpoint,
            root_keys=("state_dict", "model_state_dict", "model", "module"),
            strip_prefixes=("module.",),
            ignore_prefixes=("data_preprocessor.",),
            key_replacements=(
                ("model.backbone.", "backbone."),
                ("model.cls_head.", "cls_head."),
                (".S_Adapter.D_fc1.", ".s_adapter.fc1."),
                (".S_Adapter.D_fc2.", ".s_adapter.fc2."),
                (".T_Adapter.D_fc1.", ".t_adapter.fc1."),
                (".T_Adapter.D_fc2.", ".t_adapter.fc2."),
                (".T_Adapter_in.D_fc1.", ".t_adapter_in.fc1."),
                (".T_Adapter_in.D_fc2.", ".t_adapter_in.fc2."),
                (".MLP_Adapter.D_fc1.", ".mlp_adapter.fc1."),
                (".MLP_Adapter.D_fc2.", ".mlp_adapter.fc2."),
            ),
        )
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        missing = [k for k in missing if not k.endswith("num_batches_tracked")]
        required = {"cls_head.fc_cls.weight", "cls_head.fc_cls.bias"}
        if required.intersection(missing):
            raise RuntimeError(f"OnePeace checkpoint load failed, missing keys: {sorted(required.intersection(missing))}")


MODEL_CLASS = OnePeaceClassifier


def create(**kwargs) -> OnePeaceClassifier:
    return OnePeaceClassifier(**kwargs)

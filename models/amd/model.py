from __future__ import annotations

from pathlib import Path

import torch

from .vendor import vit_base_patch16_224
from ..base import BaseVideoClassifier
from ..pipeline_config import PipelineStage


class AmdK400Classifier(BaseVideoClassifier):
    model_name = "amd"
    raw_input_format = "NTHWC"
    preprocessed_format = "NCTHW"
    MODEL_CAPABILITIES = {
        "backbones": {
            "vitb": {
                "display_name": "ViT-B/16",
                "datasets": {
                    "kinetics400": {
                        "num_classes": 400,
                        "checkpoint_relpath": "Classifiers/AMD/data/vitb_k400_finetune_90e.pth",
                    },
                },
            },
        },
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
        self.backbone = backbone or "vitb"
        self.weights_dataset = weights_dataset or "kinetics400"
        if grad_forward_chunk_size is not None:
            self.grad_forward_chunk_size = int(grad_forward_chunk_size)
        else:
            # AMD test-time preprocessing expands 3 temporal clips into 9 views via three-crop.
            # Keeping inference chunked by 3 matches the legacy benchmark's per-clip evaluation
            # and avoids VRAM spikes when a heavy defence (e.g. VideoPure) is resident on GPU.
            self.grad_forward_chunk_size = 3
        weight_spec = self.get_weight_spec(self.backbone, self.weights_dataset)
        self.num_classes = int(num_classes if num_classes is not None else weight_spec.get("num_classes", 400))
        self.loading_configs, self.preprocessing_configs = self._build_stage_config_dicts()
        self._active_stage: PipelineStage = "test"
        self._bind_stage_functions("test")

        self.model = self._create_backbone_model(self.backbone, num_classes=self.num_classes)
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
        self.model.eval()
        self.model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False

    @staticmethod
    def _create_backbone_model(backbone: str, num_classes: int):
        key = str(backbone).lower()
        if key == "vitb":
            return vit_base_patch16_224(num_classes=num_classes)
        raise ValueError(f"Unsupported AMD backbone: {backbone}")

    @classmethod
    def _default_checkpoint_path(cls, backbone: str, weights_dataset: str) -> str | None:
        spec = cls.get_weight_spec(backbone, weights_dataset)
        rel = spec.get("checkpoint_relpath")
        if rel:
            repo_root = Path(__file__).resolve().parents[2]
            candidate = repo_root / Path(rel)
            if candidate.exists():
                return str(candidate)
        return super()._default_checkpoint_path(backbone, weights_dataset)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"AMD checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = checkpoint.get("module", checkpoint)
        self.model.load_state_dict(state_dict, strict=True)

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
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
        }
        loading = {
            "train": {**common_loading, "num_clips": 1},
            "val": {**common_loading, "num_clips": 1},
            "test": {**common_loading, "num_clips": 3},
            "attack": {**common_loading, "num_clips": 3},
        }
        preprocessing = {
            "train": {**common_pre, "spatial_strategy": "train_random_crop"},
            "val": {**common_pre, "spatial_strategy": "center_crop"},
            "test": {**common_pre, "spatial_strategy": "three_crop"},
            "attack": {**common_pre, "spatial_strategy": "center_crop"},
        }
        return loading, preprocessing

def create(**kwargs) -> AmdK400Classifier:
    return AmdK400Classifier(**kwargs)

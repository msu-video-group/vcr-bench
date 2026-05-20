from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download

from ..base import BaseVideoClassifier
from ..pipeline_config import PipelineStage
from ._impl import vit_base_patch16_224


class InternVideoClassifier(BaseVideoClassifier):
    model_name = "internvideo"
    raw_input_format = "NTHWC"
    preprocessed_format = "NCTHW"
    MODEL_CAPABILITIES: dict[str, Any] = {
        "backbones": {
            "vit_base_p16": {
                "display_name": "InternVideo ViT-B/16",
                "datasets": {
                    "kinetics400": {
                        "num_classes": 400,
                        "checkpoint_url": "https://huggingface.co/vcr-bench/checkpoints-internvideo/resolve/main/internvideo_vit_base_p16_kinetics400.bin",
                        "checkpoint_filename": "internvideo_vit_base_p16_kinetics400.bin",
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
        self.backbone = backbone or "vit_base_p16"
        self.weights_dataset = weights_dataset or "kinetics400"
        if grad_forward_chunk_size is not None:
            self.grad_forward_chunk_size = int(grad_forward_chunk_size)
        weight_spec = self.get_weight_spec(self.backbone, self.weights_dataset)
        self.num_classes = int(num_classes if num_classes is not None else weight_spec.get("num_classes", 400))
        self.loading_configs, self.preprocessing_configs = self._build_stage_config_dicts()
        self._active_stage: PipelineStage = "test"  # type: ignore[assignment]
        self._bind_stage_functions("test")
        self.model = self._build_model()
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
        for param in self.model.parameters():
            param.requires_grad = False

    def _build_model(self) -> torch.nn.Module:
        return vit_base_patch16_224(
            num_classes=self.num_classes,
            all_frames=16,
            tubelet_size=2,
            use_mean_pooling=True,
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
        state_dict = self._read_checkpoint_state_dict(checkpoint_path)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        missing = [key for key in missing if not key.endswith("num_batches_tracked")]
        if "head.weight" in missing or "head.bias" in missing:
            repaired_path = self._redownload_classifier_checkpoint(checkpoint_path)
            if repaired_path is not None:
                state_dict = self._read_checkpoint_state_dict(repaired_path)
                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                missing = [key for key in missing if not key.endswith("num_batches_tracked")]
            if "head.weight" in missing or "head.bias" in missing:
                raise RuntimeError(
                    "InternVideo checkpoint is missing classifier weights. "
                    "Remove the MAE pretrain file or pass a Kinetics-400 fine-tuned checkpoint."
                )
        if unexpected and len(unexpected) == len(state_dict):
            raise RuntimeError("InternVideo checkpoint keys did not match the model")

    def _read_checkpoint_state_dict(self, checkpoint_path: str) -> dict[str, torch.Tensor]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        return self.extract_tensor_state_dict(
            checkpoint,
            root_keys=("state_dict", "model", "module"),
            strip_prefixes=("module.", "backbone."),
        )

    def _redownload_classifier_checkpoint(self, checkpoint_path: str) -> str | None:
        path = Path(checkpoint_path)
        if path.name != "internvideo_vit_base_p16_kinetics400.bin":
            return None
        try:
            return str(
                hf_hub_download(
                    repo_id="vcr-bench/checkpoints-internvideo",
                    filename=path.name,
                    revision="main",
                    repo_type="model",
                    local_dir=str(path.parent),
                    force_download=True,
                )
            )
        except Exception:
            return None


MODEL_CLASS = InternVideoClassifier


def create(**kwargs) -> InternVideoClassifier:
    return InternVideoClassifier(**kwargs)

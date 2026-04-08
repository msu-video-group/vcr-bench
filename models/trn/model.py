from __future__ import annotations

from typing import Any

from ..tsn.model import TSNClassifier


class TRNClassifier(TSNClassifier):
    model_name = "trn"
    MODEL_CAPABILITIES: dict[str, Any] = {
        "backbones": {
            "r50": {
                "display_name": "TRN R50",
                "datasets": {
                    "sthv2": {
                        "num_classes": 174,
                        "checkpoint_url": (
                            "https://download.openmmlab.com/mmaction/v1.0/recognition/trn/"
                            "trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb/"
                            "trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb_20220815-e01617db.pth"
                        ),
                        "checkpoint_filename": "trn_r50_sthv2_mmaction.pth",
                    }
                },
            }
        }
    }

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("backbone", "r50")
        super().__init__(*args, weights_dataset="sthv2", num_classes=174, **kwargs)

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
            "resize_short": 256,
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
            "train": {**common_pre, "spatial_strategy": "center_crop"},
            "val": {**common_pre, "spatial_strategy": "center_crop"},
            "test": {**common_pre, "crop_size": 256, "spatial_strategy": "three_crop"},
            "attack": {**common_pre, "spatial_strategy": "center_crop"},
        }
        return loading, preprocessing


MODEL_CLASS = TRNClassifier


def create(**kwargs) -> TRNClassifier:
    return TRNClassifier(**kwargs)

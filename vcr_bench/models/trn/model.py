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
                        "checkpoint_url": "https://huggingface.co/maxv65/vcr-bench/resolve/main/trn_r50_sthv2.pth",
                        "checkpoint_filename": "trn_r50_sthv2.pth",
                    }
                },
            }
        }
    }

    def __init__(self, *args, **kwargs) -> None:
        backbone = kwargs.pop("backbone", None)
        weights_dataset = kwargs.pop("weights_dataset", None)
        num_classes = kwargs.pop("num_classes", None)
        super().__init__(
            *args,
            backbone="r50" if backbone is None else backbone,
            weights_dataset="sthv2" if weights_dataset is None else weights_dataset,
            num_classes=174 if num_classes is None else num_classes,
            **kwargs,
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

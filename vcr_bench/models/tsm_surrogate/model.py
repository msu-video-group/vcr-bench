from __future__ import annotations

from typing import Any

from ..tsm.model import TSMClassifier


class TSMSurrogateClassifier(TSMClassifier):
    model_name = "tsm-surrogate"
    MODEL_CAPABILITIES: dict[str, Any] = {
        "backbones": {
            "r50": {
                "display_name": "TSM R50 Surrogate",
                "datasets": {
                    "kinetics400": {
                        "num_classes": 400,
                        "checkpoint_url": (
                            "https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/"
                            "tsm_imagenet-pretrained-r50_8xb16-1x1x8-100e_kinetics400-rgb/"
                            "tsm_imagenet-pretrained-r50_8xb16-1x1x8-100e_kinetics400-rgb_20220831-a6db1e5d.pth"
                        ),
                        "checkpoint_filename": "tsm_surrogate_r50_kinetics400_mmaction.pth",
                    }
                },
            }
        }
    }

    def _num_segments(self) -> int:
        return 8


MODEL_CLASS = TSMSurrogateClassifier


def create(**kwargs) -> TSMSurrogateClassifier:
    return TSMSurrogateClassifier(**kwargs)

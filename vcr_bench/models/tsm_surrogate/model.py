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
                        "checkpoint_url": "https://huggingface.co/maxv65/vcr-bench/resolve/main/tsm_surrogate_r50_kinetics400.pth",
                        "checkpoint_filename": "tsm_surrogate_r50_kinetics400.pth",
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

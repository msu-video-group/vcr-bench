from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class VideoSampleRef:
    id: str
    path: str
    label: int | None
    label_name: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionBundle:
    logits: torch.Tensor
    probs: torch.Tensor
    pred_label: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedDatasetItem:
    sample: VideoSampleRef
    tensor: torch.Tensor
    input_format: str
    sampled: bool = True
    preprocessed: bool = False
    full_tensor: torch.Tensor | None = None
    sampled_frame_ids: torch.Tensor | None = None

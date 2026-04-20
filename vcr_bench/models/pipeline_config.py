from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PipelineStage = Literal["train", "val", "test"]


@dataclass(frozen=True)
class VideoLoadingConfig:
    raw_input_format: str = "NTHWC"
    sampled_format: str = "NTHWC"
    clip_len: int = 16
    num_clips: int = 1
    frame_interval: int = 1
    full_videos: bool = False
    temporal_strategy: str = "uniform"
    sample_range: int = 64
    num_sample_positions: int = 10


@dataclass(frozen=True)
class VideoPreprocessingConfig:
    preprocessed_format: str = "NCTHW"
    resize_short: int = 224
    crop_size: int = 224
    input_range: float = 255.0
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    spatial_strategy: str = "three_crop"  # e.g. train_random_crop, center_crop, three_crop


@dataclass(frozen=True)
class ModelDataPipelineConfig:
    stage: PipelineStage
    loading: VideoLoadingConfig
    preprocessing: VideoPreprocessingConfig

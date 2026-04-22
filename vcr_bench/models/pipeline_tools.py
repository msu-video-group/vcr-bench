from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .pipeline_config import ModelDataPipelineConfig, VideoLoadingConfig, VideoPreprocessingConfig
from vcr_bench.utils.preprocessing import (
    center_crop,
    ensure_nthwc,
    normalize_channels,
    resize_short_side,
    temporal_sample_nthwc,
    ten_crop,
    three_crop,
    to_format,
)


def make_pipeline_config(stage: str, loading: dict, preprocessing: dict) -> ModelDataPipelineConfig:
    return ModelDataPipelineConfig(
        stage=stage,  # type: ignore[arg-type]
        loading=VideoLoadingConfig(**loading),
        preprocessing=VideoPreprocessingConfig(**preprocessing),
    )


@dataclass
class RuntimePipeline:
    config: ModelDataPipelineConfig
    load_video_fn: Callable[[str], torch.Tensor]
    sample_video_fn: Callable[[torch.Tensor, str], tuple[torch.Tensor, torch.Tensor] | torch.Tensor]
    preprocess_sampled_fn: Callable[[torch.Tensor, str], torch.Tensor]

    def __post_init__(self) -> None:
        self.raw_input_format = self.config.loading.raw_input_format
        self.sampled_format = self.config.loading.sampled_format
        self.preprocessed_format = self.config.preprocessing.preprocessed_format
        self.clip_len = self.config.loading.clip_len
        self.num_clips = self.config.loading.num_clips
        self.frame_interval = self.config.loading.frame_interval
        self.full_videos = self.config.loading.full_videos

    def load_video(self, path: str) -> torch.Tensor:
        return self.load_video_fn(path)

    def sample_video(self, x: torch.Tensor, input_format: str = "NTHWC"):
        return self.sample_video_fn(x, input_format)

    def preprocess_sampled(self, x: torch.Tensor, input_format: str = "NTHWC") -> torch.Tensor:
        return self.preprocess_sampled_fn(x, input_format)


def make_runtime_pipeline(
    config: ModelDataPipelineConfig,
    *,
    load_video_fn: Callable[[str], torch.Tensor],
    sample_video_fn: Callable[[torch.Tensor, str], tuple[torch.Tensor, torch.Tensor] | torch.Tensor],
    preprocess_sampled_fn: Callable[[torch.Tensor, str], torch.Tensor],
) -> RuntimePipeline:
    return RuntimePipeline(
        config=config,
        load_video_fn=load_video_fn,
        sample_video_fn=sample_video_fn,
        preprocess_sampled_fn=preprocess_sampled_fn,
    )


def preprocess_sampled_nthwc_from_config(
    x: torch.Tensor,
    *,
    input_format: str,
    preprocessing_cfg: VideoPreprocessingConfig,
) -> torch.Tensor:
    x = ensure_nthwc(x, input_format)
    was_uint8 = x.dtype == torch.uint8
    x = x.float()
    if preprocessing_cfg.input_range <= 0:
        raise ValueError(f"input_range must be positive, got {preprocessing_cfg.input_range}")
    if was_uint8 or preprocessing_cfg.input_range > 1.0:
        x = x / preprocessing_cfg.input_range
    x = resize_short_side(x, preprocessing_cfg.resize_short, input_format="NTHWC")
    if preprocessing_cfg.spatial_strategy == "three_crop":
        x = three_crop(x, (preprocessing_cfg.crop_size, preprocessing_cfg.crop_size), input_format="NTHWC")
    elif preprocessing_cfg.spatial_strategy == "ten_crop":
        x = ten_crop(x, (preprocessing_cfg.crop_size, preprocessing_cfg.crop_size), input_format="NTHWC")
    elif preprocessing_cfg.spatial_strategy in {"center_crop", "train_random_crop"}:
        # Placeholder keeps train/val batching deterministic until true train aug pipeline is added.
        x = center_crop(x, (preprocessing_cfg.crop_size, preprocessing_cfg.crop_size), input_format="NTHWC")
    else:
        raise ValueError(f"Unsupported spatial_strategy: {preprocessing_cfg.spatial_strategy}")
    x = normalize_channels(x, list(preprocessing_cfg.mean), list(preprocessing_cfg.std), input_format="NTHWC")
    x = to_format(x, "NTHWC", preprocessing_cfg.preprocessed_format)
    _validate_preprocessed_tensor_for_config(x, preprocessing_cfg.preprocessed_format)
    return x.contiguous()


def _validate_preprocessed_tensor_for_config(x: torch.Tensor, preprocessed_format: str) -> None:
    if x.ndim != 5:
        raise ValueError(f"Expected 5D preprocessed tensor, got {tuple(x.shape)}")
    fmt = preprocessed_format.upper()
    if fmt == "NCTHW":
        if x.shape[1] != 3:
            raise ValueError(f"Expected NCTHW tensor with channel dim at 1, got shape {tuple(x.shape)}")
    elif fmt == "NTCHW":
        if x.shape[2] != 3:
            raise ValueError(f"Expected NTCHW tensor with channel dim at 2, got shape {tuple(x.shape)}")


def sample_video_nthwc_from_loading_config(
    x: torch.Tensor,
    *,
    input_format: str,
    loading_cfg: VideoLoadingConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = ensure_nthwc(x, input_format)
    sampled, frame_ids = temporal_sample_nthwc(
        x,
        clip_len=loading_cfg.clip_len,
        num_clips=loading_cfg.num_clips,
        frame_interval=loading_cfg.frame_interval,
        temporal_strategy=loading_cfg.temporal_strategy,
        sample_range=loading_cfg.sample_range,
        num_sample_positions=loading_cfg.num_sample_positions,
        return_indices=True,
    )
    return sampled.contiguous(), frame_ids.contiguous()


def bind_stage_runtime_functions(
    model_obj,
    stage: str,
    *,
    build_pipeline_config_fn: Callable[[str], ModelDataPipelineConfig],
    load_video_factory: Callable[[ModelDataPipelineConfig], Callable[[str], torch.Tensor]],
    sample_video_factory: Callable[[ModelDataPipelineConfig], Callable[[torch.Tensor, str], tuple[torch.Tensor, torch.Tensor] | torch.Tensor]],
    preprocess_sampled_factory: Callable[[ModelDataPipelineConfig], Callable[[torch.Tensor, str], torch.Tensor]],
) -> None:
    cfg = build_pipeline_config_fn(stage)
    setattr(model_obj, "_active_stage", stage)
    # Optional dict mirrors for debugging/CLI introspection.
    if hasattr(cfg, "loading"):
        setattr(model_obj, "loading_config", dict(getattr(cfg.loading, "__dict__", {})))
    if hasattr(cfg, "preprocessing"):
        setattr(model_obj, "preprocessing_config", dict(getattr(cfg.preprocessing, "__dict__", {})))
    setattr(model_obj, "load_video_fn", load_video_factory(cfg))
    setattr(model_obj, "sample_video_fn", sample_video_factory(cfg))
    setattr(model_obj, "preprocess_sampled_fn", preprocess_sampled_factory(cfg))

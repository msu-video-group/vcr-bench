from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset

from vg_videobench.types import LoadedDatasetItem, VideoSampleRef


class BaseVideoDataset(Dataset, ABC):
    _loading_pipeline: Any = None
    _instant_preprocessing: bool = False

    def configure_loading(self, data_pipeline: Any, instant_preprocessing: bool = False) -> "BaseVideoDataset":
        pipeline = copy.copy(data_pipeline)
        for attr in ("clip_len", "num_clips", "frame_interval", "full_videos"):
            if hasattr(self, attr) and hasattr(pipeline, attr):
                value = getattr(self, attr)
                if value is not None:
                    setattr(pipeline, attr, value)
        self._loading_pipeline = pipeline
        self._instant_preprocessing = bool(instant_preprocessing)
        return self

    def clear_loading(self) -> "BaseVideoDataset":
        self._loading_pipeline = None
        self._instant_preprocessing = False
        return self

    def _materialize_sample(self, sample: VideoSampleRef) -> VideoSampleRef | LoadedDatasetItem:
        if self._loading_pipeline is None:
            return sample
        full_tensor = self._loading_pipeline.load_video(sample.path)
        input_format = getattr(self._loading_pipeline, "raw_input_format", "NTHWC")
        sampled_frame_ids = None
        x = full_tensor
        if hasattr(self._loading_pipeline, "sample_video"):
            x, sampled_frame_ids = self._loading_pipeline.sample_video(full_tensor, input_format=input_format)
            input_format = getattr(self._loading_pipeline, "sampled_format", input_format)
        elif hasattr(self._loading_pipeline, "sample_raw"):
            x = self._loading_pipeline.sample_raw(full_tensor, input_format=input_format)
            input_format = getattr(self._loading_pipeline, "sampled_format", input_format)
        preprocessed = False
        if self._instant_preprocessing:
            if hasattr(self._loading_pipeline, "preprocess_sampled"):
                x = self._loading_pipeline.preprocess_sampled(x, input_format=input_format)
                input_format = getattr(self._loading_pipeline, "preprocessed_format", input_format)
                preprocessed = True
            elif hasattr(self._loading_pipeline, "preprocess"):
                x = self._loading_pipeline.preprocess(x, input_format=input_format)
                input_format = getattr(self._loading_pipeline, "preprocessed_format", input_format)
                preprocessed = True
        keep_full = bool(getattr(self._loading_pipeline, "full_videos", False))
        return LoadedDatasetItem(
            sample=sample,
            tensor=x,
            input_format=input_format,
            sampled=True,
            preprocessed=preprocessed,
            full_tensor=full_tensor if keep_full else None,
            sampled_frame_ids=sampled_frame_ids,
        )

    @abstractmethod
    def __getitem__(self, index: int) -> VideoSampleRef | LoadedDatasetItem:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def class_names(self) -> list[str]:
        raise NotImplementedError

    def num_classes(self) -> int:
        return len(self.class_names())

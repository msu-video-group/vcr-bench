from __future__ import annotations

import copy
from typing import Any

import torch
from torch.utils.data import Dataset

from vcr_bench.types import LoadedDatasetItem, VideoSampleRef


class ModelLoadedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, data_pipeline: Any, instant_preprocessing: bool = False) -> None:
        self.base_dataset = base_dataset
        self.data_pipeline = self._apply_dataset_loading_config(data_pipeline)
        self.instant_preprocessing = bool(instant_preprocessing)

    def _apply_dataset_loading_config(self, data_pipeline: Any) -> Any:
        pipeline = copy.copy(data_pipeline)
        config_source = getattr(self.base_dataset, "dataset", self.base_dataset)
        for attr in ("clip_len", "num_clips", "frame_interval", "full_videos"):
            if hasattr(config_source, attr) and hasattr(pipeline, attr):
                value = getattr(config_source, attr)
                if value is not None:
                    setattr(pipeline, attr, value)
        return pipeline

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> LoadedDatasetItem:
        sample = self.base_dataset[index]
        if not isinstance(sample, VideoSampleRef):
            raise TypeError(f"Expected VideoSampleRef from base dataset, got {type(sample)}")
        full_tensor = self.data_pipeline.load_video(sample.path)
        input_format = getattr(self.data_pipeline, "raw_input_format", "NTHWC")
        sampled_frame_ids = None
        x = full_tensor
        if hasattr(self.data_pipeline, "sample_video"):
            x, sampled_frame_ids = self.data_pipeline.sample_video(full_tensor, input_format=input_format)
            input_format = getattr(self.data_pipeline, "sampled_format", input_format)
        preprocessed = False
        if self.instant_preprocessing:
            if hasattr(self.data_pipeline, "preprocess_sampled"):
                x = self.data_pipeline.preprocess_sampled(x, input_format=input_format)
                input_format = getattr(self.data_pipeline, "preprocessed_format", input_format)
                preprocessed = True
            elif hasattr(self.data_pipeline, "preprocess"):
                x = self.data_pipeline.preprocess(x, input_format=input_format)
                input_format = getattr(self.data_pipeline, "preprocessed_format", input_format)
                preprocessed = True
        return LoadedDatasetItem(
            sample=sample,
            tensor=x,
            input_format=input_format,
            sampled=True,
            preprocessed=preprocessed,
            full_tensor=full_tensor if bool(getattr(self.data_pipeline, "full_videos", False)) else None,
            sampled_frame_ids=sampled_frame_ids,
        )


def collate_loaded_video_batch(batch: list[LoadedDatasetItem]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Empty batch")
    tensors = [item.tensor for item in batch]
    shapes = [tuple(t.shape) for t in tensors]
    stackable = all(shape == shapes[0] for shape in shapes)
    x = torch.stack(tensors, dim=0) if stackable else tensors
    return {
        "samples": [item.sample for item in batch],
        "tensor": x,
        "input_format": batch[0].input_format,
        "sampled": batch[0].sampled,
        "preprocessed": batch[0].preprocessed,
        "stackable": stackable,
        "full_tensors": [item.full_tensor for item in batch],
        "sampled_frame_ids": [item.sampled_frame_ids for item in batch],
    }

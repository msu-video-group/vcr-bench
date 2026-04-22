from __future__ import annotations

import copy
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from vcr_bench.datasets.base import BaseVideoDataset
from vcr_bench.models.base import BaseVideoClassifier
from vcr_bench.types import LoadedDatasetItem, PredictionBundle, VideoSampleRef


_CSV_FIELDS = [
    "timestamp",
    "model",
    "model_name",
    "backbone",
    "weights_dataset",
    "dataset",
    "dataset_subset",
    "pipeline_stage",
    "sample_index",
    "video_path",
    "pass_type",
    "device",
    "cuda_device_name",
    "input_shape",
    "input_dtype",
    "input_mb",
    "baseline_allocated_mb",
    "baseline_reserved_mb",
    "peak_allocated_mb",
    "peak_reserved_mb",
    "delta_allocated_mb",
    "delta_reserved_mb",
    "elapsed_sec",
    "status",
    "error",
]


@dataclass
class VramProfileContext:
    model_arg: str
    dataset_arg: str
    dataset_subset: str | None
    backbone: str | None
    weights_dataset: str | None
    pipeline_stage: str
    seed: int
    sample_index: int | None = None


def append_vram_profile_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists() or out.stat().st_size == 0
    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in _CSV_FIELDS})


def profile_model_vram_for_one_video(
    *,
    model: BaseVideoClassifier,
    dataset: BaseVideoDataset,
    context: VramProfileContext,
) -> list[dict[str, Any]]:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    base_row = {
        "timestamp": timestamp,
        "model": context.model_arg,
        "model_name": getattr(model, "model_name", "unknown"),
        "backbone": context.backbone,
        "weights_dataset": context.weights_dataset,
        "dataset": context.dataset_arg,
        "dataset_subset": context.dataset_subset,
        "pipeline_stage": context.pipeline_stage,
    }
    if not torch.cuda.is_available():
        return [
            {
                **base_row,
                "pass_type": pass_type,
                "device": str(getattr(model, "device", "cpu")),
                "status": "skipped_no_cuda",
            }
            for pass_type in ("no_grad_forward", "with_grad_forward_backward")
        ]

    model_device = getattr(model, "device", torch.device("cuda"))
    if not isinstance(model_device, torch.device):
        model_device = torch.device(str(model_device))
    if model_device.type != "cuda":
        return [
            {
                **base_row,
                "pass_type": pass_type,
                "device": str(model_device),
                "status": "skipped_non_cuda_device",
            }
            for pass_type in ("no_grad_forward", "with_grad_forward_backward")
        ]

    pipeline = model.build_data_pipeline(context.pipeline_stage)
    configured_dataset = copy.copy(dataset).configure_loading(pipeline, instant_preprocessing=False)
    sample_index = _choose_sample_index(len(configured_dataset), context.sample_index, context.seed)
    item = configured_dataset[sample_index]
    sample, sampled_video, input_format = _extract_loaded_item(item)
    sampled_video = sampled_video.to(model_device)
    rows = [
        _profile_pass(
            base_row=base_row,
            model=model,
            sample=sample,
            sample_index=sample_index,
            x=sampled_video,
            input_format=input_format,
            pass_type="no_grad_forward",
            run_fn=lambda video: _run_no_grad_forward(model, video, input_format),
        ),
        _profile_pass(
            base_row=base_row,
            model=model,
            sample=sample,
            sample_index=sample_index,
            x=sampled_video,
            input_format=input_format,
            pass_type="with_grad_forward_backward",
            run_fn=lambda video: _run_with_grad_forward_backward(model, video, input_format),
        ),
    ]
    del sampled_video
    torch.cuda.empty_cache()
    return rows


def _choose_sample_index(dataset_len: int, explicit_index: int | None, seed: int) -> int:
    if dataset_len <= 0:
        raise ValueError("Cannot profile VRAM on an empty dataset")
    if explicit_index is not None:
        index = int(explicit_index)
        if index < 0 or index >= dataset_len:
            raise IndexError(f"VRAM profile sample index out of range: {index} for dataset size {dataset_len}")
        return index
    indices = list(range(dataset_len))
    random.Random(seed).shuffle(indices)
    return indices[0]


def _extract_loaded_item(item: LoadedDatasetItem | VideoSampleRef) -> tuple[VideoSampleRef, torch.Tensor, str]:
    if isinstance(item, LoadedDatasetItem):
        return item.sample, item.tensor, item.input_format
    raise TypeError("VRAM profiling requires a dataset configured for model loading")


def _profile_pass(
    *,
    base_row: dict[str, Any],
    model: BaseVideoClassifier,
    sample: VideoSampleRef,
    sample_index: int,
    x: torch.Tensor,
    input_format: str,
    pass_type: str,
    run_fn: Callable[[torch.Tensor], None],
) -> dict[str, Any]:
    device = x.device
    row = {
        **base_row,
        "sample_index": sample_index,
        "video_path": sample.path,
        "pass_type": pass_type,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "input_shape": "x".join(str(v) for v in tuple(x.shape)),
        "input_dtype": str(x.dtype).replace("torch.", ""),
        "input_mb": _bytes_to_mb(x.numel() * x.element_size()),
    }
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
        start = time.time()
        run_fn(x)
        torch.cuda.synchronize(device)
        row.update(
            {
                "baseline_allocated_mb": _bytes_to_mb(baseline_allocated),
                "baseline_reserved_mb": _bytes_to_mb(baseline_reserved),
                "peak_allocated_mb": _bytes_to_mb(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_mb": _bytes_to_mb(torch.cuda.max_memory_reserved(device)),
                "delta_allocated_mb": _bytes_to_mb(torch.cuda.max_memory_allocated(device) - baseline_allocated),
                "delta_reserved_mb": _bytes_to_mb(torch.cuda.max_memory_reserved(device) - baseline_reserved),
                "elapsed_sec": round(time.time() - start, 6),
                "status": "ok",
                "error": "",
            }
        )
    except Exception as e:
        row.update({"status": "error", "error": f"{type(e).__name__}: {e}"})
    finally:
        torch.cuda.empty_cache()
    return row


def _run_no_grad_forward(model: BaseVideoClassifier, x: torch.Tensor, input_format: str) -> None:
    with torch.no_grad():
        out = model.predict(x, input_format=input_format, enable_grad=False)
        if isinstance(out, PredictionBundle):
            _ = out.probs.detach()
        else:
            _ = out.detach()


def _run_with_grad_forward_backward(model: BaseVideoClassifier, x: torch.Tensor, input_format: str) -> None:
    x_grad = x.detach().float().clone().requires_grad_(True)
    out = model.predict(x_grad, input_format=input_format, enable_grad=True)
    probs = out.probs if isinstance(out, PredictionBundle) else out
    loss = probs.reshape(-1).max()
    grad = torch.autograd.grad(loss, x_grad, retain_graph=False, allow_unused=False)[0]
    _ = grad.detach()
    del x_grad, out, probs, loss, grad


def _bytes_to_mb(value: int | float) -> float:
    return round(float(value) / (1024.0 * 1024.0), 3)

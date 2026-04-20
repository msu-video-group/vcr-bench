from __future__ import annotations

import contextlib
from typing import Iterable

import torch
import torch.nn.functional as F

from vcr_bench.models.base import BaseVideoClassifier


def freeze_model_weights(model: BaseVideoClassifier) -> None:
    module = getattr(model, "model", None)
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad_(False)


def predict_probs(
    model: BaseVideoClassifier,
    x: torch.Tensor,
    *,
    input_format: str = "NTHWC",
    enable_grad: bool,
) -> torch.Tensor:
    probs = model.predict(x, input_format=input_format, enable_grad=enable_grad)
    if probs.ndim != 1:
        probs = probs.reshape(-1)
    return probs


def cross_entropy_from_probs(probs: torch.Tensor, target_label: int) -> torch.Tensor:
    target = torch.tensor([int(target_label)], device=probs.device, dtype=torch.long)
    return F.nll_loss(torch.log(probs.clamp_min(1e-12)).unsqueeze(0), target)


def safe_cuda_empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_cuda_reset_peak_memory_stats(device: str | torch.device) -> None:
    try:
        dev = torch.device(device)
    except Exception:
        dev = torch.device("cpu")
    if dev.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def best_effort_import(name: str):
    with contextlib.suppress(Exception):
        return __import__(name, fromlist=["__name__"])
    return None


def first_label(probs: torch.Tensor, *, targeted: bool) -> int:
    reducer = torch.argmin if targeted else torch.argmax
    return int(reducer(probs).item())


def detach_cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu()


def iter_dataset_items(dataset: object, indices: Iterable[int]):
    for idx in indices:
        yield idx, dataset[idx]


def flatten_temporal_video(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    if x.ndim == 4:
        return x, tuple(x.shape)
    if x.ndim == 5:
        shape = tuple(x.shape)
        return x.reshape(shape[0] * shape[1], *shape[-3:]), shape
    raise ValueError(f"Expected 4D/5D video tensor, got shape {tuple(x.shape)}")


def restore_temporal_video(x: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    if len(shape) == 4:
        return x.reshape(shape)
    if len(shape) == 5:
        return x.reshape(shape)
    raise ValueError(f"Unsupported target shape: {shape}")

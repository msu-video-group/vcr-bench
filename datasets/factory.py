from __future__ import annotations

import importlib

from .base import BaseVideoDataset


def create_dataset(name: str, **kwargs) -> BaseVideoDataset:
    key = name.lower()
    try:
        module = importlib.import_module(f"vg_videobench.datasets.{key}.data")
    except ModuleNotFoundError as exc:
        if exc.name == f"vg_videobench.datasets.{key}" or exc.name == f"vg_videobench.datasets.{key}.data":
            raise ValueError(f"Unknown dataset: {name}") from exc
        raise

    if hasattr(module, "create"):
        dataset = module.create(**kwargs)
    elif hasattr(module, "DATASET_CLASS"):
        dataset = getattr(module, "DATASET_CLASS")(**kwargs)
    else:
        raise ValueError(
            f"Dataset module vg_videobench.datasets.{key}.data must expose create(), "
            "or DATASET_CLASS"
        )
    if not isinstance(dataset, BaseVideoDataset):
        raise TypeError(f"Dataset factory returned unexpected type: {type(dataset)}")
    return dataset

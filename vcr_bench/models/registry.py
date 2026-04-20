from __future__ import annotations

import importlib
from typing import Callable

from .base import BaseVideoClassifier


MODEL_REGISTRY: dict[str, Callable[..., BaseVideoClassifier]] = {}


def _module_key(name: str) -> str:
    return str(name).lower().replace("-", "_")


def _load_model_module(name: str):
    key = _module_key(name)
    try:
        return importlib.import_module(f"vcr_bench.models.{key}.model")
    except ModuleNotFoundError as exc:
        if exc.name == f"vcr_bench.models.{key}" or exc.name == f"vcr_bench.models.{key}.model":
            raise ValueError(f"Unknown model: {name}. Available dynamic plugins under vcr_bench.models/<name>/model.py") from exc
        raise


def get_model_options(name: str) -> dict:
    module = _load_model_module(name)
    if hasattr(module, "get_model_options"):
        opts = module.get_model_options()
        if isinstance(opts, dict):
            return opts
    cls = getattr(module, "MODEL_CLASS", None)
    if cls is None and hasattr(module, "create"):
        # Fallback: instantiate lightweight preview if plugin does not expose class.
        try:
            model = module.create(device="cpu", load_weights=False)
            if isinstance(model, BaseVideoClassifier):
                return {
                    "model": getattr(model, "model_name", name),
                    "backbones": model.available_backbones(),
                    "weight_datasets": {
                        b: model.available_weight_datasets(b) for b in model.available_backbones()
                    },
                    "capabilities": type(model).model_capabilities(),
                }
        except Exception:
            return {"model": name, "backbones": [], "weight_datasets": {}, "capabilities": {}}
    if cls is not None and issubclass(cls, BaseVideoClassifier):
        backs = cls.available_backbones()
        return {
            "model": getattr(cls, "model_name", name),
            "backbones": backs,
            "weight_datasets": {b: cls.available_weight_datasets(b) for b in backs},
            "capabilities": cls.model_capabilities(),
        }
    return {"model": name, "backbones": [], "weight_datasets": {}, "capabilities": {}}


def create_model(name: str, **kwargs) -> BaseVideoClassifier:
    key = _module_key(name)
    try:
        module = _load_model_module(key)
    except ValueError as exc:
        if key not in MODEL_REGISTRY:
            raise
        return MODEL_REGISTRY[key](**kwargs)

    if hasattr(module, "create"):
        model = module.create(**kwargs)
    elif hasattr(module, "MODEL_CLASS"):
        model = getattr(module, "MODEL_CLASS")(**kwargs)
    else:
        raise ValueError(
            f"Model module vcr_bench.models.{key}.model must expose create() or MODEL_CLASS"
        )
    if not isinstance(model, BaseVideoClassifier):
        raise TypeError(f"Model factory returned unexpected type: {type(model)}")
    return model

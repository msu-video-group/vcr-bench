from __future__ import annotations

import importlib

from .base import BaseVideoDefence


def _normalize_key(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def create_defence(name: str, **kwargs) -> BaseVideoDefence:
    key = _normalize_key(name)
    try:
        module = importlib.import_module(f"vcr_bench.defences.{key}.defence")
    except ModuleNotFoundError as exc:
        if exc.name in (f"vcr_bench.defences.{key}", f"vcr_bench.defences.{key}.defence"):
            raise ValueError(
                f"Unknown defence: {name}. "
                f"Available dynamic plugins under vcr_bench/defences/<name>/defence.py"
            ) from exc
        raise

    if hasattr(module, "create"):
        defence = module.create(**kwargs)
    elif hasattr(module, "DEFENCE_CLASS"):
        defence = getattr(module, "DEFENCE_CLASS")(**kwargs)
    else:
        raise ValueError(
            f"Defence module vcr_bench.defences.{key}.defence must expose create() or DEFENCE_CLASS"
        )
    if not isinstance(defence, BaseVideoDefence):
        raise TypeError(f"Defence factory returned unexpected type: {type(defence)}")
    return defence

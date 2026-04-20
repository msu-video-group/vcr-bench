from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def config_root() -> Path:
    return repo_root() / "configs"


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(current, value)
        else:
            merged[key] = value
    return merged


def _resolve_declared_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if path.is_absolute():
        return path
    return repo_root() / path


@lru_cache(maxsize=8)
def load_settings(extra_config: str | None = None) -> dict[str, Any]:
    cfg = _load_toml(config_root() / "defaults.toml")
    local_override = _load_toml(config_root() / "local.toml")
    if extra_config:
        local_override = _merge_dicts(local_override, _load_toml(Path(extra_config)))
    cfg = _merge_dicts(cfg, local_override)
    paths = cfg.setdefault("paths", {})
    for key in ("cache_dir", "data_dir", "results_dir", "remote_results_dir"):
        if key in paths:
            paths[f"{key}_abs"] = str(_resolve_declared_path(paths[key]))
    return cfg


@lru_cache(maxsize=4)
def load_checkpoint_registry(extra_config: str | None = None) -> dict[str, dict[str, Any]]:
    registry = _load_toml(config_root() / "checkpoints.toml").get("checkpoints", {})
    local = _load_toml(config_root() / "local.toml")
    if extra_config:
        local = _merge_dicts(local, _load_toml(Path(extra_config)))
    registry = _merge_dicts(registry, local.get("checkpoints", {}))
    return registry


@lru_cache(maxsize=4)
def load_dataset_registry(extra_config: str | None = None) -> dict[str, dict[str, Any]]:
    registry = _load_toml(config_root() / "datasets.toml").get("datasets", {})
    local = _load_toml(config_root() / "local.toml")
    if extra_config:
        local = _merge_dicts(local, _load_toml(Path(extra_config)))
    registry = _merge_dicts(registry, local.get("datasets", {}))
    return registry


@lru_cache(maxsize=4)
def load_benchmark_registry(extra_config: str | None = None) -> dict[str, Any]:
    registry = _load_toml(config_root() / "benchmarks.toml")
    local = _load_toml(config_root() / "local.toml")
    if extra_config:
        local = _merge_dicts(local, _load_toml(Path(extra_config)))
    return _merge_dicts(registry, {k: v for k, v in local.items() if k in ("accuracy_suites", "attack_suites")})


def get_path(name: str, *, extra_config: str | None = None) -> Path:
    settings = load_settings(extra_config)
    paths = settings.get("paths", {})
    explicit = paths.get(f"{name}_abs")
    if explicit:
        return Path(explicit)
    raw = paths.get(name)
    if raw is None:
        raise KeyError(f"Unknown configured path: {name}")
    resolved = _resolve_declared_path(raw)
    if resolved is None:
        raise KeyError(f"Unknown configured path: {name}")
    return resolved

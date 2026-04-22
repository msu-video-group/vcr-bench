from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .config import config_root, repo_root


PRESET_ROOT = config_root()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Preset must be a JSON object: {path}")
    return data


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _coerce_value(raw: str) -> Any:
    text = str(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _set_path(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [part for part in str(dotted_key).split(".") if part]
    if not parts:
        raise ValueError("Override key cannot be empty")
    cur = target
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override must have key=value format: {item}")
        key, value = item.split("=", 1)
        _set_path(out, key.strip(), _coerce_value(value.strip()))
    return out


def _preset_path(kind: str, name: str) -> Path:
    raw = Path(name)
    if raw.suffix == ".json" or raw.exists():
        return raw if raw.is_absolute() else repo_root() / raw
    direct = PRESET_ROOT / f"{kind}s" / f"{name}.json"
    if direct.exists():
        return direct
    return PRESET_ROOT / "presets" / f"{kind}s" / f"{name}.json"


def load_preset(kind: str, name: str) -> dict[str, Any]:
    path = _preset_path(kind, name)
    data = _load_json(path)
    actual_kind = data.get("kind")
    if actual_kind != kind:
        raise ValueError(f"Preset {path} has kind={actual_kind!r}, expected {kind!r}")
    data["_preset_path"] = str(path)
    return data


def resolve_entity_preset(
    kind: str,
    name: str,
    *,
    variant: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    preset = load_preset(kind, name)
    variants = preset.get("variants", preset.get("presets", {}))
    if not isinstance(variants, dict) or not variants:
        raise ValueError(f"{kind} preset has no variants: {name}")
    variant_name = variant or str(preset.get("default_variant") or next(iter(variants)))
    if variant_name not in variants:
        raise ValueError(f"Unknown variant {variant_name!r} for {kind} preset {name!r}")
    resolved_variant = _resolve_variant(variants, variant_name)
    resolved = {
        "kind": kind,
        "name": preset.get("name", name),
        "variant": variant_name,
        "factory_name": resolved_variant.get("factory_name", preset.get("factory_name", preset.get("name", name))),
        "params": resolved_variant.get("params", {}),
        "metadata": {k: v for k, v in preset.items() if k not in {"variants", "presets", "default_variant"}},
    }
    if overrides:
        resolved = _merge_dicts(resolved, overrides)
    return resolved


def _resolve_variant(variants: dict[str, Any], variant_name: str, seen: set[str] | None = None) -> dict[str, Any]:
    seen = set() if seen is None else seen
    if variant_name in seen:
        raise ValueError(f"Cycle in preset variant inheritance: {variant_name}")
    seen.add(variant_name)
    raw = variants[variant_name]
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid variant {variant_name}: expected object")
    parent_name = raw.get("inherits")
    if parent_name:
        parent = _resolve_variant(variants, str(parent_name), seen)
        return _merge_dicts(parent, {k: v for k, v in raw.items() if k != "inherits"})
    return copy.deepcopy(raw)


def resolve_run_preset(
    name: str,
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run = load_preset("run", name)
    resolved = {k: copy.deepcopy(v) for k, v in run.items() if not k.startswith("_")}
    resolved.setdefault("models", [])
    resolved.setdefault("runtime", {})
    resolved.setdefault("dataset", {})
    resolved.setdefault("output", {})
    if overrides:
        resolved = _merge_dicts(resolved, overrides)
    return resolved


def apply_namespace_overrides(
    base_overrides: dict[str, Any],
    namespace: str,
) -> dict[str, Any]:
    value = base_overrides.get(namespace, {})
    return value if isinstance(value, dict) else {}


def apply_resolved_to_namespace(args: Any, namespace: str, values: dict[str, Any]) -> None:
    for key, value in values.items():
        attr = key.replace("-", "_")
        if hasattr(args, attr):
            current = getattr(args, attr)
            if current is None or current is False:
                setattr(args, attr, value)


def apply_run_preset_to_args(args: Any, run: dict[str, Any]) -> None:
    task = run.get("task")
    dataset = run.get("dataset", {})
    runtime = run.get("runtime", {})
    output = run.get("output", {})
    if isinstance(dataset, dict):
        mapping = {
            "name": "dataset",
            "subset": "dataset_subset",
            "num_videos": "num_videos",
            "split": "split",
            "video_root": "video_root",
            "annotations": "annotations",
            "labels": "labels",
        }
        for key, attr in mapping.items():
            if key in dataset and hasattr(args, attr):
                current = getattr(args, attr)
                if current is None or (attr == "num_videos" and current == 25):
                    setattr(args, attr, dataset[key])
    if isinstance(runtime, dict):
        for key, value in runtime.items():
            attr = key.replace("-", "_")
            if hasattr(args, attr):
                current = getattr(args, attr)
                if current is None or current is False:
                    setattr(args, attr, value)
    if isinstance(output, dict):
        out_map = {"results_root": "results_root", "run_name": "attack_name"}
        for key, attr in out_map.items():
            if key in output and hasattr(args, attr):
                current = getattr(args, attr)
                if current in (None, "", "attacks", "debug"):
                    setattr(args, attr, output[key])
    if task and hasattr(args, "task"):
        setattr(args, "task", task)


def first_run_model(run: dict[str, Any]) -> dict[str, Any] | None:
    models = run.get("models", [])
    if isinstance(models, list) and models:
        first = models[0]
        return first if isinstance(first, dict) else {"name": str(first)}
    return None


def first_run_attack(run: dict[str, Any]) -> dict[str, Any] | None:
    attacks = run.get("attacks", [])
    if isinstance(attacks, list) and attacks:
        first = attacks[0]
        return first if isinstance(first, dict) else {"name": str(first)}
    attack = run.get("attack")
    return attack if isinstance(attack, dict) else None


def first_run_defence(run: dict[str, Any]) -> dict[str, Any] | None:
    defence = run.get("defence")
    return defence if isinstance(defence, dict) else None

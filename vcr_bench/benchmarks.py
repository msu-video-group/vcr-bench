from __future__ import annotations

from typing import Any

from .config import load_benchmark_registry


def get_accuracy_suite(name: str = "default") -> dict[str, Any]:
    suites = load_benchmark_registry().get("accuracy_suites", {})
    suite = suites.get(name)
    if not isinstance(suite, dict):
        raise ValueError(f"Unknown accuracy suite: {name}")
    return dict(suite)


def get_attack_suite(name: str = "default") -> dict[str, Any]:
    suites = load_benchmark_registry().get("attack_suites", {})
    suite = suites.get(name)
    if not isinstance(suite, dict):
        raise ValueError(f"Unknown attack suite: {name}")
    return dict(suite)


def expand_accuracy_suite(
    name: str = "default",
    *,
    model_filter: set[str] | None = None,
) -> dict[str, Any]:
    suite = get_accuracy_suite(name)
    models = [str(item) for item in suite.get("models", [])]
    if model_filter:
        models = [model for model in models if model in model_filter]
    suite["models"] = models
    return suite


def expand_attack_suite(
    name: str = "default",
    *,
    model_filter: set[str] | None = None,
    attack_filter: set[str] | None = None,
) -> dict[str, Any]:
    suite = get_attack_suite(name)
    entries = []
    for entry in suite.get("entries", []):
        if not isinstance(entry, dict):
            continue
        model = str(entry.get("model", ""))
        attack = str(entry.get("attack", ""))
        if model_filter and model not in model_filter:
            continue
        if attack_filter and attack not in attack_filter:
            continue
        entries.append(dict(entry))
    suite["entries"] = entries
    return suite

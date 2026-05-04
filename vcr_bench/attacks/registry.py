from __future__ import annotations

import importlib
from typing import Callable

from .base import BaseVideoAttack


ATTACK_REGISTRY: dict[str, Callable[..., BaseVideoAttack]] = {}


def _normalize_attack_key(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def load_attack_module(name: str):
    key = _normalize_attack_key(name)
    try:
        return importlib.import_module(f"vcr_bench.attacks.{key}.attack")
    except ModuleNotFoundError as exc:
        if exc.name == f"vcr_bench.attacks.{key}" or exc.name == f"vcr_bench.attacks.{key}.attack":
            if key not in ATTACK_REGISTRY:
                raise ValueError(
                    f"Unknown attack: {name}. Available dynamic plugins under vcr_bench.attacks/<name>/attack.py"
                ) from exc
            raise
        raise


def get_attack_class(name: str) -> type[BaseVideoAttack]:
    key = _normalize_attack_key(name)
    try:
        module = load_attack_module(key)
    except ValueError:
        factory = ATTACK_REGISTRY.get(key)
        if factory is None:
            raise
        attack = factory()
        if not isinstance(attack, BaseVideoAttack):
            raise TypeError(f"Attack factory returned unexpected type: {type(attack)}")
        return type(attack)

    if hasattr(module, "ATTACK_CLASS"):
        cls = getattr(module, "ATTACK_CLASS")
        if isinstance(cls, type) and issubclass(cls, BaseVideoAttack):
            return cls
    if hasattr(module, "create"):
        attack = module.create()
        if isinstance(attack, BaseVideoAttack):
            return type(attack)
    raise ValueError(
        f"Attack module vcr_bench.attacks.{key}.attack must expose create() or ATTACK_CLASS"
    )


def create_attack(name: str, **kwargs) -> BaseVideoAttack:
    key = _normalize_attack_key(name)
    try:
        module = load_attack_module(key)
    except ModuleNotFoundError as exc:
        if key not in ATTACK_REGISTRY:
            raise ValueError(
                f"Unknown attack: {name}. Available dynamic plugins under vcr_bench.attacks/<name>/attack.py"
            ) from exc
        return ATTACK_REGISTRY[key](**kwargs)
    except ValueError:
        if key not in ATTACK_REGISTRY:
            raise
        return ATTACK_REGISTRY[key](**kwargs)

    if hasattr(module, "create"):
        attack = module.create(**kwargs)
    elif hasattr(module, "ATTACK_CLASS"):
        attack = getattr(module, "ATTACK_CLASS")(**kwargs)
    else:
        raise ValueError(
            f"Attack module vcr_bench.attacks.{key}.attack must expose create() or ATTACK_CLASS"
        )
    if not isinstance(attack, BaseVideoAttack):
        raise TypeError(f"Attack factory returned unexpected type: {type(attack)}")
    return attack

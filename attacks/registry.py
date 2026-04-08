from __future__ import annotations

import importlib
from typing import Callable

from .base import BaseVideoAttack


ATTACK_REGISTRY: dict[str, Callable[..., BaseVideoAttack]] = {}


def _normalize_attack_key(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def create_attack(name: str, **kwargs) -> BaseVideoAttack:
    key = _normalize_attack_key(name)
    try:
        module = importlib.import_module(f"vg_videobench.attacks.{key}.attack")
    except ModuleNotFoundError as exc:
        if exc.name == f"vg_videobench.attacks.{key}" or exc.name == f"vg_videobench.attacks.{key}.attack":
            if key not in ATTACK_REGISTRY:
                raise ValueError(
                    f"Unknown attack: {name}. Available dynamic plugins under vg_videobench.attacks/<name>/attack.py"
                ) from exc
            return ATTACK_REGISTRY[key](**kwargs)
        raise

    if hasattr(module, "create"):
        attack = module.create(**kwargs)
    elif hasattr(module, "ATTACK_CLASS"):
        attack = getattr(module, "ATTACK_CLASS")(**kwargs)
    else:
        raise ValueError(
            f"Attack module vg_videobench.attacks.{key}.attack must expose create() or ATTACK_CLASS"
        )
    if not isinstance(attack, BaseVideoAttack):
        raise TypeError(f"Attack factory returned unexpected type: {type(attack)}")
    return attack

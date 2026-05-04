from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, get_args, get_origin, get_type_hints

from .base import BaseVideoAttack
from .registry import get_attack_class


@dataclass(frozen=True)
class AttackOptionSpec:
    param_name: str
    option_strings: tuple[str, ...]
    value_type: type[Any]
    default: Any
    help_text: str
    choices: tuple[Any, ...] = ()
    metavar: str | None = None
    boolean: bool = False


@dataclass(frozen=True)
class AttackConfigSpec:
    attack_name: str
    attack_class: type[BaseVideoAttack]
    options: tuple[AttackOptionSpec, ...]

    def default_params(self) -> dict[str, Any]:
        return {
            option.param_name: option.default
            for option in self.options
            if option.default is not inspect._empty
        }


_COMMON_HELP: dict[str, str] = {
    "eps": "L-infinity perturbation budget.",
    "alpha": "Attack step size.",
    "steps": "Number of attack iterations.",
    "random_start": "Use a randomized initialization before the main loop.",
    "sample_chunk_size": "Process sampled clips/views in chunks to reduce peak memory usage.",
    "query_budget": "Maximum number of model queries allowed per sample.",
    "n_samples": "Number of noise samples or NES directions per iteration.",
    "num_epochs": "Number of offline preparation or training epochs.",
    "save_root": "Directory for cached artifacts produced by the attack.",
    "style_steps": "Number of neural style optimization steps used by StyleFool.",
    "style_content_weight": "Content loss weight used by StyleFool neural style transfer.",
    "style_style_weight": "Style loss weight used by StyleFool neural style transfer.",
}

_COMMON_ALIASES: dict[str, tuple[str, ...]] = {
    "steps": ("--iter",),
    "sample_chunk_size": ("--attack-sample-chunk-size",),
    "save_root": ("--bmtc-save-root",),
}


def _sanitize_metavar(name: str) -> str:
    return str(name).upper().replace("-", "_")


def _normalize_type(annotation: Any, default: Any) -> type[Any]:
    if annotation is inspect._empty:
        if isinstance(default, bool):
            return bool
        if isinstance(default, int) and not isinstance(default, bool):
            return int
        if isinstance(default, float):
            return float
        if isinstance(default, str):
            return str
        return str
    origin = get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type):
            return annotation
        return str
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if len(args) == 1 and isinstance(args[0], type):
        return args[0]
    return str


def _option_strings(param_name: str) -> tuple[str, ...]:
    dashed = param_name.replace("_", "-")
    primary = f"--{dashed}"
    aliases = _COMMON_ALIASES.get(param_name, ())
    return (primary, *aliases)


def _help_text(param_name: str, default: Any) -> str:
    base = _COMMON_HELP.get(param_name, f"Attack-specific option: {param_name.replace('_', ' ')}.")
    if default is inspect._empty:
        return base
    return f"{base} Default: {default!r}."


def get_attack_spec(name: str) -> AttackConfigSpec:
    cls = get_attack_class(name)
    hints = get_type_hints(cls.__init__)
    signature = inspect.signature(cls.__init__)
    options: list[AttackOptionSpec] = []

    for param_name, parameter in signature.parameters.items():
        if param_name == "self":
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param_name.startswith("_"):
            continue

        annotation = hints.get(param_name, parameter.annotation)
        value_type = _normalize_type(annotation, parameter.default)
        boolean = value_type is bool or isinstance(parameter.default, bool)
        options.append(
            AttackOptionSpec(
                param_name=param_name,
                option_strings=_option_strings(param_name),
                value_type=value_type,
                default=parameter.default,
                help_text=_help_text(param_name, parameter.default),
                metavar=None if boolean else _sanitize_metavar(param_name),
                boolean=boolean,
            )
        )

    return AttackConfigSpec(
        attack_name=getattr(cls, "attack_name", str(name)),
        attack_class=cls,
        options=tuple(options),
    )


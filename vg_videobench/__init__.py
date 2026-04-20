from __future__ import annotations

import importlib
import sys

from vcr_bench import *  # noqa: F401,F403

_ALIASES = [
    "attacks",
    "benchmarks",
    "cli",
    "config",
    "artifacts",
    "datasets",
    "defences",
    "models",
    "remote",
    "types",
    "utils",
]

for _name in _ALIASES:
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"vcr_bench.{_name}")

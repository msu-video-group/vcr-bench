from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import importlib
import sys


def _matches_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


@contextmanager
def isolated_import_paths(
    *paths: str | Path,
    purge_modules: tuple[str, ...] = (),
):
    resolved_paths = [str(Path(path).resolve()) for path in paths]
    original_path = list(sys.path)
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if _matches_prefix(name, purge_modules)
    }
    for name in list(saved_modules):
        sys.modules.pop(name, None)
    sys.path[:0] = resolved_paths
    try:
        yield
    finally:
        sys.path[:] = original_path
        for name in list(sys.modules):
            if _matches_prefix(name, purge_modules):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


def import_attr(
    module_name: str,
    attr_name: str,
    *paths: str | Path,
    purge_modules: tuple[str, ...] = (),
):
    with isolated_import_paths(*paths, purge_modules=purge_modules):
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)

from __future__ import annotations

from pathlib import Path
from typing import Any

from vcr_bench.config import config_root, get_path, load_settings


def default_config_path() -> Path:
    return config_root() / "local.toml"


def load_remote_config(path: str | None = None, remote_name: str | None = None) -> dict[str, Any]:
    settings = load_settings(path)
    remote_cfg = settings.get("remote", {})
    remotes = remote_cfg.get("remotes", {})
    if not isinstance(remotes, dict) or not remotes:
        raise ValueError(
            "No remotes configured. Remote execution is optional; add a [remote] section to "
            "configs/local.toml or pass --config <path> with a remote definition."
        )
    selected = remote_name or remote_cfg.get("default")
    if not selected:
        raise ValueError(
            "No remote selected and no [remote].default configured. Pass --remote <name> or "
            "set [remote].default in your config."
        )
    data = remotes.get(selected)
    if not isinstance(data, dict):
        raise ValueError(f"Unknown remote: {selected}")
    slurm = dict(data.get("slurm", {}))
    return {
        "name": selected,
        "ssh_host": data["ssh_host"],
        "repo_path": data["repo_path"],
        "git_ref": data.get("git_ref", "origin/main"),
        "script_root": data.get("script_root", "scripts").rstrip("/"),
        "attack_results_root": data.get("attack_results_root", "results/remote_attacks"),
        "accuracy_results_root": data.get("accuracy_results_root", "results/accuracy/remote"),
        "sync_strategy": str(data.get("sync_strategy", "detach")),
        "local_results_dir": str(get_path("remote_results_dir", extra_config=path)),
        "slurm": slurm,
    }

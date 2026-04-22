from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path


def q(value: str) -> str:
    return shlex.quote(str(value))


def remote_path_expr(path: str) -> str:
    if path == "~":
        return "$HOME"
    if path.startswith("~/"):
        suffix = path[2:].replace('"', '\\"')
        return f"\"$HOME/{suffix}\""
    return q(path)


def remote_env_expr(value: str) -> str:
    text = str(value)
    def dq(inner: str) -> str:
        escaped = inner.replace('"', '\\"')
        return f"\"{escaped}\""
    def expand_remote_tilde(path: str) -> str:
        if path == "~":
            return "$HOME"
        if path.startswith("~/"):
            return f"$HOME/{path[2:]}"
        return path
    if "," in text and ":" in text:
        specs = []
        for spec in text.split(","):
            left, sep, right = spec.partition(":")
            specs.append(f"{expand_remote_tilde(left)}{sep}{right}" if sep else expand_remote_tilde(left))
        return dq(",".join(specs))
    if ":" in text and not text.startswith(("http://", "https://")):
        left, right = text.split(":", 1)
        if left == "~":
            return dq(f"$HOME:{right}")
        if left.startswith("~/"):
            return dq(f"$HOME/{left[2:]}:{right}")
    if text == "~":
        return dq("$HOME")
    if text.startswith("~/"):
        return dq(f"$HOME/{text[2:]}")
    return q(text)


def run_local(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    timeout_sec: float | None = None,
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        cmd,
        cwd=None if cwd is None else str(cwd),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_sec,
    )
    if check and result.returncode != 0:
        msg = (result.stderr or "").strip() or (result.stdout or "").strip()
        raise RuntimeError(msg or f"command failed: {' '.join(cmd)}")
    return result


def run_ssh(host: str, remote_cmd: str, *, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["ssh", "-o", "RequestTTY=no", "-o", "RemoteCommand=none", host, remote_cmd],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        msg = (result.stderr or "").strip() or (result.stdout or "").strip()
        raise RuntimeError(msg or f"ssh failed for {host}")
    return result


def as_json(data: dict) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def current_branch(cwd: Path) -> str:
    return run_local(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd).stdout.strip()


def current_commit(cwd: Path) -> str:
    return run_local(["git", "rev-parse", "HEAD"], cwd=cwd).stdout.strip()


def tracking_branch(cwd: Path) -> str:
    result = run_local(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], cwd=cwd, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def build_force_sync_command(repo_path: str, git_ref: str, script_root: str) -> str:
    repo_expr = remote_path_expr(repo_path)
    force_sync = (
        f"cd {repo_expr} && "
        "git fetch --all --prune && "
        f"git reset --hard {q(git_ref)} && "
        "git clean -fd "
        "-e vcr_bench/Classifiers "
        "-e vcr_bench/Classifiers/** "
        "-e vcr_bench/models/*/checkpoints "
        "-e vcr_bench/models/*/checkpoints/** && "
        f"chmod +x ./{script_root}/batch_attack.sh ./{script_root}/test_models.sh ./{script_root}/batch_prepare.sh || true"
    )
    return f"bash -lc {q(force_sync)}"


def parse_sbatch_output(text: str) -> int:
    match = re.search(r"Submitted batch job (\d+)", text)
    if not match:
        raise ValueError(f"Unable to parse sbatch output: {text.strip()}")
    return int(match.group(1))


def normalize_slurm_state(state: str) -> str:
    value = str(state).strip().upper()
    if value in {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "SUSPENDED"}:
        return "running"
    if value in {"COMPLETED"}:
        return "finished"
    if value in {"CANCELLED", "CANCELLED+", "STOPPED", "TIMEOUT"}:
        return "canceled"
    if value in {"FAILED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}:
        return "failed"
    return "unknown"


def build_remote_log_paths(repo_path: str, job_id: int) -> dict[str, str]:
    base = repo_path.rstrip("/")
    return {
        "attack_stdout": f"{base}/logs/job_{job_id}.log",
        "attack_stderr": f"{base}/logs/job_{job_id}.err",
        "test_stdout": f"{base}/logs/test_job_{job_id}.log",
        "test_stderr": f"{base}/logs/test_job_{job_id}.err",
    }


def shell_exports(env: dict[str, str | None]) -> str:
    chunks: list[str] = []
    for key, value in env.items():
        if value is None or value == "":
            continue
        chunks.append(f"export {key}={remote_env_expr(value)}")
    return " && ".join(chunks)

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def log(msg: str) -> None:
    print(f"[launch_remote] {msg}")


def run_local(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    log(f"local cmd={' '.join(cmd)} cwd={cwd}")
    result = subprocess.run(cmd, cwd=str(cwd), check=False, capture_output=True, text=True)
    if result.stdout.strip():
        log(f"local stdout:\n{result.stdout.rstrip()}")
    if result.stderr.strip():
        log(f"local stderr:\n{result.stderr.rstrip()}")
    log(f"local exit_code={result.returncode}")
    if check and result.returncode != 0:
        msg = (result.stderr or "").strip() or (result.stdout or "").strip()
        raise RuntimeError(f"local command failed (exit {result.returncode}): {msg}")
    return result


def open_source_control(cwd: Path) -> None:
    candidates = []
    for name in ["code", "code.cmd", "cursor", "cursor.cmd"]:
        resolved = shutil.which(name)
        if resolved:
            candidates.append(resolved)
    localapp = Path.home() / "AppData" / "Local" / "Programs" / "Microsoft VS Code" / "bin"
    for p in [localapp / "code.cmd", localapp / "code"]:
        if p.exists():
            candidates.append(str(p))
    for exe in candidates:
        try:
            subprocess.Popen([exe, "--reuse-window", str(cwd)], cwd=str(cwd))
            time.sleep(1.2)
            exe_name = Path(exe).name.lower()
            if "cursor" in exe_name:
                subprocess.Popen(["cmd", "/c", "start", "", "cursor://command/workbench.view.scm"], cwd=str(cwd))
            else:
                subprocess.Popen(["cmd", "/c", "start", "", "vscode://command/workbench.view.scm"], cwd=str(cwd))
            log(f"opened Source Control via: {exe}")
            return
        except FileNotFoundError:
            continue
    run_local(["cmd", "/c", "start", "", str(cwd)], cwd=cwd, check=False)
    log("VS Code CLI not found. Open Source Control manually.")


def run_ssh(host: str, remote_cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    log(f"ssh host={host} cmd={remote_cmd}")
    result = subprocess.run(
        ["ssh", "-o", "RequestTTY=no", "-o", "RemoteCommand=none", host, remote_cmd],
        check=False, capture_output=True, text=True,
    )
    if result.stdout.strip():
        log(f"stdout:\n{result.stdout.rstrip()}")
    if result.stderr.strip():
        log(f"stderr:\n{result.stderr.rstrip()}")
    log(f"exit_code={result.returncode}")
    if check and result.returncode != 0:
        msg = (result.stderr or "").strip() or (result.stdout or "").strip()
        raise RuntimeError(f"ssh failed for host '{host}' (exit {result.returncode}): {msg}")
    return result


def q(value: str) -> str:
    return shlex.quote(value)


def remote_path_expr(path: str) -> str:
    if path == "~":
        return "$HOME"
    if path.startswith("~/"):
        rest = path[2:].replace('"', '\\"')
        return f"\"$HOME/{rest}\""
    return q(path)


def repo_home_relative(path: str) -> str:
    if path == "~":
        return ""
    if path.startswith("~/"):
        return path[2:]
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scripts/autolaunch/configs/main.json")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--skip-local-git", action="store_true")
    parser.add_argument("--window-name", default="auto_launch")
    parser.add_argument("--commit-message", default=None)
    parser.add_argument("--service-arg", action="append", default=[])
    args = parser.parse_args()

    repo_root_local = Path(__file__).resolve().parent.parent.parent
    config_arg = Path(args.config)
    config_path_local = config_arg if config_arg.is_absolute() else (repo_root_local / config_arg).resolve()
    config_path_remote = config_arg.as_posix()
    log(f"loading config: {config_path_local}")
    cfg = load_config(config_path_local)

    ssh_host = cfg.get("ssh_host", "")
    tmux_session = cfg.get("tmux_session", "auto_launch")
    repo_path = cfg.get("repo_path", "")

    if not ssh_host or not repo_path:
        print("ssh_host or repo_path missing in config", file=sys.stderr)
        return 1

    log(f"ssh_host={ssh_host} tmux_session={tmux_session} repo_path={repo_path}")
    repo_path_remote = remote_path_expr(repo_path)

    service_parts = [
        "python3",
        "scripts/autolaunch/service.py",
        "--config",
        config_path_remote,
    ]
    if args.test_mode:
        service_parts.append("--test-mode")
    service_parts.extend(args.service_arg)
    service_cmd = " ".join(q(p) for p in service_parts)

    force_sync = (
        f"cd {repo_path_remote} && "
        "git fetch --all --prune && "
        "git reset --hard origin/master && "
        "git clean -fd "
        "-e vcr_bench/Classifiers -e 'vcr_bench/Classifiers/**' "
        "-e 'vcr_bench/models/*/checkpoints' -e 'vcr_bench/models/*/checkpoints/**'"
    )
    git_pull_cmd = f"bash -lc {q(force_sync)}"
    ensure_exec_bits = (
        f"cd {repo_path_remote} && "
        "chmod +x ./scripts/batch_attack.sh ./scripts/test_models.sh ./scripts/batch_prepare.sh || true"
    )
    ensure_exec_bits_cmd = f"bash -lc {q(ensure_exec_bits)}"

    ts = int(time.time())
    window_name = args.window_name
    remote_log_rel = f"scripts/autolaunch/logs/launch_remote_{ts}.log"
    remote_log_abs = f"$HOME/{repo_home_relative(repo_path).rstrip('/')}/{remote_log_rel}"
    tmux_payload = (
        f"cd {repo_path_remote} && "
        "mkdir -p scripts/autolaunch/logs && "
        f"{service_cmd} >> {q(remote_log_rel)} 2>&1"
    )
    tmux_service_cmd = f"bash -lc {q(tmux_payload)}"

    try:
        if not args.skip_local_git:
            log("checking local git status")
            status = run_local(["git", "status", "--porcelain"], cwd=repo_root_local)
            if status.stdout.strip():
                if not args.commit_message:
                    log("local changes found and --commit-message missing; opening VS Code Source Control")
                    open_source_control(repo_root_local)
                    print("Local changes detected. Commit manually, then rerun.")
                    return 2
                run_local(["git", "add", "-A"], cwd=repo_root_local)
                run_local(["git", "commit", "--no-verify", "-m", args.commit_message], cwd=repo_root_local)
            else:
                log("no local changes")
            run_local(["git", "push"], cwd=repo_root_local)

        log("force-syncing remote repo")
        run_ssh(ssh_host, git_pull_cmd)
        log("ensuring executable bits on remote scripts")
        run_ssh(ssh_host, ensure_exec_bits_cmd, check=False)

        has_session = run_ssh(ssh_host, f"tmux has-session -t {q(tmux_session)}", check=False)
        if has_session.returncode == 0:
            log("tmux session exists; reusing")
            run_ssh(ssh_host, "tmux set-option -g remain-on-exit on")
            run_ssh(ssh_host, f"tmux kill-window -t {q(tmux_session)}:{q(window_name)}", check=False)
            created = run_ssh(
                ssh_host,
                f"tmux new-window -d -P -F '#S:#I:#W' -t {q(tmux_session)} -n {q(window_name)} {q(tmux_service_cmd)}",
            )
        else:
            log("tmux session missing; creating")
            run_ssh(ssh_host, "tmux set-option -g remain-on-exit on")
            created = run_ssh(
                ssh_host,
                f"tmux new-session -d -P -F '#S:#I:#W' -s {q(tmux_session)} -n {q(window_name)} {q(tmux_service_cmd)}",
            )
        if created.stdout.strip():
            log(f"created window: {created.stdout.strip()}")
        run_ssh(ssh_host, f"tmux select-window -t {q(tmux_session)}:{q(window_name)}", check=False)
        run_ssh(ssh_host, f"tmux list-windows -t {q(tmux_session)}")
    except RuntimeError as err:
        print(str(err), file=sys.stderr)
        return 1

    print(f"Started service in tmux session '{tmux_session}' on {ssh_host} (window '{window_name}')")
    print(f"Remote service log: {remote_log_abs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

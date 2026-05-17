import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


SERVICE_PATTERNS = [
    "scripts/autolaunch/[s]ervice.py",
]


def log(msg: str) -> None:
    print(f"[stop_remote] {msg}")


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def q(value: str) -> str:
    return shlex.quote(value)


def bash_lc(script: str) -> str:
    return f"bash -lc {q(script)}"


def remote_path_expr(path: str) -> str:
    if path == "~":
        return "$HOME"
    if path.startswith("~/"):
        rest = path[2:].replace('"', '\\"')
        return f"\"$HOME/{rest}\""
    return q(path)


def run_ssh(host: str, remote_cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    log(f"ssh host={host} cmd={remote_cmd}")
    result = subprocess.run(
        ["ssh", "-T", "-o", "RemoteCommand=none", host, remote_cmd],
        check=False,
        capture_output=True,
        text=True,
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


def service_pid_script() -> str:
    return (
        "for p in "
        + " ".join(q(pat) for pat in SERVICE_PATTERNS)
        + "; do pgrep -f \"$p\" || true; done | sort -u"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Stop the remote autolaunch service.")
    parser.add_argument("--config", default="scripts/autolaunch/configs/main.json")
    parser.add_argument("--wait-seconds", type=int, default=30)
    parser.add_argument("--kill-session", action="store_true")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    ssh_host = cfg.get("ssh_host", "")
    tmux_session = cfg.get("tmux_session", "auto_launch")
    repo_path = cfg.get("repo_path", "")
    repo_path_remote = remote_path_expr(repo_path)

    if not ssh_host or not repo_path:
        print("ssh_host or repo_path missing in config", file=sys.stderr)
        return 1

    terminate_script = (
        "set +e; "
        "pids=\"$(" + service_pid_script() + " | xargs)\"; "
        "if [ -n \"$pids\" ]; then kill -TERM $pids || true; echo \"sent TERM to: $pids\"; "
        "else echo \"no running service process\"; fi"
    )

    try:
        run_ssh(ssh_host, bash_lc(terminate_script), check=False)

        deadline = time.time() + max(1, args.wait_seconds)
        while time.time() < deadline:
            p = run_ssh(ssh_host, bash_lc(service_pid_script()), check=False)
            if not p.stdout.strip():
                log("service process stopped")
                break
            time.sleep(1)
        else:
            log("wait timeout reached; service process still appears to be running")

        state_info_script = (
            f"cd {repo_path_remote} && "
            "if [ -f scripts/autolaunch/runtime/state.json ]; then "
            "echo \"state file: scripts/autolaunch/runtime/state.json\"; "
            "python3 -c "
            "\"import json;from pathlib import Path;"
            "s=json.loads(Path('scripts/autolaunch/runtime/state.json').read_text());"
            "print('last_saved_at:', s.get('last_saved_at'));"
            "print('running_jobs:', len(s.get('running', {})))\"; "
            "else echo \"state file not found\"; fi"
        )
        run_ssh(ssh_host, bash_lc(state_info_script), check=False)

        if args.kill_session:
            run_ssh(ssh_host, f"tmux kill-session -t {q(tmux_session)}", check=False)
    except RuntimeError as err:
        print(str(err), file=sys.stderr)
        return 1

    print(f"Stop command sent for service on {ssh_host}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

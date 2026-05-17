import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_VERSION = "2026-05-17.1"
STATE_PATH = "scripts/autolaunch/runtime/state.json"
SERVICE_PATTERNS = [
    "scripts/autolaunch/[s]ervice.py",
]


def log(msg: str) -> None:
    print(f"[stop_calculations] {msg}")


def q(value: str) -> str:
    return shlex.quote(value)


def bash_lc(script: str) -> str:
    return f"bash -lc {q(script)}"


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _service_pid_script() -> str:
    return (
        "for p in "
        + " ".join(q(pat) for pat in SERVICE_PATTERNS)
        + "; do pgrep -f \"$p\" || true; done | sort -u"
    )


def get_tracked_running_ids(host: str, repo_path_remote: str) -> list[str]:
    script = (
        f"cd {repo_path_remote} && "
        "python3 -c "
        "\"import json;from pathlib import Path;"
        f"p=Path('{STATE_PATH}');"
        "s=json.loads(p.read_text()) if p.exists() else {};"
        "print(' '.join(str(k) for k in s.get('running', {}).keys()))\""
    )
    result = run_ssh(host, bash_lc(script), check=True)
    return sorted({x for x in result.stdout.strip().split() if x.strip()})


def get_job_name_ids(host: str, job_name: str) -> list[str]:
    if not job_name:
        return []
    result = run_ssh(
        host,
        bash_lc("squeue -h -o %i -n " + q(job_name) + " 2>/dev/null || true"),
        check=False,
    )
    return sorted({x for x in result.stdout.strip().split() if x.strip()})


def stop_service(host: str, timeout_sec: int) -> None:
    term_script = (
        "set +e; "
        "pids=\"$(" + _service_pid_script() + " | xargs)\"; "
        "if [ -n \"$pids\" ]; then kill -TERM $pids || true; echo \"sent TERM to: $pids\"; "
        "else echo \"no running service process\"; fi"
    )
    run_ssh(host, bash_lc(term_script), check=False)

    deadline = time.time() + max(1, timeout_sec)
    alive = []
    while time.time() < deadline:
        p = run_ssh(host, bash_lc(_service_pid_script()), check=False)
        alive = [x for x in p.stdout.splitlines() if x.strip()]
        if not alive:
            log("service process stopped")
            return
        time.sleep(1)

    kill_script = (
        "set +e; "
        "pids=\"$(" + _service_pid_script() + " | xargs)\"; "
        "if [ -n \"$pids\" ]; then kill -KILL $pids || true; echo \"sent KILL to: $pids\"; fi"
    )
    log(f"service still alive after timeout: {alive}; sending SIGKILL")
    run_ssh(host, bash_lc(kill_script), check=False)


def cancel_jobs(host: str, job_ids: list[str]) -> None:
    if not job_ids:
        log("no running job ids to cancel")
        return
    run_ssh(host, bash_lc("scancel " + " ".join(q(j) for j in job_ids)), check=True)


def backup_and_clear_state(host: str, repo_path_remote: str) -> None:
    script = (
        f"cd {repo_path_remote} && "
        "python3 -c "
        "\"import json,time,os,shutil;from pathlib import Path;"
        f"p=Path('{STATE_PATH}');"
        "p.parent.mkdir(parents=True, exist_ok=True);"
        "s=json.loads(p.read_text()) if p.exists() else {};"
        "((lambda b:(shutil.copy2(p,b),print('state backup saved:',b)))"
        "(str(p)+'.bak.'+time.strftime('%Y%m%d_%H%M%S', time.localtime())+'.'+str(os.getpid()))) if p.exists() else None;"
        "s.setdefault('running', {});"
        "s.setdefault('job_miss_polls', {});"
        "s['running']={};"
        "s['job_miss_polls']={};"
        "s['last_saved_at']=int(time.time());"
        "p.write_text(json.dumps(s, indent=2));"
        "print('state running info cleared')\""
    )
    run_ssh(host, bash_lc(script), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stop remote autolaunch service, scancel attack jobs, clear running state."
    )
    parser.add_argument("--config", default="scripts/autolaunch/configs/main.json")
    parser.add_argument("--service-timeout-sec", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    ssh_host = cfg.get("ssh_host", "")
    repo_path = cfg.get("repo_path", "")
    job_name = cfg.get("job_name", "")
    repo_path_remote = remote_path_expr(repo_path)
    if not ssh_host or not repo_path:
        print("ssh_host or repo_path missing in config", file=sys.stderr)
        return 1

    log(f"script_version={SCRIPT_VERSION}")
    try:
        run_ssh(ssh_host, "echo connected", check=True)
        tracked_ids = get_tracked_running_ids(ssh_host, repo_path_remote)
        named_ids = get_job_name_ids(ssh_host, job_name)
        job_ids = sorted(set(tracked_ids) | set(named_ids))
        log(f"tracked running job ids in state: {tracked_ids}")
        log(f"running job ids with name {job_name!r}: {named_ids}")
        log(f"job ids to cancel: {job_ids}")
        if args.dry_run:
            log("dry-run: no changes applied")
            return 0

        stop_service(ssh_host, args.service_timeout_sec)
        cancel_jobs(ssh_host, job_ids)
        backup_and_clear_state(ssh_host, repo_path_remote)
        log("done")
        return 0
    except RuntimeError as err:
        print(str(err), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

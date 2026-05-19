import argparse
import json
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path


REMOTE_MONITOR = r"""
import argparse
import os
import re
import shlex
import subprocess


def run(cmd, timeout=15):
    try:
        return subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None


def shell(cmd, timeout=15):
    return run(["bash", "-lc", cmd], timeout=timeout)


def parse_mem_to_mb(value):
    value = str(value or "").strip()
    if not value or value in {"Unknown", "n/a"}:
        return ""
    m = re.match(r"^([0-9.]+)([KMGTP]?)$", value, re.I)
    if not m:
        return value
    number = float(m.group(1))
    unit = m.group(2).upper()
    scale = {"": 1 / (1024 * 1024), "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}.get(unit, 1)
    mb = number * scale
    return f"{mb:.0f}M"


def squeue_rows(job_names):
    rows = []
    seen = set()
    fmt = "%i|%j|%T|%M|%D|%R|%b|%C|%m"
    for name in job_names:
        if not name:
            continue
        res = run(["squeue", "-h", "-o", fmt, "-n", name], timeout=10)
        if res is None:
            continue
        for line in res.stdout.splitlines():
            parts = line.split("|")
            if len(parts) < 9:
                continue
            job_id = parts[0].strip()
            if not job_id or job_id in seen:
                continue
            seen.add(job_id)
            rows.append(
                {
                    "jobid": job_id,
                    "name": parts[1].strip(),
                    "state": parts[2].strip(),
                    "time": parts[3].strip(),
                    "nodes": parts[4].strip(),
                    "reason": parts[5].strip(),
                    "gres": parts[6].strip(),
                    "cpus": parts[7].strip(),
                    "mem": parts[8].strip(),
                }
            )
    return rows


def sstat_for_job(job_id):
    fields = "AveCPU,AveRSS,MaxRSS"
    for step in (f"{job_id}.batch", job_id):
        res = run(["sstat", "-n", "-P", "-j", step, f"--format={fields}"], timeout=10)
        if res is None or res.returncode != 0:
            continue
        lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        if not lines:
            continue
        parts = lines[0].split("|")
        if len(parts) >= 3:
            return {
                "ave_cpu": parts[0].strip(),
                "ave_rss": parse_mem_to_mb(parts[1]),
                "max_rss": parse_mem_to_mb(parts[2]),
            }
    return {"ave_cpu": "", "ave_rss": "", "max_rss": ""}


def gpu_for_job(job_id, enabled=True):
    if not enabled:
        return "off"
    query = "index,utilization.gpu,memory.used,memory.total"
    cmd = (
        "srun --overlap --jobid "
        + shlex.quote(str(job_id))
        + " -N1 -n1 nvidia-smi --query-gpu="
        + shlex.quote(query)
        + " --format=csv,noheader,nounits"
    )
    res = shell(cmd, timeout=8)
    if res is None or res.returncode != 0:
        return "n/a"
    vals = []
    for line in res.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            vals.append(f"{parts[0]}:{parts[1]}% {parts[2]}/{parts[3]}M")
    return "; ".join(vals) if vals else "n/a"


def short(value, width):
    value = str(value or "")
    if len(value) <= width:
        return value
    return value[: max(0, width - 1)] + "~"


def print_table(rows):
    columns = [
        ("JOBID", "jobid", 8),
        ("NAME", "name", 17),
        ("STATE", "state", 10),
        ("TIME", "time", 9),
        ("NODE", "reason", 12),
        ("CPU_ALLOC", "cpus", 9),
        ("AVE_CPU", "ave_cpu", 10),
        ("AVE_RSS", "ave_rss", 8),
        ("MAX_RSS", "max_rss", 8),
        ("GRES", "gres", 10),
        ("GPU", "gpu", 32),
    ]
    if not rows:
        print("No matching jobs.")
        return
    print(" ".join(title.ljust(width) for title, _, width in columns))
    print(" ".join("-" * width for _, _, width in columns))
    for row in rows:
        print(" ".join(short(row.get(key, ""), width).ljust(width) for _, key, width in columns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", action="append", default=[])
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    rows = squeue_rows(args.job_name)
    for row in rows:
        row.update(sstat_for_job(row["jobid"]))
        row["gpu"] = gpu_for_job(row["jobid"], enabled=not args.no_gpu and row["state"] == "RUNNING")
    print_table(rows)


if __name__ == "__main__":
    main()
"""


def q(value: str) -> str:
    return shlex.quote(value)


def remote_path_expr(path: str) -> str:
    if path == "~":
        return "$HOME"
    if path.startswith("~/"):
        rest = path[2:].replace('"', '\\"')
        return f"\"$HOME/{rest}\""
    return q(path)


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Print Slurm CPU/GPU load for current autolaunch jobs.")
    parser.add_argument("--config", default="scripts/autolaunch/configs/main.json")
    parser.add_argument("--job-name", action="append", default=[], help="Slurm job name to include. Can be repeated.")
    parser.add_argument("--no-gpu", action="store_true", help="Skip per-job nvidia-smi probes.")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    host = cfg.get("ssh_host", "")
    repo_path = cfg.get("repo_path", "")
    if not host or not repo_path:
        print("ssh_host or repo_path missing in config", file=sys.stderr)
        return 1

    job_names = list(args.job_name)
    if not job_names:
        job_names.append("attack_vcr_bench")
        configured_name = str(cfg.get("job_name", "")).strip()
        if configured_name and configured_name not in job_names:
            job_names.append(configured_name)

    remote_args = []
    for name in job_names:
        remote_args.extend(["--job-name", name])
    if args.no_gpu:
        remote_args.append("--no-gpu")

    remote_cmd = (
        f"cd {remote_path_expr(repo_path)} && "
        "python3 - "
        + " ".join(q(part) for part in remote_args)
        + " <<'PY'\n"
        + REMOTE_MONITOR
        + "\nPY"
    )
    result = subprocess.run(
        ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "RemoteCommand=none", host, f"bash -lc {q(remote_cmd)}"],
        check=False,
        text=True,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

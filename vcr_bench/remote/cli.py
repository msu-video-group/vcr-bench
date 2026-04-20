from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from vcr_bench.benchmarks import expand_accuracy_suite, expand_attack_suite

from .config import default_config_path, load_remote_config
from .core import (
    as_json,
    build_force_sync_command,
    build_remote_log_paths,
    current_branch,
    current_commit,
    normalize_slurm_state,
    parse_sbatch_output,
    q,
    remote_path_expr,
    repo_root,
    run_local,
    run_ssh,
    shell_exports,
    tracking_branch,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vcr_bench remote tools")
    parser.add_argument("--config", default=str(default_config_path()))
    parser.add_argument("--remote", default=None, help="Named remote from configs/defaults.toml or configs/local.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    push = sub.add_parser("push-code")
    push.add_argument("--commit-message", default=None)
    push.add_argument("--stage-all", action="store_true", help="Force full-tree staging with git add -A")
    push.add_argument("--dry-run", action="store_true")

    sync = sub.add_parser("sync-remote")
    sync.add_argument("--dry-run", action="store_true")

    launch = sub.add_parser("launch-job")
    launch.add_argument("--mode", choices=["test", "attack"], required=True)
    launch.add_argument("--model", default=None)
    launch.add_argument(
        "--model-spec",
        action="append",
        default=[],
        help="Attack-mode flexible entry: model[:attack[:target[:defence[:adaptive]]]]. Repeat to submit multiple entries in one sbatch.",
    )
    launch.add_argument("--dataset", default="kinetics400")
    launch.add_argument("--dataset-subset", default=None)
    launch.add_argument("--num-videos", type=int, default=8)
    launch.add_argument("--device", default="cuda")
    launch.add_argument("--backbone", default=None)
    launch.add_argument("--weights-dataset", default=None)
    launch.add_argument("--checkpoint", default=None)
    launch.add_argument("--attack", default=None)
    launch.add_argument("--attack-name", default="debug")
    launch.add_argument("--eps", type=float, default=None)
    launch.add_argument("--iter", type=int, default=None)
    launch.add_argument("--target", action="store_true")
    launch.add_argument("--comment", default="")
    launch.add_argument("--results-root", default=None)
    launch.add_argument("--dump-freq", type=int, default=None)
    launch.add_argument("--defence", default=None)
    launch.add_argument("--lite-attack", action="store_true")
    launch.add_argument("--save-defence-stages", action="store_true")
    launch.add_argument("--dry-run", action="store_true")

    accuracy_suite = sub.add_parser("launch-accuracy-suite")
    accuracy_suite.add_argument("--suite", default="default")
    accuracy_suite.add_argument("--model", action="append", default=[])
    accuracy_suite.add_argument("--dry-run", action="store_true")

    attack_suite = sub.add_parser("launch-attack-suite")
    attack_suite.add_argument("--suite", default="default")
    attack_suite.add_argument("--model", action="append", default=[])
    attack_suite.add_argument("--attack", action="append", default=[])
    attack_suite.add_argument("--dry-run", action="store_true")

    status = sub.add_parser("job-status")
    status.add_argument("--job-id", type=int, required=True)
    status.add_argument("--dry-run", action="store_true")

    fetch = sub.add_parser("fetch-job-artifacts")
    fetch.add_argument("--job-id", type=int, required=True)
    fetch.add_argument("--attack-name", default=None)
    fetch.add_argument("--result-subdir", default=None)
    fetch.add_argument("--dry-run", action="store_true")
    return parser


def _emit(payload: dict[str, Any]) -> None:
    print(as_json(payload))


def cmd_push_code(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    del cfg
    root = repo_root()
    branch = current_branch(root)
    upstream = tracking_branch(root)
    commit_message = args.commit_message or "update"
    commands: list[list[str]] = []

    untracked_paths: list[str] = []
    untracked_scan_timed_out = False
    if args.stage_all:
        commands.append(["git", "add", "-A"])
        if not args.dry_run:
            run_local(commands[-1], cwd=root)
    else:
        commands.append(["git", "add", "-u"])
        if not args.dry_run:
            run_local(commands[-1], cwd=root)
        try:
            untracked = run_local(
                ["git", "ls-files", "--others", "--exclude-standard", "--directory"],
                cwd=root,
                timeout_sec=3.0,
            )
            untracked_paths = [line.strip() for line in untracked.stdout.splitlines() if line.strip()]
        except Exception:
            untracked_scan_timed_out = True

        if untracked_paths:
            commands.append(["git", "add", "--", *untracked_paths])
            if not args.dry_run:
                run_local(commands[-1], cwd=root)

    staged = run_local(["git", "diff", "--cached", "--name-only"], cwd=root)
    if staged.stdout.strip():
        commands.append(["git", "commit", "--no-gpg-sign", "--no-verify", "-m", commit_message])
    commands.append(["git", "push"])
    if not args.dry_run:
        for cmd in commands[2 if untracked_paths else 1 :]:
            run_local(cmd, cwd=root)
    _emit(
        {
            "command": "push-code",
            "branch": branch,
            "tracking_branch": upstream,
            "commit_sha": current_commit(root) if not args.dry_run else None,
            "commands": [" ".join(q(part) for part in cmd) for cmd in commands],
            "dirty": bool(staged.stdout.strip()),
            "untracked_paths_sample": untracked_paths[:20],
            "untracked_scan_timed_out": bool(untracked_scan_timed_out),
            "stage_all": bool(args.stage_all),
            "dry_run": bool(args.dry_run),
        }
    )


def cmd_sync_remote(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    remote_cmd = build_force_sync_command(cfg["repo_path"], cfg["git_ref"], cfg["script_root"])
    if not args.dry_run:
        run_ssh(cfg["ssh_host"], remote_cmd)
    _emit(
        {
            "command": "sync-remote",
            "remote_name": cfg["name"],
            "remote_host": cfg["ssh_host"],
            "remote_repo": cfg["repo_path"],
            "git_ref": cfg["git_ref"],
            "remote_command": remote_cmd,
            "dry_run": bool(args.dry_run),
        }
    )


def _remote_env(cfg: dict[str, Any]) -> dict[str, str | None]:
    slurm = cfg.get("slurm", {})
    return {
        "VCR_BENCH_CONTAINER_IMAGE": slurm.get("container_image"),
        "VCR_BENCH_CONTAINER_MOUNTS": slurm.get("container_mounts"),
    }


def _build_launch_remote_command(args: argparse.Namespace, cfg: dict[str, Any]) -> tuple[str, str]:
    repo_expr = remote_path_expr(cfg["repo_path"])
    script_root = cfg["script_root"].rstrip("/")
    exports = shell_exports(_remote_env(cfg))
    if args.mode == "test":
        sbatch_parts = [
            "sbatch",
            f"./{script_root}/test_models.sh",
            "--dataset",
            args.dataset,
            "--num-videos",
            str(args.num_videos),
            "--device",
            args.device,
            "--results-root",
            args.results_root or cfg["accuracy_results_root"],
        ]
        if args.model:
            sbatch_parts.extend(["--model", args.model])
        if args.attack_name and args.attack_name != "debug":
            sbatch_parts.extend(["--folder-name", args.attack_name])
        if args.dataset_subset:
            sbatch_parts.extend(["--dataset-subset", args.dataset_subset])
        if args.backbone:
            sbatch_parts.extend(["--backbone", args.backbone])
        if args.weights_dataset:
            sbatch_parts.extend(["--weights-dataset", args.weights_dataset])
        if args.checkpoint:
            sbatch_parts.extend(["--checkpoint", args.checkpoint])
    else:
        model_specs = [str(item).strip() for item in getattr(args, "model_spec", []) if str(item).strip()]
        if not args.attack and not model_specs:
            raise ValueError("--attack is required in attack mode unless --model-spec provides per-entry attacks")
        if not args.model and not model_specs:
            raise ValueError("--model or --model-spec is required in attack mode")
        sbatch_parts = [
            "sbatch",
            f"./{script_root}/batch_attack.sh",
            "--attack-name",
            args.attack_name,
            "--dataset",
            args.dataset,
            "--num-videos",
            str(args.num_videos),
            "--device",
            args.device,
            "--results-root",
            args.results_root or cfg["attack_results_root"],
        ]
        if args.dataset_subset:
            sbatch_parts.extend(["--dataset-subset", args.dataset_subset])
        if args.attack:
            sbatch_parts.extend(["--attack-type", args.attack])
        else:
            sbatch_parts.extend(["--attack-type", "ifgsm"])
        if model_specs:
            sbatch_parts.extend(["--model", ",".join(model_specs)])
        else:
            sbatch_parts.extend(["--model", args.model])
        if args.eps is not None:
            sbatch_parts.extend(["--eps", str(args.eps)])
        if args.iter is not None:
            sbatch_parts.extend(["--iter", str(args.iter)])
        if args.target:
            sbatch_parts.append("--target")
        if args.comment:
            sbatch_parts.extend(["--comment", args.comment])
        if getattr(args, "dump_freq", None) is not None:
            sbatch_parts.extend(["--dump-freq", str(args.dump_freq)])
        if getattr(args, "defence", None):
            sbatch_parts.extend(["--defence", args.defence])
        if getattr(args, "lite_attack", False):
            sbatch_parts.append("--lite-attack")
        if getattr(args, "save_defence_stages", False):
            sbatch_parts.append("--save-defence-stages")
    sbatch_cmd = " ".join(q(part) for part in sbatch_parts)
    chained = f"cd {repo_expr}"
    if exports:
        chained = f"{chained} && {exports}"
    chained = f"{chained} && {sbatch_cmd}"
    return sbatch_cmd, f"bash -lc {q(chained)}"


def cmd_launch_job(args: argparse.Namespace, cfg: dict[str, Any], *, emit: bool = True) -> dict[str, Any]:
    sbatch_cmd, remote_cmd = _build_launch_remote_command(args, cfg)
    if args.dry_run:
        job_id = 0
    else:
        result = run_ssh(cfg["ssh_host"], remote_cmd)
        job_id = parse_sbatch_output(result.stdout)
    payload = {
        "command": "launch-job",
        "mode": args.mode,
        "job_id": int(job_id),
        "remote_name": cfg["name"],
        "remote_host": cfg["ssh_host"],
        "remote_repo": cfg["repo_path"],
        "submitted_command": sbatch_cmd,
        "remote_command": remote_cmd,
        "log_paths": build_remote_log_paths(cfg["repo_path"], int(job_id)),
        "dry_run": bool(args.dry_run),
    }
    if emit:
        _emit(payload)
    return payload


def cmd_launch_accuracy_suite(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    suite = expand_accuracy_suite(args.suite, model_filter=set(args.model) if args.model else None)
    launch_args = argparse.Namespace(
        mode="test",
        model=",".join(suite.get("models", [])),
        model_spec=[],
        dataset=suite["dataset"],
        dataset_subset=suite.get("dataset_subset"),
        num_videos=int(suite.get("num_videos", 8)),
        device=str(suite.get("device", "cuda")),
        backbone=suite.get("backbone"),
        weights_dataset=suite.get("weights_dataset"),
        checkpoint=suite.get("checkpoint"),
        attack=None,
        attack_name=f"{args.suite}_accuracy",
        eps=None,
        iter=None,
        target=False,
        comment="",
        results_root=suite.get("results_root", cfg["accuracy_results_root"]),
        dump_freq=None,
        defence=None,
        lite_attack=False,
        save_defence_stages=False,
        dry_run=bool(args.dry_run),
    )
    payload = cmd_launch_job(launch_args, cfg, emit=False)
    payload["suite"] = args.suite
    payload["suite_models"] = suite.get("models", [])
    _emit(payload)


def cmd_launch_attack_suite(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    suite = expand_attack_suite(
        args.suite,
        model_filter=set(args.model) if args.model else None,
        attack_filter=set(args.attack) if args.attack else None,
    )
    specs = []
    for entry in suite.get("entries", []):
        parts = [entry["model"], entry["attack"]]
        if entry.get("target"):
            parts.append("target")
        if entry.get("defence"):
            while len(parts) < 3:
                parts.append("")
            parts.append(str(entry["defence"]))
        if entry.get("adaptive"):
            while len(parts) < 4:
                parts.append("")
            parts.append("adaptive")
        specs.append(":".join(parts))
    launch_args = argparse.Namespace(
        mode="attack",
        model=None,
        model_spec=specs,
        dataset=suite["dataset"],
        dataset_subset=suite.get("dataset_subset"),
        num_videos=int(suite.get("num_videos", 8)),
        device=str(suite.get("device", "cuda")),
        backbone=suite.get("backbone"),
        weights_dataset=suite.get("weights_dataset"),
        checkpoint=suite.get("checkpoint"),
        attack=None,
        attack_name=str(suite.get("attack_name", args.suite)),
        eps=suite.get("eps"),
        iter=suite.get("iter"),
        target=False,
        comment="",
        results_root=suite.get("results_root", cfg["attack_results_root"]),
        dump_freq=suite.get("dump_freq"),
        defence=None,
        lite_attack=bool(suite.get("lite_attack", False)),
        save_defence_stages=bool(suite.get("save_defence_stages", False)),
        dry_run=bool(args.dry_run),
    )
    payload = cmd_launch_job(launch_args, cfg, emit=False)
    payload["suite"] = args.suite
    payload["suite_entries"] = suite.get("entries", [])
    _emit(payload)


def cmd_job_status(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    log_paths = build_remote_log_paths(cfg["repo_path"], args.job_id)
    squeue_cmd = f"squeue -j {int(args.job_id)} -h -o '%T|%N'"
    sacct_cmd = f"sacct -j {int(args.job_id)} --format=State,ExitCode -P -n"
    if args.dry_run:
        state = "unknown"
        node = ""
        exit_code = ""
    else:
        sq = run_ssh(cfg["ssh_host"], squeue_cmd, check=False)
        if sq.returncode == 0 and sq.stdout.strip():
            raw_state, _, node = sq.stdout.strip().partition("|")
            state = normalize_slurm_state(raw_state)
            exit_code = ""
        else:
            sa = run_ssh(cfg["ssh_host"], sacct_cmd, check=False)
            raw_state = ""
            exit_code = ""
            node = ""
            for line in sa.stdout.splitlines():
                parts = line.strip().split("|")
                if len(parts) >= 2 and parts[0]:
                    raw_state = parts[0]
                    exit_code = parts[1]
                    break
            state = normalize_slurm_state(raw_state)
    _emit(
        {
            "command": "job-status",
            "job_id": int(args.job_id),
            "state": state,
            "exit_code": exit_code,
            "node": node,
            "remote_name": cfg["name"],
            "remote_host": cfg["ssh_host"],
            "log_paths": log_paths,
            "dry_run": bool(args.dry_run),
        }
    )


def cmd_fetch_job_artifacts(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    local_root = Path(cfg["local_results_dir"]).resolve() / cfg["name"] / f"job_{int(args.job_id)}"
    local_root.mkdir(parents=True, exist_ok=True)
    repo = cfg["repo_path"].rstrip("/")
    rsync_cmds: list[list[str]] = [
        ["rsync", "-avz", f"{cfg['ssh_host']}:{repo}/logs/job_{int(args.job_id)}.log", str(local_root / "attack_job.log")],
        ["rsync", "-avz", f"{cfg['ssh_host']}:{repo}/logs/job_{int(args.job_id)}.err", str(local_root / "attack_job.err")],
        ["rsync", "-avz", f"{cfg['ssh_host']}:{repo}/logs/test_job_{int(args.job_id)}.log", str(local_root / "test_job.log")],
        ["rsync", "-avz", f"{cfg['ssh_host']}:{repo}/logs/test_job_{int(args.job_id)}.err", str(local_root / "test_job.err")],
    ]
    if args.attack_name:
        remote_dir = f"{repo}/{args.result_subdir}" if args.result_subdir else f"{repo}/{cfg['attack_results_root']}/{args.attack_name}"
        rsync_cmds.append(["rsync", "-avz", f"{cfg['ssh_host']}:{remote_dir}", str(local_root / "results")])
    if not args.dry_run:
        for cmd in rsync_cmds:
            run_local(cmd, check=False)
    _emit(
        {
            "command": "fetch-job-artifacts",
            "job_id": int(args.job_id),
            "remote_name": cfg["name"],
            "local_dir": str(local_root),
            "commands": [" ".join(q(part) for part in cmd) for cmd in rsync_cmds],
            "dry_run": bool(args.dry_run),
        }
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_remote_config(args.config, args.remote)
    if args.command == "push-code":
        cmd_push_code(args, cfg)
    elif args.command == "sync-remote":
        cmd_sync_remote(args, cfg)
    elif args.command == "launch-job":
        cmd_launch_job(args, cfg)
    elif args.command == "launch-accuracy-suite":
        cmd_launch_accuracy_suite(args, cfg)
    elif args.command == "launch-attack-suite":
        cmd_launch_attack_suite(args, cfg)
    elif args.command == "job-status":
        cmd_job_status(args, cfg)
    elif args.command == "fetch-job-artifacts":
        cmd_fetch_job_artifacts(args, cfg)
    else:
        raise ValueError(f"Unsupported command: {args.command}")

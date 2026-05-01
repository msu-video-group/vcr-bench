import argparse
import logging
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from launcher_lib import (
    build_combos,
    candidate_log_paths,
    combo_id,
    compute_dump_freq,
    expected_paths,
    is_completed,
    is_completed_from_log,
    load_main_config,
    load_tracks_config,
)

import json

RUNTIME_DIR = Path("scripts/autolaunch/runtime")
LOG_DIR = Path("scripts/autolaunch/logs")
STATE_FILE = RUNTIME_DIR / "state.json"
STOP_REQUESTED = False


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_DIR / "service.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_state():
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
    else:
        state = {}
    state.setdefault("running", {})
    state.setdefault("failures", {})
    state.setdefault("stoplist", {})
    state.setdefault("completed", {})
    state.setdefault("last_saved_at", 0)
    state.setdefault("job_miss_polls", {})
    return state


def save_state(state):
    state["last_saved_at"] = int(time.time())
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def handle_stop_signal(signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    logging.info("received stop signal: %s", signum)


def get_running_job_ids(job_name=None, tracked_job_ids=None):
    if tracked_job_ids:
        ids_arg = ",".join(str(jid) for jid in tracked_job_ids if str(jid).strip())
        if not ids_arg:
            return set()
        cmd = ["squeue", "-h", "-o", "%i", "-j", ids_arg]
    elif job_name:
        cmd = ["squeue", "-h", "-o", "%i", "-n", job_name]
    else:
        return set()
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            logging.warning("squeue failed: %s", result.stderr.strip())
            return None
        return set(line.strip() for line in result.stdout.splitlines() if line.strip())
    except FileNotFoundError:
        logging.error("squeue not found")
        return None


def get_job_queue_info(tracked_job_ids):
    ids_arg = ",".join(str(jid) for jid in tracked_job_ids if str(jid).strip())
    if not ids_arg:
        return {}
    try:
        result = subprocess.run(
            ["squeue", "-h", "-o", "%i|%T|%R", "-j", ids_arg],
            check=False, capture_output=True, text=True,
        )
        if result.returncode != 0:
            return {}
        info = {}
        for line in result.stdout.splitlines():
            parts = line.strip().split("|", 2)
            if len(parts) == 3:
                info[parts[0]] = {"state": parts[1], "reason": parts[2]}
        return info
    except FileNotFoundError:
        return {}


def _extract_batch_attack_passthrough(extra_args):
    allowed_no_value = {"--vmaf", "--no-vmaf", "-f", "--framewise-metrics", "--allow-misclassified", "--separate-logs"}
    allowed_with_value = {"--eps", "--iter", "--alpha", "--attack-sample-chunk-size", "--grad-forward-chunk-size"}
    passthrough = []
    i = 0
    while i < len(extra_args or []):
        arg = extra_args[i]
        if arg in allowed_no_value:
            passthrough.append(arg)
        elif arg in allowed_with_value and i + 1 < len(extra_args):
            passthrough.extend([arg, str(extra_args[i + 1])])
            i += 1
        i += 1
    return passthrough


def _stable_unique(values):
    seen = set()
    out = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _batch_combo_compat_key(combo):
    return (
        combo.get("attack_root"),
        combo.get("dataset"),
        combo.get("num_videos"),
        compute_dump_freq(combo.get("num_videos", 0), combo.get("save_videos", 0)),
        combo.get("comment", ""),
        bool(combo.get("full_video", False)),
        tuple(_extract_batch_attack_passthrough(combo.get("extra_args", []))),
    )


def build_batch_attack_multi_sbatch_cmd(main_cfg, combos):
    if not combos:
        raise ValueError("combos must not be empty")
    first = combos[0]
    slurm_cfg = main_cfg.get("slurm", {})
    script = str(slurm_cfg.get("batch_attack_script", "./scripts/batch_attack.sh"))
    dump_freq = compute_dump_freq(first["num_videos"], first.get("save_videos", 0))
    results_root = first.get("results_root") or main_cfg.get("results_root", "results/remote_attacks")
    logs_root = first.get("logs_root") or main_cfg.get("logs_root", "attack_logs")

    model_specs = _stable_unique([
        "{}:{}:{}:{}:{}".format(
            c["model"],
            c["attack"],
            "target" if c.get("target") else "untarget",
            c.get("defence", "no_defence"),
            "adaptive" if c.get("adaptive") else "non-adaptive",
        )
        for c in combos
    ])

    cmd = [
        "sbatch", script,
        "--attack-name", first["attack_root"],
        "--attack-type", first["attack"],
        "--dataset", first["dataset"],
        "--num-videos", str(first["num_videos"]),
        "--model", ",".join(model_specs),
        "--dump-freq", str(dump_freq),
        "--results-root", results_root,
        "--logs-root", logs_root,
    ]
    if first.get("full_video"):
        cmd.append("--full-videos")
    if first.get("comment"):
        cmd.extend(["--comment", first["comment"]])
    cmd.extend(_extract_batch_attack_passthrough(first.get("extra_args", [])))
    return cmd


def submit_batch_attack_combos(main_cfg, combos, test_mode=False):
    cmd = build_batch_attack_multi_sbatch_cmd(main_cfg, combos)
    cmd_str = " ".join(shlex.quote(part) for part in cmd)
    logging.info("batch_attack sbatch command: %s", cmd_str)
    if test_mode:
        print("TEST MODE: would run", cmd_str)
        logging.info("test mode: skipped sbatch (%d combos)", len(combos))
        return "test-mode"
    try:
        Path("logs").mkdir(parents=True, exist_ok=True)
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        logging.info("sbatch stdout: %s", result.stdout.strip())
        logging.info("sbatch stderr: %s", result.stderr.strip())
        if result.returncode != 0:
            logging.error("sbatch failed: %s", result.stderr.strip())
            return None
        stdout = result.stdout.strip()
        parts = stdout.split()
        if len(parts) >= 4 and parts[0].lower() == "submitted":
            return parts[3]
        if parts:
            return parts[-1]
        logging.error("unable to parse sbatch output: %s", stdout)
        return None
    except FileNotFoundError:
        logging.error("sbatch not found")
        return None


def update_state_for_finished_jobs(state, running_job_ids, completed, max_failures):
    if running_job_ids is None:
        logging.warning("skip finished-job update: running job ids unavailable")
        return
    miss = state.setdefault("job_miss_polls", {})
    finished = []
    for jid in list(state["running"]):
        if jid in running_job_ids:
            miss[jid] = 0
            continue
        miss[jid] = int(miss.get(jid, 0)) + 1
        if miss[jid] >= 2:
            finished.append(jid)
        else:
            logging.info("job %s not visible in squeue yet (miss_polls=%s)", jid, miss[jid])
    for jid in finished:
        for cid in state["running"].get(jid, []):
            if cid in completed:
                continue
            failures = state["failures"].get(cid, 0) + 1
            state["failures"][cid] = failures
            if failures >= max_failures:
                state["stoplist"][cid] = failures
        state["running"].pop(jid, None)
        miss.pop(jid, None)


def reconcile_running_jobs_on_start(state, running_job_ids):
    if running_job_ids is None:
        return
    stale = [jid for jid in list(state.get("running", {})) if jid not in running_job_ids]
    if not stale:
        return
    miss = state.setdefault("job_miss_polls", {})
    for jid in stale:
        state["running"].pop(jid, None)
        miss.pop(jid, None)
    logging.warning("startup reconcile: removed stale jobs: %s", stale)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scripts/autolaunch/configs/main.json")
    parser.add_argument("--test-mode", action="store_true")
    args = parser.parse_args()

    setup_logging()
    logging.info("service start")
    signal.signal(signal.SIGTERM, handle_stop_signal)
    signal.signal(signal.SIGINT, handle_stop_signal)

    main_cfg_path = Path(args.config)
    last_main_mtime = None
    last_tracks_mtime = None
    state = load_state()
    startup_reconciled = False

    while not STOP_REQUESTED:
        try:
            main_cfg = load_main_config(main_cfg_path)
            test_mode = args.test_mode or bool(main_cfg.get("test_mode", False))
            tracks_cfg_path = Path(main_cfg["tracks_config"])
            main_mtime = main_cfg_path.stat().st_mtime
            tracks_mtime = tracks_cfg_path.stat().st_mtime if tracks_cfg_path.exists() else 0

            if last_main_mtime != main_mtime or last_tracks_mtime != tracks_mtime:
                logging.info("config reload")
                last_main_mtime = main_mtime
                last_tracks_mtime = tracks_mtime

            tracks = load_tracks_config(tracks_cfg_path)
            combos = build_combos(tracks)
            combos_map = {combo_id(c): c for c in combos}

            completed = set()
            queue = []
            skipped_completed = skipped_stoplist = skipped_running = 0

            for combo in combos:
                cid = combo_id(combo)
                csv_path, _, attack_folder = expected_paths(combo, main_cfg)
                done_csv = is_completed(csv_path, expected_rows=combo.get("num_videos"))
                done_log = False
                if not done_csv:
                    for lp in candidate_log_paths(combo, main_cfg, attack_folder):
                        if is_completed_from_log(lp, combo["model"], attack_folder,
                                                 expected_total=combo.get("num_videos"), allowed_shortfall=1):
                            done_log = True
                            break
                if done_csv or done_log:
                    completed.add(cid)
                    skipped_completed += 1
                    state["completed"][cid] = {
                        "completed_at": int(time.time()),
                        "csv_path": csv_path,
                        "completion_source": "csv" if done_csv else "log",
                    }
                    continue
                if cid in state["stoplist"]:
                    skipped_stoplist += 1
                    continue
                if any(cid in cids for cids in state["running"].values()):
                    skipped_running += 1
                    continue
                queue.append(cid)

            tracked_job_ids = list(state["running"].keys())
            running_job_ids = get_running_job_ids(
                job_name=main_cfg["job_name"],
                tracked_job_ids=tracked_job_ids,
            )
            if running_job_ids is not None:
                logging.info("running: tracked=%s active=%s", tracked_job_ids, sorted(running_job_ids))
                for jid, info in get_job_queue_info(tracked_job_ids).items():
                    logging.info("job %s state=%s reason=%s", jid, info["state"], info["reason"])

            if not startup_reconciled:
                reconcile_running_jobs_on_start(state, running_job_ids)
                startup_reconciled = True

            update_state_for_finished_jobs(state, running_job_ids, completed, main_cfg["max_failures"])

            current_running = len(state["running"])
            available_slots = max(0, main_cfg["max_simultaneous_jobs"] - current_running)
            batch_size = max(1, int(main_cfg["max_attacks_per_job"]))
            logging.info(
                "scheduler: total=%d queue=%d completed=%d stoplisted=%d running=%d slots=%d test=%s",
                len(combos), len(queue), skipped_completed, skipped_stoplist,
                current_running, available_slots, test_mode,
            )

            for _ in range(available_slots):
                if not queue:
                    break
                first_cid = queue.pop(0)
                batch_ids = [first_cid]
                first_key = _batch_combo_compat_key(combos_map[first_cid])
                rest = []
                for cid in queue:
                    if len(batch_ids) < batch_size and _batch_combo_compat_key(combos_map[cid]) == first_key:
                        batch_ids.append(cid)
                    else:
                        rest.append(cid)
                queue = rest

                batch_combos = [combos_map[cid] for cid in batch_ids]
                job_id = submit_batch_attack_combos(main_cfg, batch_combos, test_mode=test_mode)
                if job_id:
                    if not test_mode:
                        state["running"][job_id] = batch_ids
                    logging.info("submitted job %s for %d combos", job_id, len(batch_ids))
                else:
                    logging.error("failed to submit job for %s", batch_ids)

            save_state(state)

            sleep_seconds = int(main_cfg["poll_interval_sec"])
            for _ in range(sleep_seconds):
                if STOP_REQUESTED:
                    break
                time.sleep(1)
        except Exception as e:
            logging.exception("service error: %s", e)
            for _ in range(10):
                if STOP_REQUESTED:
                    break
                time.sleep(1)

    save_state(state)
    logging.info("service stopped gracefully")


if __name__ == "__main__":
    main()

import csv
import json
import os
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text())


def load_main_config(path):
    cfg = load_json(path)
    cfg.setdefault("job_name", "vcr_bench")
    cfg.setdefault("max_simultaneous_jobs", 7)
    cfg.setdefault("max_attacks_per_job", 27)
    cfg.setdefault("max_failures", 2)
    cfg.setdefault("poll_interval_sec", 180)
    cfg.setdefault("tracks_config", "scripts/autolaunch/configs/tracks.json")
    cfg.setdefault("results_root", "results/remote_attacks")
    cfg.setdefault("logs_root", "attack_logs")
    cfg.setdefault("test_mode", False)
    cfg.setdefault("slurm", {})
    cfg["slurm"].setdefault("batch_attack_script", "./scripts/batch_attack.sh")
    cfg["slurm"].setdefault("container_image", "")
    cfg["slurm"].setdefault("container_mounts", "")
    cfg["slurm"].setdefault("avoid_node_names", [])
    return cfg


def load_tracks_config(path):
    return load_json(path).get("tracks", [])


def _normalize_name(value):
    if value is None:
        return ""
    return value.strip()


def parse_extra_args(extra_args):
    comment = ""
    full_video = False
    if not extra_args:
        return {"comment": comment, "full_video": full_video}
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        if arg == "--comment":
            if i + 1 < len(extra_args):
                comment = extra_args[i + 1]
                i += 1
        elif arg.startswith("--comment="):
            comment = arg.split("=", 1)[1]
        elif arg in ("--full-video", "--full-videos"):
            full_video = True
        i += 1
    return {"comment": comment, "full_video": full_video}


def sanitize_suffix(value):
    if not value:
        return ""
    return value.replace(os.sep, "_").replace(" ", "_")


def attack_folder_name(attack_name, target, defence_name, adaptive, comment):
    attack_type = "target" if target else "untarget"
    defence_name = defence_name or "no_defence"
    defence_type = "adaptive" if adaptive else "non-adaptive"
    name = f"{attack_name}_{attack_type}_{defence_name}_{defence_type}"
    comment = sanitize_suffix(comment)
    if comment:
        name = f"{name}_{comment}"
    return name


def compute_dump_freq(num_videos, save_videos):
    if save_videos is None:
        return 0
    try:
        save_videos = int(save_videos)
    except Exception:
        return 0
    if save_videos <= 0 or num_videos <= 0:
        return 0
    return max(1, num_videos // save_videos)


def build_combos(tracks):
    combos = []
    for track in tracks:
        track_name = track.get("name")
        attack_root = track.get("attack_name", track_name)
        dataset = track.get("dataset")
        num_videos = int(track.get("num_videos", 0))
        save_videos = track.get("save_videos", 0)
        extra_args = track.get("extra_args", [])
        extra_info = parse_extra_args(extra_args)
        attacks = track.get("attacks", [])
        defences = track.get("defences", [{"name": "no_defence", "adaptive": False}])
        models = track.get("models", [])
        track_results_root = track.get("results_root")

        for model in models:
            for attack in attacks:
                attack_name = attack.get("name")
                target = bool(attack.get("target", False))
                for defence in defences:
                    defence_name = _normalize_name(defence.get("name", "no_defence")) or "no_defence"
                    adaptive = bool(defence.get("adaptive", False))
                    combos.append({
                        "track_name": track_name,
                        "attack_root": attack_root,
                        "dataset": dataset,
                        "num_videos": num_videos,
                        "save_videos": save_videos,
                        "model": model,
                        "attack": attack_name,
                        "target": target,
                        "defence": defence_name,
                        "adaptive": adaptive,
                        "extra_args": extra_args,
                        "comment": extra_info["comment"],
                        "full_video": extra_info["full_video"],
                        "results_root": track_results_root,
                    })
    return combos


def combo_id(combo):
    return "|".join([
        combo["track_name"],
        combo["model"],
        combo["attack"],
        "target" if combo["target"] else "untarget",
        combo["defence"],
        "adaptive" if combo["adaptive"] else "non-adaptive",
    ])


def _effective_results_root(combo, main_cfg):
    return combo.get("results_root") or main_cfg["results_root"]


def expected_paths(combo, main_cfg):
    attack_folder = attack_folder_name(
        combo["attack"], combo["target"], combo["defence"], combo["adaptive"], combo["comment"]
    )
    results_root = _effective_results_root(combo, main_cfg)
    csv_path = os.path.join(results_root, combo["attack_root"], attack_folder, f"{combo['model']}.csv")
    log_path = os.path.join(results_root, combo["attack_root"], attack_folder, f"{combo['model']}.log")
    return csv_path, log_path, attack_folder


def candidate_log_paths(combo, main_cfg, attack_folder=None):
    if attack_folder is None:
        attack_folder = attack_folder_name(
            combo["attack"], combo["target"], combo["defence"], combo["adaptive"], combo["comment"]
        )
    results_root = _effective_results_root(combo, main_cfg)
    attack_root = combo["attack_root"]
    per_model_log = os.path.join(results_root, attack_root, attack_folder, f"log_{combo['model']}.csv")
    merged_log = os.path.join(results_root, attack_root, attack_folder, f"log_{attack_folder}.csv")
    shared_log = os.path.join(results_root, attack_root, f"log_{attack_root}.csv")
    return [per_model_log, merged_log, shared_log]


def is_completed(csv_path, expected_rows=None):
    if not os.path.isfile(csv_path):
        return False
    try:
        if os.path.getsize(csv_path) <= 0:
            return False
        if expected_rows is None:
            return True
        try:
            expected_rows = int(expected_rows)
        except Exception:
            return True
        if expected_rows <= 0:
            return True
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            row_count = sum(1 for _ in reader)
        return row_count >= expected_rows
    except OSError:
        return False


def is_completed_from_log(log_path, model_name, attack_name, expected_total=None, allowed_shortfall=0):
    if not os.path.isfile(log_path):
        return False
    try:
        with open(log_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("model") != model_name:
                    continue
                if row.get("attack") != attack_name:
                    continue
                if expected_total is None:
                    return True
                try:
                    expected_total = int(expected_total)
                except Exception:
                    return True
                if expected_total <= 0:
                    return True
                try:
                    allowed_shortfall = max(0, int(allowed_shortfall))
                except Exception:
                    allowed_shortfall = 0
                try:
                    num_total = int(float(row.get("num_total", 0)))
                except Exception:
                    num_total = 0
                if num_total >= max(0, expected_total - allowed_shortfall):
                    return True
        return False
    except OSError:
        return False

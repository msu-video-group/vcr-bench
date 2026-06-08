from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class TrackConfig:
    """Describes one benchmark track (white-box, black-box, ...).

    A track is a set of result folders named ``{attack}_{target_mode}_{defence}_{adaptive}``
    that all share the same per-video CSV layout. Each track is rendered into its own
    GitHub Pages data sub-directory and its own ``website_cache`` JSON file so the
    explorer can show tracks separately.
    """

    name: str
    source_id: str
    attacks: tuple[str, ...]
    defences: tuple[tuple[str, str], ...]
    target_mode: str
    cache_file: str
    default_results_root: str
    extra_mean_metrics: tuple[str, ...] = field(default_factory=tuple)


BASE_MEAN_METRICS = [
    "eps",
    "max_iters",
    "psnr",
    "ssim",
    "lpips",
    "dists",
    "vmaf",
    "mse",
    "iter_count",
    "time",
    "mean_time",
    "mean_time_ms",
    "mean_iterations",
]


TRACKS: dict[str, TrackConfig] = {
    "full_white_box": TrackConfig(
        name="full_white_box",
        source_id="full_white_box",
        attacks=(
            "amifgsm",
            "ifgsm",
            "mifgsm",
            "korhonen_et_al",
            "zhang_ssim",
            "zhang_dists",
            "zhang_lpips",
            "stadv",
        ),
        defences=(
            ("no_defence", "non-adaptive"),
            ("crop_resize", "adaptive"),
            ("flip", "adaptive"),
            ("gaussian_blur", "adaptive"),
            ("rotate", "adaptive"),
            ("shuffle", "adaptive"),
            ("temporal_median", "adaptive"),
            ("diff_jpeg", "adaptive"),
        ),
        target_mode="target",
        cache_file="website_cache.json",
        default_results_root="remote_results/autolaunch_full_white_box/results",
    ),
    "blackbox_full": TrackConfig(
        name="blackbox_full",
        source_id="blackbox_full",
        attacks=(
            "gradestv2",
            "bmtc",
            "stylefool",
            "square",
        ),
        defences=(
            ("no_defence", "non-adaptive"),
            ("crop_resize", "adaptive"),
            ("flip", "adaptive"),
            ("gaussian_blur", "adaptive"),
            ("rotate", "adaptive"),
            ("shuffle", "adaptive"),
            ("temporal_median", "adaptive"),
        ),
        target_mode="untarget",
        cache_file="website_cache_blackbox.json",
        default_results_root="remote_results/autolaunch_blackbox_full/results",
        extra_mean_metrics=("query_count",),
    ),
}


def parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_of(rows: list[dict[str, str]], col: str) -> Optional[float]:
    vals = [parse_float(row.get(col)) for row in rows]
    nums = [value for value in vals if value is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def compute_counts_from_rows(rows: list[dict[str, str]]) -> dict[str, float]:
    clear_correct = 0
    target_success = 0
    attacked_success = 0
    for row in rows:
        gt_class = parse_float(row.get("gt_class"))
        clear_class = parse_float(row.get("clear_class"))
        target_class = parse_float(row.get("target_class"))
        attacked_class = parse_float(row.get("attacked_class"))
        if gt_class is not None and clear_class is not None and int(gt_class) == int(clear_class):
            clear_correct += 1
        if target_class is not None and int(target_class) != -1:
            if attacked_class is not None and int(attacked_class) == int(target_class):
                target_success += 1
                attacked_success += 1
        elif attacked_class is not None and clear_class is not None and int(attacked_class) != int(clear_class):
            attacked_success += 1
    return {
        "clear_correct": float(clear_correct),
        "target_success": float(target_success),
        "attacked_success": float(attacked_success),
    }


def attack_folder(track: TrackConfig, attack: str, defence: str, adaptive: str) -> str:
    return f"{attack}_{track.target_mode}_{defence}_{adaptive}"


def ordinary_folders(track: TrackConfig) -> set[str]:
    return {
        attack_folder(track, attack, defence, adaptive)
        for attack in track.attacks
        for defence, adaptive in track.defences
    }


def parse_attack_meta(track: TrackConfig, attack_folder_name: str) -> dict[str, object]:
    for attack in track.attacks:
        prefix = f"{attack}_{track.target_mode}_"
        if not attack_folder_name.startswith(prefix):
            continue
        rest = attack_folder_name[len(prefix) :]
        for defence, adaptive in track.defences:
            suffix = f"{defence}_{adaptive}"
            if rest == suffix:
                return {
                    "attack": attack,
                    "targetMode": track.target_mode,
                    "defence": defence,
                    "adaptiveMode": adaptive,
                    "fullVideo": False,
                }
    return {
        "attack": attack_folder_name,
        "targetMode": track.target_mode,
        "defence": "unknown",
        "adaptiveMode": "non-adaptive",
        "fullVideo": False,
    }


def build_run(
    track: TrackConfig,
    log_row: dict[str, str],
    rows: list[dict[str, str]],
    csv_rel_path: str,
) -> dict[str, object]:
    attack_folder_name = (log_row.get("attack") or "").strip()
    model = (log_row.get("model") or "").strip()
    meta = parse_attack_meta(track, attack_folder_name)

    counts_from_rows = compute_counts_from_rows(rows)
    clear_correct = parse_float(log_row.get("clear_correct"))
    attacked_success = parse_float(log_row.get("attacked_success"))
    target_success = parse_float(log_row.get("target_success"))
    num_total = parse_float(log_row.get("num_total"))

    clear_correct = counts_from_rows["clear_correct"] if clear_correct is None else clear_correct
    attacked_success = counts_from_rows["attacked_success"] if attacked_success is None else attacked_success
    target_success = counts_from_rows["target_success"] if target_success is None else target_success
    num_total = float(len(rows)) if num_total is None or num_total <= 0 else num_total

    mean_metrics = list(BASE_MEAN_METRICS) + list(track.extra_mean_metrics)
    metrics: dict[str, Optional[float]] = {}
    for metric in mean_metrics:
        if metric == "eps":
            metrics[metric] = parse_float(log_row.get("eps"))
        elif metric == "max_iters":
            metrics[metric] = parse_float(log_row.get("max_iters"))
            if metrics[metric] is None:
                metrics[metric] = parse_float(log_row.get("iter"))
        elif metric in ("mean_time", "mean_time_ms", "mean_iterations"):
            metrics[metric] = parse_float(log_row.get(metric))
        elif metric == "psnr":
            metrics[metric] = parse_float(log_row.get("mean_psnr"))
        elif metric == "vmaf":
            metrics[metric] = parse_float(log_row.get("mean_vmaf"))
        else:
            metrics[metric] = mean_of(rows, metric)
        if metrics[metric] is None:
            metrics[metric] = mean_of(rows, metric)

    clear_den = clear_correct if clear_correct and clear_correct > 0 else None
    asr = (100.0 * attacked_success / clear_den) if clear_den else None
    target_sr = (100.0 * target_success / clear_den) if clear_den else None
    clear_acc = (100.0 * clear_correct / num_total) if num_total > 0 else None
    adv_acc = (100.0 * (clear_correct - attacked_success) / num_total) if num_total > 0 else None

    return {
        "key": f"{track.source_id}|{attack_folder_name}|{model}",
        "sourceId": track.source_id,
        "sourceFolder": track.name,
        "attackFolder": attack_folder_name,
        "attack": meta["attack"],
        "model": model,
        "defence": meta["defence"],
        "targetMode": meta["targetMode"],
        "adaptiveMode": meta["adaptiveMode"],
        "fullVideo": meta["fullVideo"],
        "nVideos": len(rows),
        "numTotal": num_total,
        "clearCorrectCount": clear_correct,
        "attackedSuccessCount": attacked_success,
        "targetSuccessCount": target_success,
        "metrics": metrics,
        "asr": asr,
        "target_sr": target_sr,
        "clear_acc": clear_acc,
        "adv_acc": adv_acc,
        "csvPath": csv_rel_path.replace("\\", "/"),
    }


def build_site_data(track: TrackConfig, results_root: Path, out_data: Path) -> dict[str, object]:
    expected_folders = ordinary_folders(track)
    out_root = out_data / track.name
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    merged_rows: list[dict[str, str]] = []
    merged_fields: list[str] = []
    copied_csv = 0
    missing_folders = []

    for folder_name in sorted(expected_folders):
        src_folder = results_root / folder_name
        if not src_folder.is_dir():
            missing_folders.append(folder_name)
            continue
        dst_folder = out_root / folder_name
        dst_folder.mkdir(parents=True, exist_ok=True)

        log_path = src_folder / f"log_{folder_name}.csv"
        if log_path.is_file():
            rows = read_csv_rows(log_path)
            if rows and not merged_fields:
                merged_fields = list(rows[0].keys())
            for row in rows:
                if not merged_fields:
                    merged_fields = list(row.keys())
                merged_rows.append({field: row.get(field, "") for field in merged_fields})

        for csv_path in sorted(src_folder.glob("*.csv")):
            if csv_path.name.startswith("log_"):
                continue
            shutil.copy2(csv_path, dst_folder / csv_path.name)
            copied_csv += 1

    if merged_fields:
        write_csv_rows(out_root / f"log_{track.name}.csv", merged_rows, merged_fields)

    runs = []
    missing_csv = 0
    rows_by_combo: dict[tuple[str, str], dict[str, str]] = {}
    for row in merged_rows:
        attack = (row.get("attack") or "").strip()
        model = (row.get("model") or "").strip()
        if attack and model:
            rows_by_combo[(attack, model)] = row

    for (attack_name, model), log_row in sorted(rows_by_combo.items()):
        csv_path = out_root / attack_name / f"{model}.csv"
        if not csv_path.is_file():
            missing_csv += 1
            continue
        rows = read_csv_rows(csv_path)
        if not rows:
            continue
        runs.append(build_run(track, log_row, rows, str(csv_path.relative_to(out_data))))

    cache = {
        "schema_version": 1,
        "track": track.name,
        "generated_at": int(time.time()),
        "results_root": str(out_data),
        "runs": runs,
        "stats": {
            "track": track.name,
            "ordinary_folders": len(expected_folders),
            "missing_folders": missing_folders,
            "merged_log_rows": len(merged_rows),
            "copied_csv": copied_csv,
            "processed_runs": len(runs),
            "missing_csv": missing_csv,
        },
    }
    out_data.mkdir(parents=True, exist_ok=True)
    with (out_data / track.cache_file).open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return cache["stats"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build GitHub Pages data for a benchmark track.")
    parser.add_argument(
        "--track",
        default="full_white_box",
        choices=sorted(TRACKS.keys()),
        help="Which benchmark track to build.",
    )
    parser.add_argument(
        "--results-root",
        default=None,
        help="Pulled remote results directory for the track (defaults per track).",
    )
    parser.add_argument("--out-data", default="docs/data", help="GitHub Pages data directory.")
    args = parser.parse_args()

    track = TRACKS[args.track]
    results_root = args.results_root or track.default_results_root
    stats = build_site_data(track, Path(results_root).resolve(), Path(args.out_data).resolve())
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

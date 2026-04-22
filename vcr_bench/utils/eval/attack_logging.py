from __future__ import annotations

import csv
from pathlib import Path


ATTACK_RESULT_FIELDNAMES = [
    "video_name",
    "gt_label",
    "gt_class",
    "clear_label",
    "clear_class",
    "clear_conf",
    "target_label",
    "target_class",
    "attacked_label",
    "attacked_class",
    "attacked_conf",
    "benign_conf",
    "psnr",
    "ssim",
    "mse",
    "lpips",
    "vmaf",
    "time",
    "iter_count",
    "query_count",
]

ATTACK_RESULT_DEFAULTS = {
    "video_name": "",
    "gt_label": "unknown",
    "gt_class": -1,
    "clear_label": "unknown",
    "clear_class": -1,
    "clear_conf": 0.0,
    "target_label": "unknown",
    "target_class": -1,
    "attacked_label": "unknown",
    "attacked_class": -1,
    "attacked_conf": 0.0,
    "benign_conf": 0.0,
    "psnr": 0.0,
    "ssim": 0.0,
    "mse": 0.0,
    "lpips": 0.0,
    "vmaf": 0.0,
    "time": 0.0,
    "iter_count": 0,
    "query_count": 0,
}

ATTACK_LOG_FIELDNAMES = [
    "model",
    "attack",
    "test_dataset",
    "time",
    "mean_time",
    "mean_iterations",
    "eps",
    "iter",
    "mean_psnr",
    "mean_vmaf",
    "clear_correct",
    "attacked_success",
    "target_success",
    "num_total",
]


def atomic_write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


class AttackLogger:
    def __init__(
        self,
        log_path: str | Path | None,
        attack_type: str,
        model_name: str,
        test_dataset: str,
        eps: float | int | None,
        iters: int | None,
    ) -> None:
        self.log_path = None if log_path is None else Path(log_path)
        self.attack_type = attack_type
        self.model_name = model_name
        self.test_dataset = test_dataset
        self.eps = "default" if eps is None else eps
        self.iters = "default" if iters is None else iters

    def __call__(
        self,
        total_time: float,
        mean_time: float,
        mean_iter: float,
        mean_psnr: float,
        mean_vmaf: float,
        clear_correct: int,
        attacked_success: int,
        target_success: int,
        total: int,
    ) -> None:
        if self.log_path is None:
            return
        rows: list[dict] = []
        replaced = False
        if self.log_path.exists():
            with self.log_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for key in ATTACK_LOG_FIELDNAMES:
                        row.setdefault(key, "unknown")
                    if (
                        not replaced
                        and row.get("model") == self.model_name
                        and row.get("attack") == self.attack_type
                    ):
                        rows.append(self._row(total_time, mean_time, mean_iter, mean_psnr, mean_vmaf, clear_correct, attacked_success, target_success, total))
                        replaced = True
                    else:
                        rows.append({k: row.get(k, "") for k in ATTACK_LOG_FIELDNAMES})
        if not replaced:
            rows.append(self._row(total_time, mean_time, mean_iter, mean_psnr, mean_vmaf, clear_correct, attacked_success, target_success, total))
        atomic_write_csv(self.log_path, ATTACK_LOG_FIELDNAMES, rows)

    def _row(
        self,
        total_time: float,
        mean_time: float,
        mean_iter: float,
        mean_psnr: float,
        mean_vmaf: float,
        clear_correct: int,
        attacked_success: int,
        target_success: int,
        total: int,
    ) -> dict:
        return {
            "model": self.model_name,
            "attack": self.attack_type,
            "test_dataset": self.test_dataset,
            "time": round(float(total_time), 4),
            "mean_time": round(float(mean_time), 4),
            "mean_iterations": round(float(mean_iter), 2),
            "eps": self.eps,
            "iter": self.iters,
            "mean_psnr": float(mean_psnr),
            "mean_vmaf": float(mean_vmaf),
            "clear_correct": int(clear_correct),
            "attacked_success": int(attacked_success),
            "target_success": int(target_success),
            "num_total": int(total),
        }


def read_existing_results(save_path: Path) -> tuple[list[dict], set[str]]:
    if not save_path.exists():
        return [], set()
    rows: list[dict] = []
    skip_paths: set[str] = set()
    with save_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            normalized = {k: row.get(k, ATTACK_RESULT_DEFAULTS[k]) for k in ATTACK_RESULT_FIELDNAMES}
            rows.append(normalized)
            if normalized["video_name"]:
                skip_paths.add(str(normalized["video_name"]))
    return rows, skip_paths


def write_result_rows(save_path: Path, rows: list[dict]) -> None:
    normalized = []
    for row in rows:
        full = dict(ATTACK_RESULT_DEFAULTS)
        full.update(row)
        normalized.append({k: full[k] for k in ATTACK_RESULT_FIELDNAMES})
    atomic_write_csv(save_path, ATTACK_RESULT_FIELDNAMES, normalized)


def append_result_rows(save_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = save_path.exists()
    with save_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ATTACK_RESULT_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            full = dict(ATTACK_RESULT_DEFAULTS)
            full.update(row)
            writer.writerow({k: full[k] for k in ATTACK_RESULT_FIELDNAMES})


def row_is_clean_correct(row: dict) -> bool:
    try:
        return int(row.get("gt_class", -1)) == int(row.get("clear_class", -2))
    except Exception:
        return False

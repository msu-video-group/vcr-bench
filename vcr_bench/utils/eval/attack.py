from __future__ import annotations

import os
import random
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from vcr_bench.attacks.base import BaseVideoAttack
from vcr_bench.attacks.query_base import BaseQueryVideoAttack
from vcr_bench.datasets.base import BaseVideoDataset
from vcr_bench.defences.base import BaseVideoDefence
from vcr_bench.models.base import BaseVideoClassifier
from vcr_bench.types import LoadedDatasetItem, VideoSampleRef
from vcr_bench.utils.eval.attack_logging import (
    ATTACK_RESULT_DEFAULTS,
    AttackLogger,
    append_result_rows as append_result_rows,
    read_existing_results as read_existing_results,
    row_is_clean_correct as row_is_clean_correct,
    write_result_rows as _write_result_rows,
)
from vcr_bench.utils.metrics import PSNR, MSE, SSIM, VMAF, LPIPS
from vcr_bench.utils.metrics import write_frame_metrics_csv_attack as _write_frame_metrics_csv
from vcr_bench.utils.video_dump import (
    probe_video_fps,
    save_video_lossless_like_original,
    to_uint8_thwc,
)


def _label_name(dataset: BaseVideoDataset, idx: int | None) -> str:
    if idx is None or idx < 0:
        return "unknown"
    try:
        names = dataset.class_names()
        return names[int(idx)] if 0 <= int(idx) < len(names) else "unknown"
    except Exception:
        return "unknown"


def _safe_conf(probs: torch.Tensor, label: int) -> float:
    if probs.ndim != 1:
        probs = probs.reshape(-1)
    if label < 0 or label >= probs.numel():
        return 0.0
    return float(probs[label].item())


def _predict_label_counted(
    model: BaseVideoClassifier,
    x: torch.Tensor,
    *,
    input_format: str,
    counter: dict[str, int],
) -> int:
    counter["count"] = int(counter.get("count", 0)) + 1
    return int(model.predict_label(x, input_format=input_format))

def _compute_attack_artifacts_task(
    *,
    sample_path: str,
    sampled_video_cpu: torch.Tensor,
    attacked_video_cpu: torch.Tensor,
    defended_video_cpu: torch.Tensor | None,
    full_video_cpu: torch.Tensor | None,
    sampled_frame_ids_cpu: torch.Tensor | None,
    clean_label: int,
    clean_label_name: str,
    clean_conf: float,
    attacked_label: int,
    attacked_label_name: str,
    attacked_conf: float,
    dump_video_dir: Path | None,
    calc_vmaf: bool,
    calc_lpips: bool,
    calc_frame_metrics: bool,
    frames_csv: Path | None,
) -> dict:
    mse_v = float(MSE(sampled_video_cpu, attacked_video_cpu))
    psnr_v = float(PSNR(sampled_video_cpu, attacked_video_cpu))
    try:
        ssim_v = float(SSIM(sampled_video_cpu, attacked_video_cpu))
    except Exception:
        ssim_v = 0.0
    try:
        lpips_v = float(LPIPS(sampled_video_cpu, attacked_video_cpu, device="cpu")) if calc_lpips else 0.0
    except Exception:
        lpips_v = 0.0
    try:
        vmaf_v = float(VMAF(sampled_video_cpu, attacked_video_cpu)) if calc_vmaf else 0.0
    except Exception:
        vmaf_v = 0.0

    if dump_video_dir is not None:
        try:
            source_fps = probe_video_fps(sample_path, default=16.0)

            def _safe_name(name: str) -> str:
                return str(name).replace(" ", "_").replace("/", "_").replace("\\", "_")

            save_video_lossless_like_original(
                sampled_video_cpu,
                dump_video_dir / f"clear_{_safe_name(clean_label_name)}_{clean_conf:.2f}.mp4",
                full_tensor=full_video_cpu,
                sampled_frame_ids=sampled_frame_ids_cpu,
                fps=source_fps,
            )
            save_video_lossless_like_original(
                attacked_video_cpu,
                dump_video_dir / f"attacked_{_safe_name(attacked_label_name)}_{attacked_conf:.2f}.mp4",
                full_tensor=full_video_cpu,
                sampled_frame_ids=sampled_frame_ids_cpu,
                fps=source_fps,
            )
            if defended_video_cpu is not None:
                save_video_lossless_like_original(
                    defended_video_cpu,
                    dump_video_dir / f"defended_{_safe_name(attacked_label_name)}_{attacked_conf:.2f}.mp4",
                    full_tensor=None,
                    sampled_frame_ids=None,
                    fps=source_fps,
                )
        except Exception:
            pass

    if calc_frame_metrics and frames_csv is not None:
        try:
            _write_frame_metrics_csv(
                frames_csv=frames_csv,
                sample_path=sample_path,
                clean_label=clean_label,
                clean_conf=clean_conf,
                attacked_label=attacked_label,
                attacked_conf=attacked_conf,
                clear_video=sampled_video_cpu,
                attacked_video=attacked_video_cpu,
            )
        except Exception:
            pass

    return {
        "mse": float(mse_v),
        "psnr": float(psnr_v),
        "ssim": float(ssim_v),
        "lpips": float(lpips_v),
        "vmaf": float(vmaf_v),
    }



def _pick_target_label(clear_label: int, num_classes: int) -> int:
    if num_classes <= 1:
        return clear_label
    return (int(clear_label) + 1) % int(num_classes)


def _extract_sample_item(x: VideoSampleRef | LoadedDatasetItem) -> tuple[VideoSampleRef, torch.Tensor, str]:
    if isinstance(x, LoadedDatasetItem):
        return x.sample, x.tensor, x.input_format
    raise TypeError("Attack runner requires dataset.configure_loading(...) so __getitem__ returns LoadedDatasetItem")


def run_attack(
    model: BaseVideoClassifier,
    attack: BaseVideoAttack,
    dataset: BaseVideoDataset,
    *,
    save_path: str | Path,
    log_path: str | Path | None,
    attack_name: str,
    num_videos: int,
    seed: int = 42,
    target: bool = False,
    allow_misclassified: bool = False,
    pipeline_stage: str = "test",
    instant_preprocessing: bool = False,
    skip_existing: bool = True,
    log_every: int = 20,
    verbose: bool = False,
    dump_freq: int = 0,
    dump_path: str | Path | None = None,
    calc_vmaf: bool = False,
    calc_lpips: bool = False,
    calc_frame_metrics: bool = False,
    metric_queue_limit: int | None = None,
    defence: BaseVideoDefence | None = None,
    adaptive: bool = False,
    save_defence_stages: bool = False,
) -> dict:
    if instant_preprocessing:
        raise ValueError("Attack runner requires sampled non-preprocessed input; use --instant-preprocessing off")

    save_path = Path(save_path)
    log_path_path = None if log_path is None else Path(log_path)
    dump_path_path = None if dump_path is None else Path(dump_path)

    # Adaptive defence: permanently install on model so attack gradient flows through it.
    if defence is not None and adaptive:
        defence.install(model)

    model.build_data_pipeline(pipeline_stage)
    configured_dataset = dataset.configure_loading(model.build_data_pipeline(pipeline_stage), instant_preprocessing=False)
    metric_executor = ThreadPoolExecutor(max_workers=1)
    pending_metric_jobs: list[tuple[Future, dict]] = []
    if metric_queue_limit is None:
        _cpu = os.cpu_count() or 2
        _workers = min(8, max(1, _cpu // 2))
        metric_queue_limit = max(1, _workers * 2)
    else:
        metric_queue_limit = max(1, int(metric_queue_limit))
    append_buffer: list[dict] = []
    append_flush_every = 5

    existing_rows, skip_paths = read_existing_results(save_path)
    _write_result_rows(save_path, existing_rows)

    indices = list(range(len(configured_dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    indices = indices[: min(num_videos, len(indices))]
    attack.prepare(
        model=model,
        dataset=configured_dataset,
        indices=indices,
        seed=seed,
        targeted=bool(target),
        pipeline_stage=str(pipeline_stage),
        verbose=bool(verbose),
    )

    logger = AttackLogger(
        log_path=log_path_path,
        attack_type=attack_name,
        model_name=getattr(model, "model_name", "unknown"),
        test_dataset=dataset.__class__.__name__,
        eps=getattr(attack, "eps", None),
        iters=getattr(attack, "steps", None),
    )

    all_rows = list(existing_rows)
    total_seen = len(existing_rows)
    clear_correct = 0
    attacked_success = 0
    target_success = 0
    time_sum = 0.0
    iter_sum = 0.0
    psnr_sum = 0.0
    vmaf_sum = 0.0
    metrics_num = 0

    for row in existing_rows:
        try:
            if row_is_clean_correct(row):
                clear_correct += 1
                attacked_success += int(int(row["clear_class"]) != int(row["attacked_class"]))
                target_success += int(int(row["target_class"]) >= 0 and int(row["target_class"]) == int(row["attacked_class"]))
                time_sum += float(row["time"])
                iter_sum += float(row["iter_count"])
                psnr_sum += float(row["psnr"])
                vmaf_sum += float(row["vmaf"])
                metrics_num += 1
        except Exception:
            pass

    start_total = time.time()
    processed_now = 0
    queued_attackable = 0
    skipped_misclassified = 0
    skipped_existing_count = 0
    skipped_error_count = 0

    def _flush_metric_jobs(*, block: bool, drain_all: bool = False) -> None:
        nonlocal all_rows, skip_paths
        nonlocal clear_correct, attacked_success, target_success
        nonlocal time_sum, iter_sum, psnr_sum, vmaf_sum, metrics_num, processed_now
        nonlocal skipped_error_count
        nonlocal append_buffer
        if not pending_metric_jobs:
            return
        if block:
            all_futures = [f for f, _ in pending_metric_jobs]
            if drain_all:
                wait(all_futures)
                done_futures = set(all_futures)
            else:
                done_futures, _ = wait(all_futures, return_when=FIRST_COMPLETED)
                done_futures = set(done_futures)
                if not done_futures:
                    return
        else:
            done_futures = {f for f, _ in pending_metric_jobs if f.done()}
            if not done_futures:
                return

        remaining: list[tuple[Future, dict]] = []
        for future, ctx in pending_metric_jobs:
            if future not in done_futures:
                remaining.append((future, ctx))
                continue
            try:
                metrics = future.result()
            except Exception as e:
                skipped_error_count += 1
                if verbose:
                    print(f"[attack][metrics-error] {ctx['sample_path']}: {type(e).__name__}: {e}", flush=True)
                    traceback.print_exc()
                metrics = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "vmaf": 0.0}

            row = dict(ctx["row"])
            row.update(metrics)
            all_rows.append(row)
            append_buffer.append(row)
            if len(append_buffer) >= append_flush_every:
                append_result_rows(save_path, append_buffer)
                append_buffer = []
            skip_paths.add(ctx["sample_path"])

            clean_label = int(ctx["clean_label"])
            attacked_label = int(ctx["attacked_label"])
            target_label = int(ctx["target_label"])
            elapsed = float(ctx["elapsed"])
            clear_correct += 1
            attacked_success += int(clean_label != attacked_label)
            target_success += int(ctx["is_targeted"] and attacked_label == target_label)
            time_sum += elapsed
            iter_sum += float(row["iter_count"])
            psnr_v = float(row["psnr"])
            vmaf_v = float(row["vmaf"])
            psnr_sum += float(psnr_v if psnr_v != float("inf") else 0.0)
            vmaf_sum += vmaf_v
            metrics_num += 1
            processed_now += 1

            if verbose:
                print(
                    f"[attack][ok] {ctx['sample_path']} clean={clean_label} attacked={attacked_label} "
                    f"target={target_label} clean_conf={float(row['clear_conf']):.4f} "
                    f"attacked_conf={float(row['attacked_conf']):.4f} benign_conf={float(row['benign_conf']):.4f} "
                    f"queries={int(row['query_count'])} time={elapsed:.2f}s",
                    flush=True,
                )

            if log_every and (processed_now % log_every == 0):
                wall_elapsed = time.time() - start_total
                mean_time = time_sum / max(clear_correct, 1)
                mean_iter = iter_sum / max(metrics_num, 1)
                mean_psnr = psnr_sum / max(metrics_num, 1)
                mean_vmaf = vmaf_sum / max(metrics_num, 1)
                logger(wall_elapsed, mean_time, mean_iter, mean_psnr, mean_vmaf, clear_correct, attacked_success, target_success, total_seen)

        pending_metric_jobs[:] = remaining

    for idx in tqdm(indices, total=len(indices), desc="attack"):
        _flush_metric_jobs(block=False)
        item = configured_dataset[idx]
        sample, sampled_video, input_format = _extract_sample_item(item)
        full_video_for_dump = item.full_tensor if isinstance(item, LoadedDatasetItem) else None
        sampled_frame_ids_for_dump = item.sampled_frame_ids if isinstance(item, LoadedDatasetItem) else None
        if skip_existing and sample.path in skip_paths:
            skipped_existing_count += 1
            if verbose:
                print(f"[attack][skip-existing] {sample.path}", flush=True)
            continue
        total_seen += 1
        t0 = time.time()
        model_device = getattr(model, "device", None)
        sampled_video_on_device = sampled_video.to(model_device) if model_device is not None else sampled_video

        try:
            if defence is not None and not adaptive:
                defence.install(model)
            try:
                clean_probs = model.predict(sampled_video_on_device, input_format=input_format)
            finally:
                if defence is not None and not adaptive:
                    defence.uninstall(model)
        except Exception as e:
            skipped_error_count += 1
            if verbose:
                print(f"[attack][clean-predict-error] {sample.path}: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        clean_label = int(clean_probs.argmax().item())
        clean_conf = _safe_conf(clean_probs, clean_label)
        gt_class = int(sample.label) if sample.label is not None else -1

        if (not allow_misclassified) and gt_class >= 0 and clean_label != gt_class:
            skipped_misclassified += 1
            elapsed = time.time() - t0
            skipped_row = {
                "video_name": sample.path,
                "gt_label": sample.label_name or _label_name(dataset, gt_class),
                "gt_class": gt_class,
                "clear_label": _label_name(dataset, clean_label),
                "clear_class": clean_label,
                "clear_conf": clean_conf,
                "target_label": "unknown",
                "target_class": -1,
                "attacked_label": _label_name(dataset, clean_label),
                "attacked_class": clean_label,
                "attacked_conf": clean_conf,
                "benign_conf": clean_conf,
                "psnr": 0.0,
                "ssim": 0.0,
                "mse": 0.0,
                "lpips": 0.0,
                "vmaf": 0.0,
                "time": elapsed,
                "iter_count": 0,
                "query_count": 0,
            }
            all_rows.append(skipped_row)
            append_buffer.append(skipped_row)
            if len(append_buffer) >= append_flush_every:
                append_result_rows(save_path, append_buffer)
                append_buffer = []
            skip_paths.add(sample.path)
            if verbose:
                print(
                    f"[attack][skip-misclassified] {sample.path} gt={gt_class} clean={clean_label} conf={clean_conf:.4f}",
                    flush=True,
                )
            continue

        target_label = -1
        y = None

        query_counter = {"count": 0}
        try:
            if isinstance(attack, BaseQueryVideoAttack):
                attacked_video = attack.attack_query(
                    model,
                    sampled_video_on_device,
                    query_label=lambda video: _predict_label_counted(
                        model,
                        video,
                        input_format=input_format,
                        counter=query_counter,
                    ),
                    input_format=input_format,
                    y=y,
                    targeted=target,
                    video_path=sample.path,
                )
            else:
                attacked_video = attack(model, sampled_video_on_device, input_format=input_format, y=y, targeted=target)
            if defence is not None and not adaptive:
                defence.install(model)
            try:
                attacked_probs = model.predict(attacked_video, input_format=input_format)
            finally:
                if defence is not None and not adaptive:
                    defence.uninstall(model)
        except Exception as e:
            skipped_error_count += 1
            if verbose:
                print(f"[attack][attack-error] {sample.path}: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        attacked_label = int(attacked_probs.argmax().item())
        attacked_conf = _safe_conf(attacked_probs, attacked_label)
        attacked_benign_conf = _safe_conf(attacked_probs, clean_label)
        elapsed = time.time() - t0
        attack_meta = dict(getattr(attack, "last_result", {}) or {})
        if target:
            target_label = int(attack_meta.get("target_label", _pick_target_label(clean_label, dataset.num_classes())))
        queued_attackable += 1
        should_dump = bool(dump_path_path is not None and dump_freq and (queued_attackable % dump_freq == 0))
        video_stem = Path(sample.path).stem
        dump_video_dir = (dump_path_path / video_stem) if should_dump else None
        frames_csv = save_path.with_name(save_path.stem + "_frames.csv") if calc_frame_metrics and dump_path_path is not None else None

        row = {
            "video_name": sample.path,
            "gt_label": sample.label_name or _label_name(dataset, gt_class),
            "gt_class": gt_class,
            "clear_label": _label_name(dataset, clean_label),
            "clear_class": clean_label,
            "clear_conf": clean_conf,
            "target_label": _label_name(dataset, target_label),
            "target_class": target_label,
            "attacked_label": _label_name(dataset, attacked_label),
            "attacked_class": attacked_label,
            "attacked_conf": attacked_conf,
            "benign_conf": float(attack_meta.get("benign_conf", attacked_benign_conf)),
            "psnr": 0.0,
            "ssim": 0.0,
            "mse": 0.0,
            "lpips": 0.0,
            "vmaf": 0.0,
            "time": elapsed,
            "iter_count": int(attack_meta.get("iter_count", getattr(attack, "steps", 0))),
            "query_count": int(attack_meta.get("query_count", query_counter["count"])),
        }

        sampled_video_cpu = sampled_video.detach().cpu()
        attacked_video_cpu = attacked_video.detach().cpu()
        full_video_cpu = (
            full_video_for_dump.detach().cpu()
            if should_dump and torch.is_tensor(full_video_for_dump)
            else None
        )
        sampled_frame_ids_cpu = (
            sampled_frame_ids_for_dump.detach().cpu()
            if should_dump and torch.is_tensor(sampled_frame_ids_for_dump)
            else None
        )
        defended_video_cpu: torch.Tensor | None = None
        if should_dump and save_defence_stages and defence is not None:
            try:
                with torch.no_grad():
                    defended_video_cpu = defence.transform(attacked_video.float()).detach().cpu()
            except Exception:
                defended_video_cpu = None
        pending_metric_jobs.append(
            (
                metric_executor.submit(
                    _compute_attack_artifacts_task,
                    sample_path=sample.path,
                    sampled_video_cpu=sampled_video_cpu,
                    attacked_video_cpu=attacked_video_cpu,
                    defended_video_cpu=defended_video_cpu,
                    full_video_cpu=full_video_cpu,
                    sampled_frame_ids_cpu=sampled_frame_ids_cpu,
                    clean_label=clean_label,
                    clean_label_name=_label_name(dataset, clean_label),
                    clean_conf=clean_conf,
                    attacked_label=attacked_label,
                    attacked_label_name=_label_name(dataset, attacked_label),
                    attacked_conf=attacked_conf,
                    dump_video_dir=dump_video_dir,
                    calc_vmaf=calc_vmaf,
                    calc_lpips=calc_lpips,
                    calc_frame_metrics=calc_frame_metrics,
                    frames_csv=frames_csv,
                ),
                {
                    "row": row,
                    "sample_path": sample.path,
                    "clean_label": clean_label,
                    "attacked_label": attacked_label,
                    "target_label": target_label,
                    "elapsed": elapsed,
                    "is_targeted": bool(target),
                },
            )
        )

        while len(pending_metric_jobs) >= metric_queue_limit:
            _flush_metric_jobs(block=True)

        del sampled_video_cpu, attacked_video_cpu, defended_video_cpu, full_video_cpu, sampled_frame_ids_cpu, attacked_video, attacked_probs, clean_probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _flush_metric_jobs(block=True, drain_all=True)
    metric_executor.shutdown(wait=True)
    if append_buffer:
        append_result_rows(save_path, append_buffer)
        append_buffer = []

    total_time = time.time() - start_total
    mean_time = time_sum / max(clear_correct, 1)
    mean_iter = iter_sum / max(metrics_num, 1)
    mean_psnr = psnr_sum / max(metrics_num, 1)
    mean_vmaf = vmaf_sum / max(metrics_num, 1)
    logger(total_time, mean_time, mean_iter, mean_psnr, mean_vmaf, clear_correct, attacked_success, target_success, total_seen)

    return {
        "model_name": getattr(model, "model_name", "unknown"),
        "dataset_name": dataset.__class__.__name__,
        "attack_name": attack_name,
        "targeted": bool(target),
        "processed": processed_now,
        "clear_correct": clear_correct,
        "attacked_success": attacked_success,
        "target_success": target_success,
        "total_seen": total_seen,
        "skipped_misclassified": skipped_misclassified,
        "skipped_existing": skipped_existing_count,
        "skipped_errors": skipped_error_count,
        "mean_time": mean_time,
        "mean_iterations": mean_iter,
        "mean_psnr": mean_psnr,
        "mean_vmaf": mean_vmaf,
        "save_path": str(save_path),
        "log_path": None if log_path_path is None else str(log_path_path),
        "dump_path": None if dump_path_path is None else str(dump_path_path),
    }


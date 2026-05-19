from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from vcr_bench.cli.common import resolve_model_selection_for_dataset
from vcr_bench.models import create_model
from vcr_bench.presets import (
    apply_run_preset_to_args,
    first_run_model,
    parse_overrides,
    resolve_entity_preset,
    resolve_run_preset,
)


VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".webm", ".mov", ".y4m", ".m4v"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify every video in a directory and save predictions."
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--run-preset", default=None, help="JSON run preset name or path")
    parser.add_argument("--model-preset", default=None, help="JSON model preset name or path")
    parser.add_argument("--model-preset-name", dest="model_preset_name", default=None)
    parser.add_argument("--model-variant", dest="model_preset_name", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--override", action="append", default=[], help="Preset override: dotted.key=json_value")
    parser.add_argument("--print-resolved-preset", action="store_true")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    parser.add_argument("--backbone", default=None, help="Model backbone name")
    parser.add_argument("--weights-dataset", default=None, help="Weights dataset name for the selected model/backbone")
    parser.add_argument("--grad-forward-chunk-size", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pipeline-stage", default="test", choices=["train", "val", "test", "attack"])
    parser.add_argument("--video-dir", required=True, help="Directory with input videos")
    parser.add_argument("--recursive", action="store_true", help="Search videos recursively")
    parser.add_argument("--suffix", action="append", default=None, help="Video suffix filter, e.g. .mp4")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of processed videos")
    parser.add_argument("--top-k", type=int, default=5, help="How many classes to store per video")
    parser.add_argument("--labels", default=None, help="Optional label map file, one class per line")
    parser.add_argument("--output-json", default=None, help="Write full results JSON; defaults to results/classify/<auto>.json")
    parser.add_argument("--output-csv", default=None, help="Write flattened CSV summary")
    parser.add_argument("--no-skip-errors", action="store_true", help="Raise on first per-video error instead of skipping")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _apply_presets(args: argparse.Namespace) -> dict[str, Any]:
    overrides = parse_overrides(args.override)
    resolved: dict[str, Any] = {}
    if args.run_preset:
        run = resolve_run_preset(
            args.run_preset,
            overrides=overrides.get("run") if isinstance(overrides.get("run"), dict) else None,
        )
        apply_run_preset_to_args(args, run)
        resolved["run"] = run
        model_ref = first_run_model(run)
        if model_ref and not args.model_preset and not args.model:
            args.model_preset = model_ref.get("preset") or model_ref.get("name")
            args.model_preset_name = model_ref.get("preset_name", model_ref.get("variant", args.model_preset_name))
    if args.model_preset:
        model_overrides = overrides.get("model") if isinstance(overrides.get("model"), dict) else None
        model_spec = resolve_entity_preset(
            "model",
            args.model_preset,
            preset_name=args.model_preset_name,
            overrides=model_overrides,
        )
        resolved["model"] = model_spec
        args.model = args.model or str(model_spec["factory_name"])
        for key, value in model_spec.get("params", {}).items():
            attr = key.replace("-", "_")
            if hasattr(args, attr):
                current = getattr(args, attr)
                if current is None or current is False or (attr == "device" and current == "cuda"):
                    setattr(args, attr, value)
    if args.model is None:
        raise SystemExit("--model is required unless provided by --model-preset or --run-preset")
    return resolved


def _normalize_suffixes(raw_suffixes: list[str] | None) -> set[str]:
    if not raw_suffixes:
        return set(VIDEO_SUFFIXES)
    suffixes: set[str] = set()
    for item in raw_suffixes:
        text = str(item).strip().lower()
        if not text:
            continue
        suffixes.add(text if text.startswith(".") else f".{text}")
    return suffixes or set(VIDEO_SUFFIXES)


def _collect_videos(video_dir: Path, *, recursive: bool, suffixes: set[str]) -> list[Path]:
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    if not video_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {video_dir}")
    iterator = video_dir.rglob("*") if recursive else video_dir.glob("*")
    return sorted(
        path for path in iterator
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _load_labels(path: str | None) -> list[str] | None:
    if not path:
        return None
    labels_path = Path(path)
    labels: list[str] = []
    with labels_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) > 1 and parts[0].isdigit():
                labels.append(" ".join(parts[1:]))
            else:
                labels.append(line)
    return labels


def _label_name(labels: list[str] | None, index: int) -> str | None:
    if labels is None or index < 0 or index >= len(labels):
        return None
    return labels[index]


def _default_output_json_path(model_name: str, video_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = video_dir.name or "videos"
    return Path("results") / "classify" / f"{model_name}_{slug}_{timestamp}.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_path",
        "video_name",
        "status",
        "pred_label",
        "pred_label_name",
        "pred_score",
        "top_k",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = build_parser().parse_args()
    resolved_preset = _apply_presets(args)
    if args.print_resolved_preset:
        print(json.dumps(resolved_preset, indent=2, ensure_ascii=False))
        return

    video_dir = Path(args.video_dir).resolve()
    suffixes = _normalize_suffixes(args.suffix)
    video_paths = _collect_videos(video_dir, recursive=args.recursive, suffixes=suffixes)
    if args.limit is not None:
        video_paths = video_paths[: max(0, args.limit)]
    if not video_paths:
        raise SystemExit(f"No videos found in {video_dir} for suffixes: {sorted(suffixes)}")

    labels = _load_labels(args.labels)
    resolve_model_selection_for_dataset(args)
    model = create_model(
        args.model,
        checkpoint_path=args.checkpoint,
        backbone=args.backbone,
        weights_dataset=args.weights_dataset,
        grad_forward_chunk_size=args.grad_forward_chunk_size,
        device=args.device,
    )
    model.build_data_pipeline(args.pipeline_stage)

    output_json_path = Path(args.output_json) if args.output_json else _default_output_json_path(args.model, video_dir)
    output_csv_path = Path(args.output_csv) if args.output_csv else None

    results: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    skipped = 0

    for video_path in tqdm(video_paths, desc="classify"):
        try:
            video_tensor = model.load_video(str(video_path))
            pred = model.predict(video_tensor, input_format="NTHWC", return_full=True)
            probs = pred.probs.detach().cpu()
            k = min(max(1, args.top_k), int(probs.numel()))
            top_scores, top_indices = probs.topk(k)
            top_k = [
                {
                    "label": int(label_idx),
                    "label_name": _label_name(labels, int(label_idx)),
                    "score": float(score),
                }
                for score, label_idx in zip(top_scores.tolist(), top_indices.tolist())
            ]
            top1 = top_k[0]
            result = {
                "video_path": str(video_path),
                "video_name": video_path.name,
                "status": "ok",
                "pred_label": top1["label"],
                "pred_label_name": top1["label_name"],
                "pred_score": top1["score"],
                "top_k": top_k,
                "error": None,
            }
        except Exception as exc:
            if args.no_skip_errors:
                raise
            skipped += 1
            result = {
                "video_path": str(video_path),
                "video_name": video_path.name,
                "status": "error",
                "pred_label": None,
                "pred_label_name": None,
                "pred_score": None,
                "top_k": [],
                "error": str(exc),
            }
        results.append(result)
        csv_rows.append(
            {
                "video_path": result["video_path"],
                "video_name": result["video_name"],
                "status": result["status"],
                "pred_label": result["pred_label"],
                "pred_label_name": result["pred_label_name"],
                "pred_score": result["pred_score"],
                "top_k": json.dumps(result["top_k"], ensure_ascii=False),
                "error": result["error"],
            }
        )
        if args.verbose:
            print(json.dumps(result, ensure_ascii=False))

    payload = {
        "model": args.model,
        "backbone": args.backbone,
        "weights_dataset": args.weights_dataset,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "pipeline_stage": args.pipeline_stage,
        "video_dir": str(video_dir),
        "recursive": bool(args.recursive),
        "suffixes": sorted(suffixes),
        "labels_path": str(Path(args.labels).resolve()) if args.labels else None,
        "total_found": len(video_paths),
        "processed": len(results) - skipped,
        "skipped": skipped,
        "results": results,
    }
    _write_json(output_json_path, payload)
    if output_csv_path is not None:
        _write_csv(output_csv_path, csv_rows)

    summary = {
        "model": args.model,
        "total_found": len(video_paths),
        "processed": len(results) - skipped,
        "skipped": skipped,
        "output_json": str(output_json_path.resolve()),
        "output_csv": str(output_csv_path.resolve()) if output_csv_path is not None else None,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

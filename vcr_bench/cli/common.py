from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vcr_bench.datasets import create_dataset
from vcr_bench.models import create_model, get_model_options


@dataclass
class CliResolvedContext:
    preview_model: Any
    model_pipeline: Any
    dataset_kwargs: dict[str, Any]


def build_model_dataset_context(args: Any, *, pipeline_stage_override: str | None = None) -> CliResolvedContext:
    stage = pipeline_stage_override or getattr(args, "pipeline_stage", "test")
    preview_model = create_model(
        args.model,
        checkpoint_path=getattr(args, "checkpoint", None),
        backbone=getattr(args, "backbone", None),
        weights_dataset=getattr(args, "weights_dataset", None),
        grad_forward_chunk_size=getattr(args, "grad_forward_chunk_size", None),
        device="cpu",
        load_weights=False,
    )
    model_pipeline = preview_model.build_data_pipeline(stage)
    dataset_kwargs = dict(
        video_root=getattr(args, "video_root", None),
        annotations_csv=getattr(args, "annotations", None),
        labels_txt=getattr(args, "labels", None),
        dataset_subset=getattr(args, "dataset_subset", None),
        split=getattr(args, "split", "val"),
        clip_len=getattr(model_pipeline, "clip_len", None),
        num_clips=getattr(model_pipeline, "num_clips", None),
        frame_interval=getattr(model_pipeline, "frame_interval", 1),
        full_videos=bool(getattr(args, "full_videos", False)),
    )
    return CliResolvedContext(preview_model=preview_model, model_pipeline=model_pipeline, dataset_kwargs=dataset_kwargs)


def build_default_resolution_payload(
    *,
    args: Any,
    preview_model: Any,
    dataset_kwargs: dict[str, Any],
    extra_resolved: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_preview = create_dataset(args.dataset, **dataset_kwargs)
    try:
        model_options = get_model_options(args.model)
    except Exception:
        model_options = None
    payload = {
        "dataset": getattr(args, "dataset", None),
        "model": getattr(args, "model", None),
        "model_selection": {
            "backbone": getattr(preview_model, "backbone", getattr(args, "backbone", None)),
            "weights_dataset": getattr(preview_model, "weights_dataset", getattr(args, "weights_dataset", None)),
        },
        "resolved": {
            "dataset_subset": getattr(args, "dataset_subset", None),
            "video_root": str(dataset_preview.video_root),
            "annotations": str(dataset_preview.annotations_csv),
            "labels": str(dataset_preview.labels_txt),
            "checkpoint": str(getattr(preview_model, "checkpoint_path", getattr(args, "checkpoint", None))),
            "clip_len": getattr(dataset_preview, "clip_len", None),
            "num_clips": getattr(dataset_preview, "num_clips", None),
            "frame_interval": getattr(dataset_preview, "frame_interval", None),
            "full_videos": getattr(dataset_preview, "full_videos", None),
        },
    }
    if model_options is not None:
        payload["model_options"] = model_options
    if extra_resolved:
        payload["resolved"].update(extra_resolved)
    return payload


def print_defaults_payload(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2))


def print_model_options_payload(model_name: str) -> None:
    print(json.dumps(get_model_options(model_name), indent=2))


def write_json_output(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_single_row_csv_output(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(payload.keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(payload)

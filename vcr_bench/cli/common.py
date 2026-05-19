from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vcr_bench.artifacts import default_dataset_subset
from vcr_bench.datasets import create_dataset
from vcr_bench.models import create_model, get_model_options


@dataclass
class CliResolvedContext:
    preview_model: Any
    model_pipeline: Any
    dataset_kwargs: dict[str, Any]


_WEIGHTS_DATASET_ALIASES = {
    "ssv2": ("ssv2", "sthv2"),
    "sthv2": ("sthv2", "ssv2"),
    "something-something-v2": ("sthv2", "ssv2"),
    "something_something_v2": ("sthv2", "ssv2"),
}


def _weights_dataset_candidates(dataset: str | None) -> list[str]:
    if not dataset:
        return []
    raw = str(dataset).strip()
    lowered = raw.lower()
    aliases = _WEIGHTS_DATASET_ALIASES.get(lowered, (raw,))
    out: list[str] = []
    for value in aliases:
        if value and value not in out:
            out.append(value)
    return out


def _first_available_model_selection(weight_datasets: dict[str, list[str]], backbone: str | None) -> tuple[str | None, str | None]:
    if backbone and backbone in weight_datasets and weight_datasets.get(backbone):
        return backbone, str(weight_datasets[backbone][0])
    for candidate_backbone, datasets in weight_datasets.items():
        if datasets:
            return str(candidate_backbone), str(datasets[0])
    return backbone, None


def _find_weights_dataset(
    weight_datasets: dict[str, list[str]],
    weights_dataset: str,
    backbone: str | None = None,
) -> tuple[str | None, str | None]:
    if backbone:
        datasets = weight_datasets.get(backbone, [])
        if weights_dataset in datasets:
            return backbone, weights_dataset
        return None, None
    for candidate_backbone, datasets in weight_datasets.items():
        if weights_dataset in datasets:
            return str(candidate_backbone), weights_dataset
    return None, None


def resolve_model_selection_for_dataset(args: Any, *, warn: bool = True) -> None:
    model_name = getattr(args, "model", None)
    if not model_name:
        return
    try:
        options = get_model_options(model_name)
    except Exception:
        return

    raw_weight_datasets = options.get("weight_datasets", {}) or {}
    if not isinstance(raw_weight_datasets, dict) or not raw_weight_datasets:
        return
    weight_datasets: dict[str, list[str]] = {
        str(backbone): [str(ds) for ds in (datasets or [])]
        for backbone, datasets in raw_weight_datasets.items()
    }

    requested_backbone = getattr(args, "backbone", None)
    requested_weights = getattr(args, "weights_dataset", None)
    dataset_name = getattr(args, "dataset", None)
    dataset_candidates = _weights_dataset_candidates(dataset_name)

    selected_backbone: str | None = None
    selected_weights: str | None = None
    warning_reason = ""

    if requested_weights:
        selected_backbone, selected_weights = _find_weights_dataset(
            weight_datasets,
            str(requested_weights),
            str(requested_backbone) if requested_backbone else None,
        )
        if selected_weights is None and not requested_backbone:
            selected_backbone, selected_weights = _find_weights_dataset(weight_datasets, str(requested_weights))
        if selected_weights is None:
            selected_backbone, selected_weights = _first_available_model_selection(
                weight_datasets,
                str(requested_backbone) if requested_backbone else None,
            )
            warning_reason = f"requested weights_dataset={requested_weights!r} is unavailable"
        elif requested_backbone and selected_backbone != str(requested_backbone):
            warning_reason = f"requested backbone={requested_backbone!r} has no weights_dataset={requested_weights!r}"
    else:
        for candidate in dataset_candidates:
            selected_backbone, selected_weights = _find_weights_dataset(
                weight_datasets,
                candidate,
                str(requested_backbone) if requested_backbone else None,
            )
            if selected_weights is not None:
                break
        if selected_weights is None and not requested_backbone:
            for candidate in dataset_candidates:
                selected_backbone, selected_weights = _find_weights_dataset(weight_datasets, candidate)
                if selected_weights is not None:
                    break
        if selected_weights is None:
            selected_backbone, selected_weights = _first_available_model_selection(
                weight_datasets,
                str(requested_backbone) if requested_backbone else None,
            )
            if dataset_name:
                warning_reason = f"no checkpoint is configured for dataset={dataset_name!r}"

    if selected_backbone:
        setattr(args, "backbone", selected_backbone)
    if selected_weights:
        setattr(args, "weights_dataset", selected_weights)

    if (
        warn
        and selected_weights
        and dataset_name
        and selected_weights not in dataset_candidates
    ):
        reason = f" ({warning_reason})" if warning_reason else ""
        print(
            "WARNING: "
            f"model {model_name!r} has no checkpoint matching dataset {dataset_name!r}{reason}; "
            f"using backbone={selected_backbone!r}, weights_dataset={selected_weights!r}. "
            "Predicted class names may belong to the weights dataset, not the evaluation dataset.",
            file=sys.stderr,
            flush=True,
        )


def apply_default_dataset_subset(args: Any) -> None:
    if (
        getattr(args, "dataset", None)
        and getattr(args, "dataset_subset", None) is None
        and not any(getattr(args, attr, None) for attr in ("video_root", "annotations", "labels"))
    ):
        subset = default_dataset_subset(getattr(args, "dataset"), getattr(args, "split", "val"))
        if subset:
            setattr(args, "dataset_subset", subset)


def build_dataset_kwargs_from_pipeline(args: Any, model_pipeline: Any) -> dict[str, Any]:
    apply_default_dataset_subset(args)
    return dict(
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


def build_dataset_kwargs_for_model(args: Any, model: Any, *, pipeline_stage_override: str | None = None) -> tuple[Any, dict[str, Any]]:
    stage = pipeline_stage_override or getattr(args, "pipeline_stage", "test")
    model_pipeline = model.build_data_pipeline(stage)
    return model_pipeline, build_dataset_kwargs_from_pipeline(args, model_pipeline)


def build_model_dataset_context(args: Any, *, pipeline_stage_override: str | None = None) -> CliResolvedContext:
    stage = pipeline_stage_override or getattr(args, "pipeline_stage", "test")
    apply_default_dataset_subset(args)
    resolve_model_selection_for_dataset(args)
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
    dataset_kwargs = build_dataset_kwargs_from_pipeline(args, model_pipeline)
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

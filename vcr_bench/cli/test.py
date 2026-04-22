from __future__ import annotations

import argparse
import json
from typing import Any

from vcr_bench.cli.common import (
    build_default_resolution_payload,
    build_model_dataset_context,
    print_model_options_payload,
    print_defaults_payload,
    write_single_row_csv_output,
    write_json_output,
)
from vcr_bench.datasets import create_dataset
from vcr_bench.utils.eval import run_accuracy
from vcr_bench.utils.vram import VramProfileContext, append_vram_profile_csv, profile_model_vram_for_one_video
from vcr_bench.models import create_model
from vcr_bench.presets import (
    apply_run_preset_to_args,
    first_run_model,
    parse_overrides,
    resolve_entity_preset,
    resolve_run_preset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vcr_bench accuracy test (phase 1)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--run-preset", default=None, help="JSON run preset name or path")
    parser.add_argument("--model-preset", default=None, help="JSON model preset name or path")
    parser.add_argument("--model-variant", default=None)
    parser.add_argument("--override", action="append", default=[], help="Preset override: dotted.key=json_value")
    parser.add_argument("--print-resolved-preset", action="store_true")
    parser.add_argument("--dataset-subset", default=None, help="Named dataset subset manifest to auto-download and resolve")
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    parser.add_argument("--backbone", default=None, help="Model backbone variant")
    parser.add_argument("--weights-dataset", default=None, help="Weights dataset variant for the selected model/backbone")
    parser.add_argument("--grad-forward-chunk-size", type=int, default=None, help="Chunk views during gradient forward to reduce VRAM (mainly attacks)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-videos", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dataset-loading", action="store_true", help="Decode/preprocess inside Dataset using model data pipeline")
    parser.add_argument("--instant-preprocessing", action="store_true", help="Also run model preprocessing inside Dataset (default off)")
    parser.add_argument("--full-videos", action="store_true", help="Keep full decoded video tensor in dataset items")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pipeline-stage", default="test", choices=["train", "val", "test"])
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None, help="Write one-row summary CSV")
    parser.add_argument("--vram-profile-csv", default=None, help="Append one-video VRAM profile rows to a separate CSV")
    parser.add_argument("--vram-profile-index", type=int, default=None, help="Dataset index used for --vram-profile-csv; defaults to seeded random")
    parser.add_argument("--return-full", action="store_true", help="Use rich output internally (debug)")
    parser.add_argument("--no-skip-errors", action="store_true", help="Raise on first per-sample error instead of skipping")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--print-defaults", action="store_true", help="Print resolved defaults and exit")
    parser.add_argument("--list-model-options", action="store_true", help="Print available backbones/weights datasets for --model and exit")
    return parser


def _apply_presets(args: argparse.Namespace) -> dict[str, Any]:
    overrides = parse_overrides(args.override)
    resolved: dict[str, Any] = {}
    run = None
    if args.run_preset:
        run = resolve_run_preset(args.run_preset, overrides=overrides.get("run") if isinstance(overrides.get("run"), dict) else None)
        apply_run_preset_to_args(args, run)
        resolved["run"] = run
        model_ref = first_run_model(run)
        if model_ref and not args.model_preset and not args.model:
            args.model_preset = model_ref.get("preset") or model_ref.get("name")
            args.model_variant = model_ref.get("variant", args.model_variant)
    if args.model_preset:
        model_overrides = overrides.get("model") if isinstance(overrides.get("model"), dict) else None
        model_spec = resolve_entity_preset("model", args.model_preset, variant=args.model_variant, overrides=model_overrides)
        resolved["model"] = model_spec
        args.model = args.model or str(model_spec["factory_name"])
        for key, value in model_spec.get("params", {}).items():
            attr = key.replace("-", "_")
            if hasattr(args, attr):
                current = getattr(args, attr)
                if current is None or current is False or (attr == "device" and current == "cuda"):
                    setattr(args, attr, value)
    if args.dataset is None:
        raise SystemExit("--dataset is required unless provided by --run-preset")
    if args.model is None:
        raise SystemExit("--model is required unless provided by --model-preset or --run-preset")
    return resolved


def main() -> None:
    args = build_parser().parse_args()
    resolved_preset = _apply_presets(args)
    if args.print_resolved_preset:
        print(json.dumps(resolved_preset, indent=2, sort_keys=True))
        return
    if args.list_model_options:
        print_model_options_payload(args.model)
        return
    ctx = build_model_dataset_context(args)
    preview_model = ctx.preview_model
    dataset_kwargs = ctx.dataset_kwargs

    if args.print_defaults:
        print_defaults_payload(
            build_default_resolution_payload(
                args=args,
                preview_model=preview_model,
                dataset_kwargs=dataset_kwargs,
            )
        )
        return

    dataset = create_dataset(args.dataset, **dataset_kwargs)
    model = create_model(
        args.model,
        checkpoint_path=args.checkpoint,
        backbone=args.backbone,
        weights_dataset=args.weights_dataset,
        grad_forward_chunk_size=args.grad_forward_chunk_size,
        device=args.device,
    )
    if args.vram_profile_csv:
        rows = profile_model_vram_for_one_video(
            model=model,
            dataset=dataset,
            context=VramProfileContext(
                model_arg=args.model,
                dataset_arg=args.dataset,
                dataset_subset=args.dataset_subset,
                backbone=args.backbone,
                weights_dataset=args.weights_dataset,
                pipeline_stage=args.pipeline_stage,
                seed=args.seed,
                sample_index=args.vram_profile_index,
            ),
        )
        append_vram_profile_csv(args.vram_profile_csv, rows)
    if args.verbose and hasattr(model, "preprocessed_format"):
        print(f"model={args.model} raw={model.raw_input_format} preprocessed={model.preprocessed_format}")
    summary = run_accuracy(
        model=model,
        dataset=dataset,
        num_videos=args.num_videos,
        seed=args.seed,
        return_full=args.return_full,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset_loading=args.dataset_loading,
        instant_preprocessing=args.instant_preprocessing,
        pipeline_stage=args.pipeline_stage,
        skip_decode_errors=not args.no_skip_errors,
    )
    print(json.dumps(summary, indent=2))
    write_json_output(args.output_json, summary)
    write_single_row_csv_output(args.output_csv, summary)


if __name__ == "__main__":
    main()

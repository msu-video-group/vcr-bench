from __future__ import annotations

import argparse
import json

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
from vcr_bench.models import create_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vcr_bench accuracy test (phase 1)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
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
    parser.add_argument("--return-full", action="store_true", help="Use rich output internally (debug)")
    parser.add_argument("--no-skip-errors", action="store_true", help="Raise on first per-sample error instead of skipping")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--print-defaults", action="store_true", help="Print resolved defaults and exit")
    parser.add_argument("--list-model-options", action="store_true", help="Print available backbones/weights datasets for --model and exit")
    return parser


def main() -> None:
    args = build_parser().parse_args()
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

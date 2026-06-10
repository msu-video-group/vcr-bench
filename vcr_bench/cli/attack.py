from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Defences that run a diffusion model per frame; for these we cap classifier
# frame sampling (see _apply_diffusion_frame_cap in models/base.py).
DIFFUSION_DEFENCES = {"freqpure", "videopure"}
DEFAULT_DIFFUSION_MAX_FRAMES = 32

from vcr_bench.attacks import create_attack, get_attack_spec
from vcr_bench.cli.common import (
    build_dataset_kwargs_for_model,
    build_default_resolution_payload,
    build_model_dataset_context,
    print_model_options_payload,
    print_defaults_payload,
    resolve_model_selection_for_dataset,
    write_json_output,
)
from vcr_bench.datasets import create_dataset
from vcr_bench.models import create_model
from vcr_bench.presets import (
    apply_run_preset_to_args,
    first_run_attack,
    first_run_defence,
    first_run_model,
    parse_overrides,
    resolve_entity_preset,
    resolve_run_preset,
)
from vcr_bench.utils.eval import run_attack
from vcr_bench.utils.vram import VramProfileContext, append_vram_profile_csv, profile_model_vram_for_one_video


def build_base_parser(*, add_help: bool) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="vcr_bench attack eval", add_help=add_help)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--num-workers", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--attack-name", type=str, default="attacks", help="Name for the logging root.")
    p.add_argument("--model", default=None)
    p.add_argument("--attack", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--run-preset", default=None, help="JSON run preset name or path")
    p.add_argument("--model-preset", default=None, help="JSON model preset name or path")
    p.add_argument("--model-preset-name", dest="model_preset_name", default=None)
    p.add_argument("--model-variant", dest="model_preset_name", default=None, help=argparse.SUPPRESS)
    p.add_argument("--attack-preset", default=None, help="JSON attack preset name or path")
    p.add_argument("--attack-preset-name", dest="attack_preset_name", default=None)
    p.add_argument("--attack-variant", dest="attack_preset_name", default=None, help=argparse.SUPPRESS)
    p.add_argument("--defence-preset", default=None, help="JSON defence preset name or path")
    p.add_argument("--defence-preset-name", dest="defence_preset_name", default=None)
    p.add_argument("--defence-variant", dest="defence_preset_name", default=None, help=argparse.SUPPRESS)
    p.add_argument("--override", action="append", default=[], help="Preset override: dotted.key=json_value")
    p.add_argument("--print-resolved-preset", action="store_true")
    p.add_argument("--print-attack-spec", action="store_true", help="Print the resolved attack option schema and exit.")
    p.add_argument("--dataset-subset", default=None, help="Named dataset subset manifest to auto-download and resolve")
    p.add_argument("--video-root", default=None)
    p.add_argument("--annotations", default=None)
    p.add_argument("--labels", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--backbone", default=None, help="Model backbone name")
    p.add_argument("--weights-dataset", default=None, help="Weights dataset name for the selected model/backbone")
    p.add_argument("--grad-forward-chunk-size", type=int, default=None, help="Chunk views during gradient forward to reduce VRAM")
    p.add_argument("--num-videos", type=int, default=25)
    p.add_argument("--target", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--allow-misclassified",
        dest="allow_misclassified",
        action="store_true",
        default=True,
        help="Attack all videos, including samples the clean model misclassifies. This is the default.",
    )
    p.add_argument(
        "--skip-misclassified",
        dest="allow_misclassified",
        action="store_false",
        help="Only attack videos that the clean model classifies correctly.",
    )
    p.add_argument("--full-videos", action="store_true")
    p.add_argument("--split", default="test")
    p.add_argument("--pipeline-stage", default="test", choices=["train", "val", "test"])
    p.add_argument("--lite-attack", action="store_true", help="Use lite attack pipeline (no three-crop) when model supports it")
    p.add_argument("--dump-freq", type=int, default=0)
    p.add_argument("--save-defence-stages", action="store_true", help="When dumping videos with a defence, also save the pre-defence attacked video and the post-defence defended video separately")
    p.add_argument("--vmaf", dest="vmaf", action="store_true", default=True, help=argparse.SUPPRESS)
    p.add_argument("--no-vmaf", dest="vmaf", action="store_false", help="Disable VMAF metric calculation")
    p.add_argument("--lpips", dest="lpips", action="store_true", default=True, help=argparse.SUPPRESS)
    p.add_argument("--no-lpips", dest="lpips", action="store_false", help="Disable LPIPS metric calculation")
    p.add_argument("--framewise-metrics", action="store_true")
    p.add_argument("--metric-workers", default="auto", help="Parallel artifact/VMAF workers: auto or a positive integer. Default: auto.")
    p.add_argument("--defence", default=None, help="Defence name to apply.")
    p.add_argument("--adaptive", action="store_true", help="Apply defence adaptively before attack gradient computation.")
    p.add_argument(
        "--diffusion-max-frames",
        type=int,
        default=None,
        help="Diffusion-defence mode: cap classifier sampling to roughly this many frames "
        "(reduce-only num_clips). Default: auto (32) for diffusion defences (freqpure/videopure), "
        "off otherwise. Pass 0 to disable.",
    )
    p.add_argument("--separate-logs", action="store_true")
    p.add_argument("--comment", default="")
    p.add_argument("--results-root", default="results", help="Root folder for attack result CSV outputs")
    p.add_argument("--logs-root", default="attack_logs", help="Root folder for attack summary CSV/stdout logs")
    p.add_argument("--artifacts-root", dest="results_root", help="Deprecated alias for --results-root")
    p.add_argument("--vram-profile-csv", default=None, help="Append VRAM profile rows to a separate CSV, including attack peaks")
    p.add_argument("--vram-profile-index", type=int, default=None, help="Dataset index used for --vram-profile-csv; defaults to seeded random")
    p.add_argument("--output-json", default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--print-defaults", action="store_true")
    p.add_argument("--list-model-options", action="store_true", help="Print available backbones/weights datasets for --model and exit")
    return p


def _add_attack_spec_args(parser: argparse.ArgumentParser, attack_name: str | None) -> None:
    if not attack_name:
        return
    spec = get_attack_spec(attack_name)
    group = parser.add_argument_group(f"{spec.attack_name} options")
    for option in spec.options:
        kwargs: dict[str, Any] = {
            "dest": option.param_name,
            "default": None,
            "help": option.help_text,
        }
        if option.boolean:
            group.add_argument(*option.option_strings, action="store_true", **kwargs)
            primary = option.option_strings[0]
            if primary.startswith("--"):
                group.add_argument(
                    f"--no-{primary[2:]}",
                    dest=option.param_name,
                    default=None,
                    action="store_false",
                    help=f"Disable {option.param_name.replace('_', ' ')}.",
                )
        else:
            if option.value_type is not str:
                kwargs["type"] = option.value_type
            if option.metavar:
                kwargs["metavar"] = option.metavar
            if option.choices:
                kwargs["choices"] = option.choices
            group.add_argument(*option.option_strings, **kwargs)


def _apply_presets(args: argparse.Namespace, *, require_complete: bool) -> dict[str, Any]:
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
        attack_ref = first_run_attack(run)
        defence_ref = first_run_defence(run)
        if model_ref and not args.model_preset and not args.model:
            args.model_preset = model_ref.get("preset") or model_ref.get("name")
            args.model_preset_name = model_ref.get("preset_name", model_ref.get("variant", args.model_preset_name))
        if attack_ref and not args.attack_preset and not args.attack:
            args.attack_preset = attack_ref.get("preset") or attack_ref.get("name")
            args.attack_preset_name = attack_ref.get("preset_name", attack_ref.get("variant", args.attack_preset_name))
        if defence_ref and not args.defence_preset and not args.defence:
            args.defence_preset = defence_ref.get("preset") or defence_ref.get("name")
            args.defence_preset_name = defence_ref.get("preset_name", defence_ref.get("variant", args.defence_preset_name))
    if args.model_preset:
        model_spec = resolve_entity_preset(
            "model",
            args.model_preset,
            preset_name=args.model_preset_name,
            overrides=overrides.get("model") if isinstance(overrides.get("model"), dict) else None,
        )
        resolved["model"] = model_spec
        args.model = args.model or str(model_spec["factory_name"])
        for key, value in model_spec.get("params", {}).items():
            attr = key.replace("-", "_")
            if hasattr(args, attr):
                current = getattr(args, attr)
                if current is None or current is False or (attr == "device" and current == "cuda"):
                    setattr(args, attr, value)
    if args.attack_preset:
        attack_spec = resolve_entity_preset(
            "attack",
            args.attack_preset,
            preset_name=args.attack_preset_name,
            overrides=overrides.get("attack") if isinstance(overrides.get("attack"), dict) else None,
        )
        resolved["attack"] = attack_spec
        args.attack = args.attack or str(attack_spec["factory_name"])
        params = attack_spec.get("params", {})
        for key, value in params.items():
            attr = key.replace("-", "_")
            if hasattr(args, attr):
                current = getattr(args, attr)
                if current is None or current is False:
                    setattr(args, attr, value)
    if args.defence_preset:
        defence_spec = resolve_entity_preset(
            "defence",
            args.defence_preset,
            preset_name=args.defence_preset_name,
            overrides=overrides.get("defence") if isinstance(overrides.get("defence"), dict) else None,
        )
        resolved["defence"] = defence_spec
        args.defence = args.defence or str(defence_spec["factory_name"])
        args.defence_params = dict(defence_spec.get("params", {}) or {})
    if require_complete:
        if args.dataset is None:
            raise SystemExit("--dataset is required unless provided by --run-preset")
        if args.model is None:
            raise SystemExit("--model is required unless provided by --model-preset or --run-preset")
        if args.attack is None:
            raise SystemExit("--attack is required unless provided by --attack-preset or --run-preset")
    return resolved


def _build_paths(args: argparse.Namespace, model_name: str) -> tuple[Path, Path, Path]:
    attack_root_name = args.attack_name or args.attack
    attack_type = "target" if args.target else "untarget"
    defence_name = args.defence or "no_defence"
    defence_type = "adaptive" if (args.adaptive and args.defence is not None) else "non-adaptive"
    save_attack_name = f"{args.attack}_{attack_type}_{defence_name}_{defence_type}"
    comment = args.comment.strip()
    if comment:
        safe_comment = comment.replace(" ", "_").replace("/", "_").replace("\\", "_")
        save_attack_name = f"{save_attack_name}_{safe_comment}"

    results_root = Path(args.results_root)
    save_path = results_root / attack_root_name / save_attack_name / f"{model_name}.csv"
    if args.separate_logs:
        log_path = results_root / attack_root_name / save_attack_name / f"log_{save_attack_name}.csv"
    else:
        log_path = results_root / attack_root_name / f"log_{attack_root_name}.csv"
    dump_path = results_root / "attacked_videos" / attack_root_name / save_attack_name / model_name
    return save_path, log_path, dump_path


def _build_attack_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if args.attack is None:
        return {}
    spec = get_attack_spec(args.attack)
    kwargs: dict[str, Any] = {}
    for option in spec.options:
        value = getattr(args, option.param_name, None)
        if value is not None:
            kwargs[option.param_name] = value
    return kwargs


def _attack_spec_payload(attack_name: str) -> dict[str, Any]:
    spec = get_attack_spec(attack_name)
    return {
        "attack": spec.attack_name,
        "class_name": spec.attack_class.__name__,
        "options": [
            {
                "param": option.param_name,
                "flags": list(option.option_strings),
                "type": option.value_type.__name__ if hasattr(option.value_type, "__name__") else str(option.value_type),
                "default": option.default,
                "boolean": option.boolean,
                "help": option.help_text,
            }
            for option in spec.options
        ],
    }


def parse_attack_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, dict[str, Any]]:
    argv = list(sys.argv[1:] if argv is None else argv)
    pre_parser = build_base_parser(add_help=False)
    pre_args, _ = pre_parser.parse_known_args(argv)
    _apply_presets(pre_args, require_complete=False)
    parser = build_base_parser(add_help=True)
    _add_attack_spec_args(parser, getattr(pre_args, "attack", None))
    args = parser.parse_args(argv)
    require_complete = not bool(getattr(args, "print_attack_spec", False))
    resolved = _apply_presets(args, require_complete=require_complete)
    return args, resolved


def main(argv: list[str] | None = None) -> None:
    args, resolved_preset = parse_attack_args(argv)
    if args.print_resolved_preset:
        print(json.dumps(resolved_preset, indent=2, sort_keys=True))
        return
    if args.print_attack_spec:
        if args.attack is None:
            raise SystemExit("--attack or --attack-preset is required for --print-attack-spec")
        print(json.dumps(_attack_spec_payload(args.attack), indent=2, sort_keys=True))
        return
    if args.list_model_options:
        print_model_options_payload(args.model)
        return

    effective_stage = "attack" if args.lite_attack else args.pipeline_stage

    # Diffusion-defence mode: cap classifier frame sampling so the per-frame
    # diffusion purifier isn't fed the full multi-clip test sampling (up to 320
    # frames). Auto-on for diffusion defences; --diffusion-max-frames overrides
    # (0 disables). Consumed via env by models.base._apply_diffusion_frame_cap.
    diffusion_cap = args.diffusion_max_frames
    if diffusion_cap is None:
        is_diffusion = (args.defence or "").strip().lower() in DIFFUSION_DEFENCES
        diffusion_cap = DEFAULT_DIFFUSION_MAX_FRAMES if is_diffusion else 0
    if diffusion_cap and diffusion_cap > 0:
        os.environ["VCR_BENCH_DIFFUSION_MAX_FRAMES"] = str(int(diffusion_cap))
        print(f"[attack] diffusion-defence mode: capping sampling to ~{int(diffusion_cap)} frames")

    if args.print_defaults:
        ctx = build_model_dataset_context(args, pipeline_stage_override=effective_stage)
        preview_model = ctx.preview_model
        dataset_kwargs = ctx.dataset_kwargs
        save_path, log_path, dump_path = _build_paths(args, getattr(preview_model, "model_name", args.model))
        payload = build_default_resolution_payload(
            args=args,
            preview_model=preview_model,
            dataset_kwargs=dataset_kwargs,
            extra_resolved={
                "save_path": str(save_path),
                "log_path": str(log_path),
                "dump_path": str(dump_path),
                "pipeline_stage": effective_stage,
                "lite_attack": bool(args.lite_attack),
                "attack_kwargs": _build_attack_kwargs(args),
            },
        )
        print_defaults_payload(payload)
        return

    attack = create_attack(args.attack, **_build_attack_kwargs(args))

    defence = None
    if args.defence is not None:
        from vcr_bench.defences import create_defence

        defence = create_defence(args.defence, **dict(getattr(args, "defence_params", {}) or {}))

    resolve_model_selection_for_dataset(args)
    model = create_model(
        args.model,
        checkpoint_path=args.checkpoint,
        backbone=args.backbone,
        weights_dataset=args.weights_dataset,
        grad_forward_chunk_size=args.grad_forward_chunk_size,
        device=args.device,
    )
    _, dataset_kwargs = build_dataset_kwargs_for_model(args, model, pipeline_stage_override=effective_stage)
    dataset = create_dataset(args.dataset, **dataset_kwargs)
    vram_context = None
    if args.vram_profile_csv:
        vram_context = VramProfileContext(
            model_arg=args.model,
            dataset_arg=args.dataset,
            dataset_subset=args.dataset_subset,
            backbone=args.backbone,
            weights_dataset=args.weights_dataset,
            pipeline_stage=effective_stage,
            seed=args.seed,
            sample_index=args.vram_profile_index,
        )
        rows = profile_model_vram_for_one_video(
            model=model,
            dataset=dataset,
            context=vram_context,
        )
        append_vram_profile_csv(args.vram_profile_csv, rows)

    save_path, log_path, dump_path = _build_paths(args, getattr(model, "model_name", args.model))

    if args.verbose:
        print(
            f"model={args.model} attack={args.attack} dataset={args.dataset} "
            f"raw={getattr(model, 'raw_input_format', 'unknown')} pre={getattr(model, 'preprocessed_format', 'unknown')} "
            f"stage={effective_stage}"
        )

    summary = run_attack(
        model=model,
        attack=attack,
        dataset=dataset,
        save_path=save_path,
        log_path=log_path,
        attack_name=save_path.parent.name,
        num_videos=args.num_videos,
        seed=args.seed,
        target=args.target,
        allow_misclassified=args.allow_misclassified,
        pipeline_stage=effective_stage,
        instant_preprocessing=False,
        verbose=args.verbose,
        dump_freq=args.dump_freq,
        dump_path=dump_path,
        calc_vmaf=args.vmaf,
        calc_lpips=args.lpips,
        calc_frame_metrics=args.framewise_metrics,
        metric_workers=args.metric_workers,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        defence=defence,
        adaptive=args.adaptive,
        save_defence_stages=args.save_defence_stages,
        vram_profile_csv=args.vram_profile_csv,
        vram_profile_context=vram_context,
    )
    print(json.dumps(summary, indent=2))
    write_json_output(args.output_json, summary)


if __name__ == "__main__":
    main()

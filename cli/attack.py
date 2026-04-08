from __future__ import annotations

import argparse
import json
from pathlib import Path

from vg_videobench.attacks import create_attack
from vg_videobench.cli.common import (
    build_default_resolution_payload,
    build_model_dataset_context,
    print_model_options_payload,
    print_defaults_payload,
    write_json_output,
)
from vg_videobench.datasets import create_dataset
from vg_videobench.models import create_model
from vg_videobench.utils.eval import run_attack


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="vg_videobench attack eval")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--num-workers", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--attack-name", type=str, default="attacks", help="name for logging root")
    p.add_argument("--model", required=True)
    p.add_argument("--attack", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--video-root", default=None)
    p.add_argument("--annotations", default=None)
    p.add_argument("--labels", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--backbone", default=None, help="Model backbone variant")
    p.add_argument("--weights-dataset", default=None, help="Weights dataset variant for the selected model/backbone")
    p.add_argument("--grad-forward-chunk-size", type=int, default=None, help="Chunk views during gradient forward to reduce VRAM")
    p.add_argument("--num-videos", type=int, default=25)
    p.add_argument("--target", action="store_true")
    p.add_argument("--eps", type=float, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--iter", type=int, default=None)
    p.add_argument("--random-start", dest="random_start", action="store_true", default=None, help="Enable random start when the selected attack supports it")
    p.add_argument("--no-random-start", dest="random_start", action="store_false", help="Disable random start when the selected attack supports it")
    p.add_argument("--random-start-trials", type=int, default=None, help="GradEstV2/Square: number of random-start candidates to try before the main loop")
    p.add_argument("--beta", type=float, default=None, help="TenAd finite-difference smoothing parameter")
    p.add_argument("--binary-search-steps", type=int, default=None, help="TenAd binary-search refinement steps")
    p.add_argument("--max-expand-steps", type=int, default=None, help="TenAd geometric expansion steps")
    p.add_argument("--lambda-init", type=float, default=None, help="TenAd initial lambda for boundary search")
    p.add_argument("--query-budget", type=int, default=None, help="TenAd/GradEst/Square per-sample query budget")
    p.add_argument("--max-patch-frac", type=float, default=None, help="Square: cap patch size to this fraction of min(H,W) (default 0.2)")
    p.add_argument("--scout-queries", type=int, default=None, help="Square: number of coarse scout proposals before the main loop")
    p.add_argument("--elite-window-count", type=int, default=None, help="Square: number of best proposal windows to keep for reuse")
    p.add_argument("--elite-candidates-per-step", type=int, default=None, help="Square: number of per-step candidates sampled around elite windows")
    p.add_argument("--elite-jitter-frac", type=float, default=None, help="Square: relative jitter radius when resampling around elite windows")
    p.add_argument("--guide-samples", type=int, default=None, help="Square: NES guidance sample pairs per step before patch proposals")
    p.add_argument("--guide-sigma", type=float, default=None, help="Square: smoothing scale for NES guidance noise")
    p.add_argument("--guide-patch-prob", type=float, default=None, help="Square: probability of using guidance-derived patch sign instead of random sign")
    p.add_argument("--guide-momentum", type=float, default=None, help="Square: EMA momentum for the latent guidance field")
    p.add_argument("--candidates-per-step", type=int, default=None, help="Square: number of candidate patch mutations per early step")
    p.add_argument("--candidates-per-step-late", type=int, default=None, help="Square: number of candidate patch mutations per late step")
    p.add_argument("--bootstrap-guide-samples", type=int, default=None, help="Square: NES bootstrap sample pairs before random start")
    p.add_argument("--bootstrap-guide-sigma", type=float, default=None, help="Square: NES bootstrap smoothing scale")
    p.add_argument("--bootstrap-primitives", type=int, default=None, help="Square: number of bootstrap-guided primitives in the initial seed")
    p.add_argument("--eval-batch-size", type=int, default=None, help="Square: GPU eval chunk size for batched candidate queries")
    p.add_argument("--boost-second", type=float, default=None, help="GradEst/GradEstV2: weight on -CE(second_label) in NES loss (0 = disabled)")
    p.add_argument("--boost-top2", type=float, default=None, help="GradEstV2/Square: accept/reject bonus weight on improvement of mean top-2 non-primary logits (base weight; ramps to 4x by the final step)")
    p.add_argument("--boost-top10", type=float, default=None, help="GradEstV2/Square: accept/reject bonus weight on improvement of mean top-10 non-primary logits")
    p.add_argument("--boost-top5", type=float, default=None, help="Deprecated alias for GradEstV2/Square top-k bonus; used as boost-top10 when boost-top10 is omitted")
    p.add_argument("--temporal-tiles", type=int, default=None, help="GradEstV2/Square: number of independent temporal groups")
    p.add_argument("--smooth-scale", type=float, default=None, help="GradEstV2: noise generated at N*smooth_scale resolution, upsampled (0<scale<=1)")
    p.add_argument("--num-restarts", type=int, default=None, help="TenAd restart count")
    p.add_argument("--spatial-stride", type=int, default=None, help="TenAd spatial downsampling stride for rank-1 factors (default: auto = min(H,W)//32)")
    p.add_argument("--bmtc-save-root", default=None, help="BMTC artifacts directory (mixer checkpoint, background frames)")
    p.add_argument("--n-samples", type=int, default=None, help="StyleFool/query attack: NES samples per step")
    p.add_argument("--sub-num", type=int, default=None, help="StyleFool: antithetic batch size (must be even, divide n-samples)")
    p.add_argument("--lr-annealing", action="store_true", default=False, help="StyleFool: enable LR halving when confidence falls")
    p.add_argument("--style-steps", type=int, default=None, help="StyleFool: neural style transfer steps used during prepare (must match prepare --style-steps)")
    p.add_argument("--style-content-weight", type=float, default=None, help="StyleFool: content loss weight for neural style transfer")
    p.add_argument("--style-style-weight", type=float, default=None, help="StyleFool: style loss weight for neural style transfer")
    p.add_argument("--attack-sample-chunk-size", type=int, default=None, help="Process sampled clips in chunks during attack (memory-saving; disabled by default)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-misclassified", action="store_true")
    p.add_argument("--full-videos", action="store_true")
    p.add_argument("--split", default="val")
    p.add_argument("--pipeline-stage", default="test", choices=["train", "val", "test"])
    p.add_argument("--lite-attack", action="store_true", help="Use lite attack pipeline (no three-crop) when model supports it")
    p.add_argument("--dump-freq", type=int, default=0)  # reserved for compatibility
    p.add_argument("--save-defence-stages", action="store_true", help="When dumping videos with a defence, also save the pre-defence attacked video and the post-defence defended video separately")
    p.add_argument("--vmaf", dest="vmaf", action="store_true", default=True, help=argparse.SUPPRESS)
    p.add_argument("--no-vmaf", dest="vmaf", action="store_false", help="Disable VMAF metric calculation")
    p.add_argument("--lpips", dest="lpips", action="store_true", default=True, help=argparse.SUPPRESS)
    p.add_argument("--no-lpips", dest="lpips", action="store_false", help="Disable LPIPS metric calculation")
    p.add_argument("--framewise-metrics", action="store_true")  # reserved for compatibility
    p.add_argument("--defence", default=None, help="Defence name to apply (e.g. blur, temporal_median, rs)")
    p.add_argument("--adaptive", action="store_true", help="Apply defence adaptively (before attack gradient computation)")
    p.add_argument("--separate-logs", action="store_true")
    p.add_argument("--comment", default="")
    p.add_argument("--results-root", default="vg_videobench/results", help="Root folder for attack result CSV/log outputs")
    p.add_argument("--artifacts-root", dest="results_root", help="Deprecated alias for --results-root")
    p.add_argument("--output-json", default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--print-defaults", action="store_true")
    p.add_argument("--list-model-options", action="store_true", help="Print available backbones/weights datasets for --model and exit")
    return p


def _build_paths(args, model_name: str) -> tuple[Path, Path, Path]:
    attack_root_name = args.attack_name or args.attack
    attack_type = "target" if args.target else "untarget"
    defence_name = args.defence or "no_defence"
    defence_type = "adaptive" if (args.adaptive and args.defence is not None) else "non-adaptive"
    save_attack_name = f"{args.attack}_{attack_type}_{defence_name}_{defence_type}"
    comment = args.comment.strip()
    if comment:
        safe_comment = comment.replace(" ", "_").replace("/", "_").replace("\\", "_")
        save_attack_name = f"{save_attack_name}_{safe_comment}"

    root = Path(args.results_root)
    save_path = root / attack_root_name / save_attack_name / f"{model_name}.csv"
    if args.separate_logs:
        log_path = root / attack_root_name / save_attack_name / f"log_{model_name}.csv"
    else:
        log_path = root / attack_root_name / f"log_{attack_root_name}.csv"
    dump_path = root / "attacked_videos" / attack_root_name / save_attack_name / model_name
    return save_path, log_path, dump_path


def main() -> None:
    args = build_parser().parse_args()
    if args.list_model_options:
        print_model_options_payload(args.model)
        return
    effective_stage = "attack" if args.lite_attack else args.pipeline_stage
    ctx = build_model_dataset_context(args, pipeline_stage_override=effective_stage)
    preview_model = ctx.preview_model
    dataset_kwargs = ctx.dataset_kwargs

    if args.print_defaults:
        save_path, log_path, dump_path = _build_paths(args, getattr(preview_model, "model_name", args.model))
        print_defaults_payload(
            build_default_resolution_payload(
                args=args,
                preview_model=preview_model,
                dataset_kwargs=dataset_kwargs,
                extra_resolved={
                    "save_path": str(save_path),
                    "log_path": str(log_path),
                    "dump_path": str(dump_path),
                    "pipeline_stage": effective_stage,
                    "lite_attack": bool(args.lite_attack),
                },
            )
        )
        return

    attack_kwargs = {}
    if args.eps is not None:
        attack_kwargs["eps"] = args.eps
    if args.alpha is not None:
        attack_kwargs["alpha"] = args.alpha
    if args.iter is not None:
        attack_kwargs["steps"] = args.iter
    if getattr(args, "random_start", None) is not None and str(args.attack).strip().lower().replace("-", "_") in ("ifgsm", "gradest", "gradestv2", "square"):
        attack_kwargs["random_start"] = bool(args.random_start)
    if args.attack_sample_chunk_size is not None:
        attack_kwargs["sample_chunk_size"] = args.attack_sample_chunk_size
    if getattr(args, "n_samples", None) is not None:
        attack_kwargs["n_samples"] = args.n_samples
    if getattr(args, "sub_num", None) is not None:
        attack_kwargs["sub_num"] = args.sub_num
    if getattr(args, "lr_annealing", False):
        attack_kwargs["lr_annealing"] = True
    if getattr(args, "style_steps", None) is not None:
        attack_kwargs["style_steps"] = args.style_steps
    if getattr(args, "style_content_weight", None) is not None:
        attack_kwargs["style_content_weight"] = args.style_content_weight
    if getattr(args, "style_style_weight", None) is not None:
        attack_kwargs["style_style_weight"] = args.style_style_weight
    if str(args.attack).strip().lower().replace("-", "_") == "bmtc":
        if getattr(args, "bmtc_save_root", None) is not None:
            attack_kwargs["save_root"] = args.bmtc_save_root
    if str(args.attack).strip().lower().replace("-", "_") == "square":
        if args.query_budget is not None:
            attack_kwargs["query_budget"] = args.query_budget
        if getattr(args, "max_patch_frac", None) is not None:
            attack_kwargs["max_patch_frac"] = args.max_patch_frac
        if getattr(args, "scout_queries", None) is not None:
            attack_kwargs["scout_queries"] = args.scout_queries
        if getattr(args, "elite_window_count", None) is not None:
            attack_kwargs["elite_window_count"] = args.elite_window_count
        if getattr(args, "elite_candidates_per_step", None) is not None:
            attack_kwargs["elite_candidates_per_step"] = args.elite_candidates_per_step
        if getattr(args, "elite_jitter_frac", None) is not None:
            attack_kwargs["elite_jitter_frac"] = args.elite_jitter_frac
        if getattr(args, "guide_samples", None) is not None:
            attack_kwargs["guide_samples"] = args.guide_samples
        if getattr(args, "guide_sigma", None) is not None:
            attack_kwargs["guide_sigma"] = args.guide_sigma
        if getattr(args, "guide_patch_prob", None) is not None:
            attack_kwargs["guide_patch_prob"] = args.guide_patch_prob
        if getattr(args, "guide_momentum", None) is not None:
            attack_kwargs["guide_momentum"] = args.guide_momentum
        if getattr(args, "candidates_per_step", None) is not None:
            attack_kwargs["candidates_per_step"] = args.candidates_per_step
        if getattr(args, "candidates_per_step_late", None) is not None:
            attack_kwargs["candidates_per_step_late"] = args.candidates_per_step_late
        if getattr(args, "bootstrap_guide_samples", None) is not None:
            attack_kwargs["bootstrap_guide_samples"] = args.bootstrap_guide_samples
        if getattr(args, "bootstrap_guide_sigma", None) is not None:
            attack_kwargs["bootstrap_guide_sigma"] = args.bootstrap_guide_sigma
        if getattr(args, "bootstrap_primitives", None) is not None:
            attack_kwargs["bootstrap_primitives"] = args.bootstrap_primitives
        if getattr(args, "eval_batch_size", None) is not None:
            attack_kwargs["eval_batch_size"] = args.eval_batch_size
    if str(args.attack).strip().lower().replace("-", "_") in ("gradest", "gradestv2"):
        if args.query_budget is not None:
            attack_kwargs["query_budget"] = args.query_budget
        if getattr(args, "boost_second", None) is not None:
            attack_kwargs["boost_second"] = args.boost_second
    if str(args.attack).strip().lower().replace("-", "_") in ("gradestv2", "square"):
        if getattr(args, "random_start_trials", None) is not None:
            attack_kwargs["random_start_trials"] = args.random_start_trials
        if getattr(args, "boost_top2", None) is not None:
            attack_kwargs["boost_top2"] = args.boost_top2
        if getattr(args, "boost_top10", None) is not None:
            attack_kwargs["boost_top10"] = args.boost_top10
        if getattr(args, "boost_top5", None) is not None:
            attack_kwargs["boost_top5"] = args.boost_top5
        if getattr(args, "temporal_tiles", None) is not None:
            attack_kwargs["temporal_tiles"] = args.temporal_tiles
    if str(args.attack).strip().lower().replace("-", "_") == "gradestv2":
        if getattr(args, "smooth_scale", None) is not None:
            attack_kwargs["smooth_scale"] = args.smooth_scale
    if str(args.attack).strip().lower().replace("-", "_") == "tenad":
        if args.beta is not None:
            attack_kwargs["beta"] = args.beta
        if args.binary_search_steps is not None:
            attack_kwargs["binary_search_steps"] = args.binary_search_steps
        if args.max_expand_steps is not None:
            attack_kwargs["max_expand_steps"] = args.max_expand_steps
        if args.lambda_init is not None:
            attack_kwargs["lambda_init"] = args.lambda_init
        if args.query_budget is not None:
            attack_kwargs["query_budget"] = args.query_budget
        if args.num_restarts is not None:
            attack_kwargs["num_restarts"] = args.num_restarts
        if args.spatial_stride is not None:
            attack_kwargs["spatial_stride"] = args.spatial_stride
    attack = create_attack(args.attack, **attack_kwargs)

    defence = None
    if args.defence is not None:
        from vg_videobench.defences import create_defence
        defence = create_defence(args.defence)

    dataset = create_dataset(args.dataset, **dataset_kwargs)
    model = create_model(
        args.model,
        checkpoint_path=args.checkpoint,
        backbone=args.backbone,
        weights_dataset=args.weights_dataset,
        grad_forward_chunk_size=args.grad_forward_chunk_size,
        device=args.device,
    )
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
        defence=defence,
        adaptive=args.adaptive,
        save_defence_stages=args.save_defence_stages,
    )
    print(json.dumps(summary, indent=2))
    write_json_output(args.output_json, summary)


if __name__ == "__main__":
    main()

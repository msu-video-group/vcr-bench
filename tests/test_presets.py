from __future__ import annotations

from argparse import Namespace
import subprocess

from vcr_bench.cli.common import build_model_dataset_context
from vcr_bench.cli.attack import build_base_parser
from vcr_bench.presets import parse_overrides, resolve_entity_preset, resolve_run_preset


def test_attack_preset_inheritance() -> None:
    spec = resolve_entity_preset("attack", "square", preset_name="guided")
    params = spec["params"]
    assert params["query_budget"] == 1000
    assert params["guide_samples"] == 20
    assert spec["factory_name"] == "square"


def test_override_parsing_json_values() -> None:
    overrides = parse_overrides(["attack.params.query_budget=2000", "runtime.device=\"cpu\"", "flag=true"])
    assert overrides["attack"]["params"]["query_budget"] == 2000
    assert overrides["runtime"]["device"] == "cpu"
    assert overrides["flag"] is True


def test_run_preset_resolution() -> None:
    run = resolve_run_preset("accuracy_amd_100")
    assert run["task"] == "accuracy"
    assert run["dataset"]["subset"] == "k400_test"
    assert run["models"][0]["preset"] == "amd"
    assert run["models"][0]["preset_name"] == "vitb_k400"


def test_missing_attack_preset_is_generated_from_signature() -> None:
    spec = resolve_entity_preset("attack", "mifgsm")
    assert spec["factory_name"] == "mifgsm"
    assert spec["params"]["eps"] == 8.0
    assert spec["params"]["steps"] == 20


def test_missing_defence_preset_is_generated_from_signature() -> None:
    spec = resolve_entity_preset("defence", "temporal_median")
    assert spec["factory_name"] == "temporal_median"
    assert spec["params"] == {}


def test_missing_model_preset_is_generated_from_capabilities() -> None:
    spec = resolve_entity_preset("model", "timesformer")
    assert spec["factory_name"] == "timesformer"
    assert "backbone" in spec["params"]
    assert "weights_dataset" in spec["params"]


def test_test_cli_print_resolved_preset() -> None:
    result = subprocess.run(
        [
            "python3",
            "-m",
            "vcr_bench.cli.test",
            "--run-preset",
            "accuracy_amd_100",
            "--print-resolved-preset",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert '"accuracy_amd_100"' in result.stdout
    assert '"factory_name": "amd"' in result.stdout


def test_attack_cli_print_resolved_preset() -> None:
    result = subprocess.run(
        [
            "python3",
            "-m",
            "vcr_bench.cli.attack",
            "--run-preset",
            "attack_x3d_ifgsm_debug",
            "--override",
            "attack.params.steps=3",
            "--print-resolved-preset",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert '"attack_x3d_ifgsm_debug"' in result.stdout
    assert '"steps": 3' in result.stdout


def test_attack_cli_dynamic_help_lists_attack_specific_flags() -> None:
    result = subprocess.run(
        [
            "python3",
            "-m",
            "vcr_bench.cli.attack",
            "--attack",
            "square",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--query-budget" in result.stdout
    assert "--guide-samples" in result.stdout
    assert "--iter" in result.stdout


def test_attack_cli_attacks_misclassified_by_default() -> None:
    parser = build_base_parser(add_help=False)
    args = parser.parse_args([])
    assert args.allow_misclassified is True

    args = parser.parse_args(["--skip-misclassified"])
    assert args.allow_misclassified is False


def test_cli_context_defaults_to_registered_dataset_subset() -> None:
    args = Namespace(
        model="x3d",
        checkpoint=None,
        backbone="m",
        weights_dataset="kinetics400",
        grad_forward_chunk_size=3,
        dataset="kinetics400",
        dataset_subset=None,
        video_root=None,
        annotations=None,
        labels=None,
        split="val",
        pipeline_stage="test",
        full_videos=False,
    )
    context = build_model_dataset_context(args)
    assert args.dataset_subset == "k400_val"
    assert context.dataset_kwargs["dataset_subset"] == "k400_val"


def test_remote_launch_preset_dry_run() -> None:
    result = subprocess.run(
        [
            "python3",
            "-m",
            "vcr_bench.remote",
            "--config",
            "configs/local.toml.example",
            "--remote",
            "main",
            "launch-preset",
            "--preset",
            "accuracy_amd_100",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "scripts/test_models.sh" in result.stdout
    assert "--model amd" in result.stdout

from __future__ import annotations

from argparse import Namespace
import subprocess

from vcr_bench.cli.common import build_model_dataset_context
from vcr_bench.presets import parse_overrides, resolve_entity_preset, resolve_run_preset


def test_attack_variant_inheritance() -> None:
    spec = resolve_entity_preset("attack", "square", variant="guided")
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
    assert run["dataset"]["subset"] == "k400_val"
    assert run["models"][0]["preset"] == "amd"


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


def test_cli_context_defaults_to_registered_dataset_subset() -> None:
    args = Namespace(
        model="amd",
        checkpoint=None,
        backbone="vitb",
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

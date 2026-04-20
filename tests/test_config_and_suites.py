from __future__ import annotations

from pathlib import Path

from vcr_bench.artifacts import get_checkpoint_artifact, list_dataset_subsets
from vcr_bench.benchmarks import expand_accuracy_suite, expand_attack_suite
from vcr_bench.config import get_path, load_settings


def test_settings_paths_resolve_to_repo_paths() -> None:
    settings = load_settings()
    assert Path(settings["paths"]["cache_dir_abs"]).is_absolute()
    assert get_path("results_dir").is_absolute()


def test_checkpoint_manifest_lookup() -> None:
    entry = get_checkpoint_artifact("x3d", "m", "kinetics400")
    assert entry is not None
    assert entry["repo_id"] == "vcr-bench/checkpoints-x3d"


def test_amd_checkpoint_manifest_lookup() -> None:
    entry = get_checkpoint_artifact("amd", "vitb", "kinetics400")
    assert entry is not None
    assert entry["repo_id"] == "maxv65/vcr-bench"
    assert entry["filename"] == "amd_vitb_k400_finetune_90e.pth"


def test_custom_checkpoint_manifest_lookups() -> None:
    expected = {
        ("ila", "vit_b16", "kinetics400"): "ila_k400_B_16_8_rel.pth",
        ("internvideo2", "vit_1b_p14", "kinetics400"): "internvideo2_1B_ft_k710_ft_k400_f16.pth",
        ("onepeace", "vit_l40", "kinetics400"): "onepeace_video_k400.pth",
        ("tadaformer", "large_14", "kinetics400"): "tadaformer_l14_clip+k710_k400_64f_89.9.pyth",
        ("umt", "vit_large_p16", "kinetics400"): "umt_l16_ptk710_ftk710_ftk400_f8_res224.pth",
    }
    for (model, backbone, weights_dataset), filename in expected.items():
        entry = get_checkpoint_artifact(model, backbone, weights_dataset)
        assert entry is not None
        assert entry["repo_id"] == "maxv65/vcr-bench"
        assert entry["filename"] == filename


def test_dataset_subsets_are_registered() -> None:
    subsets = list_dataset_subsets("kinetics400")
    assert "k400_val" in subsets
    assert "k400_test" in subsets
    assert "kinetics400_mini_val" in subsets


def test_accuracy_suite_expansion() -> None:
    suite = expand_accuracy_suite("default", model_filter={"timesformer"})
    assert suite["models"] == ["timesformer"]


def test_attack_suite_expansion_filters_entries() -> None:
    suite = expand_attack_suite("default", model_filter={"x3d"}, attack_filter={"ifgsm"})
    assert len(suite["entries"]) == 1
    assert suite["entries"][0]["model"] == "x3d"
    assert suite["entries"][0]["attack"] == "ifgsm"

from __future__ import annotations

from pathlib import Path

from vcr_bench.artifacts import get_checkpoint_artifact, list_dataset_subsets
from vcr_bench.benchmarks import expand_accuracy_suite, expand_attack_suite
from vcr_bench.config import config_root, get_path, load_settings, repo_root


def test_settings_paths_resolve_to_repo_paths() -> None:
    settings = load_settings()
    assert Path(settings["paths"]["cache_dir_abs"]).is_absolute()
    assert get_path("results_dir").is_absolute()


def test_checkpoint_manifest_lookup() -> None:
    entry = get_checkpoint_artifact("x3d", "m", "kinetics400")
    assert entry is not None
    assert entry["repo_id"] == "maxv65/vcr-bench"


def test_no_license_weights_are_bring_your_own() -> None:
    # AMD and ILA upstreams ship no license; VCR-Bench does not redistribute their
    # weights, so they must NOT have an auto-download manifest entry. See docs/licenses.md.
    assert get_checkpoint_artifact("amd", "vitb", "kinetics400") is None
    assert get_checkpoint_artifact("ila", "vit_b16", "kinetics400") is None


def test_c3d_checkpoint_manifest_lookup() -> None:
    entry = get_checkpoint_artifact("c3d", "c3d", "ucf101")
    assert entry is not None
    assert entry["repo_id"] == "maxv65/vcr-bench"
    assert entry["filename"] == "c3d_ucf101.pth"


def test_custom_checkpoint_manifest_lookups() -> None:
    expected = {
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


def test_all_builtin_components_have_checked_in_presets() -> None:
    cfg_root = config_root()
    repo = repo_root()

    attack_dirs = {
        p.name for p in (repo / "vcr_bench" / "attacks").iterdir()
        if p.is_dir() and p.name != "__pycache__"
    }
    defence_dirs = {
        p.name for p in (repo / "vcr_bench" / "defences").iterdir()
        if p.is_dir() and p.name != "__pycache__"
    }
    model_dirs = {
        p.name for p in (repo / "vcr_bench" / "models").iterdir()
        if p.is_dir() and p.name not in {"__pycache__", "vendor_mmaction"}
    }

    attack_cfgs = {p.stem for p in (cfg_root / "attacks").glob("*.json")}
    defence_cfgs = {p.stem for p in (cfg_root / "defences").glob("*.json")}
    model_cfgs = {p.stem for p in (cfg_root / "models").glob("*.json")}

    assert attack_dirs <= attack_cfgs
    assert defence_dirs <= defence_cfgs
    assert model_dirs <= model_cfgs

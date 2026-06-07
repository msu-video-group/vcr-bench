from __future__ import annotations

from pathlib import Path

import pytest

from vcr_bench.datasets import create_dataset


@pytest.mark.skipif(
    not (Path("data/ucf101/videos").exists() and Path("data/ucf101/annotations/classInd.txt").exists()),
    reason="UCF-101 raw videos are not redistributed; provide them under data/ucf101/ to run this test.",
)
def test_ucf101_dataset_loads_from_local_defaults() -> None:
    dataset = create_dataset("ucf101")
    assert dataset.video_root == Path("data/ucf101/videos").resolve()
    assert dataset.labels_txt == Path("data/ucf101/annotations/classInd.txt").resolve()
    assert len(dataset) > 0
    sample = dataset[0]
    assert sample.label is not None
    assert sample.label_name is not None


def test_sthv2_dataset_can_load_directory_layout_without_json_annotations(tmp_path: Path) -> None:
    video_root = tmp_path / "videos"
    label_name = "Approaching something with your camera"
    class_dir = video_root / label_name
    class_dir.mkdir(parents=True)
    (class_dir / "sample.mp4").write_bytes(b"video")

    labels = tmp_path / "label_map.txt"
    labels.write_text(f"{label_name}\nHolding something\n", encoding="utf-8")

    dataset = create_dataset("sthv2", video_root=str(video_root), labels_txt=str(labels))
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample.label == 0
    assert sample.label_name == label_name


def test_sthv2_alias_resolves_to_same_dataset(tmp_path: Path) -> None:
    video_root = tmp_path / "videos"
    label_name = "Approaching something with your camera"
    class_dir = video_root / label_name
    class_dir.mkdir(parents=True)
    (class_dir / "sample.mp4").write_bytes(b"video")

    labels = tmp_path / "label_map.txt"
    labels.write_text(f"{label_name}\n", encoding="utf-8")

    dataset = create_dataset("something_something_v2", video_root=str(video_root), labels_txt=str(labels))
    assert len(dataset) == 1

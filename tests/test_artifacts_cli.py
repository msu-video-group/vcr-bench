from __future__ import annotations

import zipfile
from pathlib import Path

from vcr_bench.artifacts import build_dataset_archive, list_checkpoint_artifacts


def test_list_checkpoint_artifacts_contains_manifest_key() -> None:
    entries = list_checkpoint_artifacts(model="x3d")
    assert entries
    assert entries[0]["manifest_key"] == "x3d_m_kinetics400"


def test_build_dataset_archive(tmp_path: Path) -> None:
    source = tmp_path / "subset"
    (source / "videos").mkdir(parents=True)
    (source / "annotations").mkdir(parents=True)
    (source / "videos" / "sample.mp4").write_bytes(b"video")
    (source / "annotations" / "val.csv").write_text("label,id\n", encoding="utf-8")
    archive = build_dataset_archive(
        dataset="kinetics400",
        subset="kinetics400_mini_val",
        source_dir=source,
        output_path=tmp_path / "subset.zip",
        force=False,
    )
    assert archive.exists()
    with zipfile.ZipFile(archive, "r") as zf:
        names = sorted(zf.namelist())
    assert "annotations/val.csv" in names
    assert "videos/sample.mp4" in names

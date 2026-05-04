from __future__ import annotations

import json
from pathlib import Path

from vcr_bench.types import LoadedDatasetItem, VideoSampleRef

from ..base import BaseVideoDataset


VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".webm", ".mov", ".y4m"}


class STHV2Dataset(BaseVideoDataset):
    def __init__(
        self,
        video_root: str | None = None,
        annotations_csv: str | None = None,
        labels_txt: str | None = None,
        dataset_subset: str | None = None,
        split: str = "val",
        clip_len: int | None = None,
        num_clips: int | None = None,
        frame_interval: int = 1,
        frame_intervals: int | None = None,
        full_videos: bool = False,
    ) -> None:
        del dataset_subset
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.frame_interval = frame_interval if frame_intervals is None else frame_intervals
        self.full_videos = bool(full_videos)
        self.split = split

        repo_root = Path(__file__).resolve().parents[3]
        data_root = repo_root / "data" / "sthv2"
        split_name = "train" if "train" in split.lower() else "validation"

        self.video_root = Path(video_root or (data_root / "videos"))
        self.annotations_csv = Path(annotations_csv or (data_root / "annotations" / f"something-something-v2-{split_name}-with-label.json"))
        self.labels_txt = Path(labels_txt or (data_root / "annotations" / "label_map.txt"))

        self._class_names = self._load_labels(self.labels_txt)
        self._label_to_idx = {name: idx for idx, name in enumerate(self._class_names)}
        self._samples = self._build_samples()

    def _load_labels(self, path: Path) -> list[str]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _resolve_video_path(self, sample_id: str) -> Path | None:
        for suffix in sorted(VIDEO_SUFFIXES):
            path = self.video_root / f"{sample_id}{suffix}"
            if path.exists():
                return path
        return None

    def _build_samples_from_annotations(self) -> list[VideoSampleRef]:
        if not self.annotations_csv.exists():
            return []
        with self.annotations_csv.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            return []
        samples: list[VideoSampleRef] = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            sample_id = str(entry.get("id", "")).strip()
            if not sample_id:
                continue
            path = self._resolve_video_path(sample_id)
            if path is None:
                continue
            label_name = str(entry.get("template", "")).replace("[", "").replace("]", "").strip() or None
            label_idx = entry.get("label_idx")
            label = int(label_idx) if isinstance(label_idx, int) else self._label_to_idx.get(label_name or "")
            samples.append(
                VideoSampleRef(
                    id=sample_id,
                    path=str(path),
                    label=label,
                    label_name=label_name,
                    metadata={"split": self.split},
                )
            )
        return samples

    def _build_samples_from_dirs(self) -> list[VideoSampleRef]:
        if not self.video_root.exists():
            raise FileNotFoundError(f"Video root not found: {self.video_root}")
        samples: list[VideoSampleRef] = []
        for path in sorted(self.video_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            label_name = path.parent.name
            label = self._label_to_idx.get(label_name)
            if label is None:
                continue
            samples.append(
                VideoSampleRef(
                    id=path.stem,
                    path=str(path),
                    label=label,
                    label_name=label_name,
                    metadata={"split": self.split},
                )
            )
        return samples

    def _build_samples(self) -> list[VideoSampleRef]:
        samples = self._build_samples_from_annotations()
        if samples:
            return samples
        return self._build_samples_from_dirs()

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> VideoSampleRef | LoadedDatasetItem:
        return self._materialize_sample(self._samples[index])

    def class_names(self) -> list[str]:
        return list(self._class_names)


def create(**kwargs) -> STHV2Dataset:
    return STHV2Dataset(**kwargs)

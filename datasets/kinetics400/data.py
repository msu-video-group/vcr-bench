from __future__ import annotations

import csv
from pathlib import Path

from vg_videobench.types import LoadedDatasetItem, VideoSampleRef

from ..base import BaseVideoDataset


VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".webm", ".mov", ".y4m"}


def _default_k400_video_id_from_name(path: Path) -> str | None:
    parts = path.stem.split("_")
    if len(parts) < 3:
        return None
    return parts[-3]


class Kinetics400Dataset(BaseVideoDataset):
    def __init__(
        self,
        video_root: str | None = None,
        annotations_csv: str | None = None,
        labels_txt: str | None = None,
        split: str = "val",
        clip_len: int | None = None,
        num_clips: int | None = None,
        frame_interval: int = 1,
        frame_intervals: int | None = None,
        full_videos: bool = False,
    ) -> None:
        self.split = split
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.frame_interval = frame_interval if frame_intervals is None else frame_intervals
        self.full_videos = bool(full_videos)
        repo_root = Path(__file__).resolve().parents[3]
        mmact_dir = "m" "maction2"
        split_defaults = {
            "val": {
                "video_root": repo_root / mmact_dir / "data" / "kinetics400" / "val",
                "annotations_csv": repo_root / mmact_dir / "data" / "kinetics400" / "annotations" / "val.csv",
            },
            "test": {
                "video_root": repo_root / mmact_dir / "data" / "kinetics400" / "test",
                "annotations_csv": repo_root / mmact_dir / "data" / "kinetics400" / "annotations" / "test.csv",
            },
        }
        labels_default = repo_root / mmact_dir / "tools" / "data" / "kinetics" / "label_map_k400.txt"
        current_split = split_defaults.get(split, split_defaults["val"])
        video_root = video_root or str(current_split["video_root"])
        annotations_csv = annotations_csv or str(current_split["annotations_csv"])
        labels_txt = labels_txt or str(labels_default)

        if not video_root or not annotations_csv or not labels_txt:
            raise ValueError(
                "Kinetics400Dataset requires video_root, annotations_csv and labels_txt "
                "(either via args or constructor defaults)"
            )

        self.video_root = Path(video_root)
        self.annotations_csv = Path(annotations_csv)
        self.labels_txt = Path(labels_txt)

        self._class_names = self._load_labels(self.labels_txt)
        self._label_to_idx = {name: i for i, name in enumerate(self._class_names)}
        self._id_to_label = self._load_annotations(self.annotations_csv, self._label_to_idx)
        self._samples = self._build_samples()

    def _load_labels(self, path: Path) -> list[str]:
        labels = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(line)
        return labels

    def _load_annotations(self, path: Path, label_to_idx: dict[str, int]) -> dict[str, int]:
        mapping: dict[str, int] = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                label_name, video_id = row[0], row[1]
                if label_name in label_to_idx:
                    mapping[video_id] = label_to_idx[label_name]
        return mapping

    def _iter_videos(self):
        if not self.video_root.exists():
            raise FileNotFoundError(f"Video root not found: {self.video_root}")
        for path in sorted(self.video_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                yield path

    def _build_samples(self) -> list[VideoSampleRef]:
        samples: list[VideoSampleRef] = []
        for path in self._iter_videos():
            video_id = _default_k400_video_id_from_name(path)
            if video_id is None:
                continue
            label = self._id_to_label.get(video_id)
            if label is None:
                continue
            samples.append(
                VideoSampleRef(
                    id=video_id,
                    path=str(path),
                    label=label,
                    label_name=self._class_names[label],
                    metadata={
                        "split": self.split,
                        "clip_len": self.clip_len,
                        "num_clips": self.num_clips,
                        "frame_interval": self.frame_interval,
                        "full_videos": self.full_videos,
                    },
                )
            )
        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> VideoSampleRef | LoadedDatasetItem:
        return self._materialize_sample(self._samples[index])

    def class_names(self) -> list[str]:
        return list(self._class_names)


def create(**kwargs) -> Kinetics400Dataset:
    return Kinetics400Dataset(**kwargs)

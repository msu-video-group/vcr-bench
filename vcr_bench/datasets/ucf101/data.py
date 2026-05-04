from __future__ import annotations

from pathlib import Path

from vcr_bench.types import LoadedDatasetItem, VideoSampleRef

from ..base import BaseVideoDataset


VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".webm", ".mov", ".y4m"}


class UCF101Dataset(BaseVideoDataset):
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
        data_root = repo_root / "data" / "ucf101"
        split_key = "train" if "train" in split.lower() else "val"
        split_file = "ucf101_train_split_1_videos.txt" if split_key == "train" else "ucf101_val_split_1_videos.txt"

        self.video_root = Path(video_root or (data_root / "videos"))
        self.annotations_csv = Path(annotations_csv or (data_root / split_file))
        self.labels_txt = Path(labels_txt or (data_root / "annotations" / "classInd.txt"))

        self._class_names = self._load_labels(self.labels_txt)
        self._label_to_idx = {name: idx for idx, name in enumerate(self._class_names)}
        self._samples = self._build_samples()

    def _load_labels(self, path: Path) -> list[str]:
        labels: list[str] = []
        if not path.exists():
            return labels
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                labels.append(" ".join(parts[1:]) if len(parts) > 1 and parts[0].isdigit() else line)
        return labels

    def _build_samples_from_annotations(self) -> list[VideoSampleRef]:
        if not self.annotations_csv.exists():
            return []
        samples: list[VideoSampleRef] = []
        with self.annotations_csv.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                rel_path = parts[0]
                path = self.video_root / rel_path
                if not path.exists():
                    continue
                label_name = Path(rel_path).parent.name
                label = None
                if len(parts) > 1:
                    try:
                        label = int(parts[1])
                    except ValueError:
                        label = None
                if label is None and label_name in self._label_to_idx:
                    label = self._label_to_idx[label_name]
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

    def _build_samples_from_dirs(self) -> list[VideoSampleRef]:
        if not self.video_root.exists():
            raise FileNotFoundError(f"Video root not found: {self.video_root}")
        samples: list[VideoSampleRef] = []
        for path in sorted(self.video_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            label_name = path.parent.name
            label = self._label_to_idx.get(label_name)
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


def create(**kwargs) -> UCF101Dataset:
    return UCF101Dataset(**kwargs)

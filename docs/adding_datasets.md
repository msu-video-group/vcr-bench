# Adding a Dataset

## Directory layout

```
vcr_bench/datasets/<name>/
    __init__.py    # empty or re-export
    data.py        # dataset implementation
configs/datasets.toml    # add subset entries here
```

The registry finds your dataset by importing `vcr_bench.datasets.<name>.data` and looking for `create()` or `DATASET_CLASS`.

## 1. Implement the dataset

```python
# vcr_bench/datasets/mydataset/data.py
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch

from ..base import BaseVideoDataset
from vcr_bench.types import VideoSampleRef, LoadedDatasetItem


class MyDataset(BaseVideoDataset):

    def __init__(
        self,
        video_root: str | None = None,
        annotations: str | None = None,
        labels: str | None = None,
        dataset_subset: str | None = None,
        split: str = "val",
        num_classes: int = 400,
        **kwargs,
    ) -> None:
        # Resolve paths — either passed directly or resolved from dataset_subset name
        if dataset_subset and (video_root is None or annotations is None):
            video_root, annotations, labels = self._resolve_subset(dataset_subset)

        self._video_root = Path(video_root)
        self._class_names_list: list[str] = self._load_labels(labels)

        # Build sample list — each entry is a VideoSampleRef
        self._samples: list[VideoSampleRef] = self._load_annotations(annotations, split)

        # Pipeline placeholders (set by configure_loading() when used with a model)
        self._loading_pipeline = None

    def _load_labels(self, labels_path: str | None) -> list[str]:
        if not labels_path:
            return [str(i) for i in range(400)]
        with open(labels_path) as f:
            return [line.strip() for line in f if line.strip()]

    def _load_annotations(self, annotations_path: str, split: str) -> list[VideoSampleRef]:
        samples = []
        with open(annotations_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if split and row.get("split", split) != split:
                    continue
                path = str(self._video_root / row["filename"])
                label = int(row["label"]) if "label" in row else None
                label_name = row.get("label_name")
                samples.append(
                    VideoSampleRef(
                        id=row.get("id", row["filename"]),
                        path=path,
                        label=label,
                        label_name=label_name,
                        metadata={},
                    )
                )
        return samples

    def _resolve_subset(self, subset_name: str) -> tuple[str, str, str]:
        # Download and extract subset archive from HuggingFace if needed.
        # See vcr_bench/datasets/kinetics400/data.py for the full resolve_dataset_subset pattern.
        from vcr_bench.artifacts import resolve_dataset_subset_artifact
        info = resolve_dataset_subset_artifact("mydataset", subset_name)
        return str(info["video_root"]), str(info["annotations"]), str(info["labels"])

    # ------------------------------------------------------------------
    # BaseVideoDataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> VideoSampleRef | LoadedDatasetItem:
        sample = self._samples[index]
        if self._loading_pipeline is None:
            return sample
        return self._loading_pipeline(sample)

    def class_names(self) -> list[str]:
        return self._class_names_list

    def num_classes(self) -> int:
        return len(self._class_names_list)

    def configure_loading(self, data_pipeline: Any, instant_preprocessing: bool = False) -> None:
        self._loading_pipeline = data_pipeline

    def clear_loading(self) -> None:
        self._loading_pipeline = None


DATASET_CLASS = MyDataset


def create(**kwargs) -> MyDataset:
    return MyDataset(**kwargs)
```

## 2. Register subsets in datasets.toml

```toml
# configs/datasets.toml

[datasets.mydataset.mini_val]
repo_id = "your-hf-org/vcr-bench"
filename = "mydataset_mini_val.zip"
annotations_filename = "mydataset_mini_val.csv"
labels_filename = "mydataset_label_map.txt"
extract_subdir = "mydataset_mini_val"
video_root = "."
annotations = "annotations/mydataset_mini_val.csv"
labels = "annotations/mydataset_label_map.txt"
split = "val"
```

Fields:
- `repo_id` — HuggingFace repo containing the archive.
- `filename` — ZIP archive filename in the repo.
- `annotations_filename` / `labels_filename` — files inside the ZIP to extract to `data/`.
- `extract_subdir` — subdirectory inside `data/` to extract videos into.
- `video_root` / `annotations` / `labels` — paths relative to `data/<extract_subdir>/` passed to `create()`.
- `split` — passed as `split=` to `create()`.

## 3. Verify

```bash
# Check the dataset loads without a model
python -c "
from vcr_bench.datasets import create_dataset
ds = create_dataset('mydataset', video_root='data/...', annotations='...', labels='...')
print(len(ds), ds.class_names()[:3])
print(ds[0])
"

# Run accuracy test
vcr-bench-test --model x3d --dataset mydataset \
  --dataset-subset mini_val --num-videos 8
```

## Annotations CSV format

No strict schema is enforced — the dataset class parses it however it needs. The kinetics400 convention is:

```csv
id,filename,label,label_name,split
v_Abseiling_g01_c01,k400_val/v_Abseiliig.mp4,0,abseiling,val
```

- `filename` — path to video, relative to `video_root`
- `label` — integer class index
- `split` — `train` / `val` / `test`

## Notes

- `__getitem__` must return `VideoSampleRef` when `_loading_pipeline` is `None` (bare dataset mode, used by the attack CLI to enumerate samples). Return `LoadedDatasetItem` when the pipeline is set.
- `configure_loading` is called by `ModelLoadedDataset` in `vcr_bench/datasets/loaded.py` before the dataloader starts.
- Subset auto-download goes through `vcr_bench/artifacts.py` → HuggingFace Hub. If you don't need remote download, just accept explicit `video_root`/`annotations`/`labels` args and skip `_resolve_subset`.

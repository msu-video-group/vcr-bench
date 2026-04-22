from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

from .config import get_path, load_checkpoint_registry, load_dataset_registry, load_settings


def _hf_download(*, repo_id: str, filename: str, revision: str, repo_type: str, local_dir: Path) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    target = local_dir / filename
    if target.exists():
        return target
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        repo_type=repo_type,
        local_dir=str(local_dir),
    )
    return Path(path)


def get_checkpoint_artifact(model: str, backbone: str, weights_dataset: str) -> dict[str, Any] | None:
    key = f"{model}_{backbone}_{weights_dataset}".replace("-", "_")
    entry = load_checkpoint_registry().get(key)
    return dict(entry) if isinstance(entry, dict) else None


def list_checkpoint_artifacts(*, model: str | None = None) -> list[dict[str, Any]]:
    entries = []
    for key, value in load_checkpoint_registry().items():
        if not isinstance(value, dict):
            continue
        item = dict(value)
        item["manifest_key"] = key
        if model is not None and item.get("model") != model:
            continue
        entries.append(item)
    return sorted(entries, key=lambda item: (str(item.get("model", "")), str(item.get("backbone", "")), str(item.get("weights_dataset", ""))))


def resolve_checkpoint_artifact(
    *,
    model: str,
    backbone: str,
    weights_dataset: str,
    auto_download: bool,
) -> Path | None:
    entry = get_checkpoint_artifact(model, backbone, weights_dataset)
    if not entry:
        return None
    target_dir = get_path("cache_dir") / str(entry.get("local_subdir", f"checkpoints/{model}"))
    target = target_dir / str(entry["filename"])
    if target.exists() or not auto_download:
        return target
    try:
        return _hf_download(
            repo_id=str(entry["repo_id"]),
            filename=str(entry["filename"]),
            revision=str(entry.get("revision", "main")),
            repo_type="model",
            local_dir=target_dir,
        )
    except (HfHubHTTPError, RepositoryNotFoundError):
        return None


def list_dataset_subsets(dataset: str) -> list[str]:
    registry = load_dataset_registry().get(dataset, {})
    return sorted(registry.keys()) if isinstance(registry, dict) else []


def default_dataset_subset(dataset: str, split: str = "val") -> str | None:
    registry = load_dataset_registry().get(dataset, {})
    if not isinstance(registry, dict):
        return None
    split = str(split or "val")
    matching = [
        name
        for name, entry in registry.items()
        if isinstance(entry, dict) and str(entry.get("split", "val")) == split
    ]
    if not matching:
        return None
    return sorted(matching, key=lambda name: (0 if name.endswith(f"_{split}") else 1, name))[0]


def get_dataset_subset_entry(dataset: str, subset: str) -> dict[str, Any] | None:
    registry = load_dataset_registry().get(dataset, {})
    if not isinstance(registry, dict):
        return None
    entry = registry.get(subset)
    return dict(entry) if isinstance(entry, dict) else None


def resolve_dataset_subset(dataset: str, subset: str) -> dict[str, str]:
    entry = get_dataset_subset_entry(dataset, subset)
    if entry is None:
        raise ValueError(f"Unknown dataset subset: {dataset}/{subset}")
    cache_root = get_path("cache_dir") / "dataset_archives" / dataset / subset
    data_root = get_path("data_dir") / dataset / subset
    extract_root = data_root / str(entry.get("extract_subdir", subset))
    marker = extract_root / ".vcr_extract_complete.json"
    archive_filename = entry.get("filename")
    archive_path: Path | None = None
    if archive_filename:
        archive_path = _hf_download(
            repo_id=str(entry["repo_id"]),
            filename=str(archive_filename),
            revision=str(entry.get("revision", "main")),
            repo_type="model",
            local_dir=cache_root,
        )
    if archive_path is not None and not marker.exists():
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_root)
        marker.write_text(
            json.dumps({"archive": str(archive_path), "subset": subset}, indent=2),
            encoding="utf-8",
        )
    else:
        extract_root.mkdir(parents=True, exist_ok=True)

    annotations_path = extract_root / str(entry["annotations"])
    labels_path = extract_root / str(entry["labels"])
    annotations_filename = entry.get("annotations_filename")
    labels_filename = entry.get("labels_filename")
    if annotations_filename and not annotations_path.exists():
        annotations_src = _hf_download(
            repo_id=str(entry["repo_id"]),
            filename=str(annotations_filename),
            revision=str(entry.get("revision", "main")),
            repo_type="model",
            local_dir=cache_root,
        )
        annotations_path.parent.mkdir(parents=True, exist_ok=True)
        annotations_path.write_bytes(annotations_src.read_bytes())
    if labels_filename and not labels_path.exists():
        labels_src = _hf_download(
            repo_id=str(entry["repo_id"]),
            filename=str(labels_filename),
            revision=str(entry.get("revision", "main")),
            repo_type="model",
            local_dir=cache_root,
        )
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        labels_path.write_bytes(labels_src.read_bytes())
    split = str(entry.get("split", "val"))
    return {
        "video_root": str(extract_root / str(entry["video_root"])),
        "annotations": str(annotations_path),
        "labels": str(labels_path),
        "split": split,
    }


def hf_token_env_name() -> str:
    settings = load_settings()
    return str(settings.get("artifacts", {}).get("hf_token_env", "HF_TOKEN"))


def _hf_api() -> HfApi:
    return HfApi()


def create_hf_repo(*, repo_id: str, repo_type: str, private: bool = False, exist_ok: bool = True) -> str:
    api = _hf_api()
    created = api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=exist_ok)
    return str(created)


def upload_file_to_hf(
    *,
    repo_id: str,
    repo_type: str,
    local_path: Path,
    path_in_repo: str,
    commit_message: str,
) -> str:
    api = _hf_api()
    return str(
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message,
        )
    )


def build_dataset_archive(
    *,
    dataset: str,
    subset: str,
    source_dir: Path,
    output_path: Path,
    force: bool = False,
) -> Path:
    entry = get_dataset_subset_entry(dataset, subset)
    if entry is None:
        raise ValueError(f"Unknown dataset subset: {dataset}/{subset}")
    if output_path.exists() and not force:
        raise FileExistsError(f"Archive already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(source_dir.rglob("*")):
            if path.is_dir():
                continue
            rel = path.relative_to(source_dir)
            zf.write(path, arcname=str(rel))
    return output_path

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vcr_bench.artifacts import (
    build_dataset_archive,
    create_hf_repo,
    get_checkpoint_artifact,
    get_dataset_subset_entry,
    list_checkpoint_artifacts,
    list_dataset_subsets,
    resolve_checkpoint_artifact,
    resolve_dataset_subset,
    upload_file_to_hf,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vcr_bench artifact management")
    sub = parser.add_subparsers(dest="command", required=True)

    list_ckpt = sub.add_parser("list-checkpoints")
    list_ckpt.add_argument("--model", default=None)

    list_ds = sub.add_parser("list-dataset-subsets")
    list_ds.add_argument("--dataset", required=True)

    resolve_ckpt = sub.add_parser("resolve-checkpoint")
    resolve_ckpt.add_argument("--model", required=True)
    resolve_ckpt.add_argument("--backbone", required=True)
    resolve_ckpt.add_argument("--weights-dataset", required=True)
    resolve_ckpt.add_argument("--download", action="store_true")

    resolve_ds = sub.add_parser("resolve-dataset-subset")
    resolve_ds.add_argument("--dataset", required=True)
    resolve_ds.add_argument("--subset", required=True)

    create_repo = sub.add_parser("create-hf-repo")
    create_repo.add_argument("--repo-id", required=True)
    create_repo.add_argument("--repo-type", choices=["model", "dataset"], required=True)
    create_repo.add_argument("--private", action="store_true")
    create_repo.add_argument("--exist-ok", action="store_true")

    upload_ckpt = sub.add_parser("upload-checkpoint")
    upload_ckpt.add_argument("--model", required=True)
    upload_ckpt.add_argument("--backbone", required=True)
    upload_ckpt.add_argument("--weights-dataset", required=True)
    upload_ckpt.add_argument("--local-path", required=True)
    upload_ckpt.add_argument("--path-in-repo", default=None)
    upload_ckpt.add_argument("--create-repo", action="store_true")
    upload_ckpt.add_argument("--private", action="store_true")

    build_zip = sub.add_parser("build-dataset-archive")
    build_zip.add_argument("--dataset", required=True)
    build_zip.add_argument("--subset", required=True)
    build_zip.add_argument("--source-dir", required=True)
    build_zip.add_argument("--output", default=None)
    build_zip.add_argument("--force", action="store_true")

    upload_ds = sub.add_parser("upload-dataset-archive")
    upload_ds.add_argument("--dataset", required=True)
    upload_ds.add_argument("--subset", required=True)
    upload_ds.add_argument("--archive-path", default=None)
    upload_ds.add_argument("--source-dir", default=None)
    upload_ds.add_argument("--path-in-repo", default=None)
    upload_ds.add_argument("--create-repo", action="store_true")
    upload_ds.add_argument("--private", action="store_true")
    upload_ds.add_argument("--keep-archive", action="store_true")
    return parser


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def cmd_list_checkpoints(args: argparse.Namespace) -> None:
    entries = list_checkpoint_artifacts(model=args.model)
    _print({"command": "list-checkpoints", "count": len(entries), "entries": entries})


def cmd_list_dataset_subsets(args: argparse.Namespace) -> None:
    subsets = list_dataset_subsets(args.dataset)
    _print({"command": "list-dataset-subsets", "dataset": args.dataset, "subsets": subsets})


def cmd_resolve_checkpoint(args: argparse.Namespace) -> None:
    entry = get_checkpoint_artifact(args.model, args.backbone, args.weights_dataset)
    path = resolve_checkpoint_artifact(
        model=args.model,
        backbone=args.backbone,
        weights_dataset=args.weights_dataset,
        auto_download=bool(args.download),
    )
    _print(
        {
            "command": "resolve-checkpoint",
            "artifact": entry,
            "resolved_path": str(path) if path is not None else None,
            "downloaded": bool(args.download),
        }
    )


def cmd_resolve_dataset_subset(args: argparse.Namespace) -> None:
    resolved = resolve_dataset_subset(args.dataset, args.subset)
    _print({"command": "resolve-dataset-subset", "dataset": args.dataset, "subset": args.subset, "resolved": resolved})


def cmd_create_hf_repo(args: argparse.Namespace) -> None:
    url = create_hf_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=bool(args.private),
        exist_ok=bool(args.exist_ok),
    )
    _print({"command": "create-hf-repo", "repo_id": args.repo_id, "repo_type": args.repo_type, "url": url})


def cmd_upload_checkpoint(args: argparse.Namespace) -> None:
    entry = get_checkpoint_artifact(args.model, args.backbone, args.weights_dataset)
    if entry is None:
        raise ValueError(
            f"No checkpoint manifest entry for {args.model}/{args.backbone}/{args.weights_dataset}"
        )
    if args.create_repo:
        create_hf_repo(
            repo_id=str(entry["repo_id"]),
            repo_type="model",
            private=bool(args.private),
            exist_ok=True,
        )
    path_in_repo = args.path_in_repo or str(entry["filename"])
    url = upload_file_to_hf(
        repo_id=str(entry["repo_id"]),
        repo_type="model",
        local_path=Path(args.local_path),
        path_in_repo=path_in_repo,
        commit_message=f"Upload checkpoint {args.model}/{args.backbone}/{args.weights_dataset}",
    )
    _print(
        {
            "command": "upload-checkpoint",
            "repo_id": entry["repo_id"],
            "path_in_repo": path_in_repo,
            "url": url,
        }
    )


def cmd_build_dataset_archive(args: argparse.Namespace) -> None:
    entry = get_dataset_subset_entry(args.dataset, args.subset)
    if entry is None:
        raise ValueError(f"No dataset subset manifest entry for {args.dataset}/{args.subset}")
    output_path = Path(args.output) if args.output else Path.cwd() / str(entry["filename"])
    archive_path = build_dataset_archive(
        dataset=args.dataset,
        subset=args.subset,
        source_dir=Path(args.source_dir),
        output_path=output_path,
        force=bool(args.force),
    )
    _print(
        {
            "command": "build-dataset-archive",
            "dataset": args.dataset,
            "subset": args.subset,
            "archive_path": str(archive_path),
        }
    )


def cmd_upload_dataset_archive(args: argparse.Namespace) -> None:
    entry = get_dataset_subset_entry(args.dataset, args.subset)
    if entry is None:
        raise ValueError(f"No dataset subset manifest entry for {args.dataset}/{args.subset}")
    cleanup_archive = False
    if args.archive_path:
        archive_path = Path(args.archive_path)
    else:
        if not args.source_dir:
            raise ValueError("--source-dir is required when --archive-path is not provided")
        archive_path = build_dataset_archive(
            dataset=args.dataset,
            subset=args.subset,
            source_dir=Path(args.source_dir),
            output_path=Path.cwd() / str(entry["filename"]),
            force=True,
        )
        cleanup_archive = not args.keep_archive
    if args.create_repo:
        create_hf_repo(
            repo_id=str(entry["repo_id"]),
            repo_type="dataset",
            private=bool(args.private),
            exist_ok=True,
        )
    path_in_repo = args.path_in_repo or str(entry["filename"])
    try:
        url = upload_file_to_hf(
            repo_id=str(entry["repo_id"]),
            repo_type="dataset",
            local_path=archive_path,
            path_in_repo=path_in_repo,
            commit_message=f"Upload dataset subset {args.dataset}/{args.subset}",
        )
    finally:
        if cleanup_archive and archive_path.exists():
            archive_path.unlink()
    _print(
        {
            "command": "upload-dataset-archive",
            "repo_id": entry["repo_id"],
            "path_in_repo": path_in_repo,
            "url": url,
        }
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "list-checkpoints":
        cmd_list_checkpoints(args)
    elif args.command == "list-dataset-subsets":
        cmd_list_dataset_subsets(args)
    elif args.command == "resolve-checkpoint":
        cmd_resolve_checkpoint(args)
    elif args.command == "resolve-dataset-subset":
        cmd_resolve_dataset_subset(args)
    elif args.command == "create-hf-repo":
        cmd_create_hf_repo(args)
    elif args.command == "upload-checkpoint":
        cmd_upload_checkpoint(args)
    elif args.command == "build-dataset-archive":
        cmd_build_dataset_archive(args)
    elif args.command == "upload-dataset-archive":
        cmd_upload_dataset_archive(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")

#!/usr/bin/env python3
import argparse
import os
from contextlib import contextmanager

from huggingface_hub import snapshot_download

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_ID = "damo-vilab/text-to-video-ms-1.7b"
DEFAULT_CACHE_DIR = os.path.join(THIS_DIR, "hf-cache")
DEFAULT_LOCK_PATH = os.path.join(THIS_DIR, ".videopure_model_download.lock")


@contextmanager
def _download_lock(lock_path=DEFAULT_LOCK_PATH):
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    lock_file = open(lock_path, "w")
    try:
        try:
            import fcntl
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except ImportError:
            pass
        yield
    finally:
        try:
            import fcntl
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except ImportError:
            pass
        lock_file.close()


def ensure_model_downloaded(model_id=DEFAULT_MODEL_ID, cache_dir=DEFAULT_CACHE_DIR, quiet=False):
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    if os.getenv("HF_HUB_DISABLE_PROGRESS_BARS") is None:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

    with _download_lock():
        try:
            local_path = snapshot_download(
                repo_id=model_id,
                cache_dir=cache_dir,
                local_files_only=True,
            )
            if not quiet:
                print(f"[VideoPure] Cache hit for {model_id}")
                print(f"[VideoPure] Snapshot path: {local_path}")
            return local_path, False
        except Exception:
            if not quiet:
                print(f"[VideoPure] Cache miss for {model_id}. Downloading...")
                print(f"[VideoPure] Cache dir: {cache_dir}")
            path = snapshot_download(
                repo_id=model_id,
                cache_dir=cache_dir,
                local_files_only=False,
            )
            if not quiet:
                print(f"[VideoPure] Download complete. Snapshot path: {path}")
            return path, True


def main():
    parser = argparse.ArgumentParser(description="Predownload VideoPure diffusers weights to local cache.")
    parser.add_argument(
        "--model-id",
        default=os.getenv("VIDEOPURE_MODEL_ID", DEFAULT_MODEL_ID),
        help="Hugging Face model repo id",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("VIDEOPURE_CACHE_DIR", DEFAULT_CACHE_DIR),
        help="Optional custom HF cache directory",
    )
    args = parser.parse_args()

    print(f"[VideoPure] Ensuring local snapshot: {args.model_id}")
    _, downloaded = ensure_model_downloaded(model_id=args.model_id, cache_dir=args.cache_dir, quiet=False)
    if downloaded:
        print("[VideoPure] Status: downloaded")
    else:
        print("[VideoPure] Status: already present")


if __name__ == "__main__":
    main()

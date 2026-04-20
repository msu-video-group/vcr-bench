from __future__ import annotations

import random
import time
import copy

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from vcr_bench.datasets.base import BaseVideoDataset
from vcr_bench.datasets.loaded import collate_loaded_video_batch
from vcr_bench.models.base import BaseVideoClassifier
from vcr_bench.types import PredictionBundle


def run_accuracy(
    model: BaseVideoClassifier,
    dataset: BaseVideoDataset,
    num_videos: int,
    seed: int = 42,
    skip_decode_errors: bool = True,
    log_every: int = 10,
    return_full: bool = False,
    batch_size: int = 1,
    num_workers: int = 0,
    dataset_loading: bool = False,
    instant_preprocessing: bool = False,
    pipeline_stage: str = "test",
) -> dict:
    total = min(num_videos, len(dataset))
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    indices = indices[:total]

    correct = 0
    count = 0
    skipped = 0
    start = time.time()
    # Ensure the model uses the requested stage config for preprocessing paths.
    model.build_data_pipeline(pipeline_stage)

    if dataset_loading or batch_size > 1 or num_workers > 0:
        # Work on a shallow copy so we don't leave loading configured on the caller's dataset instance.
        configured_dataset = copy.copy(dataset).configure_loading(
            model.build_data_pipeline(pipeline_stage),
            instant_preprocessing=instant_preprocessing,
        )
        subset = Subset(configured_dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=max(1, batch_size),
            shuffle=False,
            num_workers=max(0, num_workers),
            collate_fn=collate_loaded_video_batch,
        )
        seen = 0
        for batch in tqdm(loader, total=len(loader), desc="eval"):
            samples = batch["samples"]
            try:
                if batch["stackable"]:
                    probs_batch = model.predict(batch["tensor"], input_format=batch["input_format"])
                else:
                    probs_batch = torch.stack([model.predict(t, input_format=batch["input_format"]) for t in batch["tensor"]], dim=0)
            except Exception:
                if not skip_decode_errors:
                    raise
                skipped += len(samples)
                continue

            for j, sample in enumerate(samples):
                probs = probs_batch[j]
                pred_label = int(probs.argmax().item())
                if sample.label is not None and pred_label == int(sample.label):
                    correct += 1
                count += 1
                seen += 1
                if log_every and seen % log_every == 0:
                    _ = correct / max(count, 1)
    else:
        for i, idx in enumerate(tqdm(indices, total=len(indices), desc="eval"), start=1):
            sample = dataset[idx]
            try:
                x = model.load_video(sample.path)
                pred = model.predict(x, input_format="NTHWC", return_full=return_full)
            except Exception:
                if skip_decode_errors:
                    skipped += 1
                    continue
                raise

            if isinstance(pred, PredictionBundle):
                probs = pred.probs
            else:
                probs = pred
            pred_label = int(probs.argmax().item())
            if sample.label is not None and pred_label == int(sample.label):
                correct += 1
            count += 1

            if log_every and i % log_every == 0:
                _ = correct / max(count, 1)

    elapsed = time.time() - start
    accuracy = (correct / count) if count else 0.0
    return {
        "correct": correct,
        "count": count,
        "accuracy": accuracy,
        "skipped": skipped,
        "elapsed_sec": elapsed,
        "model_name": getattr(model, "model_name", "unknown"),
        "dataset_name": dataset.__class__.__name__,
    }

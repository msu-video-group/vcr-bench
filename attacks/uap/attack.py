from __future__ import annotations

import gc

import torch

from vg_videobench.attacks.base import BaseVideoAttack
from vg_videobench.attacks.utils import cross_entropy_from_probs, flatten_temporal_video, iter_dataset_items, predict_probs, restore_temporal_video, safe_cuda_empty_cache
from vg_videobench.types import LoadedDatasetItem


def apply_centered_trigger(video_tensor: torch.Tensor, trigger: torch.Tensor) -> torch.Tensor:
    if video_tensor.ndim == 5:
        shape = tuple(video_tensor.shape)
        flat = video_tensor.reshape(shape[0] * shape[1], *shape[-3:])
        return apply_centered_trigger(flat, trigger).reshape(shape)
    t, h, w, _ = video_tensor.shape
    trigger_t, trigger_h, trigger_w, _ = trigger.shape
    start_t = max(0, (t - trigger_t) // 2)
    start_h = max(0, (h - trigger_h) // 2)
    start_w = max(0, (w - trigger_w) // 2)
    end_t = min(t, start_t + trigger_t)
    end_h = min(h, start_h + trigger_h)
    end_w = min(w, start_w + trigger_w)
    trigger_start_t = max(0, -((t - trigger_t) // 2))
    trigger_start_h = max(0, -((h - trigger_h) // 2))
    trigger_start_w = max(0, -((w - trigger_w) // 2))
    trigger_end_t = trigger_start_t + (end_t - start_t)
    trigger_end_h = trigger_start_h + (end_h - start_h)
    trigger_end_w = trigger_start_w + (end_w - start_w)
    perturbed_video = video_tensor.clone()
    if end_t > start_t and end_h > start_h and end_w > start_w:
        perturbed_video[start_t:end_t, start_h:end_h, start_w:end_w, :] += trigger[
            trigger_start_t:trigger_end_t,
            trigger_start_h:trigger_end_h,
            trigger_start_w:trigger_end_w,
            :,
        ]
    return perturbed_video


class UAPAttack(BaseVideoAttack):
    attack_name = "uap"

    def __init__(self, eps: float = 10.0, alpha: float = 10.0, steps: int = 10, sample_chunk_size: int | None = None) -> None:
        super().__init__(eps=eps, alpha=alpha, steps=steps, clip_min=0.0, clip_max=255.0, sample_chunk_size=sample_chunk_size)
        self.trained_trigger: torch.Tensor | None = None

    def prepare(self, *, model, dataset, indices, seed, targeted, pipeline_stage, verbose) -> None:
        del seed, targeted, pipeline_stage
        device = torch.device(getattr(model, "device", "cpu"))
        current_uap: torch.Tensor | None = None
        epochs = 1
        learning_rate = self.alpha if indices else 30.0
        if indices:
            learning_rate = 30.0 / (len(indices) * epochs)
        for _ in range(epochs):
            for _, item in iter_dataset_items(dataset, indices):
                if not isinstance(item, LoadedDatasetItem):
                    continue
                video = item.tensor.detach().clone().float()
                flat_video, flat_shape = flatten_temporal_video(video)
                probs = predict_probs(model, restore_temporal_video(flat_video.to(device), flat_shape), input_format=item.input_format, enable_grad=False)
                original_label = int(torch.argmax(probs).item())
                if current_uap is None:
                    current_uap = torch.zeros((flat_video.shape[0], 480, 720, 3), requires_grad=True, device=device)
                else:
                    current_uap = current_uap.clone().detach().requires_grad_(True).to(device)
                optimizer = torch.optim.Adam([current_uap], lr=learning_rate)
                perturbed_video = apply_centered_trigger(restore_temporal_video(flat_video.to(device), flat_shape), current_uap).clamp(0.0, 255.0)
                adv_probs = predict_probs(model, perturbed_video, input_format=item.input_format, enable_grad=True)
                loss = -cross_entropy_from_probs(adv_probs, original_label)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    current_uap.data.clamp_(-self.eps, self.eps)
                safe_cuda_empty_cache()
                gc.collect()
        self.trained_trigger = None if current_uap is None else current_uap.detach().cpu()
        if verbose and self.trained_trigger is None:
            print("[uap] prepare finished without training samples", flush=True)

    def attack_sampled(self, model, x, *, input_format="NTHWC", y=None, targeted=False):
        del y, targeted
        if self.trained_trigger is None:
            raise RuntimeError("UAP attack requires prepare() to run before attack()")
        init_tensor = x.detach().clone().float().to(getattr(model, "device", x.device))
        attacked = apply_centered_trigger(init_tensor, self.trained_trigger.to(init_tensor.device)).clamp(0.0, 255.0)
        benign_probs = predict_probs(model, init_tensor, input_format=input_format, enable_grad=False)
        final_probs = predict_probs(model, attacked, input_format=input_format, enable_grad=False)
        self.last_result = {
            "iter_count": 1,
            "target_label": -1,
            "benign_label": int(torch.argmax(benign_probs).item()),
            "final_label": int(torch.argmax(final_probs).item()),
        }
        return attacked.detach().cpu()


ATTACK_CLASS = UAPAttack


def create(**kwargs) -> UAPAttack:
    return UAPAttack(**kwargs)

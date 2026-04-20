from __future__ import annotations

import torch

from vcr_bench.attacks.utils import cross_entropy_from_probs, flatten_temporal_video, freeze_model_weights, predict_probs, restore_temporal_video, safe_cuda_reset_peak_memory_stats


def metric_attack(
    model,
    x: torch.Tensor,
    metric,
    *,
    k: float,
    eps: float,
    steps: int,
    targeted: bool,
    input_format: str,
    y=None,
    target_conf: float = 0.5,
):
    freeze_model_weights(model)
    safe_cuda_reset_peak_memory_stats(getattr(model, "device", x.device))
    init_tensor = x.detach().clone().float().to(getattr(model, "device", x.device))
    _, restore_shape = flatten_temporal_video(init_tensor)
    pert = torch.zeros_like(init_tensor, requires_grad=True)
    init_probs = predict_probs(model, init_tensor, input_format=input_format, enable_grad=False)
    benign_label = int(torch.argmax(init_probs).item())
    benign_conf = float(init_probs[benign_label].item())
    target_label = int(torch.argmin(init_probs).item()) if targeted and y is None else (int(y) if y is not None else -1)

    final_label = benign_label
    final_conf = benign_conf
    iteration = 0
    for iteration in range(1, int(steps) + 1):
        attacked = init_tensor + pert
        probs = predict_probs(model, attacked, input_format=input_format, enable_grad=True)
        class_loss = cross_entropy_from_probs(probs, target_label if targeted else benign_label) * (1 if targeted else -1)
        class_loss.backward()
        save_grad = pert.grad.clone()
        pert.grad = None

        attacked = init_tensor + pert
        attacked_flat, _ = flatten_temporal_video(attacked)
        init_flat, _ = flatten_temporal_video(init_tensor)
        current_metric = metric(attacked_flat, init_flat)
        metric_loss = 0.05 * float(k) * current_metric
        metric_loss.backward()
        grad = pert.grad + save_grad
        pert.data -= torch.sign(grad)
        pert.data.clamp_(-float(eps), float(eps))
        pert.grad = None

        attacked = restore_temporal_video((init_tensor + pert).clamp(0.0, 255.0), restore_shape)
        cur_probs = predict_probs(model, attacked, input_format=input_format, enable_grad=False)
        final_label = int(torch.argmax(cur_probs).item())
        final_conf = float(cur_probs[final_label].item())
        if (targeted and final_label == target_label and final_conf >= target_conf) or ((not targeted) and final_label != benign_label):
            break

    meta = {
        "iter_count": int(iteration),
        "target_label": int(target_label),
        "benign_label": int(benign_label),
        "benign_conf": float(benign_conf),
        "final_label": int(final_label),
        "final_conf": float(final_conf),
    }
    return restore_temporal_video((init_tensor + pert).clamp(0.0, 255.0), restore_shape).detach().cpu(), meta

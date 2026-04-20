from __future__ import annotations

import torch

from vcr_bench.defences.base import BaseVideoDefence


class BilateralDefence(BaseVideoDefence):
    voting_count = 1

    def __init__(self, sigma_color: float = 5.0, sigma_time: float = 0.1, window_radius: int = 1):
        self.sigma_color = sigma_color
        self.sigma_time = sigma_time
        self.window_radius = window_radius

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, H, W, C] float32 [0..255]
        N, T, H, W, C = x.shape
        time_range = torch.arange(T, device=x.device).float()
        filtered = torch.zeros_like(x)

        for n in range(N):
            for t in range(T):
                center_frame = x[n, t]
                weights_sum = torch.zeros(C, device=x.device)
                values_sum = torch.zeros_like(center_frame)

                for k in range(max(0, t - self.window_radius), min(T, t + self.window_radius + 1)):
                    if k == t:
                        continue
                    current_frame = x[n, k]
                    time_diff = torch.abs(time_range[t] - time_range[k])
                    time_weight = torch.exp(-time_diff ** 2 / (2 * self.sigma_time ** 2))
                    color_diff = torch.abs(center_frame - current_frame)
                    color_weight = torch.exp(-color_diff ** 2 / (2 * self.sigma_color ** 2))
                    total_weight = time_weight * color_weight
                    values_sum = values_sum + total_weight * current_frame
                    weights_sum = weights_sum + total_weight.mean(dim=(0, 1))

                if weights_sum.sum() > 0:
                    filtered[n, t] = values_sum / weights_sum.view(1, 1, C)
                else:
                    filtered[n, t] = center_frame

        return filtered


def create(**kwargs) -> BilateralDefence:
    return BilateralDefence(**kwargs)

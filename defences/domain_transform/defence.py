from __future__ import annotations

import torch

from vg_videobench.defences.base import BaseVideoDefence


class DomainTransformDefence(BaseVideoDefence):
    voting_count = 1

    def __init__(self, sigma_r: float = 0.1, sigma_t: float = 2.0, iterations: int = 3):
        self.sigma_r = sigma_r
        self.sigma_t = sigma_t
        self.iterations = iterations

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, H, W, C]
        imgs = x.clone()
        N, T, H, W, C = imgs.shape
        device = imgs.device

        for _ in range(self.iterations):
            transformed_coords = torch.arange(T, dtype=torch.float32, device=device).view(1, T, 1, 1, 1)
            transformed_coords = transformed_coords.expand(N, T, H, W, 1).clone()

            for t in range(1, T):
                intensity_diff = torch.abs(imgs[:, t] - imgs[:, t - 1]).mean(dim=-1, keepdim=True)
                transformed_coords[:, t] = transformed_coords[:, t - 1] + 1 + self.sigma_t / (self.sigma_r + 1e-8) * intensity_diff

            time_values = torch.zeros(N, T, H, W, C, device=device)
            time_weights = torch.zeros(N, T, H, W, 1, device=device)

            sorted_indices = torch.argsort(transformed_coords.squeeze(-1).view(N, T, -1), dim=1)
            sorted_indices = sorted_indices.view(N, T, H, W)

            for n in range(N):
                for t in range(T):
                    sorted_idx = sorted_indices[n, t]
                    h_idx = sorted_idx // W
                    w_idx = sorted_idx % W
                    frame_value = imgs[n, :, h_idx, w_idx]
                    time_values[n, t] += frame_value.mean(dim=0)
                    time_weights[n, t] += 1.0

            time_values = time_values / (time_weights + 1e-8)

            for n in range(N):
                for t in range(T):
                    sorted_idx = sorted_indices[n, t]
                    h_idx = sorted_idx // W
                    w_idx = sorted_idx % W
                    imgs[n, t, h_idx, w_idx] = time_values[n, t]

        return imgs


def create(**kwargs) -> DomainTransformDefence:
    return DomainTransformDefence(**kwargs)

from __future__ import annotations

import random

import torch

from vcr_bench.defences.base import BaseVideoDefence


class RandomizedSmoothingDefence(BaseVideoDefence):
    voting_count = 1

    def __init__(self, sigma_range: tuple[float, float] = (5.0, 25.0)):
        self.sigma_range = sigma_range

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, H, W, C]
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
        noise = torch.randn_like(x) * sigma
        return torch.clamp(x + noise, 0.0, 255.0)


def create(**kwargs) -> RandomizedSmoothingDefence:
    return RandomizedSmoothingDefence(**kwargs)

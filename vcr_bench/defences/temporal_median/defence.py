from __future__ import annotations

import torch

from vcr_bench.defences.base import BaseVideoDefence


class TemporalMedianDefence(BaseVideoDefence):
    voting_count = 1

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, H, W, C]
        N, T, H, W, C = x.shape
        filtered = torch.zeros_like(x)
        for i in range(T):
            start = max(0, i - 1)
            end = min(T, i + 2)
            window = x[:, start:end, :, :, :]  # [N, window, H, W, C]
            filtered[:, i] = torch.median(window, dim=1)[0]
        return filtered


def create(**kwargs) -> TemporalMedianDefence:
    return TemporalMedianDefence(**kwargs)

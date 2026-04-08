from __future__ import annotations

import random

import torch

from vg_videobench.defences.base import BaseVideoDefence


class FlipDefence(BaseVideoDefence):
    voting_count = 1

    def __init__(self, flip_prob: float = 1.0):
        self.flip_prob = flip_prob

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, H, W, C]
        if random.random() < self.flip_prob:
            return x.flip(dims=[3])  # horizontal flip along W axis
        return x


def create(**kwargs) -> FlipDefence:
    return FlipDefence(**kwargs)

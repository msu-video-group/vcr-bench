from __future__ import annotations

import random

import torch

from vcr_bench.defences.base import BaseVideoDefence


class ShuffleDefence(BaseVideoDefence):
    voting_count = 1

    def __init__(self, h1: int = 2, h2: int = 4):
        self.h1 = h1
        self.h2 = h2

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, H, W, C]
        N, T, H, W, C = x.shape
        # Work in [N, T, C, H, W] for easier frame indexing
        x_perm = x.permute(0, 1, 4, 2, 3)
        shuffled = torch.zeros_like(x_perm)

        for n in range(N):
            frames = x_perm[n]  # [T, C, H, W]
            result_frames = []
            t = 0
            while t < T:
                possible = [t + off for off in range(-self.h1, self.h1 + 1) if off != 0 and 0 <= t + off < T]
                if possible:
                    chosen = random.choice(possible)
                    end = min(chosen + self.h2, T)
                    chunk = frames[chosen:end]
                    for frame in chunk:
                        if t < T:
                            result_frames.append(frame.unsqueeze(0))
                            t += 1
                        else:
                            break
                else:
                    result_frames.append(frames[t].unsqueeze(0))
                    t += 1

            if result_frames:
                batch_shuffled = torch.cat(result_frames, dim=0)
                if batch_shuffled.shape[0] < T:
                    pad = batch_shuffled[-1:].repeat(T - batch_shuffled.shape[0], 1, 1, 1)
                    batch_shuffled = torch.cat([batch_shuffled, pad], dim=0)
                shuffled[n] = batch_shuffled

        return shuffled.permute(0, 1, 3, 4, 2)  # [N, T, H, W, C]


def create(**kwargs) -> ShuffleDefence:
    return ShuffleDefence(**kwargs)

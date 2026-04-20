from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F

from vcr_bench.defences.base import BaseVideoDefence


class RotateDefence(BaseVideoDefence):
    voting_count = 1

    def __init__(self, angle_range: tuple[float, float] = (-30.0, 30.0)):
        self.angle_range = angle_range

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, H, W, C]
        N, T, H, W, C = x.shape
        angle = random.uniform(self.angle_range[0], self.angle_range[1])
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        rot_mat = torch.tensor(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]],
            dtype=torch.float32,
            device=x.device,
        )
        rot_mat = rot_mat.unsqueeze(0).expand(N * T, -1, -1)

        x_perm = x.permute(0, 1, 4, 2, 3).reshape(N * T, C, H, W)  # [N*T, C, H, W]
        grid = F.affine_grid(rot_mat, size=(N * T, C, H, W), align_corners=False)
        rotated = F.grid_sample(x_perm, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        rotated = rotated.reshape(N, T, C, H, W).permute(0, 1, 3, 4, 2)  # [N, T, H, W, C]
        return rotated


def create(**kwargs) -> RotateDefence:
    return RotateDefence(**kwargs)

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from vcr_bench.defences.base import BaseVideoDefence


class GaussianBlurDefence(BaseVideoDefence):
    voting_count = 1

    def __init__(
        self,
        sigma_range: tuple[float, float] = (5.0, 25.0),
        kernel_size_range: tuple[int, int] = (3, 11),
    ):
        self.sigma_range = sigma_range
        self.kernel_size_range = kernel_size_range

    @staticmethod
    def _gaussian_kernel_2d(kernel_size: int, sigma: float) -> torch.Tensor:
        kernel = torch.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                x = i - center
                y = j - center
                kernel[i, j] = torch.exp(torch.tensor(-(x ** 2 + y ** 2) / (2 * sigma ** 2)))
        kernel /= kernel.sum()
        return kernel

    def _blur(self, x: torch.Tensor, sigma: float, kernel_size: int) -> torch.Tensor:
        # x: [N, T, H, W, C]
        N, T, H, W, C = x.shape
        x_perm = x.permute(0, 1, 4, 2, 3)  # [N, T, C, H, W]
        kernel = self._gaussian_kernel_2d(kernel_size, sigma).to(x.device)
        kernel = kernel.view(1, 1, 1, kernel_size, kernel_size).expand(C, 1, 1, kernel_size, kernel_size)
        padding = kernel_size // 2
        blurred_batches = []
        for n in range(N):
            blurred_frames = []
            for t in range(T):
                frame = x_perm[n, t].unsqueeze(0).unsqueeze(2)  # [1, C, 1, H, W]
                frame = F.conv3d(frame, kernel, padding=(0, padding, padding), groups=C)
                blurred_frames.append(frame)
            blurred_batches.append(torch.cat(blurred_frames, dim=2))
        blurred = torch.cat(blurred_batches, dim=0)
        blurred = torch.clamp(blurred, 0, 255)
        return blurred.permute(0, 2, 3, 4, 1)  # [N, T, H, W, C]

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
        ks_low, ks_high = self.kernel_size_range
        odd_sizes = [k for k in range(ks_low, ks_high + 1, 2)]
        kernel_size = random.choice(odd_sizes)
        return self._blur(x, sigma, kernel_size)


def create(**kwargs) -> GaussianBlurDefence:
    return GaussianBlurDefence(**kwargs)

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from vcr_bench.defences.base import BaseVideoDefence


class CropResizeDefence(BaseVideoDefence):
    voting_count = 1

    def __init__(self, crop_range: tuple[int, int] = (2, 20)):
        self.crop_range = crop_range

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, H, W, C]
        N, T, H, W, C = x.shape
        crop_pixels = random.randint(self.crop_range[0], self.crop_range[1])
        start_h = random.randint(0, crop_pixels)
        start_w = random.randint(0, crop_pixels)
        end_h = H - (crop_pixels - start_h)
        end_w = W - (crop_pixels - start_w)

        cropped = x[:, :, start_h:end_h, start_w:end_w, :]  # [N, T, h, w, C]
        cropped = cropped.permute(0, 1, 4, 2, 3)  # [N, T, C, h, w]
        h_crop = end_h - start_h
        w_crop = end_w - start_w
        resized = F.interpolate(
            cropped.reshape(-1, C, h_crop, w_crop),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        resized = resized.reshape(N, T, C, H, W).permute(0, 1, 3, 4, 2)  # [N, T, H, W, C]
        return resized


def create(**kwargs) -> CropResizeDefence:
    return CropResizeDefence(**kwargs)

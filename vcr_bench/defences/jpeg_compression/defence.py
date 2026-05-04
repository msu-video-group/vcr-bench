from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image

from vcr_bench.defences.base import BaseVideoDefence


class JpegCompressionDefence(BaseVideoDefence):
    voting_count = 1

    def __init__(self, quality: int = 75):
        """
        Args:
            quality: JPEG quality factor in [1, 95]. Lower = more compression/destruction.
                     75 preserves clean accuracy well; 30 is stronger; 10 is very aggressive.
        """
        if not (1 <= quality <= 95):
            raise ValueError(f"quality must be in [1, 95], got {quality}")
        self.quality = quality

    def _compress_frame(self, frame: np.ndarray) -> np.ndarray:
        buf = io.BytesIO()
        Image.fromarray(frame, mode="RGB").save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        return np.array(Image.open(buf))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, H, W, C], float32, values in [0, 255]
        device = x.device
        x_uint8 = x.clamp(0, 255).byte().cpu().numpy()
        N, T = x_uint8.shape[:2]
        out = np.empty_like(x_uint8)
        for n in range(N):
            for t in range(T):
                out[n, t] = self._compress_frame(x_uint8[n, t])
        return torch.from_numpy(out).float().to(device)


def create(**kwargs) -> JpegCompressionDefence:
    return JpegCompressionDefence(**kwargs)

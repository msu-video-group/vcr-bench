from __future__ import annotations

import numpy as np
import torch

from vcr_bench.defences.base import BaseVideoDefence

# --------------------------------------------------------------------------
# Standard JPEG quantization tables (transposed, as in the reference DiffJPEG
# implementation by mlomnitz, github.com/mlomnitz/DiffJPEG).
# --------------------------------------------------------------------------
_Y_TABLE = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float32,
).T

_C_TABLE = np.full((8, 8), 99.0, dtype=np.float32)
_C_TABLE[:4, :4] = np.array(
    [
        [17, 18, 24, 47],
        [18, 21, 26, 66],
        [24, 26, 56, 99],
        [47, 66, 99, 99],
    ],
    dtype=np.float32,
).T

# RGB <-> YCbCr matrices (channels-last, [0, 255] range).
_RGB2YCBCR = np.array(
    [
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312],
    ],
    dtype=np.float32,
).T
_YCBCR2RGB = np.array(
    [
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0],
    ],
    dtype=np.float32,
).T


def _build_dct_tensor() -> np.ndarray:
    tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for x in range(8):
        for y in range(8):
            for u in range(8):
                for v in range(8):
                    tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
                        (2 * y + 1) * v * np.pi / 16
                    )
    return tensor


_DCT_TENSOR = _build_dct_tensor()
_ALPHA = np.ones(8, dtype=np.float32)
_ALPHA[0] = 1.0 / np.sqrt(2)
_DCT_SCALE = np.outer(_ALPHA, _ALPHA) * 0.25  # [8, 8]
# IDCT uses the alpha weights on the (u, v) frequency axes.
_IDCT_SCALE = np.outer(_ALPHA, _ALPHA)  # [8, 8]


def quality_to_factor(quality: float) -> float:
    """Map a JPEG quality factor in (0, 100) to a quantization scale."""
    if quality < 50:
        q = 5000.0 / quality
    else:
        q = 200.0 - quality * 2.0
    return q / 100.0


def diff_round(x: torch.Tensor) -> torch.Tensor:
    """Differentiable approximation of rounding (Shin & Song, 2017).

    ``round(x) + (x - round(x))**3`` matches integer rounding in the forward
    pass while keeping a non-zero, smooth gradient everywhere, so adaptive
    white-box attacks can backprop through the quantization step.
    """
    r = torch.round(x)
    return r + (x - r) ** 3


class DiffJpegDefence(BaseVideoDefence):
    """Differentiable JPEG compression defence.

    Implements the full JPEG pipeline (RGB->YCbCr, 4:2:0 chroma subsampling,
    8x8 block DCT, quantization, dequantization, IDCT, reconstruction) with a
    soft, differentiable rounding step. Unlike the PIL-based ``jpeg_compression``
    defence, gradients flow through ``transform``, so it is meaningful as an
    *adaptive* (white-box) compression defence.
    """

    voting_count = 1
    differentiable = True

    def __init__(self, quality: int = 75):
        """
        Args:
            quality: JPEG quality factor in [1, 99]. Lower = more compression.
                     75 preserves clean accuracy well; 30 is stronger.
        """
        if not (1 <= quality <= 99):
            raise ValueError(f"quality must be in [1, 99], got {quality}")
        self.quality = int(quality)
        self.factor = quality_to_factor(self.quality)
        # Cached device/dtype-resident constants, keyed by (device, dtype).
        self._consts: dict[tuple, dict[str, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    def _get_consts(self, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        key = (device, dtype)
        c = self._consts.get(key)
        if c is None:
            to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dtype)
            c = {
                "y_table": to(_Y_TABLE),
                "c_table": to(_C_TABLE),
                "rgb2ycbcr": to(_RGB2YCBCR),
                "ycbcr2rgb": to(_YCBCR2RGB),
                "shift": to(np.array([0.0, 128.0, 128.0], dtype=np.float32)),
                "dct": to(_DCT_TENSOR),
                "dct_scale": to(_DCT_SCALE),
                "idct_scale": to(_IDCT_SCALE),
            }
            self._consts[key] = c
        return c

    # ----- compression -------------------------------------------------
    @staticmethod
    def _block_split(plane: torch.Tensor) -> torch.Tensor:
        # plane: [B, H, W] -> [B, num_blocks, 8, 8]
        B, H, W = plane.shape
        x = plane.view(B, H // 8, 8, W // 8, 8)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        return x.view(B, -1, 8, 8)

    @staticmethod
    def _block_merge(blocks: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # blocks: [B, num_blocks, 8, 8] -> [B, H, W]
        B = blocks.shape[0]
        x = blocks.view(B, height // 8, width // 8, 8, 8)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        return x.view(B, height, width)

    def _dct(self, blocks: torch.Tensor, c: dict) -> torch.Tensor:
        # blocks: [B, N, 8, 8], range [0, 255]
        shifted = blocks - 128.0
        coeff = torch.tensordot(shifted, c["dct"], dims=2)  # [B, N, 8, 8]
        return coeff * c["dct_scale"]

    def _idct(self, coeff: torch.Tensor, c: dict) -> torch.Tensor:
        # Inverse: weight the frequency axes by alpha, then inverse cosine sum.
        weighted = coeff * c["idct_scale"]
        out = 0.25 * torch.tensordot(weighted, c["dct"].permute(2, 3, 0, 1), dims=2)
        return out + 128.0

    def _compress_plane(self, plane: torch.Tensor, table: torch.Tensor, c: dict) -> torch.Tensor:
        blocks = self._block_split(plane)
        coeff = self._dct(blocks, c)
        q = coeff / (table * self.factor)
        q = diff_round(q)
        # dequantize + inverse
        coeff_r = q * (table * self.factor)
        rec = self._idct(coeff_r, c)
        return self._block_merge(rec, plane.shape[1], plane.shape[2])

    def _jpeg(self, image: torch.Tensor, c: dict) -> torch.Tensor:
        # image: [B, H, W, 3], range [0, 255]
        ycbcr = image @ c["rgb2ycbcr"] + c["shift"]
        y = ycbcr[..., 0]
        cb = ycbcr[..., 1]
        cr = ycbcr[..., 2]

        # 4:2:0 chroma subsampling via 2x2 average pooling.
        def _down(p: torch.Tensor) -> torch.Tensor:
            B, H, W = p.shape
            return p.view(B, H // 2, 2, W // 2, 2).mean(dim=(2, 4))

        cb_d = _down(cb)
        cr_d = _down(cr)

        y_r = self._compress_plane(y, c["y_table"], c)
        cb_r = self._compress_plane(cb_d, c["c_table"], c)
        cr_r = self._compress_plane(cr_d, c["c_table"], c)

        # Nearest-neighbour chroma upsampling (matches subsampling block size).
        cb_u = cb_r.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
        cr_u = cr_r.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)

        ycbcr_r = torch.stack([y_r, cb_u, cr_u], dim=-1)
        rgb = (ycbcr_r - c["shift"]) @ c["ycbcr2rgb"]
        return rgb

    # ------------------------------------------------------------------
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, H, W, C], float32, values in [0, 255]
        N, T, H, W, C = x.shape
        dtype = x.dtype if x.dtype.is_floating_point else torch.float32
        c = self._get_consts(x.device, dtype)

        img = x.reshape(N * T, H, W, C).to(dtype)

        # JPEG needs H, W divisible by 16 (8x8 blocks after 2x chroma downsample).
        pad_h = (-H) % 16
        pad_w = (-W) % 16
        if pad_h or pad_w:
            # replicate-pad on the spatial axes (pad expects NCHW-style last dims)
            img = img.permute(0, 3, 1, 2)
            img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode="replicate")
            img = img.permute(0, 2, 3, 1)

        rec = self._jpeg(img, c)

        if pad_h or pad_w:
            rec = rec[:, :H, :W, :]

        rec = torch.clamp(rec, 0, 255)
        return rec.reshape(N, T, H, W, C).to(x.dtype)


def create(**kwargs) -> DiffJpegDefence:
    return DiffJpegDefence(**kwargs)

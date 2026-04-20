from __future__ import annotations

import torch
import torch.nn.functional as F

from .converters import ensure_nthwc, to_format


def to_float_nthwc(x: torch.Tensor, scale_if_uint8: bool = True) -> torch.Tensor:
    x = ensure_nthwc(x, "NTHWC")
    if x.dtype == torch.uint8:
        x = x.float()
        if scale_if_uint8:
            x = x / 255.0
        return x
    x = x.float()
    if scale_if_uint8:
        x = x / 255.0
    return x


def temporal_sample_nthwc(
    x: torch.Tensor,
    clip_len: int,
    num_clips: int,
    frame_interval: int = 1,
    temporal_strategy: str = "uniform",
    sample_range: int = 64,
    num_sample_positions: int = 10,
    return_indices: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    x = ensure_nthwc(x, "NTHWC")
    if x.ndim != 5:
        raise ValueError("Expected NTHWC tensor")
    if x.shape[0] != 1:
        raise ValueError(f"Expected a single decoded video (N=1), got {tuple(x.shape)}")
    t = int(x.shape[1])
    needed = clip_len * frame_interval
    if t <= 0:
        raise ValueError("Video has zero frames")

    if temporal_strategy == "dense":
        sample_position = max(1, 1 + t - int(sample_range))
        interval = int(sample_range) // int(num_clips)
        start_list = torch.linspace(
            0,
            sample_position - 1,
            steps=max(1, int(num_sample_positions)),
            dtype=torch.long,
        )
        base_offsets = torch.arange(int(num_clips), dtype=torch.long) * interval
        starts = [(base_offsets + start).remainder(t) for start in start_list]
        starts = torch.cat(starts, dim=0).tolist()
    else:
        max_start = max(0, t - needed)
        if num_clips == 1:
            starts = [max_start // 2]
        else:
            starts = [round(i * max_start / (num_clips - 1)) for i in range(num_clips)]

    clips = []
    all_inds = []
    for start in starts:
        inds = torch.arange(start, start + needed, frame_interval, dtype=torch.long)
        inds = torch.clamp(inds, 0, t - 1)
        all_inds.append(inds)
        clips.append(x[0, inds])
    clips_tensor = torch.stack(clips, dim=0).contiguous()  # NTHWC, N=num_clips
    if not return_indices:
        return clips_tensor
    inds_tensor = torch.stack(all_inds, dim=0).contiguous()
    return clips_tensor, inds_tensor


def _apply_spatial_frames_ntchw(x_ntchw: torch.Tensor, fn) -> torch.Tensor:
    n, t, c, h, w = x_ntchw.shape
    y = x_ntchw.reshape(n * t, c, h, w)
    y = fn(y)
    h2, w2 = y.shape[-2:]
    return y.reshape(n, t, c, h2, w2)


def resize_short_side(x: torch.Tensor, size: int, input_format: str = "NTHWC") -> torch.Tensor:
    x_nthwc = ensure_nthwc(x, input_format)
    x_ntchw = to_format(x_nthwc, "NTHWC", "NTCHW")

    def fn(y: torch.Tensor) -> torch.Tensor:
        h, w = y.shape[-2:]
        if min(h, w) == size:
            return y
        scale = float(size) / float(min(h, w))
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        return F.interpolate(y, size=(new_h, new_w), mode="bilinear", align_corners=False)

    out = _apply_spatial_frames_ntchw(x_ntchw, fn)
    return to_format(out, "NTCHW", "NTHWC")


def center_crop(x: torch.Tensor, size: int | tuple[int, int], input_format: str = "NTHWC") -> torch.Tensor:
    x_nthwc = ensure_nthwc(x, input_format)
    x_ntchw = to_format(x_nthwc, "NTHWC", "NTCHW")
    crop_h, crop_w = (size, size) if isinstance(size, int) else size

    def fn(y: torch.Tensor) -> torch.Tensor:
        h, w = y.shape[-2:]
        if h < crop_h or w < crop_w:
            y = F.interpolate(y, size=(max(crop_h, h), max(crop_w, w)), mode="bilinear", align_corners=False)
            h, w = y.shape[-2:]
        top = max(0, (h - crop_h) // 2)
        left = max(0, (w - crop_w) // 2)
        return y[..., top : top + crop_h, left : left + crop_w]

    out = _apply_spatial_frames_ntchw(x_ntchw, fn)
    return to_format(out, "NTCHW", "NTHWC")


def three_crop(x: torch.Tensor, size: int | tuple[int, int], input_format: str = "NTHWC") -> torch.Tensor:
    x_nthwc = ensure_nthwc(x, input_format)
    x_ntchw = to_format(x_nthwc, "NTHWC", "NTCHW")
    crop_h, crop_w = (size, size) if isinstance(size, int) else size
    n, t, c, h, w = x_ntchw.shape
    y = x_ntchw.reshape(n * t, c, h, w)

    if h < crop_h or w < crop_w:
        scale = max(float(crop_h) / h, float(crop_w) / w)
        y = F.interpolate(y, size=(int(round(h * scale)), int(round(w * scale))), mode="bilinear", align_corners=False)
        h, w = y.shape[-2:]

    crops = []
    if w >= h:
        top = max(0, (h - crop_h) // 2)
        xs = [0, max(0, (w - crop_w) // 2), max(0, w - crop_w)]
        for left in xs:
            crops.append(y[..., top : top + crop_h, left : left + crop_w])
    else:
        left = max(0, (w - crop_w) // 2)
        ys = [0, max(0, (h - crop_h) // 2), max(0, h - crop_h)]
        for top in ys:
            crops.append(y[..., top : top + crop_h, left : left + crop_w])

    stacked = torch.stack(crops, dim=1)  # [N*T, 3, C, H, W]
    stacked = stacked.reshape(n, t, 3, c, crop_h, crop_w).permute(0, 2, 1, 3, 4, 5).contiguous()
    stacked = stacked.reshape(n * 3, t, c, crop_h, crop_w)
    return to_format(stacked, "NTCHW", "NTHWC")


def ten_crop(x: torch.Tensor, size: int | tuple[int, int], input_format: str = "NTHWC") -> torch.Tensor:
    x_nthwc = ensure_nthwc(x, input_format)
    x_ntchw = to_format(x_nthwc, "NTHWC", "NTCHW")
    crop_h, crop_w = (size, size) if isinstance(size, int) else size
    n, t, c, h, w = x_ntchw.shape
    y = x_ntchw.reshape(n * t, c, h, w)

    if h < crop_h or w < crop_w:
        scale = max(float(crop_h) / h, float(crop_w) / w)
        y = F.interpolate(y, size=(int(round(h * scale)), int(round(w * scale))), mode="bilinear", align_corners=False)
        h, w = y.shape[-2:]

    top = 0
    left = 0
    bottom = max(0, h - crop_h)
    right = max(0, w - crop_w)
    center_top = max(0, (h - crop_h) // 2)
    center_left = max(0, (w - crop_w) // 2)
    coords = [
        (top, left),
        (top, right),
        (bottom, left),
        (bottom, right),
        (center_top, center_left),
    ]

    crops = [y[..., t0 : t0 + crop_h, l0 : l0 + crop_w] for t0, l0 in coords]
    flips = [torch.flip(crop, dims=(-1,)) for crop in crops]
    stacked = torch.stack(crops + flips, dim=1)
    stacked = stacked.reshape(n, t, 10, c, crop_h, crop_w).permute(0, 2, 1, 3, 4, 5).contiguous()
    stacked = stacked.reshape(n * 10, t, c, crop_h, crop_w)
    return to_format(stacked, "NTCHW", "NTHWC")


def normalize_channels(
    x: torch.Tensor,
    mean: list[float] | tuple[float, float, float],
    std: list[float] | tuple[float, float, float],
    input_format: str = "NTHWC",
) -> torch.Tensor:
    x_nthwc = ensure_nthwc(x, input_format).float()
    mean_t = torch.tensor(mean, dtype=x_nthwc.dtype, device=x_nthwc.device).view(1, 1, 1, 1, 3)
    std_t = torch.tensor(std, dtype=x_nthwc.dtype, device=x_nthwc.device).view(1, 1, 1, 1, 3)
    return (x_nthwc - mean_t) / std_t

from __future__ import annotations

from pathlib import Path
from fractions import Fraction

import torch


def flatten_video_to_thwc(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return x
    if x.ndim == 5:
        # [Nclips, T, H, W, C] -> [Nclips*T, H, W, C]
        return x.reshape(-1, *x.shape[-3:])
    raise ValueError(f"Unsupported video tensor shape for saving/metrics: {tuple(x.shape)}")


def to_uint8_thwc(x: torch.Tensor) -> torch.Tensor:
    x = flatten_video_to_thwc(x).detach().cpu()
    if x.dtype == torch.uint8:
        return x
    x = x.float()
    # Heuristic: attacks often operate in pixel space [0,255], but handle [0,1] too.
    if x.numel() and float(x.max().item()) <= 1.5:
        x = x * 255.0
    return x.clamp(0, 255).round().to(torch.uint8)


def save_video_sampled(x: torch.Tensor, out_path: Path, fps: int = 16) -> None:
    import av

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = to_uint8_thwc(x).numpy()
    container = av.open(str(out_path), mode="w")
    rate = Fraction(str(float(fps))).limit_denominator(1000)
    if rate <= 0:
        rate = Fraction(16, 1)
    stream = container.add_stream("libx264", rate=rate)
    stream.width = int(frames.shape[2])
    stream.height = int(frames.shape[1])
    stream.pix_fmt = "yuv420p"
    for frame in frames:
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in stream.encode(av_frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def compose_full_video_with_replaced_frames(
    sampled_or_attacked: torch.Tensor,
    *,
    full_tensor: torch.Tensor | None,
    sampled_frame_ids: torch.Tensor | None,
    mark_replaced: bool = True,
) -> torch.Tensor:
    if full_tensor is None or sampled_frame_ids is None:
        return to_uint8_thwc(sampled_or_attacked)
    full = to_uint8_thwc(full_tensor).clone()
    repl = to_uint8_thwc(sampled_or_attacked)
    frame_ids = sampled_frame_ids.detach().cpu().reshape(-1).to(torch.long)
    if frame_ids.numel() == 0 or full.shape[0] == 0:
        return full
    frame_ids = frame_ids.clamp(min=0, max=max(int(full.shape[0]) - 1, 0))
    replace_count = min(int(frame_ids.numel()), int(repl.shape[0]))
    for i in range(replace_count):
        idx = int(frame_ids[i].item())
        frame = repl[i].clone()
        if mark_replaced:
            h, w = int(frame.shape[0]), int(frame.shape[1])
            size = max(8, min(h, w) // 20)
            y0, x0 = 4, max(0, w - size - 4)
            frame[y0 : y0 + size, x0 : x0 + size] = torch.tensor([255, 0, 0], dtype=frame.dtype)
        full[idx] = frame
    return full


def save_video_lossless_like_original(
    sampled_or_attacked: torch.Tensor,
    out_path: Path,
    *,
    full_tensor: torch.Tensor | None,
    sampled_frame_ids: torch.Tensor | None,
    fps: float = 16.0,
) -> None:
    frames = compose_full_video_with_replaced_frames(
        sampled_or_attacked,
        full_tensor=full_tensor,
        sampled_frame_ids=sampled_frame_ids,
        mark_replaced=True,
    )
    # Keep sampled-only dump behavior for short 16fps clips.
    # For reconstructed full videos, preserve original duration without padding.
    is_full_reconstruction = full_tensor is not None and sampled_frame_ids is not None
    if (not is_full_reconstruction) and int(fps) == 16 and frames.ndim == 4 and 0 < int(frames.shape[0]) < 160:
        t = int(frames.shape[0])
        reps = (160 + t - 1) // t
        frames = frames.repeat((reps, 1, 1, 1))[:160]
    save_video_sampled(frames, out_path, fps=fps)


def probe_video_fps(path: str, default: float = 16.0) -> float:
    import av

    try:
        container = av.open(str(path))
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else float(default)
        container.close()
        return fps if fps > 0 else float(default)
    except Exception:
        return float(default)


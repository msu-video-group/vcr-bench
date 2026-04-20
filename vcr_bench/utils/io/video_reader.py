from __future__ import annotations

from pathlib import Path

import torch


def decode_video_nthwc(path: str, return_info: bool = False):
    import av

    container = av.open(path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 0.0
    frames = []
    for frame in container.decode(stream):
        arr = frame.to_rgb().to_ndarray()
        frames.append(torch.frombuffer(arr.tobytes(), dtype=torch.uint8).reshape(arr.shape))
    container.close()

    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")

    tensor = torch.stack(frames, dim=0)
    tensor = tensor.unsqueeze(0)  # NTHWC, N=1 for a single video
    if return_info:
        return tensor, {
            "fps": fps,
            "total_frames": int(tensor.shape[1]),
            "path": str(Path(path)),
        }
    return tensor

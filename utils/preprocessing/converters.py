from __future__ import annotations

import torch


_SUPPORTED = {"THWC", "TCHW", "NTHWC", "NTCHW", "NCTHW"}


def _permute_by_labels(x: torch.Tensor, src: str, dst: str) -> torch.Tensor:
    src_idx = {ch: i for i, ch in enumerate(src)}
    return x.permute(*(src_idx[ch] for ch in dst))


def to_format(x: torch.Tensor, src_format: str, dst_format: str) -> torch.Tensor:
    src_format = src_format.upper()
    dst_format = dst_format.upper()
    if src_format not in _SUPPORTED or dst_format not in _SUPPORTED:
        raise ValueError(f"Unsupported format conversion: {src_format} -> {dst_format}")
    if src_format == dst_format:
        return x

    if x.ndim != len(src_format):
        raise ValueError(f"Tensor ndim {x.ndim} does not match src format {src_format}")

    if src_format in {"THWC", "TCHW"} and dst_format.startswith("N"):
        x = x.unsqueeze(0)
        src_format = f"N{src_format}"
    elif src_format.startswith("N") and dst_format in {"THWC", "TCHW"}:
        if x.shape[0] != 1:
            raise ValueError(f"Cannot drop batch dim for N != 1: shape={tuple(x.shape)}")
        x = x.squeeze(0)
        src_format = src_format[1:]

    if len(src_format) != len(dst_format):
        raise ValueError(f"Rank mismatch after normalization: {src_format} -> {dst_format}")
    if set(src_format) != set(dst_format):
        raise ValueError(f"Incompatible axes: {src_format} -> {dst_format}")
    return _permute_by_labels(x, src_format, dst_format).contiguous()


def ensure_nthwc(x: torch.Tensor, input_format: str = "NTHWC") -> torch.Tensor:
    return to_format(x, input_format, "NTHWC")

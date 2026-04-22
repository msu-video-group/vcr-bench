from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from vcr_bench.defences.base import BaseVideoDefence

_VIDEOPURE_DIR = str(Path(__file__).resolve().parents[3] / "Defences" / "VideoPure")

_denoiser = None
_denoiser_lock = threading.Lock()


def _vp_log(msg: str) -> None:
    print(msg, flush=True)


def _start_heartbeat(label: str, interval_sec: int = 30):
    stop_event = threading.Event()
    t0 = time.time()

    def _run():
        while not stop_event.wait(interval_sec):
            elapsed = time.time() - t0
            if torch.cuda.is_available():
                try:
                    used_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                    reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
                    _vp_log(
                        f"[VideoPure] {label} still running... "
                        f"{elapsed:.1f}s elapsed | cuda alloc={used_gb:.2f}GB reserved={reserved_gb:.2f}GB"
                    )
                except Exception:
                    _vp_log(f"[VideoPure] {label} still running... {elapsed:.1f}s elapsed")
            else:
                _vp_log(f"[VideoPure] {label} still running... {elapsed:.1f}s elapsed")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    def _stop():
        stop_event.set()
        thread.join(timeout=0.2)

    return _stop


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_denoiser():
    global _denoiser
    if _denoiser is not None:
        return _denoiser
    with _denoiser_lock:
        if _denoiser is not None:
            return _denoiser
        _vp_log("Initializing VideoPure denoiser...")
        _denoiser = _init_denoiser()
        _vp_log("Denoiser initialized.")
    return _denoiser


def _init_denoiser():
    if _VIDEOPURE_DIR not in sys.path:
        sys.path.insert(0, _VIDEOPURE_DIR)

    from main import Denoiser  # type: ignore[import]
    from pipe_defense import Video_Diffpure  # type: ignore[import]
    from core.raft_arch import RAFT_SR  # type: ignore[import]
    from flow_net import Flow_models  # type: ignore[import]
    from predownload_diffuser import DEFAULT_MODEL_ID, DEFAULT_CACHE_DIR, ensure_model_downloaded  # type: ignore[import]

    model_id = os.getenv("VIDEOPURE_MODEL_ID", DEFAULT_MODEL_ID)
    cache_dir = os.getenv("VIDEOPURE_CACHE_DIR", DEFAULT_CACHE_DIR)
    local_only = _env_flag("VIDEOPURE_LOCAL_FILES_ONLY", default=True)
    auto_download = _env_flag("VIDEOPURE_AUTO_DOWNLOAD", default=True)

    _vp_log(f"[VideoPure] Loading diffusion model: {model_id}")
    if cache_dir:
        _vp_log(f"[VideoPure] Cache dir: {cache_dir}")
    _vp_log(f"[VideoPure] Local files only: {local_only}")
    _vp_log(f"[VideoPure] Auto download on cache miss: {auto_download}")

    start = time.time()
    if auto_download:
        ensure_model_downloaded(model_id=model_id, cache_dir=cache_dir, quiet=False)

    _vp_log("[VideoPure] Building diffusion pipeline (from_pretrained)...")
    t_pre = time.time()
    try:
        pipe = Video_Diffpure.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=cache_dir,
            local_files_only=local_only,
        )
    except Exception as exc:
        raise RuntimeError(
            "VideoPure model init failed. If cache is missing, run: "
            "python3 Defences/VideoPure/predownload_diffuser.py "
            "or set VIDEOPURE_AUTO_DOWNLOAD=1."
        ) from exc
    _vp_log(f"[VideoPure] Pipeline constructed in {time.time() - t_pre:.1f}s")

    _vp_log("[VideoPure] Moving pipeline to CUDA...")
    t_cuda = time.time()
    pipe.to("cuda")
    _vp_log(f"[VideoPure] Pipeline moved to CUDA in {time.time() - t_cuda:.1f}s")

    _vp_log("[VideoPure] Initializing RAFT flow model...")
    t_raft = time.time()
    flow_model = Flow_models(RAFT_SR())
    _vp_log(f"[VideoPure] RAFT flow model ready in {time.time() - t_raft:.1f}s")
    _vp_log(f"[VideoPure] Diffusion model ready in {time.time() - start:.1f}s total init time")

    denoiser = Denoiser(
        model=pipe,
        classifier=None,
        t=5,
        flow_models=flow_model,
        denoise_type="videopure",
        classifier_name="",
    )
    denoiser.eval()
    return denoiser


def _normalize_denoiser_output(processed_video: torch.Tensor) -> torch.Tensor:
    _vp_log(f"[VideoPure] Raw denoiser output shape: {tuple(processed_video.shape)}")
    guard = 0
    while processed_video.dim() > 4 and guard < 8:
        if processed_video.shape[0] == 1:
            processed_video = processed_video.squeeze(0)
            _vp_log(f"[VideoPure] Squeezed dim0 -> {tuple(processed_video.shape)}")
        else:
            processed_video = processed_video[-1]
            _vp_log(f"[VideoPure] Selected last sample on dim0 -> {tuple(processed_video.shape)}")
        guard += 1
    if processed_video.dim() != 4:
        raise RuntimeError(
            f"Unexpected denoiser output rank after normalization: "
            f"shape={tuple(processed_video.shape)}, dim={processed_video.dim()}"
        )
    return processed_video


def _ensure_0_255_range(video_thwc: torch.Tensor) -> torch.Tensor:
    x = video_thwc.detach()
    x_min = float(torch.amin(x))
    x_max = float(torch.amax(x))
    _vp_log(f"[VideoPure] Output value range before fix: min={x_min:.4f}, max={x_max:.4f}")
    if x_max <= 1.5:
        x = x * 255.0
        _vp_log("[VideoPure] Scaled denoiser output from [0,1] to [0,255].")
    x = torch.clamp(x, 0.0, 255.0).float()
    x2_min = float(torch.amin(x))
    x2_max = float(torch.amax(x))
    _vp_log(f"[VideoPure] Output value range after fix: min={x2_min:.4f}, max={x2_max:.4f}")
    return x


def _match_frame_count(video_thwc: torch.Tensor, target_t: int) -> torch.Tensor:
    t = video_thwc.shape[0]
    if t == target_t:
        return video_thwc
    if t <= 0:
        raise RuntimeError(f"Invalid denoiser output frame count: {t}")
    idx = torch.linspace(0, t - 1, steps=target_t, device=video_thwc.device).round().long()
    out = video_thwc.index_select(0, idx)
    _vp_log(f"[VideoPure] Resampled frame count from {t} -> {target_t}")
    return out


def _match_spatial_size(video_thwc: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    h, w = int(video_thwc.shape[1]), int(video_thwc.shape[2])
    if h == target_h and w == target_w:
        return video_thwc
    x = video_thwc.permute(0, 3, 1, 2)  # [T,H,W,C] -> [T,C,H,W]
    x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
    x = x.permute(0, 2, 3, 1)  # [T,C,H,W] -> [T,H,W,C]
    _vp_log(f"[VideoPure] Resized spatial size from ({h},{w}) -> ({target_h},{target_w})")
    return x


def _log_defence_delta(original_thwc: torch.Tensor, defended_thwc: torch.Tensor) -> None:
    if original_thwc.shape != defended_thwc.shape:
        _vp_log(
            f"[VideoPure] Delta stats skipped due to shape mismatch: "
            f"orig={tuple(original_thwc.shape)} defended={tuple(defended_thwc.shape)}"
        )
        return
    delta = (defended_thwc - original_thwc).abs()
    mae = float(delta.mean())
    max_abs = float(delta.max())
    _vp_log(f"[VideoPure] Defence delta stats: MAE={mae:.4f}, max_abs={max_abs:.4f}")


class DenoiseDefence(BaseVideoDefence):
    """VideoPure diffusion-based denoiser defence.

    Applies the same denoising pipeline as the original benchmark's
    Defences/VideoPure/Denoiser.py wrapper, operating on NTHWC tensors.
    """

    voting_count = 1

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Denoise each video in the batch independently.

        Args:
            x: [N, T, H, W, C] float32, values in [0, 255].

        Returns:
            Denoised tensor of the same shape.
        """
        denoiser = _get_denoiser()
        N, T, H, W, C = x.shape
        input_device = x.device
        defence_device = torch.device("cuda" if torch.cuda.is_available() else input_device)

        # Flatten [N, T, H, W, C] -> [N*T, H, W, C], matching old benchmark where all
        # clips are passed as one flat [T, H, W, C] sequence to the denoiser.
        flat_thwc = x.reshape(N * T, H, W, C).contiguous().to(defence_device).float()
        _vp_log(
            f"[VideoPure] Flattened sampled clips into one sequence: "
            f"clips={N} clip_len={T} total_frames={N * T}"
        )

        # [T, H, W, C] -> [C, T, H, W]  (same permute as old Denoiser.py wrapper)
        original_video = flat_thwc.permute(3, 0, 1, 2)
        input_t = int(original_video.shape[1])
        input_h = int(original_video.shape[2])
        input_w = int(original_video.shape[3])
        _vp_log(f"[VideoPure] Input video tensor shape after permute: {tuple(original_video.shape)}")

        stop_heartbeat = _start_heartbeat("denoise pass", interval_sec=30)
        t_denoise = time.time()
        with torch.no_grad():
            processed_video = denoiser.denoise_method((original_video / 255.0).unsqueeze(0))
        stop_heartbeat()
        _vp_log(f"[VideoPure] Denoise pass finished in {time.time() - t_denoise:.1f}s")

        processed_video = _normalize_denoiser_output(processed_video)
        if processed_video.dim() == 4:
            processed_video = processed_video.permute(1, 2, 3, 0)  # [C, T, H, W] -> [T, H, W, C]
        processed_video = _ensure_0_255_range(processed_video)
        processed_video = _match_frame_count(processed_video, input_t)
        processed_video = _match_spatial_size(processed_video, input_h, input_w)

        original_thwc = original_video.permute(1, 2, 3, 0).float()
        _log_defence_delta(original_thwc, processed_video)
        _vp_log(f"[VideoPure] Output video tensor shape: {tuple(processed_video.shape)}")
        torch.cuda.empty_cache()

        return processed_video.reshape(N, T, H, W, C).to(input_device).contiguous()


DEFENCE_CLASS = DenoiseDefence


def create(**kwargs) -> DenoiseDefence:
    return DenoiseDefence(**kwargs)

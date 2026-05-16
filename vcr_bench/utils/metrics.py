from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
try:
    from pytorch_msssim import SSIM as diff_ssim
except Exception:
    diff_ssim = None
import torch
import contextlib
import threading
try:
    from IQA_pytorch import LPIPSvgg
    from IQA_pytorch import DISTS
except Exception:
    LPIPSvgg = None
    DISTS = None
import gc
import warnings
import numpy as np
import cv2
import tempfile
import os
import av
import csv
import io
import json
import subprocess
import time
import shutil
import torch.nn.functional as F

def to_torch(x, device='cpu'):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if len(x.shape) == 3:
            x = x.permute(2, 0, 1).unsqueeze(0)
        else:
            x = x.permute(0, 3, 1, 2)
        x = x.type(torch.FloatTensor).to(device)
    return x

def to_numpy(x):
    if torch.is_tensor(x):
        x = x.cpu().detach().numpy()
    if len(x.shape) == 5:
        # Standard benchmark sampled format: [Nclips, T, H, W, C] -> [Nclips*T, H, W, C]
        return x.reshape(-1, *x.shape[-3:])
    return x if len(x.shape) == 4 else x[np.newaxis]


def to_uint8_video(x):
    x = to_numpy(x)
    if x.dtype == np.uint8:
        return np.ascontiguousarray(x)
    x = x.astype(np.float32, copy=False)
    if x.size and float(np.nanmax(x)) <= 1.5:
        x = x * 255.0
    return np.ascontiguousarray(np.clip(np.rint(x), 0, 255).astype(np.uint8))

def PSNR(x, y):
    return peak_signal_noise_ratio(to_numpy(x), to_numpy(y), data_range=255)


def SSIM(x, y):
    frames_x = to_numpy(x)
    frames_y = to_numpy(y)
    ssim_sum = 0
    for img1, img2 in zip(frames_x, frames_y):
        ssim_sum += structural_similarity(img1, img2, channel_axis=2, data_range=255)
    return ssim_sum / len(frames_x)


def MSE(x, y):
    return mean_squared_error(to_numpy(x), to_numpy(y))

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

_lpips_thread_local = threading.local()
_lpips_metric_cuda = None
_dists_metric_cuda = None
_ssim_metric_cuda = None
_dists_metric_cpu = None
_ssim_metric_cpu = None


def _normalize_device(device):
    d = torch.device(device)
    if d.type == 'cuda' and d.index is None:
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return d


def _module_device(module, fallback):
    try:
        p = next(module.parameters())
        return p.device
    except StopIteration:
        pass
    try:
        b = next(module.buffers())
        return b.device
    except StopIteration:
        return torch.device(fallback)


def _make_lpips_model(device):
    """Create LPIPSvgg on the target device.

    IQA_pytorch stores self.weights as a plain Python list (not a registered
    buffer), so calling .to(device) on the module does NOT move those tensors.
    If LPIPSvgg.pt was saved on CUDA they remain on CUDA even after .to('cpu'),
    causing a device-mismatch RuntimeError on every CPU forward pass and
    silently falling back to MSE.  We fix this by explicitly moving each
    weight tensor to the target device after construction.
    """
    model = LPIPSvgg().to(device)
    model.weights = [(n, w.to(device)) for n, w in model.weights]
    model.eval()
    return model


def _get_lpips_model(device):
    global _lpips_metric_cuda
    if LPIPSvgg is None:
        raise ImportError("IQA_pytorch is required for LPIPS")
    device = _normalize_device(device)
    if device.type == 'cpu':
        model = getattr(_lpips_thread_local, "model", None)
        if model is None or next(model.parameters()).device.type != 'cpu':
            model = _make_lpips_model(device)
            _lpips_thread_local.model = model
        return model
    if _lpips_metric_cuda is None or next(_lpips_metric_cuda.parameters()).device != device:
        _lpips_metric_cuda = _make_lpips_model(device)
    return _lpips_metric_cuda


def _get_dists_model(device):
    global _dists_metric_cuda, _dists_metric_cpu
    if DISTS is None:
        raise ImportError("IQA_pytorch is required for DISTS")
    device = _normalize_device(device)
    if device.type == 'cpu':
        if _dists_metric_cpu is None:
            _dists_metric_cpu = DISTS().to(device)
            _dists_metric_cpu.eval()
        return _dists_metric_cpu
    if _dists_metric_cuda is None or _module_device(_dists_metric_cuda, device) != device:
        _dists_metric_cuda = DISTS().to(device)
        _dists_metric_cuda.eval()
    return _dists_metric_cuda


def _get_ssim_model(device):
    global _ssim_metric_cuda, _ssim_metric_cpu
    if diff_ssim is None:
        raise ImportError("pytorch_msssim is required for video_ssim")
    device = _normalize_device(device)
    if device.type == 'cpu':
        if _ssim_metric_cpu is None:
            _ssim_metric_cpu = diff_ssim().to(device)
            _ssim_metric_cpu.eval()
        return _ssim_metric_cpu
    if _ssim_metric_cuda is None or _module_device(_ssim_metric_cuda, device) != device:
        _ssim_metric_cuda = diff_ssim().to(device)
        _ssim_metric_cuda.eval()
    return _ssim_metric_cuda

def video_metric(vid1, vid2, metric, batch_size=4):
    """
    Вычисляет Метрику по видео с оптимизацией памяти
    """
    # Отключаем градиенты для vid2
    vid2 = vid2.detach().requires_grad_(False)

    # Переставляем размерности: [B, H, W, C] -> [B, C, H, W]
    vid1 = vid1.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
    vid2 = vid2.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

    total_loss = None
    len_ = vid1.shape[0]

    metric_device = vid1.device
    if isinstance(metric, torch.nn.Module):
        metric.eval()
        # Force metric to the same device as inputs to avoid mixed-device errors.
        metric = metric.to(vid1.device)
        try:
            metric_device = next(metric.parameters()).device
        except StopIteration:
            metric_device = vid1.device

    for i in range(0, len_, batch_size):
        end_idx = min(i + batch_size, len_)

        batch1 = vid1[i:end_idx].to(metric_device)
        batch2 = vid2[i:end_idx].to(metric_device)

        autocast_ctx = torch.amp.autocast(metric_device.type) if metric_device.type == 'cuda' else contextlib.nullcontext()
        with autocast_ctx:
            if (LPIPSvgg is not None and isinstance(metric, LPIPSvgg)) or (DISTS is not None and isinstance(metric, DISTS)):
                # LPIPSvgg/DISTS expect inputs in [0, 1]; our tensors are in [0, 255].
                batch_loss = metric(batch1 / 255.0, batch2 / 255.0, as_loss=True)
            else:
                batch_loss = metric(batch1, batch2)

        # Keep this path differentiable w.r.t. vid1 for attack optimization.
        weighted_loss = batch_loss * float(end_idx - i)
        total_loss = weighted_loss if total_loss is None else (total_loss + weighted_loss)

        del batch1, batch2, batch_loss, weighted_loss

    if total_loss is None:
        return torch.zeros((), device=vid1.device, requires_grad=True)
    return total_loss / float(len_)


def video_ssim(vid1, vid2):
    metric = _get_ssim_model(vid1.device)
    return video_metric(vid1, vid2, metric)


def video_lpips(vid1, vid2, device='cuda', batch_size=4):
    device = _normalize_device(device)
    lpips_metric_local = _get_lpips_model(device)
    try:
        metric_device = next(lpips_metric_local.parameters()).device
    except StopIteration:
        metric_device = device
    if metric_device != device:
        lpips_metric_local = lpips_metric_local.to(device)
    return video_metric(vid1.to(device), vid2.to(device), lpips_metric_local, batch_size=batch_size)


def video_dists(vid1, vid2):
    metric = _get_dists_model(vid1.device)
    return video_metric(vid1, vid2, metric)


def _downscale_video_if_needed(vid, max_size):
    if max_size is None:
        return vid
    # vid: [T, H, W, C]
    h, w = vid.shape[1], vid.shape[2]
    if h <= max_size and w <= max_size:
        return vid
    scale = max_size / float(max(h, w))
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    vid_bchw = vid.permute(0, 3, 1, 2).contiguous()
    vid_bchw = F.interpolate(vid_bchw, size=(new_h, new_w), mode='bilinear', align_corners=False)
    return vid_bchw.permute(0, 2, 3, 1).contiguous()


def LPIPS(vid1, vid2, device='cpu', max_size=224, batch_size=1):
    with torch.no_grad():
        device = torch.device('cpu')
        v1 = vid1.detach().to(device)
        v2 = vid2.detach().to(device)
        if v1.ndim == 5:
            v1 = v1.reshape(-1, *v1.shape[-3:])
        if v2.ndim == 5:
            v2 = v2.reshape(-1, *v2.shape[-3:])
        v1 = _downscale_video_if_needed(v1, max_size)
        v2 = _downscale_video_if_needed(v2, max_size)
        return video_lpips(v1, v2, device=device, batch_size=batch_size).item()
        


def save_video(tensor, path, fps=30):
    t, h, w, c = tensor.shape
    container = av.open(path, mode='w')
    stream = container.add_stream('libx264', rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = 'yuv444p'
    stream.options = {'crf': '0', 'preset': 'veryslow'}
    for frame in tensor:
        frame_rgb = frame if frame.dtype == np.uint8 else (frame * 255).astype(np.uint8)
        av_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
        for packet in stream.encode(av_frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _parse_ffmpeg_vmaf_log(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pooled = data.get("pooled_metrics", {})
    vmaf_metric = pooled.get("vmaf", {})
    if isinstance(vmaf_metric, dict):
        mean = vmaf_metric.get("mean")
        if mean is not None:
            return float(mean)

    scores = []
    for frame in data.get("frames", []):
        metrics = frame.get("metrics", {})
        if "vmaf" in metrics:
            scores.append(float(metrics["vmaf"]))
    return float(np.mean(scores)) if scores else 0.0


_FFMPEG_LIBVMAF_AVAILABLE = None


def _subprocess_path(executable, path):
    path = str(path)
    if os.name != "nt" and str(executable).lower().endswith(".exe"):
        try:
            result = subprocess.run(
                ["wslpath", "-w", path],
                shell=False,
                capture_output=True,
                timeout=5,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
    return path


def _ffmpeg_filter_path(executable, path):
    path = _subprocess_path(executable, path)
    path = path.replace("\\", "/")
    if len(path) >= 2 and path[1] == ":":
        path = f"{path[0]}\\\\:{path[2:]}"
    return path


def _ffmpeg_supports_libvmaf(ffmpeg):
    global _FFMPEG_LIBVMAF_AVAILABLE
    if _FFMPEG_LIBVMAF_AVAILABLE is not None:
        return _FFMPEG_LIBVMAF_AVAILABLE
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-filters"],
            shell=False,
            capture_output=True,
            timeout=10,
            text=True,
        )
        _FFMPEG_LIBVMAF_AVAILABLE = result.returncode == 0 and " libvmaf " in result.stdout
    except Exception:
        _FFMPEG_LIBVMAF_AVAILABLE = False
    return _FFMPEG_LIBVMAF_AVAILABLE


def _run_ffmpeg_vmaf(ffmpeg, ref_np, dist_np, log_path, timeout_sec):
    ref_np = np.ascontiguousarray(ref_np)
    dist_np = np.ascontiguousarray(dist_np)
    if ref_np.shape != dist_np.shape:
        raise ValueError(f"VMAF inputs must have the same shape, got {ref_np.shape} and {dist_np.shape}")
    if ref_np.ndim != 4 or ref_np.shape[-1] != 3:
        raise ValueError(f"VMAF inputs must be THWC RGB tensors, got {ref_np.shape}")

    _, h, w, _ = ref_np.shape
    ref_raw = tempfile.NamedTemporaryFile(suffix='.rgb', delete=False)
    dist_raw = tempfile.NamedTemporaryFile(suffix='.rgb', delete=False)
    try:
        ref_raw.write(ref_np.tobytes())
        dist_raw.write(dist_np.tobytes())
        ref_raw.close()
        dist_raw.close()

        raw_input = [
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            "30",
        ]
        filter_graph = (
            "[0:v]format=yuv444p[dist];"
            "[1:v]format=yuv444p[ref];"
            f"[dist][ref]libvmaf=log_path={_ffmpeg_filter_path(ffmpeg, log_path)}:log_fmt=json"
        )
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            *raw_input,
            "-i",
            _subprocess_path(ffmpeg, dist_raw.name),
            *raw_input,
            "-i",
            _subprocess_path(ffmpeg, ref_raw.name),
            "-filter_complex",
            filter_graph,
            "-f",
            "null",
            "-",
        ]
        result = subprocess.run(cmd, shell=False, capture_output=True, timeout=timeout_sec, text=True)
        if result.returncode != 0 and ("Option 'log_path' not found" in result.stderr or "Error applying option" in result.stderr):
            filter_graph = (
                "[0:v]format=yuv444p[dist];"
                "[1:v]format=yuv444p[ref];"
                f"[dist][ref]libvmaf=log_fmt=json:log_path={_ffmpeg_filter_path(ffmpeg, log_path)}"
            )
            cmd[cmd.index("-filter_complex") + 1] = filter_graph
            result = subprocess.run(cmd, shell=False, capture_output=True, timeout=timeout_sec, text=True)
    finally:
        for raw_file in (ref_raw, dist_raw):
            try:
                os.unlink(raw_file.name)
            except Exception:
                pass

    if result.returncode != 0:
        print(f"VMAF calculation via ffmpeg failed with return code {result.returncode}")
        if "No such filter: 'libvmaf'" in result.stderr:
            print("VMAF calculation error: ffmpeg was built without libvmaf support")
        return None
    return _parse_ffmpeg_vmaf_log(log_path)


def _run_vqmt_vmaf(vqmt, ref_video_path, dist_video_path, csv_path, timeout_sec):
    cmd = [vqmt, "-metr", "vmaf", "-orig", ref_video_path, "-in", dist_video_path, "-csv-file", csv_path]
    result = subprocess.run(cmd, shell=False, capture_output=True, timeout=timeout_sec, text=True)
    if result.returncode != 0:
        print(f"VMAF calculation via vqmt failed with return code {result.returncode}")
        return None
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        scores = []
        for row in reader:
            try:
                if 'vmaf' in row:
                    scores.append(float(row['vmaf']))
                if 'Netflix VMAF_VMAF061_float' in row:
                    scores.append(float(row['Netflix VMAF_VMAF061_float']))
            except (ValueError, TypeError):
                pass
    return float(np.mean(scores)) if scores else 0.0


def VMAF(x, y):
    start_time = time.time()
    timeout_sec = int(os.getenv("VMAF_TIMEOUT_SEC", os.getenv("VQMT_TIMEOUT_SEC", "120")))
    backend = os.getenv("VMAF_BACKEND", "auto").strip().lower()
    if backend not in {"auto", "ffmpeg", "vqmt"}:
        print(f"Unknown VMAF_BACKEND={backend!r}; expected auto, ffmpeg, or vqmt")
        return 0.0
    ref_np = to_uint8_video(x)
    dist_np = to_uint8_video(y)
    ffmpeg = shutil.which(os.getenv("FFMPEG_BIN", "ffmpeg"))
    vqmt = shutil.which(os.getenv("VQMT_BIN", "vqmt"))
    ref_video = None
    dist_video = None
    log_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
    csv_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
    log_file.close()
    csv_file.close()
    try:
        attempts = []
        if backend in {"auto", "ffmpeg"}:
            if ffmpeg is not None and _ffmpeg_supports_libvmaf(ffmpeg):
                attempts.append(("ffmpeg", lambda: _run_ffmpeg_vmaf(ffmpeg, ref_np, dist_np, log_file.name, timeout_sec)))
            elif backend == "ffmpeg":
                print("VMAF calculation error: ffmpeg/libvmaf is not available")
        if backend in {"auto", "vqmt"}:
            if vqmt is not None:
                ref_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
                dist_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
                save_video(ref_np, ref_video.name)
                save_video(dist_np, dist_video.name)
                attempts.append(("vqmt", lambda: _run_vqmt_vmaf(vqmt, ref_video.name, dist_video.name, csv_file.name, timeout_sec)))
            elif backend == "vqmt":
                print("VMAF calculation error: vqmt executable not found")
        if not attempts:
            print("VMAF calculation error: neither ffmpeg/libvmaf nor vqmt is available")
            mean_vmaf = 0.0
        else:
            mean_vmaf = 0.0
            for name, attempt in attempts:
                value = attempt()
                if value is not None:
                    mean_vmaf = value
                    break
                if backend != "auto":
                    break
    except subprocess.TimeoutExpired:
        print(f"VMAF calculation timed out after {time.time() - start_time:.2f}s (timeout={timeout_sec}s)")
        mean_vmaf = 0.0
    except Exception as e:
        print(f"VMAF calculation error: {e}")
        mean_vmaf = 0.0
    finally:
        for temp_file in (ref_video, dist_video, log_file, csv_file):
            if temp_file is None:
                continue
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass
    return mean_vmaf


_WARNED_ONCE = set()


def _warn_once(key, msg):
    if key in _WARNED_ONCE:
        return
    _WARNED_ONCE.add(key)
    print(msg, flush=True)


def _standardize_attack_video_tensor(x):
    """
    Standard benchmark video metric tensor format helper.
    Assumes attack/eval tensors are video clips in [..., H, W, C] and flattens leading clip dims.
    Produces THWC-like tensor expected by original metric implementations.
    """
    x = torch.as_tensor(x)
    return x.reshape(-1, *x.shape[-3:])


def write_frame_metrics_csv_attack(
    *,
    frames_csv,
    sample_path: str,
    clean_label: int,
    clean_conf: float,
    attacked_label: int,
    attacked_conf: float,
    clear_video,
    attacked_video,
):
    """
    Write per-frame PSNR/SSIM/MSE rows for a single attacked sample.
    Uses original metric implementations and standard benchmark tensor format assumptions.
    """
    a_frames = to_numpy(_standardize_attack_video_tensor(clear_video))
    b_frames = to_numpy(_standardize_attack_video_tensor(attacked_video))
    n = min(len(a_frames), len(b_frames))
    with open(frames_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([sample_path, "meta", clean_label, clean_conf, attacked_label, attacked_conf])
        psnr_list, ssim_list, mse_list = [], [], []
        for fi in range(n):
            fa = a_frames[fi]
            fb = b_frames[fi]
            psnr_i = float(peak_signal_noise_ratio(fa, fb, data_range=255))
            ssim_i = float(structural_similarity(fa, fb, channel_axis=2, data_range=255))
            mse_i = float(mean_squared_error(fa, fb))
            psnr_list.append(psnr_i)
            ssim_list.append(ssim_i)
            mse_list.append(mse_i)
        writer.writerow(["PSNR", *psnr_list])
        writer.writerow(["SSIM", *ssim_list])
        writer.writerow(["MSE", *mse_list])

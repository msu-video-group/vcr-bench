"""StyleFool (untargeted) for vcr_bench.

This implementation follows the original three-stage recipe more closely:
style selection, style transfer, then NES-based adversarial generation.

Offline preparation can still precompute fallback neural style caches, but the
attack itself always starts from a stylized video instead of silently degrading
to a no-style initialization.
"""
from __future__ import annotations

import colorsys
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from vcr_bench.attacks.query_base import BaseQueryVideoAttack
from vcr_bench.attacks.utils import flatten_temporal_video, restore_temporal_video
from vcr_bench.models.base import BaseVideoClassifier


# ---------------------------------------------------------------------------
# Cache / path helpers
# ---------------------------------------------------------------------------

def _cache_root() -> Path:
    return Path(os.getenv("STYLEFOOL_CACHE_DIR", "video_classifiers/methods/artifacts/stylefool_cache"))


def _stylefool_original_root() -> Path:
    explicit = os.getenv("STYLEFOOL_ORIGINAL_ROOT", "").strip()
    if explicit:
        return Path(explicit)
    for candidate in (Path("attacks/StyleFool"), Path("Attacks/StyleFool")):
        if candidate.exists():
            return candidate
    return Path("attacks/StyleFool")


def _style_images_dir() -> Path:
    explicit = os.getenv("STYLEFOOL_STYLES_DIR", "").strip()
    if explicit:
        return Path(explicit)
    return _stylefool_original_root() / "styles"


def _file_sig(path: Path) -> str:
    try:
        st = path.stat()
        return f"{path.resolve()}|{int(st.st_mtime)}|{st.st_size}"
    except Exception:
        return str(path)


def _sha1_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _sha1_tensor(x: torch.Tensor) -> str:
    arr = x.detach().cpu().contiguous().numpy()
    h = hashlib.sha1()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Style descriptor helpers
# ---------------------------------------------------------------------------

_MC_THEME_COUNT = 3
_HSV_CYL_RADIUS = 100.0
_HSV_CYL_ANGLE_DEG = 30.0
_HSV_CYL_H0 = _HSV_CYL_RADIUS * math.cos(math.radians(_HSV_CYL_ANGLE_DEG))
_HSV_CYL_R0 = _HSV_CYL_RADIUS * math.sin(math.radians(_HSV_CYL_ANGLE_DEG))


def _median_cut_themes_rgb(img_rgb: np.ndarray, n_colors: int = _MC_THEME_COUNT) -> np.ndarray:
    """Approximate the paper's median-cut color themes."""
    pixels = np.asarray(img_rgb, dtype=np.float32).reshape(-1, 3)
    if pixels.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    buckets: list[np.ndarray] = [pixels]
    while len(buckets) < n_colors:
        split_idx = max(
            range(len(buckets)),
            key=lambda i: float((buckets[i].max(axis=0) - buckets[i].min(axis=0)).max()) if buckets[i].shape[0] > 1 else -1.0,
        )
        bucket = buckets.pop(split_idx)
        if bucket.shape[0] <= 1:
            buckets.append(bucket)
            break
        channel = int(np.argmax(bucket.max(axis=0) - bucket.min(axis=0)))
        order = np.argsort(bucket[:, channel], kind="mergesort")
        bucket = bucket[order]
        mid = bucket.shape[0] // 2
        left, right = bucket[:mid], bucket[mid:]
        if left.shape[0] == 0 or right.shape[0] == 0:
            buckets.append(bucket)
            break
        buckets.extend([left, right])
        if all(b.shape[0] <= 1 for b in buckets):
            break

    themes = [np.median(bucket, axis=0) for bucket in buckets[:n_colors] if bucket.size > 0]
    if not themes:
        return np.zeros((0, 3), dtype=np.float32)
    out = np.stack(themes, axis=0).astype(np.float32)
    if out.shape[0] < n_colors:
        out = np.concatenate([out, np.repeat(out[-1:], n_colors - out.shape[0], axis=0)], axis=0)
    return np.clip(out, 0.0, 255.0)


def _rgb_themes_to_hsv(rgb_themes: np.ndarray) -> np.ndarray:
    hsv = []
    for rgb in np.asarray(rgb_themes, dtype=np.float32):
        r, g, b = (float(rgb[0]) / 255.0, float(rgb[1]) / 255.0, float(rgb[2]) / 255.0)
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        hsv.append([360.0 * h, s, v])
    return np.asarray(hsv, dtype=np.float32)


def _hsv_themes_to_xyz(hsv_themes: np.ndarray) -> np.ndarray:
    xyz = []
    for h, s, v in np.asarray(hsv_themes, dtype=np.float32):
        rad = math.radians(float(h))
        x = _HSV_CYL_R0 * float(v) * float(s) * math.cos(rad)
        y = _HSV_CYL_R0 * float(v) * float(s) * math.sin(rad)
        z = _HSV_CYL_H0 * (1.0 - float(v))
        xyz.append([x, y, z])
    return np.asarray(xyz, dtype=np.float32)


def _theme_descriptor_from_rgb_image(img_rgb: np.ndarray) -> dict:
    rgb_themes = _median_cut_themes_rgb(img_rgb, n_colors=_MC_THEME_COUNT)
    hsv_themes = _rgb_themes_to_hsv(rgb_themes)
    xyz_themes = _hsv_themes_to_xyz(hsv_themes)
    return {
        "rgb_themes": rgb_themes.tolist(),
        "hsv_themes": hsv_themes.tolist(),
        "xyz_themes": xyz_themes.tolist(),
    }


def _video_desc_from_file(path: Path) -> dict | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return _theme_descriptor_from_rgb_image(rgb)


def _video_desc_from_tensor(x: torch.Tensor) -> dict:
    arr = x.detach().cpu().numpy().astype(np.float32)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Expected THWC RGB tensor, got {tuple(arr.shape)}")
    frame0 = arr[0]
    if frame0.max() <= 1.5:
        frame0 = frame0 * 255.0
    return _theme_descriptor_from_rgb_image(frame0)


def _video_desc_cache_path(video_path: str) -> Path:
    key = _sha1_str(f"themes_v2|{_file_sig(Path(video_path))}")
    d = _cache_root() / "video_desc"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.json"


def get_or_compute_video_desc(video_path: str) -> dict | None:
    cache = _video_desc_cache_path(video_path)
    if cache.is_file():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "xyz_themes" in data:
                return data
        except Exception:
            pass
    data = _video_desc_from_file(Path(video_path))
    if data is not None:
        try:
            cache.write_text(json.dumps(data), encoding="utf-8")
        except Exception:
            pass
    return data


# ---------------------------------------------------------------------------
# Style pool index
# ---------------------------------------------------------------------------

def _style_pool_sig(paths: list[Path]) -> str:
    h = hashlib.sha1()
    h.update(b"themes_v2")
    for p in paths:
        try:
            st = p.stat()
            h.update(f"{p.resolve()}|{int(st.st_mtime)}|{st.st_size}".encode())
        except Exception:
            h.update(str(p).encode())
    return h.hexdigest()


def _build_style_index(paths: list[Path]) -> list[dict]:
    index = []
    for p in paths:
        try:
            img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            index.append({"path": str(p), "name": p.name, **_theme_descriptor_from_rgb_image(img_rgb)})
        except Exception:
            pass
    return index


def load_style_index() -> list[dict]:
    styles_dir = _style_images_dir()
    if not styles_dir.is_dir():
        return []
    paths = sorted([p for p in styles_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not paths:
        return []
    cache = _cache_root() / "style_index" / f"{_style_pool_sig(paths)}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.is_file():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except Exception:
            pass
    data = _build_style_index(paths)
    try:
        cache.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass
    return data


def _style_distance(vd: dict, sd: dict) -> float:
    vx = np.asarray(vd.get("xyz_themes", []), dtype=np.float32)
    sx = np.asarray(sd.get("xyz_themes", []), dtype=np.float32)
    n = min(int(vx.shape[0]), int(sx.shape[0]), _MC_THEME_COUNT)
    if n <= 0:
        return float("inf")
    return float(np.linalg.norm(vx[:n] - sx[:n], axis=1).sum())


def select_style(cache_key: str, video_desc: dict | None) -> Path | None:
    explicit = os.getenv("STYLEFOOL_STYLE_IMAGE", "").strip()
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
    styles_dir = _style_images_dir()
    if not styles_dir.is_dir():
        return None
    styles = sorted([p for p in styles_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not styles:
        return None
    index = load_style_index()
    if not index or video_desc is None:
        return styles[int(cache_key[:8], 16) % len(styles)]
    ranked = sorted((round(_style_distance(video_desc, item), 6), str(item["path"])) for item in index)
    if not ranked:
        return styles[int(cache_key[:8], 16) % len(styles)]
    return Path(ranked[0][1])


# ---------------------------------------------------------------------------
# VGG neural style transfer  (Gatys et al., CVPR 2016)
# ---------------------------------------------------------------------------

# VGG-19 feature layer indices (in .features sequential):
#   relu1_1=1, relu2_1=6, relu3_1=11, relu4_1=20  → style Gram matrices
#   relu4_2=22                                      → content features
_VGG_STYLE_IDXS  = [1, 6, 11, 20]
_VGG_CONTENT_IDX = 22
_VGG_MEAN = [0.485, 0.456, 0.406]
_VGG_STD  = [0.229, 0.224, 0.225]


def _load_vgg(device: str) -> torch.nn.Module | None:
    try:
        import torchvision.models as m
        vgg = m.vgg19(weights=m.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        return vgg
    except Exception:
        return None


def _vgg_features(
    vgg: torch.nn.Module,
    x: torch.Tensor,    # (1, 3, H, W), VGG-normalised
) -> tuple[dict[int, torch.Tensor], torch.Tensor | None]:
    """Run VGG up to relu4_2; collect Gram matrices at style layers and content feature."""
    grams: dict[int, torch.Tensor] = {}
    content: torch.Tensor | None = None
    for i, layer in enumerate(vgg):
        x = layer(x)
        if i in _VGG_STYLE_IDXS:
            c = x.shape[1]
            flat = x.view(c, -1)
            grams[i] = (flat @ flat.t()) / flat.numel()
        if i == _VGG_CONTENT_IDX:
            content = x
            break
    return grams, content


def _neural_style_transfer(
    content_thwc: torch.Tensor,   # (T, H, W, C) float in [0, hi]
    style_path: Path,
    hi: float,
    device: str,
    steps: int = 200,
    content_weight: float = 1.0,
    style_weight: float = 1e5,
) -> torch.Tensor:
    """Per-frame Gram-matrix style transfer.  Falls back to AdaIN blend if VGG unavailable."""
    vgg = _load_vgg(device)
    if vgg is None:
        return _adain_blend(content_thwc, style_path, hi)

    vgg_mean = torch.tensor(_VGG_MEAN, device=device).view(1, 3, 1, 1)
    vgg_std  = torch.tensor(_VGG_STD,  device=device).view(1, 3, 1, 1)

    img_bgr = cv2.imread(str(style_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return content_thwc.clone()

    t, h, w, _c = content_thwc.shape
    style_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    style_rgb = cv2.resize(style_rgb, (w, h), interpolation=cv2.INTER_AREA)
    style_chw = torch.from_numpy(style_rgb / hi).to(device, dtype=torch.float32).permute(2, 0, 1)

    with torch.no_grad():
        style_grams, _ = _vgg_features(vgg, (style_chw.unsqueeze(0) - vgg_mean) / vgg_std)

    result = content_thwc.clone().to(device)
    for i in range(t):
        frame_chw = content_thwc[i].to(device).permute(2, 0, 1).float() / hi
        frame_norm = (frame_chw.unsqueeze(0) - vgg_mean) / vgg_std
        with torch.no_grad():
            _, content_feat = _vgg_features(vgg, frame_norm)

        x = frame_chw.clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=0.01)
        for _ in range(steps):
            opt.zero_grad()
            x_n = (x.clamp(0.0, 1.0).unsqueeze(0) - vgg_mean) / vgg_std
            cur_grams, cur_feat = _vgg_features(vgg, x_n)
            s_loss = sum(F.mse_loss(cur_grams[k], style_grams[k]) for k in style_grams)
            c_loss = F.mse_loss(cur_feat, content_feat)
            (style_weight * s_loss + content_weight * c_loss).backward()
            opt.step()

        result[i] = x.detach().clamp(0.0, 1.0).permute(1, 2, 0) * hi

    return result


# ---------------------------------------------------------------------------
# Pixel-level AdaIN blend  (fast fallback, no VGG required)
# ---------------------------------------------------------------------------

def _adain_blend(orig: torch.Tensor, style_path: Path | None, hi: float) -> torch.Tensor:
    """Shift each frame's pixel mean/std to match the style image statistics."""
    if style_path is None or not style_path.is_file():
        return orig.clone()
    img_bgr = cv2.imread(str(style_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return orig.clone()
    t, h, w, c = orig.shape
    style_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    style_rgb = cv2.resize(style_rgb, (w, h), interpolation=cv2.INTER_AREA)
    style_t   = torch.from_numpy(style_rgb).to(orig.device, dtype=orig.dtype)
    s_mean = style_t.mean(dim=(0, 1), keepdim=True)
    s_std  = style_t.std(dim=(0, 1),  keepdim=True).clamp_min(1e-6)
    x      = orig.clone()
    f_mean = x.mean(dim=(1, 2), keepdim=True)
    f_std  = x.std(dim=(1, 2),  keepdim=True).clamp_min(1e-6)
    adain  = (x - f_mean) / f_std * s_std.unsqueeze(0) + s_mean.unsqueeze(0)
    alpha  = min(max(float(os.getenv("STYLEFOOL_INIT_BLEND", "0.35")), 0.0), 1.0)
    return ((1 - alpha) * x + alpha * adain).clamp(0.0, hi)


# ---------------------------------------------------------------------------
# Stylized-init cache helpers
# ---------------------------------------------------------------------------

def _original_backend_enabled() -> bool:
    raw = os.getenv("STYLEFOOL_USE_ORIGINAL_BACKEND", "auto").strip().lower()
    root = _stylefool_original_root()
    script_ok = (root / "stylizeVideo_untarget.sh").is_file()
    th_ok = shutil.which("th") is not None
    ffmpeg_ok = shutil.which("ffmpeg") is not None or shutil.which("avconv") is not None
    available = script_ok and th_ok and ffmpeg_ok
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return available
    return available


def _resize_styled_to_match(styled: torch.Tensor, ref: torch.Tensor) -> torch.Tensor | None:
    if not torch.is_tensor(styled) or styled.dim() != 4 or ref.dim() != 4:
        return None
    if styled.shape[-1] != ref.shape[-1]:
        return None
    out = styled.to(ref.device, dtype=ref.dtype)
    if out.shape[0] != ref.shape[0]:
        idx = torch.linspace(0, out.shape[0] - 1, ref.shape[0], device=out.device)
        idx = torch.round(idx).long().clamp(0, out.shape[0] - 1)
        out = out.index_select(0, idx)
    if out.shape[1] != ref.shape[1] or out.shape[2] != ref.shape[2]:
        out = F.interpolate(
            out.permute(0, 3, 1, 2),
            size=(ref.shape[1], ref.shape[2]),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)
    return out


def _sampled_cache_path(sample_key: str, style_name: str, backend_tag: str, steps: int) -> Path:
    d = _cache_root() / backend_tag / "sampled" / sample_key[:2] / sample_key
    d.mkdir(parents=True, exist_ok=True)
    suffix = f"{style_name}_s{steps}.pt" if steps > 0 else f"{style_name}.pt"
    return d / suffix


def _write_temp_video(orig_thwc: torch.Tensor, out_path: Path, fps: int = 5) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = orig_thwc.detach().cpu().numpy()
    if arr.max() <= 1.5:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    height, width = int(arr.shape[1]), int(arr.shape[2])
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create temporary style-transfer video: {out_path}")
    try:
        for frame in arr:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _try_original_style_transfer(
    orig_thwc: torch.Tensor,
    sample_key: str,
    style_path: Path | None,
    *,
    hi: float,
    verbose: bool = False,
) -> torch.Tensor | None:
    if not _original_backend_enabled() or style_path is None or not style_path.is_file():
        return None

    root = _stylefool_original_root()
    script = root / "stylizeVideo_untarget.sh"
    if not script.is_file():
        return None

    work_tmp = _cache_root() / "_tmp"
    work_tmp.mkdir(parents=True, exist_ok=True)
    video_name = f"sf_{sample_key[:16]}"
    temp_video = work_tmp / f"{video_name}.avi"

    (root / "styled_npy" / "v_untarget").mkdir(parents=True, exist_ok=True)
    (root / "v_untarget").mkdir(parents=True, exist_ok=True)

    try:
        _write_temp_video(orig_thwc, temp_video, fps=5)
        proc = subprocess.run(
            ["bash", str(script), str(temp_video), str(style_path)],
            cwd=str(root),
            check=False,
            capture_output=True,
            text=True,
            timeout=int(os.getenv("STYLEFOOL_ORIGINAL_TIMEOUT_SEC", "3600")),
        )
        if proc.returncode != 0:
            if verbose:
                err = (proc.stderr or proc.stdout or "").strip()
                print(f"[stylefool] original_backend FAIL rc={proc.returncode} err={err[-400:]}", flush=True)
            return None

        style_name = style_path.stem.replace("%", "x")
        styled_npy = root / "styled_npy" / "v_untarget" / f"{video_name}-stylized_{style_name}.npy"
        if not styled_npy.is_file():
            if verbose:
                print(f"[stylefool] original_backend missing output: {styled_npy}", flush=True)
            return None

        arr = np.load(styled_npy)
        styled = torch.from_numpy(arr).to(orig_thwc.device, dtype=orig_thwc.dtype)
        styled = _resize_styled_to_match(styled, orig_thwc)
        if styled is None:
            return None
        return styled.clamp(0.0, hi)
    except Exception as e:
        if verbose:
            print(f"[stylefool] original_backend exception: {type(e).__name__}: {e}", flush=True)
        return None
    finally:
        try:
            if temp_video.exists():
                temp_video.unlink()
        except Exception:
            pass

def _video_key(video_path: str | None, orig_thwc: torch.Tensor) -> str:
    """Stable cache key: file-path sig when path available, otherwise tensor hash."""
    if video_path:
        return _sha1_str(_file_sig(Path(video_path)))
    return _sha1_tensor(orig_thwc)


def _neural_cache_path(video_key: str, style_name: str, steps: int) -> Path:
    d = _cache_root() / "neural" / video_key[:2] / video_key
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{style_name}_s{steps}.pt"


def _adain_cache_path(video_key: str, style_name: str) -> Path:
    d = _cache_root() / "adain" / video_key[:2] / video_key
    d.mkdir(parents=True, exist_ok=True)
    return d / f"init_{style_name}.pt"


def get_or_compute_stylized_init(
    orig_thwc: torch.Tensor,
    hi: float,
    video_path: str | None = None,
    style_steps: int = 0,
    device: str | None = None,
    content_weight: float = 1.0,
    style_weight: float = 1e5,
    verbose: bool = False,
) -> torch.Tensor:
    """Load or compute the mandatory stylized initialization."""
    key = _video_key(video_path, orig_thwc)
    sample_key = _sha1_tensor(orig_thwc)
    video_desc = get_or_compute_video_desc(video_path) if video_path else None
    if video_desc is None:
        video_desc = _video_desc_from_tensor(orig_thwc)
    style_path = select_style(key, video_desc)
    style_name = style_path.stem if style_path else "nostyle"
    device = device or str(orig_thwc.device)

    if verbose:
        print(
            f"[stylefool] style_select  key={key[:8]}  style={style_name}  "
            f"path={style_path}",
            flush=True,
        )

    def _load(p: Path) -> torch.Tensor | None:
        try:
            t = torch.load(p, map_location=orig_thwc.device)
            if torch.is_tensor(t):
                if tuple(t.shape) == tuple(orig_thwc.shape):
                    return t.to(orig_thwc.device, dtype=orig_thwc.dtype)
                matched = _resize_styled_to_match(t, orig_thwc)
                if matched is not None:
                    return matched.to(orig_thwc.device, dtype=orig_thwc.dtype)
        except Exception:
            pass
        return None

    exact_cache_candidates = [
        _sampled_cache_path(sample_key, style_name, "original", style_steps),
        _sampled_cache_path(sample_key, style_name, "neural", style_steps),
        _sampled_cache_path(sample_key, style_name, "adain", 0),
    ]
    for cache_path in exact_cache_candidates:
        if cache_path.is_file():
            cached = _load(cache_path)
            if cached is not None:
                if verbose:
                    print(f"[stylefool] style_source exact_cache  file={cache_path.name}", flush=True)
                return cached.clamp(0.0, hi)

    if style_steps > 0:
        nc = _neural_cache_path(key, style_name, style_steps)
        if nc.is_file():
            cached = _load(nc)
            if cached is not None:
                if verbose:
                    print(f"[stylefool] style_source neural_cache  steps={style_steps}  file={nc.name}", flush=True)
                try:
                    torch.save(cached.detach().cpu(), _sampled_cache_path(sample_key, style_name, "neural", style_steps))
                except Exception:
                    pass
                return cached
        if verbose:
            print(f"[stylefool] style_source neural_cache MISS  steps={style_steps}  (prepare not run?)", flush=True)

    ac = _adain_cache_path(key, style_name)
    if ac.is_file():
        cached = _load(ac)
        if cached is not None:
            if verbose:
                print(f"[stylefool] style_source adain_cache  file={ac.name}", flush=True)
            try:
                torch.save(cached.detach().cpu(), _sampled_cache_path(sample_key, style_name, "adain", 0))
            except Exception:
                pass
            return cached

    backend_tag = "adain"
    styled = _try_original_style_transfer(orig_thwc, sample_key, style_path, hi=hi, verbose=verbose)
    if styled is not None:
        backend_tag = "original"
    elif style_steps > 0 and style_path is not None:
        styled = _neural_style_transfer(
            orig_thwc,
            style_path,
            hi=hi,
            device=device,
            steps=style_steps,
            content_weight=content_weight,
            style_weight=style_weight,
        )
        backend_tag = "neural"
    else:
        styled = _adain_blend(orig_thwc, style_path, hi)

    exact_cache = _sampled_cache_path(sample_key, style_name, backend_tag, style_steps if backend_tag != "adain" else 0)
    try:
        torch.save(styled.detach().cpu(), exact_cache)
    except Exception:
        pass
    if verbose:
        print(f"[stylefool] style_source computed  backend={backend_tag}  cached={exact_cache.name}", flush=True)
    return styled.clamp(0.0, hi)


# ---------------------------------------------------------------------------
# NES gradient estimation  (matches original apply_NES, untargeted)
# ---------------------------------------------------------------------------

def _nes_grad(
    model: BaseVideoClassifier,
    x_thwc: torch.Tensor,
    benign_label: int,
    input_format: str,
    restore_shape: tuple,
    sigma: float,
    n_samples: int,
    sub_num: int = 4,
    lower: torch.Tensor | None = None,
    upper: torch.Tensor | None = None,
) -> tuple[float | None, torch.Tensor | None, torch.Tensor | None, int]:
    """NES gradient via antithetic pairs.

    This mirrors the original untargeted StyleFool loss more closely:
    use `-max(score)` on noisy samples that still predict the benign label,
    and treat already-misclassified noisy samples as lucky shots.

    Returns `(mean_loss, grad, lucky_adv, used_queries)`.
    """
    assert n_samples % sub_num == 0 and sub_num % 2 == 0
    grad = torch.zeros_like(x_thwc)
    loss_total = 0.0
    count_in = 0
    used_queries = 0

    for _ in range(n_samples // sub_num):
        half = sub_num // 2
        noise = torch.randn((half,) + x_thwc.shape, device=x_thwc.device) * sigma
        all_noise = torch.cat([noise, -noise], 0)  # antithetic pairs (sub_num, ...)

        for j in range(sub_num):
            v = x_thwc + all_noise[j]
            if lower is not None and upper is not None:
                v = torch.max(torch.min(v, upper), lower)
            restored = restore_temporal_video(v, restore_shape)
            with torch.no_grad():
                s = model(restored, input_format=input_format, enable_grad=False).reshape(-1)
            used_queries += 1
            top = int(s.argmax().item())
            if top != benign_label:
                # Lucky shot: noisy sample already fools the model
                return None, None, v, used_queries
            loss = -float(s.max().item())
            count_in += 1
            loss_total += loss
            grad = grad + all_noise[j] / sigma * loss

    if count_in == 0:
        return None, None, None, used_queries
    return loss_total / count_in, grad, None, used_queries


# ---------------------------------------------------------------------------
# Main attack class
# ---------------------------------------------------------------------------

class StyleFoolAttack(BaseQueryVideoAttack):
    """Untargeted StyleFool with mandatory style-transfer initialization."""
    attack_name = "stylefool"

    def __init__(
        self,
        eps: float = 12.75,
        alpha: float = 1.0,
        steps: int = 200,
        sigma: float | None = None,
        n_samples: int = 64,
        sub_num: int = 4,
        step_mult: float = 8.0,
        lr_annealing: bool = True,
        query_budget: int = 300_000,
        style_steps: int = 200,           # neural style transfer iters in prepare (0 = AdaIN only)
        style_content_weight: float = 1.0,
        style_style_weight: float = 1e5,
    ) -> None:
        super().__init__(eps=eps, alpha=alpha, steps=steps)
        self.sigma_override        = None if sigma is None else float(sigma)
        self.n_samples             = int(n_samples)
        self.sub_num               = int(sub_num)
        self.step_mult             = float(step_mult)
        self.lr_annealing          = bool(lr_annealing)
        self.query_budget          = int(query_budget)
        self.style_steps           = int(style_steps)
        self.style_content_weight  = float(style_content_weight)
        self.style_style_weight    = float(style_style_weight)
        self.verbose               = False

    def prepare(self, *, model, dataset, indices, seed, targeted, pipeline_stage, verbose) -> None:
        del model, dataset, indices, seed, targeted, pipeline_stage
        self.verbose = bool(verbose)

    # ------------------------------------------------------------------

    def supports_offline_prepare(self) -> bool:
        return True

    def prepare_offline(
        self,
        *,
        dataset,
        indices: list[int],
        seed: int,
        targeted: bool,
        pipeline_stage: str,
        verbose: bool,
        device: str = "cpu",
        **kwargs,
    ) -> dict:
        """Precompute VGG neural style transfer for each video and save to cache."""
        index = load_style_index()
        if verbose:
            print(f"[stylefool prepare] style index: {len(index)} styles from {_style_images_dir()}")

        ok = fail = skip = 0
        for idx in indices:
            try:
                item = dataset[idx]   # VideoSampleRef — path only, no tensor
                path = str(item.path)

                key        = _sha1_str(_file_sig(Path(path)))
                video_desc = get_or_compute_video_desc(path)
                style_path = select_style(key, video_desc)
                style_name = style_path.stem if style_path else "nostyle"

                nc = _neural_cache_path(key, style_name, self.style_steps)
                if nc.is_file():
                    skip += 1
                    continue

                if style_path is None:
                    if verbose:
                        print(f"[stylefool prepare] idx={idx}: no style images found, skipping", flush=True)
                    skip += 1
                    continue

                if verbose:
                    print(f"[stylefool prepare] idx={idx} style={style_name} steps={self.style_steps}", flush=True)

                # Load video frames from file using OpenCV
                cap = cv2.VideoCapture(path)
                frames: list[np.ndarray] = []
                while len(frames) < 16:
                    ok_read, frame = cap.read()
                    if not ok_read:
                        break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
                if not frames:
                    fail += 1
                    continue

                arr = np.stack(frames).astype(np.float32)          # (T, H, W, 3) uint8 range
                x_flat = torch.from_numpy(arr)                     # (T, H, W, C) float32 [0,255]
                hi = 255.0
                styled = _neural_style_transfer(
                    x_flat, style_path, hi=hi, device=device,
                    steps=self.style_steps,
                    content_weight=self.style_content_weight,
                    style_weight=self.style_style_weight,
                )
                torch.save(styled.detach().cpu(), nc)
                ok += 1
            except Exception as e:
                if verbose:
                    print(f"[stylefool prepare] idx={idx} failed: {e}", flush=True)
                fail += 1

        if verbose:
            print(f"[stylefool prepare] done: ok={ok} skip={skip} fail={fail}")
        return {"style_index_size": len(index), "styled_ok": ok, "styled_skip": skip, "styled_fail": fail}

    # ------------------------------------------------------------------

    def attack_query(
        self,
        model: BaseVideoClassifier,
        x: torch.Tensor,
        *,
        query_label,
        input_format: str = "NTHWC",
        y=None,
        targeted: bool = False,
        video_path: str | None = None,
    ) -> torch.Tensor:
        del query_label, y
        if targeted:
            raise ValueError("StyleFoolAttack only supports untargeted mode.")

        x_ref = self._ensure_float_attack_tensor(x.detach().clone())
        x_flat, restore_shape = flatten_temporal_video(x_ref)
        orig_thwc = x_flat  # (T, H, W, C)

        hi  = 255.0 if float(orig_thwc.max().item()) > 1.5 else 1.0
        lo  = 0.0
        eps_sc = self.eps / 255.0 if hi <= 1.0 else self.eps
        sigma  = self.sigma_override if self.sigma_override is not None else (1e-3 if hi <= 1.0 else 1.0)
        device = orig_thwc.device

        query_count = 0

        def call_model(v_thwc: torch.Tensor) -> torch.Tensor:
            nonlocal query_count
            query_count += 1
            restored = restore_temporal_video(v_thwc, restore_shape)
            with torch.no_grad():
                s = model(restored, input_format=input_format, enable_grad=False)
            return s.reshape(-1)

        benign_scores = call_model(orig_thwc)
        benign_label  = int(benign_scores.argmax().item())

        if self.verbose:
            top3_vals, top3_idx = torch.topk(benign_scores, min(3, benign_scores.numel()))
            top3_str = "  ".join(f"{int(i)}:{float(v):.4f}" for v, i in zip(top3_vals, top3_idx))
            print(
                f"[stylefool] start  video={video_path or 'tensor'}  "
                f"shape={tuple(orig_thwc.shape)}  hi={hi:.0f}  "
                f"eps={eps_sc:.2f}  sigma={sigma:.4f}  steps={self.steps}  "
                f"n_samples={self.n_samples}  sub_num={self.sub_num}  "
                f"step_mult={self.step_mult}  style_steps={self.style_steps}  "
                f"benign={benign_label}  top3=[{top3_str}]",
                flush=True,
            )

        # Stylized init — load from prepare cache (keyed by file path) or fall back to AdaIN
        styled_init = get_or_compute_stylized_init(
            orig_thwc,
            hi=hi,
            video_path=video_path,
            style_steps=self.style_steps,
            device=str(device),
            content_weight=self.style_content_weight,
            style_weight=self.style_style_weight,
            verbose=self.verbose,
        ).to(device)

        styled_scores = call_model(styled_init)
        styled_label = int(styled_scores.argmax().item())
        styled_benign_conf = float(styled_scores[benign_label].item())

        # Untargeted StyleFool constrains NES around the stylized video xs.
        lower = torch.clamp(styled_init - eps_sc, lo, hi)
        upper = torch.clamp(styled_init + eps_sc, lo, hi)
        adv_thwc = styled_init.clone().clamp(lo, hi)

        if self.verbose:
            styled_linf = float((styled_init - orig_thwc).abs().max().item())
            print(
                f"[stylefool] init  styled_vs_orig_linf={styled_linf:.2f}  "
                f"styled_label={styled_label}  benign_conf_on_styled={styled_benign_conf:.4f}",
                flush=True,
            )

        if styled_label != benign_label:
            self.last_result = {
                "iter_count": 0,
                "query_count": query_count,
                "target_label": -1,
                "benign_label": benign_label,
                "final_label": styled_label,
            }
            if self.verbose:
                print("[stylefool] early_success  stylized video already fools the model", flush=True)
            return restore_temporal_video(adv_thwc, restore_shape).detach()

        cur_step = (eps_sc / max(1, self.steps)) * self.step_mult
        min_step = cur_step / 4.0

        final_label = benign_label
        iters_done  = 0
        last_score: list[float] = []
        last_p:     list[float] = []

        for i in range(self.steps):
            iters_done = i + 1
            if query_count >= self.query_budget:
                break

            l, g, lucky, used_queries = _nes_grad(
                model, adv_thwc, benign_label, input_format, restore_shape,
                sigma=sigma, n_samples=self.n_samples, sub_num=self.sub_num,
                lower=lower, upper=upper,
            )
            query_count += used_queries

            if lucky is not None:
                # A noisy sample already misclassified — use it directly
                adv_thwc    = lucky.clamp(lo, hi)
                cur_scores  = call_model(adv_thwc)
                final_label = int(cur_scores.argmax().item())
                benign_cur  = float(cur_scores[benign_label].item())
                if self.verbose:
                    delta_linf = float((adv_thwc - styled_init).abs().max().item())
                    top2_vals, top2_idx = torch.topk(cur_scores, min(2, cur_scores.numel()))
                    top2_str = "  ".join(f"{int(idx)}:{float(v):.4f}" for v, idx in zip(top2_vals, top2_idx))
                    print(
                        f"[stylefool] iter={i + 1}/{self.steps} queries={query_count}/{self.query_budget} "
                        f"label={final_label} benign_conf={benign_cur:.4f} "
                        f"delta_linf_xs={delta_linf:.2f}/{eps_sc:.2f} top2=[{top2_str}] [lucky]",
                        flush=True,
                    )
                if final_label != benign_label:
                    break
                continue

            if l is None or g is None:
                continue
            if not torch.isfinite(g).all():
                break

            adv_thwc = adv_thwc + cur_step * g.sign()
            adv_thwc = torch.max(torch.min(adv_thwc, upper), lower).clamp(lo, hi)

            cur_scores  = call_model(adv_thwc)
            final_label = int(cur_scores.argmax().item())
            benign_cur  = float(cur_scores[benign_label].item())

            if self.verbose:
                delta_linf = float((adv_thwc - styled_init).abs().max().item())
                top2_vals, top2_idx = torch.topk(cur_scores, min(2, cur_scores.numel()))
                top2_str = "  ".join(f"{int(idx)}:{float(v):.4f}" for v, idx in zip(top2_vals, top2_idx))
                print(
                    f"[stylefool] iter={i + 1}/{self.steps} queries={query_count}/{self.query_budget} "
                    f"loss_est={l:.4f} label={final_label} benign_conf={benign_cur:.4f} "
                    f"delta_linf_xs={delta_linf:.2f}/{eps_sc:.2f} top2=[{top2_str}] lr={cur_step:.2e}",
                    flush=True,
                )

            if final_label != benign_label:
                break

            # Stagnation: stop if benign confidence on the current adversarial video does not drop.
            last_score.append(benign_cur)
            last_score = last_score[-200:]
            if len(last_score) == 200 and last_score[-1] >= last_score[0]:
                if self.verbose:
                    print("[stylefool] stagnation detected, stopping", flush=True)
                break

            if self.lr_annealing:
                last_p.append(benign_cur)
                last_p = last_p[-20:]
                if len(last_p) == 20 and last_p[-1] <= last_p[0]:
                    if cur_step > min_step:
                        cur_step = max(cur_step / 2.0, min_step)
                    last_p = []

        self.last_result = {
            "iter_count":   iters_done,
            "query_count":  query_count,
            "target_label": -1,
            "benign_label": benign_label,
            "final_label":  final_label,
        }
        if self.verbose:
            final_linf = float((adv_thwc - styled_init).abs().max().item())
            print(
                f"[stylefool] done  iters={iters_done}  queries={query_count}  "
                f"benign={benign_label}  final={final_label}  success={final_label != benign_label}  "
                f"delta_linf_xs={final_linf:.2f}/{eps_sc:.2f}",
                flush=True,
            )
        return restore_temporal_video(adv_thwc, restore_shape).detach()


ATTACK_CLASS = StyleFoolAttack


def create(**kwargs) -> StyleFoolAttack:
    return StyleFoolAttack(**kwargs)

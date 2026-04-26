from __future__ import annotations

import os
import time
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F

from vcr_bench.defences.base import BaseVideoDefence
from .guided_diffusion.unet import UNetModel

_CHECKPOINT_URL = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"


def _ensure_checkpoint(path: str) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lock_path = f"{path}.lock"
    tmp_path = f"{path}.tmp"
    owns_lock = False
    while not owns_lock:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            owns_lock = True
        except FileExistsError:
            if os.path.exists(path):
                return
            print(f"[freqpure] Waiting for checkpoint download lock: {lock_path}", flush=True)
            time.sleep(10)

    last_pct = {"value": -1}
    print(f"[freqpure] Downloading checkpoint to {path} ...", flush=True)

    def _progress(count, block_size, total_size):
        if total_size > 0:
            pct = count * block_size * 100 // total_size
            if pct != last_pct["value"]:
                last_pct["value"] = pct
                print(f"[freqpure] download {pct}%", flush=True)

    try:
        urllib.request.urlretrieve(_CHECKPOINT_URL, tmp_path, reporthook=_progress)
        os.replace(tmp_path, path)
    finally:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass
        if not os.path.exists(path):
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass


def _build_unet() -> UNetModel:
    # Config for 256x256_diffusion_uncond.pt (OpenAI guided-diffusion, learn_sigma=True).
    # attention_resolutions: image_size // res for res in "32,16,8" → 8, 16, 32
    return UNetModel(
        image_size=256,
        in_channels=3,
        model_channels=256,
        out_channels=6,  # learn_sigma=True → 3 noise + 3 sigma
        num_res_blocks=2,
        attention_resolutions=(8, 16, 32),
        dropout=0.0,
        channel_mult=(1, 1, 2, 2, 4, 4),
        use_fp16=True,
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
    )


class FreqPureDefence(BaseVideoDefence):
    """Diffusion-based adversarial purification with FFT amplitude/phase exchange.

    Ports the FreqPure method (https://github.com/GaozhengPei/FreqPure) to the
    BaseVideoDefence interface. Each video frame is purified independently via:
      1. Add noise at ``timestep`` using a 1000-step linear beta schedule.
      2. Iteratively denoise (DDIM/DDPM) while exchanging low-frequency amplitude
         and phase components from a lightly-denoised reference image.

    Requires the 256x256 unconditional DDPM checkpoint from OpenAI guided-diffusion
    (``256x256_diffusion_uncond.pt``, ~2 GB) at ``checkpoint_path``.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str = "vcr_bench/defences/freqpure/256x256_diffusion_uncond.pt",
        timestep: int = 100,
        amplitude_cut_range: float = 50.0,
        phase_cut_range: float = 50.0,
        delta: float = 0.6,
        forward_noise_steps: int = 5,
        sampling_method: str = "ddim",
        frame_batch_size: int = 1,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.timestep = timestep
        self.amplitude_cut_range = amplitude_cut_range
        self.phase_cut_range = phase_cut_range
        self.delta = delta
        self.forward_noise_steps = forward_noise_steps
        self.frame_batch_size = max(1, int(frame_batch_size))
        assert sampling_method in ("ddim", "ddpm")
        self._eta = 0.0 if sampling_method == "ddim" else 1.0
        self._model: UNetModel | None = None
        self._betas: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self, device: torch.device) -> None:
        if self._model is not None:
            return
        _ensure_checkpoint(self.checkpoint_path)
        model = _build_unet()
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.convert_to_fp16()
        model.eval().to(device)
        self._model = model
        betas = torch.linspace(1e-4, 2e-2, 1000, dtype=torch.float32).to(device)
        self._betas = betas

    # ------------------------------------------------------------------
    # Diffusion helpers
    # ------------------------------------------------------------------

    def _compute_alpha(self, t: torch.Tensor) -> torch.Tensor:
        # t: long tensor [N] — returns alpha_bar_t: [N, 1, 1, 1]
        beta = torch.cat([torch.zeros(1, device=self._betas.device), self._betas])
        return (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)

    def _get_noised_x(self, x: torch.Tensor, t: int) -> torch.Tensor:
        e = torch.randn_like(x)
        t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        a = (1 - self._betas).cumprod(dim=0).index_select(0, t_tensor).view(-1, 1, 1, 1)
        return x * a.sqrt() + e * (1.0 - a).sqrt()

    def _predict_noise(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t: float tensor [N]
        et = self._model(xt, t)
        et, _ = torch.split(et, 3, dim=1)  # discard learned sigma
        return et

    # ------------------------------------------------------------------
    # Frequency exchange
    # ------------------------------------------------------------------

    def _freq_mask(self, rows: int, cols: int, cutoff: float, device: torch.device) -> torch.Tensor:
        u = np.arange(-cols // 2, cols // 2)
        v = np.arange(-rows // 2, rows // 2)
        U, V = np.meshgrid(u, v)
        return torch.from_numpy(np.sqrt(U ** 2 + V ** 2) <= cutoff).to(device)

    def _compute_fft(self, image: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        # image: [C, H, W] float, values in [0, 255]
        amp, phase = [], []
        for c in range(3):
            fshift = torch.fft.fftshift(torch.fft.fft2(image[c]))
            amp.append(torch.abs(fshift))
            phase.append(torch.angle(fshift) + torch.pi)
        return amp, phase

    def _reconstruct(self, amp: list[torch.Tensor], phase: list[torch.Tensor]) -> torch.Tensor:
        # Returns [H, W, C] in [0, 1]
        channels = []
        for c in range(3):
            fshift = amp[c] * torch.exp(1j * (phase[c] - torch.pi))
            img = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(fshift)))
            channels.append(torch.clamp(img, 0, 255) / 255.0)
        return torch.stack(channels, dim=2)

    def _amplitude_phase_exchange(self, ori_x: torch.Tensor, x0_t: torch.Tensor) -> torch.Tensor:
        # ori_x, x0_t: [N, 3, H, W] in diffusion space [-1, 1]
        # Compute a lightly-denoised reference from ori_x
        x_t_ref = self._get_noised_x(ori_x, self.forward_noise_steps)
        t_ref = torch.full((ori_x.shape[0],), self.forward_noise_steps, device=ori_x.device, dtype=torch.long)
        at_ref = self._compute_alpha(t_ref)
        et_ref = self._predict_noise(x_t_ref, t_ref.float())
        x_ref = (x_t_ref - et_ref * (1 - at_ref).sqrt()) / at_ref.sqrt()

        # Convert both to [0, 255]
        x_ref_255 = torch.clamp((x_ref / 2 + 0.5) * 255, 0, 255)
        x0_t_255 = torch.clamp((x0_t / 2 + 0.5) * 255, 0, 255)

        N, _, H, W = x_ref_255.shape
        amp_mask = self._freq_mask(H, W, self.amplitude_cut_range, ori_x.device)
        phase_mask = self._freq_mask(H, W, self.phase_cut_range, ori_x.device)

        result = torch.zeros(N, H, W, 3, device=ori_x.device)
        for b in range(N):
            amp_ref, phase_ref = self._compute_fft(x_ref_255[b])
            amp_tgt, phase_tgt = self._compute_fft(x0_t_255[b])

            # Amplitude: replace low-freq of target with reference
            amp_exc = [torch.where(amp_mask, amp_ref[c], amp_tgt[c]) for c in range(3)]

            # Phase: replace low-freq and clip to delta-neighbourhood of reference
            phase_exc = []
            for c in range(3):
                p = torch.where(phase_mask, phase_ref[c], phase_tgt[c])
                p[phase_mask] = torch.clamp(
                    p[phase_mask],
                    phase_ref[c][phase_mask] - self.delta,
                    phase_ref[c][phase_mask] + self.delta,
                )
                phase_exc.append(p)

            result[b] = self._reconstruct(amp_exc, phase_exc)  # [H, W, C] in [0, 1]

        # Back to [N, 3, H, W] diffusion space [-1, 1]
        result = result.permute(0, 3, 1, 2).float()
        return torch.clamp((result - 0.5) * 2, -1, 1)

    # ------------------------------------------------------------------
    # Denoising loop
    # ------------------------------------------------------------------

    def _denoise(self, ori_x: torch.Tensor, noised: torch.Tensor, seq: list[int]) -> torch.Tensor:
        n = noised.shape[0]
        seq_next = [-1] + list(seq[:-1])
        xt = noised
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = torch.full((n,), i, device=noised.device, dtype=torch.long)
            next_t = torch.full((n,), j, device=noised.device, dtype=torch.long)
            at = self._compute_alpha(t)
            at_next = self._compute_alpha(next_t)
            et = self._predict_noise(xt, t.float())
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_t = self._amplitude_phase_exchange(ori_x, x0_t)
            c1 = self._eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt = at_next.sqrt() * x0_t + c1 * torch.randn_like(noised) + c2 * et
        return xt

    # ------------------------------------------------------------------
    # BaseVideoDefence interface
    # ------------------------------------------------------------------

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, H, W, C] float32 in [0, 255]
        device = x.device
        self._load_model(device)

        N, T, H, W, C = x.shape

        # Flatten to [N*T, C, H, W] and normalise to [0, 1]
        frames = x.reshape(N * T, H, W, C).permute(0, 3, 1, 2) / 255.0

        # Resize to 256×256 (DDPM input size)
        frames_256 = F.interpolate(frames, size=(256, 256), mode="bilinear", align_corners=False)

        # Convert to diffusion space [-1, 1]
        x_diff = torch.clamp((frames_256 - 0.5) * 2, -1, 1)

        purified_chunks = []
        seq = list(range(0, self.timestep, max(1, self.timestep // 10)))
        with torch.no_grad():
            for start in range(0, x_diff.shape[0], self.frame_batch_size):
                chunk = x_diff[start : start + self.frame_batch_size]
                noised = self._get_noised_x(chunk, self.timestep)
                purified_chunks.append(self._denoise(chunk, noised, seq))
        purified_diff = torch.cat(purified_chunks, dim=0)

        # Back to [0, 1], resize to original resolution, scale to [0, 255]
        purified_01 = torch.clamp(purified_diff / 2 + 0.5, 0, 1)
        if H != 256 or W != 256:
            purified_01 = F.interpolate(purified_01, size=(H, W), mode="bilinear", align_corners=False)

        purified = (purified_01 * 255.0).clamp(0, 255)
        return purified.permute(0, 2, 3, 1).reshape(N, T, H, W, C)


def create(**kwargs) -> FreqPureDefence:
    return FreqPureDefence(**kwargs)

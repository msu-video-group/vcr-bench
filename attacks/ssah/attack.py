from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import MSELoss

from vg_videobench.attacks.base import BaseVideoAttack
from vg_videobench.attacks.utils import cross_entropy_from_probs, flatten_temporal_video, predict_probs, restore_temporal_video, safe_cuda_empty_cache


class DWT3DTiny(nn.Module):
    def __init__(self, wavename: str = "haar"):
        super().__init__()
        import pywt

        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_length = len(self.band_low)
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix_for_dim(self, dim_size: int, device: torch.device) -> torch.Tensor:
        l = math.floor(dim_size / 2)
        matrix_h = np.zeros((l, dim_size + self.band_length - 2))
        end = None if self.band_length_half == 1 else (-self.band_length_half + 1)
        index = 0
        for i in range(l):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h = matrix_h[:, (self.band_length_half - 1) : end]
        return torch.tensor(matrix_h, device=device, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 4:
            x = x.permute(3, 0, 1, 2)
        c, t, h, w = x.shape
        mt = self.get_matrix_for_dim(t, x.device)
        mh = self.get_matrix_for_dim(h, x.device)
        mw = self.get_matrix_for_dim(w, x.device)
        x_t = x.reshape(c, t, h * w)
        l_t = torch.matmul(mt, x_t).reshape(c, t // 2, h, w)
        x_h = l_t.permute(0, 1, 3, 2).reshape(c * (t // 2) * w, h)
        l_h = torch.matmul(mh, x_h.t()).t().reshape(c, t // 2, w, h // 2).permute(0, 1, 3, 2)
        x_w = l_h.reshape(c * (t // 2) * (h // 2), w)
        ll = torch.matmul(mw, x_w.t()).t().reshape(c, t // 2, h // 2, w // 2)
        return ll.permute(1, 2, 3, 0)


class IDWT3DTiny(nn.Module):
    def __init__(self, wavename: str = "haar"):
        super().__init__()
        import pywt

        wavelet = pywt.Wavelet(wavename)
        self.band_low = list(wavelet.dec_lo)
        self.band_low.reverse()
        self.band_length = len(self.band_low)
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix_for_dim(self, dim_size: int, device: torch.device) -> torch.Tensor:
        l = math.floor(dim_size / 2)
        matrix_h = np.zeros((l, dim_size + self.band_length - 2))
        end = None if self.band_length_half == 1 else (-self.band_length_half + 1)
        index = 0
        for i in range(l):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h = matrix_h[:, (self.band_length_half - 1) : end]
        return torch.tensor(matrix_h, device=device, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 4:
            x = x.permute(3, 0, 1, 2)
        c, t, h, w = x.shape
        mt = self.get_matrix_for_dim(t * 2, x.device).t()
        mh = self.get_matrix_for_dim(h * 2, x.device).t()
        mw = self.get_matrix_for_dim(w * 2, x.device).t()
        x_w = x.reshape(c * t * h, w)
        recon_w = torch.matmul(mw, x_w.t()).t().reshape(c, t, h, w * 2)
        x_h = recon_w.permute(0, 1, 3, 2).reshape(c * t * (w * 2), h)
        recon_h = torch.matmul(mh, x_h.t()).t().reshape(c, t, w * 2, h * 2).permute(0, 1, 3, 2)
        x_t = recon_h.reshape(c, t, (h * 2) * (w * 2))
        recon_t = torch.matmul(mt, x_t).reshape(c, t * 2, h * 2, w * 2)
        return recon_t.permute(1, 2, 3, 0)


class SSAHAttack(BaseVideoAttack):
    attack_name = "ssah"

    def __init__(
        self,
        eps: float = 8.0,
        alpha: float = 1.0,
        steps: int = 20,
        target_conf: float = 0.5,
        sample_chunk_size: int | None = None,
    ) -> None:
        super().__init__(eps=eps, alpha=alpha, steps=steps, clip_min=0.0, clip_max=255.0, sample_chunk_size=sample_chunk_size)
        self.target_conf = float(target_conf)

    def attack_sampled(self, model, x, *, input_format="NTHWC", y=None, targeted=False):
        source_tensor = x.detach().clone().float().to(getattr(model, "device", x.device))
        init_tensor, restore_shape = flatten_temporal_video(source_tensor)
        pert = torch.zeros_like(init_tensor, requires_grad=True)
        init_probs = predict_probs(model, restore_temporal_video(init_tensor, restore_shape), input_format=input_format, enable_grad=False)
        benign_label = int(torch.argmax(init_probs).item())
        target_label = int(self._coerce_target(model, init_tensor, input_format=input_format, y=y, targeted=targeted).item()) if targeted else -1
        dwt_3d = DWT3DTiny().to(init_tensor.device)
        idwt_3d = IDWT3DTiny().to(init_tensor.device)
        lambda_lf = 0.1
        with torch.no_grad():
            inputs_ll = dwt_3d(init_tensor)
            inputs_ll_recon = idwt_3d(inputs_ll)
        final_label = benign_label
        final_conf = float(init_probs[benign_label].item())
        iteration = 0
        for iteration in range(1, self.steps + 1):
            if pert.grad is not None:
                pert.grad.zero_()
            attacked = restore_temporal_video(init_tensor + pert, restore_shape)
            probs = predict_probs(model, attacked, input_format=input_format, enable_grad=True)
            final_label = int(torch.argmax(probs).item())
            final_conf = float(probs[final_label].item())
            if (targeted and final_label == target_label and final_conf >= self.target_conf) or ((not targeted) and final_label != benign_label):
                break
            adv_loss = cross_entropy_from_probs(probs, target_label if targeted else benign_label) * (1 if targeted else -1)
            adv_loss.backward()
            adv_grad = pert.grad.clone()
            pert.grad.zero_()
            adv_ll = dwt_3d(init_tensor + pert)
            adv_ll_recon = idwt_3d(adv_ll)
            lf_loss = lambda_lf * nn.SmoothL1Loss()(adv_ll_recon, inputs_ll_recon)
            lf_loss.backward()
            lf_grad = pert.grad.clone()
            pert.grad.zero_()
            grad = adv_grad + lf_grad
            pert.data -= self.alpha * torch.sign(grad)
            pert.data.clamp_(-self.eps, self.eps)
            safe_cuda_empty_cache()
        self.last_result = {"iter_count": int(iteration), "target_label": int(target_label)}
        return restore_temporal_video((init_tensor + pert).clamp(0.0, 255.0), restore_shape).detach().cpu()


ATTACK_CLASS = SSAHAttack


def create(**kwargs) -> SSAHAttack:
    return SSAHAttack(**kwargs)

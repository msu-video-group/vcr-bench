from __future__ import annotations

import numpy as np
import torch

from vcr_bench.attacks.base import BaseVideoAttack
from vcr_bench.attacks.utils import cross_entropy_from_probs, flatten_temporal_video, freeze_model_weights, predict_probs, restore_temporal_video, safe_cuda_empty_cache


def rgb2ycbcr(im_rgb: np.ndarray) -> np.ndarray:
    import cv2

    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
    im_ycbcr = im_ycrcb[:, :, (0, 2, 1)].astype(np.float32)
    im_ycbcr[:, :, 0] = (im_ycbcr[:, :, 0] * (235 - 16) + 16) / 255.0
    im_ycbcr[:, :, 1:] = (im_ycbcr[:, :, 1:] * (240 - 16) + 16) / 255.0
    return im_ycbcr


def make_spatial_activity_map(frame: np.ndarray) -> torch.Tensor:
    """frame: (H, W, C) float32 numpy array"""
    import cv2
    from scipy import ndimage

    im = rgb2ycbcr(frame)
    im_sob = ndimage.sobel(im[:, :, 0])
    im_zero = np.zeros_like(im_sob)
    im_zero[1:-1, 1:-1] = im_sob[1:-1, 1:-1]
    maxval = float(im_zero.max())
    if maxval == 0:
        im_zero = im_zero + 1
        maxval = 1.0
    im_sob = im_zero / maxval
    kernel = np.array([[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]], dtype="uint8")
    out_im = cv2.dilate(im_sob, kernel)
    return torch.from_numpy(out_im.astype(np.float32)).unsqueeze(-1)


class KorhonenEtAlAttack(BaseVideoAttack):
    attack_name = "korhonen-et-al"

    def __init__(
        self,
        eps: float = 8.0,
        alpha: float = 0.1,
        steps: int = 20,
        target_conf: float = 0.5,
        sample_chunk_size: int | None = None,
    ) -> None:
        super().__init__(eps=eps, alpha=alpha, steps=steps, clip_min=0.0, clip_max=255.0, sample_chunk_size=sample_chunk_size)
        self.target_conf = float(target_conf)

    def attack_sampled(self, model, x, *, input_format="NTHWC", y=None, targeted=False):
        freeze_model_weights(model)
        source = self._ensure_float_attack_tensor(x.detach().clone()).to(getattr(model, "device", x.device))
        init_flat, restore_shape = flatten_temporal_video(source)  # (T, H, W, C)
        sp_maps = torch.stack(
            [make_spatial_activity_map(init_flat[i].detach().cpu().numpy()) for i in range(init_flat.shape[0])]
        ).to(init_flat.device) * 255.0  # (T, H, W, 1)
        pert = torch.zeros_like(init_flat, requires_grad=True)

        init_probs = predict_probs(model, source, input_format=input_format, enable_grad=False)
        benign_label = int(torch.argmax(init_probs).item())
        benign_conf = float(init_probs[benign_label].item())
        target_label = int(self._coerce_target(model, source, input_format=input_format, y=y, targeted=targeted).item()) if targeted else -1

        final_label = benign_label
        final_conf = benign_conf
        iteration = 0
        for iteration in range(1, self.steps + 1):
            attacked = restore_temporal_video((init_flat + pert).clamp(0.0, 255.0), restore_shape)
            probs = predict_probs(model, attacked, input_format=input_format, enable_grad=True)
            loss = cross_entropy_from_probs(probs, target_label if targeted else benign_label) * (1 if targeted else -1)
            loss.backward()
            grad = torch.nan_to_num(pert.grad) * sp_maps
            pert.data -= self.alpha * torch.sign(grad)
            pert.grad.zero_()
            cur_probs = predict_probs(model, attacked.detach(), input_format=input_format, enable_grad=False)
            final_label = int(torch.argmax(cur_probs).item())
            final_conf = float(cur_probs[final_label].item())
            if (targeted and final_label == target_label and final_conf >= self.target_conf) or (
                (not targeted) and final_label != benign_label
            ):
                break
            safe_cuda_empty_cache()

        self.last_result = {"iter_count": int(iteration), "target_label": int(target_label)}
        return restore_temporal_video((init_flat + pert).clamp(0.0, 255.0), restore_shape).detach().cpu()


ATTACK_CLASS = KorhonenEtAlAttack


def create(**kwargs) -> KorhonenEtAlAttack:
    return KorhonenEtAlAttack(**kwargs)

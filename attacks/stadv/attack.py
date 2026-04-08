from __future__ import annotations

import torch

from vg_videobench.attacks.base import BaseVideoAttack
from vg_videobench.attacks.utils import cross_entropy_from_probs, flatten_temporal_video, freeze_model_weights, predict_probs, restore_temporal_video, safe_cuda_empty_cache


def flow_st(images: torch.Tensor, flows: torch.Tensor) -> torch.Tensor:
    _, _, h, w = images.size()
    basegrid = torch.stack(torch.meshgrid(torch.arange(0, h, device=images.device), torch.arange(0, w, device=images.device), indexing="ij"))
    sampling_grid = basegrid.unsqueeze(0).float() + flows
    sampling_grid_x = torch.clamp(sampling_grid[:, 1], 0.0, w - 1.0)
    sampling_grid_y = torch.clamp(sampling_grid[:, 0], 0.0, h - 1.0)
    x0 = torch.floor(sampling_grid_x).long()
    x1 = torch.clamp(x0 + 1, 0, w - 1)
    y0 = torch.floor(sampling_grid_y).long()
    y1 = torch.clamp(y0 + 1, 0, h - 1)
    x0 = torch.clamp(x0, 0, w - 2)
    y0 = torch.clamp(y0, 0, h - 2)
    Ia = images[:, :, y0[0], x0[0]]
    Ib = images[:, :, y1[0], x0[0]]
    Ic = images[:, :, y0[0], x1[0]]
    Id = images[:, :, y1[0], x1[0]]
    x0f = x0.float()
    x1f = x1.float()
    y0f = y0.float()
    y1f = y1.float()
    wa = (x1f - sampling_grid_x) * (y1f - sampling_grid_y)
    wb = (x1f - sampling_grid_x) * (sampling_grid_y - y0f)
    wc = (sampling_grid_x - x0f) * (y1f - sampling_grid_y)
    wd = (sampling_grid_x - x0f) * (sampling_grid_y - y0f)
    return wa.unsqueeze(1) * Ia + wb.unsqueeze(1) * Ib + wc.unsqueeze(1) * Ic + wd.unsqueeze(1) * Id


def flow_loss(flows: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    padded = torch.nn.functional.pad(flows, (1, 1, 1, 1), mode="constant", value=0)
    shifted = [
        padded[:, :, 2:, 2:],
        padded[:, :, 2:, :-2],
        padded[:, :, :-2, 2:],
        padded[:, :, :-2, :-2],
    ]
    loss = 0
    for cur in shifted:
        loss += torch.sum(torch.square(flows[:, 1] - cur[:, 1]) + torch.square(flows[:, 0] - cur[:, 0]) + epsilon)
    return loss.float().mean()


class STAdvAttack(BaseVideoAttack):
    attack_name = "stadv"

    def __init__(
        self,
        eps: float = 0.25,
        alpha: float = 0.01,
        steps: int = 20,
        phi: float = 1.0,
        beta: float = 1e-7,
        target_conf: float = 0.5,
        sample_chunk_size: int | None = None,
    ) -> None:
        super().__init__(eps=eps, alpha=alpha, steps=steps, sample_chunk_size=sample_chunk_size)
        self.phi = float(phi)
        self.beta = float(beta)
        self.target_conf = float(target_conf)

    def attack_sampled(self, model, x, *, input_format="NTHWC", y=None, targeted=False):
        freeze_model_weights(model)
        source_tensor = x.detach().clone().float().to(getattr(model, "device", x.device))
        init_tensor, restore_shape = flatten_temporal_video(source_tensor)
        init_video = init_tensor.permute(0, 3, 1, 2)
        flows = torch.nn.Parameter(torch.zeros((1, 2) + init_video.shape[2:], device=init_video.device))
        optimizer = torch.optim.Adam([flows], lr=self.alpha)
        benign_probs = predict_probs(model, restore_temporal_video(init_video.permute(0, 2, 3, 1), restore_shape), input_format=input_format, enable_grad=False)
        benign_label = int(torch.argmax(benign_probs).item())
        target_label = int(torch.argmin(benign_probs).item()) if targeted and y is None else (int(y) if y is not None else -1)
        final_label = benign_label
        final_conf = float(benign_probs[benign_label].item())
        iteration = 0
        for iteration in range(1, self.steps + 1):
            optimizer.zero_grad()
            attacked_video = flow_st(init_video, flows).clamp(0.0, 255.0)
            attacked = restore_temporal_video(attacked_video.permute(0, 2, 3, 1), restore_shape)
            probs = predict_probs(model, attacked, input_format=input_format, enable_grad=True)
            l_adv = cross_entropy_from_probs(probs, target_label if targeted else benign_label) * (1 if targeted else -1)
            l_flow = flow_loss(flows)
            loss = self.phi * l_adv + self.beta * l_flow
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                flows.clamp_(-self.eps, self.eps)
            cur_probs = predict_probs(model, attacked.detach(), input_format=input_format, enable_grad=False)
            final_label = int(torch.argmax(cur_probs).item())
            final_conf = float(cur_probs[final_label].item())
            if (targeted and final_label == target_label and final_conf >= self.target_conf) or ((not targeted) and final_label != benign_label):
                break
            safe_cuda_empty_cache()
        self.last_result = {"iter_count": int(iteration), "target_label": int(target_label)}
        return restore_temporal_video(flow_st(init_video, flows).permute(0, 2, 3, 1).clamp(0.0, 255.0), restore_shape).detach().cpu()


ATTACK_CLASS = STAdvAttack


def create(**kwargs) -> STAdvAttack:
    return STAdvAttack(**kwargs)

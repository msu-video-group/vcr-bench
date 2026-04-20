from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from vcr_bench.attacks.query_base import BaseQueryVideoAttack
from vcr_bench.attacks.utils import flatten_temporal_video, restore_temporal_video, safe_cuda_empty_cache, safe_cuda_reset_peak_memory_stats
from vcr_bench.models.base import BaseVideoClassifier
from vcr_bench.types import PredictionBundle


def _p_selection(p_init: float, it: int, n_iters: int) -> float:
    """Decay square fraction over the attack horizon."""
    frac = int(it / n_iters * 10_000)
    thresholds = [10, 50, 200, 500, 1000, 2000, 4000, 6000, 8000]
    divisors = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    for thresh, div in zip(thresholds, divisors[1:]):
        if frac <= thresh:
            return p_init / div
    return p_init / divisors[-1]


@dataclass
class SquarePrimitive:
    t0: int
    t1: int
    h0: int
    w0: int
    size: int
    sign: torch.Tensor
    mag: float
    texture: torch.Tensor | None = None

    def clone(self) -> "SquarePrimitive":
        return SquarePrimitive(
            t0=int(self.t0),
            t1=int(self.t1),
            h0=int(self.h0),
            w0=int(self.w0),
            size=int(self.size),
            sign=self.sign.clone(),
            mag=float(self.mag),
            texture=None if self.texture is None else self.texture.clone(),
        )


@dataclass
class EliteWindow:
    t0: int
    t1: int
    h0: int
    w0: int
    size: int
    sign: torch.Tensor
    mag: float
    score: float


class SquareAttack(BaseQueryVideoAttack):
    """Discrete square-patch black-box attack with GradEstV2-style scoring."""

    attack_name = "square"

    def __init__(
        self,
        eps: float = 64.0,
        alpha: float | None = None,
        steps: int = 200,
        p_init: float = 0.1,
        max_patch_frac: float = 0.2,
        spatial_size: int | None = 224,
        temporal_tiles: int = 8,
        query_budget: int = 100_000,
        random_start: bool = True,
        random_start_trials: int = 4,
        coarse_to_fine: bool = True,
        coarse_spatial_scale: float = 0.5,
        coarse_temporal_scale: float = 0.5,
        plateau_patience: int = 12,
        reject_alpha_decay: float = 0.5,
        accept_alpha_growth: float = 1.05,
        boost_top2: float = 1.0,
        boost_top10: float = 0.15,
        boost_top5: float | None = None,
        boost_rival: float = 1.25,
        boost_conf: float = 0.0,
        candidates_per_step: int = 2,
        candidates_per_step_late: int = 2,
        scout_queries: int = 0,
        elite_window_count: int = 8,
        elite_candidates_per_step: int = 1,
        elite_jitter_frac: float = 0.20,
        neighborhood_candidates_per_step: int = 1,
        neighborhood_size_jitter: float = 0.35,
        neighborhood_add_prob: float = 0.80,
        neighborhood_walk_steps: int = 2,
        guide_samples: int = 0,
        guide_sigma: float = 0.0,
        guide_patch_prob: float = 0.0,
        guide_momentum: float = 0.0,
        bootstrap_guide_samples: int = 0,
        bootstrap_guide_sigma: float = 0.12,
        bootstrap_primitives: int = 10,
        eval_batch_size: int = 1,
        patch_noise_scale: float = 0.5,
        patch_noise_blend: float = 0.35,
        rescue_steps: int = 0,
        rescue_samples: int = 2,
        rescue_sigma: float = 0.10,
        min_primitives: int = 12,
        max_primitives: int = 32,
    ) -> None:
        alpha_v = alpha if alpha is not None else (eps / max(1, steps) * 2.5)
        super().__init__(eps=eps, alpha=alpha_v, steps=steps, random_start=random_start)
        self.p_init = float(p_init)
        self.max_patch_frac = float(max_patch_frac)
        self.spatial_size = None if spatial_size is None else int(spatial_size)
        self.temporal_tiles = max(1, int(temporal_tiles))
        self.query_budget = int(query_budget)
        self.random_start_trials = max(1, int(random_start_trials))
        self.coarse_to_fine = bool(coarse_to_fine)
        self.coarse_spatial_scale = float(coarse_spatial_scale)
        self.coarse_temporal_scale = float(coarse_temporal_scale)
        self.plateau_patience = max(0, int(plateau_patience))
        self.reject_alpha_decay = float(reject_alpha_decay)
        self.accept_alpha_growth = float(accept_alpha_growth)
        self.boost_top2 = float(boost_top2)
        self.boost_top10 = float(
            boost_top10 if boost_top5 is None else (boost_top10 if boost_top10 != 0.0 else boost_top5)
        )
        self.boost_rival = float(boost_rival)
        self.boost_conf = float(boost_conf)
        self.candidates_per_step = max(1, int(candidates_per_step))
        self.candidates_per_step_late = max(1, int(candidates_per_step_late))
        self.min_primitives = max(1, int(min_primitives))
        self.max_primitives = max(self.min_primitives, int(max_primitives))
        self.scout_queries = max(0, int(scout_queries))
        self.elite_window_count = max(0, int(elite_window_count))
        self.elite_candidates_per_step = max(0, int(elite_candidates_per_step))
        self.elite_jitter_frac = float(elite_jitter_frac)
        self.neighborhood_candidates_per_step = max(0, int(neighborhood_candidates_per_step))
        self.neighborhood_size_jitter = max(0.0, float(neighborhood_size_jitter))
        self.neighborhood_add_prob = float(neighborhood_add_prob)
        self.neighborhood_walk_steps = max(1, int(neighborhood_walk_steps))
        self.guide_samples = max(0, int(guide_samples))
        self.guide_sigma = float(guide_sigma)
        self.guide_patch_prob = float(guide_patch_prob)
        self.guide_momentum = float(guide_momentum)
        self.bootstrap_guide_samples = max(0, int(bootstrap_guide_samples))
        self.bootstrap_guide_sigma = float(bootstrap_guide_sigma)
        self.bootstrap_primitives = max(1, int(bootstrap_primitives))
        self.eval_batch_size = max(1, int(eval_batch_size))
        self.patch_noise_scale = float(patch_noise_scale)
        self.patch_noise_blend = float(patch_noise_blend)
        self.rescue_steps = max(0, int(rescue_steps))
        self.rescue_samples = max(1, int(rescue_samples))
        self.rescue_sigma = float(rescue_sigma)
        self.verbose = False

    def _dbg(self, msg: str) -> None:
        if self.verbose:
            print(f"[square] {msg}", flush=True)

    def prepare(self, *, model, dataset, indices, seed, targeted, pipeline_stage, verbose) -> None:
        del model, dataset, indices, seed, targeted, pipeline_stage
        self.verbose = bool(verbose)

    @staticmethod
    def _aggregate_logits(logits: torch.Tensor) -> torch.Tensor:
        x = logits.float()
        if x.ndim == 1:
            return x
        return x.reshape(-1, x.shape[-1]).mean(dim=0)

    @staticmethod
    def _tile_window(num_frames: int, num_tiles: int, tile_idx: int) -> tuple[int, int]:
        if num_tiles <= 1:
            return 0, num_frames
        start = int(math.floor(tile_idx * num_frames / num_tiles))
        stop = int(math.ceil((tile_idx + 1) * num_frames / num_tiles))
        return start, max(start + 1, min(num_frames, stop))

    @staticmethod
    def _secondary_label(probs: torch.Tensor, primary_label: int) -> tuple[int, float]:
        probs = probs.reshape(-1)
        if probs.numel() <= 1:
            return int(primary_label), float(probs[primary_label].item())
        top2 = torch.topk(probs, 2)
        first = int(top2.indices[0].item())
        second = int(top2.indices[1].item())
        label = second if first == int(primary_label) else first
        return label, float(probs[label].item())

    @staticmethod
    def _clone_primitives(primitives: list[SquarePrimitive]) -> list[SquarePrimitive]:
        return [primitive.clone() for primitive in primitives]

    def _random_sign(self, *, channels: int, device: torch.device) -> torch.Tensor:
        sign = torch.empty(channels, device=device)
        sign.bernoulli_(0.5)
        return sign.mul_(2.0).sub_(1.0)

    def _guided_sign(
        self,
        guide_field: torch.Tensor | None,
        *,
        t0: int,
        t1: int,
        h0: int,
        w0: int,
        size: int,
        channels: int,
        device: torch.device,
    ) -> torch.Tensor:
        if guide_field is None or guide_field.numel() == 0:
            return self._random_sign(channels=channels, device=device)
        patch = guide_field[t0:t1, :, h0 : h0 + size, w0 : w0 + size]
        if patch.numel() == 0:
            return self._random_sign(channels=channels, device=device)
        pooled = patch.mean(dim=(0, 2, 3))
        sign = torch.sign(pooled)
        zero_mask = sign.abs() < 1e-12
        if bool(zero_mask.any()):
            fallback = self._random_sign(channels=channels, device=device)
            sign = torch.where(zero_mask, fallback, sign)
        return sign.to(device=device)

    def _make_guidance_noise(
        self,
        *,
        num_tiles: int,
        grid_size: int,
        channels: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        small_n = max(4, grid_size // 2)
        small = torch.randn(num_tiles * channels, 1, small_n, small_n, device=device, dtype=dtype)
        up = F.interpolate(small, size=(grid_size, grid_size), mode="bilinear", align_corners=False)
        return up.reshape(num_tiles, channels, grid_size, grid_size).contiguous()

    def _make_patch_texture(
        self,
        *,
        size: int,
        channels: int,
        device: torch.device,
        dtype: torch.dtype,
        base_sign: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tex_size = max(1, int(size))
        if self.patch_noise_scale < 1.0:
            small_n = max(2, int(round(tex_size * max(0.15, self.patch_noise_scale))))
            small = torch.randn(channels, small_n, small_n, device=device, dtype=dtype)
            texture = F.interpolate(
                small.unsqueeze(0),
                size=(tex_size, tex_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            texture = torch.randn(channels, tex_size, tex_size, device=device, dtype=dtype)
        texture = texture / texture.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        if base_sign is not None:
            base = base_sign.to(device=device, dtype=dtype).view(channels, 1, 1).expand(-1, tex_size, tex_size)
            texture = ((1.0 - self.patch_noise_blend) * base) + (self.patch_noise_blend * texture)
            texture = texture / texture.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        return texture.contiguous()

    def _render_primitives(
        self,
        primitives: list[SquarePrimitive],
        *,
        num_tiles: int,
        channels: int,
        grid_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        latent = torch.zeros(num_tiles, channels, grid_size, grid_size, device=device, dtype=dtype)
        for primitive in primitives:
            t0 = max(0, min(num_tiles - 1, int(primitive.t0)))
            t1 = max(t0 + 1, min(num_tiles, int(primitive.t1)))
            size = max(1, min(grid_size, int(primitive.size)))
            max_h0 = max(0, grid_size - size)
            max_w0 = max(0, grid_size - size)
            h0 = max(0, min(max_h0, int(primitive.h0)))
            w0 = max(0, min(max_w0, int(primitive.w0)))
            if primitive.texture is not None:
                tex = primitive.texture.to(device=device, dtype=dtype)
                if tex.shape[-2:] != (size, size):
                    tex = F.interpolate(tex.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)
                step = tex.unsqueeze(0).mul(float(primitive.mag))
            else:
                step = primitive.sign.to(device=device, dtype=dtype).view(1, channels, 1, 1).mul(float(primitive.mag))
            latent[t0:t1, :, h0 : h0 + size, w0 : w0 + size] += step
        return torch.clamp(latent, min=-self.eps, max=self.eps)

    def _latent_to_video(
        self,
        latent: torch.Tensor,
        *,
        num_frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        num_tiles, channels, n_h, n_w = latent.shape
        full_t = torch.empty(num_frames, channels, n_h, n_w, device=latent.device, dtype=latent.dtype)
        for tile_idx in range(num_tiles):
            t0, t1 = self._tile_window(num_frames, num_tiles, tile_idx)
            full_t[t0:t1] = latent[tile_idx].unsqueeze(0).expand(t1 - t0, -1, -1, -1)
        if full_t.shape[-2:] != (height, width):
            full_t = F.interpolate(full_t, size=(height, width), mode="nearest")
        return full_t.permute(0, 2, 3, 1).contiguous()

    def _apply_latent(
        self,
        x_flat: torch.Tensor,
        latent: torch.Tensor,
        *,
        lower: torch.Tensor,
        upper: torch.Tensor,
        lo: float,
        hi: float,
    ) -> torch.Tensor:
        num_frames, height, width, _ = x_flat.shape
        adv = x_flat + self._latent_to_video(latent, num_frames=num_frames, height=height, width=width)
        adv = torch.clamp(adv, lo, hi)
        return torch.max(torch.min(adv, upper), lower)

    def _random_temporal_span(self, *, num_tiles: int) -> tuple[int, int]:
        if num_tiles <= 1:
            return 0, 1
        mode = str(np.random.choice(["all", "single", "adjacent", "random_span"], p=[0.15, 0.35, 0.25, 0.25]))
        if mode == "all":
            return 0, num_tiles
        if mode == "single":
            start = int(np.random.randint(0, num_tiles))
            return start, start + 1
        if mode == "adjacent":
            if num_tiles <= 2:
                return 0, num_tiles
            start = int(np.random.randint(0, num_tiles - 1))
            return start, min(num_tiles, start + 2)
        span = int(np.random.randint(1, num_tiles + 1))
        start = int(np.random.randint(0, num_tiles - span + 1))
        return start, start + span

    @staticmethod
    def _sample_center_biased_start(max_start: int) -> int:
        if max_start <= 0:
            return 0
        if np.random.rand() < 0.80:
            centerish = np.random.beta(2.8, 2.8)
            return int(round(centerish * max_start))
        return int(np.random.randint(0, max_start + 1))

    def _random_primitive(
        self,
        *,
        num_tiles: int,
        grid_size: int,
        channels: int,
        device: torch.device,
        patch_lo: int,
        patch_hi: int,
        mag_lo: float,
        mag_hi: float,
        guide_field: torch.Tensor | None = None,
        guide_patch_prob: float = 0.0,
    ) -> SquarePrimitive:
        size = max(1, min(grid_size, int(np.random.randint(max(1, patch_lo), max(patch_lo, patch_hi) + 1))))
        max_h0 = max(0, grid_size - size)
        max_w0 = max(0, grid_size - size)
        h0 = self._sample_center_biased_start(max_h0)
        w0 = self._sample_center_biased_start(max_w0)
        t0, t1 = self._random_temporal_span(num_tiles=num_tiles)
        mag = float(np.random.uniform(mag_lo, max(mag_lo, mag_hi)))
        use_guidance = bool(guide_field is not None and guide_patch_prob > 0.0 and np.random.rand() < guide_patch_prob)
        sign = (
            self._guided_sign(
                guide_field,
                t0=t0,
                t1=t1,
                h0=h0,
                w0=w0,
                size=size,
                channels=channels,
                device=device,
            )
            if use_guidance
            else self._random_sign(channels=channels, device=device)
        )
        return SquarePrimitive(
            t0=t0,
            t1=t1,
            h0=h0,
            w0=w0,
            size=size,
            sign=sign,
            mag=max(1e-3, min(float(self.eps), mag)),
        )

    def _random_start_primitives(
        self,
        *,
        num_tiles: int,
        grid_size: int,
        channels: int,
        device: torch.device,
        max_patch: int,
    ) -> list[SquarePrimitive]:
        patch_lo = max(1, max_patch // 3)
        patch_hi = max(patch_lo, max_patch)
        count_hi = min(self.max_primitives, max(self.min_primitives, num_tiles + 4))
        count = int(np.random.randint(self.min_primitives, count_hi + 1))
        return [
            self._random_primitive(
                num_tiles=num_tiles,
                grid_size=grid_size,
                channels=channels,
                device=device,
                patch_lo=patch_lo,
                patch_hi=patch_hi,
                mag_lo=0.55 * self.eps,
                mag_hi=self.eps,
            )
            for _ in range(count)
        ]

    def _guide_to_primitives(
        self,
        guide_field: torch.Tensor,
        *,
        num_tiles: int,
        grid_size: int,
        channels: int,
        device: torch.device,
        max_patch: int,
    ) -> list[SquarePrimitive]:
        count = max(self.min_primitives, min(self.max_primitives, self.bootstrap_primitives))
        patch_size = max(1, min(max_patch, max_patch if max_patch <= 2 else int(round(max_patch * 0.85))))
        radius = max(1, patch_size // 2)
        score = guide_field.abs().mean(dim=1)
        work = score.clone()
        primitives: list[SquarePrimitive] = []
        for _ in range(count):
            flat_idx = int(work.reshape(-1).argmax().item())
            if float(work.reshape(-1)[flat_idx].item()) <= 1e-12:
                break
            tile_idx = flat_idx // (grid_size * grid_size)
            rem = flat_idx % (grid_size * grid_size)
            center_h = rem // grid_size
            center_w = rem % grid_size
            h0 = max(0, min(grid_size - patch_size, int(center_h - (patch_size // 2))))
            w0 = max(0, min(grid_size - patch_size, int(center_w - (patch_size // 2))))
            temporal_strength = score[:, max(0, center_h - radius) : min(grid_size, center_h + radius + 1), max(0, center_w - radius) : min(grid_size, center_w + radius + 1)].mean(dim=(1, 2))
            active_tiles = torch.nonzero(temporal_strength >= (0.6 * temporal_strength.max()), as_tuple=False).reshape(-1)
            if active_tiles.numel() > 0:
                t0 = int(active_tiles.min().item())
                t1 = int(active_tiles.max().item()) + 1
            else:
                t0 = int(tile_idx)
                t1 = min(num_tiles, t0 + 1)
            primitives.append(
                SquarePrimitive(
                    t0=t0,
                    t1=max(t0 + 1, t1),
                    h0=h0,
                    w0=w0,
                    size=patch_size,
                    sign=self._guided_sign(
                        guide_field,
                        t0=t0,
                        t1=max(t0 + 1, t1),
                        h0=h0,
                        w0=w0,
                        size=patch_size,
                        channels=channels,
                        device=device,
                    ),
                    mag=float(self.eps),
                )
            )
            work[max(0, t0 - 1) : min(num_tiles, t1 + 1), max(0, center_h - radius) : min(grid_size, center_h + radius + 1), max(0, center_w - radius) : min(grid_size, center_w + radius + 1)] = 0
        return primitives

    def _window_score(
        self,
        primitive: SquarePrimitive,
        *,
        guide_field: torch.Tensor | None,
        num_tiles: int,
        grid_size: int,
    ) -> float:
        t0 = max(0, min(num_tiles - 1, int(primitive.t0)))
        t1 = max(t0 + 1, min(num_tiles, int(primitive.t1)))
        size = max(1, min(grid_size, int(primitive.size)))
        h0 = max(0, min(max(0, grid_size - size), int(primitive.h0)))
        w0 = max(0, min(max(0, grid_size - size), int(primitive.w0)))
        span = max(1, t1 - t0)
        base = float(abs(primitive.mag)) * float(size * size * span)
        if guide_field is None or guide_field.numel() == 0:
            return base
        patch = guide_field[t0:t1, :, h0 : h0 + size, w0 : w0 + size]
        if patch.numel() == 0:
            return base
        guide_score = float(patch.abs().mean().item())
        return base * (1.0 + guide_score)

    def _merge_elite_windows(
        self,
        elite_windows: list[EliteWindow],
        primitives: list[SquarePrimitive],
        *,
        guide_field: torch.Tensor | None,
        num_tiles: int,
        grid_size: int,
    ) -> list[EliteWindow]:
        if self.elite_window_count <= 0:
            return []
        merged = list(elite_windows)
        for primitive in primitives:
            merged.append(
                EliteWindow(
                    t0=int(primitive.t0),
                    t1=int(primitive.t1),
                    h0=int(primitive.h0),
                    w0=int(primitive.w0),
                    size=int(primitive.size),
                    sign=primitive.sign.clone(),
                    mag=float(primitive.mag),
                    score=self._window_score(
                        primitive,
                        guide_field=guide_field,
                        num_tiles=num_tiles,
                        grid_size=grid_size,
                    ),
                )
            )
        merged.sort(key=lambda window: float(window.score), reverse=True)
        pruned: list[EliteWindow] = []
        for window in merged:
            duplicate = False
            for kept in pruned:
                close_t = abs(window.t0 - kept.t0) <= 1 and abs(window.t1 - kept.t1) <= 1
                close_s = abs(window.size - kept.size) <= max(1, int(round(0.2 * max(window.size, kept.size))))
                close_xy = abs(window.h0 - kept.h0) <= max(1, window.size // 3) and abs(window.w0 - kept.w0) <= max(1, window.size // 3)
                if close_t and close_s and close_xy:
                    duplicate = True
                    break
            if duplicate:
                continue
            pruned.append(window)
            if len(pruned) >= self.elite_window_count:
                break
        return pruned

    def _elite_primitive(
        self,
        elite_windows: list[EliteWindow],
        *,
        num_tiles: int,
        grid_size: int,
        channels: int,
        device: torch.device,
        patch_lo: int,
        patch_hi: int,
        alpha_t: float,
        guide_field: torch.Tensor | None = None,
    ) -> SquarePrimitive | None:
        if not elite_windows:
            return None
        weights = np.asarray([max(1e-6, float(window.score)) for window in elite_windows], dtype=np.float64)
        weights = weights / weights.sum()
        anchor = elite_windows[int(np.random.choice(len(elite_windows), p=weights))]
        jitter = max(1, int(round(max(anchor.size, patch_hi) * max(0.05, self.elite_jitter_frac))))
        size = int(round(anchor.size * float(np.random.uniform(0.85, 1.20))))
        size = max(1, min(grid_size, max(patch_lo, min(patch_hi, size))))
        t_span = max(1, anchor.t1 - anchor.t0)
        new_t0 = anchor.t0 + int(np.random.randint(-1, 2))
        new_t1 = anchor.t1 + int(np.random.randint(-1, 2))
        if np.random.rand() < 0.35:
            delta_span = int(np.random.randint(-1, 2))
            t_span = max(1, min(num_tiles, t_span + delta_span))
            new_t1 = new_t0 + t_span
        new_t0 = max(0, min(num_tiles - 1, new_t0))
        new_t1 = max(new_t0 + 1, min(num_tiles, new_t1))
        h0 = max(0, min(max(0, grid_size - size), anchor.h0 + int(np.random.randint(-jitter, jitter + 1))))
        w0 = max(0, min(max(0, grid_size - size), anchor.w0 + int(np.random.randint(-jitter, jitter + 1))))
        if guide_field is not None and np.random.rand() < 0.7:
            sign = self._guided_sign(
                guide_field,
                t0=new_t0,
                t1=new_t1,
                h0=h0,
                w0=w0,
                size=size,
                channels=channels,
                device=device,
            )
        else:
            sign = anchor.sign.clone().to(device=device)
            if np.random.rand() < 0.25:
                flip_idx = int(np.random.randint(0, channels))
                sign[flip_idx] = -sign[flip_idx]
        mag = float(np.clip(anchor.mag + np.random.uniform(-alpha_t, alpha_t), max(alpha_t, 0.25 * self.eps), self.eps))
        return SquarePrimitive(
            t0=new_t0,
            t1=new_t1,
            h0=h0,
            w0=w0,
            size=size,
            sign=sign,
            mag=mag,
        )

    def _neighborhood_mutation(
        self,
        primitives: list[SquarePrimitive],
        *,
        num_tiles: int,
        grid_size: int,
        channels: int,
        device: torch.device,
        patch_lo: int,
        patch_hi: int,
        alpha_t: float,
        guide_field: torch.Tensor | None = None,
    ) -> tuple[list[SquarePrimitive], str]:
        if not primitives:
            return self._mutate_add(
                primitives,
                num_tiles=num_tiles,
                grid_size=grid_size,
                channels=channels,
                device=device,
                patch_lo=patch_lo,
                patch_hi=patch_hi,
                alpha_t=alpha_t,
                guide_field=guide_field,
                guide_patch_prob=max(self.guide_patch_prob, 0.5),
            ), "neighbor_seed"

        ranked = sorted(
            primitives,
            key=lambda primitive: self._window_score(
                primitive,
                guide_field=guide_field,
                num_tiles=num_tiles,
                grid_size=grid_size,
            ),
            reverse=True,
        )
        anchor = ranked[int(np.random.randint(0, min(len(ranked), max(1, 3))))]
        out = self._clone_primitives(primitives)
        ops: list[str] = []

        size_scale = float(np.random.uniform(1.0 - self.neighborhood_size_jitter, 1.0 + self.neighborhood_size_jitter))
        size = int(round(anchor.size * size_scale))
        size = max(1, min(grid_size, max(patch_lo, min(patch_hi, size))))
        max_coord = max(1, max(anchor.size, size))
        radius = max(1, int(round(max_coord * max(0.15, self.elite_jitter_frac))))
        center_h = anchor.h0 + (anchor.size // 2)
        center_w = anchor.w0 + (anchor.size // 2)
        h0 = max(0, min(max(0, grid_size - size), center_h - (size // 2) + int(np.random.randint(-radius, radius + 1))))
        w0 = max(0, min(max(0, grid_size - size), center_w - (size // 2) + int(np.random.randint(-radius, radius + 1))))

        t0 = int(anchor.t0)
        t1 = int(anchor.t1)
        for _ in range(self.neighborhood_walk_steps):
            if np.random.rand() < 0.6:
                shift = int(np.random.randint(-1, 2))
                span = max(1, t1 - t0)
                new_t0 = max(0, min(num_tiles - span, t0 + shift))
                t0, t1 = new_t0, min(num_tiles, new_t0 + span)
            if np.random.rand() < 0.35:
                span = max(1, t1 - t0)
                span = max(1, min(num_tiles, span + int(np.random.randint(-1, 2))))
                t0 = max(0, min(num_tiles - span, t0))
                t1 = min(num_tiles, t0 + span)
        t1 = max(t0 + 1, min(num_tiles, t1))

        if guide_field is not None and np.random.rand() < 0.75:
            sign = self._guided_sign(
                guide_field,
                t0=t0,
                t1=t1,
                h0=h0,
                w0=w0,
                size=size,
                channels=channels,
                device=device,
            )
        else:
            sign = anchor.sign.clone().to(device=device)
            if np.random.rand() < 0.35:
                sign[int(np.random.randint(0, channels))] *= -1.0
        mag = float(np.clip(anchor.mag + np.random.uniform(-alpha_t, alpha_t), max(alpha_t, 0.2 * self.eps), self.eps))

        neighbor = SquarePrimitive(
            t0=t0,
            t1=t1,
            h0=h0,
            w0=w0,
            size=size,
            sign=sign,
            mag=mag,
        )
        if len(out) < self.max_primitives and np.random.rand() < self.neighborhood_add_prob:
            out.append(neighbor)
            ops.append("neighbor_add")
        else:
            replace_pool = min(len(out), max(1, 3))
            replace_idx = int(np.random.randint(0, replace_pool))
            out[replace_idx] = neighbor
            ops.append("neighbor_replace")

        if out and np.random.rand() < 0.65:
            idx = int(np.random.randint(0, len(out)))
            primitive = out[idx]
            walk_scale = max(1, int(round(max(primitive.size, anchor.size) * 0.35)))
            primitive.h0 = max(0, min(max(0, grid_size - primitive.size), primitive.h0 + int(np.random.randint(-walk_scale, walk_scale + 1))))
            primitive.w0 = max(0, min(max(0, grid_size - primitive.size), primitive.w0 + int(np.random.randint(-walk_scale, walk_scale + 1))))
            if np.random.rand() < 0.35:
                primitive.size = max(1, min(grid_size, max(patch_lo, min(patch_hi, int(round(primitive.size * np.random.uniform(0.8, 1.25)))))))
                primitive.h0 = max(0, min(max(0, grid_size - primitive.size), primitive.h0))
                primitive.w0 = max(0, min(max(0, grid_size - primitive.size), primitive.w0))
            ops.append("neighbor_walk")

        return out, "+".join(ops)

    def _mutate_add(
        self,
        primitives: list[SquarePrimitive],
        *,
        num_tiles: int,
        grid_size: int,
        channels: int,
        device: torch.device,
        patch_lo: int,
        patch_hi: int,
        alpha_t: float,
        guide_field: torch.Tensor | None = None,
        guide_patch_prob: float = 0.0,
    ) -> list[SquarePrimitive]:
        out = self._clone_primitives(primitives)
        if len(out) >= self.max_primitives:
            return out
        out.append(
            self._random_primitive(
                num_tiles=num_tiles,
                grid_size=grid_size,
                channels=channels,
                device=device,
                patch_lo=patch_lo,
                patch_hi=patch_hi,
                mag_lo=max(alpha_t, 0.35 * self.eps),
                mag_hi=self.eps,
                guide_field=guide_field,
                guide_patch_prob=guide_patch_prob,
            )
        )
        return out

    def _mutate_remove(self, primitives: list[SquarePrimitive]) -> list[SquarePrimitive]:
        out = self._clone_primitives(primitives)
        if len(out) <= 1:
            return out
        del out[int(np.random.randint(0, len(out)))]
        return out

    def _mutate_move(
        self,
        primitives: list[SquarePrimitive],
        *,
        grid_size: int,
        base_patch: int,
    ) -> list[SquarePrimitive]:
        out = self._clone_primitives(primitives)
        if not out:
            return out
        idx = int(np.random.randint(0, len(out)))
        primitive = out[idx]
        max_h0 = max(0, grid_size - primitive.size)
        max_w0 = max(0, grid_size - primitive.size)
        if np.random.rand() < 0.30:
            primitive.h0 = 0 if max_h0 == 0 else int(np.random.randint(0, max_h0 + 1))
            primitive.w0 = 0 if max_w0 == 0 else int(np.random.randint(0, max_w0 + 1))
            return out
        jitter = max(1, min(grid_size, max(base_patch, primitive.size) // 2))
        steps = 1 if np.random.rand() < 0.5 else self.neighborhood_walk_steps
        for _ in range(steps):
            primitive.h0 = max(0, min(max_h0, primitive.h0 + int(np.random.randint(-jitter, jitter + 1))))
            primitive.w0 = max(0, min(max_w0, primitive.w0 + int(np.random.randint(-jitter, jitter + 1))))
        return out

    def _mutate_resize(
        self,
        primitives: list[SquarePrimitive],
        *,
        grid_size: int,
        patch_lo: int,
        patch_hi: int,
    ) -> list[SquarePrimitive]:
        out = self._clone_primitives(primitives)
        if not out:
            return out
        idx = int(np.random.randint(0, len(out)))
        primitive = out[idx]
        scale = float(np.random.uniform(0.7, 1.35))
        new_size = int(round(primitive.size * scale))
        primitive.size = max(1, min(grid_size, max(patch_lo, min(patch_hi, new_size))))
        primitive.h0 = max(0, min(max(0, grid_size - primitive.size), primitive.h0))
        primitive.w0 = max(0, min(max(0, grid_size - primitive.size), primitive.w0))
        return out

    def _mutate_temporal(
        self,
        primitives: list[SquarePrimitive],
        *,
        num_tiles: int,
    ) -> list[SquarePrimitive]:
        out = self._clone_primitives(primitives)
        if not out:
            return out
        idx = int(np.random.randint(0, len(out)))
        primitive = out[idx]
        mode = str(np.random.choice(["shift", "expand", "shrink", "resample"], p=[0.35, 0.25, 0.20, 0.20]))
        if mode == "resample":
            primitive.t0, primitive.t1 = self._random_temporal_span(num_tiles=num_tiles)
            return out
        span = max(1, primitive.t1 - primitive.t0)
        if mode == "shift":
            shift = int(np.random.randint(-max(1, span), max(1, span) + 1))
            new_t0 = max(0, min(num_tiles - span, primitive.t0 + shift))
            primitive.t0 = new_t0
            primitive.t1 = new_t0 + span
            return out
        if mode == "expand":
            grow = int(np.random.randint(1, max(2, span + 1)))
            primitive.t0 = max(0, primitive.t0 - grow // 2)
            primitive.t1 = min(num_tiles, primitive.t1 + (grow - grow // 2))
            if primitive.t1 <= primitive.t0:
                primitive.t1 = min(num_tiles, primitive.t0 + 1)
            return out
        if span <= 1:
            primitive.t0, primitive.t1 = self._random_temporal_span(num_tiles=num_tiles)
            return out
        shrink = int(np.random.randint(1, span + 1))
        new_span = max(1, span - shrink)
        if primitive.t0 + new_span > num_tiles:
            new_t0 = max(0, num_tiles - new_span)
        else:
            new_t0 = primitive.t0 + int(np.random.randint(0, span - new_span + 1))
        primitive.t0 = new_t0
        primitive.t1 = min(num_tiles, new_t0 + new_span)
        return out

    def _mutate_sign_or_mag(
        self,
        primitives: list[SquarePrimitive],
        *,
        channels: int,
        device: torch.device,
        dtype: torch.dtype,
        alpha_t: float,
        guide_field: torch.Tensor | None = None,
        guide_patch_prob: float = 0.0,
    ) -> list[SquarePrimitive]:
        out = self._clone_primitives(primitives)
        if not out:
            return out
        edit_count = min(len(out), int(np.random.choice([1, 2, 3], p=[0.20, 0.50, 0.30])))
        edit_indices = list(np.random.choice(len(out), size=edit_count, replace=False))
        step_scale = float(np.clip(2.0 * alpha_t / max(float(self.eps), 1e-6), 0.08, 0.60))
        for idx in edit_indices:
            primitive = out[idx]
            if np.random.rand() < 0.7:
                use_guidance = bool(guide_field is not None and guide_patch_prob > 0.0 and np.random.rand() < guide_patch_prob)
                if use_guidance:
                    new_sign = self._guided_sign(
                        guide_field,
                        t0=primitive.t0,
                        t1=primitive.t1,
                        h0=primitive.h0,
                        w0=primitive.w0,
                        size=primitive.size,
                        channels=channels,
                        device=device,
                    )
                    primitive.sign = new_sign
                    if primitive.texture is not None:
                        guided_texture = self._make_patch_texture(
                            size=primitive.size,
                            channels=channels,
                            device=device,
                            dtype=dtype,
                            base_sign=new_sign,
                        )
                        primitive.texture = primitive.texture.to(device=device, dtype=dtype) + (step_scale * guided_texture)
                        primitive.texture = primitive.texture / primitive.texture.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
                else:
                    mode = str(np.random.choice(["flip_all", "flip_one", "resample", "texture"], p=[0.15, 0.15, 0.15, 0.55]))
                    if mode == "flip_all":
                        primitive.sign = -primitive.sign
                        if primitive.texture is not None:
                            primitive.texture = -primitive.texture
                    elif mode == "flip_one":
                        ch = int(np.random.randint(0, channels))
                        primitive.sign[ch] = -primitive.sign[ch]
                        if primitive.texture is not None:
                            primitive.texture[ch] = -primitive.texture[ch]
                    elif mode == "texture":
                        base_sign = primitive.sign
                        noise_texture = self._make_patch_texture(
                            size=primitive.size,
                            channels=channels,
                            device=device,
                            dtype=dtype,
                            base_sign=base_sign,
                        )
                        if primitive.texture is None:
                            primitive.texture = step_scale * noise_texture
                        else:
                            primitive.texture = primitive.texture.to(device=device, dtype=dtype) + (step_scale * noise_texture)
                            primitive.texture = primitive.texture / primitive.texture.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
                    else:
                        primitive.sign = self._random_sign(channels=channels, device=device)
                        if primitive.texture is not None and np.random.rand() < 0.7:
                            resampled_texture = self._make_patch_texture(
                                size=primitive.size,
                                channels=channels,
                                device=device,
                                dtype=dtype,
                                base_sign=primitive.sign,
                            )
                            primitive.texture = primitive.texture.to(device=device, dtype=dtype) + (step_scale * resampled_texture)
                            primitive.texture = primitive.texture / primitive.texture.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            else:
                delta = float(np.random.uniform(-1.5 * alpha_t, 1.5 * alpha_t))
                primitive.mag = max(1e-3, min(float(self.eps), primitive.mag + delta))
        return out

    def _mutate_primitives(
        self,
        primitives: list[SquarePrimitive],
        *,
        num_tiles: int,
        grid_size: int,
        channels: int,
        device: torch.device,
        dtype: torch.dtype,
        patch_lo: int,
        patch_hi: int,
        base_patch: int,
        alpha_t: float,
        aggressive: bool = False,
        guide_field: torch.Tensor | None = None,
        guide_patch_prob: float = 0.0,
    ) -> tuple[list[SquarePrimitive], str]:
        if not primitives:
            return self._mutate_add(
                primitives,
                num_tiles=num_tiles,
                grid_size=grid_size,
                channels=channels,
                device=device,
                patch_lo=patch_lo,
                patch_hi=patch_hi,
                alpha_t=alpha_t,
                guide_field=guide_field,
                guide_patch_prob=guide_patch_prob,
            ), "add"

        out = self._clone_primitives(primitives)
        applied_ops: list[str] = []
        num_ops = int(np.random.choice([1, 2, 3], p=[0.20, 0.45, 0.35])) if aggressive else int(np.random.choice([1, 2], p=[0.55, 0.45]))
        step_mode = str(np.random.choice(["signmag", "move"], p=[0.85, 0.15] if aggressive else [0.75, 0.25]))
        for _ in range(num_ops):
            op = step_mode
            if step_mode == "move":
                out = self._mutate_move(out, grid_size=grid_size, base_patch=base_patch)
            else:
                out = self._mutate_sign_or_mag(
                    out,
                    channels=channels,
                    device=device,
                    dtype=dtype,
                    alpha_t=alpha_t,
                    guide_field=guide_field,
                    guide_patch_prob=guide_patch_prob,
                )
                op = "signmag"
            applied_ops.append(op)
        return out, "+".join(applied_ops)

    def _primitive_effective_texture(
        self,
        primitive: SquarePrimitive,
        *,
        channels: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if primitive.texture is not None:
            return primitive.texture.to(device=device, dtype=dtype)
        return primitive.sign.to(device=device, dtype=dtype).view(channels, 1, 1).expand(-1, primitive.size, primitive.size).contiguous()

    def _apply_proportional_step(
        self,
        base_primitives: list[SquarePrimitive],
        proposal_primitives: list[SquarePrimitive],
        *,
        channels: int,
        device: torch.device,
        dtype: torch.dtype,
        step_score: float,
        objective_ref: float,
    ) -> tuple[list[SquarePrimitive], float]:
        if len(base_primitives) != len(proposal_primitives):
            return self._clone_primitives(proposal_primitives), 1.0
        positive_step = step_score >= 0.0
        if positive_step:
            denom = max(1e-4, abs(objective_ref) * 0.0025)
            delta_scale = float(np.clip(np.tanh(step_score / denom), 0.0, 1.0))
            if delta_scale < 0.35:
                delta_scale = 0.35
            spatial_scale = 1.0
            noise_scale = 2.0
        else:
            denom = max(1e-4, abs(objective_ref) * 0.0100)
            delta_scale = float(np.clip(np.tanh(step_score / denom), -0.20, 0.0))
            if delta_scale > -0.08:
                delta_scale = -0.08
            spatial_scale = 0.35
            noise_scale = 0.50
        blended: list[SquarePrimitive] = []
        for base, proposal in zip(base_primitives, proposal_primitives):
            merged = base.clone()
            merged.h0 = int(round(base.h0 + (spatial_scale * delta_scale * (proposal.h0 - base.h0))))
            merged.w0 = int(round(base.w0 + (spatial_scale * delta_scale * (proposal.w0 - base.w0))))
            merged.t0 = int(round(base.t0 + (spatial_scale * delta_scale * (proposal.t0 - base.t0))))
            merged.t1 = int(round(base.t1 + (spatial_scale * delta_scale * (proposal.t1 - base.t1))))
            merged.h0 = max(0, merged.h0)
            merged.w0 = max(0, merged.w0)
            merged.t0 = max(0, merged.t0)
            merged.t1 = max(merged.t0 + 1, merged.t1)
            merged.mag = max(1e-3, min(float(self.eps), base.mag + (noise_scale * delta_scale * (proposal.mag - base.mag))))
            base_tex = self._primitive_effective_texture(base, channels=channels, device=device, dtype=dtype)
            prop_tex = self._primitive_effective_texture(proposal, channels=channels, device=device, dtype=dtype)
            if base_tex.shape[-2:] != (proposal.size, proposal.size):
                base_tex = F.interpolate(base_tex.unsqueeze(0), size=(proposal.size, proposal.size), mode="bilinear", align_corners=False).squeeze(0)
            merged_tex = base_tex + (noise_scale * delta_scale * (prop_tex - base_tex))
            merged_tex = merged_tex / merged_tex.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            merged.texture = merged_tex.contiguous()
            sign_source = torch.sign(merged_tex.mean(dim=(1, 2)))
            zero_mask = sign_source.abs() < 1e-12
            if bool(zero_mask.any()):
                fallback = base.sign.to(device=device, dtype=dtype)
                sign_source = torch.where(zero_mask, fallback, sign_source)
            merged.sign = sign_source.to(device=device)
            blended.append(merged)
        return blended, delta_scale

    def _scale_primitives(
        self,
        primitives: list[SquarePrimitive],
        *,
        src_tiles: int,
        dst_tiles: int,
        src_size: int,
        dst_size: int,
    ) -> list[SquarePrimitive]:
        if src_tiles == dst_tiles and src_size == dst_size:
            return self._clone_primitives(primitives)
        out: list[SquarePrimitive] = []
        tile_scale = float(dst_tiles) / float(max(1, src_tiles))
        spatial_scale = float(dst_size) / float(max(1, src_size))
        for primitive in primitives:
            size = max(1, min(dst_size, int(round(primitive.size * spatial_scale))))
            h0 = max(0, min(max(0, dst_size - size), int(round(primitive.h0 * spatial_scale))))
            w0 = max(0, min(max(0, dst_size - size), int(round(primitive.w0 * spatial_scale))))
            t0 = max(0, min(dst_tiles - 1, int(math.floor(primitive.t0 * tile_scale))))
            t1 = max(t0 + 1, min(dst_tiles, int(math.ceil(primitive.t1 * tile_scale))))
            out.append(
                SquarePrimitive(
                    t0=t0,
                    t1=t1,
                    h0=h0,
                    w0=w0,
                    size=size,
                    sign=primitive.sign.clone(),
                    mag=float(primitive.mag),
                )
            )
        return out

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
        del query_label, video_path
        attack_start = time.perf_counter()

        x_ref = self._ensure_float_attack_tensor(x.detach().clone())
        x_flat, restore_shape = flatten_temporal_video(x_ref)
        num_frames, height, width, channels = x_flat.shape
        device = x_flat.device
        safe_cuda_reset_peak_memory_stats(device)
        lo = 0.0
        hi = 255.0 if float(x_flat.max().item()) > 1.5 else 1.0
        eps = float(self.eps)

        max_n = min(height, width) if self.spatial_size is None else min(int(self.spatial_size), height, width)
        max_tiles = min(self.temporal_tiles, num_frames)
        if self.coarse_to_fine:
            start_n = max(4, min(max_n, int(round(max_n * self.coarse_spatial_scale))))
            start_tiles = max(1, min(max_tiles, int(round(max_tiles * self.coarse_temporal_scale))))
        else:
            start_n = max_n
            start_tiles = max_tiles

        lower = torch.clamp(x_flat - eps, lo, hi)
        upper = torch.clamp(x_flat + eps, lo, hi)
        query_count = 0

        def query_bundle(flat: torch.Tensor) -> PredictionBundle:
            nonlocal query_count
            query_count += 1
            restored = restore_temporal_video(flat, restore_shape)
            with torch.no_grad():
                bundle = model(restored, input_format=input_format, enable_grad=False, return_full=True)
            if not isinstance(bundle, PredictionBundle):
                raise TypeError(f"Expected PredictionBundle, got {type(bundle)}")
            return bundle

        def query_probs_batch(flats: list[torch.Tensor]) -> torch.Tensor:
            nonlocal query_count
            if not flats:
                return torch.empty(0, device=device, dtype=x_flat.dtype)
            outputs: list[torch.Tensor] = []
            for start in range(0, len(flats), self.eval_batch_size):
                chunk = flats[start : start + self.eval_batch_size]
                query_count += len(chunk)
                restored_batch = None
                probs_batch = None
                try:
                    restored_batch = torch.stack([restore_temporal_video(flat, restore_shape) for flat in chunk], dim=0)
                    with torch.no_grad():
                        probs_batch = model(restored_batch, input_format=input_format, enable_grad=False, return_full=False)
                except torch.OutOfMemoryError:
                    safe_cuda_empty_cache()
                    raise
                if isinstance(probs_batch, PredictionBundle):
                    raise TypeError("Expected batched probabilities, got PredictionBundle")
                outputs.append(probs_batch)
                del restored_batch, probs_batch
                if device.type == "cuda":
                    safe_cuda_empty_cache()
            return torch.cat(outputs, dim=0)

        benign_bundle = query_bundle(x_flat)
        benign_probs = benign_bundle.probs.reshape(-1)
        benign_label = int(benign_probs.argmax().item())
        second_label_init, _ = self._secondary_label(benign_probs, benign_label)
        rival_label = second_label_init

        if targeted:
            if y is None:
                target_label = int(benign_probs.argmin().item())
            elif isinstance(y, int):
                target_label = y
            else:
                target_label = int(y.item())
        else:
            target_label = benign_label

        def objective_from_probs(probs: torch.Tensor) -> float:
            logp = torch.log(probs.reshape(-1).float().clamp_min(1e-12))
            if targeted:
                main = logp[target_label]
                others = logp.clone()
                others[target_label] = float("-inf")
                return float((main - others.max()).item())
            main = logp[benign_label]
            others = logp.clone()
            others[benign_label] = float("-inf")
            rival_term = 0.0
            if 0 <= rival_label < logp.numel() and rival_label != benign_label:
                rival_term = self.boost_rival * float((logp[rival_label] - main).item())
            return float((others.max() - main).item() + rival_term)

        def objective(bundle: PredictionBundle) -> float:
            return objective_from_probs(bundle.probs)

        def topk_support_from_probs(probs: torch.Tensor, k: int) -> float:
            logp = torch.log(probs.reshape(-1).float().clamp_min(1e-12))
            pivot = target_label if targeted else benign_label
            others = logp.clone()
            others[pivot] = float("-inf")
            if others.numel() <= 1:
                return 0.0
            count = min(int(k), others.numel() - 1)
            if count <= 0:
                return 0.0
            return float(torch.topk(others, count).values.mean().item())

        def topk_support(bundle: PredictionBundle, k: int) -> float:
            return topk_support_from_probs(bundle.probs, k)

        def success(bundle: PredictionBundle) -> bool:
            pred = int(bundle.pred_label)
            return pred == target_label if targeted else pred != benign_label

        def success_from_probs(probs: torch.Tensor) -> bool:
            pred = int(probs.reshape(-1).argmax().item())
            return pred == target_label if targeted else pred != benign_label

        def benign_conf(probs: torch.Tensor) -> float:
            return float(probs[target_label].item()) if targeted else float(probs[benign_label].item())

        def dense_rescue(
            *,
            base_latent: torch.Tensor,
            base_objective: float,
            base_top2: float,
            base_top10: float,
            base_conf: float,
            num_tiles: int,
            grid_size: int,
        ) -> dict[str, object] | None:
            if self.rescue_steps <= 0 or self.rescue_samples <= 0 or self.rescue_sigma <= 0.0:
                return None
            if query_count >= self.query_budget:
                return None
            rescue_latent = base_latent.clone()
            rescue_velocity = torch.zeros_like(rescue_latent)
            best_rescue: dict[str, object] | None = None
            sigma_t = max(1e-3, self.rescue_sigma * max(1.0, eps))
            for _ in range(self.rescue_steps):
                if query_count >= self.query_budget:
                    break
                grad = torch.zeros_like(rescue_latent)
                actual_pairs = 0
                for _ in range(self.rescue_samples):
                    if query_count + 2 > self.query_budget:
                        break
                    noise = self._make_guidance_noise(
                        num_tiles=num_tiles,
                        grid_size=grid_size,
                        channels=channels,
                        device=device,
                        dtype=x_flat.dtype,
                    )
                    plus_latent = torch.clamp(rescue_latent + (sigma_t * noise), min=-eps, max=eps)
                    minus_latent = torch.clamp(rescue_latent - (sigma_t * noise), min=-eps, max=eps)
                    pair_probs = query_probs_batch(
                        [
                            self._apply_latent(x_flat, plus_latent, lower=lower, upper=upper, lo=lo, hi=hi),
                            self._apply_latent(x_flat, minus_latent, lower=lower, upper=upper, lo=lo, hi=hi),
                        ]
                    )
                    grad += (objective_from_probs(pair_probs[0]) - objective_from_probs(pair_probs[1])) * noise
                    actual_pairs += 1
                if actual_pairs == 0:
                    break
                grad = grad / (2.0 * actual_pairs * sigma_t)
                rescue_velocity = 0.8 * rescue_velocity + 0.2 * grad
                direction = torch.sign(rescue_velocity)
                if direction.abs().max() < 1e-12:
                    continue
                candidate_latent = torch.clamp(rescue_latent + (max(alpha_t, 1.0) * direction), min=-eps, max=eps)
                candidate_flat = self._apply_latent(x_flat, candidate_latent, lower=lower, upper=upper, lo=lo, hi=hi)
                candidate_bundle = query_bundle(candidate_flat)
                candidate_probs = candidate_bundle.probs.reshape(-1)
                candidate_objective = objective(candidate_bundle)
                candidate_top2 = topk_support(candidate_bundle, 2)
                candidate_top10 = topk_support(candidate_bundle, 10)
                candidate_conf = benign_conf(candidate_probs)
                if (
                    best_rescue is None
                    or candidate_objective > float(best_rescue["objective"])
                    or success(candidate_bundle)
                ):
                    best_rescue = {
                        "latent": candidate_latent,
                        "bundle": candidate_bundle,
                        "objective": candidate_objective,
                        "top2": candidate_top2,
                        "top10": candidate_top10,
                        "benign_conf": candidate_conf,
                        "success": success(candidate_bundle),
                        "guide": rescue_velocity.clone(),
                    }
                rescue_latent = candidate_latent
                if success(candidate_bundle):
                    break
            if best_rescue is None:
                return None
            improved = bool(
                float(best_rescue["objective"]) > base_objective + 1e-6
                or float(best_rescue["benign_conf"]) < base_conf - 1e-6
                or bool(best_rescue["success"])
            )
            if not improved:
                return None
            rescue_primitives = self._guide_to_primitives(
                best_rescue["guide"],  # type: ignore[arg-type]
                num_tiles=num_tiles,
                grid_size=grid_size,
                channels=channels,
                device=device,
                max_patch=max(1, min(grid_size, int(round(self.max_patch_frac * grid_size)))),
            )
            return {
                "primitives": rescue_primitives,
                "latent": best_rescue["latent"],
                "bundle": best_rescue["bundle"],
                "objective": best_rescue["objective"],
                "top2": best_rescue["top2"],
                "top10": best_rescue["top10"],
                "benign_conf": best_rescue["benign_conf"],
                "success": best_rescue["success"],
            }

        cur_n = start_n
        cur_tiles = start_tiles
        max_patch = max(1, min(cur_n, int(round(self.max_patch_frac * cur_n))))
        cur_primitives: list[SquarePrimitive] = []
        cur_bundle = benign_bundle
        elite_windows: list[EliteWindow] = []

        selected_start_trial = 0
        attempted_start_trials = 0
        if self.random_start and query_count < self.query_budget:
            max_trials = min(self.random_start_trials, max(self.query_budget - query_count, 0))
            attempted_start_trials = max_trials
            best_start_primitives: list[SquarePrimitive] | None = None
            best_start_bundle: PredictionBundle | None = None
            best_start_objective = float("-inf")
            if self.bootstrap_guide_samples > 0 and self.bootstrap_guide_sigma > 0.0 and query_count + 2 <= self.query_budget:
                guide_sigma = max(self.bootstrap_guide_sigma * max(1.0, eps), 1e-3)
                guide_grad = torch.zeros(cur_tiles, channels, cur_n, cur_n, device=device, dtype=x_flat.dtype)
                actual_pairs = 0
                for _ in range(self.bootstrap_guide_samples):
                    if query_count + 2 > self.query_budget:
                        break
                    noise = self._make_guidance_noise(
                        num_tiles=cur_tiles,
                        grid_size=cur_n,
                        channels=channels,
                        device=device,
                        dtype=x_flat.dtype,
                    )
                    plus_latent = torch.clamp(guide_sigma * noise, min=-eps, max=eps)
                    minus_latent = torch.clamp(-guide_sigma * noise, min=-eps, max=eps)
                    plus_flat = self._apply_latent(x_flat, plus_latent, lower=lower, upper=upper, lo=lo, hi=hi)
                    minus_flat = self._apply_latent(x_flat, minus_latent, lower=lower, upper=upper, lo=lo, hi=hi)
                    pair_probs = query_probs_batch([plus_flat, minus_flat])
                    guide_grad += (objective_from_probs(pair_probs[0]) - objective_from_probs(pair_probs[1])) * noise
                    actual_pairs += 1
                if actual_pairs > 0:
                    guide_grad = guide_grad / (2.0 * actual_pairs * guide_sigma)
                    dense_direction = torch.sign(guide_grad)
                    if dense_direction.abs().max() > 0:
                        dense_latent = torch.clamp(eps * dense_direction, min=-eps, max=eps)
                        dense_flat = self._apply_latent(x_flat, dense_latent, lower=lower, upper=upper, lo=lo, hi=hi)
                        dense_bundle = query_bundle(dense_flat)
                        if success(dense_bundle):
                            dense_probs = dense_bundle.probs.reshape(-1)
                            second_label, second_conf = self._secondary_label(dense_probs, benign_label)
                            attack_time_sec = float(time.perf_counter() - attack_start)
                            self.last_result = {
                                "iter_count": 0,
                                "query_count": query_count,
                                "target_label": target_label if targeted else -1,
                                "benign_label": benign_label,
                                "final_label": int(dense_bundle.pred_label),
                                "benign_conf": float(dense_probs[benign_label].item()),
                                "second_label": second_label,
                                "second_conf": second_conf,
                                "effective_eps": eps,
                                "attack_time_sec": attack_time_sec,
                            }
                            self._dbg(
                                f"done iters=0 queries={query_count} benign={benign_label} "
                                f"final={int(dense_bundle.pred_label)} found=True time={attack_time_sec:.2f}s [bootstrap-dense]"
                            )
                            return restore_temporal_video(dense_flat, restore_shape).detach()
                    bootstrap_primitives = self._guide_to_primitives(
                        guide_grad,
                        num_tiles=cur_tiles,
                        grid_size=cur_n,
                        channels=channels,
                        device=device,
                        max_patch=max_patch,
                    )
                    if bootstrap_primitives:
                        bootstrap_latent = self._render_primitives(
                            bootstrap_primitives,
                            num_tiles=cur_tiles,
                            channels=channels,
                            grid_size=cur_n,
                            device=device,
                            dtype=x_flat.dtype,
                        )
                        bootstrap_flat = self._apply_latent(x_flat, bootstrap_latent, lower=lower, upper=upper, lo=lo, hi=hi)
                        bootstrap_bundle = query_bundle(bootstrap_flat)
                        bootstrap_objective = objective(bootstrap_bundle)
                        best_start_primitives = self._clone_primitives(bootstrap_primitives)
                        best_start_bundle = bootstrap_bundle
                        best_start_objective = bootstrap_objective
                        elite_windows = self._merge_elite_windows(
                            elite_windows,
                            bootstrap_primitives,
                            guide_field=guide_grad,
                            num_tiles=cur_tiles,
                            grid_size=cur_n,
                        )
                        selected_start_trial = -1
                        if success(bootstrap_bundle):
                            cur_primitives = best_start_primitives
                            cur_bundle = best_start_bundle
                            max_trials = 0
            trial_primitives_batch: list[list[SquarePrimitive]] = []
            trial_flats: list[torch.Tensor] = []
            for _ in range(max_trials):
                trial_primitives = self._random_start_primitives(
                    num_tiles=cur_tiles,
                    grid_size=cur_n,
                    channels=channels,
                    device=device,
                    max_patch=max_patch,
                )
                trial_latent = self._render_primitives(
                    trial_primitives,
                    num_tiles=cur_tiles,
                    channels=channels,
                    grid_size=cur_n,
                    device=device,
                    dtype=x_flat.dtype,
                )
                trial_primitives_batch.append(trial_primitives)
                trial_flats.append(self._apply_latent(x_flat, trial_latent, lower=lower, upper=upper, lo=lo, hi=hi))
            if trial_flats:
                trial_probs_batch = query_probs_batch(trial_flats)
                for trial_idx, (trial_primitives, trial_probs) in enumerate(zip(trial_primitives_batch, trial_probs_batch), start=1):
                    trial_probs = trial_probs.reshape(-1).detach()
                    trial_objective = objective_from_probs(trial_probs)
                    trial_bundle = PredictionBundle(
                        logits=torch.log(trial_probs.float().clamp_min(1e-12)),
                        probs=trial_probs,
                        pred_label=int(trial_probs.argmax().item()),
                    )
                    if best_start_bundle is None or trial_objective > best_start_objective:
                        best_start_primitives = self._clone_primitives(trial_primitives)
                        best_start_bundle = trial_bundle
                        best_start_objective = trial_objective
                        elite_windows = self._merge_elite_windows(
                            elite_windows,
                            trial_primitives,
                            guide_field=None,
                            num_tiles=cur_tiles,
                            grid_size=cur_n,
                        )
                        selected_start_trial = trial_idx
                    if success_from_probs(trial_probs):
                        best_start_primitives = self._clone_primitives(trial_primitives)
                        best_start_bundle = trial_bundle
                        best_start_objective = trial_objective
                        elite_windows = self._merge_elite_windows(
                            elite_windows,
                            trial_primitives,
                            guide_field=None,
                            num_tiles=cur_tiles,
                            grid_size=cur_n,
                        )
                        selected_start_trial = trial_idx
                        break
            if best_start_primitives is not None and best_start_bundle is not None:
                cur_primitives = best_start_primitives
                cur_bundle = best_start_bundle

        scout_base_objective = objective(cur_bundle)
        if (
            self.scout_queries > 0
            and query_count < self.query_budget
            and cur_n > 0
            and cur_tiles > 0
        ):
            scout_count = min(self.scout_queries, max(self.query_budget - query_count, 0))
            scout_trials: list[dict[str, object]] = []
            scout_flats: list[torch.Tensor] = []
            for _ in range(scout_count):
                scout_primitive = self._random_primitive(
                    num_tiles=cur_tiles,
                    grid_size=cur_n,
                    channels=channels,
                    device=device,
                    patch_lo=max(1, max_patch // 2),
                    patch_hi=max(1, max_patch),
                    mag_lo=max(1.0, 0.5 * eps),
                    mag_hi=eps,
                )
                scout_primitives = self._clone_primitives(cur_primitives)
                if scout_primitives and np.random.rand() < 0.5:
                    replace_idx = int(np.random.randint(0, len(scout_primitives)))
                    scout_primitives[replace_idx] = scout_primitive
                elif len(scout_primitives) < self.max_primitives:
                    scout_primitives.append(scout_primitive)
                else:
                    scout_primitives[int(np.random.randint(0, len(scout_primitives)))] = scout_primitive
                scout_latent = self._render_primitives(
                    scout_primitives,
                    num_tiles=cur_tiles,
                    channels=channels,
                    grid_size=cur_n,
                    device=device,
                    dtype=x_flat.dtype,
                )
                scout_trials.append({"primitives": scout_primitives, "latent": scout_latent})
                scout_flats.append(self._apply_latent(x_flat, scout_latent, lower=lower, upper=upper, lo=lo, hi=hi))
            if scout_flats:
                scout_probs_batch = query_probs_batch(scout_flats)
                for scout_trial, scout_probs in zip(scout_trials, scout_probs_batch):
                    scout_probs = scout_probs.reshape(-1).detach()
                    scout_bundle = PredictionBundle(
                        logits=torch.log(scout_probs.float().clamp_min(1e-12)),
                        probs=scout_probs,
                        pred_label=int(scout_probs.argmax().item()),
                    )
                    scout_objective = objective_from_probs(scout_probs)
                    elite_windows = self._merge_elite_windows(
                        elite_windows,
                        scout_trial["primitives"],  # type: ignore[arg-type]
                        guide_field=None,
                        num_tiles=cur_tiles,
                        grid_size=cur_n,
                    )
                    if scout_objective > scout_base_objective + 1e-6:
                        cur_primitives = self._clone_primitives(scout_trial["primitives"])  # type: ignore[arg-type]
                        cur_latent = scout_trial["latent"]  # type: ignore[assignment]
                        cur_bundle = scout_bundle
                        scout_base_objective = scout_objective
                    if success(scout_bundle):
                        cur_primitives = self._clone_primitives(scout_trial["primitives"])  # type: ignore[arg-type]
                        cur_latent = scout_trial["latent"]  # type: ignore[assignment]
                        cur_bundle = scout_bundle
                        scout_base_objective = scout_objective
                        break
                del scout_flats, scout_probs_batch, scout_trials
                if device.type == "cuda":
                    safe_cuda_empty_cache()

        cur_latent = self._render_primitives(
            cur_primitives,
            num_tiles=cur_tiles,
            channels=channels,
            grid_size=cur_n,
            device=device,
            dtype=x_flat.dtype,
        )
        clean_objective = objective(benign_bundle)
        cur_objective = objective(cur_bundle)
        cur_probs = cur_bundle.probs.reshape(-1)
        cur_top2 = topk_support(cur_bundle, 2)
        cur_top10 = topk_support(cur_bundle, 10)
        cur_benign_conf = benign_conf(cur_probs)
        best_benign_conf = float(benign_probs[benign_label].item())
        elite_windows = self._merge_elite_windows(
            elite_windows,
            cur_primitives,
            guide_field=None,
            num_tiles=cur_tiles,
            grid_size=cur_n,
        )

        best_primitives = []
        best_bundle = benign_bundle
        best_objective = clean_objective
        best_n = cur_n
        best_tiles = cur_tiles
        if cur_objective > best_objective or success(cur_bundle):
            best_primitives = self._clone_primitives(cur_primitives)
            best_bundle = cur_bundle
            best_objective = cur_objective
            best_benign_conf = cur_benign_conf
            best_n = cur_n
            best_tiles = cur_tiles

        alpha_t = float(self.alpha)
        alpha_min = max(1e-3, float(self.alpha) * 0.1)
        alpha_max = max(alpha_min, float(self.alpha) * 3.0)
        steps_no_improve = 0
        found = bool(self.random_start and success(cur_bundle))
        iters_done = 0
        guide_field_state = torch.zeros_like(cur_latent) if self.guide_samples > 0 and self.guide_sigma > 0.0 else None

        self._dbg(
            f"start shape={tuple(x_flat.shape)} eps={eps:.1f} alpha={self.alpha:.2f} "
            f"N={cur_n}->{max_n} T_tiles={cur_tiles}->{max_tiles} steps={self.steps} "
            f"query_budget={self.query_budget} p_init={self.p_init:.4f} "
            f"candidates={self.candidates_per_step}/{self.candidates_per_step_late} "
            f"neighborhood={self.neighborhood_candidates_per_step} "
            f"random_start={self.random_start} random_start_trials={self.random_start_trials} "
            f"boost_top2={self.boost_top2} boost_top10={self.boost_top10} boost_conf={self.boost_conf} "
            f"bootstrap_guide_samples={self.bootstrap_guide_samples} bootstrap_guide_sigma={self.bootstrap_guide_sigma:.4f} "
            f"guide_samples={self.guide_samples} guide_sigma={self.guide_sigma:.4f} "
            f"guide_patch_prob={self.guide_patch_prob:.2f} guide_momentum={self.guide_momentum:.2f} "
            f"benign={benign_label} second_init={second_label_init}"
        )
        if self.random_start:
            init_delta_linf = float(cur_latent.abs().max().item())
            self._dbg(
                f"init random-start objective={cur_objective:.4f} benign_conf={cur_benign_conf:.4f} "
                f"trial={selected_start_trial}/{max(1, attempted_start_trials)} "
                f"primitives={len(cur_primitives)} delta_linf={init_delta_linf:.4f}/{eps:.4f} "
                f"elapsed={time.perf_counter() - attack_start:.2f}s"
            )

        if found:
            best_latent = self._render_primitives(
                best_primitives,
                num_tiles=best_tiles,
                channels=channels,
                grid_size=best_n,
                device=device,
                dtype=x_flat.dtype,
            )
            best_flat = self._apply_latent(x_flat, best_latent, lower=lower, upper=upper, lo=lo, hi=hi)
            second_label, second_conf = self._secondary_label(best_bundle.probs.reshape(-1), benign_label)
            attack_time_sec = float(time.perf_counter() - attack_start)
            self.last_result = {
                "iter_count": 0,
                "query_count": query_count,
                "target_label": target_label if targeted else -1,
                "benign_label": benign_label,
                "final_label": int(best_bundle.pred_label),
                "benign_conf": float(best_bundle.probs.reshape(-1)[benign_label].item()),
                "second_label": second_label,
                "second_conf": second_conf,
                "effective_eps": eps,
                "attack_time_sec": attack_time_sec,
            }
            self._dbg(
                f"done iters=0 queries={query_count} benign={benign_label} "
                f"final={int(best_bundle.pred_label)} found=True time={attack_time_sec:.2f}s"
            )
            return restore_temporal_video(best_flat, restore_shape).detach()

        for step_i in range(self.steps):
            iters_done = step_i + 1
            if query_count >= self.query_budget:
                break

            p = _p_selection(self.p_init, step_i, self.steps)
            max_patch = max(1, min(cur_n, int(round(self.max_patch_frac * cur_n))))
            base_patch = max(1, int(round(math.sqrt(p * cur_n * cur_n))))
            base_patch = min(base_patch, max_patch)
            patch_lo = max(1, min(max_patch, max(1, int(round(base_patch * 0.6)))))
            patch_hi = max(patch_lo, min(max_patch, max(base_patch, int(round(base_patch * 1.75)))))
            aggressive_push = bool((not targeted) and cur_benign_conf < 0.45)
            if aggressive_push:
                patch_lo = max(patch_lo, int(round(0.75 * max_patch)))
                patch_hi = max(patch_hi, max_patch)

            remaining_budget = max(self.query_budget - query_count, 0)
            planned_candidates = self.candidates_per_step if step_i < self.steps // 2 else self.candidates_per_step_late
            if steps_no_improve > 0:
                planned_candidates += 1
            if aggressive_push:
                planned_candidates += 1
            num_candidates = max(0, min(planned_candidates, remaining_budget))
            if aggressive_push:
                num_candidates = max(num_candidates, min(2, remaining_budget))
            if num_candidates <= 0:
                break
            elite_candidate_count = 0
            neighborhood_candidate_count = 0
            random_candidate_count = num_candidates

            guide_field = None
            if (
                guide_field_state is not None
                and self.guide_samples > 0
                and self.guide_sigma > 0.0
                and query_count < self.query_budget
            ):
                guide_sigma_t = max(alpha_t, self.guide_sigma * max(1.0, eps))
                remaining_after_candidates = max(self.query_budget - query_count - num_candidates, 0)
                guide_pairs = min(self.guide_samples, remaining_after_candidates // 2)
                if guide_pairs > 0:
                    guide_grad = torch.zeros_like(cur_latent)
                    actual_pairs = 0
                    for _ in range(guide_pairs):
                        if query_count + 2 + num_candidates > self.query_budget:
                            break
                        noise = self._make_guidance_noise(
                            num_tiles=cur_tiles,
                            grid_size=cur_n,
                            channels=channels,
                            device=device,
                            dtype=x_flat.dtype,
                        )
                        plus_latent = torch.clamp(cur_latent + (guide_sigma_t * noise), min=-eps, max=eps)
                        minus_latent = torch.clamp(cur_latent - (guide_sigma_t * noise), min=-eps, max=eps)
                        plus_flat = self._apply_latent(x_flat, plus_latent, lower=lower, upper=upper, lo=lo, hi=hi)
                        minus_flat = self._apply_latent(x_flat, minus_latent, lower=lower, upper=upper, lo=lo, hi=hi)
                        pair_probs = query_probs_batch([plus_flat, minus_flat])
                        guide_grad += (objective_from_probs(pair_probs[0]) - objective_from_probs(pair_probs[1])) * noise
                        actual_pairs += 1
                    if actual_pairs > 0:
                        guide_grad = guide_grad / (2.0 * actual_pairs * guide_sigma_t)
                        guide_field_state = (
                            (self.guide_momentum * guide_field_state) + ((1.0 - self.guide_momentum) * guide_grad)
                            if self.guide_momentum > 0.0
                            else guide_grad
                        )
                        guide_field = guide_field_state

            best_trial: dict[str, object] | None = None
            candidate_trials: list[dict[str, object]] = []
            candidate_flats: list[torch.Tensor] = []
            for _ in range(num_candidates):
                candidate_primitives: list[SquarePrimitive]
                op_name: str
                candidate_primitives, op_name = self._mutate_primitives(
                    cur_primitives,
                    num_tiles=cur_tiles,
                    grid_size=cur_n,
                    channels=channels,
                    device=device,
                    dtype=x_flat.dtype,
                    patch_lo=patch_lo,
                    patch_hi=patch_hi,
                    base_patch=base_patch,
                    alpha_t=alpha_t,
                    aggressive=aggressive_push,
                    guide_field=guide_field,
                    guide_patch_prob=self.guide_patch_prob,
                )
                candidate_latent = self._render_primitives(
                    candidate_primitives,
                    num_tiles=cur_tiles,
                    channels=channels,
                    grid_size=cur_n,
                    device=device,
                    dtype=x_flat.dtype,
                )
                candidate_flat = self._apply_latent(x_flat, candidate_latent, lower=lower, upper=upper, lo=lo, hi=hi)
                candidate_trials.append(
                    {
                        "primitives": candidate_primitives,
                        "latent": candidate_latent,
                        "op": op_name,
                    }
                )
                candidate_flats.append(candidate_flat)
            if candidate_flats:
                candidate_probs_batch = query_probs_batch(candidate_flats)
                for trial, candidate_probs in zip(candidate_trials, candidate_probs_batch):
                    candidate_probs = candidate_probs.reshape(-1).detach()
                    candidate_objective = objective_from_probs(candidate_probs)
                    candidate_top2 = topk_support_from_probs(candidate_probs, 2)
                    candidate_top10 = topk_support_from_probs(candidate_probs, 10)
                    candidate_benign_conf = benign_conf(candidate_probs)
                    candidate_bundle = PredictionBundle(
                        logits=torch.log(candidate_probs.float().clamp_min(1e-12)),
                        probs=candidate_probs,
                        pred_label=int(candidate_probs.argmax().item()),
                    )
                    trial["bundle"] = candidate_bundle
                    trial["objective"] = candidate_objective
                    trial["top2"] = candidate_top2
                    trial["top10"] = candidate_top10
                    trial["benign_conf"] = candidate_benign_conf
                    progress = float(step_i) / float(max(1, self.steps - 1))
                    boost_top2_t = self.boost_top2 * (1.0 + 3.0 * progress)
                    margin_gain = candidate_objective - cur_objective
                    top2_gain = (cur_top2 - candidate_top2) if targeted else (candidate_top2 - cur_top2)
                    top10_gain = (cur_top10 - candidate_top10) if targeted else (candidate_top10 - cur_top10)
                    conf_gain = (candidate_benign_conf - cur_benign_conf) if targeted else (cur_benign_conf - candidate_benign_conf)
                    accept_score = margin_gain + (boost_top2_t * top2_gain) + (self.boost_top10 * top10_gain) + (self.boost_conf * conf_gain)
                    accept_margin_floor = -0.01 if step_i < self.steps // 3 else 0.0
                    pure_noise_step = bool(trial["op"]) and all(part == "signmag" for part in str(trial["op"]).split("+"))
                    soft_accept = bool(
                        accept_score > 1e-12
                        and conf_gain > 1e-5
                        and margin_gain > accept_margin_floor
                    )
                    if pure_noise_step:
                        soft_accept = bool(accept_score > 1e-12)
                    trial["accept_score"] = accept_score
                    trial["success"] = success_from_probs(candidate_probs)
                    trial["soft_accept"] = soft_accept
                    trial["pure_noise_step"] = pure_noise_step
                    if best_trial is None or bool(trial["success"]) or float(trial["accept_score"]) > float(best_trial["accept_score"]):
                        best_trial = trial
                    if bool(trial["success"]):
                        break
                del candidate_flats, candidate_probs_batch, candidate_trials
                if device.type == "cuda":
                    safe_cuda_empty_cache()

            if best_trial is None:
                break

            prev_objective = cur_objective
            prev_benign_conf = cur_benign_conf
            proportional_scale = None
            improved = False
            if bool(best_trial.get("pure_noise_step", False)) and query_count < self.query_budget:
                blended_primitives, proportional_scale = self._apply_proportional_step(
                    cur_primitives,
                    best_trial["primitives"],  # type: ignore[arg-type]
                    channels=channels,
                    device=device,
                    dtype=x_flat.dtype,
                    step_score=float(best_trial["accept_score"]),
                    objective_ref=cur_objective,
                )
                blended_latent = self._render_primitives(
                    blended_primitives,
                    num_tiles=cur_tiles,
                    channels=channels,
                    grid_size=cur_n,
                    device=device,
                    dtype=x_flat.dtype,
                )
                blended_flat = self._apply_latent(x_flat, blended_latent, lower=lower, upper=upper, lo=lo, hi=hi)
                blended_bundle = query_bundle(blended_flat)
                cur_primitives = self._clone_primitives(blended_primitives)
                cur_latent = blended_latent
                cur_bundle = blended_bundle
                cur_objective = objective(blended_bundle)
                cur_top2 = topk_support(blended_bundle, 2)
                cur_top10 = topk_support(blended_bundle, 10)
                cur_probs = blended_bundle.probs.reshape(-1)
                cur_benign_conf = benign_conf(cur_probs)
                best_trial["op"] = f"{best_trial['op']}[prop{proportional_scale:.2f}]"
                improved = bool(
                    success(cur_bundle)
                    or cur_objective > prev_objective + 1e-12
                    or cur_benign_conf < prev_benign_conf - 1e-6
                )
                if improved:
                    alpha_t = min(alpha_max, alpha_t * self.accept_alpha_growth)
                    steps_no_improve = 0
                else:
                    alpha_t = max(alpha_min, alpha_t * self.reject_alpha_decay)
                    steps_no_improve += 1
            else:
                improved = bool(
                    float(best_trial["objective"]) > cur_objective + 1e-12
                    or bool(best_trial["soft_accept"])
                )
                if improved or bool(best_trial["success"]):
                    cur_primitives = self._clone_primitives(best_trial["primitives"])  # type: ignore[arg-type]
                    cur_latent = best_trial["latent"]  # type: ignore[assignment]
                    cur_bundle = best_trial["bundle"]  # type: ignore[assignment]
                    cur_objective = float(best_trial["objective"])
                    cur_top2 = float(best_trial["top2"])
                    cur_top10 = float(best_trial["top10"])
                    cur_benign_conf = float(best_trial["benign_conf"])
                    cur_probs = cur_bundle.probs.reshape(-1)
                    alpha_t = min(alpha_max, alpha_t * self.accept_alpha_growth)
                    steps_no_improve = 0
                else:
                    cur_probs = cur_bundle.probs.reshape(-1)
                    alpha_t = max(alpha_min, alpha_t * self.reject_alpha_decay)
                    steps_no_improve += 1

            if (
                success(cur_bundle)
                or cur_benign_conf < best_benign_conf - 1e-6
                or cur_objective > best_objective + 1e-6
            ):
                best_objective = cur_objective
                best_primitives = self._clone_primitives(cur_primitives)
                best_bundle = cur_bundle
                best_benign_conf = cur_benign_conf
                best_n = cur_n
                best_tiles = cur_tiles
            elite_windows = self._merge_elite_windows(
                elite_windows,
                cur_primitives,
                guide_field=guide_field,
                num_tiles=cur_tiles,
                grid_size=cur_n,
            )

            final_label = int(cur_bundle.pred_label)
            second_label, second_conf = self._secondary_label(cur_probs, benign_label)
            delta_linf = float(cur_latent.abs().max().item())
            elapsed_sec = float(time.perf_counter() - attack_start)
            self._dbg(
                f"iter={step_i + 1}/{self.steps} queries={query_count}/{self.query_budget} "
                f"label={final_label} benign_conf={cur_benign_conf:.4f} second={second_label}:{second_conf:.4f} "
                f"op={best_trial['op']} squares={len(cur_primitives)} alpha_t={alpha_t:.4f} stall={steps_no_improve} "
                f"objective={cur_objective:.4f} accepted={improved} accept_score={float(best_trial['accept_score']):.4f} "
                f"top2={cur_top2:.4f} top10={cur_top10:.4f} N={cur_n} T_tiles={cur_tiles} "
                f"delta_linf={delta_linf:.4f}/{eps:.4f} elapsed={elapsed_sec:.2f}s"
            )
            if bool(best_trial["success"]):
                found = True
                break

            if self.plateau_patience > 0 and steps_no_improve >= self.plateau_patience:
                refined = False
                rescue_trial = dense_rescue(
                    base_latent=cur_latent,
                    base_objective=cur_objective,
                    base_top2=cur_top2,
                    base_top10=cur_top10,
                    base_conf=cur_benign_conf,
                    num_tiles=cur_tiles,
                    grid_size=cur_n,
                )
                if rescue_trial is not None:
                    cur_primitives = self._clone_primitives(rescue_trial["primitives"])  # type: ignore[arg-type]
                    cur_latent = rescue_trial["latent"]  # type: ignore[assignment]
                    cur_bundle = rescue_trial["bundle"]  # type: ignore[assignment]
                    cur_objective = float(rescue_trial["objective"])
                    cur_top2 = float(rescue_trial["top2"])
                    cur_top10 = float(rescue_trial["top10"])
                    cur_probs = cur_bundle.probs.reshape(-1)
                    cur_benign_conf = float(rescue_trial["benign_conf"])
                    steps_no_improve = 0
                    alpha_t = min(alpha_max, max(alpha_t, float(self.alpha)))
                    if success(cur_bundle) or cur_benign_conf < best_benign_conf - 1e-6 or cur_objective > best_objective + 1e-6:
                        best_objective = cur_objective
                        best_primitives = self._clone_primitives(cur_primitives)
                        best_bundle = cur_bundle
                        best_benign_conf = cur_benign_conf
                        best_n = cur_n
                        best_tiles = cur_tiles
                    elite_windows = self._merge_elite_windows(
                        elite_windows,
                        cur_primitives,
                        guide_field=None,
                        num_tiles=cur_tiles,
                        grid_size=cur_n,
                    )
                    self._dbg(
                        f"iter={step_i + 1}: plateau -> rescue objective={cur_objective:.4f} "
                        f"benign_conf={cur_benign_conf:.4f} squares={len(cur_primitives)} "
                        f"elapsed={time.perf_counter() - attack_start:.2f}s"
                    )
                    if success(cur_bundle):
                        found = True
                        break
                    refined = True
                    if device.type == "cuda":
                        safe_cuda_empty_cache()
                if (not refined) and self.coarse_to_fine and (cur_n < max_n or cur_tiles < max_tiles):
                    new_n = min(max_n, max(cur_n + 1, int(round(cur_n * 1.5))))
                    new_tiles = min(max_tiles, max(cur_tiles + 1, int(round(cur_tiles * 1.5))))
                    if new_n != cur_n or new_tiles != cur_tiles:
                        cur_primitives = self._scale_primitives(
                            cur_primitives,
                            src_tiles=cur_tiles,
                            dst_tiles=new_tiles,
                            src_size=cur_n,
                            dst_size=new_n,
                        )
                        cur_n = new_n
                        cur_tiles = new_tiles
                        cur_latent = self._render_primitives(
                            cur_primitives,
                            num_tiles=cur_tiles,
                            channels=channels,
                            grid_size=cur_n,
                            device=device,
                            dtype=x_flat.dtype,
                        )
                        current_flat = self._apply_latent(x_flat, cur_latent, lower=lower, upper=upper, lo=lo, hi=hi)
                        if query_count < self.query_budget:
                            cur_bundle = query_bundle(current_flat)
                            cur_objective = objective(cur_bundle)
                            cur_top2 = topk_support(cur_bundle, 2)
                            cur_top10 = topk_support(cur_bundle, 10)
                            cur_probs = cur_bundle.probs.reshape(-1)
                            cur_benign_conf = benign_conf(cur_probs)
                            if success(cur_bundle) or cur_benign_conf < best_benign_conf - 1e-6 or cur_objective > best_objective + 1e-6:
                                best_objective = cur_objective
                                best_primitives = self._clone_primitives(cur_primitives)
                                best_bundle = cur_bundle
                                best_benign_conf = cur_benign_conf
                                best_n = cur_n
                                best_tiles = cur_tiles
                            elite_windows = self._merge_elite_windows(
                                elite_windows,
                                cur_primitives,
                                guide_field=None,
                                num_tiles=cur_tiles,
                                grid_size=cur_n,
                            )
                            if success(cur_bundle):
                                found = True
                                break
                        refined = True
                        steps_no_improve = 0
                        self._dbg(
                            f"iter={step_i + 1}: plateau -> refine N={cur_n} T_tiles={cur_tiles} "
                            f"squares={len(cur_primitives)} elapsed={time.perf_counter() - attack_start:.2f}s"
                        )
                if not refined:
                    if cur_primitives:
                        keep = max(1, int(round(0.75 * len(cur_primitives))))
                        order = list(np.random.permutation(len(cur_primitives)))
                        cur_primitives = self._clone_primitives([cur_primitives[idx] for idx in order[:keep]])
                    alpha_t = min(alpha_max, max(float(self.alpha), alpha_t * 1.25))
                    steps_no_improve = 0
                    self._dbg(
                        f"iter={step_i + 1}: plateau -> prune/reset alpha_t={alpha_t:.4f} "
                        f"squares={len(cur_primitives)} elapsed={time.perf_counter() - attack_start:.2f}s"
                    )
            if device.type == "cuda" and ((step_i + 1) % 5 == 0):
                safe_cuda_empty_cache()

        best_latent = self._render_primitives(
            best_primitives,
            num_tiles=best_tiles,
            channels=channels,
            grid_size=best_n,
            device=device,
            dtype=x_flat.dtype,
        )
        best_flat = self._apply_latent(x_flat, best_latent, lower=lower, upper=upper, lo=lo, hi=hi)
        best_probs = best_bundle.probs.reshape(-1)
        second_label, second_conf = self._secondary_label(best_probs, benign_label)
        attack_time_sec = float(time.perf_counter() - attack_start)
        self.last_result = {
            "iter_count": iters_done,
            "query_count": query_count,
            "target_label": target_label if targeted else -1,
            "benign_label": benign_label,
            "final_label": int(best_bundle.pred_label),
            "benign_conf": float(best_probs[benign_label].item()),
            "second_label": second_label,
            "second_conf": second_conf,
            "effective_eps": eps,
            "attack_time_sec": attack_time_sec,
        }
        self._dbg(
            f"done iters={iters_done} queries={query_count} benign={benign_label} "
            f"final={int(best_bundle.pred_label)} found={found} time={attack_time_sec:.2f}s"
        )
        return restore_temporal_video(best_flat, restore_shape).detach()


ATTACK_CLASS = SquareAttack


def create(**kwargs) -> SquareAttack:
    return SquareAttack(**kwargs)

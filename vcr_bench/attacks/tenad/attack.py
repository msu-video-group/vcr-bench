from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from vcr_bench.attacks.query_base import BaseQueryVideoAttack
from vcr_bench.attacks.utils import flatten_temporal_video, restore_temporal_video
from vcr_bench.models.base import BaseVideoClassifier


@dataclass
class BoundarySearchResult:
    success: bool
    boundary_lambda: float
    attacked_video: torch.Tensor
    attacked_label: int


class TenAdAttack(BaseQueryVideoAttack):
    attack_name = "tenad"

    def __init__(
        self,
        eps: float = 255.0,
        alpha: float = 0.2,
        steps: int = 20,
        beta: float = 1e-2,
        binary_search_steps: int = 12,
        max_expand_steps: int = 16,
        lambda_init: float = 1.0,
        query_budget: int = 1000,
        num_restarts: int = 1,
        spatial_stride: int | None = None,
        clip_perturbation: bool = False,
    ) -> None:
        super().__init__(eps=eps, alpha=alpha, steps=steps)
        if beta <= 0:
            raise ValueError("TenAd requires beta > 0")
        if binary_search_steps < 0:
            raise ValueError("TenAd requires binary_search_steps >= 0")
        if max_expand_steps <= 0:
            raise ValueError("TenAd requires max_expand_steps > 0")
        if lambda_init <= 0:
            raise ValueError("TenAd requires lambda_init > 0")
        if query_budget <= 0:
            raise ValueError("TenAd requires query_budget > 0")
        if num_restarts <= 0:
            raise ValueError("TenAd requires num_restarts > 0")
        if spatial_stride is not None and spatial_stride < 1:
            raise ValueError("TenAd requires spatial_stride >= 1")
        self.beta = float(beta)
        self.binary_search_steps = int(binary_search_steps)
        self.max_expand_steps = int(max_expand_steps)
        self.lambda_init = float(lambda_init)
        self.query_budget = int(query_budget)
        self.num_restarts = int(num_restarts)
        self.spatial_stride = None if spatial_stride is None else int(spatial_stride)
        self.clip_perturbation = bool(clip_perturbation)
        self.verbose = False

    def prepare(self, *, model, dataset, indices, seed, targeted, pipeline_stage, verbose) -> None:
        del model, dataset, indices, seed, targeted, pipeline_stage
        self.verbose = bool(verbose)

    @staticmethod
    def infer_clip_bounds(x: torch.Tensor) -> tuple[float, float]:
        x_detached = x.detach()
        max_val = float(x_detached.max().item())
        min_val = float(x_detached.min().item())
        if min_val >= -1e-6 and max_val <= 1.5:
            return 0.0, 1.0
        return 0.0, 255.0

    def _get_spatial_stride(self, h: int, w: int) -> int:
        if self.spatial_stride is not None:
            return self.spatial_stride
        # Auto: target ~32px grid in each spatial dim
        return max(1, min(h, w) // 32)

    def init_factors(self, video_4d: torch.Tensor) -> list[torch.Tensor]:
        t, h, w, c = video_4d.shape
        device = video_4d.device
        dtype = video_4d.dtype
        stride = self._get_spatial_stride(h, w)
        h_s = max(1, h // stride)
        w_s = max(1, w // stride)
        return [
            torch.randn(t, device=device, dtype=dtype),
            torch.randn(h_s, device=device, dtype=dtype),
            torch.randn(w_s, device=device, dtype=dtype),
            torch.randn(c, device=device, dtype=dtype),
        ]

    @staticmethod
    def normalize_factors(factors: list[torch.Tensor]) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        for factor in factors:
            norm = torch.linalg.norm(factor.reshape(-1))
            if not torch.isfinite(norm) or float(norm.item()) <= 1e-12:
                out.append(torch.ones_like(factor) / max(factor.numel(), 1))
            else:
                out.append(factor / norm)
        return out

    @classmethod
    def build_rank1_perturbation(cls, factors: list[torch.Tensor]) -> torch.Tensor:
        t, h, w, c = cls.normalize_factors(factors)
        return torch.einsum("t,h,w,c->thwc", t, h, w, c)

    def _build_perturbation(
        self, factors: list[torch.Tensor], *, target_h: int, target_w: int
    ) -> torch.Tensor:
        """Build rank-1 perturbation and upsample to full spatial resolution if needed."""
        perturb = self.build_rank1_perturbation(factors)  # (T, h_s, w_s, C)
        t_dim, h_s, w_s, c_dim = perturb.shape
        if h_s == target_h and w_s == target_w:
            return perturb
        # (T, h_s, w_s, C) → (1, T*C, h_s, w_s) → interpolate → (T, H, W, C)
        x = perturb.permute(0, 3, 1, 2).reshape(1, t_dim * c_dim, h_s, w_s)
        x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return x.reshape(t_dim, c_dim, target_h, target_w).permute(0, 2, 3, 1)

    def _clip_video(self, x: torch.Tensor, *, clip_min: float, clip_max: float) -> torch.Tensor:
        return x.clamp(min=clip_min, max=clip_max)

    def _apply_lambda(
        self,
        video_4d: torch.Tensor,
        perturbation_4d: torch.Tensor,
        lam: float,
        *,
        clip_min: float,
        clip_max: float,
    ) -> torch.Tensor:
        result = video_4d + (float(lam) * perturbation_4d)
        if self.clip_perturbation:
            result = self._clip_video(result, clip_min=clip_min, clip_max=clip_max)
        return result

    def _search_boundary(
        self,
        video_4d: torch.Tensor,
        perturbation_4d: torch.Tensor,
        *,
        benign_label: int,
        query_label,
        clip_min: float,
        clip_max: float,
    ) -> BoundarySearchResult:
        max_lambda = float(self.eps) if self.eps > 0 else None
        lam = float(self.lambda_init)
        success_video = video_4d
        success_label = benign_label
        success_lambda: float | None = None

        for _ in range(self.max_expand_steps):
            if max_lambda is not None:
                lam = min(lam, max_lambda)
            candidate = self._apply_lambda(video_4d, perturbation_4d, lam, clip_min=clip_min, clip_max=clip_max)
            label = int(query_label(candidate))
            if label != benign_label:
                success_video = candidate
                success_label = label
                success_lambda = lam
                break
            if max_lambda is not None and lam >= max_lambda:
                break
            lam *= 2.0

        if success_lambda is None:
            return BoundarySearchResult(False, float("inf"), video_4d.detach().clone(), benign_label)

        low = 0.0
        high = float(success_lambda)
        best_video = success_video
        best_label = success_label
        for _ in range(self.binary_search_steps):
            mid = 0.5 * (low + high)
            candidate = self._apply_lambda(video_4d, perturbation_4d, mid, clip_min=clip_min, clip_max=clip_max)
            label = int(query_label(candidate))
            if label != benign_label:
                high = mid
                best_video = candidate
                best_label = label
            else:
                low = mid

        return BoundarySearchResult(True, float(high), best_video.detach().clone(), int(best_label))

    def _finite_difference_update(
        self,
        video_4d: torch.Tensor,
        factors: list[torch.Tensor],
        *,
        benign_label: int,
        query_label,
        clip_min: float,
        clip_max: float,
        target_h: int,
        target_w: int,
    ) -> tuple[list[torch.Tensor], BoundarySearchResult]:
        baseline = self._search_boundary(
            video_4d,
            self._build_perturbation(factors, target_h=target_h, target_w=target_w),
            benign_label=benign_label,
            query_label=query_label,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        if not baseline.success:
            return factors, baseline

        updated = [factor.detach().clone() for factor in factors]
        for idx, factor in enumerate(factors):
            direction = torch.randn_like(factor)
            candidate_factors = [cur.detach().clone() for cur in updated]
            candidate_factors[idx] = candidate_factors[idx] + (self.beta * direction)
            candidate = self._search_boundary(
                video_4d,
                self._build_perturbation(candidate_factors, target_h=target_h, target_w=target_w),
                benign_label=benign_label,
                query_label=query_label,
                clip_min=clip_min,
                clip_max=clip_max,
            )
            if not candidate.success:
                continue
            grad_scale = (candidate.boundary_lambda - baseline.boundary_lambda) / self.beta
            updated[idx] = updated[idx] - (self.alpha * grad_scale * direction)

        refined = self._search_boundary(
            video_4d,
            self._build_perturbation(updated, target_h=target_h, target_w=target_w),
            benign_label=benign_label,
            query_label=query_label,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        if refined.success and refined.boundary_lambda <= baseline.boundary_lambda:
            return updated, refined
        return updated, baseline

    def attack_query(
        self,
        model: BaseVideoClassifier,
        x: torch.Tensor,
        *,
        query_label,
        input_format: str = "NTHWC",
        y: torch.Tensor | int | None = None,
        targeted: bool = False,
        video_path: str | None = None,
    ) -> torch.Tensor:
        del y, video_path
        if targeted:
            raise ValueError("TenAd v1 supports untargeted attacks only")

        x_ref = self._ensure_float_attack_tensor(x.detach().clone())
        flat_video, restore_shape = flatten_temporal_video(x_ref)
        video_4d = flat_video.detach().clone()
        clip_min, clip_max = self.infer_clip_bounds(video_4d)
        _, target_h, target_w, _ = video_4d.shape
        stride = self._get_spatial_stride(target_h, target_w)

        total_queries = 0

        def budgeted_query(video_4d_local: torch.Tensor) -> int:
            nonlocal total_queries
            if total_queries >= self.query_budget:
                raise RuntimeError("TenAd query budget exhausted")
            total_queries += 1
            restored = restore_temporal_video(video_4d_local, restore_shape)
            return int(query_label(restored))

        def get_gt_conf(video_4d_local: torch.Tensor, gt_label: int) -> float:
            try:
                restored = restore_temporal_video(video_4d_local, restore_shape)
                with torch.no_grad():
                    probs = model.predict(restored, input_format=input_format)
                if probs.ndim != 1:
                    probs = probs.reshape(-1)
                return float(probs[gt_label].item()) if 0 <= gt_label < probs.numel() else 0.0
            except Exception:
                return float("nan")

        sample_factors = self.init_factors(video_4d)
        sample_perturb = self._build_perturbation(sample_factors, target_h=target_h, target_w=target_w)
        rank1_linf = float(sample_perturb.abs().max())
        max_lam = 2 ** (self.max_expand_steps - 1) * self.lambda_init
        if self.verbose:
            print(
                f"[tenad-debug] flat_shape={tuple(video_4d.shape)} clip=({clip_min},{clip_max}) clip_perturb={self.clip_perturbation} "
                f"pixel_range=({float(video_4d.min()):.3f},{float(video_4d.max()):.3f}) "
                f"spatial_stride={stride} coarse_hw=({sample_factors[1].numel()},{sample_factors[2].numel()}) "
                f"rank1_linf={rank1_linf:.5f} max_lambda={max_lam:.0f} max_delta={rank1_linf * max_lam:.1f}",
                flush=True,
            )
        del sample_factors, sample_perturb

        # Diagnostic: invert all pixels and check model sensitivity
        inverted = (clip_max - video_4d).clamp(clip_min, clip_max)
        try:
            restored_inv = restore_temporal_video(inverted, restore_shape)
            with torch.no_grad():
                probs_inv = model.predict(restored_inv, input_format=input_format)
            if probs_inv.ndim != 1:
                probs_inv = probs_inv.reshape(-1)
            top3 = torch.topk(probs_inv, k=min(3, probs_inv.numel()))
            top3_str = ", ".join(f"cls{int(i)}={float(v):.4f}" for v, i in zip(top3.values, top3.indices))
            if self.verbose:
                print(f"[tenad-diag] pixel-inverted video → top3: {top3_str}", flush=True)
        except Exception as e:
            if self.verbose:
                print(f"[tenad-diag] diagnostic failed: {e}", flush=True)

        benign_label = int(budgeted_query(video_4d))
        best_result = BoundarySearchResult(False, float("inf"), video_4d.detach().clone(), benign_label)
        best_factors: list[torch.Tensor] | None = None
        completed_iterations = 0

        try:
            for restart in range(self.num_restarts):
                factors = self.init_factors(video_4d)
                try:
                    for iteration in range(self.steps):
                        factors, current = self._finite_difference_update(
                            video_4d,
                            factors,
                            benign_label=benign_label,
                            query_label=budgeted_query,
                            clip_min=clip_min,
                            clip_max=clip_max,
                            target_h=target_h,
                            target_w=target_w,
                        )
                        completed_iterations = max(completed_iterations, iteration + 1)
                        if current.success and current.boundary_lambda < best_result.boundary_lambda:
                            best_result = current
                            best_factors = [factor.detach().clone() for factor in factors]
                        if best_result.success:
                            gt_conf = get_gt_conf(best_result.attacked_video, benign_label)
                            conf_tag = "adv"
                            lam_str = f"{best_result.boundary_lambda:.4f}"
                        else:
                            gt_conf = get_gt_conf(video_4d, benign_label)
                            conf_tag = "clean"
                            lam_str = "N/A"
                        if self.verbose:
                            print(
                                f"[tenad] restart={restart} iter={iteration + 1}/{self.steps} "
                                f"queries={total_queries}/{self.query_budget} "
                                f"gt_conf_{conf_tag}={gt_conf:.4f} lambda={lam_str} this_iter={current.success}",
                                flush=True,
                            )
                except RuntimeError as exc:
                    if "query budget exhausted" not in str(exc):
                        raise
                    break
        except RuntimeError as exc:
            if "query budget exhausted" not in str(exc):
                raise

        if best_factors is not None and best_result.success:
            attacked_video = restore_temporal_video(best_result.attacked_video, restore_shape).detach()
            final_label = int(best_result.attacked_label)
        else:
            attacked_video = x_ref.detach().clone()
            final_label = benign_label

        self.last_result = {
            "iter_count": int(completed_iterations),
            "query_count": int(total_queries),
            "target_label": -1,
            "benign_label": int(benign_label),
            "final_label": int(final_label),
            "final_lambda": float(best_result.boundary_lambda if best_result.success else 0.0),
        }
        return attacked_video


ATTACK_CLASS = TenAdAttack


def create(**kwargs) -> TenAdAttack:
    return TenAdAttack(**kwargs)

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from vg_videobench.attacks.query_base import BaseQueryVideoAttack
from vg_videobench.attacks.utils import flatten_temporal_video, restore_temporal_video
from vg_videobench.models.base import BaseVideoClassifier
from vg_videobench.types import PredictionBundle


class GradEstV2Attack(BaseQueryVideoAttack):
    """NES-based black-box attack with temporal variety and spatially smooth noise."""
    attack_name = "gradestv2"

    def __init__(
        self,
        eps: float = 32.0,
        alpha: float | None = None,        # per-step size; auto = eps/steps
        steps: int = 200,
        sigma: float = 0.200,
        sigma_decay: float = 0.997,
        n_samples: int = 10,
        n_samples_late: int = 5,
        spatial_size: int | None = 224,    # larger default than gradest
        temporal_tiles: int = 8,           # independent noise groups along time axis
        smooth_scale: float = 0.5,         # noise generated at N*smooth_scale, upsampled to N
        query_budget: int = 100_000,
        momentum: float = 0.9,
        boost_second: float = 0.0,
        boost_top2: float = 2.0,
        boost_top10: float = 1.0,
        boost_top5: float | None = None,
        batch_noise: bool = False,
        plateau_patience: int = 20,
        random_start: bool = True,
        random_start_trials: int = 4,
        coarse_to_fine: bool = True,
        coarse_spatial_scale: float = 0.5,
        coarse_temporal_scale: float = 0.5,
        reject_alpha_decay: float = 0.5,
        accept_alpha_growth: float = 1.05,
        adaptive_samples: bool = True,
    ) -> None:
        super().__init__(eps=eps, alpha=alpha if alpha is not None else eps / steps * 2.5, steps=steps, random_start=random_start)
        self.sigma = float(sigma)
        self.sigma_decay = float(sigma_decay)
        self.n_samples = int(n_samples)
        self.n_samples_late = int(n_samples_late)
        self.spatial_size = None if spatial_size is None else int(spatial_size)
        self.temporal_tiles = int(temporal_tiles)
        self.smooth_scale = float(smooth_scale)
        self.query_budget = int(query_budget)
        self.momentum = float(momentum)
        self.boost_second = float(boost_second)
        self.boost_top2 = float(boost_top2)
        self.boost_top10 = float(boost_top10 if boost_top5 is None else (boost_top10 if boost_top10 != 0.0 else boost_top5))
        self.batch_noise = bool(batch_noise)
        self.plateau_patience = int(plateau_patience)
        self.random_start_trials = max(1, int(random_start_trials))
        self.coarse_to_fine = bool(coarse_to_fine)
        self.coarse_spatial_scale = float(coarse_spatial_scale)
        self.coarse_temporal_scale = float(coarse_temporal_scale)
        self.reject_alpha_decay = float(reject_alpha_decay)
        self.accept_alpha_growth = float(accept_alpha_growth)
        self.adaptive_samples = bool(adaptive_samples)
        self.verbose = False

    def _dbg(self, msg: str) -> None:
        if self.verbose:
            print(f"[gradestv2] {msg}", flush=True)

    def prepare(self, *, model, dataset, indices, seed, targeted, pipeline_stage, verbose) -> None:
        del model, dataset, indices, seed, targeted, pipeline_stage
        self.verbose = bool(verbose)

    def _make_noise(self, T_tiles: int, N: int, C: int, device: torch.device) -> torch.Tensor:
        """(T_tiles, C, N, N) smooth low-res latent noise."""
        if self.smooth_scale < 1.0:
            n_small = max(4, int(N * self.smooth_scale))
            small = torch.randn(T_tiles * C, 1, n_small, n_small, device=device)
            up = F.interpolate(small, size=(N, N), mode="bilinear", align_corners=False)
            return up.reshape(T_tiles, C, N, N).contiguous()
        return torch.randn(T_tiles, C, N, N, device=device)

    def _resize_latent(
        self,
        latent: torch.Tensor,
        *,
        target_tiles: int,
        target_size: int,
    ) -> torch.Tensor:
        """Resize latent perturbation over temporal tiles and spatial grid."""
        if latent.shape[0] != target_tiles:
            src_tiles, channels, n_h, n_w = latent.shape
            tmp = latent.permute(1, 2, 3, 0).reshape(1, channels * n_h * n_w, src_tiles)
            tmp = F.interpolate(tmp, size=target_tiles, mode="linear", align_corners=False)
            latent = tmp.reshape(channels, n_h, n_w, target_tiles).permute(3, 0, 1, 2).contiguous()
        if latent.shape[-1] != target_size:
            latent = F.interpolate(latent, size=(target_size, target_size), mode="bilinear", align_corners=False)
        return latent.contiguous()

    def _latent_to_video(
        self,
        latent: torch.Tensor,
        *,
        T: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """latent: (T_tiles, C, N, N) -> (T, H, W, C) via temporal+spatial interpolation."""
        T_tiles, channels, n_h, n_w = latent.shape
        if T_tiles != T:
            tmp = latent.permute(1, 2, 3, 0).reshape(1, channels * n_h * n_w, T_tiles)
            tmp = F.interpolate(tmp, size=T, mode="linear", align_corners=False)
            full_t = tmp.reshape(channels, n_h, n_w, T).permute(3, 0, 1, 2).contiguous()
        else:
            full_t = latent
        if full_t.shape[-2:] != (H, W):
            full_t = F.interpolate(full_t, size=(H, W), mode="bilinear", align_corners=False)
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
        T, H, W, _ = x_flat.shape
        adv = x_flat + self._latent_to_video(latent, T=T, H=H, W=W)
        adv = torch.clamp(adv, lo, hi)
        return torch.max(torch.min(adv, upper), lower)

    @staticmethod
    def _aggregate_logits(logits: torch.Tensor) -> torch.Tensor:
        x = logits.float()
        if x.ndim == 1:
            return x
        return x.reshape(-1, x.shape[-1]).mean(dim=0)

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
        del query_label
        attack_start = time.perf_counter()

        x_ref = self._ensure_float_attack_tensor(x.detach().clone())
        x_flat, restore_shape = flatten_temporal_video(x_ref)
        T, H, W, C = x_flat.shape
        device = x_flat.device
        lo, hi = 0.0, 255.0
        eps = float(self.eps)
        max_N = min(H, W) if self.spatial_size is None else min(int(self.spatial_size), H, W)
        max_T_tiles = min(self.temporal_tiles, T)
        if self.coarse_to_fine:
            start_N = max(4, min(max_N, int(round(max_N * self.coarse_spatial_scale))))
            start_T_tiles = max(1, min(max_T_tiles, int(round(max_T_tiles * self.coarse_temporal_scale))))
        else:
            start_N = max_N
            start_T_tiles = max_T_tiles
        query_count = 0
        sigma = self.sigma

        def query_bundle(flat: torch.Tensor) -> PredictionBundle:
            nonlocal query_count
            query_count += 1
            restored = restore_temporal_video(flat, restore_shape)
            with torch.no_grad():
                bundle = model(restored, input_format=input_format, enable_grad=False, return_full=True)
            if not isinstance(bundle, PredictionBundle):
                raise TypeError(f"Expected PredictionBundle, got {type(bundle)}")
            return bundle

        benign_bundle = query_bundle(x_flat)
        benign_probs = benign_bundle.probs.reshape(-1)
        benign_label = int(benign_probs.argmax().item())
        second_label_init = int(torch.topk(benign_probs, 2).indices[1].item())

        if targeted:
            if y is None:
                target_label = int(benign_probs.argmin().item())
            elif isinstance(y, int):
                target_label = y
            else:
                target_label = int(y.item())
        else:
            target_label = benign_label

        lower = torch.clamp(x_flat - eps, lo, hi)
        upper = torch.clamp(x_flat + eps, lo, hi)

        cur_N = start_N
        cur_T_tiles = start_T_tiles
        latent = torch.zeros(cur_T_tiles, C, cur_N, cur_N, device=device)
        velocity = torch.zeros_like(latent)
        x_adv_flat = x_flat.clone()
        cur_bundle = benign_bundle

        def _objective(bundle: PredictionBundle) -> float:
            logits = self._aggregate_logits(bundle.logits)
            if targeted:
                main = logits[target_label]
                others = logits.clone()
                others[target_label] = float("-inf")
                obj = main - others.max()
            else:
                main = logits[benign_label]
                others = logits.clone()
                others[benign_label] = float("-inf")
                obj = others.max() - main
            if self.boost_second > 0.0 and others.numel() > 1:
                obj = obj + (self.boost_second * others.max() if not targeted else -self.boost_second * others.max())
            return float(obj.item())

        def _topk_support(bundle: PredictionBundle, k: int) -> float:
            logits = self._aggregate_logits(bundle.logits)
            if targeted:
                others = logits.clone()
                others[target_label] = float("-inf")
            else:
                others = logits.clone()
                others[benign_label] = float("-inf")
            if others.numel() <= 1:
                return float(0.0)
            topk = torch.topk(others, min(int(k), others.numel() - 1)).values
            if topk.numel() == 0:
                return float(0.0)
            return float(topk.mean().item())

        def _success(bundle: PredictionBundle) -> bool:
            pred = int(bundle.pred_label)
            return pred == target_label if targeted else pred != benign_label

        selected_start_trial = 0
        attempted_start_trials = 0
        if self.random_start and query_count < self.query_budget:
            max_trials = min(self.random_start_trials, max(self.query_budget - query_count, 0))
            attempted_start_trials = max_trials
            best_start_latent: torch.Tensor | None = None
            best_start_flat: torch.Tensor | None = None
            best_start_bundle: PredictionBundle | None = None
            best_start_objective = float("-inf")
            for trial_idx in range(max_trials):
                trial_latent = torch.empty_like(latent).uniform_(-eps, eps)
                trial_flat = self._apply_latent(x_flat, trial_latent, lower=lower, upper=upper, lo=lo, hi=hi)
                trial_bundle = query_bundle(trial_flat)
                trial_objective = _objective(trial_bundle)
                if (best_start_bundle is None) or (trial_objective > best_start_objective):
                    best_start_latent = trial_latent
                    best_start_flat = trial_flat
                    best_start_bundle = trial_bundle
                    best_start_objective = trial_objective
                    selected_start_trial = trial_idx + 1
                if _success(trial_bundle):
                    best_start_latent = trial_latent
                    best_start_flat = trial_flat
                    best_start_bundle = trial_bundle
                    best_start_objective = trial_objective
                    selected_start_trial = trial_idx + 1
                    break

            if best_start_latent is not None and best_start_flat is not None and best_start_bundle is not None:
                latent = best_start_latent
                x_adv_flat = best_start_flat
                cur_bundle = best_start_bundle

        cur_objective = _objective(cur_bundle)
        if self.random_start and _success(cur_bundle):
            attack_time_sec = float(time.perf_counter() - attack_start)
            self.last_result = {
                "iter_count": 0,
                "query_count": query_count,
                "target_label": target_label if targeted else -1,
                "benign_label": benign_label,
                "final_label": int(cur_bundle.pred_label),
                "effective_eps": eps,
                "attack_time_sec": attack_time_sec,
            }
            self._dbg(
                f"done iters=0 queries={query_count} benign={benign_label} "
                f"final={int(cur_bundle.pred_label)} found=True "
                f"time={attack_time_sec:.2f}s [random-start trial={selected_start_trial}/{max(1, attempted_start_trials)}]"
            )
            return restore_temporal_video(x_adv_flat, restore_shape).detach()

        clean_objective = _objective(benign_bundle)
        best_adv_flat = x_flat.clone()
        best_bundle = benign_bundle
        best_objective = clean_objective
        if cur_objective > best_objective or _success(cur_bundle):
            best_adv_flat = x_adv_flat.clone()
            best_bundle = cur_bundle
            best_objective = cur_objective
        iters_done = 0
        final_label = int(cur_bundle.pred_label)
        found = False
        steps_no_improve = 0
        pair_cv_ema = 1.0
        alpha_t = self.alpha
        alpha_min = max(1e-3, self.alpha * 0.1)
        alpha_max = max(alpha_min, self.alpha * 3.0)
        cur_top2_support = _topk_support(cur_bundle, 2)
        cur_top10_support = _topk_support(cur_bundle, 10)

        self._dbg(
            f"start shape={tuple(x_flat.shape)} eps={eps:.1f} alpha={self.alpha:.2f} "
            f"N={cur_N}->{max_N} T_tiles={cur_T_tiles}->{max_T_tiles} "
            f"smooth_scale={self.smooth_scale} steps={self.steps} "
            f"sigma={sigma:.4f} sigma_decay={self.sigma_decay} "
            f"n_samples={self.n_samples}/{self.n_samples_late} "
            f"momentum={self.momentum} boost_second={self.boost_second} "
            f"boost_top2={self.boost_top2} boost_top10={self.boost_top10} "
            f"random_start={self.random_start} random_start_trials={self.random_start_trials} "
            f"benign={benign_label} second_init={second_label_init}"
        )

        if self.random_start:
            init_delta_linf = float((x_adv_flat - x_flat).abs().max().item())
            init_elapsed_sec = float(time.perf_counter() - attack_start)
            self._dbg(
                f"init random-start objective={cur_objective:.4f} "
                f"trial={selected_start_trial}/{max(1, attempted_start_trials)} "
                f"delta_linf={init_delta_linf:.4f}/{eps:.4f} elapsed={init_elapsed_sec:.2f}s"
            )

        if self.batch_noise:
            self._dbg("batch_noise=True is incompatible with margin/full-bundle mode; using sequential queries")

        for step_i in range(self.steps):
            iters_done = step_i + 1
            if query_count >= self.query_budget:
                break

            sample_floor = max(2, min(self.n_samples, self.n_samples_late))
            sample_ceiling = max(sample_floor, self.n_samples, self.n_samples_late)
            remaining_budget = max(self.query_budget - query_count - 1, 0)
            planning_horizon = max(1, min(self.steps - step_i, 8))
            budget_cap = max(1, remaining_budget // max(2, 2 * planning_horizon))
            variance_ratio = pair_cv_ema / (1.0 + pair_cv_ema) if self.adaptive_samples else 0.0
            target_samples = sample_floor + int(round((sample_ceiling - sample_floor) * variance_ratio))
            n_samples = max(1, min(target_samples, budget_cap, remaining_budget // 2 if remaining_budget >= 2 else 1))

            grad = torch.zeros_like(latent)
            pair_diffs: list[float] = []
            lucky_adv_flat: torch.Tensor | None = None
            lucky_bundle: PredictionBundle | None = None

            for _ in range(n_samples):
                if query_count + 2 > self.query_budget:
                    break
                noise = self._make_noise(cur_T_tiles, cur_N, C, device)
                plus_flat = self._apply_latent(
                    x_flat,
                    latent + (sigma * noise),
                    lower=lower,
                    upper=upper,
                    lo=lo,
                    hi=hi,
                )
                plus_bundle = query_bundle(plus_flat)
                if _success(plus_bundle):
                    lucky_adv_flat = plus_flat
                    lucky_bundle = plus_bundle
                    break

                minus_flat = self._apply_latent(
                    x_flat,
                    latent - (sigma * noise),
                    lower=lower,
                    upper=upper,
                    lo=lo,
                    hi=hi,
                )
                minus_bundle = query_bundle(minus_flat)
                if _success(minus_bundle):
                    lucky_adv_flat = minus_flat
                    lucky_bundle = minus_bundle
                    break

                diff = _objective(plus_bundle) - _objective(minus_bundle)
                pair_diffs.append(diff)
                grad += diff * noise

            if lucky_adv_flat is not None and lucky_bundle is not None:
                x_adv_flat = lucky_adv_flat
                best_adv_flat = lucky_adv_flat.clone()
                best_bundle = lucky_bundle
                best_objective = _objective(lucky_bundle)
                final_label = int(lucky_bundle.pred_label)
                found = True
                elapsed_sec = float(time.perf_counter() - attack_start)
                self._dbg(
                    f"iter={step_i + 1}/{self.steps} queries={query_count}/{self.query_budget} "
                    f"label={final_label} objective={best_objective:.4f} "
                    f"elapsed={elapsed_sec:.2f}s [lucky-shot]"
                )
                break

            actual_samples = len(pair_diffs)
            if actual_samples == 0:
                break

            grad = grad / (2.0 * actual_samples * sigma)
            pair_tensor = torch.tensor(pair_diffs, device=device, dtype=torch.float32)
            mean_abs = float(pair_tensor.abs().mean().item())
            std_val = float(pair_tensor.std(unbiased=False).item()) if pair_tensor.numel() > 1 else 0.0
            pair_cv = std_val / (mean_abs + 1e-6)
            pair_cv_ema = 0.8 * pair_cv_ema + 0.2 * pair_cv

            velocity = self.momentum * velocity + (1.0 - self.momentum) * grad
            if velocity.abs().max() < 1e-12:
                sigma *= self.sigma_decay
                continue
            direction = torch.sign(velocity)

            if query_count < self.query_budget:
                progress = float(step_i) / float(max(1, self.steps - 1))
                boost_top2_t = self.boost_top2 * (1.0 + 3.0 * progress)
                candidate_latent = latent + (alpha_t * direction)
                candidate_flat = self._apply_latent(
                    x_flat,
                    candidate_latent,
                    lower=lower,
                    upper=upper,
                    lo=lo,
                    hi=hi,
                )
                candidate_bundle = query_bundle(candidate_flat)
                candidate_objective = _objective(candidate_bundle)
                candidate_top2_support = _topk_support(candidate_bundle, 2)
                candidate_top10_support = _topk_support(candidate_bundle, 10)
                margin_gain = candidate_objective - cur_objective
                top2_gain = (
                    (cur_top2_support - candidate_top2_support)
                    if targeted
                    else (candidate_top2_support - cur_top2_support)
                )
                top10_gain = (
                    (cur_top10_support - candidate_top10_support)
                    if targeted
                    else (candidate_top10_support - cur_top10_support)
                )
                accept_score = margin_gain + (boost_top2_t * top2_gain) + (self.boost_top10 * top10_gain)
                improved = accept_score > 1e-12

                if improved or _success(candidate_bundle):
                    latent = candidate_latent
                    x_adv_flat = candidate_flat
                    cur_bundle = candidate_bundle
                    cur_objective = candidate_objective
                    cur_top2_support = candidate_top2_support
                    cur_top10_support = candidate_top10_support
                    final_label = int(cur_bundle.pred_label)
                    alpha_t = min(alpha_max, alpha_t * self.accept_alpha_growth)
                    sigma *= self.sigma_decay
                    steps_no_improve = 0
                    if cur_objective > best_objective or _success(cur_bundle):
                        best_objective = cur_objective
                        best_adv_flat = x_adv_flat.clone()
                        best_bundle = cur_bundle
                else:
                    alpha_t = max(alpha_min, alpha_t * self.reject_alpha_decay)
                    steps_no_improve += 1
                    final_label = int(cur_bundle.pred_label)

                cur_probs = cur_bundle.probs.reshape(-1)
                benign_conf = float(cur_probs[benign_label].item())
                delta_linf = float((x_adv_flat - x_flat).abs().max().item())
                top2 = torch.topk(cur_probs, 2)
                second_label = int(top2.indices[1].item()) if final_label == benign_label else int(top2.indices[0].item())
                second_conf = float(cur_probs[second_label].item())
                elapsed_sec = float(time.perf_counter() - attack_start)
                self._dbg(
                    f"iter={step_i + 1}/{self.steps} queries={query_count}/{self.query_budget} "
                    f"label={final_label} benign_conf={benign_conf:.4f} second={second_label}:{second_conf:.4f} "
                    f"sigma={sigma:.4f} stall={steps_no_improve} alpha_t={alpha_t:.4f} "
                    f"objective={cur_objective:.4f} accepted={improved} "
                    f"top2={cur_top2_support:.4f} top2_gain={top2_gain:.4f} boost_top2_t={boost_top2_t:.4f} "
                    f"top10={cur_top10_support:.4f} top10_gain={top10_gain:.4f} accept_score={accept_score:.4f} "
                    f"pair_cv={pair_cv:.3f} N={cur_N} T_tiles={cur_T_tiles} "
                    f"delta_linf={delta_linf:.4f}/{eps:.4f} elapsed={elapsed_sec:.2f}s"
                )
                if _success(cur_bundle):
                    found = True
                    break

                if self.plateau_patience > 0 and steps_no_improve >= self.plateau_patience:
                    refined = False
                    if self.coarse_to_fine and (cur_N < max_N or cur_T_tiles < max_T_tiles):
                        new_N = min(max_N, max(cur_N + 1, int(round(cur_N * 1.5))))
                        new_T_tiles = min(max_T_tiles, max(cur_T_tiles + 1, int(round(cur_T_tiles * 1.5))))
                        if new_N != cur_N or new_T_tiles != cur_T_tiles:
                            latent = self._resize_latent(latent, target_tiles=new_T_tiles, target_size=new_N)
                            velocity = self._resize_latent(velocity, target_tiles=new_T_tiles, target_size=new_N)
                            cur_N = new_N
                            cur_T_tiles = new_T_tiles
                            x_adv_flat = self._apply_latent(
                                x_flat,
                                latent,
                                lower=lower,
                                upper=upper,
                                lo=lo,
                                hi=hi,
                            )
                            if query_count < self.query_budget:
                                cur_bundle = query_bundle(x_adv_flat)
                                cur_objective = _objective(cur_bundle)
                                cur_top2_support = _topk_support(cur_bundle, 2)
                                cur_top10_support = _topk_support(cur_bundle, 10)
                                final_label = int(cur_bundle.pred_label)
                                if cur_objective > best_objective or _success(cur_bundle):
                                    best_objective = cur_objective
                                    best_adv_flat = x_adv_flat.clone()
                                    best_bundle = cur_bundle
                                if _success(cur_bundle):
                                    found = True
                                    break
                            refined = True
                            steps_no_improve = 0
                            elapsed_sec = float(time.perf_counter() - attack_start)
                            self._dbg(
                                f"iter={step_i + 1}: plateau -> refine N={cur_N} T_tiles={cur_T_tiles} "
                                f"elapsed={elapsed_sec:.2f}s"
                            )
                    if not refined:
                        velocity.zero_()
                        steps_no_improve = 0
                        elapsed_sec = float(time.perf_counter() - attack_start)
                        self._dbg(f"iter={step_i + 1}: plateau -> reset velocity elapsed={elapsed_sec:.2f}s")

        attack_time_sec = float(time.perf_counter() - attack_start)
        self.last_result = {
            "iter_count": iters_done,
            "query_count": query_count,
            "target_label": target_label if targeted else -1,
            "benign_label": benign_label,
            "final_label": int(best_bundle.pred_label),
            "effective_eps": eps,
            "attack_time_sec": attack_time_sec,
        }
        self._dbg(
            f"done iters={iters_done} queries={query_count} benign={benign_label} "
            f"final={int(best_bundle.pred_label)} found={found} time={attack_time_sec:.2f}s"
        )
        return restore_temporal_video(best_adv_flat, restore_shape).detach()


ATTACK_CLASS = GradEstV2Attack


def create(**kwargs) -> GradEstV2Attack:
    return GradEstV2Attack(**kwargs)

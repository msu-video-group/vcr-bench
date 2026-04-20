from __future__ import annotations

import heapq
import itertools
import math
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from vcr_bench.attacks.query_base import BaseQueryVideoAttack
from vcr_bench.attacks.utils import flatten_temporal_video, restore_temporal_video
from vcr_bench.models.base import BaseVideoClassifier


def _nll(scores: torch.Tensor, label: int) -> float:
    probs = F.softmax(scores.float(), dim=0)
    return -float(torch.log(probs[label].clamp_min(1e-12)).item())


def _split_blocks(h: int, w: int, block_size: int) -> list[tuple[int, int, int, int, int]]:
    """Return list of (h0, h1, w0, w1, c) block descriptors."""
    blocks = []
    for h0 in range(0, h, block_size):
        for w0 in range(0, w, block_size):
            h1 = min(h0 + block_size, h)
            w1 = min(w0 + block_size, w)
            for c in range(3):
                blocks.append((h0, h1, w0, w1, c))
    return blocks


def _perturb_image(
    frame: torch.Tensor,   # (C, H, W)
    noise: torch.Tensor,   # (C, H_n, W_n)
    H: int, W: int,
) -> torch.Tensor:
    """Resize noise to (C, H, W) and add to frame."""
    noise_resized = F.interpolate(
        noise.unsqueeze(0), size=(H, W), mode="nearest"
    ).squeeze(0)
    return torch.clamp(frame + noise_resized, 0.0, frame.max().item() + 1e-3)


def _flip_noise(noise: torch.Tensor, block: tuple) -> torch.Tensor:
    """Return a copy of noise with the sign flipped on one block."""
    h0, h1, w0, w1, c = block
    n = noise.clone()
    n[c, h0:h1, w0:w1] = -n[c, h0:h1, w0:w1]
    return n


class ParsimoniousAttack(BaseQueryVideoAttack):
    """Parsimonious (local search) black-box attack.

    Operates frame-by-frame with a shared spatial noise, refined via lazy
    greedy insert/delete over block tiles with a hierarchical block schedule.
    Mirrors the original ``parsimonious.py`` implementation.

    Default hyper-parameters match the original config (eps=32, iter=10).
    Note: ``steps`` here controls the outer hierarchical iterations per frame,
    not the total query count. Use ``query_budget`` to cap the query count.
    """
    attack_name = "parsimonious"

    def __init__(
        self,
        eps: float = 32.0,
        alpha: float = 1.0,
        steps: int = 10,          # inner local-search iterations per block pass
        block_size_init: int = 32,
        noise_size: int = 256,    # internal noise canvas size (matches original)
        max_queries: int = 2_000,
        query_budget: int = 100_000,
    ) -> None:
        super().__init__(eps=eps, alpha=alpha, steps=steps)
        self.block_size_init = int(block_size_init)
        self.noise_size = int(noise_size)
        self.max_queries = int(max_queries)
        self.query_budget = int(query_budget)
        self.verbose = False

    def prepare(self, *, model, dataset, indices, seed, targeted, pipeline_stage, verbose) -> None:
        del model, dataset, indices, seed, targeted, pipeline_stage
        self.verbose = bool(verbose)

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
        del query_label

        x_ref = self._ensure_float_attack_tensor(x.detach().clone())
        x_flat, restore_shape = flatten_temporal_video(x_ref)  # (T, H, W, C)
        T, H, W, C = x_flat.shape
        device = x_flat.device

        hi = 255.0 if float(x_flat.max().item()) > 1.5 else 1.0
        lo = 0.0

        # Work in TCHW format
        x_tchw = x_flat.permute(0, 3, 1, 2).clone()  # (T, C, H, W)
        lower_tchw = torch.clamp(x_tchw - self.eps, lo, hi)
        upper_tchw = torch.clamp(x_tchw + self.eps, lo, hi)

        N_noise = self.noise_size
        query_count = 0

        def _model_call(tchw: torch.Tensor) -> torch.Tensor:
            """Call model on a TCHW tensor, return 1-D scores."""
            nonlocal query_count
            query_count += 1
            flat = tchw.permute(0, 2, 3, 1)
            restored = restore_temporal_video(flat, restore_shape)
            with torch.no_grad():
                s = model(restored, input_format=input_format, enable_grad=False)
            return s.reshape(-1)

        benign_scores = _model_call(x_tchw)
        benign_label = int(benign_scores.argmax().item())

        if targeted:
            target_label = int(benign_scores.argmin().item()) if y is None else (y if isinstance(y, int) else int(y.item()))
        else:
            target_label = benign_label

        # Shared noise canvas: (C, N_noise, N_noise), initialised to -eps
        noise = -self.eps * torch.ones(C, N_noise, N_noise, device=device)

        x_adv_tchw = x_tchw.clone()
        final_label = benign_label

        # ------------------------------------------------------------------
        # Hierarchical block schedule
        # ------------------------------------------------------------------
        block_size = self.block_size_init
        no_hier = False
        total_iters_done = 0

        while True:
            blocks = _split_blocks(N_noise, N_noise, block_size)
            n_blocks = len(blocks)
            order = np.random.permutation(n_blocks).tolist()

            # Process each selected frame
            key_frames = list(range(0, T, max(1, T // 4))) or [0]

            for frame_idx in key_frames:
                if query_count >= min(self.max_queries, self.query_budget):
                    break

                frame = x_tchw[frame_idx:frame_idx + 1]  # (1, C, H, W)
                curr_loss = _nll(
                    _model_call(x_adv_tchw),
                    target_label,
                )

                batch_size = 100
                n_batches = math.ceil(n_blocks / batch_size)
                pq: list = []  # min-heap of (−margin, idx)
                A = np.zeros(n_blocks, dtype=np.int32)  # 1 = in working set

                # Initialise A from current noise signs
                for i, (h0, h1, w0, w1, c) in enumerate(blocks):
                    if float(noise[c, h0:h1, w0:w1].mean().item()) > 0:
                        A[i] = 1

                for _it in range(self.steps):
                    total_iters_done += 1

                    # ---- lazy greedy INSERT ----
                    (not_in,) = np.where(A == 0)
                    for ibatch in range(math.ceil(len(not_in) / batch_size)):
                        bstart = ibatch * batch_size
                        bend = min(bstart + batch_size, len(not_in))
                        for k in range(bstart, bend):
                            if query_count >= min(self.max_queries, self.query_budget):
                                break
                            idx = int(not_in[k])
                            cand_noise = _flip_noise(noise, blocks[idx])
                            cand_frame = _perturb_image(frame[0], cand_noise, H, W)
                            cand_tchw = x_adv_tchw.clone()
                            cand_tchw[frame_idx] = torch.clamp(
                                cand_frame, lower_tchw[frame_idx], upper_tchw[frame_idx]
                            )
                            cand_scores = _model_call(cand_tchw)
                            cand_loss = _nll(cand_scores, target_label)
                            margin = -(cand_loss - curr_loss)  # negate for min-heap
                            heapq.heappush(pq, (margin, idx))

                    if pq:
                        best_margin, best_idx = heapq.heappop(pq)
                        curr_loss -= best_margin
                        noise = _flip_noise(noise, blocks[best_idx])
                        A[best_idx] = 1

                    while pq:
                        cand_margin, cand_idx = heapq.heappop(pq)
                        if query_count >= min(self.max_queries, self.query_budget):
                            break
                        cand_noise = _flip_noise(noise, blocks[cand_idx])
                        cand_frame = _perturb_image(frame[0], cand_noise, H, W)
                        cand_tchw = x_adv_tchw.clone()
                        cand_tchw[frame_idx] = torch.clamp(
                            cand_frame, lower_tchw[frame_idx], upper_tchw[frame_idx]
                        )
                        cand_scores = _model_call(cand_tchw)
                        cand_loss = _nll(cand_scores, target_label)
                        margin = -(cand_loss - curr_loss)
                        if not pq or margin <= pq[0][0]:
                            if margin > 0:
                                break
                            curr_loss = cand_loss
                            noise = _flip_noise(noise, blocks[cand_idx])
                            A[cand_idx] = 1
                        else:
                            heapq.heappush(pq, (margin, cand_idx))
                    pq = []

                    # ---- lazy greedy DELETE ----
                    (in_set,) = np.where(A == 1)
                    for k in range(len(in_set)):
                        if query_count >= min(self.max_queries, self.query_budget):
                            break
                        idx = int(in_set[k])
                        cand_noise = _flip_noise(noise, blocks[idx])
                        cand_frame = _perturb_image(frame[0], cand_noise, H, W)
                        cand_tchw = x_adv_tchw.clone()
                        cand_tchw[frame_idx] = torch.clamp(
                            cand_frame, lower_tchw[frame_idx], upper_tchw[frame_idx]
                        )
                        cand_scores = _model_call(cand_tchw)
                        cand_loss = _nll(cand_scores, target_label)
                        margin = -(cand_loss - curr_loss)
                        heapq.heappush(pq, (margin, idx))

                    if pq:
                        best_margin, best_idx = heapq.heappop(pq)
                        if best_margin < 0:
                            curr_loss -= best_margin
                            noise = _flip_noise(noise, blocks[best_idx])
                            A[best_idx] = 0
                    pq = []

                # Apply updated noise to this frame
                new_frame = _perturb_image(frame[0], noise, H, W)
                x_adv_tchw[frame_idx] = torch.clamp(
                    new_frame, lower_tchw[frame_idx], upper_tchw[frame_idx]
                )

                final_scores = _model_call(x_adv_tchw)
                final_label = int(final_scores.argmax().item())
                if self.verbose:
                    benign_conf = float(F.softmax(final_scores.float(), dim=0)[benign_label].item())
                    print(
                        f"[parsimonious] block_size={block_size} frame={frame_idx} "
                        f"queries={query_count} label={final_label} benign_conf={benign_conf:.4f}",
                        flush=True,
                    )
                if (not targeted and final_label != benign_label) or \
                   (targeted and final_label == target_label):
                    break

            if query_count >= min(self.max_queries, self.query_budget):
                break

            if (not targeted and final_label != benign_label) or \
               (targeted and final_label == target_label):
                break

            if not no_hier and block_size >= 2:
                block_size //= 2
            else:
                break

        x_adv_flat = x_adv_tchw.permute(0, 2, 3, 1)
        self.last_result = {
            "iter_count": total_iters_done,
            "query_count": query_count,
            "target_label": target_label if targeted else -1,
            "benign_label": benign_label,
            "final_label": final_label,
        }
        return restore_temporal_video(x_adv_flat, restore_shape).detach()


ATTACK_CLASS = ParsimoniousAttack


def create(**kwargs) -> ParsimoniousAttack:
    return ParsimoniousAttack(**kwargs)

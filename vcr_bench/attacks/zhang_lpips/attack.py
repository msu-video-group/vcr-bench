from __future__ import annotations

from vcr_bench.attacks.base import BaseVideoAttack
from vcr_bench.attacks.zhang_common import metric_attack
from vcr_bench.utils.metrics import video_lpips


class ZhangLPIPSAttack(BaseVideoAttack):
    attack_name = "zhang-lpips"

    def __init__(self, eps: float = 8.0, steps: int = 20, target_conf: float = 0.5, sample_chunk_size: int | None = None) -> None:
        super().__init__(eps=eps, alpha=1.0, steps=steps, clip_min=0.0, clip_max=255.0, sample_chunk_size=sample_chunk_size)
        self.target_conf = float(target_conf)

    def attack_sampled(self, model, x, *, input_format="NTHWC", y=None, targeted=False):
        metric = lambda a, b: video_lpips(a, b, device=getattr(model, "device", a.device))
        adv, meta = metric_attack(model, x, metric, k=-1.0, eps=self.eps, steps=self.steps, targeted=targeted, input_format=input_format, y=y, target_conf=self.target_conf)
        self.last_result = meta
        return adv


ATTACK_CLASS = ZhangLPIPSAttack


def create(**kwargs) -> ZhangLPIPSAttack:
    return ZhangLPIPSAttack(**kwargs)

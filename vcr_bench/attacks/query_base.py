from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

from vcr_bench.attacks.base import BaseVideoAttack
from vcr_bench.models.base import BaseVideoClassifier


class BaseQueryVideoAttack(BaseVideoAttack, ABC):
    """Base class for hard-label/query-driven attacks."""

    @abstractmethod
    def attack_query(
        self,
        model: BaseVideoClassifier,
        x: torch.Tensor,
        *,
        query_label: Callable[[torch.Tensor], int],
        input_format: str = "NTHWC",
        y: torch.Tensor | int | None = None,
        targeted: bool = False,
        video_path: str | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

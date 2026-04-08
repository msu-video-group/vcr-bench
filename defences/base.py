from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vg_videobench.models.base import BaseVideoClassifier

_ORIG_PREPROCESS_ATTR = "_defence_orig_preprocess_sampled"
_ORIG_PREDICT_ATTR = "_defence_orig_predict"


class BaseVideoDefence(ABC):
    """Base class for video defences.

    Each defence implements ``transform(x)`` which operates on an NTHWC tensor.
    The base class provides ``install(model)`` / ``uninstall(model)`` to wrap the
    model's ``_preprocess_sampled`` method (and optionally ``predict`` for voting)
    in the same style as the old benchmark's ``classifier.preprocess = wrapper(...)``.
    """

    voting_count: int = 1

    # ------------------------------------------------------------------
    # Core transformation — subclasses must implement this
    # ------------------------------------------------------------------

    @abstractmethod
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply defence transformation to a sampled video tensor.

        Args:
            x: Tensor of shape [N, T, H, W, C], float32, values in [0, 255].

        Returns:
            Defended tensor of the same shape.
        """
        ...

    # ------------------------------------------------------------------
    # Model installation / removal
    # ------------------------------------------------------------------

    def install(self, model: "BaseVideoClassifier") -> None:
        """Wrap model methods to apply this defence.

        - Always wraps ``model._preprocess_sampled`` to call ``transform`` before preprocessing.
        - If ``voting_count > 1``, also wraps ``model.predict`` to average over multiple calls.

        Stores originals on the model instance so ``uninstall`` can restore them.
        """
        # Wrap _preprocess_sampled
        orig_pre = model._preprocess_sampled
        object.__setattr__(model, _ORIG_PREPROCESS_ATTR, orig_pre)

        defence = self

        def _defended_preprocess_sampled(t: torch.Tensor, input_format: str = "NTHWC") -> torch.Tensor:
            orig_dtype = t.dtype
            t_float = t.float() if t.dtype != torch.float32 else t
            transformed = defence.transform(t_float)
            if orig_dtype != torch.float32:
                transformed = transformed.to(orig_dtype)
            return orig_pre(transformed, input_format)

        model._preprocess_sampled = _defended_preprocess_sampled  # type: ignore[method-assign]

        # Optionally wrap predict for voting
        if self.voting_count > 1:
            orig_predict = model.predict
            object.__setattr__(model, _ORIG_PREDICT_ATTR, orig_predict)
            n = self.voting_count

            def _voting_predict(x: torch.Tensor, **kwargs) -> torch.Tensor:
                results = [orig_predict(x, **kwargs) for _ in range(n)]
                stacked = torch.stack([r if isinstance(r, torch.Tensor) else r for r in results])
                return stacked.mean(0)

            model.predict = _voting_predict  # type: ignore[method-assign]

    def uninstall(self, model: "BaseVideoClassifier") -> None:
        """Restore model methods that were wrapped by ``install``."""
        orig_pre = getattr(model, _ORIG_PREPROCESS_ATTR, None)
        if orig_pre is not None:
            model._preprocess_sampled = orig_pre  # type: ignore[method-assign]
            try:
                object.__delattr__(model, _ORIG_PREPROCESS_ATTR)
            except AttributeError:
                pass

        orig_predict = getattr(model, _ORIG_PREDICT_ATTR, None)
        if orig_predict is not None:
            model.predict = orig_predict  # type: ignore[method-assign]
            try:
                object.__delattr__(model, _ORIG_PREDICT_ATTR)
            except AttributeError:
                pass

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
import torch
import urllib.request
from tqdm import tqdm

from .pipeline_config import ModelDataPipelineConfig, PipelineStage
from .pipeline_tools import (
    bind_stage_runtime_functions,
    make_pipeline_config,
    make_runtime_pipeline,
    preprocess_sampled_nthwc_from_config,
    sample_video_nthwc_from_loading_config,
)
from vg_videobench.utils.io import decode_video_nthwc
from vg_videobench.utils.preprocessing import to_format
from vg_videobench.types import PredictionBundle


class BaseVideoClassifier(ABC):
    raw_input_format = "NTHWC"
    preprocessed_format = "UNKNOWN"
    model_name = "unknown"
    MODEL_CAPABILITIES: dict[str, Any] = {"backbones": {}}
    grad_forward_chunk_size: int | None = None

    def train(self, mode: bool = True) -> "BaseVideoClassifier":
        module = getattr(self, "model", None)
        if hasattr(module, "train"):
            module.train(mode)
        return self

    def eval(self) -> "BaseVideoClassifier":
        return self.train(False)

    @classmethod
    def model_capabilities(cls) -> dict[str, Any]:
        caps = getattr(cls, "MODEL_CAPABILITIES", None)
        return dict(caps) if isinstance(caps, dict) else {"backbones": {}}

    @classmethod
    def _checkpoint_dir(cls) -> Path:
        module_path = Path(__import__(cls.__module__, fromlist=["__name__"]).__file__).resolve()
        ckpt_dir = module_path.parent / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir

    @classmethod
    def available_backbones(cls) -> list[str]:
        backs = cls.model_capabilities().get("backbones", {})
        return sorted(list(backs.keys())) if isinstance(backs, dict) else []

    @classmethod
    def _ensure_checkpoint_downloaded(cls, checkpoint_path: str, backbone: str, weights_dataset: str) -> str:
        path = Path(checkpoint_path)
        if path.exists():
            return str(path)
        spec = cls.get_weight_spec(backbone, weights_dataset)
        url = spec.get("checkpoint_url")
        if not url:
            raise FileNotFoundError(f"checkpoint not found and no download URL configured: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".part")
        try:
            with urllib.request.urlopen(str(url)) as resp:
                total = resp.headers.get("Content-Length")
                total_bytes = int(total) if total is not None else None
                with tmp_path.open("wb") as f, tqdm(
                    total=total_bytes,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"download {path.name}",
                    leave=True,
                ) as pbar:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))
            tmp_path.replace(path)
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to download checkpoint for {cls.model_name}/{backbone}/{weights_dataset} "
                f"from {url} -> {path}: {e}"
            ) from e
        return str(path)

    @classmethod
    def available_weight_datasets(cls, backbone: str | None = None) -> list[str]:
        caps = cls.model_capabilities()
        backs = caps.get("backbones", {})
        if not isinstance(backs, dict):
            return []
        if backbone is None:
            ds = set()
            for bspec in backs.values():
                if isinstance(bspec, dict):
                    ds_map = bspec.get("datasets", {})
                    if isinstance(ds_map, dict):
                        ds.update(ds_map.keys())
            return sorted(ds)
        bspec = backs.get(str(backbone))
        if not isinstance(bspec, dict):
            return []
        ds_map = bspec.get("datasets", {})
        return sorted(list(ds_map.keys())) if isinstance(ds_map, dict) else []

    @classmethod
    def get_weight_spec(cls, backbone: str, weights_dataset: str) -> dict[str, Any]:
        backs = cls.model_capabilities().get("backbones", {})
        if not isinstance(backs, dict) or backbone not in backs:
            raise ValueError(f"Unsupported backbone for {cls.__name__}: {backbone}")
        bspec = backs[backbone]
        if not isinstance(bspec, dict):
            raise ValueError(f"Invalid backbone spec for {cls.__name__}: {backbone}")
        ds_map = bspec.get("datasets", {})
        if not isinstance(ds_map, dict) or weights_dataset not in ds_map:
            raise ValueError(f"Unsupported weights dataset for {cls.__name__}/{backbone}: {weights_dataset}")
        spec = ds_map[weights_dataset]
        if not isinstance(spec, dict):
            raise ValueError(f"Invalid weights spec for {cls.__name__}/{backbone}/{weights_dataset}")
        return dict(spec)

    @classmethod
    def _default_checkpoint_path(cls, backbone: str, weights_dataset: str) -> str | None:
        spec = cls.get_weight_spec(backbone, weights_dataset)
        fname = spec.get("checkpoint_filename")
        if fname:
            return str(cls._checkpoint_dir() / str(fname))
        rel = spec.get("checkpoint_relpath")
        if rel:
            repo_root = Path(__file__).resolve().parents[2]
            return str(repo_root / Path(rel))
        return None

    @classmethod
    def resolve_checkpoint_path(
        cls,
        backbone: str,
        weights_dataset: str,
        checkpoint_path: str | None = None,
        *,
        auto_download: bool = False,
    ) -> str:
        resolved = checkpoint_path or cls._default_checkpoint_path(backbone, weights_dataset)
        if not resolved:
            raise ValueError(
                f"Checkpoint path is not configured for {cls.model_name}/{backbone}/{weights_dataset}"
            )
        if auto_download:
            resolved = cls._ensure_checkpoint_downloaded(resolved, backbone, weights_dataset)
        return resolved

    @staticmethod
    def _extract_checkpoint_mapping(
        checkpoint: Any,
        *,
        root_keys: tuple[str, ...] = ("state_dict", "model_state_dict", "model", "module"),
    ) -> dict[str, Any]:
        cur = checkpoint
        if isinstance(cur, dict):
            for key in root_keys:
                value = cur.get(key)
                if isinstance(value, dict):
                    cur = value
                    break
        if not isinstance(cur, dict):
            raise ValueError("Unsupported checkpoint format")
        return cur

    @classmethod
    def extract_tensor_state_dict(
        cls,
        checkpoint: Any,
        *,
        root_keys: tuple[str, ...] = ("state_dict", "model_state_dict", "model", "module"),
        strip_prefixes: tuple[str, ...] = ("module.",),
        ignore_prefixes: tuple[str, ...] = (),
        key_replacements: tuple[tuple[str, str], ...] = (),
    ) -> dict[str, torch.Tensor]:
        src = cls._extract_checkpoint_mapping(checkpoint, root_keys=root_keys)
        out: dict[str, torch.Tensor] = {}
        for k, v in src.items():
            if not torch.is_tensor(v):
                continue
            key = str(k)
            for prefix in strip_prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
            if any(key.startswith(prefix) for prefix in ignore_prefixes):
                continue
            for old, new in key_replacements:
                key = key.replace(old, new)
            out[key] = v
        return out

    def load_converted_checkpoint(
        self,
        checkpoint_path: str,
        *,
        strict: bool = False,
        root_keys: tuple[str, ...] = ("state_dict", "model_state_dict", "model", "module"),
        strip_prefixes: tuple[str, ...] = ("module.",),
        ignore_prefixes: tuple[str, ...] = (),
        key_replacements: tuple[tuple[str, str], ...] = (),
        required_keys: tuple[str, ...] = (),
        ignore_missing_suffixes: tuple[str, ...] = ("num_batches_tracked",),
    ) -> tuple[list[str], list[str]]:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = self.extract_tensor_state_dict(
            checkpoint,
            root_keys=root_keys,
            strip_prefixes=strip_prefixes,
            ignore_prefixes=ignore_prefixes,
            key_replacements=key_replacements,
        )
        missing, unexpected = self.model.load_state_dict(state_dict, strict=strict)
        missing_filtered = [k for k in missing if not any(k.endswith(sfx) for sfx in ignore_missing_suffixes)]
        if required_keys:
            missing_required = [k for k in required_keys if k in missing_filtered]
            if missing_required:
                raise RuntimeError(
                    f"{self.__class__.__name__} checkpoint load failed. Missing required keys: {missing_required[:10]}"
                )
        return list(missing_filtered), list(unexpected)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint from file. Override in subclass for custom loading logic."""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        # Handle common "module" wrapper (e.g., from DataParallel)
        state_dict = checkpoint.get("module", checkpoint)
        if not isinstance(state_dict, dict):
            raise ValueError(f"Checkpoint does not contain valid state dict: {checkpoint_path}")
        self.model.load_state_dict(state_dict, strict=True)

    def _load_weights(self, checkpoint_path: str | None = None) -> None:
        """Load model weights from checkpoint. Called during __init__ when load_weights=True."""
        if checkpoint_path is None:
            checkpoint_path = getattr(self, "checkpoint_path", None)
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required for loading weights")
        self._load_checkpoint(checkpoint_path)

    def load_video(self, path: str) -> torch.Tensor:
        # Public interface: load + subsample according to active pipeline config.
        full = self._load_video_decoded(path)
        return self._sample_video_public(full, input_format=self.raw_input_format)

    def preprocess_video(self, x: torch.Tensor, input_format: str = "NTHWC") -> torch.Tensor:
        # Public interface: accepts single/batch, sampled/raw-decoded, returns model-specific preprocessed tensor.
        if self._uses_legacy_preprocess_override():
            return type(self).preprocess(self, x, input_format=input_format)  # type: ignore[misc]
        if self._looks_preprocessed(x):
            return x
        if x.ndim == 6 and x.shape[-1] in (1, 3):
            pre_batch = [self._preprocess_single_video(x[i], input_format=input_format) for i in range(x.shape[0])]
            try:
                return torch.stack(pre_batch, dim=0)
            except RuntimeError:
                # Preserve caller-visible shape failure clearly; batching mismatch usually means variable spatial sizes.
                raise ValueError("Cannot stack preprocessed batch automatically; per-sample preprocessed shapes differ")
        return self._preprocess_single_video(x, input_format=input_format)

    def _uses_legacy_preprocess_override(self) -> bool:
        cls_preprocess = getattr(type(self), "preprocess", None)
        if cls_preprocess is None:
            return False
        if cls_preprocess is BaseVideoClassifier.__dict__.get("preprocess_video"):
            return False
        # If runtime pipeline functions exist, prefer the new pipeline path.
        return not hasattr(self, "preprocess_sampled_fn")

    def forward_preprocessed(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_preprocessed_with_grad(x, enable_grad=False)

    def _uses_legacy_forward_override(self) -> bool:
        cls_forward = getattr(type(self), "forward_preprocessed", None)
        base_forward = BaseVideoClassifier.__dict__.get("forward_preprocessed")
        return cls_forward is not None and cls_forward is not base_forward

    def forward_preprocessed_with_grad(self, x: torch.Tensor, *, enable_grad: bool = True) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected 5D preprocessed tensor, got {tuple(x.shape)}")

        target_fmt = self._target_preprocessed_format()
        x = self._adapt_preprocessed_tensor(x, target_fmt)

        device = getattr(self, "device", x.device)
        x = x.to(device, non_blocking=True)
        model = getattr(self, "model", None)
        if model is None:
            raise NotImplementedError(f"{self.__class__.__name__} must define self.model or override forward_preprocessed()")

        chunk_size = getattr(self, "grad_forward_chunk_size", None)
        should_chunk = isinstance(chunk_size, int) and chunk_size > 0 and x.ndim == 5 and x.shape[0] > chunk_size

        if enable_grad:
            if should_chunk:
                logits_chunks = []
                for start in range(0, int(x.shape[0]), int(chunk_size)):
                    x_chunk = x[start : start + int(chunk_size)]
                    cur = model(x_chunk)
                    if isinstance(cur, tuple):
                        cur = cur[0]
                    if cur.ndim == 1:
                        cur = cur.unsqueeze(0)
                    logits_chunks.append(cur)
                logits = torch.cat(logits_chunks, dim=0)
            else:
                logits = model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
        else:
            with torch.no_grad():
                if should_chunk:
                    logits_chunks = []
                    for start in range(0, int(x.shape[0]), int(chunk_size)):
                        x_chunk = x[start : start + int(chunk_size)]
                        cur = model(x_chunk)
                        if isinstance(cur, tuple):
                            cur = cur[0]
                        if cur.ndim == 1:
                            cur = cur.unsqueeze(0)
                        logits_chunks.append(cur)
                    logits = torch.cat(logits_chunks, dim=0)
                else:
                    logits = model(x)
                    if isinstance(logits, tuple):
                        logits = logits[0]
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        return logits

    def _probs_from_logits_like(self, logits_or_probs: torch.Tensor) -> torch.Tensor:
        x = logits_or_probs
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2:
            raise ValueError(f"Expected [V,C] or [C], got shape {tuple(logits_or_probs.shape)}")

        row_sums = x.sum(dim=-1)
        looks_like_probs = bool(
            torch.all(x >= 0).item()
            and torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4)
        )
        probs = x if looks_like_probs else torch.softmax(x, dim=-1)
        return probs

    def _aggregate_probs(self, probs: torch.Tensor) -> torch.Tensor:
        if probs.ndim == 1:
            return probs
        return probs.mean(dim=0)

    def _ensure_single_video_probs(self, probs: torch.Tensor) -> torch.Tensor:
        if probs.ndim == 1:
            return probs
        if probs.ndim == 2 and probs.shape[0] == 1:
            return probs[0]
        raise ValueError(f"Expected single-video probs as [C], got shape {tuple(probs.shape)}")

    def _target_preprocessed_format(self) -> str:
        try:
            cfg = self.build_pipeline_config(getattr(self, "_active_stage", "test"))
            return cfg.preprocessing.preprocessed_format
        except Exception:
            return getattr(self, "preprocessed_format", "UNKNOWN")

    def _adapt_preprocessed_tensor(self, x: torch.Tensor, target_format: str) -> torch.Tensor:
        if target_format == "NCTHW":
            if x.shape[1] == 3:
                return x
            if x.ndim == 5 and x.shape[2] == 3:
                return to_format(x, "NTCHW", "NCTHW")
        if target_format == "NTCHW":
            if x.shape[2] == 3:
                return x
            if x.ndim == 5 and x.shape[1] == 3:
                return to_format(x, "NCTHW", "NTCHW")
        return x

    def build_data_pipeline(self, stage: PipelineStage = "test"):
        cfg = self.build_pipeline_config(stage)
        self._bind_stage_functions(stage)
        if all(hasattr(self, name) for name in ("load_video_fn", "sample_video_fn", "preprocess_sampled_fn")):
            return make_runtime_pipeline(
                cfg,
                load_video_fn=self.load_video_fn,
                sample_video_fn=self.sample_video_fn,
                preprocess_sampled_fn=self.preprocess_sampled_fn,
            )
        # Fallback for legacy/simple models that expose methods directly.
        return self

    def build_pipeline_config(self, stage: PipelineStage = "test") -> ModelDataPipelineConfig:
        loading_configs = getattr(self, "loading_configs", None)
        preprocessing_configs = getattr(self, "preprocessing_configs", None)
        if not isinstance(loading_configs, dict) or not isinstance(preprocessing_configs, dict):
            raise NotImplementedError(
                f"{self.__class__.__name__} must define loading_configs/preprocessing_configs "
                "or override build_pipeline_config()"
            )
        key = str(stage).lower()
        if key not in loading_configs or key not in preprocessing_configs:
            raise ValueError(f"Unsupported pipeline stage for {self.__class__.__name__}: {stage}")
        return make_pipeline_config(key, loading_configs[key], preprocessing_configs[key])

    def _decode_video_raw(self, path: str) -> torch.Tensor:
        return decode_video_nthwc(path, return_info=False)

    def _load_video_decoded(self, path: str) -> torch.Tensor:
        if hasattr(self, "load_video_fn"):
            return self.load_video_fn(path)
        pipeline = self.build_data_pipeline(getattr(self, "_active_stage", "test"))
        if hasattr(pipeline, "load_video"):
            return pipeline.load_video(path)
        return self._decode_video_raw(path)

    def _sample_video_with_cfg(self, x: torch.Tensor, input_format: str, cfg: ModelDataPipelineConfig):
        return sample_video_nthwc_from_loading_config(x, input_format=input_format, loading_cfg=cfg.loading)

    def _preprocess_sampled_with_cfg(self, x: torch.Tensor, input_format: str, cfg: ModelDataPipelineConfig) -> torch.Tensor:
        return preprocess_sampled_nthwc_from_config(
            x,
            input_format=input_format,
            preprocessing_cfg=cfg.preprocessing,
        )

    def _make_load_video_factory(self):
        return lambda cfg: (lambda path: self._decode_video_raw(path))

    def _make_sample_video_factory(self):
        return lambda cfg: (lambda x, input_format="NTHWC": self._sample_video_with_cfg(x, input_format, cfg))

    def _make_preprocess_sampled_factory(self):
        return lambda cfg: (
            lambda x, input_format="NTHWC": self._preprocess_sampled_with_cfg(x, input_format, cfg)
        )

    def _bind_stage_functions(self, stage: PipelineStage) -> None:
        load_factory = self._make_load_video_factory()
        sample_factory = self._make_sample_video_factory()
        preprocess_factory = self._make_preprocess_sampled_factory()
        bind_stage_runtime_functions(
            self,
            str(stage),
            build_pipeline_config_fn=lambda s: self.build_pipeline_config(s),  # type: ignore[arg-type]
            load_video_factory=load_factory,
            sample_video_factory=sample_factory,
            preprocess_sampled_factory=preprocess_factory,
        )

    def _sample_raw(self, x: torch.Tensor, input_format: str = "NTHWC") -> torch.Tensor:
        if hasattr(self, "sample_video_fn"):
            sampled = self.sample_video_fn(x, input_format)
            return sampled[0] if isinstance(sampled, tuple) else sampled
        pipeline = self.build_data_pipeline(getattr(self, "_active_stage", "test"))
        if hasattr(pipeline, "sample_video"):
            sampled = pipeline.sample_video(x, input_format=input_format)
            return sampled[0] if isinstance(sampled, tuple) else sampled
        if hasattr(pipeline, "sample_raw"):
            return pipeline.sample_raw(x, input_format=input_format)
        return x

    def _sample_video_public(self, x: torch.Tensor, input_format: str = "NTHWC") -> torch.Tensor:
        sampled = self._sample_raw(x, input_format=input_format)
        return sampled

    def _clip_len_for_active_stage(self) -> int | None:
        try:
            return int(self.build_pipeline_config(getattr(self, "_active_stage", "test")).loading.clip_len)
        except Exception:
            return None

    def _num_clips_for_active_stage(self) -> int | None:
        try:
            return int(self.build_pipeline_config(getattr(self, "_active_stage", "test")).loading.num_clips)
        except Exception:
            return None

    def _looks_preprocessed(self, x: torch.Tensor) -> bool:
        target = self._target_preprocessed_format().upper()
        if x.ndim == 5:
            if target == "NCTHW":
                return x.shape[1] == 3
            if target == "NTCHW":
                return x.shape[2] == 3
        if x.ndim == 6:
            # Batched preprocessed: [B,V,C,T,H,W] or [B,V,T,C,H,W]
            if target == "NCTHW":
                return x.shape[2] == 3
            if target == "NTCHW":
                return x.shape[3] == 3
        return False

    def _looks_batched_sampled(self, x: torch.Tensor) -> bool:
        # canonical sampled batch: [B, Nclips, T, H, W, C]
        if not (x.ndim == 6 and x.shape[-1] in (1, 3)) or self._looks_preprocessed(x):
            return False
        nclips = self._num_clips_for_active_stage()
        clip_len = self._clip_len_for_active_stage()
        if nclips is not None and int(x.shape[1]) != int(nclips):
            return False
        if clip_len is not None and int(x.shape[2]) != int(clip_len):
            return False
        return True

    def _is_likely_sampled_single(self, x: torch.Tensor) -> bool:
        if x.ndim != 5 or x.shape[-1] not in (1, 3):
            return False
        nclips = self._num_clips_for_active_stage()
        clip_len = self._clip_len_for_active_stage()
        if nclips is not None and x.shape[0] == nclips:
            return True
        if clip_len is not None and x.shape[1] == clip_len and x.shape[0] != 1:
            return True
        if clip_len is not None and x.shape[0] == 1 and x.shape[1] == clip_len and (nclips == 1):
            return True
        return False

    def _preprocess_single_video(self, x: torch.Tensor, input_format: str = "NTHWC") -> torch.Tensor:
        if self._looks_preprocessed(x):
            return x
        # If it's not clearly sampled, treat it as decoded/raw and subsample first.
        sampled = x if self._is_likely_sampled_single(x) else self._sample_video_public(x, input_format=input_format)
        return self._preprocess_sampled(sampled, input_format="NTHWC")

    def _preprocess_sampled(self, x: torch.Tensor, input_format: str = "NTHWC") -> torch.Tensor:
        if hasattr(self, "preprocess_sampled_fn"):
            return self.preprocess_sampled_fn(x, input_format)
        pipeline = self.build_data_pipeline(getattr(self, "_active_stage", "test"))
        if hasattr(pipeline, "preprocess_sampled"):
            return pipeline.preprocess_sampled(x, input_format=input_format)
        if hasattr(pipeline, "preprocess"):
            return pipeline.preprocess(x, input_format=input_format)
        raise NotImplementedError(f"{self.__class__.__name__} does not provide preprocess_sampled_fn/preprocess()")

    def predict(
        self,
        x: torch.Tensor,
        *,
        input_format: str = "NTHWC",
        preprocessed: bool | None = None,
        return_full: bool = False,
        enable_grad: bool = False,
    ) -> torch.Tensor | PredictionBundle:
        # Public interface: auto-preprocess if needed, handle single and batch.
        if preprocessed is True:
            # Backward-compatible override.
            if x.ndim == 6:
                if return_full:
                    raise ValueError("return_full=True is only supported for single-video predict()")
                return self._predict_preprocessed_batch(x)
            pre_x = x
            logits = (
                self.forward_preprocessed(pre_x)
                if (not enable_grad and self._uses_legacy_forward_override())
                else self.forward_preprocessed_with_grad(pre_x, enable_grad=enable_grad)
            )
            probs_views = self._probs_from_logits_like(logits)
            probs = self._ensure_single_video_probs(self._aggregate_probs(probs_views))
            pred_label = int(torch.argmax(probs).item())
            if return_full:
                return PredictionBundle(
                    logits=logits.detach(),
                    probs=probs.detach(),
                    pred_label=pred_label,
                    metadata={"model_name": self.model_name, "raw_input_format": self.raw_input_format, "preprocessed_format": self.preprocessed_format},
                )
            return probs
        if self._looks_batched_sampled(x) or (x.ndim == 6 and self._looks_preprocessed(x)):
            probs_batch = self._predict_batch(x, input_format=input_format)
            if return_full:
                raise ValueError("return_full=True is only supported for single-video predict()")
            return probs_batch

        pre_x = x if self._looks_preprocessed(x) else self.preprocess_video(x, input_format=input_format)
        if pre_x.ndim == 6:
            if return_full:
                raise ValueError("return_full=True is only supported for single-video predict()")
            return self._predict_preprocessed_batch(pre_x)
        logits = (
            self.forward_preprocessed(pre_x)
            if (not enable_grad and self._uses_legacy_forward_override())
            else self.forward_preprocessed_with_grad(pre_x, enable_grad=enable_grad)
        )
        probs_views = self._probs_from_logits_like(logits)
        probs = self._ensure_single_video_probs(self._aggregate_probs(probs_views))
        pred_label = int(torch.argmax(probs).item())

        if return_full:
            return PredictionBundle(
                logits=logits.detach(),
                probs=probs.detach(),
                pred_label=pred_label,
                metadata={
                    "model_name": self.model_name,
                    "raw_input_format": self.raw_input_format,
                    "preprocessed_format": self.preprocessed_format,
                },
            )
        return probs

    def __call__(
        self,
        x: torch.Tensor,
        *,
        input_format: str = "NTHWC",
        preprocessed: bool | None = None,
        return_full: bool = False,
        enable_grad: bool = False,
    ) -> torch.Tensor | PredictionBundle:
        return self.predict(
            x,
            input_format=input_format,
            preprocessed=preprocessed,
            return_full=return_full,
            enable_grad=enable_grad,
        )

    def predict_label(
        self,
        x: torch.Tensor,
        *,
        input_format: str = "NTHWC",
        preprocessed: bool | None = None,
    ) -> int:
        probs = self.predict(
            x,
            input_format=input_format,
            preprocessed=preprocessed,
            return_full=False,
            enable_grad=False,
        )
        if isinstance(probs, PredictionBundle):
            return int(probs.pred_label)
        if probs.ndim != 1:
            probs = probs.reshape(-1)
        return int(torch.argmax(probs).item())

    def _predict_sampled(
        self,
        x: torch.Tensor,
        *,
        input_format: str = "NTHWC",
        return_full: bool = False,
    ) -> torch.Tensor | PredictionBundle:
        pre_x = self._preprocess_sampled(x, input_format=input_format)
        logits = self.forward_preprocessed(pre_x)
        probs_views = self._probs_from_logits_like(logits)
        probs = self._ensure_single_video_probs(self._aggregate_probs(probs_views))
        pred_label = int(torch.argmax(probs).item())
        if return_full:
            return PredictionBundle(
                logits=logits.detach(),
                probs=probs.detach(),
                pred_label=pred_label,
                metadata={
                    "model_name": self.model_name,
                    "raw_input_format": self.raw_input_format,
                    "preprocessed_format": self.preprocessed_format,
                    "sampled_input": True,
                },
            )
        return probs

    def _predict_preprocessed_batch(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(f"Expected batched preprocessed tensor, got shape {tuple(x.shape)}")
        # Common case: [B, V, ...] view dimension in dim=1 for multi-view inference.
        if x.ndim == 6:
            b, v = x.shape[:2]
            flat = x.reshape(b * v, *x.shape[2:])
            logits = self.forward_preprocessed(flat)
            probs_flat = self._probs_from_logits_like(logits)
            if probs_flat.shape[0] == b * v:
                probs = probs_flat.reshape(b, v, -1).mean(dim=1)
                return probs
        # Default generic path: iterate samples in a batch and stack final probabilities.
        outputs = []
        for i in range(x.shape[0]):
            probs = self.predict(x[i], return_full=False)
            outputs.append(probs)
        return torch.stack(outputs, dim=0)

    def _predict_sampled_batch(self, x: torch.Tensor, *, input_format: str = "NTHWC") -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(f"Expected batched sampled tensor, got shape {tuple(x.shape)}")
        if x.ndim == 6:
            pre_batch = []
            for i in range(x.shape[0]):
                pre_batch.append(self.preprocess_video(x[i], input_format=input_format))
            try:
                pre = torch.stack(pre_batch, dim=0)
            except RuntimeError:
                pre = None
            if pre is not None:
                return self._predict_preprocessed_batch(pre)
        outputs = []
        for i in range(x.shape[0]):
            probs = self.predict(x[i], input_format=input_format, return_full=False)
            outputs.append(probs)
        return torch.stack(outputs, dim=0)

    def _predict_batch(
        self,
        x: torch.Tensor,
        *,
        input_format: str = "NTHWC",
        preprocessed: bool | None = None,
        sampled: bool | None = None,
    ) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(f"Expected batched tensor, got shape {tuple(x.shape)}")
        if preprocessed is True:
            return self._predict_preprocessed_batch(x)
        if sampled is True:
            return self._predict_sampled_batch(x, input_format=input_format)
        if self._looks_preprocessed(x):
            return self._predict_preprocessed_batch(x)
        if self._looks_batched_sampled(x):
            return self._predict_sampled_batch(x, input_format=input_format)
        outputs = []
        for i in range(x.shape[0]):
            outputs.append(self.predict(x[i], input_format=input_format, return_full=False))
        return torch.stack(outputs, dim=0)

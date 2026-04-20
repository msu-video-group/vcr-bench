from __future__ import annotations

from typing import Any

import clip
import torch
import torch.nn as nn
from clip.model import CLIP, Transformer, build_model as build_clip_model

from ..base import BaseVideoClassifier
from ..ila.model import _load_labels
from ..pipeline_config import PipelineStage


class TransformerAdapter(nn.Module):
    def __init__(self, clip_model: nn.Module, num_segs: int, num_layers: int = 6) -> None:
        super().__init__()
        self.num_segs = int(num_segs)
        embed_dim = int(clip_model.text_projection.shape[1])
        transformer_width = int(clip_model.ln_final.weight.shape[0])
        transformer_heads = transformer_width // 64
        self.frame_position_embeddings = nn.Embedding(self.num_segs, embed_dim)
        self.transformer = Transformer(width=embed_dim, layers=num_layers, heads=transformer_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = x.size()
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        embeddings = self.frame_position_embeddings(position_ids)
        x_original = x
        x = x + embeddings.unsqueeze(0)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        x = x.type(x_original.dtype) + x_original
        return x.mean(dim=1)


class _ActionClipModel(nn.Module):
    def __init__(self, clip_model: nn.Module, adapter: nn.Module, prompt_tokens: torch.Tensor, num_prompt: int) -> None:
        super().__init__()
        self.clip = clip_model
        self.adapter = adapter
        self.register_buffer("prompt_tokens", prompt_tokens, persistent=False)
        self.num_prompt = int(num_prompt)
        self._cached_text_features: torch.Tensor | None = None

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        batch_size, num_segs, channels, height, width = video.shape
        video = video.view(-1, channels, height, width)
        frame_features = self.clip.encode_image(video)
        frame_features = frame_features.view(batch_size, num_segs, -1)
        return self.adapter(frame_features)

    def encode_text(self) -> torch.Tensor:
        if self._cached_text_features is not None and self._cached_text_features.device == self.prompt_tokens.device:
            return self._cached_text_features
        text_features = self.clip.encode_text(self.prompt_tokens)
        self._cached_text_features = text_features
        return text_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        video_features = self.encode_video(inputs)
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        text_features = self.encode_text()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * video_features @ text_features.T).view(inputs.shape[0], self.num_prompt, -1)
        return similarity.softmax(dim=2).mean(dim=1)


class ActionClipClassifier(BaseVideoClassifier):
    model_name = "actionclip"
    raw_input_format = "NTHWC"
    preprocessed_format = "NTCHW"
    MODEL_CAPABILITIES: dict[str, Any] = {
        "backbones": {
            "vit_b16": {
                "display_name": "ActionCLIP ViT-B/16",
                "datasets": {
                    "kinetics400": {
                        "num_classes": 400,
                        "checkpoint_url": (
                            "https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/"
                            "actionclip_vit-base-p16-res224-clip-pre_1x1x32_k400-rgb/vit-b-16-32f.pth"
                        ),
                        "checkpoint_filename": "actionclip_vit_b16_32f_kinetics400_mmaction.pth",
                    }
                },
            }
        }
    }

    def __init__(
        self,
        checkpoint_path: str | None = None,
        backbone: str | None = None,
        weights_dataset: str | None = None,
        device: str = "cuda",
        grad_forward_chunk_size: int | None = None,
        load_weights: bool | None = None,
        num_classes: int | None = None,
        auto_download: bool = True,
    ) -> None:
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.backbone = backbone or "vit_b16"
        self.weights_dataset = weights_dataset or "kinetics400"
        if grad_forward_chunk_size is not None:
            self.grad_forward_chunk_size = int(grad_forward_chunk_size)
        weight_spec = self.get_weight_spec(self.backbone, self.weights_dataset)
        self.num_classes = int(num_classes if num_classes is not None else weight_spec.get("num_classes", 400))
        self.loading_configs, self.preprocessing_configs = self._build_stage_config_dicts()
        self._active_stage: PipelineStage = "test"  # type: ignore[assignment]
        self._bind_stage_functions("test")
        self.prompt_templates = (
            "a photo of action {}",
            "a picture of action {}",
            "Human action of {}",
            "{}, an action",
            "{}, a video of action",
            "Playing action of {}",
            "{}",
            "Doing a kind of action, {}",
            "Look, the human is {}",
            "Can you recognize the action of {}?",
            "Video classification of {}",
            "A video of {}",
            "The man is {}",
            "The woman is {}",
        )
        self.prompt_tokens = self._build_prompt_tokens()
        self.model = self._build_model()
        should_load = bool(self._default_checkpoint_path(self.backbone, self.weights_dataset) or checkpoint_path) if load_weights is None else bool(load_weights)
        if should_load:
            self.checkpoint_path = self.resolve_checkpoint_path(
                self.backbone,
                self.weights_dataset,
                checkpoint_path,
                auto_download=auto_download,
            )
            if not self.checkpoint_path:
                raise ValueError("checkpoint_path is required when load_weights=True")
            self._load_checkpoint(self.checkpoint_path)
        else:
            self.checkpoint_path = checkpoint_path
        self.model.eval().to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

    def _build_prompt_tokens(self) -> torch.Tensor:
        labels = _load_labels(self.num_classes)
        prompts = [template.format(label) for template in self.prompt_templates for label in labels]
        return clip.tokenize(prompts)

    def _build_clip(self) -> nn.Module:
        return CLIP(
            512,
            224,
            12,
            768,
            16,
            77,
            49408,
            512,
            8,
            12,
        )

    def _build_model(self) -> nn.Module:
        clip_model = self._build_clip()
        adapter = TransformerAdapter(clip_model, num_segs=32, num_layers=6)
        return _ActionClipModel(
            clip_model=clip_model,
            adapter=adapter,
            prompt_tokens=self.prompt_tokens,
            num_prompt=len(self.prompt_templates),
        )

    def _build_stage_config_dicts(self) -> tuple[dict[str, dict], dict[str, dict]]:
        common_loading = {
            "raw_input_format": "NTHWC",
            "sampled_format": "NTHWC",
            "clip_len": 1,
            "frame_interval": 1,
            "full_videos": False,
        }
        common_pre = {
            "preprocessed_format": "NTCHW",
            "resize_short": 256,
            "crop_size": 224,
            "input_range": 1.0,
            "mean": (122.771, 116.746, 104.093),
            "std": (68.5, 66.632, 70.323),
        }
        loading = {
            "train": {**common_loading, "num_clips": 32},
            "val": {**common_loading, "num_clips": 32},
            "test": {**common_loading, "num_clips": 32},
            "attack": {**common_loading, "num_clips": 32},
        }
        preprocessing = {
            "train": {**common_pre, "spatial_strategy": "center_crop"},
            "val": {**common_pre, "spatial_strategy": "center_crop"},
            "test": {**common_pre, "spatial_strategy": "center_crop"},
            "attack": {**common_pre, "spatial_strategy": "center_crop"},
        }
        return loading, preprocessing

    @staticmethod
    def _normalize_checkpoint_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                checkpoint = checkpoint["state_dict"]
            elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
                checkpoint = checkpoint["model"]
        if not isinstance(checkpoint, dict):
            raise TypeError("Unsupported ActionCLIP checkpoint format")
        state_dict: dict[str, torch.Tensor] = {}
        for key, value in checkpoint.items():
            if not isinstance(value, torch.Tensor):
                continue
            normalized_key = key[7:] if key.startswith("module.") else key
            state_dict[normalized_key] = value
        return state_dict

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = self._normalize_checkpoint_state_dict(checkpoint)
        clip_state_dict = {
            key.removeprefix("clip."): value
            for key, value in state_dict.items()
            if key.startswith("clip.")
        }
        adapter_state_dict = {
            key.removeprefix("adapter."): value
            for key, value in state_dict.items()
            if key.startswith("adapter.")
        }
        if not clip_state_dict or not adapter_state_dict:
            raise RuntimeError("ActionCLIP checkpoint is missing clip or adapter weights")
        clip_model = build_clip_model(clip_state_dict)
        num_segs = int(adapter_state_dict["frame_position_embeddings.weight"].shape[0])
        num_layers = len(
            {
                key.split(".")[2]
                for key in adapter_state_dict
                if key.startswith("transformer.resblocks.") and key.endswith(".attn.in_proj_weight")
            }
        )
        adapter = TransformerAdapter(clip_model, num_segs=num_segs, num_layers=num_layers)
        adapter.load_state_dict(adapter_state_dict, strict=True)
        self.model = _ActionClipModel(
            clip_model=clip_model,
            adapter=adapter,
            prompt_tokens=self.prompt_tokens,
            num_prompt=len(self.prompt_templates),
        )


MODEL_CLASS = ActionClipClassifier


def create(**kwargs) -> ActionClipClassifier:
    return ActionClipClassifier(**kwargs)

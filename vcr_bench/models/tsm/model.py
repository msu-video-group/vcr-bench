from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torchvision.models import resnet50

from ..base import BaseVideoClassifier
from ..pipeline_config import PipelineStage
from ..vendor_mmaction.resnet3d import NonLocal3d


class _TSMHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, num_segments: int, dropout_ratio: float = 0.5) -> None:
        super().__init__()
        self.num_segments = int(num_segments)
        self.dropout = nn.Dropout(p=dropout_ratio) if dropout_ratio and dropout_ratio > 0 else nn.Identity()
        self.fc_cls = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"TSM head expected [N*T, C] features, got {tuple(x.shape)}")
        x = self.dropout(x)
        cls_score = self.fc_cls(x)
        cls_score = cls_score.view((-1, self.num_segments, cls_score.shape[-1])).mean(dim=1)
        return cls_score


class _TemporalShift:
    @staticmethod
    def shift(x: torch.Tensor, num_segments: int, shift_div: int = 8) -> torch.Tensor:
        n, c, h, w = x.size()
        x = x.view(-1, num_segments, c, h * w)
        fold = c // shift_div

        left_split = x[:, :, :fold, :]
        mid_split = x[:, :, fold : 2 * fold, :]
        right_split = x[:, :, 2 * fold :, :]

        zeros = left_split - left_split
        blank = zeros[:, :1, :, :]
        left_split = torch.cat((left_split[:, 1:, :, :], blank), dim=1)

        zeros = mid_split - mid_split
        blank = zeros[:, :1, :, :]
        mid_split = torch.cat((blank, mid_split[:, :-1, :, :]), dim=1)

        out = torch.cat((left_split, mid_split, right_split), dim=2)
        return out.view(n, c, h, w)


class _TSMResNetBackbone(nn.Module):
    def __init__(
        self,
        num_segments: int = 8,
        shift_div: int = 8,
        non_local_stages: tuple[tuple[int, ...], ...] | None = None,
        non_local_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        base = resnet50(weights=None)
        self.num_segments = int(num_segments)
        self.shift_div = int(shift_div)
        self.non_local_stages = non_local_stages or tuple()
        self.non_local_cfg = dict(non_local_cfg or {})
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self._nl_layers = nn.ModuleDict()
        if self.non_local_stages:
            for stage_idx, stage in enumerate((self.layer1, self.layer2, self.layer3, self.layer4), start=1):
                cfg = self.non_local_stages[stage_idx - 1]
                if sum(cfg) == 0:
                    continue
                for block_idx, enabled in enumerate(cfg):
                    if enabled:
                        block = stage[block_idx]
                        self._nl_layers[f"layer{stage_idx}_{block_idx}"] = NonLocal3d(
                            block.bn3.num_features,
                            **self.non_local_cfg,
                        )

    def _apply_non_local(self, stage_idx: int, block_idx: int, x: torch.Tensor) -> torch.Tensor:
        key = f"layer{stage_idx}_{block_idx}"
        if key not in self._nl_layers:
            return x
        nl = self._nl_layers[key]
        n, c, h, w = x.size()
        x = x.view(n // self.num_segments, self.num_segments, c, h, w).transpose(1, 2).contiguous()
        x = nl(x)
        return x.transpose(1, 2).contiguous().view(n, c, h, w)

    def _forward_bottleneck(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = block.conv1(_TemporalShift.shift(x, self.num_segments, self.shift_div))
        out = block.bn1(out)
        out = block.relu(out)

        out = block.conv2(out)
        out = block.bn2(out)
        out = block.relu(out)

        out = block.conv3(out)
        out = block.bn3(out)

        if block.downsample is not None:
            identity = block.downsample(x)

        out += identity
        out = block.relu(out)
        return out

    def _forward_layer(self, layer: nn.Sequential, x: torch.Tensor, stage_idx: int) -> torch.Tensor:
        for block_idx, block in enumerate(layer):
            x = self._forward_bottleneck(block, x)
            x = self._apply_non_local(stage_idx, block_idx, x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self._forward_layer(self.layer1, x, 1)
        x = self._forward_layer(self.layer2, x, 2)
        x = self._forward_layer(self.layer3, x, 3)
        x = self._forward_layer(self.layer4, x, 4)

        x = self.avgpool(x)
        return torch.flatten(x, 1)


class _SimpleTSMRecognizer(nn.Module):
    def __init__(
        self,
        num_classes: int = 400,
        num_segments: int = 8,
        shift_div: int = 8,
        non_local_stages: tuple[tuple[int, ...], ...] | None = None,
        non_local_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.backbone = _TSMResNetBackbone(
            num_segments=num_segments,
            shift_div=shift_div,
            non_local_stages=non_local_stages,
            non_local_cfg=non_local_cfg,
        )
        self.cls_head = _TSMHead(in_channels=2048, num_classes=num_classes, num_segments=num_segments, dropout_ratio=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"TSM expected 5D input, got {tuple(x.shape)}")
        if x.shape[2] == 3:  # NTCHW
            n, t, c, h, w = x.shape
            x_4d = x.reshape(n * t, c, h, w)
            feats = self.backbone(x_4d)
            return self.cls_head(feats)
        if x.shape[1] == 3:  # NCTHW
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            return self.forward(x)
        raise ValueError(f"TSM unsupported input layout for shape {tuple(x.shape)}")


class TSMClassifier(BaseVideoClassifier):
    model_name = "tsm"
    raw_input_format = "NTHWC"
    preprocessed_format = "NTCHW"
    MODEL_CAPABILITIES: dict[str, Any] = {
        "backbones": {
            "r50": {
                "display_name": "ResNet-50 TSM",
                "datasets": {
                    "kinetics400": {
                        "num_classes": 400,
                        "checkpoint_url": (
                            "https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/"
                            "tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_kinetics400-rgb/"
                            "tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_kinetics400-rgb_20220831-042b1748.pth"
                        ),
                        "checkpoint_filename": "tsm_r50_kinetics400_mmaction.pth",
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
        self.backbone = backbone or "r50"
        self.weights_dataset = weights_dataset or "kinetics400"
        if grad_forward_chunk_size is not None:
            self.grad_forward_chunk_size = int(grad_forward_chunk_size)
        weight_spec = self.get_weight_spec(self.backbone, self.weights_dataset)
        self.num_classes = int(num_classes if num_classes is not None else weight_spec.get("num_classes", 400))
        self.loading_configs, self.preprocessing_configs = self._build_stage_config_dicts()
        self._active_stage: PipelineStage = "test"  # type: ignore[assignment]
        self._bind_stage_functions("test")

        self.model = _SimpleTSMRecognizer(
            num_classes=self.num_classes,
            num_segments=self._num_segments(),
            shift_div=8,
            non_local_stages=self._non_local_stages(),
            non_local_cfg=self._non_local_cfg(),
        )

        should_load = True if load_weights is None else bool(load_weights)
        if should_load:
            self.checkpoint_path = self.resolve_checkpoint_path(
                self.backbone,
                self.weights_dataset,
                checkpoint_path,
                auto_download=auto_download,
            )
            self._load_checkpoint(self.checkpoint_path)
        else:
            self.checkpoint_path = checkpoint_path

        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False

    def _num_segments(self) -> int:
        return 16

    def _non_local_stages(self) -> tuple[tuple[int, ...], ...] | None:
        return None

    def _non_local_cfg(self) -> dict | None:
        return None

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
            "mean": (123.675, 116.28, 103.53),
            "std": (58.395, 57.12, 57.375),
        }
        loading = {
            "train": {**common_loading, "num_clips": self._num_segments()},
            "val": {**common_loading, "num_clips": self._num_segments()},
            "test": {**common_loading, "num_clips": self._num_segments()},
            "attack": {**common_loading, "num_clips": self._num_segments()},
        }
        preprocessing = {
            "train": {**common_pre, "spatial_strategy": "train_random_crop"},
            "val": {**common_pre, "spatial_strategy": "center_crop"},
            "test": {**common_pre, "spatial_strategy": "ten_crop"},
            "attack": {**common_pre, "spatial_strategy": "center_crop"},
        }
        return loading, preprocessing

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        self.load_converted_checkpoint(
            checkpoint_path,
            strict=False,
            ignore_prefixes=("data_preprocessor.",),
            key_replacements=(
                ("model.backbone.", "backbone."),
                ("model.cls_head.", "cls_head."),
                (".downsample.conv.", ".downsample.0."),
                (".downsample.bn.", ".downsample.1."),
                (".conv1.conv.net.", ".conv1."),
                (".conv1.conv.", ".conv1."),
                (".conv1.bn.", ".bn1."),
                (".conv2.conv.", ".conv2."),
                (".conv2.bn.", ".bn2."),
                (".conv3.conv.", ".conv3."),
                (".conv3.bn.", ".bn3."),
            ),
            required_keys=("cls_head.fc_cls.weight", "cls_head.fc_cls.bias"),
        )


MODEL_CLASS = TSMClassifier


def create(**kwargs) -> TSMClassifier:
    return TSMClassifier(**kwargs)

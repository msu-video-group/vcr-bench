from __future__ import annotations

from typing import Any

from ..tsm.model import TSMClassifier


class TSMNonLocalClassifier(TSMClassifier):
    model_name = "tsmnonlocal"
    MODEL_CAPABILITIES: dict[str, Any] = {
        "backbones": {
            "r50": {
                "display_name": "TSM R50 NonLocal",
                "datasets": {
                    "kinetics400": {
                        "num_classes": 400,
                        "checkpoint_url": (
                            "https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/"
                            "tsm_imagenet-pretrained-r50-nl-embedded-gaussian_8xb16-1x1x8-50e_kinetics400-rgb/"
                            "tsm_imagenet-pretrained-r50-nl-embedded-gaussian_8xb16-1x1x8-50e_kinetics400-rgb_20220831-35eddb57.pth"
                        ),
                        "checkpoint_filename": "tsmnonlocal_r50_kinetics400_mmaction.pth",
                    }
                },
            }
        }
    }

    def _num_segments(self) -> int:
        return 8

    def _non_local_stages(self) -> tuple[tuple[int, ...], ...]:
        return (
            (0, 0, 0),
            (1, 0, 1, 0),
            (1, 0, 1, 0, 1, 0),
            (0, 0, 0),
        )

    def _non_local_cfg(self) -> dict:
        return dict(
            sub_sample=True,
            use_scale=False,
            norm_cfg=dict(type="BN3d", requires_grad=True),
            mode="embedded_gaussian",
        )

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        self.load_converted_checkpoint(
            checkpoint_path,
            strict=False,
            ignore_prefixes=("data_preprocessor.",),
            key_replacements=(
                ("model.backbone.", "backbone."),
                ("model.cls_head.", "cls_head."),
                # Non-local prefix: checkpoint embeds NL inside ResNet blocks,
                # model stores them in _nl_layers ModuleDict
                ("backbone.layer2.0.non_local_block.", "backbone._nl_layers.layer2_0."),
                ("backbone.layer2.2.non_local_block.", "backbone._nl_layers.layer2_2."),
                ("backbone.layer3.0.non_local_block.", "backbone._nl_layers.layer3_0."),
                ("backbone.layer3.2.non_local_block.", "backbone._nl_layers.layer3_2."),
                ("backbone.layer3.4.non_local_block.", "backbone._nl_layers.layer3_4."),
                # .block. wrapper: NL-containing ResNet blocks use extra .block. prefix
                ("backbone.layer2.0.block.", "backbone.layer2.0."),
                ("backbone.layer2.2.block.", "backbone.layer2.2."),
                ("backbone.layer3.0.block.", "backbone.layer3.0."),
                ("backbone.layer3.2.block.", "backbone.layer3.2."),
                ("backbone.layer3.4.block.", "backbone.layer3.4."),
                # Non-local inner structure: checkpoint wraps weights in extra layers
                (".conv_out.bn.", ".bn."),
                (".conv_out.conv.", ".conv_out."),
                (".g.0.conv.", ".g."),
                (".phi.0.conv.", ".phi."),
                (".theta.conv.", ".theta."),
                # Regular TSM backbone mappings
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


MODEL_CLASS = TSMNonLocalClassifier


def create(**kwargs) -> TSMNonLocalClassifier:
    return TSMNonLocalClassifier(**kwargs)

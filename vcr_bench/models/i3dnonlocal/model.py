from __future__ import annotations

from typing import Any

from ..i3d.model import I3DClassifier
from ..vendor_mmaction.resnet3d import ResNet3d


class I3DNonLocalClassifier(I3DClassifier):
    model_name = "i3dnonlocal"
    MODEL_CAPABILITIES: dict[str, Any] = {
        "backbones": {
            "r50": {
                "display_name": "I3D R50 NonLocal",
                "datasets": {
                    "kinetics400": {
                        "num_classes": 400,
                        "checkpoint_url": (
                            "https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/"
                            "i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb/"
                            "i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb_20220812-8e1f2148.pth"
                        ),
                        "checkpoint_filename": "i3dnonlocal_r50_kinetics400_mmaction.pth",
                    }
                },
            }
        }
    }

    def _build_backbone(self) -> ResNet3d:
        return ResNet3d(
            depth=50,
            pretrained2d=False,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=2,
            pool1_stride_t=2,
            conv_cfg=dict(type="Conv3d"),
            inflate=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
            non_local=((0, 0, 0), (0, 1, 0, 1), (0, 1, 0, 1, 0, 1), (0, 0, 0)),
            non_local_cfg=dict(
                sub_sample=True,
                use_scale=False,
                norm_cfg=dict(type="BN3d", requires_grad=True),
                mode="dot_product",
            ),
            zero_init_residual=False,
            norm_eval=False,
        )


MODEL_CLASS = I3DNonLocalClassifier


def create(**kwargs) -> I3DNonLocalClassifier:
    return I3DNonLocalClassifier(**kwargs)

from __future__ import annotations
import torch
import sys
sys.path.insert(0, "/work")

# Inspect tsmnonlocal checkpoint keys
ckpt_path = "/work/.cache/vcr_bench/checkpoints/tsmnonlocal/tsmnonlocal_r50_kinetics400_mmaction.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")
if isinstance(ckpt, dict):
    for root_key in ("state_dict", "model_state_dict", "model", "module"):
        if root_key in ckpt:
            sd = ckpt[root_key]
            print("Root key:", root_key, "num_keys:", len(sd))
            keys = sorted(sd.keys())[:30]
            for k in keys:
                print("CKPT:", k)
            break

# What keys fail to match
from vcr_bench.models.tsmnonlocal.model import TSMNonLocalClassifier
m = TSMNonLocalClassifier(load_weights=False)
missing, unexpected = m.load_converted_checkpoint(
    ckpt_path,
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
print("\nMissing keys (first 20):")
for k in sorted(missing)[:20]:
    print("MISSING:", k)
print("\nUnexpected keys (first 20):")
for k in sorted(unexpected)[:20]:
    print("UNEXPECTED:", k)

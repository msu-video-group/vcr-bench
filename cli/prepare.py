from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from vg_videobench.attacks import create_attack
from vg_videobench.datasets import create_dataset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "vg_videobench offline attack preparation. "
            "Runs model-independent preprocessing (style caching, background extraction, "
            "mixer training, etc.) once so that subsequent attack runs skip it."
        )
    )
    p.add_argument("--attack", required=True, help="Attack name (e.g. stylefool, bmtc)")
    p.add_argument("--dataset", required=True, help="Dataset name (e.g. kinetics400)")
    p.add_argument("--video-root", default=None)
    p.add_argument("--annotations", default=None)
    p.add_argument("--labels", default=None)
    p.add_argument("--split", default="val")
    p.add_argument("--num-videos", type=int, default=100,
                   help="Number of videos to prepare for (same as the attack run)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed – must match the attack run seed for index alignment")
    p.add_argument("--device", default="cuda",
                   help="Device for computations that need a GPU (e.g. BMTC mixer training)")
    # Generic attack hyper-parameters forwarded to create_attack()
    p.add_argument("--eps", type=float, default=None)
    p.add_argument("--iter", type=int, default=None)
    p.add_argument("--style-steps", type=int, default=None,
                   help="StyleFool: VGG optimization steps per frame (default: 200)")
    p.add_argument("--style-content-weight", type=float, default=None,
                   help="StyleFool: content loss weight for neural style transfer")
    p.add_argument("--style-style-weight", type=float, default=None,
                   help="StyleFool: style loss weight for neural style transfer")
    # BMTC-specific
    p.add_argument("--bmtc-save-root", default=None,
                   help="Directory to save BMTC mixer artifacts (default: bmtc_artifacts/)")
    p.add_argument("--bmtc-num-epochs", type=int, default=None,
                   help="Mixer training epochs (default: 20)")
    p.add_argument("--bmtc-background-sets", default=None,
                   help="Path to pre-extracted background PNG directory")
    # Output
    p.add_argument("--output-json", default=None,
                   help="Write prepare summary JSON to this path")
    return p


def main() -> None:
    args = build_parser().parse_args()

    attack_kwargs: dict = {}
    if args.eps is not None:
        attack_kwargs["eps"] = args.eps
    if args.iter is not None:
        attack_kwargs["steps"] = args.iter
    if getattr(args, "style_steps", None) is not None:
        attack_kwargs["style_steps"] = args.style_steps
    if getattr(args, "style_content_weight", None) is not None:
        attack_kwargs["style_content_weight"] = args.style_content_weight
    if getattr(args, "style_style_weight", None) is not None:
        attack_kwargs["style_style_weight"] = args.style_style_weight
    attack = create_attack(args.attack, **attack_kwargs)

    if not attack.supports_offline_prepare():
        result = {
            "attack": args.attack,
            "status": "no_offline_prepare",
            "message": (
                f"Attack '{args.attack}' does not require offline preparation "
                "(no supports_offline_prepare() hook). Nothing to do."
            ),
        }
        print(json.dumps(result, indent=2))
        return

    dataset = create_dataset(
        args.dataset,
        video_root=args.video_root,
        annotations_csv=args.annotations,
        labels_txt=args.labels,
        split=args.split,
    )

    indices = list(range(len(dataset)))
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    indices = indices[: min(args.num_videos, len(indices))]

    prepare_kwargs: dict = dict(
        dataset=dataset,
        indices=indices,
        seed=args.seed,
        targeted=False,
        pipeline_stage="test",
        verbose=True,
        device=args.device,
    )
    # BMTC-specific kwargs
    if args.bmtc_save_root is not None:
        prepare_kwargs["save_root"] = args.bmtc_save_root
    if args.bmtc_num_epochs is not None:
        prepare_kwargs["num_epochs"] = args.bmtc_num_epochs
    if args.bmtc_background_sets is not None:
        prepare_kwargs["background_sets"] = args.bmtc_background_sets

    summary = attack.prepare_offline(**prepare_kwargs)

    output = {"attack": args.attack, "status": "ok", "result": summary}
    print(json.dumps(output, indent=2))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

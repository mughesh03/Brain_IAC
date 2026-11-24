"""Linear-probe test shared by part 2 and part 4.

Part 2 scores features from the pretrained encoder (part 1).
Part 4 scores features from the adapted encoder (part 3).
Both calls go through extract_features.
"""

import argparse
import sys
from pathlib import Path
from pprint import pprint

import torch

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from extract_features import extract_features
from linear_probe import evaluate_probe

PRETRAINED_CKPT = SRC_DIR / "checkpoints" / "BrainIAC.ckpt"
ADAPTED_CKPT = SRC_DIR / "checkpoints" / "BrainIAC_simclr_finetuned.ckpt"
PRETRAINED_FEATURES = SRC_DIR / "features" / "pretrained.npz"
ADAPTED_FEATURES = SRC_DIR / "features" / "adapted.npz"


def parse_args():
    parser = argparse.ArgumentParser(description="Test parts 2 and 4 with one linear probe")
    parser.add_argument("--pretrained-checkpoint", type=str, default=str(PRETRAINED_CKPT))
    parser.add_argument("--adapted-checkpoint", type=str, default=str(ADAPTED_CKPT))
    parser.add_argument("--pretrained-features", type=str, default=str(PRETRAINED_FEATURES))
    parser.add_argument("--adapted-features", type=str, default=str(ADAPTED_FEATURES))
    parser.add_argument("--output-dir", type=str, default=str(SRC_DIR / "results"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def run_part(title, checkpoint, features_path, output_dir, args):
    print("=" * 72)
    print(title)
    print("=" * 72)
    if args.skip_extract:
        if not Path(features_path).is_file():
            raise SystemExit(f"Missing features for {title}: {features_path}")
    else:
        extract_features(
            checkpoint,
            features_path,
            batch_size=args.batch_size,
            val_fraction=args.val_fraction,
            seed=args.seed,
            device=args.device,
        )
    report = evaluate_probe(features_path, output_dir, seed=args.seed)
    pprint(report, sort_dicts=False, width=80)
    print(f"saved {Path(output_dir) / 'metrics.txt'}")
    print()
    return report


def main():
    args = parse_args()
    parts = (
        (
            "Part 2: linear probe on pretrained features",
            args.pretrained_checkpoint,
            args.pretrained_features,
            Path(args.output_dir) / "pretrained",
        ),
        (
            "Part 4: linear probe on adapted features",
            args.adapted_checkpoint,
            args.adapted_features,
            Path(args.output_dir) / "adapted",
        ),
    )
    for title, checkpoint, features_path, output_dir in parts:
        if not args.skip_extract and not Path(checkpoint).is_file():
            raise SystemExit(f"Missing checkpoint for {title}: {checkpoint}")
        run_part(title, checkpoint, features_path, output_dir, args)


if __name__ == "__main__":
    main()

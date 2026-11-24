"""Extract frozen BrainIAC features for the train, validation, and test splits."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finetune_simclr import (
    TEST_IMAGES,
    TEST_LABELS,
    TRAIN_IMAGES,
    TRAIN_LABELS,
    VOLUME_SIZE,
    assert_splits_disjoint,
    channels_to_volume,
    load_labeled_array,
    prepare_slice,
    resize_full,
    stratified_split,
)
from load_brainiac import load_brainiac


def embed_split(model, images, indices, batch_size, device):
    model.eval()
    features = []
    indices = np.asarray(indices, dtype=np.int64)
    for start in range(0, len(indices), batch_size):
        rows = indices[start : start + batch_size]
        volumes = []
        for row in rows:
            img = prepare_slice(torch.from_numpy(np.array(images[int(row)], dtype=np.float32, copy=True)))
            volumes.append(channels_to_volume(resize_full(img, VOLUME_SIZE)))
        batch = torch.stack(volumes, dim=0).to(device)
        with torch.no_grad():
            features.append(model(batch).detach().cpu().numpy())
    return np.concatenate(features, axis=0)


def extract_features(checkpoint, output, batch_size=4, val_fraction=0.2, seed=0, device=None):
    """Freeze one BrainIAC checkpoint and save train, validation, and test features."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_images, train_labels = load_labeled_array(TRAIN_IMAGES, TRAIN_LABELS)
    test_images, test_labels = load_labeled_array(TEST_IMAGES, TEST_LABELS)
    train_idx, val_idx = stratified_split(train_labels, val_fraction, seed)
    assert_splits_disjoint(train_images, train_idx, val_idx, test_images)
    test_idx = np.arange(len(test_labels), dtype=np.int64)

    def run_on(run_device):
        model = load_brainiac(str(checkpoint), device="cpu")
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        model = model.to(run_device)
        return (
            embed_split(model, train_images, train_idx, batch_size, run_device),
            embed_split(model, train_images, val_idx, batch_size, run_device),
            embed_split(model, test_images, test_idx, batch_size, run_device),
        )

    torch_device = torch.device(device)
    try:
        train_features, val_features, test_features = run_on(torch_device)
    except torch.cuda.OutOfMemoryError:
        # The GPU is shared and another process may be holding its memory.
        if torch_device.type != "cuda":
            raise
        print(
            "CUDA ran out of memory because other processes are using this GPU. "
            "Falling back to CPU. Check `nvidia-smi` or pass --device cpu."
        )
        torch.cuda.empty_cache()
        train_features, val_features, test_features = run_on(torch.device("cpu"))

    np.savez(
        output,
        train_features=train_features,
        train_labels=train_labels[train_idx],
        val_features=val_features,
        val_labels=train_labels[val_idx],
        test_features=test_features,
        test_labels=test_labels[test_idx],
        train_idx=train_idx,
        val_idx=val_idx,
    )
    print(
        f"saved {output} train={train_features.shape} "
        f"val={val_features.shape} test={test_features.shape}"
    )
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frozen BrainIAC features")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    extract_features(
        args.checkpoint,
        args.output,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()

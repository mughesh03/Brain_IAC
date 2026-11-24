"""Contrastive adaptation of the BrainIAC ViT.

The official training array is split into train and validation. The official
test array stays untouched and is only available for the later evaluation.
"""

import argparse
import hashlib
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import ViTBackboneNet
from simclr_loss import add_contrastive_loss

DATA_DIR = SRC_DIR / "data"
TRAIN_IMAGES = DATA_DIR / "brain_train_image_final.npy"
TRAIN_LABELS = DATA_DIR / "brain_train_label.npy"
TEST_IMAGES = DATA_DIR / "brain_test_image_final.npy"
TEST_LABELS = DATA_DIR / "brain_test_label.npy"
PRETRAINED_CKPT = SRC_DIR / "checkpoints" / "BrainIAC.ckpt"
BEST_CKPT = SRC_DIR / "checkpoints" / "BrainIAC_simclr_finetuned.ckpt"
LAST_CKPT = SRC_DIR / "checkpoints" / "BrainIAC_simclr_finetuned_last.ckpt"

VOLUME_SIZE = 96
COLOR_JITTER_STRENGTH = 1.0


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_resized_crop(img, size, scale=(0.08, 1.0), ratio=(0.75, 4.0 / 3.0)):
    """SimCLR random crop: area in ``scale``, then bilinear resize to ``size``."""
    _, height, width = img.shape
    area = height * width
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        target = area * random.uniform(scale[0], scale[1])
        aspect = math.exp(random.uniform(log_ratio[0], log_ratio[1]))
        crop_w = int(round(math.sqrt(target * aspect)))
        crop_h = int(round(math.sqrt(target / aspect)))
        if 0 < crop_w <= width and 0 < crop_h <= height:
            top = random.randint(0, height - crop_h)
            left = random.randint(0, width - crop_w)
            crop = img[:, top : top + crop_h, left : left + crop_w]
            return F.interpolate(
                crop.unsqueeze(0),
                size=(size, size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
    return F.interpolate(
        img.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _brightness(img, delta):
    # SimCLRv2 brightness is multiplicative.
    factor = random.uniform(max(1.0 - delta, 0.0), 1.0 + delta)
    return img * factor


def _contrast(img, delta):
    factor = random.uniform(1.0 - delta, 1.0 + delta)
    mean = img.mean(dim=(-2, -1), keepdim=True)
    return (img - mean) * factor + mean


def color_distort(img, strength):
    """Brightness and contrast from SimCLR color jitter.

    Saturation, hue, and grayscale are RGB operations. These arrays are
    two-channel images already scaled to [0, 1], so those ops are omitted.
    """
    if random.random() >= 0.8:
        return img
    brightness_delta = 0.8 * strength
    contrast_delta = 0.8 * strength
    ops = (
        lambda x: _brightness(x, brightness_delta),
        lambda x: _contrast(x, contrast_delta),
    )
    order = [0, 1]
    random.shuffle(order)
    for index in order:
        img = ops[index](img)
    return img


def gaussian_blur(img, kernel_size, sigma):
    if kernel_size % 2 == 0:
        kernel_size += 1
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, dtype=img.dtype, device=img.device)
    kernel = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    channels = img.shape[0]
    img_b = img.unsqueeze(0)
    kernel_h = kernel.view(1, 1, 1, kernel_size).repeat(channels, 1, 1, 1)
    kernel_v = kernel.view(1, 1, kernel_size, 1).repeat(channels, 1, 1, 1)
    blurred = F.conv2d(img_b, kernel_h, padding=(0, radius), groups=channels)
    blurred = F.conv2d(blurred, kernel_v, padding=(radius, 0), groups=channels)
    return blurred.squeeze(0)


def maybe_blur(img, size):
    if random.random() >= 0.5:
        return img
    sigma = random.uniform(0.1, 2.0)
    return gaussian_blur(img, kernel_size=max(size // 10, 3), sigma=sigma)


def prepare_slice(img):
    """Keep channel 1 and remove its global offset.

    Channel 0 is constant for every slice of a class, in the training array
    and in the test array. Feeding it to the encoder gives the probe the label.
    """
    if img.ndim != 3 or img.shape[0] < 2:
        raise ValueError(f"expected (2, H, W), got {tuple(img.shape)}")
    anatomical = img[1:2]
    mean = anatomical.mean()
    std = anatomical.std().clamp(min=1e-6)
    return (anatomical - mean) / std


def resize_full(img, size):
    """Deterministic resize used for the held-out test set."""
    return F.interpolate(
        img.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def augment_view(img, size):
    """One SimCLR view of a ``(1, H, W)`` anatomical slice."""
    view = random_resized_crop(img, size)
    if random.random() < 0.5:
        view = torch.flip(view, dims=(-1,))
    view = color_distort(view, COLOR_JITTER_STRENGTH)
    view = maybe_blur(view, size)
    return view


def channels_to_volume(img):
    """Lift a 2-channel 96x96 view into the encoder's ``(1, 96, 96, 96)`` input.

    The two channels are the ends of the depth axis. Trilinear resize fills
    the 96-depth volume the pretrained 3D patch embedding expects.
    """
    volume = img.unsqueeze(0).unsqueeze(0)
    volume = F.interpolate(
        volume,
        size=(VOLUME_SIZE, VOLUME_SIZE, VOLUME_SIZE),
        mode="trilinear",
        align_corners=False,
    )
    return volume.squeeze(0)


def load_labeled_array(image_path, label_path):
    """Load one image array and its labels from ``src/data``."""
    for path in (image_path, label_path):
        if Path(path).resolve().parent != DATA_DIR.resolve():
            raise ValueError(f"Refusing to read outside {DATA_DIR}: {path}")
    images = np.load(image_path, mmap_mode="r")
    labels = np.asarray(np.load(label_path)).reshape(-1)
    if images.ndim != 4 or images.shape[1] != 2:
        raise ValueError(f"{image_path} must have shape (N, 2, H, W), got {images.shape}")
    if labels.shape[0] != images.shape[0]:
        raise ValueError(
            f"{label_path} has {labels.shape[0]} labels for {images.shape[0]} images"
        )
    return images, labels


def stratified_split(labels, val_fraction, seed):
    """Split training indices into train and validation, stratified by class.

    The official test array is not part of this split.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    rng = np.random.RandomState(seed)
    train_parts = []
    val_parts = []
    for class_id in np.unique(labels):
        class_idx = np.flatnonzero(labels == class_id)
        rng.shuffle(class_idx)
        n_val = int(round(len(class_idx) * val_fraction))
        n_val = min(max(n_val, 1), len(class_idx) - 1)
        val_parts.append(class_idx[:n_val])
        train_parts.append(class_idx[n_val:])
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def _slice_hashes(images, indices):
    """Identity of the anatomical channel only."""
    found = []
    for row in np.asarray(indices, dtype=np.int64):
        anatomical = np.ascontiguousarray(images[int(row), 1])
        found.append(hashlib.md5(anatomical.tobytes()).hexdigest())
    return found


def assert_splits_disjoint(train_images, train_idx, val_idx, test_images):
    """Train, validation, and test must not share indices or identical slices."""
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    if len(train_idx) != len(np.unique(train_idx)) or len(val_idx) != len(np.unique(val_idx)):
        raise RuntimeError("a split contains a repeated index")
    if len(np.intersect1d(train_idx, val_idx)):
        raise RuntimeError("train and validation indices overlap")
    covered = len(train_idx) + len(val_idx)
    if covered != len(train_images):
        raise RuntimeError(
            f"train+val has {covered} rows but the training array has {len(train_images)}"
        )
    test_idx = np.arange(len(test_images), dtype=np.int64)
    train_hashes = _slice_hashes(train_images, train_idx)
    val_hashes = _slice_hashes(train_images, val_idx)
    test_hashes = _slice_hashes(test_images, test_idx)
    if len(train_hashes) != len(set(train_hashes)):
        raise RuntimeError("duplicate slices inside the training split")
    if len(val_hashes) != len(set(val_hashes)):
        raise RuntimeError("duplicate slices inside the validation split")
    if len(test_hashes) != len(set(test_hashes)):
        raise RuntimeError("duplicate slices inside the test split")
    shared = (
        (set(train_hashes) & set(val_hashes))
        | (set(train_hashes) & set(test_hashes))
        | (set(val_hashes) & set(test_hashes))
    )
    if shared:
        raise RuntimeError(f"{len(shared)} identical slices appear in more than one split")


class BrainArrayContrastiveDataset(Dataset):
    """Two SimCLR views for a subset of the training array."""

    def __init__(self, images, indices):
        self.images = images
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, idx):
        row = int(self.indices[idx])
        img = prepare_slice(torch.from_numpy(np.array(self.images[row], dtype=np.float32, copy=True)))
        view1 = channels_to_volume(augment_view(img, VOLUME_SIZE))
        view2 = channels_to_volume(augment_view(img, VOLUME_SIZE))
        return view1, view2


class BrainArrayTestDataset(Dataset):
    """Official test split: one deterministic volume and its class label."""

    def __init__(self, images, labels):
        self.images = images
        self.labels = np.asarray(labels).reshape(-1)

    def __len__(self):
        return int(self.images.shape[0])

    def __getitem__(self, idx):
        img = prepare_slice(torch.from_numpy(np.array(self.images[idx], dtype=np.float32, copy=True)))
        volume = channels_to_volume(resize_full(img, VOLUME_SIZE))
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return volume, label


class ProjectionHead(nn.Module):
    """Nonlinear head stored in the BrainIAC SimCLR checkpoint.

    Linear 768 -> 768, BatchNorm, ReLU, Linear 768 -> 2048, BatchNorm.
    The last layer has no bias and no ReLU, matching SimCLR's projection head.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 768, bias=False),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Linear(768, 2048, bias=False),
            nn.BatchNorm1d(2048),
        )

    def forward(self, x):
        return self.layers(x)


class SimCLRBrainIAC(nn.Module):
    def __init__(self, checkpoint_path, freeze_encoder=False):
        super().__init__()
        self.encoder = ViTBackboneNet(str(checkpoint_path))
        self.projection_head = ProjectionHead()
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        proj_state = {
            key[len("projection_head.") :]: value
            for key, value in state.items()
            if key.startswith("projection_head.")
        }
        self.projection_head.load_state_dict(proj_state, strict=True)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.encoder(x)
        return self.projection_head(features)


def learning_rate(epoch, epochs, base_lr, warmup_epochs):
    if epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(max(warmup_epochs, 1))
    progress = (epoch - warmup_epochs) / float(max(epochs - warmup_epochs, 1))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def save_checkpoint(path, model, epoch, train_loss, val_loss, train_idx, val_idx, args):
    state = {}
    for key, value in model.encoder.backbone.state_dict().items():
        state[f"backbone.{key}"] = value.detach().cpu()
    for key, value in model.projection_head.state_dict().items():
        state[f"projection_head.{key}"] = value.detach().cpu()
    torch.save(
        {
            "epoch": epoch,
            "train_contrastive_loss": train_loss,
            "val_contrastive_loss": val_loss,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "state_dict": state,
            "hyper_parameters": {
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "temperature": args.temperature,
                "batch_size": args.batch_size,
                "val_fraction": args.val_fraction,
                "seed": args.seed,
            },
        },
        path,
    )


def run_steps(model, loader, optimizer, device, temperature, train, max_batches):
    model.train(mode=train)
    total = 0.0
    seen = 0
    for batch_idx, (view1, view2) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)
        images = torch.cat([view1, view2], dim=0)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            projected = model(images)
            loss = add_contrastive_loss(projected, hidden_norm=True, temperature=temperature)
            if train:
                loss.backward()
                optimizer.step()
        total += float(loss.item())
        seen += 1
    if seen == 0:
        raise RuntimeError("No batches were run. Lower --batch-size so at least two images remain after drop_last.")
    return total / seen, seen


def parse_args():
    parser = argparse.ArgumentParser(description="SimCLR finetune of the BrainIAC ViT")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after this many train steps. 0 runs every epoch.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 2:
        raise SystemExit("--batch-size must be at least 2")
    for path in (TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS, PRETRAINED_CKPT):
        if not path.is_file():
            raise SystemExit(f"Missing required file: {path}")

    seed_everything(args.seed)
    device = torch.device(args.device)

    train_images, train_labels = load_labeled_array(TRAIN_IMAGES, TRAIN_LABELS)
    test_images, test_labels = load_labeled_array(TEST_IMAGES, TEST_LABELS)
    train_idx, val_idx = stratified_split(train_labels, args.val_fraction, args.seed)
    assert_splits_disjoint(train_images, train_idx, val_idx, test_images)
    train_set = BrainArrayContrastiveDataset(train_images, train_idx)
    val_set = BrainArrayContrastiveDataset(train_images, val_idx)
    test_set = BrainArrayTestDataset(test_images, test_labels)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "drop_last": True,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    model = SimCLRBrainIAC(PRETRAINED_CKPT, freeze_encoder=args.freeze_encoder).to(device)
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(
        f"device={device} train={len(train_set)} val={len(val_set)} "
        f"test={len(test_set)} trainable={trainable}"
    )
    print("test set is held out for evaluation")

    best_val = float("inf")
    steps_done = 0

    for epoch in range(args.epochs):
        lr = learning_rate(epoch, args.epochs, args.lr, args.warmup_epochs)
        set_lr(optimizer, lr)
        remaining = None
        if args.max_steps:
            remaining = args.max_steps - steps_done
            if remaining <= 0:
                break
        train_loss, train_batches = run_steps(
            model,
            train_loader,
            optimizer,
            device,
            args.temperature,
            train=True,
            max_batches=remaining,
        )
        steps_done += train_batches
        val_loss, _ = run_steps(
            model,
            val_loader,
            optimizer=None,
            device=device,
            temperature=args.temperature,
            train=False,
            max_batches=None,
        )
        print(
            f"epoch {epoch:03d} lr={lr:.6g} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} steps={steps_done}"
        )
        save_checkpoint(LAST_CKPT, model, epoch, train_loss, val_loss, train_idx, val_idx, args)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(BEST_CKPT, model, epoch, train_loss, val_loss, train_idx, val_idx, args)
            print(f"saved {BEST_CKPT.name} val_loss={best_val:.4f}")
        if args.max_steps and steps_done >= args.max_steps:
            break

    print(f"best val contrastive loss={best_val:.4f}")
    print(f"last checkpoint={LAST_CKPT}")
    print(f"best checkpoint={BEST_CKPT}")


if __name__ == "__main__":
    main()

"""
Contrastive Pretraining — Vía C
================================
Trains a projection head (and optionally the last CSP block of the backbone)
on top of the YOLOv4 detector with a joint NT-Xent + SupCon loss to produce
a class-discriminative latent space for Vía B diversity sampling.

The trained head is consumed by `deep_cluster_sampler.py --projection_head`.
The YOLOv4 detector weights are **not** overwritten; this script only
produces a separate checkpoint at `checkpoints/projection_head.pth.tar`.

Pipeline (per batch)
--------------------
1. Sample B images (mix of labelled + unlabelled, weighted by availability).
2. Apply the SimCLR augmentation pipeline twice → 2B views.
3. Forward through backbone (frozen except optionally the last CSP block) →
   GAP → projection head → L2-normalized embeddings z ∈ ℝ[2B, D].
4. Compute:
       L_NTXent  : on all 2B rows.                            [Chen 2020]
       L_SupCon  : on the labelled subset (via class_mask).   [Khosla 2020]
   Joint loss:
       L = L_NTXent + λ · L_SupCon              (λ = config.CONTRASTIVE_SUPCON_LAMBDA)

Image-level class label
-----------------------
YOLO labels are object-level. For SupCon we need an image-level class. We use
the **majority class** present in the image (most common class_id across the
.txt boxes). This is a standard simplification and is honest given that the
supermarket scenes are typically dominated by one product.

Usage
-----
    python src/training/contrastive_pretrain.py --weights checkpoints/finetune_best_map.pth.tar

References
----------
Chen et al. 2020 (SimCLR), Khosla et al. 2020 (SupCon),
He et al. 2020 (MoCo), Caron et al. 2018 (DeepCluster).
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

import core.config as config
from core.contrastive import (
    ProjectionHead,
    NTXentLoss,
    SupConLoss,
    build_simclr_transform,
)
from core.model import YOLOv4, CSPBlock


# ============================================================
# DATASET — TWO VIEWS PER IMAGE
# ============================================================

class ContrastiveImageDataset(Dataset):
    """
    Loads an image and returns two independent SimCLR augmentations of it
    plus an image-level class label (or -1 if the image is unlabelled).

    The class label is derived from the YOLO .txt file (majority class).
    """

    UNLABELLED = -1

    def __init__(
        self,
        image_paths: List[Path],
        labels: List[int],
        transform_factory,
    ) -> None:
        assert len(image_paths) == len(labels)
        self.image_paths = image_paths
        self.labels = labels
        # Two independent transform instances → independent random augmentations
        self.transform_view1 = transform_factory()
        self.transform_view2 = transform_factory()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        path = self.image_paths[idx]
        img_np = np.array(Image.open(path).convert("RGB"))
        v1 = self.transform_view1(image=img_np)["image"]
        v2 = self.transform_view2(image=img_np)["image"]
        return v1, v2, int(self.labels[idx])


def _majority_class_from_label_file(label_path: Path) -> Optional[int]:
    """Return the most frequent class_id in a YOLO label file, or None if empty."""
    try:
        with open(label_path, "r") as f:
            cls_ids = [int(line.split()[0]) for line in f if line.strip()]
    except (FileNotFoundError, ValueError):
        return None
    if not cls_ids:
        return None
    return int(max(set(cls_ids), key=cls_ids.count))


def _collect_image_paths(
    labelled_dir: Path,
    unlabelled_dir: Path,
) -> Tuple[List[Path], List[int]]:
    """Collect all training images and image-level labels (-1 for unlabelled)."""
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths: List[Path] = []
    labels: List[int] = []

    if labelled_dir.exists():
        for f in sorted(os.listdir(labelled_dir)):
            if Path(f).suffix.lower() not in valid_ext:
                continue
            img_path = labelled_dir / f
            txt_path = img_path.with_suffix(".txt")
            cls = _majority_class_from_label_file(txt_path)
            if cls is None:
                # No usable label → treat as unlabelled (still useful for NT-Xent)
                image_paths.append(img_path)
                labels.append(ContrastiveImageDataset.UNLABELLED)
            else:
                image_paths.append(img_path)
                labels.append(cls)

    if unlabelled_dir.exists():
        for f in sorted(os.listdir(unlabelled_dir)):
            if Path(f).suffix.lower() not in valid_ext:
                continue
            image_paths.append(unlabelled_dir / f)
            labels.append(ContrastiveImageDataset.UNLABELLED)

    return image_paths, labels


# ============================================================
# FORWARD HELPER — frozen-or-unfrozen backbone + GAP + head
# ============================================================

def _backbone_features(model: YOLOv4, x: torch.Tensor) -> torch.Tensor:
    """Run the input through the backbone (no SPP/neck/heads) and return GAP."""
    for layer in model.backbone:
        x = layer(x)
    return x.mean(dim=(-2, -1))                       # [B, 1024]


# ============================================================
# OPTIMIZER PARAM GROUPS
# ============================================================

def _setup_trainable_params(
    model: YOLOv4,
    head: ProjectionHead,
    unfreeze_last_csp: bool,
) -> List[torch.nn.Parameter]:
    """Freeze the detector, unfreeze the last CSPBlock if requested, return trainable params."""
    for p in model.parameters():
        p.requires_grad = False

    trainable: List[torch.nn.Parameter] = list(head.parameters())

    if unfreeze_last_csp:
        # The last backbone layer is the CSPBlock with num_repeats=4 (just before SPP).
        last_csp_idx: Optional[int] = None
        for i in range(len(model.backbone) - 1, -1, -1):
            if isinstance(model.backbone[i], CSPBlock):
                last_csp_idx = i
                break
        if last_csp_idx is None:
            raise RuntimeError("Could not locate the last CSPBlock in the backbone.")
        for p in model.backbone[last_csp_idx].parameters():
            p.requires_grad = True
        trainable.extend([p for p in model.backbone[last_csp_idx].parameters()])
        print(f"  Unfrozen backbone layer index: {last_csp_idx}  "
              f"(CSPBlock, num_repeats={model.backbone[last_csp_idx].num_repeats})")

    n_train = sum(p.numel() for p in trainable)
    print(f"  Trainable parameters: {n_train:,}")
    return trainable


# ============================================================
# COSINE LR WITH WARMUP
# ============================================================

def _lr_at(epoch: int, total: int, warmup: int, base_lr: float) -> float:
    if epoch < warmup:
        return base_lr * (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Contrastive pretraining of a projection head over the frozen YOLOv4 backbone."
    )
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to YOLOv4 detector checkpoint (.pth.tar).")
    parser.add_argument("--epochs", type=int, default=config.CONTRASTIVE_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.CONTRASTIVE_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.CONTRASTIVE_LR)
    parser.add_argument("--temperature", type=float, default=config.CONTRASTIVE_TEMPERATURE)
    parser.add_argument("--proj_dim", type=int, default=config.CONTRASTIVE_PROJ_DIM)
    parser.add_argument("--hidden_dim", type=int, default=config.CONTRASTIVE_HIDDEN_DIM)
    parser.add_argument("--supcon_lambda", type=float, default=config.CONTRASTIVE_SUPCON_LAMBDA)
    parser.add_argument("--warmup_epochs", type=int, default=config.CONTRASTIVE_WARMUP_EPOCHS)
    parser.add_argument("--no_unfreeze_last_csp", action="store_true",
                        help="Disable unfreezing of the last CSPBlock (head-only training).")
    parser.add_argument("--output", type=str, default=str(config.CONTRASTIVE_CHECKPOINT))
    args = parser.parse_args()

    weights_path = (
        Path(args.weights) if os.path.isabs(args.weights)
        else Path(config.BASE_DIR) / args.weights
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unfreeze_last_csp = (not args.no_unfreeze_last_csp) and config.CONTRASTIVE_UNFREEZE_LAST_CSP

    print(f"\n{'='*60}")
    print(f"  CONTRASTIVE PRETRAINING — Vía C")
    print(f"  Detector ckpt : {weights_path}")
    print(f"  Epochs        : {args.epochs}  (warmup {args.warmup_epochs})")
    print(f"  Batch size    : {args.batch_size}  → 2× views per step")
    print(f"  LR / WD       : {args.lr} / {config.CONTRASTIVE_WEIGHT_DECAY}")
    print(f"  Temperature   : {args.temperature}")
    print(f"  Proj dim      : {args.proj_dim}  (hidden {args.hidden_dim})")
    print(f"  SupCon λ      : {args.supcon_lambda}")
    print(f"  Unfreeze CSP4 : {unfreeze_last_csp}")
    print(f"  Output ckpt   : {output_path}")
    print(f"{'='*60}\n")

    device = config.DEVICE

    # --- Load detector ---
    model = YOLOv4(num_classes=config.SPECIFIC_NUM_CLASSES).to(device)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print("  Detector loaded.")

    # --- Build projection head ---
    head = ProjectionHead(in_dim=1024, hidden_dim=args.hidden_dim, out_dim=args.proj_dim).to(device)
    print(f"  Projection head: 1024 → {args.hidden_dim} → {args.proj_dim}")

    # --- Setup trainable params ---
    trainable = _setup_trainable_params(model, head, unfreeze_last_csp)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=config.CONTRASTIVE_WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda")

    # --- Collect data ---
    print("\n  Collecting image paths...")
    img_paths, labels = _collect_image_paths(
        labelled_dir=Path(config.IMG_DIR),
        unlabelled_dir=Path(config.UNLABELLED_IMG_DIR),
    )
    n_total = len(img_paths)
    n_lab = sum(1 for c in labels if c != ContrastiveImageDataset.UNLABELLED)
    n_unl = n_total - n_lab
    print(f"  Total images   : {n_total}")
    print(f"    labelled     : {n_lab}")
    print(f"    unlabelled   : {n_unl}\n")
    if n_total == 0:
        print("  No images found — aborting.")
        return

    # Up-weight labelled samples so each batch sees a non-trivial number of them.
    # Without this, SupCon would barely fire (many batches without enough labelled rows).
    target_lab_share = 0.4
    if n_lab > 0 and n_unl > 0:
        w_lab = (target_lab_share * n_total) / n_lab
        w_unl = ((1 - target_lab_share) * n_total) / n_unl
    else:
        w_lab = w_unl = 1.0
    sample_weights = np.array(
        [w_lab if c != ContrastiveImageDataset.UNLABELLED else w_unl for c in labels],
        dtype=np.float64,
    )

    # Two transforms with independent RNG so the two views differ
    transform_factory = lambda: build_simclr_transform(image_size=config.IMAGE_SIZE)

    dataset = ContrastiveImageDataset(img_paths, labels, transform_factory)
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=n_total, replacement=True
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )

    # --- Losses ---
    ntxent_loss = NTXentLoss(temperature=args.temperature).to(device)
    supcon_loss = SupConLoss(temperature=args.temperature).to(device)

    # --- Training mode ---
    # Frozen backbone layers must keep eval()-mode BatchNorm (stats frozen) to
    # avoid drift; only the unfrozen last CSP block (if any) runs in train()
    # mode so its BN updates running stats from contrastive batches.
    model.eval()
    head.train()
    if unfreeze_last_csp:
        last_csp_idx = None
        for i in range(len(model.backbone) - 1, -1, -1):
            if isinstance(model.backbone[i], CSPBlock):
                last_csp_idx = i
                break
        model.backbone[last_csp_idx].train()

    best_loss = float("inf")
    history = []

    for epoch in range(args.epochs):
        lr_now = _lr_at(epoch, args.epochs, args.warmup_epochs, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        epoch_total, epoch_nt, epoch_sup, epoch_n = 0.0, 0.0, 0.0, 0
        n_supcon_batches = 0

        pbar = tqdm(loader, desc=f"  Epoch {epoch+1}/{args.epochs} (lr={lr_now:.2e})", leave=False)
        for v1, v2, lbls in pbar:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            x = torch.cat([v1, v2], dim=0)                          # [2B, 3, H, W]
            doubled_labels = torch.cat([lbls, lbls], dim=0)         # [2B]
            class_mask = doubled_labels != ContrastiveImageDataset.UNLABELLED

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                features = _backbone_features(model, x)              # [2B, 1024]
                z = head(features)                                   # [2B, D] L2-norm
                z = torch.nan_to_num(z, nan=0.0)

                loss_nt = ntxent_loss(z)
                if class_mask.sum() >= 2:
                    loss_sup = supcon_loss(z, doubled_labels, class_mask)
                else:
                    loss_sup = torch.zeros((), device=device)

                loss = loss_nt + args.supcon_lambda * loss_sup

            if torch.isnan(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            bsz = v1.shape[0]
            epoch_total += loss.item() * bsz
            epoch_nt    += loss_nt.item() * bsz
            epoch_sup   += loss_sup.item() * bsz
            epoch_n     += bsz
            if class_mask.sum() >= 2:
                n_supcon_batches += 1

            pbar.set_postfix({
                "loss":   f"{loss.item():.3f}",
                "NTXent": f"{loss_nt.item():.3f}",
                "SupCon": f"{loss_sup.item():.3f}",
            })

        if epoch_n == 0:
            continue
        avg_total = epoch_total / epoch_n
        avg_nt    = epoch_nt    / epoch_n
        avg_sup   = epoch_sup   / epoch_n
        history.append({
            "epoch": epoch + 1, "lr": lr_now,
            "total": avg_total, "ntxent": avg_nt, "supcon": avg_sup,
            "supcon_batches": n_supcon_batches,
        })
        print(f"  Epoch {epoch+1:>3}/{args.epochs}  "
              f"loss={avg_total:.4f}  NTXent={avg_nt:.4f}  "
              f"SupCon={avg_sup:.4f}  "
              f"(SupCon active in {n_supcon_batches} batches)")

        # --- Save best ---
        if avg_total < best_loss:
            best_loss = avg_total
            # Save only what is needed by deep_cluster_sampler: head + (optionally)
            # the unfrozen last-CSP-block weights.
            backbone_state = None
            if unfreeze_last_csp:
                # Find last CSPBlock index again — same scan as in _setup_trainable_params
                last_csp_idx = None
                for i in range(len(model.backbone) - 1, -1, -1):
                    if isinstance(model.backbone[i], CSPBlock):
                        last_csp_idx = i
                        break
                backbone_state = {
                    "last_csp_idx": last_csp_idx,
                    "state_dict": model.backbone[last_csp_idx].state_dict(),
                }
            torch.save({
                "projection_head": head.state_dict(),
                "head_config": {
                    "in_dim":     1024,
                    "hidden_dim": args.hidden_dim,
                    "out_dim":    args.proj_dim,
                },
                "last_csp_block": backbone_state,
                "args":    vars(args),
                "history": history,
                "best_loss": best_loss,
            }, output_path)

    print(f"\n  ✅ Training finished.  Best joint loss = {best_loss:.4f}")
    print(f"  Checkpoint → {output_path}\n")
    print(f"  Next step: run diversity sampling with the new latent space:")
    print(f"    python src/training/deep_cluster_sampler.py \\")
    print(f"        --weights {args.weights} \\")
    print(f"        --projection_head {output_path} \\")
    print(f"        --strategy coreset --budget {config.DIVERSITY_BUDGET}\n")


if __name__ == "__main__":
    main()

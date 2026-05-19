"""
YOLOv4 Fine-Tuning Pipeline
============================

Loads model pretrained with generic dataset and adapts it to the specific
supermarket dataset through a 3-phase progressive unfreezing:

    Phase 1 — Heads only   (backbone + SPP + neck frozen)
    ─────────────────────────────────────────────────────
    • Goal: stabilize new heads (random weights)
    • High LR (1e-4) because no risk of destroying frozen backbone
    • No scheduler: gradients are chaotic at first

    Phase 2 — Neck + SPP + heads   (backbone still frozen)
    ─────────────────────────────────────────────────────
    • Goal: fine-tune neck and SPP with new classes
    • Low LR (5e-6) + CosineAnnealingLR for smooth descent
    • Early stopping: saves best checkpoint if val_loss improves
    • Evaluates mAP every MAP_EVAL_FREQ epochs

    Phase 3 — Full model   (backbone unfrozen)
    ─────────────────────────────────────────────────────
    • Goal: global fine-tuning to adapt backbone to new domain
    • Very low LR (1e-6) to avoid catastrophic forgetting
    • Early stopping continues from Phase 2 best val_loss

Usage
-----
    Full pipeline (phases 1-2-3):
        python src/training/finetune.py

    Resume at phase 3 only (fresh LR schedule from existing checkpoint):
        python src/training/finetune.py --resume_phase3 checkpoints/finetune_best.pth.tar
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from core.model import YOLOv4, ScalePrediction, initialize_weights
from core.loss import YoloLoss
import core.config as config
from training.train import train_fn, val_fn
from core.utils import (
    get_loaders,
    save_checkpoint,
    check_class_accuracy,
    get_evaluation_bboxes,
    mean_average_precision,
)

import wandb
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True


# ============================================================
# FINE-TUNING HYPERPARAMETERS
# ============================================================
FASE1_EPOCHS  = 15        # Heads only
FASE2_EPOCHS  = 180       # Neck + SPP + heads
LR_FASE1      = 1e-4      # Higher: heads start from random weights
LR_FASE2      = 1e-5      # Lower: fine-tune without destroying learning
MAP_EVAL_FREQ = 5         # Evaluate mAP every N epochs
PATIENCE      = 15        # Early stopping: epochs without val_loss improvement
FASE3_EPOCHS  = 120        # Full unfreezing
LR_FASE3      = 1e-6      # Very low to not break backbone


# ============================================================
# FREEZING HELPERS
# ============================================================

def freeze_all_except_heads(model: nn.Module) -> None:
    """Freezes backbone, SPP and neck. Only heads remain trainable."""
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.spp.parameters():
        param.requires_grad = False
    for param in model.neck.parameters():
        param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters (heads only): {trainable:,}")


def unfreeze_neck_and_spp(model: nn.Module) -> None:
    """Unfreezes neck and SPP for Phase 2. Backbone still frozen."""
    for param in model.spp.parameters():
        param.requires_grad = True
    for param in model.neck.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters (neck + SPP + heads): {trainable:,}")


def unfreeze_all(model: nn.Module) -> None:
    """Unfreezes backbone for final fine-tuning."""
    for param in model.backbone.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters (ENTIRE MODEL): {trainable:,}")


# ============================================================
# CLI
# ============================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "YOLOv4 Fine-Tuning Pipeline. "
            "Default: run all 3 phases from the generic pretrained checkpoint. "
            "Use --resume_phase3 to skip phases 1-2 and restart phase 3 from "
            "an existing checkpoint with a fresh cosine-annealing LR schedule."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--resume_phase3",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help=(
            "Path to a .pth.tar checkpoint already trained with the current "
            "class set (e.g. checkpoints/finetune_best.pth.tar). "
            "Skips phases 1 and 2. The val_loss and mAP baselines are "
            "computed automatically from the val set so the phase-3 "
            "early-stopping threshold is always correct."
        ),
    )
    return p.parse_args()


# ============================================================
# PHASE 3 CORE LOOP  (shared by full pipeline and --resume_phase3)
# ============================================================

def _run_phase3(
    model: nn.Module,
    train_loader,
    val_loader,
    loss_fn: YoloLoss,
    scaled_anchors: torch.Tensor,
    best_val_loss: float,
    best_map: float,
    epoch_offset: int,
) -> None:
    """
    Phase 3: full-model fine-tuning with a fresh cosine-annealing LR schedule.

    Assumes the model is already fully unfrozen (unfreeze_all called by caller)
    and wandb has already been initialised.

    Args:
        best_val_loss:  baseline val_loss — only saves finetune_best.pth.tar
                        when the current epoch improves on this value.
        best_map:       baseline mAP — only saves finetune_best_map.pth.tar
                        when the current epoch improves on this value.
        epoch_offset:   added to the epoch index for wandb x-axis continuity
                        (pass 0 when this is a standalone resume run).
    """
    print(f"\n{'─'*60}")
    print(f"  PHASE 3: Global backbone fine-tuning  ({FASE3_EPOCHS} epochs)")
    print(f"  LR={LR_FASE3}  |  FULLY UNFROZEN  |  patience={PATIENCE}")
    print(f"  Baseline  val_loss={best_val_loss:.4f}  |  mAP={best_map:.4f}")
    print(f"{'─'*60}\n")

    scaler      = torch.amp.GradScaler("cuda")
    optimizer   = optim.AdamW(model.parameters(), lr=LR_FASE3, weight_decay=config.WEIGHT_DECAY)
    scheduler   = CosineAnnealingLR(optimizer, T_max=FASE3_EPOCHS, eta_min=1e-7)

    epochs_no_improve = 0

    for epoch in range(FASE3_EPOCHS):
        print(f"[Phase 3]  Epoch {epoch+1}/{FASE3_EPOCHS}  |  LR={scheduler.get_last_lr()[0]:.2e}")

        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
        val_loss   = val_fn(val_loader,   model, loss_fn, scaled_anchors)
        scheduler.step()

        log_dict = {
            "phase":      3,
            "epoch":      epoch_offset + epoch + 1,
            "train/loss": train_loss,
            "val/loss":   val_loss,
            "lr":         scheduler.get_last_lr()[0],
        }

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, filename=config.FINETUNE_BEST)
            print(f"  ✅ New best val_loss: {best_val_loss:.4f} → finetune_best.pth.tar")
        else:
            epochs_no_improve += 1
            print(f"  ⏳ No improvement: {epochs_no_improve}/{PATIENCE}")

        if (epoch + 1) % MAP_EVAL_FREQ == 0:
            class_acc, noobj_acc, obj_acc = check_class_accuracy(
                model, val_loader, threshold=config.MAP_CONF_THRESHOLD
            )
            pred_boxes, true_boxes = get_evaluation_bboxes(
                val_loader, model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.MAP_CONF_THRESHOLD,
                device=config.DEVICE,
            )
            map_val = mean_average_precision(
                pred_boxes, true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.SPECIFIC_NUM_CLASSES,
            ).item()
            print(f"  mAP@{config.MAP_IOU_THRESH}: {map_val:.4f}")
            if map_val > best_map:
                best_map = map_val
                save_checkpoint(model, optimizer, filename=config.FINETUNE_BEST_MAP)
                print(f"  ✅ New best mAP: {best_map:.4f} → finetune_best_map.pth.tar")
            log_dict.update({
                "eval/mAP":       map_val,
                "eval/class_acc": class_acc,
                "eval/obj_acc":   obj_acc,
                "eval/noobj_acc": noobj_acc,
            })
            model.train()

        wandb.log(log_dict)

        if epochs_no_improve >= PATIENCE:
            print(f"\n  Early stopping triggered after {PATIENCE} epochs without improvement.")
            break

    print(f"\n{'='*60}")
    print(f"  Phase 3 complete.")
    print(f"  Best val_loss : {best_val_loss:.4f}  →  {config.FINETUNE_BEST}")
    print(f"  Best mAP      : {best_map:.4f}  →  {config.FINETUNE_BEST_MAP}")
    print(f"{'='*60}\n")


# ============================================================
# --resume_phase3 ENTRY POINT
# ============================================================

def _main_resume_phase3(checkpoint_path: str) -> None:
    """
    Skips phases 1-2 and runs phase 3 only from the given checkpoint.

    The val_loss and mAP baselines are measured on the val set so
    early-stopping thresholds are always accurate regardless of which
    checkpoint is passed.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_absolute():
        ckpt_path = config.BASE_DIR / ckpt_path

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Pass a path relative to the project root or an absolute path."
        )

    print(f"\n{'='*60}")
    print(f"  YOLOv4 Fine-Tuning — PHASE 3 RESUME")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Device     : {config.DEVICE}")
    print(f"  Classes    : {config.SPECIFIC_NUM_CLASSES}")
    print(f"{'='*60}\n")

    # Build model with the target class count (not the generic 1-class backbone)
    model = YOLOv4(num_classes=config.SPECIFIC_NUM_CLASSES).to(config.DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print(f"  ✅ Weights loaded from: {ckpt_path.name}\n")

    # Dataloaders
    for csv_path in (config.TRAIN_CSV, config.VAL_CSV):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"CSV not found: {csv_path}\n"
                "Run: python src/scripts/generate_csv.py --dataset specific"
            )

    train_loader, val_loader, _ = get_loaders(
        train_csv_path=config.TRAIN_CSV,
        val_csv_path=config.VAL_CSV,
        train_img_dir=config.IMG_DIR,
        train_label_dir=config.LABEL_DIR,
        val_img_dir=config.VAL_IMG_DIR,
        val_label_dir=config.VAL_LABEL_DIR,
        mosaic_prob=0.5,
    )
    print(f"  Dataloaders ready:  train={len(train_loader.dataset)} imgs | "
          f"val={len(val_loader.dataset)} imgs\n")

    loss_fn = YoloLoss()
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    # Compute val_loss baseline from the loaded checkpoint (source of truth —
    # avoids hardcoding numbers that become stale as training progresses).
    print("  Computing val_loss baseline from checkpoint...")
    model.eval()
    baseline_val_loss = val_fn(val_loader, model, loss_fn, scaled_anchors)
    print(f"  ✅ Baseline val_loss = {baseline_val_loss:.4f}\n")

    # Compute mAP baseline so finetune_best_map.pth.tar is only overwritten
    # by a genuinely better model, not just the first phase-3 mAP eval.
    print("  Computing mAP baseline from checkpoint...")
    pred_boxes, true_boxes = get_evaluation_bboxes(
        val_loader, model,
        iou_threshold=config.NMS_IOU_THRESH,
        anchors=config.ANCHORS,
        threshold=config.MAP_CONF_THRESHOLD,
        device=config.DEVICE,
    )
    baseline_map = mean_average_precision(
        pred_boxes, true_boxes,
        iou_threshold=config.MAP_IOU_THRESH,
        box_format="midpoint",
        num_classes=config.SPECIFIC_NUM_CLASSES,
    ).item()
    print(f"  ✅ Baseline mAP@{config.MAP_IOU_THRESH} = {baseline_map:.4f}\n")
    model.train()

    # Unfreeze entire model for phase 3
    unfreeze_all(model)

    wandb.init(
        project="yolov4-finetune",
        name=f"phase3-resume-{ckpt_path.stem}",
        config={
            "mode":               "resume_phase3",
            "checkpoint":         str(ckpt_path),
            "fase3_epochs":       FASE3_EPOCHS,
            "lr_fase3":           LR_FASE3,
            "batch_size":         config.BATCH_SIZE,
            "num_classes":        config.SPECIFIC_NUM_CLASSES,
            "patience":           PATIENCE,
            "baseline_val_loss":  baseline_val_loss,
            "baseline_map":       baseline_map,
        },
    )

    _run_phase3(
        model, train_loader, val_loader, loss_fn, scaled_anchors,
        best_val_loss=baseline_val_loss,
        best_map=baseline_map,
        epoch_offset=0,
    )

    wandb.finish()


# ============================================================
# FULL PIPELINE ENTRY POINT  (phases 1 → 2 → 3)
# ============================================================

def main() -> None:
    args = _parse_args()

    # --resume_phase3 takes a completely separate code path
    if args.resume_phase3:
        _main_resume_phase3(args.resume_phase3)
        return

    print(f"\n{'='*60}")
    print(f"  YOLOv4 Fine-Tuning  |  device: {config.DEVICE}")
    print(f"  Dataset: data/yolo_dataset  |  Classes: {config.GENERIC_NUM_CLASSES} → {config.SPECIFIC_NUM_CLASSES}")
    print(f"{'='*60}\n")

    # ----------------------------------------------------------
    # 1. LOAD MODEL WITH PRETRAINING ARCHITECTURE
    # ----------------------------------------------------------
    model = YOLOv4(num_classes=config.GENERIC_NUM_CLASSES).to(config.DEVICE)

    print("Loading pretrained weights from generic dataset...")
    if not os.path.exists(config.CHECKPOINT_FILE):
        raise FileNotFoundError(
            f"Checkpoint not found: {config.CHECKPOINT_FILE}\n"
            "Make sure you trained first with the generic dataset."
        )
    checkpoint = torch.load(config.CHECKPOINT_FILE, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    print("  ✅ Weights loaded successfully.\n")

    # ----------------------------------------------------------
    # 2. REPLACE DETECTION HEADS (generic → specific classes)
    # ----------------------------------------------------------
    print(f"Replacing heads: {config.GENERIC_NUM_CLASSES} → {config.SPECIFIC_NUM_CLASSES} classes...")
    model.head_large  = ScalePrediction(256, config.SPECIFIC_NUM_CLASSES).to(config.DEVICE)
    model.head_medium = ScalePrediction(256, config.SPECIFIC_NUM_CLASSES).to(config.DEVICE)
    model.head_small  = ScalePrediction(128, config.SPECIFIC_NUM_CLASSES).to(config.DEVICE)

    initialize_weights(model.head_large)
    initialize_weights(model.head_medium)
    initialize_weights(model.head_small)
    print("  ✅ Heads initialized.\n")

    # ----------------------------------------------------------
    # 3. DATALOADERS
    # ----------------------------------------------------------
    for csv_path in (config.TRAIN_CSV, config.VAL_CSV):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"CSV not found: {csv_path}\n"
                "Run: python src/scripts/generate_csv.py --dataset specific"
            )

    train_loader, val_loader, _ = get_loaders(
        train_csv_path=config.TRAIN_CSV,
        val_csv_path=config.VAL_CSV,
        train_img_dir=config.IMG_DIR,
        train_label_dir=config.LABEL_DIR,
        val_img_dir=config.VAL_IMG_DIR,
        val_label_dir=config.VAL_LABEL_DIR,
        mosaic_prob=0.5,
    )
    print(f"  Dataloaders ready:  train={len(train_loader.dataset)} imgs | "
          f"val={len(val_loader.dataset)} imgs\n")

    # ----------------------------------------------------------
    # 4. SHARED COMPONENTS
    # ----------------------------------------------------------
    loss_fn = YoloLoss()
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    # ----------------------------------------------------------
    # 5. WANDB
    # ----------------------------------------------------------
    wandb.init(
        project="yolov4-finetune",
        config={
            "fase1_epochs": FASE1_EPOCHS,
            "fase2_epochs": FASE2_EPOCHS,
            "lr_fase1":     LR_FASE1,
            "lr_fase2":     LR_FASE2,
            "batch_size":   config.BATCH_SIZE,
            "num_classes":  config.SPECIFIC_NUM_CLASSES,
            "image_size":   config.IMAGE_SIZE,
            "device":       config.DEVICE,
            "patience":     PATIENCE,
        },
    )

    # ──────────────────────────────────────────────────────────
    # PHASE 1: Heads only
    # ──────────────────────────────────────────────────────────
    print(f"{'─'*60}")
    print(f"  PHASE 1: Training heads only  ({FASE1_EPOCHS} epochs)")
    print(f"  LR={LR_FASE1}  |  backbone+SPP+neck FROZEN")
    print(f"{'─'*60}\n")

    freeze_all_except_heads(model)

    scaler      = torch.amp.GradScaler("cuda")
    optimizer_f1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_FASE1,
        weight_decay=config.WEIGHT_DECAY,
    )

    for epoch in range(FASE1_EPOCHS):
        print(f"[Phase 1]  Epoch {epoch+1}/{FASE1_EPOCHS}")
        train_loss = train_fn(train_loader, model, optimizer_f1, loss_fn, scaler, scaled_anchors)
        val_loss   = val_fn(val_loader, model, loss_fn, scaled_anchors)

        wandb.log({
            "phase":      1,
            "epoch":      epoch + 1,
            "train/loss": train_loss,
            "val/loss":   val_loss,
            "lr":         optimizer_f1.param_groups[0]["lr"],
        })

    if config.SAVE_MODEL:
        save_checkpoint(model, optimizer_f1, filename=config.FINETUNE_CHECKPOINT)
        print("  Phase 1 checkpoint saved.\n")

    # ──────────────────────────────────────────────────────────
    # PHASE 2: Neck + SPP + heads
    # ──────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  PHASE 2: Fine-tuning neck + SPP + heads  ({FASE2_EPOCHS} epochs)")
    print(f"  LR={LR_FASE2}  |  backbone FROZEN  |  early stopping (patience={PATIENCE})")
    print(f"{'─'*60}\n")

    unfreeze_neck_and_spp(model)

    scaler      = torch.amp.GradScaler("cuda")
    optimizer_f2 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_FASE2,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler_f2 = CosineAnnealingLR(optimizer_f2, T_max=FASE2_EPOCHS, eta_min=1e-7)

    best_val_loss     = float("inf")
    best_map          = 0.0
    epochs_no_improve = 0
    phase2_epochs_run = 0

    for epoch in range(FASE2_EPOCHS):
        print(f"[Phase 2]  Epoch {epoch+1}/{FASE2_EPOCHS}  |  LR={scheduler_f2.get_last_lr()[0]:.2e}")
        train_loss = train_fn(train_loader, model, optimizer_f2, loss_fn, scaler, scaled_anchors)
        val_loss   = val_fn(val_loader, model, loss_fn, scaled_anchors)
        scheduler_f2.step()
        phase2_epochs_run += 1

        log_dict = {
            "phase":      2,
            "epoch":      FASE1_EPOCHS + epoch + 1,
            "train/loss": train_loss,
            "val/loss":   val_loss,
            "lr":         scheduler_f2.get_last_lr()[0],
        }

        if config.SAVE_MODEL:
            save_checkpoint(model, optimizer_f2, filename=config.FINETUNE_CHECKPOINT)

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            save_checkpoint(model, optimizer_f2, filename=config.FINETUNE_BEST)
            print(f"  ✅ Best val_loss: {best_val_loss:.4f} → finetune_best.pth.tar")
        else:
            epochs_no_improve += 1
            print(f"  ⏳ No improvement: {epochs_no_improve}/{PATIENCE}")

        if (epoch + 1) % MAP_EVAL_FREQ == 0:
            class_acc, noobj_acc, obj_acc = check_class_accuracy(
                model, val_loader, threshold=config.MAP_CONF_THRESHOLD
            )
            pred_boxes, true_boxes = get_evaluation_bboxes(
                val_loader, model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.MAP_CONF_THRESHOLD,
                device=config.DEVICE,
            )
            map_val = mean_average_precision(
                pred_boxes, true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.SPECIFIC_NUM_CLASSES,
            ).item()
            print(f"  mAP@{config.MAP_IOU_THRESH}: {map_val:.4f}")
            if map_val > best_map:
                best_map = map_val
                save_checkpoint(model, optimizer_f2, filename=config.FINETUNE_BEST_MAP)
                print(f"  ✅ Best mAP: {best_map:.4f} → finetune_best_map.pth.tar")
            log_dict.update({
                "eval/mAP":       map_val,
                "eval/class_acc": class_acc,
                "eval/obj_acc":   obj_acc,
                "eval/noobj_acc": noobj_acc,
            })
            model.train()

        wandb.log(log_dict)

        if epochs_no_improve >= PATIENCE:
            print(f"\n  Early stopping triggered after {PATIENCE} epochs without improvement.")
            break

    # ──────────────────────────────────────────────────────────
    # PHASE 3: Full model (total unfreezing)
    # Reload the best phase-2 state so the backbone starts from
    # the strongest supervised baseline, not from wherever early
    # stopping happened to leave the weights.
    # ──────────────────────────────────────────────────────────
    print("\n  Reloading best Phase 2 checkpoint before backbone unfreezing...")
    ckpt = torch.load(config.FINETUNE_BEST, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print(f"  ✅ Restored best val_loss={best_val_loss:.4f} state.\n")

    unfreeze_all(model)

    _run_phase3(
        model, train_loader, val_loader, loss_fn, scaled_anchors,
        best_val_loss=best_val_loss,
        best_map=best_map,
        epoch_offset=FASE1_EPOCHS + phase2_epochs_run,
    )

    wandb.finish()


if __name__ == "__main__":
    main()

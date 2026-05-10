"""
YOLOv4 Fine-Tuning Pipeline
============================

Loads model pretrained with generic dataset and adapts it to specific
supermarket dataset (20 classes) through a 2-phase process:

    Phase 1 — Heads only   (backbone + SPP + neck frozen)
    ─────────────────────────────────────────────────────
    • Goal: stabilize new heads (random weights)
    • High LR (1e-4) because no risk of destroying frozen backbone
    • No scheduler: gradients are chaotic at first

    Phase 2 — Neck + SPP + heads   (backbone still frozen)
    ─────────────────────────────────────────────────────
    • Goal: fine-tune neck and SPP with new classes
    • Low LR (1e-5) + CosineAnnealingLR for smooth descent
    • Early stopping: saves best checkpoint if val_loss improves
    • Evaluates mAP every MAP_EVAL_FREQ epochs

Backbone remains frozen throughout fine-tuning to preserve
visual features learned from generic dataset.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from core.model import YOLOv4, ScalePrediction, initialize_weights
from core.loss import YoloLoss
from tqdm import tqdm
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
FASE1_EPOCHS   = 15       # Heads only
FASE2_EPOCHS   = 100      # Neck + SPP + heads
LR_FASE1       = 1e-4     # Higher: heads start from random weights
LR_FASE2       = 5e-6     # Lower: fine-tune without destroying learning
MAP_EVAL_FREQ  = 5        # Evaluate mAP every N epochs in phase 2
PATIENCE       = 15       # Early stopping: epochs without val_loss improvement
FASE3_EPOCHS   = 30       # Full unfreezing
LR_FASE3       = 1e-6     # Very low to not break backbone

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
# MAIN
# ============================================================

def main():
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
    # 2. REPLACE DETECTION HEADS (85 → 20 classes)
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
    # 3. DATALOADERS FOR SPECIFIC DATASET
    # ----------------------------------------------------------
    # Check if CSVs exist; if not, generate them
    for csv_path in (config.TRAIN_CSV, config.VAL_CSV):
        if not os.path.exists(csv_path):
            print(f"CSV not found: {csv_path}")
            print("Run first: python generate_csv.py --dataset specific")
            raise FileNotFoundError(csv_path)

    train_loader, val_loader, _ = get_loaders(
        train_csv_path=config.TRAIN_CSV,
        val_csv_path=config.VAL_CSV,
        train_img_dir=config.IMG_DIR,
        train_label_dir=config.LABEL_DIR,
        val_img_dir=config.VAL_IMG_DIR,
        val_label_dir=config.VAL_LABEL_DIR,
    )
    print(f"Dataloaders listos:  train={len(train_loader.dataset)} imgs | "
          f"val={len(val_loader.dataset)} imgs\n")

    # ----------------------------------------------------------
    # 4. COMPONENTES COMUNES
    # ----------------------------------------------------------
    loss_fn = YoloLoss()

    scaler = torch.amp.GradScaler("cuda")

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
            "fase1_epochs":  FASE1_EPOCHS,
            "fase2_epochs":  FASE2_EPOCHS,
            "lr_fase1":      LR_FASE1,
            "lr_fase2":      LR_FASE2,
            "batch_size":    config.BATCH_SIZE,
            "num_classes":   config.SPECIFIC_NUM_CLASSES,
            "image_size":    config.IMAGE_SIZE,
            "device":        config.DEVICE,
            "patience":      PATIENCE,
        },
    )

    # ──────────────────────────────────────────────────────────
    # FASE 1: Solo cabezas
    # ──────────────────────────────────────────────────────────
    print(f"{'─'*60}")
    print(f"  FASE 1: Entrenando solo las cabezas  ({FASE1_EPOCHS} epochs)")
    print(f"  LR={LR_FASE1}  |  backbone+SPP+neck CONGELADOS")
    print(f"{'─'*60}\n")

    freeze_all_except_heads(model)

    optimizer_f1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_FASE1,
        weight_decay=config.WEIGHT_DECAY,
    )

    for epoch in range(FASE1_EPOCHS):
        print(f"[Fase 1]  Epoch {epoch+1}/{FASE1_EPOCHS}")
        train_loss = train_fn(train_loader, model, optimizer_f1, loss_fn, scaler, scaled_anchors)
        val_loss   = val_fn(val_loader, model, loss_fn, scaled_anchors)

        wandb.log({
            "fase": 1,
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "val/loss":   val_loss,
            "lr": optimizer_f1.param_groups[0]["lr"],
        })

        if config.SAVE_MODEL:
            save_checkpoint(model, optimizer_f1, filename=config.FINETUNE_CHECKPOINT)

    # ──────────────────────────────────────────────────────────
    # FASE 2: Neck + SPP + cabezas
    # ──────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  FASE 2: Afinando neck + SPP + cabezas  ({FASE2_EPOCHS} epochs)")
    print(f"  LR={LR_FASE2}  |  backbone CONGELADO  |  early stopping (patience={PATIENCE})")
    print(f"{'─'*60}\n")

    unfreeze_neck_and_spp(model)

    optimizer_f2 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_FASE2,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer_f2, T_max=FASE2_EPOCHS, eta_min=1e-7)

    best_val_loss    = float("inf")
    epochs_no_improve = 0

    for epoch in range(FASE2_EPOCHS):
        print(f"[Fase 2]  Epoch {epoch+1}/{FASE2_EPOCHS}  |  LR={scheduler.get_last_lr()[0]:.2e}")
        train_loss = train_fn(train_loader, model, optimizer_f2, loss_fn, scaler, scaled_anchors)
        val_loss   = val_fn(val_loader, model, loss_fn, scaled_anchors)
        scheduler.step()

        log_dict = {
            "fase": 2,
            "epoch": FASE1_EPOCHS + epoch + 1,
            "train/loss": train_loss,
            "val/loss":   val_loss,
            "lr": scheduler.get_last_lr()[0],
        }

        # Guardar checkpoint periódico
        if config.SAVE_MODEL:
            save_checkpoint(model, optimizer_f2, filename=config.FINETUNE_CHECKPOINT)

        # Guardar el mejor checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(model, optimizer_f2, filename=config.FINETUNE_BEST)
            print(f"  ✅ Mejor val_loss: {best_val_loss:.4f} → guardado como finetune_best.pth.tar")
        else:
            epochs_no_improve += 1
            print(f"  ⏳ Sin mejora: {epochs_no_improve}/{PATIENCE}")

        # Evaluar mAP periódicamente
        if (epoch + 1) % MAP_EVAL_FREQ == 0:
            class_acc, noobj_acc, obj_acc = check_class_accuracy(
                model, val_loader, threshold=config.CONF_THRESHOLD
            )
            pred_boxes, true_boxes = get_evaluation_bboxes(
                val_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
                device=config.DEVICE,
            )
            map_val = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.SPECIFIC_NUM_CLASSES,
            )
            print(f"  mAP@{config.MAP_IOU_THRESH}: {map_val.item():.4f}")
            log_dict.update({
                "eval/mAP":      map_val.item(),
                "eval/class_acc": class_acc,
                "eval/obj_acc":   obj_acc,
                "eval/noobj_acc": noobj_acc,
            })
            model.train()

        wandb.log(log_dict)

        # Early stopping
        if epochs_no_improve >= PATIENCE:
            print(f"\n  Early stopping activado tras {PATIENCE} epochs sin mejora.")
            break


    # ──────────────────────────────────────────────────────────
    # FASE 3: Todo el modelo (Descongelación total)
    # ──────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  FASE 3: Fine-tuning global del Backbone  ({FASE3_EPOCHS} epochs)")
    print(f"  LR={LR_FASE3}  |  TODO DESCONGELADO")
    print(f"{'─'*60}\n")

    unfreeze_all(model)

    optimizer_f3 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_FASE3,
        weight_decay=config.WEIGHT_DECAY,
    )
    
    scheduler_f3 = CosineAnnealingLR(optimizer_f3, T_max=FASE3_EPOCHS, eta_min=1e-7)

    epochs_no_improve = 0 # Reiniciamos la paciencia

    for epoch in range(FASE3_EPOCHS):
        print(f"[Fase 3]  Epoch {epoch+1}/{FASE3_EPOCHS}  |  LR={scheduler_f3.get_last_lr()[0]:.2e}")
        train_loss = train_fn(train_loader, model, optimizer_f3, loss_fn, scaler, scaled_anchors)
        val_loss   = val_fn(val_loader, model, loss_fn, scaled_anchors)
        scheduler_f3.step()

        # Guardar el mejor checkpoint global
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(model, optimizer_f3, filename=config.FINETUNE_BEST)
            print(f"  ✅ Nuevo mejor val_loss global: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1

        wandb.log({
            "fase": 3,
            "epoch": FASE1_EPOCHS + FASE2_EPOCHS + epoch + 1,
            "train/loss": train_loss,
            "val/loss":   val_loss,
            "lr": scheduler_f3.get_last_lr()[0],
        })

        if epochs_no_improve >= PATIENCE:
            print("  Early stopping activado en Fase 3.")
            break

    print(f"\n{'='*60}")
    print(f"  Fine-tuning completado.")
    print(f"  Mejor val_loss: {best_val_loss:.4f}")
    print(f"  Checkpoint guardado en: {config.FINETUNE_BEST}")
    print(f"{'='*60}\n")

    wandb.finish()


if __name__ == "__main__":
    main()
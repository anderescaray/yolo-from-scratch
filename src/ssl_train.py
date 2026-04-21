"""
YOLOv4 SSL Training Pipeline
==============================

Entrena el modelo YOLOv4 con datos labelled + pseudo-labelled combinados
según el framework STAC.

Parte del mejor checkpoint del fine-tuning supervisado y continúa
entrenando con el dataset mixto (labelled + pseudo-labels con strong augmentation).

Uso:
    python src/ssl_train.py
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from model import YOLOv4
from loss import YoloLoss
from train import train_fn, val_fn
from ssl_dataset import get_ssl_loader
from utils import (
    save_checkpoint,
    check_class_accuracy,
    get_evaluation_bboxes,
    mean_average_precision,
)

import wandb
import warnings
import os
import cv2

cv2.setNumThreads(0)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True


# ============================================================
# HIPERPARÁMETROS SSL
# ============================================================
SSL_EPOCHS     = 50
SSL_LR         = 1e-5
PATIENCE       = 10
MAP_EVAL_FREQ  = 5


def main():
    print(f"\n{'='*60}")
    print(f"  YOLOv4 SSL Training (STAC)")
    print(f"  Device: {config.DEVICE}")
    print(f"  Clases: {config.SPECIFIC_NUM_CLASSES}")
    print(f"{'='*60}\n")

    # ----------------------------------------------------------
    # 1. CARGAR MODELO DESDE FINETUNE BEST
    # ----------------------------------------------------------
    model = YOLOv4(num_classes=config.SPECIFIC_NUM_CLASSES).to(config.DEVICE)

    weights_path = config.FINETUNE_BEST
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"No se encontró el checkpoint: {weights_path}\n"
            "Asegúrate de haber completado el fine-tuning supervisado primero."
        )

    print(f"Cargando pesos desde: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    print("  ✅ Pesos cargados.\n")

    # ----------------------------------------------------------
    # 2. DATALOADERS COMBINADOS
    # ----------------------------------------------------------
    if not os.path.exists(config.PSEUDO_CSV):
        raise FileNotFoundError(
            f"No se encontró: {config.PSEUDO_CSV}\n"
            "Ejecuta primero: python src/pseudo_labeler.py --weights checkpoints/finetune_best.pth.tar"
        )

    ssl_loader, val_loader = get_ssl_loader()

    # ----------------------------------------------------------
    # 3. COMPONENTES
    # ----------------------------------------------------------
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=SSL_LR,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=SSL_EPOCHS, eta_min=1e-7)

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    # ----------------------------------------------------------
    # 4. WANDB
    # ----------------------------------------------------------
    wandb.init(
        project="yolov4-ssl",
        config={
            "ssl_epochs":   SSL_EPOCHS,
            "ssl_lr":       SSL_LR,
            "batch_size":   config.BATCH_SIZE,
            "num_classes":  config.SPECIFIC_NUM_CLASSES,
            "tau":          config.SSL_TAU,
            "patience":     PATIENCE,
        },
    )

    # ----------------------------------------------------------
    # 5. BUCLE DE ENTRENAMIENTO SSL
    # ----------------------------------------------------------
    print(f"\n{'─'*60}")
    print(f"  SSL Training  ({SSL_EPOCHS} epochs)")
    print(f"  LR={SSL_LR}  |  patience={PATIENCE}")
    print(f"{'─'*60}\n")

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(SSL_EPOCHS):
        print(f"[SSL]  Epoch {epoch+1}/{SSL_EPOCHS}  |  LR={scheduler.get_last_lr()[0]:.2e}")

        train_loss = train_fn(ssl_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
        val_loss   = val_fn(val_loader, model, loss_fn, scaled_anchors)
        scheduler.step()

        log_dict = {
            "epoch":       epoch + 1,
            "train/loss":  train_loss,
            "val/loss":    val_loss,
            "lr":          scheduler.get_last_lr()[0],
        }

        # Best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, filename=config.SSL_BEST)
            print(f"  ✅ Mejor val_loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  ⏳ Sin mejora: {epochs_no_improve}/{PATIENCE}")

        # mAP cada N epochs
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
                "eval/mAP":       map_val.item(),
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

    print(f"\n{'='*60}")
    print(f"  SSL Training completado.")
    print(f"  Mejor val_loss: {best_val_loss:.4f}")
    print(f"  Checkpoint guardado en: {config.SSL_BEST}")
    print(f"{'='*60}\n")

    wandb.finish()


if __name__ == "__main__":
    main()

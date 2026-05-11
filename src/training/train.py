"""
YOLOv4 Training Pipeline
========================

This script combines model, data, loss and optimizer to train the model.

Uses float16 instead of float32 to double speed and reduce GPU VRAM usage.
Calculates loss for 3 scales simultaneously.
Calculates mAP (Mean Average Precision) periodically.

"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import core.config as config
import torch
import torch.optim as optim
from core.model import YOLOv4
from tqdm import tqdm 
from core.loss import YoloLoss
import warnings
import wandb
import cv2

cv2.setNumThreads(0)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from core.utils import (
    mean_average_precision,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)

warnings.filterwarnings("ignore")

# So PyTorch finds the fastest conv algorithm for available hardware
torch.backends.cudnn.benchmark = True


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    """
    Execute one training epoch.

    Args:
        train_loader: DataLoader providing batches
        model: YOLOv4 model
        optimizer: AdamW optimizer updating weights
        loss_fn: YoloLoss
        scaler: GradScaler for FP16
        scaled_anchors: anchors adjusted to grid size (13, 26, 52)
    """
    # Progress bar to see progress
    loop = tqdm(train_loader, leave=True)
    losses = []

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE, non_blocking=True)
        y0, y1, y2 = (
            y[0].to(config.DEVICE, non_blocking=True),
            y[1].to(config.DEVICE, non_blocking=True),
            y[2].to(config.DEVICE, non_blocking=True),
        )
        with torch.amp.autocast("cuda"):
            out = model(x)

        # OUTSIDE autocast, to convert predictions to Float32
        # to prevent w/h^2 from being w/0.0 and giving inf
        out0 = out[0].float()
        out1 = out[1].float()
        out2 = out[2].float()

        loss = (
            loss_fn(out0, y0, scaled_anchors[0])
            + loss_fn(out1, y1, scaled_anchors[1])
            + loss_fn(out2, y2, scaled_anchors[2])
        )

        # Backpropagation
        losses.append(loss.item())
        optimizer.zero_grad() # Clear previous gradients

        # Scale loss (necessary for float16 to prevent underflow)
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        # Gradient clipping: if any gradient > 1.0, limit it
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer) # Update weights
        scaler.update()

        # Update progress bar
        # Show current average error
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)

    # Return mean loss for the epoch
    return mean_loss

def val_fn(val_loader, model, loss_fn, scaled_anchors):
    """
    Calculate loss on validation set without updating weights.
    Runs in eval() mode to deactivate dropout and BN train mode.
    Allows comparing train loss vs val loss to detect overfitting.

    Args:
        val_loader:     Validation set DataLoader
        model:          YOLOv4 model
        loss_fn:        YoloLoss
        scaled_anchors: anchors adjusted to grid size (13, 26, 52)

    Returns:
        mean_loss: mean loss over entire validation set
    """
    model.eval() # Deactivate BatchNorm train mode and dropout
    losses = []

    # torch.no_grad() to not build gradient graph (faster and less memory)
    with torch.no_grad():
        loop = tqdm(val_loader, leave=True, desc="Validation")
        for x, y in loop:
            x = x.to(config.DEVICE, non_blocking=True)
            y0, y1, y2 = (
                y[0].to(config.DEVICE, non_blocking=True),
                y[1].to(config.DEVICE, non_blocking=True),
                y[2].to(config.DEVICE, non_blocking=True),
            )
            with torch.amp.autocast("cuda"):
                out = model(x)
            out0 = out[0].float()
            out1 = out[1].float()
            out2 = out[2].float()
            loss = (
                loss_fn(out0, y0, scaled_anchors[0]) # large
                + loss_fn(out1, y1, scaled_anchors[1]) # medium
                + loss_fn(out2, y2, scaled_anchors[2]) # small
            )
            losses.append(loss.item())
            mean_loss = sum(losses) / len(losses)
            loop.set_postfix(val_loss=mean_loss)

    model.train() # Return to train mode for next epoch
    return mean_loss

def main():
    """Main configuration function and epoch loop"""

    # train.py always operates on the generic dataset.
    assert config.DATASET_TYPE == "generic", (
        "train.py requires DATASET_TYPE='generic' in config.py. "
        f"Current value: '{config.DATASET_TYPE}'."
    )

    # Initialize wandb
    # view results at https://wandb.ai
    wandb.init(
        project="yolov4-supermarket",  # project name in wandb
        config={
            "learning_rate": config.LEARNING_RATE,
            "weight_decay": config.WEIGHT_DECAY,
            "batch_size": config.BATCH_SIZE,
            "epochs": config.NUM_EPOCHS,
            "num_classes": config.GENERIC_NUM_CLASSES,
            "image_size": config.IMAGE_SIZE,
            "device": config.DEVICE,
        }
    )

    # Initialize model, optimizer and loss
    model = YOLOv4(num_classes=config.GENERIC_NUM_CLASSES).to(config.DEVICE)

    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
        )

    loss_fn = YoloLoss()
    scaler = torch.amp.GradScaler("cuda")

    # Load generic dataset (DATASET_TYPE = "generic" in config.py)
    train_loader, val_loader, train_eval_loader = get_loaders(
        train_csv_path=config.TRAIN_CSV,
        val_csv_path=config.VAL_CSV,
    )

    # Load checkpoint (if you want to continue training a saved one)
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )

    # Anchor scaling
    # Anchors in config are normalized (0-1)
    # And Loss Function needs them in grid units
    # --> Anchor * Grid = Grid units
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    patience = 15
    patience_counter = 0
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch: {epoch+1}/{config.NUM_EPOCHS}")

        # Train one complete pass
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
        val_loss = val_fn(val_loader, model, loss_fn, scaled_anchors)

        # Log both losses together in wandb to compare in same graph
        wandb.log({
            "train/loss": train_loss,
            "val/loss":   val_loss,
            "epoch":      epoch + 1,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save model
            if config.SAVE_MODEL:
                save_checkpoint(model, optimizer, filename=config.CHECKPOINT_FILE)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Evaluate accuracy (mAP) every 5 epochs (slow so don't do it always)
        if epoch > 0 and epoch % 5 == 0:
            class_acc, noobj_acc, obj_acc = check_class_accuracy(model, val_loader, threshold=config.CONF_THRESHOLD)
            pred_boxes, true_boxes = get_evaluation_bboxes(
                val_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            map_val = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.GENERIC_NUM_CLASSES,
            )
            print(f"mAP: {map_val.item()}")

            wandb.log({
                "eval/mAP":         map_val.item(),
                "eval/class_acc":   class_acc,
                "eval/obj_acc":     obj_acc,
                "eval/noobj_acc":   noobj_acc,
                "epoch":            epoch + 1,
            })

            model.train()
    wandb.finish()

if __name__ == "__main__":
    # train.py opera siempre sobre el dataset genérico.
    assert config.DATASET_TYPE == "generic", (
        "train.py requiere DATASET_TYPE='generic' en config.py. "
        f"Valor actual: '{config.DATASET_TYPE}'."
    )
    print(f"Using device: {config.DEVICE}")
    main()
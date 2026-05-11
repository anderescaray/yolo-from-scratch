"""
Test Visual — Generates images with model predictions
==============================================

Loads a checkpoint, predicts on test images and saves
images with drawn bounding boxes in saved_images/.

Change WEIGHTS_PATH to choose which checkpoint to use:
  - config.CHECKPOINT_FILE   → Generic pretraining
  - config.FINETUNE_BEST     → Best supervised fine-tuning
  - config.SSL_BEST          → Best SSL
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import core.config as config
import torch
import torch.optim as optim
from core.model import YOLOv4
from core.utils import load_checkpoint, get_loaders, plot_couple_examples

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ============================================================
# CONFIGURATION — Change the checkpoint you want to use here
# ============================================================
WEIGHTS_PATH = config.FINETUNE_BEST       # ← Change this line
NUM_CLASSES  = config.SPECIFIC_NUM_CLASSES # ← Change if using other dataset


def main():
    print(f"Loading model for visualization...")
    print(f"  Checkpoint: {WEIGHTS_PATH}")
    print(f"  Classes: {NUM_CLASSES}\n")

    # Create model
    model = YOLOv4(num_classes=NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # Load weights
    load_checkpoint(WEIGHTS_PATH, model, optimizer, config.LEARNING_RATE)

    # Test data
    _, test_loader, _ = get_loaders(
        train_csv_path=config.TRAIN_CSV,
        val_csv_path=config.TEST_CSV,
        val_img_dir=config.TEST_IMG_DIR,
        val_label_dir=config.TEST_LABEL_DIR,
    )

    # Create output folder
    os.makedirs(os.path.join(config.BASE_DIR, "saved_images"), exist_ok=True)

    # Generate and save images
    print("Generating predictions...")
    plot_couple_examples(model, test_loader, 0.7, config.NMS_IOU_THRESH, config.ANCHORS, config.specific_class_labels)

    print(f"\nLook for 'prediction_test_X.png' files in saved_images/")


if __name__ == "__main__":
    main()
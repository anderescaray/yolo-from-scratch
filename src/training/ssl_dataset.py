"""
SSL Dataset — Combines labeled and pseudo-labeled data
======================================================

Creates a ConcatDataset that mixes:
  1. Labeled dataset   → uses train_transforms (standard augmentation)
  2. Pseudo dataset    → uses strong_transforms (aggressive STAC augmentation)

Each batch contains samples from both sources mixed randomly
to prevent catastrophic forgetting.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import core.config as config
from core.dataset import YOLODataset
from torch.utils.data import ConcatDataset, DataLoader


def get_ssl_loader():
    """
    Builds a combined labeled + pseudo-labeled DataLoader.

    Returns:
        ssl_loader: Mixed DataLoader
        val_loader: Validation DataLoader (unchanged)
    """
    IMAGE_SIZE = config.IMAGE_SIZE
    S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
    num_classes = config.SPECIFIC_NUM_CLASSES

    # --- Dataset 1: Labeled (standard augmentation) ---
    labelled_dataset = YOLODataset(
        csv_file=config.TRAIN_CSV,
        transform=config.train_transforms,
        S=S,
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
        C=num_classes,
    )

    # --- Dataset 2: Pseudo-labeled (aggressive STAC augmentation) ---
    pseudo_dataset = YOLODataset(
        csv_file=config.PSEUDO_CSV,
        transform=config.strong_transforms,
        S=S,
        img_dir=config.UNLABELLED_IMG_DIR,    # original unlabeled images
        label_dir=config.PSEUDO_LABEL_DIR,    # generated pseudo-labels
        anchors=config.ANCHORS,
        C=num_classes,
    )

    # --- Concatenate both datasets ---
    combined_dataset = ConcatDataset([labelled_dataset, pseudo_dataset])

    ssl_loader = DataLoader(
        dataset=combined_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,           # Mix labeled + pseudo in each epoch
        drop_last=False,
        persistent_workers=True if config.NUM_WORKERS > 0 else False,
    )

    # --- Validation (labeled only, unchanged) ---
    val_dataset = YOLODataset(
        csv_file=config.VAL_CSV,
        transform=config.test_transforms,
        S=S,
        img_dir=config.VAL_IMG_DIR,
        label_dir=config.VAL_LABEL_DIR,
        anchors=config.ANCHORS,
        C=num_classes,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
        persistent_workers=True if config.NUM_WORKERS > 0 else False,
    )

    print(f"  SSL DataLoader:")
    print(f"    Labeled:         {len(labelled_dataset)} imgs")
    print(f"    Pseudo-labeled:  {len(pseudo_dataset)} imgs")
    print(f"    Total combined:  {len(combined_dataset)} imgs")
    print(f"    Validation:      {len(val_dataset)} imgs")

    return ssl_loader, val_loader

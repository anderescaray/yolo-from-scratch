"""
YOLOv4 Configuration and Hyperparameters
=========================================

This module centralizes all constants, hyperparameters and model configurations.
Supports two dataset modes:

    DATASET_TYPE = "generic"   → Pretraining with generic dataset (85 classes)
    DATASET_TYPE = "specific"  → Fine-tuning with specific dataset (20 classes)

Change DATASET_TYPE here and all paths and classes adapt automatically.
"""
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

# ============================================================
# CENTRAL DATASET CONTROL
# ============================================================
# Options: "generic" | "specific"
DATASET_TYPE = "specific"

# ============================================================
# BASE PATHS
# ============================================================
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True

BASE_DIR = Path(__file__).resolve().parent.parent.parent

if DATASET_TYPE == "generic":
    DATASET_DIR      = BASE_DIR / "data" / "generic_dataset"
    IMG_DIR          = DATASET_DIR / "train"
    LABEL_DIR        = DATASET_DIR / "train"
    VAL_IMG_DIR      = DATASET_DIR / "val"
    VAL_LABEL_DIR    = DATASET_DIR / "val"
    TRAIN_CSV        = DATASET_DIR / "train.csv"
    VAL_CSV          = DATASET_DIR / "val.csv"
else:  # "specific"
    DATASET_DIR      = BASE_DIR / "data" / "yolo_dataset"
    IMG_DIR          = DATASET_DIR / "train" / "labelled"
    LABEL_DIR        = DATASET_DIR / "train" / "labelled"
    VAL_IMG_DIR      = DATASET_DIR / "val"
    VAL_LABEL_DIR    = DATASET_DIR / "val"
    TRAIN_CSV        = DATASET_DIR / "train.csv"
    VAL_CSV          = DATASET_DIR / "val.csv"
    TEST_CSV         = DATASET_DIR / "test.csv"
    TEST_IMG_DIR     = DATASET_DIR / "test"
    TEST_LABEL_DIR   = DATASET_DIR / "test"
    # SSL
    UNLABELLED_IMG_DIR  = DATASET_DIR / "train" / "unlabelled"
    PSEUDO_LABEL_DIR    = DATASET_DIR / "train" / "pseudo_labelled"
    PSEUDO_CSV          = DATASET_DIR / "pseudo_train.csv"

# Checkpoint from generic pretraining (input for finetune)
CHECKPOINT_FILE     = BASE_DIR / "checkpoints" / "checkpoint.pth.tar"
FINETUNE_CHECKPOINT = BASE_DIR / "checkpoints" / "finetune_checkpoint.pth.tar"
FINETUNE_BEST       = BASE_DIR / "checkpoints" / "finetune_best.pth.tar"
SSL_BEST            = BASE_DIR / "checkpoints" / "ssl_best.pth.tar"

# ============================================================
# HYPERPARAMETERS
# ============================================================
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
BATCH_SIZE    = 24       
NUM_EPOCHS    = 300

CONF_THRESHOLD  = 0.5
NMS_IOU_THRESH  = 0.45
MAP_IOU_THRESH  = 0.5

# SSL Hyperparams
SSL_TAU         = 0.9   # Confidence threshold for pseudo-labels (WBF)
SSL_LOSS_WEIGHT = 1.0    # Relative weight of pseudo loss vs labeled

# ============================================================
# CLASSES
# ============================================================

# Specific dataset (fine-tuning) — 20 supermarket classes
specific_class_labels = [
    "coca_cola_bottle", "coca_cola_can", "orange_fanta_bottle", "heineken_can",
    "whole_milk", "semi_skimmed_milk", "skimmed_milk", "banana", "orange",
    "green_apple", "red_apple", "natural_yogurt", "stracciatella_yogurt",
    "shampoo_hs", "shampoo_hacendado", "ketchup", "mayonnaise",
    "fried_tomato", "york_ham", "turkey_ham"
]

GENERIC_NUM_CLASSES = 1 # Class-Agnostic
SPECIFIC_NUM_CLASSES = len(specific_class_labels)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# ARCHITECTURE
# ============================================================
IMAGE_SIZE = 416
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]  # [13, 26, 52]

# Tuples are convolutional blocks
# Lists are CSP residual blocks
CONFIG = [
    # --- BACKBONE CSPDarknet53 ---
    (32, 3, 1),                    # 416×416×32
    (64, 3, 2),                    # 208×208×64
    ["C", 1],

    (128, 3, 2),                   # 104×104×128
    ["C", 2],

    (256, 3, 2),                   # 52×52×256
    ["C", 8],                      # <-- Route connection (small objects)

    (512, 3, 2),                   # 26×26×512
    ["C", 8],                      # <-- Route connection (medium objects)

    (1024, 3, 2),                  # 13×13×1024
    ["C", 4],                      # SPP input
]

# ============================================================
# ANCHORS (COCO, scaled to [0,1])
# ============================================================
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],   # scale 13×13 (large objects)
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],  # scale 26×26 (medium)
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],  # scale 52×52 (small)
]

# ============================================================
# TRANSFORMATIONS
# ============================================================
SCALE = 1.2

train_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=int(IMAGE_SIZE * SCALE)),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5),
        A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

# Validation/test transformations (no augmentation)
test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

# Aggressive transformations for pseudo-labels (STAC)
strong_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=int(IMAGE_SIZE * SCALE)),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.15, p=0.8),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.CoarseDropout(
            num_holes_range=(2, 8),
            hole_height_range=(10, 40),
            hole_width_range=(10, 40),
            p=0.5),
        A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)
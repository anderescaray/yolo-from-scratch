"""
YOLOv4 Configuración e Hiperparámetros
========================================

Este módulo centraliza todas las constantes, hiperparámetros y configuraciones
del modelo. Soporta dos modos de dataset:

    DATASET_TYPE = "generic"   → Pre-entrenamiento con dataset genérico (85 clases)
    DATASET_TYPE = "specific"  → Fine-tuning con dataset específico (20 clases)

Cambia DATASET_TYPE aquí y todas las rutas y clases se adaptan automáticamente.
"""
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

# ============================================================
# CONTROL CENTRAL DE DATASET
# ============================================================
# Opciones: "generic" | "specific"
DATASET_TYPE = "specific"

# ============================================================
# RUTAS BASE
# ============================================================
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if DATASET_TYPE == "generic":
    DATASET_DIR      = os.path.join(BASE_DIR, "data", "generic_dataset")
    IMG_DIR          = os.path.join(DATASET_DIR, "train")
    LABEL_DIR        = os.path.join(DATASET_DIR, "train")
    VAL_IMG_DIR      = os.path.join(DATASET_DIR, "valid")
    VAL_LABEL_DIR    = os.path.join(DATASET_DIR, "valid")
    TRAIN_CSV        = os.path.join(DATASET_DIR, "train.csv")
    VAL_CSV          = os.path.join(DATASET_DIR, "test.csv")
else:  # "specific"
    DATASET_DIR      = os.path.join(BASE_DIR, "data", "yolo_dataset")
    IMG_DIR          = os.path.join(DATASET_DIR, "train", "labelled")
    LABEL_DIR        = os.path.join(DATASET_DIR, "train", "labelled")
    VAL_IMG_DIR      = os.path.join(DATASET_DIR, "val")
    VAL_LABEL_DIR    = os.path.join(DATASET_DIR, "val")
    TRAIN_CSV        = os.path.join(DATASET_DIR, "train.csv")
    VAL_CSV          = os.path.join(DATASET_DIR, "val.csv")
    TEST_CSV         = os.path.join(DATASET_DIR, "test.csv")
    TEST_IMG_DIR     = os.path.join(DATASET_DIR, "test")
    TEST_LABEL_DIR   = os.path.join(DATASET_DIR, "test")
    # SSL
    UNLABELLED_IMG_DIR  = os.path.join(DATASET_DIR, "train", "unlabelled")
    PSEUDO_LABEL_DIR    = os.path.join(DATASET_DIR, "train", "pseudo_labelled")
    PSEUDO_CSV          = os.path.join(DATASET_DIR, "pseudo_train.csv")

# Checkpoint del preentrenamiento genérico (input del finetune)
CHECKPOINT_FILE         = os.path.join(BASE_DIR, "checkpoints", "checkpoint.pth.tar")
# Checkpoint de salida del finetune
FINETUNE_CHECKPOINT     = os.path.join(BASE_DIR, "checkpoints", "finetune_checkpoint.pth.tar")
# Mejor checkpoint durante finetune (basado en val_loss)
FINETUNE_BEST           = os.path.join(BASE_DIR, "checkpoints", "finetune_best.pth.tar")
# Mejor checkpoint durante SSL
SSL_BEST                = os.path.join(BASE_DIR, "checkpoints", "ssl_best.pth.tar")

# ============================================================
# HIPERPARÁMETROS
# ============================================================
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
BATCH_SIZE    = 16       # Reducido para imágenes grandes del dataset específico
NUM_EPOCHS    = 300

CONF_THRESHOLD  = 0.5
NMS_IOU_THRESH  = 0.45
MAP_IOU_THRESH  = 0.5

# SSL Hyperparams
SSL_TAU         = 0.85   # Umbral de confianza para pseudo-labels (WBF)
SSL_LOSS_WEIGHT = 1.0    # Peso relativo de la loss pseudo vs labelled

# ============================================================
# CLASES
# ============================================================

# Dataset genérico (preentrenamiento)
generic_class_labels = [
    "-", "Apple", "Apple -Green-", "Apple -Red-", "Artichoke", "Asparagus",
    "Avocado", "Banana", "Beans", "Bell Pepper", "Blackberries", "Blueberries",
    "Book", "Boxed Food", "Bread", "Broccoli", "Brussel Sprouts", "Butter",
    "Cabbage", "Canned Fish", "Canned Food", "Cantaloupe", "Carrots", "Cauliflower",
    "Cereal", "Cerealbox", "Cheese", "Clementine", "Coffee", "Condiment",
    "Corn", "Creamer", "Cucumber", "Detergent", "Drinks", "Egg",
    "Eggplant", "Eggs", "Fish", "Galia", "Garlic", "Grains",
    "Grapes", "Honeydew", "Jar", "Juice", "Kiwi", "Lemon",
    "Lettuce", "Meat", "Meat -Red-", "Milk", "Mushroom", "Mushrooms",
    "Nectarine", "Noodles", "Oats", "Orange", "Oranges", "Peach",
    "Peanut Butter", "Pear", "Pickled Food -Jar-", "Pineapple", "Plum", "Pomegranate",
    "Popcorn", "Potatoes -Package-", "Raspberries", "Rice", "Salad", "Sauce",
    "Seasoning -Thin Package-", "Snack", "Soda", "Spinach", "Squash", "Strawberries",
    "Strawberry", "Tofu", "Tomatoes", "Water", "Watermelon", "Yogurt", "Zucchini"
]

# Dataset específico (fine-tuning) — 20 clases del supermercado
specific_class_labels = [
    "coca_cola_bottle", "coca_cola_can", "orange_fanta_bottle", "heineken_can",
    "whole_milk", "semi_skimmed_milk", "skimmed_milk", "banana", "orange",
    "green_apple", "red_apple", "natural_yogurt", "stracciatella_yogurt",
    "shampoo_hs", "shampoo_hacendado", "ketchup", "mayonnaise",
    "fried_tomato", "york_ham", "turkey_ham"
]

GENERIC_NUM_CLASSES = len(generic_class_labels)
SPECIFIC_NUM_CLASSES = len(specific_class_labels)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# ARQUITECTURA
# ============================================================
IMAGE_SIZE = 416
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]  # [13, 26, 52]

# Las tuplas son bloques convolucionales
# Las listas son bloques residuales CSP
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
# ANCHORS (COCO, escalados a [0,1])
# ============================================================
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],   # escala 13×13 (objetos grandes)
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],  # escala 26×26 (medianos)
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],  # escala 52×52 (pequeños)
]

# ============================================================
# TRANSFORMACIONES
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

# Transformaciones de validación/test (sin augmentation)
test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

# Transformaciones agresivas para pseudo-labels (STAC)
strong_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=int(IMAGE_SIZE * SCALE)),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.15, p=0.8),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.CoarseDropout(max_holes=8, max_height=40, max_width=40,
                        min_holes=2, min_height=10, min_width=10,
                        fill_value=0, p=0.5),
        A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)
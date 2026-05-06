"""
Test Visual — Genera imágenes con las predicciones del modelo
==============================================================

Carga un checkpoint, predice sobre imágenes de test y guarda
las imágenes con las bounding boxes dibujadas en saved_images/.

Cambia WEIGHTS_PATH para elegir qué checkpoint usar:
  - config.CHECKPOINT_FILE   → Preentrenamiento genérico
  - config.FINETUNE_BEST     → Mejor fine-tuning supervisado
  - config.SSL_BEST          → Mejor SSL
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import core.config as config
import torch
import torch.optim as optim
from core.model import YOLOv4
from core.utils import load_checkpoint, get_loaders, plot_couple_examples

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ============================================================
# CONFIGURACIÓN — Cambia aquí el checkpoint que quieras usar
# ============================================================
WEIGHTS_PATH = config.FINETUNE_BEST       # ← Cambia esta línea
NUM_CLASSES  = config.SPECIFIC_NUM_CLASSES # ← Cambia si usas otro dataset


def main():
    print(f"Cargando modelo para visualización...")
    print(f"  Checkpoint: {WEIGHTS_PATH}")
    print(f"  Clases: {NUM_CLASSES}\n")

    # Crear modelo
    model = YOLOv4(num_classes=NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # Cargar los pesos
    load_checkpoint(WEIGHTS_PATH, model, optimizer, config.LEARNING_RATE)

    # Datos de test
    _, test_loader, _ = get_loaders(
        train_csv_path=config.TRAIN_CSV,
        val_csv_path=config.TEST_CSV,
        val_img_dir=config.TEST_IMG_DIR,
        val_label_dir=config.TEST_LABEL_DIR,
    )

    # Crear carpeta de salida
    os.makedirs(os.path.join(config.BASE_DIR, "saved_images"), exist_ok=True)

    # Generar y guardar imágenes
    print("Generando predicciones...")
    plot_couple_examples(model, test_loader, 0.7, config.NMS_IOU_THRESH, config.ANCHORS, config.specific_class_labels)

    print(f"\nBusca los archivos 'prediccion_test_X.png' en saved_images/")


if __name__ == "__main__":
    main()
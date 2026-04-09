"""
SSL Dataset — Combina datos etiquetados y pseudo-etiquetados
=============================================================

Crea un ConcatDataset que mezcla:
  1. Dataset labelled   → usa train_transforms (augmentación estándar)
  2. Dataset pseudo     → usa strong_transforms (augmentación agresiva STAC)

Cada batch contiene samples de ambos orígenes mezclados aleatoriamente
para evitar catastrophic forgetting.
"""

import config
from dataset import YOLODataset
from torch.utils.data import ConcatDataset, DataLoader


def get_ssl_loader():
    """
    Construye un DataLoader combinado labelled + pseudo-labelled.

    Returns:
        ssl_loader: DataLoader mezclado
        val_loader: DataLoader de validación (sin cambios)
    """
    IMAGE_SIZE = config.IMAGE_SIZE
    S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
    num_classes = config.SPECIFIC_NUM_CLASSES

    # --- Dataset 1: Labelled (augmentación estándar) ---
    labelled_dataset = YOLODataset(
        csv_file=config.TRAIN_CSV,
        transform=config.train_transforms,
        S=S,
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
        C=num_classes,
    )

    # --- Dataset 2: Pseudo-labelled (augmentación agresiva STAC) ---
    pseudo_dataset = YOLODataset(
        csv_file=config.PSEUDO_CSV,
        transform=config.strong_transforms,
        S=S,
        img_dir=config.UNLABELLED_IMG_DIR,    # imágenes originales sin etiquetar
        label_dir=config.PSEUDO_LABEL_DIR,     # pseudo-labels generadas
        anchors=config.ANCHORS,
        C=num_classes,
    )

    # --- Concatenar ambos datasets ---
    combined_dataset = ConcatDataset([labelled_dataset, pseudo_dataset])

    ssl_loader = DataLoader(
        dataset=combined_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,           # Mezcla labelled + pseudo en cada epoch
        drop_last=False,
        persistent_workers=True if config.NUM_WORKERS > 0 else False,
    )

    # --- Validación (solo labelled, sin cambios) ---
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
    print(f"    Labelled:       {len(labelled_dataset)} imgs")
    print(f"    Pseudo-labelled: {len(pseudo_dataset)} imgs")
    print(f"    Total combinado: {len(combined_dataset)} imgs")
    print(f"    Validación:      {len(val_dataset)} imgs")

    return ssl_loader, val_loader

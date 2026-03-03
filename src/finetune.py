"""
YOLOv4 Fine-Tuning Pipeline
===========================

Este script carga el modelo preentrenado con el dataset genérico,
congela el backbone y adapta la capa de salida para entrenar con el
dataset específico del supermercado (dataset semilla).

El fine-tuning se divide en 2 fases:
    Fase 1: Solo se entrenan las cabezas nuevas (backbone + SPP + neck congelados)
            El gradiente de las cabezas aleatorias no contamina los pesos preentrenados
    Fase 2: Se descongela el neck y el SPP para que se afinen junto a las cabezas
            El backbone sigue congelado para preservar las features aprendidas
"""

import torch
import torch.nn as nn
import torch.optim as optim
from model import YOLOv4, ScalePrediction, initialize_weights
from loss import YoloLoss
from tqdm import tqdm
import config
from train import train_fn # Reutilizamos la función de entrenamiento de train.py
from utils import get_loaders, save_checkpoint, check_class_accuracy, get_evaluation_bboxes, mean_average_precision

import warnings
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

# Epochs que dura cada fase
FASE1_EPOCHS = 15   # Solo cabezas, con todo lo demás congelado
FASE2_EPOCHS = config.NUM_EPOCHS  # Neck + SPP + cabezas, backbone sigue congelado

# Learning rate más bajo que en preentrenamiento para no destruir los pesos aprendidos
FINETUNE_LR = 1e-5


def freeze_all_except_heads(model):
    """Congela backbone, SPP y neck. Solo las cabezas quedan entrenables."""
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.spp.parameters():
        param.requires_grad = False
    for param in model.neck.parameters():
        param.requires_grad = False


def unfreeze_neck_and_spp(model):
    """Descongela el neck y el SPP para la Fase 2. El backbone sigue congelado."""
    for param in model.spp.parameters():
        param.requires_grad = True
    for param in model.neck.parameters():
        param.requires_grad = True


def main():
    print(f"Using device: {config.DEVICE}")

    # Número de clases que tenía el dataset genérico
    OLD_NUM_CLASSES = 85 
    
    # Instanciar el modelo con la configuración ANTIGUA
    model = YOLOv4(num_classes=OLD_NUM_CLASSES).to(config.DEVICE)
    
    print("Cargando modelo preentrenado...")
    print("=> Loading checkpoint")
    checkpoint = torch.load(config.CHECKPOINT_FILE, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    print("Pesos preentrenados cargados con éxito.")

    ### REEMPLAZAR LAS CABEZAS DE DETECCIÓN
    print(f"Adaptando las cabezas de detección a {config.NUM_CLASSES} clases...")
    model.head_large  = ScalePrediction(256, config.NUM_CLASSES).to(config.DEVICE)
    model.head_medium = ScalePrediction(256, config.NUM_CLASSES).to(config.DEVICE)
    model.head_small  = ScalePrediction(128, config.NUM_CLASSES).to(config.DEVICE)

    initialize_weights(model.head_large)
    initialize_weights(model.head_medium)
    initialize_weights(model.head_small)

    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    # Cargar los loaders del NUEVO dataset
    train_loader, test_loader, _ = get_loaders(
        train_csv_path=config.DATASET + "/nuevo_train.csv", 
        test_csv_path=config.DATASET + "/nuevo_test.csv"
    )

    # Escalado de las anclas
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    # -----------------------------------------------------------------------
    ### FASE 1: Solo cabezas (backbone + SPP + neck congelados)
    # El gradiente de las cabezas nuevas (pesos aleatorios) podría distorsionar
    # los pesos preentrenados del neck y el SPP si los dejásemos libres desde el inicio
    # -----------------------------------------------------------------------
    print(f"\n--- FASE 1: Entrenando solo las cabezas ({FASE1_EPOCHS} epochs) ---")
    freeze_all_except_heads(model)

    # El optimizador solo ve los parámetros entrenables en este momento (las cabezas)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FINETUNE_LR,
        weight_decay=config.WEIGHT_DECAY
    )

    for epoch in range(FASE1_EPOCHS):
        print(f"Epoch (Fase 1): {epoch+1}/{FASE1_EPOCHS}")
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        if config.SAVE_MODEL:
            save_checkpoint(model, optimizer, filename="checkpoints/finetune_checkpoint.pth.tar")

    # -----------------------------------------------------------------------
    ### FASE 2: Neck + SPP + cabezas (backbone sigue congelado)
    # Ahora que las cabezas tienen gradientes estables, descongelamos el neck y el SPP
    # para que se afinen junto a las cabezas con pasos pequeños
    # -----------------------------------------------------------------------
    print(f"\n--- FASE 2: Afinando neck + SPP + cabezas ({FASE2_EPOCHS} epochs) ---")
    unfreeze_neck_and_spp(model)

    # Recreamos el optimizador para que incluya los parámetros recién descongelados
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FINETUNE_LR,
        weight_decay=config.WEIGHT_DECAY
    )

    for epoch in range(FASE2_EPOCHS):
        print(f"Epoch (Fase 2): {epoch+1}/{FASE2_EPOCHS}")
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        if config.SAVE_MODEL:
            save_checkpoint(model, optimizer, filename="checkpoints/finetune_checkpoint.pth.tar")

        # Evaluamos mAP cada 5 epochs
        if epoch > 0 and epoch % 5 == 0:
            check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
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
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP: {map_val.item()}")
            model.train()

if __name__ == "__main__":
    main()
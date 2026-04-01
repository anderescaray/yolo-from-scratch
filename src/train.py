"""
YOLOv4 Training Pipeline
========================

Este script junta el modelo, los datos, la loss y el optimizador para entrenar el modelo

Se usa float16 en lugar de float32 para duplicar la velocidad y reducir el uso de VRAM en la GPU
Se calcula la pérdida en las 3 escalas simultáneamente
Se calcula mAP (Mean Average Precision) periódicamente

"""

import config
import torch
import torch.optim as optim
from model import YOLOv4
from tqdm import tqdm 
from loss import YoloLoss
import warnings
import wandb
import os
import cv2

cv2.setNumThreads(0)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from utils import (
    mean_average_precision,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)

warnings.filterwarnings("ignore")

# Así pyTorch busca el algoritmo de conv más rápido para el hardware disponible
torch.backends.cudnn.benchmark = True


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    """
    Ejecuta una epoch de entrenamiento 
    
    Args:
        train_loader: el Dataloader que nos da los batchs
        model: el modelo YOLOv4
        optimizer: el AdamW que actualiza los pesos
        loss_fn: YoloLoss
        scaler: GradScaler para FP16
        scaled_anchors: las anclas ajustadas al tamaño de la rejilla (13, 26, 52)
    """
    # Barra de progreso para ver cuánto falta
    loop = tqdm(train_loader, leave=True)
    losses = []

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE, non_blocking=True)
        y0, y1, y2 = (
            y[0].to(config.DEVICE, non_blocking=True),
            y[1].to(config.DEVICE, non_blocking=True),
            y[2].to(config.DEVICE, non_blocking=True),
        )
        with torch.cuda.amp.autocast():
            out = model(x)
            
        # FUERA del autocast, para convertir a las predicciones a Float32
        # para evitar que w/h^2 sea w/0.0 y de inf
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
        optimizer.zero_grad() # Se limpian los gradientes anteriores
        
        # Se escala la pérdida (necesario para float16 para evitar underflow)
        scaler.scale(loss).backward() 

        scaler.unscale_(optimizer)
        
        # Gradient clipping: si algún gradiente es > 1.0, lo frena
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer) # Se actualizan los pesos
        scaler.update()

        # Actualización de la barra de progreso
        # Mostramos el promedio de error actual
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)

    # Devolvemos la pérdida media de la epoch
    return mean_loss

def val_fn(val_loader, model, loss_fn, scaled_anchors):
    """
    Calcula la loss sobre el conjunto de validación sin actualizar pesos.
    Se ejecuta en modo eval() para desactivar dropout y BN en modo train.
    Permite comparar train loss vs val loss para detectar overfitting.

    Args:
        val_loader:     Dataloader del conjunto de validación
        model:          El modelo YOLOv4
        loss_fn:        YoloLoss
        scaled_anchors: Las anclas ajustadas al tamaño de la rejilla (13, 26, 52)

    Returns:
        mean_loss: loss media sobre todo el conjunto de validación
    """
    model.eval() # Desactivamos BatchNorm en modo train y dropout
    losses = []

    # torch.no_grad() para no construir el grafo de gradientes (más rápido y menos memoria)
    with torch.no_grad():
        loop = tqdm(val_loader, leave=True, desc="Validación")
        for x, y in loop:
            x = x.to(config.DEVICE, non_blocking=True)
            y0, y1, y2 = (
                y[0].to(config.DEVICE, non_blocking=True),
                y[1].to(config.DEVICE, non_blocking=True),
                y[2].to(config.DEVICE, non_blocking=True),
            )
            with torch.cuda.amp.autocast():
                out = model(x)
                loss = (
                    loss_fn(out[0], y0, scaled_anchors[0]) # objetos grandes
                    + loss_fn(out[1], y1, scaled_anchors[1]) # medianos
                    + loss_fn(out[2], y2, scaled_anchors[2]) # pequeños
                )
            losses.append(loss.item())
            mean_loss = sum(losses) / len(losses)
            loop.set_postfix(val_loss=mean_loss)

    model.train() # Volvemos a modo train para la siguiente epoch
    return mean_loss

def main():
    """Función principal de configuración y bucle de epochs"""
    
    # Inicializar wandb
    # ver los resultados en https://wandb.ai
    wandb.init(
        project="yolov4-supermercado",  # nombre del proyecto en wandb
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
    
    # Inicializar modelo, optimizador y pérdida
    model = YOLOv4(num_classes=config.GENERIC_NUM_CLASSES).to(config.DEVICE)
    #optimizer = optim.Adam(
    #    model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    #)

    # Pendiente de probar:
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
        )
    
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler() # Para fp16

    # Cargar datos del dataset genérico (DATASET_TYPE = "generic" en config.py)
    train_loader, val_loader, train_eval_loader = get_loaders(
        train_csv_path=config.TRAIN_CSV,
        val_csv_path=config.VAL_CSV,
    )

    # Cargar checkpoint (Por si se quiere seguir entrenando uno guardado)
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )

    # Escalado de las anclas
    # Las anclas en config vienen normalizadas (0-1)
    # Y la Loss Function necesita que estén en unidades de la grid 
    # --> Ancla * Grid = Unidades de la grid
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    # Bucle de entrenamiento
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch: {epoch+1}/{config.NUM_EPOCHS}")
        
        # Entrenar una vuelta completa
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
        val_loss = val_fn(val_loader, model, loss_fn, scaled_anchors)

        # Logear ambas losses juntas en wandb para poder compararlas en la misma gráfica
        wandb.log({
            "train/loss": train_loss,
            "val/loss":   val_loss,
            "epoch":      epoch + 1,
        })

        # Guardar el modelo
        if config.SAVE_MODEL:
            save_checkpoint(model, optimizer, filename=f"checkpoints/checkpoint.pth.tar")

        # Evaluar precisión (mAP) cada 5 epochs (es lento por eso mejor no hacerlo siempre)
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
    print(f"Using device: {config.DEVICE}")
    main()
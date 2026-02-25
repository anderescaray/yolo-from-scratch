"""
YOLOv3 Training Pipeline
========================

Este script junta el modelo, los datos, la loss y el optimizador para entrenar el modelo

Se usa float16 en lugar de float32 para duplicar la velocidad y reducir el uso de VRAM en la GPU
Se calcula la pérdida en las 3 escalas simultáneamente
Se calcula mAP (Mean Average Precision) periódicamente

"""

import config
import torch
import torch.optim as optim
from model import YOLOv3
from tqdm import tqdm 
from loss import YoloLoss
import warnings


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
        model: el modelo YOLOv3
        optimizer: el Adam que actualiza los pesos
        loss_fn: YoloLoss
        scaler: Herramienta para Mixed Precision (fp16)
        scaled_anchors: las anclas ajustadas al tamaño de la rejilla (13, 26, 52)
    """
    # Barra de progreso para ver cuánto falta
    loop = tqdm(train_loader, leave=True)
    losses = []

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        # Forward pass con Mixed Precision
        # 'autocast' permite a la GPU usar float16 donde sea seguro (más rápido)
        with torch.cuda.amp.autocast():
            # out[0]=13x13, out[1]=26x26, out[2]=52x52
            out = model(x)
            
            # Cálculo de loss
            # Sumando el error de las 3 escalas
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0]) # error de objetos grandes
                + loss_fn(out[1], y1, scaled_anchors[1]) # medianos
                + loss_fn(out[2], y2, scaled_anchors[2]) # pequeños
            )

        # Backpropagation
        losses.append(loss.item())
        optimizer.zero_grad() # Se limpian los gradientes anteriores
        
        # Se escala la pérdida (necesario para float16 para evitar underflow)
        scaler.scale(loss).backward() 
        scaler.step(optimizer) # Se actualizan los pesos
        scaler.update() # Se actualiza el factor de escala

        # Actualización de la barra de progreso
        # Mostramos el promedio de error actual
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)


def main():
    """Función principal de configuración y bucle de epochs"""
    
    # Inicializar modelo, optimizador y pérdida
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Pendiente de probar:
    #optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler() # Para fp16

    # Cargar datos
    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.DATASET + "/train.csv", 
        test_csv_path=config.DATASET + "/test.csv"
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
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        # Guardar el modelo
        if config.SAVE_MODEL:
            save_checkpoint(model, optimizer, filename=f"checkpoints/checkpoint.pth.tar")

        # Evaluar precisión (mAP) cada 3 épocas (es lento por eso mejor no hacerlo siempre)
        if epoch > 0 and epoch % 3 == 0:
            check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
            '''pred_boxes, true_boxes = get_evaluation_bboxes(
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
            print(f"MAP: {map_val.item()}")'''
            
            model.train()

if __name__ == "__main__":
    print(f"Using device: {config.DEVICE}")
    main()
"""
YOLOv4 Fine-Tuning Pipeline
===========================

Este script carga el modelo preentrenado con el dataset genérico,
congela el backbone y adapta la capa de salida para entrenar con el
dataset específico del supermercado (dataset semilla).
"""

import torch
import torch.optim as optim
from model import YOLOv4, ScalePrediction, initialize_weights # ¡Asegúrate de importar ScalePrediction e initialize_weights!
from loss import YoloLoss
from tqdm import tqdm
import config
from train import train_fn # Reutilizamos la función de entrenamiento de train.py
from utils import get_loaders, load_checkpoint, save_checkpoint, check_class_accuracy, get_evaluation_bboxes, mean_average_precision

import warnings
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

def main():
    print(f"Using device: {config.DEVICE}")

    # 1. Definir los parámetros del modelo preentrenado
    # SUSTITUYE 80 por el número real de clases que tenía tu dataset genérico
    OLD_NUM_CLASSES = 80 
    
    # 2. Instanciar el modelo con la configuración ANTIGUA
    model = YOLOv4(num_classes=OLD_NUM_CLASSES).to(config.DEVICE)
    
    # 3. Cargar los pesos del preentrenamiento
    # Suponemos que tienes el archivo .pth.tar en config.CHECKPOINT_FILE
    print("Cargando modelo preentrenado...")
    optimizer_dummy = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE) # Optimizador temporal solo para que la función load_checkpoint no falle
    load_checkpoint(config.CHECKPOINT_FILE, model, optimizer_dummy, config.LEARNING_RATE)
    print("Pesos preentrenados cargados con éxito.")

    # 4. CONGELAR EL BACKBONE
    # Esto evita que los pesos de las capas convolucionales iniciales se modifiquen
    print("Congelando el backbone...")
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Opcional: También podrías congelar el cuello (SPP y PANet) si tienes muy pocas imágenes
    # for param in model.spp.parameters():
    #     param.requires_grad = False
    # for param in model.neck.parameters():
    #     param.requires_grad = False

    # 5. REEMPLAZAR LAS CABEZAS DE DETECCIÓN
    # Creamos nuevas capas ScalePrediction con el NUEVO número de clases (config.NUM_CLASSES)
    print(f"Adaptando las cabezas de detección a {config.NUM_CLASSES} clases...")
    model.head_large = ScalePrediction(256, config.NUM_CLASSES).to(config.DEVICE)
    model.head_medium = ScalePrediction(256, config.NUM_CLASSES).to(config.DEVICE)
    model.head_small = ScalePrediction(128, config.NUM_CLASSES).to(config.DEVICE)

    # Inicializamos SOLO estas nuevas capas (para no arrastrar pesos aleatorios malos)
    model.head_large.apply(initialize_weights)
    model.head_medium.apply(initialize_weights)
    model.head_small.apply(initialize_weights)

    # 6. CONFIGURAR EL OPTIMIZADOR SOLO PARA LAS CAPAS NO CONGELADAS
    # filter(lambda p: p.requires_grad, ...) asegura que solo le pasamos los parámetros que queremos entrenar
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config.LEARNING_RATE, # Quizás quieras usar un LR un poco más bajo para fine-tuning, ej: 1e-5
        weight_decay=config.WEIGHT_DECAY
    )

    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    # 7. Cargar los loaders de tu NUEVO dataset (el dataset semilla del supermercado)
    # ¡Asegúrate de que las rutas en config.py apunten a los nuevos CSVs!
    train_loader, test_loader, _ = get_loaders(
        train_csv_path=config.DATASET + "/tu_nuevo_train.csv", 
        test_csv_path=config.DATASET + "/tu_nuevo_test.csv"
    )

    # Escalado de las anclas
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    # 8. Bucle de Entrenamiento (Fine-Tuning)
    # Puede que necesites menos epochs que en el preentrenamiento (ej: 50-100)
    print("Iniciando Fine-Tuning...")
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch (Fine-Tuning): {epoch+1}/{config.NUM_EPOCHS}")
        
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        if config.SAVE_MODEL:
            # Guarda con un nombre diferente para no sobrescribir tu checkpoint genérico
            save_checkpoint(model, optimizer, filename=f"checkpoints/finetune_checkpoint.pth.tar")

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
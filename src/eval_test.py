"""
Script de Evaluación del Test Set
=========================================

"""

import argparse
import torch
import config
from model import YOLOv4
from utils import (
    get_loaders,
    check_class_accuracy,
    get_evaluation_bboxes,
    mean_average_precision,
)

def evaluate_model(weights_path):
    print(f"\n{'='*60}")
    print(f"  EVALUACIÓN EN CONJUNTO DE TEST")
    print(f"  Modelo: {weights_path}")
    print(f"{'='*60}\n")

    model = YOLOv4(num_classes=config.SPECIFIC_NUM_CLASSES).to(config.DEVICE)

    # Cargar los pesos del checkpoint
    print("Cargando pesos...")
    checkpoint = torch.load(weights_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(" ✅ Pesos cargados correctamente.\n")

    # Cargar el dataLoader de test
    _, test_loader, _ = get_loaders(
        train_csv_path=config.TRAIN_CSV, # Relleno
        val_csv_path=config.TEST_CSV,    
        train_img_dir=config.IMG_DIR,
        train_label_dir=config.LABEL_DIR,
        val_img_dir=config.TEST_IMG_DIR,
        val_label_dir=config.TEST_LABEL_DIR,
    )

    # Evaluación
    print("Calculando métricas en el set de Test...")
    with torch.no_grad():
        class_acc, noobj_acc, obj_acc = check_class_accuracy(
            model, test_loader, threshold=config.CONF_THRESHOLD
        )
        pred_boxes, true_boxes = get_evaluation_bboxes(
            test_loader,
            model,
            iou_threshold=config.NMS_IOU_THRESH,
            anchors=config.ANCHORS,
            threshold=config.CONF_THRESHOLD,
            device=config.DEVICE,
        )
        map_test = mean_average_precision(
            pred_boxes,
            true_boxes,
            iou_threshold=config.MAP_IOU_THRESH,
            box_format="midpoint",
            num_classes=config.SPECIFIC_NUM_CLASSES,
        )

    # Reporte Final
    print(f"\n  RESULTADOS FINALES")
    print(f"  mAP@{config.MAP_IOU_THRESH}: {map_test.item():.4f}")
    print(f"  Precisión de Clase: {class_acc:.2f}%")
    print(f"  Acierto con Objeto: {obj_acc:.2f}%")
    print(f"  Acierto sin Objeto (Fondo): {noobj_acc:.2f}%")
    print(f"{'='*60}\n")

if __name__ == "__main__":

# example: python src/eval_test.py --weights checkpoints/finetune_best.pth.tar

    parser = argparse.ArgumentParser(description="Evaluar modelo YOLO en conjunto de Test")
    parser.add_argument(
        "--weights", 
        type=str, 
        required=True, 
        help="Ruta al archivo .pth.tar que quieres evaluar"
    )
    args = parser.parse_args()
    
    evaluate_model(args.weights)
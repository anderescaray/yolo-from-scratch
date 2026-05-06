"""
Pseudo-Label Generator (TTA + WBF)
====================================

Genera pseudo-labels offline para imágenes sin etiquetar usando:
  1. Test-Time Augmentation (TTA): original + horizontal flip
  2. Weighted Boxes Fusion (WBF): fusiona las cajas coincidentes
  3. Filtrado por umbral tau para quedarse solo con las de alta confianza

Output:
    - Archivos .txt en formato YOLO en data/yolo_dataset/train/pseudo_labelled/
    - CSV pseudo_train.csv con pares (imagen, label)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import csv
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import core.config as config
from core.model import YOLOv4
from core.utils import cells_to_bboxes, non_max_suppression


# ============================================================
# WBF (Weighted Boxes Fusion) — implementación ligera
# ============================================================

def _iou_single(box_a, box_b):
    """IoU entre dos cajas en formato [x_center, y_center, w, h] normalizado."""
    ax1 = box_a[0] - box_a[2] / 2
    ay1 = box_a[1] - box_a[3] / 2
    ax2 = box_a[0] + box_a[2] / 2
    ay2 = box_a[1] + box_a[3] / 2

    bx1 = box_b[0] - box_b[2] / 2
    by1 = box_b[1] - box_b[3] / 2
    bx2 = box_b[0] + box_b[2] / 2
    by2 = box_b[1] + box_b[3] / 2

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area + 1e-8

    return inter_area / union


def weighted_boxes_fusion(boxes_list, iou_threshold=0.55):
    """
    WBF simplificado para fusionar cajas de N pasadas de TTA.

    Args:
        boxes_list: lista de listas, cada sub-lista contiene cajas
                    en formato [class, score, x, y, w, h]
        iou_threshold: umbral IoU para considerar que dos cajas son la misma

    Returns:
        fused_boxes: lista de cajas fusionadas [class, score, x, y, w, h]
    """
    # Juntamos todas las cajas en una sola lista
    all_boxes = []
    for boxes in boxes_list:
        all_boxes.extend(boxes)

    if len(all_boxes) == 0:
        return []

    # Ordenar por confianza descendente
    all_boxes = sorted(all_boxes, key=lambda b: b[1], reverse=True)

    # Clusters de cajas fusionadas
    clusters = []       # cada cluster es una lista de cajas que se solapan
    fused_boxes = []    # la caja promedio resultante de cada cluster

    for box in all_boxes:
        matched = False
        for i, fused in enumerate(fused_boxes):
            # Solo fusionar si son de la misma clase
            if int(box[0]) == int(fused[0]):
                iou_val = _iou_single(box[2:], fused[2:])
                if iou_val > iou_threshold:
                    # Añadir al cluster existente
                    clusters[i].append(box)
                    # Recalcular la caja fusionada como media ponderada por confianza
                    cluster = clusters[i]
                    total_weight = sum(b[1] for b in cluster)
                    avg_score = total_weight / len(cluster) # confianza media (si alucina en todas, pueden pasar pseudo-labels malas)
                    avg_x = sum(b[1] * b[2] for b in cluster) / total_weight
                    avg_y = sum(b[1] * b[3] for b in cluster) / total_weight
                    avg_w = sum(b[1] * b[4] for b in cluster) / total_weight
                    avg_h = sum(b[1] * b[5] for b in cluster) / total_weight
                    fused_boxes[i] = [int(box[0]), avg_score, avg_x, avg_y, avg_w, avg_h]
                    matched = True
                    break

        if not matched:
            clusters.append([box])
            fused_boxes.append(list(box))

    return fused_boxes


# ============================================================
# INFERENCIA CON TTA
# ============================================================

def decode_predictions(model, image_tensor, anchors, device):
    """
    Pasa una imagen por el modelo y decodifica las predicciones de las 3 escalas
    a formato [class, score, x, y, w, h] normalizado (0-1).
    """
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            predictions = model(image_tensor)

    all_boxes = []
    for i in range(3):
        S = predictions[i].shape[2]
        anchor = torch.tensor([*anchors[i]]).to(device) * S
        boxes = cells_to_bboxes(predictions[i], anchor, S=S, is_preds=True)
        all_boxes.extend(boxes[0])  # Solo 1 imagen en el batch

    return all_boxes


def flip_boxes_horizontal(boxes):
    """Invierte la coordenada X de las cajas para deshacer un flip horizontal."""
    flipped = []
    for box in boxes:
        cls, score, x, y, w, h = box
        flipped.append([cls, score, 1.0 - x, y, w, h])
    return flipped    

def get_eval_transform(new_size):
    """Crea una transformación dinámica al tamaño requerido para las distintas escalas."""
    return A.Compose([
        A.Resize(height=new_size, width=new_size),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2(),
    ])

def predict_with_tta(model, image_np, anchors, device, conf_threshold=0.3):
    """
    Genera predicciones con Multi-Scale Test-Time Augmentation (4 pasadas).
    Evalúa la imagen en diferentes condiciones para maximizar la robustez:
      1. Original (1.0x)
      2. Horizontal Flip (Espejo)
      3. Scale Up (~1.25x)
      4. Scale Down (~0.75x)

    Args:
        model: modelo YOLOv4 en eval()
        image_np: imagen numpy HxWxC (0-255)
        anchors: config.ANCHORS
        device: cuda/cpu
        conf_threshold: umbral mínimo para pre-filtrar cajas basura

    Returns:
        Lista con 4 sub-listas de predicciones: 
        [boxes_original, boxes_flipped_corrected, boxes_scale_up, boxes_scale_down]
    """
    base_size = config.IMAGE_SIZE
    # YOLO requiere que las dimensiones de entrada sean múltiplos de 32
    size_up = int(base_size * 1.25) // 32 * 32
    size_down = int(base_size * 0.75) // 32 * 32

    # --- Pasada 1: Original ---
    t_orig = get_eval_transform(base_size)
    img_orig = t_orig(image=image_np)["image"]
    boxes_orig = decode_predictions(model, img_orig, anchors, device)
    boxes_orig = [b for b in boxes_orig if b[1] > conf_threshold]

    # --- Pasada 2: Horizontal Flip ---
    image_flipped = np.fliplr(image_np).copy()
    img_flip = t_orig(image=image_flipped)["image"]
    boxes_flip = decode_predictions(model, img_flip, anchors, device)
    boxes_flip = [b for b in boxes_flip if b[1] > conf_threshold]
    boxes_flip = flip_boxes_horizontal(boxes_flip) # Deshacer el flip

    # --- Pasada 3: Scale Up ---
    t_up = get_eval_transform(size_up)
    img_up = t_up(image=image_np)["image"]
    boxes_up = decode_predictions(model, img_up, anchors, device)
    boxes_up = [b for b in boxes_up if b[1] > conf_threshold]

    # --- Pasada 4: Scale Down ---
    t_down = get_eval_transform(size_down)
    img_down = t_down(image=image_np)["image"]
    boxes_down = decode_predictions(model, img_down, anchors, device)
    boxes_down = [b for b in boxes_down if b[1] > conf_threshold]

    return [boxes_orig, boxes_flip, boxes_up, boxes_down]


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generar pseudo-labels con TTA + WBF")
    parser.add_argument("--weights", type=str, required=True,
                        help="Ruta al checkpoint .pth.tar (ej: checkpoints/finetune_best.pth.tar)")
    parser.add_argument("--tau", type=float, default=config.SSL_TAU,
                        help=f"Umbral de confianza para pseudo-labels (default: {config.SSL_TAU})")
    parser.add_argument("--iou-wbf", type=float, default=0.55,
                        help="Umbral IoU para WBF (default: 0.55)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  PSEUDO-LABEL GENERATOR (TTA + WBF)")
    # Resolver el path del checkpoint relativo a la raíz del proyecto
    weights_path = args.weights
    if not os.path.isabs(weights_path):
        weights_path = os.path.join(config.BASE_DIR, weights_path)

    print(f"  Modelo: {weights_path}")
    print(f"  Tau: {args.tau}  |  IoU WBF: {args.iou_wbf}")
    print(f"{'='*60}\n")

    # ------- Cargar modelo -------
    model = YOLOv4(num_classes=config.SPECIFIC_NUM_CLASSES).to(config.DEVICE)
    checkpoint = torch.load(weights_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print("  ✅ Modelo cargado.\n")

    # ------- Encontrar imágenes sin etiquetar -------
    unlabelled_dir = config.UNLABELLED_IMG_DIR
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted([
        f for f in os.listdir(unlabelled_dir)
        if os.path.splitext(f)[1].lower() in valid_ext
    ])
    print(f"  Imágenes sin etiquetar encontradas: {len(image_files)}")

    if len(image_files) == 0:
        print("  ⚠️  No hay imágenes en UNLABELLED_IMG_DIR. Abortando.")
        return

    # ------- Crear directorio de salida -------
    output_dir = config.PSEUDO_LABEL_DIR
    os.makedirs(output_dir, exist_ok=True)

    # ------- Procesar cada imagen -------
    csv_rows = []
    total_labels = 0
    skipped = 0

    for img_name in tqdm(image_files, desc="Generando pseudo-labels"):
        img_path = os.path.join(unlabelled_dir, img_name)
        image_np = np.array(Image.open(img_path).convert("RGB"))

        # TTA: obtener cajas de las 4 pasadas (Multi-Scale + H-Flip)
        tta_boxes_list = predict_with_tta(
            model, image_np, config.ANCHORS, config.DEVICE
        )

        # WBF: fusionar la lista de listas
        fused = weighted_boxes_fusion(
            tta_boxes_list,
            iou_threshold=args.iou_wbf,
        )

        # Filtrar por tau
        confident_boxes = [b for b in fused if b[1] >= args.tau]

        if len(confident_boxes) == 0:
            skipped += 1
            continue

        # Guardar como .txt YOLO
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(output_dir, label_name)

        with open(label_path, "w") as f:
            for box in confident_boxes:
                cls_id = int(box[0])
                x, y, w, h = box[2], box[3], box[4], box[5]

                # Convertir a esquinas, clampear a [0,1] y reconvertir a centro
                # Esto es lo único que garantiza que y_center + h/2 <= 1.0
                x1 = max(0.0, x - w / 2)
                y1 = max(0.0, y - h / 2)
                x2 = min(1.0, x + w / 2)
                y2 = min(1.0, y + h / 2)

                # Descartar cajas degeneradas tras el clamp
                if x2 <= x1 or y2 <= y1:
                    continue

                xc = (x1 + x2) / 2
                yc = (y1 + y2) / 2
                wc = x2 - x1
                hc = y2 - y1

                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {wc:.6f} {hc:.6f}\n")

        total_labels += len(confident_boxes)
        csv_rows.append([img_name, label_name])

    # ------- Generar CSV -------
    csv_path = config.PSEUDO_CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in csv_rows:
            writer.writerow(row)

    # ------- Resumen -------
    print(f"\n{'='*60}")
    print(f"  RESUMEN")
    print(f"  Imágenes procesadas:  {len(image_files)}")
    print(f"  Imágenes con labels:  {len(csv_rows)}")
    print(f"  Imágenes descartadas: {skipped}")
    print(f"  Total pseudo-labels:  {total_labels}")
    print(f"  Labels guardadas en:  {output_dir}")
    print(f"  CSV generado en:      {csv_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

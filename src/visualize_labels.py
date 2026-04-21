"""
Visualizador de Labels del Dataset Etiquetado
==============================================

Dibuja las bounding boxes ground truth sobre las imágenes etiquetadas
y las guarda en 'label_viz/' para comprobar que las labels son correctas.

Uso:
    python src/visualize_labels.py                 # todas las imágenes de train/labelled
    python src/visualize_labels.py --split val     # imágenes de val
    python src/visualize_labels.py --split test    # imágenes de test
    python src/visualize_labels.py --max 50        # limitar a 50 imágenes
"""

import argparse
import os
import cv2
import numpy as np
from PIL import Image

import config

# Colores BGR distintos por clase
COLORS = [
    (255,  56,  56), (255, 157, 151), (255, 112,  31), (255, 178,  29),
    (207, 210,  49), ( 72, 249,  10), (146, 204,  23), ( 61, 219, 134),
    ( 26, 147,  52), (  0, 212, 187), ( 44, 153, 168), (  0, 194, 255),
    ( 52,  69, 147), (100, 115, 255), (  0,  24, 236), (132,  56, 255),
    ( 82,   0, 133), (203,  56, 255), (255, 149, 200), (255,  55, 199),
]


def draw_boxes(image_np, boxes, class_names):
    """
    Dibuja bboxes en formato YOLO [cls, x_center, y_center, w, h] sobre la imagen.
    """
    h, w = image_np.shape[:2]
    img = image_np.copy()

    for box in boxes:
        cls_id, xc, yc, bw, bh = box
        cls_id = int(cls_id)

        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        color = COLORS[cls_id % len(COLORS)]
        name = class_names[cls_id] if cls_id < len(class_names) else f"cls{cls_id}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Fondo del texto
        (lw, lh), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - lh - 6), (x1 + lw, y1), color, -1)
        cv2.putText(img, name, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def load_label(txt_path):
    """Carga un .txt YOLO. Devuelve lista de (cls, x, y, w, h)."""
    boxes = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls = int(float(parts[0]))
                x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                boxes.append((cls, x, y, w, h))
    return boxes


def main():
    parser = argparse.ArgumentParser(description="Visualizar ground truth labels del dataset etiquetado")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test"],
                        help="Split a visualizar: train | val | test (default: train)")
    parser.add_argument("--max", type=int, default=100,
                        help="Número máximo de imágenes a visualizar (default: 100)")
    args = parser.parse_args()

    # Seleccionar directorios según el split
    if args.split == "train":
        img_dir   = config.IMG_DIR
        label_dir = config.LABEL_DIR
    elif args.split == "val":
        img_dir   = config.VAL_IMG_DIR
        label_dir = config.VAL_LABEL_DIR
    else:  # test
        img_dir   = config.TEST_IMG_DIR
        label_dir = config.TEST_LABEL_DIR

    output_dir = os.path.join(config.BASE_DIR, f"label_viz_{args.split}")
    os.makedirs(output_dir, exist_ok=True)

    class_names = config.specific_class_labels
    valid_ext   = {".jpg", ".jpeg", ".png", ".bmp"}

    img_files = sorted([
        f for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in valid_ext
    ])[:args.max]

    print(f"\n{'='*60}")
    print(f"  VISUALIZADOR DE LABELS  |  split: {args.split}")
    print(f"  Imágenes a procesar: {len(img_files)}")
    print(f"  Salida: {output_dir}")
    print(f"{'='*60}\n")

    sin_label  = 0
    sin_objeto = 0
    con_objeto = 0

    for img_name in img_files:
        img_path   = os.path.join(img_dir, img_name)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)

        image_np = np.array(Image.open(img_path).convert("RGB"))

        if not os.path.exists(label_path):
            sin_label += 1
            # Marcar la imagen como sin label
            vis = image_np.copy()
            cv2.putText(vis, "SIN LABEL", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            boxes = load_label(label_path)
            vis   = draw_boxes(image_np, boxes, class_names)

            status = f"{len(boxes)} objeto(s)"
            color  = (0, 200, 0) if boxes else (0, 100, 255)
            cv2.putText(vis, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if boxes:
                con_objeto += 1
            else:
                sin_objeto += 1

        cv2.putText(vis, img_name, (10, image_np.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        out_path = os.path.join(output_dir, f"viz_{img_name}")
        cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print(f"  Con objetos:  {con_objeto}")
    print(f"  Label vacía:  {sin_objeto}")
    print(f"  Sin label:    {sin_label}")
    print(f"\n  ✅ Guardado en: {output_dir}\n")


if __name__ == "__main__":
    main()

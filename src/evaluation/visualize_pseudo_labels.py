"""
Pseudo-Labels Visualizer
=========================

Draws bounding boxes of generated pseudo-labels on original images
and saves them in 'pseudo_label_viz/' for visual inspection.

Usage:
    python src/visualize_pseudo_labels.py
    python src/visualize_pseudo_labels.py --max 30   # limit to 30 images
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import cv2
import numpy as np
from PIL import Image

import core.config as config

OUTPUT_DIR = os.path.join(config.BASE_DIR, "pseudo_label_viz")

# Different BGR colors per class
COLORS = [
    (255, 56,  56),  (255, 157,  151), (255, 112,  31),  (255, 178, 29),
    (207, 210,  49),  (72, 249, 10),   (146, 204, 23),   (61, 219, 134),
    (26, 147, 52),   (0, 212, 187),    (44, 153, 168),   (0, 194, 255),
    (52,  69, 147),  (100,  115, 255), (0,  24, 236),    (132,  56, 255),
    (82,   0, 133),  (203,  56, 255),  (255, 149, 200),  (255, 55, 199),
]


def draw_boxes(image_np, boxes, class_names):
    """
    Draws YOLO bboxes [class, score, x_center, y_center, w, h] on image.
    boxes: list of tuples (cls_id, score, xc, yc, w, h)
    """
    h, w = image_np.shape[:2]
    img = image_np.copy()

    for box in boxes:
        cls_id, score, xc, yc, bw, bh = box

        # Convert from normalized to pixels
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        color = COLORS[int(cls_id) % len(COLORS)]
        label = f"{class_names[int(cls_id)]} {score:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Fondo del texto
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - lh - 6), (x1 + lw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def load_pseudo_label(txt_path):
    """Carga un .txt de pseudo-labels. Devuelve lista de (cls, score, x, y, w, h)."""
    boxes = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                # Formato YOLO puro sin score — ponemos score=1.0 para labels manuales
                cls, x, y, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                boxes.append((cls, 1.0, x, y, w, h))
            elif len(parts) == 6:
                cls, score, x, y, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                boxes.append((cls, score, x, y, w, h))
    return boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=50,
                        help="Número máximo de imágenes a visualizar (default: 50)")
    args = parser.parse_args()

    img_dir     = config.UNLABELLED_IMG_DIR
    label_dir   = config.PSEUDO_LABEL_DIR
    class_names = config.specific_class_labels

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Iterate over the .txt files in pseudo_labelled/ — only images that actually
    # got a pseudo-label are shown (no point showing the ones that were skipped).
    valid_ext   = {".jpg", ".jpeg", ".png", ".bmp"}
    label_files = sorted([
        f for f in os.listdir(label_dir)
        if f.endswith(".txt")
    ])

    total = len(label_files)
    to_viz = min(args.max, total)

    print(f"\n{'='*60}")
    print(f"  PSEUDO-LABELS VISUALIZER")
    print(f"  Labels found in pseudo_labelled/: {total}")
    print(f"  Visualizing: {to_viz}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    found    = 0
    missing  = 0

    for label_name in label_files[:args.max]:
        base      = os.path.splitext(label_name)[0]
        label_path = os.path.join(label_dir, label_name)

        # Find the matching image (try all valid extensions)
        img_path = None
        for ext in valid_ext:
            candidate = os.path.join(img_dir, base + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            print(f"  ⚠️  Image not found for label: {label_name}")
            missing += 1
            continue

        image_np = np.array(Image.open(img_path).convert("RGB"))
        boxes    = load_pseudo_label(label_path)
        found   += 1

        vis = draw_boxes(image_np, boxes, class_names)

        img_name = os.path.basename(img_path)
        status   = f"{len(boxes)} detections"
        cv2.putText(vis, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis, img_name, (10, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        out_path = os.path.join(OUTPUT_DIR, f"viz_{img_name}")
        cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print(f"  Visualized:           {found}")
    print(f"  Image not found:      {missing}")
    print(f"\n  ✅ Images saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()

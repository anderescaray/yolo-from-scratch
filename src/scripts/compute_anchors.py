"""
compute_anchors.py
==================
K-means clustering (IoU-based distance) over the labeled training set
to compute the 9 optimal YOLO anchors for this specific dataset.

Run:
    python src/scripts/compute_anchors.py

Output:
    - Prints the 9 anchors ready to paste into config.py
    - Reports average IoU before/after to quantify the improvement
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch
from pathlib import Path

import core.config as config
from core.utils import iou_width_height


# ============================================================
# IoU-based K-means (standard YOLO anchor clustering)
# ============================================================

def _iou_distance(box: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """
    Distance = 1 - IoU between one box and each cluster centroid.
    Boxes are (w, h) only — centers are assumed aligned (YOLO-style).
    """
    min_w = np.minimum(box[0], clusters[:, 0])
    min_h = np.minimum(box[1], clusters[:, 1])
    intersection = min_w * min_h
    union = box[0] * box[1] + clusters[:, 0] * clusters[:, 1] - intersection
    return 1.0 - intersection / (union + 1e-9)


def kmeans_iou(boxes: np.ndarray, k: int, n_iter: int = 500, seed: int = 42) -> np.ndarray:
    """
    K-means with 1-IoU as distance metric.

    Args:
        boxes:  (N, 2) array of normalized (w, h) values
        k:      number of clusters (9 for YOLO: 3 scales × 3 anchors)
        n_iter: max iterations
        seed:   random seed for reproducibility

    Returns:
        clusters: (k, 2) array of anchor (w, h) values
    """
    np.random.seed(seed)
    idx = np.random.choice(len(boxes), k, replace=False)
    clusters = boxes[idx].copy()
    assignments = np.zeros(len(boxes), dtype=int)

    for iteration in range(n_iter):
        distances = np.stack([_iou_distance(box, clusters) for box in boxes])
        new_assignments = distances.argmin(axis=1)

        if np.all(new_assignments == assignments):
            print(f"  K-means converged at iteration {iteration + 1}")
            break
        assignments = new_assignments

        for j in range(k):
            members = boxes[assignments == j]
            if len(members) > 0:
                clusters[j] = members.mean(axis=0)

    return clusters, assignments


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"\n{'='*60}")
    print("  ANCHOR COMPUTATION — K-means (IoU distance)")
    print(f"  Dataset: {config.TRAIN_CSV}")
    print(f"{'='*60}\n")

    # ------ Collect all (w, h) from labeled training set ------
    label_dir = Path(config.LABEL_DIR)
    train_csv = config.TRAIN_CSV

    if not Path(train_csv).exists():
        print(f"ERROR: {train_csv} not found. Run generate_csv.py first.")
        return

    df = pd.read_csv(train_csv)
    label_files = df.iloc[:, 1].tolist()

    boxes_wh = []
    missing = 0
    for label_file in label_files:
        label_path = label_dir / label_file
        if not label_path.exists():
            missing += 1
            continue
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    w, h = float(parts[3]), float(parts[4])
                    if w > 1e-4 and h > 1e-4:
                        boxes_wh.append([w, h])

    if len(boxes_wh) == 0:
        print("ERROR: No bounding boxes found. Check LABEL_DIR and TRAIN_CSV.")
        return

    boxes = np.array(boxes_wh, dtype=np.float32)
    print(f"  Bounding boxes collected: {len(boxes)}")
    print(f"  Label files missing:      {missing}")
    print(f"  Width  — mean: {boxes[:, 0].mean():.4f}  std: {boxes[:, 0].std():.4f}")
    print(f"  Height — mean: {boxes[:, 1].mean():.4f}  std: {boxes[:, 1].std():.4f}\n")

    # ------ Baseline: average IoU with current COCO anchors ------
    coco_anchors = np.array(
        config.ANCHORS[0] + config.ANCHORS[1] + config.ANCHORS[2], dtype=np.float32
    )
    baseline_ious = []
    for box in boxes:
        b = torch.tensor(box)
        a = torch.tensor(coco_anchors)
        baseline_ious.append(iou_width_height(b, a).max().item())
    baseline_mean_iou = float(np.mean(baseline_ious))
    print(f"  Baseline avg IoU (COCO anchors):   {baseline_mean_iou:.4f}")

    # ------ K-means clustering ------
    print(f"\n  Running K-means (k=9)...")
    clusters, assignments = kmeans_iou(boxes, k=9)

    # ------ Sort by area and split into 3 scales ------
    areas = clusters[:, 0] * clusters[:, 1]
    clusters = clusters[areas.argsort()]   # ascending: smallest first

    small  = clusters[0:3]   # → 52×52 head (small objects)
    medium = clusters[3:6]   # → 26×26 head (medium objects)
    large  = clusters[6:9]   # → 13×13 head (large objects)

    # ------ Average IoU with new anchors ------
    new_anchors_np = np.vstack([large, medium, small])
    new_ious = []
    for box in boxes:
        b = torch.tensor(box)
        a = torch.tensor(new_anchors_np.astype(np.float32))
        new_ious.append(iou_width_height(b, a).max().item())
    new_mean_iou = float(np.mean(new_ious))
    print(f"  New     avg IoU (k-means anchors): {new_mean_iou:.4f}")
    print(f"  Improvement: +{(new_mean_iou - baseline_mean_iou):.4f}\n")

    # ------ Print results ------
    def fmt(arr):
        return [(round(float(w), 4), round(float(h), 4)) for w, h in arr]

    print(f"{'='*60}")
    print("  RESULT — paste into src/core/config.py")
    print(f"{'='*60}")
    print("ANCHORS = [")
    print(f"    {fmt(large)},   # scale 13×13 (large objects)")
    print(f"    {fmt(medium)},  # scale 26×26 (medium objects)")
    print(f"    {fmt(small)},   # scale 52×52 (small objects)")
    print("]")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

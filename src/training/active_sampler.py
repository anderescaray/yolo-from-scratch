"""
Active Sampler — Uncertainty-Based Query Strategy
==================================================

Scores all unlabeled images by model uncertainty and outputs a ranked CSV
of the most informative candidates for manual annotation.

Uncertainty is estimated from two complementary signals over N TTA passes:

  1. Classification entropy  H(p) = -Σ p_i log p_i             [Settles 2009]
     Measures how spread the softmax distribution is over classes.
     High entropy → model does not know which class the object is.

  2. TTA score variance  σ(conf_1, …, conf_N)               [Lakshminarayanan 2017]
     Measures how much the model's detection confidence fluctuates across
     different augmentations of the same image (analogous to ensemble
     disagreement).  High variance → aleatoric / epistemic uncertainty.

  3. TTA class disagreement  (1 - agreement_ratio)
     Fraction of TTA passes that disagree on the predicted class label for
     the same detected object.  Complements entropy when the model is
     confident but wrong on half the passes.

  4. Localization instability  (1 - mean_pairwise_IoU)
     Complement of the mean pairwise IoU across all TTA-pass boxes that
     were fused into the same cluster.  High value → the model is uncertain
     about WHERE the object is, not just what class it belongs to.
     Singleton clusters (detected in only one TTA pass) receive maximum
     instability (1.0).  Reference: Kao et al. ACCV 2018.

Per-box uncertainty:
    u_box = α·H_norm + β·σ_score + γ·(1 - agreement) + δ·loc_instability
    with α=0.35, β=0.20, γ=0.15, δ=0.30  and H normalised by log(C).

Per-image uncertainty:
    u_image = confidence-weighted mean of u_box over all detected objects.
    (Low-confidence detections are down-weighted to reduce noise.)

References
----------
[1] Settles, B. (2009). Active Learning Literature Survey.
    University of Wisconsin–Madison Tech. Report 1648.
[2] Lakshminarayanan, B. et al. (2017). Simple and Scalable Predictive
    Uncertainty Estimation using Deep Ensembles. NeurIPS.
[3] Beluch, W. H. et al. (2018). The Power of Ensembles for Active
    Learning in Image Classification. CVPR.
[4] Wang, G. et al. (2019). Aleatoric uncertainty estimation with
    test-time augmentation for medical image segmentation. Comput. Biol.
    Med. (establishes TTA std as uncertainty proxy for detection tasks).

Usage
-----
    python src/training/active_sampler.py \\
        --weights checkpoints/finetune_best.pth.tar \\
        --top_k   50 \\
        --output  active_query.csv
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import core.config as config
from core.model import YOLOv4


# ============================================================
# UNCERTAINTY WEIGHTS  (tune if needed)
# ============================================================
W_ENTROPY      = 0.35  # classification entropy                [Settles 2009]
W_SCORE_STD    = 0.20  # TTA confidence variance               [Lakshminarayanan 2017]
W_DISAGREEMENT = 0.15  # TTA class disagreement ratio          [Beluch 2018]
W_LOC_INSTAB   = 0.30  # localization instability (1 - pairwise IoU)  [Kao 2018]


# ============================================================
# DECODE PREDICTIONS WITH CLASS ENTROPY
# ============================================================

def _decode_with_entropy(
    model: YOLOv4,
    image_tensor: torch.Tensor,
    anchors: list,
    device: str,
    conf_threshold: float = 0.1,
) -> List[List[float]]:
    """
    Decode raw model output into boxes and compute per-box class entropy.

    Returns list of [class_idx, score, x, y, w, h, class_entropy].
    class_entropy is Shannon entropy of the softmax class distribution,
    normalised by log(C) so it lives in [0, 1]  [Settles 2009, §2.1].
    """
    image_tensor = image_tensor.unsqueeze(0).to(device)
    num_classes = config.SPECIFIC_NUM_CLASSES
    log_C = math.log(num_classes)

    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            raw = model(image_tensor)

    all_boxes: List[List[float]] = []

    for scale_idx in range(3):
        pred = raw[scale_idx]           # [1, 3, S, S, 5+C]
        S = pred.shape[2]
        anchor = (
            torch.tensor(anchors[scale_idx], dtype=torch.float32)
            .to(device)
            .reshape(1, 3, 1, 1, 2)
        ) * S

        # --- Objectness and class probabilities ---
        obj_prob    = torch.sigmoid(pred[..., 0])           # [1, 3, S, S]
        class_probs = torch.softmax(pred[..., 5:], dim=-1)  # [1, 3, S, S, C]

        best_class_prob, best_class = class_probs.max(dim=-1)  # [1, 3, S, S]
        score = obj_prob * best_class_prob                      # [1, 3, S, S]

        # Shannon entropy normalised by log(C)  →  [0, 1]
        eps = 1e-9
        entropy = -(class_probs * torch.log(class_probs + eps)).sum(dim=-1) / log_C

        # --- Box coordinate decoding (mirrors cells_to_bboxes) ---
        xy_offset = torch.sigmoid(pred[..., 1:3])          # [1, 3, S, S, 2]
        wh_scaled = torch.exp(pred[..., 3:5]) * anchor     # [1, 3, S, S, 2]

        # Grid indices: col → x, row → y
        col_idx = torch.arange(S, device=device).view(1, 1, 1, S)
        row_idx = torch.arange(S, device=device).view(1, 1, S, 1)

        x_abs = (xy_offset[..., 0] + col_idx) / S          # [1, 3, S, S]
        y_abs = (xy_offset[..., 1] + row_idx) / S
        w_abs = wh_scaled[..., 0] / S
        h_abs = wh_scaled[..., 1] / S

        # Flatten everything to [3*S*S]
        score_np   = score.reshape(-1).cpu().numpy()
        class_np   = best_class.reshape(-1).cpu().numpy()
        entropy_np = entropy.reshape(-1).cpu().numpy()
        x_np = x_abs.reshape(-1).cpu().numpy()
        y_np = y_abs.reshape(-1).cpu().numpy()
        w_np = w_abs.reshape(-1).cpu().numpy()
        h_np = h_abs.reshape(-1).cpu().numpy()

        for i in range(len(score_np)):
            if score_np[i] > conf_threshold:
                all_boxes.append([
                    float(class_np[i]),
                    float(score_np[i]),
                    float(x_np[i]),
                    float(y_np[i]),
                    float(w_np[i]),
                    float(h_np[i]),
                    float(entropy_np[i]),
                ])

    return all_boxes


# ============================================================
# TTA HELPERS  (independent of pseudo_labeler to keep modules decoupled)
# ============================================================

def _get_transform(size: int) -> A.Compose:
    return A.Compose([
        A.Resize(height=size, width=size),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2(),
    ])


def _flip_x(boxes: List[List[float]]) -> List[List[float]]:
    """Mirror x-coordinate to undo horizontal flip (entropy field preserved)."""
    return [[b[0], b[1], 1.0 - b[2], b[3], b[4], b[5], b[6]] for b in boxes]


def _predict_tta(
    model: YOLOv4,
    image_np: np.ndarray,
    anchors: list,
    device: str,
    conf_threshold: float = 0.1,
) -> List[List[List[float]]]:
    """
    4-pass TTA: original, H-flip, scale-up ×1.25, scale-down ×0.75.
    Returns list of 4 box-lists, each in [cls, score, x, y, w, h, entropy].
    """
    base  = config.IMAGE_SIZE
    s_up  = int(base * 1.25) // 32 * 32
    s_dn  = int(base * 0.75) // 32 * 32

    t_base = _get_transform(base)
    t_up   = _get_transform(s_up)
    t_dn   = _get_transform(s_dn)

    def _run(img_np, transform):
        tensor = transform(image=img_np)["image"]
        return [b for b in _decode_with_entropy(model, tensor, anchors, device, conf_threshold)
                if b[1] > conf_threshold]

    boxes_orig = _run(image_np, t_base)
    boxes_flip = _flip_x(_run(np.fliplr(image_np).copy(), t_base))
    boxes_up   = _run(image_np, t_up)
    boxes_dn   = _run(image_np, t_dn)

    return [boxes_orig, boxes_flip, boxes_up, boxes_dn]


# ============================================================
# IoU HELPER
# ============================================================

def _iou(a: List[float], b: List[float]) -> float:
    """IoU between two boxes in [x_center, y_center, w, h] format."""
    ax1, ay1 = a[2] - a[4] / 2, a[3] - a[5] / 2
    ax2, ay2 = a[2] + a[4] / 2, a[3] + a[5] / 2
    bx1, by1 = b[2] - b[4] / 2, b[3] - b[5] / 2
    bx2, by2 = b[2] + b[4] / 2, b[3] + b[5] / 2

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / (union + 1e-8)


def _mean_pairwise_iou(boxes: List[List[float]]) -> float:
    """
    Mean pairwise IoU across all box pairs in a TTA cluster.

    High mean IoU → boxes agree on location → low localization instability.
    Low mean IoU → boxes disagree on location → high localization instability.

    Singleton clusters (detected in only one TTA pass) return 0.0, which maps
    to maximum instability (1.0) after the complement is taken by the caller.
    This is intentional: a box seen in just one pass cannot be location-verified.

    Reference: Kao et al. 2018 (localization tightness/stability criterion).
    """
    n = len(boxes)
    if n <= 1:
        return 0.0
    total = sum(_iou(boxes[i], boxes[j]) for i in range(n) for j in range(i + 1, n))
    return total / (n * (n - 1) / 2)


# ============================================================
# UNCERTAINTY-AWARE WBF
# ============================================================

def _wbf_uncertainty(
    tta_boxes_list: List[List[List[float]]],
    iou_threshold: float = 0.55,
) -> List[Dict]:
    """
    Weighted Boxes Fusion augmented with per-cluster uncertainty statistics.

    For each cluster of overlapping boxes (same object seen across TTA passes),
    computes:
      - fused_box     : confidence-weighted average position
      - mean_entropy  : mean classification entropy over cluster members
      - score_std     : std of confidence scores  (TTA disagreement)  [Lakshminarayanan 2017]
      - class_agreement: fraction of boxes whose class == majority class

    Returns list of dicts with keys:
        class, score, x, y, w, h, mean_entropy, score_std, class_agreement
    """
    all_boxes: List[List[float]] = []
    for boxes in tta_boxes_list:
        all_boxes.extend(boxes)

    if not all_boxes:
        return []

    all_boxes.sort(key=lambda b: b[1], reverse=True)

    clusters: List[List[List[float]]] = []
    fused:    List[List[float]] = []        # [cls, score, x, y, w, h]

    for box in all_boxes:
        matched = False
        for i, rep in enumerate(fused):
            if int(box[0]) == int(rep[0]) and _iou(box, rep) > iou_threshold:
                clusters[i].append(box)
                c = clusters[i]
                total_w = sum(b[1] for b in c)
                fused[i] = [
                    int(box[0]),
                    total_w / len(c),                           # avg score
                    sum(b[1] * b[2] for b in c) / total_w,     # w-avg x
                    sum(b[1] * b[3] for b in c) / total_w,     # w-avg y
                    sum(b[1] * b[4] for b in c) / total_w,     # w-avg w
                    sum(b[1] * b[5] for b in c) / total_w,     # w-avg h
                ]
                matched = True
                break
        if not matched:
            clusters.append([box])
            fused.append(list(box[:6]))

    results = []
    for i, c in enumerate(clusters):
        scores    = [b[1] for b in c]
        entropies = [b[6] for b in c]
        classes   = [int(b[0]) for b in c]
        majority  = max(set(classes), key=classes.count)
        agreement = classes.count(majority) / len(classes)

        # Localization instability: complement of mean pairwise IoU.
        # 1.0 → boxes completely disagree on location across TTA passes.
        # 0.0 → all TTA passes predict the exact same box.
        loc_instability = 1.0 - _mean_pairwise_iou(c)

        results.append({
            "class":            int(fused[i][0]),
            "score":            float(fused[i][1]),
            "x": float(fused[i][2]), "y": float(fused[i][3]),
            "w": float(fused[i][4]), "h": float(fused[i][5]),
            "mean_entropy":     float(np.mean(entropies)),
            "score_std":        float(np.std(scores)) if len(scores) > 1 else 0.0,
            "class_agreement":  float(agreement),
            "loc_instability":  float(loc_instability),
        })

    return results


# ============================================================
# IMAGE-LEVEL UNCERTAINTY SCORE
# ============================================================

def _image_uncertainty(clusters: List[Dict]) -> Tuple[float, Dict]:
    """
    Aggregate per-box uncertainty into a single image-level score.

    Uses confidence-weighted mean of per-box uncertainty  [Settles 2009, §2.3],
    so that high-confidence detections (more reliable uncertainty estimates)
    have greater influence than marginal ones.

    Returns (uncertainty_score, stats_dict).
    """
    if not clusters:
        return 0.0, {
            "n_detections": 0, "mean_entropy": 0.0, "mean_score_std": 0.0,
            "mean_class_agreement": 1.0, "mean_loc_instability": 0.0,
        }

    scores   = np.array([c["score"]            for c in clusters])
    entropies= np.array([c["mean_entropy"]      for c in clusters])
    stds     = np.array([c["score_std"]         for c in clusters])
    agrs     = np.array([c["class_agreement"]   for c in clusters])
    locs     = np.array([c["loc_instability"]   for c in clusters])

    # Per-box combined uncertainty: classification entropy + TTA score variance
    # + class disagreement + localization instability  [Kao 2018]
    per_box_u = (W_ENTROPY       * entropies
                 + W_SCORE_STD   * stds
                 + W_DISAGREEMENT * (1.0 - agrs)
                 + W_LOC_INSTAB  * locs)

    # Confidence-weighted mean
    weight_sum = scores.sum() + 1e-9
    img_score  = float((scores * per_box_u).sum() / weight_sum)

    stats = {
        "n_detections":          len(clusters),
        "mean_entropy":          float(entropies.mean()),
        "mean_score_std":        float(stds.mean()),
        "mean_class_agreement":  float(agrs.mean()),
        "mean_loc_instability":  float(locs.mean()),
    }
    return img_score, stats


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Uncertainty-based active learning query for YOLOv4."
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to finetuned checkpoint (.pth.tar).",
    )
    parser.add_argument(
        "--top_k", type=int, default=50,
        help="Number of images to recommend for manual annotation (default: 50).",
    )
    parser.add_argument(
        "--output", type=str, default="active_query.csv",
        help="Output CSV filename (saved next to checkpoints/, default: active_query.csv).",
    )
    parser.add_argument(
        "--conf", type=float, default=0.1,
        help="Minimum confidence to consider a prediction (default: 0.1).",
    )
    parser.add_argument(
        "--iou_wbf", type=float, default=0.55,
        help="IoU threshold for WBF clustering (default: 0.55).",
    )
    args = parser.parse_args()

    weights_path = args.weights
    if not os.path.isabs(weights_path):
        weights_path = os.path.join(config.BASE_DIR, weights_path)

    output_path = Path(config.BASE_DIR) / args.output

    print(f"\n{'='*60}")
    print("  ACTIVE SAMPLER — Uncertainty-Based Query")
    print(f"  Model   : {weights_path}")
    print(f"  Top-K   : {args.top_k}")
    print(f"  Output  : {output_path}")
    print(f"  Signals : entropy×{W_ENTROPY} + score_std×{W_SCORE_STD}"
          f" + disagreement×{W_DISAGREEMENT} + loc_instab×{W_LOC_INSTAB}")
    print(f"{'='*60}\n")

    # --- Load model ---
    model = YOLOv4(num_classes=config.SPECIFIC_NUM_CLASSES).to(config.DEVICE)
    ckpt  = torch.load(weights_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print("  Model loaded.\n")

    # --- Find unlabeled images ---
    unlabelled_dir = config.UNLABELLED_IMG_DIR
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted([
        f for f in os.listdir(unlabelled_dir)
        if Path(f).suffix.lower() in valid_ext
    ])

    if not image_files:
        print("  No images found in UNLABELLED_IMG_DIR. Aborting.")
        return

    print(f"  Unlabeled images: {len(image_files)}\n")

    # --- Score each image ---
    results: List[Dict] = []

    for img_name in tqdm(image_files, desc="Scoring uncertainty"):
        img_path = os.path.join(unlabelled_dir, img_name)
        image_np = np.array(Image.open(img_path).convert("RGB"))

        tta_boxes = _predict_tta(
            model, image_np, config.ANCHORS, config.DEVICE, conf_threshold=args.conf
        )
        clusters = _wbf_uncertainty(tta_boxes, iou_threshold=args.iou_wbf)
        score, stats = _image_uncertainty(clusters)

        results.append({
            "image":               img_name,
            "uncertainty_score":   score,
            **stats,
        })

    # --- Sort descending by uncertainty ---
    results.sort(key=lambda r: r["uncertainty_score"], reverse=True)

    # --- Save full CSV ---
    fieldnames = [
        "image", "uncertainty_score", "n_detections",
        "mean_entropy", "mean_score_std", "mean_class_agreement", "mean_loc_instability",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in fieldnames})

    # --- Print top-K recommendations ---
    top_k = min(args.top_k, len(results))
    print(f"\n{'='*60}")
    print(f"  TOP-{top_k} IMAGES FOR MANUAL ANNOTATION")
    print(f"{'='*60}")
    print(f"  {'#':<4}  {'Uncertainty':>11}  {'Detections':>10}  {'Entropy':>7}  Image")
    print(f"  {'-'*4}  {'-'*11}  {'-'*10}  {'-'*7}  {'-'*30}")
    for rank, row in enumerate(results[:top_k], start=1):
        print(f"  {rank:<4}  {row['uncertainty_score']:>11.4f}  "
              f"{row['n_detections']:>10}  "
              f"{row['mean_entropy']:>7.4f}  "
              f"{row['image']}")

    print(f"\n  Full ranking saved to: {output_path}")
    print(f"  Annotate images and add them to data/../train/labelled/")
    print(f"  Then re-run finetune.py to incorporate new annotations.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

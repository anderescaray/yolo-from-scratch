"""
Test Set Evaluation Script
=================================

"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import torch
import core.config as config
from core.model import YOLOv4
from core.utils import (
    get_loaders,
    check_class_accuracy,
    get_evaluation_bboxes,
    mean_average_precision,
)

def evaluate_model(weights_path):
    # Resolver el path relativo desde la raíz del proyecto
    import os
    if not os.path.isabs(weights_path):
        weights_path = os.path.join(config.BASE_DIR, weights_path)

    print(f"\n{'='*60}")
    print(f"  TEST SET EVALUATION")
    print(f"  Model: {weights_path}")
    print(f"{'='*60}\n")

    model = YOLOv4(num_classes=config.SPECIFIC_NUM_CLASSES).to(config.DEVICE)

    # Load weights from checkpoint
    print("Loading weights...")
    checkpoint = torch.load(weights_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(" ✅ Weights loaded successfully.\n")

    # Load test DataLoader
    _, test_loader, _ = get_loaders(
        train_csv_path=config.TRAIN_CSV, # Placeholder
        val_csv_path=config.TEST_CSV,
        train_img_dir=config.IMG_DIR,
        train_label_dir=config.LABEL_DIR,
        val_img_dir=config.TEST_IMG_DIR,
        val_label_dir=config.TEST_LABEL_DIR,
    )

    # Evaluation
    print("Calculating metrics on test set...")
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

    # Final Report
    print(f"\n  FINAL RESULTS")
    print(f"  mAP@{config.MAP_IOU_THRESH}: {map_test.item():.4f}")
    print(f"  Class Accuracy: {class_acc:.2f}%")
    print(f"  Object Detection Accuracy: {obj_acc:.2f}%")
    print(f"  No Object Accuracy (Background): {noobj_acc:.2f}%")
    print(f"{'='*60}\n")

if __name__ == "__main__":

# example: python src/eval_test.py --weights checkpoints/finetune_best.pth.tar
# checkpoints/ssl_best.pth.tar

    parser = argparse.ArgumentParser(description="Evaluar modelo YOLO en conjunto de Test")
    parser.add_argument(
        "--weights", 
        type=str, 
        required=True, 
        help="Ruta al archivo .pth.tar que quieres evaluar"
    )
    args = parser.parse_args()
    
    evaluate_model(args.weights)
"""
YOLOv4 Loss Function
====================

This module calculates the error between model prediction and ground truth.
Combines 4 different losses:
    1. Box Loss: Error in predicting bounding box (x, y) center and (w, h) size
    2. Object Loss: Error in detecting whether object exists or not (confidence) with IoU-aware
    3. No Object Loss: Penalty for detecting objects where nothing exists (background)
    4. Class Loss: Error in object class
            --> Uses label smoothing=0.1 to avoid overconfidence (YOLOv4 BoF)

Total Loss Formula:
    Loss = λ_box * L_box + λ_obj * L_obj + λ_noobj * L_noobj + λ_class * L_class
"""

import torch
import torch.nn as nn
from core.utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # For coordinates (x,y,w,h) and Classes we use Mean Squared Error or Cross Entropy
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss() # Binary Cross Entropy (includes Sigmoid)
        self.entropy = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Constants (Lambdas) to balance importance of each loss
        self.lambda_class = 1
        self.lambda_noobj = 0.5
        self.lambda_obj = 1
        self.lambda_box = 5   

    def ciou_loss(self, pred_boxes, target_boxes):
        """
        CIoU Loss (Complete IoU) - YOLOv4 improvement over YOLOv3 MSE

        Instead of treating x, y, w, h as 4 independent errors (as with MSE),
        CIoU directly measures geometric quality between two boxes with 3 terms:

            1. IoU:      How much the two boxes overlap (main objective)
            2. Distance: Penalizes distance between box centers
            3. Aspect:   Penalizes different aspect ratios (w/h)

        CIoU = IoU - (distance²/diagonal²) - α * v
            where v measures aspect difference and α weights it by current IoU

        The more boxes match, the closer CIoU approaches 1.
        Loss is: 1 - CIoU (so 0 = perfect boxes, 2 = opposite boxes)

        Args:
            pred_boxes:   Tensor (N, 4) with [x, y, w, h] already in grid scale
            target_boxes: Tensor (N, 4) with [x, y, w, h] already in grid scale
        Returns:
            loss: Scalar with mean CIoU Loss
        """
        # Convert from midpoints (tx, ty, w, h) to (x1, y1, x2, y2) corners
        # (upper left and lower right corners)
        # This makes it easier to calculate areas and intersections
        pred_x1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2 # tx - w/2
        pred_y1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2 # ty - h/2
        pred_x2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
        pred_y2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2

        tgt_x1 = target_boxes[..., 0] - target_boxes[..., 2] / 2
        tgt_y1 = target_boxes[..., 1] - target_boxes[..., 3] / 2
        tgt_x2 = target_boxes[..., 0] + target_boxes[..., 2] / 2
        tgt_y2 = target_boxes[..., 1] + target_boxes[..., 3] / 2

        # IoU
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)
        # clamp(0) so if no intersection area is 0 not negative
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        tgt_area  = (tgt_x2  - tgt_x1) * (tgt_y2  - tgt_y1)
        union_area = pred_area + tgt_area - inter_area + 1e-6

        iou = inter_area / union_area.clamp(min=1e-4)

        # Distance between centers / diagonal of enclosing box
        # Enclosing box is the minimal box containing both
        enclose_x1 = torch.min(pred_x1, tgt_x1)
        enclose_y1 = torch.min(pred_y1, tgt_y1)
        enclose_x2 = torch.max(pred_x2, tgt_x2)
        enclose_y2 = torch.max(pred_y2, tgt_y2)
        # c^2 = diagonal^2
        c2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + 1e-6

        # d^2 = distance^2 between centers of pred and target
        center_dist2 = (
            (pred_boxes[..., 0] - target_boxes[..., 0]) ** 2 +
            (pred_boxes[..., 1] - target_boxes[..., 1]) ** 2
        )

        # Aspect ratio consistency
        # v measures how different the arctangents of aspect ratio w/h are between boxes
        # If they have same ratio v = 0
        v = ((4 / (torch.pi ** 2)) * (
            torch.atan(target_boxes[..., 2] / (target_boxes[..., 3].clamp(min=1e-6))) -
            torch.atan(pred_boxes[..., 2] / (pred_boxes[..., 3].clamp(min=1e-6)))
        ) ** 2)

        # alpha weights v by current IoU: if IoU is high, aspect ratio matters more
        with torch.no_grad():
            # is a weighting coefficient, not a variable we want to differentiate.
            # If included in gradient graph we'd create circular dependencies in calculation
            # Low IoU -> small alpha -> aspect matters little
            # High IoU -> large alpha -> aspect matters more
            denominator_alpha = (1 - iou + v).clamp(min=1e-4)
            alpha = v / denominator_alpha

        # Final CIoU
        ciou = iou - (center_dist2 / c2.clamp(min=1e-4)) - alpha * v

        # Loss is 1 - CIoU: better box = lower loss
        return (1 - ciou).mean()

    def forward(self, predictions, target, anchors):
        """
        Calculates loss for ONE specific scale.
        predictions: Tensor (N, 3, S, S, 5+C) from model output
        target: Tensor (N, 3, S, S, 6) from dataset
        anchors: The 3 anchors for this scale
        """

        # Identify where objects exist and where they don't in target
        # Look only at index 0 of last dimension (confidence)
        # Create two matrices with shape (N,3,S,S)
        # obj is true where object actually exists, noobj is true where not (empty cells=background)
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0

    ### noobj loss ###
        # If no object, network should predict confidence 0.
        # Only look at position [..., 0:1] which is confidence
        no_object_loss = self.bce((predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]))

    ### obj loss ###
        # If object exists, network should predict confidence 1 (or actual IoU).
        anchors = anchors.reshape(1, 3, 1, 1, 2) # Format anchors for operations

        # Convert network predictions (bboxes) to normal format
        # and select only dimensions and concatenate
        box_preds = torch.cat([torch.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5].clamp(min=-10, max=10)) * anchors], dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()

        # So predicted confidence is like IoU
        object_loss = self.mse(torch.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

    ### box loss ###
        # only where obj = True

        # Build predicted boxes without modifying original predictions tensor
        # Work with local variables to not corrupt data between different grid scales
        pred_xy = torch.sigmoid(predictions[..., 1:3])           # center (x,y) between 0 and 1
        pred_wh = torch.exp(predictions[..., 3:5].clamp(min=-10, max=10)) * anchors    # (w,h) in grid scale
        pred_boxes = torch.cat([pred_xy, pred_wh], dim=-1)      # (x, y, w, h)

        # Also build target boxes in same format as pred_boxes
        # Dataset already stores w,h in grid scale format
        target_boxes = target[..., 1:5] # (x, y, w, h)

        # Calculate CIoU only for cells where object exists
        box_loss = self.ciou_loss(pred_boxes[obj], target_boxes[obj])

    ### class loss ###
        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long())
        )

    ### total loss ###
        total_loss = (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )

        if torch.isnan(total_loss):
            print("\nNAN DETECTED ALERT!")
            print(f"Box Loss: {box_loss.item()}")
            print(f"Obj Loss: {object_loss.item()}")
            print(f"NoObj Loss: {no_object_loss.item()}")
            print(f"Class Loss: {class_loss.item()}")

        return total_loss


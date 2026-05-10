"""
YOLOv3 Dataloader
============================

Class to load images and labels in the specific format required by YOLOv3

    1. Load images and labels
    2. Use Data Augmentation
    3. Build the TARGETS (Training matrices):
       Converts boxes [x, y, w, h] into matrices for 3 scales (13, 26, 52),
       assigning each object to the cell and Anchor Box most appropriate by IoU

The output returned by __getitem__:
    - image: Normalized image tensor
    - targets: Tuple with 3 tensors (one per scale) containing ground truths
"""
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from core.utils import iou_width_height as iou
from core.config import ANCHORS

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size=416,
        S=[13, 26, 52], # The sizes of the 3 grids
        C=20, # number of classes
        transform=None,
    ):
        df = pd.read_csv(csv_file) # Read the CSV
        self.img_files = df.iloc[:, 0].tolist()
        self.label_files = df.iloc[:, 1].tolist()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform # =config.train_transforms
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # Join all 9 anchors in one list
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C

        # Ignore boxes with warnings
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        """
        Load image and labels for a given index.

        Args:
            index (int): Index in the data list.

        Returns:
            image: Processed image.
            tuple(targets): Tuple of 3 tensors with labels mapped to grid.
        """
        # 1. LOAD IMAGE AND LABEL
        label_path = os.path.join(self.label_dir, self.label_files[index])

        # Fast native reading without np.loadtxt or pandas
        bboxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    # [class, x, y, w, h] -> [x, y, w, h, class]
                    class_label = int(float(parts[0]))
                    box = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), class_label]
                    bboxes.append(box)

        img_path = os.path.join(self.img_dir, self.img_files[index])
        image = np.array(Image.open(img_path).convert("RGB"))

        # 2. DATA AUGMENTATION
        # Clamp boxes to [0,1] range by converting to corners.
        # Single coordinate clamp is NOT sufficient: if y_center=1e-7 and h=0.56,
        # albumentations calculates y_min = y_center - h/2 < 0 and fails.
        clamped_bboxes = []
        for box in bboxes:
            x, y, w, h, cls = box[0], box[1], box[2], box[3], box[4]
            x1 = max(0.0, x - w / 2)
            y1 = max(0.0, y - h / 2)
            x2 = min(1.0, x + w / 2)
            y2 = min(1.0, y + h / 2)
            if x2 > x1 and y2 > y1:  # discard degenerate boxes
                clamped_bboxes.append(
                    [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, cls]
                )
        bboxes = clamped_bboxes

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # 3. PREPARE TARGETS (Training matrices)
        # Create 3 empty matrices one for each size 13, 26 and 52
        # Each cell will have 6 values: [prob_obj, x, y, w, h, class]
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]

        # 4. ASSIGN BOXES TO CELLS AND ANCHORS
        # iterate over each object
        for box in bboxes:
            # box: [x, y, w, h, class]
            # Calculate which of 9 anchors best matches this box (by IoU)
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors) # box[2:4] is width and height
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)

            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # Ensure each scale has max 1 anchor per object

            # Iterate through anchors (best to worst match)
            for anchor_idx in anchor_indices:
                # Which scale does this anchor belong to? (0=Large, 1=Medium, 2=Small)
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale

                S = self.S[scale_idx] # Current grid size

                # Which cell i,j does object center fall into?
                i, j = int(S * y), int(S * x) # y=row, x=column

                # Check if box is valid and cell not already taken
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchor[scale_idx]:
                    # if object exists confidence=1
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1

                    # Coordinates relative to cell
                    x_cell, y_cell = S * x - j, S * y - i
                    width_cell, height_cell = (width * S, height * S)

                    # Save coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 1] = x_cell
                    targets[scale_idx][anchor_on_scale, i, j, 2] = y_cell
                    targets[scale_idx][anchor_on_scale, i, j, 3] = width_cell
                    targets[scale_idx][anchor_on_scale, i, j, 4] = height_cell

                    # Save class
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)

                    has_anchor[scale_idx] = True

                # If anchor is not best but decent (>0.5 IoU), we ignore it
                # (set -1 to say "don't penalize this, it's ambiguous")
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # Ignore prediction

        return image, tuple(targets)
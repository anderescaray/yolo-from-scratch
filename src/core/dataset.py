"""
YOLOv4 Dataset
==============

Dataset class that loads images and labels in the format required by YOLOv4.

Pipeline per sample:
    1. Load image and bounding boxes from disk (raw, no transform).
    2. Optionally apply Mosaic augmentation (4 images combined into one canvas).
    3. Apply the Albumentations transform pipeline (augmentation + normalization).
    4. Build YOLO target tensors: convert boxes [x, y, w, h] into 3-scale grid
       matrices by assigning each object to its best-matching anchor and cell.

Output of __getitem__:
    - image : normalized image tensor  [3, H, W]
    - targets: tuple of 3 tensors, one per scale (13×13, 26×26, 52×52),
               each shaped [num_anchors_per_scale, S, S, 6]
               where the last dim is [obj_conf, x, y, w, h, class].
"""

import random
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
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
        S=[13, 26, 52],
        C=20,
        transform=None,
        mosaic_prob: float = 0.0,
    ):
        df = pd.read_csv(csv_file)
        self.img_files  = df.iloc[:, 0].tolist()
        self.label_files = df.iloc[:, 1].tolist()
        self.img_dir    = img_dir
        self.label_dir  = label_dir
        self.image_size = image_size
        self.transform  = transform
        self.S          = S
        self.anchors    = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.mosaic_prob = mosaic_prob

        # Threshold above which an anchor is considered a positive-but-ambiguous
        # match and should be ignored in the loss (not penalized as a false positive).
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.img_files)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_raw(self, index: int):
        """
        Load a single image and its bounding boxes from disk with no transforms.

        Returns:
            image   : numpy array  [H, W, 3]  uint8
            bboxes  : list of [x_center, y_center, w, h, class]  (all normalized [0, 1])
        """
        label_path = os.path.join(self.label_dir, self.label_files[index])
        bboxes = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls = int(float(parts[0]))
                    bboxes.append(
                        [float(parts[1]), float(parts[2]),
                         float(parts[3]), float(parts[4]), cls]
                    )

        img_path = os.path.join(self.img_dir, self.img_files[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        return image, bboxes

    def _clamp_bboxes(self, bboxes: list) -> list:
        """
        Clamp boxes to [0, 1] by converting to corner format, clamping, then
        converting back to center format.  Degenerate boxes are discarded.

        This is required because Albumentations computes corners internally and
        raises an error if any corner falls outside [0, 1].
        """
        valid = []
        for box in bboxes:
            x, y, w, h, cls = box
            x1 = max(0.0, x - w / 2)
            y1 = max(0.0, y - h / 2)
            x2 = min(1.0, x + w / 2)
            y2 = min(1.0, y + h / 2)
            if x2 > x1 and y2 > y1:
                valid.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, cls])
        return valid

    def _build_mosaic(self, index: int):
        """
        Build a Mosaic sample by tiling 4 images on a single IMAGE_SIZE×IMAGE_SIZE canvas.

        A random split point (cx, cy) divides the canvas into four quadrants.
        Each image is resized to fill its quadrant exactly, and the bounding boxes
        are remapped to the canvas coordinate space accordingly.

        The split point is constrained to [S/4, 3S/4] so that every quadrant
        has at least 25 % of the canvas area, preventing degenerate tiles.

        Args:
            index: index of the primary image (always placed in the top-left quadrant).

        Returns:
            canvas  : numpy array [IMAGE_SIZE, IMAGE_SIZE, 3]
            bboxes  : list of remapped boxes [x_center, y_center, w, h, class]
                      normalized to [0, 1] in canvas space.
        """
        S  = self.image_size
        cx = random.randint(S // 4, 3 * S // 4)   # horizontal split
        cy = random.randint(S // 4, 3 * S // 4)   # vertical split

        # Gray fill (neutral background for areas not covered by any image)
        canvas = np.full((S, S, 3), 114, dtype=np.uint8)
        combined_bboxes = []

        # Primary image in top-left; three additional random images for the rest
        indices = [index] + random.choices(range(len(self)), k=3)

        # (row_start, row_end, col_start, col_end) for each quadrant
        quadrants = [
            (0,  cy, 0,  cx),   # top-left
            (0,  cy, cx, S),    # top-right
            (cy, S,  0,  cx),   # bottom-left
            (cy, S,  cx, S),    # bottom-right
        ]

        for idx, (r0, r1, c0, c1) in zip(indices, quadrants):
            img, bboxes = self._load_raw(idx)
            qh, qw = r1 - r0, c1 - c0          # quadrant pixel dimensions

            # Resize image to fit the quadrant exactly
            img_resized = np.array(
                Image.fromarray(img).resize((qw, qh), Image.BILINEAR)
            )
            canvas[r0:r1, c0:c1] = img_resized

            # Remap each box from per-image normalized space to canvas normalized space
            for box in bboxes:
                bx, by, bw, bh, cls = box

                # Map box center and size from [0,1] within the quadrant
                # to absolute pixel coordinates in the full canvas
                cx_abs = c0 + bx * qw
                cy_abs = r0 + by * qh
                cw_abs = bw * qw
                ch_abs = bh * qh

                # Re-normalize to [0, 1] relative to the full canvas
                combined_bboxes.append([
                    cx_abs / S,
                    cy_abs / S,
                    cw_abs / S,
                    ch_abs / S,
                    cls,
                ])

        return canvas, combined_bboxes

    # ------------------------------------------------------------------
    # Main item loader
    # ------------------------------------------------------------------

    def __getitem__(self, index: int):
        """
        Load and preprocess one training sample.

        If mosaic_prob > 0 and the random draw succeeds, the sample is a
        mosaic of 4 images.  Otherwise it is a standard single-image sample.
        In both cases the same transform pipeline and target-building logic apply.

        Args:
            index: dataset index.

        Returns:
            image  : processed image tensor.
            targets: tuple of 3 YOLO grid tensors.
        """
        # 1. LOAD IMAGE AND BOUNDING BOXES
        if self.mosaic_prob > 0.0 and random.random() < self.mosaic_prob:
            image, bboxes = self._build_mosaic(index)
        else:
            image, bboxes = self._load_raw(index)

        # 2. CLAMP AND VALIDATE BOXES
        # Ensures all coordinates are strictly within [0, 1] before passing
        # to Albumentations, which raises errors on out-of-range corners.
        bboxes = self._clamp_bboxes(bboxes)

        # 3. DATA AUGMENTATION
        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image  = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # 4. BUILD YOLO TARGET TENSORS
        # One zero-initialized tensor per scale; each cell stores
        # [obj_conf, x_cell, y_cell, w_cell, h_cell, class].
        targets = [torch.zeros((self.num_anchors_per_scale, S, S, 6)) for S in self.S]

        for box in bboxes:
            # Find which of the 9 anchors best matches this box by IoU on w/h only
            iou_anchors   = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)

            x, y, width, height, class_label = box
            has_anchor = [False] * 3   # one flag per scale

            for anchor_idx in anchor_indices:
                scale_idx       = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx  % self.num_anchors_per_scale
                S_cur           = self.S[scale_idx]

                # Grid cell that contains the object center
                i, j = int(S_cur * y), int(S_cur * x)   # row, col

                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1

                    # Coordinates relative to the cell (offset from top-left corner)
                    targets[scale_idx][anchor_on_scale, i, j, 1] = S_cur * x - j
                    targets[scale_idx][anchor_on_scale, i, j, 2] = S_cur * y - i

                    # Width and height in cell units
                    targets[scale_idx][anchor_on_scale, i, j, 3] = width  * S_cur
                    targets[scale_idx][anchor_on_scale, i, j, 4] = height * S_cur

                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    # Decent but not best anchor — ignore rather than penalize
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return image, tuple(targets)

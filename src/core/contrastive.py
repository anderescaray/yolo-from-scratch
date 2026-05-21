"""
Contrastive Learning Module — Projection Head + Losses
======================================================

Trainable add-on for the frozen YOLOv4 backbone. Used by the contrastive
pretraining stage (`src/training/contrastive_pretrain.py`) to learn a
class-discriminative embedding space for Vía B diversity sampling, **without
touching the detector**.

Pipeline:
    Image [B, 3, 416, 416]
      → frozen backbone (CSPDarknet53)
      → feature map [B, 1024, 13, 13]
      → Global Average Pooling
      → vector [B, 1024]
      → ProjectionHead (this module)
      → L2-normalized vector [B, D]    D = CONTRASTIVE_PROJ_DIM

Losses
------
NTXentLoss (Chen et al. ICML 2020, SimCLR):
    Instance-discrimination. Two augmentations of the same image are positives;
    all other images in the batch are negatives. Self-supervised — no labels.

SupConLoss (Khosla et al. NeurIPS 2020):
    Supervised contrastive. All views of all images with the same class label
    are positives. Strong signal when (small) labelled set is available.

Joint loss used in `contrastive_pretrain.py`:
    L = L_NTXent(all images, 2 views) + λ · L_SupCon(labelled subset)

References
----------
[1] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A Simple
    Framework for Contrastive Learning of Visual Representations. ICML 2020.
    arXiv:2002.05709.
[2] Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P.,
    Maschinot, A., Liu, C., & Krishnan, D. (2020). Supervised Contrastive
    Learning. NeurIPS 2020. arXiv:2004.11362.
[3] He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). Momentum
    Contrast for Unsupervised Visual Representation Learning. CVPR 2020.
    (Referenced for the contrastive paradigm; we use SimCLR-style batches
     instead of a memory queue.)
[4] van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation Learning
    with Contrastive Predictive Coding. arXiv:1807.03748. (InfoNCE origin.)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# PROJECTION HEAD
# ============================================================

class ProjectionHead(nn.Module):
    """
    2-layer MLP that maps pooled backbone features into a low-dimensional
    space where contrastive distances are meaningful.

    Architecture (SimCLR g(·)):
        Linear(in_dim → hidden_dim) → BatchNorm1d → ReLU → Linear(hidden_dim → out_dim)

    L2-normalization is applied at the output so that cosine similarity reduces
    to a simple dot product. The projection head is discarded at evaluation
    time; the YOLOv4 detector keeps its original architecture untouched.
    """

    def __init__(self, in_dim: int = 1024, hidden_dim: int = 512, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=1, eps=1e-8)


# ============================================================
# LOSS 1 — NT-XENT (SimCLR)
# ============================================================

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy.
    [Chen et al. ICML 2020 — SimCLR]

    Given 2 augmented views of B images stacked as [2B, D] with view-1 in rows
    [0..B-1] and view-2 in rows [B..2B-1], for each anchor i the unique
    positive is at index (i + B) mod 2B; every other row is a negative.

    Loss for anchor i:
        L_i = -log [ exp(sim(i, pos_i) / τ) / Σ_{k ≠ i} exp(sim(i, k) / τ) ]
    where sim(a, b) = a · b   (z's are L2-normalized).

    Total loss is averaged over all 2B anchors.
    """

    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [2B, D] L2-normalized
        n = z.shape[0]
        assert n % 2 == 0, "NTXentLoss expects 2B rows (view1 stacked over view2)."
        b = n // 2
        device = z.device

        sim = (z @ z.t()) / self.temperature                       # [2B, 2B]
        self_mask = torch.eye(n, dtype=torch.bool, device=device)
        sim.masked_fill_(self_mask, float("-inf"))                 # remove self-similarity

        # Positive index for row i is (i + B) mod 2B
        pos_idx = torch.arange(n, device=device)
        pos_idx = (pos_idx + b) % n

        # log-softmax across the row (denominator) → pick the positive (numerator)
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)  # [2B, 2B]
        loss = -log_prob[torch.arange(n, device=device), pos_idx]   # [2B]
        return loss.mean()


# ============================================================
# LOSS 2 — SUPERVISED CONTRASTIVE (SupCon)
# ============================================================

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    [Khosla et al. NeurIPS 2020]

    Given [2B, D] embeddings and class labels [2B] (the same label repeated
    for both views of an image), the positives for anchor i are *all* rows
    j ≠ i that share the same label. Negatives are all rows with a different
    label.

    Loss for anchor i:
        L_i = -(1 / |P(i)|) · Σ_{p ∈ P(i)}
                              log[ exp(sim(i, p) / τ) / Σ_{a ≠ i} exp(sim(i, a) / τ) ]
    Anchors with |P(i)| = 0 are skipped.

    If `class_mask` is supplied (bool tensor of shape [2B]), only the rows
    where class_mask == True contribute to the loss (used to mask out
    unlabelled rows when the batch mixes labelled + unlabelled samples).
    """

    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        class_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # z: [N, D] L2-normalized   N = 2B
        # labels: [N]   (class id; arbitrary value where class_mask == False)
        n = z.shape[0]
        device = z.device

        if class_mask is None:
            class_mask = torch.ones(n, dtype=torch.bool, device=device)

        if class_mask.sum() < 2:
            return torch.zeros((), device=device, requires_grad=True)

        sim = (z @ z.t()) / self.temperature                                # [N, N]
        self_mask = torch.eye(n, dtype=torch.bool, device=device)
        sim.masked_fill_(self_mask, float("-inf"))

        # Positive mask: same label AND both rows are labelled AND not self
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)              # [N, N]
        both_lab  = class_mask.unsqueeze(0) & class_mask.unsqueeze(1)       # [N, N]
        pos_mask  = labels_eq & both_lab & (~self_mask)                     # [N, N]

        # log-softmax across the row (all non-self entries are valid negatives,
        # including unlabelled ones — they push labelled anchors apart from
        # arbitrary unlabelled negatives, which is the desired contrastive signal)
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)          # [N, N]

        # Mean log-prob over positives, then mean over anchors that have ≥ 1 positive
        pos_count = pos_mask.sum(dim=1)                                     # [N]
        valid     = (pos_count > 0) & class_mask
        if valid.sum() == 0:
            return torch.zeros((), device=device, requires_grad=True)

        # Sum log_prob over positives row-wise, divide by |P(i)|
        log_prob_pos_sum = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob)).sum(dim=1)  # [N]
        mean_log_prob_pos = log_prob_pos_sum[valid] / pos_count[valid].clamp(min=1)

        return -mean_log_prob_pos.mean()


# ============================================================
# SIMCLR AUGMENTATION PIPELINE
# ============================================================

def build_simclr_transform(image_size: int = 416) -> "object":
    """
    SimCLR-style augmentation pipeline (Chen et al. ICML 2020, Table 1 / §B.9):
        RandomResizedCrop → HorizontalFlip → strong ColorJitter (p=0.8)
        → RandomGrayscale (p=0.2) → GaussianBlur (p=0.5) → Normalize → ToTensor.

    Two independent applications of this transform produce the positive pair
    used by NTXentLoss / SupConLoss.

    Albumentations is used to stay consistent with the rest of the repository.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.5, 1.0), ratio=(0.75, 1.33), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        A.ToGray(p=0.2),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2(),
    ])

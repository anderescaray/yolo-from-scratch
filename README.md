# 🛒 Custom YOLO & Active Semi-Supervised Learning: Vision-Based Self-Checkout
### Building an Object Detection pipeline from scratch with Active Semi-Supervised Learning to minimize manual annotation

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch" />
  <img src="https://img.shields.io/badge/YOLO-v3%2Fv4%20from%20scratch-green?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" />
</p>

> **Bachelor's Final Thesis (TFG)** — A custom PyTorch implementation of YOLO combined with an Active Semi-Supervised Learning pipeline. This project solves the "data bottleneck" in computer vision by drastically reducing the manual labeling required to deploy a real-time, vision-only self-checkout system.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Active Semi-Supervised Learning Pipeline](#-active-semi-supervised-learning-pipeline)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Future Work](#-future-work)

---

## 🔍 Overview

Traditional self-checkout systems rely on barcodes. This project explores a fully **vision-based** alternative: a camera observes products placed one at a time on a surface, the model identifies them, and a digital ticket is automatically built. The user pays at the end with no scanning required.

The core challenge is **data**. Labelling thousands of product images by hand is impractical. To solve this, the project combines:

- A **custom YOLO detector** (inspired by YOLOv3/v4) trained first on a generic product dataset and then fine-tuned on specific products.
- An **Active Semi-Supervised Learning loop** that uses model confidence and deep clustering to maximise the value of every manual annotation.

The result is a system capable of learning to recognise a new set of products with only a **small initial set of manually labelled images**, expanding its own dataset iteratively.

---

## ✨ Key Features

- **YOLO from scratch** — Full reimplementation in PyTorch, no black-box dependencies.
- **Two-stage training** — Generic pre-training followed by domain-specific fine-tuning with frozen backbone.
- **Pseudo-labelling** — High-confidence detections on unlabelled images are automatically added to the training set.
- **Deep Clustering for Active Learning** — Object crops are embedded via the backbone; similar products cluster together in latent space, allowing a human annotator to label only the cluster centroid and propagate labels to the rest.
- **Real-time webcam inference** — Live product detection and ticket generation.
- **Iterative self-improvement loop** — Each training cycle produces a better model, which in turn generates better pseudo-labels and more coherent clusters.

---

## 🏗️ Architecture

### YOLO Model

The detector is a hybrid of YOLOv3 and YOLOv4, implemented entirely from scratch in PyTorch. It uses:

- A **Darknet-inspired backbone** for feature extraction, pre-trained on a generic dataset.
- **Multi-scale prediction heads** for detecting objects at different sizes.
- A custom **loss function** combining objectness, bounding box regression (CIoU), and classification losses.

### Two-Stage Training

```
Stage 1 ─ Generic Dataset
    │  Train full model (backbone + heads)
    ▼
Stage 2 ─ Specific Products
    │  Freeze backbone weights
    │  Replace output layer (new number of classes)
    │  Fine-tune on small labelled set
    ▼
  Base Model
```

---

## 🔄 Active Semi-Supervised Learning Pipeline

This is the core contribution of the project. The goal is to build a large, high-quality labelled dataset while minimising human effort.

```
┌─────────────────────────────────────────────────────────┐
│                   ITERATIVE LOOP                        │
│                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │  Train   │───▶│  Inference   │───▶│  Sort by      │  │
│  │  Model   │    │  on unlabels │    │  confidence   │  │
│  └──────────┘    └──────────────┘    └───────────────┘  │
│       ▲                                    │             │
│       │               ┌────────────────────┤             │
│       │               │                    │             │
│       │        High confidence         Low confidence    │
│       │               │                    │             │
│       │               ▼                    ▼             │
│       │        Pseudo-labels       Crop → Backbone       │
│       │        (auto-add to        → Latent space        │
│       │         train set)         → Deep Clustering     │
│       │               │                    │             │
│       │               │            Human labels          │
│       │               │            centroid only         │
│       │               │                    │             │
│       └───────────────┴────────────────────┘             │
└─────────────────────────────────────────────────────────┘
```

**Semi-supervised branch:** Detections with confidence above a threshold are treated as ground truth and added directly to the training set as pseudo-labels.

**Active learning branch:** For uncertain detections, the crop of the detected object is passed through the backbone to extract its latent representation. Deep clustering groups these embeddings — products of the same class naturally cluster together, even when the classifier fails. An annotator is then asked to label only the images closest to each cluster centroid. These few labels are propagated to the rest of the cluster.

Each iteration of the loop expands the dataset and improves the model, requiring progressively less human intervention.

---

## 📁 Project Structure

```
project/
├── README.md
├── .gitignore
├── requirements.txt
│
├── src/
│   ├── config.py          # Hyperparameters, paths, anchor configurations
│   ├── dataset.py         # Dataset loading, augmentation, CSV parsing
│   ├── generate_csv.py    # Generates train/test CSV files from data folder
│   ├── model.py           # YOLO architecture (backbone + neck + heads)
│   ├── loss.py            # Custom loss: objectness + CIoU + classification
│   ├── train.py           # Training loop with checkpoint saving
│   ├── test_visual.py     # Visual evaluation with bounding box rendering
│   ├── webcam.py          # Real-time inference via webcam
│   └── utils.py           # NMS, mAP, IoU, plotting utilities
│
├── data/
│   ├── train/             # Training images + YOLO-format .txt labels
│   ├── test/              # Test images + labels
│   ├── valid/             # Validation images + labels
│   ├── train_csv.csv      # Manifest with image paths and annotations
│   └── test_csv.csv
│
├── checkpoints/           # Saved model weights and training checkpoints
└── saved_images/          # Output images from test_visual.py
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/anderescaray/yolo-from-scratch.git
cd your-repo-name

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.x
- OpenCV
- NumPy, Pandas, Matplotlib, tqdm

---

## 🚀 Usage

### 1. Prepare the dataset

Place images and YOLO-format `.txt` label files in `data/train/`, `data/test/`, and `data/valid/`. Then generate the CSV manifests:

```bash
python src/generate_csv.py
```

### 2. Configure the model

Edit `src/config.py` to set the number of classes, anchor sizes, image resolution, and training hyperparameters.

### 3. Train

```bash
python src/train.py
```

Checkpoints are saved automatically to `checkpoints/`.

### 4. Evaluate visually

```bash
python src/test_visual.py
```

Output images with predicted bounding boxes are saved to `saved_images/`.

### 5. Run real-time inference

```bash
python src/webcam.py
```

Point your webcam at a product and observe live detections.

---

## 📊 Results

> *(Results will be updated upon project completion)*

| Metric | Generic Dataset | Fine-tuned (manual labels only) | After SSL Loop |
|--------|----------------|----------------------------------|----------------|
| mAP@0.5 | — | — | — |
| Precision | — | — | — |
| Recall | — | — | — |
| FPS (webcam) | — | — | — |

**Dataset growth per iteration:**

| Iteration | Manual labels | Pseudo-labels added | Total training samples |
|-----------|--------------|---------------------|------------------------|
| 0 | — | — | — |
| 1 | — | — | — |
| 2 | — | — | — |

---

## 🔮 Future Work

### Pipeline improvements

- **Per-class dynamic confidence threshold** — Replace the fixed pseudo-label threshold with a per-class adaptive one (computed as the mean confidence of each class plus a margin). This prevents dominant classes from flooding the training set while underrepresented ones stagnate, and avoids the need to manually tune a single global value.

- **Consistency filtering for pseudo-labels** — Before accepting a pseudo-label, run inference on multiple augmented versions of the same image (flip, brightness, crop). Only add it to the training set if predictions are consistent across augmentations. This filters out cases where the model is accidentally confident and reduces noise in the training set.

- **Margin sampling for active selection** — Instead of selecting uncertain samples purely by low confidence score, prioritise images where the gap between the top-1 and top-2 class probabilities (the margin) is smallest. These samples sit on the decision boundary and are the most informative for the annotator per label spent.

- **Projection head with contrastive learning (SimCLR)** — The YOLO backbone produces embeddings optimised for detection, not clustering. Adding a small projection head (2–3 linear layers) trained with a contrastive objective — treating differently-augmented crops of the same object as positive pairs — would yield a much cleaner latent space, leading to more coherent clusters and more reliable label propagation without requiring any extra annotations.

### System extensions

- Build a full end-to-end checkout UI with price lookup and payment simulation.
- Evaluate generalisation to other product domains beyond supermarket items.
- Explore consistency regularisation (e.g. MeanTeacher) as an additional semi-supervised signal alongside pseudo-labelling.

---

## 👤 Author

**Ander Escaray**  
Bachelor's Degree in Data Science — UPNA, 2026  
[linkedin.com/in/ander-escaray-354641389](https://www.linkedin.com) · [github.com/anderescaray](https://github.com)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
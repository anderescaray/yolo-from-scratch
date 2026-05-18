"""
Deep Cluster Sampler — Diversity-Based Query Strategy
=====================================================
Selects unlabeled images for annotation by maximizing coverage of the backbone
feature space (diversity). Complements the uncertainty-based Vía A
(active_sampler.py) for the central AL comparison experiment of the TFG.

Embedding pipeline (shared by all strategies):
    Image [3,416,416] → backbone (CSPDarknet53, frozen, partial forward)
    → feature map [1024,13,13] → Global Average Pooling → [1024]
    → L2-normalize → PCA whitening → [D]

Image-level GAP is used instead of crop-level to guarantee a consistent
embedding space for both labeled and unlabeled images: unlabeled crops would
depend on the model's bounding-box predictions (mAP=0.15), producing a
different noise profile from GT crops used for labeled images, which biases
Core-Set distance comparisons.

Three selection strategies (--strategy):

  coreset  [Sener & Savarese, ICLR 2018]
    Greedy k-center / farthest-first traversal. Seeds S with the already-labeled
    set so selected points are maximally far from what the model has seen.
    2-approximation of the optimal k-center solution.

  kmeans   [Arthur & Vassilvitskii, SODA 2007]
    k-means++ clustering + balanced nearest-to-centroid selection. K selected
    automatically via silhouette score [Rousseeuw 1987] if --k 0 is passed.

  clue     [Prabhu et al., ICCV 2021]
    Uncertainty-weighted k-means: scales each embedding by the uncertainty score
    from active_query.csv before clustering. Bridges Vía A and Vía B.

Output CSV (diversity_query.csv):
    Selected images appear first (rows 1..B, diversity_score = B..1).
    Non-selected images follow (diversity_score = 0).
    Fully compatible with prepare_al_datasets.py --score_col diversity_score.

References
----------
[1] Sener & Savarese. Active Learning for CNNs: A Core-Set Approach. ICLR 2018.
[2] Caron et al. Deep Clustering for Unsupervised Learning. ECCV 2018.
[3] Arthur & Vassilvitskii. k-means++: Careful Seeding. SODA 2007.
[4] Rousseeuw. Silhouettes. J. Comp. Appl. Math., 20, 1987.
[5] Jolliffe. Principal Component Analysis. Springer, 2002.
[6] Prabhu et al. CLUE: Active Domain Adaptation. ICCV 2021.
[7] van der Maaten & Hinton. Visualizing Data using t-SNE. JMLR 9, 2008.
[8] McInnes et al. UMAP. arXiv:1802.03426, 2018.

Usage
-----
    python src/training/deep_cluster_sampler.py \\
        --weights checkpoints/finetune_best_map.pth.tar \\
        --strategy coreset --budget 75
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import core.config as config
from core.model import YOLOv4

# ============================================================
# CONSTANTS  (all overridable via CLI; sourced from config)
# ============================================================
D_BUDGET        = config.DIVERSITY_BUDGET
D_STRATEGY      = config.DIVERSITY_STRATEGY
D_PCA_DIM       = config.PCA_DIM             # [Jolliffe 2002]
D_K             = config.DIVERSITY_K          # [Arthur & Vassilvitskii 2007]
D_K_RANGE       = config.SILHOUETTE_K_RANGE   # [Rousseeuw 1987]
_EMBED_BATCH    = 64


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def _build_transform() -> A.Compose:
    return A.Compose([
        A.Resize(height=config.IMAGE_SIZE, width=config.IMAGE_SIZE),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2(),
    ])


def _extract_embeddings(
    model: YOLOv4,
    image_paths: List[Path],
    device: str,
    desc: str = "Extracting embeddings",
) -> np.ndarray:
    """
    Partial backbone forward → GAP → L2-normalize → [N, 1024].

    Runs only model.backbone (11 layers, no SPP/neck/heads).
    Feature map [B, 1024, 13, 13] → GAP → [B, 1024] → L2-norm.
    Same pipeline for both labeled and unlabeled images ensures a consistent
    embedding space for distance-based selection. [Caron et al. ECCV 2018]
    """
    transform = _build_transform()
    all_emb: List[np.ndarray] = []

    for i in tqdm(range(0, len(image_paths), _EMBED_BATCH), desc=desc, leave=False):
        batch_paths = image_paths[i : i + _EMBED_BATCH]
        tensors = []
        for p in batch_paths:
            img_np = np.array(Image.open(p).convert("RGB"))
            tensors.append(transform(image=img_np)["image"])

        batch = torch.stack(tensors).to(device, non_blocking=True)

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                x = batch
                for layer in model.backbone:   # backbone only, no SPP/neck/heads
                    x = layer(x)
                emb = x.float().mean(dim=(-2, -1))  # GAP: [B, 1024, 13, 13] → [B, 1024]

        norms = emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
        all_emb.append((emb / norms).cpu().numpy())

    return np.concatenate(all_emb, axis=0)  # [N, 1024]


def _pca_whiten(
    unlabelled_emb: np.ndarray,
    labelled_emb: Optional[np.ndarray],
    n_components: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    PCA whitening fitted on all available embeddings, applied to both sets.
    Reduces 1024-d to D-d to combat the curse of dimensionality before L2
    distance computation. [Jolliffe 2002; Sener & Savarese 2018]
    """
    fit_data = (
        np.vstack([unlabelled_emb, labelled_emb])
        if labelled_emb is not None and len(labelled_emb) > 0
        else unlabelled_emb
    )
    n_comp = min(n_components, fit_data.shape[0] - 1, fit_data.shape[1])

    pca = PCA(n_components=n_comp, whiten=True, random_state=42)
    pca.fit(fit_data)

    u_red = pca.transform(unlabelled_emb)
    l_red = pca.transform(labelled_emb) if labelled_emb is not None else None
    expl  = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA {fit_data.shape[1]}d → {n_comp}d  ({expl * 100:.1f}% variance explained)")
    return u_red, l_red


# ============================================================
# DISTANCE UTILITY
# ============================================================

def _min_sq_dists(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Minimum squared L2 distance from each point to its nearest center.
    Uses ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>. Result shape: [N_points].
    """
    pts_sq  = (points  ** 2).sum(axis=1)[:, None]   # [N, 1]
    ctrs_sq = (centers ** 2).sum(axis=1)[None, :]   # [1, M]
    cross   = points @ centers.T                     # [N, M]
    dists   = pts_sq + ctrs_sq - 2.0 * cross
    np.maximum(dists, 0.0, out=dists)
    return dists.min(axis=1)                         # [N]


# ============================================================
# STRATEGY: CORESET (GREEDY K-CENTER)
# ============================================================

def _coreset(
    unlabelled_emb: np.ndarray,
    labelled_emb:   Optional[np.ndarray],
    budget:         int,
) -> List[int]:
    """
    Greedy k-center / farthest-first traversal. [Sener & Savarese, ICLR 2018]

    Iteratively picks the unlabeled point farthest from the current selected set
    S. Seeded with the already-labeled images: selects points maximally far from
    what the model has already seen. O(budget * N * D) — negligible at N~3600.
    """
    rng = np.random.default_rng(42)

    if labelled_emb is not None and len(labelled_emb) > 0:
        centers = labelled_emb.copy()
    else:
        seed = int(rng.integers(len(unlabelled_emb)))
        centers = unlabelled_emb[seed : seed + 1]

    min_dists = _min_sq_dists(unlabelled_emb, centers)
    selected:  List[int] = []

    for _ in range(budget):
        idx = int(np.argmax(min_dists))
        selected.append(idx)
        new_dists = _min_sq_dists(unlabelled_emb, unlabelled_emb[idx : idx + 1])
        np.minimum(min_dists, new_dists, out=min_dists)

    return selected


# ============================================================
# SHARED HELPER: BALANCED CLUSTER ALLOCATION
# ============================================================

def _balanced_alloc(cluster_labels: np.ndarray, k: int, budget: int) -> np.ndarray:
    """Distribute budget across clusters proportionally to cluster size."""
    sizes  = np.bincount(cluster_labels, minlength=k).astype(float)
    alloc  = np.floor(budget * sizes / sizes.sum()).astype(int)
    remain = budget - int(alloc.sum())
    fracs  = budget * sizes / sizes.sum() - alloc
    for ci in np.argsort(-fracs)[:remain]:
        alloc[ci] += 1
    return alloc


def _nearest_to_centroids(
    emb: np.ndarray,
    cluster_labels: np.ndarray,
    centroids: np.ndarray,
    alloc: np.ndarray,
) -> List[int]:
    """For each cluster, select the `alloc[c]` points nearest to the centroid."""
    selected: List[int] = []
    for c_id, n in enumerate(alloc):
        if n == 0:
            continue
        mask = np.where(cluster_labels == c_id)[0]
        dists = np.linalg.norm(emb[mask] - centroids[c_id], axis=1)
        selected.extend(mask[np.argsort(dists)[:n]].tolist())
    return selected


# ============================================================
# STRATEGY: KMEANS++ + BALANCED SELECTION
# ============================================================

def _kmeans_select(
    unlabelled_emb: np.ndarray,
    budget:         int,
    k:              int,
    k_range:        Tuple[int, int],
) -> Tuple[List[int], np.ndarray]:
    """
    k-means++ clustering + balanced nearest-to-centroid selection.
    [Arthur & Vassilvitskii, SODA 2007; Rousseeuw 1987 silhouette for K]
    """
    sample_size = min(2000, len(unlabelled_emb))

    if k == 0:
        best_k, best_sil = k_range[0], -1.0
        print(f"  Silhouette search K={k_range[0]}..{k_range[1]}:")
        for ck in range(k_range[0], k_range[1] + 1):
            labels_tmp = KMeans(n_clusters=ck, init="k-means++", n_init=5,
                                random_state=42).fit_predict(unlabelled_emb)
            sil = silhouette_score(unlabelled_emb, labels_tmp,
                                   sample_size=sample_size, random_state=42)
            print(f"    K={ck}: silhouette={sil:.4f}")
            if sil > best_sil:
                best_k, best_sil = ck, sil
        print(f"  → Best K={best_k} (silhouette={best_sil:.4f})")
        k = best_k

    km             = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    cluster_labels = km.fit_predict(unlabelled_emb)
    alloc          = _balanced_alloc(cluster_labels, k, budget)
    selected       = _nearest_to_centroids(unlabelled_emb, cluster_labels,
                                            km.cluster_centers_, alloc)
    return selected, cluster_labels


# ============================================================
# STRATEGY: CLUE (UNCERTAINTY-WEIGHTED K-MEANS)
# ============================================================

def _clue_select(
    unlabelled_emb:    np.ndarray,
    uncertainty_scores: np.ndarray,
    budget:            int,
    k:                 int,
) -> Tuple[List[int], np.ndarray]:
    """
    Uncertainty-weighted k-means + nearest-to-centroid selection.
    [Prabhu et al., ICCV 2021 — CLUE]

    Each embedding is scaled by its uncertainty score before clustering so that
    high-uncertainty regions attract cluster centroids, bridging Vía A and Vía B.
    Uncertainty is normalized to [0.1, 1.0] to avoid collapsing low-uncertainty
    points to the origin.
    """
    u = uncertainty_scores.copy()
    u_min, u_max = float(u.min()), float(u.max())
    if u_max > u_min:
        u = 0.1 + 0.9 * (u - u_min) / (u_max - u_min)
    else:
        u = np.ones_like(u)

    weighted_emb   = unlabelled_emb * u[:, np.newaxis]
    km             = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    cluster_labels = km.fit_predict(weighted_emb)
    alloc          = _balanced_alloc(cluster_labels, k, budget)
    selected       = _nearest_to_centroids(weighted_emb, cluster_labels,
                                            km.cluster_centers_, alloc)
    return selected, cluster_labels


# ============================================================
# VISUALIZATION
# ============================================================

def _save_visualization(
    unlabelled_emb: np.ndarray,
    labelled_emb:   Optional[np.ndarray],
    selected_idx:   List[int],
    cluster_labels: Optional[np.ndarray],
    strategy:       str,
    output_path:    Path,
) -> None:
    """
    2D scatter of backbone embeddings for qualitative coverage inspection.
    Uses UMAP [McInnes et al. 2018] if installed, falls back to PCA 2D.
    Saved to saved_images/ for the thesis memoria coverage figure.
    [van der Maaten & Hinton, JMLR 2008]
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    all_emb = (
        np.vstack([unlabelled_emb, labelled_emb])
        if labelled_emb is not None and len(labelled_emb) > 0
        else unlabelled_emb
    )

    try:
        import umap as umap_lib
        reducer = umap_lib.UMAP(n_components=2, random_state=42, n_jobs=1)
        print("  Computing UMAP projection...")
    except ImportError:
        reducer = PCA(n_components=2)
        print("  umap-learn not found — using PCA 2D for visualization.")

    coords   = reducer.fit_transform(all_emb)
    u_coords = coords[: len(unlabelled_emb)]
    l_coords = coords[len(unlabelled_emb) :] if labelled_emb is not None else None

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(u_coords[:, 0], u_coords[:, 1],
               c="lightgray", s=8, alpha=0.5, label="Unlabelled", zorder=1)

    if cluster_labels is not None:
        k      = int(cluster_labels.max()) + 1
        colors = cm.tab10(np.linspace(0, 1, min(k, 10)))
        for c_id in range(k):
            mask = cluster_labels == c_id
            ax.scatter(u_coords[mask, 0], u_coords[mask, 1],
                       c=[colors[c_id % 10]], s=10, alpha=0.35, zorder=2)

    sel_arr = np.array(selected_idx)
    ax.scatter(u_coords[sel_arr, 0], u_coords[sel_arr, 1],
               c="red", s=70, edgecolors="darkred", linewidths=0.5,
               label=f"Selected ({len(selected_idx)})", zorder=4)

    if l_coords is not None:
        ax.scatter(l_coords[:, 0], l_coords[:, 1],
                   c="royalblue", s=40, marker="^", alpha=0.75,
                   label=f"Already labelled ({len(l_coords)})", zorder=3)

    ax.set_title(f"Diversity sampling: {strategy.upper()} — backbone embeddings (2D)")
    ax.legend(loc="best", fontsize=9)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {output_path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diversity-based active learning query for YOLOv4 (Vía B)."
    )
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to finetuned checkpoint (.pth.tar).")
    parser.add_argument("--strategy", type=str, default=D_STRATEGY,
                        choices=["coreset", "kmeans", "clue"],
                        help="Selection strategy (default: coreset).")
    parser.add_argument("--budget", type=int, default=D_BUDGET,
                        help=f"Images to select for annotation (default: {D_BUDGET}).")
    parser.add_argument("--k", type=int, default=D_K,
                        help="Clusters for kmeans/clue. 0 = auto via silhouette.")
    parser.add_argument("--pca_dim", type=int, default=D_PCA_DIM,
                        help=f"PCA output dimensionality (default: {D_PCA_DIM}).")
    parser.add_argument("--uncertainty_csv", type=str,
                        default=str(Path(config.BASE_DIR) / "active_query.csv"),
                        help="Path to active_query.csv (required for --strategy clue).")
    parser.add_argument("--output", type=str, default="diversity_query.csv",
                        help="Output CSV filename (default: diversity_query.csv).")
    args = parser.parse_args()

    weights_path = (
        args.weights if os.path.isabs(args.weights)
        else os.path.join(config.BASE_DIR, args.weights)
    )
    output_path = Path(config.BASE_DIR) / args.output

    print(f"\n{'='*60}")
    print(f"  DEEP CLUSTER SAMPLER — Diversity Query (Vía B)")
    print(f"  Model    : {weights_path}")
    print(f"  Strategy : {args.strategy}")
    print(f"  Budget   : {args.budget}")
    print(f"  PCA dim  : {args.pca_dim}")
    print(f"  Output   : {output_path}")
    print(f"{'='*60}\n")

    # --- Load model (backbone only used; SPP/neck/heads loaded but bypassed) ---
    model = YOLOv4(num_classes=config.SPECIFIC_NUM_CLASSES).to(config.DEVICE)
    ckpt  = torch.load(weights_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print("  Model loaded (backbone frozen).\n")

    # --- Unlabelled image list ---
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    unlabelled_dir = config.UNLABELLED_IMG_DIR
    unlabelled_files: List[Path] = sorted([
        unlabelled_dir / f for f in os.listdir(unlabelled_dir)
        if Path(f).suffix.lower() in valid_ext
    ])
    if not unlabelled_files:
        print("  No images found in UNLABELLED_IMG_DIR. Aborting.")
        return
    print(f"  Unlabelled images : {len(unlabelled_files)}")

    # --- Labelled image list (seed for coreset; same pipeline → consistent space) ---
    labelled_files: List[Path] = []
    if args.strategy == "coreset":
        labelled_dir = config.IMG_DIR
        if labelled_dir.exists():
            labelled_files = sorted([
                labelled_dir / f for f in os.listdir(labelled_dir)
                if Path(f).suffix.lower() in valid_ext
            ])
        print(f"  Labelled seed     : {len(labelled_files)} images")
    print()

    # --- Extract embeddings ---
    u_emb_raw = _extract_embeddings(model, unlabelled_files, config.DEVICE,
                                    desc="Unlabelled embeddings")
    l_emb_raw: Optional[np.ndarray] = None
    if labelled_files:
        l_emb_raw = _extract_embeddings(model, labelled_files, config.DEVICE,
                                        desc="Labelled embeddings  ")

    # --- PCA whitening ---
    print("\n  PCA whitening...")
    u_emb, l_emb = _pca_whiten(u_emb_raw, l_emb_raw, args.pca_dim)
    print()

    # --- Run selected strategy ---
    cluster_labels: Optional[np.ndarray] = None

    if args.strategy == "coreset":
        print("  Core-Set greedy k-center  [Sener & Savarese, ICLR 2018]")
        selected_idx = _coreset(u_emb, l_emb, args.budget)

    elif args.strategy == "kmeans":
        print(f"  k-means++ (K={args.k if args.k > 0 else 'auto'})  "
              f"[Arthur & Vassilvitskii, SODA 2007]")
        selected_idx, cluster_labels = _kmeans_select(
            u_emb, args.budget, args.k, D_K_RANGE
        )

    else:  # clue
        unc_csv_path = Path(args.uncertainty_csv)
        if not unc_csv_path.exists():
            print(f"  [ERROR] uncertainty_csv not found: {unc_csv_path}")
            print("  Run active_sampler.py first, or use --strategy coreset|kmeans.")
            return
        unc_map: Dict[str, float] = {}
        with open(unc_csv_path, newline="") as f:
            for row in csv.DictReader(f):
                unc_map[row["image"]] = float(row["uncertainty_score"])
        unc_scores = np.array([unc_map.get(p.name, 0.5) for p in unlabelled_files])
        print(f"  CLUE (uncertainty-weighted k-means, K={args.k})  "
              f"[Prabhu et al., ICCV 2021]")
        selected_idx, cluster_labels = _clue_select(
            u_emb, unc_scores, args.budget, args.k
        )

    selected_set = set(selected_idx)
    budget_used  = len(selected_idx)

    # --- Console summary ---
    print(f"\n{'='*60}")
    print(f"  TOP-{budget_used} SELECTED IMAGES  (strategy={args.strategy})")
    print(f"{'='*60}")
    print(f"  {'Rank':<5}  {'Cluster':>7}  Image")
    print(f"  {'-'*5}  {'-'*7}  {'-'*40}")
    for rank, idx in enumerate(selected_idx, start=1):
        cid = int(cluster_labels[idx]) if cluster_labels is not None else -1
        print(f"  {rank:<5}  {cid:>7}  {unlabelled_files[idx].name}")

    # --- Build CSV rows ---
    # diversity_score: budget..1 for selected (descending by selection order),
    # 0 for non-selected. This ordering makes prepare_al_datasets.py --top_k B
    # pick exactly the selected images when --score_col diversity_score is used.
    div_scores = np.zeros(len(unlabelled_files))
    for rank, idx in enumerate(selected_idx, start=1):
        div_scores[idx] = float(budget_used + 1 - rank)

    rows_selected = [
        {
            "image":           unlabelled_files[idx].name,
            "diversity_score": float(div_scores[idx]),
            "selected":        1,
            "selected_rank":   rank,
            "strategy":        args.strategy,
            "cluster_id":      int(cluster_labels[idx]) if cluster_labels is not None else -1,
        }
        for rank, idx in enumerate(selected_idx, start=1)
    ]
    rows_rest = [
        {
            "image":           unlabelled_files[i].name,
            "diversity_score": 0.0,
            "selected":        0,
            "selected_rank":   0,
            "strategy":        args.strategy,
            "cluster_id":      int(cluster_labels[i]) if cluster_labels is not None else -1,
        }
        for i in range(len(unlabelled_files)) if i not in selected_set
    ]

    fieldnames = ["image", "diversity_score", "selected", "selected_rank",
                  "strategy", "cluster_id"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_selected + rows_rest)

    print(f"\n  Full CSV → {output_path}")

    # --- 2D coverage figure ---
    viz_path = Path(config.BASE_DIR) / "saved_images" / f"diversity_{args.strategy}.png"
    print(f"\n  Generating coverage figure...")
    _save_visualization(u_emb, l_emb, selected_idx, cluster_labels,
                        args.strategy, viz_path)

    print(f"\n  Next steps:")
    print(f"  1. Annotate the {budget_used} images listed above.")
    print(f"  2. Add .jpg + .txt to data/yolo_dataset/train/labelled/")
    print(f"  3. Run: python src/scripts/prepare_al_datasets.py \\")
    print(f"             --csv {output_path.name} \\")
    print(f"             --score_col diversity_score --top_k {budget_used}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

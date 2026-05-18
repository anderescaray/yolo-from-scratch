"""
Prepare AL Datasets
===================
Copies the full yolo_dataset twice, excluding the 75 most uncertain / most
certain images from each copy's unlabelled folder.

Output:
    data/yolo_dataset_uncertain/   ← full dataset minus top-K uncertain images
    data/yolo_dataset_certain/     ← full dataset minus bottom-K certain images

The original data/yolo_dataset/ is NEVER modified.

After running this script:
  1. Annotate annotation_batches/uncertain/ → copy .jpg + .txt into
     data/yolo_dataset_uncertain/train/labelled/
  2. Annotate annotation_batches/certain/   → copy .jpg + .txt into
     data/yolo_dataset_certain/train/labelled/
  3. Rename the folder you want to use to yolo_dataset and run finetune.py

Usage
-----
    python src/scripts/prepare_al_datasets.py
    python src/scripts/prepare_al_datasets.py --top_k 75
    python src/scripts/prepare_al_datasets.py --csv active_query.csv --top_k 50
"""

import argparse
import csv
import shutil
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from typing import Set

import core.config as config


def _read_excluded_sets(csv_path: Path, k: int) -> tuple[Set[str], Set[str]]:
    """Return (uncertain_stems, certain_stems) — stems to EXCLUDE from each copy."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["image"], float(row["uncertainty_score"])))
    rows.sort(key=lambda r: r[1], reverse=True)

    if len(rows) < k * 2:
        raise ValueError(
            f"CSV has only {len(rows)} rows — cannot extract {k} uncertain + {k} certain "
            f"without overlap. Reduce --top_k to at most {len(rows) // 2}."
        )

    uncertain = {Path(rows[i][0]).stem for i in range(k)}
    certain   = {Path(rows[-(i + 1)][0]).stem for i in range(k)}
    return uncertain, certain


def _copy_dataset(
    src: Path,
    dst: Path,
    exclude_stems: Set[str],
    unlabelled_subdir: str = "train/unlabelled",
) -> None:
    """
    Copy src tree to dst, skipping files whose stem is in exclude_stems
    only when they are inside unlabelled_subdir.
    All other files and directories are copied verbatim.
    """
    unlabelled_abs = (src / unlabelled_subdir).resolve()

    if dst.exists():
        print(f"  [WARN] {dst} already exists — removing it first.")
        shutil.rmtree(dst)

    copied = skipped = 0

    for src_file in src.rglob("*"):
        if src_file.is_dir():
            continue

        # Decide whether this file is inside the unlabelled folder
        try:
            src_file.resolve().relative_to(unlabelled_abs)
            in_unlabelled = True
        except ValueError:
            in_unlabelled = False

        if in_unlabelled and src_file.stem in exclude_stems:
            skipped += 1
            continue

        dst_file = dst / src_file.relative_to(src)
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        copied += 1

    print(f"    Copied : {copied} files")
    print(f"    Skipped: {skipped} files (excluded from unlabelled/)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare two AL dataset copies.")
    parser.add_argument(
        "--csv", type=str, default=str(Path(config.BASE_DIR) / "active_query.csv"),
        help="Path to active_query.csv (default: <project_root>/active_query.csv).",
    )
    parser.add_argument(
        "--top_k", type=int, default=75,
        help="Number of images to exclude per copy (default: 75).",
    )
    parser.add_argument(
        "--source", type=str, default=str(config.DATASET_DIR),
        help="Source dataset directory (default: DATASET_DIR from config).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    src      = Path(args.source)
    k        = args.top_k

    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        print("  Run first: python src/training/active_sampler.py "
              "--weights checkpoints/finetune_best_map.pth.tar")
        sys.exit(1)

    if not src.exists():
        print(f"[ERROR] Source dataset not found: {src}")
        sys.exit(1)

    print(f"\n{'='*58}")
    print(f"  Prepare AL Datasets")
    print(f"  Source : {src}")
    print(f"  CSV    : {csv_path}")
    print(f"  K      : {k} images excluded per copy")
    print(f"{'='*58}\n")

    uncertain_stems, certain_stems = _read_excluded_sets(csv_path, k)

    dst_uncertain = src.parent / f"{src.name}_uncertain"
    dst_certain   = src.parent / f"{src.name}_certain"

    print(f"  Building yolo_dataset_uncertain  (excluding top-{k} uncertain)...")
    _copy_dataset(src, dst_uncertain, uncertain_stems)
    print(f"  → {dst_uncertain}\n")

    print(f"  Building yolo_dataset_certain    (excluding bottom-{k} certain)...")
    _copy_dataset(src, dst_certain, certain_stems)
    print(f"  → {dst_certain}\n")

    print(f"{'='*58}")
    print(f"  Done. Next steps:")
    print(f"  1. Annotate annotation_batches/uncertain/ and paste")
    print(f"     .jpg + .txt into {dst_uncertain}/train/labelled/")
    print(f"  2. Same for certain/ → {dst_certain}/train/labelled/")
    print(f"  3. Rename the folder to yolo_dataset and run finetune.py")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()

"""
CSV Generator for YOLOv4 pipeline
=================================

Generates CSV files (image, label) needed for DataLoaders.
Supports both project datasets.

Usage:
    python generate_csv.py --dataset generic    # Pretraining
    python generate_csv.py --dataset specific   # Fine-tuning

CSVs are saved inside corresponding dataset folder:
    data/generic_dataset/train.csv, test.csv
    data/yolo_dataset/train.csv, val.csv
"""

import os
import argparse
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def scan_folder(folder_path: str, output_csv: str) -> None:
    """
    Scans folder looking for (image, .txt) pairs and generates CSV.
    Only includes images with corresponding .txt file that is not empty.
    """
    if not os.path.exists(folder_path):
        print(f"  ⚠️  Folder not found, skipping: '{folder_path}'")
        return

    filenames = sorted(os.listdir(folder_path))
    data = []
    skipped_no_label = 0
    skipped_empty    = 0

    for filename in filenames:
        if not (filename.endswith(".jpg") or filename.endswith(".png")):
            continue

        label_file = filename.replace(".jpg", ".txt").replace(".png", ".txt")
        label_path = os.path.join(folder_path, label_file)

        if not os.path.exists(label_path):
            skipped_no_label += 1
            continue

        # Skip empty labels (images without annotations)
        if os.path.getsize(label_path) == 0:
            skipped_empty += 1
            continue

        data.append([filename, label_file])

    if data:
        df = pd.DataFrame(data, columns=["img", "label"])
        df.to_csv(output_csv, index=False)
        print(f"  ✅ {output_csv}  →  {len(data)} pairs  "
              f"(no label: {skipped_no_label}, empty label: {skipped_empty})")
    else:
        print(f"  ⚠️  {folder_path} contains no valid image+label pairs.")


def generate_generic() -> None:
    """Generates CSVs for generic pretraining dataset."""
    print("Generating CSVs for generic dataset...")
    dataset_dir = os.path.join(BASE_DIR, "data", "generic_dataset")

    scan_folder(
        folder_path=os.path.join(dataset_dir, "train"),
        output_csv=os.path.join(dataset_dir, "train.csv"),
    )
    scan_folder(
        folder_path=os.path.join(dataset_dir, "val"),
        output_csv=os.path.join(dataset_dir, "val.csv"),
    )


def generate_specific() -> None:
    """Generates CSVs for the specific fine-tuning dataset."""
    print("Generating CSVs for specific dataset...")
    dataset_dir = os.path.join(BASE_DIR, "data", "yolo_dataset")

    # Train: labeled images only
    scan_folder(
        folder_path=os.path.join(dataset_dir, "train", "labelled"),
        output_csv=os.path.join(dataset_dir, "train.csv"),
    )
    # Validation: val folder
    scan_folder(
        folder_path=os.path.join(dataset_dir, "val"),
        output_csv=os.path.join(dataset_dir, "val.csv"),
    )
    # Final test (optional — skipped if folder does not exist or is empty)
    scan_folder(
        folder_path=os.path.join(dataset_dir, "test"),
        output_csv=os.path.join(dataset_dir, "test.csv"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSV generator for YOLOv4")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["generic", "specific", "all"],
        default="all",
        help="Which dataset to process: 'generic', 'specific' or 'all' (default: all)",
    )
    args = parser.parse_args()

    if args.dataset in ("generic", "all"):
        generate_generic()
    if args.dataset in ("specific", "all"):
        generate_specific()

    print("\nDone.")
"""
filter_to_10classes.py
======================
Filters the 20-class yolo_dataset down to 10 visually distinct classes,
remaps label indices (0-19 → 0-9), and copies everything into a new
yolo_dataset_10c directory.  Original data is never modified.

Output structure:
    data/yolo_dataset_10c/
    ├── train/
    │   ├── labelled/     images + remapped .txt labels
    │   └── unlabelled/   images only (no labels)
    ├── val/              images + remapped .txt labels
    ├── test/             images + remapped .txt labels  (if source exists)
    ├── train.csv
    ├── val.csv
    └── test.csv

Run from the project root:
    python src/scripts/filter_to_10classes.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import core.config as config

# ============================================================
# CONFIGURATION
# Edit KEEP_CLASSES to change which classes survive and their new index.
# Keys must match entries in config.specific_class_labels exactly.
# ============================================================

KEEP_CLASSES: Dict[str, int] = {
    "coca_cola_bottle": 0,
    "coca_cola_can":    1,
    "heineken_can":     2,
    "whole_milk":       3,
    "banana":           4,
    "orange":           5,
    "green_apple":      6,
    "natural_yogurt":   7,
    "ketchup":          8,
    "mayonnaise":       9,
}

# Hardcoded source labels — always the original 20-class list regardless of config state.
# These must match the order in which classes were annotated in the source .txt files.
_ORIGINAL_LABELS: List[str] = [
    "coca_cola_bottle", "coca_cola_can", "orange_fanta_bottle", "heineken_can",
    "whole_milk", "semi_skimmed_milk", "skimmed_milk", "banana", "orange",
    "green_apple", "red_apple", "natural_yogurt", "stracciatella_yogurt",
    "shampoo_hs", "shampoo_hacendado", "ketchup", "mayonnaise",
    "fried_tomato", "york_ham", "turkey_ham",
]

OLD_TO_NEW: Dict[int, int] = {
    _ORIGINAL_LABELS.index(name): new_idx
    for name, new_idx in KEEP_CLASSES.items()
}

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


# ============================================================
# HELPERS
# ============================================================

def class_from_stem(stem: str) -> Optional[str]:
    """
    Extracts the class name from an image filename stem.
    Checks ALL original labels (kept + excluded) sorted longest-first so that
    'orange_fanta_bottle_001' matches 'orange_fanta_bottle' before 'orange'.
    Returns None if the longest match is not a kept class.
    """
    for name in sorted(_ORIGINAL_LABELS, key=len, reverse=True):
        if stem.startswith(name + "_"):
            return name if name in KEEP_CLASSES else None
    return None


def remap_label_file(src_path: Path) -> List[str]:
    """
    Reads a YOLO .txt label file and returns lines with remapped class indices.
    Lines whose class is not in the kept set are silently dropped.
    """
    kept: List[str] = []
    with open(src_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            old_cls = int(float(parts[0]))
            if old_cls in OLD_TO_NEW:
                new_cls = OLD_TO_NEW[old_cls]
                kept.append(f"{new_cls} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
    return kept


# ============================================================
# DIRECTORY PROCESSORS
# ============================================================

def process_labelled(
    src_img_dir: Path,
    src_lbl_dir: Path,
    dst_dir: Path,
) -> List[Tuple[str, str]]:
    """
    Copies images and rewrites label files for the kept classes.

    Returns a list of (image_filename, label_filename) rows for the CSV.
    Skips images whose class is not in KEEP_CLASSES.
    Skips images whose label file becomes empty after filtering.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    csv_rows: List[Tuple[str, str]] = []
    n_copied = n_skip_class = n_skip_empty = 0

    for img_path in sorted(src_img_dir.iterdir()):
        if img_path.suffix.lower() not in VALID_EXT:
            continue

        if class_from_stem(img_path.stem) is None:
            n_skip_class += 1
            continue

        lbl_path = src_lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            n_skip_class += 1
            continue

        lines = remap_label_file(lbl_path)
        if not lines:
            n_skip_empty += 1
            continue

        shutil.copy2(img_path, dst_dir / img_path.name)

        dst_lbl = dst_dir / lbl_path.name
        with open(dst_lbl, "w") as f:
            f.writelines(lines)

        csv_rows.append((img_path.name, lbl_path.name))
        n_copied += 1

    print(f"    Copied: {n_copied}  |  "
          f"Skipped (wrong class): {n_skip_class}  |  "
          f"Skipped (empty label after remap): {n_skip_empty}")
    return csv_rows


def process_unlabelled(src_dir: Path, dst_dir: Path) -> int:
    """Copies unlabelled images for the kept classes (no .txt files)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    n_copied = n_skipped = 0

    for img_path in sorted(src_dir.iterdir()):
        if img_path.suffix.lower() not in VALID_EXT:
            continue
        if class_from_stem(img_path.stem) is not None:
            shutil.copy2(img_path, dst_dir / img_path.name)
            n_copied += 1
        else:
            n_skipped += 1

    print(f"    Copied: {n_copied}  |  Skipped: {n_skipped}")
    return n_copied


def write_csv(path: Path, rows: List[Tuple[str, str]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    src_root = config.BASE_DIR / "data" / "yolo_dataset"
    dst_root = config.BASE_DIR / "data" / "yolo_dataset_10c"

    print(f"\n{'='*60}")
    print("  FILTER 20 → 10 CLASSES")
    print(f"  Source : {src_root}")
    print(f"  Dest   : {dst_root}")
    print(f"  Keeping: {list(KEEP_CLASSES.keys())}")
    print(f"  Old→New: {OLD_TO_NEW}")
    print(f"{'='*60}\n")

    # ── train/labelled ──────────────────────────────────────
    print("train/labelled ...")
    train_rows = process_labelled(
        src_img_dir=src_root / "train" / "labelled",
        src_lbl_dir=src_root / "train" / "labelled",
        dst_dir=dst_root / "train" / "labelled",
    )
    write_csv(dst_root / "train.csv", train_rows)
    print(f"    → train.csv  ({len(train_rows)} images)\n")

    # ── train/unlabelled ────────────────────────────────────
    print("train/unlabelled ...")
    n_unlab = process_unlabelled(
        src_dir=src_root / "train" / "unlabelled",
        dst_dir=dst_root / "train" / "unlabelled",
    )
    print()

    # ── val ─────────────────────────────────────────────────
    print("val/ ...")
    val_rows = process_labelled(
        src_img_dir=src_root / "val",
        src_lbl_dir=src_root / "val",
        dst_dir=dst_root / "val",
    )
    write_csv(dst_root / "val.csv", val_rows)
    print(f"    → val.csv  ({len(val_rows)} images)\n")

    # ── test (optional) ─────────────────────────────────────
    test_src = src_root / "test"
    if test_src.exists():
        print("test/ ...")
        test_rows = process_labelled(
            src_img_dir=test_src,
            src_lbl_dir=test_src,
            dst_dir=dst_root / "test",
        )
        write_csv(dst_root / "test.csv", test_rows)
        print(f"    → test.csv  ({len(test_rows)} images)\n")

    # ── Summary ─────────────────────────────────────────────
    print(f"{'='*60}")
    print("  DONE — paste the following into config.py:")
    print(f"{'='*60}")
    print()
    print('    DATASET_DIR = BASE_DIR / "data" / "yolo_dataset_10c"')
    print()
    print("    specific_class_labels = [")
    for name in KEEP_CLASSES:
        print(f'        "{name}",')
    print("    ]")
    print()
    print("    SPECIFIC_NUM_CLASSES = 10")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

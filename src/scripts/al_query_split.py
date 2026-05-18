"""
AL Query Split
==============
Reads active_query.csv (output of active_sampler.py) and copies the top-K
most uncertain and bottom-K most certain images into separate folders for
manual annotation and the AL comparison experiment.

Output layout:
    <output_dir>/uncertain/   ← top-K images (highest uncertainty score)
    <output_dir>/certain/     ← bottom-K images (lowest uncertainty score)

Usage
-----
    python src/scripts/al_query_split.py
    python src/scripts/al_query_split.py --csv active_query.csv --top_k 75
    python src/scripts/al_query_split.py --top_k 50 --output_dir annotation_batches
"""

import argparse
import csv
import shutil
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from typing import List

import core.config as config


def _read_csv_ranked(csv_path: Path) -> List[str]:
    """Return image filenames sorted descending by uncertainty_score."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["image"], float(row["uncertainty_score"])))
    rows.sort(key=lambda r: r[1], reverse=True)
    return [r[0] for r in rows]


def _copy_images(
    filenames: List[str],
    src_dir: Path,
    dst_dir: Path,
    label: str,
) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing = 0
    for fname in filenames:
        src = src_dir / fname
        if not src.exists():
            print(f"  [WARN] Not found: {src}")
            missing += 1
            continue
        shutil.copy2(src, dst_dir / fname)
        copied += 1
    print(f"  {label}: {copied} images copied to {dst_dir}"
          + (f"  ({missing} missing)" if missing else ""))
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Split AL query CSV into annotation batches.")
    parser.add_argument(
        "--csv", type=str, default=str(Path(config.BASE_DIR) / "active_query.csv"),
        help="Path to active_query.csv (default: <project_root>/active_query.csv).",
    )
    parser.add_argument(
        "--src_dir", type=str, default=str(config.UNLABELLED_IMG_DIR),
        help="Directory containing unlabelled images (default: UNLABELLED_IMG_DIR from config).",
    )
    parser.add_argument(
        "--top_k", type=int, default=75,
        help="Number of images per group (default: 75).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(Path(config.BASE_DIR) / "annotation_batches"),
        help="Root output directory (default: <project_root>/annotation_batches).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    src_dir  = Path(args.src_dir)
    out_dir  = Path(args.output_dir)
    k        = args.top_k

    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        print("  Run active_sampler.py first:")
        print("  python src/training/active_sampler.py --weights checkpoints/finetune_best_map.pth.tar")
        sys.exit(1)

    if not src_dir.exists():
        print(f"[ERROR] Source directory not found: {src_dir}")
        sys.exit(1)

    ranked = _read_csv_ranked(csv_path)
    total  = len(ranked)

    if total < k * 2:
        print(f"[WARN] Only {total} images in CSV — reducing top_k from {k} to {total // 2}")
        k = total // 2

    uncertain_names = ranked[:k]
    certain_names   = ranked[-k:]

    print(f"\n{'='*55}")
    print(f"  AL Query Split")
    print(f"  CSV        : {csv_path}")
    print(f"  Source dir : {src_dir}")
    print(f"  Total imgs : {total}")
    print(f"  K per group: {k}")
    print(f"  Output dir : {out_dir}")
    print(f"{'='*55}\n")

    _copy_images(uncertain_names, src_dir, out_dir / "uncertain", "Uncertain (top-K)")
    _copy_images(certain_names,   src_dir, out_dir / "certain",   "Certain  (bot-K)")

    print(f"\n  Done. Annotate each folder and add labelled images + .txt files")
    print(f"  to data/yolo_dataset/train/labelled/ before re-running finetune.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()

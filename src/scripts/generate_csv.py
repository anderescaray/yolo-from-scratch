"""
Generador de CSVs para el pipeline YOLOv4
==========================================

Genera los archivos CSV (imagen, label) necesarios para los DataLoaders.
Soporta ambos datasets del proyecto.

Uso:
    python generate_csv.py --dataset generic    # Pre-entrenamiento
    python generate_csv.py --dataset specific   # Fine-tuning

Los CSVs se guardan dentro de la carpeta del dataset correspondiente:
    data/generic_dataset/train.csv, test.csv
    data/yolo_dataset/train.csv, val.csv
"""

import os
import argparse
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def scan_folder(folder_path: str, output_csv: str) -> None:
    """
    Escanea una carpeta buscando pares (imagen, .txt) y genera un CSV.
    Solo incluye imágenes que tienen su .txt correspondiente y que no está vacío.
    """
    if not os.path.exists(folder_path):
        print(f"ERROR: La carpeta '{folder_path}' no existe.")
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

        # Saltamos labels vacíos (imágenes sin anotaciones)
        if os.path.getsize(label_path) == 0:
            skipped_empty += 1
            continue

        data.append([filename, label_file])

    if data:
        df = pd.DataFrame(data, columns=["img", "label"])
        df.to_csv(output_csv, index=False)
        print(f"  ✅ {output_csv}  →  {len(data)} pares  "
              f"(sin label: {skipped_no_label}, label vacío: {skipped_empty})")
    else:
        print(f"  ⚠️  {folder_path} no contiene pares imagen+label válidos.")


def generate_generic() -> None:
    """Genera los CSVs del dataset genérico de preentrenamiento."""
    print("Generando CSVs del dataset genérico...")
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
    """Genera los CSVs del dataset específico para fine-tuning."""
    print("Generando CSVs del dataset específico...")
    dataset_dir = os.path.join(BASE_DIR, "data", "yolo_dataset")

    # Train: solo las imágenes etiquetadas
    scan_folder(
        folder_path=os.path.join(dataset_dir, "train", "labelled"),
        output_csv=os.path.join(dataset_dir, "train.csv"),
    )
    # Validación: carpeta val
    scan_folder(
        folder_path=os.path.join(dataset_dir, "val"),
        output_csv=os.path.join(dataset_dir, "val.csv"),
    )
    # Test final
    scan_folder(
        folder_path=os.path.join(dataset_dir, "test"),
        output_csv=os.path.join(dataset_dir, "test.csv"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de CSVs para YOLOv4")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["generic", "specific", "all"],
        default="all",
        help="Qué dataset procesar: 'generic', 'specific' o 'all' (por defecto: all)",
    )
    args = parser.parse_args()

    if args.dataset in ("generic", "all"):
        generate_generic()
    if args.dataset in ("specific", "all"):
        generate_specific()

    print("\nDone.")
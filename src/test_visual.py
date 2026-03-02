import config
import torch
import torch.optim as optim
from model import YOLOv4
from utils import load_checkpoint, get_loaders, plot_couple_examples
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    print("Cargando modelo para visualización...")
    
    # Crear estructura del modelo
    model = YOLOv4(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # Cargar los pesos entrenados
    load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE)
    
    # Datos de test
    _, test_loader, _ = get_loaders(
        train_csv_path=config.DATASET + "/train.csv",
        test_csv_path=config.DATASET + "/test.csv"
    )
    
    # Generar y guardar imágenes
    print("Generando predicciones...")
    # Umbral 0.5 (Confianza 50%)
    plot_couple_examples(model, test_loader, 0.45, config.NMS_IOU_THRESH, config.ANCHORS)
    
    print("\nBusca los archivos 'prediccion_test_X.png' en tu carpeta.")

if __name__ == "__main__":
    main()
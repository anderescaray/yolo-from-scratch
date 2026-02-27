import os
import pandas as pd
import config

def generate_csv(folder, output_name):
    
    path = os.path.join(config.DATASET, folder)
    
    print(f"Buscando en: {path}")

    if not os.path.exists(path):
        print(f"ERROR: La carpeta '{path}' no existe.")
        return

    filenames = os.listdir(path)
    data = []
    
    for filename in filenames:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_file = filename
            label_file = filename.replace(".jpg", ".txt").replace(".png", ".txt")
            
            if os.path.exists(os.path.join(path, label_file)):
                data.append([image_file, label_file])
            else:
                print(f"Aviso: {filename} no tiene .txt")

    if len(data) > 0:
        df = pd.DataFrame(data, columns=["img", "label"])
        output_path = os.path.join(base_dir, "data", output_name)
        df.to_csv(output_path, index=False)
        print(f"✅ ¡Éxito! Generado {output_name} con {len(data)} datos.")
    else:
        print(f"⚠️ La carpeta {folder} estaba vacía o no tenía parejas jpg+txt.")

if __name__ == "__main__":
    generate_csv("train", "train.csv")
    generate_csv("valid", "test.csv")
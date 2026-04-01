"""
YOLOv3 Dataloader
============================

Clase para cargar imágenes y etiquetas en el formato específico que requiere YOLOv3

    1. Cargar imágenes y etiquetas
    2. Usar Data Augmentation
    3. Construir los TARGETS (Matrices de entrenamiento):
       Convierte las cajas [x, y, w, h] en matrices de 3 escalas (13, 26, 52),
       asignando cada objeto a la celda y al Anchor Box más apropiado por IoU

El output que devuelve __getitem__:
    - image: Tensor normalizado de la imagen
    - targets: Tupla con 3 tensores (uno por escala) conteniendo los ground truths
"""
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from utils import iou_width_height as iou
from config import ANCHORS

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size=416,
        S=[13, 26, 52], # Los tamaños de las 3 grids
        C=20, # n de clases
        transform=None,
    ):
        df = pd.read_csv(csv_file) # Leemos el CSV
        self.img_files = df.iloc[:, 0].tolist()
        self.label_files = df.iloc[:, 1].tolist()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform # =config.train_transforms
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # Juntamos las 9 anclas en una lista
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        
        # Ignoramos cajas con advertencias 
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        """
        Carga una imagen y sus etiquetas para un índice dado.
        
        Args:
            index (int): Índice de la lista de datos.
            
        Returns:
            image: Imagen procesada.
            tuple(targets): Tupla de 3 tensores con las etiquetas mapeadas a la rejilla.
        """
        # 1. CARGAR IMAGEN Y ETIQUETA
        label_path = os.path.join(self.label_dir, self.label_files[index])
        
        # Lectura rápida nativa sin np.loadtxt ni pandas
        bboxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    # [class, x, y, w, h] -> [x, y, w, h, class]
                    class_label = int(float(parts[0]))
                    box = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), class_label]
                    bboxes.append(box)
        
        img_path = os.path.join(self.img_dir, self.img_files[index])
        image = np.array(Image.open(img_path).convert("RGB"))

        # 2. DATA AUGMENTATION
        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # 3. PREPARAR LOS OBJETIVOS (TARGETS)
        # Creamos 3 matrices vacías una para cada tamaño 13, 26 y 52
        # Cada celda tendrá 6 valores que serán [prob_obj, x, y, w, h, clase] 
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        
        # 4. ASIGNAR CAJAS A CELDAS Y ANCLAS
        # iteramos sobre cada objeto
        for box in bboxes:
            # box: [x, y, w, h, class]
            # Calculamos cuál de las 9 anclas se parece más a esta caja (IoU)
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors) # box[2:4] es ancho y alto
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # Para asegurar que cada escala tiene máx 1 ancla por objeto

            # Iteramos sobre las anclas ordenadas (de mejor a peor encaje)
            for anchor_idx in anchor_indices:
                # ¿A qué escala pertenece esta ancla? (0=Grande, 1=Mediana, 2=Pequeña)
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                
                S = self.S[scale_idx] # Grid size actual 
                
                # ¿En qué celda i,j cae el centro del objeto?
                i, j = int(S * y), int(S * x) # y=fila, x=columna
                
                # Verificamos que la caja es válida y que esa celda no está cogida ya
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                
                if not anchor_taken and not has_anchor[scale_idx]:
                    #si hay objeto confianza=1
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    
                    # Coordenadas relativas a la celda
                    x_cell, y_cell = S * x - j, S * y - i 
                    width_cell, height_cell = (width * S, height * S) 
                    
                    # Guardamos coordenadas
                    targets[scale_idx][anchor_on_scale, i, j, 1] = x_cell
                    targets[scale_idx][anchor_on_scale, i, j, 2] = y_cell
                    targets[scale_idx][anchor_on_scale, i, j, 3] = width_cell
                    targets[scale_idx][anchor_on_scale, i, j, 4] = height_cell
                    
                    # Guardamos la clase
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    
                    has_anchor[scale_idx] = True

                # Si el ancla no es la mejor pero es decente (>0.5 IoU), la ignoramos
                # (ponemos -1 para decir "no penalices esto, es ambiguo")
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # Ignorar predicción

        return image, tuple(targets)
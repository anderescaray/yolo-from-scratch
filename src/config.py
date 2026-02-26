"""
YOLOv3 Configuración e Hiperparámetros
========================================

Este módulo centraliza todas las constantes, hiperparámetros y configuraciones
del modelo YOLOv3. Este archivo se usa para todo el pipeline de train y val
"""
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os


NUM_WORKERS = 0  
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = True
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_FILE = os.path.join(BASE_DIR, "checkpoints", "checkpoint.pth.tar")

# Directorio donde están los csv 
DATASET = os.path.join(BASE_DIR, "data")
# Directorio donde están las imágenes y los label

IMG_DIR = os.path.join(DATASET, "train")
LABEL_DIR = os.path.join(DATASET, "train")

LEARNING_RATE = 1e-4 
WEIGHT_DECAY = 1e-4   
BATCH_SIZE = 8        
NUM_EPOCHS = 100      

CONF_THRESHOLD = 0.5
NMS_IOU_THRESH = 0.45   # Para limpiar cajas duplicadas con nms
MAP_IOU_THRESH = 0.5    # Umbral para calcular la nota mAP


class_labels = [
    "-", "Apple", "Apple -Green-", "Apple -Red-", "Artichoke", "Asparagus", 
    "Avocado", "Banana", "Beans", "Bell Pepper", "Blackberries", "Blueberries", 
    "Book", "Boxed Food", "Bread", "Broccoli", "Brussel Sprouts", "Butter", 
    "Cabbage", "Canned Fish", "Canned Food", "Cantaloupe", "Carrots", "Cauliflower", 
    "Cereal", "Cerealbox", "Cheese", "Clementine", "Coffee", "Condiment", 
    "Corn", "Creamer", "Cucumber", "Detergent", "Drinks", "Egg", 
    "Eggplant", "Eggs", "Fish", "Galia", "Garlic", "Grains", 
    "Grapes", "Honeydew", "Jar", "Juice", "Kiwi", "Lemon", 
    "Lettuce", "Meat", "Meat -Red-", "Milk", "Mushroom", "Mushrooms", 
    "Nectarine", "Noodles", "Oats", "Orange", "Oranges", "Peach", 
    "Peanut Butter", "Pear", "Pickled Food -Jar-", "Pineapple", "Plum", "Pomegranate", 
    "Popcorn", "Potatoes -Package-", "Raspberries", "Rice", "Salad", "Sauce", 
    "Seasoning -Thin Package-", "Snack", "Soda", "Spinach", "Squash", "Strawberries", 
    "Strawberry", "Tofu", "Tomatoes", "Water", "Watermelon", "Yogurt", "Zucchini"
]

NUM_CLASSES = len(class_labels)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Arquitectura del modelo YOLOv3
# Tupla: (out_channels, kernel_size, stride) -> Convolución
# Lista ["B", num_repeats]: Bloque Residual
# "S": Scale Prediction (que será el output)
# "U": Upsampling 
IMAGE_SIZE = 416
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
CONFIG = [
    # si la entrada es 416 x 416
    (32, 3, 1),                    # salida 416 x 416 x 32
    (64, 3, 2),                    # stride 2 -> reducimos 208 x 208 x 64
    ["B", 1],                      # resblcok
    (128, 3, 2),                   # 104 x 104 x 128
    ["B", 2],                      # x2 resblocks
    (256, 3, 2),                   # 52 x 52 x 256
    ["B", 8],                      # x8 resblocks      aquí guardaremos una copia de este mapa de características en route connections
                                   #                para luego detectar objetos pequeños (aquí la red ya ve cosas pequeñas pero todavía con detalle de posición)
    
    (512, 3, 2),                   # 26 x 26 x 512
    ["B", 8],                      #  x8 resblocks  --> guardar en route connections para luego detectar objetos medianos
    (1024, 3, 2),                  # 13 x 13 x 1024   aquí la red sabe mucho pero no ve la imagen, ve conceptos semánticos
    ["B", 4],       # Backbone Darknet-53 termina aquí
    # PROBAR A CAMBIAR ESTE ULTIMO BLOQUE RESIDUAL POR CPSDarknet53

    # Aquí ya hemos acabado con la primera fase de embudo donde hemos comprimido la imagen
    # falta reconstruirla para saber dónde están los objetos que hemos detectado en la fase embudo 

    (512, 1, 1),             # Reducimos canales a la mitad para ahorrar cómputo
    (1024, 3, 1),              # Convolución final de procesamiento
    "S",            # <--- OUTPUT 1 (Objetos Grandes - Predicción con Grid 13x13)

    (256, 1, 1),            # Preparamos los canales antes de subir hacia el paso para detectar obj medianos
    "U",                    # <--- Upsample 

    (256, 1, 1),                # Procesamos la imagen fusionada (Upsample + Route 2)
    (512, 3, 1),
    "S",                            # <--- OUTPUT 2 (Objetos Medianos - Predicción con Grid 26x26)

    (128, 1, 1),            # Preparamos los canales antes de subir a objs pequeños
    "U",            # <--- Upsample (Agrandar)

    (128, 1, 1),                # Procesamos la imagen fusionada (Upsample + Route 1)
    (256, 3, 1),
    "S",            # <--- OUTPUT 3 (Objetos Pequeños - Grid 52x52)
]

#### DATA AUGMENTATION ####

# ANCHORS
# Estos son los tamaños de caja predefinidos (ancho, alto) calculados clusterizando 
# las bounding boxes con K-Means en el dataset COCO del paper
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],  # Para los objetos grnades (13x13)
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],  # medianos (26x26)
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],  # pequeños (52x52)
]

# Transformaciones para train
# Opción 1 (Imagen entera con padding)
'''train_transforms = A.Compose(
    [
        # Para que sea del tamaño correcto
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT),
        
        # Para cambiar el brillo, contraste, saturación aleatoriamente
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.5),

        # Para mover, escalar y rotar 
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        
        # Para hacer flips
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        
        # Normalizamos valores de pixeles 
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    # Para que ajuste las bounding boxes también igual que las imágenes
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)'''

# Opción 2 (Zoom + Recorte)
SCALE = 1.2 
train_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=int(IMAGE_SIZE * SCALE)),

        # Recortamos un cuadrado de 416x416 en cualquier lugar aleatorio
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        
        # Variaciones de color
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5),
        
        # Rotación ligera por si la cámara está torcida
        A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT),

        # Flips
        A.HorizontalFlip(p=0.5),
        
        # Normalización de valores de pixeles 
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

# Transformaciones para test/val de solo redimensionar
test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)
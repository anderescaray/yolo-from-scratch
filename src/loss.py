"""
YOLOv3 Loss Function
====================

Este módulo calcula el error entre la predicción del modelo y el ground truth
Combina 4 pérdidas diferentes:
    1. Box Loss: Error al predecir la bounding box (x, y) centro y (w, h) tamaño
    2. Object Loss: Error al detectar si hay objeto o no (confianza) con IoU-aware
    3. No Object Loss: Penalización por detectar objetos donde no hay nada (Fondo)
    4. Class Loss: Error en la clase del objeto

Fórmula de la Loss total:
    Loss = λ_box * L_box + λ_obj * L_obj + λ_noobj * L_noobj + λ_class * L_class
"""
import random
import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Para coordenadas (x,y,w,h) y Clases usamos Error Cuadrático o Entropía Cruzada
        self.mse = nn.MSELoss() 
        self.bce = nn.BCEWithLogitsLoss() # Binary Cross Entropy (incluye Sigmoide)
        self.entropy = nn.CrossEntropyLoss()
        
        # Constantes (Lambdas) para equilibrar la importancia de cada pérdida
        self.lambda_class = 1
        self.lambda_noobj = 10  # Castigamos mucho los falsos positivos en el fondo
        self.lambda_obj = 1
        self.lambda_box = 10    # Las coordenadas deben ser muy precisas

    def forward(self, predictions, target, anchors):
        """
        Calcula la pérdida para UNA escala específica.
        predictions: Tensor (N, 3, S, S, 5+C) que sale del modelo
        target: Tensor (N, 3, S, S, 6) que viene del dataset
        anchors: Las 3 anclas de esta escala
        """
        # Identificamos dónde hay objeto y dónde no en el target
        # miramos solo el índice 0 de la última dimensión, la confianza
        # creamos dos matrices con la misma forma (N,3,S,S)
        # obj será true donde de verdad hay un objeto y noobj será true donde de verdad no lo hay (celdas vacías=fondo)
        obj = target[..., 0] == 1 
        noobj = target[..., 0] == 0

    ### noobj loss ###
        # Si no hay objeto, la red debe predecir confianza 0.
        # Solo miramos la posición [..., 0:1] que es la confianza 
        no_object_loss = self.bce((predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]))

    ### obj loss ###
        # Si hay objeto, la red debe predecir confianza 1 (o el IoU real).
        anchors = anchors.reshape(1, 3, 1, 1, 2) # Formateamos anclas para operar
        
        # Pasamos las preds de la red (de las bboxes) a formato normal
        # y cogem solo las dimensiones y las concatenamos
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        
        # Para que la confianza predicha sea como el IoU
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

    ### box loss ###
        # solo donde obj = True
        
        # centro (x,y)
        # La red los predice con cualquier nº y los ponemos para que estén entre 0 y 1
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3]) 
        # Este ya está entre 0 y 1
        target[..., 1:3] = target[..., 1:3] 
        
        # Dimensiones (w,h)
        # La red predice el exponente para estirar el ancla: w = ancla * e^pred
        # Para comparar lo mismo con lo mismo, pasamos el Target al formato de las preds
        # t_w = log(w_real / w_ancla)
        target[..., 3:5] = torch.log(
            # usamos el 1e-16 para evitar ln(0)=-inf y que explote el grad
            (1e-16 + target[..., 3:5] / anchors)
        ) 
        
        # Calculamos el error total de coordenadas
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

    ### class loss ###
        # Predicción de qué objeto es 
        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long())
        )

    ### loss total ###
        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )

    def sigmoid(self, x):
        return torch.sigmoid(x)
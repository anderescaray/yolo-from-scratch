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
        self.lambda_noobj = 5  
        self.lambda_obj = 2
        self.lambda_box = 5   

    def ciou_loss(self, pred_boxes, target_boxes):
        """
        CIoU Loss (Complete IoU) - Mejora de YOLOv4 sobre el MSE de YOLOv3
        
        En lugar de tratar x, y, w, h como 4 errores independientes (MSE),
        CIoU mide directamente la calidad geométrica entre las dos cajas con 3 términos:
        
            1. IoU:      Cuánto se solapan las dos cajas (el objetivo principal)
            2. Distance: Penaliza la distancia entre los centros de las dos cajas
            3. Aspect:   Penaliza que las proporciones (w/h) sean distintas
        
        CIoU = IoU - (distancia²/diagonal²) - α * v
            donde v mide la diferencia de aspecto y α lo pondera según el IoU actual

        Cuanto más se parezcan las cajas, más se acerca CIoU a 1
        La loss es: 1 - CIoU  (así 0 = cajas perfectas, 2 = cajas opuestas)

        Args:
            pred_boxes:   Tensor (N, 4) con [x, y, w, h] ya en escala de la grid
            target_boxes: Tensor (N, 4) con [x, y, w, h] ya en escala de la grid
        Returns:
            loss: Escalar con la media del CIoU Loss
        """
        # Convertimos de midpoints (tx, ty, w, h) a (x1, y1, x2, y2) corners
        # (esquina superior izqda y inferior dcha)
        # Esto nos facilita calcular áreas e intersecciones
        pred_x1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2 # tx - w/2
        pred_y1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2 # ty - h/2
        pred_x2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
        pred_y2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2

        tgt_x1 = target_boxes[..., 0] - target_boxes[..., 2] / 2
        tgt_y1 = target_boxes[..., 1] - target_boxes[..., 3] / 2
        tgt_x2 = target_boxes[..., 0] + target_boxes[..., 2] / 2
        tgt_y2 = target_boxes[..., 1] + target_boxes[..., 3] / 2

        # IoU
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)
        # clamp(0) para que si no hay intersección el área sea 0 y no negativa
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        tgt_area  = (tgt_x2  - tgt_x1) * (tgt_y2  - tgt_y1)
        union_area = pred_area + tgt_area - inter_area + 1e-7

        iou = inter_area / union_area

        # Distancia entre centros / diagonal de la enclosing box
        # La enclosing box es la mínima caja que contiene a las dos
        enclose_x1 = torch.min(pred_x1, tgt_x1)
        enclose_y1 = torch.min(pred_y1, tgt_y1)
        enclose_x2 = torch.max(pred_x2, tgt_x2)
        enclose_y2 = torch.max(pred_y2, tgt_y2)
        # c^2 = diagonal^2 
        c2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + 1e-7

        # d^2 = distancia^2 entre los centros de pred y target
        center_dist2 = (
            (pred_boxes[..., 0] - target_boxes[..., 0]) ** 2 +
            (pred_boxes[..., 1] - target_boxes[..., 1]) ** 2
        )

        # Consistencia de proporción
        # v mide cuánto difieren los arcotangentes del aspecto w/h entre las dos cajas
        # Si tienen la misma proporción v = 0
        v = ((4 / (torch.pi ** 2)) # constante de normalización del paper original para que v esté entre 0 y 1
            * (
                torch.atan(target_boxes[..., 2] / (target_boxes[..., 3] + 1e-7)) -
                torch.atan(pred_boxes[..., 2]  / (pred_boxes[..., 3]  + 1e-7))
            ) ** 2
        )

        # alpha pondera v según el IoU actual: si el IoU ya es alto, el aspecto importa más
        with torch.no_grad(): # 
            # es un coeficiente de ponderación, no una variable que queramos diferenciar. 
            # Si lo incluyéramos en el grafo de gradientes crearíamos dependencias circulares en el cálculo
            # Iou bajo -> alpha pequeño -> aspecto importa poco
            # Iou alto -> alpha grande ->aspecto toma + importancia
            alpha = v / (1 - iou + v + 1e-7)

        # CIoU final
        ciou = iou - (center_dist2 / c2) - alpha * v

        # Para ver si fallaba aqui
        if torch.isnan(1 - ciou).any():
            print("WARNING: NaN en ciou_loss")
            return torch.tensor(0.0, requires_grad=True).to(pred_boxes.device)

        # La loss es 1 - CIoU: cuanto mejor la caja, menor la pérdida
        return (1 - ciou).mean()

    def forward(self, predictions, target, anchors):
        """
        Calcula la pérdida para UNA escala específica.
        predictions: Tensor (N, 3, S, S, 5+C) que sale del modelo
        target: Tensor (N, 3, S, S, 6) que viene del dataset
        anchors: Las 3 anclas de esta escala
        """
        # Para ver si fallaba aqui
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            print("WARNING: NaN/Inf en predictions")
        if torch.isnan(target).any() or torch.isinf(target).any():
            print("WARNING: NaN/Inf en target")
        
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
        
        # Construimos las cajas predichas sin modificar el tensor original de predictions
        # Trabajamos con variables locales para no corromper los datos entre diferentes escalas de la grid
        pred_xy = self.sigmoid(predictions[..., 1:3])           # centro (x,y) entre 0 y 1
        pred_wh = torch.exp(predictions[..., 3:5]) * anchors    # (w,h) en escala de la grid
        pred_boxes = torch.cat([pred_xy, pred_wh], dim=-1)      # (x, y, w, h)

        # Construimos también las cajas del target en el mismo formato que pred_boxes
        # El dataset ya guarda w,h en el formato de la escala de la grid 
        target_boxes = target[..., 1:5] #(x, y, w, h)

        # Calculamos CIoU solo para las celdas donde hay objeto
        box_loss = self.ciou_loss(pred_boxes[obj], target_boxes[obj])

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
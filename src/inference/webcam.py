import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import cv2
import torch
import numpy as np
import core.config as config
from core.model import YOLOv3
from core.utils import cells_to_bboxes, non_max_suppression

def main():
    # 1. Cargar el modelo que acabas de entrenar
    print("Cargando modelo...")
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    
    # Cargar los pesos (Asegúrate de tener el archivo checkpoint.pth.tar)
    checkpoint = torch.load("checkpoint.pth.tar", map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval() # Modo evaluación (MUY IMPORTANTE)

    # 2. Iniciar la webcam (0 suele ser la cámara integrada de tu portátil)
    cap = cv2.VideoCapture(0)
    print("Iniciando webcam... Pulsa 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Preprocesar el frame para YOLO (416x416)
        # Guardamos el tamaño original para luego re-escalar las cajas
        H_orig, W_orig, _ = frame.shape 
        
        # OpenCV usa BGR, PyTorch usa RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Aplicamos las transformaciones de test que tienes en config.py
        transform = config.test_transforms
        transformed = transform(image=image_rgb)
        x = transformed["image"].unsqueeze(0).to(config.DEVICE) # Añadir dimensión de Batch (1, 3, 416, 416)

        # 4. Predicción
        with torch.no_grad():
            out = model(x)
            bboxes = []
            for i in range(3):
                batch_size, A, S, _, _ = out[i].shape
                anchor = torch.tensor(config.ANCHORS[i]).to(config.DEVICE) * S
                boxes_scale_i = cells_to_bboxes(
                    out[i], anchor, S=S, is_preds=True
                )
                for box in boxes_scale_i[0]: # Solo el primer elemento del batch
                    bboxes.append(box)

        # 5. Non-Max Suppression para limpiar cajas repetidas
        nms_boxes = non_max_suppression(
            bboxes, 
            iou_threshold=config.NMS_IOU_THRESH, 
            threshold=0.5, # Umbral de confianza (cámbialo si quieres ver más o menos cajas)
            box_format="midpoint"
        )

        # 6. Dibujar las cajas en el frame original
        for box in nms_boxes:
            class_pred = int(box[0])
            prob = box[1]
            # Las cajas vienen en formato midpoint (x, y, w, h) relativos de 0 a 1
            box_x, box_y, box_w, box_h = box[2], box[3], box[4], box[5]

            # Convertir a coordenadas de píxeles reales de la webcam
            x1 = int((box_x - box_w / 2) * W_orig)
            y1 = int((box_y - box_h / 2) * H_orig)
            x2 = int((box_x + box_w / 2) * W_orig)
            y2 = int((box_y + box_h / 2) * H_orig)

            # Dibujar rectángulo y texto
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            texto = f"{config.class_labels[class_pred]}: {prob:.2f}"
            cv2.putText(frame, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 7. Mostrar el frame
        cv2.imshow("YOLOv3 - Webcam TFG", frame)

        # Salir si se pulsa la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
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
    # 1. Load the trained model
    print("Loading model...")
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)

    # Load weights (make sure you have checkpoint.pth.tar file)
    checkpoint = torch.load("checkpoint.pth.tar", map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval() # Evaluation mode (VERY IMPORTANT)

    # 2. Start webcam (0 is usually your laptop's integrated camera)
    cap = cv2.VideoCapture(0)
    print("Starting webcam... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Preprocess frame for YOLO (416x416)
        # Save original size to re-scale boxes later
        H_orig, W_orig, _ = frame.shape

        # OpenCV uses BGR, PyTorch uses RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply test transformations from config.py
        transform = config.test_transforms
        transformed = transform(image=image_rgb)
        x = transformed["image"].unsqueeze(0).to(config.DEVICE) # Add batch dimension (1, 3, 416, 416)

        # 4. Prediction
        with torch.no_grad():
            out = model(x)
            bboxes = []
            for i in range(3):
                batch_size, A, S, _, _ = out[i].shape
                anchor = torch.tensor(config.ANCHORS[i]).to(config.DEVICE) * S
                boxes_scale_i = cells_to_bboxes(
                    out[i], anchor, S=S, is_preds=True
                )
                for box in boxes_scale_i[0]: # Only first element of batch
                    bboxes.append(box)

        # 5. Non-Max Suppression to clean duplicate boxes
        nms_boxes = non_max_suppression(
            bboxes,
            iou_threshold=config.NMS_IOU_THRESH,
            threshold=0.5, # Confidence threshold (change if you want more/less boxes)
            box_format="midpoint"
        )

        # 6. Draw boxes on original frame
        for box in nms_boxes:
            class_pred = int(box[0])
            prob = box[1]
            # Boxes come in midpoint format (x, y, w, h) relative 0 to 1
            box_x, box_y, box_w, box_h = box[2], box[3], box[4], box[5]

            # Convert to actual pixel coordinates of webcam
            x1 = int((box_x - box_w / 2) * W_orig)
            y1 = int((box_y - box_h / 2) * H_orig)
            x2 = int((box_x + box_w / 2) * W_orig)
            y2 = int((box_y + box_h / 2) * H_orig)

            # Draw rectangle and text
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
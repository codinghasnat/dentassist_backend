from ultralytics import YOLO
from PIL import Image
import torch
import os

# Load YOLOv8 model
model = YOLO('models/tooth_classification/yolo_detector/yolo_detector.pt')  # adjust to your model path

def detect_and_crop(image_path):
    results = model(image_path, conf=0.005)[0]  # get the first result
    img = Image.open(image_path).convert("RGB")
    img_width, img_height = img.size
    crops = []
    boxes_info = []  # Store box coordinates

    # Get top 40 boxes sorted by confidence
    boxes = results.boxes
    confs = boxes.conf.cpu().numpy()
    sorted_indices = confs.argsort()[::-1][:40]

    for idx in sorted_indices:
        x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy()
        w = x2 - x1
        h = y2 - y1

        expansion_ratio = 0.1
        x1_exp = max(0, x1 - expansion_ratio * w)
        y1_exp = max(0, y1 - expansion_ratio * h)
        x2_exp = min(img_width, x2 + expansion_ratio * w)
        y2_exp = min(img_height, y2 + expansion_ratio * h)

        # Store original (non-expanded) box coordinates
        boxes_info.append({
            'x1': int(x1),
            'y1': int(y1),
            'x2': int(x2),
            'y2': int(y2)
        })

        # Crop and append
        crop = img.crop((int(x1_exp), int(y1_exp), int(x2_exp), int(y2_exp)))
        crops.append(crop)

    return crops, boxes_info

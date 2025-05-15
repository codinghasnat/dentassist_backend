import numpy as np

# === Extract coords from dict box ===
def box_to_xyxy(box):
    return [box["x1"], box["y1"], box["x2"], box["y2"]]

# === IOU Calculation ===
def compute_iou(boxA, boxB):
    boxA = box_to_xyxy(boxA)
    boxB = box_to_xyxy(boxB)

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# === IOU Filtering ===
def bounding_box_filter_iou(boxes, crops, iou_threshold=0.1):
    keep_boxes = []
    keep_crops = []

    # Sort by box area descending
    sorted_items = sorted(zip(boxes, crops), key=lambda x: 
        (x[0]["x2"] - x[0]["x1"]) * (x[0]["y2"] - x[0]["y1"]), reverse=True)

    while sorted_items:
        current_box, current_crop = sorted_items.pop(0)
        keep_boxes.append(current_box)
        keep_crops.append(current_crop)

        sorted_items = [
            (box, crop) for box, crop in sorted_items
            if compute_iou(current_box, box) < iou_threshold
        ]

    return keep_boxes, keep_crops

# === Midpoint Filtering ===
def get_center(box):
    cx = (box["x1"] + box["x2"]) / 2
    cy = (box["y1"] + box["y2"]) / 2
    return cx, cy

def bounding_box_filter_center(boxes, crops, min_dist=100):
    # The bigger min_dist is, the more filtering occurs and the less duplicates we have.
    keep_boxes = []
    keep_crops = []
    centers = []

    for box, crop in zip(boxes, crops):
        cx, cy = get_center(box)
        too_close = any(np.hypot(cx - px, cy - py) < min_dist for px, py in centers)
        if not too_close:
            keep_boxes.append(box)
            keep_crops.append(crop)
            centers.append((cx, cy))

    return keep_boxes, keep_crops

# === Hybrid Filtering ===
def hybrid_filter(boxes, crops, iou_threshold=0.5, min_dist=100):
    print("[DEBUG] Running hybrid filtering...")
    boxes_iou, crops_iou = bounding_box_filter_iou(boxes, crops, iou_threshold)
    boxes_final, crops_final = bounding_box_filter_center(boxes_iou, crops_iou, min_dist)
    return boxes_final, crops_final

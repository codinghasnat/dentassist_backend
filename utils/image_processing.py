import os
from PIL import Image, ImageDraw

def get_disease_color(disease):
    """Return RGB color tuple based on disease classification."""
    color_map = {
        "Healthy": (76, 175, 80),  # green
        "Caries": (244, 67, 54),  # rose/red
        "Deeper Caries": (211, 47, 47),  # deeper rose/red
        "Periapical Lesion": (255, 152, 0),  # orange
        "Impacted": (255, 193, 7),  # amber
        "Fractured": (3, 169, 244),  # sky blue
        "BDC/BDR": (156, 39, 176),  # fuchsia/purple
        "Unknown": (158, 158, 158),  # neutral gray
    }
    return color_map.get(disease, (255, 0, 0))  # default to red if not found

def draw_boxes(image_path, boxes, output_path, width=10):
    """Draw bounding boxes on an image and save it."""
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    for box in boxes:
        color = get_disease_color(box.get('disease', 'Unknown'))
        draw.rectangle(
            [(box['x1'], box['y1']), (box['x2'], box['y2'])],
            outline=color,
            width=width
        )
    
    img.save(output_path)
    return img

def save_annotated_images(image_path, boxes, output_dir="annotated_xrays_single_tooth"):
    """Save individual annotated images for each tooth and a final combined image."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual tooth annotations
    for idx, box in enumerate(boxes):
        output_path = os.path.join(output_dir, f"tooth_{idx}.jpg")
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        color = get_disease_color(box.get('disease', 'Unknown'))
        draw.rectangle(
            [(box['x1'], box['y1']), (box['x2'], box['y2'])],
            outline=color,
            width=10
        )
        img.save(output_path)
    
    # Save final image with all boxes
    final_output = os.path.join(output_dir, "final_yolo_output.jpg")
    draw_boxes(image_path, boxes, final_output)

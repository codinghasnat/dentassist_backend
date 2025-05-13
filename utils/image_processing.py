import os
from PIL import Image, ImageDraw

def draw_boxes(image_path, boxes, output_path, color=(255, 0, 0), width=3):
    """Draw bounding boxes on an image and save it."""
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    for box in boxes:
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
        draw.rectangle(
            [(box['x1'], box['y1']), (box['x2'], box['y2'])],
            outline=(255, 0, 0),
            width=3
        )
        img.save(output_path)
    
    # Save final image with all boxes
    final_output = os.path.join(output_dir, "final_yolo_output.jpg")
    draw_boxes(image_path, boxes, final_output)

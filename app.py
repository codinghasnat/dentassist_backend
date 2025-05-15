from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import os
import base64
from PIL import Image
from io import BytesIO
import shutil

from detector import detect_and_crop
from binary_classifier import binary_filter_teeth
from disease_classifier import classify_teeth
from utils.image_processing import save_annotated_images
from bb_filering import bounding_box_filter_iou, bounding_box_filter_center, hybrid_filter

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
STORAGE_FOLDER = 'stored_xrays'  # Permanent storage for original xrays
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STORAGE_FOLDER, exist_ok=True)

def encode_image_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

@app.route('/disease_classify', methods=['POST'])
def disease_classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    filename = datetime.now().strftime("%Y%m%d%H%M%S") + '_' + image.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image.save(filepath)

    # Single tooth classification
    prediction = classify_teeth(filepath)  # Assuming classify_teeth can handle single images
    
    # Convert the image to base64
    image_base64 = encode_image_base64(Image.open(filepath).convert("RGB"))

    # Return single result
    return jsonify({
        "originalImage": image_base64,
        "annotatedImage": "",  # you can replace this later
        "detectedTeeth": [{
            "id": 0,
            "image": image_base64,
            "disease": prediction["disease"],
            "confidence": round(prediction["confidence"], 4)
        }]
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        image = request.files['image']
        filename = datetime.now().strftime("%Y%m%d%H%M%S") + '_' + image.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        storage_path = os.path.join(STORAGE_FOLDER, filename)
        image.save(filepath)

        # Step 1: YOLO detection
        initial_crops, boxes_info = detect_and_crop(filepath)

        # Step 2: Binary classifier filtering
        filtered = binary_filter_teeth(initial_crops)
        filtered_indices = [idx for idx, _ in filtered]
        filtered_crops = [crop for _, crop in filtered]
        filtered_boxes = [boxes_info[i] for i in filtered_indices]

        # Step 3: Bounding box filtering: 3 different options
        # Option A: IOU only
        filtered_boxes, filtered_crops = bounding_box_filter_iou(filtered_boxes, filtered_crops)

        # Option B: Midpoint only
        # filtered_boxes, filtered_crops = bounding_box_filter_center(filtered_boxes, filtered_crops)

        # Option C: Hybrid
        # filtered_boxes, filtered_crops = hybrid_filter(filtered_boxes, filtered_crops)
        # print("[DEBUG] After bounding box filtering")

        
        # Save annotated images of NEW filtered boxes
        save_annotated_images(filepath, filtered_boxes)
        # print("[DEBUG] After save_annotated_images")
        
        # Save cropped teeth of NEW filtered teeth
        save_cropped_teeth(filtered_crops)
        # print("[DEBUG] After save_cropped_teeth")

        # Step 4: Multiclass disease classification
        predictions = classify_teeth(filtered_crops)
        # print("[DEBUG] After classify_teeth")

        # Step 5: Attach base64-encoded images
        results = []
        for idx, (crop, pred) in enumerate(zip(filtered_crops, predictions)):
            # Get the individual annotated image for this tooth
            individual_annotated_path = os.path.join('annotated_xrays_single_tooth', f'tooth_{idx}.jpg')
            individual_annotated = Image.open(individual_annotated_path)
            
            results.append({
                "id": idx,
                "image": encode_image_base64(crop),
                "annotatedImage": encode_image_base64(individual_annotated),
                "disease": pred["disease"],
                "confidence": round(pred["confidence"], 4)
            })

        # Get the final annotated image
        final_annotated = Image.open(os.path.join('annotated_xrays_single_tooth', 'final_yolo_output.jpg'))
        
        # Move original file to permanent storage
        shutil.move(filepath, storage_path)
        
        return jsonify({
            "originalImage": encode_image_base64(Image.open(storage_path).convert("RGB")),
            "annotatedImage": encode_image_base64(final_annotated),
            "detectedTeeth": results
        })

    except Exception as e:
        # Clean up upload file only if error occurs and file exists
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

def save_cropped_teeth(crops, output_dir="saved_crops"):
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    for idx, crop in enumerate(crops):
        filename = f"tooth_{idx}.jpg"
        path = os.path.join(output_dir, filename)
        crop.save(path)
        saved_paths.append(path)
    return saved_paths


if __name__ == '__main__':
    app.run(debug=True, port=5000)

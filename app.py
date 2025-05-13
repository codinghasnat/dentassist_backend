from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import os
import base64
from PIL import Image
from io import BytesIO

from detector import detect_and_crop
from binary_classifier import filter_teeth
from disease_classifier import classify_teeth

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def encode_image_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    filename = datetime.now().strftime("%Y%m%d%H%M%S") + '_' + image.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image.save(filepath)

    # Step 1: YOLO detection
    initial_crops = detect_and_crop(filepath)

    # Step 2: Binary classifier filtering
    filtered_crops = filter_teeth(initial_crops)
    save_cropped_teeth(filtered_crops)


    # Step 3: Multiclass disease classification
    predictions = classify_teeth(filtered_crops)

    # Step 4: Attach base64-encoded images
    results = []
    for idx, (crop, pred) in enumerate(zip(filtered_crops, predictions)):
      image_base64 = encode_image_base64(crop)

      results.append({
          "id": idx,
          "image": image_base64,
          "disease": pred["disease"],
          "confidence": round(pred["confidence"], 4)
      })
    # Optional: base64 encode original image
    original_image_base64 = encode_image_base64(Image.open(filepath).convert("RGB"))


    print(f"[INFO] YOLO detected: {len(initial_crops)} potential teeth")
    print(f"[INFO] Binary filter kept: {len(filtered_crops)} teeth")

    return jsonify({
        "originalImage": encode_image_base64(Image.open(filepath).convert("RGB")),
        "annotatedImage": "",  # you can replace this later
        "detectedTeeth": results
        })

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

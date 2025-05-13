import torch
from torchvision import transforms, models
from torch import nn
from PIL import Image

# === Class Names ===
class_names = [
    "Caries",
    "Periapical Lesion",
    "Impacted",
    "Fractured",
    "BDC/BDR",
    "Healthy",
    "Deeper Caries"
]

# === Model Definition ===
model = models.resnet18(weights=None)  # or "IMAGENET1K_V1" if you want pretrained
model.fc = nn.Linear(model.fc.in_features, len(class_names))  # 7 outputs

# === Load state dict ===
state_dict = torch.load("models/disease_classification/multiclass_classifier.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# === Image transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Run classification on list of cropped PIL Images ===
def classify_teeth(input_data):
    predictions = []
    
    # Handle both single image and list of images
    if isinstance(input_data, str):  # If input is a filepath
        image = Image.open(input_data).convert('RGB')
        images = [image]
    elif isinstance(input_data, Image.Image):  # If input is a single PIL Image
        images = [input_data]
    elif isinstance(input_data, list):  # If input is a list of images
        images = input_data
    else:
        raise TypeError(f"Unexpected input type: {type(input_data)}")

    for idx, img in enumerate(images):
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
            pred_class = output.argmax(dim=1).item()
            confidence = torch.softmax(output, dim=1)[0][pred_class].item()

            predictions.append({
                "id": idx,
                "disease": class_names[pred_class],
                "confidence": round(confidence, 4),
            })

    # If single image was passed, return just the first prediction
    if isinstance(input_data, (str, Image.Image)):
        return predictions[0]
    return predictions

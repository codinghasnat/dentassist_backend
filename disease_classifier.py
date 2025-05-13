import torch
from torchvision import transforms, models
from torch import nn

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
def classify_teeth(crops):
    predictions = []
    for idx, crop in enumerate(crops):
        img_tensor = transform(crop).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
            pred_class = output.argmax(dim=1).item()
            confidence = torch.softmax(output, dim=1)[0][pred_class].item()

            predictions.append({
                "id": idx,
                "image": "",  # You can populate this later if saving or encoding images
                "disease": class_names[pred_class],
                "confidence": round(confidence, 4),
            })

    return predictions

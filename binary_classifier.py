import torch
from torchvision import transforms, models
from torch import nn

# === Reconstruct model ===
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)

# === Load weights ===
state_dict = torch.load('models/tooth_classification/binary_classifier/binary_tooth.pt', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

# === Image transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Inference ===
def filter_teeth(crops):
    filtered = []
    for crop in crops:
        img_tensor = transform(crop).unsqueeze(0)  # [1, 3, 224, 224]
        with torch.no_grad():
            logit = model(img_tensor)
            prob = torch.sigmoid(logit).item()
            print(f"[DEBUG] Crop {crop}: prob = {prob:.4f}")
            if prob >= 0.15:  # Classify as "tooth"
                filtered.append(crop)

    return filtered

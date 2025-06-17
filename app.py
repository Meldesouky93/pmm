
import gradio as gr
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load("pneumonia_resnet18.pth", map_location=device))
model.eval()
model.to(device)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def predict(images):
    results = []
    for file in images:
        image = Image.open(file).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.sigmoid(output).item()
            label = "PNEUMONIA" if prob > 0.5 else "NORMAL"
            results.append(f"{file.name}: {label} (Confidence: {prob:.2f})")
    return "\n".join(results)

gr.Interface(
    fn=predict,
    inputs=gr.File(file_types=["image"], file_count="multiple", label="Upload Chest X-ray Images"),
    outputs=gr.Textbox(label="Results"),
    title="Batch Pneumonia Detector",
    description="Upload multiple chest X-ray images to detect Pneumonia"
).launch()


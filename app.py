import os
import time
import base64
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
from PIL import Image
from io import BytesIO
from typing import List
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

# Define class names
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load model
def load_model():
    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load('fashion_mnist_resnet18.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Define transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- ROUTES ---

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict_single(file: UploadFile = File(...)):
    start_time = time.time()
    img = Image.open(file.file).convert('L')
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()

    processing_time = (time.time() - start_time) * 1000

    return {
        "prediction": {
            "class_id": predicted_class,
            "class_name": CLASS_NAMES[predicted_class],
            "confidence": round(confidence, 2),
            "processing_time_ms": round(processing_time, 2)
        },
        "status": "success"
    }

class BatchRequest(BaseModel):
    images: List[str]  # base64-encoded images

@app.post("/predict/batch")
async def predict_batch(payload: BatchRequest):
    start_time = time.time()
    predictions = []

    for image_data in payload.images:
        img_start_time = time.time()
        image_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(image_bytes)).convert('L')
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()

        predictions.append({
            "class_id": predicted_class,
            "class_name": CLASS_NAMES[predicted_class],
            "confidence": round(confidence, 2),
            "processing_time_ms": round((time.time() - img_start_time) * 1000, 2)
        })

    return {
        "predictions": predictions,
        "status": "success",
        "total_processing_time_ms": round((time.time() - start_time) * 1000, 2)
    }

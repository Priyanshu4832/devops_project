from fastapi import FastAPI, UploadFile, File
from transformers import pipeline
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import logging

# Setup Logging 
logging.basicConfig(filename="api_logs.log", level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="Multi-Modal AI DevOps Pipeline", version="1.0")

# Load Models 
logging.info("Loading AI Models...")

# Text Model: Hugging Face Sentiment Analysis
sentiment_model = pipeline("sentiment-analysis")

# Image Model: ResNet18 (Standard)

image_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
image_model.eval() 

# Image transformations needed for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

logging.info("Models loaded successfully.")

# 3. Create Endpoints

@app.get("/")
def home():
    return {"message": "AI DevOps Pipeline is Active. Go to /docs to test the API."}

@app.post("/sentiment")
def analyze_sentiment(text: str):
    logging.info(f"Sentiment request received for text: '{text}'")
    result = sentiment_model(text)
    return {"text": text, "prediction": result, "model_version": "hf-distilbert-v1"}

@app.post("/classify-image")
def classify_image(file: UploadFile = File(...)):
    logging.info(f"Image classification request received: {file.filename}")
    
    # Read and transform the image
    image = Image.open(file.file).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)

    # Run the model
    with torch.no_grad():
        outputs = image_model(img_tensor)
    
    # Get the predicted class ID (To keep it simple, we just return the raw class ID number)
    _, predicted = torch.max(outputs, 1)
    
    return {"filename": file.filename, "class_id": predicted.item(), "model_version": "resnet18-v1"}
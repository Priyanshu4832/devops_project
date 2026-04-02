from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from transformers import pipeline
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import logging

# Setup Logging 
logging.basicConfig(filename="api_logs.log", level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="Multi-Modal AI DevOps Pipeline", version="2.1")


logging.info("Loading AI Models...")
sentiment_model = pipeline("sentiment-analysis")


weights = models.ResNet18_Weights.DEFAULT
image_model = models.resnet18(weights=weights)
image_model.eval() 

# Extract the 1,000 human-readable class names from PyTorch
categories = weights.meta["categories"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
logging.info("Models loaded successfully.")

# --- API ENDPOINTS ---

@app.post("/sentiment")
def analyze_sentiment(text: str):
    logging.info(f"Sentiment request received for text: '{text}'")
    result = sentiment_model(text)
    return {"text": text, "prediction": result, "model_version": "hf-distilbert-v1"}

@app.post("/classify-image")
def classify_image(file: UploadFile = File(...)):
    logging.info(f"Image classification request received: {file.filename}")
    image = Image.open(file.file).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = image_model(img_tensor)
    
    _, predicted = torch.max(outputs, 1)
    class_id = predicted.item()
    
    # LOOKUP THE HUMAN READABLE NAME
    class_name = categories[class_id]
    
    return {
        "filename": file.filename, 
        "class_id": class_id, 
        "class_name": class_name, 
        "model_version": "resnet18-v1"
    }

# THE FRONTEND DASHBOARD 

@app.get("/", response_class=HTMLResponse)
def home():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>AI DevOps Pipeline</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #eef2f7, #f8fafc);
                color: #2d3436;
            }

            .container {
                background: #ffffff;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 6px 20px rgba(0,0,0,0.06);
            }

            h1 {
                color: #1f2d3d;
                text-align: center;
                border-bottom: 2px solid #e6ecf1;
                padding-bottom: 10px;
            }

            .card {
                border: 1px solid #e6ecf1;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 10px;
                background: #f9fbfd;
            }

            button {
                background: linear-gradient(135deg, #3498db, #5dade2);
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 6px;
                cursor: pointer;
                font-weight: bold;
                transition: all 0.2s ease;
                margin-top: 10px;
            }

            button:hover {
                background: linear-gradient(135deg, #2c80b4, #4a9fd6);
                transform: translateY(-1px);
            }

            input[type="text"],
            input[type="file"] {
                width: 100%;
                padding: 10px;
                margin: 10px 0;
                box-sizing: border-box;
                border: 1px solid #dcdfe6;
                border-radius: 6px;
                background: #ffffff;
            }

            pre {
                background: #1e293b;
                color: #e2e8f0;
                padding: 15px;
                border-radius: 6px;
                overflow-x: auto;
                font-size: 14px;
            }

            .badge {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 12px;
                background: #2ecc71;
                color: white;
                font-size: 12px;
                margin-bottom: 10px;
            }

            .highlight {
                color: #f39c12;
                font-weight: bold;
                font-size: 16px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Live AI Deployment Dashboard</h1>
            
            <div class="card">
                <span class="badge">Text Model</span>
                <h3>Sentiment Analysis</h3>
                <input type="text" id="textInput" placeholder="Enter a sentence to analyze...">
                <button onclick="analyzeSentiment()">Analyze Sentiment</button>
                <pre id="sentimentResult">Waiting for input...</pre>
            </div>

            <div class="card">
                <span class="badge">Vision Model</span>
                <h3>Image Classification</h3>
                <input type="file" id="imageInput" accept="image/*">
                <button onclick="classifyImage()">Upload & Classify</button>
                <pre id="imageResult">Waiting for input...</pre>
            </div>
        </div>

        <script>
            async function analyzeSentiment() {
                const text = document.getElementById('textInput').value;
                if(!text) return alert("Please enter text.");
                document.getElementById('sentimentResult').innerText = "Processing...";
                try {
                    const res = await fetch(`/sentiment?text=${encodeURIComponent(text)}`, { method: 'POST' });
                    document.getElementById('sentimentResult').innerText = JSON.stringify(await res.json(), null, 2);
                } catch(e) { document.getElementById('sentimentResult').innerText = "Error connecting to API"; }
            }
            async function classifyImage() {
                const fileInput = document.getElementById('imageInput');
                if(!fileInput.files[0]) return alert("Please select an image first.");
                document.getElementById('imageResult').innerHTML = "Processing...";
                const formData = new FormData();
                formData.append("file", fileInput.files[0]);
                try {
                    const res = await fetch('/classify-image', { method: 'POST', body: formData });
                    const data = await res.json();
                    
                    // Format the output to clearly highlight the English name
                    let displayHtml = `<span class="highlight">DETECTED: ${data.class_name.toUpperCase()}</span>\\n\\n`;
                    displayHtml += JSON.stringify(data, null, 2);
                    
                    document.getElementById('imageResult').innerHTML = displayHtml;
                } catch(e) { document.getElementById('imageResult').innerText = "Error connecting to API"; }
            }
        </script>
    </body>
    </html>
    """
    return html_content
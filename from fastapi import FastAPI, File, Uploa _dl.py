from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

# Load model
model = load_model("fashion_classifier.keras")

# Categories (same order as your training)
categories = ['Tops', 'Dresses', 'Footwear', 'Handbags', 'Jackets', 'Bottoms']

app = FastAPI(
    title="Fashion Image Classifier API",
    description="Predicts fashion product categories using a deep learning model.",
    version="1.0"
)

# OPTIONAL: Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image_bytes):
    """Convert uploaded image to model input tensor"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image bytes
    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)

    # Predict
    preds = model.predict(img_array)
    class_idx = int(np.argmax(preds))
    confidence = float(preds[0][class_idx])
    label = categories[class_idx]

    return {
        "filename": file.filename,
        "predicted_class": label,
        "confidence": confidence
    }

# Run using: uvicorn main:app --reload
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

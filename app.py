from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Load your model
model = load_model(r"C:\Users\diyau\OneDrive\Documents\dl_project\fashion_classifier.keras")

categories = ['Tops', 'Dresses', 'Handbags', 'Jackets']

# Flask app
app = Flask(__name__)

def prepare_image(img_bytes):
    """Preprocess the image to feed into the model"""
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    img_array = prepare_image(file.read())
    
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    class_label = categories[class_idx]
    confidence = float(preds[0][class_idx])
    
    return jsonify({"class": class_label, "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True)
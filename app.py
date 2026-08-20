import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from disease_advisor import get_advice

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin requests

# 1. Load Trained MobileNetV2 Model
MODEL_PATH = 'rose_model.h5'

if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: '{MODEL_PATH}' not found! Please run 'python train.py' first.")
else:
    print(f"✓ Loading AI Model from '{MODEL_PATH}'...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✓ Model loaded successfully!")

# 2. Class Names (must match alphabetical order from training directory)
CLASS_NAMES = ['blackspot', 'healthy', 'mildew', 'rust']

def preprocess_image(image: Image.Image):
    """
    Preprocess image to match MobileNetV2 input requirements (224x224, normalized 0-1).
    """
    image = image.convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
    return img_array

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "Online",
        "service": "Rose Leaf Disease Detection & Smart Advisor API",
        "supported_classes": CLASS_NAMES
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided with key "image"'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400

    try:
        # Open and preprocess image
        image = Image.open(file.stream)
        processed_img = preprocess_image(image)

        # Run inference
        predictions = model.predict(processed_img)[0]
        max_idx = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[max_idx]
        confidence = float(predictions[max_idx]) * 100

        # Retrieve smart agronomy advice
        advice = get_advice(predicted_class)

        response = {
            "disease": predicted_class,
            "confidence": f"{confidence:.2f}%",
            "status": advice["status"],
            "description": advice["description"],
            "fertilizer_recommendation": advice["fertilizer_recommendation"],
            "preventive_measures": advice["preventive_measures"]
        }
        
        print(f"✓ Prediction: {predicted_class} ({confidence:.2f}%)")
        return jsonify(response), 200

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run on 0.0.0.0:5000 so the Android Emulator and physical devices can access it
    app.run(host='0.0.0.0', port=5000, debug=True)

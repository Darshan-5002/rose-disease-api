import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from disease_advisor import get_advice

app = Flask(__name__)
CORS(app)

# 1. Load Trained Model
MODEL_PATH = 'rose_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ['blackspot', 'healthy', 'mildew', 'rust']

def is_valid_plant_image(image: Image.Image) -> bool:
    """
    Checks if the uploaded image actually contains plant/leaf foliage colors.
    Rejects text banners, screenshots, graphics, and random non-plant objects.
    """
    # Convert image to HSV
    img_hsv = image.convert('HSV')
    hsv_array = np.array(img_hsv)
    
    h = hsv_array[:, :, 0]  # Hue (0-255 in PIL)
    s = hsv_array[:, :, 1]  # Saturation
    v = hsv_array[:, :, 2]  # Value / Brightness

    # Plant foliage hue ranges in PIL HSV (Green, Yellow-Green, Brownish-green)
    # Green/Yellow-Green hue roughly maps to 25 to 110 (out of 255) with decent saturation
    plant_mask = ((h >= 20) & (h <= 115) & (s >= 30) & (v >= 30))
    plant_pixel_ratio = np.sum(plant_mask) / (hsv_array.shape[0] * hsv_array.shape[1])

    # If less than 12% of the image pixels have plant-like colors, reject it
    return plant_pixel_ratio >= 0.12

def preprocess_image(image: Image.Image):
    image = image.convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "Online",
        "service": "Rose Leaf Disease Detection & Smart Advisor API"
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        image = Image.open(file.stream)

        # Step 1: Check if the image is actually a plant/leaf
        if not is_valid_plant_image(image):
            return jsonify({
                "disease": "invalid",
                "confidence": "0.00%",
                "status": "Invalid / Non-Plant Image",
                "description": "The uploaded image does not appear to contain a rose leaf or plant foliage.",
                "fertilizer_recommendation": "N/A - Please scan an actual rose leaf.",
                "preventive_measures": [
                    "Take a clear, focused photo of a rose leaf.",
                    "Avoid screenshots, documents, text, or non-plant objects.",
                    "Ensure adequate natural lighting."
                ]
            }), 200

        # Step 2: Run CNN Inference
        processed_img = preprocess_image(image)
        predictions = model.predict(processed_img)[0]
        max_idx = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[max_idx]
        confidence = float(predictions[max_idx]) * 100

        # Step 3: Confidence threshold check (must be at least 80% confident)
        if confidence < 80.0:
            return jsonify({
                "disease": "invalid",
                "confidence": f"{confidence:.2f}%",
                "status": "Uncertain / Unclear Leaf",
                "description": "The model could not identify the disease with sufficient confidence.",
                "fertilizer_recommendation": "N/A - Please retake the photo with better focus.",
                "preventive_measures": [
                    "Ensure the leaf fills most of the camera viewfinder.",
                    "Avoid blurry or extreme close-up angles."
                ]
            }), 200

        # Step 4: Return Valid Disease & Treatment
        advice = get_advice(predicted_class)
        return jsonify({
            "disease": predicted_class,
            "confidence": f"{confidence:.2f}%",
            "status": advice["status"],
            "description": advice["description"],
            "fertilizer_recommendation": advice["fertilizer_recommendation"],
            "preventive_measures": advice["preventive_measures"]
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

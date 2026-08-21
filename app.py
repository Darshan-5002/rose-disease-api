import os
import io
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from disease_advisor import get_advice
from leaf_segmentation import segment_and_crop_leaf

app = Flask(__name__)
CORS(app)

# 1. Load Trained MobileNetV2 Model
MODEL_PATH = 'rose_model.h5'
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: '{MODEL_PATH}' not found! Please run 'python train.py' first.")
else:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✓ AI Model loaded successfully!")

CLASS_NAMES = ['blackspot', 'healthy', 'mildew', 'rust']

def check_image_quality(image: Image.Image):
    open_cv_image = np.array(image.convert('RGB'))
    img_bgr = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 20.0:
        return False, "Image is too blurry. Please hold camera steady."

    mean_brightness = np.mean(gray)
    if mean_brightness < 30:
        return False, "Image is too dark. Please take photo in good lighting."
    if mean_brightness > 235:
        return False, "Image is overexposed/too bright."

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    plant_mask = ((hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 115) & (hsv[:, :, 1] >= 25) & (hsv[:, :, 2] >= 25))
    plant_ratio = np.sum(plant_mask) / (hsv.shape[0] * hsv.shape[1])

    if plant_ratio < 0.10:
        return False, "No plant or rose foliage detected in image."

    return True, "Quality OK"

def preprocess_image(image: Image.Image):
    image = image.convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def pil_to_base64(image: Image.Image):
    """Converts PIL image to base64 string for Android display"""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "Online",
        "service": "Adaptive Rose Disease Detection & Smart Advisor API"
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        raw_image = Image.open(file.stream)

        # Stage 0: Quality Check
        is_quality_good, quality_msg = check_image_quality(raw_image)
        if not is_quality_good:
            return jsonify({
                "disease": "invalid",
                "confidence": "0.00%",
                "status": "Image Quality Warning",
                "severity": "N/A",
                "segmented_image_base64": None,
                "description": quality_msg,
                "fertilizer_recommendation": "N/A - Please recapture with better lighting and focus.",
                "preventive_measures": [
                    "Ensure adequate natural daylight.",
                    "Keep the camera steady to avoid blur.",
                    "Position the rose leaf in the center of the frame."
                ]
            }), 200

        # Stage 1: Leaf Segmentation & Severity
        segmented_leaf, severity_pct, severity_level = segment_and_crop_leaf(raw_image)
        segmented_base64 = pil_to_base64(segmented_leaf)

        # Stage 2: CNN Inference
        processed_img = preprocess_image(segmented_leaf)
        predictions = model.predict(processed_img)[0]
        max_idx = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[max_idx]
        confidence = float(predictions[max_idx]) * 100

        # Confidence Guard
        if confidence < 70.0:
            return jsonify({
                "disease": "invalid",
                "confidence": f"{confidence:.2f}%",
                "status": "Uncertain / Unclear Leaf",
                "severity": "Unknown",
                "segmented_image_base64": segmented_base64,
                "description": "The AI model is uncertain about this leaf image.",
                "fertilizer_recommendation": "N/A - Please retake the photo with the leaf clearly in focus.",
                "preventive_measures": [
                    "Ensure the rose leaf fills most of the frame.",
                    "Avoid capturing background clutter, multiple leaves, or fingers."
                ]
            }), 200

        if predicted_class.lower() == "healthy":
            severity_str = "0% (Healthy Foliage)"
        else:
            severity_str = f"{severity_level} [~{severity_pct:.1f}% lesion area]"

        advice = get_advice(predicted_class)

        return jsonify({
            "disease": predicted_class,
            "confidence": f"{confidence:.2f}%",
            "status": advice["status"],
            "severity": severity_str,
            "segmented_image_base64": segmented_base64,
            "description": advice["description"],
            "fertilizer_recommendation": advice["fertilizer_recommendation"],
            "preventive_measures": advice["preventive_measures"]
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

import os
import cv2
import base64
import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from disease_advisor import get_advice
from leaf_segmentation import segment_and_crop_leaf
from history_db import init_db, analyze_progression_and_save, get_all_scans

app = Flask(__name__)
CORS(app)

# Initialize SQLite Database Table
init_db()

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

    # Blur Detection
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 20.0:
        return False, "Image is too blurry. Please hold camera steady."

    # Brightness Check
    mean_brightness = np.mean(gray)
    if mean_brightness < 30:
        return False, "Image is too dark. Please take photo in good lighting."
    if mean_brightness > 235:
        return False, "Image is overexposed/too bright."

    # Plant Foliage Check in HSV
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

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "Online",
        "service": "Adaptive Rose Disease Detection & Progression Tracker API",
        "features": [
            "Image Quality Assessment",
            "Automatic Leaf Segmentation",
            "Two-Stage AI Inference",
            "Disease Severity Estimation",
            "Temporal Progression & Recovery Tracker (SQLite)"
        ]
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Optional plant ID (defaults to 'rose_bush_1' if not provided)
    plant_id = request.form.get('plant_id', 'rose_bush_1')

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
                "progression_trend": "N/A",
                "description": quality_msg,
                "fertilizer_recommendation": "N/A - Please recapture with better lighting and focus.",
                "preventive_measures": [
                    "Ensure adequate natural daylight.",
                    "Keep the camera steady to avoid blur.",
                    "Position the rose leaf in the center of the frame."
                ]
            }), 200

        # Stage 1: Leaf Segmentation for CNN
        _, cnn_leaf, _, _ = segment_and_crop_leaf(raw_image)

        # Stage 2: MobileNetV2 CNN Inference
        processed_img = preprocess_image(cnn_leaf)
        predictions = model.predict(processed_img)[0]
        max_idx = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[max_idx]
        confidence = float(predictions[max_idx]) * 100

        # Confidence Fallback
        if confidence < 70.0:
            return jsonify({
                "disease": "invalid",
                "confidence": f"{confidence:.2f}%",
                "status": "Uncertain / Unclear Leaf",
                "severity": "Unknown",
                "progression_trend": "N/A",
                "description": "The AI model is uncertain about this leaf image.",
                "fertilizer_recommendation": "N/A - Please retake the photo with the leaf clearly in focus.",
                "preventive_measures": [
                    "Ensure the rose leaf fills most of the frame.",
                    "Avoid capturing background clutter, multiple leaves, or fingers."
                ]
            }), 200

        # Stage 2.5: Disease-Specific Visual Lesion Segmentation
        visual_leaf, _, severity_pct, severity_level = segment_and_crop_leaf(raw_image, disease_hint=predicted_class)

        # Severity Formatter
        if predicted_class.lower() == "healthy":
            severity_str = "0% (Healthy Foliage)"
            severity_num = 0.0
            advice = get_advice("healthy", "Healthy")
        else:
            severity_str = f"{severity_level} [~{severity_pct:.1f}% lesion area]"
            severity_num = severity_pct
            advice = get_advice(predicted_class, severity_level)

        # Stage 3: Temporal Disease Progression Tracking (SQLite)
        trend, trend_feedback = analyze_progression_and_save(
            plant_id=plant_id,
            current_disease=predicted_class,
            confidence_val=confidence,
            current_severity_pct=severity_num,
            severity_level=severity_level
        )

        # Combine measures with progression feedback and climate guidance
        measures = [f"📈 Treatment Response: {trend_feedback}"]
        measures.extend(advice.get("preventive_measures", []))
        if "climate_advisory" in advice:
            measures.append("🌦️ Climate Guidance: " + advice["climate_advisory"])

        # Encode visual segmented/highlighted leaf image as base64 for the app
        seg_buffer = io.BytesIO()
        visual_leaf.save(seg_buffer, format='JPEG', quality=85)
        seg_base64 = base64.b64encode(seg_buffer.getvalue()).decode('utf-8')

        response = {
            "disease": predicted_class,
            "confidence": f"{confidence:.2f}%",
            "status": advice["status"],
            "severity": severity_str,
            "progression_trend": trend,
            "description": advice["description"],
            "fertilizer_recommendation": advice["fertilizer_recommendation"],
            "preventive_measures": measures,
            "segmented_image": seg_base64
        }

        print(f"✓ Diagnosis: {predicted_class} ({confidence:.2f}%) | Trend: {trend} | Severity: {severity_str}")
        return jsonify(response), 200

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    plant_id = request.args.get('plant_id', 'rose_bush_1')
    records = get_all_scans(plant_id)
    return jsonify(records), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

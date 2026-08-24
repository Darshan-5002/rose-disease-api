# 🌹 Rose Leaf Disease Detection & Smart Advisor API

An AI-powered REST API that detects rose leaf diseases using a **Convolutional Neural Network (CNN)** based on **MobileNetV2** transfer learning architecture. The system classifies leaf images into 4 categories and provides intelligent fertilizer recommendations and preventive treatment measures.

---

## 🧠 AI Model

| Attribute | Details |
| :--- | :--- |
| **Architecture** | MobileNetV2 (Transfer Learning) |
| **Framework** | TensorFlow / Keras |
| **Input Size** | 224 × 224 × 3 (RGB) |
| **Training Dataset** | 7,203 rose leaf images |
| **Validation Accuracy** | 96-99.9% (1,441 test images) |
| **Precision / Recall / F1** | 1.0000 |
| **Model Format** | Keras HDF5 (`.h5`) |

---

## 🌿 Supported Disease Classes

| Class | Description |
| :--- | :--- |
| `blackspot` | Black Spot (*Diplocarpon rosae*) |
| `healthy` | Healthy rose leaf |
| `mildew` | Powdery Mildew (*Podosphaera pannosa*) |
| `rust` | Rose Rust (*Phragmidium tuberculatum*) |

---

## 🚀 API Endpoints

### `GET /`
Health check endpoint.

**Response:**
```json
{
  "status": "Online",
  "service": "Rose Leaf Disease Detection & Smart Advisor API",
  "supported_classes": ["blackspot", "healthy", "mildew", "rust"]
}
```

### `POST /predict`
Upload a rose leaf image for AI diagnosis.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `image` (file)

**Response:**
```json
{
  "disease": "blackspot",
  "confidence": "98.75%",
  "status": "Infected: Black Spot (Diplocarpon rosae)",
  "description": "A serious fungal disease causing circular black/dark-brown spots with fringed margins, leading to premature leaf drop.",
  "fertilizer_recommendation": "Apply Potassium-rich fertilizer (SOP or wood ash) to strengthen cell walls; avoid excessive nitrogen.",
  "preventive_measures": [
    "Immediately pick and dispose of infected leaves (do not use in compost).",
    "Spray with Neem Oil, Mancozeb, or Copper-based organic fungicide early in the morning.",
    "Avoid evening and overhead watering so leaves do not remain wet overnight."
  ]
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Deep Learning** | TensorFlow, Keras, MobileNetV2 |
| **Backend API** | Python, Flask, Flask-CORS |
| **Deployment** | Render (Gunicorn WSGI Server) |
| **Frontend** | Android (Java), Retrofit, Glide |

---

## 📁 Project Structure

```
rose-disease-api/
├── app.py                  # Flask REST API server
├── disease_advisor.py      # Smart fertilizer & treatment knowledge base
├── rose_model.h5           # Trained MobileNetV2 CNN model
├── requirements.txt        # Python dependencies
├── Procfile                # Render deployment configuration
├── .python-version         # Python version specification
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation
```

---

## ⚡ Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/Darshan-5002/rose-disease-api.git
cd rose-disease-api

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Flask server
python app.py
```

The API will be available at `http://127.0.0.1:5000/`

---

## 🧪 Testing with Postman

1. Open **Postman** and create a new `POST` request.
2. URL: `http://127.0.0.1:5000/predict`
3. Body Tab → Select **form-data**
4. Key: `image` (set type to **File**)
5. Value: Select any rose leaf image from your device.
6. Click **Send** and view the JSON response with diagnosis and treatment advice.

---

## 📱 Android App Integration

The API is consumed by a companion Android app built with **Java**, **Retrofit**, and **Glide**. The app provides:

- 📷 **Live Camera Capture** — Take photos of rose leaves in real-time
- 📁 **Gallery Selection** — Pick existing leaf photos from device storage
- 🔍 **Real-Time AI Diagnosis** — Instant disease classification with confidence score
- 🌱 **Smart Fertilizer Recommendations** — Targeted fertilizer advice per disease
- 🛡️ **Preventive Action Plans** — Step-by-step treatment and prevention measures

---

## 📊 Model Performance

### Training Accuracy & Loss
The model was trained using **MobileNetV2 Transfer Learning** with data augmentation on 5,762 training images and validated on 1,441 unseen test images.

- **Validation Accuracy:** 100%
- **Validation Loss:** < 0.008
- **Epochs to Convergence:** 7 (with EarlyStopping)

### Confusion Matrix Results

| | Predicted: blackspot | Predicted: healthy | Predicted: mildew | Predicted: rust |
|---|:---:|:---:|:---:|:---:|
| **Actual: blackspot** | 258 | 0 | 0 | 0 |
| **Actual: healthy** | 0 | 493 | 0 | 0 |
| **Actual: mildew** | 0 | 0 | 200 | 0 |
| **Actual: rust** | 0 | 0 | 0 | 490 |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| blackspot | 1.0000 | 1.0000 | 1.0000 | 258 |
| healthy | 1.0000 | 1.0000 | 1.0000 | 493 |
| mildew | 1.0000 | 1.0000 | 1.0000 | 200 |
| rust | 1.0000 | 1.0000 | 1.0000 | 490 |
| **Overall Accuracy** | | | **1.0000** | **1,441** |

---

## 🌐 Live Deployment

The API is deployed on **Render** and accessible at:

```
https://rose-disease-api.onrender.com/
```

> **Note:** The free tier instance may sleep after 15 minutes of inactivity. The first request after sleep takes ~30-60 seconds to wake up.

---

## 📄 License

This project is developed as part of a **B.E. Final Year Project** under **Visvesvaraya Technological University (VTU)**.

**Project Title:** Rose Leaf Disease Detection and Smart Advisor using Convolutional Neural Network (CNN)

---

**Developed by Darshan** | 2026

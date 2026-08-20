# Agronomy and Treatment Knowledge Base
ADVISORY_DATABASE = {
    "healthy": {
        "status": "Healthy Plant",
        "description": "Your rose leaf looks vibrant and free from visible fungal or bacterial infections.",
        "fertilizer_recommendation": "Maintain balanced N-P-K (10-10-10 or 20-20-20) or feed organic compost/cow manure once a month.",
        "preventive_measures": [
            "Always water at the base/root of the plant, not on foliage.",
            "Ensure the rose receives at least 6 to 8 hours of direct daily sunlight.",
            "Prune inner, crowded branches to promote healthy air circulation."
        ]
    },
    "blackspot": {
        "status": "Infected: Black Spot (Diplocarpon rosae)",
        "description": "A serious fungal disease causing circular black/dark-brown spots with fringed margins, leading to premature leaf drop.",
        "fertilizer_recommendation": "Apply Potassium-rich fertilizer (SOP or wood ash) to strengthen cell walls; avoid excessive nitrogen.",
        "preventive_measures": [
            "Immediately pick and dispose of infected leaves (do not use in compost).",
            "Spray with Neem Oil, Mancozeb, or Copper-based organic fungicide early in the morning.",
            "Avoid evening and overhead watering so leaves do not remain wet overnight."
        ]
    },
    "mildew": {
        "status": "Infected: Powdery Mildew (Podosphaera pannosa)",
        "description": "Fungal disease characterized by a white-to-gray talcum-like powdery coating on leaves, shoots, and young flower buds.",
        "fertilizer_recommendation": "Use slow-release balanced organic fertilizer. Reduce high-nitrogen fertilizers which promote susceptible lush growth.",
        "preventive_measures": [
            "Spray a solution of baking soda (1 teaspoon per liter of water with a few drops of mild liquid soap).",
            "Apply wettable sulfur spray or bio-fungicide during early disease onset.",
            "Ensure good sunlight and spacing between plants to lower humidity around foliage."
        ]
    },
    "rust": {
        "status": "Infected: Rose Rust (Phragmidium tuberculatum)",
        "description": "A fungal infection causing bright orange, yellow, or rusty pustules on the lower leaf surface, causing foliage distortion.",
        "fertilizer_recommendation": "Apply Phosphorus and Potassium fertilizer (such as bone meal + potash) to stimulate root vigor and disease resistance.",
        "preventive_measures": [
            "Prune off heavily infected stems and leaves and discard them carefully.",
            "Apply fungicides containing Myclobutanil, Mancozeb, or Sulfur every 7 to 10 days.",
            "Clear away all fallen autumn leaves around the base to eliminate overwintering fungal spores."
        ]
    }
}

def get_advice(disease_name):
    """
    Returns agronomy advice for a given disease class.
    """
    key = disease_name.lower().strip()
    return ADVISORY_DATABASE.get(key, {
        "status": f"Detected: {disease_name.capitalize()}",
        "description": "Disease details are not available in the database.",
        "fertilizer_recommendation": "Consult a local agricultural extension officer.",
        "preventive_measures": [
            "Isolate the plant to prevent spreading.",
            "Ensure clean growing conditions and proper watering."
        ]
    })

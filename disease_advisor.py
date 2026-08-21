# Adaptive Agronomy Knowledge Base: Disease + Severity Level

ADVISORY_DATABASE = {
    "healthy": {
        "status": "Healthy Rose Plant",
        "description": "Foliage exhibits vigorous chlorophyll levels with no detectable fungal, bacterial, or necrotic lesions.",
        "fertilizer_recommendation": "Apply balanced N-P-K (10-10-10 or 20-20-20) or compost tea once every 4 weeks to sustain steady growth and blooming.",
        "preventive_measures": [
            "Water directly at the root zone in the morning; keep leaves dry.",
            "Ensure 6–8 hours of daily direct sunlight.",
            "Maintain 2–3 feet spacing between bushes for natural airflow."
        ],
        "climate_advisory": "Ideal conditions. Continue routine morning watering and maintain mulch layer."
    },

    "blackspot": {
        "Mild": {
            "status": "Infected: Black Spot (Mild Stage)",
            "description": "Early stage fungal infection with small, isolated black spots (<15% leaf surface area).",
            "fertilizer_recommendation": "Feed Potassium Sulphate (SOP) or organic wood ash to harden outer cell walls and slow spore entry. Avoid excess quick-release nitrogen.",
            "preventive_measures": [
                "Pluck off the few spotted leaves immediately and discard in trash (do not compost).",
                "Spray cold-pressed Neem Oil (5ml/liter with a drop of liquid soap) every 7 days.",
                "Disinfect pruning shears with 70% rubbing alcohol between cuts."
            ],
            "climate_advisory": "High humidity accelerates spores. Switch to drip irrigation and avoid overhead sprinklers."
        },
        "Moderate": {
            "status": "Infected: Black Spot (Moderate Stage)",
            "description": "Active fungal propagation with spreading yellow-haloed lesions across 15–40% of foliage.",
            "fertilizer_recommendation": "Apply Potassium-rich foliar feed + Kelp extract to reduce plant stress. Halt all nitrogen fertilizer until infection is controlled.",
            "preventive_measures": [
                "Prune all moderately infected leaves and thin out overcrowded center branches.",
                "Apply Copper-based or Sulfur fungicide spray early in the morning before heat rises.",
                "Collect and burn all fallen leaves from the soil underneath the bush."
            ],
            "climate_advisory": "Spores travel in splashing raindrops. Apply 2 inches of pine bark mulch around base to prevent ground splashback."
        },
        "Severe": {
            "status": "Infected: Black Spot (Severe Outbreak)",
            "description": "Severe defoliation hazard with extensive necrotic black patches spanning >40% of leaf area.",
            "fertilizer_recommendation": "Emergency potassium & micronutrient drench (Iron + Zinc + Silica). Strictly zero nitrogen.",
            "preventive_measures": [
                "Aggressively prune and destroy all heavily infected stems down to healthy green wood.",
                "Apply systemic chemical fungicide (Mancozeb 75% WP or Chlorothalonil) every 7–10 days for 3 cycles.",
                "Isolate the plant immediately if potted to prevent nursery-wide infection."
            ],
            "climate_advisory": "Critical risk. Keep plant in full sun and ensure rapid air movement to dry foliage."
        }
    },

    "mildew": {
        "Mild": {
            "status": "Infected: Powdery Mildew (Mild Stage)",
            "description": "Initial powdery white spore patches (<15% leaf area) on upper leaf surface and young shoots.",
            "fertilizer_recommendation": "Use slow-release organic fertilizer. Avoid high-nitrogen fertilizers which create soft, vulnerable tissue.",
            "preventive_measures": [
                "Spray with Potassium Bicarbonate or Baking Soda solution (1 tsp baking soda + 1/2 tsp dish soap per liter of water).",
                "Wipe young infected leaves gently with a damp cotton cloth to remove superficial spores.",
                "Ensure plant receives strong morning sunlight."
            ],
            "climate_advisory": "Powdery mildew thrives in warm, dry days with humid nights. Prune neighboring foliage for ventilation."
        },
        "Moderate": {
            "status": "Infected: Powdery Mildew (Moderate Stage)",
            "description": "Thick talcum-like fungal carpet covering 15–40% of foliage, causing curling of leaves.",
            "fertilizer_recommendation": "Feed soluble Potassium Silicate to strengthen epidermis layer against mycelial penetration.",
            "preventive_measures": [
                "Prune curled and distorted leaf clusters.",
                "Spray wettable sulfur (0.2%) or horticultural mineral oil in the late evening (avoid hot sun).",
                "Avoid late evening watering."
            ],
            "climate_advisory": "Do not crowd pots or plant beds. Space plants at least 3 feet apart."
        },
        "Severe": {
            "status": "Infected: Powdery Mildew (Severe Outbreak)",
            "description": "Choking white fungal coating spanning >40% of foliage, stunting shoot and bud development.",
            "fertilizer_recommendation": "Halt all feeding until new healthy shoots emerge. Apply root biostimulants.",
            "preventive_measures": [
                "Hard prune heavily colonized stems and young flower buds.",
                "Apply systemic bio-fungicide (Myclobutanil or Azoxystrobin) for deep tissue cure.",
                "Sterilize the surrounding topsoil with sulfur powder."
            ],
            "climate_advisory": "Ensure maximum ventilation and relocate potted plants to high-sunlight locations."
        }
    },

    "rust": {
        "Mild": {
            "status": "Infected: Rose Rust (Mild Stage)",
            "description": "Small, scattered orange/yellow pustules (<15% leaf area) on lower leaf surface.",
            "fertilizer_recommendation": "Boost with Phosphorus & Potassium (bone meal + potash) to stimulate root vigor and disease resistance.",
            "preventive_measures": [
                "Manually remove leaves showing orange powder underneath.",
                "Apply Sulfur-based organic fungicide spray weekly.",
                "Keep the base of the bush clear of debris."
            ],
            "climate_advisory": "Rust spores require continuous moisture for 2–4 hours to germinate. Keep foliage dry."
        },
        "Moderate": {
            "status": "Infected: Rose Rust (Moderate Stage)",
            "description": "Spreading rusty pustules across 15–40% of leaf undersides, causing upper leaf chlorosis.",
            "fertilizer_recommendation": "Foliar feed Potassium Phosphite to trigger natural systemic acquired resistance (SAR).",
            "preventive_measures": [
                "Prune out infected branches showing stem lesions.",
                "Spray Mancozeb (2g/liter) or Copper Oxychloride every 10 days.",
                "Clean and bag all fallen leaves immediately."
            ],
            "climate_advisory": "Cool, damp, overcast weather accelerates rust. Maximize sun exposure."
        },
        "Severe": {
            "status": "Infected: Rose Rust (Severe Outbreak)",
            "description": "Extensive rust pustules with severe leaf yellowing and early defoliation (>40% leaf surface area).",
            "fertilizer_recommendation": "Soil drench with potassium and humic acid to recover root vitality.",
            "preventive_measures": [
                "Cut back infected stems by one-third down to healthy unaffected wood.",
                "Apply systemic fungicide (Tebuconazole or Propiconazole) every 7 days for 2–3 applications.",
                "Isolate surrounding rose bushes and monitor closely."
            ],
            "climate_advisory": "High spore contamination. Rake away 1 inch of topsoil mulch and replace with fresh mulch."
        }
    }
}

def get_advice(disease_name: str, severity_level_str: str = "Mild"):
    """
    Returns personalized agronomy advice adapted to both Disease and Severity Level.
    """
    key = disease_name.lower().strip()
    
    # Extract simple tier: "Mild", "Moderate", "Severe"
    tier = "Mild"
    if "moderate" in severity_level_str.lower():
        tier = "Moderate"
    elif "severe" in severity_level_str.lower():
        tier = "Severe"

    if key == "healthy":
        return ADVISORY_DATABASE["healthy"]

    disease_dict = ADVISORY_DATABASE.get(key)
    if disease_dict:
        advice = disease_dict.get(tier, disease_dict["Mild"])
        return advice

    return {
        "status": f"Detected: {disease_name.capitalize()}",
        "description": "Disease details not found in database.",
        "fertilizer_recommendation": "Consult local agricultural extension officer.",
        "preventive_measures": ["Isolate plant and inspect closely."],
        "climate_advisory": "Maintain clean growing conditions."
    }

import cv2
import numpy as np
from PIL import Image

def segment_and_crop_leaf(pil_image: Image.Image):
    """
    Takes a raw PIL image, segments the rose leaf by removing background noise,
    and returns:
      1. Segmented leaf (PIL Image with black background)
      2. Leaf bounding box cropped image
      3. Total leaf pixel area
      4. Infected lesion pixel area & severity percentage
    """
    # 1. Convert PIL to OpenCV BGR format
    open_cv_image = np.array(pil_image.convert('RGB'))
    img_bgr = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

    # 2. Convert to HSV for robust plant segmentation
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 3. Create mask for plant material (greens, yellows, necrotic browns)
    # Lower: yellowish-green, Upper: dark forest green/brownish tones
    lower_plant = np.array([15, 30, 20])
    upper_plant = np.array([100, 255, 255])
    plant_mask = cv2.inRange(hsv, lower_plant, upper_plant)

    # 4. Morphological operations (remove salt & pepper noise, close holes)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, kernel)
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN, kernel)

    # 5. Find contours and select the largest contour (the primary leaf)
    contours, _ = cv2.findContours(plant_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback if no clear contour found
        return pil_image, 0.0, "Mild"

    largest_contour = max(contours, key=cv2.contourArea)
    leaf_area = cv2.contourArea(largest_contour)

    # Create a refined mask for the main leaf only
    clean_leaf_mask = np.zeros_like(plant_mask)
    cv2.drawContours(clean_leaf_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # 6. Apply mask to isolate the leaf
    segmented_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=clean_leaf_mask)

    # 7. Disease Lesion / Severity Calculation
    # Instead of counting everything non-green as diseased (which over-counts),
    # we specifically detect disease color signatures within the leaf area.

    total_leaf_pixels = cv2.countNonZero(clean_leaf_mask)

    # --- Detect specific disease markers ---
    # Black Spot: very dark regions on the leaf (low value in HSV)
    blackspot_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 70]))
    blackspot_on_leaf = cv2.bitwise_and(blackspot_mask, clean_leaf_mask)

    # Powdery Mildew: white/grayish powdery patches (low saturation, high value)
    mildew_mask = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 50, 255]))
    mildew_on_leaf = cv2.bitwise_and(mildew_mask, clean_leaf_mask)

    # Rust: orange/yellow-brown pustules
    rust_mask = cv2.inRange(hsv, np.array([5, 80, 80]), np.array([25, 255, 255]))
    rust_on_leaf = cv2.bitwise_and(rust_mask, clean_leaf_mask)

    # Yellowing / Chlorosis: yellow areas indicating stress
    yellow_mask = cv2.inRange(hsv, np.array([20, 60, 100]), np.array([35, 255, 255]))
    yellow_on_leaf = cv2.bitwise_and(yellow_mask, clean_leaf_mask)

    # Combine all disease masks (avoid double-counting with bitwise OR)
    combined_disease = cv2.bitwise_or(blackspot_on_leaf, mildew_on_leaf)
    combined_disease = cv2.bitwise_or(combined_disease, rust_on_leaf)
    combined_disease = cv2.bitwise_or(combined_disease, yellow_on_leaf)

    diseased_pixels = cv2.countNonZero(combined_disease)

    severity_percentage = 0.0
    if total_leaf_pixels > 0:
        severity_percentage = round((diseased_pixels / total_leaf_pixels) * 100.0, 1)

    # Categorize severity
    if severity_percentage < 15.0:
        severity_level = "Mild"
    elif severity_percentage < 40.0:
        severity_level = "Moderate"
    else:
        severity_level = "Severe"

    # Convert back to PIL image for CNN input
    segmented_rgb = cv2.cvtColor(segmented_bgr, cv2.COLOR_BGR2RGB)
    segmented_pil = Image.fromarray(segmented_rgb)

    return segmented_pil, severity_percentage, severity_level

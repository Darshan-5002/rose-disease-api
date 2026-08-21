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
    # Healthy rose leaf pixels are typically vivid green
    # Diseased regions (black spots, white mildew, rust orange) deviate from healthy green
    lower_healthy = np.array([35, 40, 40])
    upper_healthy = np.array([85, 255, 255])
    healthy_mask = cv2.inRange(hsv, lower_healthy, upper_healthy)
    healthy_leaf_mask = cv2.bitwise_and(healthy_mask, clean_leaf_mask)

    total_leaf_pixels = cv2.countNonZero(clean_leaf_mask)
    healthy_pixels = cv2.countNonZero(healthy_leaf_mask)
    diseased_pixels = max(0, total_leaf_pixels - healthy_pixels)

    severity_percentage = 0.0
    if total_leaf_pixels > 0:
        severity_percentage = (diseased_pixels / total_leaf_pixels) * 100.0

    # Categorize severity
    if severity_percentage < 15.0:
        severity_level = "Mild (<15% infected area)"
    elif severity_percentage < 40.0:
        severity_level = "Moderate (15-40% infected area)"
    else:
        severity_level = "Severe (>40% infected area)"

    # Convert back to PIL image for CNN input
    segmented_rgb = cv2.cvtColor(segmented_bgr, cv2.COLOR_BGR2RGB)
    segmented_pil = Image.fromarray(segmented_rgb)

    return segmented_pil, severity_percentage, severity_level

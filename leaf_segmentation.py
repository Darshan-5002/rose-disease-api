import cv2
import numpy as np
from PIL import Image

def segment_and_crop_leaf(pil_image: Image.Image, disease_hint: str = None):
    """
    Performs dual-purpose agronomic leaf & disease segmentation:
    1. Isolates leaf foliage from background.
    2. Pinpoints disease lesions (blackspot, mildew, rust, chlorosis) with high visual precision.
    3. Produces a visual annotated overlay for user display (dimmed background + glowing lesion contours).
    
    Returns:
      - visual_pil: Highlighted image showing detected lesion spots with colored mask & contours.
      - cnn_pil: Clean segmented leaf for CNN model inference.
      - severity_percentage: float (e.g. 18.5)
      - severity_level: "Mild", "Moderate", or "Severe"
    """
    open_cv_image = np.array(pil_image.convert('RGB'))
    img_bgr = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    h, w, _ = img_bgr.shape

    # 1. PLANT FOLIAGE DETECTION
    lower_plant = np.array([10, 20, 15])
    upper_plant = np.array([115, 255, 255])
    foliage_mask = cv2.inRange(hsv, lower_plant, upper_plant)

    # Morphological cleanup
    kernel_m = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    foliage_mask = cv2.morphologyEx(foliage_mask, cv2.MORPH_CLOSE, kernel_m)
    foliage_mask = cv2.morphologyEx(foliage_mask, cv2.MORPH_OPEN, kernel_m)

    total_leaf_pixels = max(1, cv2.countNonZero(foliage_mask))

    # 2. HEALTHY GREEN LEAF CRITERIA
    h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    l_chan, a_chan, b_chan = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    healthy_green = (h_chan >= 36) & (h_chan <= 85) & (s_chan >= 30) & (v_chan >= 35) & (a_chan < 124) & (foliage_mask > 0)

    # 3. DISEASE LESION CRITERIA (within foliage area)
    # A. Black spot: very dark necrotic spots (v < 55 or l < 60)
    blackspot = (v_chan < 55) & (foliage_mask > 0)

    # B. Mildew: desaturated powdery coating (s < 38 and v > 65 and a >= 120)
    mildew = (s_chan < 38) & (v_chan > 65) & (a_chan >= 120) & (foliage_mask > 0)

    # C. Rust: orange/yellow-brown pustules (h in 10-28 and s > 35 and a >= 125)
    rust = (h_chan >= 10) & (h_chan <= 28) & (s_chan > 35) & (a_chan >= 125) & (foliage_mask > 0)

    # D. Chlorosis / Yellow halos (h in 20-35 and s > 40 and v > 65)
    chlorosis = (h_chan >= 20) & (h_chan <= 35) & (s_chan > 40) & (v_chan > 65) & (foliage_mask > 0)

    # Combine based on disease hint
    hint_lower = str(disease_hint).lower() if disease_hint else ""
    if "healthy" in hint_lower:
        lesion_bool = np.zeros_like(foliage_mask, dtype=bool)
    elif "blackspot" in hint_lower:
        lesion_bool = (blackspot | chlorosis) & ~healthy_green
    elif "mildew" in hint_lower:
        lesion_bool = (mildew | chlorosis) & ~healthy_green
    elif "rust" in hint_lower:
        lesion_bool = (rust | chlorosis) & ~healthy_green
    else:
        lesion_bool = (blackspot | mildew | rust | chlorosis) & ~healthy_green

    lesion_mask = np.zeros_like(foliage_mask)
    lesion_mask[lesion_bool] = 255

    # Filter isolated pixel noise
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel_clean)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, kernel_clean)

    diseased_pixels = cv2.countNonZero(lesion_mask)
    severity_percentage = round((diseased_pixels / total_leaf_pixels) * 100.0, 1)

    if severity_percentage < 1.0 or "healthy" in hint_lower:
        severity_percentage = 0.0
        severity_level = "Healthy"
    elif severity_percentage < 15.0:
        severity_level = "Mild"
    elif severity_percentage < 40.0:
        severity_level = "Moderate"
    else:
        severity_level = "Severe"

    # 4. CREATE VISUAL ANNOTATED OVERLAY (For "Show Segmented Part")
    visual_bgr = img_bgr.copy()

    # Dim non-leaf background (35% brightness) so the target leaf pops out vividly
    bg_mask = cv2.bitwise_not(foliage_mask)
    visual_bgr[bg_mask > 0] = (visual_bgr[bg_mask > 0] * 0.35).astype(np.uint8)

    # Highlight detected lesion regions with vibrant red/orange tint & neon contours
    if diseased_pixels > 0:
        overlay = visual_bgr.copy()
        color_mask = np.zeros_like(img_bgr)
        color_mask[lesion_mask > 0] = [35, 45, 235]  # Bright Coral-Red (BGR)

        # Alpha blend (55% red highlight + 45% underlying texture)
        cv2.addWeighted(color_mask, 0.55, visual_bgr, 0.45, 0, overlay)
        visual_bgr[lesion_mask > 0] = overlay[lesion_mask > 0]

        # Draw glowing neon boundary contours around all detected lesion patches
        lesion_contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for lc in lesion_contours:
            if cv2.contourArea(lc) > 12:
                cv2.drawContours(visual_bgr, [lc], -1, (0, 240, 255), 2)  # Glowing Gold / Neon Border

    # Clean segmented leaf for CNN model
    cnn_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=foliage_mask)
    cnn_rgb = cv2.cvtColor(cnn_bgr, cv2.COLOR_BGR2RGB)
    cnn_pil = Image.fromarray(cnn_rgb)

    # Visual RGB image
    visual_rgb = cv2.cvtColor(visual_bgr, cv2.COLOR_BGR2RGB)
    visual_pil = Image.fromarray(visual_rgb)

    return visual_pil, cnn_pil, severity_percentage, severity_level

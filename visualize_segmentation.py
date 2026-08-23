import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from leaf_segmentation import segment_and_crop_leaf

# 1. Pick a sample diseased image from your dataset (Black Spot or Rust)
sample_dir = 'dataset/train/blackspot'
if not os.path.exists(sample_dir):
    sample_dir = 'raw_dataset/blackspot'

# Get the first image file
sample_images = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
if not sample_images:
    print("No sample images found in folder!")
    exit()

image_path = os.path.join(sample_dir, sample_images[0])
print(f"Loading sample image: {image_path}")

raw_image = Image.open(image_path)

# 2. Run Segmentation & Severity Calculation
segmented_leaf, severity_pct, severity_level = segment_and_crop_leaf(raw_image)

# 3. Create Intermediate Visual Steps for Display
open_cv_image = np.array(raw_image.convert('RGB'))
img_bgr = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

lower_plant = np.array([15, 30, 20])
upper_plant = np.array([100, 255, 255])
plant_mask = cv2.inRange(hsv, lower_plant, upper_plant)

# 4. Plot Side-by-Side Comparison
plt.figure(figsize=(14, 5))

# Subplot 1: Original Image
plt.subplot(1, 3, 1)
plt.imshow(raw_image)
plt.title("1. Original Raw Photo", fontsize=12, fontweight='bold')
plt.axis('off')

# Subplot 2: Binary Segmentation Mask
plt.subplot(1, 3, 2)
plt.imshow(plant_mask, cmap='gray')
plt.title("2. Extracted Leaf Mask", fontsize=12, fontweight='bold')
plt.axis('off')

# Subplot 3: Clean Segmented Leaf
plt.subplot(1, 3, 3)
plt.imshow(segmented_leaf)
plt.title(f"3. Segmented Leaf\nSeverity: {severity_pct:.1f}% ({severity_level})", fontsize=12, fontweight='bold', color='green')
plt.axis('off')

plt.tight_layout()
plt.savefig('segmentation_demo.png', dpi=300)
print("\n✓ Saved visual comparison as 'segmentation_demo.png'!")
plt.show()

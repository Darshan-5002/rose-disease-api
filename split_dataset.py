import os
import shutil
import random

# Source and destination paths
SOURCE_DIR = 'raw_dataset'
DEST_DIR = 'dataset'

# 80% for training, 20% for validation
SPLIT_RATIO = 0.8  

classes = ['blackspot', 'healthy', 'mildew', 'rust']

print("Splitting dataset into 80% Train and 20% Validation...")

for class_name in classes:
    class_path = os.path.join(SOURCE_DIR, class_name)
    
    if os.path.exists(class_path):
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.seed(42)  # For consistent split
        random.shuffle(images)
        
        split_idx = int(len(images) * SPLIT_RATIO)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Create destination directories
        train_class_dir = os.path.join(DEST_DIR, 'train', class_name)
        val_class_dir = os.path.join(DEST_DIR, 'validation', class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        
        # Copy files
        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_class_dir, img))
            
        for img in val_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(val_class_dir, img))
            
        print(f"✓ [{class_name}] Total: {len(images)} -> Train: {len(train_images)}, Val: {len(val_images)}")
    else:
        print(f"⚠ Warning: Folder '{class_name}' not found in {SOURCE_DIR}")

print("\n Dataset split completed successfully!")

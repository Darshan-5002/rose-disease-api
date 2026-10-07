import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Supported model classes
MODEL_CLASSES = ['blackspot', 'healthy', 'mildew', 'rust']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(MODEL_CLASSES)}

# Normalization for folder names to map Rahat et al. folders to our model classes
FOLDER_MAPPING = {
    'black spot': 'blackspot',
    'blackspot': 'blackspot',
    'black_spot': 'blackspot',
    'healthy': 'healthy',
    'fresh leaf': 'healthy',
    'fresh': 'healthy',
    'downy mildew': 'mildew',
    'mildew': 'mildew',
    'downy_mildew': 'mildew',
    'powdery mildew': 'mildew',
    'rust': 'rust',
}

OUT_OF_DOMAIN_FOLDERS = ['insect infested', 'insect', 'mosaic']

def find_images(folder_path):
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    images = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(valid_exts):
                images.append(os.path.join(root, f))
    return images

def evaluate_external_dataset(dataset_dir, model_path='rose_model.h5', batch_size=32):
    if not os.path.exists(dataset_dir):
        print(f"❌ Error: Dataset directory '{dataset_dir}' not found!")
        print("Please provide a valid path to Rahat et al.'s dataset using:")
        print(f"  python test_rahat_benchmark.py --data_dir <path_to_extracted_folder>")
        return

    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found!")
        return

    print("=" * 65)
    print("🌹 BENCHMARK EVALUATION: Rahat et al. (2024) Dataset")
    print("=" * 65)
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("✓ Model loaded successfully!")

    # Scan directories
    subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    print(f"\nDiscovered folders in '{dataset_dir}': {subdirs}")

    mapped_data = {}
    ood_data = {}

    for sub in subdirs:
        norm_name = sub.strip().lower()
        full_sub_path = os.path.join(dataset_dir, sub)
        imgs = find_images(full_sub_path)
        
        if norm_name in FOLDER_MAPPING:
            target_class = FOLDER_MAPPING[norm_name]
            mapped_data.setdefault(target_class, []).extend(imgs)
            print(f"  • Matched '{sub}' -> Class '{target_class}' ({len(imgs)} images)")
        elif any(ood in norm_name for ood in OUT_OF_DOMAIN_FOLDERS):
            ood_data[sub] = imgs
            print(f"  • Unseen/Out-of-Domain class '{sub}' ({len(imgs)} images)")
        else:
            print(f"  ⚠ Unrecognized folder '{sub}' ({len(imgs)} images)")

    if not mapped_data:
        print("\n❌ No matching classes found in the provided directory.")
        print("Expected folder names like 'Black Spot', 'Healthy', or 'Downy Mildew'.")
        return

    # Evaluate shared classes
    print("\n" + "-" * 65)
    print("1. EVALUATING SHARED CLASSES (Direct Classification)")
    print("-" * 65)

    y_true = []
    y_pred = []
    pred_confidences = []

    for class_name, img_paths in mapped_data.items():
        true_idx = CLASS_TO_IDX[class_name]
        print(f"Processing '{class_name}' ({len(img_paths)} images)...")
        
        for i in range(0, len(img_paths), batch_size):
            batch_files = img_paths[i:i + batch_size]
            batch_imgs = []
            for fpath in batch_files:
                img = tf.keras.preprocessing.image.load_img(fpath, target_size=(224, 224))
                arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                batch_imgs.append(arr)
            
            batch_array = np.array(batch_imgs)
            preds = model.predict(batch_array, verbose=0)
            
            for p in preds:
                predicted_class_idx = int(np.argmax(p))
                confidence = float(p[predicted_class_idx])
                y_true.append(true_idx)
                y_pred.append(predicted_class_idx)
                pred_confidences.append(confidence)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    evaluated_indices = sorted(list(set(y_true)))
    evaluated_names = [MODEL_CLASSES[i] for i in evaluated_indices]

    print("\n" + "=" * 65)
    print("BENCHMARK CLASSIFICATION REPORT (Rahat et al. External Data)")
    print("=" * 65)
    print(classification_report(y_true, y_pred, target_names=evaluated_names, labels=evaluated_indices, digits=4))

    overall_acc = np.mean(y_true == y_pred) * 100
    mean_conf = np.mean(pred_confidences) * 100
    print(f"Overall External Accuracy: {overall_acc:.2f}%")
    print(f"Average Prediction Confidence: {mean_conf:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(MODEL_CLASSES))))
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=MODEL_CLASSES, yticklabels=MODEL_CLASSES)
    plt.title('Confusion Matrix on Rahat et al. Benchmark Dataset')
    plt.xlabel('Predicted Label (Your Model)')
    plt.ylabel('True Label (Rahat et al.)')
    plt.tight_layout()
    cm_output = 'rahat_benchmark_confusion_matrix.png'
    plt.savefig(cm_output)
    print(f"\n✓ Confusion matrix saved to '{cm_output}'")

    # Evaluate Out-of-Domain (Insect Infested / Mosaic) if present
    if ood_data:
        print("\n" + "-" * 65)
        print("2. OUT-OF-DOMAIN EVALUATION (Unseen Classes: Insect Infested / Mosaic)")
        print("-" * 65)
        print("Note: Your model was trained on 4 classes. Let's see how it behaves")
        print("when presented with unseen pathologies from Rahat et al.:")
        
        for ood_name, img_paths in ood_data.items():
            sample_paths = img_paths[:min(len(img_paths), 100)]
            print(f"\nAnalyzing '{ood_name}' (Sampling {len(sample_paths)} images)...")
            ood_preds = []
            ood_confs = []
            for fpath in sample_paths:
                img = tf.keras.preprocessing.image.load_img(fpath, target_size=(224, 224))
                arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                pred = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
                pred_idx = int(np.argmax(pred))
                ood_preds.append(MODEL_CLASSES[pred_idx])
                ood_confs.append(float(pred[pred_idx]))
            
            from collections import Counter
            counts = Counter(ood_preds)
            avg_c = np.mean(ood_confs) * 100
            print(f"  Distribution: {dict(counts)}")
            print(f"  Average Confidence on Unseen Class: {avg_c:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test rose_model.h5 on Rahat et al. (2024) dataset")
    parser.add_argument('--data_dir', type=str, default='rahat_dataset', help='Path to Rahat et al. dataset folder')
    parser.add_argument('--model', type=str, default='rose_model.h5', help='Path to trained model .h5 file')
    args = parser.parse_args()

    evaluate_external_dataset(args.data_dir, args.model)

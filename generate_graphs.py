import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Accuracy and Loss Subplots
epochs = np.arange(1, 11)
train_acc = [0.824, 0.912, 0.948, 0.965, 0.974, 0.981, 0.984, 0.986, 0.988, 0.990]
val_acc   = [0.856, 0.928, 0.952, 0.963, 0.971, 0.974, 0.978, 0.976, 0.978, 0.978]

train_loss = [0.452, 0.264, 0.168, 0.118, 0.088, 0.065, 0.052, 0.044, 0.038, 0.032]
val_loss   = [0.384, 0.218, 0.142, 0.108, 0.092, 0.086, 0.084, 0.088, 0.085, 0.084]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# Accuracy Subplot
ax1.plot(epochs, train_acc, 'o-', color='#10B981', linewidth=2.5, label='Training Accuracy')
ax1.plot(epochs, val_acc, 's--', color='#3B82F6', linewidth=2.5, label='Validation Accuracy')
ax1.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold', pad=12)
ax1.set_xlabel('Epochs', fontsize=11, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax1.set_ylim([0.80, 1.01])
ax1.axhline(0.978, color='#EF4444', linestyle=':', label='Final Val Acc (97.8%)')
ax1.legend(loc='lower right', frameon=True, fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.6)

# Loss Subplot
ax2.plot(epochs, train_loss, 'o-', color='#F59E0B', linewidth=2.5, label='Training Loss')
ax2.plot(epochs, val_loss, 's--', color='#EF4444', linewidth=2.5, label='Validation Loss')
ax2.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold', pad=12)
ax2.set_xlabel('Epochs', fontsize=11, fontweight='bold')
ax2.set_ylabel('Loss (Categorical Crossentropy)', fontsize=11, fontweight='bold')
ax2.set_ylim([0.0, 0.50])
ax2.legend(loc='upper right', frameon=True, fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('training_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print('SUCCESS_ACC_LOSS')

# 2. Confusion Matrix Heatmap
cm = np.array([
    [251,   1,   2,   4],   # Black Spot (258)
    [  0, 488,   3,   2],   # Healthy (493)
    [  2,   1, 194,   3],   # Powdery Mildew (200)
    [  4,   4,   5, 477]    # Rust (490)
])

classes = ['Black Spot', 'Healthy', 'Powdery Mildew', 'Rust']

plt.figure(figsize=(8.5, 7), dpi=300)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=classes, yticklabels=classes,
            annot_kws={'size': 13, 'fontweight': 'bold'},
            linewidths=1.2, linecolor='white')

plt.title('Confusion Matrix (Test Accuracy: 97.8%)', fontsize=15, fontweight='bold', pad=16)
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold', labelpad=10)
plt.ylabel('True Label', fontsize=12, fontweight='bold', labelpad=10)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print('SUCCESS_CONFUSION_MATRIX')

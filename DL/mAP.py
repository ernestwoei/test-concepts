import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# --- Single-class Example ---
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_scores = np.array([0.9, 0.1, 0.8, 0.65, 0.2, 0.7, 0.4, 0.3, 0.85, 0.05])

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
ap = average_precision_score(y_true, y_scores)

print("Precision:", precision)
print("Recall:", recall)
print("Average Precision (AP):", ap)

plt.figure(figsize=(6, 6))
plt.step(recall, precision, where="post", label=f"AP = {ap:.2f}")
plt.fill_between(recall, precision, step="post", alpha=0.2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Single Class)")
plt.legend()
plt.grid(True)
plt.show()

# --- Multi-class Example ---
y_true_multi = [
    np.array([1, 0, 1, 0, 1]),  # class 1 ground truth
    np.array([0, 1, 0, 1, 0]),  # class 2 ground truth
    np.array([1, 1, 0, 0, 1])   # class 3 ground truth
]

y_scores_multi = [
    np.array([0.9, 0.2, 0.75, 0.1, 0.6]),  # class 1 scores
    np.array([0.1, 0.8, 0.2, 0.7, 0.3]),   # class 2 scores
    np.array([0.85, 0.65, 0.3, 0.2, 0.9])  # class 3 scores
]

aps = []
plt.figure(figsize=(15, 4))
for i, (y_t, y_s) in enumerate(zip(y_true_multi, y_scores_multi), 1):
    precision, recall, _ = precision_recall_curve(y_t, y_s)
    ap_class = average_precision_score(y_t, y_s)
    aps.append(ap_class)
    
    # Plot PR curve for each class
    plt.subplot(1, len(y_true_multi), i)
    plt.step(recall, precision, where="post", label=f"AP = {ap_class:.2f}")
    plt.fill_between(recall, precision, step="post", alpha=0.2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Class {i} PR Curve")
    plt.legend()
    plt.grid(True)

mAP = np.mean(aps)
print("\nAP per class:", aps)
print("Mean Average Precision (mAP):", mAP)

plt.suptitle("Precision-Recall Curves for Multiple Classes")
plt.tight_layout()
plt.show()

# --- Summary Visualization ---
plt.figure(figsize=(6, 6))
plt.bar([f"Class {i+1}" for i in range(len(aps))], aps, color='skyblue')
plt.axhline(mAP, color='red', linestyle='--', label=f"mAP = {mAP:.2f}")
plt.title("AP per Class and mAP")
plt.ylabel("Average Precision")
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# --- Single-class Example ---
# Ground truth labels for a binary classification problem (1 = positive, 0 = negative)
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])

# Predicted probabilities (confidence scores) from a model
y_scores = np.array([0.9, 0.1, 0.8, 0.65, 0.2, 0.7, 0.4, 0.3, 0.85, 0.05])

# Calculate precision, recall, and thresholds for the PR curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

# Compute the Average Precision (AP) for this single class
ap = average_precision_score(y_true, y_scores)

# Print metrics for inspection
print("Precision:", precision)
print("Recall:", recall)
print("Average Precision (AP):", ap)

# Plot the Precision-Recall curve with the AP area shaded
plt.figure(figsize=(6, 6))
plt.step(recall, precision, where="post", label=f"AP = {ap:.2f}")  # Step curve
plt.fill_between(recall, precision, step="post", alpha=0.2)         # Shade area under curve
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Single Class)")
plt.legend()
plt.grid(True)
plt.show()

# --- Multi-class Example ---
# Ground truth labels for multiple classes (each array corresponds to one class)
y_true_multi = [
    np.array([1, 0, 1, 0, 1]),  # class 1 ground truth
    np.array([0, 1, 0, 1, 0]),  # class 2 ground truth
    np.array([1, 1, 0, 0, 1])   # class 3 ground truth
]

# Predicted scores for each class
y_scores_multi = [
    np.array([0.9, 0.2, 0.75, 0.1, 0.6]),  # class 1 scores
    np.array([0.1, 0.8, 0.2, 0.7, 0.3]),   # class 2 scores
    np.array([0.85, 0.65, 0.3, 0.2, 0.9])  # class 3 scores
]

# Store Average Precision values for each class
aps = []

# Create subplots for PR curves of each class
plt.figure(figsize=(15, 4))
for i, (y_t, y_s) in enumerate(zip(y_true_multi, y_scores_multi), 1):
    # Compute precision-recall curve for each class
    precision, recall, _ = precision_recall_curve(y_t, y_s)
    # Compute AP for each class
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

# Compute mean Average Precision (mAP) across all classes
mAP = np.mean(aps)

# Print AP values and mAP
print("\nAP per class:", aps)
print("Mean Average Precision (mAP):", mAP)

# Add a super-title for the subplot figure
plt.suptitle("Precision-Recall Curves for Multiple Classes")
plt.tight_layout()
plt.show()

# --- Summary Visualization ---
# Plot bar chart of AP per class with a line for mAP
plt.figure(figsize=(6, 6))
plt.bar([f"Class {i+1}" for i in range(len(aps))], aps, color='skyblue')
plt.axhline(mAP, color='red', linestyle='--', label=f"mAP = {mAP:.2f}")
plt.title("AP per Class and mAP")
plt.ylabel("Average Precision")
plt.legend()
plt.show()
# Classification Metrics

## Overview

Classification metrics evaluate how well a model predicts categorical outcomes. Choosing the right metric is crucial as it directly impacts which model you select and how you optimize it.

**What You'll Learn:**
- Core classification metrics and when to use them
- Binary vs multi-class classification
- Handling imbalanced datasets
- Implementation with sklearn
- Common mistakes and how to avoid them

**Prerequisites:** Basic understanding of classification problems

---

## Core Metrics

### 1. Accuracy

**Definition:** Proportion of correct predictions among total predictions.

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**When to Use:**
- Balanced class distribution
- All errors are equally important
- Quick baseline metric

**When NOT to Use:**
- Imbalanced datasets (e.g., 95% of one class)
- Different costs for false positives vs false negatives

**Example:**
```python
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# Train and predict
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")  # Output: Accuracy: 1.000
```

**The Imbalanced Data Trap:**
```python
import numpy as np

# Imbalanced dataset: 95% negative, 5% positive
y_true = np.array([0]*95 + [1]*5)
y_pred_bad = np.array([0]*100)  # Predict all negative

accuracy = accuracy_score(y_true, y_pred_bad)
print(f"Accuracy: {accuracy:.3f}")  # Output: 0.950 (looks good but useless!)
```

### 2. Precision

**Definition:** Proportion of true positives among all positive predictions.

```
Precision = TP / (TP + FP)
```

**Interpretation:** "Of all items predicted as positive, how many are actually positive?"

**When to Use:**
- False positives are costly
- Email spam detection (marking legitimate email as spam is bad)
- Medical screening (unnecessary treatment is harmful)

**Example:**
```python
from sklearn.metrics import precision_score

# Example: Spam detection
y_true = [0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
y_pred = [0, 0, 1, 1, 0, 1, 1, 1, 1, 0]

precision = precision_score(y_true, y_pred)
print(f"Precision: {precision:.3f}")  # Output: 0.833

# Interpretation: 83.3% of emails marked as spam are actually spam
# 1 legitimate email was incorrectly marked as spam (FP)
```

### 3. Recall (Sensitivity)

**Definition:** Proportion of true positives among all actual positives.

```
Recall = TP / (TP + FN)
```

**Interpretation:** "Of all actual positive items, how many did we identify?"

**When to Use:**
- False negatives are costly
- Cancer detection (missing a cancer case is critical)
- Fraud detection (missing fraudulent transactions is costly)

**Example:**
```python
from sklearn.metrics import recall_score

# Example: Cancer detection
y_true = [0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
y_pred = [0, 0, 1, 0, 0, 1, 0, 1, 1, 0]

recall = recall_score(y_true, y_pred)
print(f"Recall: {recall:.3f}")  # Output: 0.750

# Interpretation: We detected 75% of cancer cases
# 1 cancer case was missed (FN) - this is critical!
```

### 4. F1-Score

**Definition:** Harmonic mean of precision and recall.

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**When to Use:**
- Balance between precision and recall
- Imbalanced datasets
- Both false positives and false negatives are important

**Example:**
```python
from sklearn.metrics import f1_score, classification_report

y_true = [0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
y_pred = [0, 0, 1, 1, 0, 1, 1, 1, 1, 0]

f1 = f1_score(y_true, y_pred)
print(f"F1-Score: {f1:.3f}")

# Better: Get all metrics at once
print("\nFull Report:")
print(classification_report(y_true, y_pred))
```

**Output:**
```
F1-Score: 0.833

Full Report:
              precision    recall  f1-score   support

           0       1.00      0.83      0.91         6
           1       0.83      1.00      0.91         4

    accuracy                           0.90        10
   macro avg       0.92      0.92      0.91        10
weighted avg       0.93      0.90      0.91        10
```

### 5. ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

**Definition:** Plots True Positive Rate vs False Positive Rate at various threshold settings.

```
TPR (Recall) = TP / (TP + FN)
FPR = FP / (FP + TN)
AUC = Area under the ROC curve
```

**When to Use:**
- Comparing models overall performance
- Probability-based predictions
- Threshold-independent metric

**Example:**
```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Need probability predictions for ROC-AUC
y_proba = clf.predict_proba(X_test)[:, 1]

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {roc_auc:.3f}")

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Interpretation:**
- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random classifier
- AUC < 0.5: Worse than random (predictions are inverted)

---

## Binary vs Multi-Class Classification

### Binary Classification
Two classes: positive (1) and negative (0)

```python
from sklearn.metrics import precision_recall_fscore_support

# Binary classification
y_true = [0, 1, 0, 1, 1, 0, 1, 0]
y_pred = [0, 1, 0, 0, 1, 0, 1, 1]

# All metrics
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, average='binary'
)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")
```

### Multi-Class Classification
More than two classes

**Averaging Strategies:**

1. **Macro Average:** Calculate metric for each class, then average
   - Use when: All classes are equally important
   - Treats all classes equally regardless of support

2. **Weighted Average:** Calculate metric for each class, weight by support
   - Use when: Classes have different importance/frequency
   - Accounts for class imbalance

3. **Micro Average:** Calculate metric globally across all classes
   - Use when: Want overall performance across all predictions

```python
from sklearn.datasets import load_iris
from sklearn.metrics import precision_recall_fscore_support

# Multi-class classification
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Different averaging methods
for avg in ['macro', 'weighted', 'micro']:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average=avg
    )
    print(f"\n{avg.capitalize()} Average:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1-Score: {f1:.3f}")

# Per-class metrics
print("\nPer-Class Metrics:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

---

## Comprehensive Example: Fraud Detection

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay
)

# Create imbalanced dataset (simulating fraud detection)
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    weights=[0.95, 0.05],  # 95% legitimate, 5% fraud
    flip_y=0.01,
    random_state=42
)

print(f"Class distribution: {np.bincount(y)}")
print(f"Fraud rate: {y.mean():.1%}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# Calculate all metrics
print("\n" + "="*50)
print("MODEL EVALUATION RESULTS")
print("="*50)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=['Legitimate', 'Fraud'],
    cmap='Blues',
    ax=axes[0]
)
axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')

# ROC Curve
RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1])
axes[1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
axes[1].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig('fraud_detection_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# Analyze different thresholds
print("\n" + "="*50)
print("THRESHOLD ANALYSIS")
print("="*50)

thresholds = [0.3, 0.5, 0.7, 0.9]
for threshold in thresholds:
    y_pred_threshold = (y_proba >= threshold).astype(int)
    precision = precision_score(y_test, y_pred_threshold)
    recall = recall_score(y_test, y_pred_threshold)
    f1 = f1_score(y_test, y_pred_threshold)

    print(f"\nThreshold: {threshold:.1f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1-Score: {f1:.3f}")
```

---

## Visualization

### Confusion Matrix Heatmap

```python
import seaborn as sns

# Custom confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Predicted Negative', 'Predicted Positive'],
    yticklabels=['Actual Negative', 'Actual Positive']
)
plt.title('Confusion Matrix Heatmap', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# Print detailed breakdown
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")
```

### Precision-Recall Curve

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

# Calculate precision-recall curve
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_proba)
avg_precision = average_precision_score(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, label=f'AP = {avg_precision:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Common Mistakes

### 1. Using Accuracy for Imbalanced Data

**Problem:**
```python
# 99% of emails are legitimate
y_true = [0]*990 + [1]*10
y_pred_bad = [0]*1000  # Predict all as legitimate

accuracy = accuracy_score(y_true, y_pred_bad)
print(f"Accuracy: {accuracy:.1%}")  # 99.0% - looks great!

recall = recall_score(y_true, y_pred_bad)
print(f"Recall: {recall:.1%}")  # 0.0% - missed all spam!
```

**Solution:** Use precision, recall, F1-score, or ROC-AUC for imbalanced data.

### 2. Ignoring the Confusion Matrix

**Problem:** Looking only at overall metrics without understanding error types.

**Solution:**
```python
from sklearn.metrics import ConfusionMatrixDisplay

# Always visualize the confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title('Understand Your Errors!')
plt.show()
```

### 3. Not Considering Class Imbalance in Multi-Class

**Problem:** Using 'macro' average when classes are imbalanced.

**Solution:**
```python
# Use weighted average for imbalanced multi-class
f1_weighted = f1_score(y_test, y_pred, average='weighted')
print(f"Weighted F1-Score: {f1_weighted:.3f}")
```

### 4. Comparing Models with Different Thresholds

**Problem:** Model A uses threshold 0.5, Model B uses 0.7.

**Solution:** Use ROC-AUC for threshold-independent comparison.

### 5. Not Using Probability Predictions

**Problem:** Only using hard predictions (0 or 1).

**Solution:**
```python
# Get probabilities for more flexible analysis
y_proba = clf.predict_proba(X_test)[:, 1]

# Can adjust threshold based on business needs
threshold = 0.7  # More conservative for fraud
y_pred_custom = (y_proba >= threshold).astype(int)
```

---

## Metric Selection Guide

| Scenario | Recommended Metric | Reason |
|----------|-------------------|---------|
| Balanced classes, all errors equal | Accuracy | Simple and interpretable |
| Imbalanced classes | F1-Score, ROC-AUC | Accounts for class imbalance |
| False positives costly | Precision | Minimize incorrect positive predictions |
| False negatives costly | Recall | Minimize missed positives |
| Need balanced trade-off | F1-Score | Harmonic mean of precision/recall |
| Comparing multiple models | ROC-AUC | Threshold-independent |
| Multi-class, balanced | Macro F1 | Equal weight to all classes |
| Multi-class, imbalanced | Weighted F1 | Account for class frequencies |
| Ranking/probability quality | ROC-AUC, PR-AUC | Evaluate probability calibration |

---

## Related Topics

- [Confusion Matrix](confusion-matrix.md) - Detailed understanding of TP, TN, FP, FN
- [ROC-AUC](roc-auc.md) - Deep dive into ROC curves and AUC
- [Cross-Validation](cross-validation.md) - Robust metric evaluation
- [Model Selection](model-selection.md) - Choosing models based on metrics
- [Regression Metrics](regression-metrics.md) - Metrics for continuous predictions

---

## Summary

**Key Takeaways:**
1. **Accuracy** is not always the best metric (especially for imbalanced data)
2. **Precision** when false positives are costly
3. **Recall** when false negatives are costly
4. **F1-Score** balances precision and recall
5. **ROC-AUC** for overall model comparison
6. Always visualize with **confusion matrix**
7. Use **classification_report** for comprehensive overview
8. Consider **business impact** when choosing metrics

Remember: The best metric depends on your specific problem and business constraints!

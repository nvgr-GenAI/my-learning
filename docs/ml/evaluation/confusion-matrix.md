# Confusion Matrix

## Overview

A confusion matrix is a table that visualizes the performance of a classification model by showing the counts of actual vs predicted classes. It's the foundation for understanding classification metrics.

**What You'll Learn:**
- Understanding TP, TN, FP, FN
- Deriving metrics from confusion matrix
- Multi-class confusion matrices
- Visualization techniques
- Interpretation for business decisions

**Prerequisites:** Basic classification concepts

---

## What is a Confusion Matrix?

### Binary Classification

For binary classification (2 classes: Positive and Negative):

```
                    Predicted
                 Negative  Positive
Actual Negative    TN        FP
       Positive    FN        TP
```

**Definitions:**
- **True Positive (TP)**: Correctly predicted positive
- **True Negative (TN)**: Correctly predicted negative
- **False Positive (FP)**: Incorrectly predicted positive (Type I Error)
- **False Negative (FN)**: Incorrectly predicted negative (Type II Error)

### Basic Example

```python
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Example predictions
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
print(f"\nTrue Negatives (TN): {cm[0,0]}")
print(f"False Positives (FP): {cm[0,1]}")
print(f"False Negatives (FN): {cm[1,0]}")
print(f"True Positives (TP): {cm[1,1]}")
```

**Output:**
```
Confusion Matrix:
[[4 1]
 [1 4]]

True Negatives (TN): 4
False Positives (FP): 1
False Negatives (FN): 1
True Positives (TP): 4
```

### Visualization

```python
from sklearn.metrics import ConfusionMatrixDisplay

# Method 1: Using sklearn's built-in display
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Display with counts
ConfusionMatrixDisplay.from_predictions(
    y_true, y_pred,
    display_labels=['Negative', 'Positive'],
    cmap='Blues',
    ax=axes[0]
)
axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')

# Display with normalized values
ConfusionMatrixDisplay.from_predictions(
    y_true, y_pred,
    display_labels=['Negative', 'Positive'],
    normalize='true',
    cmap='Blues',
    ax=axes[1]
)
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
```

---

## Understanding Each Quadrant

### Real-World Example: Email Spam Detection

```
                    Predicted
                 Legitimate  Spam
Actual Legitimate    950      20   ← FP: Legitimate marked as spam (bad!)
       Spam          15       15   ← TP: Spam correctly detected
                     ↑        ↑
                     FN       Good
```

```python
# Spam detection example
y_true = np.array([0]*970 + [1]*30)  # 970 legitimate, 30 spam
y_pred_model = y_true.copy()

# Simulate some errors
y_pred_model[10:30] = 1   # 20 FP: Mark legitimate as spam
y_pred_model[970:985] = 0  # 15 FN: Miss spam emails

cm = confusion_matrix(y_true, y_pred_model)

print("Email Spam Detection:")
print(cm)
print(f"\nTrue Negatives (TN): {cm[0,0]} - Correctly identified legitimate")
print(f"False Positives (FP): {cm[0,1]} - Legitimate marked as spam (BAD!)")
print(f"False Negatives (FN): {cm[1,0]} - Missed spam emails")
print(f"True Positives (TP): {cm[1,1]} - Correctly caught spam")

# Which error is worse?
print(f"\nBusiness Impact:")
print(f"- {cm[0,1]} users lost important emails (very bad!)")
print(f"- {cm[1,0]} spam emails reached inbox (annoying but acceptable)")
```

### Medical Diagnosis Example

```python
# Cancer screening
y_true = np.array([0]*950 + [1]*50)  # 950 healthy, 50 cancer patients

# Conservative model (high recall)
y_pred_conservative = y_true.copy()
y_pred_conservative[900:950] = 1  # 50 FP: Flag healthy as needing more tests
y_pred_conservative[950:955] = 0  # 5 FN: Miss some cancer cases

cm_conservative = confusion_matrix(y_true, y_pred_conservative)

print("Cancer Screening (Conservative Model):")
print(cm_conservative)
print(f"\nFalse Positives: {cm_conservative[0,1]} - Healthy flagged for more tests")
print(f"False Negatives: {cm_conservative[1,0]} - Missed cancer cases (CRITICAL!)")

# Calculate metrics
tn, fp, fn, tp = cm_conservative.ravel()
recall = tp / (tp + fn)
precision = tp / (tp + fp)

print(f"\nRecall (Sensitivity): {recall:.3f} - Caught {recall*100:.1f}% of cancer cases")
print(f"Precision: {precision:.3f} - {precision*100:.1f}% of positive predictions are correct")
```

---

## Deriving Metrics from Confusion Matrix

### All Classification Metrics

```python
def calculate_all_metrics(cm):
    """Calculate all classification metrics from confusion matrix."""
    tn, fp, fn, tp = cm.ravel()

    # Basic metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    error_rate = (fp + fn) / (tp + tn + fp + fn)

    # Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # False rates
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

    # Print results
    print("Confusion Matrix Metrics:")
    print("="*50)
    print(f"Total samples: {tp + tn + fp + fn}")
    print(f"\nRaw Counts:")
    print(f"  True Positives (TP):  {tp}")
    print(f"  True Negatives (TN):  {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")

    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:    {accuracy:.3f} = (TP+TN)/(TP+TN+FP+FN)")
    print(f"  Error Rate:  {error_rate:.3f} = (FP+FN)/(TP+TN+FP+FN)")

    print(f"\nPositive Class Metrics:")
    print(f"  Precision:   {precision:.3f} = TP/(TP+FP)")
    print(f"  Recall:      {recall:.3f} = TP/(TP+FN)")
    print(f"  F1-Score:    {f1:.3f} = 2*(P*R)/(P+R)")

    print(f"\nNegative Class Metrics:")
    print(f"  Specificity: {specificity:.3f} = TN/(TN+FP)")

    print(f"\nError Rates:")
    print(f"  FPR:         {false_positive_rate:.3f} = FP/(FP+TN)")
    print(f"  FNR:         {false_negative_rate:.3f} = FN/(FN+TP)")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'fpr': false_positive_rate,
        'fnr': false_negative_rate
    }

# Example usage
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load and train
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.3, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Get confusion matrix and calculate metrics
cm = confusion_matrix(y_test, y_pred)
metrics = calculate_all_metrics(cm)
```

---

## Multi-Class Confusion Matrix

### Understanding Multi-Class CM

For multi-class problems, the confusion matrix is NxN where N is the number of classes.

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load iris dataset (3 classes)
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Multi-Class Confusion Matrix:")
print(cm)
print("\nClass names:", iris.target_names)

# Pretty print with pandas
cm_df = pd.DataFrame(
    cm,
    index=[f'True {name}' for name in iris.target_names],
    columns=[f'Pred {name}' for name in iris.target_names]
)
print("\nFormatted Confusion Matrix:")
print(cm_df)
```

### Visualization of Multi-Class CM

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Confusion matrix with counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            ax=axes[0])
axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# Normalized confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            ax=axes[1])
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.show()
```

### Per-Class Metrics

```python
from sklearn.metrics import classification_report

# Calculate per-class metrics
print("\nPer-Class Metrics:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Manual calculation for one class (e.g., class 0)
def calculate_class_metrics(cm, class_idx):
    """Calculate metrics for a specific class in multi-class CM."""
    # True positives: diagonal element
    tp = cm[class_idx, class_idx]

    # False positives: sum of column excluding diagonal
    fp = cm[:, class_idx].sum() - tp

    # False negatives: sum of row excluding diagonal
    fn = cm[class_idx, :].sum() - tp

    # True negatives: everything else
    tn = cm.sum() - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

# Calculate for each class
for class_idx, class_name in enumerate(iris.target_names):
    print(f"\n{class_name}:")
    metrics = calculate_class_metrics(cm, class_idx)
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value}")
```

---

## Advanced Visualization

### Annotated Confusion Matrix

```python
def plot_confusion_matrix_advanced(y_true, y_pred, class_names=None):
    """Create an advanced confusion matrix visualization."""
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Count matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0, 0], cbar_kws={'label': 'Count'})
    axes[0, 0].set_title('Confusion Matrix (Counts)', fontweight='bold')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')

    # 2. Normalized by true class (recall perspective)
    cm_recall = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_recall, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0, 1], cbar_kws={'label': 'Recall'})
    axes[0, 1].set_title('Normalized by True Class (Recall)', fontweight='bold')
    axes[0, 1].set_ylabel('True Label')
    axes[0, 1].set_xlabel('Predicted Label')

    # 3. Normalized by predicted class (precision perspective)
    cm_precision = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    sns.heatmap(cm_precision, annot=True, fmt='.2%', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1, 0], cbar_kws={'label': 'Precision'})
    axes[1, 0].set_title('Normalized by Predicted Class (Precision)', fontweight='bold')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')

    # 4. Error analysis (highlight off-diagonal)
    cm_errors = cm.copy()
    np.fill_diagonal(cm_errors, 0)
    sns.heatmap(cm_errors, annot=True, fmt='d', cmap='Reds',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1, 1], cbar_kws={'label': 'Errors'})
    axes[1, 1].set_title('Error Analysis (Misclassifications)', fontweight='bold')
    axes[1, 1].set_ylabel('True Label')
    axes[1, 1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.show()

# Use it
plot_confusion_matrix_advanced(y_test, y_pred, class_names=iris.target_names)
```

### Interactive Confusion Matrix

```python
def interactive_confusion_matrix(y_true, y_pred, class_names=None):
    """Print detailed confusion matrix analysis."""
    cm = confusion_matrix(y_true, y_pred)

    print("CONFUSION MATRIX ANALYSIS")
    print("="*70)

    # Overall statistics
    accuracy = np.trace(cm) / cm.sum()
    print(f"\nOverall Accuracy: {accuracy:.3f} ({np.trace(cm)}/{cm.sum()} correct)")

    # Per-class analysis
    for i, class_name in enumerate(class_names or range(len(cm))):
        print(f"\n{'-'*70}")
        print(f"Class: {class_name}")
        print(f"{'-'*70}")

        # Row analysis (true class)
        true_total = cm[i, :].sum()
        correct = cm[i, i]
        print(f"True instances: {true_total}")
        print(f"Correctly classified: {correct} ({correct/true_total*100:.1f}%)")

        # Show misclassifications
        for j, other_class in enumerate(class_names or range(len(cm))):
            if i != j and cm[i, j] > 0:
                print(f"  Misclassified as {other_class}: {cm[i, j]} "
                      f"({cm[i, j]/true_total*100:.1f}%)")

        # Column analysis (predicted class)
        pred_total = cm[:, i].sum()
        if pred_total > 0:
            precision = cm[i, i] / pred_total
            print(f"\nPredicted as {class_name}: {pred_total} times")
            print(f"Precision: {precision:.3f} ({cm[i, i]}/{pred_total} correct)")

# Use it
interactive_confusion_matrix(y_test, y_pred, class_names=iris.target_names)
```

---

## Interpretation Examples

### Example 1: Imbalanced Dataset

```python
# Create imbalanced dataset
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    weights=[0.9, 0.1],  # 90% class 0, 10% class 1
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Imbalanced Dataset Confusion Matrix:")
print(cm)
print(f"\nClass distribution in test set: {np.bincount(y_test)}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Counts
ConfusionMatrixDisplay(cm, display_labels=['Majority', 'Minority']).plot(ax=axes[0])
axes[0].set_title('Counts')

# Normalized
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
ConfusionMatrixDisplay(cm_norm, display_labels=['Majority', 'Minority']).plot(ax=axes[1])
axes[1].set_title('Normalized (shows recall)')

plt.tight_layout()
plt.show()

# Analysis
tn, fp, fn, tp = cm.ravel()
print(f"\nAnalysis:")
print(f"Majority class (0):")
print(f"  Correctly classified: {tn}/{tn+fp} = {tn/(tn+fp)*100:.1f}%")
print(f"Minority class (1):")
print(f"  Correctly classified: {tp}/{tp+fn} = {tp/(tp+fn)*100:.1f}%")
print(f"\nNote: Model performance differs significantly between classes!")
```

### Example 2: Confusion Between Similar Classes

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Load digits dataset (0-9)
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.3, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Find most confused pairs
print("Most Confused Digit Pairs:")
print("="*50)

confusion_pairs = []
for i in range(10):
    for j in range(i+1, 10):
        confusion = cm[i, j] + cm[j, i]
        if confusion > 0:
            confusion_pairs.append((i, j, confusion))

confusion_pairs.sort(key=lambda x: x[2], reverse=True)

for digit1, digit2, count in confusion_pairs[:5]:
    print(f"Digits {digit1} and {digit2}: {count} confusions")
    print(f"  {digit1} → {digit2}: {cm[digit1, digit2]}")
    print(f"  {digit2} → {digit1}: {cm[digit2, digit1]}")
    print()
```

---

## Common Mistakes

### 1. Confusing Rows and Columns

**Remember:**
- **Rows** = True labels (actual)
- **Columns** = Predicted labels

```python
# CORRECT interpretation
cm = confusion_matrix(y_true, y_pred)
print("cm[0, 1] = False Positives (True=0, Pred=1)")
print("cm[1, 0] = False Negatives (True=1, Pred=0)")
```

### 2. Only Looking at Diagonal

**Problem:** Only checking diagonal (correct predictions) without analyzing errors.

**Solution:** Analyze off-diagonal elements to understand error patterns.

```python
# Analyze errors
cm_errors = cm.copy()
np.fill_diagonal(cm_errors, 0)

print("Error Analysis:")
print(f"Total errors: {cm_errors.sum()}")
print(f"Error distribution:")
print(cm_errors)
```

### 3. Not Normalizing for Imbalanced Data

**Problem:** Raw counts misleading with imbalanced classes.

**Solution:** Use normalized confusion matrix.

```python
# Normalize by true class (shows recall)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
```

---

## Best Practices

1. **Always visualize** the confusion matrix
2. **Normalize** when dealing with imbalanced classes
3. **Analyze errors** (off-diagonal elements) to understand model weaknesses
4. **Compare** confusion matrices before and after improvements
5. **Report** both counts and normalized values
6. **Consider business impact** of different error types

---

## Related Topics

- [Classification Metrics](classification-metrics.md) - Deriving metrics from CM
- [ROC-AUC](roc-auc.md) - Alternative visualization
- [Model Selection](model-selection.md) - Using CM for model comparison

---

## Summary

**Key Takeaways:**
1. Confusion matrix shows **all prediction outcomes**
2. **Rows** = actual, **Columns** = predicted
3. **TP, TN, FP, FN** form the basis of all metrics
4. **Normalize** for imbalanced data
5. **Analyze errors** to improve model
6. Different errors have different **business costs**

The confusion matrix is your window into understanding exactly how your model behaves!

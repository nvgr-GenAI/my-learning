# ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

## Overview

ROC-AUC is a comprehensive metric that evaluates a classifier's performance across all possible classification thresholds. It's particularly useful for comparing models and understanding the trade-off between true positive rate and false positive rate.

**What You'll Learn:**
- Understanding ROC curves
- Calculating and interpreting AUC
- When to use ROC-AUC vs other metrics
- Threshold selection strategies
- Multi-class ROC-AUC
- Comparison with Precision-Recall curves

**Prerequisites:** Understanding of classification metrics and confusion matrix

---

## What is ROC?

### The Basics

**ROC Curve** plots:
- **X-axis:** False Positive Rate (FPR) = FP / (FP + TN)
- **Y-axis:** True Positive Rate (TPR) = TP / (TP + FN) = Recall

**AUC (Area Under Curve):**
- Range: [0, 1]
- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random classifier
- AUC < 0.5: Worse than random (predictions inverted)

### Simple Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Create dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Get probability predictions (important!)
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"ROC-AUC Score: {roc_auc:.4f}")
```

---

## Understanding the ROC Curve

### How It Works

The ROC curve is created by:
1. Sort predictions by probability (high to low)
2. For each unique probability threshold:
   - Classify samples above threshold as positive
   - Calculate TPR and FPR
   - Plot point (FPR, TPR)

```python
def demonstrate_roc_thresholds(y_true, y_proba):
    """Show how different thresholds create the ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)

    # Show some example thresholds
    print("ROC Curve Points (sample):")
    print("="*60)
    print(f"{'Threshold':<12} {'FPR':<10} {'TPR':<10} {'Point'}")
    print("-"*60)

    # Show first, middle, and last points
    indices = [0, len(thresholds)//4, len(thresholds)//2,
               3*len(thresholds)//4, -1]

    for idx in indices:
        if idx >= len(thresholds):
            continue
        print(f"{thresholds[idx]:<12.3f} {fpr[idx]:<10.3f} "
              f"{tpr[idx]:<10.3f} ({fpr[idx]:.3f}, {tpr[idx]:.3f})")

    print("-"*60)
    print(f"{'Very High':<12} {fpr[0]:<10.3f} {tpr[0]:<10.3f} "
          f"← Few positives predicted")
    print(f"{'Very Low':<12} {fpr[-1]:<10.3f} {tpr[-1]:<10.3f} "
          f"← Most predicted positive")

# Demonstrate
demonstrate_roc_thresholds(y_test, y_proba)
```

### Visualizing Different Classifiers

```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Train multiple models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42)
}

plt.figure(figsize=(10, 8))

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)

    # Predict probabilities
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Plot
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)

# Random classifier
plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)', linewidth=2)

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print AUC scores
print("\nAUC Scores:")
print("="*40)
for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"{name:<25} {auc:.4f}")
```

---

## Interpreting ROC-AUC

### What AUC Means

**Probabilistic Interpretation:**
AUC = Probability that the model ranks a random positive example higher than a random negative example.

```python
def explain_auc(y_true, y_proba):
    """Demonstrate probabilistic interpretation of AUC."""
    # Calculate AUC
    auc = roc_auc_score(y_true, y_proba)

    # Manual calculation: randomly sample pairs
    n_samples = 10000
    correct_rankings = 0

    pos_indices = np.where(y_true == 1)[0]
    neg_indices = np.where(y_true == 0)[0]

    for _ in range(n_samples):
        pos_idx = np.random.choice(pos_indices)
        neg_idx = np.random.choice(neg_indices)

        if y_proba[pos_idx] > y_proba[neg_idx]:
            correct_rankings += 1

    empirical_auc = correct_rankings / n_samples

    print(f"Sklearn AUC: {auc:.4f}")
    print(f"Empirical AUC (from random pairs): {empirical_auc:.4f}")
    print(f"\nInterpretation: {auc*100:.1f}% chance that a random positive")
    print(f"example is ranked higher than a random negative example.")

explain_auc(y_test, y_proba)
```

### AUC Score Interpretation

| AUC Range | Interpretation | Quality |
|-----------|---------------|---------|
| 0.90 - 1.00 | Excellent | Outstanding discrimination |
| 0.80 - 0.90 | Good | Acceptable discrimination |
| 0.70 - 0.80 | Fair | Poor discrimination |
| 0.60 - 0.70 | Poor | Minimal discrimination |
| 0.50 - 0.60 | Fail | No discrimination |

---

## Threshold Selection

### Finding the Optimal Threshold

```python
from sklearn.metrics import precision_recall_curve, f1_score

def find_optimal_threshold(y_true, y_proba, metric='f1'):
    """Find optimal classification threshold."""
    # Calculate metrics for different thresholds
    thresholds = np.linspace(0, 1, 100)
    scores = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'youden':
            # Youden's J statistic = TPR - FPR
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            idx = np.argmax(tpr - fpr)
            return thresholds[idx], tpr[idx] - fpr[idx]

        scores.append(score)

    # Find best threshold
    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx]
    best_score = scores[best_idx]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, scores, linewidth=2)
    plt.axvline(best_threshold, color='r', linestyle='--',
                label=f'Best Threshold = {best_threshold:.3f}')
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel(f'{metric.upper()} Score', fontsize=12)
    plt.title(f'Threshold vs {metric.upper()} Score', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return best_threshold, best_score

# Find optimal threshold
best_threshold, best_f1 = find_optimal_threshold(y_test, y_proba, metric='f1')
print(f"Optimal Threshold: {best_threshold:.3f}")
print(f"F1-Score at optimal threshold: {best_f1:.3f}")

# Compare with default threshold (0.5)
y_pred_default = (y_proba >= 0.5).astype(int)
y_pred_optimal = (y_proba >= best_threshold).astype(int)

from sklearn.metrics import classification_report

print("\nDefault Threshold (0.5):")
print(classification_report(y_test, y_pred_default))

print(f"\nOptimal Threshold ({best_threshold:.3f}):")
print(classification_report(y_test, y_pred_optimal))
```

### Threshold Selection Strategies

```python
def compare_threshold_strategies(y_true, y_proba):
    """Compare different threshold selection strategies."""
    from sklearn.metrics import precision_score, recall_score, f1_score

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)

    strategies = {
        'Default (0.5)': 0.5,
        'Youden Index': thresholds[np.argmax(tpr - fpr)],
        'Max F1': None,  # Calculated below
        'High Precision (90%)': None,  # Calculated below
        'High Recall (90%)': None  # Calculated below
    }

    # Find threshold for max F1
    f1_scores = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1_scores.append(f1_score(y_true, y_pred))
    strategies['Max F1'] = thresholds[np.argmax(f1_scores)]

    # Find threshold for high precision
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        if precision_score(y_true, y_pred) >= 0.90:
            strategies['High Precision (90%)'] = threshold
            break

    # Find threshold for high recall
    for threshold in reversed(thresholds):
        y_pred = (y_proba >= threshold).astype(int)
        if recall_score(y_true, y_pred) >= 0.90:
            strategies['High Recall (90%)'] = threshold
            break

    # Compare strategies
    print("Threshold Selection Strategies:")
    print("="*80)
    print(f"{'Strategy':<25} {'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-"*80)

    for strategy, threshold in strategies.items():
        if threshold is not None:
            y_pred = (y_proba >= threshold).astype(int)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            print(f"{strategy:<25} {threshold:<12.3f} {precision:<12.3f} "
                  f"{recall:<12.3f} {f1:<12.3f}")

compare_threshold_strategies(y_test, y_proba)
```

---

## ROC-AUC vs Precision-Recall Curve

### When to Use Each

**ROC-AUC:**
- Balanced datasets
- Care about both FPR and TPR
- Want threshold-independent metric

**PR Curve (Precision-Recall):**
- Imbalanced datasets
- Positive class is rare and important
- Focus on positive predictions

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

# Create imbalanced dataset
X_imb, y_imb = make_classification(
    n_samples=1000, n_features=20,
    weights=[0.95, 0.05],  # 95% negative, 5% positive
    random_state=42
)

X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
    X_imb, y_imb, test_size=0.3, random_state=42, stratify=y_imb
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_imb, y_train_imb)
y_proba_imb = model.predict_proba(X_test_imb)[:, 1]

# Calculate both curves
fpr, tpr, _ = roc_curve(y_test_imb, y_proba_imb)
precision, recall, _ = precision_recall_curve(y_test_imb, y_proba_imb)

roc_auc = roc_auc_score(y_test_imb, y_proba_imb)
pr_auc = average_precision_score(y_test_imb, y_proba_imb)

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC Curve
axes[0].plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.3f}')
axes[0].plot([0, 1], [0, 1], 'k--', linewidth=2)
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)
axes[0].set_title('ROC Curve\n(Looks good despite imbalance!)', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# PR Curve
baseline = y_test_imb.sum() / len(y_test_imb)
axes[1].plot(recall, precision, linewidth=2, label=f'AP = {pr_auc:.3f}')
axes[1].axhline(baseline, color='k', linestyle='--', linewidth=2,
                label=f'Baseline = {baseline:.3f}')
axes[1].set_xlabel('Recall', fontsize=12)
axes[1].set_ylabel('Precision', fontsize=12)
axes[1].set_title('PR Curve\n(More realistic for imbalanced data)', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Class distribution: {np.bincount(y_test_imb)}")
print(f"ROC-AUC: {roc_auc:.4f} (looks great!)")
print(f"PR-AUC: {pr_auc:.4f} (more realistic)")
```

### Comparison Table

| Aspect | ROC-AUC | PR Curve |
|--------|---------|----------|
| X-axis | False Positive Rate | Recall |
| Y-axis | True Positive Rate | Precision |
| Baseline | 0.5 (diagonal) | Positive class frequency |
| Imbalanced data | Can be optimistic | More informative |
| Interpretation | FPR vs TPR trade-off | Precision vs Recall trade-off |
| Use when | Balanced classes | Imbalanced classes |

---

## Multi-Class ROC-AUC

### One-vs-Rest (OvR) Strategy

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier

# Load multi-class dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# Binarize labels for multi-class ROC
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

# Train model with One-vs-Rest
model = OneVsRestClassifier(RandomForestClassifier(random_state=42))
model.fit(X_train, y_train)
y_proba_multi = model.predict_proba(X_test)

# Calculate ROC curve for each class
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Individual class ROC curves
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba_multi[:, i])
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, linewidth=2,
                 label=f'{iris.target_names[i]} (AUC = {roc_auc:.3f})')

axes[0].plot([0, 1], [0, 1], 'k--', linewidth=2)
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)
axes[0].set_title('Multi-Class ROC (One-vs-Rest)', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Plot 2: Micro and Macro average
# Micro-average (aggregate all classes)
fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_proba_multi.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

# Macro-average (average of individual curves)
all_fpr = np.unique(np.concatenate([roc_curve(y_test_bin[:, i], y_proba_multi[:, i])[0]
                                     for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba_multi[:, i])
    mean_tpr += np.interp(all_fpr, fpr, tpr)
mean_tpr /= n_classes
roc_auc_macro = auc(all_fpr, mean_tpr)

axes[1].plot(fpr_micro, tpr_micro, linewidth=2,
             label=f'Micro-average (AUC = {roc_auc_micro:.3f})')
axes[1].plot(all_fpr, mean_tpr, linewidth=2, linestyle='--',
             label=f'Macro-average (AUC = {roc_auc_macro:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', linewidth=2)
axes[1].set_xlabel('False Positive Rate', fontsize=12)
axes[1].set_ylabel('True Positive Rate', fontsize=12)
axes[1].set_title('Averaged Multi-Class ROC', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Micro-average AUC: {roc_auc_micro:.4f}")
print(f"Macro-average AUC: {roc_auc_macro:.4f}")
```

### Multi-Class AUC with sklearn

```python
from sklearn.metrics import roc_auc_score

# Automatic multi-class AUC calculation
# One-vs-Rest (OvR)
auc_ovr = roc_auc_score(y_test, y_proba_multi, multi_class='ovr', average='macro')
print(f"Multi-class AUC (OvR, macro): {auc_ovr:.4f}")

# One-vs-One (OvO)
auc_ovo = roc_auc_score(y_test, y_proba_multi, multi_class='ovo', average='macro')
print(f"Multi-class AUC (OvO, macro): {auc_ovo:.4f}")

# Weighted average (by class support)
auc_weighted = roc_auc_score(y_test, y_proba_multi, multi_class='ovr', average='weighted')
print(f"Multi-class AUC (OvR, weighted): {auc_weighted:.4f}")
```

---

## Common Mistakes

### 1. Using Predicted Labels Instead of Probabilities

**WRONG:**
```python
# Using hard predictions (0/1) - ROC will have only 2 points!
y_pred = model.predict(X_test)  # WRONG!
roc_auc = roc_auc_score(y_test, y_pred)  # Not meaningful
```

**CORRECT:**
```python
# Use probability predictions
y_proba = model.predict_proba(X_test)[:, 1]  # CORRECT!
roc_auc = roc_auc_score(y_test, y_proba)
```

### 2. Using ROC-AUC for Imbalanced Data Without Caution

**Problem:** ROC-AUC can be optimistic for imbalanced datasets.

**Solution:** Also check Precision-Recall AUC.

```python
# For imbalanced data, check both
roc_auc = roc_auc_score(y_test_imb, y_proba_imb)
pr_auc = average_precision_score(y_test_imb, y_proba_imb)

print(f"ROC-AUC: {roc_auc:.3f} (may be optimistic)")
print(f"PR-AUC: {pr_auc:.3f} (more realistic)")
```

### 3. Comparing AUC Without Confidence Intervals

**Problem:** Not accounting for uncertainty in AUC estimates.

**Solution:** Use bootstrap or cross-validation.

```python
from sklearn.utils import resample

def bootstrap_auc(y_true, y_proba, n_bootstrap=1000):
    """Calculate bootstrap confidence interval for AUC."""
    auc_scores = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = resample(range(len(y_true)), n_samples=len(y_true))
        auc = roc_auc_score(y_true[indices], y_proba[indices])
        auc_scores.append(auc)

    auc_scores = np.array(auc_scores)
    mean_auc = auc_scores.mean()
    ci_lower = np.percentile(auc_scores, 2.5)
    ci_upper = np.percentile(auc_scores, 97.5)

    return mean_auc, ci_lower, ci_upper

mean_auc, ci_lower, ci_upper = bootstrap_auc(y_test, y_proba)
print(f"AUC: {mean_auc:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
```

---

## When to Use ROC-AUC

### Use ROC-AUC When:
- Balanced class distribution
- Both TPR and FPR are important
- Comparing models (threshold-independent)
- Need single metric for model selection
- Probability calibration is good

### Don't Use ROC-AUC (or use with caution) When:
- Severely imbalanced datasets (use PR-AUC)
- Only care about positive class performance
- FPR not meaningful (use precision/recall)
- Model outputs poorly calibrated probabilities

---

## Related Topics

- [Classification Metrics](classification-metrics.md) - Other classification metrics
- [Confusion Matrix](confusion-matrix.md) - Understanding TP, FP, TN, FN
- [Model Selection](model-selection.md) - Using ROC-AUC for selection
- [Cross-Validation](cross-validation.md) - Robust AUC estimation

---

## Summary

**Key Takeaways:**
1. ROC-AUC evaluates **threshold-independent** performance
2. Always use **probability predictions**, not hard labels
3. AUC = **probability of correct ranking** (positive > negative)
4. For **imbalanced data**, also check **PR-AUC**
5. Use **threshold selection** to optimize for specific needs
6. Report **confidence intervals** for AUC
7. ROC-AUC is great for **model comparison**

ROC-AUC is one of the most comprehensive metrics for evaluating classifier performance!

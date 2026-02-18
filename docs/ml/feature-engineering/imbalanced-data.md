# Imbalanced Data

## Overview

Imbalanced data occurs when the classes in a classification problem are not represented equally. This is extremely common in real-world applications like fraud detection, disease diagnosis, and anomaly detection. Standard machine learning algorithms often perform poorly on imbalanced datasets, requiring specialized techniques.

**Key Principle**: Accuracy is misleading for imbalanced data. A 99% accurate model that predicts "no fraud" for everything is useless if we need to catch the 1% fraudulent transactions.

## Why Imbalanced Data is a Problem

1. **Bias Toward Majority**: Models learn to predict the majority class
2. **Misleading Metrics**: High accuracy masks poor minority class performance
3. **Poor Generalization**: Fails to learn patterns in minority class
4. **Business Impact**: Missing rare events (fraud, disease) is costly
5. **Learning Difficulty**: Less data to learn minority class patterns

## Understanding the Problem

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Generate imbalanced dataset
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    weights=[0.99, 0.01],  # 99% class 0, 1% class 1
    random_state=42
)

print("Class distribution:")
unique, counts = np.unique(y, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"  Class {cls}: {count} ({count/len(y)*100:.1f}%)")

# Visualize class imbalance
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.bar(['Majority (0)', 'Minority (1)'], counts, color=['blue', 'red'])
plt.ylabel('Count')
plt.title('Class Distribution')

plt.subplot(1, 2, 2)
plt.pie(counts, labels=['Majority (0)', 'Minority (1)'], autopct='%1.1f%%',
        colors=['blue', 'red'])
plt.title('Class Proportion')

plt.tight_layout()
plt.show()

# Train naive model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# The problem with accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("Sounds good, right? But let's look deeper...")

# Confusion matrix reveals the truth
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Detailed metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Majority', 'Minority']))

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix: Naive Model on Imbalanced Data')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# Demonstrate the "predict all majority" baseline
y_pred_all_majority = np.zeros_like(y_test)
baseline_accuracy = accuracy_score(y_test, y_pred_all_majority)
print(f"\n'Predict all majority' baseline accuracy: {baseline_accuracy:.4f}")
print("Our model barely beats predicting everything as majority class!")
```

## Proper Evaluation Metrics

```python
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt

# Calculate various metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Need probability scores for ROC-AUC and PR-AUC
y_pred_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
pr_auc = average_precision_score(y_test, y_pred_proba)

print("=== Metrics for Imbalanced Data ===")
print(f"Accuracy:  {accuracy:.4f}  ← Misleading!")
print(f"Precision: {precision:.4f}  ← Of predicted positives, how many are correct?")
print(f"Recall:    {recall:.4f}  ← Of actual positives, how many did we catch?")
print(f"F1 Score:  {f1:.4f}  ← Harmonic mean of precision & recall")
print(f"ROC-AUC:   {roc_auc:.4f}  ← Area under ROC curve")
print(f"PR-AUC:    {pr_auc:.4f}  ← Area under Precision-Recall curve")

# Visualize metrics
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Metrics comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC']
values = [accuracy, precision, recall, f1, roc_auc, pr_auc]
colors = ['lightgray', 'green', 'green', 'green', 'green', 'green']

axes[0].barh(metrics, values, color=colors)
axes[0].set_xlabel('Score')
axes[0].set_title('Metrics Comparison')
axes[0].set_xlim([0, 1])
axes[0].axvline(x=0.5, color='r', linestyle='--', alpha=0.3)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
axes[2].plot(recall_vals, precision_vals, linewidth=2, label=f'PR (AUC = {pr_auc:.3f})')
axes[2].axhline(y=y_test.mean(), color='k', linestyle='--', label='Random')
axes[2].set_xlabel('Recall')
axes[2].set_ylabel('Precision')
axes[2].set_title('Precision-Recall Curve')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n=== Which Metrics to Use? ===")
print("• Precision: When false positives are costly (e.g., spam detection)")
print("• Recall: When false negatives are costly (e.g., disease detection)")
print("• F1: When you need balance between precision and recall")
print("• ROC-AUC: Good for comparing models, less affected by threshold")
print("• PR-AUC: Better than ROC-AUC for highly imbalanced data")
```

## Handling Techniques

### 1. Resampling Methods

#### Oversampling (Increase Minority Class)

```python
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from collections import Counter

print("Original distribution:", Counter(y_train))

# Random Oversampling
ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X_train, y_train)
print(f"\nRandom Oversampling: {Counter(y_ros)}")

# SMOTE (Synthetic Minority Oversampling Technique)
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
print(f"SMOTE: {Counter(y_smote)}")

# ADASYN (Adaptive Synthetic Sampling)
adasyn = ADASYN(random_state=42)
X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)
print(f"ADASYN: {Counter(y_adasyn)}")

# Visualize (using PCA for 2D projection)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_smote_pca = pca.transform(X_smote)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Original data
majority_mask = y_train == 0
minority_mask = y_train == 1
axes[0].scatter(X_train_pca[majority_mask, 0], X_train_pca[majority_mask, 1],
               alpha=0.3, label='Majority', s=10)
axes[0].scatter(X_train_pca[minority_mask, 0], X_train_pca[minority_mask, 1],
               alpha=0.6, label='Minority', s=20, color='red')
axes[0].set_title('Original Imbalanced Data')
axes[0].legend()

# After SMOTE
majority_mask_smote = y_smote == 0
minority_mask_smote = y_smote == 1
axes[1].scatter(X_smote_pca[majority_mask_smote, 0], X_smote_pca[majority_mask_smote, 1],
               alpha=0.3, label='Majority', s=10)
axes[1].scatter(X_smote_pca[minority_mask_smote, 0], X_smote_pca[minority_mask_smote, 1],
               alpha=0.6, label='Minority (includes synthetic)', s=20, color='red')
axes[1].set_title('After SMOTE')
axes[1].legend()

plt.tight_layout()
plt.show()

# Compare model performance
models_results = []

for name, X_resampled, y_resampled in [
    ('Original', X_train, y_train),
    ('Random Oversampling', X_ros, y_ros),
    ('SMOTE', X_smote, y_smote),
    ('ADASYN', X_adasyn, y_adasyn)
]:
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_resampled, y_resampled)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    models_results.append({
        'Method': name,
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    })

results_df = pd.DataFrame(models_results)
print("\n=== Oversampling Methods Comparison ===")
print(results_df.to_string(index=False))
```

#### Undersampling (Decrease Majority Class)

```python
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks

print("Original distribution:", Counter(y_train))

# Random Undersampling
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X_train, y_train)
print(f"\nRandom Undersampling: {Counter(y_rus)}")

# NearMiss
nm = NearMiss(version=1)
X_nm, y_nm = nm.fit_resample(X_train, y_train)
print(f"NearMiss: {Counter(y_nm)}")

# Tomek Links (removes noisy samples)
tl = TomekLinks()
X_tl, y_tl = tl.fit_resample(X_train, y_train)
print(f"Tomek Links: {Counter(y_tl)}")

# Compare performance
undersampling_results = []

for name, X_resampled, y_resampled in [
    ('Original', X_train, y_train),
    ('Random Undersampling', X_rus, y_rus),
    ('NearMiss', X_nm, y_nm),
    ('Tomek Links', X_tl, y_tl)
]:
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_resampled, y_resampled)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    undersampling_results.append({
        'Method': name,
        'Training Samples': len(y_resampled),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    })

results_df = pd.DataFrame(undersampling_results)
print("\n=== Undersampling Methods Comparison ===")
print(results_df.to_string(index=False))
```

#### Combined Methods

```python
from imblearn.combine import SMOTETomek, SMOTEENN

# SMOTETomek: SMOTE + Tomek links
smote_tomek = SMOTETomek(random_state=42)
X_st, y_st = smote_tomek.fit_resample(X_train, y_train)
print(f"SMOTETomek: {Counter(y_st)}")

# SMOTEENN: SMOTE + Edited Nearest Neighbors
smote_enn = SMOTEENN(random_state=42)
X_se, y_se = smote_enn.fit_resample(X_train, y_train)
print(f"SMOTEENN: {Counter(y_se)}")
```

### 2. Class Weights

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

# Compute class weights
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))

print("Class weights:")
print(class_weight_dict)
print(f"Minority class weight is {class_weights[1]/class_weights[0]:.1f}x higher")

# Compare models with and without class weights
models = [
    ('Logistic Regression', LogisticRegression(random_state=42)),
    ('Logistic (weighted)', LogisticRegression(class_weight='balanced', random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('Random Forest (weighted)', RandomForestClassifier(class_weight='balanced', random_state=42)),
]

class_weight_results = []

for name, model in models:
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    class_weight_results.append({
        'Model': name,
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    })

results_df = pd.DataFrame(class_weight_results)
print("\n=== Class Weights Comparison ===")
print(results_df.to_string(index=False))

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

metrics = ['Precision', 'Recall', 'F1', 'ROC-AUC']
x = np.arange(len(metrics))
width = 0.2

for i, (name, _) in enumerate(models):
    values = [class_weight_results[i][metric] for metric in metrics]
    axes[0].bar(x + i*width, values, width, label=name)

axes[0].set_xlabel('Metrics')
axes[0].set_ylabel('Score')
axes[0].set_title('Class Weights Impact on Performance')
axes[0].set_xticks(x + width * 1.5)
axes[0].set_xticklabels(metrics)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ROC curves
for name, model in models:
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[1].plot(fpr, tpr, label=name, linewidth=2)

axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curves Comparison')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 3. Threshold Adjustment

```python
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Get probability predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate precision and recall for different thresholds
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Find optimal threshold (maximize F1)
f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last element
optimal_threshold = thresholds[optimal_idx]

print(f"Default threshold: 0.5")
print(f"Optimal threshold (max F1): {optimal_threshold:.3f}")

# Compare predictions at different thresholds
thresholds_to_test = [0.1, 0.3, 0.5, 0.7, 0.9, optimal_threshold]

threshold_results = []
for threshold in thresholds_to_test:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)

    threshold_results.append({
        'Threshold': threshold,
        'Precision': precision_score(y_test, y_pred_threshold),
        'Recall': recall_score(y_test, y_pred_threshold),
        'F1': f1_score(y_test, y_pred_threshold)
    })

results_df = pd.DataFrame(threshold_results)
print("\n=== Threshold Impact ===")
print(results_df.to_string(index=False))

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Precision-Recall vs Threshold
axes[0].plot(thresholds, precision_vals[:-1], label='Precision', linewidth=2)
axes[0].plot(thresholds, recall_vals[:-1], label='Recall', linewidth=2)
axes[0].plot(thresholds[:-1], f1_scores[:-1], label='F1', linewidth=2)
axes[0].axvline(x=optimal_threshold, color='r', linestyle='--',
               label=f'Optimal: {optimal_threshold:.3f}')
axes[0].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default: 0.5')
axes[0].set_xlabel('Threshold')
axes[0].set_ylabel('Score')
axes[0].set_title('Metrics vs Threshold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Metrics comparison at different thresholds
axes[1].plot(results_df['Threshold'], results_df['Precision'], 'o-', label='Precision')
axes[1].plot(results_df['Threshold'], results_df['Recall'], 's-', label='Recall')
axes[1].plot(results_df['Threshold'], results_df['F1'], '^-', label='F1')
axes[1].set_xlabel('Threshold')
axes[1].set_ylabel('Score')
axes[1].set_title('Threshold Selection Impact')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 4. Ensemble Methods

```python
from sklearn.ensemble import RandomForestClassifier, BalancedRandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier, EasyEnsembleClassifier, RUSBoostClassifier

# Standard Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Balanced Random Forest (built-in class balancing)
brf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)

# Balanced Bagging
bb = BalancedBaggingClassifier(
    estimator=LogisticRegression(),
    n_estimators=10,
    random_state=42
)

# Easy Ensemble
ee = EasyEnsembleClassifier(n_estimators=10, random_state=42)

# RUSBoost
rus_boost = RUSBoostClassifier(n_estimators=10, random_state=42)

ensemble_models = [
    ('Random Forest', rf),
    ('Balanced Random Forest', brf),
    ('Balanced Bagging', bb),
    ('Easy Ensemble', ee),
    ('RUSBoost', rus_boost)
]

ensemble_results = []

for name, model in ensemble_models:
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    ensemble_results.append({
        'Model': name,
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    })

results_df = pd.DataFrame(ensemble_results)
print("\n=== Ensemble Methods Comparison ===")
print(results_df.to_string(index=False))

# Visualize
plt.figure(figsize=(12, 6))
x = np.arange(len(ensemble_models))
width = 0.2

metrics = ['Precision', 'Recall', 'F1', 'ROC-AUC']
for i, metric in enumerate(metrics):
    values = [result[metric] for result in ensemble_results]
    plt.bar(x + i*width, values, width, label=metric)

plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Ensemble Methods Comparison')
plt.xticks(x + width * 1.5, [name for name, _ in ensemble_models], rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
```

## Comprehensive Comparison

```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import time

# Collect all methods
all_methods = [
    ('Original (No handling)', X_train, y_train),
    ('Random Oversampling', X_ros, y_ros),
    ('SMOTE', X_smote, y_smote),
    ('Random Undersampling', X_rus, y_rus),
]

# Base model
base_model = LogisticRegression(random_state=42, max_iter=1000)

final_results = []

for name, X_method, y_method in all_methods:
    start = time.time()

    # Train and evaluate
    base_model.fit(X_method, y_method)
    y_pred = base_model.predict(X_test)
    y_pred_proba = base_model.predict_proba(X_test)[:, 1]

    training_time = time.time() - start

    final_results.append({
        'Method': name,
        'Training Samples': len(y_method),
        'Training Time (s)': training_time,
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    })

# Add class weight method
model_weighted = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
start = time.time()
model_weighted.fit(X_train, y_train)
y_pred = model_weighted.predict(X_test)
y_pred_proba = model_weighted.predict_proba(X_test)[:, 1]
training_time = time.time() - start

final_results.append({
    'Method': 'Class Weights',
    'Training Samples': len(y_train),
    'Training Time (s)': training_time,
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1': f1_score(y_test, y_pred),
    'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
})

results_df = pd.DataFrame(final_results)
print("\n=== COMPREHENSIVE COMPARISON ===")
print(results_df.to_string(index=False))

# Find best method for each metric
print("\n=== Best Method for Each Metric ===")
for metric in ['Precision', 'Recall', 'F1', 'ROC-AUC']:
    best_idx = results_df[metric].idxmax()
    print(f"{metric}: {results_df.loc[best_idx, 'Method']} ({results_df.loc[best_idx, metric]:.4f})")
```

## Best Practices

1. **Understand the Problem**: Know which errors are more costly
2. **Use Appropriate Metrics**: Never rely solely on accuracy
3. **Try Multiple Techniques**: Different problems need different solutions
4. **Stratified Split**: Maintain class distribution in train/test split
5. **Cross-Validation**: Use stratified k-fold cross-validation
6. **Start Simple**: Begin with class weights before complex resampling
7. **Validate on Original Distribution**: Test on imbalanced test set
8. **Monitor Both Classes**: Track performance on both majority and minority

## Common Mistakes

1. **Using Accuracy**: Misleading metric for imbalanced data
2. **Oversampling Before Split**: Causes data leakage
3. **Balancing Test Set**: Test set should reflect real-world distribution
4. **Ignoring Cost**: Not considering business cost of errors
5. **Over-balancing**: Perfectly balanced data may not be optimal
6. **Not Trying Multiple Methods**: One method doesn't fit all
7. **Forgetting Threshold Tuning**: Default 0.5 may not be optimal

## Decision Guide

```python
def recommend_imbalance_handling(imbalance_ratio, dataset_size, computational_budget):
    """
    Recommend approach based on problem characteristics

    Parameters:
    - imbalance_ratio: majority_count / minority_count
    - dataset_size: total number of samples
    - computational_budget: 'low', 'medium', 'high'
    """

    print("=== Imbalanced Data Handling Recommendations ===\n")

    print(f"Imbalance Ratio: {imbalance_ratio:.1f}:1")
    print(f"Dataset Size: {dataset_size}")
    print(f"Computational Budget: {computational_budget}\n")

    if imbalance_ratio < 3:
        print("✓ MILD IMBALANCE (< 3:1)")
        print("  Recommendation: Class weights or no special handling")
        print("  Reason: Standard algorithms handle this well")

    elif imbalance_ratio < 10:
        print("✓ MODERATE IMBALANCE (3:1 to 10:1)")
        print("  Primary: Class weights")
        print("  Alternative: SMOTE if enough minority samples")
        print("  Reason: Balance between simplicity and effectiveness")

    elif imbalance_ratio < 100:
        print("✓ HIGH IMBALANCE (10:1 to 100:1)")
        if dataset_size < 10000:
            print("  Primary: SMOTE (generates synthetic samples)")
            print("  Alternative: Combined SMOTE + Tomek Links")
        else:
            print("  Primary: Random Undersampling + Ensemble")
            print("  Alternative: Balanced ensemble methods")
        print("  Reason: Need to address severe imbalance")

    else:
        print("✓ EXTREME IMBALANCE (> 100:1)")
        print("  Primary: Anomaly detection approaches")
        print("  Alternative: Cost-sensitive learning with heavy weighting")
        print("  Consider: Treating as anomaly detection problem")
        print("  Reason: Standard classification may not work well")

    print("\n=== Computational Considerations ===")
    if computational_budget == 'low':
        print("• Use class weights (fast, no resampling)")
        print("• Avoid wrapper methods like RFE")
    elif computational_budget == 'medium':
        print("• Try SMOTE or random undersampling")
        print("• Use balanced ensemble methods")
    else:  # high
        print("• Experiment with all methods")
        print("• Use RFECV and complex ensembles")
        print("• Try nested cross-validation")

    print("\n=== Always Remember ===")
    print("• Use stratified splitting")
    print("• Evaluate with precision, recall, F1, and ROC-AUC")
    print("• Test on original (imbalanced) distribution")
    print("• Consider business cost of errors")

# Example usage
# recommend_imbalance_handling(imbalance_ratio=99, dataset_size=10000, computational_budget='medium')
```

## Related Topics

- [Feature Selection](./feature-selection.md) - Select features for imbalanced data
- [Model Evaluation](../evaluation/index.md) - Proper evaluation metrics
- Classification - Classification algorithms

## Summary

Handling imbalanced data requires understanding the problem domain, using appropriate evaluation metrics, and applying suitable techniques like resampling, class weights, or specialized algorithms. Never rely on accuracy alone. The best approach depends on the severity of imbalance, dataset size, and business costs of different types of errors. Always validate on the original distribution and focus on metrics that matter for your specific use case.

---

*Last Updated: 2026-02-09*

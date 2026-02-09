# Model Selection

## Overview

Model selection is the process of choosing the best model from a set of candidates. This involves comparing models fairly, using appropriate metrics, and validating that the selected model generalizes well to unseen data.

**What You'll Learn:**
- How to compare models fairly
- Statistical tests for model comparison
- Bias-variance trade-off considerations
- Cross-validated model selection
- Best practices for model selection

**Prerequisites:** Understanding of cross-validation and evaluation metrics

---

## The Model Selection Process

### Standard Workflow

```
1. Define Problem & Metrics
          ↓
2. Prepare Data (train/val/test split)
          ↓
3. Train Multiple Models
          ↓
4. Evaluate with Cross-Validation
          ↓
5. Compare Performance
          ↓
6. Select Best Model
          ↓
7. Final Evaluation on Test Set
```

### Complete Example

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split: Development (train+val) and Test
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Development set: {len(X_dev)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Class distribution: {np.bincount(y_dev)}")

# Define models with preprocessing pipelines
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ]),
    'Decision Tree': Pipeline([
        ('clf', DecisionTreeClassifier(random_state=42, max_depth=5))
    ]),
    'Random Forest': Pipeline([
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    'Gradient Boosting': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(random_state=42))
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(probability=True, random_state=42))
    ]),
    'K-Nearest Neighbors': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=5))
    ])
}

# Define metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate all models
results = []

print("\n" + "="*80)
print("MODEL EVALUATION WITH CROSS-VALIDATION")
print("="*80)

for name, model in models.items():
    print(f"\nEvaluating {name}...")

    # Cross-validation
    cv_results = cross_validate(
        model, X_dev, y_dev,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    # Store results
    result = {'Model': name}
    for metric in scoring.keys():
        result[f'{metric}_mean'] = cv_results[f'test_{metric}'].mean()
        result[f'{metric}_std'] = cv_results[f'test_{metric}'].std()
        result[f'train_{metric}_mean'] = cv_results[f'train_{metric}'].mean()

    results.append(result)

    # Print summary
    print(f"  ROC-AUC: {result['roc_auc_mean']:.3f} (±{result['roc_auc_std']:.3f})")
    print(f"  F1-Score: {result['f1_mean']:.3f} (±{result['f1_std']:.3f})")

# Create results DataFrame
df_results = pd.DataFrame(results)

# Display sorted by ROC-AUC
print("\n" + "="*80)
print("RESULTS SUMMARY (sorted by ROC-AUC)")
print("="*80)
df_sorted = df_results.sort_values('roc_auc_mean', ascending=False)
print(df_sorted[['Model', 'roc_auc_mean', 'roc_auc_std', 'f1_mean', 'f1_std']].to_string(index=False))

# Select best model
best_model_name = df_sorted.iloc[0]['Model']
best_model = models[best_model_name]

print(f"\n{'='*80}")
print(f"BEST MODEL: {best_model_name}")
print(f"{'='*80}")

# Train on full development set
best_model.fit(X_dev, y_dev)

# Final evaluation on held-out test set
from sklearn.metrics import classification_report, roc_auc_score

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_proba)

print("\nFinal Test Set Performance:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {test_auc:.4f}")
```

---

## Fair Model Comparison

### Key Principles

1. **Same data:** All models evaluated on identical data
2. **Same splits:** Use same cross-validation splits
3. **Same metrics:** Evaluate using same criteria
4. **Same preprocessing:** Fair comparison of architectures
5. **Statistical testing:** Account for variance

### Ensuring Fair Comparison

```python
from sklearn.model_selection import StratifiedKFold
import time

def fair_model_comparison(models, X, y, cv_splits=5, random_state=42):
    """
    Perform fair model comparison with same CV splits.
    """
    # Create CV splitter (same for all models)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    results = []

    for name, model in models.items():
        print(f"\nEvaluating {name}...")

        # Track time
        start_time = time.time()

        # Cross-validation with same splits
        cv_results = cross_validate(
            model, X, y,
            cv=cv,  # Same splits for all models!
            scoring='roc_auc',
            return_train_score=True,
            return_estimator=True
        )

        elapsed_time = time.time() - start_time

        # Calculate statistics
        test_scores = cv_results['test_score']
        train_scores = cv_results['train_score']

        results.append({
            'Model': name,
            'Test Mean': test_scores.mean(),
            'Test Std': test_scores.std(),
            'Train Mean': train_scores.mean(),
            'Overfitting Gap': train_scores.mean() - test_scores.mean(),
            'Time (s)': elapsed_time,
            'Scores': test_scores
        })

    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('Test Mean', ascending=False)

    return df

# Use it
df_comparison = fair_model_comparison(models, X_dev, y_dev)

print("\n" + "="*80)
print("FAIR MODEL COMPARISON")
print("="*80)
print(df_comparison[['Model', 'Test Mean', 'Test Std', 'Overfitting Gap', 'Time (s)']].to_string(index=False))
```

---

## Statistical Tests for Model Comparison

### Paired t-test

Use when comparing two models on same CV folds.

```python
from scipy import stats

def paired_ttest_comparison(scores1, scores2, model1_name, model2_name):
    """
    Perform paired t-test to compare two models.

    H0: No difference between models
    H1: Models are different
    """
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(scores1, scores2)

    print(f"\nPaired t-test: {model1_name} vs {model2_name}")
    print("="*60)
    print(f"{model1_name}: {scores1.mean():.4f} (±{scores1.std():.4f})")
    print(f"{model2_name}: {scores2.mean():.4f} (±{scores2.std():.4f})")
    print(f"Difference: {scores1.mean() - scores2.mean():.4f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")

    alpha = 0.05
    if p_value < alpha:
        winner = model1_name if scores1.mean() > scores2.mean() else model2_name
        print(f"\nResult: {winner} is significantly better (p < {alpha})")
    else:
        print(f"\nResult: No significant difference (p >= {alpha})")

    return t_stat, p_value

# Example: Compare top 2 models
model1_scores = df_comparison.iloc[0]['Scores']
model2_scores = df_comparison.iloc[1]['Scores']
model1_name = df_comparison.iloc[0]['Model']
model2_name = df_comparison.iloc[1]['Model']

paired_ttest_comparison(model1_scores, model2_scores, model1_name, model2_name)
```

### Multiple Model Comparison (Friedman Test + Post-hoc)

```python
from scipy.stats import friedmanchisquare
from itertools import combinations

def friedman_test_comparison(results_dict):
    """
    Friedman test for comparing multiple models.

    H0: All models perform equally
    H1: At least one model is different
    """
    # Extract scores for each model (rows=folds, cols=models)
    model_names = list(results_dict.keys())
    scores_matrix = np.array([results_dict[name] for name in model_names]).T

    # Friedman test
    statistic, p_value = friedmanchisquare(*scores_matrix.T)

    print("Friedman Test (Multiple Model Comparison)")
    print("="*60)
    print(f"Chi-square statistic: {statistic:.4f}")
    print(f"p-value: {p_value:.4f}")

    alpha = 0.05
    if p_value < alpha:
        print(f"\nResult: Models are significantly different (p < {alpha})")
        print("Proceed with post-hoc pairwise tests.")
    else:
        print(f"\nResult: No significant difference among models (p >= {alpha})")

    return statistic, p_value

# Prepare scores dictionary
scores_dict = {}
for _, row in df_comparison.iterrows():
    scores_dict[row['Model']] = row['Scores']

friedman_test_comparison(scores_dict)

# Post-hoc pairwise comparisons
print("\n" + "="*60)
print("POST-HOC PAIRWISE COMPARISONS")
print("="*60)

model_names = list(scores_dict.keys())
for model1, model2 in combinations(model_names[:3], 2):  # Compare top 3
    scores1 = scores_dict[model1]
    scores2 = scores_dict[model2]
    paired_ttest_comparison(scores1, scores2, model1, model2)
```

---

## Bias-Variance Trade-off

### Understanding Overfitting vs Underfitting

```python
import matplotlib.pyplot as plt

def plot_bias_variance_analysis(models, X_train, y_train, X_test, y_test):
    """
    Visualize bias-variance trade-off across models.
    """
    results = []

    for name, model in models.items():
        # Cross-validation on training set
        cv_results = cross_validate(
            model, X_train, y_train,
            cv=5,
            scoring='accuracy',
            return_train_score=True
        )

        train_score = cv_results['train_score'].mean()
        val_score = cv_results['test_score'].mean()

        # Final test score
        model.fit(X_train, y_train)
        test_score = model.score(X_test, y_test)

        results.append({
            'Model': name,
            'Train': train_score,
            'Validation': val_score,
            'Test': test_score,
            'Bias': 1 - val_score,  # Approximate
            'Variance': train_score - val_score  # Overfitting gap
        })

    df = pd.DataFrame(results)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Train vs Validation scores
    x = np.arange(len(df))
    width = 0.35

    axes[0].bar(x - width/2, df['Train'], width, label='Train', alpha=0.8)
    axes[0].bar(x + width/2, df['Validation'], width, label='Validation', alpha=0.8)
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Train vs Validation Accuracy', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df['Model'], rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Plot 2: Bias vs Variance
    axes[1].scatter(df['Bias'], df['Variance'], s=100, alpha=0.6)
    for idx, row in df.iterrows():
        axes[1].annotate(row['Model'], (row['Bias'], row['Variance']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)

    axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1].axvline(0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_xlabel('Bias (1 - Validation Score)')
    axes[1].set_ylabel('Variance (Train - Validation)')
    axes[1].set_title('Bias-Variance Trade-off', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Add quadrants
    axes[1].text(0.01, 0.01, 'Good Fit', transform=axes[1].transAxes,
                fontsize=10, alpha=0.5)
    axes[1].text(0.7, 0.01, 'High Bias', transform=axes[1].transAxes,
                fontsize=10, alpha=0.5)
    axes[1].text(0.01, 0.9, 'High Variance', transform=axes[1].transAxes,
                fontsize=10, alpha=0.5)

    plt.tight_layout()
    plt.show()

    return df

# Use it
df_bias_variance = plot_bias_variance_analysis(models, X_dev, y_dev, X_test, y_test)
print("\nBias-Variance Analysis:")
print(df_bias_variance.to_string(index=False))
```

---

## Visualization of Model Comparison

### Comprehensive Comparison Plot

```python
def plot_model_comparison(df_results):
    """
    Create comprehensive visualization of model comparison.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    models = df_results['Model'].values
    x = np.arange(len(models))

    # Plot 1: Test scores with error bars
    test_means = df_results['roc_auc_mean'].values
    test_stds = df_results['roc_auc_std'].values

    axes[0, 0].barh(x, test_means, xerr=test_stds, alpha=0.7, capsize=5)
    axes[0, 0].set_yticks(x)
    axes[0, 0].set_yticklabels(models)
    axes[0, 0].set_xlabel('ROC-AUC Score')
    axes[0, 0].set_title('Model Performance (with std)', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='x')

    # Plot 2: Multiple metrics comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_data = np.array([[df_results[df_results['Model']==model][f'{m}_mean'].values[0]
                             for m in metrics] for model in models])

    x_metrics = np.arange(len(metrics))
    width = 0.15
    for i, model in enumerate(models[:5]):  # Top 5 models
        axes[0, 1].bar(x_metrics + i*width, metric_data[i], width, label=model, alpha=0.8)

    axes[0, 1].set_xlabel('Metric')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Multiple Metrics Comparison', fontweight='bold')
    axes[0, 1].set_xticks(x_metrics + width * 2)
    axes[0, 1].set_xticklabels([m.capitalize() for m in metrics], rotation=45)
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Plot 3: Train vs Test (overfitting check)
    train_means = [df_results[df_results['Model']==model][f'train_roc_auc_mean'].values[0]
                   for model in models]
    test_means = df_results['roc_auc_mean'].values

    axes[1, 0].scatter(test_means, train_means, s=100, alpha=0.6)
    axes[1, 0].plot([0.5, 1.0], [0.5, 1.0], 'k--', alpha=0.3)  # Perfect fit line

    for i, model in enumerate(models):
        axes[1, 0].annotate(model, (test_means[i], train_means[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)

    axes[1, 0].set_xlabel('Test ROC-AUC')
    axes[1, 0].set_ylabel('Train ROC-AUC')
    axes[1, 0].set_title('Overfitting Analysis', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Score distribution (box plot)
    score_data = [df_results[df_results['Model']==model]['Scores'].values[0]
                  if 'Scores' in df_results.columns else []
                  for model in models]

    if score_data[0] is not None and len(score_data[0]) > 0:
        axes[1, 1].boxplot(score_data, labels=models, vert=True)
        axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
        axes[1, 1].set_ylabel('ROC-AUC Score')
        axes[1, 1].set_title('Score Distribution (CV Folds)', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

# Use it
plot_model_comparison(df_results)
```

---

## Best Practices

### 1. Always Hold Out a Test Set

```python
# Correct workflow
# Split 1: Separate test set (never touch until final evaluation)
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Split 2: Use CV on development set for model selection
cv_scores = cross_val_score(model, X_dev, y_dev, cv=5)

# Select best model based on CV scores
best_model.fit(X_dev, y_dev)

# Final evaluation on held-out test set (only once!)
final_score = best_model.score(X_test, y_test)
```

### 2. Use Nested Cross-Validation for Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Nested CV: Outer for evaluation, Inner for tuning
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Define parameter grid
param_grid = {
    'clf__n_estimators': [50, 100, 200],
    'clf__max_depth': [5, 10, None]
}

# Inner CV for hyperparameter tuning
model = Pipeline([
    ('clf', RandomForestClassifier(random_state=42))
])

grid_search = GridSearchCV(
    model, param_grid,
    cv=inner_cv,
    scoring='roc_auc'
)

# Outer CV for unbiased evaluation
nested_scores = cross_val_score(
    grid_search, X_dev, y_dev,
    cv=outer_cv,
    scoring='roc_auc'
)

print(f"Nested CV Score: {nested_scores.mean():.3f} (±{nested_scores.std():.3f})")
```

### 3. Report Multiple Metrics

```python
# Don't rely on single metric
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

for metric in scoring:
    scores = cross_val_score(model, X_dev, y_dev, cv=5, scoring=metric)
    print(f"{metric}: {scores.mean():.3f} (±{scores.std():.3f})")
```

### 4. Consider Computational Cost

```python
import time

def compare_models_with_time(models, X, y):
    """Compare models including training time."""
    results = []

    for name, model in models.items():
        start = time.time()
        scores = cross_val_score(model, X, y, cv=5)
        elapsed = time.time() - start

        results.append({
            'Model': name,
            'Score': scores.mean(),
            'Std': scores.std(),
            'Time (s)': elapsed,
            'Time per fold (s)': elapsed / 5
        })

    df = pd.DataFrame(results)
    df['Score/Time'] = df['Score'] / df['Time (s)']  # Efficiency metric

    return df.sort_values('Score', ascending=False)

# Consider both performance and speed
df_with_time = compare_models_with_time(models, X_dev, y_dev)
print(df_with_time.to_string(index=False))
```

---

## Model Selection Checklist

- [ ] Split data properly (train/val/test or dev/test with CV)
- [ ] Use same data and splits for all models
- [ ] Evaluate multiple metrics, not just one
- [ ] Use cross-validation for robust estimates
- [ ] Report mean and standard deviation
- [ ] Check for overfitting (train vs validation scores)
- [ ] Perform statistical tests when comparing models
- [ ] Consider computational cost and complexity
- [ ] Hold out final test set for unbiased evaluation
- [ ] Use nested CV if tuning hyperparameters during selection
- [ ] Visualize results for better understanding
- [ ] Document the selection process

---

## Common Mistakes

### 1. Selecting Model Based on Test Set

**WRONG:**
```python
# Evaluating multiple models on test set and selecting best
best_score = 0
for model in models:
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)  # WRONG! Using test set for selection
    if score > best_score:
        best_model = model
```

**CORRECT:**
```python
# Use CV on development set for selection
best_score = 0
for model in models:
    scores = cross_val_score(model, X_dev, y_dev, cv=5)
    if scores.mean() > best_score:
        best_model = model

# Evaluate selected model on test set only once
best_model.fit(X_dev, y_dev)
final_score = best_model.score(X_test, y_test)
```

### 2. Not Using Same CV Splits

**WRONG:**
```python
# Different random splits for each model
score1 = cross_val_score(model1, X, y, cv=5, random_state=42)
score2 = cross_val_score(model2, X, y, cv=5, random_state=123)  # Different splits!
```

**CORRECT:**
```python
# Same CV splitter for all models
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
score1 = cross_val_score(model1, X, y, cv=cv)
score2 = cross_val_score(model2, X, y, cv=cv)
```

### 3. Ignoring Statistical Significance

**Problem:** Selecting model with marginally better score without checking significance.

**Solution:** Use statistical tests.

```python
# Check if difference is significant
from scipy.stats import ttest_rel

t_stat, p_value = ttest_rel(scores1, scores2)
if p_value < 0.05:
    print("Difference is statistically significant")
else:
    print("No significant difference - choose simpler model")
```

---

## Related Topics

- [Cross-Validation](cross-validation.md) - Robust evaluation technique
- [Hyperparameter Tuning](../optimization/hyperparameter-tuning.md) - Optimizing selected model
- [Classification Metrics](classification-metrics.md) - Metrics for selection
- [Regression Metrics](regression-metrics.md) - Metrics for regression

---

## Summary

**Key Takeaways:**
1. Always **hold out a test set** for final evaluation
2. Use **cross-validation** on development set for selection
3. Ensure **fair comparison** (same data, same splits, same metrics)
4. Perform **statistical tests** to validate differences
5. Consider **bias-variance trade-off**
6. Report **multiple metrics** and confidence intervals
7. **Visualize** results for better understanding
8. Use **nested CV** when tuning hyperparameters

Model selection is not just about the highest score - it's about finding the model that generalizes best!

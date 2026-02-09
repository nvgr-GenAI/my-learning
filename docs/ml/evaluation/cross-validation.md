# Cross-Validation

## Overview

Cross-validation is a robust technique for evaluating model performance by training and testing on different subsets of data. It provides more reliable performance estimates than a single train/test split.

**What You'll Learn:**
- Different cross-validation techniques
- When to use each technique
- Implementation with sklearn
- Nested cross-validation for hyperparameter tuning
- Common pitfalls and how to avoid them

**Prerequisites:** Understanding of train/test split and overfitting

---

## Why Cross-Validation?

### The Problem with Single Train/Test Split

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

# Single split - results vary based on random_state!
for random_state in range(5):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    print(f"Random State {random_state}: {score:.3f}")
```

**Output:**
```
Random State 0: 1.000
Random State 1: 0.978
Random State 2: 0.978
Random State 3: 0.933
Random State 4: 0.956
```

**Problem:** Results vary significantly! Which score do you trust?

**Solution:** Use cross-validation to get robust estimates.

---

## K-Fold Cross-Validation

### Concept

Split data into K equal folds:
1. Train on K-1 folds
2. Validate on remaining fold
3. Repeat K times (each fold used once as validation)
4. Average the results

```
Fold 1: [Test][Train][Train][Train][Train]
Fold 2: [Train][Test][Train][Train][Train]
Fold 3: [Train][Train][Test][Train][Train]
Fold 4: [Train][Train][Train][Test][Train]
Fold 5: [Train][Train][Train][Train][Test]
```

### Implementation

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import numpy as np

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Initialize model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Method 1: Simple cross_val_score
scores = cross_val_score(model, X, y, cv=5)

print("K-Fold Cross-Validation (K=5)")
print(f"Scores: {scores}")
print(f"Mean: {scores.mean():.3f}")
print(f"Std: {scores.std():.3f}")
print(f"95% CI: [{scores.mean() - 2*scores.std():.3f}, "
      f"{scores.mean() + 2*scores.std():.3f}]")
```

**Output:**
```
K-Fold Cross-Validation (K=5)
Scores: [1.    0.967 0.933 0.933 1.   ]
Mean: 0.967
Std: 0.033
95% CI: [0.900, 1.033]
```

### Manual K-Fold for Understanding

```python
from sklearn.model_selection import KFold

# Manual K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    # Split data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train and evaluate
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    scores.append(score)
    print(f"Fold {fold}: {score:.3f} "
          f"(train size: {len(train_idx)}, test size: {len(test_idx)})")

print(f"\nMean: {np.mean(scores):.3f}")
print(f"Std: {np.std(scores):.3f}")
```

### Choosing K

| K Value | Pros | Cons | When to Use |
|---------|------|------|-------------|
| K=5 | Fast, good bias-variance balance | Moderate variance | Default choice |
| K=10 | Lower bias, more reliable | Slower | Standard datasets |
| K=n (LOO) | Lowest bias | Very slow, high variance | Small datasets |
| K=3 | Very fast | Higher variance | Quick prototyping |

**Rule of thumb:** K=5 or K=10 for most cases.

---

## Stratified K-Fold Cross-Validation

### Why Stratified?

**Problem:** Regular K-Fold may create imbalanced folds.

```python
# Imbalanced dataset example
y_imbalanced = np.array([0]*90 + [1]*10)

# Regular K-Fold (might create unbalanced folds)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kf.split(y_imbalanced), 1):
    print(f"Fold {fold} - Class 1 in test: {y_imbalanced[test_idx].sum()}")
```

**Output:**
```
Fold 1 - Class 1 in test: 3
Fold 2 - Class 1 in test: 1  # Imbalanced!
Fold 3 - Class 1 in test: 2
Fold 4 - Class 1 in test: 2
Fold 5 - Class 1 in test: 2
```

### Solution: Stratified K-Fold

**Stratified K-Fold** maintains class distribution in each fold.

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Stratified K-Fold:")
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_imbalanced), 1):
    train_dist = np.bincount(y_imbalanced[train_idx])
    test_dist = np.bincount(y_imbalanced[test_idx])

    print(f"Fold {fold}:")
    print(f"  Train: {train_dist} ({train_dist[1]/len(train_idx)*100:.1f}% class 1)")
    print(f"  Test:  {test_dist} ({test_dist[1]/len(test_idx)*100:.1f}% class 1)")

# Use with cross_val_score
scores = cross_val_score(model, X, y_imbalanced, cv=skf)
print(f"\nMean Score: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

**When to Use Stratified K-Fold:**
- Classification problems (always!)
- Imbalanced datasets (essential!)
- Multi-class classification (maintains all class proportions)

---

## Leave-One-Out Cross-Validation (LOO)

### Concept

- K = number of samples
- Each sample used once as validation
- Most computationally expensive
- Lowest bias, but high variance

```python
from sklearn.model_selection import LeaveOneOut

# Leave-One-Out CV
loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)

print(f"LOO CV (n={len(scores)} iterations)")
print(f"Mean: {scores.mean():.3f}")
print(f"Std: {scores.std():.3f}")
print(f"Accuracy: {scores.sum()}/{len(scores)} = {scores.mean():.3f}")
```

**When to Use:**
- Very small datasets (n < 100)
- Need lowest bias estimate
- Computational cost is acceptable

**When NOT to Use:**
- Large datasets (too slow)
- Need fast iterations
- High variance is a concern

---

## Time Series Cross-Validation

### The Problem with Regular CV on Time Series

**WRONG:**
```python
# Regular K-Fold on time series - WRONG!
# This leaks future information into the past!
kf = KFold(n_splits=5, shuffle=True)  # shuffle=True is WRONG for time series
```

### Solution: TimeSeriesSplit

**Maintains temporal order:**

```
Fold 1: [Train            ][Test]
Fold 2: [Train                 ][Test]
Fold 3: [Train                      ][Test]
Fold 4: [Train                           ][Test]
Fold 5: [Train                                ][Test]
```

```python
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import matplotlib.pyplot as plt

# Create time series data
dates = pd.date_range('2020-01-01', periods=100, freq='D')
y_ts = np.sin(np.arange(100) * 0.1) + np.random.randn(100) * 0.1

# Time Series Split
tscv = TimeSeriesSplit(n_splits=5)

plt.figure(figsize=(14, 8))
for fold, (train_idx, test_idx) in enumerate(tscv.split(y_ts), 1):
    plt.subplot(5, 1, fold)
    plt.plot(train_idx, y_ts[train_idx], 'b-', label='Train', alpha=0.6)
    plt.plot(test_idx, y_ts[test_idx], 'r-', label='Test', alpha=0.6)
    plt.title(f'Fold {fold}')
    plt.legend(loc='upper left')
    plt.tight_layout()

plt.show()

# Evaluate with TimeSeriesSplit
scores = cross_val_score(model, X_ts, y_ts, cv=tscv)
print(f"Time Series CV Mean: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

**Key Points:**
- Never shuffle time series data
- Training set always precedes test set
- Each fold adds more historical data
- Test set simulates future predictions

### Advanced: Rolling Window CV

```python
def rolling_window_cv(X, y, window_size, n_splits):
    """Rolling window cross-validation for time series."""
    scores = []
    total_size = len(X)

    for i in range(n_splits):
        # Define train and test indices
        test_start = window_size + i * (total_size - window_size) // n_splits
        test_end = test_start + (total_size - window_size) // n_splits

        train_idx = np.arange(test_start - window_size, test_start)
        test_idx = np.arange(test_start, test_end)

        # Train and evaluate
        model = RandomForestRegressor(random_state=42)
        model.fit(X[train_idx], y[train_idx])
        score = model.score(X[test_idx], y[test_idx])

        scores.append(score)
        print(f"Fold {i+1}: Score = {score:.3f}, "
              f"Train [{train_idx[0]}:{train_idx[-1]}], "
              f"Test [{test_idx[0]}:{test_idx[-1]}]")

    return np.array(scores)

# Example usage
X_ts = np.arange(1000).reshape(-1, 1)
y_ts = np.sin(X_ts.ravel() * 0.01) + np.random.randn(1000) * 0.1

scores = rolling_window_cv(X_ts, y_ts, window_size=200, n_splits=4)
print(f"\nRolling Window CV Mean: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

---

## Multiple Metrics with Cross-Validation

```python
from sklearn.model_selection import cross_validate
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

# Load data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Define multiple scoring metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

# Cross-validation with multiple metrics
model = RandomForestClassifier(random_state=42)
cv_results = cross_validate(
    model, X, y,
    cv=5,
    scoring=scoring,
    return_train_score=True
)

# Display results
print("Cross-Validation Results:")
print("="*60)
for metric in scoring.keys():
    train_scores = cv_results[f'train_{metric}']
    test_scores = cv_results[f'test_{metric}']

    print(f"\n{metric.upper()}:")
    print(f"  Train: {train_scores.mean():.3f} (+/- {train_scores.std():.3f})")
    print(f"  Test:  {test_scores.mean():.3f} (+/- {test_scores.std():.3f})")

    # Check for overfitting
    gap = train_scores.mean() - test_scores.mean()
    if gap > 0.05:
        print(f"  ⚠️  Warning: Possible overfitting (gap: {gap:.3f})")
```

---

## Nested Cross-Validation

### Why Nested CV?

**Problem:** Using CV for hyperparameter tuning and evaluation gives optimistically biased results.

**Solution:** Nested CV - outer loop for evaluation, inner loop for tuning.

```
Outer Loop (Evaluation):
  Fold 1:
    Inner Loop (Hyperparameter Tuning):
      [Train on subset] -> [Validate] -> [Select best params]
    [Train on full train fold with best params]
    [Test on outer test fold]
  Fold 2:
    ...
```

### Implementation

```python
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

# Load data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Define model and parameter grid
model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

# Nested CV
# Inner loop: GridSearchCV (finds best hyperparameters)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
clf = GridSearchCV(model, param_grid, cv=inner_cv, scoring='roc_auc')

# Outer loop: cross_val_score (evaluates generalization)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
nested_scores = cross_val_score(clf, X, y, cv=outer_cv, scoring='roc_auc')

print("Nested Cross-Validation Results:")
print(f"ROC-AUC: {nested_scores.mean():.3f} (+/- {nested_scores.std():.3f})")
print(f"Scores: {nested_scores}")

# Compare with non-nested (biased) approach
non_nested_scores = cross_val_score(
    clf.fit(X, y).best_estimator_, X, y, cv=outer_cv, scoring='roc_auc'
)
print(f"\nNon-nested (biased): {non_nested_scores.mean():.3f}")
print(f"Difference: {non_nested_scores.mean() - nested_scores.mean():.3f}")
```

**Complete Nested CV Example:**

```python
def nested_cross_validation(X, y, model, param_grid, outer_cv=5, inner_cv=3):
    """
    Perform nested cross-validation.

    Outer loop: Evaluate model performance
    Inner loop: Tune hyperparameters
    """
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    import numpy as np

    # Outer CV
    outer_cv_splits = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=42)
    outer_scores = []
    best_params_list = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv_splits.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Inner CV for hyperparameter tuning
        inner_cv_splits = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            model, param_grid,
            cv=inner_cv_splits,
            scoring='roc_auc',
            n_jobs=-1
        )

        # Fit inner CV
        grid_search.fit(X_train, y_train)

        # Evaluate on outer test fold
        score = grid_search.score(X_test, y_test)
        outer_scores.append(score)
        best_params_list.append(grid_search.best_params_)

        print(f"Fold {fold}: Score = {score:.3f}, Best params = {grid_search.best_params_}")

    print(f"\n{'='*60}")
    print(f"Nested CV Mean: {np.mean(outer_scores):.3f} (+/- {np.std(outer_scores):.3f})")
    print(f"{'='*60}")

    return np.array(outer_scores), best_params_list

# Use it
scores, best_params = nested_cross_validation(
    X, y,
    RandomForestClassifier(random_state=42),
    param_grid,
    outer_cv=5,
    inner_cv=3
)
```

---

## Common Pitfalls

### 1. Data Leakage in Preprocessing

**WRONG:**
```python
# Scaling on entire dataset before CV - LEAKS information!
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # WRONG!

scores = cross_val_score(model, X_scaled, y, cv=5)
```

**CORRECT:**
```python
# Use Pipeline to ensure scaling happens within each fold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Method 1: Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])

scores = cross_val_score(pipe, X, y, cv=5)
print(f"Correct CV with Pipeline: {scores.mean():.3f}")

# Method 2: Manual (for understanding)
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Only transform test data

    # Train and evaluate
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    score = model.score(X_test_scaled, y_test)
    scores.append(score)

print(f"Manual correct CV: {np.mean(scores):.3f}")
```

### 2. Using shuffle=True for Time Series

**WRONG:**
```python
# Shuffling time series breaks temporal order!
kf = KFold(n_splits=5, shuffle=True)  # WRONG for time series
```

**CORRECT:**
```python
# Use TimeSeriesSplit for time series
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X_ts, y_ts, cv=tscv)
```

### 3. Not Using Stratified CV for Classification

**WRONG:**
```python
# Regular KFold on imbalanced classification
kf = KFold(n_splits=5)
scores = cross_val_score(model, X, y_imbalanced, cv=kf)
```

**CORRECT:**
```python
# Always use StratifiedKFold for classification
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y_imbalanced, cv=skf)
```

### 4. Using CV Results for Final Model Selection

**WRONG:**
```python
# Using CV scores to select model, then reporting CV score
scores_rf = cross_val_score(rf_model, X, y, cv=5)
scores_svm = cross_val_score(svm_model, X, y, cv=5)

# Select best model based on CV
if scores_rf.mean() > scores_svm.mean():
    final_model = rf_model
    final_score = scores_rf.mean()  # WRONG! Optimistically biased

print(f"Final model score: {final_score:.3f}")  # Biased!
```

**CORRECT:**
```python
# Use nested CV or hold out final test set
from sklearn.model_selection import train_test_split

# Hold out final test set
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Use CV on development set for model selection
scores_rf = cross_val_score(rf_model, X_dev, y_dev, cv=5)
scores_svm = cross_val_score(svm_model, X_dev, y_dev, cv=5)

# Select best model
if scores_rf.mean() > scores_svm.mean():
    final_model = rf_model
else:
    final_model = svm_model

# Train on full development set and evaluate on held-out test set
final_model.fit(X_dev, y_dev)
final_score = final_model.score(X_test, y_test)

print(f"Final test score: {final_score:.3f}")  # Unbiased!
```

---

## Best Practices

### 1. Always Use Pipeline for Preprocessing

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# Correct way: Pipeline ensures no data leakage
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('classifier', RandomForestClassifier(random_state=42))
])

scores = cross_val_score(pipe, X, y, cv=5)
```

### 2. Report Mean and Standard Deviation

```python
scores = cross_val_score(model, X, y, cv=5)

print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
print(f"Individual scores: {scores}")
print(f"95% CI: [{scores.mean() - 2*scores.std():.3f}, "
      f"{scores.mean() + 2*scores.std():.3f}]")
```

### 3. Use Stratified CV for Classification

```python
# Always default to stratified for classification
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv)
```

### 4. Set Random Seeds for Reproducibility

```python
# Set random seeds everywhere
model = RandomForestClassifier(random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv)
```

---

## CV Selection Guide

| Scenario | Recommended CV | Why |
|----------|---------------|-----|
| Classification | Stratified K-Fold (K=5) | Maintains class distribution |
| Regression | K-Fold (K=5) | Standard approach |
| Small dataset | LOO or K-Fold (K=10) | Maximum data usage |
| Time series | TimeSeriesSplit | Respects temporal order |
| Imbalanced data | Stratified K-Fold | Essential for balance |
| Hyperparameter tuning | Nested CV | Unbiased evaluation |
| Quick prototyping | K-Fold (K=3) | Fast iterations |

---

## Related Topics

- [Model Selection](model-selection.md) - Using CV for model comparison
- [Hyperparameter Tuning](../optimization/hyperparameter-tuning.md) - Using CV for tuning
- [Classification Metrics](classification-metrics.md) - Metrics to use with CV
- [Regression Metrics](regression-metrics.md) - Metrics for regression CV

---

## Summary

**Key Takeaways:**
1. **Always use cross-validation** for robust performance estimates
2. **K-Fold (K=5 or K=10)** is the default choice
3. **Stratified K-Fold** for classification (especially imbalanced)
4. **TimeSeriesSplit** for time series data (never shuffle!)
5. **Use Pipeline** to prevent data leakage in preprocessing
6. **Nested CV** for unbiased hyperparameter tuning evaluation
7. **Report mean and standard deviation** for transparency
8. **Set random seeds** for reproducibility

Remember: A single train/test split can be misleading - always use cross-validation!

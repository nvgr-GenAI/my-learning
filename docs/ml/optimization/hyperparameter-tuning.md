# Hyperparameter Tuning

## Overview

Hyperparameter tuning is the process of finding the optimal set of hyperparameters for a machine learning model. Unlike model parameters (learned during training), hyperparameters are set before training and control the learning process.

**What You'll Learn:**
- Grid search, random search, Bayesian optimization
- When to use each method
- Implementation with sklearn and Optuna
- Search space definition
- Best practices and common pitfalls

**Prerequisites:** Understanding of model training and cross-validation

---

## What are Hyperparameters?

### Examples

**Model Hyperparameters:**
- Number of layers, neurons
- Learning rate
- Regularization strength (alpha)
- Tree depth, number of trees

**Training Hyperparameters:**
- Batch size
- Number of epochs
- Optimizer choice

```python
# Example: These are NOT learned from data
model = RandomForestClassifier(
    n_estimators=100,        # Hyperparameter
    max_depth=10,             # Hyperparameter
    min_samples_split=5,      # Hyperparameter
    random_state=42
)
```

---

## 1. Grid Search

### Concept

Try **all combinations** of hyperparameters in a predefined grid.

```
Learning Rate: [0.001, 0.01, 0.1]
Batch Size: [16, 32, 64]

Total combinations: 3 × 3 = 9
```

### Implementation

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create model
rf = RandomForestClassifier(random_state=42)

# Grid search with cross-validation
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,                    # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,               # Use all CPUs
    verbose=2,
    return_train_score=True
)

# Fit
print("Starting Grid Search...")
print(f"Total combinations: {np.prod([len(v) for v in param_grid.values()])}")
grid_search.fit(X_train, y_train)

# Best parameters
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Test performance
test_score = grid_search.score(X_test, y_test)
print(f"Test score: {test_score:.4f}")

# Results DataFrame
results = pd.DataFrame(grid_search.cv_results_)
results = results.sort_values('rank_test_score')

print("\nTop 5 configurations:")
print(results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']].head())
```

### Pros and Cons

**Pros:**
- Exhaustive (guaranteed to find best in grid)
- Reproducible
- Easy to parallelize

**Cons:**
- Computationally expensive
- Exponential growth with parameters
- Doesn't explore between grid points

**When to Use:**
- Few hyperparameters (2-3)
- Small search spaces
- Need guaranteed best combination

---

## 2. Random Search

### Concept

Sample **random combinations** of hyperparameters from distributions.

**Key Insight:** Often finds good solutions faster than grid search!

### Implementation

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distributions
param_distributions = {
    'n_estimators': randint(50, 300),           # Discrete
    'max_depth': [None] + list(range(5, 50)),   # Mixed
    'min_samples_split': randint(2, 20),        # Discrete
    'min_samples_leaf': randint(1, 10),         # Discrete
    'max_features': uniform(0.1, 0.9)           # Continuous
}

# Random search
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=50,           # Number of random combinations to try
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    random_state=42,
    return_train_score=True
)

print("Starting Random Search...")
random_search.fit(X_train, y_train)

print(f"\nBest parameters: {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_:.4f}")
print(f"Test score: {random_search.score(X_test, y_test):.4f}")
```

### Grid Search vs Random Search

```python
import matplotlib.pyplot as plt

# Visualization of coverage
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Grid Search
ax1 = axes[0]
grid_x = [0.001, 0.01, 0.1]
grid_y = [10, 50, 100]
for x in grid_x:
    for y in grid_y:
        ax1.scatter(x, y, s=100, c='blue', marker='s', alpha=0.6)
ax1.set_xscale('log')
ax1.set_xlabel('Learning Rate')
ax1.set_ylabel('N Estimators')
ax1.set_title('Grid Search (9 combinations)', fontweight='bold')
ax1.grid(True, alpha=0.3)

# Random Search
ax2 = axes[1]
np.random.seed(42)
random_x = 10 ** np.random.uniform(-3, -1, 9)
random_y = np.random.uniform(10, 100, 9)
ax2.scatter(random_x, random_y, s=100, c='red', marker='o', alpha=0.6)
ax2.set_xscale('log')
ax2.set_xlabel('Learning Rate')
ax2.set_ylabel('N Estimators')
ax2.set_title('Random Search (9 combinations)', fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**When to Use:**
- Many hyperparameters (4+)
- Large search spaces
- Limited compute budget
- Want to explore broadly

---

## 3. Bayesian Optimization

### Concept

Intelligently selects next hyperparameters to try based on past results. Builds a probabilistic model of the objective function.

**Process:**
1. Try a few random configurations
2. Build probabilistic model
3. Select next point that maximizes expected improvement
4. Repeat

### Implementation with Optuna

```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    """
    Objective function for Optuna to optimize.
    """
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 5, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_float('max_features', 0.1, 1.0)

    # Create model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )

    # Cross-validation score
    score = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1).mean()

    return score

# Create study
study = optuna.create_study(
    direction='maximize',  # Maximize accuracy
    sampler=optuna.samplers.TPESampler(seed=42)
)

# Optimize
print("Starting Bayesian Optimization with Optuna...")
study.optimize(objective, n_trials=50, show_progress_bar=True)

# Best parameters
print(f"\nBest parameters: {study.best_params}")
print(f"Best cross-validation score: {study.best_value:.4f}")

# Train final model
best_params = study.best_params
final_model = RandomForestClassifier(**best_params, random_state=42)
final_model.fit(X_train, y_train)
print(f"Test score: {final_model.score(X_test, y_test):.4f}")
```

### Visualization

```python
# Plot optimization history
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Optimization history
trials_df = study.trials_dataframe()
axes[0].plot(trials_df['number'], trials_df['value'], 'o-', alpha=0.6)
axes[0].axhline(study.best_value, color='r', linestyle='--', label='Best')
axes[0].set_xlabel('Trial')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Optimization History', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Parameter importance
from optuna.visualization import plot_param_importances
param_importance = optuna.importance.get_param_importances(study)
params = list(param_importance.keys())
importances = list(param_importance.values())

axes[1].barh(params, importances)
axes[1].set_xlabel('Importance')
axes[1].set_title('Hyperparameter Importance', fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()
```

**When to Use:**
- Expensive evaluations (deep learning)
- Many hyperparameters
- Want efficient search
- Have moderate compute budget

---

## Comparison of Methods

| Method | Efficiency | Coverage | Use Case |
|--------|-----------|----------|----------|
| **Grid Search** | Low | Complete within grid | Few params, small space |
| **Random Search** | Medium | Broad exploration | Many params, large space |
| **Bayesian** | High | Intelligent exploration | Expensive evaluations |

### Computational Comparison

```python
import time

# Simulate search times
methods = ['Grid', 'Random', 'Bayesian']
times = []
scores = []

# Grid Search (baseline)
start = time.time()
grid_search.fit(X_train, y_train)
times.append(time.time() - start)
scores.append(grid_search.best_score_)

# Random Search
start = time.time()
random_search.fit(X_train, y_train)
times.append(time.time() - start)
scores.append(random_search.best_score_)

# Bayesian (Optuna already run)
times.append(15.0)  # Example time
scores.append(study.best_value)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(methods, times)
axes[0].set_ylabel('Time (seconds)')
axes[0].set_title('Computation Time', fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(methods, scores)
axes[1].set_ylabel('Best Accuracy')
axes[1].set_title('Best Score Found', fontweight='bold')
axes[1].set_ylim([min(scores)*0.95, max(scores)*1.01])
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

---

## Search Space Definition

### Good Practices

```python
# Logarithmic scale for learning rate
param_dist = {
    'learning_rate': uniform(1e-5, 1e-1),  # Better: loguniform
}

# Optuna logarithmic
def objective(trial):
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)  # Log scale
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    n_layers = trial.suggest_int('n_layers', 1, 5)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)

# Conditional parameters
def objective_conditional(trial):
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd'])

    if optimizer_name == 'adam':
        beta1 = trial.suggest_float('adam_beta1', 0.8, 0.99)
        beta2 = trial.suggest_float('adam_beta2', 0.9, 0.999)
    else:
        momentum = trial.suggest_float('sgd_momentum', 0.0, 0.99)
```

### Common Search Ranges

| Hyperparameter | Typical Range | Scale |
|----------------|---------------|-------|
| Learning Rate | [1e-5, 1e-1] | Log |
| Batch Size | [16, 32, 64, 128, 256] | Categorical |
| L2 Regularization | [1e-5, 1e-1] | Log |
| Dropout Rate | [0.0, 0.5] | Linear |
| N Estimators | [50, 500] | Linear or Log |
| Max Depth | [3, 30] | Linear |
| Hidden Units | [32, 256] | Power of 2 |

---

## Nested Cross-Validation

### Why?

Using same data for hyperparameter tuning and evaluation gives **optimistically biased** results.

### Solution

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Outer CV for evaluation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Inner CV for hyperparameter tuning
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Grid search (inner CV)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=inner_cv,  # Inner CV
    scoring='accuracy'
)

# Nested CV (outer CV)
nested_scores = cross_val_score(
    grid_search,
    X_train,
    y_train,
    cv=outer_cv,  # Outer CV
    scoring='accuracy'
)

print(f"Nested CV Scores: {nested_scores}")
print(f"Mean: {nested_scores.mean():.4f} (+/- {nested_scores.std():.4f})")
```

---

## Best Practices

### 1. Start Simple

```python
# Phase 1: Coarse search (broad)
param_grid_coarse = {
    'learning_rate': [1e-4, 1e-3, 1e-2],
    'batch_size': [32, 64, 128]
}

# Phase 2: Fine search (narrow around best)
param_grid_fine = {
    'learning_rate': [5e-4, 1e-3, 2e-3],  # Around best from phase 1
    'batch_size': [48, 64, 80]
}
```

### 2. Use Resource-Based Pruning

```python
# Optuna: Early stopping for bad trials
def objective_with_pruning(trial):
    model = create_model(trial)

    for epoch in range(100):
        score = train_one_epoch(model)

        # Report and check for pruning
        trial.report(score, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return final_score

# Create study with pruner
study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
)
```

### 3. Log Everything

```python
import mlflow

def objective_with_logging(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 30)
    }

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)

        # Train model
        model = RandomForestClassifier(**params)
        score = cross_val_score(model, X_train, y_train, cv=5).mean()

        # Log metrics
        mlflow.log_metric('cv_score', score)

    return score
```

---

## Common Mistakes

### 1. Not Using Cross-Validation

**WRONG:**
```python
# Single train/test split for tuning
grid_search = GridSearchCV(model, param_grid, cv=None)  # Wrong!
```

**CORRECT:**
```python
grid_search = GridSearchCV(model, param_grid, cv=5)  # Use CV!
```

### 2. Data Leakage

**WRONG:**
```python
# Scale before split
X_scaled = scaler.fit_transform(X)  # Leakage!
grid_search.fit(X_scaled, y)
```

**CORRECT:**
```python
# Use pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
```

### 3. Overfitting to Validation Set

**Problem:** Tuning too much on same validation set.

**Solution:** Use nested CV or hold out final test set.

---

## Hyperparameter Tuning Checklist

- [ ] Define clear objective metric
- [ ] Use cross-validation (not single split)
- [ ] Start with coarse grid, then refine
- [ ] Use logarithmic scale for learning rates
- [ ] Consider computational budget
- [ ] Log all experiments
- [ ] Use nested CV for unbiased evaluation
- [ ] Hold out final test set
- [ ] Document best hyperparameters
- [ ] Verify on independent test set

---

## Related Topics

- [Cross-Validation](../evaluation/cross-validation.md) - Robust evaluation
- [Model Selection](../evaluation/model-selection.md) - Choosing models
- [Learning Rate Scheduling](learning-rate-scheduling.md) - Dynamic LR
- [Regularization](regularization.md) - Hyperparameters to tune

---

## Summary

**Key Takeaways:**
1. **Grid Search**: Exhaustive but expensive
2. **Random Search**: More efficient than grid for large spaces
3. **Bayesian Optimization**: Most efficient for expensive evaluations
4. **Always use CV** for hyperparameter selection
5. **Start coarse, refine later**
6. **Use log scale** for learning rates and regularization
7. **Nested CV** for unbiased evaluation
8. **Optuna** is excellent for complex tuning

Remember: Good hyperparameters can improve accuracy by 5-20% - worth the effort!

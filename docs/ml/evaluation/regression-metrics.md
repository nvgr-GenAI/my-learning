# Regression Metrics

## Overview

Regression metrics evaluate how well a model predicts continuous values. Unlike classification, regression deals with numerical predictions where the distance from the true value matters.

**What You'll Learn:**
- Core regression metrics and when to use them
- Understanding error magnitudes and units
- Handling outliers in evaluation
- Implementation with sklearn
- Visualization techniques for regression

**Prerequisites:** Basic understanding of regression problems

---

## Core Metrics

### 1. Mean Squared Error (MSE)

**Definition:** Average of squared differences between predictions and actual values.

```
MSE = (1/n) * Σ(y_true - y_pred)²
```

**Properties:**
- Always positive (squared errors)
- Heavily penalizes large errors (due to squaring)
- Unit: squared units of target variable
- Range: [0, ∞), lower is better

**When to Use:**
- Large errors are particularly undesirable
- Want to penalize outliers heavily
- Standard metric for optimization (differentiable)

**When NOT to Use:**
- Outliers exist (they dominate the metric)
- Need interpretable units (MSE is in squared units)

**Example:**
```python
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

# Load California housing dataset
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.4f}")  # Units: (100k$)²

# Manual calculation
mse_manual = np.mean((y_test - y_pred)**2)
print(f"MSE (manual): {mse_manual:.4f}")
```

### 2. Root Mean Squared Error (RMSE)

**Definition:** Square root of MSE.

```
RMSE = √(MSE) = √[(1/n) * Σ(y_true - y_pred)²]
```

**Properties:**
- Same units as target variable (more interpretable than MSE)
- Still penalizes large errors heavily
- Range: [0, ∞), lower is better
- Most popular regression metric

**When to Use:**
- Default choice for most regression problems
- Need interpretable error magnitude
- Large errors are particularly bad

**Example:**
```python
from sklearn.metrics import mean_squared_error
import numpy as np

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f} (in 100k$)")

# Or use sklearn directly
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse:.4f}")

# Interpretation: On average, predictions are off by ~$57k
```

### 3. Mean Absolute Error (MAE)

**Definition:** Average of absolute differences between predictions and actual values.

```
MAE = (1/n) * Σ|y_true - y_pred|
```

**Properties:**
- Linear penalty for errors
- Same units as target variable
- More robust to outliers than MSE/RMSE
- Range: [0, ∞), lower is better

**When to Use:**
- Outliers present in data
- All errors should be weighted equally
- Need robust metric
- More interpretable than RMSE

**Example:**
```python
from sklearn.metrics import mean_absolute_error

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.4f}")

# Manual calculation
mae_manual = np.mean(np.abs(y_test - y_pred))
print(f"MAE (manual): {mae_manual:.4f}")

# Compare with RMSE
print(f"\nRMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Difference: {rmse - mae:.4f}")
# If RMSE >> MAE, there are large outliers
```

**MAE vs RMSE Comparison:**
```python
# Demonstrate difference with outliers
errors = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 50])  # One outlier

mae = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(errors**2))

print(f"MAE: {mae:.2f}")   # 5.90
print(f"RMSE: {rmse:.2f}")  # 15.97 - much higher due to outlier!
```

### 4. R² (R-Squared / Coefficient of Determination)

**Definition:** Proportion of variance in the dependent variable explained by the model.

```
R² = 1 - (SS_res / SS_tot)
   = 1 - [Σ(y_true - y_pred)² / Σ(y_true - ȳ)²]
```

Where:
- SS_res: Sum of squared residuals
- SS_tot: Total sum of squares
- ȳ: Mean of true values

**Properties:**
- Range: (-∞, 1], higher is better
- R² = 1: Perfect predictions
- R² = 0: Model as good as predicting mean
- R² < 0: Model worse than predicting mean
- Unitless (scale-independent)

**When to Use:**
- Compare models on different datasets
- Understand explained variance
- Communicate with non-technical stakeholders

**When NOT to Use:**
- Absolute error magnitude is important
- Adding features always increases R² (use adjusted R²)

**Example:**
```python
from sklearn.metrics import r2_score

# Calculate R²
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

# Interpretation
print(f"\nThe model explains {r2*100:.2f}% of the variance in the target.")

# Manual calculation
ss_res = np.sum((y_test - y_pred)**2)
ss_tot = np.sum((y_test - y_test.mean())**2)
r2_manual = 1 - (ss_res / ss_tot)
print(f"R² (manual): {r2_manual:.4f}")

# Compare with baseline (predicting mean)
y_pred_baseline = np.full_like(y_test, y_test.mean())
r2_baseline = r2_score(y_test, y_pred_baseline)
print(f"Baseline R²: {r2_baseline:.4f}")  # Should be 0.0
```

**Adjusted R²** (for multiple features):
```python
def adjusted_r2(r2, n_samples, n_features):
    """Calculate adjusted R² that penalizes adding features."""
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

n_samples = len(y_test)
n_features = X_test.shape[1]

adj_r2 = adjusted_r2(r2, n_samples, n_features)
print(f"Adjusted R²: {adj_r2:.4f}")
```

### 5. Mean Absolute Percentage Error (MAPE)

**Definition:** Average of absolute percentage errors.

```
MAPE = (100/n) * Σ|((y_true - y_pred) / y_true)|
```

**Properties:**
- Output in percentage (easy to interpret)
- Scale-independent
- Sensitive to small denominators
- Cannot handle zero values in y_true

**When to Use:**
- Need percentage error
- Comparing across different scales
- Business stakeholders prefer percentages

**When NOT to Use:**
- Target values contain zeros
- Very small values in target (unstable)
- Asymmetric (penalizes over-prediction more)

**Example:**
```python
def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE, handling zero values."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAPE: {mape:.2f}%")

# Or use sklearn (available in newer versions)
from sklearn.metrics import mean_absolute_percentage_error
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAPE (sklearn): {mape*100:.2f}%")
```

**Symmetric MAPE (sMAPE)** - More balanced alternative:
```python
def symmetric_mape(y_true, y_pred):
    """Symmetric MAPE - treats over/under-prediction equally."""
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
    )

smape = symmetric_mape(y_test, y_pred)
print(f"sMAPE: {smape:.2f}%")
```

---

## Comprehensive Example: House Price Prediction

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error
)

# Load data
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

print("Dataset Info:")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {X_train.shape[1]}")
print(f"Target range: [{y_test.min():.2f}, {y_test.max():.2f}]")
print(f"Target mean: {y_test.mean():.2f}")

# Train multiple models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = []

print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'Model': name,
        'Train RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'Train MAE': mean_absolute_error(y_train, y_pred_train),
        'Test MAE': mean_absolute_error(y_test, y_pred_test),
        'Train R²': r2_score(y_train, y_pred_train),
        'Test R²': r2_score(y_test, y_pred_test),
        'Test MAPE': mean_absolute_percentage_error(y_test, y_pred_test) * 100
    }
    results.append(metrics)

    print(f"\n{name}:")
    print(f"  RMSE: {metrics['Test RMSE']:.4f}")
    print(f"  MAE:  {metrics['Test MAE']:.4f}")
    print(f"  R²:   {metrics['Test R²']:.4f}")
    print(f"  MAPE: {metrics['Test MAPE']:.2f}%")

# Create results DataFrame
df_results = pd.DataFrame(results)
print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(df_results.to_string(index=False))

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for idx, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)

    # 1. Actual vs Predicted
    axes[0, idx].scatter(y_test, y_pred, alpha=0.5, s=10)
    axes[0, idx].plot([y_test.min(), y_test.max()],
                      [y_test.min(), y_test.max()],
                      'r--', lw=2, label='Perfect Prediction')
    axes[0, idx].set_xlabel('Actual Values')
    axes[0, idx].set_ylabel('Predicted Values')
    axes[0, idx].set_title(f'{name}\nActual vs Predicted')
    axes[0, idx].legend()
    axes[0, idx].grid(True, alpha=0.3)

    # 2. Residual Plot
    residuals = y_test - y_pred
    axes[1, idx].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[1, idx].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, idx].set_xlabel('Predicted Values')
    axes[1, idx].set_ylabel('Residuals')
    axes[1, idx].set_title(f'{name}\nResidual Plot')
    axes[1, idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regression_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Error distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    errors = y_test - y_pred

    axes[idx].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[idx].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[idx].set_xlabel('Prediction Error')
    axes[idx].set_ylabel('Frequency')
    axes[idx].set_title(f'{name}\nError Distribution')
    axes[idx].grid(True, alpha=0.3, axis='y')

    # Add statistics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    axes[idx].text(0.05, 0.95, f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}',
                   transform=axes[idx].transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('error_distributions.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## Visualization Techniques

### 1. Actual vs Predicted Plot

```python
def plot_predictions(y_true, y_pred, title='Actual vs Predicted'):
    """Visualize actual vs predicted values."""
    plt.figure(figsize=(8, 8))

    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5, s=30)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val],
             'r--', lw=2, label='Perfect Prediction')

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(f'{title}\nRMSE: {rmse:.3f}, R²: {r2:.3f}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Use it
plot_predictions(y_test, y_pred, 'House Price Prediction')
```

### 2. Residual Plot

```python
def plot_residuals(y_true, y_pred, title='Residual Plot'):
    """Visualize residuals to check for patterns."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=30)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Predicted')
    axes[0].grid(True, alpha=0.3)

    # Residual histogram
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Use it
plot_residuals(y_test, y_pred, 'Model Residual Analysis')
```

### 3. Error Metrics Comparison

```python
def compare_metrics(y_true, predictions_dict):
    """Compare multiple models using different metrics."""
    metrics_data = []

    for name, y_pred in predictions_dict.items():
        metrics_data.append({
            'Model': name,
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R²': r2_score(y_true, y_pred),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100
        })

    df = pd.DataFrame(metrics_data)

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = ['RMSE', 'MAE', 'R²', 'MAPE']
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        df.plot(x='Model', y=metric, kind='bar', ax=ax, legend=False)
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

    return df

# Use it
predictions_dict = {
    'Linear': model1.predict(X_test),
    'Ridge': model2.predict(X_test),
    'Random Forest': model3.predict(X_test)
}
metrics_df = compare_metrics(y_test, predictions_dict)
print(metrics_df)
```

---

## Common Mistakes

### 1. Using MSE Instead of RMSE for Interpretation

**Problem:**
```python
mse = mean_squared_error(y_test, y_pred)
print(f"Error: {mse:.2f}")  # Units are squared - hard to interpret!
```

**Solution:**
```python
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Error: {rmse:.2f} (same units as target)")
```

### 2. Ignoring Outliers

**Problem:** RMSE heavily influenced by outliers.

**Solution:**
```python
# Compare RMSE and MAE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")

# Large difference indicates outliers
if rmse / mae > 1.5:
    print("Warning: Significant outliers detected!")
    print("Consider using MAE or investigating outliers.")
```

### 3. Using MAPE with Zero Values

**Problem:**
```python
y_true = np.array([0, 1, 2, 3, 4])
y_pred = np.array([0.5, 1.2, 2.1, 2.9, 4.2])

# This will raise warning or give incorrect results
mape = mean_absolute_percentage_error(y_true, y_pred)
```

**Solution:**
```python
# Filter out zeros
mask = y_true != 0
if mask.sum() > 0:
    mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask])
    print(f"MAPE: {mape*100:.2f}%")
else:
    print("Cannot calculate MAPE: all true values are zero")
```

### 4. Not Checking R² on Training vs Test

**Problem:** High training R², low test R² indicates overfitting.

**Solution:**
```python
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f"R² Train: {r2_train:.3f}")
print(f"R² Test: {r2_test:.3f}")

if r2_train - r2_test > 0.1:
    print("Warning: Possible overfitting detected!")
```

### 5. Comparing Models on Different Scales

**Problem:** Comparing RMSE across datasets with different scales.

**Solution:** Use R² or MAPE for scale-independent comparison.

```python
# Instead of comparing RMSE directly
# Use R² which is scale-independent
print(f"Model 1 R²: {r2_model1:.3f}")
print(f"Model 2 R²: {r2_model2:.3f}")
```

---

## Metric Selection Guide

| Scenario | Recommended Metric | Reason |
|----------|-------------------|---------|
| Default choice | RMSE | Interpretable units, standard metric |
| Outliers present | MAE | More robust to outliers |
| Need % error | MAPE | Easy for stakeholders to understand |
| Comparing across scales | R², MAPE | Scale-independent |
| Heavily penalize large errors | MSE, RMSE | Quadratic penalty |
| All errors equally bad | MAE | Linear penalty |
| Understanding explained variance | R² | Shows model effectiveness |
| Optimization objective | MSE | Differentiable, standard loss |
| Business reporting | MAPE, RMSE | Interpretable for non-technical audience |

---

## Related Topics

- [Classification Metrics](classification-metrics.md) - Metrics for categorical predictions
- [Cross-Validation](cross-validation.md) - Robust metric evaluation
- [Model Selection](model-selection.md) - Choosing models based on metrics
- [Evaluation Index](index.md) - Overview of model evaluation

---

## Summary

**Key Takeaways:**
1. **RMSE** is the default choice (same units as target)
2. **MAE** when outliers are present (more robust)
3. **R²** for understanding explained variance (scale-independent)
4. **MAPE** for percentage error (avoid with zeros)
5. Always visualize with **actual vs predicted** and **residual plots**
6. Compare **RMSE vs MAE** to detect outliers (large ratio = outliers)
7. Check metrics on both **training and test sets** (detect overfitting)
8. Use **multiple metrics** for complete evaluation

Remember: Choose metrics based on your data characteristics and business needs!

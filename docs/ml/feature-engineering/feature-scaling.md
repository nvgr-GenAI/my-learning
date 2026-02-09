# Feature Scaling

## Overview

Feature scaling transforms features to similar ranges or distributions, improving model performance and training stability. Many machine learning algorithms are sensitive to the scale of input features, making proper scaling essential for optimal results.

**Key Principle**: Scale features to ensure no single feature dominates due to its magnitude, not its predictive power.

## Why Feature Scaling Matters

1. **Faster Convergence**: Gradient descent converges faster with scaled features
2. **Better Performance**: Distance-based algorithms require similar scales
3. **Numerical Stability**: Prevents overflow/underflow in computations
4. **Feature Importance**: Enables fair comparison of coefficients/weights
5. **Regularization**: L1/L2 regularization works better with scaled features

## When Scaling is Required

### Algorithms That NEED Scaling

```python
# These algorithms are scale-sensitive:
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

# All distance-based and gradient-based algorithms require scaling
```

### Algorithms That DON'T Need Scaling

```python
# These algorithms are scale-invariant:
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, XGBClassifier
from sklearn.naive_bayes import GaussianNB

# Tree-based algorithms split on feature values, not distances
```

## Impact Demonstration

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Create dataset with different scales
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                           n_informative=2, random_state=42)

# Scale features differently
X[:, 0] = X[:, 0] * 1000  # Feature 1: large scale
X[:, 1] = X[:, 1] * 0.01  # Feature 2: small scale

df = pd.DataFrame(X, columns=['feature1', 'feature2'])
df['target'] = y

print("Feature statistics:")
print(df[['feature1', 'feature2']].describe())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model WITHOUT scaling
model_unscaled = LogisticRegression(random_state=42, max_iter=1000)
model_unscaled.fit(X_train, y_train)
score_unscaled = model_unscaled.score(X_test, y_test)

# Model WITH scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = LogisticRegression(random_state=42, max_iter=1000)
model_scaled.fit(X_train_scaled, y_train)
score_scaled = model_scaled.score(X_test_scaled, y_test)

print(f"\nAccuracy WITHOUT scaling: {score_unscaled:.4f}")
print(f"Accuracy WITH scaling: {score_scaled:.4f}")
print(f"Improvement: {(score_scaled - score_unscaled)*100:.2f}%")

# Visualize feature scales
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Original data
axes[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.5)
axes[0].set_xlabel('Feature 1 (large scale)')
axes[0].set_ylabel('Feature 2 (small scale)')
axes[0].set_title('Original Data (Different Scales)')

# Scaled data
axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, alpha=0.5)
axes[1].set_xlabel('Feature 1 (scaled)')
axes[1].set_ylabel('Feature 2 (scaled)')
axes[1].set_title('Scaled Data (Similar Scales)')

# Feature ranges comparison
feature_ranges = pd.DataFrame({
    'Original': [X_train[:, 0].std(), X_train[:, 1].std()],
    'Scaled': [X_train_scaled[:, 0].std(), X_train_scaled[:, 1].std()]
}, index=['Feature 1', 'Feature 2'])

feature_ranges.plot(kind='bar', ax=axes[2])
axes[2].set_title('Standard Deviation Comparison')
axes[2].set_ylabel('Standard Deviation')
plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=0)

plt.tight_layout()
plt.show()
```

## Scaling Methods

### 1. StandardScaler (Z-score Normalization)

Transforms features to have mean=0 and standard deviation=1.

**Formula**: `z = (x - μ) / σ`

```python
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Sample data
data = np.array([[1, 200], [2, 300], [3, 400], [4, 500], [5, 600]])
df = pd.DataFrame(data, columns=['age', 'income'])

print("Original data:")
print(df)
print(f"\nOriginal stats:\n{df.describe()}")

# Apply StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
df_scaled = pd.DataFrame(scaled_data, columns=['age_scaled', 'income_scaled'])

print("\nScaled data:")
print(df_scaled)
print(f"\nScaled stats:\n{df_scaled.describe()}")

# Check: mean should be ~0, std should be ~1
print(f"\nVerification:")
print(f"Means: {df_scaled.mean().values}")
print(f"Std devs: {df_scaled.std().values}")

# Visualize
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Original
df.plot(kind='box', ax=axes[0])
axes[0].set_title('Original Data')
axes[0].set_ylabel('Value')

# Scaled
df_scaled.plot(kind='box', ax=axes[1])
axes[1].set_title('StandardScaler')
axes[1].set_ylabel('Scaled Value')

plt.tight_layout()
plt.show()
```

**When to use:**
- Features are normally distributed
- Want to preserve outlier information
- Using algorithms sensitive to feature variance (SVM, neural networks, PCA)
- When you need interpretable z-scores

**Pros:**
- Preserves outliers (important if they're informative)
- Centers data around 0 (good for gradient descent)
- Less affected by outliers than MinMaxScaler

**Cons:**
- Not bounded to specific range
- Sensitive to outliers (they affect mean and std)

### 2. MinMaxScaler (Normalization)

Scales features to a fixed range, typically [0, 1].

**Formula**: `x_scaled = (x - x_min) / (x_max - x_min)`

```python
from sklearn.preprocessing import MinMaxScaler

# Apply MinMaxScaler
scaler_minmax = MinMaxScaler()
scaled_data = scaler_minmax.fit_transform(df)
df_minmax = pd.DataFrame(scaled_data, columns=['age_scaled', 'income_scaled'])

print("MinMaxScaler (0-1 range):")
print(df_minmax)
print(f"\nMin values: {df_minmax.min().values}")
print(f"Max values: {df_minmax.max().values}")

# Custom range [0, 10]
scaler_custom = MinMaxScaler(feature_range=(0, 10))
scaled_custom = scaler_custom.fit_transform(df)
df_custom = pd.DataFrame(scaled_custom, columns=['age_scaled', 'income_scaled'])

print("\nMinMaxScaler (0-10 range):")
print(df_custom)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Original
axes[0].scatter(df['age'], df['income'])
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Income')
axes[0].set_title('Original Data')

# [0, 1] scaled
axes[1].scatter(df_minmax['age_scaled'], df_minmax['income_scaled'])
axes[1].set_xlabel('Age (scaled)')
axes[1].set_ylabel('Income (scaled)')
axes[1].set_title('MinMaxScaler [0, 1]')

# [0, 10] scaled
axes[2].scatter(df_custom['age_scaled'], df_custom['income_scaled'])
axes[2].set_xlabel('Age (scaled)')
axes[2].set_ylabel('Income (scaled)')
axes[2].set_title('MinMaxScaler [0, 10]')

plt.tight_layout()
plt.show()
```

**When to use:**
- Features have clear min/max boundaries
- Distribution is not Gaussian
- Need features in specific range (e.g., [0, 1] for neural networks)
- When you want to preserve zero values

**Pros:**
- Bounded range (predictable output)
- Preserves relationships between values
- Works well with sparse data

**Cons:**
- Very sensitive to outliers (they become the bounds)
- New data outside training range will be outside [0, 1]
- Doesn't center data around 0

### 3. RobustScaler

Scales features using statistics robust to outliers (median and IQR).

**Formula**: `x_scaled = (x - median) / IQR`

```python
from sklearn.preprocessing import RobustScaler
import numpy as np

# Data with outliers
data_outliers = np.array([
    [1, 200],
    [2, 300],
    [3, 400],
    [4, 500],
    [5, 600],
    [100, 10000]  # Outlier
])
df_outliers = pd.DataFrame(data_outliers, columns=['age', 'income'])

print("Data with outliers:")
print(df_outliers)

# Compare scalers
scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler()
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, scaler) in enumerate(scalers.items()):
    scaled = scaler.fit_transform(df_outliers)
    df_scaled = pd.DataFrame(scaled, columns=['age', 'income'])

    axes[idx].scatter(df_scaled['age'], df_scaled['income'])
    axes[idx].scatter(df_scaled['age'].iloc[-1], df_scaled['income'].iloc[-1],
                     color='red', s=200, marker='x', label='Outlier')
    axes[idx].set_title(name)
    axes[idx].set_xlabel('Age (scaled)')
    axes[idx].set_ylabel('Income (scaled)')
    axes[idx].legend()

plt.tight_layout()
plt.show()

# Compare impact on normal points
scaler_robust = RobustScaler()
df_robust = pd.DataFrame(
    scaler_robust.fit_transform(df_outliers),
    columns=['age_scaled', 'income_scaled']
)

print("\nRobustScaler result:")
print(df_robust)
print("\nNotice: Regular points (-5) maintain reasonable scale despite outlier")
```

**When to use:**
- Data contains outliers
- Distribution is not normal
- Want to reduce impact of extreme values
- Working with skewed distributions

**Pros:**
- Robust to outliers
- Preserves structure of data
- Good for skewed distributions

**Cons:**
- Not bounded to specific range
- Less commonly used (may confuse collaborators)
- Doesn't center at 0

### 4. MaxAbsScaler

Scales by maximum absolute value to range [-1, 1].

**Formula**: `x_scaled = x / max(|x|)`

```python
from sklearn.preprocessing import MaxAbsScaler

# Data with negative values
data_neg = np.array([
    [-10, -100],
    [-5, 0],
    [0, 50],
    [5, 100],
    [10, 200]
])
df_neg = pd.DataFrame(data_neg, columns=['feature1', 'feature2'])

print("Data with negative values:")
print(df_neg)

scaler_maxabs = MaxAbsScaler()
df_maxabs = pd.DataFrame(
    scaler_maxabs.fit_transform(df_neg),
    columns=['feature1_scaled', 'feature2_scaled']
)

print("\nMaxAbsScaler result:")
print(df_maxabs)
print("\nNote: Preserves sign, scales to [-1, 1]")
```

**When to use:**
- Data is sparse (many zeros)
- Want to preserve sparsity
- Data already centered at zero
- Features have positive and negative values

**Pros:**
- Preserves sparsity (zeros stay zeros)
- Preserves sign of values
- Bounded to [-1, 1]

**Cons:**
- Sensitive to outliers
- Less commonly needed
- Assumes data is already centered

### 5. Normalizer (L1/L2 Normalization)

Scales individual samples (rows) to have unit norm.

```python
from sklearn.preprocessing import Normalizer

# Sample data: feature vectors
data_vectors = np.array([
    [3, 4],      # Magnitude: 5
    [5, 12],     # Magnitude: 13
    [8, 15]      # Magnitude: 17
])

print("Original vectors:")
print(data_vectors)
print("\nMagnitudes:", [np.linalg.norm(row) for row in data_vectors])

# L2 normalization (Euclidean norm)
normalizer_l2 = Normalizer(norm='l2')
data_l2 = normalizer_l2.fit_transform(data_vectors)

print("\nL2 Normalized:")
print(data_l2)
print("Magnitudes:", [np.linalg.norm(row) for row in data_l2])

# L1 normalization (Manhattan norm)
normalizer_l1 = Normalizer(norm='l1')
data_l1 = normalizer_l1.fit_transform(data_vectors)

print("\nL1 Normalized:")
print(data_l1)
print("L1 norms:", [np.sum(np.abs(row)) for row in data_l1])

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Original
axes[0].scatter(data_vectors[:, 0], data_vectors[:, 1], s=100)
for i, (x, y) in enumerate(data_vectors):
    axes[0].arrow(0, 0, x, y, head_width=0.5, alpha=0.5)
axes[0].set_title('Original Vectors')
axes[0].set_xlim(-1, 20)
axes[0].set_ylim(-1, 20)
axes[0].grid(True)

# L2 normalized
axes[1].scatter(data_l2[:, 0], data_l2[:, 1], s=100)
for i, (x, y) in enumerate(data_l2):
    axes[1].arrow(0, 0, x, y, head_width=0.05, alpha=0.5)
circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--')
axes[1].add_patch(circle)
axes[1].set_title('L2 Normalized (unit circle)')
axes[1].set_xlim(-1.5, 1.5)
axes[1].set_ylim(-1.5, 1.5)
axes[1].grid(True)
axes[1].set_aspect('equal')

# L1 normalized
axes[2].scatter(data_l1[:, 0], data_l1[:, 1], s=100)
for i, (x, y) in enumerate(data_l1):
    axes[2].arrow(0, 0, x, y, head_width=0.05, alpha=0.5)
axes[2].plot([1, 0, -1, 0, 1], [0, 1, 0, -1, 0], 'r--', label='L1 norm = 1')
axes[2].set_title('L1 Normalized')
axes[2].set_xlim(-1.5, 1.5)
axes[2].set_ylim(-1.5, 1.5)
axes[2].grid(True)
axes[2].legend()

plt.tight_layout()
plt.show()
```

**When to use:**
- Text data (TF-IDF vectors)
- When direction matters more than magnitude
- Comparing document similarity
- Neural networks with L2 normalization

**Pros:**
- Focuses on direction, not scale
- Good for similarity metrics
- Each sample scaled independently

**Cons:**
- Different from other scalers (works on rows, not columns)
- Not appropriate for most tabular data
- Can lose important magnitude information

## Comparing Scalers

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                   MaxAbsScaler, Normalizer)
import matplotlib.pyplot as plt

# Create diverse dataset
np.random.seed(42)
data = {
    'normal': np.random.normal(100, 15, 100),
    'skewed': np.random.exponential(50, 100),
    'with_outliers': np.concatenate([np.random.normal(50, 10, 95), [200, 210, 180, 190, 175]])
}
df = pd.DataFrame(data)

# Apply all scalers
scalers = {
    'Original': None,
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler(),
    'MaxAbsScaler': MaxAbsScaler()
}

fig, axes = plt.subplots(len(data), len(scalers), figsize=(20, 12))

for col_idx, (feature, values) in enumerate(data.items()):
    for row_idx, (scaler_name, scaler) in enumerate(scalers.items()):
        ax = axes[col_idx, row_idx]

        if scaler is None:
            scaled_values = values
        else:
            scaled_values = scaler.fit_transform(values.reshape(-1, 1)).flatten()

        ax.hist(scaled_values, bins=30, edgecolor='black', alpha=0.7)
        ax.set_title(f'{feature}\n{scaler_name}')
        ax.set_ylabel('Frequency')

        # Add statistics
        stats_text = f'Mean: {scaled_values.mean():.2f}\nStd: {scaled_values.std():.2f}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# Numerical comparison
comparison = pd.DataFrame()
for scaler_name, scaler in scalers.items():
    if scaler is None:
        continue

    for feature in data.keys():
        scaled = scaler.fit_transform(df[[feature]]).flatten()
        comparison = pd.concat([comparison, pd.DataFrame({
            'Feature': feature,
            'Scaler': scaler_name,
            'Mean': scaled.mean(),
            'Std': scaled.std(),
            'Min': scaled.min(),
            'Max': scaled.max()
        }, index=[0])], ignore_index=True)

print("\nScaler Comparison:")
print(comparison)
```

## Practical Implementation

### Complete Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Sample data
X = np.random.randn(1000, 5)
X[:, 0] *= 1000  # Different scales
X[:, 1] *= 0.01
y = (X[:, 0] + X[:, 1] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define different scaling for different features
preprocessor = ColumnTransformer(
    transformers=[
        ('standard', StandardScaler(), [0, 1, 2]),  # StandardScaler for features 0-2
        ('minmax', MinMaxScaler(), [3, 4])          # MinMaxScaler for features 3-4
    ])

# Create pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)

print(f"Training accuracy: {train_score:.4f}")
print(f"Test accuracy: {test_score:.4f}")
```

### Inverse Transform

```python
# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Later, convert back to original scale
X_original = scaler.inverse_transform(X_scaled)

print("Original range:", X_train.min(), "to", X_train.max())
print("Scaled range:", X_scaled.min(), "to", X_scaled.max())
print("Inverse range:", X_original.min(), "to", X_original.max())
```

## Common Mistakes and How to Avoid Them

### 1. Data Leakage (Fitting on Entire Dataset)

```python
# WRONG: Fit on entire dataset before splitting
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Uses test data statistics!
X_train, X_test = train_test_split(X_scaled)

# CORRECT: Split first, fit only on training
X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use training statistics
```

### 2. Scaling Target Variable

```python
# Usually DON'T scale target for classification
# Only scale target for regression if needed

# Classification: NO scaling
y = [0, 1, 0, 1]  # Keep as is

# Regression: Consider scaling if target has extreme values
y_reg = [1000000, 2000000, 3000000]
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(np.array(y_reg).reshape(-1, 1))
# Remember to inverse transform predictions!
```

### 3. Scaling Tree-Based Models

```python
from sklearn.ensemble import RandomForestClassifier

# Tree-based models DON'T need scaling
# Scaling won't hurt, but wastes computation

# No need to scale
rf = RandomForestClassifier()
rf.fit(X_train, y_train)  # Works fine without scaling
```

### 4. Not Saving the Scaler

```python
# WRONG: Fit new scaler on test/production data
scaler_test = StandardScaler()
X_test_scaled = scaler_test.fit_transform(X_test)  # Different statistics!

# CORRECT: Save and reuse training scaler
import joblib

# Save scaler after training
joblib.dump(scaler, 'scaler.pkl')

# Load and use in production
scaler_loaded = joblib.load('scaler.pkl')
X_new_scaled = scaler_loaded.transform(X_new)
```

## Decision Guide

```python
def recommend_scaler(X, algorithm_type):
    """Recommend scaler based on data and algorithm"""

    recommendations = []

    # Check if scaling is needed
    if algorithm_type in ['tree', 'random_forest', 'xgboost']:
        return "No scaling needed (tree-based algorithm)"

    # Check for outliers
    from scipy import stats
    z_scores = np.abs(stats.zscore(X, nan_policy='omit'))
    has_outliers = (z_scores > 3).any()

    if has_outliers:
        recommendations.append("RobustScaler (data has outliers)")
    else:
        recommendations.append("StandardScaler (no significant outliers)")

    # Check distribution
    from scipy.stats import normaltest
    for i in range(X.shape[1]):
        stat, p_value = normaltest(X[:, i])
        if p_value < 0.05:
            recommendations.append(f"Feature {i} is non-normal: consider MinMaxScaler")

    # Check for negative values
    if (X < 0).any():
        recommendations.append("Data has negative values: avoid MinMaxScaler if you want to preserve them")

    # Check sparsity
    sparsity = (X == 0).sum() / X.size
    if sparsity > 0.5:
        recommendations.append(f"Data is sparse ({sparsity*100:.1f}% zeros): consider MaxAbsScaler")

    return recommendations

# Example usage
print("Scaling recommendations:")
for rec in recommend_scaler(X_train, 'svm'):
    print(f"- {rec}")
```

## Best Practices Summary

1. **Always split data first**: Fit scaler only on training data
2. **Choose scaler based on data**: Consider distribution, outliers, sparsity
3. **Use pipelines**: Prevent leakage and simplify code
4. **Save scalers**: Reuse for test/production data
5. **Document choice**: Record why you chose a specific scaler
6. **Validate impact**: Check if scaling improves performance
7. **Consider algorithm**: Some don't need scaling

## Related Topics

- [Data Cleaning](./data-cleaning.md) - Clean data before scaling
- [Outlier Detection](./outlier-detection.md) - Handle outliers before scaling
- [Feature Creation](./feature-creation.md) - Create features, then scale

## Summary

Feature scaling is essential for many machine learning algorithms. Choose StandardScaler for normal distributions, MinMaxScaler for bounded ranges, and RobustScaler when outliers are present. Always fit scalers on training data only to prevent data leakage, and save fitted scalers for consistent preprocessing in production.

**Next Step**: Learn about [encoding categorical variables](./encoding-categorical.md) for non-numeric features.

---

*Last Updated: 2026-02-09*

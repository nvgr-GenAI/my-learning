# Regularization (L1, L2, Elastic Net)

## Overview

Regularization prevents overfitting by adding a penalty term to the loss function, discouraging complex models. It's one of the most important techniques for building models that generalize well.

**What You'll Learn:**
- L1 (Lasso) regularization
- L2 (Ridge) regularization
- Elastic Net (L1 + L2)
- When to use each
- Regularization strength (alpha/lambda)
- Implementation with sklearn and deep learning
- Early stopping as regularization

**Prerequisites:** Understanding of overfitting and loss functions

---

## Why Regularization?

### The Overfitting Problem

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate data with noise
np.random.seed(42)
X = np.sort(np.random.rand(50, 1) * 10, axis=0)
y = 2 * X.squeeze() + 3 + np.random.randn(50) * 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit high-degree polynomial (prone to overfitting)
poly = PolynomialFeatures(degree=15)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

# Plot
X_plot = np.linspace(0, 10, 200).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)
y_plot = model.predict(X_plot_poly)

plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, label='Training data', alpha=0.6)
plt.scatter(X_test, y_test, label='Test data', alpha=0.6, color='red')
plt.plot(X_plot, y_plot, 'g-', label='15th degree polynomial (overfit)', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Overfitting Example', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-5, 30)
plt.show()

print(f"Training Score: {model.score(X_train_poly, y_train):.4f}")
print(f"Test Score: {model.score(X_test_poly, y_test):.4f}")
print(f"Number of features: {X_train_poly.shape[1]}")
```

**Solution:** Add regularization to prevent extreme weights.

---

## L2 Regularization (Ridge)

### Concept

Add penalty proportional to **square of weights** to the loss function.

**Modified Loss:**
```
J(θ) = Original_Loss + λ * Σ(θ²)
     = Original_Loss + λ * ||θ||²₂
```

Where:
- λ (lambda/alpha): Regularization strength
- Larger λ → More regularization → Simpler model
- λ = 0 → No regularization (standard regression)

### Properties

- **Shrinks weights** toward zero (but not exactly zero)
- **Smooth penalty** (differentiable everywhere)
- **Prefers many small weights** over few large weights
- **Works well** when many features are relevant

### Implementation

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# L2 Regularization with sklearn
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

# Different regularization strengths
alphas = [0, 0.01, 0.1, 1, 10, 100]

plt.figure(figsize=(14, 10))

for idx, alpha in enumerate(alphas, 1):
    # Train Ridge model
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)

    # Plot
    plt.subplot(2, 3, idx)

    X_plot_scaled = scaler.transform(X_plot_poly)
    y_plot = ridge.predict(X_plot_scaled)

    plt.scatter(X_train, y_train, alpha=0.6, label='Train')
    plt.scatter(X_test, y_test, alpha=0.6, color='red', label='Test')
    plt.plot(X_plot, y_plot, 'g-', linewidth=2)
    plt.title(f'Ridge α={alpha}\nTrain R²={ridge.score(X_train_scaled, y_train):.3f}, '
              f'Test R²={ridge.score(X_test_scaled, y_test):.3f}',
              fontweight='bold')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-5, 30)

plt.tight_layout()
plt.show()

# Compare coefficients
print("\nCoefficient magnitudes:")
for alpha in [0, 1, 10]:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    print(f"α={alpha:3}: Max |coef| = {np.abs(ridge.coef_).max():.2f}, "
          f"L2 norm = {np.linalg.norm(ridge.coef_):.2f}")
```

---

## L1 Regularization (Lasso)

### Concept

Add penalty proportional to **absolute value of weights** to the loss function.

**Modified Loss:**
```
J(θ) = Original_Loss + λ * Σ|θ|
     = Original_Loss + λ * ||θ||₁
```

### Properties

- **Sparse solutions**: Sets some weights to exactly zero
- **Feature selection**: Automatically selects important features
- **Non-differentiable** at zero (requires special optimization)
- **Prefers few large weights** over many small weights

### Implementation

```python
from sklearn.linear_model import Lasso

# L1 Regularization with sklearn
alphas = [0.001, 0.01, 0.1, 1, 10, 100]

plt.figure(figsize=(14, 10))

for idx, alpha in enumerate(alphas, 1):
    # Train Lasso model
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)

    # Plot
    plt.subplot(2, 3, idx)

    y_plot = lasso.predict(X_plot_scaled)

    plt.scatter(X_train, y_train, alpha=0.6, label='Train')
    plt.scatter(X_test, y_test, alpha=0.6, color='red', label='Test')
    plt.plot(X_plot, y_plot, 'g-', linewidth=2)

    # Count non-zero coefficients
    n_nonzero = np.sum(lasso.coef_ != 0)

    plt.title(f'Lasso α={alpha}\nNon-zero coefs: {n_nonzero}/{len(lasso.coef_)}\n'
              f'Test R²={lasso.score(X_test_scaled, y_test):.3f}',
              fontweight='bold')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-5, 30)

plt.tight_layout()
plt.show()

# Analyze sparsity
print("\nLasso Sparsity Analysis:")
for alpha in [0.01, 0.1, 1, 10]:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    n_nonzero = np.sum(lasso.coef_ != 0)
    print(f"α={alpha:5}: {n_nonzero:2}/{len(lasso.coef_)} non-zero coefficients")
```

---

## Elastic Net (L1 + L2)

### Concept

Combines both L1 and L2 regularization - best of both worlds.

**Modified Loss:**
```
J(θ) = Original_Loss + λ₁ * Σ|θ| + λ₂ * Σ(θ²)
     = Original_Loss + α * (l1_ratio * ||θ||₁ + (1-l1_ratio) * ||θ||²₂)
```

### Properties

- **Balances** L1 sparsity and L2 stability
- **Feature selection** like L1
- **Groups correlated features** like L2
- **More robust** than pure L1 or L2

### Implementation

```python
from sklearn.linear_model import ElasticNet

# Elastic Net with different l1_ratio
l1_ratios = [0, 0.25, 0.5, 0.75, 1.0]  # 0=Ridge, 1=Lasso

plt.figure(figsize=(14, 10))

for idx, l1_ratio in enumerate(l1_ratios, 1):
    # Train Elastic Net
    elastic = ElasticNet(alpha=0.1, l1_ratio=l1_ratio, max_iter=10000)
    elastic.fit(X_train_scaled, y_train)

    # Plot
    plt.subplot(2, 3, idx)

    y_plot = elastic.predict(X_plot_scaled)

    plt.scatter(X_train, y_train, alpha=0.6, label='Train')
    plt.scatter(X_test, y_test, alpha=0.6, color='red', label='Test')
    plt.plot(X_plot, y_plot, 'g-', linewidth=2)

    n_nonzero = np.sum(elastic.coef_ != 0)
    penalty_type = 'Ridge' if l1_ratio == 0 else 'Lasso' if l1_ratio == 1 else 'Elastic Net'

    plt.title(f'{penalty_type} (l1_ratio={l1_ratio})\n'
              f'Non-zero: {n_nonzero}/{len(elastic.coef_)}\n'
              f'Test R²={elastic.score(X_test_scaled, y_test):.3f}',
              fontweight='bold')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-5, 30)

plt.tight_layout()
plt.show()
```

---

## Comparison: L1 vs L2 vs Elastic Net

### Visual Comparison

```python
import pandas as pd

# Train all models with same alpha
alpha = 0.1

models = {
    'No Regularization': LinearRegression(),
    'L2 (Ridge)': Ridge(alpha=alpha),
    'L1 (Lasso)': Lasso(alpha=alpha, max_iter=10000),
    'Elastic Net': ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000)
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

    results.append({
        'Model': name,
        'Train R²': model.score(X_train_scaled, y_train),
        'Test R²': model.score(X_test_scaled, y_test),
        'Non-zero coefs': np.sum(model.coef_ != 0) if hasattr(model, 'coef_') else 'N/A',
        'L2 norm': np.linalg.norm(model.coef_) if hasattr(model, 'coef_') else 'N/A'
    })

df_results = pd.DataFrame(results)
print("\nModel Comparison:")
print("="*70)
print(df_results.to_string(index=False))
```

### Decision Guide

| Use Case | Recommended | Reason |
|----------|-------------|---------|
| Many relevant features | **L2 (Ridge)** | Shrinks all weights smoothly |
| Few relevant features | **L1 (Lasso)** | Automatic feature selection |
| Correlated features | **Elastic Net** | Groups correlated features |
| Feature selection needed | **L1 or Elastic Net** | Creates sparse solutions |
| Stability important | **L2 or Elastic Net** | Less sensitive to data |
| Don't know | **Elastic Net** | Good compromise |

---

## Regularization in Deep Learning

### L2 Regularization in Neural Networks

```python
import torch
import torch.nn as nn
import torch.optim as optim

# PyTorch implementation
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# L2 regularization via weight_decay parameter
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)  # L2 reg

# Training loop
criterion = nn.MSELoss()

for epoch in range(100):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Keras/TensorFlow Implementation

```python
from tensorflow import keras
from tensorflow.keras import regularizers

# L2 regularization
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu',
                       kernel_regularizer=regularizers.l2(0.01)),  # L2
    keras.layers.Dense(1)
])

# L1 regularization
model_l1 = keras.Sequential([
    keras.layers.Dense(64, activation='relu',
                       kernel_regularizer=regularizers.l1(0.01)),  # L1
    keras.layers.Dense(1)
])

# Elastic Net (L1 + L2)
model_elastic = keras.Sequential([
    keras.layers.Dense(64, activation='relu',
                       kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
```

---

## Early Stopping

### Concept

Stop training when validation loss stops improving - acts as regularization by preventing overfitting.

### Implementation

```python
from sklearn.model_selection import train_test_split

# Split data: train, validation, test
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42
)

# Keras with early stopping
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,  # Wait 10 epochs for improvement
    restore_best_weights=True  # Restore best model
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=1000,
    callbacks=[early_stop],
    verbose=0
)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.axvline(x=early_stop.stopped_epoch, color='r', linestyle='--',
            label=f'Early Stop (epoch {early_stop.stopped_epoch})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Early Stopping', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Choosing Regularization Strength (λ/α)

### Cross-Validation for Optimal Alpha

```python
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

# Ridge with automatic alpha selection
ridge_cv = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100], cv=5)
ridge_cv.fit(X_train_scaled, y_train)

print(f"Best alpha (Ridge): {ridge_cv.alpha_}")
print(f"Test R²: {ridge_cv.score(X_test_scaled, y_test):.4f}")

# Lasso with automatic alpha selection
lasso_cv = LassoCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1], cv=5, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)

print(f"\nBest alpha (Lasso): {lasso_cv.alpha_:.6f}")
print(f"Test R²: {lasso_cv.score(X_test_scaled, y_test):.4f}")

# Elastic Net with automatic alpha selection
elastic_cv = ElasticNetCV(
    alphas=[0.0001, 0.001, 0.01, 0.1, 1],
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
    cv=5,
    max_iter=10000
)
elastic_cv.fit(X_train_scaled, y_train)

print(f"\nBest alpha (Elastic Net): {elastic_cv.alpha_:.6f}")
print(f"Best l1_ratio: {elastic_cv.l1_ratio_}")
print(f"Test R²: {elastic_cv.score(X_test_scaled, y_test):.4f}")
```

### Regularization Path

```python
from sklearn.linear_model import ridge_regression, lasso_path

# Ridge regularization path
alphas = np.logspace(-2, 2, 50)
coefs_ridge = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    coefs_ridge.append(ridge.coef_)

coefs_ridge = np.array(coefs_ridge)

# Lasso regularization path
alphas_lasso, coefs_lasso, _ = lasso_path(X_train_scaled, y_train, alphas=alphas)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Ridge path
axes[0].plot(alphas, coefs_ridge)
axes[0].set_xscale('log')
axes[0].set_xlabel('Alpha (λ)')
axes[0].set_ylabel('Coefficients')
axes[0].set_title('Ridge Regularization Path', fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Lasso path
axes[1].plot(alphas_lasso, coefs_lasso.T)
axes[1].set_xscale('log')
axes[1].set_xlabel('Alpha (λ)')
axes[1].set_ylabel('Coefficients')
axes[1].set_title('Lasso Regularization Path (Note sparsity)', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Common Mistakes

### 1. Not Scaling Features

**Problem:** Regularization penalizes large coefficients, but feature scale affects coefficient magnitude.

**Solution:** Always scale features before regularization.

```python
# WRONG
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)  # Unscaled features

# CORRECT
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
ridge.fit(X_train_scaled, y_train)
```

### 2. Regularizing the Bias Term

**Problem:** Bias term should not be regularized.

**Solution:** sklearn handles this correctly by default (fit_intercept=True).

### 3. Using Same Alpha for All Features

**Problem:** Different features may need different regularization.

**Solution:** Use Elastic Net or feature-specific regularization in deep learning.

---

## Best Practices

1. **Always scale features** before regularization
2. **Start with Elastic Net** (good default)
3. **Use cross-validation** to select alpha
4. **Monitor both training and validation loss**
5. **Combine with early stopping** for deep learning
6. **Feature engineering first**, regularization second
7. **Larger alpha → simpler model** (start large, decrease if underfitting)

---

## Related Topics

- [Ridge Regression](../supervised-learning/regression/ridge-regression.md) - L2 in detail
- [Lasso Regression](../supervised-learning/regression/lasso-regression.md) - L1 in detail
- [Dropout](dropout.md) - Regularization for neural networks
- [Batch Normalization](batch-normalization.md) - Also acts as regularization

---

## Summary

**Key Takeaways:**
1. **L2 (Ridge)**: Shrinks weights, good for many relevant features
2. **L1 (Lasso)**: Creates sparsity, automatic feature selection
3. **Elastic Net**: Best of both, recommended default
4. **Alpha controls strength**: larger → more regularization
5. **Always scale features** before applying regularization
6. **Use CV** to select optimal alpha
7. **Early stopping** is also a form of regularization

Regularization is essential for preventing overfitting and building models that generalize!

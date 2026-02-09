# Gradient Descent

## Overview

Gradient descent is the fundamental optimization algorithm used to train machine learning models. It iteratively adjusts model parameters to minimize a loss function by following the negative gradient.

**What You'll Learn:**
- Core gradient descent algorithm
- Variants: Batch, Mini-batch, Stochastic
- Learning rate importance
- Convergence analysis
- Implementation from scratch
- Visualization techniques
- Common issues and solutions

**Prerequisites:** Basic calculus (derivatives), understanding of loss functions

---

## The Intuition

### Mountain Analogy

Imagine you're on a mountain in fog and want to reach the valley (minimum):
- You can't see far ahead (no global view)
- You feel the slope under your feet (gradient)
- You take small steps downhill (learning rate)
- Eventually reach the bottom (convergence)

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a simple 2D loss surface
def loss_surface(w1, w2):
    return w1**2 + w2**2 + 0.5

# Create grid
w1 = np.linspace(-5, 5, 100)
w2 = np.linspace(-5, 5, 100)
W1, W2 = np.meshgrid(w1, w2)
Z = loss_surface(W1, W2)

# Plot
fig = plt.figure(figsize=(14, 6))

# 3D surface
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(W1, W2, Z, cmap='viridis', alpha=0.8)
ax1.set_xlabel('Weight 1')
ax1.set_ylabel('Weight 2')
ax1.set_zlabel('Loss')
ax1.set_title('Loss Surface (3D)', fontweight='bold')

# Contour plot
ax2 = fig.add_subplot(122)
contour = ax2.contour(W1, W2, Z, levels=20, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)
ax2.set_xlabel('Weight 1')
ax2.set_ylabel('Weight 2')
ax2.set_title('Loss Surface (Contour)', fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## The Algorithm

### Mathematical Formulation

**Goal:** Minimize loss function J(θ)

**Update Rule:**
```
θ_new = θ_old - α * ∇J(θ)
```

Where:
- θ: Model parameters (weights)
- α: Learning rate (step size)
- ∇J(θ): Gradient of loss w.r.t. parameters

### Simple Implementation

```python
def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    """
    Simple gradient descent for linear regression.

    Minimize: J(w, b) = (1/2m) * Σ(h(x) - y)²
    where h(x) = w*x + b
    """
    m, n = X.shape
    w = np.zeros(n)  # Initialize weights
    b = 0  # Initialize bias

    loss_history = []

    for iteration in range(n_iterations):
        # Forward pass (predictions)
        y_pred = X @ w + b

        # Compute loss
        loss = (1/(2*m)) * np.sum((y_pred - y)**2)
        loss_history.append(loss)

        # Compute gradients
        dw = (1/m) * X.T @ (y_pred - y)
        db = (1/m) * np.sum(y_pred - y)

        # Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Print progress
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Loss = {loss:.4f}")

    return w, b, loss_history

# Example: Simple linear regression
np.random.seed(42)
X = np.random.randn(100, 1)
y = 3 * X.squeeze() + 2 + np.random.randn(100) * 0.5

# Run gradient descent
w, b, loss_history = gradient_descent(X, y, learning_rate=0.1, n_iterations=1000)

print(f"\nFinal parameters:")
print(f"Weight: {w[0]:.4f} (true: 3.0)")
print(f"Bias: {b:.4f} (true: 2.0)")

# Plot convergence
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Gradient Descent Convergence', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Variants of Gradient Descent

### 1. Batch Gradient Descent

**Uses:** All training samples for each update

**Pros:**
- Stable convergence
- Can vectorize efficiently
- Theoretical guarantees

**Cons:**
- Slow for large datasets
- Memory intensive
- Can get stuck in local minima

```python
def batch_gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    """
    Batch GD: Use all samples for each update.
    """
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    loss_history = []

    for iteration in range(n_iterations):
        # Use ALL samples
        y_pred = X @ w + b
        loss = (1/(2*m)) * np.sum((y_pred - y)**2)
        loss_history.append(loss)

        # Gradients from all samples
        dw = (1/m) * X.T @ (y_pred - y)
        db = (1/m) * np.sum(y_pred - y)

        # Single update per iteration
        w = w - learning_rate * dw
        b = b - learning_rate * db

    return w, b, loss_history
```

### 2. Stochastic Gradient Descent (SGD)

**Uses:** One random sample for each update

**Pros:**
- Fast updates
- Can escape local minima (noise helps)
- Works well for large datasets
- Online learning possible

**Cons:**
- Noisy convergence
- Never truly converges (oscillates)
- Requires learning rate decay

```python
def stochastic_gradient_descent(X, y, learning_rate=0.01, n_epochs=10):
    """
    SGD: Use one random sample for each update.
    """
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    loss_history = []

    for epoch in range(n_epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(m)

        for i in indices:
            # Use only ONE sample
            x_i = X[i:i+1]
            y_i = y[i:i+1]

            # Prediction
            y_pred = x_i @ w + b

            # Gradients from single sample
            dw = x_i.T @ (y_pred - y_i)
            db = (y_pred - y_i)[0]

            # Update immediately
            w = w - learning_rate * dw.flatten()
            b = b - learning_rate * db

        # Calculate loss at end of epoch
        y_pred_all = X @ w + b
        loss = (1/(2*m)) * np.sum((y_pred_all - y)**2)
        loss_history.append(loss)

        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    return w, b, loss_history
```

### 3. Mini-Batch Gradient Descent

**Uses:** Small batch of samples for each update

**Pros:**
- Best of both worlds (batch & SGD)
- Efficient (GPU parallelization)
- Stable yet fast
- Most commonly used in practice

**Cons:**
- Requires choosing batch size
- Memory usage depends on batch size

```python
def mini_batch_gradient_descent(X, y, learning_rate=0.01, n_epochs=10, batch_size=32):
    """
    Mini-batch GD: Use small batches for each update.
    """
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    loss_history = []

    for epoch in range(n_epochs):
        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Process mini-batches
        for start_idx in range(0, m, batch_size):
            end_idx = min(start_idx + batch_size, m)

            # Get mini-batch
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            batch_m = len(X_batch)

            # Predictions
            y_pred = X_batch @ w + b

            # Gradients from mini-batch
            dw = (1/batch_m) * X_batch.T @ (y_pred - y_batch)
            db = (1/batch_m) * np.sum(y_pred - y_batch)

            # Update
            w = w - learning_rate * dw
            b = b - learning_rate * db

        # Calculate loss at end of epoch
        y_pred_all = X @ w + b
        loss = (1/(2*m)) * np.sum((y_pred_all - y)**2)
        loss_history.append(loss)

        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    return w, b, loss_history
```

### Comparison

```python
# Compare all three variants
np.random.seed(42)
X = np.random.randn(1000, 1)
y = 3 * X.squeeze() + 2 + np.random.randn(1000) * 0.5

# Run all variants
print("Batch Gradient Descent:")
w_batch, b_batch, loss_batch = batch_gradient_descent(
    X, y, learning_rate=0.1, n_iterations=100
)

print("\nStochastic Gradient Descent:")
w_sgd, b_sgd, loss_sgd = stochastic_gradient_descent(
    X, y, learning_rate=0.01, n_epochs=10
)

print("\nMini-Batch Gradient Descent:")
w_mini, b_mini, loss_mini = mini_batch_gradient_descent(
    X, y, learning_rate=0.05, n_epochs=10, batch_size=32
)

# Plot comparison
plt.figure(figsize=(12, 6))
plt.plot(loss_batch, label='Batch GD', linewidth=2)
plt.plot(loss_sgd, label='SGD', linewidth=2)
plt.plot(loss_mini, label='Mini-Batch GD', linewidth=2)
plt.xlabel('Iteration/Epoch')
plt.ylabel('Loss')
plt.title('Comparison of Gradient Descent Variants', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.show()

# Results
results = pd.DataFrame({
    'Method': ['Batch', 'SGD', 'Mini-Batch'],
    'Weight': [w_batch[0], w_sgd[0], w_mini[0]],
    'Bias': [b_batch, b_sgd, b_mini],
    'Final Loss': [loss_batch[-1], loss_sgd[-1], loss_mini[-1]]
})
print("\nFinal Results:")
print(results.to_string(index=False))
```

---

## Learning Rate

### The Critical Hyperparameter

**Too Small:** Slow convergence, may never reach minimum
**Too Large:** Divergence, oscillation, instability
**Just Right:** Fast and stable convergence

```python
def demonstrate_learning_rates(X, y):
    """
    Show effect of different learning rates.
    """
    learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]

    plt.figure(figsize=(14, 8))

    for lr in learning_rates:
        _, _, loss_history = batch_gradient_descent(
            X, y, learning_rate=lr, n_iterations=100
        )

        plt.plot(loss_history, label=f'LR = {lr}', linewidth=2)

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Effect of Learning Rate on Convergence', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

# Demonstrate
demonstrate_learning_rates(X, y)
```

### Finding Good Learning Rate

```python
def learning_rate_finder(X, y, start_lr=1e-6, end_lr=10, num_iter=100):
    """
    Learning rate finder: gradually increase LR to find optimal range.
    """
    m, n = X.shape
    w = np.zeros(n)
    b = 0

    # Exponentially increase learning rate
    lrs = np.logspace(np.log10(start_lr), np.log10(end_lr), num_iter)
    losses = []

    for lr in lrs:
        # Forward pass
        y_pred = X @ w + b
        loss = (1/(2*m)) * np.sum((y_pred - y)**2)
        losses.append(loss)

        # If loss is exploding, stop
        if len(losses) > 1 and losses[-1] > losses[-2] * 4:
            break

        # Backward pass
        dw = (1/m) * X.T @ (y_pred - y)
        db = (1/m) * np.sum(y_pred - y)

        # Update
        w = w - lr * dw
        b = b - lr * db

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(lrs[:len(losses)], losses, linewidth=2)
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Learning Rate Finder', fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Find optimal LR (steepest descent)
    gradients = np.gradient(losses)
    optimal_idx = np.argmin(gradients)
    optimal_lr = lrs[optimal_idx]

    plt.axvline(optimal_lr, color='r', linestyle='--',
                label=f'Optimal LR ≈ {optimal_lr:.4f}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Suggested learning rate: {optimal_lr:.6f}")
    return optimal_lr

# Find optimal learning rate
optimal_lr = learning_rate_finder(X, y)
```

---

## Visualization

### Gradient Descent Path

```python
def visualize_gd_path(learning_rate=0.1):
    """
    Visualize gradient descent path on 2D loss surface.
    """
    # Simple quadratic loss: J(w1, w2) = w1^2 + w2^2
    def loss(w):
        return w[0]**2 + w[1]**2

    def gradient(w):
        return np.array([2*w[0], 2*w[1]])

    # Initialize
    w = np.array([4.0, 4.0])
    path = [w.copy()]

    # Run GD
    for _ in range(50):
        grad = gradient(w)
        w = w - learning_rate * grad
        path.append(w.copy())

    path = np.array(path)

    # Create loss surface
    w1 = np.linspace(-5, 5, 100)
    w2 = np.linspace(-5, 5, 100)
    W1, W2 = np.meshgrid(w1, w2)
    Z = W1**2 + W2**2

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 3D path
    ax1 = axes[0]
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(W1, W2, Z, alpha=0.3, cmap='viridis')
    loss_path = [loss(w) for w in path]
    ax1.plot(path[:, 0], path[:, 1], loss_path, 'r.-', linewidth=2, markersize=8)
    ax1.set_xlabel('Weight 1')
    ax1.set_ylabel('Weight 2')
    ax1.set_zlabel('Loss')
    ax1.set_title(f'GD Path (LR={learning_rate})', fontweight='bold')

    # 2D contour with path
    ax2 = axes[1]
    contour = ax2.contour(W1, W2, Z, levels=30, cmap='viridis', alpha=0.6)
    ax2.plot(path[:, 0], path[:, 1], 'r.-', linewidth=2, markersize=8)
    ax2.scatter([0], [0], c='green', s=200, marker='*', label='Optimum')
    ax2.scatter(path[0,0], path[0,1], c='red', s=100, marker='o', label='Start')
    ax2.set_xlabel('Weight 1')
    ax2.set_ylabel('Weight 2')
    ax2.set_title('GD Path (Contour)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Visualize different learning rates
for lr in [0.05, 0.1, 0.5]:
    visualize_gd_path(learning_rate=lr)
```

---

## Common Issues

### 1. Divergence (Learning Rate Too Large)

```python
# Demonstrate divergence
X_div = np.random.randn(100, 1)
y_div = 3 * X_div.squeeze() + 2

print("With appropriate learning rate:")
_, _, loss_good = batch_gradient_descent(X_div, y_div, learning_rate=0.1, n_iterations=50)

print("\nWith too large learning rate:")
_, _, loss_bad = batch_gradient_descent(X_div, y_div, learning_rate=1.0, n_iterations=50)

plt.figure(figsize=(10, 5))
plt.plot(loss_good, label='LR=0.1 (Good)', linewidth=2)
plt.plot(loss_bad, label='LR=1.0 (Diverging)', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Divergence with Large Learning Rate', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 2. Slow Convergence (Learning Rate Too Small)

**Solution:** Learning rate scheduling or adaptive optimizers

### 3. Local Minima

**Solution:**
- Use SGD (noise helps escape)
- Multiple random initializations
- Advanced optimizers (Adam, etc.)

### 4. Vanishing/Exploding Gradients

**Solution:**
- Gradient clipping
- Batch normalization
- Proper weight initialization
- Better activation functions

```python
def gradient_clipping(gradients, max_norm=1.0):
    """Clip gradients to prevent explosion."""
    norm = np.linalg.norm(gradients)
    if norm > max_norm:
        gradients = gradients * (max_norm / norm)
    return gradients

# Example usage in GD
def gradient_descent_with_clipping(X, y, learning_rate=0.01, n_iterations=1000, clip_norm=1.0):
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    loss_history = []

    for iteration in range(n_iterations):
        y_pred = X @ w + b
        loss = (1/(2*m)) * np.sum((y_pred - y)**2)
        loss_history.append(loss)

        # Compute gradients
        dw = (1/m) * X.T @ (y_pred - y)
        db = (1/m) * np.sum(y_pred - y)

        # Clip gradients
        dw = gradient_clipping(dw, max_norm=clip_norm)

        # Update
        w = w - learning_rate * dw
        b = b - learning_rate * db

    return w, b, loss_history
```

---

## Convergence Analysis

### Monitoring Convergence

```python
def check_convergence(loss_history, tolerance=1e-6, patience=10):
    """
    Check if gradient descent has converged.
    """
    if len(loss_history) < patience + 1:
        return False

    # Check if loss improvement is below tolerance for 'patience' iterations
    recent_losses = loss_history[-patience-1:]
    improvements = [recent_losses[i] - recent_losses[i+1]
                   for i in range(len(recent_losses)-1)]

    return all(imp < tolerance for imp in improvements)

# Example
_, _, loss_history = batch_gradient_descent(X, y, learning_rate=0.1, n_iterations=1000)

for i in range(0, len(loss_history), 100):
    converged = check_convergence(loss_history[:i+1])
    print(f"Iteration {i}: Converged = {converged}, Loss = {loss_history[i]:.6f}")
```

---

## Best Practices

1. **Start with reasonable learning rate** (0.001 - 0.1)
2. **Use learning rate finder** to determine optimal range
3. **Monitor loss during training** (should decrease)
4. **Use mini-batch GD** for best performance
5. **Implement early stopping** to prevent overfitting
6. **Normalize/standardize features** for faster convergence
7. **Use momentum or adaptive optimizers** (Adam) for better convergence

---

## Related Topics

- [Optimizers](optimizers.md) - Advanced optimization algorithms
- [Learning Rate Scheduling](learning-rate-scheduling.md) - Adaptive learning rates
- [Batch Normalization](batch-normalization.md) - Stabilizing training
- [Regularization](regularization.md) - Preventing overfitting

---

## Summary

**Key Takeaways:**
1. Gradient descent **minimizes loss** by following negative gradient
2. **Three variants:** Batch (stable), SGD (fast), Mini-batch (best)
3. **Learning rate** is critical: too large → divergence, too small → slow
4. **Mini-batch GD** is most commonly used in practice
5. **Monitor convergence** and use early stopping
6. **Feature scaling** helps convergence
7. Modern frameworks provide **advanced optimizers** (Adam, RMSprop)

Gradient descent is the workhorse of machine learning optimization!

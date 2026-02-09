# Optimizers (SGD, Momentum, Adam, RMSprop, AdaGrad)

## Overview

Optimizers are advanced algorithms that improve upon basic gradient descent by adapting learning rates and using momentum. They make training faster, more stable, and help escape local minima.

**What You'll Learn:**
- SGD with Momentum
- RMSprop, AdaGrad, Adam optimizers
- When to use each optimizer
- Hyperparameter tuning for optimizers
- Implementation and comparison
- Best practices

**Prerequisites:** Understanding of gradient descent

---

## The Problem with Vanilla Gradient Descent

**Issues:**
1. Same learning rate for all parameters
2. Can oscillate in steep dimensions
3. Slow convergence in gentle slopes
4. Gets stuck in saddle points
5. Sensitive to learning rate choice

**Solution:** Advanced optimizers that adapt!

---

## 1. SGD with Momentum

### Intuition

Like a ball rolling downhill - gains momentum in consistent directions, dampens oscillations.

**Update Rule:**
```
v_t = β * v_{t-1} + (1-β) * ∇J(θ)
θ_t = θ_{t-1} - α * v_t
```

Where:
- v: Velocity (momentum term)
- β: Momentum coefficient (typically 0.9)
- α: Learning rate

### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

class SGDMomentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, params, grads):
        if self.velocity is None:
            self.velocity = {k: np.zeros_like(v) for k, v in params.items()}

        for key in params:
            # Update velocity
            self.velocity[key] = (self.momentum * self.velocity[key]
                                 - self.lr * grads[key])
            # Update parameters
            params[key] += self.velocity[key]

        return params
```

**When to Use:**
- Default choice over vanilla SGD
- Works well for most problems
- Helps escape local minima

**Hyperparameters:**
- β = 0.9 (typical default)
- Can try 0.95 or 0.99 for more momentum

---

## 2. AdaGrad (Adaptive Gradient)

### Intuition

Adapts learning rate for each parameter - larger updates for infrequent parameters, smaller for frequent ones.

**Update Rule:**
```
G_t = G_{t-1} + (∇J(θ))²    # Accumulate squared gradients
θ_t = θ_{t-1} - (α / √(G_t + ε)) * ∇J(θ)
```

### Implementation

```python
class AdaGrad:
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self.lr = learning_rate
        self.epsilon = epsilon
        self.G = None  # Sum of squared gradients

    def update(self, params, grads):
        if self.G is None:
            self.G = {k: np.zeros_like(v) for k, v in params.items()}

        for key in params:
            # Accumulate squared gradients
            self.G[key] += grads[key] ** 2

            # Adaptive learning rate
            adapted_lr = self.lr / (np.sqrt(self.G[key]) + self.epsilon)

            # Update parameters
            params[key] -= adapted_lr * grads[key]

        return params
```

**When to Use:**
- Sparse data (NLP, recommender systems)
- When different features have very different update frequencies

**Pros:**
- No need to manually tune learning rate
- Works well for sparse features

**Cons:**
- Learning rate can become very small
- May stop learning too early

---

## 3. RMSprop (Root Mean Square Propagation)

### Intuition

Fixes AdaGrad's diminishing learning rate by using exponential moving average of squared gradients.

**Update Rule:**
```
E[g²]_t = β * E[g²]_{t-1} + (1-β) * (∇J(θ))²
θ_t = θ_{t-1} - (α / √(E[g²]_t + ε)) * ∇J(θ)
```

### Implementation

```python
class RMSprop:
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        self.lr = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.E = None  # Moving average of squared gradients

    def update(self, params, grads):
        if self.E is None:
            self.E = {k: np.zeros_like(v) for k, v in params.items()}

        for key in params:
            # Exponential moving average of squared gradients
            self.E[key] = (self.beta * self.E[key] +
                          (1 - self.beta) * grads[key]**2)

            # Adaptive learning rate
            adapted_lr = self.lr / (np.sqrt(self.E[key]) + self.epsilon)

            # Update parameters
            params[key] -= adapted_lr * grads[key]

        return params
```

**When to Use:**
- Recurrent Neural Networks (RNNs)
- Non-stationary problems
- Better than AdaGrad for most cases

**Hyperparameters:**
- β = 0.9 (typical default)
- α = 0.001 (learning rate)

---

## 4. Adam (Adaptive Moment Estimation)

### Intuition

Combines the best of Momentum and RMSprop - uses both first and second moments of gradients.

**Update Rule:**
```
m_t = β1 * m_{t-1} + (1-β1) * ∇J(θ)           # First moment (momentum)
v_t = β2 * v_{t-1} + (1-β2) * (∇J(θ))²        # Second moment (RMSprop)

m̂_t = m_t / (1 - β1^t)                        # Bias correction
v̂_t = v_t / (1 - β2^t)                        # Bias correction

θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
```

### Implementation

```python
class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step

    def update(self, params, grads):
        if self.m is None:
            self.m = {k: np.zeros_like(v) for k, v in params.items()}
            self.v = {k: np.zeros_like(v) for k, v in params.items()}

        self.t += 1

        for key in params:
            # Update biased first moment
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]

            # Update biased second moment
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key]**2

            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)

            # Update parameters
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return params
```

**When to Use:**
- **Default choice for deep learning**
- Works well out-of-the-box
- Good for most problems

**Hyperparameters (defaults work well):**
- α = 0.001 (learning rate)
- β1 = 0.9 (first moment decay)
- β2 = 0.999 (second moment decay)

---

## Optimizer Selection Guide

| Optimizer | When to Use | Pros | Cons | Default LR |
|-----------|-------------|------|------|-----------|
| **SGD** | Simple problems | Simple, well understood | Slow, sensitive to LR | 0.01 |
| **Momentum** | Better than vanilla SGD | Faster convergence | One more hyperparameter | 0.01 |
| **AdaGrad** | Sparse data (NLP) | Adaptive LR per parameter | LR decay too aggressive | 0.01 |
| **RMSprop** | RNNs, non-stationary | Fixes AdaGrad's decay | Can be unstable | 0.001 |
| **Adam** | **Default for deep learning** | Works well out-of-box | Memory overhead | 0.001 |

---

## Using with PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# Different optimizers
optimizers_pytorch = {
    'SGD': optim.SGD(model.parameters(), lr=0.01),
    'SGD+Momentum': optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    'AdaGrad': optim.Adagrad(model.parameters(), lr=0.01),
    'RMSprop': optim.RMSprop(model.parameters(), lr=0.001),
    'Adam': optim.Adam(model.parameters(), lr=0.001),
    'AdamW': optim.AdamW(model.parameters(), lr=0.001)  # Adam with weight decay
}

# Training loop
def train_with_optimizer(model, optimizer, X, y, n_epochs=100):
    model.train()
    criterion = nn.MSELoss()

    for epoch in range(n_epochs):
        # Forward pass
        predictions = model(X)
        loss = criterion(predictions, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item():.4f}')
```

---

## Using with Keras/TensorFlow

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1)
])

# Different optimizers
optimizers_keras = {
    'SGD': keras.optimizers.SGD(learning_rate=0.01),
    'SGD+Momentum': keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    'AdaGrad': keras.optimizers.Adagrad(learning_rate=0.01),
    'RMSprop': keras.optimizers.RMSprop(learning_rate=0.001),
    'Adam': keras.optimizers.Adam(learning_rate=0.001)
}

# Compile and train
for name, optimizer in optimizers_keras.items():
    print(f"\nTraining with {name}")
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    # history = model.fit(X_train, y_train, epochs=100, verbose=0)
```

---

## Best Practices

1. **Start with Adam** - works well out-of-the-box
2. **Use default hyperparameters** first, then tune if needed
3. **Monitor training loss** - should decrease smoothly
4. **Learning rate is most important** hyperparameter
5. **Combine with learning rate scheduling** for best results
6. **Use gradient clipping** for RNNs (prevent explosion)
7. **AdamW for regularization** (Adam with weight decay)

---

## Common Hyperparameter Values

```python
# Good starting points
configs = {
    'SGD': {'lr': 0.01, 'momentum': 0.9},
    'Adam': {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999},
    'RMSprop': {'lr': 0.001, 'beta': 0.9},
    'AdaGrad': {'lr': 0.01}
}
```

---

## Related Topics

- [Gradient Descent](gradient-descent.md) - Foundation
- [Learning Rate Scheduling](learning-rate-scheduling.md) - Adaptive LR
- [Hyperparameter Tuning](hyperparameter-tuning.md) - Finding optimal settings
- [Batch Normalization](batch-normalization.md) - Stabilize training

---

## Summary

**Key Takeaways:**
1. **Adam is the default choice** for deep learning
2. **Momentum helps** escape local minima and accelerate convergence
3. **Adaptive optimizers** (RMSprop, Adam) adjust learning rate per parameter
4. **Start with defaults**, tune only if necessary
5. **Use with learning rate scheduling** for best results
6. **Different optimizers for different problems** (Adam for general, RMSprop for RNNs)

Remember: Adam works 90% of the time. Start there!

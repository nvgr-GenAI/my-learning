# Batch Normalization

## Overview

Batch Normalization normalizes the inputs of each layer to have zero mean and unit variance within each mini-batch. It accelerates training, acts as regularization, and allows higher learning rates.

**What You'll Learn:**
- How batch normalization works
- Why it's effective
- Implementation in Keras/PyTorch
- BatchNorm vs LayerNorm
- When and where to apply
- Common mistakes

**Prerequisites:** Understanding of neural networks and normalization

---

## The Problem: Internal Covariate Shift

### What is Internal Covariate Shift?

As network trains, distribution of inputs to each layer changes (shifts), making training unstable and slow.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate activations without normalization
layer1_output = np.random.randn(1000) * 0.5
layer2_output = np.random.randn(1000) * 2.0  # Different scale!
layer3_output = np.random.randn(1000) * 5.0  # Even more different!

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (layer_name, data) in enumerate([
    ('Layer 1', layer1_output),
    ('Layer 2', layer2_output),
    ('Layer 3', layer3_output)
]):
    axes[idx].hist(data, bins=50, alpha=0.7, edgecolor='black')
    axes[idx].set_title(f'{layer_name}\nMean: {data.mean():.2f}, Std: {data.std():.2f}',
                       fontweight='bold')
    axes[idx].set_xlabel('Activation Value')
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Internal Covariate Shift (Different Distributions)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

**Problem:** Gradients can vanish or explode due to unstable distributions.

---

## How Batch Normalization Works

### Algorithm

For each mini-batch during training:

```
1. Calculate batch mean: μ_B = (1/m) * Σx_i
2. Calculate batch variance: σ²_B = (1/m) * Σ(x_i - μ_B)²
3. Normalize: x̂_i = (x_i - μ_B) / √(σ²_B + ε)
4. Scale and shift: y_i = γ * x̂_i + β
```

Where:
- μ_B, σ²_B: Mean and variance of current batch
- ε: Small constant for numerical stability (1e-5)
- γ, β: Learnable parameters (scale and shift)

### Implementation from Scratch

```python
class BatchNormalization:
    def __init__(self, num_features, epsilon=1e-5, momentum=0.9):
        self.epsilon = epsilon
        self.momentum = momentum

        # Learnable parameters
        self.gamma = np.ones(num_features)  # Scale
        self.beta = np.zeros(num_features)  # Shift

        # Running statistics (for inference)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, X, training=True):
        if training:
            # Batch statistics
            batch_mean = X.mean(axis=0)
            batch_var = X.var(axis=0)

            # Normalize
            X_normalized = (X - batch_mean) / np.sqrt(batch_var + self.epsilon)

            # Update running statistics
            self.running_mean = (self.momentum * self.running_mean +
                                (1 - self.momentum) * batch_mean)
            self.running_var = (self.momentum * self.running_var +
                               (1 - self.momentum) * batch_var)
        else:
            # Use running statistics for inference
            X_normalized = ((X - self.running_mean) /
                          np.sqrt(self.running_var + self.epsilon))

        # Scale and shift
        output = self.gamma * X_normalized + self.beta

        return output

# Example
np.random.seed(42)
X_batch = np.random.randn(32, 10) * 5 + 3  # Batch of 32 samples, 10 features

bn = BatchNormalization(num_features=10)

# Training
X_normalized = bn.forward(X_batch, training=True)

print(f"Before BatchNorm:")
print(f"  Mean: {X_batch.mean(axis=0)[:3]}")
print(f"  Std:  {X_batch.std(axis=0)[:3]}")

print(f"\nAfter BatchNorm (training):")
print(f"  Mean: {X_normalized.mean(axis=0)[:3]}")
print(f"  Std:  {X_normalized.std(axis=0)[:3]}")
```

---

## Benefits of Batch Normalization

### 1. Faster Training

Allows higher learning rates without divergence.

### 2. Reduces Sensitivity to Initialization

Network less dependent on careful weight initialization.

### 3. Acts as Regularization

Slight regularization effect (reduces need for dropout).

### 4. Enables Deeper Networks

Stable gradients allow training very deep networks.

### Visualization

```python
# Simulate training curves
epochs = np.arange(1, 51)

# Without BatchNorm
loss_no_bn = 2 * np.exp(-epochs/20) + 0.3 + np.random.randn(50) * 0.1
acc_no_bn = (1 - np.exp(-epochs/15)) * 0.9

# With BatchNorm
loss_with_bn = 2 * np.exp(-epochs/10) + 0.1 + np.random.randn(50) * 0.05
acc_with_bn = (1 - np.exp(-epochs/8)) * 0.95

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(epochs, loss_no_bn, label='Without BatchNorm', linewidth=2)
axes[0].plot(epochs, loss_with_bn, label='With BatchNorm', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(epochs, acc_no_bn, label='Without BatchNorm', linewidth=2)
axes[1].plot(epochs, acc_with_bn, label='With BatchNorm', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training Accuracy', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Batch Normalization Benefits', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

---

## Implementation

### Keras/TensorFlow

```python
import tensorflow as tf
from tensorflow import keras

# Method 1: Sequential API
model = keras.Sequential([
    keras.layers.Dense(128, input_shape=(784,)),
    keras.layers.BatchNormalization(),  # After dense, before activation
    keras.layers.Activation('relu'),

    keras.layers.Dense(64),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),

    keras.layers.Dense(10, activation='softmax')
])

# Method 2: Functional API
inputs = keras.Input(shape=(784,))
x = keras.layers.Dense(128)(inputs)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.Dense(64)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
outputs = keras.layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

### PyTorch

```python
import torch
import torch.nn as nn

class ModelWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.bn1 = nn.BatchNorm1d(128)  # BatchNorm after linear layer

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  # Apply BatchNorm
        x = torch.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)

        x = self.fc3(x)
        return x

model = ModelWithBatchNorm()

# Training mode (uses batch statistics)
model.train()

# Evaluation mode (uses running statistics)
model.eval()
```

---

## Where to Place BatchNorm?

### Common Placements

**Option 1: After linear layer, before activation** (recommended)
```python
x = Dense(128)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
```

**Option 2: After activation**
```python
x = Dense(128)(x)
x = Activation('relu')(x)
x = BatchNormalization()(x)
```

**Current consensus:** Place before activation (Option 1).

### For CNNs

```python
# Convolutional layers
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)),
    keras.layers.BatchNormalization(),  # After conv, before activation
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(64, (3, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])
```

---

## BatchNorm Variants

### 1. Layer Normalization

Normalize across features (not batch dimension).

**Use case:** RNNs, Transformers (where batch size varies)

```python
# Keras
layer_norm = keras.layers.LayerNormalization()

# PyTorch
layer_norm = nn.LayerNorm(normalized_shape=128)
```

### 2. Instance Normalization

Normalize each sample independently.

**Use case:** Style transfer, GANs

```python
# PyTorch
instance_norm = nn.InstanceNorm1d(num_features=128)
```

### 3. Group Normalization

Divide features into groups and normalize within each group.

**Use case:** Small batch sizes

```python
# PyTorch
group_norm = nn.GroupNorm(num_groups=32, num_channels=128)
```

### Comparison

| Normalization | Normalizes Over | Best For |
|---------------|-----------------|----------|
| Batch Norm | Batch dimension | CNNs, large batches |
| Layer Norm | Feature dimension | RNNs, Transformers |
| Instance Norm | Each sample | Style transfer |
| Group Norm | Feature groups | Small batches |

---

## BatchNorm vs Dropout

### Can they be combined?

**Yes, but with care:**

```python
# Order matters!
model = keras.Sequential([
    keras.layers.Dense(128),
    keras.layers.BatchNormalization(),  # First BatchNorm
    keras.layers.Activation('relu'),
    keras.layers.Dropout(0.3),  # Then Dropout

    keras.layers.Dense(64),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(10, activation='softmax')
])
```

**Recommendation:**
- Use BatchNorm primarily
- Add Dropout only if still overfitting
- Lower dropout rate when using BatchNorm (0.2-0.3 instead of 0.5)

---

## Common Mistakes

### 1. Wrong Order of Operations

**WRONG:**
```python
x = Dense(128)(x)
x = Activation('relu')(x)
x = BatchNormalization()(x)  # Too late! After activation
```

**CORRECT:**
```python
x = Dense(128)(x)
x = BatchNormalization()(x)  # Before activation
x = Activation('relu')(x)
```

### 2. Not Using Training Mode Correctly

**WRONG (PyTorch):**
```python
model.train()
# ... training ...
predictions = model(X_test)  # Still in training mode!
```

**CORRECT:**
```python
model.train()
# ... training ...
model.eval()  # Switch to eval mode
with torch.no_grad():
    predictions = model(X_test)
```

### 3. Very Small Batch Sizes

**Problem:** BatchNorm unstable with batch_size < 16

**Solutions:**
- Use larger batch size
- Use Group Normalization or Layer Normalization
- Use synchronized BatchNorm (for distributed training)

### 4. Freezing BatchNorm During Fine-Tuning

**Problem:** When fine-tuning, frozen BatchNorm layers may cause issues.

**Solution:**
```python
# Keras: Set training=False for frozen layers
for layer in base_model.layers:
    layer.trainable = False
    if isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = False  # Keep using running stats
```

---

## Best Practices

1. **Place before activation** (current best practice)
2. **Use with moderate learning rates** (0.001-0.01)
3. **Batch size ≥ 16** for stability
4. **Always switch to eval mode** during inference
5. **Reduce dropout rates** when using BatchNorm
6. **Fine-tuning**: Be careful with frozen BatchNorm layers
7. **Small batches**: Consider Layer Norm or Group Norm

---

## When to Use Batch Normalization

### Use BatchNorm When:
- Training deep networks (> 5 layers)
- Using CNNs
- Batch size ≥ 16
- Want faster training
- Need higher learning rates

### Don't Use BatchNorm When:
- Very small batch sizes (< 8)
- RNNs (use Layer Norm instead)
- Online learning (single samples)
- Inference speed is critical

---

## Hyperparameters

```python
# Keras
BatchNormalization(
    momentum=0.99,  # Moving average momentum (default: 0.99)
    epsilon=1e-3,   # Numerical stability (default: 1e-3)
    center=True,    # Learn β (shift) parameter
    scale=True      # Learn γ (scale) parameter
)

# PyTorch
nn.BatchNorm1d(
    num_features=128,
    eps=1e-5,        # Numerical stability
    momentum=0.1,    # Moving average (different convention!)
    affine=True      # Learn γ and β
)
```

**Note:** PyTorch momentum is (1 - Keras momentum)!

---

## Related Topics

- [Dropout](dropout.md) - Complementary regularization
- [Regularization](regularization.md) - General regularization techniques
- [Neural Networks Basics](../deep-learning/neural-networks-basics.md) - Foundation
- [Activation Functions](../deep-learning/activation-functions.md) - Used with BatchNorm

---

## Summary

**Key Takeaways:**
1. **BatchNorm normalizes** activations to have mean=0, std=1
2. **Faster training** and allows higher learning rates
3. **Place before activation** (recommended)
4. **Requires batch size ≥ 16** for stability
5. **Switch to eval mode** during inference
6. **Reduces need for dropout** (has regularization effect)
7. **Use Layer Norm for RNNs** and small batches

Batch Normalization is one of the most important innovations in deep learning - use it in almost every deep network!

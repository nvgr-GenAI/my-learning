# Learning Rate Scheduling

## Overview

Learning rate scheduling adjusts the learning rate during training to improve convergence and final performance. Starting with a high learning rate and gradually decreasing it often leads to better results than using a fixed learning rate.

**What You'll Learn:**
- Different scheduling strategies
- When to use each strategy
- Implementation in Keras and PyTorch
- Learning rate warmup
- Visualization and monitoring
- Best practices

**Prerequisites:** Understanding of gradient descent and learning rate

---

## Why Schedule Learning Rate?

### The Problem with Fixed Learning Rate

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate loss landscape
def loss_function(w):
    return w**2 + 0.1 * np.sin(10*w)

# Training with different fixed learning rates
weights_history = {}

for lr_name, lr in [('Too High', 0.5), ('Good', 0.1), ('Too Low', 0.01)]:
    w = 5.0
    history = [w]

    for _ in range(50):
        grad = 2*w + np.cos(10*w)  # Gradient
        w = w - lr * grad
        history.append(w)

    weights_history[lr_name] = history

# Plot
w_range = np.linspace(-6, 6, 200)
loss_range = loss_function(w_range)

plt.figure(figsize=(12, 5))

for lr_name, history in weights_history.items():
    plt.plot(history, [loss_function(w) for w in history],
             'o-', label=lr_name, alpha=0.7, markersize=3)

plt.plot(w_range, loss_range, 'k-', alpha=0.3, linewidth=2)
plt.xlabel('Weight Value')
plt.ylabel('Loss')
plt.title('Fixed Learning Rates - One Size Does NOT Fit All', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Solution:** Start high (fast progress), then decrease (fine-tuning).

---

## 1. Step Decay

### Concept

Reduce learning rate by a factor every N epochs.

```
LR = LR_0 * drop_rate^(floor(epoch / step_size))
```

### Implementation

```python
import tensorflow as tf
from tensorflow import keras

# Keras implementation
initial_lr = 0.1
step_decay_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=10,  # Decay every 10 epochs
    decay_rate=0.5,  # Multiply by 0.5
    staircase=True   # Step function (not smooth)
)

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1)
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=step_decay_schedule),
              loss='mse')

# PyTorch implementation
import torch
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,  # Decay every 10 epochs
    gamma=0.5      # Multiply by 0.5
)

# In training loop:
for epoch in range(100):
    # train_one_epoch()
    scheduler.step()  # Update learning rate
```

### Visualization

```python
# Visualize step decay
epochs = np.arange(0, 100)
initial_lr = 0.1
step_size = 10
gamma = 0.5

lrs = initial_lr * (gamma ** (epochs // step_size))

plt.figure(figsize=(10, 5))
plt.plot(epochs, lrs, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Step Decay Schedule', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.show()
```

**When to Use:**
- Simple baseline
- Know when to decay (based on domain knowledge)
- Want predictable schedule

---

## 2. Exponential Decay

### Concept

Smooth exponential decrease.

```
LR = LR_0 * decay_rate^epoch
```

### Implementation

```python
# Keras
exp_decay_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.1,
    decay_steps=1,       # Decay every epoch
    decay_rate=0.96,     # Multiply by 0.96 each epoch
    staircase=False      # Smooth decay
)

# PyTorch
scheduler = optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=0.96  # Multiply by 0.96 each epoch
)
```

### Comparison with Step Decay

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

epochs = np.arange(0, 50)

# Step Decay
step_lrs = 0.1 * (0.5 ** (epochs // 10))
axes[0].plot(epochs, step_lrs, linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Learning Rate')
axes[0].set_title('Step Decay (Discrete)', fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Exponential Decay
exp_lrs = 0.1 * (0.96 ** epochs)
axes[1].plot(epochs, exp_lrs, linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Learning Rate')
axes[1].set_title('Exponential Decay (Smooth)', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 3. Cosine Annealing

### Concept

Learning rate follows a cosine curve - smooth decrease with potential restarts.

```
LR = LR_min + 0.5 * (LR_max - LR_min) * (1 + cos(π * epoch / T_max))
```

### Implementation

```python
# Keras
cosine_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.1,
    decay_steps=100,  # Total epochs
    alpha=0.0         # Minimum learning rate factor
)

# PyTorch
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,     # Period of annealing
    eta_min=0.0    # Minimum learning rate
)
```

### Visualization

```python
def cosine_annealing(epoch, lr_max=0.1, lr_min=0.0, T_max=50):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / T_max))

epochs = np.arange(0, 100)
lrs = [cosine_annealing(e, T_max=50) for e in epochs]

plt.figure(figsize=(10, 5))
plt.plot(epochs, lrs, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Cosine Annealing Schedule', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.show()
```

**When to Use:**
- Training deep networks
- Want smooth decay
- Using with restarts (SGDR)

---

## 4. Reduce on Plateau

### Concept

Reduce learning rate when validation loss **stops improving**.

**Most adaptive** - responds to actual training progress!

### Implementation

```python
# Keras
reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,           # Multiply LR by 0.5
    patience=5,           # Wait 5 epochs without improvement
    min_lr=1e-7,          # Don't go below this
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    callbacks=[reduce_lr_callback]
)

# PyTorch
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',           # Minimize metric
    factor=0.5,           # Multiply LR by 0.5
    patience=5,           # Wait 5 epochs
    min_lr=1e-7,
    verbose=True
)

# In training loop:
for epoch in range(100):
    # train_loss = train_one_epoch()
    # val_loss = validate()
    scheduler.step(val_loss)  # Pass validation loss
```

**When to Use:**
- Don't know when to decrease LR
- Want adaptive schedule
- Have validation set
- **Recommended default**

---

## 5. Cyclic Learning Rate (CLR)

### Concept

Cyclically vary learning rate between bounds - helps escape local minima.

```python
def triangular_clr(epoch, base_lr=0.001, max_lr=0.006, step_size=20):
    """Triangular cyclic learning rate."""
    cycle = np.floor(1 + epoch / (2 * step_size))
    x = np.abs(epoch / step_size - 2 * cycle + 1)
    return base_lr + (max_lr - base_lr) * max(0, 1 - x)

# Visualization
epochs = np.arange(0, 200)
lrs = [triangular_clr(e) for e in epochs]

plt.figure(figsize=(12, 5))
plt.plot(epochs, lrs, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Cyclic Learning Rate (Triangular)', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.show()
```

### PyTorch Implementation

```python
# PyTorch
scheduler = optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=0.001,
    max_lr=0.006,
    step_size_up=20,      # Epochs to increase LR
    mode='triangular'     # triangular, triangular2, exp_range
)
```

**When to Use:**
- Training CNNs
- Want to escape local minima
- Have compute budget for longer training

---

## 6. One Cycle Policy

### Concept

**Single cycle** of increasing then decreasing LR. Used by fast.ai, very effective!

```
Phase 1 (0-40% epochs): LR increases from base to max
Phase 2 (40-90% epochs): LR decreases from max to base
Phase 3 (90-100% epochs): LR decreases from base to very low
```

### Implementation

```python
# PyTorch
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,
    epochs=100,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,        # 30% warmup
    anneal_strategy='cos'  # Cosine annealing
)

# In training loop:
for epoch in range(100):
    for batch in train_loader:
        # train_on_batch()
        scheduler.step()  # Update every batch!
```

### Visualization

```python
def one_cycle_lr(step, total_steps, max_lr=0.1, pct_start=0.3, div_factor=25):
    """Simulate one cycle LR."""
    warmup_steps = int(total_steps * pct_start)

    if step < warmup_steps:
        # Warmup phase
        return (max_lr / div_factor) + (max_lr - max_lr/div_factor) * step / warmup_steps
    else:
        # Annealing phase
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return max_lr * (1 + np.cos(np.pi * progress)) / 2

total_steps = 1000
steps = np.arange(0, total_steps)
lrs = [one_cycle_lr(s, total_steps) for s in steps]

plt.figure(figsize=(12, 5))
plt.plot(steps, lrs, linewidth=2)
plt.xlabel('Training Step')
plt.ylabel('Learning Rate')
plt.title('One Cycle Learning Rate Policy', fontweight='bold')
plt.axvline(300, color='r', linestyle='--', alpha=0.5, label='End of Warmup')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**When to Use:**
- State-of-the-art technique
- Training from scratch
- Want fast convergence
- **Highly recommended**

---

## Learning Rate Warmup

### Concept

**Start with very low LR**, gradually increase to target LR. Prevents early instability.

### Implementation

```python
class WarmupSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, target_lr):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)

        # Linear warmup
        warmup_lr = self.initial_lr + (self.target_lr - self.initial_lr) * step / warmup_steps

        # Use warmup LR if still in warmup phase
        return tf.cond(step < warmup_steps,
                      lambda: warmup_lr,
                      lambda: self.target_lr)

# Use it
warmup_schedule = WarmupSchedule(
    initial_lr=1e-6,
    warmup_steps=1000,
    target_lr=1e-3
)

# PyTorch: Use with LambdaLR
def warmup_lambda(step, warmup_steps=1000):
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
```

**When to Use:**
- Training large models (Transformers)
- Using Adam optimizer
- Large batch sizes
- **Always for Transformers**

---

## Comparison of Strategies

```python
# Compare all schedules
epochs = np.arange(0, 100)

schedules = {
    'Step Decay': 0.1 * (0.5 ** (epochs // 20)),
    'Exponential': 0.1 * (0.95 ** epochs),
    'Cosine': [cosine_annealing(e, T_max=100) for e in epochs],
    'Reduce on Plateau': 0.1 * np.exp(-epochs / 30),  # Simulated
}

plt.figure(figsize=(12, 6))

for name, lrs in schedules.items():
    plt.plot(epochs, lrs, linewidth=2, label=name)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.title('Learning Rate Schedules Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.show()
```

| Strategy | Complexity | Adaptiveness | Use Case |
|----------|------------|--------------|----------|
| **Step Decay** | Low | None | Simple baseline |
| **Exponential** | Low | None | Smooth decay needed |
| **Cosine** | Medium | None | Deep learning, smooth |
| **Reduce on Plateau** | Low | **High** | **Recommended default** |
| **Cyclic LR** | Medium | Medium | CNNs, escaping minima |
| **One Cycle** | Medium | None | **State-of-the-art** |

---

## Monitoring Learning Rate

### TensorBoard Integration

```python
# Keras: Log LR to TensorBoard
tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./logs')

class LRLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(self.model.optimizer.iterations)
        logs['learning_rate'] = lr

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    callbacks=[tensorboard_callback, LRLogger()]
)
```

### Plot LR History

```python
# Keras: Plot learning rate from history
# history = model.fit(..., callbacks=[LRLogger()])

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# plt.plot(history.history['learning_rate'])
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.tight_layout()
plt.show()
```

---

## Best Practices

### 1. Start with Reduce on Plateau

```python
# Simple and effective default
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)
```

### 2. Combine Warmup + Decay

```python
# For large models (Transformers)
# 1. Warmup for first 10% of training
# 2. Then apply cosine decay

def warmup_cosine_decay(step, warmup_steps, total_steps, max_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return max_lr * 0.5 * (1 + np.cos(np.pi * progress))
```

### 3. Use One Cycle for Maximum Performance

```python
# State-of-the-art for CNNs
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,
    epochs=100,
    steps_per_epoch=len(train_loader)
)
```

### 4. Always Monitor

```python
# Log learning rate every epoch
import matplotlib.pyplot as plt

def train_with_monitoring(model, optimizer, scheduler, epochs):
    lrs = []
    losses = []

    for epoch in range(epochs):
        # Training
        loss = train_one_epoch(model, optimizer)
        losses.append(loss)

        # Get current LR
        lrs.append(optimizer.param_groups[0]['lr'])

        # Update scheduler
        scheduler.step()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(losses)
    axes[0].set_title('Loss')
    axes[1].plot(lrs)
    axes[1].set_title('Learning Rate')
    plt.show()
```

---

## Common Mistakes

### 1. Not Using Any Schedule

**Problem:** Fixed LR leaves performance on table.

**Solution:** At minimum, use Reduce on Plateau.

### 2. Decreasing LR Too Early

**Problem:** Model hasn't converged yet.

**Solution:** Use larger patience or start with higher LR.

### 3. Decreasing LR Too Much

**Problem:** LR becomes too small to make progress.

**Solution:** Set sensible `min_lr`.

### 4. Wrong Scheduler for Problem

**Problem:** Using Step Decay without knowing when to decay.

**Solution:** Use Reduce on Plateau (adaptive).

---

## Schedule Selection Guide

### Decision Tree

```
Start Here
│
├─ Training Transformer?
│  └─ Yes → Warmup + Cosine Decay
│
├─ Have validation set?
│  └─ Yes → Reduce on Plateau
│
├─ Want state-of-the-art?
│  └─ Yes → One Cycle Policy
│
├─ Training CNN?
│  └─ Yes → Cosine Annealing or One Cycle
│
└─ Don't know?
   └─ Use Reduce on Plateau (safe default)
```

---

## Related Topics

- [Optimizers](optimizers.md) - Optimization algorithms
- [Gradient Descent](gradient-descent.md) - Foundation
- [Hyperparameter Tuning](hyperparameter-tuning.md) - Finding optimal LR
- [Neural Networks Basics](../deep-learning/neural-networks-basics.md) - Context

---

## Summary

**Key Takeaways:**
1. **Learning rate scheduling improves performance**
2. **Reduce on Plateau**: Safe default choice
3. **One Cycle Policy**: State-of-the-art for many tasks
4. **Warmup**: Essential for Transformers and large models
5. **Always monitor** learning rate during training
6. **Start high, end low** for best results
7. **Combine with good optimizer** (Adam, AdamW)

Remember: A good learning rate schedule can improve final accuracy by 1-5% - always use one!

# Chapter 5: Learning Through Optimization

How does a machine actually "learn"? We've seen that models are functions with parameters, but how do we find the right parameter values? The answer is optimization—a mathematical process of finding the best solution to a problem.

In machine learning, learning means adjusting parameters to minimize errors. This chapter explores the heart of machine learning: how algorithms iteratively improve by following gradients downhill toward better solutions, the different strategies for doing this efficiently, and the challenges that arise along the way.

---

## 5.1 What Does "Learning" Mean?

### Learning as Parameter Adjustment

**Learning** in machine learning is the process of finding parameter values that make your model's predictions as accurate as possible. It's an iterative process: start with random parameters, measure how wrong they are, adjust them to be less wrong, and repeat.

```python
import numpy as np

# Conceptual: What learning looks like
def simple_model(x, w, b):
    """Model: y = w*x + b"""
    return w * x + b

# True relationship: y = 3x + 2
X_true = np.array([1, 2, 3, 4, 5])
y_true = 3 * X_true + 2

# Start with random parameters
w = 0.5  # Random initial weight
b = 1.0  # Random initial bias

print("Learning process:")
print("True function: y = 3x + 2")
print(f"Starting parameters: w={w}, b={b}")

# Make predictions with initial (bad) parameters
y_pred = simple_model(X_true, w, b)
error = np.mean((y_true - y_pred)**2)
print(f"Initial error: {error:.2f}")

# After learning (we'll see HOW later)
w_learned = 3.0
b_learned = 2.0
y_pred_learned = simple_model(X_true, w_learned, b_learned)
error_learned = np.mean((y_true - y_pred_learned)**2)

print(f"\nAfter learning: w={w_learned}, b={b_learned}")
print(f"Final error: {error_learned:.2f}")
print("\nLearning = finding parameters that minimize error")
```

### Key Training Concepts

**Training** is the complete process of learning from data. During training, the model sees examples and adjusts its parameters.

**Epoch**: One complete pass through the entire training dataset.

```python
# Conceptual: Epochs
training_data_size = 1000
print(f"Training data: {training_data_size} examples")
print(f"\n1 epoch = seeing all {training_data_size} examples once")
print(f"5 epochs = seeing all {training_data_size} examples five times")
print("\nMore epochs = more opportunities to learn")
print("But too many epochs can lead to overfitting")
```

**Batch**: A subset of training data processed together before updating parameters.

**Mini-batch**: A small batch (common: 32, 64, 128, 256 examples).

**Iteration** (or step): One parameter update. If you have 1000 examples and batch size 100, one epoch has 10 iterations.

```python
# Training terminology
total_examples = 1000
batch_size = 100
epochs = 5

iterations_per_epoch = total_examples // batch_size
total_iterations = epochs * iterations_per_epoch

print(f"Dataset size: {total_examples} examples")
print(f"Batch size: {batch_size} examples per batch")
print(f"Epochs: {epochs}")
print(f"\nIterations per epoch: {iterations_per_epoch}")
print(f"Total iterations: {total_iterations}")

print("\nEach iteration:")
print("  1. Take a batch of data")
print("  2. Make predictions")
print("  3. Calculate error")
print("  4. Update parameters")
```

**Full Example: Training Loop**

```python
from sklearn.linear_model import SGDRegressor
import numpy as np

# Generate data
np.random.seed(42)
X = np.random.rand(1000, 3)
y = X.sum(axis=1) + np.random.randn(1000) * 0.1

# Create model with explicit iteration control
model = SGDRegressor(max_iter=1, warm_start=True, random_state=42)

print("Training progress:")
print("Epoch | Error")
print("-" * 20)

# Train for multiple epochs
for epoch in range(10):
    model.fit(X, y)
    predictions = model.predict(X)
    error = np.mean((y - predictions)**2)
    print(f"{epoch+1:5d} | {error:.4f}")

print("\nError decreases as model learns")
```

---

## 5.2 Loss Functions

A loss function measures how wrong your model's predictions are. It quantifies the gap between predictions and true values. The goal of learning is to minimize this loss.

### Loss vs Cost vs Objective

These terms are often used interchangeably, but have subtle distinctions:

**Loss Function**: Error for a single example.

**Cost Function**: Average loss across all examples (or a batch).

**Objective Function**: The function we're trying to minimize (or maximize).

```python
import numpy as np

# Single example
y_true_single = 5.0
y_pred_single = 4.0
loss = (y_true_single - y_pred_single)**2
print(f"Loss (single example): {loss}")

# Multiple examples
y_true = np.array([5, 3, 7, 2])
y_pred = np.array([4, 3.5, 6, 2.5])
losses = (y_true - y_pred)**2
cost = np.mean(losses)

print(f"\nIndividual losses: {losses}")
print(f"Cost (average loss): {cost:.2f}")

print("\nLoss = per-example error")
print("Cost = average over all examples")
print("Objective = what we minimize (usually the cost)")
```

### Mean Squared Error (MSE)

**MSE** measures the average squared difference between predictions and true values. Common for regression problems.

Formula: `MSE = (1/n) Σ(y_true - y_pred)²`

```python
import numpy as np

def mean_squared_error(y_true, y_pred):
    """Calculate MSE manually"""
    return np.mean((y_true - y_pred)**2)

# Example
y_true = np.array([3, 5, 7, 9])
y_pred = np.array([2.8, 5.2, 6.9, 9.1])

mse = mean_squared_error(y_true, y_pred)
print(f"Predictions: {y_pred}")
print(f"True values: {y_true}")
print(f"MSE: {mse:.4f}")

# Using sklearn
from sklearn.metrics import mean_squared_error as sklearn_mse
print(f"Sklearn MSE: {sklearn_mse(y_true, y_pred):.4f}")

print("\nMSE characteristics:")
print("  - Always positive")
print("  - Larger errors are penalized more (squared)")
print("  - Units are squared (if y is in dollars, MSE is in dollars²)")
```

**Why squaring?**

```python
# Without squaring: errors cancel out
y_true = np.array([5, 5, 5, 5])
y_pred = np.array([4, 6, 4, 6])  # Always off by 1

errors = y_true - y_pred
mean_error = np.mean(errors)
print(f"Errors: {errors}")
print(f"Mean error (no squaring): {mean_error}")
print("  Positive and negative errors cancel!")

# With squaring: all errors count
squared_errors = (y_true - y_pred)**2
mse = np.mean(squared_errors)
print(f"\nSquared errors: {squared_errors}")
print(f"MSE: {mse}")
print("  All errors contribute positively")
```

### Mean Absolute Error (MAE)

**MAE** measures the average absolute difference. Less sensitive to outliers than MSE.

Formula: `MAE = (1/n) Σ|y_true - y_pred|`

```python
def mean_absolute_error(y_true, y_pred):
    """Calculate MAE manually"""
    return np.mean(np.abs(y_true - y_pred))

y_true = np.array([3, 5, 7, 9])
y_pred = np.array([2.8, 5.2, 6.9, 9.1])

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")

# MAE vs MSE with outlier
y_true_outlier = np.array([3, 5, 7, 100])  # Outlier!
y_pred_outlier = np.array([2.8, 5.2, 6.9, 80])

mae_outlier = mean_absolute_error(y_true_outlier, y_pred_outlier)
mse_outlier = mean_squared_error(y_true_outlier, y_pred_outlier)

print(f"\nWith outlier:")
print(f"MAE: {mae_outlier:.4f}")
print(f"MSE: {mse_outlier:.4f}")
print("\nMSE heavily penalizes large errors (outliers)")
print("MAE treats all errors proportionally")
```

### Cross-Entropy Loss

**Cross-Entropy** is used for classification. It measures how different the predicted probability distribution is from the true distribution.

**Binary Cross-Entropy** (for binary classification):

```python
import numpy as np

def binary_cross_entropy(y_true, y_pred_prob):
    """
    y_true: actual labels (0 or 1)
    y_pred_prob: predicted probabilities
    """
    epsilon = 1e-15  # Avoid log(0)
    y_pred_prob = np.clip(y_pred_prob, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred_prob) +
                    (1 - y_true) * np.log(1 - y_pred_prob))

# Examples
y_true = np.array([1, 0, 1, 1, 0])  # True labels

# Good predictions (confident and correct)
y_pred_good = np.array([0.9, 0.1, 0.85, 0.95, 0.05])
loss_good = binary_cross_entropy(y_true, y_pred_good)

# Bad predictions (confident but wrong)
y_pred_bad = np.array([0.1, 0.9, 0.2, 0.15, 0.95])
loss_bad = binary_cross_entropy(y_true, y_pred_bad)

print("True labels:", y_true)
print(f"\nGood predictions: {y_pred_good}")
print(f"Loss: {loss_good:.4f}")

print(f"\nBad predictions: {y_pred_bad}")
print(f"Loss: {loss_bad:.4f}")

print("\nCross-entropy penalizes confident wrong predictions heavily")
```

**Categorical Cross-Entropy** (for multi-class):

```python
def categorical_cross_entropy(y_true, y_pred_probs):
    """
    y_true: true class labels (one-hot encoded)
    y_pred_probs: predicted probabilities for each class
    """
    epsilon = 1e-15
    y_pred_probs = np.clip(y_pred_probs, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred_probs), axis=1))

# Example: 3 classes
y_true = np.array([
    [1, 0, 0],  # Class 0
    [0, 1, 0],  # Class 1
    [0, 0, 1]   # Class 2
])

# Predicted probabilities
y_pred = np.array([
    [0.8, 0.1, 0.1],  # Confident about class 0 ✓
    [0.2, 0.6, 0.2],  # Confident about class 1 ✓
    [0.1, 0.2, 0.7]   # Confident about class 2 ✓
])

loss = categorical_cross_entropy(y_true, y_pred)
print("Categorical Cross-Entropy Loss:", loss)
```

---

## 5.3 Gradient Descent

Gradient descent is the workhorse algorithm for training machine learning models. It's how we actually minimize the loss function and find better parameters.

### The Gradient

The **gradient** is a vector of partial derivatives. It points in the direction of steepest increase of a function. For loss minimization, we move in the opposite direction (downhill).

```python
import numpy as np

# Simple example: f(x) = x²
# Gradient (derivative): f'(x) = 2x

def function(x):
    return x**2

def gradient(x):
    return 2 * x

# Start at x = 5
x = 5.0
print(f"Starting at x = {x}")
print(f"Function value: f({x}) = {function(x)}")
print(f"Gradient: {gradient(x)}")
print("  Gradient is positive → function increases to the right")
print("  To minimize, move LEFT (opposite of gradient)")

# Move opposite to gradient
learning_rate = 0.1
x_new = x - learning_rate * gradient(x)
print(f"\nAfter one step: x = {x_new}")
print(f"Function value: f({x_new}) = {function(x_new)}")
print("Value decreased! We're moving toward the minimum")
```

### Gradient Descent Algorithm

**Gradient Descent** iteratively updates parameters by moving in the direction opposite to the gradient.

Update rule: `θ_new = θ_old - α * ∇L(θ)`

Where:
- `θ` = parameters
- `α` = learning rate
- `∇L(θ)` = gradient of loss with respect to parameters

```python
import numpy as np
import matplotlib.pyplot as plt

# Minimize f(x) = (x - 3)²
# Minimum at x = 3

def loss(x):
    return (x - 3)**2

def gradient(x):
    return 2 * (x - 3)

# Gradient descent
x = 0.0  # Start far from minimum
learning_rate = 0.1
history = [x]

print("Gradient Descent:")
print("Step | x | Loss | Gradient")
print("-" * 40)

for step in range(10):
    grad = gradient(x)
    x = x - learning_rate * grad
    history.append(x)
    print(f"{step+1:4d} | {x:5.2f} | {loss(x):5.2f} | {grad:8.2f}")

print(f"\nConverged close to optimal x = 3.0")
```

### Learning Rate

The **learning rate** controls how big each step is. Too large and you overshoot; too small and learning is slow.

```python
def gradient_descent_demo(learning_rate, steps=20):
    x = 0.0
    path = [x]

    for _ in range(steps):
        grad = 2 * (x - 3)
        x = x - learning_rate * grad
        path.append(x)

    return path

# Try different learning rates
lr_small = 0.05
lr_good = 0.3
lr_large = 1.1

path_small = gradient_descent_demo(lr_small)
path_good = gradient_descent_demo(lr_good)
path_large = gradient_descent_demo(lr_large, steps=10)

print("Learning Rate Comparison:")
print(f"\nSmall LR ({lr_small}): reaches {path_small[-1]:.4f} after 20 steps")
print(f"Good LR ({lr_good}): reaches {path_good[-1]:.4f} after 20 steps")
print(f"Large LR ({lr_large}): reaches {path_large[-1]:.4f} after 10 steps")

print("\nToo small: slow convergence")
print("Too large: might overshoot or diverge")
print("Just right: fast and stable convergence")
```

### Variants of Gradient Descent

**Batch Gradient Descent**: Uses all training data to compute gradient.

```python
# Conceptual
def batch_gradient_descent(X, y, learning_rate, epochs):
    """Use ALL data for each update"""
    n_samples = len(X)
    w = 0.0  # Initialize parameter

    for epoch in range(epochs):
        # Compute gradient using ALL samples
        predictions = w * X
        gradient = (-2/n_samples) * np.sum(X * (y - predictions))

        # Update parameter
        w = w - learning_rate * gradient

    return w

print("Batch GD: Uses entire dataset per update")
print("  Pros: Stable, smooth convergence")
print("  Cons: Slow for large datasets, needs all data in memory")
```

**Stochastic Gradient Descent (SGD)**: Uses one example at a time.

```python
def stochastic_gradient_descent(X, y, learning_rate, epochs):
    """Use ONE example for each update"""
    n_samples = len(X)
    w = 0.0

    for epoch in range(epochs):
        for i in range(n_samples):
            # Compute gradient using ONE sample
            prediction = w * X[i]
            gradient = -2 * X[i] * (y[i] - prediction)

            # Update parameter
            w = w - learning_rate * gradient

    return w

print("\nSGD: Uses one example per update")
print("  Pros: Fast, can escape local minima")
print("  Cons: Noisy updates, less stable")
```

**Mini-batch Gradient Descent**: Uses a small batch of examples.

```python
def minibatch_gradient_descent(X, y, learning_rate, epochs, batch_size=32):
    """Use a BATCH of examples for each update"""
    n_samples = len(X)
    w = 0.0

    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)

        # Process in batches
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            # Compute gradient using BATCH
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            predictions = w * X_batch
            gradient = (-2/len(X_batch)) * np.sum(X_batch * (y_batch - predictions))

            # Update parameter
            w = w - learning_rate * gradient

    return w

print("\nMini-batch GD: Uses batches of examples")
print("  Pros: Balance of speed and stability")
print("  Cons: Adds batch_size hyperparameter")
print("\nMost common in practice: Mini-batch with size 32-256")
```

---

## 5.4 Advanced Optimizers

Plain gradient descent works, but advanced optimizers can train faster and reach better solutions. They add mechanisms like momentum and adaptive learning rates.

### Momentum

**Momentum** adds a "velocity" term that accumulates gradients over time. This helps accelerate in consistent directions and dampen oscillations.

```python
import numpy as np

def gradient_descent_with_momentum(x_start, learning_rate, momentum, steps):
    """GD with momentum"""
    x = x_start
    velocity = 0.0
    history = [x]

    for _ in range(steps):
        # Gradient of f(x) = (x - 3)²
        grad = 2 * (x - 3)

        # Update velocity
        velocity = momentum * velocity - learning_rate * grad

        # Update position
        x = x + velocity
        history.append(x)

    return history

# Compare with and without momentum
history_no_momentum = gradient_descent_demo(0.1, steps=20)
history_momentum = gradient_descent_with_momentum(0.0, 0.1, 0.9, steps=20)

print("Convergence comparison:")
print(f"Without momentum (step 20): x = {history_no_momentum[-1]:.4f}")
print(f"With momentum (step 20): x = {history_momentum[-1]:.4f}")

print("\nMomentum benefits:")
print("  - Faster convergence")
print("  - Can push through small local minima")
print("  - Smooths out noisy gradients")
```

### Adam (Adaptive Moment Estimation)

**Adam** is the most popular optimizer. It combines momentum with adaptive learning rates for each parameter.

```python
from sklearn.neural_network import MLPRegressor
import numpy as np

# Generate data
np.random.seed(42)
X = np.random.rand(200, 5)
y = X.sum(axis=1) + np.random.randn(200) * 0.1

# Compare optimizers
sgd_model = MLPRegressor(solver='sgd', learning_rate_init=0.01,
                         max_iter=100, random_state=42)

adam_model = MLPRegressor(solver='adam', learning_rate_init=0.001,
                          max_iter=100, random_state=42)

sgd_model.fit(X, y)
adam_model.fit(X, y)

print("Optimizer Comparison:")
print(f"SGD final loss: {sgd_model.loss_:.4f}")
print(f"Adam final loss: {adam_model.loss_:.4f}")

print("\nAdam advantages:")
print("  - Adaptive learning rates per parameter")
print("  - Combines momentum and RMSprop")
print("  - Works well with default hyperparameters")
print("  - Industry standard for deep learning")
```

### RMSprop and Adagrad

**Adagrad**: Adapts learning rate based on past gradients (rarely used now).

**RMSprop**: Like Adagrad but with exponential moving average (better for non-convex problems).

```python
print("Optimizer Family:")
print("\nAdagrad:")
print("  - Adapts learning rate per parameter")
print("  - Good: Parameters with small gradients get larger updates")
print("  - Bad: Learning rate always decreases, can stop learning")

print("\nRMSprop:")
print("  - Like Adagrad but uses moving average")
print("  - Good: Learning rate doesn't monotonically decrease")
print("  - Use case: RNNs and recurrent architectures")

print("\nAdam:")
print("  - Combines RMSprop + Momentum")
print("  - Good: Works well in most situations")
print("  - Default choice for most deep learning")
```

### Learning Rate Scheduling

**Learning Rate Scheduling** changes the learning rate during training.

```python
from sklearn.neural_network import MLPRegressor

# Step decay: reduce learning rate every N epochs
print("Learning Rate Schedules:")
print("\n1. Step Decay:")
print("   Start: lr = 0.1")
print("   Epoch 10: lr = 0.01")
print("   Epoch 20: lr = 0.001")

print("\n2. Exponential Decay:")
print("   lr = initial_lr * exp(-decay_rate * epoch)")

print("\n3. Cosine Annealing:")
print("   lr follows cosine curve")

print("\nWhy schedule learning rate?")
print("  - Start with large LR: make fast initial progress")
print("  - Reduce LR later: fine-tune solution")
```

### Weight Decay (L2 Regularization)

**Weight Decay** adds a penalty for large weights to the loss function.

```python
from sklearn.linear_model import Ridge
import numpy as np

# Generate data
X = np.random.rand(100, 5)
y = np.random.rand(100)

# Without weight decay
model_no_decay = Ridge(alpha=0.0)
model_no_decay.fit(X, y)

# With weight decay
model_with_decay = Ridge(alpha=1.0)
model_with_decay.fit(X, y)

print("Weight Decay (L2 Regularization):")
print(f"\nWithout decay: weights = {model_no_decay.coef_}")
print(f"Max weight: {np.abs(model_no_decay.coef_).max():.4f}")

print(f"\nWith decay: weights = {model_with_decay.coef_}")
print(f"Max weight: {np.abs(model_with_decay.coef_).max():.4f}")

print("\nWeight decay encourages smaller weights")
print("  - Prevents overfitting")
print("  - Acts as regularization")
```

---

## 5.5 Convergence

**Convergence** is when training reaches a point where further updates don't significantly improve the model.

### Detecting Convergence

```python
import numpy as np

# Simulate training
losses = []
for epoch in range(100):
    # Simulated loss that converges
    loss = 10 * np.exp(-epoch/20) + np.random.randn() * 0.1
    losses.append(loss)

print("Convergence indicators:")
print(f"\nLoss at epoch 10: {losses[10]:.4f}")
print(f"Loss at epoch 50: {losses[50]:.4f}")
print(f"Loss at epoch 90: {losses[90]:.4f}")

# Check convergence
recent_losses = losses[-10:]
if max(recent_losses) - min(recent_losses) < 0.01:
    print("\nConverged: Loss is stable")
else:
    print("\nNot converged: Loss still changing")

print("\nSigns of convergence:")
print("  - Loss stops decreasing")
print("  - Validation error stops improving")
print("  - Gradients become very small")
```

### Local vs Global Minima

**Local Minimum**: A point lower than nearby points, but not the absolute lowest.

**Global Minimum**: The absolute lowest point of the loss function.

```python
# Example: Non-convex function with multiple minima
def non_convex_function(x):
    """Function with multiple local minima"""
    return x**4 - 4*x**3 + 4*x**2 + 1

# Different starting points lead to different minima
start_points = [-1.0, 0.5, 3.0]

print("Local vs Global Minima:")
print("\nStarting from different points:")

for start in start_points:
    x = start
    for _ in range(50):
        grad = 4*x**3 - 12*x**2 + 8*x
        x = x - 0.01 * grad

    print(f"Start: {start:5.1f} → Converged to: {x:5.2f} (loss: {non_convex_function(x):.2f})")

print("\nDifferent starting points can reach different local minima")
print("Global minimum might not be reached!")

print("\nIn practice:")
print("  - Deep learning: many local minima are good enough")
print("  - Convex problems: only one minimum (guaranteed global)")
print("  - Non-convex: use tricks like momentum, restarts")
```

---

## 5.6 Backpropagation (for neural networks)

Backpropagation is the algorithm that efficiently computes gradients in neural networks by applying the chain rule from calculus.

### Forward and Backward Pass

**Forward Pass**: Input flows through the network to produce output.

**Backward Pass**: Errors flow backward through the network to compute gradients.

```python
import numpy as np

# Simple 2-layer network
class SimpleNetwork:
    def __init__(self):
        # Initialize weights
        self.w1 = np.random.randn(2, 3) * 0.1  # Input to hidden
        self.w2 = np.random.randn(3, 1) * 0.1  # Hidden to output

    def forward(self, x):
        """Forward pass"""
        self.x = x
        self.hidden = np.maximum(0, np.dot(x, self.w1))  # ReLU activation
        self.output = np.dot(self.hidden, self.w2)
        return self.output

    def backward(self, y_true, learning_rate=0.01):
        """Backward pass (backpropagation)"""
        # Gradient of loss with respect to output
        d_output = 2 * (self.output - y_true)

        # Gradient with respect to w2
        d_w2 = np.dot(self.hidden.T, d_output)

        # Backpropagate through hidden layer
        d_hidden = np.dot(d_output, self.w2.T)
        d_hidden[self.hidden <= 0] = 0  # ReLU gradient

        # Gradient with respect to w1
        d_w1 = np.dot(self.x.T, d_hidden)

        # Update weights
        self.w2 -= learning_rate * d_w2
        self.w1 -= learning_rate * d_w1

# Example usage
net = SimpleNetwork()
X = np.array([[1, 2]])
y = np.array([[5]])

print("Backpropagation Demo:")
print("Initial prediction:", net.forward(X)[0, 0])

# Train
for epoch in range(100):
    output = net.forward(X)
    net.backward(y)

print("After training:", net.forward(X)[0, 0])
print("Target:", y[0, 0])

print("\nBackpropagation:")
print("  1. Forward pass: compute output")
print("  2. Compute loss")
print("  3. Backward pass: compute gradients using chain rule")
print("  4. Update weights")
```

### Vanishing and Exploding Gradients

**Vanishing Gradient**: Gradients become extremely small, making learning impossible.

**Exploding Gradient**: Gradients become extremely large, causing unstable updates.

```python
# Vanishing gradient example
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Deep network with sigmoid
print("Vanishing Gradient Problem:")
print("\nSigmoid derivative at different values:")
for x in [0, 2, 5, 10]:
    deriv = sigmoid_derivative(x)
    print(f"  x={x:2d}: derivative = {deriv:.6f}")

print("\nIn deep networks:")
print("  Layer 1 gradient: 0.25")
print("  Layer 2 gradient: 0.25 * 0.25 = 0.0625")
print("  Layer 10 gradient: 0.25^10 ≈ 0.000001")
print("  Early layers barely learn!")

print("\nSolutions:")
print("  - Use ReLU instead of sigmoid")
print("  - Batch normalization")
print("  - Residual connections (ResNet)")
print("  - Gradient clipping (for exploding gradients)")
```

**Example: ReLU vs Sigmoid**

```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

print("\nReLU gradient:")
for x in [-2, 0, 2, 100]:
    deriv = relu_derivative(x)
    print(f"  x={x:4d}: derivative = {deriv:.1f}")

print("\nReLU advantages:")
print("  - Gradient is 1 (for x>0) → no vanishing")
print("  - Fast to compute")
print("  - Helps train deep networks")
```

---

## Summary

Learning in machine learning is fundamentally an optimization process:

**Learning as Optimization:** Learning means finding parameter values that minimize error. This happens iteratively through training, where the model processes data in epochs (complete passes), batches (subsets), and iterations (individual updates). Each iteration: take data, predict, calculate loss, update parameters.

**Loss Functions:** Loss functions quantify prediction errors. MSE for regression (penalizes large errors heavily), MAE for regression (robust to outliers), Cross-Entropy for classification (penalizes confident wrong predictions). The choice of loss function shapes what patterns the model learns.

**Gradient Descent:** The core algorithm for learning. Compute the gradient (direction of steepest increase), move opposite to it (downhill). Learning rate controls step size—too large and you overshoot, too small and learning is slow. Three variants: batch (all data), stochastic (one example), mini-batch (small batches)—mini-batch wins in practice.

**Advanced Optimizers:** Beyond basic gradient descent: Momentum accumulates velocity for faster convergence, Adam combines momentum with adaptive learning rates (industry standard), Learning rate schedules start fast then slow down for fine-tuning. Weight decay prevents overfitting by penalizing large weights.

**Convergence and Backpropagation:** Convergence is when loss stops improving—can reach local minima (good enough) or global minimum (best possible). Backpropagation efficiently computes gradients in neural networks via chain rule (forward pass computes outputs, backward pass computes gradients). Watch for vanishing gradients (too small) and exploding gradients (too large)—ReLU activations help prevent vanishing.

With optimization mastered, we turn to a critical challenge: ensuring models perform well on new data, not just training data. That's the generalization problem, our next chapter's focus.

---

[← Previous: Chapter 4](04-models-as-functions.md) | [Back to Index](index.md) | [Next: Chapter 6 →](06-generalization.md)

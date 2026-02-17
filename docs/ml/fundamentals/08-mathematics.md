# Chapter 8: Mathematical Foundation

Machine learning algorithms are fundamentally built on mathematical principles. While modern libraries hide much of the complexity, understanding the underlying mathematics helps you choose appropriate models, debug problems, tune hyperparameters, and ultimately build better systems.

This chapter covers the essential mathematical concepts that underpin machine learning: linear algebra for representing and manipulating data, calculus for optimization, probability for modeling uncertainty, and information theory for measuring information content. We'll focus on intuition and practical application rather than mathematical rigor.

## 8.1 Linear Algebra Essentials

Linear algebra provides the language for working with multi-dimensional data. Nearly every operation in machine learning—from data representation to model training—involves vectors and matrices.

### Vectors: The Building Blocks

A vector is an ordered collection of numbers. In machine learning, vectors typically represent data points or features.

```python
import numpy as np

# A vector representing a house: [size, bedrooms, age]
house = np.array([1500, 3, 10])

print(f"House features: {house}")
print(f"Vector dimension: {house.shape}")  # (3,)
```

Geometrically, a vector can be visualized as an arrow in space. A 2D vector has two components (x, y), a 3D vector has three (x, y, z), and so on.

```python
import matplotlib.pyplot as plt

# 2D vector visualization
v = np.array([3, 2])

plt.figure(figsize=(6, 6))
plt.arrow(0, 0, v[0], v[1], head_width=0.2, head_length=0.2, fc='blue', ec='blue')
plt.xlim(0, 4)
plt.ylim(0, 3)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Vector [3, 2]')
plt.show()
```

#### Vector Operations

**Vector Addition**: Combine two vectors element-wise.

```python
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Element-wise addition
v_sum = v1 + v2
print(f"v1 + v2 = {v_sum}")  # [5, 7, 9]
```

Geometrically, vector addition places the tail of the second vector at the head of the first.

**Scalar Multiplication**: Multiply every element by a number (scalar).

```python
v = np.array([1, 2, 3])
scalar = 2

# Scale vector
v_scaled = scalar * v
print(f"2 * v = {v_scaled}")  # [2, 4, 6]
```

This stretches or shrinks the vector, changing its magnitude but not its direction (unless the scalar is negative).

**Dot Product**: A key operation that produces a single number from two vectors.

```python
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Dot product
dot = np.dot(v1, v2)
print(f"v1 · v2 = {dot}")  # 1*4 + 2*5 + 3*6 = 32
```

The dot product formula: v₁ · v₂ = Σ(v₁ᵢ * v₂ᵢ) = v₁₁*v₂₁ + v₁₂*v₂₂ + ... + v₁ₙ*v₂ₙ

Geometrically, the dot product measures how much two vectors point in the same direction:

- **Large positive value**: Vectors point in similar directions
- **Zero**: Vectors are perpendicular (orthogonal)
- **Large negative value**: Vectors point in opposite directions

```python
# Example: Measuring similarity
user_preferences = np.array([5, 1, 3])  # [action, romance, comedy]
movie_profile = np.array([4, 0, 5])

similarity = np.dot(user_preferences, movie_profile)
print(f"Movie similarity score: {similarity}")  # Higher = more similar
```

**Vector Magnitude (Length)**: The Euclidean length of a vector.

```python
v = np.array([3, 4])

# Magnitude: ||v|| = sqrt(v₁² + v₂² + ... + vₙ²)
magnitude = np.linalg.norm(v)
print(f"||v|| = {magnitude}")  # sqrt(3² + 4²) = 5
```

**Unit Vector (Normalization)**: A vector with magnitude 1, pointing in the same direction.

```python
v = np.array([3, 4])

# Normalize: divide by magnitude
v_normalized = v / np.linalg.norm(v)
print(f"Normalized v: {v_normalized}")  # [0.6, 0.8]
print(f"Magnitude: {np.linalg.norm(v_normalized)}")  # 1.0
```

Normalization is crucial in machine learning for:

- Feature scaling (making features comparable)
- Computing cosine similarity
- Gradient descent stability

### Matrices: Organizing Multiple Vectors

A matrix is a 2D array of numbers. In machine learning, matrices typically represent datasets (rows = examples, columns = features) or transformations (weights).

```python
# Dataset as matrix: 3 houses, 4 features each
# [size, bedrooms, bathrooms, age]
houses = np.array([
    [1500, 3, 2, 10],
    [2000, 4, 3, 5],
    [1200, 2, 1, 15]
])

print(f"Dataset shape: {houses.shape}")  # (3, 4) = 3 rows, 4 columns
print(f"First house: {houses[0]}")       # [1500, 3, 2, 10]
print(f"All sizes: {houses[:, 0]}")      # [1500, 2000, 1200]
```

#### Matrix Operations

**Matrix-Vector Multiplication**: Apply a transformation to a vector.

```python
# Weight matrix W (2x3)
W = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# Input vector x (3,)
x = np.array([1, 0, 1])

# Matrix-vector product: y = W @ x
y = W @ x  # or np.dot(W, x)
print(f"Output: {y}")  # [1*1 + 2*0 + 3*1, 4*1 + 5*0 + 6*1] = [4, 10]
```

This is the fundamental operation in neural networks: each layer performs y = Wx + b.

**Matrix-Matrix Multiplication**: Combine transformations.

```python
A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

# Matrix product C = A @ B
C = A @ B
print(f"A @ B:\n{C}")
# [[1*5 + 2*7, 1*6 + 2*8],    [[19, 22],
#  [3*5 + 4*7, 3*6 + 4*8]]  =  [43, 50]]
```

**Important**: Matrix multiplication is NOT element-wise. The number of columns in A must equal the number of rows in B. The result has shape (rows of A, columns of B).

**Transpose**: Flip rows and columns.

```python
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# Transpose: swap dimensions
A_T = A.T
print(f"Original shape: {A.shape}")    # (2, 3)
print(f"Transposed shape: {A_T.shape}")  # (3, 2)
print(f"A^T:\n{A_T}")
# [[1, 4],
#  [2, 5],
#  [3, 6]]
```

The transpose is essential for:

- Converting row vectors to column vectors
- Computing gradients in backpropagation
- Formulating linear regression solutions

**Identity Matrix**: The matrix equivalent of 1.

```python
# 3x3 identity matrix
I = np.eye(3)
print(f"Identity matrix:\n{I}")
# [[1, 0, 0],
#  [0, 1, 0],
#  [0, 0, 1]]

# Multiplying by identity doesn't change the matrix
A = np.array([[1, 2], [3, 4]])
print(f"A @ I = A:\n{A @ I[:2, :2]}")  # Same as A
```

**Matrix Inverse**: The matrix that "undoes" another matrix.

```python
A = np.array([
    [4, 7],
    [2, 6]
])

# Inverse: A @ A_inv = I
A_inv = np.linalg.inv(A)
print(f"A inverse:\n{A_inv}")

# Verify: A @ A_inv ≈ I
print(f"A @ A_inv:\n{A @ A_inv}")
# [[1, 0],
#  [0, 1]]
```

The inverse is used in closed-form solutions (e.g., linear regression: θ = (XᵀX)⁻¹Xᵀy), though in practice we use iterative methods for large matrices.

### Why Linear Algebra Matters for ML

**1. Data Representation**: Datasets are matrices where rows are examples and columns are features.

```python
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

print(f"Dataset shape: {X.shape}")  # (150, 4) = 150 flowers, 4 features
print(f"Feature names: {iris.feature_names}")
```

**2. Model Parameters**: Model weights are vectors or matrices.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)

print(f"Weights shape: {model.coef_.shape}")  # (4,) = one weight per feature
print(f"Weights: {model.coef_}")
```

**3. Predictions**: Making predictions involves matrix-vector products.

```python
# Linear model: ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
# In matrix form: ŷ = X @ w + b

X_sample = X[:5]  # First 5 examples
predictions = X_sample @ model.coef_ + model.intercept_
print(f"Predictions: {predictions}")
```

**4. Dimensionality Reduction**: Techniques like PCA use eigenvalue decomposition.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print(f"Original shape: {X.shape}")      # (150, 4)
print(f"Reduced shape: {X_reduced.shape}")  # (150, 2)
print(f"Principal components:\n{pca.components_}")  # Eigenvectors
```

**5. Neural Networks**: Each layer is a matrix transformation.

```python
# Simple neural network layer
def layer(X, W, b):
    """
    X: input (batch_size, input_dim)
    W: weights (input_dim, output_dim)
    b: bias (output_dim,)
    """
    return X @ W + b

# Example: 3 inputs → 5 hidden units
X_batch = np.random.randn(10, 3)  # 10 examples, 3 features
W = np.random.randn(3, 5)         # 3 inputs, 5 outputs
b = np.random.randn(5)

output = layer(X_batch, W, b)
print(f"Output shape: {output.shape}")  # (10, 5)
```

Understanding linear algebra allows you to:

- Understand model architectures (especially deep learning)
- Debug dimension mismatches
- Implement custom models
- Interpret model behavior

## 8.2 Calculus for Optimization

Calculus is the mathematics of change. In machine learning, we use calculus to find model parameters that minimize loss—the process of optimization.

### Derivatives: Measuring Rate of Change

The derivative measures how a function changes as its input changes. It's the slope of the function at a point.

```python
import numpy as np
import matplotlib.pyplot as plt

# Function: f(x) = x²
def f(x):
    return x**2

# Derivative: f'(x) = 2x
def f_prime(x):
    return 2 * x

# Visualize
x = np.linspace(-3, 3, 100)
y = f(x)

plt.figure(figsize=(10, 4))

# Plot function
plt.subplot(1, 2, 1)
plt.plot(x, y, label='f(x) = x²')
plt.scatter([1], [f(1)], color='red', s=100, zorder=5)
plt.title('Function f(x) = x²')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()

# Plot derivative
plt.subplot(1, 2, 2)
plt.plot(x, f_prime(x), label="f'(x) = 2x", color='orange')
plt.scatter([1], [f_prime(1)], color='red', s=100, zorder=5)
plt.title("Derivative f'(x) = 2x")
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

print(f"At x=1: f(1) = {f(1)}, f'(1) = {f_prime(1)}")
# At x=1: f(1) = 1, f'(1) = 2 (slope is 2)
```

**Interpretation**:

- **Positive derivative**: Function is increasing
- **Negative derivative**: Function is decreasing
- **Zero derivative**: Function is at a local minimum, maximum, or inflection point

**Common Derivatives**:

```python
# f(x) = x^n  →  f'(x) = n*x^(n-1)
# f(x) = x²   →  f'(x) = 2x
# f(x) = x³   →  f'(x) = 3x²

# f(x) = e^x  →  f'(x) = e^x
# f(x) = ln(x) → f'(x) = 1/x

# f(x) = sin(x) → f'(x) = cos(x)
# f(x) = cos(x) → f'(x) = -sin(x)
```

### Finding Minima: The Goal of Training

In machine learning, we want to minimize a loss function. The minimum occurs where the derivative is zero (or close to zero).

```python
# Loss function: L(w) = (w - 3)²
def loss(w):
    return (w - 3)**2

def loss_gradient(w):
    return 2 * (w - 3)

# Find minimum
w = np.linspace(0, 6, 100)
L = loss(w)

plt.plot(w, L)
plt.scatter([3], [loss(3)], color='red', s=100, label='Minimum at w=3')
plt.xlabel('w')
plt.ylabel('Loss')
plt.title('Loss Function')
plt.legend()
plt.grid(True)
plt.show()

print(f"Gradient at w=3: {loss_gradient(3)}")  # 0 (minimum!)
print(f"Gradient at w=1: {loss_gradient(1)}")  # -4 (negative → move right)
print(f"Gradient at w=5: {loss_gradient(5)}")  # 4 (positive → move left)
```

### Gradient Descent: Following the Slope Downhill

Gradient descent uses the derivative to iteratively move toward the minimum:

1. Start at a random point
2. Compute the gradient (derivative)
3. Take a step in the opposite direction (downhill)
4. Repeat until convergence

```python
def gradient_descent(initial_w, learning_rate=0.1, iterations=50):
    w = initial_w
    history = [w]

    for i in range(iterations):
        # Compute gradient
        grad = loss_gradient(w)

        # Update: move opposite to gradient
        w = w - learning_rate * grad
        history.append(w)

    return w, history

# Run gradient descent
initial_w = 0.0
final_w, history = gradient_descent(initial_w, learning_rate=0.1, iterations=20)

print(f"Started at w = {initial_w}")
print(f"Converged to w = {final_w:.4f} (true minimum: 3)")

# Visualize convergence
plt.figure(figsize=(12, 4))

# Loss landscape
plt.subplot(1, 2, 1)
w_range = np.linspace(0, 6, 100)
plt.plot(w_range, loss(w_range), label='Loss')
plt.plot(history, [loss(w) for w in history], 'ro-', label='GD path')
plt.xlabel('w')
plt.ylabel('Loss')
plt.title('Gradient Descent Path')
plt.legend()
plt.grid(True)

# Convergence over iterations
plt.subplot(1, 2, 2)
plt.plot(history, marker='o')
plt.axhline(y=3, color='r', linestyle='--', label='True minimum')
plt.xlabel('Iteration')
plt.ylabel('w')
plt.title('Parameter Convergence')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

### Partial Derivatives: Multiple Variables

Most machine learning models have many parameters. We need to compute the derivative with respect to each parameter separately—these are partial derivatives.

```python
# Loss with two parameters: L(w₁, w₂) = (w₁ - 2)² + (w₂ - 3)²
def loss_2d(w1, w2):
    return (w1 - 2)**2 + (w2 - 3)**2

# Partial derivatives
def grad_w1(w1, w2):
    return 2 * (w1 - 2)

def grad_w2(w1, w2):
    return 2 * (w2 - 3)

# Gradient vector: [∂L/∂w₁, ∂L/∂w₂]
def gradient_2d(w1, w2):
    return np.array([grad_w1(w1, w2), grad_w2(w1, w2)])

# Gradient descent in 2D
def gradient_descent_2d(initial_w, learning_rate=0.1, iterations=20):
    w = np.array(initial_w)
    history = [w.copy()]

    for i in range(iterations):
        grad = gradient_2d(w[0], w[1])
        w = w - learning_rate * grad
        history.append(w.copy())

    return w, np.array(history)

# Run optimization
initial = [0.0, 0.0]
final, history = gradient_descent_2d(initial, learning_rate=0.2)

print(f"Started at: {initial}")
print(f"Converged to: {final} (true minimum: [2, 3])")

# Visualize 2D optimization
w1_range = np.linspace(-1, 5, 100)
w2_range = np.linspace(-1, 6, 100)
W1, W2 = np.meshgrid(w1_range, w2_range)
L = loss_2d(W1, W2)

plt.figure(figsize=(10, 8))
plt.contour(W1, W2, L, levels=20, cmap='viridis')
plt.colorbar(label='Loss')
plt.plot(history[:, 0], history[:, 1], 'ro-', linewidth=2, markersize=8)
plt.scatter([2], [3], color='red', s=200, marker='*', label='Minimum')
plt.xlabel('w₁')
plt.ylabel('w₂')
plt.title('Gradient Descent in 2D')
plt.legend()
plt.grid(True)
plt.show()
```

### The Chain Rule: Composing Derivatives

The chain rule is essential for backpropagation in neural networks. It tells us how to compute derivatives of composite functions.

**Rule**: If y = f(g(x)), then dy/dx = (dy/dg) × (dg/dx)

```python
# Example: y = (x² + 1)³
# Let u = x² + 1, then y = u³
# dy/dx = (dy/du) × (du/dx) = 3u² × 2x = 3(x² + 1)² × 2x

def f(x):
    u = x**2 + 1
    y = u**3
    return y

def f_derivative(x):
    u = x**2 + 1
    # dy/du = 3u²
    dy_du = 3 * u**2
    # du/dx = 2x
    du_dx = 2 * x
    # Chain rule: dy/dx = dy/du × du/dx
    dy_dx = dy_du * du_dx
    return dy_dx

x = 2.0
print(f"f({x}) = {f(x)}")
print(f"f'({x}) = {f_derivative(x)}")

# Verify with numerical derivative
eps = 1e-5
numerical_deriv = (f(x + eps) - f(x)) / eps
print(f"Numerical derivative: {numerical_deriv:.4f}")
```

### Backpropagation: Chain Rule in Action

Neural networks use the chain rule to compute gradients through multiple layers.

```python
# Simple 2-layer network: y = ReLU(x @ W1) @ W2
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Forward pass
def forward(x, W1, W2):
    z1 = x @ W1          # Linear layer 1
    a1 = relu(z1)        # Activation
    z2 = a1 @ W2         # Linear layer 2
    return z2, (x, z1, a1)  # Return output and intermediates

# Backward pass (chain rule)
def backward(dL_dz2, cache, W1, W2):
    x, z1, a1 = cache

    # Gradient at output layer
    dL_dW2 = a1.T @ dL_dz2                    # ∂L/∂W2

    # Gradient flows back through layer 1
    dL_da1 = dL_dz2 @ W2.T                    # ∂L/∂a1
    dL_dz1 = dL_da1 * relu_derivative(z1)     # ∂L/∂z1 (chain rule!)
    dL_dW1 = x.T @ dL_dz1                     # ∂L/∂W1

    return dL_dW1, dL_dW2

# Example
np.random.seed(42)
x = np.random.randn(1, 3)   # 1 example, 3 features
W1 = np.random.randn(3, 4)  # 3 → 4 hidden units
W2 = np.random.randn(4, 1)  # 4 → 1 output

# Forward
output, cache = forward(x, W1, W2)
print(f"Output: {output}")

# Backward (assuming gradient at output is 1)
dL_doutput = np.array([[1.0]])
dL_dW1, dL_dW2 = backward(dL_doutput, cache, W1, W2)

print(f"Gradient W1 shape: {dL_dW1.shape}")  # (3, 4)
print(f"Gradient W2 shape: {dL_dW2.shape}")  # (4, 1)
```

### Why Calculus Matters for ML

**1. Training Algorithms**: Gradient descent and its variants (SGD, Adam) are the workhorses of ML.

**2. Backpropagation**: The chain rule enables training deep neural networks.

**3. Understanding Convergence**: Derivatives tell us if we're at a minimum, maximum, or saddle point.

**4. Hyperparameter Tuning**: Learning rate relates to derivative magnitude—too large and we overshoot, too small and we converge slowly.

**5. Custom Models**: Implementing new architectures requires computing gradients correctly.

## 8.3 Probability and Statistics

Machine learning is fundamentally about learning from uncertain data and making predictions under uncertainty. Probability theory provides the mathematical framework for this.

### Random Variables and Probability Distributions

A random variable is a variable whose value is uncertain. A probability distribution describes how likely different values are.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Discrete distribution: Coin flips
# P(X = heads) = 0.5, P(X = tails) = 0.5
coin_flips = np.random.choice(['H', 'T'], size=1000, p=[0.5, 0.5])
print(f"Heads: {(coin_flips == 'H').sum()/1000:.2%}")

# Discrete distribution: Dice rolls
dice_rolls = np.random.randint(1, 7, size=1000)
plt.hist(dice_rolls, bins=np.arange(0.5, 7.5, 1), edgecolor='black')
plt.xlabel('Dice Value')
plt.ylabel('Frequency')
plt.title('Dice Roll Distribution (Uniform)')
plt.show()
```

#### Common Distributions

**1. Normal (Gaussian) Distribution**: The most important distribution in ML.

```python
# Normal distribution: N(μ, σ²)
mu, sigma = 0, 1  # Mean and standard deviation

# Generate samples
samples = np.random.normal(mu, sigma, 10000)

# Plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(samples, bins=50, density=True, alpha=0.7, edgecolor='black')
x = np.linspace(-4, 4, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='True PDF')
plt.xlabel('x')
plt.ylabel('Density')
plt.title(f'Normal Distribution N({mu}, {sigma}²)')
plt.legend()

# Different parameters
plt.subplot(1, 2, 2)
x = np.linspace(-10, 10, 200)
for mu, sigma in [(0, 1), (0, 2), (2, 1)]:
    plt.plot(x, stats.norm.pdf(x, mu, sigma), label=f'μ={mu}, σ={sigma}')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Normal Distributions with Different Parameters')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Properties
print(f"Mean: {np.mean(samples):.2f} (true: {mu})")
print(f"Std: {np.std(samples):.2f} (true: {sigma})")
print(f"68% of data within 1σ: {np.sum(np.abs(samples) < 1) / len(samples):.2%}")
print(f"95% of data within 2σ: {np.sum(np.abs(samples) < 2) / len(samples):.2%}")
```

**Why it matters**: Many natural phenomena follow normal distributions. The Central Limit Theorem states that sums of many independent random variables tend toward normal distributions. Many ML algorithms assume normality.

**2. Bernoulli Distribution**: Binary outcomes (success/failure).

```python
# Bernoulli: P(X = 1) = p, P(X = 0) = 1-p
p = 0.3  # Success probability

samples = np.random.binomial(1, p, 10000)
print(f"P(X=1) = {samples.mean():.2f} (true: {p})")

# Application: Binary classification
# Model outputs probability p, sample to get prediction
```

**3. Binomial Distribution**: Number of successes in n trials.

```python
# Binomial: n trials, each with success probability p
n, p = 10, 0.3

samples = np.random.binomial(n, p, 10000)
plt.hist(samples, bins=np.arange(-0.5, n+1.5, 1), density=True, alpha=0.7, edgecolor='black')
x = np.arange(0, n+1)
plt.plot(x, stats.binom.pmf(x, n, p), 'ro-', label='True PMF')
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.title(f'Binomial Distribution: n={n}, p={p}')
plt.legend()
plt.show()

print(f"Expected successes: {n * p} (mean: {samples.mean():.2f})")
```

**4. Categorical Distribution**: Multiple discrete outcomes (generalization of Bernoulli).

```python
# Categorical: K classes with probabilities p1, p2, ..., pK
classes = ['dog', 'cat', 'bird']
probs = [0.5, 0.3, 0.2]

samples = np.random.choice(classes, size=1000, p=probs)
for cls, prob in zip(classes, probs):
    actual = (samples == cls).sum() / len(samples)
    print(f"P({cls}) = {prob:.1f}, actual: {actual:.2f}")

# Application: Multi-class classification
```

### Expected Value and Variance

**Expected Value** (mean): The average outcome if we repeated the experiment many times.

```python
# Discrete: E[X] = Σ x * P(X = x)
# For dice: E[X] = 1*(1/6) + 2*(1/6) + ... + 6*(1/6) = 3.5

dice_samples = np.random.randint(1, 7, 100000)
print(f"Dice expected value: {dice_samples.mean():.2f} (true: 3.5)")

# Continuous: E[X] = ∫ x * p(x) dx
normal_samples = np.random.normal(5, 2, 100000)
print(f"Normal expected value: {normal_samples.mean():.2f} (true: 5)")
```

**Variance**: Measures spread or dispersion.

```python
# Variance: Var(X) = E[(X - μ)²]
# Standard deviation: σ = sqrt(Var(X))

samples = np.random.normal(0, 2, 10000)
variance = np.var(samples)
std = np.std(samples)

print(f"Variance: {variance:.2f} (true: 4)")
print(f"Std dev: {std:.2f} (true: 2)")

# Low vs high variance
plt.figure(figsize=(10, 4))
x = np.linspace(-10, 10, 200)
plt.plot(x, stats.norm.pdf(x, 0, 1), label='Low variance (σ=1)')
plt.plot(x, stats.norm.pdf(x, 0, 3), label='High variance (σ=3)')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Variance Controls Spread')
plt.legend()
plt.grid(True)
plt.show()
```

### Bayes' Theorem: Updating Beliefs

Bayes' theorem describes how to update probabilities given new evidence. It's fundamental to probabilistic machine learning.

**Formula**: P(A|B) = P(B|A) × P(A) / P(B)

- P(A|B): Posterior probability (probability of A given B)
- P(B|A): Likelihood (probability of observing B given A)
- P(A): Prior probability (initial belief about A)
- P(B): Evidence (probability of observing B)

```python
# Example: Medical diagnosis
# Disease D, Test T
# P(D) = 0.01 (1% of population has disease) - PRIOR
# P(T=pos|D) = 0.95 (test detects disease 95% of time) - LIKELIHOOD
# P(T=pos|¬D) = 0.05 (5% false positive rate)

# Question: If test is positive, what's probability of having disease?

# Compute P(T=pos)
P_D = 0.01
P_not_D = 0.99
P_T_pos_given_D = 0.95
P_T_pos_given_not_D = 0.05

P_T_pos = P_T_pos_given_D * P_D + P_T_pos_given_not_D * P_not_D

# Bayes' theorem: P(D|T=pos)
P_D_given_T_pos = (P_T_pos_given_D * P_D) / P_T_pos

print(f"Prior P(Disease) = {P_D:.1%}")
print(f"Posterior P(Disease|Test+) = {P_D_given_T_pos:.1%}")
print("\nEven with positive test, only 16% chance of disease!")
print("This is because disease is rare (low prior)")
```

**Naive Bayes Classifier**: Applies Bayes' theorem for classification.

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Train Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

# Predict with probabilities
sample = X_test[0:1]
prediction = nb.predict(sample)[0]
probabilities = nb.predict_proba(sample)[0]

print(f"Prediction: {iris.target_names[prediction]}")
print("Probabilities:")
for name, prob in zip(iris.target_names, probabilities):
    print(f"  P({name}|features) = {prob:.3f}")
```

### Maximum Likelihood Estimation (MLE)

MLE is a method for estimating model parameters by finding values that maximize the likelihood of observing the data.

```python
# Example: Estimating coin bias
# We observe: H H T H H (4 heads, 1 tail)
# What's the probability p of heads?

# Likelihood: P(data|p) = p^4 * (1-p)^1
def likelihood(p, heads, tails):
    return (p ** heads) * ((1 - p) ** tails)

# Find p that maximizes likelihood
p_values = np.linspace(0, 1, 100)
likelihoods = [likelihood(p, 4, 1) for p in p_values]

plt.plot(p_values, likelihoods)
plt.xlabel('p (probability of heads)')
plt.ylabel('Likelihood')
plt.title('Likelihood Function')
plt.axvline(x=0.8, color='r', linestyle='--', label='MLE: p=0.8')
plt.legend()
plt.grid(True)
plt.show()

# MLE estimate: p = heads / total = 4/5 = 0.8
mle_p = 4 / 5
print(f"MLE estimate: p = {mle_p}")
```

**MLE for Gaussian**: Estimate mean and variance from data.

```python
# Given data, find μ and σ² that maximize likelihood
data = np.array([2.3, 1.8, 3.1, 2.5, 2.0, 2.8])

# MLE estimates
mu_mle = np.mean(data)
sigma_mle = np.std(data, ddof=0)  # Use N, not N-1

print(f"MLE μ = {mu_mle:.2f}")
print(f"MLE σ = {sigma_mle:.2f}")

# Verify: These should maximize the likelihood
x = np.linspace(0, 5, 100)
plt.hist(data, bins=10, density=True, alpha=0.5, label='Data')
plt.plot(x, stats.norm.pdf(x, mu_mle, sigma_mle), 'r-', linewidth=2, label='MLE Gaussian')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('MLE Gaussian Fit')
plt.legend()
plt.show()
```

### Why Probability Matters for ML

**1. Uncertainty Quantification**: Models output probabilities, not just predictions.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Probability outputs
sample = X_test[0:1]
probs = rf.predict_proba(sample)[0]
print(f"Probabilities: {probs}")
print(f"Prediction: {iris.target_names[probs.argmax()]} (confidence: {probs.max():.2%})")
```

**2. Loss Functions**: Cross-entropy loss is derived from likelihood maximization.

**3. Generative Models**: Explicitly model probability distributions (VAEs, GANs).

**4. Bayesian Methods**: Incorporate prior knowledge and update beliefs with data.

**5. A/B Testing**: Statistical hypothesis testing for comparing models.

## 8.4 Information Theory Basics

Information theory quantifies information content and provides tools for measuring the difference between probability distributions. It's fundamental to understanding neural networks and many ML concepts.

### Entropy: Measuring Uncertainty

Entropy measures the average amount of information (or surprise) in a random variable. High entropy means high uncertainty; low entropy means low uncertainty.

**Formula**: H(X) = -Σ P(x) log₂ P(x)

```python
import numpy as np
import matplotlib.pyplot as plt

def entropy(probs):
    """Compute entropy of discrete distribution"""
    # Filter out zero probabilities (0 log 0 = 0 by convention)
    probs = np.array(probs)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

# Example 1: Fair coin (maximum uncertainty)
fair_coin = [0.5, 0.5]
H_fair = entropy(fair_coin)
print(f"Fair coin entropy: {H_fair:.2f} bits")

# Example 2: Biased coin (less uncertainty)
biased_coin = [0.9, 0.1]
H_biased = entropy(biased_coin)
print(f"Biased coin entropy: {H_biased:.2f} bits")

# Example 3: Certain outcome (no uncertainty)
certain = [1.0, 0.0]
H_certain = entropy(certain)
print(f"Certain outcome entropy: {H_certain:.2f} bits")

# Visualize entropy vs coin bias
p_values = np.linspace(0.01, 0.99, 100)
entropies = [entropy([p, 1-p]) for p in p_values]

plt.plot(p_values, entropies)
plt.xlabel('P(Heads)')
plt.ylabel('Entropy (bits)')
plt.title('Coin Entropy vs Bias')
plt.axhline(y=1, color='r', linestyle='--', label='Maximum (p=0.5)')
plt.grid(True)
plt.legend()
plt.show()
```

**Interpretation**:

- Fair coin (p=0.5): H = 1 bit (maximum uncertainty for binary)
- Biased coin (p=0.9): H = 0.47 bits (less uncertainty)
- Certain (p=1.0): H = 0 bits (no uncertainty)

**Multi-class Entropy**:

```python
# Uniform distribution: maximum entropy
uniform = [0.25, 0.25, 0.25, 0.25]  # 4 classes
H_uniform = entropy(uniform)
print(f"Uniform (4 classes) entropy: {H_uniform:.2f} bits")

# Peaked distribution: low entropy
peaked = [0.7, 0.1, 0.1, 0.1]
H_peaked = entropy(peaked)
print(f"Peaked distribution entropy: {H_peaked:.2f} bits")

# Maximum entropy for K classes: log₂(K)
print(f"Maximum entropy for 4 classes: {np.log2(4):.2f} bits")
```

### Cross-Entropy: Measuring Distribution Mismatch

Cross-entropy measures the average number of bits needed to encode data from distribution P using a code optimized for distribution Q. It's the most common loss function for classification.

**Formula**: H(P, Q) = -Σ P(x) log Q(x)

```python
def cross_entropy(p_true, q_pred):
    """
    Cross-entropy between true distribution p and predicted distribution q
    """
    p_true = np.array(p_true)
    q_pred = np.array(q_pred)
    # Clip to avoid log(0)
    q_pred = np.clip(q_pred, 1e-10, 1)
    return -np.sum(p_true * np.log(q_pred))

# Example: Binary classification
# True label: class 0 (represented as [1, 0])
y_true = [1, 0]

# Good prediction: [0.9, 0.1]
y_pred_good = [0.9, 0.1]
ce_good = cross_entropy(y_true, y_pred_good)
print(f"Cross-entropy (good prediction): {ce_good:.3f}")

# Bad prediction: [0.2, 0.8]
y_pred_bad = [0.2, 0.8]
ce_bad = cross_entropy(y_true, y_pred_bad)
print(f"Cross-entropy (bad prediction): {ce_bad:.3f}")

# Perfect prediction: [1.0, 0.0]
y_pred_perfect = [1.0, 0.0]
ce_perfect = cross_entropy(y_true, y_pred_perfect)
print(f"Cross-entropy (perfect prediction): {ce_perfect:.3f}")
```

**In Neural Networks**:

```python
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3,
                           n_informative=15, random_state=42)

# Neural network with cross-entropy loss
nn = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, random_state=42)
nn.fit(X, y)

# Prediction with probabilities
sample = X[0:1]
probs = nn.predict_proba(sample)[0]
true_label = y[0]

# One-hot encode true label
y_true_onehot = np.zeros(3)
y_true_onehot[true_label] = 1

ce = cross_entropy(y_true_onehot, probs)
print(f"\nTrue label: {true_label}")
print(f"Predicted probabilities: {probs}")
print(f"Cross-entropy loss: {ce:.3f}")
```

### KL Divergence: Measuring Distance Between Distributions

Kullback-Leibler (KL) divergence measures how one probability distribution differs from another. It's asymmetric and always non-negative.

**Formula**: D_KL(P || Q) = Σ P(x) log(P(x) / Q(x)) = H(P, Q) - H(P)

```python
def kl_divergence(p, q):
    """KL divergence from Q to P: D_KL(P || Q)"""
    p = np.array(p)
    q = np.array(q)
    # Filter out zeros
    mask = (p > 0) & (q > 0)
    p = p[mask]
    q = q[mask]
    return np.sum(p * np.log(p / q))

# Example: Compare distributions
p = [0.4, 0.6]       # True distribution
q1 = [0.5, 0.5]      # Similar to p
q2 = [0.1, 0.9]      # Different from p

kl1 = kl_divergence(p, q1)
kl2 = kl_divergence(p, q2)

print(f"D_KL(P || Q1) = {kl1:.3f}")  # Small divergence
print(f"D_KL(P || Q2) = {kl2:.3f}")  # Large divergence

# Note: KL divergence is asymmetric
kl_forward = kl_divergence(p, q2)
kl_reverse = kl_divergence(q2, p)
print(f"\nD_KL(P || Q) = {kl_forward:.3f}")
print(f"D_KL(Q || P) = {kl_reverse:.3f}")
print("Not equal! KL divergence is asymmetric")
```

**Visualization**:

```python
# Compare two distributions
p = [0.3, 0.3, 0.4]
q = [0.2, 0.5, 0.3]

x = np.arange(len(p))
width = 0.35

plt.bar(x - width/2, p, width, label='P (true)', alpha=0.7)
plt.bar(x + width/2, q, width, label='Q (predicted)', alpha=0.7)
plt.xlabel('Class')
plt.ylabel('Probability')
plt.title(f'KL Divergence: D_KL(P || Q) = {kl_divergence(p, q):.3f}')
plt.xticks(x, ['Class 0', 'Class 1', 'Class 2'])
plt.legend()
plt.grid(True, axis='y')
plt.show()
```

### Why Information Theory Matters for ML

**1. Loss Functions**: Cross-entropy is the standard classification loss.

```python
import torch
import torch.nn as nn

# PyTorch cross-entropy loss
loss_fn = nn.CrossEntropyLoss()

# Predictions (logits)
logits = torch.tensor([[2.0, 1.0, 0.1]])  # Raw scores

# True label
target = torch.tensor([0])  # Class 0

loss = loss_fn(logits, target)
print(f"Cross-entropy loss: {loss.item():.3f}")
```

**2. Model Evaluation**: Entropy measures prediction confidence.

```python
# Low entropy = confident predictions
confident_pred = [0.9, 0.05, 0.05]
print(f"Confident prediction entropy: {entropy(confident_pred):.2f} bits")

# High entropy = uncertain predictions
uncertain_pred = [0.4, 0.3, 0.3]
print(f"Uncertain prediction entropy: {entropy(uncertain_pred):.2f} bits")
```

**3. Decision Trees**: Split criterion uses entropy or Gini (related concept).

```python
from sklearn.tree import DecisionTreeClassifier

# Decision tree with entropy criterion
dt = DecisionTreeClassifier(criterion='entropy', max_depth=3)
dt.fit(X, y)

print(f"Tree uses entropy to find best splits")
print(f"Tree depth: {dt.get_depth()}")
```

**4. Variational Autoencoders (VAEs)**: Use KL divergence in the loss function.

**5. Generative Models**: Minimize KL divergence between model and data distribution.

**6. Mutual Information**: Measures dependency between variables (feature selection).

---

In this final chapter, we've covered the mathematical foundations of machine learning. Linear algebra gives us the language for representing data and models. Calculus enables optimization through gradient descent. Probability provides a framework for uncertainty. And information theory offers tools for measuring information content and distribution differences.

These mathematical concepts aren't just theoretical—they directly impact how you build, train, debug, and improve machine learning systems. Understanding them deepens your intuition and empowers you to go beyond using ML as a black box.

With this foundation in place, you're well-equipped to explore advanced topics, implement custom models, and tackle real-world machine learning challenges with confidence.

---

[← Previous: Chapter 7](07-types-of-ml.md) | [Back to Index](index.md)

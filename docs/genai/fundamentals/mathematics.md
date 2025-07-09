# Mathematics for GenAI

!!! abstract "Mathematical Foundations"
    Master the essential mathematical concepts that power generative AI systems. This section focuses on intuitive understanding backed by practical examples.

## Why Mathematics Matters in GenAI

Understanding the mathematics behind GenAI isn't just academic—it's practical:

- **Better Prompting**: Understand how models process information
- **Efficient Implementation**: Optimize performance and memory usage
- **Debugging**: Diagnose issues when things go wrong
- **Innovation**: Build upon existing techniques for new applications

## Linear Algebra Essentials

### Vectors: The Building Blocks

Everything in GenAI starts with vectors—lists of numbers that represent information.

```python
import numpy as np
import matplotlib.pyplot as plt

# A word embedding is just a vector
word_embedding = np.array([0.2, -0.1, 0.8, 0.3, -0.5])
print(f"Word embedding: {word_embedding}")

# Similarity between words = similarity between vectors
def cosine_similarity(vec1, vec2):
    """Calculate similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    return dot_product / (magnitude1 * magnitude2)

# Example: Similar words have similar embeddings
king = np.array([0.5, 0.8, 0.2, 0.9, 0.1])
queen = np.array([0.4, 0.7, 0.3, 0.8, 0.2])
cat = np.array([-0.3, 0.1, 0.6, -0.2, 0.8])

print(f"King-Queen similarity: {cosine_similarity(king, queen):.3f}")
print(f"King-Cat similarity: {cosine_similarity(king, cat):.3f}")
```

### Matrices: Transforming Information

Matrices transform vectors from one representation to another.

```python
# Neural network layer: matrix multiplication
input_dim = 4
hidden_dim = 3

# Weight matrix transforms input to hidden representation
W = np.random.randn(input_dim, hidden_dim)
b = np.zeros(hidden_dim)

# Input vector
x = np.array([1.0, 0.5, -0.3, 0.8])

# Linear transformation
hidden = np.dot(x, W) + b
print(f"Input: {x}")
print(f"Hidden: {hidden}")
```

### The Attention Matrix

The heart of transformers is matrix operations that compute attention.

```python
# Simplified attention calculation
def simple_attention(query, key, value):
    """
    Args:
        query: What we're looking for [seq_len, d_model]
        key: What we're comparing against [seq_len, d_model]
        value: What we actually use [seq_len, d_model]
    """
    # Step 1: Calculate attention scores
    scores = np.dot(query, key.T)  # [seq_len, seq_len]
    
    # Step 2: Normalize scores
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    
    # Step 3: Apply attention to values
    output = np.dot(attention_weights, value)
    
    return output, attention_weights

# Example with 3 words
seq_len, d_model = 3, 4
query = np.random.randn(seq_len, d_model)
key = np.random.randn(seq_len, d_model)
value = np.random.randn(seq_len, d_model)

output, weights = simple_attention(query, key, value)
print(f"Attention weights:\n{weights}")
```

## Probability and Statistics

### Why Probability Matters

AI models are fundamentally probabilistic—they predict the **likelihood** of different outcomes.

```python
# Language model predicts next word probability
vocabulary = ['the', 'cat', 'sat', 'on', 'mat']
probabilities = [0.3, 0.2, 0.25, 0.15, 0.1]

# Higher probability = more likely next word
print("Next word predictions:")
for word, prob in zip(vocabulary, probabilities):
    print(f"'{word}': {prob:.1%}")
```

### Distributions in GenAI

```python
# Softmax: Convert numbers to probabilities
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Numerical stability
    return exp_x / np.sum(exp_x)

# Model outputs (logits)
logits = np.array([2.0, 1.0, 0.5, 3.0])
probs = softmax(logits)

print(f"Logits: {logits}")
print(f"Probabilities: {probs}")
print(f"Sum: {np.sum(probs):.3f}")  # Should be 1.0
```

## Calculus for Optimization

### Gradients: The Direction of Improvement

Models learn by following gradients—the direction that reduces error.

```python
# Simple loss function
def loss_function(w):
    return (w - 2) ** 2  # Minimum at w = 2

# Gradient (derivative)
def gradient(w):
    return 2 * (w - 2)

# Gradient descent
w = 0.0  # Starting point
learning_rate = 0.1

for step in range(10):
    grad = gradient(w)
    w = w - learning_rate * grad
    loss = loss_function(w)
    print(f"Step {step}: w={w:.3f}, loss={loss:.3f}")
```

### Backpropagation: Chain Rule in Action

```python
# Simple 2-layer network
def forward_pass(x, w1, w2):
    h = np.maximum(0, x * w1)  # ReLU activation
    y = h * w2
    return y, h

def backward_pass(x, y_pred, y_true, w1, w2, h):
    # Loss gradient
    dy = 2 * (y_pred - y_true)
    
    # Gradients through layers
    dw2 = dy * h
    dh = dy * w2
    
    # ReLU gradient
    dw1 = dh * x if h > 0 else 0
    
    return dw1, dw2

# Training example
x, y_true = 1.0, 3.0
w1, w2 = 0.5, 0.5

for epoch in range(5):
    y_pred, h = forward_pass(x, w1, w2)
    dw1, dw2 = backward_pass(x, y_pred, y_true, w1, w2, h)
    
    # Update weights
    w1 -= 0.1 * dw1
    w2 -= 0.1 * dw2
    
    loss = (y_pred - y_true) ** 2
    print(f"Epoch {epoch}: loss={loss:.3f}, w1={w1:.3f}, w2={w2:.3f}")
```

## Information Theory

### Entropy: Measuring Uncertainty

```python
# Entropy measures information content
def entropy(probabilities):
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

# Low entropy = predictable
certain_probs = np.array([0.9, 0.05, 0.05])
print(f"Certain distribution entropy: {entropy(certain_probs):.3f}")

# High entropy = unpredictable
uncertain_probs = np.array([0.33, 0.33, 0.34])
print(f"Uncertain distribution entropy: {entropy(uncertain_probs):.3f}")
```

### Cross-Entropy Loss

```python
# Cross-entropy: How far off are our predictions?
def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-10))

# True answer (one-hot encoded)
true_label = np.array([0, 1, 0])  # Word 2 is correct

# Model predictions
good_prediction = np.array([0.1, 0.8, 0.1])
bad_prediction = np.array([0.4, 0.3, 0.3])

print(f"Good prediction loss: {cross_entropy_loss(true_label, good_prediction):.3f}")
print(f"Bad prediction loss: {cross_entropy_loss(true_label, bad_prediction):.3f}")
```

## Practical Applications

### Attention Computation

```python
# Scaled dot-product attention (simplified)
def scaled_dot_product_attention(Q, K, V):
    """
    Args:
        Q: Query matrix [seq_len, d_k]
        K: Key matrix [seq_len, d_k]
        V: Value matrix [seq_len, d_v]
    """
    d_k = Q.shape[-1]
    
    # Attention scores
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    
    # Attention weights
    attention_weights = softmax(scores)
    
    # Weighted values
    output = np.dot(attention_weights, V)
    
    return output, attention_weights

# Example usage
seq_len, d_k, d_v = 4, 8, 8
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_v)

output, attention = scaled_dot_product_attention(Q, K, V)
print(f"Attention pattern:\n{attention}")
```

### Embedding Arithmetic

```python
# Famous example: King - Man + Woman ≈ Queen
king = np.array([0.5, 0.8, 0.2, 0.9])
man = np.array([0.3, 0.4, 0.1, 0.6])
woman = np.array([0.2, 0.7, 0.3, 0.8])

# Vector arithmetic
result = king - man + woman
print(f"King - Man + Woman = {result}")

# This should be close to queen embedding
queen = np.array([0.4, 0.9, 0.4, 0.9])
similarity = cosine_similarity(result, queen)
print(f"Similarity to Queen: {similarity:.3f}")
```

## Key Takeaways

!!! success "Mathematical Intuition"
    - **Vectors** represent information (words, images, concepts)
    - **Matrices** transform information between representations
    - **Attention** weights determine what information to focus on
    - **Gradients** guide learning in the right direction
    - **Probability** quantifies uncertainty and prediction confidence

## Next Steps

Now that you understand the mathematical foundations:

1. **Practice**: Try the code examples and experiment with different values
2. **Visualize**: Use matplotlib to plot attention weights and embeddings
3. **Connect**: See how these concepts appear in transformer architectures
4. **Build**: Apply these principles in your own GenAI projects

**Ready to dive deeper?** Continue with [Neural Networks](neural-networks.md) to see how these mathematical concepts combine to create intelligent systems.
A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Principal Component Analysis (PCA) example
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
data = np.random.randn(100, 50)  # 100 samples, 50 features

# Apply PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)

print(f"Original shape: {data.shape}")
print(f"Reduced shape: {reduced_data.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
```

## Calculus for Optimization

### Gradients and Derivatives

The foundation of backpropagation and gradient descent.

#### Single Variable Calculus

```python
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define symbolic variable
x = sp.Symbol('x')

# Example function: f(x) = x^2 + 2x + 1
f = x**2 + 2*x + 1

# Compute derivative
f_prime = sp.diff(f, x)
print(f"f(x) = {f}")
print(f"f'(x) = {f_prime}")

# Find critical points
critical_points = sp.solve(f_prime, x)
print(f"Critical points: {critical_points}")

# Numerical example
x_vals = np.linspace(-3, 1, 100)
y_vals = [float(f.subs(x, val)) for val in x_vals]
y_prime_vals = [float(f_prime.subs(x, val)) for val in x_vals]

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(x_vals, y_vals, label='f(x)')
plt.title('Function')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_vals, y_prime_vals, label="f'(x)", color='red')
plt.title('Derivative')
plt.legend()
plt.tight_layout()
plt.show()
```

#### Multivariable Calculus

```python
# Multivariable function
x, y = sp.symbols('x y')
f = x**2 + y**2 + 2*x*y

# Partial derivatives
df_dx = sp.diff(f, x)
df_dy = sp.diff(f, y)

print(f"f(x,y) = {f}")
print(f"∂f/∂x = {df_dx}")
print(f"∂f/∂y = {df_dy}")

# Gradient vector
gradient = [df_dx, df_dy]
print(f"∇f = {gradient}")

# Hessian matrix (second derivatives)
hessian = sp.Matrix([[sp.diff(f, x, 2), sp.diff(f, x, y)],
                     [sp.diff(f, y, x), sp.diff(f, y, 2)]])
print(f"Hessian matrix:\n{hessian}")
```

#### Chain Rule in Neural Networks

```python
# Chain rule example for backpropagation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Simple neural network layer
def forward_backward_example():
    # Input
    x = 0.5
    
    # Weights and bias
    w1, w2 = 0.8, 0.3
    b = 0.1
    
    # Forward pass
    z1 = w1 * x + b  # Linear transformation
    a1 = sigmoid(z1)  # Activation
    z2 = w2 * a1      # Next layer
    
    print(f"Forward pass:")
    print(f"z1 = {z1}, a1 = {a1}, z2 = {z2}")
    
    # Backward pass (chain rule)
    # Assume loss L = z2^2 (simple quadratic loss)
    dL_dz2 = 2 * z2
    dz2_da1 = w2
    da1_dz1 = sigmoid_derivative(z1)
    dz1_dw1 = x
    
    # Chain rule: dL/dw1 = dL/dz2 * dz2/da1 * da1/dz1 * dz1/dw1
    dL_dw1 = dL_dz2 * dz2_da1 * da1_dz1 * dz1_dw1
    
    print(f"\nBackward pass:")
    print(f"dL/dw1 = {dL_dw1}")
    
    return dL_dw1

gradient_w1 = forward_backward_example()
```

### Optimization Theory

#### Gradient Descent Variants

```python
class GradientDescentOptimizer:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
    
    def step(self, params, gradients):
        """Vanilla gradient descent"""
        return [param - self.lr * grad for param, grad in zip(params, gradients)]

class MomentumOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None
    
    def step(self, params, gradients):
        """Gradient descent with momentum"""
        if self.velocity is None:
            self.velocity = [np.zeros_like(grad) for grad in gradients]
        
        # Update velocity
        self.velocity = [self.momentum * v + grad 
                        for v, grad in zip(self.velocity, gradients)]
        
        # Update parameters
        return [param - self.lr * v for param, v in zip(params, self.velocity)]

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step
    
    def step(self, params, gradients):
        """Adam optimizer"""
        self.t += 1
        
        if self.m is None:
            self.m = [np.zeros_like(grad) for grad in gradients]
            self.v = [np.zeros_like(grad) for grad in gradients]
        
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, gradients)):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            # Update parameters
            updated_param = param - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            updated_params.append(updated_param)
        
        return updated_params

# Example optimization
def quadratic_loss(x):
    return x**2 + 2*x + 1

def quadratic_gradient(x):
    return 2*x + 2

# Compare optimizers
optimizers = {
    'SGD': GradientDescentOptimizer(0.1),
    'Momentum': MomentumOptimizer(0.1, 0.9),
    'Adam': AdamOptimizer(0.1)
}

for name, optimizer in optimizers.items():
    x = [2.0]  # Starting point
    history = [x[0]]
    
    for _ in range(20):
        grad = [quadratic_gradient(x[0])]
        x = optimizer.step(x, grad)
        history.append(x[0])
    
    print(f"{name}: Final x = {x[0]:.6f}, Loss = {quadratic_loss(x[0]):.6f}")
```

## Probability and Statistics

### Probability Distributions

#### Fundamental Distributions

```python
import scipy.stats as stats
import matplotlib.pyplot as plt

# Normal Distribution
x = np.linspace(-4, 4, 100)
normal_pdf = stats.norm.pdf(x, 0, 1)
normal_cdf = stats.norm.cdf(x, 0, 1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(x, normal_pdf, label='PDF')
plt.title('Normal Distribution PDF')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, normal_cdf, label='CDF')
plt.title('Normal Distribution CDF')
plt.legend()
plt.tight_layout()
plt.show()

# Sampling from distributions
normal_samples = np.random.normal(0, 1, 1000)
exponential_samples = np.random.exponential(2, 1000)

print(f"Normal samples - Mean: {np.mean(normal_samples):.3f}, Std: {np.std(normal_samples):.3f}")
print(f"Exponential samples - Mean: {np.mean(exponential_samples):.3f}")
```

#### Bayesian Inference

```python
# Bayesian coin flip example
def bayesian_coin_flip():
    # Prior: Beta(1, 1) - uniform prior
    alpha_prior = 1
    beta_prior = 1
    
    # Observed data: 7 heads out of 10 flips
    heads = 7
    tails = 3
    
    # Posterior: Beta(alpha_prior + heads, beta_prior + tails)
    alpha_posterior = alpha_prior + heads
    beta_posterior = beta_prior + tails
    
    # Plot prior and posterior
    x = np.linspace(0, 1, 100)
    prior = stats.beta.pdf(x, alpha_prior, beta_prior)
    posterior = stats.beta.pdf(x, alpha_posterior, beta_posterior)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, prior, label='Prior', alpha=0.7)
    plt.plot(x, posterior, label='Posterior', alpha=0.7)
    plt.axvline(0.5, color='red', linestyle='--', label='Fair coin')
    plt.xlabel('Probability of heads')
    plt.ylabel('Density')
    plt.title('Bayesian Coin Flip Analysis')
    plt.legend()
    plt.show()
    
    # Posterior mean and credible interval
    posterior_mean = alpha_posterior / (alpha_posterior + beta_posterior)
    credible_interval = stats.beta.interval(0.95, alpha_posterior, beta_posterior)
    
    print(f"Posterior mean: {posterior_mean:.3f}")
    print(f"95% Credible interval: ({credible_interval[0]:.3f}, {credible_interval[1]:.3f})")

bayesian_coin_flip()
```

### Information Theory

#### Entropy and Mutual Information

```python
def entropy(probabilities):
    """Calculate Shannon entropy"""
    p = np.array(probabilities)
    p = p[p > 0]  # Remove zero probabilities
    return -np.sum(p * np.log2(p))

def cross_entropy(true_probs, pred_probs):
    """Calculate cross entropy"""
    true_p = np.array(true_probs)
    pred_p = np.array(pred_probs)
    pred_p = np.clip(pred_p, 1e-15, 1 - 1e-15)  # Avoid log(0)
    return -np.sum(true_p * np.log2(pred_p))

def kl_divergence(p, q):
    """Calculate KL divergence"""
    p = np.array(p)
    q = np.array(q)
    q = np.clip(q, 1e-15, 1 - 1e-15)  # Avoid division by 0
    return np.sum(p * np.log2(p / q))

# Examples
uniform_dist = [0.25, 0.25, 0.25, 0.25]
skewed_dist = [0.7, 0.2, 0.05, 0.05]
predicted_dist = [0.6, 0.25, 0.1, 0.05]

print(f"Entropy of uniform distribution: {entropy(uniform_dist):.3f} bits")
print(f"Entropy of skewed distribution: {entropy(skewed_dist):.3f} bits")
print(f"Cross entropy: {cross_entropy(skewed_dist, predicted_dist):.3f} bits")
print(f"KL divergence: {kl_divergence(skewed_dist, predicted_dist):.3f} bits")
```

#### Mutual Information in Feature Selection

```python
from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import make_classification

# Generate sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, 
                         n_redundant=5, random_state=42)

# Calculate mutual information
mi_scores = mutual_info_classif(X, y)

# Visualize feature importance
plt.figure(figsize=(12, 6))
plt.bar(range(len(mi_scores)), mi_scores)
plt.xlabel('Feature Index')
plt.ylabel('Mutual Information')
plt.title('Feature Importance via Mutual Information')
plt.show()

print(f"Top 5 features by mutual information: {np.argsort(mi_scores)[-5:]}")
```

## Advanced Mathematical Concepts

### Matrix Calculus

#### Gradients of Matrix Functions

```python
# Matrix calculus examples for neural networks
def matrix_gradient_examples():
    # Example 1: Gradient of quadratic form
    # f(x) = x^T A x, gradient is 2Ax (assuming A is symmetric)
    
    A = np.array([[2, 1], [1, 3]])
    x = np.array([1, 2])
    
    # Function value
    f_val = x.T @ A @ x
    
    # Gradient
    gradient = 2 * A @ x
    
    print(f"f(x) = {f_val}")
    print(f"∇f(x) = {gradient}")
    
    # Example 2: Gradient of matrix trace
    # f(X) = tr(AX), gradient is A^T
    
    A = np.random.randn(3, 3)
    X = np.random.randn(3, 3)
    
    # Function value
    f_val = np.trace(A @ X)
    
    # Gradient
    gradient = A.T
    
    print(f"\nTrace example:")
    print(f"f(X) = {f_val}")
    print(f"∇f(X) shape = {gradient.shape}")

matrix_gradient_examples()
```

### Fourier Analysis

#### Discrete Fourier Transform

```python
# Fourier analysis for understanding attention patterns
def fourier_analysis_example():
    # Create a signal
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 0.1, len(t))
    
    # Compute FFT
    fft_values = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), t[1] - t[0])
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(t, signal)
    plt.title('Original Signal')
    plt.xlabel('Time')
    
    plt.subplot(2, 2, 2)
    plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_values)[:len(fft_values)//2])
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency')
    
    # Attention pattern analysis (conceptual)
    attention_weights = np.random.rand(100, 100)
    attention_fft = np.fft.fft2(attention_weights)
    
    plt.subplot(2, 2, 3)
    plt.imshow(attention_weights, cmap='viridis')
    plt.title('Attention Pattern')
    
    plt.subplot(2, 2, 4)
    plt.imshow(np.log(np.abs(attention_fft) + 1e-8), cmap='viridis')
    plt.title('Frequency Analysis of Attention')
    
    plt.tight_layout()
    plt.show()

fourier_analysis_example()
```

## Applications in GenAI

### Attention Mechanism Mathematics

```python
def attention_mathematics():
    """Mathematical breakdown of attention mechanism"""
    
    # Dimensions
    seq_len = 10
    d_model = 64
    d_k = d_model // 8  # Multi-head attention
    
    # Input embeddings
    X = np.random.randn(seq_len, d_model)
    
    # Query, Key, Value projections
    W_q = np.random.randn(d_model, d_k)
    W_k = np.random.randn(d_model, d_k)
    W_v = np.random.randn(d_model, d_k)
    
    # Compute Q, K, V
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    
    # Attention scores
    scores = Q @ K.T / np.sqrt(d_k)
    
    # Softmax attention weights
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    
    # Attended values
    output = attention_weights @ V
    
    print(f"Input shape: {X.shape}")
    print(f"Q, K, V shapes: {Q.shape}, {K.shape}, {V.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Output shape: {output.shape}")
    
    # Visualize attention pattern
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_weights, cmap='Blues')
    plt.colorbar()
    plt.title('Attention Weights Heatmap')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.show()
    
    return attention_weights, output

attention_weights, attention_output = attention_mathematics()
```

### Embedding Space Geometry

```python
def embedding_geometry():
    """Analyze the geometry of embedding spaces"""
    
    # Simulate word embeddings
    vocab_size = 1000
    embedding_dim = 300
    
    # Random embeddings (in practice, these would be learned)
    embeddings = np.random.randn(vocab_size, embedding_dim)
    
    # Normalize embeddings (common practice)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Compute pairwise similarities
    similarity_matrix = embeddings @ embeddings.T
    
    # Analyze distribution of similarities
    similarities = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(similarities, bins=50, alpha=0.7)
    plt.title('Distribution of Cosine Similarities')
    plt.xlabel('Cosine Similarity')
    
    # Dimensionality reduction for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings[:100])  # First 100 words
    
    plt.subplot(1, 3, 2)
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
    plt.title('2D PCA of Embeddings')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    # Analyze embedding norms
    norms = np.linalg.norm(embeddings, axis=1)
    
    plt.subplot(1, 3, 3)
    plt.hist(norms, bins=50, alpha=0.7)
    plt.title('Distribution of Embedding Norms')
    plt.xlabel('L2 Norm')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Mean cosine similarity: {np.mean(similarities):.3f}")
    print(f"Std cosine similarity: {np.std(similarities):.3f}")
    print(f"Mean embedding norm: {np.mean(norms):.3f}")

embedding_geometry()
```

---

!!! tip "Key Takeaways"
    - **Linear algebra** provides the computational foundation for all neural operations
    - **Calculus** enables optimization through gradient-based methods
    - **Probability theory** underlies uncertainty quantification and Bayesian methods
    - **Information theory** helps understand model capacity and compression

!!! warning "Implementation Notes"
    - Always consider numerical stability when implementing mathematical operations
    - Use appropriate libraries (NumPy, SciPy) for efficient computation
    - Understand the computational complexity of different operations
    - Gradient computation is the heart of neural network training

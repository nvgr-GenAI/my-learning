# Mathematics for Machine Learning

**Build the mathematical foundation for understanding ML algorithms.** Master linear algebra, calculus, and probability - the three pillars of machine learning.

**Difficulty:** 🟡 Intermediate | **Time:** 1-2 weeks | **Prerequisites:** High school math, Basic Python

---

## Quick Reference

| Area | Key Topics | Why Important | Priority |
|------|-----------|---------------|----------|
| **Linear Algebra** | Vectors, matrices, operations | Data representation, computations | 🔥🔥🔥 Critical |
| **Calculus** | Derivatives, gradients, chain rule | Optimization, backpropagation | 🔥🔥🔥 Critical |
| **Probability** | Distributions, Bayes, expectation | Uncertainty, inference | 🔥🔥 Important |
| **Statistics** | Mean, variance, hypothesis testing | Understanding data | 🔥🔥 Important |

---

## Complete Guide

=== "📖 Overview"
    ## Why Mathematics Matters

    **Core Idea:** ML algorithms are mathematical functions that learn from data.

    **You don't need to be a mathematician**, but understanding the basics helps you:
    - Debug models when things go wrong
    - Tune hyperparameters intelligently
    - Read research papers and documentation
    - Design custom solutions
    - Interview successfully

    ---

    ## The Three Pillars

    ```mermaid
    graph TB
        ML[Machine Learning] --> LA[Linear Algebra<br/>Data Representation]
        ML --> Calc[Calculus<br/>Optimization]
        ML --> Prob[Probability<br/>Uncertainty]

        LA --> LA1[Vectors & Matrices]
        LA --> LA2[Matrix Operations]
        LA --> LA3[Eigenvalues]

        Calc --> C1[Derivatives]
        Calc --> C2[Gradients]
        Calc --> C3[Chain Rule]

        Prob --> P1[Distributions]
        Prob --> P2[Bayes Theorem]
        Prob --> P3[Expectation]

        style ML fill:#e1f5ff
        style LA fill:#ccffcc
        style Calc fill:#ffffcc
        style Prob fill:#ffcccc
    ```

    ---

    ## How Much Math Do You Need?

    **Minimum (Can Use ML Libraries):**
    - Understand what vectors and matrices represent
    - Know what derivatives measure (rate of change)
    - Basic probability concepts

    **Intermediate (Can Tune Models):**
    - Matrix operations and their meanings
    - Gradients and how gradient descent works
    - Probability distributions and their properties

    **Advanced (Can Research/Innovate):**
    - Matrix decompositions (SVD, eigenvalues)
    - Multivariable calculus and optimization theory
    - Advanced probability (Bayesian inference, information theory)

    **This guide:** Focuses on intermediate level - practical understanding.

=== "🧮 Linear Algebra"
    ## Why Linear Algebra?

    **Core principle:** Machine learning is all about manipulating large matrices efficiently.

    - **Data** is a matrix (rows = samples, columns = features)
    - **Model parameters** are vectors/matrices
    - **Computations** are matrix operations
    - **Transformations** are matrix multiplications

    ---

    ## Vectors

    ### Definition
    A vector is an ordered list of numbers.

    $$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix}$$

    **In ML:** A single data point with multiple features

    **Example:**
    ```
    House: [2000 sq ft, 3 bedrooms, 15 years old]
    Vector: [2000, 3, 15]
    ```

    ---

    ### Vector Operations

    **Addition:**
    $$\begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 3 \\ 4 \end{bmatrix} = \begin{bmatrix} 4 \\ 6 \end{bmatrix}$$

    **Scalar Multiplication:**
    $$3 \times \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 3 \\ 6 \end{bmatrix}$$

    **Dot Product (Inner Product):**
    $$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i$$

    Example: $[1, 2, 3] \cdot [4, 5, 6] = 1×4 + 2×5 + 3×6 = 32$

    **Meaning in ML:** Similarity between vectors, predictions in linear models

    ---

    ### Vector Norms

    **L2 Norm (Euclidean length):**
    $$\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{n} v_i^2}$$

    **L1 Norm (Manhattan distance):**
    $$\|\mathbf{v}\|_1 = \sum_{i=1}^{n} |v_i|$$

    **Uses in ML:**
    - L2 regularization (Ridge)
    - L1 regularization (Lasso)
    - Distance metrics

    ---

    ## Matrices

    ### Definition
    A matrix is a 2D array of numbers.

    $$\mathbf{X} = \begin{bmatrix}
    x_{11} & x_{12} & x_{13} \\
    x_{21} & x_{22} & x_{23} \\
    x_{31} & x_{32} & x_{33}
    \end{bmatrix}$$

    **In ML:** Dataset with m samples and n features

    $$\mathbf{X} = \begin{bmatrix}
    — & \mathbf{x}^{(1)} & — \\
    — & \mathbf{x}^{(2)} & — \\
    & \vdots & \\
    — & \mathbf{x}^{(m)} & —
    \end{bmatrix} \in \mathbb{R}^{m \times n}$$

    ---

    ### Matrix Operations

    **Transpose:**
    $$(\mathbf{A}^T)_{ij} = \mathbf{A}_{ji}$$

    Example:
    $$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}^T = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}$$

    ---

    **Matrix Multiplication:**
    $$(\mathbf{AB})_{ij} = \sum_{k} A_{ik}B_{kj}$$

    **Requirements:** Number of columns in A = number of rows in B

    **Example:**
    $$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 \\ 6 \end{bmatrix} = \begin{bmatrix} 1×5 + 2×6 \\ 3×5 + 4×6 \end{bmatrix} = \begin{bmatrix} 17 \\ 39 \end{bmatrix}$$

    **In ML:** Core operation for predictions
    $$\mathbf{y} = \mathbf{X}\mathbf{w}$$

    ---

    **Identity Matrix:**
    $$\mathbf{I} = \begin{bmatrix}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 1
    \end{bmatrix}$$

    Property: $\mathbf{AI} = \mathbf{IA} = \mathbf{A}$

    ---

    **Inverse Matrix:**
    $$\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$$

    **In ML:** Solving normal equations
    $$\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

    ---

    ## Key Concepts for ML

    ### Eigenvalues and Eigenvectors

    For square matrix $\mathbf{A}$:
    $$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$

    Where:
    - $\mathbf{v}$ is eigenvector
    - $\lambda$ is eigenvalue

    **Meaning:** Direction that only gets scaled, not rotated

    **Uses:**
    - Principal Component Analysis (PCA)
    - Understanding matrix behavior
    - Dimensionality reduction

    ---

    ### Matrix Decomposition

    **Singular Value Decomposition (SVD):**
    $$\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$$

    **Uses:**
    - PCA
    - Recommender systems
    - Data compression

    ---

    ## Python Implementation

    ```python
    import numpy as np

    # ========================================
    # Vectors
    # ========================================

    # Create vectors
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])

    # Vector addition
    v_sum = v1 + v2
    print(f"Sum: {v_sum}")  # [5, 7, 9]

    # Scalar multiplication
    v_scaled = 3 * v1
    print(f"Scaled: {v_scaled}")  # [3, 6, 9]

    # Dot product
    dot_product = np.dot(v1, v2)
    print(f"Dot product: {dot_product}")  # 32

    # Norms
    l2_norm = np.linalg.norm(v1)  # L2 norm
    l1_norm = np.linalg.norm(v1, ord=1)  # L1 norm
    print(f"L2 norm: {l2_norm:.3f}")
    print(f"L1 norm: {l1_norm:.3f}")

    # ========================================
    # Matrices
    # ========================================

    # Create matrix (2x3)
    X = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    print(f"Shape: {X.shape}")  # (2, 3)

    # Transpose
    X_T = X.T
    print(f"Transpose shape: {X_T.shape}")  # (3, 2)

    # Matrix multiplication
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    C = np.matmul(A, B)  # or A @ B
    print(f"Matrix multiplication:\n{C}")

    # Identity matrix
    I = np.eye(3)
    print(f"Identity:\n{I}")

    # Matrix inverse
    A_inv = np.linalg.inv(A)
    print(f"Inverse:\n{A_inv}")

    # Verify: A * A_inv = I
    result = np.matmul(A, A_inv)
    print(f"A * A_inv:\n{result}")

    # ========================================
    # ML Example: Linear Regression
    # ========================================

    # Data: 100 samples, 5 features
    X = np.random.randn(100, 5)
    y = np.random.randn(100)

    # Add bias column (column of 1s)
    X_b = np.c_[np.ones((100, 1)), X]

    # Normal equation: θ = (X^T X)^-1 X^T y
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    print(f"\nLearned parameters: {theta}")

    # Predictions
    y_pred = X_b @ theta
    print(f"Predictions shape: {y_pred.shape}")
    ```

=== "📐 Calculus"
    ## Why Calculus?

    **Core principle:** Machine learning is optimization - finding parameters that minimize error.

    **Calculus gives us:** Tools to find minimum of functions efficiently.

    ---

    ## Derivatives

    ### Single Variable

    **Definition:** Rate of change of function

    $$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

    **Geometric interpretation:** Slope of tangent line

    ---

    ### Common Derivatives

    | Function | Derivative |
    |----------|------------|
    | $f(x) = c$ | $f'(x) = 0$ |
    | $f(x) = x$ | $f'(x) = 1$ |
    | $f(x) = x^n$ | $f'(x) = nx^{n-1}$ |
    | $f(x) = e^x$ | $f'(x) = e^x$ |
    | $f(x) = \ln(x)$ | $f'(x) = \frac{1}{x}$ |
    | $f(x) = \sin(x)$ | $f'(x) = \cos(x)$ |

    ---

    ### Rules

    **Sum Rule:**
    $$(f + g)' = f' + g'$$

    **Product Rule:**
    $$(fg)' = f'g + fg'$$

    **Chain Rule:**
    $$(f(g(x)))' = f'(g(x)) \cdot g'(x)$$

    **Chain rule is critical for neural networks!**

    ---

    ## Partial Derivatives

    **For multivariable functions:** Derivative with respect to one variable

    $$\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x+h, y) - f(x, y)}{h}$$

    **Example:**
    $$f(x, y) = x^2 + 2xy + y^2$$

    $$\frac{\partial f}{\partial x} = 2x + 2y$$
    $$\frac{\partial f}{\partial y} = 2x + 2y$$

    ---

    ## Gradient

    **Definition:** Vector of all partial derivatives

    $$\nabla f = \begin{bmatrix}
    \frac{\partial f}{\partial x_1} \\
    \frac{\partial f}{\partial x_2} \\
    \vdots \\
    \frac{\partial f}{\partial x_n}
    \end{bmatrix}$$

    **Meaning:** Direction of steepest ascent

    **For ML:** We follow negative gradient (steepest descent)

    ---

    ## Gradient Descent

    **Goal:** Minimize loss function $J(\theta)$

    **Algorithm:**
    $$\theta := \theta - \alpha \nabla J(\theta)$$

    Where:
    - $\theta$ = parameters
    - $\alpha$ = learning rate
    - $\nabla J(\theta)$ = gradient

    **Intuition:** Roll downhill to find minimum

    ---

    ### Example: Linear Regression

    **Loss function:**
    $$J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})^2$$

    Where $h_\theta(x) = \theta^T x$

    **Gradient:**
    $$\frac{\partial J}{\partial \theta_j} = \frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$$

    **Update rule:**
    $$\theta_j := \theta_j - \alpha \frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$$

    ---

    ## Chain Rule (Backpropagation)

    **Critical for neural networks!**

    **Setup:** Nested functions
    $$z = f(y), \quad y = g(x)$$

    **Chain rule:**
    $$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$$

    **Example:**
    $$z = (x^2 + 1)^3$$

    Let $y = x^2 + 1$, so $z = y^3$

    $$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx} = 3y^2 \cdot 2x = 3(x^2+1)^2 \cdot 2x$$

    ---

    ## Python Implementation

    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    # ========================================
    # Numerical Derivatives
    # ========================================

    def derivative(f, x, h=1e-5):
        """Numerical derivative"""
        return (f(x + h) - f(x)) / h

    # Example function
    f = lambda x: x**2

    # Derivative at x=3
    x = 3
    df_dx = derivative(f, x)
    print(f"f(x) = x^2")
    print(f"f'(3) ≈ {df_dx:.4f}")  # Should be ~6
    print(f"Analytical: {2*x}")

    # ========================================
    # Gradient Descent
    # ========================================

    def gradient_descent_1d(f, df, x0, learning_rate=0.1, n_iterations=100):
        """
        Minimize function using gradient descent

        Args:
            f: Function to minimize
            df: Derivative of f
            x0: Starting point
            learning_rate: Step size
            n_iterations: Number of steps
        """
        x = x0
        history = [x]

        for i in range(n_iterations):
            gradient = df(x)
            x = x - learning_rate * gradient
            history.append(x)

        return x, history

    # Function: f(x) = (x-3)^2
    f = lambda x: (x - 3)**2
    df = lambda x: 2*(x - 3)

    # Find minimum starting from x=0
    x_min, history = gradient_descent_1d(f, df, x0=0, learning_rate=0.1)

    print(f"\nGradient Descent:")
    print(f"Found minimum at x = {x_min:.4f}")
    print(f"True minimum at x = 3.0")

    # Visualize
    x_plot = np.linspace(-1, 7, 100)
    y_plot = f(x_plot)

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x)')
    plt.plot(history, [f(x) for x in history], 'ro-',
            markersize=8, label='GD steps')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Gradient Descent Optimization')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # ========================================
    # Multivariable: Gradient Descent for Linear Regression
    # ========================================

    # Generate data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    # Add bias term
    X_b = np.c_[np.ones((100, 1)), X]

    def compute_cost(X, y, theta):
        """MSE cost function"""
        m = len(y)
        predictions = X @ theta
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        return cost

    def compute_gradient(X, y, theta):
        """Gradient of MSE"""
        m = len(y)
        predictions = X @ theta
        gradient = (1/m) * X.T @ (predictions - y)
        return gradient

    def gradient_descent(X, y, theta, learning_rate, n_iterations):
        """Gradient descent for linear regression"""
        costs = []
        for i in range(n_iterations):
            gradient = compute_gradient(X, y, theta)
            theta = theta - learning_rate * gradient
            cost = compute_cost(X, y, theta)
            costs.append(cost)

            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}")

        return theta, costs

    # Initialize
    theta = np.random.randn(2, 1)

    # Run gradient descent
    theta_final, costs = gradient_descent(
        X_b, y, theta, learning_rate=0.1, n_iterations=1000
    )

    print(f"\nFinal parameters: {theta_final.ravel()}")
    print(f"True parameters: [4, 3]")

    # Plot cost over time
    plt.figure(figsize=(10, 6))
    plt.plot(costs)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Function During Training')
    plt.grid(alpha=0.3)
    plt.show()
    ```

=== "📊 Probability & Statistics"
    ## Why Probability?

    **Core principle:** Real-world data is noisy and uncertain.

    **Probability helps us:**
    - Model uncertainty
    - Make predictions with confidence
    - Understand generalization
    - Build probabilistic models

    ---

    ## Basic Probability

    ### Definitions

    **Probability:** $P(A)$ = likelihood of event A

    **Properties:**
    - $0 \leq P(A) \leq 1$
    - $P(\text{certain event}) = 1$
    - $P(\text{impossible event}) = 0$

    ---

    ### Conditional Probability

    $$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

    **Meaning:** Probability of A given that B occurred

    **Example:** P(spam | contains "free money")

    ---

    ### Bayes' Theorem

    $$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

    **In ML notation:**
    $$P(\text{hypothesis}|\text{data}) = \frac{P(\text{data}|\text{hypothesis})P(\text{hypothesis})}{P(\text{data})}$$

    **Terms:**
    - $P(A|B)$ = Posterior (what we want)
    - $P(B|A)$ = Likelihood
    - $P(A)$ = Prior
    - $P(B)$ = Evidence

    **Uses:**
    - Naive Bayes classifier
    - Bayesian inference
    - Spam filtering

    ---

    ## Random Variables

    **Discrete:** Takes countable values (coin flip, dice)

    **Continuous:** Takes any value in range (height, temperature)

    ---

    ## Probability Distributions

    ### Bernoulli Distribution

    **Binary outcome:** success (1) or failure (0)

    $$P(X = 1) = p, \quad P(X = 0) = 1-p$$

    **Example:** Coin flip, binary classification

    ---

    ### Normal (Gaussian) Distribution

    **Most important distribution in ML!**

    $$P(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

    **Parameters:**
    - $\mu$ = mean (center)
    - $\sigma^2$ = variance (spread)

    **Properties:**
    - Bell-shaped curve
    - 68% within 1 std dev
    - 95% within 2 std dev
    - 99.7% within 3 std dev

    **Uses:** Noise in data, Central Limit Theorem, many ML assumptions

    ---

    ## Expectation and Variance

    ### Expectation (Mean)

    **Discrete:**
    $$\mathbb{E}[X] = \sum_x x \cdot P(X=x)$$

    **Continuous:**
    $$\mathbb{E}[X] = \int x \cdot p(x) dx$$

    **Meaning:** Average value

    ---

    ### Variance

    $$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

    **Meaning:** Spread around mean

    **Standard Deviation:** $\sigma = \sqrt{\text{Var}(X)}$

    ---

    ## Maximum Likelihood Estimation

    **Goal:** Find parameters that maximize probability of observed data

    $$\theta^* = \arg\max_\theta P(\text{data}|\theta)$$

    **Log-likelihood:** (easier to optimize)
    $$\theta^* = \arg\max_\theta \log P(\text{data}|\theta)$$

    **Example:** Linear regression with Gaussian noise
    - Maximizing likelihood = Minimizing MSE!

    ---

    ## Python Implementation

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    # ========================================
    # Probability Distributions
    # ========================================

    # Normal distribution
    mu, sigma = 0, 1
    x = np.linspace(-4, 4, 100)
    pdf = stats.norm.pdf(x, mu, sigma)

    plt.figure(figsize=(12, 4))

    # Plot 1: PDF
    plt.subplot(1, 3, 1)
    plt.plot(x, pdf, 'b-', linewidth=2)
    plt.fill_between(x, pdf, alpha=0.3)
    plt.axvline(mu, color='r', linestyle='--', label=f'Mean = {mu}')
    plt.axvline(mu + sigma, color='g', linestyle='--',
                label=f'Std = {sigma}')
    plt.axvline(mu - sigma, color='g', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.title('Normal Distribution PDF')
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot 2: Different means
    plt.subplot(1, 3, 2)
    for mu in [-2, 0, 2]:
        pdf = stats.norm.pdf(x, mu, sigma)
        plt.plot(x, pdf, linewidth=2, label=f'μ={mu}')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.title('Effect of Mean')
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot 3: Different variances
    plt.subplot(1, 3, 3)
    mu = 0
    for sigma in [0.5, 1, 2]:
        pdf = stats.norm.pdf(x, mu, sigma)
        plt.plot(x, pdf, linewidth=2, label=f'σ={sigma}')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.title('Effect of Standard Deviation')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ========================================
    # Sampling and Statistics
    # ========================================

    # Generate samples
    samples = np.random.normal(mu=5, sigma=2, size=1000)

    # Calculate statistics
    sample_mean = np.mean(samples)
    sample_std = np.std(samples)
    sample_var = np.var(samples)

    print(f"True mean: 5, Sample mean: {sample_mean:.3f}")
    print(f"True std: 2, Sample std: {sample_std:.3f}")

    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=50, density=True, alpha=0.7, label='Samples')

    # Overlay true distribution
    x = np.linspace(samples.min(), samples.max(), 100)
    pdf = stats.norm.pdf(x, 5, 2)
    plt.plot(x, pdf, 'r-', linewidth=2, label='True Distribution')

    plt.axvline(sample_mean, color='g', linestyle='--',
                linewidth=2, label='Sample Mean')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Sampling from Normal Distribution')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # ========================================
    # Bayes' Theorem Example
    # ========================================

    def naive_bayes_example():
        """
        Example: Spam detection using Bayes' theorem
        """
        # Prior probabilities
        P_spam = 0.3  # 30% of emails are spam
        P_not_spam = 0.7

        # Likelihood: P(word|class)
        P_free_given_spam = 0.8  # 80% of spam contains "free"
        P_free_given_not_spam = 0.1  # 10% of non-spam contains "free"

        # Calculate P(free)
        P_free = (P_free_given_spam * P_spam +
                  P_free_given_not_spam * P_not_spam)

        # Bayes' theorem: P(spam|free)
        P_spam_given_free = (
            P_free_given_spam * P_spam / P_free
        )

        print("\nBayes' Theorem Example: Spam Detection")
        print(f"P(spam) = {P_spam}")
        print(f"P('free'|spam) = {P_free_given_spam}")
        print(f"P(spam|'free') = {P_spam_given_free:.3f}")

    naive_bayes_example()

    # ========================================
    # Maximum Likelihood Estimation
    # ========================================

    # Generate data from normal distribution
    true_mu, true_sigma = 3, 2
    data = np.random.normal(true_mu, true_sigma, size=100)

    # MLE estimates
    mle_mu = np.mean(data)
    mle_sigma = np.std(data, ddof=1)  # Unbiased estimator

    print(f"\nMaximum Likelihood Estimation:")
    print(f"True μ: {true_mu}, MLE μ: {mle_mu:.3f}")
    print(f"True σ: {true_sigma}, MLE σ: {mle_sigma:.3f}")
    ```

=== "🎯 Practice"
    ## Beginner Problems

    **Problem 1: Vector Operations**

    Implement from scratch (no NumPy):
    - Vector addition
    - Dot product
    - L2 norm
    - Verify with NumPy

    ---

    **Problem 2: Matrix Multiplication**

    - Implement matrix multiplication manually
    - Compare with NumPy
    - Visualize computation for 2x2 matrices

    ---

    **Problem 3: Gradient Descent**

    Minimize $f(x) = x^2 + 4x + 4$:
    - Derive gradient analytically
    - Implement gradient descent
    - Visualize convergence

    ---

    ## Intermediate Problems

    **Problem 4: Linear Regression from Scratch**

    Using only NumPy:
    - Implement gradient descent
    - Implement normal equation
    - Compare convergence and results

    ---

    **Problem 5: Probability Distributions**

    - Sample from different distributions
    - Estimate parameters using MLE
    - Visualize and compare

    ---

    ## Advanced Problems

    **Problem 6: Backpropagation**

    For 2-layer neural network:
    - Implement forward pass
    - Derive gradients using chain rule
    - Implement backward pass
    - Verify with numerical gradients

---

## Essential Formulas Cheat Sheet

### Linear Algebra

| Operation | Formula | Use in ML |
|-----------|---------|-----------|
| Dot Product | $\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i$ | Similarity, predictions |
| Matrix Mult | $(\mathbf{AB})_{ij} = \sum_k A_{ik}B_{kj}$ | Neural network layers |
| Transpose | $(\mathbf{A}^T)_{ij} = \mathbf{A}_{ji}$ | Gradient computation |
| Inverse | $\mathbf{AA}^{-1} = \mathbf{I}$ | Normal equation |

### Calculus

| Concept | Formula | Use in ML |
|---------|---------|-----------|
| Derivative | $f'(x) = \lim_{h \to 0} \frac{f(x+h)-f(x)}{h}$ | Optimization |
| Gradient | $\nabla f = [\frac{\partial f}{\partial x_1}, ..., \frac{\partial f}{\partial x_n}]$ | Gradient descent |
| Chain Rule | $\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$ | Backpropagation |
| Gradient Descent | $\theta := \theta - \alpha \nabla J(\theta)$ | Training models |

### Probability

| Concept | Formula | Use in ML |
|---------|---------|-----------|
| Bayes' Theorem | $P(A\|B) = \frac{P(B\|A)P(A)}{P(B)}$ | Naive Bayes, inference |
| Expectation | $\mathbb{E}[X] = \sum_x x P(x)$ | Average prediction |
| Variance | $\text{Var}(X) = \mathbb{E}[(X-\mu)^2]$ | Uncertainty |
| Normal Distribution | $P(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | Modeling noise |

---

## Interview Preparation

### Common Questions

**Q1: Why is linear algebra important for machine learning?**

**Strong Answer:** "Linear algebra is fundamental because ML operates on large matrices. Data is represented as matrices (rows=samples, columns=features), model parameters are vectors/matrices, and predictions are matrix multiplications. For example, in linear regression, predictions are $\mathbf{y} = \mathbf{Xw}$ - a matrix-vector multiplication. Understanding linear algebra helps you optimize computations, debug shape mismatches, and design efficient models."

**Q2: Explain gradient descent.**

**Strong Answer:** "Gradient descent is an optimization algorithm to minimize loss functions. It works by computing the gradient (direction of steepest ascent) and moving in the opposite direction. The update rule is $\theta := \theta - \alpha \nabla J(\theta)$, where $\alpha$ is learning rate. Think of it as rolling downhill - the gradient tells you which way is down, and you take steps in that direction until you reach the bottom (local minimum)."

**Q3: What is the chain rule and why is it important?**

**Strong Answer:** "Chain rule lets us compute derivatives of nested functions: $\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$. It's critical for backpropagation in neural networks - we compute gradients layer by layer, propagating them backward through the network. Without chain rule, we couldn't train deep neural networks efficiently. It's the mathematical foundation of how models learn."

**Q4: Explain the difference between L1 and L2 norms.**

**Strong Answer:** "L1 norm (Manhattan distance) is sum of absolute values: $\sum|x_i|$. L2 norm (Euclidean distance) is square root of sum of squares: $\sqrt{\sum x_i^2}$. In ML, L1 regularization (Lasso) tends to create sparse solutions (many coefficients exactly zero) - good for feature selection. L2 regularization (Ridge) shrinks all coefficients but doesn't zero them - good for handling correlated features. L2 is also easier to differentiate."

**Q5: What is Bayes' theorem and how is it used in ML?**

**Strong Answer:** "Bayes' theorem relates conditional probabilities: $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$. In ML, it's used for updating beliefs given evidence. Naive Bayes classifier uses it to compute $P(\text{class}|\text{features})$ from $P(\text{features}|\text{class})$. It's also fundamental to Bayesian inference - starting with prior beliefs and updating them with data to get posterior beliefs. Common in spam filtering, medical diagnosis, and probabilistic models."

---

## Related Topics

- [Optimization](../optimization/) - Advanced optimization techniques
- [Linear Regression](../supervised-learning/regression/linear-regression.md) - Applies linear algebra
- [Neural Networks](../deep-learning/neural-networks.md) - Uses calculus and backpropagation
- [Naive Bayes](../supervised-learning/classification/naive-bayes.md) - Applies probability
- [PCA](../unsupervised-learning/dimensionality-reduction/pca.md) - Uses eigenvalues

---

## References

1. **Book:** *Mathematics for Machine Learning* by Deisenroth et al. - [Free PDF](https://mml-book.github.io/)
2. **Course:** [MIT 18.06 Linear Algebra](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/) by Gilbert Strang
3. **Course:** [Khan Academy - Multivariable Calculus](https://www.khanacademy.org/math/multivariable-calculus)
4. **Book:** *Linear Algebra and Its Applications* by Gilbert Strang
5. **3Blue1Brown:** [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) - Visual explanations
6. **StatQuest:** Probability and statistics videos

---

**Next Steps:**

- Practice implementing algorithms from scratch to solidify understanding
- Work through [3Blue1Brown's series](https://www.youtube.com/c/3blue1brown) for visual intuition
- Apply math concepts in real ML projects
- Read research papers to see advanced applications

**Remember:** You don't need to be a math expert to use ML, but understanding the fundamentals makes you a better practitioner!

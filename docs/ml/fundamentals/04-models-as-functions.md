# Chapter 4: Models as Functions

At its heart, machine learning is about finding functions. Given inputs, we want to predict outputs. A spam detector is a function mapping emails to "spam" or "not spam." A house price predictor is a function mapping features (size, location, age) to prices. Every machine learning model, no matter how complex, is fundamentally a mathematical function.

Understanding models as functions helps demystify machine learning. In this chapter, we'll explore what models really are mathematically, the difference between parameters we learn and hyperparameters we choose, the vast space of possible functions we could learn, and how different model types capture different kinds of relationships.

---

## 4.1 What is a Model?

### Models as Mathematical Functions

**A model is a mathematical function** that maps inputs (features) to outputs (predictions). We write this as:

```text
ŷ = f(x)
```

Where:
- `x` = input features (what we know)
- `f` = the model (the function we're learning)
- `ŷ` = predicted output (what we want to know)

The hat (^) on ŷ indicates it's a prediction, not the true value `y`.

```python
import numpy as np

# Conceptual: A model is a function
def model(x):
    """
    A simple model function
    Input: x (features)
    Output: prediction
    """
    # This could be as simple as...
    return 2 * x + 1

# Or as complex as a neural network with millions of parameters

# Using the model
x = np.array([1, 2, 3, 4, 5])
predictions = model(x)

print("Inputs:", x)
print("Predictions:", predictions)
print("\nThe model is just a function: f(x) = 2x + 1")
```

### From Data to Function

Machine learning's goal is to find the right function `f` that best describes the relationship between inputs and outputs in your data.

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Training data: relationship between study hours and test scores
hours_studied = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
test_scores = np.array([50, 55, 65, 70, 80])

# Learn the function from data
model = LinearRegression()
model.fit(hours_studied, test_scores)

# The learned function
print("Learned function: f(x) = {:.2f}x + {:.2f}".format(
    model.coef_[0], model.intercept_
))

# Use the function to predict
new_hours = np.array([[6], [7]])
predicted_scores = model.predict(new_hours)

print("\nPredictions:")
print("6 hours → score:", predicted_scores[0])
print("7 hours → score:", predicted_scores[1])
```

### Different Models, Different Functions

Different algorithms learn different types of functions. Linear models learn straight lines, decision trees learn step functions, neural networks learn highly complex nonlinear functions.

```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np

# Same data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])  # y = x²

# Different models learn different function shapes
linear_model = LinearRegression()
tree_model = DecisionTreeRegressor(max_depth=3)
neural_model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)

# Train all three
linear_model.fit(X, y)
tree_model.fit(X, y)
neural_model.fit(X, y)

# Predict
test_x = np.array([[3.5]])
print("Input: 3.5")
print(f"True value: {3.5**2}")
print(f"Linear model: {linear_model.predict(test_x)[0]:.2f}")
print(f"Tree model: {tree_model.predict(test_x)[0]:.2f}")
print(f"Neural model: {neural_model.predict(test_x)[0]:.2f}")
print("\nDifferent models = different function approximations")
```

### Components of a Model Function

Most models can be written as:

```text
f(x) = g(w·x + b)
```

Where:
- `x` = input features
- `w` = weights (learned parameters)
- `b` = bias (learned parameter)
- `g` = transformation function

```python
import numpy as np

def simple_model(x, w, b, activation='linear'):
    """
    Basic model structure: f(x) = g(w·x + b)
    """
    # Linear combination
    z = np.dot(w, x) + b

    # Apply activation function g
    if activation == 'linear':
        return z
    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-z))
    elif activation == 'relu':
        return np.maximum(0, z)

# Example
x = np.array([1, 2, 3])  # Input features
w = np.array([0.5, 0.3, 0.2])  # Learned weights
b = 1.0  # Learned bias

print("Input:", x)
print("Linear output:", simple_model(x, w, b, 'linear'))
print("Sigmoid output:", simple_model(x, w, b, 'sigmoid'))
print("ReLU output:", simple_model(x, w, b, 'relu'))
```

---

## 4.2 Parameters vs Hyperparameters

Understanding the difference between parameters and hyperparameters is crucial for effective model development.

### Parameters: What the Model Learns

**Parameters** are the internal variables that the model learns from data during training. These define the function. You don't set these manually—the learning algorithm finds them.

**Weights** and **biases** are the most common parameters.

**Weights** determine how much each feature influences the prediction. Larger weights mean greater influence.

**Biases** (or intercepts) allow the model to shift the output up or down, independent of inputs.

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Training data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([3, 5, 7, 9])

# Train model
model = LinearRegression()
model.fit(X, y)

# Parameters (learned from data)
print("Parameters learned from data:")
print(f"  Weights: {model.coef_}")
print(f"  Bias: {model.intercept_}")

# The model function
print(f"\nLearned function:")
print(f"  f(x1, x2) = {model.coef_[0]:.2f}·x1 + {model.coef_[1]:.2f}·x2 + {model.intercept_:.2f}")

# These parameters define how the model makes predictions
new_data = np.array([[2, 4]])
prediction = model.predict(new_data)
print(f"\nPrediction for [2, 4]: {prediction[0]:.2f}")
```

**Example: Neural Network Parameters**

Neural networks have millions of parameters (weights and biases for each connection between neurons).

```python
from sklearn.neural_network import MLPRegressor

# Simple neural network
model = MLPRegressor(hidden_layer_sizes=(10, 5), random_state=42)
model.fit(X, y)

# Count parameters
n_params = sum(layer.size for layer in model.coefs_) + sum(layer.size for layer in model.intercepts_)
print(f"Total parameters: {n_params}")

# Structure
print(f"Input layer: 2 features")
print(f"Hidden layer 1: 10 neurons")
print(f"Hidden layer 2: 5 neurons")
print(f"Output layer: 1 neuron")
print(f"\nWeights (parameters) connect every neuron to every neuron in next layer")
```

### Hyperparameters: What You Choose

**Hyperparameters** are settings you configure before training. They control how the learning process works and the model's structure. The learning algorithm doesn't find these—you must choose them.

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# Different hyperparameters for different models

# Decision Tree hyperparameters
tree = DecisionTreeRegressor(
    max_depth=5,           # Maximum tree depth
    min_samples_split=10,  # Minimum samples to split a node
    min_samples_leaf=5     # Minimum samples in a leaf
)

# Random Forest hyperparameters
forest = RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum depth of each tree
    max_features='sqrt',   # Number of features to consider for splits
    random_state=42
)

# Neural Network hyperparameters
neural_net = MLPRegressor(
    hidden_layer_sizes=(100, 50),  # Network architecture
    activation='relu',              # Activation function
    learning_rate_init=0.001,       # Initial learning rate
    max_iter=1000,                  # Number of training iterations
    random_state=42
)

print("Hyperparameters are chosen BEFORE training")
print("Parameters are learned DURING training")
```

### Key Differences

| Aspect | Parameters | Hyperparameters |
|--------|-----------|----------------|
| **Set by** | Learning algorithm | You (the practitioner) |
| **When** | During training | Before training |
| **Examples** | Weights, biases | Learning rate, tree depth, number of layers |
| **Purpose** | Define the model function | Control the learning process |
| **Count** | Usually many (thousands to billions) | Usually few (5-20) |

```python
from sklearn.linear_model import Ridge
import numpy as np

# Generate data
np.random.seed(42)
X = np.random.rand(100, 5)
y = np.random.rand(100)

# Hyperparameter: regularization strength (YOU choose this)
alpha = 1.0

# Create and train model
model = Ridge(alpha=alpha)  # alpha is a hyperparameter
model.fit(X, y)

# Parameters: learned from data
print("Hyperparameter (you chose):")
print(f"  alpha (regularization): {alpha}")

print("\nParameters (model learned):")
print(f"  Weights: {model.coef_}")
print(f"  Bias: {model.intercept_:.4f}")

print("\nYou set hyperparameters")
print("Model learns parameters")
```

### Finding Good Hyperparameters

Since hyperparameters aren't learned automatically, you must search for good values. This is called **hyperparameter tuning**.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# Generate data
X = np.random.rand(100, 3)
y = np.random.rand(100)

# Define hyperparameters to try
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Try all combinations
grid_search = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2'
)

grid_search.fit(X, y)

print("Best hyperparameters found:")
print(grid_search.best_params_)
print(f"\nBest score: {grid_search.best_score_:.3f}")

# These hyperparameters produced the best-performing model
```

---

## 4.3 The Hypothesis Space

When we train a model, we're searching through a space of possible functions. This space is called the **hypothesis space**.

### All Possible Functions

The **hypothesis space** is the set of all functions that your chosen algorithm can represent. Different algorithms have different hypothesis spaces.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = 2 * X.flatten() + 1 + np.random.randn(50) * 2

# Linear model hypothesis space: all straight lines
from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(X, y)

# Polynomial model hypothesis space: all polynomials of degree 2
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
poly = make_pipeline(PolynomialFeatures(2), LinearRegression())
poly.fit(X, y)

print("Linear hypothesis space: {a·x + b | a, b ∈ ℝ}")
print("  Can represent: straight lines only")
print()
print("Polynomial (degree 2) hypothesis space: {a·x² + b·x + c | a, b, c ∈ ℝ}")
print("  Can represent: parabolas and straight lines")
print()
print("More complex models = larger hypothesis space")
print("Larger hypothesis space = more functions you can represent")
```

### Model Complexity

**Model complexity** refers to how flexible a model is—how many different patterns it can capture. More complex models have larger hypothesis spaces.

**Simple model**: Few parameters, limited flexibility, can only represent simple patterns.

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Data with nonlinear pattern
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])  # y = x²

# Simple linear model (low complexity)
simple_model = LinearRegression()
simple_model.fit(X, y)

# Predictions
predictions = simple_model.predict(X)

print("Simple model (linear):")
print(f"  Parameters: 2 (weight + bias)")
print(f"  Can represent: straight lines only")
print(f"  Training error: {np.mean((y - predictions)**2):.2f}")
print("  Too simple for this data (underfits)")
```

**Complex model**: Many parameters, high flexibility, can represent complex patterns.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# Complex polynomial model (high complexity)
complex_model = make_pipeline(
    PolynomialFeatures(degree=10),
    LinearRegression()
)
complex_model.fit(X, y)

# Predictions
predictions = complex_model.predict(X)

print("\nComplex model (degree-10 polynomial):")
print(f"  Parameters: 11 (weights for x¹, x², ..., x¹⁰ + bias)")
print(f"  Can represent: very curvy functions")
print(f"  Training error: {np.mean((y - predictions)**2):.2f}")
print("  Might be too complex (could overfit on new data)")
```

### Model Capacity

**Model capacity** is the model's ability to fit a wide variety of functions. It's closely related to complexity but emphasizes what the model can learn.

- **Low capacity**: Can only learn simple patterns
- **High capacity**: Can learn complex, intricate patterns

```python
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# Generate data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = np.sin(X).ravel() + np.random.randn(100) * 0.1

# Low capacity: shallow tree
low_capacity = DecisionTreeRegressor(max_depth=2)
low_capacity.fit(X, y)

# High capacity: deep tree
high_capacity = DecisionTreeRegressor(max_depth=20)
high_capacity.fit(X, y)

print("Low capacity model (max_depth=2):")
print(f"  Number of leaf nodes: {low_capacity.get_n_leaves()}")
print("  Can learn: simple step functions")

print("\nHigh capacity model (max_depth=20):")
print(f"  Number of leaf nodes: {high_capacity.get_n_leaves()}")
print("  Can learn: complex, detailed patterns")

print("\nHigher capacity = can fit more complex patterns")
print("But risk: might memorize noise instead of learning signal")
```

### The Right Complexity

The goal is to find the right balance—enough complexity to capture the true pattern, but not so much that you memorize noise.

```python
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# Generate data
np.random.seed(42)
X = np.random.rand(200, 1) * 10
y_true = np.sin(X).ravel()
y = y_true + np.random.randn(200) * 0.3

# Try different complexities
complexities = [2, 5, 10, 20]

print("Model Complexity vs Performance:")
print("-" * 50)

for depth in complexities:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X, y)

    train_error = np.mean((y - model.predict(X))**2)

    print(f"Max depth {depth:2d}: Train error = {train_error:.4f}")

print("\nSweet spot: Not too simple, not too complex")
```

---

## 4.4 Linear vs Nonlinear Models

Models can be categorized by the type of function they learn: linear or nonlinear. This distinction affects what patterns they can capture and how they make decisions.

### Linear Models

**Linear models** assume the relationship between inputs and output is linear. The output is a weighted sum of inputs plus a bias.

Form: `y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Simple linear relationship
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([3, 5, 7, 9, 11])  # y = x1 + x2 + 1

# Linear model
model = LinearRegression()
model.fit(X, y)

print("Linear model equation:")
print(f"y = {model.coef_[0]:.2f}·x1 + {model.coef_[1]:.2f}·x2 + {model.intercept_:.2f}")

# Visualize the function
test_X = np.array([[3, 3], [4, 4], [5, 5]])
predictions = model.predict(test_X)

print("\nPredictions:")
for x, pred in zip(test_X, predictions):
    print(f"  x1={x[0]}, x2={x[1]} → ŷ={pred:.2f}")

print("\nLinear: output changes proportionally with inputs")
```

**Examples of Linear Models:**

- Linear Regression
- Logistic Regression (linear in the transformed space)
- Support Vector Machines with linear kernel
- Perceptron

**Characteristics:**

```python
import numpy as np

# What makes a model linear?

def linear_model(x1, x2, w1=2, w2=3, b=1):
    """Linear model: output is weighted sum of inputs"""
    return w1 * x1 + w2 * x2 + b

# Property 1: Superposition
x1a, x2a = 1, 2
x1b, x2b = 3, 4

result_separate = (linear_model(x1a, x2a) + linear_model(x1b, x2b))
result_combined = linear_model(x1a + x1b, x2a + x2b)

print("Property: Superposition")
print(f"  f(a) + f(b) = {result_separate:.1f}")
print(f"  f(a+b) = {result_combined:.1f}")
print(f"  Equal? {np.isclose(result_separate, result_combined)}")

# Property 2: Homogeneity
scaling = 2
result_scaled_input = linear_model(scaling * x1a, scaling * x2a)
result_scaled_output = scaling * linear_model(x1a, x2a)

print("\nProperty: Homogeneity (with b=0)")
# Note: bias breaks strict homogeneity, but the feature transformation is still linear
```

**Strengths:**
- Fast to train
- Interpretable (can see feature importance)
- Work well with limited data
- Computationally efficient

**Limitations:**
- Can only model linear relationships
- Struggle with complex patterns

```python
# When linear models struggle
X = np.array([[x] for x in range(-10, 11)])
y_linear = 2 * X.flatten() + 1
y_quadratic = X.flatten() ** 2

linear_model = LinearRegression()

# Works great for linear data
linear_model.fit(X, y_linear)
linear_score = linear_model.score(X, y_linear)

# Struggles with nonlinear data
linear_model.fit(X, y_quadratic)
quadratic_score = linear_model.score(X, y_quadratic)

print("Linear model performance:")
print(f"  On linear data: R² = {linear_score:.3f}")
print(f"  On quadratic data: R² = {quadratic_score:.3f}")
print("\nLinear models cannot capture nonlinear relationships")
```

### Nonlinear Models

**Nonlinear models** can capture curved, complex relationships. The relationship between inputs and outputs is not constrained to a straight line or plane.

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np

# Nonlinear data: sine wave
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).ravel()

# Different nonlinear models
tree = DecisionTreeRegressor(max_depth=5)
forest = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
neural_net = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)

# Train all
tree.fit(X, y)
forest.fit(X, y)
neural_net.fit(X, y)

# Compare performance
print("Performance on nonlinear (sine) data:")
print(f"  Decision Tree: R² = {tree.score(X, y):.3f}")
print(f"  Random Forest: R² = {forest.score(X, y):.3f}")
print(f"  Neural Network: R² = {neural_net.score(X, y):.3f}")

print("\nNonlinear models can capture curves and complex patterns")
```

**Examples of Nonlinear Models:**

- Decision Trees
- Random Forests
- Gradient Boosting
- Neural Networks
- Support Vector Machines with nonlinear kernels
- K-Nearest Neighbors

**Characteristics:**

```python
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# Decision tree: nonlinear through piece-wise constant regions
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([1, 4, 2, 8, 3, 9, 4, 10])  # Irregular pattern

tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)

test_X = np.linspace(1, 8, 50).reshape(-1, 1)
predictions = tree.predict(test_X)

print("Decision tree creates step functions")
print("  Changes prediction at decision boundaries")
print("  Can model non-smooth relationships")
print(f"\nNumber of regions (leaves): {tree.get_n_leaves()}")
```

**Strengths:**
- Can model complex, nonlinear patterns
- No assumptions about relationship shape
- Can capture interactions between features automatically

**Limitations:**
- Need more data
- Longer training time
- Less interpretable
- Risk of overfitting

### Making Linear Models Nonlinear

You can give linear models nonlinear capacity by transforming features.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import numpy as np

# Nonlinear data
X = np.array([[x] for x in range(1, 6)])
y = np.array([1, 4, 9, 16, 25])  # y = x²

# Standard linear model (fails)
linear = LinearRegression()
linear.fit(X, y)

# Linear model with polynomial features (succeeds!)
poly_linear = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)
poly_linear.fit(X, y)

print("Standard linear model:")
print(f"  R² score: {linear.score(X, y):.3f}")

print("\nLinear model with polynomial features:")
print(f"  R² score: {poly_linear.score(X, y):.3f}")

print("\nPolynomial features: [x] → [1, x, x²]")
print("Still uses linear regression, but on transformed features")
print("Linear in the transformed space = nonlinear in original space")
```

### Choosing Between Linear and Nonlinear

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

# Generate different types of data
np.random.seed(42)

# Linear data
X_linear = np.random.rand(200, 5)
y_linear = X_linear.sum(axis=1) + np.random.randn(200) * 0.1

# Nonlinear data (with interactions)
X_nonlinear = np.random.rand(200, 5)
y_nonlinear = (X_nonlinear[:, 0] * X_nonlinear[:, 1] +
               np.sin(X_nonlinear[:, 2]) +
               np.random.randn(200) * 0.1)

# Compare models
linear_model = LinearRegression()
nonlinear_model = RandomForestRegressor(random_state=42)

# On linear data
linear_on_linear = cross_val_score(linear_model, X_linear, y_linear, cv=5).mean()
nonlinear_on_linear = cross_val_score(nonlinear_model, X_linear, y_linear, cv=5).mean()

# On nonlinear data
linear_on_nonlinear = cross_val_score(linear_model, X_nonlinear, y_nonlinear, cv=5).mean()
nonlinear_on_nonlinear = cross_val_score(nonlinear_model, X_nonlinear, y_nonlinear, cv=5).mean()

print("Performance Comparison:")
print("\nOn linear data:")
print(f"  Linear model: {linear_on_linear:.3f}")
print(f"  Nonlinear model: {nonlinear_on_linear:.3f}")

print("\nOn nonlinear data:")
print(f"  Linear model: {linear_on_nonlinear:.3f}")
print(f"  Nonlinear model: {nonlinear_on_nonlinear:.3f}")

print("\nRule of thumb:")
print("  - Start with linear for interpretability and speed")
print("  - Use nonlinear if linear models underperform")
print("  - Nonlinear models need more data to shine")
```

---

## Summary

Models are fundamentally functions that map inputs to outputs:

**Models as Functions:** Every machine learning model is a mathematical function f(x) → ŷ that transforms input features into predictions. Understanding this helps demystify even the most complex models. The goal of training is to find the function that best describes the pattern in your data.

**Parameters vs Hyperparameters:** Parameters (weights, biases) are learned automatically during training—they define the specific function. Hyperparameters (learning rate, tree depth, architecture) are settings you choose before training—they control how learning happens and the model's structure. Parameters are found by the algorithm; hyperparameters are found by you (often through search).

**Hypothesis Space:** The hypothesis space is the set of all functions your algorithm can represent. Simple models (linear regression) have small hypothesis spaces (all straight lines). Complex models (deep neural networks) have vast hypothesis spaces (incredibly complex functions). Model complexity and capacity determine how flexible your model is—too simple and you underfit, too complex and you overfit.

**Linear vs Nonlinear:** Linear models assume the output is a weighted sum of inputs—they're fast, interpretable, but limited to linear relationships. Nonlinear models can capture curves and complex patterns—they're more flexible but need more data and are less interpretable. You can give linear models nonlinear power by transforming features (like polynomial features).

With a solid understanding of models as functions, we're ready to explore how machines actually learn these functions—the optimization process that finds the best parameters. That's the focus of our next chapter.

---

[← Previous: Chapter 3](03-numerical-representation.md) | [Back to Index](index.md) | [Next: Chapter 5 →](05-optimization.md)

# Boosting

**Sequentially build strong learners by focusing on mistakes.** Boosting is the secret behind competition-winning models like XGBoost and LightGBM.

**Difficulty:** 🟡 Intermediate to Advanced | **Time:** 4-5 hours | **Prerequisites:** Decision Trees, Gradient Descent

---

## Overview

Boosting trains models sequentially, where each new model focuses on correcting the errors made by previous models. Unlike bagging which reduces variance, boosting reduces bias by building complexity incrementally.

**Core idea:** Learn from mistakes - each model focuses on what previous models got wrong.

**Use cases:** Kaggle competitions, fraud detection, ranking systems, any high-accuracy requirement

---

## Intuition 💡

### The Big Idea

Imagine learning to shoot basketball free throws:
1. **First attempt:** You miss most shots (weak learner)
2. **Analyze:** Notice you aim too high
3. **Second attempt:** Adjust aim, but now you're too low on some
4. **Analyze:** Focus on those specific mistakes
5. **Repeat:** Each iteration improves on specific weaknesses

Boosting works the same way - each model learns from previous mistakes!

```mermaid
graph LR
    A[Training Data<br/>All equal weights] --> B1[Model 1<br/>Weak Learner]
    B1 --> C1[Errors: 30%<br/>Increase weights on errors]
    C1 --> B2[Model 2<br/>Focus on hard examples]
    B2 --> C2[Errors: 20%<br/>Increase weights]
    C2 --> B3[Model 3<br/>Refine further]
    B3 --> C3[Errors: 10%]
    C3 --> D[Weighted Combination<br/>Final Strong Learner]

    style A fill:#e1f5ff
    style C1 fill:#ffcccc
    style C2 fill:#ffddcc
    style C3 fill:#ffeecc
    style D fill:#ccffcc
```

### Real-World Analogy

**Team of Students Learning:**
- **Student 1 (Weak):** Studies basics, gets 60% correct
- **Student 2:** Focuses only on problems Student 1 got wrong, masters those
- **Student 3:** Focuses on problems Students 1 & 2 still miss
- **Final Exam:** Combine all their knowledge → 95% correct!

### Sequential vs Parallel (Bagging)

```
Bagging (Parallel):
Model 1 ──┐
Model 2 ──┤──→ Average ──→ Prediction
Model 3 ──┘

Boosting (Sequential):
Data → Model 1 → [Focus on errors] → Model 2 → [Focus on errors] → Model 3 → Weighted Sum
```

---

## When to Use ✅ / When to Avoid ❌

### ✅ Good For

| Scenario | Why It Works |
|----------|--------------|
| **Need maximum accuracy** | Often achieves best performance on tabular data |
| **High bias models** | Corrects underfitting by building complexity |
| **Tabular/structured data** | Dominates Kaggle competitions on structured data |
| **Feature importance** | Provides robust importance scores |
| **Handling complex patterns** | Captures non-linear relationships |
| **Imbalanced data** | Can weight classes appropriately |

**Examples:**
- Kaggle competitions (XGBoost wins most)
- Fraud detection (needs high accuracy)
- Credit scoring
- Search ranking (LambdaMART)
- Click-through rate prediction

### ❌ Not Good For

| Scenario | Why It Fails |
|----------|--------------|
| **Noisy data** | Can overfit by focusing on noise |
| **Need fast training** | Sequential training is slow |
| **Real-time prediction** | Need to run all models sequentially |
| **Need interpretability** | Complex ensemble hard to explain |
| **Very limited data** | Risk of overfitting |
| **Outliers present** | Overly focuses on outliers |

**When to use instead:**
- Noisy data: Bagging (Random Forest)
- Need speed: Single model or Bagging (parallelizable)
- Need interpretability: Decision Tree, Linear Model
- Limited data: Regularized models, simpler algorithms

---

## Mathematical Foundation

### General Boosting Framework

**Additive Model:**

$$F(x) = \sum_{m=1}^{M}\alpha_m h_m(x)$$

Where:
- $F(x)$ = final strong learner
- $h_m(x)$ = weak learner $m$
- $\alpha_m$ = weight for learner $m$
- $M$ = number of iterations

**Weak Learner:** Model that performs slightly better than random guessing.

**Strong Learner:** Combination of weak learners that performs very well.

### Key Algorithms

We'll cover three main boosting algorithms:

1. **AdaBoost** - Adjusts sample weights
2. **Gradient Boosting** - Fits to residuals
3. **XGBoost** - Optimized gradient boosting

---

## AdaBoost (Adaptive Boosting)

### Algorithm Overview

AdaBoost adjusts sample weights after each iteration, making misclassified samples more important.

**Algorithm Steps:**

1. Initialize sample weights: $w_i = \frac{1}{n}$ for all samples
2. For $m = 1$ to $M$:
   - Train weak learner $h_m$ on weighted data
   - Calculate weighted error: $\epsilon_m = \sum_{i: h_m(x_i) \neq y_i} w_i$
   - Calculate learner weight: $\alpha_m = \frac{1}{2}\log\frac{1-\epsilon_m}{\epsilon_m}$
   - Update sample weights: $w_i \leftarrow w_i \cdot e^{-\alpha_m y_i h_m(x_i)}$
   - Normalize weights
3. Final prediction: $F(x) = \text{sign}\left(\sum_{m=1}^{M}\alpha_m h_m(x)\right)$

### Mathematical Details

**Learner Weight ($\alpha_m$):**
- If $\epsilon_m \to 0$ (perfect): $\alpha_m \to \infty$ (high weight)
- If $\epsilon_m = 0.5$ (random): $\alpha_m = 0$ (ignore)
- If $\epsilon_m > 0.5$ (worse than random): $\alpha_m < 0$ (reverse)

**Sample Weight Update:**
- Correctly classified: $w_i \cdot e^{-\alpha_m}$ (decrease weight)
- Misclassified: $w_i \cdot e^{\alpha_m}$ (increase weight)

### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Generate data
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Base weak learner: shallow tree (stump)
weak_learner = DecisionTreeClassifier(max_depth=1)  # Decision stump

# AdaBoost
ada_model = AdaBoostClassifier(
    base_estimator=weak_learner,
    n_estimators=50,
    learning_rate=1.0,
    algorithm='SAMME',  # Discrete AdaBoost
    random_state=42
)

ada_model.fit(X_train, y_train)

# Evaluate
y_pred = ada_model.predict(X_test)
print(f"AdaBoost Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# Compare with single weak learner
weak_learner.fit(X_train, y_train)
print(f"Single Weak Learner: {accuracy_score(y_test, weak_learner.predict(X_test)):.3f}")

# Visualize learning progress
train_scores = []
test_scores = []

for i, y_pred_train in enumerate(ada_model.staged_predict(X_train)):
    train_scores.append(accuracy_score(y_train, y_pred_train))

for i, y_pred_test in enumerate(ada_model.staged_predict(X_test)):
    test_scores.append(accuracy_score(y_test, y_pred_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_scores) + 1), train_scores, label='Train', linewidth=2)
plt.plot(range(1, len(test_scores) + 1), test_scores, label='Test', linewidth=2)
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('AdaBoost Learning Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### AdaBoost from Scratch

```python
class AdaBoostScratch:
    """AdaBoost implementation from scratch"""

    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.models = []

    def fit(self, X, y):
        """Train AdaBoost"""
        n_samples = len(X)

        # Initialize weights uniformly
        weights = np.ones(n_samples) / n_samples

        for m in range(self.n_estimators):
            # Train weak learner on weighted data
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X, y, sample_weight=weights)

            # Make predictions
            predictions = stump.predict(X)

            # Calculate weighted error
            incorrect = predictions != y
            error = np.sum(weights[incorrect]) / np.sum(weights)

            # Avoid division by zero
            error = np.clip(error, 1e-10, 1 - 1e-10)

            # Calculate learner weight
            alpha = 0.5 * np.log((1 - error) / error)

            # Update sample weights
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)  # Normalize

            # Store model and weight
            self.models.append(stump)
            self.alphas.append(alpha)

            if m % 10 == 0:
                print(f"Iteration {m}: Error = {error:.4f}, Alpha = {alpha:.4f}")

        return self

    def predict(self, X):
        """Make predictions"""
        # Weighted sum of weak learners
        predictions = np.sum([
            alpha * model.predict(X)
            for alpha, model in zip(self.alphas, self.models)
        ], axis=0)

        return np.sign(predictions)

# Usage
y_train_signed = np.where(y_train == 0, -1, 1)
y_test_signed = np.where(y_test == 0, -1, 1)

ada_scratch = AdaBoostScratch(n_estimators=50)
ada_scratch.fit(X_train, y_train_signed)
y_pred_scratch = ada_scratch.predict(X_test)

scratch_acc = accuracy_score(y_test_signed, y_pred_scratch)
print(f"\nFrom Scratch Accuracy: {scratch_acc:.3f}")
```

---

## Gradient Boosting

### Algorithm Overview

Gradient Boosting fits each new model to the **residuals** (errors) of the previous predictions, using gradient descent in function space.

**Key Insight:** Each model predicts the residual error, gradually reducing overall error.

**Algorithm Steps:**

1. Initialize: $F_0(x) = \arg\min_\gamma \sum_{i=1}^n L(y_i, \gamma)$
2. For $m = 1$ to $M$:
   - Compute negative gradient (residuals):
     $$r_i = -\frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)}$$
   - Fit weak learner $h_m$ to residuals: $h_m(x) \approx r$
   - Find optimal step size: $\gamma_m = \arg\min_\gamma \sum_{i=1}^n L(y_i, F_{m-1}(x_i) + \gamma h_m(x_i))$
   - Update: $F_m(x) = F_{m-1}(x) + \nu \gamma_m h_m(x)$
3. Final: $F(x) = F_M(x)$

Where $\nu$ is the learning rate (shrinkage).

### Loss Functions

Different tasks use different loss functions:

**Regression:**
- MSE: $L(y, F) = \frac{1}{2}(y - F)^2$ → residual = $y - F$
- MAE: $L(y, F) = |y - F|$ → residual = $\text{sign}(y - F)$

**Classification:**
- Log loss: $L(y, F) = \log(1 + e^{-yF})$ → residual = $\frac{y}{1 + e^{yF}}$

### Implementation

```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

# Classification
gb_clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,  # Stochastic GB
    random_state=42
)

gb_clf.fit(X_train, y_train)
print(f"Gradient Boosting (Classification): {gb_clf.score(X_test, y_test):.3f}")

# Regression example
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

gb_reg = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb_reg.fit(X_train_r, y_train_r)
y_pred_reg = gb_reg.predict(X_test_r)
print(f"Gradient Boosting (Regression) MSE: {mean_squared_error(y_test_r, y_pred_reg):.2f}")
```

### Visualizing Residual Reduction

```python
def plot_residual_reduction():
    """Show how gradient boosting reduces residuals"""
    # Simple regression example
    X_simple = np.linspace(0, 10, 100).reshape(-1, 1)
    y_simple = np.sin(X_simple).ravel() + np.random.normal(0, 0.1, 100)

    # Train gradient boosting step by step
    gb = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1,
                                   max_depth=3, random_state=42)
    gb.fit(X_simple, y_simple)

    # Plot predictions at different stages
    stages = [1, 5, 10, 50]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for idx, stage in enumerate(stages):
        ax = axes[idx]

        # Predictions at this stage
        y_pred_stage = np.zeros(len(X_simple))
        for i, pred in enumerate(gb.staged_predict(X_simple)):
            y_pred_stage = pred
            if i + 1 == stage:
                break

        # Plot
        ax.scatter(X_simple, y_simple, alpha=0.5, label='Data')
        ax.plot(X_simple, y_pred_stage, 'r-', linewidth=2, label='Prediction')

        # Residuals
        residuals = y_simple - y_pred_stage
        mse = np.mean(residuals**2)

        ax.set_title(f'After {stage} estimators (MSE: {mse:.4f})')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.legend()

    plt.tight_layout()
    plt.savefig('gradient_boosting_residuals.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_residual_reduction()
```

---

## XGBoost (Extreme Gradient Boosting)

### Why XGBoost is Special

XGBoost is an optimized implementation of gradient boosting with:

1. **Regularization:** L1 and L2 penalties to prevent overfitting
2. **Tree pruning:** Max_depth first, then prune back (more efficient)
3. **Built-in cross-validation**
4. **Parallel processing:** Parallelizes tree construction
5. **Handles missing values:** Learns best direction for missing values
6. **Sparsity awareness:** Efficient handling of sparse features
7. **Cache optimization:** Better hardware utilization

### Objective Function

XGBoost minimizes:

$$\text{Obj}^{(t)} = \sum_{i=1}^{n}L(y_i, \hat{y}_i^{(t)}) + \sum_{k=1}^{t}\Omega(f_k)$$

Where:
- $L$ = loss function
- $\Omega(f) = \gamma T + \frac{1}{2}\lambda\sum_{j=1}^{T}w_j^2$ = regularization term
- $T$ = number of leaves, $w_j$ = leaf weights

### Implementation

```python
import xgboost as xgb
from sklearn.metrics import roc_auc_score

# XGBoost Classifier
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,              # Minimum loss reduction
    reg_alpha=0,          # L1 regularization
    reg_lambda=1,         # L2 regularization
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric='auc',
    early_stopping_rounds=10,
    verbose=False
)

# Predictions
y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb):.3f}")
print(f"XGBoost AUC: {roc_auc_score(y_test, y_pred_proba_xgb):.3f}")

# Feature importance
feature_importance = xgb_model.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1][:10]

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), [f'Feature {i}' for i in sorted_idx])
plt.xlabel('Importance')
plt.title('XGBoost Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

### Using Native XGBoost API

```python
# Native API for more control
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'binary:logistic',
    'max_depth': 3,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'auc',
}

evals = [(dtrain, 'train'), (dtest, 'test')]
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=evals,
    early_stopping_rounds=10,
    verbose_eval=10
)

# Predict
y_pred_native = bst.predict(dtest)
y_pred_native_class = (y_pred_native > 0.5).astype(int)
print(f"Native XGBoost Accuracy: {accuracy_score(y_test, y_pred_native_class):.3f}")
```

---

## LightGBM and CatBoost

### LightGBM (Light Gradient Boosting Machine)

**Key Features:**
- **Faster** than XGBoost
- **Leaf-wise growth** (vs level-wise in XGBoost)
- **Gradient-based One-Side Sampling (GOSS)**
- **Exclusive Feature Bundling (EFB)**
- Better for large datasets

```python
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    num_leaves=31,
    random_state=42
)

lgb_model.fit(X_train, y_train)
print(f"LightGBM Accuracy: {lgb_model.score(X_test, y_test):.3f}")
```

### CatBoost

**Key Features:**
- **Handles categorical features** natively
- **Ordered boosting** (reduces overfitting)
- **Symmetric trees**
- No need for extensive hyperparameter tuning

```python
from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=3,
    verbose=False,
    random_state=42
)

cat_model.fit(X_train, y_train)
print(f"CatBoost Accuracy: {cat_model.score(X_test, y_test):.3f}")
```

### Comparison

```python
# Compare all boosting methods
models = {
    'AdaBoost': AdaBoostClassifier(n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, use_label_encoder=False),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100),
    'CatBoost': CatBoostClassifier(iterations=100, verbose=False)
}

import time

results = {}
for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    accuracy = model.score(X_test, y_test)
    results[name] = {'Accuracy': accuracy, 'Train Time': train_time}

    print(f"{name:20s} - Accuracy: {accuracy:.3f}, Time: {train_time:.2f}s")
```

---

## Hyperparameters

### Critical Parameters

| Parameter | What It Does | Typical Range | Tips |
|-----------|--------------|---------------|------|
| `n_estimators` | Number of boosting rounds | 50-1000 | More = better but slower; use early stopping |
| `learning_rate` | Shrinkage factor | 0.01-0.3 | Lower = need more trees but better performance |
| `max_depth` | Maximum tree depth | 3-10 | Higher = more complex but overfits |
| `subsample` | Fraction of samples per tree | 0.5-1.0 | <1.0 = stochastic boosting, reduces overfitting |
| `colsample_bytree` | Fraction of features per tree | 0.5-1.0 | Adds randomness, reduces overfitting |
| `min_child_weight` | Minimum sum of weights in leaf | 1-10 | Higher = more conservative |
| `gamma` | Minimum loss reduction for split | 0-5 | Higher = more conservative |
| `reg_alpha` | L1 regularization | 0-1 | Feature selection |
| `reg_lambda` | L2 regularization | 0-10 | Smoothing |

### Learning Rate vs N_Estimators Trade-off

```python
# Lower learning rate + more estimators = better performance
configs = [
    {'learning_rate': 0.3, 'n_estimators': 50},
    {'learning_rate': 0.1, 'n_estimators': 150},
    {'learning_rate': 0.01, 'n_estimators': 500},
]

for config in configs:
    model = xgb.XGBClassifier(**config, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"LR={config['learning_rate']}, N={config['n_estimators']}: {score:.3f}")
```

### Grid Search Example

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

grid_search = GridSearchCV(
    xgb.XGBClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
print(f"Test score: {grid_search.score(X_test, y_test):.3f}")
```

---

## Visualization

### Training Progress

```python
# Plot training and validation curves
xgb_model_eval = xgb.XGBClassifier(n_estimators=200, learning_rate=0.1)

eval_set = [(X_train, y_train), (X_test, y_test)]
xgb_model_eval.fit(
    X_train, y_train,
    eval_set=eval_set,
    eval_metric='error',
    verbose=False
)

results = xgb_model_eval.evals_result()

plt.figure(figsize=(10, 6))
plt.plot(results['validation_0']['error'], label='Train')
plt.plot(results['validation_1']['error'], label='Test')
plt.xlabel('Number of Trees')
plt.ylabel('Error Rate')
plt.title('XGBoost Learning Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Tree Visualization

```python
# Visualize a single tree
xgb.plot_tree(xgb_model, num_trees=0)
plt.gcf().set_size_inches(20, 10)
plt.show()

# Feature importance plot
xgb.plot_importance(xgb_model, max_num_features=10)
plt.gcf().set_size_inches(10, 6)
plt.show()
```

---

## Common Pitfalls

### 1. Overfitting with Too Many Trees

!!! warning "Model Memorizes Training Data"
    **Problem:** Using 1000+ trees with high learning rate leads to overfitting.

    **Solution:** Use early stopping with validation set.

    ```python
    xgb_model = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.1)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=10,  # Stop if no improvement for 10 rounds
        verbose=False
    )
    print(f"Best iteration: {xgb_model.best_iteration}")
    ```

### 2. Not Tuning Learning Rate

!!! warning "Using Default Learning Rate"
    **Problem:** Default learning rate might be too high or too low.

    **Rule of thumb:** Lower learning rate with more trees = better performance

    ```python
    # Start with moderate LR, then reduce
    # High LR + Few trees: Fast but suboptimal
    fast_model = xgb.XGBClassifier(n_estimators=50, learning_rate=0.3)

    # Low LR + Many trees: Slow but better
    accurate_model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.01)
    ```

### 3. Ignoring Regularization

!!! warning "Not Using Regularization Parameters"
    **Problem:** Boosting can overfit, especially with deep trees.

    **Solution:** Use regularization parameters.

    ```python
    xgb_regularized = xgb.XGBClassifier(
        max_depth=3,              # Limit tree depth
        min_child_weight=5,       # Minimum samples per leaf
        gamma=0.1,                # Minimum loss reduction
        reg_alpha=0.1,            # L1 regularization
        reg_lambda=1.0,           # L2 regularization
        subsample=0.8,            # Sample 80% of data
        colsample_bytree=0.8      # Use 80% of features
    )
    ```

### 4. Sensitive to Noisy Data and Outliers

!!! warning "Overfocusing on Noisy Samples"
    **Problem:** Boosting keeps focusing on hard-to-classify samples, which might be noise.

    **Solution:**
    - Clean data first
    - Use robust loss functions
    - Try bagging instead (more robust)

    ```python
    # For regression with outliers, use MAE instead of MSE
    xgb_robust = xgb.XGBRegressor(objective='reg:pseudohubererror')
    ```

### 5. Not Scaling Features (for some algorithms)

!!! warning "Feature Scaling for AdaBoost with Linear Models"
    **Problem:** If base learner needs scaling (e.g., linear models), AdaBoost will struggle.

    **Solution:** Scale features or use tree-based weak learners (don't need scaling).

    ```python
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # If using linear base learners
    from sklearn.linear_model import LogisticRegression

    ada_scaled = Pipeline([
        ('scaler', StandardScaler()),
        ('adaboost', AdaBoostClassifier(
            base_estimator=LogisticRegression()
        ))
    ])
    ```

### 6. Class Imbalance

!!! warning "Biased Toward Majority Class"
    **Problem:** Boosting might ignore minority class.

    **Solution:** Use `scale_pos_weight` parameter or class weights.

    ```python
    # Calculate ratio
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos

    xgb_balanced = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight  # Balance classes
    )
    ```

---

## Practice Problems

### 🟢 Beginner

!!! example "Problem 1: AdaBoost with Different Weak Learners"
    **Goal:** Understand weak learner impact

    **Tasks:**
    1. Train AdaBoost with different max_depth: 1, 2, 3, 5
    2. Compare accuracy and training time
    3. Plot learning curves for each
    4. Which depth is optimal?

!!! example "Problem 2: Learning Rate Effect"
    **Goal:** See learning rate vs n_estimators trade-off

    **Tasks:**
    1. Train XGBoost with (LR=0.3, n=50), (LR=0.1, n=150), (LR=0.01, n=500)
    2. Compare accuracy
    3. Plot convergence speed
    4. Which configuration is best?

### 🟡 Intermediate

!!! example "Problem 3: Early Stopping"
    **Goal:** Prevent overfitting with early stopping

    **Tasks:**
    1. Train XGBoost with 1000 trees
    2. Use early_stopping_rounds=10
    3. Compare with model using all 1000 trees
    4. Plot train vs validation curves
    5. Find optimal stopping point

!!! example "Problem 4: Feature Engineering for Boosting"
    **Goal:** Improve boosting with better features

    **Tasks:**
    1. Load raw dataset
    2. Create polynomial features
    3. Create interaction features
    4. Train XGBoost on different feature sets
    5. Compare feature importance

### 🔴 Advanced

!!! example "Problem 5: Hyperparameter Optimization"
    **Goal:** Find optimal XGBoost configuration

    **Tasks:**
    1. Use RandomizedSearchCV or Optuna
    2. Tune 8+ parameters simultaneously
    3. Use 5-fold CV
    4. Compare with default parameters
    5. Analyze which parameters matter most

!!! example "Problem 6: Custom Loss Function"
    **Goal:** Implement custom objective for XGBoost

    **Tasks:**
    1. Define custom loss function (e.g., focal loss)
    2. Implement gradient and hessian
    3. Train XGBoost with custom objective
    4. Compare with standard log loss
    5. Evaluate on imbalanced dataset

---

## Real-World Applications

### Industry Use Cases

| Domain | Application | Algorithm | Why Boosting |
|--------|-------------|-----------|-------------|
| **Finance** | Credit scoring | XGBoost | Need highest accuracy for risk |
| **E-commerce** | Click prediction | LightGBM | Fast inference on large scale |
| **Healthcare** | Disease prediction | CatBoost | Handles categorical data well |
| **Fraud Detection** | Transaction monitoring | XGBoost | Detects complex patterns |
| **Search Ranking** | Result ordering | LambdaMART | Optimizes ranking metrics |

### Competition Winners

**Kaggle competitions won by boosting:**
- Netflix Prize: Gradient Boosting ensemble
- Microsoft Malware: XGBoost primary model
- Home Credit Default: LightGBM + CatBoost
- Porto Seguro: XGBoost with custom features
- Most tabular competitions: XGBoost/LightGBM

---

## Complexity Analysis

### Time Complexity

| Algorithm | Training | Prediction |
|-----------|----------|------------|
| **AdaBoost** | $O(M \cdot T \cdot n \log n)$ | $O(M \cdot \log n)$ |
| **Gradient Boosting** | $O(M \cdot T \cdot n \log n \cdot m)$ | $O(M \cdot \log n)$ |
| **XGBoost** | Similar but optimized | Similar |
| **LightGBM** | Faster (GOSS + EFB) | Similar |

Where: $M$ = n_estimators, $T$ = tree depth, $n$ = samples, $m$ = features

**Note:** Sequential training cannot be parallelized across trees (only within tree construction).

### Space Complexity

- **Storage:** $O(M \cdot T)$ for tree structures
- **Memory:** Can be high for large datasets (XGBoost caches)

---

## Related Topics

- [Bagging](bagging.md) - Parallel ensemble for reducing variance
- [Stacking](stacking.md) - Meta-learning ensemble
- [Decision Trees](../classification/decision-trees.md) - Base learner for boosting
- [Gradient Descent](../../optimization/gradient-descent.md) - Optimization algorithm
- [Ensemble Methods Overview](index.md) - Compare all approaches

---

## References

1. **Original Papers:**
   - AdaBoost: Freund & Schapire (1997) "A Decision-Theoretic Generalization of On-Line Learning"
   - Gradient Boosting: Friedman (2001) "Greedy Function Approximation: A Gradient Boosting Machine"
   - XGBoost: Chen & Guestrin (2016) "XGBoost: A Scalable Tree Boosting System"

2. **Documentation:**
   - [Scikit-learn Gradient Boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
   - [XGBoost Official Docs](https://xgboost.readthedocs.io/)
   - [LightGBM Documentation](https://lightgbm.readthedocs.io/)
   - [CatBoost Documentation](https://catboost.ai/docs/)

3. **Books:**
   - *The Elements of Statistical Learning* Chapter 10
   - *Hands-On Machine Learning* Chapter 7

4. **Videos:**
   - [StatQuest: Gradient Boosting](https://www.youtube.com/watch?v=3CC4N4z3GJc)
   - [XGBoost Tutorial Series](https://www.youtube.com/watch?v=OtD8wVaFm6E)

---

**Next Steps:**
- Master [XGBoost](xgboost.md) - Industry standard boosting
- Learn [Stacking](stacking.md) - Combine boosting with other models
- Explore [Feature Engineering](../../feature-engineering/index.md) - Boost your boosting!

**Ready to dominate Kaggle competitions?** Practice with the problems above! 🏆

# Overfitting & Underfitting

**The two most common problems in machine learning and how to fix them.** Learn to recognize when your model is too simple or too complex, and apply proven solutions.

**Difficulty:** 🟢 Beginner | **Time:** 2-3 hours | **Prerequisites:** [What is ML?](what-is-ml.md), [Bias-Variance Tradeoff](bias-variance-tradeoff.md)

---

## Quick Reference

| Property | Underfitting | Just Right | Overfitting |
|----------|--------------|------------|-------------|
| **Aka** | High Bias | Optimal | High Variance |
| **Model Complexity** | Too simple | Balanced | Too complex |
| **Training Error** | High | Low | Very low |
| **Test Error** | High | Low | High |
| **Gap** | Small | Small | Large |
| **Problem** | Misses patterns | Captures patterns | Memorizes noise |
| **Visual** | Straight line for curve | Smooth fit | Wiggly, through every point |

---

## Complete Guide

=== "📖 Overview"
    ## What are Overfitting and Underfitting?

    **Underfitting:** Model is too simple to capture the underlying pattern in the data.

    **Overfitting:** Model is too complex and memorizes training data, including noise.

    **Goal:** Find the "Goldilocks" model - not too simple, not too complex, just right!

    ---

    ## Visual Understanding

    ```
    Underfitting          Just Right         Overfitting
    (Too Simple)          (Perfect)          (Too Complex)

    ●  ●  ●               ●  ●  ●            ●  ●  ●
     ●  ●  ●     vs       ●  ●  ●     vs      ●  ●  ●
    ──────────            ╱─────╲            ╱╲╱╲╱╲╱╲

    Misses pattern        Captures           Memorizes
                         pattern            noise
    ```

    ---

    ## Underfitting (High Bias)

    **Definition:** Model too simple to learn the underlying structure.

    **Characteristics:**
    - **Poor training performance:** High training error
    - **Poor test performance:** High test error
    - **Small gap:** Train and test errors similar (both bad)
    - **Consistent failure:** Systematic errors

    **Example:** Using a straight line to predict a curved relationship

    **Analogy:** Studying only 1 hour for a complex exam - you won't do well on practice tests OR the real exam.

    ---

    ### Signs of Underfitting

    1. **High training error** (> acceptable threshold)
    2. **Training error ≈ Test error** (both high)
    3. **Model performs poorly on all data**
    4. **Learning curves plateau early at high error**
    5. **Predictions are systematically wrong**

    ---

    ### Causes of Underfitting

    - Model too simple (linear for non-linear data)
    - Insufficient features
    - Over-regularization (too much penalty)
    - Insufficient training time/iterations
    - Data quality issues (but model can't express this)

    ---

    ## Overfitting (High Variance)

    **Definition:** Model too complex, memorizes training data including noise.

    **Characteristics:**
    - **Excellent training performance:** Very low training error
    - **Poor test performance:** High test error
    - **Large gap:** Huge difference between train and test error
    - **Noise fitting:** Model captures random fluctuations

    **Example:** Using a 15th-degree polynomial when data is actually linear

    **Analogy:** Memorizing practice test answers without understanding - you ace practice tests but fail real tests with different questions.

    ---

    ### Signs of Overfitting

    1. **Very low training error** (near 0%)
    2. **High test error** (much worse than training)
    3. **Large gap** between train and test performance
    4. **Model performs well only on training data**
    5. **Complex model with many parameters**
    6. **Learning curves show growing gap**

    ---

    ### Causes of Overfitting

    - Model too complex (many parameters)
    - Too few training examples
    - Training too long (especially neural networks)
    - Too many features relative to samples
    - No regularization
    - Noise in training data

    ---

    ## The Spectrum

    ```
    Underfitting          Sweet Spot          Overfitting
    ◄─────────────────────────○─────────────────────────►
    Too Simple           Just Right           Too Complex

    High Bias                                High Variance
    Can't learn                              Memorizes

    Solution:                                Solution:
    ↓ Increase complexity                    ↓ Reduce complexity
    ↓ Add features                           ↓ Get more data
    ↓ Reduce regularization                  ↓ Add regularization
    ↓ Train longer                           ↓ Early stopping
    ```

=== "🧮 Theory & Math"
    ## Mathematical Framework

    ### Expected Error Decomposition

    $$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

    **Underfitting:** High bias dominates
    **Overfitting:** High variance dominates

    ---

    ## Learning Curves Theory

    **Training Error Curve:**
    - Starts low (easy to fit small dataset)
    - Increases as more data added (harder to fit everything)
    - Plateaus at model's capability

    **Validation Error Curve:**
    - Starts high (few samples, can't generalize)
    - Decreases as more data added (better generalization)
    - Plateaus at model's generalization ability

    ---

    ### Underfitting Learning Curves

    ```
    Error
      ↑
      |
      |  Training ───────
      |           Validation
      |  ─────────
      |
      |  Both errors high and close
      └──────────────────────────→
                Dataset Size
    ```

    **Interpretation:** Both curves plateau at high error. More data won't help - need more complex model!

    ---

    ### Overfitting Learning Curves

    ```
    Error
      ↑
      |        Validation
      |  ─────╱‾‾‾‾‾‾‾‾‾
      |      ╱
      |     ╱
      |  ──╱── Training (very low)
      |
      |  Large gap between curves
      └──────────────────────────→
                Dataset Size
    ```

    **Interpretation:** Large gap between curves. Training error very low, validation high. More data helps!

    ---

    ### Just Right Learning Curves

    ```
    Error
      ↑
      |        Validation
      |  ─────╱────────
      |     ╱
      |  ──╱──── Training
      |
      |  Small gap, both errors low
      └──────────────────────────→
                Dataset Size
    ```

    **Interpretation:** Small gap, both errors acceptable. Model generalizes well!

    ---

    ## Model Complexity and Error

    $$\text{Training Error}(C) = \text{decreasing function of complexity}$$
    $$\text{Test Error}(C) = \text{U-shaped function of complexity}$$

    Where $C$ is model complexity.

    **Optimal Complexity:** Minimizes test error

    ```
    Error
      ↑
      |           Test Error
      |          ╱‾‾‾‾╲
      |        ╱        ╲
      |      ╱            ╲
      |    ╱                ╲
      |  ╱  Training Error    ╲
      | ╱                      ╲___
      |╱___________________________
      |        ↑
      |     Optimal
      └──────────────────────────→
          Model Complexity
    ```

=== "💻 Implementation"
    ## Detecting Overfitting and Underfitting

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split, learning_curve
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    # Generate synthetic data
    np.random.seed(42)
    X = np.sort(np.random.uniform(0, 10, 200))
    y_true = np.sin(X)
    y = y_true + np.random.normal(0, 0.2, 200)

    X = X.reshape(-1, 1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    def evaluate_model(degree, X_train, X_test, y_train, y_test):
        """Train and evaluate polynomial model"""
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        # Train model
        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        # Predictions
        y_train_pred = model.predict(X_train_poly)
        y_test_pred = model.predict(X_test_poly)

        # Errors
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # Diagnosis
        gap = test_mse - train_mse
        if train_mse > 0.1 and test_mse > 0.1:
            diagnosis = "UNDERFITTING (High Bias)"
        elif gap > 0.3:
            diagnosis = "OVERFITTING (High Variance)"
        else:
            diagnosis = "GOOD FIT"

        return {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'gap': gap,
            'diagnosis': diagnosis,
            'model': model,
            'poly': poly
        }

    # Test different complexities
    degrees = [1, 3, 9, 15]
    results = {}

    print("Degree | Train MSE | Test MSE | Gap    | Diagnosis")
    print("-------|-----------|----------|--------|------------------")

    for degree in degrees:
        results[degree] = evaluate_model(
            degree, X_train, X_test, y_train, y_test
        )
        r = results[degree]
        print(f"  {degree:2d}   | {r['train_mse']:9.4f} | "
              f"{r['test_mse']:8.4f} | {r['gap']:6.3f} | {r['diagnosis']}")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    X_plot = np.linspace(0, 10, 300).reshape(-1, 1)

    for idx, degree in enumerate(degrees):
        r = results[degree]
        X_plot_poly = r['poly'].transform(X_plot)
        y_plot = r['model'].predict(X_plot_poly)

        axes[idx].scatter(X_train, y_train, alpha=0.6, label='Training')
        axes[idx].scatter(X_test, y_test, alpha=0.6, label='Test')
        axes[idx].plot(X_plot, y_plot, 'r-', linewidth=2, label='Model')
        axes[idx].set_ylim(-2, 2)
        axes[idx].legend()
        axes[idx].set_title(f'{r["diagnosis"]}\nDegree {degree} | '
                           f'Train: {r["train_mse"]:.3f}, '
                           f'Test: {r["test_mse"]:.3f}')

    plt.tight_layout()
    plt.show()
    ```

    ---

    ## Learning Curves for Diagnosis

    ```python
    from sklearn.model_selection import learning_curve

    def plot_learning_curves(estimator, X, y, title):
        """
        Plot learning curves to diagnose overfitting/underfitting
        """
        train_sizes, train_scores, val_scores = learning_curve(
            estimator, X, y,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        # Convert to MSE (positive)
        train_scores = -train_scores
        val_scores = -val_scores

        # Calculate statistics
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue',
                label='Training error', linewidth=2, markersize=8)
        plt.plot(train_sizes, val_mean, 'o-', color='red',
                label='Validation error', linewidth=2, markersize=8)

        plt.fill_between(train_sizes,
                         train_mean - train_std,
                         train_mean + train_std,
                         alpha=0.15, color='blue')
        plt.fill_between(train_sizes,
                         val_mean - val_std,
                         val_mean + val_std,
                         alpha=0.15, color='red')

        # Diagnose
        final_gap = val_mean[-1] - train_mean[-1]
        if train_mean[-1] > 0.15:
            diagnosis = "UNDERFITTING: Both errors high"
            solution = "→ Use more complex model, add features"
            color = 'orange'
        elif final_gap > 0.15:
            diagnosis = "OVERFITTING: Large gap"
            solution = "→ Get more data, regularize, simplify model"
            color = 'red'
        else:
            diagnosis = "GOOD FIT: Low error, small gap"
            solution = "→ Model is working well!"
            color = 'green'

        plt.text(0.5, 0.95, diagnosis,
                transform=plt.gca().transAxes,
                fontsize=12, fontweight='bold',
                color=color, ha='center')
        plt.text(0.5, 0.88, solution,
                transform=plt.gca().transAxes,
                fontsize=10, color=color, ha='center')

        plt.xlabel('Training Set Size', fontsize=12)
        plt.ylabel('Mean Squared Error', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=11)
        plt.grid(alpha=0.3)
        plt.show()

    # Test with different models
    # Underfitting example (too simple)
    poly_1 = PolynomialFeatures(degree=1)
    X_poly_1 = poly_1.fit_transform(X)
    plot_learning_curves(LinearRegression(), X_poly_1, y,
                        'Underfitting: Linear Model')

    # Good fit example
    poly_3 = PolynomialFeatures(degree=3)
    X_poly_3 = poly_3.fit_transform(X)
    from sklearn.linear_model import Ridge
    plot_learning_curves(Ridge(alpha=1.0), X_poly_3, y,
                        'Good Fit: Degree 3 with Regularization')

    # Overfitting example (too complex)
    poly_15 = PolynomialFeatures(degree=15)
    X_poly_15 = poly_15.fit_transform(X)
    plot_learning_curves(LinearRegression(), X_poly_15, y,
                        'Overfitting: Degree 15 Polynomial')
    ```

    ---

    ## Fixing Overfitting with Regularization

    ```python
    from sklearn.linear_model import Ridge, Lasso

    # Data with many features (prone to overfitting)
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=100, n_features=50,
                          n_informative=10, noise=20,
                          random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    models = {
        'No Regularization': LinearRegression(),
        'Ridge (L2)': Ridge(alpha=1.0),
        'Lasso (L1)': Lasso(alpha=0.1)
    }

    print("\nModel                | Train R² | Test R² | Overfitting?")
    print("---------------------|----------|---------|-------------")

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_r2 = model.score(X_train, y_train)
        test_r2 = model.score(X_test, y_test)
        gap = train_r2 - test_r2

        overfit = "YES" if gap > 0.2 else "NO"
        print(f"{name:20s} | {train_r2:8.3f} | {test_r2:7.3f} | "
              f"{overfit:12s}")

    # Regularization strength comparison
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    train_scores = []
    test_scores = []

    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        train_scores.append(model.score(X_train, y_train))
        test_scores.append(model.score(X_test, y_test))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.semilogx(alphas, train_scores, 'o-', label='Training R²',
                linewidth=2)
    plt.semilogx(alphas, test_scores, 'o-', label='Test R²',
                linewidth=2)
    plt.xlabel('Regularization Strength (alpha)', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.title('Regularization Effect on Overfitting', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.show()
    ```

    ---

    ## Early Stopping (Neural Networks)

    ```python
    from sklearn.neural_network import MLPRegressor

    # Train with early stopping
    model = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        max_iter=1000,
        early_stopping=True,  # Stop when validation error increases
        validation_fraction=0.2,
        n_iter_no_change=10,  # Patience
        random_state=42
    )

    model.fit(X_train, y_train)

    print(f"\nEarly stopping at iteration: {model.n_iter_}")
    print(f"Best validation score: {model.best_validation_score_:.3f}")
    ```

=== "📊 Visualization"
    ## Diagnostic Flowchart

    ```mermaid
    flowchart TD
        Start[Train Model] --> Eval[Evaluate on Train & Test]
        Eval --> TrainErr{Training Error?}

        TrainErr -->|High| TestErrHigh{Test Error?}
        TrainErr -->|Low| TestErrLow{Test Error?}

        TestErrHigh -->|High| Under[UNDERFITTING<br/>High Bias]
        TestErrHigh -->|Low| Impossible[Impossible!<br/>Check your code]

        TestErrLow -->|High| Over[OVERFITTING<br/>High Variance]
        TestErrLow -->|Low| Good[GOOD FIT!]

        Under --> UnderSol[Solutions:<br/>• More complex model<br/>• More features<br/>• Less regularization]
        Over --> OverSol[Solutions:<br/>• More data<br/>• Regularization<br/>• Simpler model]
        Good --> Deploy[Deploy model!]

        style Start fill:#e1f5ff
        style Under fill:#ffcccc
        style Over fill:#ffffcc
        style Good fill:#ccffcc
    ```

    ---

    ## Visual Comparison

    ```
    Dataset:  ● ● ● ● ● ● ● ●

    Underfitting:
    ──────────────────────
    (Straight line, misses curve)
    Train Error: High ❌
    Test Error: High ❌

    Just Right:
    ╱──────────╲
    (Smooth curve, captures pattern)
    Train Error: Low ✓
    Test Error: Low ✓

    Overfitting:
    ╱╲╱╲╱╲╱╲╱╲╱╲
    (Wiggly, through every point)
    Train Error: Very Low ✓
    Test Error: High ❌
    ```

=== "🎯 Practice"
    ## Beginner Problems

    **Problem 1: Diagnose the Issue**

    Given these results, identify the problem:

    1. Train accuracy: 95%, Test accuracy: 94%
    2. Train accuracy: 55%, Test accuracy: 54%
    3. Train accuracy: 99%, Test accuracy: 65%
    4. Train accuracy: 85%, Test accuracy: 84%

    **Answers:**
    1. Good fit (high accuracy, small gap)
    2. Underfitting (both accuracies low)
    3. Overfitting (large gap)
    4. Good fit

    ---

    **Problem 2: Plot Learning Curves**

    Load the Iris dataset:
    - Train a decision tree with max_depth=2 (simple)
    - Train a decision tree with max_depth=20 (complex)
    - Plot learning curves for both
    - Identify which overfits

    ---

    ## Intermediate Problems

    **Problem 3: Fix Overfitting**

    Given an overfitting model:
    - Try 3 different regularization strengths
    - Implement early stopping
    - Add dropout (if neural network)
    - Compare results

    ---

    **Problem 4: Polynomial Degree Selection**

    For a regression problem:
    - Try polynomial degrees 1-20
    - Plot train/test error vs degree
    - Find optimal degree using cross-validation

    ---

    ## Advanced Problems

    **Problem 5: Ensemble to Reduce Overfitting**

    Demonstrate variance reduction through ensembles:
    - Train 100 decision trees (each overfits)
    - Show individual predictions (high variance)
    - Show ensemble average (reduced variance)
    - Compare with single tree

    ---

    **Problem 6: Synthetic Data Experiment**

    Create synthetic data with known function:
    - Generate y = f(x) + noise
    - Intentionally create underfitting model
    - Intentionally create overfitting model
    - Show learning curves for both
    - Find optimal complexity

---

## Solutions Summary Table

### For Underfitting (High Bias)

| Solution | Description | When to Use | Example |
|----------|-------------|-------------|---------|
| **More Complex Model** | Use model with more capacity | Simple model failing | Linear → Polynomial |
| **Add Features** | Include more informative variables | Missing key information | Add interaction terms |
| **Reduce Regularization** | Decrease penalty strength | Over-regularized | alpha: 10 → 0.1 |
| **Train Longer** | More iterations/epochs | Model hasn't converged | epochs: 10 → 100 |
| **Remove Feature Selection** | Use all available features | Too few features | Don't drop important features |

---

### For Overfitting (High Variance)

| Solution | Description | When to Use | Example |
|----------|-------------|-------------|---------|
| **More Training Data** | Collect additional samples | Small dataset | 100 → 10,000 samples |
| **Regularization** | Add L1/L2 penalty | Model too complex | Ridge, Lasso |
| **Dropout** | Randomly drop neurons | Neural networks | dropout=0.5 |
| **Early Stopping** | Stop training before overfitting | Iterative training | Monitor validation error |
| **Reduce Model Complexity** | Fewer parameters | Too many parameters | max_depth: 20 → 5 |
| **Feature Selection** | Remove irrelevant features | Many features | SelectKBest, RFE |
| **Data Augmentation** | Generate more training data | Images, text | Rotation, cropping |
| **Ensemble Methods** | Combine multiple models | Single model overfits | Random Forest, Bagging |
| **Cross-Validation** | Use k-fold validation | Better evaluation | 5-fold CV |

---

## Detection Methods

| Method | What It Shows | How to Use |
|--------|---------------|------------|
| **Train/Test Split** | Basic performance gap | Compare train vs test error |
| **Learning Curves** | Performance vs dataset size | Plot error as function of training size |
| **Validation Curves** | Performance vs complexity | Plot error as function of hyperparameter |
| **Cross-Validation** | Robust error estimates | Use k-fold to avoid lucky splits |
| **Residual Analysis** | Pattern in errors | Plot residuals, check randomness |

---

## Code Snippets for Quick Diagnosis

```python
# Quick diagnosis function
def diagnose_model(model, X_train, X_test, y_train, y_test):
    """Diagnose overfitting/underfitting"""
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    gap = train_score - test_score

    print(f"Training Score: {train_score:.3f}")
    print(f"Test Score: {test_score:.3f}")
    print(f"Gap: {gap:.3f}")

    if train_score < 0.7:
        print("\n⚠️ UNDERFITTING: Low training score")
        print("Solutions: More complex model, add features")
    elif gap > 0.1:
        print("\n⚠️ OVERFITTING: Large train-test gap")
        print("Solutions: Regularization, more data, simpler model")
    else:
        print("\n✅ GOOD FIT: Model generalizes well!")

    return train_score, test_score, gap
```

---

## Interview Preparation

### Common Questions

**Q1: What's the difference between overfitting and underfitting?**

**Strong Answer:** "Underfitting is when your model is too simple to capture the underlying pattern - high training error and high test error. Like using a straight line to fit a curve. Overfitting is when your model is too complex and memorizes training data including noise - very low training error but high test error. Like using a 15th-degree polynomial for linear data. The key diagnostic is the gap: small gap with high errors = underfitting, large gap = overfitting."

**Q2: How do you detect overfitting?**

**Strong Answer:** "Several ways: (1) Compare train vs test error - large gap indicates overfitting. (2) Plot learning curves - if training error is very low but validation error is high and they diverge, it's overfitting. (3) Check model complexity - too many parameters relative to data suggests risk. (4) Use cross-validation - consistent poor performance on different folds indicates overfitting. In practice, I start by evaluating on a held-out test set."

**Q3: How do you fix overfitting?**

**Strong Answer:** "Multiple approaches: (1) Get more training data - most effective but not always possible. (2) Add regularization - L1/L2 penalties constrain model complexity. (3) Reduce model complexity - fewer parameters, shallower trees, smaller networks. (4) Use dropout for neural networks. (5) Early stopping - stop training before memorizing. (6) Ensemble methods like Random Forest - averaging reduces variance. Choose based on your specific situation and constraints."

**Q4: Why is more data helpful for overfitting?**

**Strong Answer:** "More data makes it harder to memorize - the model must learn genuine patterns that hold across many examples rather than noise. Mathematically, more data reduces variance because the model's predictions become more stable. However, more data only helps with high variance (overfitting), not high bias (underfitting). If your model is too simple, more data won't help - you need a more complex model first."

**Q5: What are learning curves and how do you interpret them?**

**Strong Answer:** "Learning curves plot training and validation error vs training set size. They diagnose: (1) Underfitting - both curves plateau at high error, close together. More data won't help, need more complex model. (2) Overfitting - large gap between curves, training error very low. More data would help. (3) Good fit - both errors low, small gap. In practice, I plot these early in a project to guide whether I need more data or different model complexity."

**Q6: When would you prefer a slightly underfit model over a perfectly fit one?**

**Strong Answer:** "When interpretability or speed matters more than perfect accuracy. A simpler model is easier to understand, explain to stakeholders, and deploy. For example, in regulated industries like healthcare or finance, a slightly less accurate but interpretable linear model might be preferred over a black-box neural network. Also, simpler models train and predict faster, important for real-time systems. The tradeoff depends on your specific requirements."

---

## Related Topics

- [Bias-Variance Tradeoff](bias-variance-tradeoff.md) - Theoretical foundation
- [Regularization](../optimization/regularization.md) - Techniques to prevent overfitting
- [Cross-Validation](../evaluation/cross-validation.md) - Robust evaluation
- [Ensemble Methods](../supervised-learning/ensemble/) - Reduce variance
- [Feature Selection](../feature-engineering/feature-selection.md) - Reduce overfitting

---

## References

1. **Andrew Ng's ML Course:** [Coursera](https://www.coursera.org/learn/machine-learning) - Lectures on overfitting
2. **Book:** *Hands-On Machine Learning* by Aurélien Géron - Chapter 4
3. **Scikit-learn:** [Learning Curves Documentation](https://scikit-learn.org/stable/modules/learning_curve.html)
4. **StatQuest:** [Overfitting Video](https://www.youtube.com/watch?v=EuBBz3bI-aA)
5. **Google's ML Course:** [Overfitting Module](https://developers.google.com/machine-learning/crash-course/regularization-for-simplicity)
6. **Book:** *The Elements of Statistical Learning* - Chapter 7

---

**Next Steps:**

- Master [Regularization](../optimization/regularization.md) techniques
- Learn [Cross-Validation](../evaluation/cross-validation.md) for robust evaluation
- Study [Ensemble Methods](../supervised-learning/ensemble/) to reduce variance
- Practice diagnosing real models with learning curves

**Remember:** Almost every ML project involves fighting overfitting - master this and you'll build better models!

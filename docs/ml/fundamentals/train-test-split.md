# Train/Test Split

**The fundamental practice for evaluating machine learning models.** Learn how to properly split data to measure model performance and avoid common pitfalls.

**Difficulty:** 🟢 Beginner | **Time:** 2-3 hours | **Prerequisites:** [What is ML?](what-is-ml.md)

---

## Quick Reference

| Property | Value |
|----------|-------|
| **Purpose** | Evaluate model generalization to unseen data |
| **Standard Split** | 70-80% train, 20-30% test |
| **Three-Way Split** | 60-70% train, 15-20% validation, 15-20% test |
| **Golden Rule** | Never train on test data! |
| **When to Split** | Before any preprocessing or feature engineering |
| **Common Ratios** | Small data: 60/20/20, Large data: 98/1/1 |
| **Key Technique** | Stratification for imbalanced data |
| **Advanced** | Cross-validation for robust evaluation |

---

## Complete Guide

=== "📖 Overview"
    ## Why Split Data?

    **Core Problem:** If you evaluate a model on the same data it was trained on, you're testing memorization, not learning.

    **Goal:** Measure how well the model generalizes to new, unseen data.

    **Analogy:** You don't study using the same questions that will be on the exam - you practice with different questions to test true understanding.

    ---

    ## The Three Datasets

    ```mermaid
    graph LR
        A[Complete Dataset] --> B[Training Set<br/>60-80%]
        A --> C[Validation Set<br/>10-20%]
        A --> D[Test Set<br/>10-20%]

        B --> E[Learn Patterns]
        C --> F[Tune Hyperparameters]
        D --> G[Final Evaluation]

        style A fill:#e1f5ff
        style B fill:#ccffcc
        style C fill:#ffffcc
        style D fill:#ffcccc
    ```

    ---

    ### 1. Training Set (60-80%)

    **Purpose:** Learn patterns, fit model parameters.

    **Usage:**
    - Train model on this data
    - Model sees these examples during learning
    - Adjust weights/parameters to minimize error

    **Analogy:** Homework problems you practice with

    ---

    ### 2. Validation Set (10-20%)

    **Purpose:** Tune hyperparameters, select best model.

    **Usage:**
    - Evaluate different model configurations
    - Choose learning rate, regularization strength, etc.
    - Select between different algorithms
    - Prevent overfitting to training data

    **Analogy:** Practice tests to see what you need to study more

    **Critical:** Can use validation set multiple times during development

    ---

    ### 3. Test Set (10-20%)

    **Purpose:** Final, unbiased evaluation of model performance.

    **Usage:**
    - Use only once at the very end
    - Reports final model performance
    - Simulates real-world deployment
    - Never tune based on test results!

    **Analogy:** The actual exam you only take once

    **Critical:** Test set must remain "unseen" throughout development

    ---

    ## Critical Rules

    !!! danger "Golden Rules of Data Splitting"
        1. **Never train on test data** - This is data leakage!
        2. **Never tune hyperparameters on test data** - Use validation set
        3. **Split before preprocessing** - Prevent information leakage
        4. **Test only once** - At the very end
        5. **Stratify for imbalanced data** - Maintain class distribution

    ---

    ## Common Split Ratios

    ### Based on Dataset Size

    | Dataset Size | Train | Validation | Test | Reasoning |
    |--------------|-------|------------|------|-----------|
    | **Small (< 1K)** | 60% | 20% | 20% | Need enough samples in each set |
    | **Medium (1K-100K)** | 70% | 15% | 15% | Balanced approach |
    | **Large (100K-1M)** | 80% | 10% | 10% | More data for training |
    | **Very Large (> 1M)** | 98% | 1% | 1% | Even small % is many samples |

    ### Two-Way Split (Train/Test Only)

    Use when:
    - Not tuning hyperparameters
    - Using cross-validation instead
    - Very small dataset

    **Ratio:** 80/20 or 70/30

    ---

    ## When Things Go Wrong

    **Problem 1: Training on Test Data** (Data Leakage)
    ```python
    # WRONG ❌
    model.fit(X, y)  # Trains on all data
    score = model.score(X, y)  # Tests on same data
    # Result: Overly optimistic score
    ```

    **Problem 2: Fitting Scaler on All Data**
    ```python
    # WRONG ❌
    X_scaled = scaler.fit_transform(X)  # Uses info from all data
    X_train, X_test = train_test_split(X_scaled)  # Too late!
    # Problem: Test data statistics leaked into training
    ```

    **Problem 3: Tuning on Test Data**
    ```python
    # WRONG ❌
    for alpha in [0.1, 1.0, 10.0]:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)  # Using test data!
        # Select best alpha based on test scores
    # Problem: Test set is no longer "unseen"
    ```

=== "🧮 Theory & Math"
    ## Statistical Foundation

    ### Generalization Error

    We want to estimate the expected error on new data:

    $$\text{Generalization Error} = \mathbb{E}_{(x,y) \sim P}[\mathcal{L}(f(x), y)]$$

    Where:
    - $P$ is the true data distribution
    - $\mathcal{L}$ is the loss function
    - $f$ is our learned model

    **Problem:** We don't have access to $P$, only a finite sample.

    **Solution:** Split sample into train (learn) and test (estimate generalization).

    ---

    ## Train/Test Split as Sampling

    Given dataset $D = \{(x^{(i)}, y^{(i)})\}_{i=1}^{m}$

    **Training Set:** $D_{\text{train}} \subset D$, size $m_{\text{train}}$

    **Test Set:** $D_{\text{test}} = D \setminus D_{\text{train}}$, size $m_{\text{test}}$

    **Objective:**
    1. Learn $\hat{f}$ on $D_{\text{train}}$
    2. Estimate error: $\hat{\mathcal{E}} = \frac{1}{m_{\text{test}}}\sum_{(x,y) \in D_{\text{test}}}\mathcal{L}(\hat{f}(x), y)$

    ---

    ## Bias-Variance of Test Error Estimate

    The test error estimate itself has:

    **Bias:** Low if test set is representative
    **Variance:** Decreases with larger test set

    $$\text{Var}[\hat{\mathcal{E}}] \propto \frac{1}{m_{\text{test}}}$$

    **Tradeoff:**
    - Larger train set → Better model
    - Larger test set → Better error estimate

    ---

    ## Stratified Sampling

    For imbalanced datasets, ensure class proportions are maintained.

    **Population:** Class A: 90%, Class B: 10%

    **Random Split Risk:** Might get test set with 95% A, 5% B (unrepresentative)

    **Stratified Split:** Guarantees test set has 90% A, 10% B

    **Formula for Stratified Split:**

    For each class $c$:
    $$n_{\text{train}}^{(c)} = \text{split\_ratio} \times n^{(c)}$$
    $$n_{\text{test}}^{(c)} = (1 - \text{split\_ratio}) \times n^{(c)}$$

    ---

    ## Time Series Considerations

    **Standard Split:** Random assignment

    **Time Series:** Cannot randomly split!

    **Why:** Future data can't be used to predict past (temporal leak)

    **Solution:** Split chronologically

    ```
    Time: ──────────────────────────────────────→
          |         Train          | Val |Test|
          ←─────────────────────────→
    ```

    **Validation Strategy:** Walk-forward validation
    - Train on [t₁, t₂]
    - Validate on [t₂+1, t₃]
    - Test on [t₃+1, t₄]

=== "💻 Implementation"
    ## Basic Train/Test Split

    ```python
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    # Load data
    data = load_iris()
    X, y = data.data, data.target

    print(f"Total samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {np.unique(y)}")

    # ========================================
    # Proper Train/Test Split
    # ========================================

    # Split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,      # 20% for testing
        random_state=42,    # Reproducibility
        stratify=y          # Maintain class proportions
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Check stratification worked
    print(f"\nOriginal class distribution: {np.bincount(y) / len(y)}")
    print(f"Train class distribution: {np.bincount(y_train) / len(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test) / len(y_test)}")

    # ========================================
    # Train Model (Only on Training Data!)
    # ========================================

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)  # Learn from training data only

    # ========================================
    # Evaluate on Both Sets
    # ========================================

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"\nTraining Accuracy: {train_accuracy:.3f}")
    print(f"Test Accuracy: {test_accuracy:.3f}")
    print(f"Gap: {train_accuracy - test_accuracy:.3f}")

    # Diagnose overfitting
    if train_accuracy - test_accuracy > 0.1:
        print("\n⚠️ Warning: Possible overfitting (large gap)")
    else:
        print("\n✅ Good generalization (small gap)")
    ```

    ---

    ## Three-Way Split (Train/Val/Test)

    ```python
    from sklearn.model_selection import train_test_split

    # Load data
    X, y = load_iris(return_X_y=True)

    # ========================================
    # Method 1: Two Sequential Splits
    # ========================================

    # First split: separate test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Second split: train/validation from remaining 80%
    # 0.25 * 0.8 = 0.2 (20% of original for validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    print(f"Training: {len(X_train)} samples ({len(X_train)/len(X)*100:.0f}%)")
    print(f"Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.0f}%)")
    print(f"Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.0f}%)")

    # ========================================
    # Method 2: Custom Split Function
    # ========================================

    def train_val_test_split(X, y, train_ratio=0.6, val_ratio=0.2,
                             test_ratio=0.2, random_state=42):
        """
        Split data into train, validation, and test sets

        Args:
            X: Features
            y: Labels
            train_ratio: Proportion for training (default 0.6)
            val_ratio: Proportion for validation (default 0.2)
            test_ratio: Proportion for testing (default 0.2)
            random_state: Random seed
        """
        assert train_ratio + val_ratio + test_ratio == 1.0

        # First split: test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=random_state,
            stratify=y
        )

        # Second split: train and validation
        val_size = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size,
            random_state=random_state, stratify=y_temp
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    # Use custom function
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
    )
    ```

    ---

    ## Proper Preprocessing with Split

    ```python
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    # ========================================
    # CORRECT ✅
    # ========================================

    # 1. Split FIRST
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Fit scaler on training data ONLY
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 3. Transform test data using training statistics
    X_test_scaled = scaler.transform(X_test)  # Note: transform, not fit_transform!

    # 4. Train model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # 5. Evaluate
    test_accuracy = model.score(X_test_scaled, y_test)
    print(f"Test Accuracy: {test_accuracy:.3f}")

    # ========================================
    # WRONG ❌ (Data Leakage)
    # ========================================

    # Scaling before split
    X_scaled = scaler.fit_transform(X)  # Uses info from ALL data!
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

    # Problem: Test data statistics influenced the scaling
    # Result: Overly optimistic performance estimate
    ```

    ---

    ## Hyperparameter Tuning with Validation Set

    ```python
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt

    # Generate regression data
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=1000, n_features=20,
                          noise=10, random_state=42)

    # Three-way split
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    # ========================================
    # Tune hyperparameter using VALIDATION set
    # ========================================

    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    val_errors = []
    train_errors = []

    for alpha in alphas:
        # Train with this alpha
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        # Evaluate on VALIDATION set (not test!)
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        train_error = mean_squared_error(y_train, y_train_pred)
        val_error = mean_squared_error(y_val, y_val_pred)

        train_errors.append(train_error)
        val_errors.append(val_error)

    # Select best alpha based on validation error
    best_idx = np.argmin(val_errors)
    best_alpha = alphas[best_idx]

    print(f"Best alpha: {best_alpha}")

    # ========================================
    # Final evaluation on TEST set (only once!)
    # ========================================

    final_model = Ridge(alpha=best_alpha)
    final_model.fit(X_train, y_train)  # Could also use train + val

    y_test_pred = final_model.predict(X_test)
    test_error = mean_squared_error(y_test, y_test_pred)

    print(f"\nFinal Test Error: {test_error:.2f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.semilogx(alphas, train_errors, 'o-', label='Training Error')
    plt.semilogx(alphas, val_errors, 's-', label='Validation Error')
    plt.axvline(best_alpha, color='red', linestyle='--',
                label=f'Best Alpha: {best_alpha}')
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.title('Hyperparameter Tuning using Validation Set')
    plt.grid(alpha=0.3)
    plt.show()
    ```

    ---

    ## Time Series Split

    ```python
    from sklearn.model_selection import TimeSeriesSplit
    import pandas as pd

    # Generate time series data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    X = np.arange(100).reshape(-1, 1)  # Time index as feature
    y = np.sin(X.ravel() * 0.1) + np.random.randn(100) * 0.1

    # ========================================
    # Time Series Split (No Random Shuffle!)
    # ========================================

    tscv = TimeSeriesSplit(n_splits=5)

    print("Time Series Cross-Validation Splits:")
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"\nFold {i+1}:")
        print(f"  Train: indices {train_idx[0]} to {train_idx[-1]}")
        print(f"  Test:  indices {test_idx[0]} to {test_idx[-1]}")

    # ========================================
    # Manual Time-Based Split
    # ========================================

    train_size = int(0.8 * len(X))

    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    print(f"\nManual split:")
    print(f"Train: {len(X_train)} samples (up to index {train_size-1})")
    print(f"Test: {len(X_test)} samples (from index {train_size})")

    # Visualize split
    plt.figure(figsize=(12, 6))
    plt.plot(X_train, y_train, 'b.', label='Training Data', alpha=0.6)
    plt.plot(X_test, y_test, 'r.', label='Test Data', alpha=0.6)
    plt.axvline(train_size, color='green', linestyle='--',
                linewidth=2, label='Split Point')
    plt.xlabel('Time Index')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Time Series Train/Test Split')
    plt.show()
    ```

=== "📊 Visualization"
    ## Split Visualization

    ```mermaid
    graph TB
        A[Complete Dataset<br/>100%] --> B[Training Set<br/>70%]
        A --> C[Validation Set<br/>15%]
        A --> D[Test Set<br/>15%]

        B --> E[Learn Patterns<br/>Fit Parameters]
        C --> F[Tune Hyperparameters<br/>Model Selection]
        D --> G[Final Evaluation<br/>Report Performance]

        E --> H[Use many times]
        F --> I[Use during development]
        G --> J[Use ONCE at end]

        style A fill:#e1f5ff
        style B fill:#ccffcc
        style C fill:#ffffcc
        style D fill:#ffcccc
        style J fill:#ff6666
    ```

    ---

    ## Common Mistakes

    ```
    ❌ WRONG: Train on everything, test on everything
    ┌────────────────────────────────┐
    │    All Data (Train & Test)     │
    │         ↓                       │
    │    Same data used for both     │
    └────────────────────────────────┘
    Result: Overly optimistic scores!

    ✅ CORRECT: Separate train and test
    ┌──────────────────┬─────────────┐
    │   Training Set   │  Test Set   │
    │        ↓         │      ↓      │
    │    Learn here    │  Evaluate   │
    └──────────────────┴─────────────┘
    Result: Realistic performance!
    ```

=== "🎯 Practice"
    ## Beginner Problems

    **Problem 1: Basic Split**

    Load the Iris dataset and:
    - Split into 80% train, 20% test
    - Train a logistic regression model
    - Report train and test accuracy
    - Check for overfitting

    ---

    **Problem 2: Stratified Split**

    For an imbalanced dataset:
    - Create dataset with 90% class 0, 10% class 1
    - Split with and without stratification
    - Compare class distributions in test sets
    - Show why stratification matters

    ---

    ## Intermediate Problems

    **Problem 3: Three-Way Split**

    Implement a complete ML pipeline:
    - 60% train, 20% validation, 20% test
    - Try 5 different regularization values
    - Select best using validation set
    - Report final performance on test set

    ---

    **Problem 4: Preprocessing Leakage**

    Demonstrate data leakage:
    - Show WRONG way (scale before split)
    - Show CORRECT way (scale after split)
    - Compare test scores
    - Explain the difference

    ---

    ## Advanced Problems

    **Problem 5: Time Series Split**

    For stock price data:
    - Implement walk-forward validation
    - Train on past data, test on future
    - Show why random split fails
    - Compare with standard cross-validation

    ---

    **Problem 6: Nested Cross-Validation**

    Implement nested CV:
    - Outer loop for model evaluation
    - Inner loop for hyperparameter tuning
    - Compare with simple train/test split
    - Explain when to use each approach

---

## Decision Matrix

| Scenario | Recommended Split | Reasoning |
|----------|------------------|-----------|
| **Small dataset (< 1000)** | 60/20/20 | Need sufficient samples in each set |
| **Medium dataset (1K-100K)** | 70/15/15 | Balanced approach |
| **Large dataset (> 100K)** | 80/10/10 or 98/1/1 | More data for training |
| **Imbalanced classes** | Use stratification | Maintain class proportions |
| **Time series** | Chronological split | Prevent temporal leakage |
| **No hyperparameter tuning** | 80/20 (no validation) | Simpler when not tuning |
| **Extensive tuning** | 60/20/20 | Need validation set |

---

## Common Pitfalls and Solutions

| Pitfall | Problem | Solution |
|---------|---------|----------|
| **Training on test data** | Data leakage, overly optimistic scores | Split before any training |
| **Tuning on test data** | Test set no longer "unseen" | Use validation set for tuning |
| **Scaling before split** | Test statistics leak into training | Split first, then scale |
| **Random split for time series** | Future data predicts past | Use chronological split |
| **Not stratifying** | Unrepresentative test set | Use stratify parameter |
| **Testing multiple times** | Implicit overfitting to test | Test only once at end |
| **Too small test set** | Unreliable error estimates | Ensure min 100-200 samples |

---

## Interview Preparation

### Common Questions

**Q1: Why do we split data into train and test sets?**

**Strong Answer:** "We split data to measure how well our model generalizes to new, unseen data. If we evaluate on training data, we're testing memorization, not learning. The test set simulates real-world deployment where the model encounters new examples. This helps us detect overfitting - if training performance is great but test performance is poor, the model memorized rather than learned patterns."

**Q2: What's the difference between validation and test sets?**

**Strong Answer:** "Validation set is used during development to tune hyperparameters and select models - you can use it multiple times. Test set is for final evaluation only, used once at the very end to report model performance. Think of validation as practice tests you use to improve, and test as the final exam. If you tune based on test performance, it's no longer truly 'unseen' and your reported performance will be overly optimistic."

**Q3: What ratio should you use for splitting?**

**Strong Answer:** "Depends on dataset size. Small datasets (< 1K): 60/20/20 train/val/test to ensure enough samples in each. Medium (1-100K): 70/15/15 or 80/10/10. Large (> 100K): 80/10/10 or even 98/1/1 since 1% of a million is still 10K samples. The key is having enough test samples for reliable error estimates (minimum 100-200) while maximizing training data."

**Q4: When should you use stratified sampling?**

**Strong Answer:** "Use stratification when you have imbalanced classes or important subgroups. It ensures the class distribution in train/test matches the original dataset. For example, if your data is 90% class A and 10% class B, random splitting might give you a test set with 95% A and 5% B, which is unrepresentative. Stratification guarantees both sets have the same 90/10 split, leading to more reliable evaluation."

**Q5: What's data leakage and how do you prevent it?**

**Strong Answer:** "Data leakage is when information from test data influences training, leading to overly optimistic performance. Common causes: (1) training on test data directly, (2) fitting preprocessing (scaling, imputation) on all data before splitting, (3) using test data to tune hyperparameters. Prevent it by: split data first thing, fit preprocessing only on training data, use validation set for tuning, and test only once at the end."

**Q6: How do you handle time series data splitting?**

**Strong Answer:** "Time series requires special treatment because data has temporal dependency. You can't randomly split because that would use future data to predict the past. Instead, split chronologically - train on earlier data, test on later data. Use walk-forward validation: train on [t1, t2], validate on [t2+1, t3], test on [t3+1, t4]. This simulates real-world usage where you predict future from past."

**Q7: What if you have a very small dataset?**

**Strong Answer:** "For small datasets, use cross-validation instead of or in addition to train/test split. K-fold CV uses all data for both training and testing by rotating which fold is the test set. This gives more reliable performance estimates. For tiny datasets (< 100 samples), consider leave-one-out CV where each sample is the test set once. However, still hold out a final test set if possible, or at least use nested CV."

---

## Related Topics

- [Cross-Validation](../evaluation/cross-validation.md) - More robust evaluation strategy
- [Overfitting & Underfitting](overfitting-underfitting.md) - What train/test split helps detect
- [Data Preprocessing](../feature-engineering/preprocessing.md) - When to scale and transform
- [Model Evaluation](../evaluation/) - Metrics for train/test comparison
- [Bias-Variance Tradeoff](bias-variance-tradeoff.md) - Theoretical foundation

---

## References

1. **Scikit-learn:** [train_test_split Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
2. **Andrew Ng's ML Course:** Lectures on train/test/validation sets
3. **Book:** *Hands-On Machine Learning* by Aurélien Géron - Chapter 2
4. **Google's ML Course:** [Training and Test Sets](https://developers.google.com/machine-learning/crash-course/training-and-test-sets)
5. **Paper:** "A Few Useful Things to Know About Machine Learning" by Pedro Domingos
6. **Scikit-learn:** [Cross-validation Guide](https://scikit-learn.org/stable/modules/cross_validation.html)

---

**Next Steps:**

- Master [Cross-Validation](../evaluation/cross-validation.md) for robust evaluation
- Learn [Model Evaluation Metrics](../evaluation/) specific to your problem
- Study [Overfitting Detection](overfitting-underfitting.md) using learning curves
- Practice with real datasets on [Kaggle](https://www.kaggle.com/)

**Remember:** Proper data splitting is the foundation of honest ML evaluation - get this right or everything else is built on sand!

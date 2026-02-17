# Self-Training

**Use a model's own predictions to generate training labels for unlabeled data.**

Self-training is the simplest semi-supervised learning approach where a model is iteratively trained on its own most confident predictions.

---

## 🎯 What is Self-Training?

### Core Idea

**The Self-Teaching Process:**
1. Train a model on the small labeled dataset
2. Use the model to predict labels for unlabeled data
3. Select the most confident predictions
4. Add these as "pseudo-labeled" examples to the training set
5. Retrain the model on the expanded dataset
6. Repeat until convergence or all unlabeled data is labeled

**Intuition:** If a model is confident about its prediction, it's likely correct. We can use these confident predictions as additional training data.

---

## 🔑 Key Components

### 1. Initial Model

**Requirements:**
- Trained on labeled data only
- Produces probability estimates (not just class labels)
- Reasonable baseline performance (>60% accuracy typically)

**Example:**
```python
# Train initial model on labeled data
model = LogisticRegression()
model.fit(X_labeled, y_labeled)

# Evaluate baseline
baseline_accuracy = model.score(X_val, y_val)
print(f"Baseline: {baseline_accuracy:.2%}")  # e.g., 72%
```

### 2. Confidence Threshold

**Purpose:** Filter predictions to keep only high-confidence examples

**Common Thresholds:**
- **Conservative:** 0.95+ (very confident)
- **Moderate:** 0.90-0.95
- **Aggressive:** 0.80-0.90 (risky, may include mistakes)

**Example:**
```python
# Get predictions with probabilities
predictions = model.predict_proba(X_unlabeled)
max_probs = predictions.max(axis=1)

# Select confident predictions
confident_mask = max_probs >= 0.95
X_confident = X_unlabeled[confident_mask]
y_confident = predictions[confident_mask].argmax(axis=1)

print(f"Selected {len(X_confident)} confident examples")
```

### 3. Selection Strategy

**How many examples to add per iteration?**

**Fixed Number:**
- Add top-k most confident examples (e.g., k=100)
- Ensures consistent growth

**Threshold-Based:**
- Add all examples above confidence threshold
- Variable growth per iteration

**Balanced:**
- Add equal number per class
- Prevents class imbalance

---

## 📊 The Self-Training Algorithm

### Standard Algorithm

```
Algorithm: Self-Training

Input:
  - L: Labeled dataset {(x, y)}
  - U: Unlabeled dataset {x}
  - θ: Confidence threshold
  - max_iter: Maximum iterations

Output:
  - Trained model f

1. Train initial model f on L
2. For iteration = 1 to max_iter:
     a. Predict labels for U: predictions = f(U)
     b. Select confident: C = {(x, ŷ) | confidence(ŷ) ≥ θ}
     c. If C is empty: break
     d. Add C to L: L = L ∪ C
     e. Remove C from U: U = U \ C
     f. Retrain f on updated L
3. Return f
```

### Implementation

```python
def self_training(X_labeled, y_labeled, X_unlabeled,
                  model, threshold=0.95, max_iter=10):
    """
    Self-training semi-supervised learning

    Args:
        X_labeled: Labeled features
        y_labeled: Labeled targets
        X_unlabeled: Unlabeled features
        model: Base classifier with predict_proba
        threshold: Confidence threshold
        max_iter: Maximum iterations

    Returns:
        Trained model
    """
    # Train initial model
    model.fit(X_labeled, y_labeled)

    for iteration in range(max_iter):
        # Get predictions with confidence
        probs = model.predict_proba(X_unlabeled)
        max_probs = probs.max(axis=1)
        predicted_labels = probs.argmax(axis=1)

        # Select confident predictions
        confident_mask = max_probs >= threshold

        if not confident_mask.any():
            print(f"No confident predictions at iteration {iteration}")
            break

        # Add confident examples to labeled set
        X_confident = X_unlabeled[confident_mask]
        y_confident = predicted_labels[confident_mask]

        X_labeled = np.vstack([X_labeled, X_confident])
        y_labeled = np.hstack([y_labeled, y_confident])

        # Remove from unlabeled set
        X_unlabeled = X_unlabeled[~confident_mask]

        print(f"Iteration {iteration}: Added {confident_mask.sum()} examples")
        print(f"  Training set size: {len(X_labeled)}")
        print(f"  Unlabeled remaining: {len(X_unlabeled)}")

        # Retrain model
        model.fit(X_labeled, y_labeled)

    return model
```

---

## 🎓 Practical Example: Text Classification

### Scenario: Sentiment Analysis

**Dataset:**
- 500 labeled movie reviews (250 positive, 250 negative)
- 10,000 unlabeled movie reviews
- Goal: Classify reviews as positive/negative

### Step-by-Step Process

**Step 1: Train Initial Model**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Prepare data
vectorizer = TfidfVectorizer(max_features=1000)
X_labeled = vectorizer.fit_transform(labeled_reviews)
y_labeled = labeled_sentiment  # [0 or 1]

# Train baseline
model = LogisticRegression()
model.fit(X_labeled, y_labeled)

# Evaluate
baseline_acc = model.score(X_val, y_val)
print(f"Baseline accuracy: {baseline_acc:.2%}")  # e.g., 78%
```

**Step 2: Apply Self-Training**
```python
# Transform unlabeled data
X_unlabeled = vectorizer.transform(unlabeled_reviews)

# Run self-training
model_st = self_training(
    X_labeled, y_labeled, X_unlabeled,
    model=LogisticRegression(),
    threshold=0.95,
    max_iter=10
)

# Evaluate
final_acc = model_st.score(X_val, y_val)
print(f"After self-training: {final_acc:.2%}")  # e.g., 83%
print(f"Improvement: {final_acc - baseline_acc:.2%}")  # e.g., +5%
```

**Iteration Results:**
```
Iteration 0: Added 1,250 examples (12.5% of unlabeled)
  Training set: 1,750
  Unlabeled remaining: 8,750

Iteration 1: Added 980 examples
  Training set: 2,730
  Unlabeled remaining: 7,770

Iteration 2: Added 750 examples
  Training set: 3,480
  Unlabeled remaining: 7,020

...

Final accuracy: 83% (+5% over baseline)
```

---

## ✅ When Self-Training Works Well

### Ideal Conditions

**1. Confident Model**
- Initial model has >70% accuracy
- Clear decision boundaries
- Well-separated classes

**2. Unlabeled Data Quality**
- Same distribution as labeled data
- Relatively clean (few outliers)
- Large quantity available

**3. Problem Characteristics**
- Natural clusters in data
- High-confidence regions exist
- Classes are separable

### Success Stories

**Image Classification:**
- Object recognition with limited labels
- Medical image analysis (X-rays, MRIs)
- Satellite imagery classification

**Text Classification:**
- Sentiment analysis
- Topic categorization
- Spam detection

**Speech Recognition:**
- Accent detection
- Speaker identification
- Language classification

---

## ⚠️ Common Pitfalls and Solutions

### 1. Confirmation Bias

**Problem:** Model reinforces its own mistakes

**Example:**
```
Initial model misclassifies some examples
→ Adds these misclassified examples with wrong labels
→ Retrained model makes same mistakes more strongly
→ Error propagates and amplifies
```

**Solutions:**
- **High Thresholds:** Use conservative thresholds (0.95+)
- **Periodic Reset:** Restart from original labeled set occasionally
- **Ensemble Voting:** Use multiple models, add only if they agree
- **Validation Monitoring:** Stop if validation performance degrades

```python
def self_training_with_monitoring(X_labeled, y_labeled, X_unlabeled,
                                   X_val, y_val, model, threshold=0.95):
    best_model = None
    best_score = 0

    for iteration in range(max_iter):
        # Self-training step
        model = self_training_step(...)

        # Validate
        val_score = model.score(X_val, y_val)

        if val_score > best_score:
            best_score = val_score
            best_model = model
        elif val_score < best_score - 0.02:  # Degrading
            print("Performance degrading, stopping")
            break

    return best_model
```

### 2. Class Imbalance

**Problem:** Model becomes biased toward confident class

**Example:**
```
Initial: 50% positive, 50% negative
After iteration 1: 60% positive, 40% negative  # Imbalance grows
After iteration 5: 80% positive, 20% negative  # Severe imbalance
```

**Solutions:**
- **Balanced Sampling:** Add equal number per class
- **Class-Specific Thresholds:** Higher threshold for majority class
- **Stratified Selection:** Maintain class distribution

```python
def balanced_selection(probs, labels, n_per_class=100):
    """Select equal number of confident examples per class"""
    selected_indices = []

    for class_label in np.unique(labels):
        class_mask = (labels == class_label)
        class_probs = probs[class_mask]

        # Get top-n most confident for this class
        top_n = np.argsort(class_probs.max(axis=1))[-n_per_class:]
        selected_indices.extend(np.where(class_mask)[0][top_n])

    return np.array(selected_indices)
```

### 3. Distribution Shift

**Problem:** Unlabeled data from different distribution

**Example:**
- Training: Product reviews from Amazon
- Unlabeled: Product reviews from Reddit
- Different writing style, vocabulary, context

**Solutions:**
- **Domain Adaptation:** Align distributions first
- **Careful Selection:** Filter unlabeled data for similarity
- **Two-Stage Approach:** Self-train on similar examples first

### 4. Poor Initial Model

**Problem:** Weak baseline leads to poor pseudo-labels

**Rule of Thumb:**
- Baseline < 60%: Self-training unlikely to help
- Baseline 60-75%: May help with careful tuning
- Baseline > 75%: Good candidate for self-training

**Solutions:**
- Improve initial model first (more features, better algorithm)
- Get more labeled data if possible
- Try other semi-supervised methods

---

## 🎯 Best Practices

### 1. Start Conservative

```python
# Begin with high threshold
initial_threshold = 0.95

# Gradually relax if needed
for iteration in range(max_iter):
    threshold = max(0.80, initial_threshold - 0.01 * iteration)
    # ... self-training step
```

### 2. Monitor Everything

```python
def self_training_with_logging(X_labeled, y_labeled, X_unlabeled, X_val, y_val):
    history = {
        'train_size': [],
        'val_accuracy': [],
        'confident_ratio': [],
        'class_distribution': []
    }

    for iteration in range(max_iter):
        # Training step
        # ...

        # Log metrics
        history['train_size'].append(len(X_labeled))
        history['val_accuracy'].append(model.score(X_val, y_val))
        history['confident_ratio'].append(confident_mask.mean())
        history['class_distribution'].append(
            np.bincount(y_labeled) / len(y_labeled)
        )

    return model, history
```

### 3. Ensemble Approach

**Idea:** Use multiple models, add only examples where all agree

```python
def ensemble_self_training(X_labeled, y_labeled, X_unlabeled, models):
    """Self-training with ensemble agreement"""

    # Train multiple models
    for model in models:
        model.fit(X_labeled, y_labeled)

    # Get predictions from all models
    all_predictions = [m.predict(X_unlabeled) for m in models]

    # Select examples where all models agree
    agreement_mask = np.all(
        [pred == all_predictions[0] for pred in all_predictions],
        axis=0
    )

    # Add agreed-upon examples
    X_confident = X_unlabeled[agreement_mask]
    y_confident = all_predictions[0][agreement_mask]

    return X_confident, y_confident
```

### 4. Gradual Addition

**Idea:** Add small batches iteratively rather than all at once

```python
# Instead of:
# Add all confident examples at once

# Do:
def gradual_addition(X_labeled, y_labeled, X_unlabeled,
                     batch_size=100, max_batches=50):
    for batch in range(max_batches):
        # Train model
        model.fit(X_labeled, y_labeled)

        # Get top-k most confident
        probs = model.predict_proba(X_unlabeled)
        top_k = np.argsort(probs.max(axis=1))[-batch_size:]

        # Add batch
        X_labeled = np.vstack([X_labeled, X_unlabeled[top_k]])
        y_labeled = np.hstack([y_labeled, probs[top_k].argmax(axis=1)])
        X_unlabeled = np.delete(X_unlabeled, top_k, axis=0)

    return model
```

---

## 📊 Performance Analysis

### Expected Improvements

**Typical Results:**

| Initial Labeled | Unlabeled | Baseline | Self-Training | Improvement |
|-----------------|-----------|----------|---------------|-------------|
| 100 | 10,000 | 65% | 72% | +7% |
| 500 | 10,000 | 75% | 81% | +6% |
| 1,000 | 10,000 | 80% | 84% | +4% |
| 5,000 | 10,000 | 85% | 87% | +2% |

**Key Insight:** Improvement is larger when initial labeled set is smaller (most benefit when labels are scarce).

### Comparison with Fully Supervised

**Example: CIFAR-10 Classification**
- Fully supervised (50,000 labels): 90% accuracy
- Self-training (4,000 labels + 46,000 unlabeled): 85% accuracy
- Supervised only (4,000 labels): 75% accuracy

**Value:** Self-training achieves 85% accuracy with only 4,000 labels, compared to 75% without unlabeled data.

---

## 🔬 Variants and Extensions

### 1. Yarowsky Algorithm

**Specific for NLP, bootstrapping from seed rules**

```
1. Start with seed rules (e.g., "bank" near "river" → geographic)
2. Label examples matching rules
3. Train classifier
4. Add confident predictions
5. Iterate
```

### 2. Curriculum Self-Training

**Add examples in order of increasing difficulty**

```python
def curriculum_self_training(X_labeled, y_labeled, X_unlabeled):
    # Start with very high threshold
    thresholds = [0.99, 0.97, 0.95, 0.93, 0.90]

    for threshold in thresholds:
        model.fit(X_labeled, y_labeled)
        probs = model.predict_proba(X_unlabeled)

        # Add examples at current difficulty level
        confident = (probs.max(axis=1) >= threshold) & \
                   (probs.max(axis=1) < threshold + 0.02)

        # Update datasets
        # ...
```

### 3. Multi-View Self-Training

**Combine with co-training ideas**

```python
def multiview_self_training(X1, X2, y_labeled, X1_unlabeled, X2_unlabeled):
    """Two views of same data"""
    model1 = train_on_view1(X1, y_labeled)
    model2 = train_on_view2(X2, y_labeled)

    # Each model labels for the other
    labels1 = model1.predict(X1_unlabeled)
    labels2 = model2.predict(X2_unlabeled)

    # Add where both agree
    agreement = (labels1 == labels2)
    # ...
```

---

## 📚 References and Further Reading

**Classic Papers:**
- Yarowsky, D. (1995). "Unsupervised word sense disambiguation"
- Blum, A., & Mitchell, T. (1998). "Combining labeled and unlabeled data with co-training"
- Zhu, X. (2005). "Semi-supervised learning literature survey"

**Modern Applications:**
- Pseudo-Labeling for Deep Learning
- Self-training for neural networks
- Active + Self-training hybrid approaches

**Libraries:**
- scikit-learn: `SelfTrainingClassifier`
- Semi-supervised learning frameworks
- Custom implementations

---

## 🚀 Next Steps

**Try It Yourself:**
1. Implement self-training on a simple dataset
2. Experiment with different thresholds
3. Monitor validation performance
4. Compare with supervised baseline

**Explore Related Methods:**
- [Co-Training](co-training.md) - Multiple views of data
- [Pseudo-Labeling](pseudo-labeling.md) - Modern deep learning approach
- [Main Index](index.md) - Overview of all methods

---

**Key Takeaway:** Self-training is simple yet powerful. It works best with confident initial models, high-quality unlabeled data, and careful monitoring to avoid confirmation bias.

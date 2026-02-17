# Co-Training

**Train two models on different views of data that teach each other through confident predictions.**

Co-training leverages multiple independent views of the same data to create a more robust semi-supervised learning algorithm.

---

## 🎯 What is Co-Training?

### Core Idea

**The Two-View Learning Process:**
1. Split features into two independent "views" of the data
2. Train two separate models, one on each view
3. Each model predicts labels for unlabeled data
4. Add each model's confident predictions to the other model's training set
5. Retrain both models
6. Repeat

**Key Insight:** When two models using different feature sets agree on a prediction, it's likely correct.

### The "Teaching Each Other" Concept

**Example: Web Page Classification**

**View 1 (Page Content):**
- Words on the page
- Heading structure
- Meta tags

**View 2 (Hyperlinks):**
- Text in links pointing to this page
- Number and types of incoming links
- Domains linking to this page

**Process:**
```
Model 1 learns from page content → Labels unlabeled pages confidently
Model 2 learns from hyperlinks → Uses Model 1's labels as training data
Model 2 labels pages confidently → Model 1 uses Model 2's labels
Both models improve by teaching each other!
```

---

## 🔑 Key Requirements

### 1. Two Independent Views

**Definition:** Features can be split into two sets where each set is:
- **Sufficient:** Each view alone can train a reasonable classifier
- **Conditionally Independent:** Views are independent given the class label

**Mathematical Formulation:**
```
Given class Y:
  View 1 features (X₁) ⊥ View 2 features (X₂) | Y

This means: P(X₁, X₂ | Y) = P(X₁ | Y) · P(X₂ | Y)
```

**Intuition:** Knowing the class label, one view doesn't tell you about the other view.

### Good View Splits

**Natural Splits:**

| Domain | View 1 | View 2 |
|--------|--------|--------|
| **Text Documents** | Document content | Citation information |
| **Web Pages** | Page text | Hyperlinks/anchor text |
| **Images** | Color/texture | Shape/edges |
| **Speech** | Audio features | Visual lip movements |
| **Videos** | Visual frames | Audio track |
| **Medical Records** | Lab results | Clinical notes |
| **Product Reviews** | Review text | Reviewer history |

### Bad View Splits

❌ **Not Independent:**
- Image: Left half vs. Right half (spatially dependent)
- Text: First half vs. Second half (sequential dependency)
- Time series: Adjacent time windows

❌ **Not Sufficient:**
- All features in View 1, nothing in View 2
- Views with random features that don't predict class

---

## 📊 The Co-Training Algorithm

### Standard Algorithm

```
Algorithm: Co-Training

Input:
  - L: Small labeled dataset
  - U: Large unlabeled dataset
  - X₁, X₂: Two feature views
  - k: Number of examples to add per iteration
  - max_iter: Maximum iterations

Output:
  - Two trained models f₁, f₂

1. Split features: L = (L₁, L₂) and U = (U₁, U₂)
2. Train f₁ on L₁ and f₂ on L₂
3. For iteration = 1 to max_iter:
     a. f₁ labels U₁, select k most confident → C₁
     b. f₂ labels U₂, select k most confident → C₂
     c. Add C₁ to L₂ (f₁ teaches f₂)
     d. Add C₂ to L₁ (f₂ teaches f₁)
     e. Remove C₁, C₂ from U
     f. Retrain f₁ on updated L₁
     g. Retrain f₂ on updated L₂
4. Return ensemble of f₁ and f₂
```

### Implementation

```python
def co_training(X1_labeled, X2_labeled, y_labeled,
                X1_unlabeled, X2_unlabeled,
                model1, model2, k=10, max_iter=30):
    """
    Co-training with two views

    Args:
        X1_labeled, X2_labeled: Labeled data in both views
        y_labeled: Labels
        X1_unlabeled, X2_unlabeled: Unlabeled data in both views
        model1, model2: Base classifiers for each view
        k: Number of examples to add per iteration per class
        max_iter: Maximum iterations

    Returns:
        Two trained models
    """
    # Train initial models
    model1.fit(X1_labeled, y_labeled)
    model2.fit(X2_labeled, y_labeled)

    for iteration in range(max_iter):
        # Model 1 predicts on view 1
        probs1 = model1.predict_proba(X1_unlabeled)
        labels1 = probs1.argmax(axis=1)
        conf1 = probs1.max(axis=1)

        # Model 2 predicts on view 2
        probs2 = model2.predict_proba(X2_unlabeled)
        labels2 = probs2.argmax(axis=1)
        conf2 = probs2.max(axis=1)

        # Select k most confident per class from each model
        indices1 = select_top_k_per_class(conf1, labels1, k)
        indices2 = select_top_k_per_class(conf2, labels2, k)

        if len(indices1) == 0 and len(indices2) == 0:
            print(f"No confident predictions at iteration {iteration}")
            break

        # Model 1 teaches Model 2
        if len(indices1) > 0:
            X2_labeled = np.vstack([X2_labeled, X2_unlabeled[indices1]])
            y_labeled2 = np.hstack([y_labeled, labels1[indices1]])
            model2.fit(X2_labeled, y_labeled2)

        # Model 2 teaches Model 1
        if len(indices2) > 0:
            X1_labeled = np.vstack([X1_labeled, X1_unlabeled[indices2]])
            y_labeled1 = np.hstack([y_labeled, labels2[indices2]])
            model1.fit(X1_labeled, y_labeled1)

        # Remove labeled examples from unlabeled pool
        all_indices = np.union1d(indices1, indices2)
        X1_unlabeled = np.delete(X1_unlabeled, all_indices, axis=0)
        X2_unlabeled = np.delete(X2_unlabeled, all_indices, axis=0)

        print(f"Iteration {iteration}: Added {len(indices1) + len(indices2)} examples")

    return model1, model2


def select_top_k_per_class(confidences, labels, k):
    """Select top-k most confident examples per class"""
    indices = []
    unique_labels = np.unique(labels)

    for label in unique_labels:
        label_mask = (labels == label)
        label_confidences = confidences[label_mask]

        if len(label_confidences) == 0:
            continue

        # Get top-k indices for this class
        n_select = min(k, len(label_confidences))
        top_k_local = np.argsort(label_confidences)[-n_select:]

        # Convert to global indices
        global_indices = np.where(label_mask)[0][top_k_local]
        indices.extend(global_indices)

    return np.array(indices)
```

---

## 🎓 Practical Example: Email Classification

### Scenario: Spam Detection

**Dataset:**
- 200 labeled emails (100 spam, 100 legitimate)
- 10,000 unlabeled emails
- Goal: Build spam classifier

### Feature Views

**View 1: Email Content**
```python
# Text features from email body
content_features = [
    'word_count',
    'capitalization_ratio',
    'exclamation_count',
    'keyword_frequencies',
    'tfidf_features'
]
```

**View 2: Email Metadata**
```python
# Structural features
metadata_features = [
    'sender_domain',
    'has_attachments',
    'time_of_day',
    'recipient_count',
    'subject_length',
    'html_ratio'
]
```

### Implementation

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Prepare View 1: Content
vectorizer = TfidfVectorizer(max_features=500)
X1_labeled = vectorizer.fit_transform(email_content_labeled)
X1_unlabeled = vectorizer.transform(email_content_unlabeled)

# Prepare View 2: Metadata
X2_labeled = extract_metadata_features(emails_labeled)
X2_unlabeled = extract_metadata_features(emails_unlabeled)

# Initialize models
model1 = MultinomialNB()  # Good for text
model2 = RandomForestClassifier()  # Good for mixed features

# Run co-training
model1, model2 = co_training(
    X1_labeled, X2_labeled, y_labeled,
    X1_unlabeled, X2_unlabeled,
    model1, model2,
    k=20,  # Add 20 examples per class per iteration
    max_iter=30
)

# Make predictions (ensemble)
pred1 = model1.predict_proba(X1_test)
pred2 = model2.predict_proba(X2_test)
final_pred = (pred1 + pred2) / 2  # Average predictions
```

### Results

```
Baseline (View 1 only, 200 labels): 82% accuracy
Baseline (View 2 only, 200 labels): 76% accuracy
Co-Training (200 labels + 10,000 unlabeled): 89% accuracy
Improvement: +7% over best single view
```

---

## ✅ When Co-Training Works Well

### Ideal Conditions

**1. Natural View Splits**
- Features naturally partition into two groups
- Each view is sufficient for classification
- Views are (approximately) conditionally independent

**2. Complementary Information**
- Views capture different aspects of data
- Agreement between views is meaningful
- Disagreement indicates uncertainty

**3. Unlabeled Data Quality**
- Large quantity available
- Same distribution as labeled data
- Both views available for all examples

### Success Domains

**Web Data:**
- Page classification (content + links)
- User profiling (behavior + demographics)
- Product categorization (description + reviews)

**Multimodal Data:**
- Audio-visual speech recognition
- Video understanding (visual + audio)
- Medical diagnosis (images + text reports)

**Text Mining:**
- Document classification (content + citations)
- Named entity recognition (context + gazetteer)
- Sentiment analysis (text + metadata)

---

## ⚠️ Common Pitfalls and Solutions

### 1. Views Not Independent

**Problem:** Violates conditional independence assumption

**Example:**
```python
# Bad split: Image left/right halves
# Objects often span both halves → dependent!
view1 = image[:, :width//2]  # Left half
view2 = image[:, width//2:]  # Right half
```

**Solution:**
- Use semantically different features
- Test independence: Train on each view, check correlation of errors

```python
# Good split: Different feature types
view1 = extract_color_histogram(image)
view2 = extract_edge_features(image)
```

### 2. One View Dominates

**Problem:** One view much stronger than other

**Example:**
```
View 1 accuracy: 85% (strong)
View 2 accuracy: 60% (weak)
→ View 2 learns from View 1, but View 1 doesn't benefit
```

**Solutions:**
- Weight contributions inversely by view strength
- Use confidence-based selection
- Require agreement between views

```python
def weighted_co_training(X1, X2, y, model1, model2):
    # Evaluate view strengths
    acc1 = model1.score(X1_val, y_val)
    acc2 = model2.score(X2_val, y_val)

    # Weight by reliability
    weight1 = acc1 / (acc1 + acc2)
    weight2 = acc2 / (acc1 + acc2)

    # Use weighted ensemble
    pred = weight1 * model1.predict_proba(X) + \
           weight2 * model2.predict_proba(X)
```

### 3. Distribution Mismatch

**Problem:** Unlabeled data different from labeled

**Solution:**
- Use domain adaptation techniques
- Filter unlabeled examples by similarity
- Start with conservative selection (high confidence)

### 4. Confirmation Bias Cascade

**Problem:** Both models reinforce each other's mistakes

**Example:**
```
Model 1 makes mistake → Labels example incorrectly
Model 2 learns from mistake → Makes same mistake
Model 2 reinforces Model 1's error → Mistake amplified
```

**Solutions:**
- **Agreement Requirement:** Only add examples where both models agree

```python
def agreement_co_training(X1, X2, y, model1, model2):
    pred1 = model1.predict(X1_unlabeled)
    pred2 = model2.predict(X2_unlabeled)

    # Only add where models agree
    agreement = (pred1 == pred2)
    confident1 = model1.predict_proba(X1_unlabeled).max(axis=1) > 0.9
    confident2 = model2.predict_proba(X2_unlabeled).max(axis=1) > 0.9

    select = agreement & confident1 & confident2
    # Add selected examples
```

- **Validation Monitoring:** Stop if performance degrades
- **Periodic Reset:** Restart from original labeled data

---

## 🔬 Variants and Extensions

### 1. Tri-Training

**Idea:** Use three views instead of two

**Benefits:**
- More robust (majority voting)
- Less sensitive to single view mistakes
- Works when you have three natural views

```python
def tri_training(X1, X2, X3, y, model1, model2, model3):
    """Three views, majority voting"""

    for iteration in range(max_iter):
        # Train all three models
        model1.fit(X1_labeled, y)
        model2.fit(X2_labeled, y)
        model3.fit(X3_labeled, y)

        # Each pair labels for the third
        pred1 = model1.predict(X1_unlabeled)
        pred2 = model2.predict(X2_unlabeled)
        pred3 = model3.predict(X3_unlabeled)

        # Model 1 and 2 agree → teach Model 3
        agree_12 = (pred1 == pred2)
        X3_new = X3_unlabeled[agree_12]
        y3_new = pred1[agree_12]

        # Similarly for other pairs...
```

### 2. Co-Training with Disagreement

**Idea:** Focus on examples where views disagree

**Benefit:** Disagreement indicates the most informative examples

```python
def disagreement_sampling(model1, model2, X1, X2, y):
    """Query labels for examples where models disagree"""

    pred1 = model1.predict(X1_unlabeled)
    pred2 = model2.predict(X2_unlabeled)

    # Select disagreement examples
    disagree = (pred1 != pred2)

    # Query these for labels (active learning)
    X_query = X_unlabeled[disagree]
    return X_query
```

### 3. Multi-View Learning

**Generalization:** Learn shared representation across views

```python
# Modern approach: Neural networks
class MultiViewNetwork(nn.Module):
    def __init__(self):
        self.view1_encoder = nn.Linear(view1_dim, hidden_dim)
        self.view2_encoder = nn.Linear(view2_dim, hidden_dim)
        self.shared_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x1, x2):
        h1 = self.view1_encoder(x1)
        h2 = self.view2_encoder(x2)
        h_shared = (h1 + h2) / 2  # Average representations
        return self.shared_layer(h_shared)
```

---

## 🎯 Best Practices

### 1. Validate View Quality

**Before co-training:**
```python
# Check each view individually
model1.fit(X1_labeled, y_labeled)
model2.fit(X2_labeled, y_labeled)

acc1 = model1.score(X1_val, y_val)
acc2 = model2.score(X2_val, y_val)

print(f"View 1 accuracy: {acc1:.2%}")
print(f"View 2 accuracy: {acc2:.2%}")

# Both should be > 60% for co-training to work
if acc1 < 0.6 or acc2 < 0.6:
    print("Warning: One view is too weak")
```

### 2. Balance Class Distribution

**Add equal number per class:**
```python
def balanced_selection(probs, k_per_class):
    """Select k examples per class"""
    selected = []

    for class_idx in range(num_classes):
        class_probs = probs[:, class_idx]
        top_k = np.argsort(class_probs)[-k_per_class:]
        selected.extend(top_k)

    return np.array(selected)
```

### 3. Use Ensemble Prediction

**Combine both models:**
```python
def ensemble_predict(model1, model2, X1_test, X2_test):
    """Ensemble prediction from both views"""

    prob1 = model1.predict_proba(X1_test)
    prob2 = model2.predict_proba(X2_test)

    # Average probabilities
    ensemble_prob = (prob1 + prob2) / 2

    # Or weighted average based on validation performance
    # ensemble_prob = weight1 * prob1 + weight2 * prob2

    return ensemble_prob.argmax(axis=1)
```

### 4. Monitor Both Views

```python
def co_training_with_monitoring(X1, X2, y, X_val1, X_val2, y_val):
    history = {
        'view1_acc': [],
        'view2_acc': [],
        'ensemble_acc': []
    }

    for iteration in range(max_iter):
        # Co-training step
        # ...

        # Evaluate both views
        acc1 = model1.score(X_val1, y_val)
        acc2 = model2.score(X_val2, y_val)

        # Ensemble
        pred = ensemble_predict(model1, model2, X_val1, X_val2)
        ens_acc = accuracy_score(y_val, pred)

        history['view1_acc'].append(acc1)
        history['view2_acc'].append(acc2)
        history['ensemble_acc'].append(ens_acc)

    return model1, model2, history
```

---

## 📊 Performance Analysis

### Expected Improvements

**Typical Results:**

| Views | Baseline (Single View) | Co-Training | Improvement |
|-------|----------------------|-------------|-------------|
| Good split | 75% | 83% | +8% |
| Moderate split | 75% | 79% | +4% |
| Poor split | 75% | 76% | +1% |

**Key Insight:** Quality of view split is critical. Better splits → larger improvements.

### Comparison with Self-Training

**Co-Training vs. Self-Training:**

| Aspect | Self-Training | Co-Training |
|--------|--------------|-------------|
| **Views** | Single view | Two independent views |
| **Robustness** | Prone to confirmation bias | More robust (cross-validation) |
| **Applicability** | Any data | Requires view split |
| **Improvement** | 3-7% typical | 5-10% typical |
| **Complexity** | Simple | Moderate |

---

## 📚 References and Further Reading

**Seminal Paper:**
- Blum, A., & Mitchell, T. (1998). "Combining labeled and unlabeled data with co-training"

**Extensions:**
- Zhou, Z. H., & Li, M. (2005). "Tri-training: Exploiting unlabeled data using three classifiers"
- Christoudias, C. M., et al. (2008). "Multi-view learning in the presence of view disagreement"

**Applications:**
- Web page classification
- Natural language processing
- Multimodal learning

**Tools:**
- Custom implementations (no standard library support)
- Multi-view learning libraries
- Semi-supervised learning frameworks

---

## 🚀 Next Steps

**Try It Yourself:**
1. Identify two independent views in your data
2. Validate each view can train a reasonable classifier
3. Implement co-training
4. Compare with single-view supervised baseline

**Explore Related Methods:**
- [Self-Training](self-training.md) - Single view approach
- [Pseudo-Labeling](pseudo-labeling.md) - Deep learning approach
- [Main Index](index.md) - Overview of all methods

---

**Key Takeaway:** Co-training leverages multiple views to create robust semi-supervised learning. When views are truly independent and sufficient, co-training can significantly outperform single-view methods by allowing models to teach each other through their complementary perspectives.

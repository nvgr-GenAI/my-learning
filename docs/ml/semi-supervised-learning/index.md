# Semi-Supervised Learning

**Learn from limited labeled data by leveraging large amounts of unlabeled data.**

Semi-supervised learning combines a small amount of labeled data with a large amount of unlabeled data during training, bridging the gap between supervised and unsupervised learning.

---

## 🎯 What is Semi-Supervised Learning?

### The Challenge

**The Labeling Problem:**
- Labeled data is expensive and time-consuming to obtain
- Requires human expertise (medical diagnosis, legal documents)
- Manual annotation doesn't scale
- But we often have abundant unlabeled data

**Example: Email Spam Detection**
- Labeled: 100 emails (manually marked as spam/not spam)
- Unlabeled: 10,000 emails (no labels)
- **Question:** Can we use the 10,000 unlabeled emails to improve our spam detector?

**The Semi-Supervised Solution:**

Semi-supervised learning uses both types of data to build better models than using labeled data alone.

---

## 🔑 Key Concepts

### 1. The Semi-Supervised Assumption

Semi-supervised learning works under specific assumptions:

**Smoothness Assumption:**
- Points close to each other should have similar labels
- If two data points are in a high-density region, their labels should be similar

**Cluster Assumption:**
- Data tends to form discrete clusters
- Points in the same cluster are likely to have the same label

**Manifold Assumption:**
- High-dimensional data lies on a lower-dimensional manifold
- Unlabeled data helps discover the manifold structure

### 2. When to Use Semi-Supervised Learning

**Ideal Scenarios:**

✅ **Limited Labeled Data:**
- Medical imaging (few labeled X-rays, many unlabeled)
- Speech recognition (few transcribed audio, many untranscribed)
- Web content classification (few labeled pages, millions unlabeled)

✅ **Expensive Labeling:**
- Expert annotations required (legal, medical)
- Time-consuming manual labeling
- Privacy concerns with human labelers

✅ **Large Unlabeled Data Available:**
- Web scraping produces unlabeled data
- Sensor data continuously generated
- User-generated content without labels

**When NOT to Use:**

❌ Plenty of labeled data already available
❌ Unlabeled data violates semi-supervised assumptions
❌ Unlabeled data comes from different distribution
❌ Labeling is cheap and quick

---

## 📊 Semi-Supervised Learning Methods

### Overview of Techniques

| Method | Key Idea | Difficulty | Best For |
|--------|----------|------------|----------|
| **Self-Training** | Use model's own predictions as labels | 🟢 Easy | Simple problems, confident predictions |
| **Co-Training** | Train multiple models on different views | 🟡 Medium | Data with natural feature splits |
| **Pseudo-Labeling** | Assign confident predictions as labels | 🟢 Easy | Deep learning, large datasets |
| **Consistency Regularization** | Enforce consistent predictions | 🟡 Medium | Modern deep learning |
| **Graph-Based Methods** | Propagate labels through similarity graph | 🔴 Hard | Small to medium datasets |

---

## 🚀 Common Approaches

### 1. Self-Training (Wrapper Method)

**Process:**
1. Train initial model on labeled data
2. Use model to predict labels for unlabeled data
3. Add most confident predictions to training set
4. Retrain model
5. Repeat

**Example:**
```
Labeled: 100 spam emails
Unlabeled: 10,000 emails

Iteration 1: Train on 100 → Predict 10,000 → Add 500 confident → Retrain on 600
Iteration 2: Train on 600 → Predict 9,500 → Add 500 confident → Retrain on 1,100
...
```

[→ Learn More: Self-Training](self-training.md){ .md-button }

---

### 2. Co-Training

**Process:**
1. Split features into two views (e.g., image: color vs. texture)
2. Train two models independently on each view
3. Each model labels unlabeled data
4. Add confident predictions to other model's training set
5. Retrain both models

**Example: Web Page Classification**
- View 1: Page content (text)
- View 2: Hyperlinks pointing to the page
- Two models teach each other

[→ Learn More: Co-Training](co-training.md){ .md-button }

---

### 3. Pseudo-Labeling

**Process:**
1. Train model on labeled data
2. Generate pseudo-labels for all unlabeled data
3. Train jointly on real labels + pseudo-labels
4. Use cross-entropy loss with confidence weighting

**Modern Approach:**
- Popular in deep learning
- Often combined with data augmentation
- Used in competitions and practice

[→ Learn More: Pseudo-Labeling](pseudo-labeling.md){ .md-button }

---

## 🎓 Practical Example: Image Classification

### Scenario
- **Task:** Classify images of animals
- **Labeled data:** 1,000 images with labels
- **Unlabeled data:** 100,000 images without labels
- **Goal:** Build better classifier than using 1,000 alone

### Approach 1: Self-Training
```python
# Pseudo-code
model = train_model(labeled_data)  # Train on 1,000

for iteration in range(10):
    predictions = model.predict(unlabeled_data)
    confident = get_confident_predictions(predictions, threshold=0.95)
    labeled_data.add(confident)
    model = retrain_model(labeled_data)
```

### Approach 2: Pseudo-Labeling
```python
# Pseudo-code
model = initial_train(labeled_data)

# Generate pseudo-labels
pseudo_labels = model.predict(unlabeled_data)

# Train on combined dataset
combined_data = labeled_data + (unlabeled_data, pseudo_labels)
model = train_with_confidence_weighting(combined_data)
```

---

## 🔬 Advanced Techniques

### Consistency Regularization

**Idea:** Model should output similar predictions for perturbed versions of the same input

**Examples:**
- **Π-Model:** Encourage consistent predictions for same input with different dropout
- **Mean Teacher:** Use exponential moving average of model weights as teacher
- **UDA (Unsupervised Data Augmentation):** Consistency on strongly augmented data

### Graph-Based Methods

**Idea:** Build similarity graph, propagate labels from labeled to unlabeled nodes

**Process:**
1. Construct graph where nodes are data points
2. Edge weights represent similarity
3. Propagate labels through graph using smoothness assumption

---

## 📈 Performance Expectations

### Typical Improvements

**With Good Assumptions:**
- 5-20% accuracy improvement over supervised baseline
- Equivalent to 2-5x more labeled data

**Example Results:**
- Supervised (1,000 labels): 75% accuracy
- Semi-supervised (1,000 labels + 100,000 unlabeled): 82% accuracy
- Fully supervised (10,000 labels): 83% accuracy

**Key Insight:** Semi-supervised learning can achieve near-fully-supervised performance with far less labeling effort.

---

## ⚠️ Common Pitfalls

### 1. Confirmation Bias
**Problem:** Model reinforces its own mistakes
**Solution:** Use confidence thresholding, periodic resets

### 2. Distribution Mismatch
**Problem:** Unlabeled data from different distribution
**Example:** Training on indoor images, unlabeled outdoor images
**Solution:** Careful data inspection, domain adaptation techniques

### 3. Poor Initial Model
**Problem:** Weak initial model leads to poor pseudo-labels
**Solution:** Ensure strong baseline with labeled data first

### 4. Class Imbalance
**Problem:** Pseudo-labels may favor majority class
**Solution:** Balanced sampling, class-aware confidence thresholds

---

## 🎯 Best Practices

### 1. Start Simple
- Begin with supervised baseline
- Try self-training or pseudo-labeling first
- Measure improvement carefully

### 2. Validate Carefully
- Hold out labeled validation set
- Don't use unlabeled data for validation
- Monitor for confirmation bias

### 3. Confidence Thresholding
- Start with high thresholds (0.95+)
- Monitor pseudo-label quality
- Adjust based on validation performance

### 4. Iterative Refinement
- Add pseudo-labels gradually
- Periodically retrain from scratch
- Remove low-confidence predictions

---

## 🔗 Comparison with Other Approaches

| Approach | Labeled Data | Unlabeled Data | Use Case |
|----------|--------------|----------------|----------|
| **Supervised** | Required | Not used | Plenty of labels available |
| **Semi-Supervised** | Some required | Leveraged | Limited labels, plenty unlabeled |
| **Unsupervised** | Not used | Required | No labels available |
| **Transfer Learning** | Some required | Pre-trained on other task | Related task with labels exists |
| **Active Learning** | Iteratively acquired | Available | Can query labels strategically |

---

## 📚 Real-World Applications

### 1. Computer Vision
- **Medical Imaging:** Few labeled X-rays, many unlabeled scans
- **Satellite Imagery:** Limited ground truth, vast unlabeled data
- **Video Analysis:** Few annotated frames, hours of unlabeled video

### 2. Natural Language Processing
- **Text Classification:** Few labeled documents, millions unlabeled
- **Named Entity Recognition:** Limited annotations, large corpus
- **Sentiment Analysis:** Small labeled dataset, vast user reviews

### 3. Speech Recognition
- **Transcription:** Few transcribed audio, many untranscribed recordings
- **Speaker Identification:** Limited labeled speakers, large audio database
- **Accent Detection:** Few examples per accent, large speech corpus

### 4. Drug Discovery
- **Molecular Property Prediction:** Few tested compounds, millions untested
- **Protein Function:** Limited functional annotations, vast sequence data

---

## 🎓 Learning Path

### For Beginners
1. Start with [Self-Training](self-training.md) - simplest approach
2. Understand assumptions and when they hold
3. Practice with small datasets first

### For Intermediate
1. Learn [Co-Training](co-training.md) for multi-view data
2. Implement [Pseudo-Labeling](pseudo-labeling.md) in deep learning
3. Explore consistency regularization (Π-Model, Mean Teacher)

### For Advanced
1. Study graph-based methods
2. Implement UDA (Unsupervised Data Augmentation)
3. Research latest techniques (FixMatch, MixMatch)

---

## 📖 Additional Resources

**Classic Papers:**
- "Semi-Supervised Learning" by Chapelle et al. (2006)
- "Co-Training" by Blum and Mitchell (1998)
- "Temporal Ensembling" by Laine and Aila (2017)

**Modern Techniques:**
- FixMatch (2020) - State-of-the-art semi-supervised
- MixMatch (2019) - Combines multiple techniques
- UDA (2020) - Unsupervised Data Augmentation

**Tools & Libraries:**
- scikit-learn: `LabelPropagation`, `LabelSpreading`
- TorchSSL: Semi-supervised learning library for PyTorch
- LAMDA-SSL: Comprehensive semi-supervised learning toolkit

---

## 🚀 Next Steps

Ready to dive deeper? Choose a method to explore:

[Self-Training →](self-training.md){ .md-button .md-button--primary }
[Co-Training →](co-training.md){ .md-button }
[Pseudo-Labeling →](pseudo-labeling.md){ .md-button }

---

**Key Takeaway:** Semi-supervised learning bridges the gap between fully supervised and unsupervised learning, enabling us to build better models with limited labeled data by leveraging abundant unlabeled data.

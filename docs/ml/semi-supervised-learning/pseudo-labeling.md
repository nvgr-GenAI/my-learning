# Pseudo-Labeling

**Use model predictions as "pseudo-labels" to train on unlabeled data simultaneously with labeled data.**

Pseudo-labeling is a modern semi-supervised approach popular in deep learning that treats predicted labels as if they were true labels during training.

---

## 🎯 What is Pseudo-Labeling?

### Core Idea

**The Pseudo-Labeling Process:**
1. Train a model on labeled data
2. Use model to predict "pseudo-labels" for unlabeled data
3. **Key Difference:** Train on both labeled and unlabeled data simultaneously
4. Weight unlabeled examples by prediction confidence
5. Update model using combined loss

**vs. Self-Training:**
- **Self-Training:** Iterative (label → add → retrain → repeat)
- **Pseudo-Labeling:** Joint training (train on both datasets together)

### Intuition

**The Learning Signal:**
```
Labeled Data: We know the true label
→ Loss = CrossEntropy(prediction, true_label)

Unlabeled Data: We use predicted label as if it were true
→ Loss = CrossEntropy(prediction, pseudo_label) × confidence_weight
```

**Why It Works:**
- Model learns from both labeled and unlabeled data
- Confidence weighting reduces impact of wrong pseudo-labels
- End-to-end training (no iterative process)

---

## 🔑 Key Components

### 1. Pseudo-Label Generation

**How to create pseudo-labels:**

```python
def generate_pseudo_labels(model, X_unlabeled, threshold=0.95):
    """
    Generate pseudo-labels for unlabeled data

    Args:
        model: Trained classifier
        X_unlabeled: Unlabeled data
        threshold: Confidence threshold (optional)

    Returns:
        pseudo_labels: Predicted labels
        confidence: Prediction confidence
        mask: Boolean mask for confident predictions
    """
    # Get predictions
    probabilities = model.predict(X_unlabeled)

    # Pseudo-labels are the predicted class
    pseudo_labels = probabilities.argmax(axis=1)

    # Confidence is max probability
    confidence = probabilities.max(axis=1)

    # Optional: Filter by confidence
    mask = confidence >= threshold

    return pseudo_labels, confidence, mask
```

### 2. Confidence Weighting

**Why weight by confidence?**
- High confidence → likely correct → higher weight
- Low confidence → likely wrong → lower weight
- Reduces impact of incorrect pseudo-labels

**Weighting Strategies:**

**1. Hard Threshold:**
```python
# Only use confident predictions
weight = 1.0 if confidence > threshold else 0.0
```

**2. Soft Weighting:**
```python
# Weight proportional to confidence
weight = confidence  # [0, 1]
```

**3. Temperature Scaling:**
```python
# Sharper confidence distribution
weight = (confidence ** temperature)
```

### 3. Joint Training Loss

**Combined Loss Function:**

```
Total Loss = Supervised Loss + λ × Unsupervised Loss

Where:
  Supervised Loss = CrossEntropy(predictions_labeled, true_labels)
  Unsupervised Loss = Σ confidence[i] × CrossEntropy(predictions_unlabeled[i], pseudo_labels[i])
  λ = weight balancing labeled vs. unlabeled data
```

---

## 📊 The Pseudo-Labeling Algorithm

### Standard Algorithm

```
Algorithm: Pseudo-Labeling

Input:
  - L: Labeled dataset {(x, y)}
  - U: Unlabeled dataset {x}
  - model: Neural network
  - λ: Unsupervised loss weight
  - epochs: Training epochs

Output:
  - Trained model

1. For each epoch:
     a. Generate pseudo-labels for U:
        pseudo_labels, confidences = model.predict(U)

     b. For each mini-batch:
        - Sample labeled data: (x_l, y_l)
        - Sample unlabeled data: (x_u, ŷ_u, w_u)

        - Compute supervised loss:
          L_sup = CrossEntropy(model(x_l), y_l)

        - Compute unsupervised loss:
          L_unsup = Σ w_u × CrossEntropy(model(x_u), ŷ_u)

        - Total loss: L = L_sup + λ × L_unsup

        - Update model: model ← model - lr × ∇L

2. Return model
```

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def pseudo_labeling_train(model, labeled_loader, unlabeled_loader,
                          epochs=100, lambda_u=1.0, threshold=0.95):
    """
    Pseudo-labeling training for neural networks

    Args:
        model: PyTorch model
        labeled_loader: DataLoader for labeled data
        unlabeled_loader: DataLoader for unlabeled data
        epochs: Number of training epochs
        lambda_u: Weight for unsupervised loss
        threshold: Confidence threshold for pseudo-labels

    Returns:
        Trained model
    """
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(reduction='none')  # Per-sample loss

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        supervised_loss_sum = 0
        unsupervised_loss_sum = 0

        # Iterate through batches
        for (X_labeled, y_labeled), (X_unlabeled,) in \
            zip(labeled_loader, unlabeled_loader):

            optimizer.zero_grad()

            # Supervised loss (labeled data)
            outputs_labeled = model(X_labeled)
            loss_supervised = criterion(outputs_labeled, y_labeled).mean()

            # Generate pseudo-labels (unlabeled data)
            with torch.no_grad():
                outputs_unlabeled = model(X_unlabeled)
                probs = torch.softmax(outputs_unlabeled, dim=1)
                confidence, pseudo_labels = probs.max(dim=1)

                # Filter by confidence
                confident_mask = confidence >= threshold

            # Unsupervised loss (only for confident predictions)
            if confident_mask.sum() > 0:
                outputs_unlabeled_train = model(X_unlabeled[confident_mask])
                loss_unsupervised_per_sample = criterion(
                    outputs_unlabeled_train,
                    pseudo_labels[confident_mask]
                )

                # Weight by confidence
                weights = confidence[confident_mask]
                loss_unsupervised = (loss_unsupervised_per_sample * weights).mean()
            else:
                loss_unsupervised = torch.tensor(0.0)

            # Combined loss
            loss = loss_supervised + lambda_u * loss_unsupervised

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Track losses
            total_loss += loss.item()
            supervised_loss_sum += loss_supervised.item()
            unsupervised_loss_sum += loss_unsupervised.item()

        # Epoch summary
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Supervised Loss: {supervised_loss_sum:.4f}")
        print(f"  Unsupervised Loss: {unsupervised_loss_sum:.4f}")
        print(f"  Total Loss: {total_loss:.4f}")

    return model
```

---

## 🎓 Practical Example: Image Classification

### Scenario: CIFAR-10 Classification

**Dataset:**
- 4,000 labeled images (400 per class)
- 46,000 unlabeled images
- Goal: Classify into 10 categories

### Implementation

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Prepare data
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

labeled_dataset = CIFAR10Subset(labeled_data, transform=transform)
unlabeled_dataset = CIFAR10Subset(unlabeled_data, transform=transform)

labeled_loader = DataLoader(labeled_dataset, batch_size=64, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=256, shuffle=True)

# Train with pseudo-labeling
model = SimpleCNN()
model = pseudo_labeling_train(
    model,
    labeled_loader,
    unlabeled_loader,
    epochs=200,
    lambda_u=1.0,
    threshold=0.95
)

# Evaluate
test_accuracy = evaluate_model(model, test_loader)
print(f"Test Accuracy: {test_accuracy:.2%}")
```

### Results

```
Baseline (4,000 labels only): 65% accuracy
Pseudo-Labeling (4,000 + 46,000 unlabeled): 78% accuracy
Improvement: +13% absolute

Epoch 1:
  Supervised Loss: 2.1234
  Unsupervised Loss: 1.8765
  Confident Ratio: 12% (5,520 examples)

Epoch 50:
  Supervised Loss: 0.4567
  Unsupervised Loss: 0.6543
  Confident Ratio: 45% (20,700 examples)

Epoch 200:
  Supervised Loss: 0.1234
  Unsupervised Loss: 0.2345
  Confident Ratio: 78% (35,880 examples)
```

**Key Observation:** As model improves, more unlabeled examples become confident, creating a virtuous cycle.

---

## ✅ When Pseudo-Labeling Works Well

### Ideal Conditions

**1. Deep Learning Context**
- Neural networks (CNNs, Transformers)
- End-to-end differentiable models
- Large-scale datasets

**2. Clear Confidence Signals**
- Model produces well-calibrated probabilities
- High-confidence predictions are accurate
- Sufficient unlabeled data for low-confidence examples to learn from high-confidence ones

**3. Data Augmentation**
- Combining with augmentation is very effective
- Augmentation prevents overfitting on pseudo-labels
- Creates more diverse training samples

### Success Domains

**Computer Vision:**
- Image classification (especially with CNNs)
- Object detection
- Semantic segmentation

**Natural Language Processing:**
- Text classification
- Sequence labeling
- Language modeling fine-tuning

**Audio:**
- Speech recognition
- Audio classification
- Sound event detection

---

## ⚠️ Common Pitfalls and Solutions

### 1. Overconfident Wrong Predictions

**Problem:** Model is confident but wrong

**Example:**
```
Model predicts "dog" with 98% confidence
True label: "cat"
→ Strong incorrect gradient during training
```

**Solutions:**

**Label Smoothing:**
```python
def label_smoothing(labels, num_classes, smoothing=0.1):
    """
    Smooth pseudo-labels to reduce overconfidence

    Args:
        labels: Pseudo-labels
        num_classes: Number of classes
        smoothing: Smoothing factor (0.1 = 10%)

    Returns:
        Smoothed label distribution
    """
    confidence = 1.0 - smoothing
    smooth_value = smoothing / (num_classes - 1)

    # Create smoothed labels
    smoothed = torch.full((len(labels), num_classes), smooth_value)
    smoothed[range(len(labels)), labels] = confidence

    return smoothed
```

**Temporal Ensembling:**
```python
# Average predictions over time
ema_predictions = 0.99 * ema_predictions + 0.01 * current_predictions
pseudo_labels = ema_predictions.argmax(dim=1)
```

### 2. Class Imbalance in Pseudo-Labels

**Problem:** Model favors majority class in pseudo-labels

**Example:**
```
True distribution: 50% class A, 50% class B
Pseudo-labels: 80% class A, 20% class B
→ Bias amplifies over training
```

**Solutions:**

**Balanced Sampling:**
```python
def balanced_pseudo_label_sampling(pseudo_labels, batch_size):
    """Sample equal number per class"""
    samples_per_class = batch_size // num_classes
    indices = []

    for class_idx in range(num_classes):
        class_indices = (pseudo_labels == class_idx).nonzero(as_tuple=True)[0]
        selected = torch.randperm(len(class_indices))[:samples_per_class]
        indices.append(class_indices[selected])

    return torch.cat(indices)
```

**Class-Balanced Loss:**
```python
# Weight classes inversely by frequency
class_counts = torch.bincount(pseudo_labels)
class_weights = 1.0 / class_counts
loss_weights = class_weights[pseudo_labels]
loss = (criterion(outputs, pseudo_labels) * loss_weights).mean()
```

### 3. Confirmation Bias

**Problem:** Model reinforces initial biases

**Solution: Mixup Augmentation**
```python
def mixup(x1, y1, x2, y2, alpha=0.4):
    """
    Mixup augmentation for robustness

    Creates mixed samples: x = λ·x1 + (1-λ)·x2
    """
    lambda_ = np.random.beta(alpha, alpha)
    x_mixed = lambda_ * x1 + (1 - lambda_) * x2
    y_mixed = lambda_ * y1 + (1 - lambda_) * y2
    return x_mixed, y_mixed

# Use during training
x_mixed, y_mixed = mixup(x_labeled, y_labeled, x_unlabeled, pseudo_labels)
```

### 4. Poor Calibration

**Problem:** Confidence doesn't match accuracy

**Example:**
```
Model says 95% confident
Actual accuracy at 95% threshold: 75%
→ Many confident wrong predictions
```

**Solution: Temperature Scaling**
```python
def calibrate_probabilities(logits, temperature=2.0):
    """
    Scale logits to calibrate probabilities

    Higher temperature → softer probabilities
    """
    return torch.softmax(logits / temperature, dim=1)

# Find optimal temperature on validation set
best_temp = find_optimal_temperature(model, val_loader)
calibrated_probs = calibrate_probabilities(logits, temperature=best_temp)
```

---

## 🔬 Advanced Techniques

### 1. FixMatch (State-of-the-Art)

**Key Ideas:**
- Strong augmentation for unlabeled data
- Consistency regularization
- Confidence thresholding

```python
def fixmatch_loss(model, x_labeled, y_labeled, x_unlabeled, threshold=0.95):
    """
    FixMatch: Simplicity and effectiveness

    Args:
        model: Neural network
        x_labeled: Labeled batch
        y_labeled: True labels
        x_unlabeled: Unlabeled batch (weak augmentation)
        threshold: Confidence threshold
    """
    # Supervised loss
    logits_labeled = model(x_labeled)
    loss_supervised = F.cross_entropy(logits_labeled, y_labeled)

    # Generate pseudo-labels using weakly augmented unlabeled data
    with torch.no_grad():
        logits_weak = model(x_unlabeled)
        probs = torch.softmax(logits_weak, dim=1)
        confidence, pseudo_labels = probs.max(dim=1)
        mask = confidence >= threshold

    # Strongly augment unlabeled data
    x_unlabeled_strong = strong_augment(x_unlabeled)

    # Consistency loss
    if mask.sum() > 0:
        logits_strong = model(x_unlabeled_strong)
        loss_unsupervised = F.cross_entropy(
            logits_strong[mask],
            pseudo_labels[mask]
        )
    else:
        loss_unsupervised = torch.tensor(0.0)

    return loss_supervised + loss_unsupervised
```

### 2. MixMatch

**Combines:**
- Pseudo-labeling
- MixUp augmentation
- Sharpening predictions
- Temperature scaling

```python
def mixmatch(model, x_labeled, y_labeled, x_unlabeled, T=0.5, K=2, alpha=0.75):
    """
    MixMatch: Holistic approach combining multiple techniques

    Args:
        T: Temperature for sharpening
        K: Number of augmentations
        alpha: Mixup parameter
    """
    # Augment labeled data
    x_labeled_aug = augment(x_labeled)

    # Generate pseudo-labels with sharpening
    pseudo_labels = []
    for _ in range(K):
        x_unlabeled_aug = augment(x_unlabeled)
        with torch.no_grad():
            pred = model(x_unlabeled_aug)
            pseudo_labels.append(torch.softmax(pred, dim=1))

    # Average and sharpen
    pseudo_labels = torch.stack(pseudo_labels).mean(dim=0)
    pseudo_labels = sharpen(pseudo_labels, T)

    # MixUp both datasets
    x_all = torch.cat([x_labeled_aug, x_unlabeled])
    y_all = torch.cat([y_labeled, pseudo_labels])

    x_mixed, y_mixed = mixup(x_all, y_all, alpha=alpha)

    # Train on mixed data
    logits = model(x_mixed)
    loss = -torch.mean(torch.sum(y_mixed * F.log_softmax(logits, dim=1), dim=1))

    return loss
```

### 3. UDA (Unsupervised Data Augmentation)

**Key Idea:** Enforce consistency under strong augmentation

```python
def uda_loss(model, x_labeled, y_labeled, x_unlabeled, threshold=0.95):
    """
    UDA: Consistency under strong augmentation

    Key: Model should predict same for original and augmented
    """
    # Supervised loss
    loss_supervised = F.cross_entropy(model(x_labeled), y_labeled)

    # Generate pseudo-labels on original
    with torch.no_grad():
        logits_orig = model(x_unlabeled)
        probs = torch.softmax(logits_orig, dim=1)
        confidence, _ = probs.max(dim=1)
        mask = confidence >= threshold

    # Consistency loss with strongly augmented version
    if mask.sum() > 0:
        x_unlabeled_aug = strong_augment(x_unlabeled)
        logits_aug = model(x_unlabeled_aug)

        # KL divergence for consistency
        loss_consistency = F.kl_div(
            F.log_softmax(logits_aug[mask], dim=1),
            F.softmax(logits_orig[mask], dim=1),
            reduction='batchmean'
        )
    else:
        loss_consistency = torch.tensor(0.0)

    return loss_supervised + loss_consistency
```

---

## 🎯 Best Practices

### 1. Start with Supervised Baseline

```python
# Train supervised baseline first
model = train_supervised(X_labeled, y_labeled, epochs=100)
baseline_acc = evaluate(model, X_val, y_val)

# Then add pseudo-labeling
model_pl = pseudo_labeling_train(model, X_labeled, y_labeled, X_unlabeled)
pl_acc = evaluate(model_pl, X_val, y_val)

print(f"Improvement: {pl_acc - baseline_acc:.2%}")
```

### 2. Use Data Augmentation

```python
# Strong augmentation for unlabeled data
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor()
])
```

### 3. Anneal Unsupervised Weight

```python
def get_lambda_u(epoch, max_epochs, max_lambda=1.0, warmup_epochs=10):
    """
    Gradually increase weight of unsupervised loss

    Start low (model not confident) → increase over time
    """
    if epoch < warmup_epochs:
        return max_lambda * (epoch / warmup_epochs)
    else:
        return max_lambda

# During training
lambda_u = get_lambda_u(epoch, max_epochs)
loss = loss_supervised + lambda_u * loss_unsupervised
```

### 4. Monitor Pseudo-Label Quality

```python
def monitor_pseudo_labels(model, X_unlabeled, y_unlabeled_true):
    """
    If you have true labels, monitor pseudo-label accuracy
    (For validation/debugging only)
    """
    pseudo_labels = model.predict(X_unlabeled).argmax(axis=1)
    accuracy = (pseudo_labels == y_unlabeled_true).mean()

    print(f"Pseudo-label accuracy: {accuracy:.2%}")
    print(f"Confident ratio: {(model.predict_proba(X_unlabeled).max(axis=1) > 0.95).mean():.2%}")
```

---

## 📊 Performance Analysis

### Expected Improvements

**Typical Results:**

| Dataset | Labels | Baseline | Pseudo-Labeling | Improvement |
|---------|--------|----------|-----------------|-------------|
| CIFAR-10 | 250/class | 65% | 78% | +13% |
| CIFAR-10 | 4,000/class | 85% | 89% | +4% |
| ImageNet | 10% labels | 42% | 51% | +9% |
| Text (IMDb) | 1,000 | 82% | 88% | +6% |

**Key Insights:**
- Larger improvements with fewer labels
- Diminishing returns as labeled data increases
- Very effective for deep learning models

---

## 📚 References and Further Reading

**Foundational:**
- Lee, D. H. (2013). "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks"

**Modern Advances:**
- Sohn, K., et al. (2020). "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence"
- Berthelot, D., et al. (2019). "MixMatch: A Holistic Approach to Semi-Supervised Learning"
- Xie, Q., et al. (2020). "Unsupervised Data Augmentation for Consistency Training"

**Libraries:**
- PyTorch Lightning with semi-supervised modules
- TensorFlow Addons semi-supervised
- TorchSSL: https://github.com/TorchSSL/TorchSSL

---

## 🚀 Next Steps

**Implement It:**
1. Start with basic pseudo-labeling
2. Add confidence thresholding
3. Incorporate data augmentation
4. Try advanced methods (FixMatch, MixMatch)

**Compare Methods:**
- [Self-Training](self-training.md) - Iterative approach
- [Co-Training](co-training.md) - Multi-view approach
- [Main Index](index.md) - Overview

---

**Key Takeaway:** Pseudo-labeling is the modern way to do semi-supervised learning in deep learning. Combined with confidence thresholding, data augmentation, and consistency regularization (FixMatch, MixMatch, UDA), it achieves state-of-the-art results with limited labeled data.

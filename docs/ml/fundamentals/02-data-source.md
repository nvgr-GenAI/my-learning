# Chapter 2: Data - The Source of Intelligence

A machine learning model is only as good as the data it learns from. You can have the most sophisticated algorithm, the most powerful computer, and the best engineers—but if your data is poor, your model will fail. This fundamental truth makes data the most critical ingredient in machine learning.

In this chapter, we'll explore what makes data valuable, the different forms it takes, the quality issues that can derail your models, and how to properly evaluate performance. We'll also uncover one of the most insidious problems in machine learning: data leakage.

---

## 2.1 Data is Everything

### The Foundation of Learning

Remember from Chapter 1 that machine learning flips the traditional programming paradigm: instead of writing rules, we provide data and let the computer discover the patterns. This means the quality, quantity, and relevance of your data directly determines what your model can learn.

Think of data as the textbook from which your model studies. If the textbook is incomplete, incorrect, or irrelevant, no amount of studying will help. Similarly, if your data doesn't capture the patterns you need, your model cannot learn them.

### Structured vs Unstructured Data

Data comes in many forms, and understanding these forms helps you choose the right approaches for working with them.

**Structured Data** is organized into a fixed format with clearly defined types and relationships. It fits neatly into tables with rows and columns, where each column has a specific data type and meaning.

**Example:** Customer Database

| Customer_ID | Age | Income | Credit_Score | Purchased |
|-------------|-----|--------|--------------|-----------|
| 1001        | 25  | 45000  | 720          | Yes       |
| 1002        | 45  | 85000  | 680          | No        |
| 1003        | 35  | 65000  | 750          | Yes       |

```python
import pandas as pd

# Structured data is easy to work with
data = pd.DataFrame({
    'customer_id': [1001, 1002, 1003],
    'age': [25, 45, 35],
    'income': [45000, 85000, 65000],
    'credit_score': [720, 680, 750],
    'purchased': ['Yes', 'No', 'Yes']
})

# Access data by column
print(data['age'].mean())  # 35.0
print(data['income'].max())  # 85000
```

**Unstructured Data** lacks a predefined format. It's free-form and requires more preprocessing to extract meaningful patterns. Most of the world's data is unstructured.

Examples include:

- Text (emails, documents, social media)
- Images (photos, medical scans)
- Audio (speech, music)
- Video

**Example:** Text Data

```python
# Unstructured text data
emails = [
    "Congratulations! You've won $1,000,000. Click here now!!!",
    "Hi John, can we schedule a meeting for next Tuesday?",
    "FREE VIAGRA! NO PRESCRIPTION NEEDED!!!",
    "The quarterly report is attached. Please review."
]

# Requires processing to extract features
# We'll learn more about this in Chapter 3
```

### Types of Data

Let's explore the common data types you'll encounter in machine learning:

#### 1. Tabular Data

**Tabular data** is the most common form in machine learning. Each row represents an observation, and each column represents a feature. This is what you typically see in spreadsheets and databases.

**Example: Housing Data**

```python
import numpy as np
import pandas as pd

# Typical tabular dataset
housing_data = pd.DataFrame({
    'square_feet': [1500, 2000, 1200, 1800, 2200],
    'bedrooms': [3, 4, 2, 3, 4],
    'bathrooms': [2, 3, 1, 2, 3],
    'age_years': [10, 5, 15, 8, 3],
    'price': [300000, 450000, 250000, 380000, 520000]
})

print(housing_data.head())
print(f"\nDataset shape: {housing_data.shape[0]} rows, {housing_data.shape[1]} columns")
```

#### 2. Text Data

**Text data** includes documents, reviews, emails, social media posts, and any natural language content. Text must be converted to numerical form before machine learning algorithms can process it.

**Example: Sentiment Analysis**

```python
# Text reviews with sentiment labels
reviews = [
    ("This product is amazing! Highly recommend.", "positive"),
    ("Terrible quality. Waste of money.", "negative"),
    ("It's okay, nothing special.", "neutral"),
    ("Love it! Exceeded expectations.", "positive"),
    ("Broke after one week. Very disappointed.", "negative")
]

# Text needs special preprocessing
texts, sentiments = zip(*reviews)
print(f"Number of reviews: {len(texts)}")
print(f"Example: '{texts[0]}' -> {sentiments[0]}")
```

#### 3. Image Data

**Image data** consists of pixel arrays. Each pixel has values representing colors (RGB) or grayscale intensities. Images are high-dimensional data—a 224×224 color image has 150,528 features (224 × 224 × 3 color channels).

**Example: Image as Data**

```python
from PIL import Image
import numpy as np

# Load an image (conceptual example)
# In practice, you'd load from file: Image.open('photo.jpg')

# Simulate a small 4x4 grayscale image
image = np.array([
    [255, 200, 150, 100],
    [245, 190, 140, 95],
    [235, 180, 130, 90],
    [225, 170, 120, 85]
])

print(f"Image shape: {image.shape}")  # (4, 4)
print(f"Pixel values range: {image.min()} to {image.max()}")
print(f"Total features: {image.size}")

# For color images
color_image = np.random.randint(0, 256, size=(224, 224, 3))
print(f"\nColor image shape: {color_image.shape}")  # (224, 224, 3)
print(f"Total features: {color_image.size}")  # 150,528
```

#### 4. Time Series Data

**Time series data** consists of observations recorded over time at regular intervals. The order matters—time is a critical feature. Examples include stock prices, weather measurements, and sensor readings.

**Example: Temperature Readings**

```python
import pandas as pd

# Time series: temperature readings
dates = pd.date_range('2024-01-01', periods=10, freq='D')
temperatures = [32, 35, 31, 28, 30, 33, 36, 34, 32, 29]

time_series = pd.DataFrame({
    'date': dates,
    'temperature': temperatures
})

print(time_series)
print(f"\nData type: Time series with {len(time_series)} observations")
```

---

## 2.2 Data Quality Issues

Perfect data is rare. Real-world data is messy, incomplete, biased, and noisy. Understanding these issues is crucial because they directly impact model performance.

### Sampling Bias

**Sampling Bias** occurs when your training data doesn't represent the real-world population your model will encounter. The model learns patterns specific to the biased sample and fails to generalize.

**Example: Hiring Model Bias**

Suppose you're building a model to predict job performance based on historical data. If your company historically hired mostly from certain universities, your data is biased toward those schools. The model will incorrectly learn that attending those universities is essential for good performance.

```python
import pandas as pd

# Biased hiring data
biased_data = pd.DataFrame({
    'university': ['Stanford', 'MIT', 'Stanford', 'MIT', 'Stanford'],
    'gpa': [3.8, 3.9, 3.7, 3.85, 3.95],
    'performance': [85, 90, 82, 88, 92]
})

print("Biased sample - only prestigious schools")
print(biased_data)
print("\nProblem: Model will think university prestige predicts performance")
print("Reality: Sample doesn't include great performers from other schools")

# Better: diverse sample
diverse_data = pd.DataFrame({
    'university': ['Stanford', 'State U', 'MIT', 'Community', 'Stanford', 'State U'],
    'gpa': [3.8, 3.6, 3.9, 3.5, 3.7, 3.7],
    'performance': [85, 87, 90, 86, 82, 88]
})

print("\n\nDiverse sample - performance not tied to school prestige")
print(diverse_data)
```

### Noise

**Noise** refers to random errors or irrelevant variations in data. It can come from measurement errors, data entry mistakes, or natural randomness. Too much noise makes it hard for models to learn the true underlying patterns.

```python
import numpy as np
import matplotlib.pyplot as plt

# Clean signal: y = 2x + 1
x = np.linspace(0, 10, 50)
y_true = 2 * x + 1

# Noisy measurements
noise = np.random.normal(0, 2, size=len(x))
y_noisy = y_true + noise

# Extreme noise
extreme_noise = np.random.normal(0, 5, size=len(x))
y_very_noisy = y_true + extreme_noise

print(f"True relationship: y = 2x + 1")
print(f"Low noise example: y = {y_noisy[0]:.2f} (true: {y_true[0]:.2f})")
print(f"High noise example: y = {y_very_noisy[0]:.2f} (true: {y_true[0]:.2f})")
print(f"\nNoise makes it harder to learn the true pattern")
```

### Missing Values

**Missing values** occur when some data points lack information for certain features. This is extremely common in real-world datasets. How you handle missing values significantly impacts model performance.

**Example: Customer Data with Missing Values**

```python
import pandas as pd
import numpy as np

# Dataset with missing values
customers = pd.DataFrame({
    'age': [25, np.nan, 35, 45, np.nan, 55],
    'income': [45000, 55000, np.nan, 75000, 65000, 85000],
    'credit_score': [720, 680, 750, np.nan, 700, 780],
    'purchased': [1, 0, 1, 1, 0, 1]
})

print("Dataset with missing values:")
print(customers)
print(f"\nMissing values per column:")
print(customers.isnull().sum())

# Common strategies:

# 1. Drop rows with missing values
print(f"\nStrategy 1: Drop rows -> {len(customers.dropna())} rows remain")

# 2. Fill with mean
customers_filled = customers.copy()
customers_filled['age'].fillna(customers['age'].mean(), inplace=True)
customers_filled['income'].fillna(customers['income'].mean(), inplace=True)
customers_filled['credit_score'].fillna(customers['credit_score'].mean(), inplace=True)
print(f"\nStrategy 2: Fill with mean")
print(customers_filled)
```

### Outliers

**Outliers** are data points that differ significantly from other observations. They can be legitimate rare cases or errors. Outliers can distort model training, especially for algorithms sensitive to extreme values.

**Example: Salary Data with Outlier**

```python
import numpy as np

# Employee salaries (most are reasonable)
salaries = np.array([
    45000, 50000, 48000, 52000, 47000, 51000,
    49000, 53000, 46000, 5000000  # CEO salary - outlier!
])

print(f"Salaries: {salaries}")
print(f"Mean salary: ${salaries.mean():,.0f}")  # Distorted by outlier
print(f"Median salary: ${np.median(salaries):,.0f}")  # Robust to outlier

# Detecting outliers using IQR method
q1 = np.percentile(salaries, 25)
q3 = np.percentile(salaries, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = salaries[(salaries < lower_bound) | (salaries > upper_bound)]
print(f"\nOutliers detected: {outliers}")
```

### Data Augmentation

**Data Augmentation** is a technique to artificially increase the size of your training data by creating modified versions of existing examples. This is especially common with image data.

**Example: Image Augmentation**

```python
import numpy as np

# Simulate augmentation techniques
def augment_image(image):
    """
    Common augmentation techniques:
    - Rotation
    - Flipping
    - Brightness adjustment
    - Cropping
    - Adding noise
    """
    augmented = []

    # Original
    augmented.append(('original', image))

    # Horizontal flip
    augmented.append(('flipped', np.fliplr(image)))

    # Rotation (simplified)
    augmented.append(('rotated', np.rot90(image)))

    # Brightness adjustment (simplified)
    brighter = np.clip(image * 1.2, 0, 255)
    augmented.append(('brighter', brighter))

    return augmented

# Start with one image
original_image = np.random.randint(0, 256, size=(4, 4))
print(f"Original dataset size: 1 image")

# Augment to create 4 versions
augmented_images = augment_image(original_image)
print(f"After augmentation: {len(augmented_images)} images")
print(f"\nAugmentation techniques used:")
for name, _ in augmented_images:
    print(f"  - {name}")
```

---

## 2.3 The Evaluation Problem

How do you know if your model is any good? You need to test it—but not on the same data you used for training. This leads to one of the most important concepts in machine learning: splitting your data properly.

### The Problem with Training Data

If you evaluate your model on the same data it was trained on, you're testing memorization, not learning. A model might perfectly memorize all training examples but fail completely on new data.

**Example: Memorization vs Learning**

```python
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Simple dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 1, 1, 1])

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Test on training data
train_accuracy = model.score(X, y)
print(f"Training accuracy: {train_accuracy * 100:.0f}%")
print("But does it generalize to new data? We don't know!")
```

### Training, Validation, and Test Sets

To properly evaluate models, we split data into separate sets:

**Training Set**: Data used to train the model. The model sees these examples and learns patterns from them.

**Validation Set**: Data used to tune the model and make decisions during development (like choosing hyperparameters). The model doesn't train on this data, but we use it to guide our choices.

**Test Set**: Data held back until the very end. This gives an unbiased estimate of how the model performs on truly unseen data. Use this only once at the end.

**Example: Three-Way Split**

```python
from sklearn.model_selection import train_test_split
import numpy as np

# Generate sample data
X = np.random.rand(1000, 5)  # 1000 examples, 5 features
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Binary labels

# First split: separate test set (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Second split: separate validation set from remaining data (25% of 80% = 20%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)

print("Data split:")
print(f"Training set: {len(X_train)} examples ({len(X_train)/len(X)*100:.0f}%)")
print(f"Validation set: {len(X_val)} examples ({len(X_val)/len(X)*100:.0f}%)")
print(f"Test set: {len(X_test)} examples ({len(X_test)/len(X)*100:.0f}%)")
print(f"\nTotal: {len(X_train) + len(X_val) + len(X_test)} examples")
```

### Cross-Validation

**Cross-Validation** is a more sophisticated evaluation technique that maximizes the use of your data. Instead of a single train/test split, you divide data into multiple "folds" and train/test multiple times, rotating which fold is used for testing.

**K-Fold Cross-Validation** divides data into K equal-sized folds. The model is trained K times, each time using K-1 folds for training and 1 fold for testing. The final performance is the average across all K runs.

**Example: 5-Fold Cross-Validation**

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Sample data
X = np.random.rand(100, 5)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

# Model
model = RandomForestClassifier(n_estimators=10, random_state=42)

# 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)

print("5-Fold Cross-Validation")
print(f"Scores from each fold: {scores}")
print(f"Mean accuracy: {scores.mean():.3f}")
print(f"Standard deviation: {scores.std():.3f}")
print(f"\nThis gives more reliable estimate than single split")
```

**Why Cross-Validation?**

```python
# Single split can be lucky or unlucky
from sklearn.model_selection import train_test_split

# Same data as before
X = np.random.rand(100, 5)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

# Try 5 different random splits
print("Single split results (can vary):")
for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=i
    )
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Split {i+1}: {score:.3f}")

print("\nCross-validation averages across multiple splits")
print("More stable and reliable estimate")
```

---

## 2.4 Data Leakage

Data leakage is one of the most dangerous and subtle problems in machine learning. It occurs when information from outside the training data is used to create the model. This leads to artificially high performance during development but failure in production.

### What is Data Leakage?

**Data Leakage** happens when your training data contains information about the target variable that won't be available when making predictions on new data. The model appears to work great during testing but fails in the real world because it learned from information it shouldn't have had access to.

There are two main types:

1. **Target Leakage**: Features that include information about the target
2. **Train-Test Contamination**: Information from test data leaking into training

### Target Leakage Examples

**Example 1: Medical Diagnosis**

```python
import pandas as pd

# BAD: Data with leakage
# Suppose we're predicting pneumonia from medical records
leaky_data = pd.DataFrame({
    'age': [45, 67, 34, 56],
    'fever': [1, 1, 0, 1],
    'cough': [1, 1, 1, 1],
    'took_antibiotic': [1, 1, 0, 1],  # LEAKAGE! Only taken AFTER diagnosis
    'pneumonia': [1, 1, 0, 1]
})

print("Data with leakage:")
print(leaky_data)
print("\nProblem: 'took_antibiotic' is only known AFTER diagnosis")
print("It's a consequence of pneumonia, not a predictor")

# GOOD: Data without leakage
clean_data = pd.DataFrame({
    'age': [45, 67, 34, 56],
    'fever': [1, 1, 0, 1],
    'cough': [1, 1, 1, 1],
    'smoking_history': [1, 1, 0, 1],  # Known before diagnosis
    'pneumonia': [1, 1, 0, 1]
})

print("\n\nData without leakage:")
print(clean_data)
print("\nAll features are known BEFORE diagnosis")
```

**Example 2: Credit Card Fraud**

```python
# BAD: Leakage in fraud detection
fraud_data_bad = pd.DataFrame({
    'transaction_amount': [100, 5000, 50, 10000],
    'merchant_category': ['grocery', 'jewelry', 'gas', 'electronics'],
    'distance_from_home': [2, 500, 5, 1000],
    'dispute_filed': [0, 1, 0, 1],  # LEAKAGE! Only known after fraud detected
    'fraud': [0, 1, 0, 1]
})

print("Leaky fraud detection data:")
print(fraud_data_bad)
print("\nProblem: Dispute is filed AFTER discovering fraud")

# GOOD: No leakage
fraud_data_good = pd.DataFrame({
    'transaction_amount': [100, 5000, 50, 10000],
    'merchant_category': ['grocery', 'jewelry', 'gas', 'electronics'],
    'distance_from_home': [2, 500, 5, 1000],
    'time_since_last_transaction': [24, 2, 48, 1],  # Known at transaction time
    'fraud': [0, 1, 0, 1]
})

print("\n\nClean fraud detection data:")
print(fraud_data_good)
```

### Train-Test Contamination

This occurs when you preprocess data before splitting into train/test sets, allowing information from the test set to influence training.

**Example: Scaling Before Splitting**

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Generate data
X = np.random.randn(100, 3)
y = (X[:, 0] > 0).astype(int)

print("=== WRONG: Scaling before split (LEAKAGE) ===")
# Scale entire dataset first
scaler_bad = StandardScaler()
X_scaled_bad = scaler_bad.fit_transform(X)  # Uses ALL data

# Then split
X_train_bad, X_test_bad, y_train, y_test = train_test_split(
    X_scaled_bad, y, test_size=0.2, random_state=42
)

print("Problem: Test set statistics influenced the scaling of training data")
print(f"Training set shape: {X_train_bad.shape}")
print(f"Test set shape: {X_test_bad.shape}")

print("\n=== CORRECT: Split first, then scale (NO LEAKAGE) ===")
# Split first
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale separately
scaler_good = StandardScaler()
X_train_scaled = scaler_good.fit_transform(X_train)  # Fit only on training
X_test_scaled = scaler_good.transform(X_test)  # Apply same transformation

print("Correct: Scaler learned only from training data")
print("Test data transformed using training statistics")
```

### Detecting and Preventing Leakage

**Key Questions to Ask:**

1. Would this feature be available at prediction time?
2. Is this feature a consequence of the target, not a cause?
3. Did I process data before splitting train/test?
4. Am I using future information to predict the past?

**Example: Time Series Leakage**

```python
import pandas as pd

# Stock price prediction
stock_data = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=5, freq='D'),
    'price': [100, 102, 98, 103, 105],
    'volume': [1000, 1100, 900, 1200, 1050]
})

# BAD: Using future prices
stock_data['price_tomorrow'] = stock_data['price'].shift(-1)
print("Data with leakage (using future prices):")
print(stock_data)
print("\nProblem: Using tomorrow's price to predict today's movement!")

# GOOD: Using only past information
stock_data['price_yesterday'] = stock_data['price'].shift(1)
stock_data['avg_price_7d'] = stock_data['price'].rolling(window=7, min_periods=1).mean()
print("\n\nData without leakage (using only past):")
print(stock_data[['date', 'price', 'price_yesterday', 'avg_price_7d']])
```

---

## Summary

Data is the foundation of machine learning—everything else builds on it:

**Data Forms Matter:** Data comes in structured (tabular) and unstructured (text, images, audio) forms. Understanding these types helps you choose appropriate preprocessing and modeling approaches. Time series data adds temporal dependencies that must be respected.

**Quality Is Critical:** Real-world data suffers from sampling bias, noise, missing values, and outliers. Each quality issue requires specific strategies—from collecting more diverse samples to robust preprocessing techniques. Data augmentation can help when you need more training examples.

**Proper Evaluation:** Never evaluate on training data alone. Split your data into training, validation, and test sets. Use cross-validation for robust performance estimates. The test set should remain untouched until final evaluation to get an honest assessment of real-world performance.

**Beware of Leakage:** Data leakage is subtle but devastating. It occurs when information that won't be available at prediction time leaks into training. Always ask: "Would I have this information in production?" Process data only after splitting train/test. Respect time order in temporal data.

With a solid understanding of data—its forms, quality issues, and proper evaluation—you're ready to learn how to transform this data into a form that machines can learn from. That's the focus of our next chapter.

---

[← Previous: Chapter 1](01-learning-problem.md) | [Back to Index](index.md) | [Next: Chapter 3 →](03-numerical-representation.md)

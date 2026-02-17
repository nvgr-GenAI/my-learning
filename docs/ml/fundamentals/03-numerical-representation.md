# Chapter 3: Making Data Machine-Readable

Computers understand numbers, not concepts. They don't know what "red" means, what "California" represents, or how to interpret the word "excellent." To apply machine learning, we must transform our real-world data—with all its categories, text, and images—into numerical form that algorithms can process.

This transformation is both an art and a science. Done well, it preserves the meaningful patterns in your data. Done poorly, it can destroy the very relationships you're trying to learn. In this chapter, we'll explore how to represent different types of data numerically, why scaling matters, and how thoughtful feature engineering can dramatically improve model performance.

---

## 3.1 Feature Representation

### What Are Features?

**Features** (also called **independent variables** or **predictors**) are the measurable properties or characteristics we use as inputs to our machine learning models. They're the information the model uses to make predictions.

Think of features as the questions you'd ask to make a decision. To predict if someone will default on a loan, you might ask: What's their income? Credit score? Employment history? Each answer is a feature.

```python
import pandas as pd

# Each column (except the target) is a feature
loan_data = pd.DataFrame({
    'income': [50000, 75000, 35000, 90000],
    'credit_score': [720, 680, 650, 780],
    'years_employed': [5, 10, 2, 15],
    'existing_debt': [10000, 20000, 5000, 15000],
    'defaulted': [0, 0, 1, 0]  # Target variable (not a feature)
})

# Features: the inputs we use for prediction
features = ['income', 'credit_score', 'years_employed', 'existing_debt']
X = loan_data[features]

# Target: what we're trying to predict
y = loan_data['defaulted']

print("Features shape:", X.shape)  # (4 samples, 4 features)
print("Target shape:", y.shape)    # (4 samples)
```

### Feature Engineering, Selection, and Extraction

These three concepts are central to working with features:

**Feature Engineering** is the process of creating new features from existing data using domain knowledge. It's about extracting more meaningful information that helps the model learn better.

```python
import numpy as np

# Original features
data = pd.DataFrame({
    'price': [100, 150, 200],
    'quantity': [2, 3, 1]
})

# Feature engineering: create new features
data['total_cost'] = data['price'] * data['quantity']
data['price_per_unit'] = data['price'] / data['quantity']
data['is_bulk_purchase'] = (data['quantity'] > 2).astype(int)

print("Original features: price, quantity")
print("Engineered features: total_cost, price_per_unit, is_bulk_purchase")
print(data)
```

**Feature Selection** is choosing which features to keep and which to discard. Not all features are useful—some add noise, some are redundant, and some are irrelevant.

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.datasets import make_classification

# Generate data with many features, only some are useful
X, y = make_classification(n_samples=100, n_features=20, n_informative=5,
                          n_redundant=10, random_state=42)

print(f"Original: {X.shape[1]} features")

# Select the 5 most informative features
selector = SelectKBest(f_classif, k=5)
X_selected = selector.fit_transform(X, y)

print(f"After selection: {X_selected.shape[1]} features")
print("Keeping only the most predictive features")
```

**Feature Extraction** is transforming features into a new, typically lower-dimensional representation that captures the essential information. Unlike selection (which keeps some features and discards others), extraction creates entirely new features.

```python
from sklearn.decomposition import PCA

# Original high-dimensional data
X = np.random.rand(100, 50)  # 100 samples, 50 features

# Extract 10 principal components
pca = PCA(n_components=10)
X_extracted = pca.fit_transform(X)

print(f"Original: {X.shape[1]} features")
print(f"Extracted: {X_extracted.shape[1]} new features")
print(f"Variance captured: {pca.explained_variance_ratio_.sum():.2%}")
```

---

## 3.2 Encoding Categorical Variables

Categorical variables represent discrete categories or groups—like colors, cities, or yes/no answers. Machine learning algorithms need numbers, so we must encode these categories numerically. The encoding method you choose matters greatly.

### Label Encoding

**Label Encoding** assigns each category a unique integer. Simple, but dangerous—it implies an ordering that may not exist.

```python
from sklearn.preprocessing import LabelEncoder

# Categories: shirt sizes
sizes = ['Small', 'Medium', 'Large', 'Medium', 'Small', 'Large']

encoder = LabelEncoder()
sizes_encoded = encoder.fit_transform(sizes)

print("Original:", sizes)
print("Encoded:", sizes_encoded)
print("\nMapping:")
for i, size in enumerate(encoder.classes_):
    print(f"  {size} -> {i}")

print("\nProblem: Implies Large (2) > Medium (1) > Small (0)")
print("The model might think Large is 'twice' Small!")
```

**When to use Label Encoding:** Only for ordinal data where the order matters (e.g., "Low", "Medium", "High" or education levels).

```python
# Good use case: ordinal data
education = ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor']

# Manual ordinal encoding (preserving meaningful order)
education_map = {
    'High School': 0,
    'Bachelor': 1,
    'Master': 2,
    'PhD': 3
}

education_encoded = [education_map[e] for e in education]
print("Education levels:", education)
print("Ordinal encoding:", education_encoded)
print("Order makes sense: PhD (3) > Master (2) > Bachelor (1) > High School (0)")
```

### One-Hot Encoding

**One-Hot Encoding** creates a binary column for each category. Each sample has a 1 in one column and 0s elsewhere. This avoids implying false orderings.

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Categories: colors
colors = np.array(['Red', 'Blue', 'Green', 'Blue', 'Red']).reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)
colors_encoded = encoder.fit_transform(colors)

print("Original colors:")
print(colors.flatten())
print("\nOne-hot encoded:")
print(colors_encoded)
print("\nColumn names:", encoder.categories_[0])
print("Each color gets its own binary column")
```

**Visual Example:**

```python
import pandas as pd

# Easier to see with pandas
df = pd.DataFrame({'color': ['Red', 'Blue', 'Green', 'Blue', 'Red']})

# One-hot encoding with pandas
df_encoded = pd.get_dummies(df, columns=['color'], prefix='color')

print("Original:")
print(df)
print("\nOne-hot encoded:")
print(df_encoded)
```

**When to use One-Hot Encoding:** For nominal categories with no inherent order (colors, countries, product types).

**Caution:** Creates many features if you have many categories.

```python
# Problem: many categories = many features
cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
# If you have 1000 cities, one-hot encoding creates 1000 features!

print(f"Original: 1 feature (city)")
print(f"After one-hot: {len(cities)} features")
print("High cardinality can cause 'curse of dimensionality'")
```

### Ordinal Encoding

**Ordinal Encoding** is like label encoding but explicitly for ordered categories. You control the mapping to ensure it reflects the true order.

```python
from sklearn.preprocessing import OrdinalEncoder

# Ordered categories
satisfaction = np.array([
    ['Very Unsatisfied'],
    ['Unsatisfied'],
    ['Neutral'],
    ['Satisfied'],
    ['Very Satisfied'],
    ['Neutral']
])

# Define the order explicitly
categories = [['Very Unsatisfied', 'Unsatisfied', 'Neutral', 'Satisfied', 'Very Satisfied']]

encoder = OrdinalEncoder(categories=categories)
satisfaction_encoded = encoder.fit_transform(satisfaction)

print("Original:")
print(satisfaction.flatten())
print("\nOrdinal encoding:")
print(satisfaction_encoded.flatten())
print("\nMaintains order: 0 < 1 < 2 < 3 < 4")
```

---

## 3.3 Encoding Text and Images

Text and images are everywhere, but they require special encoding techniques to become machine-readable features.

### Encoding Text

Text is unstructured and high-dimensional. We need to convert words into numbers while preserving meaning.

#### Bag of Words (BoW)

**Bag of Words** treats text as a collection of words, ignoring grammar and order. Each unique word becomes a feature, and the value is how many times it appears.

```python
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "I love machine learning",
    "Machine learning is amazing",
    "I love Python and machine learning"
]

# Create bag of words
vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(documents)

# Show the vocabulary
print("Vocabulary (unique words):")
print(vectorizer.get_feature_names_out())

print("\nBag of Words matrix:")
print(bow_matrix.toarray())
print("\nEach row is a document, each column is a word count")

# Convert to DataFrame for clarity
import pandas as pd
bow_df = pd.DataFrame(
    bow_matrix.toarray(),
    columns=vectorizer.get_feature_names_out()
)
print("\nAs DataFrame:")
print(bow_df)
```

**Limitation:** Ignores word order. "dog bites man" and "man bites dog" look identical.

#### TF-IDF (Term Frequency-Inverse Document Frequency)

**TF-IDF** improves on Bag of Words by weighing words based on importance. Common words like "the" and "is" get lower weights; rare, informative words get higher weights.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "Cats and dogs are great pets"
]

# TF-IDF encoding
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(documents)

print("TF-IDF matrix:")
print(tfidf_matrix.toarray())
print("\nVocabulary:")
print(tfidf.get_feature_names_out())

# Show as DataFrame
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf.get_feature_names_out()
)
print("\nAs DataFrame (rounded):")
print(tfidf_df.round(2))
print("\nCommon words (the, on) have lower values")
print("Unique words (cats, log) have higher values")
```

#### Word Embeddings

**Embeddings** are dense, low-dimensional representations where similar words have similar vectors. Words like "king" and "queen" end up close together in vector space.

```python
# Conceptual example (actual embeddings require trained models)
import numpy as np

# Simplified 3D embeddings (real embeddings are 100-300 dimensions)
word_vectors = {
    'king': np.array([0.8, 0.1, 0.9]),
    'queen': np.array([0.7, 0.15, 0.85]),
    'man': np.array([0.6, 0.0, 0.5]),
    'woman': np.array([0.55, 0.05, 0.45]),
    'dog': np.array([-0.3, 0.8, 0.2])
}

# Similar words have similar vectors
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print("Similarity between words:")
print(f"king-queen: {cosine_similarity(word_vectors['king'], word_vectors['queen']):.3f}")
print(f"king-man: {cosine_similarity(word_vectors['king'], word_vectors['man']):.3f}")
print(f"king-dog: {cosine_similarity(word_vectors['king'], word_vectors['dog']):.3f}")
print("\nKing is most similar to queen, least similar to dog")
```

### Encoding Images

Images are already numbers (pixels), but they need proper representation.

```python
import numpy as np

# Grayscale image: 2D array of pixel intensities (0-255)
grayscale_image = np.array([
    [255, 200, 150],
    [200, 150, 100],
    [150, 100, 50]
])

print("Grayscale image (3x3):")
print(grayscale_image)
print(f"Shape: {grayscale_image.shape}")  # (height, width)
print(f"Pixel range: {grayscale_image.min()} to {grayscale_image.max()}")

# Color image: 3D array (height, width, channels)
color_image = np.random.randint(0, 256, size=(224, 224, 3))
print(f"\nColor image shape: {color_image.shape}")  # (height, width, 3)
print(f"Total pixels: {224 * 224}")
print(f"Total values: {224 * 224 * 3} (RGB)")

# Flatten for traditional ML (not needed for CNNs)
flattened = grayscale_image.flatten()
print(f"\nFlattened shape: {flattened.shape}")
print(f"Flattened: {flattened}")
```

**Normalization** scales pixel values to a standard range:

```python
# Original: 0-255
image = np.array([[0, 128, 255], [64, 192, 32]])

# Normalize to 0-1
normalized = image / 255.0

print("Original (0-255):")
print(image)
print("\nNormalized (0-1):")
print(normalized)
print("\nHelps neural networks train faster and more stable")
```

---

## 3.4 Feature Scaling

Different features often have vastly different ranges. Income might be in thousands, while age is in tens. This disparity can cause problems for many algorithms.

### Why Scaling Matters

Many algorithms (like gradient descent, k-NN, SVM) are sensitive to feature scales. Features with larger values can dominate the learning process.

```python
import pandas as pd
import numpy as np

# Features with different scales
data = pd.DataFrame({
    'age': [25, 35, 45, 55],
    'income': [30000, 50000, 75000, 100000],
    'credit_score': [650, 700, 750, 800]
})

print("Original data (different scales):")
print(data)
print("\nRanges:")
for col in data.columns:
    print(f"  {col}: {data[col].min()} to {data[col].max()}")

print("\nProblem: Income dominates due to large values")
print("Distance calculations heavily weighted by income")
```

### Standardization (Z-Score Normalization)

**Standardization** transforms features to have mean=0 and standard deviation=1. It preserves the shape of the distribution but changes the scale.

Formula: `z = (x - μ) / σ` where μ is mean and σ is standard deviation.

```python
from sklearn.preprocessing import StandardScaler

# Original data
X = np.array([[25, 30000], [35, 50000], [45, 75000], [55, 100000]])

scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

print("Original data:")
print(X)
print("\nStandardized data (mean=0, std=1):")
print(X_standardized)
print("\nMean per feature:", X_standardized.mean(axis=0))
print("Std per feature:", X_standardized.std(axis=0))
```

**When to use Standardization:**

- When features follow a normal (Gaussian) distribution
- For algorithms that assume normally distributed data (Linear Regression, Logistic Regression)
- When you want to preserve outliers' relative importance

### Normalization (Min-Max Scaling)

**Normalization** (Min-Max Scaling) scales features to a fixed range, typically [0, 1].

Formula: `x_scaled = (x - x_min) / (x_max - x_min)`

```python
from sklearn.preprocessing import MinMaxScaler

# Original data
X = np.array([[25, 30000], [35, 50000], [45, 75000], [55, 100000]])

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

print("Original data:")
print(X)
print("\nNormalized data (range [0, 1]):")
print(X_normalized)
print("\nMin per feature:", X_normalized.min(axis=0))
print("Max per feature:", X_normalized.max(axis=0))
```

**When to use Normalization:**

- When you need a bounded range (e.g., [0, 1])
- When features don't follow a normal distribution
- For algorithms that need bounded inputs (neural networks with sigmoid/tanh)
- When you want all features to contribute equally

### Comparison

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

# Sample data
data = np.array([[10], [20], [30], [100]])  # Note the outlier (100)

# Standardization
standard_scaler = StandardScaler()
data_standardized = standard_scaler.fit_transform(data)

# Normalization
minmax_scaler = MinMaxScaler()
data_normalized = minmax_scaler.fit_transform(data)

print("Original | Standardized | Normalized")
print("-" * 40)
for orig, stand, norm in zip(data, data_standardized, data_normalized):
    print(f"{orig[0]:8.0f} | {stand[0]:12.2f} | {norm[0]:10.2f}")

print("\nStandardization: Outlier still stands out")
print("Normalization: Outlier compressed to max value (1.0)")
```

---

## 3.5 Feature Engineering

Creating new features from existing ones can dramatically improve model performance. Good feature engineering requires domain knowledge and creativity.

### Creating Interaction Features

Sometimes the interaction between features matters more than individual features.

```python
import numpy as np
import pandas as pd

# Housing data
data = pd.DataFrame({
    'bedrooms': [2, 3, 4, 3],
    'bathrooms': [1, 2, 2, 3],
    'square_feet': [1000, 1500, 2000, 1800]
})

# Engineer new features
data['bed_bath_ratio'] = data['bedrooms'] / data['bathrooms']
data['sqft_per_bedroom'] = data['square_feet'] / data['bedrooms']
data['total_rooms'] = data['bedrooms'] + data['bathrooms']

print("Original features:")
print(data[['bedrooms', 'bathrooms', 'square_feet']])
print("\nEngineered features:")
print(data[['bed_bath_ratio', 'sqft_per_bedroom', 'total_rooms']])
```

### Domain-Specific Features

Use your knowledge of the problem to create meaningful features.

```python
import pandas as pd

# E-commerce transaction data
transactions = pd.DataFrame({
    'customer_id': [1, 1, 2, 2, 3],
    'purchase_amount': [50, 150, 30, 200, 75],
    'timestamp': pd.to_datetime([
        '2024-01-01 10:00', '2024-01-05 14:30',
        '2024-01-02 09:15', '2024-01-03 16:45',
        '2024-01-01 11:20'
    ])
})

# Engineer domain-specific features
transactions['hour_of_day'] = transactions['timestamp'].dt.hour
transactions['day_of_week'] = transactions['timestamp'].dt.dayofweek
transactions['is_weekend'] = (transactions['day_of_week'] >= 5).astype(int)
transactions['is_evening'] = (transactions['hour_of_day'] >= 18).astype(int)

print("Original data:")
print(transactions[['customer_id', 'purchase_amount', 'timestamp']])
print("\nEngineered features:")
print(transactions[['hour_of_day', 'day_of_week', 'is_weekend', 'is_evening']])
```

### Polynomial Features

Create polynomial and interaction terms automatically.

```python
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Simple features: x1, x2
X = np.array([[2, 3], [3, 4], [4, 5]])

# Create polynomial features (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

print("Original features (x1, x2):")
print(X)
print("\nPolynomial features (degree 2):")
print("Features:", poly.get_feature_names_out(['x1', 'x2']))
print(X_poly)
print("\nGenerated: x1, x2, x1², x1·x2, x2²")
```

**Caution:** Polynomial features can quickly explode the feature space.

```python
# Feature explosion
X_simple = np.random.rand(5, 3)  # 3 features

poly_2 = PolynomialFeatures(degree=2)
poly_3 = PolynomialFeatures(degree=3)

print(f"Original: {X_simple.shape[1]} features")
print(f"Degree 2: {poly_2.fit_transform(X_simple).shape[1]} features")
print(f"Degree 3: {poly_3.fit_transform(X_simple).shape[1]} features")
print("\nFeature count grows rapidly!")
```

---

## 3.6 The Curse of Dimensionality

As the number of features (dimensions) increases, several problems emerge that make machine learning harder. This phenomenon is called the **curse of dimensionality**.

### What Is It?

**The Curse of Dimensionality** refers to various phenomena that arise when working with high-dimensional data. As dimensions increase, data becomes increasingly sparse, and many concepts that work well in low dimensions break down.

```python
import numpy as np

# Demonstrate sparsity in high dimensions
def volume_of_unit_sphere(dimensions):
    """Volume of unit sphere in d dimensions (simplified)"""
    if dimensions == 1:
        return 2.0  # Line segment [-1, 1]
    elif dimensions == 2:
        return np.pi  # Circle
    elif dimensions == 3:
        return (4/3) * np.pi  # Sphere
    else:
        # Approximation for higher dimensions
        return (np.pi ** (dimensions/2)) / np.math.factorial(dimensions//2)

# Volume of cube is always 2^d (from -1 to 1 in each dimension)
print("Dimension | Cube Volume | Sphere Volume | Sphere/Cube Ratio")
print("-" * 65)
for d in [1, 2, 3, 5, 10]:
    cube_vol = 2 ** d
    sphere_vol = volume_of_unit_sphere(d)
    ratio = sphere_vol / cube_vol
    print(f"{d:9d} | {cube_vol:11.1f} | {sphere_vol:13.2f} | {ratio:17.4f}")

print("\nAs dimensions increase, data concentrates in corners of the space")
print("Points become equidistant (all far apart)")
```

### Why It Matters

**Distance becomes meaningless:** In high dimensions, the difference between the nearest and farthest neighbor becomes negligible. Algorithms relying on distance (like k-NN) fail.

```python
import numpy as np

def average_distance_ratio(n_dimensions, n_points=1000):
    """Ratio of max to min distance in random data"""
    data = np.random.rand(n_points, n_dimensions)
    distances = np.linalg.norm(data[0] - data[1:], axis=1)
    return distances.max() / distances.min()

print("Dimension | Max/Min Distance Ratio")
print("-" * 35)
for dim in [2, 10, 50, 100, 500]:
    ratio = average_distance_ratio(dim)
    print(f"{dim:9d} | {ratio:22.2f}")

print("\nIn high dimensions, all points are roughly equidistant")
print("'Nearest' neighbor is almost as far as 'farthest' neighbor")
```

**Data becomes sparse:** To maintain the same density of points, you need exponentially more data as dimensions increase.

```python
# How much data needed for same density?
def samples_needed(dimensions, density=10):
    """Samples needed to maintain density in d dimensions"""
    return density ** dimensions

print("Dimension | Samples Needed (for 10 points per dimension)")
print("-" * 55)
for d in [1, 2, 3, 5, 10]:
    samples = samples_needed(d, 10)
    print(f"{d:9d} | {samples:12,d}")

print("\nExponential growth! 10D space needs 10 billion samples!")
```

### Solutions

**1. Feature Selection:** Remove irrelevant and redundant features.

```python
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.datasets import make_regression

# Generate data: 100 features, only 10 are useful
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10,
                      noise=10, random_state=42)

print(f"Original: {X.shape[1]} features")

# Select top 10 features
selector = SelectKBest(f_regression, k=10)
X_selected = selector.fit_transform(X, y)

print(f"After selection: {X_selected.shape[1]} features")
print("Reduced dimensionality by 90%!")
```

**2. Dimensionality Reduction:** Project data to lower dimensions while preserving information.

```python
from sklearn.decomposition import PCA

# High-dimensional data
X = np.random.rand(1000, 100)  # 100 dimensions

# Reduce to 10 dimensions
pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X)

print(f"Original: {X.shape[1]} dimensions")
print(f"Reduced: {X_reduced.shape[1]} dimensions")
print(f"Variance retained: {pca.explained_variance_ratio_.sum():.2%}")
```

**3. Regularization:** Penalize model complexity to prevent overfitting in high dimensions.

```python
from sklearn.linear_model import Ridge, Lasso

# High-dimensional data
X = np.random.rand(100, 50)  # 100 samples, 50 features
y = np.random.rand(100)

# Ridge regression adds L2 penalty
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)

# Lasso adds L1 penalty (can zero out features)
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)

print(f"Ridge: Uses all {np.sum(ridge.coef_ != 0)} features")
print(f"Lasso: Uses {np.sum(lasso.coef_ != 0)} features (auto-selection)")
```

**4. Collect More Data:** The most straightforward but often most expensive solution.

---

## Summary

Transforming data into machine-readable form is a critical step that directly impacts model performance:

**Feature Representation:** Features are the measurable inputs to models. We can engineer new features (create from existing), select features (choose most relevant), or extract features (transform to new representation). Each approach serves different purposes in building effective models.

**Encoding Categorical Data:** Categories need numerical encoding. Use one-hot encoding for nominal categories (no order), ordinal encoding for ranked categories (low/medium/high), and be cautious with label encoding (implies false ordering). The wrong encoding can mislead models.

**Encoding Text and Images:** Text requires special techniques—Bag of Words for simple counting, TF-IDF for weighted importance, and embeddings for semantic meaning. Images are already numbers (pixels) but benefit from normalization. The encoding method should match your task requirements.

**Feature Scaling:** Different scales cause problems for many algorithms. Standardization (z-score) creates mean=0 and std=1, preserving distribution shape. Normalization (min-max) scales to [0,1], creating bounded ranges. Choose based on your data distribution and algorithm requirements.

**Feature Engineering:** Creating new features often provides the biggest performance gains. Combine features (interactions), extract temporal patterns (time-based features), or create polynomial terms. Domain knowledge is your most powerful tool here.

**Curse of Dimensionality:** High-dimensional spaces are sparse and counterintuitive. Distances become meaningless, and you need exponentially more data. Combat this with feature selection, dimensionality reduction, regularization, or more data collection.

With data properly represented as numbers, scaled appropriately, and engineered thoughtfully, you're ready to understand how models use these features to learn. That's what we'll explore next.

---

[← Previous: Chapter 2](02-data-source.md) | [Back to Index](index.md) | [Next: Chapter 4 →](04-models-as-functions.md)

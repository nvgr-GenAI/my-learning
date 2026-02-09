# Encoding Categorical Variables

## Overview

Machine learning algorithms require numerical input, but real-world data often contains categorical variables (text labels). Encoding transforms categorical data into numerical format while preserving the information and relationships between categories.

**Key Principle**: Choose encoding methods that preserve the categorical variable's nature - ordinal vs nominal, high vs low cardinality.

## Why Encoding Matters

1. **Model Compatibility**: Most ML algorithms only accept numerical input
2. **Information Preservation**: Proper encoding retains categorical relationships
3. **Performance Impact**: Wrong encoding can hurt model accuracy
4. **Dimensionality**: Some methods create many features (curse of dimensionality)
5. **Overfitting Risk**: Poor encoding can lead to overfitting

## Types of Categorical Variables

### 1. Nominal (No Order)

Categories have no inherent order or ranking.

```python
import pandas as pd
import numpy as np

# Examples of nominal variables
data = {
    'color': ['red', 'blue', 'green', 'red', 'yellow'],
    'city': ['NYC', 'LA', 'Chicago', 'NYC', 'Miami'],
    'product_type': ['Electronics', 'Clothing', 'Food', 'Electronics', 'Food']
}
df = pd.DataFrame(data)

print("Nominal Variables (no inherent order):")
print(df)
print("\nUnique values per column:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")
```

### 2. Ordinal (Has Order)

Categories have a meaningful order or ranking.

```python
# Examples of ordinal variables
data_ordinal = {
    'education': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor'],
    'satisfaction': ['Poor', 'Fair', 'Good', 'Excellent', 'Good'],
    'size': ['Small', 'Medium', 'Large', 'Small', 'XLarge']
}
df_ordinal = pd.DataFrame(data_ordinal)

print("Ordinal Variables (inherent order):")
print(df_ordinal)

# Define orders
education_order = ['High School', 'Bachelor', 'Master', 'PhD']
satisfaction_order = ['Poor', 'Fair', 'Good', 'Excellent']
size_order = ['Small', 'Medium', 'Large', 'XLarge']

print("\nOrders:")
print(f"Education: {education_order}")
print(f"Satisfaction: {satisfaction_order}")
print(f"Size: {size_order}")
```

## Encoding Methods

### 1. Label Encoding

Assigns each unique category an integer.

```python
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Sample data
data = {
    'color': ['red', 'blue', 'green', 'red', 'yellow', 'blue'],
    'size': ['S', 'M', 'L', 'M', 'S', 'L']
}
df = pd.DataFrame(data)

print("Original data:")
print(df)

# Apply Label Encoding
le_color = LabelEncoder()
le_size = LabelEncoder()

df['color_encoded'] = le_color.fit_transform(df['color'])
df['size_encoded'] = le_size.fit_transform(df['size'])

print("\nLabel Encoded:")
print(df)

# Show mapping
print("\nColor mapping:")
for i, label in enumerate(le_color.classes_):
    print(f"  {label} -> {i}")

print("\nSize mapping:")
for i, label in enumerate(le_size.classes_):
    print(f"  {label} -> {i}")

# Inverse transform
original_colors = le_color.inverse_transform(df['color_encoded'])
print(f"\nInverse transform works: {np.array_equal(df['color'].values, original_colors)}")
```

**When to use:**
- **Ordinal variables** with clear ordering
- Target variable encoding for classification
- Tree-based algorithms (can handle arbitrary numbers)

**Pros:**
- Simple and fast
- Maintains single column
- Reversible with inverse_transform

**Cons:**
- Implies ordinal relationship (not suitable for nominal data)
- Can mislead linear models (3 > 2 > 1)
- Arbitrary ordering affects some models

**Warning Example:**

```python
# Problem with label encoding for nominal data
from sklearn.linear_model import LinearRegression

# Nominal data: color has no order
colors = np.array(['red', 'blue', 'green', 'red', 'blue', 'green'])
prices = np.array([10, 15, 12, 11, 14, 13])

# Label encode
le = LabelEncoder()
colors_encoded = le.fit_transform(colors).reshape(-1, 1)

# Train model
model = LinearRegression()
model.fit(colors_encoded, prices)

print("Label encoding implies: blue(0) < green(1) < red(2)")
print("This creates artificial ordering where none exists!")
print("Linear model learns: as color increases, price changes")
print("This is meaningless for nominal categories!")
```

### 2. One-Hot Encoding

Creates binary column for each category.

```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Sample data
data = {
    'color': ['red', 'blue', 'green', 'red', 'yellow'],
    'size': ['S', 'M', 'L', 'M', 'S']
}
df = pd.DataFrame(data)

print("Original data:")
print(df)

# Method 1: Using pandas get_dummies
df_dummies = pd.get_dummies(df, columns=['color', 'size'], prefix=['color', 'size'])
print("\nOne-Hot Encoded (pandas):")
print(df_dummies)

# Method 2: Using sklearn OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[['color', 'size']])

# Get feature names
feature_names = encoder.get_feature_names_out(['color', 'size'])
df_encoded = pd.DataFrame(encoded, columns=feature_names)

print("\nOne-Hot Encoded (sklearn):")
print(df_encoded)

# Show that exactly one column is 1 for each original category
print("\nVerification (sum of color columns = 1):")
print(df_encoded[[col for col in df_encoded.columns if 'color' in col]].sum(axis=1))
```

**When to use:**
- **Nominal variables** (no inherent order)
- Linear models, neural networks
- Low to medium cardinality (< 10-15 categories)

**Pros:**
- No ordinal assumption
- Works well with linear models
- Clear interpretation
- sklearn version handles unknown categories

**Cons:**
- Creates many columns (high cardinality problem)
- Sparse data
- Can cause multicollinearity
- Memory intensive for large datasets

**Handling Unknown Categories:**

```python
# Handle categories not seen during training
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(df[['color']])

# New data with unknown category
new_data = pd.DataFrame({'color': ['red', 'purple']})  # 'purple' not in training
encoded_new = encoder.transform(new_data)

print("Encoded with unknown category 'purple':")
print(encoded_new)
print("Unknown category gets all zeros!")
```

### 3. Ordinal Encoding

Maps categories to integers with specified order.

```python
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

# Ordinal data with clear ordering
data = {
    'education': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'High School'],
    'satisfaction': ['Fair', 'Good', 'Excellent', 'Poor', 'Good', 'Fair']
}
df = pd.DataFrame(data)

print("Original ordinal data:")
print(df)

# Define orderings
education_categories = [['High School', 'Bachelor', 'Master', 'PhD']]
satisfaction_categories = [['Poor', 'Fair', 'Good', 'Excellent']]

# Apply ordinal encoding
encoder = OrdinalEncoder(categories=education_categories + satisfaction_categories)
df_encoded = df.copy()
df_encoded[['education', 'satisfaction']] = encoder.fit_transform(df)

print("\nOrdinal Encoded:")
print(df_encoded)

# Show mappings
print("\nEducation mapping (increasing education level):")
for i, level in enumerate(education_categories[0]):
    print(f"  {level} -> {i}")

print("\nSatisfaction mapping (increasing satisfaction):")
for i, level in enumerate(satisfaction_categories[0]):
    print(f"  {level} -> {i}")

# Verify order is preserved
print("\nVerification:")
print("PhD (3) > Master (2) > Bachelor (1) > High School (0) ✓")
print("Excellent (3) > Good (2) > Fair (1) > Poor (0) ✓")
```

**When to use:**
- **Ordinal variables** with known order
- Want to preserve ordering information
- Reduce dimensionality vs one-hot encoding

**Pros:**
- Preserves ordinal relationships
- Single column per feature
- Interpretable numeric values

**Cons:**
- Assumes equal spacing between categories
- Requires domain knowledge for ordering
- Not suitable for nominal data

### 4. Frequency Encoding

Replaces categories with their frequency/proportion.

```python
import pandas as pd
import numpy as np

# Data with varying frequencies
data = {
    'city': ['NYC', 'NYC', 'LA', 'NYC', 'Chicago', 'LA', 'NYC',
             'Miami', 'Chicago', 'NYC']
}
df = pd.DataFrame(data)

print("Original data:")
print(df['city'].value_counts())

# Frequency encoding
freq_encoding = df['city'].value_counts(normalize=False).to_dict()
df['city_freq'] = df['city'].map(freq_encoding)

# Proportion encoding
prop_encoding = df['city'].value_counts(normalize=True).to_dict()
df['city_prop'] = df['city'].map(prop_encoding)

print("\nFrequency Encoded:")
print(df)

print("\nFrequency mapping:")
for city, freq in freq_encoding.items():
    print(f"  {city} -> {freq} ({prop_encoding[city]:.1%})")
```

**When to use:**
- High cardinality categorical variables
- Frequency is informative (e.g., popular products)
- Complement to other encoding methods

**Pros:**
- Single column
- Captures category importance
- Works with high cardinality

**Cons:**
- Different categories can have same frequency
- Loses category identity
- May not work for categories with equal frequency

### 5. Target Encoding (Mean Encoding)

Replaces categories with mean of target variable for that category.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

# Sample data with target
np.random.seed(42)
data = {
    'city': ['NYC', 'LA', 'Chicago', 'NYC', 'LA', 'Chicago', 'NYC', 'LA'] * 10,
    'price': []
}

# Generate prices with city-specific patterns
for city in data['city']:
    if city == 'NYC':
        data['price'].append(np.random.normal(300, 50))
    elif city == 'LA':
        data['price'].append(np.random.normal(250, 40))
    else:  # Chicago
        data['price'].append(np.random.normal(200, 30))

df = pd.DataFrame(data)

print("Original data:")
print(df.groupby('city')['price'].agg(['mean', 'std', 'count']))

# Simple target encoding (prone to overfitting!)
target_encoding = df.groupby('city')['price'].mean().to_dict()
df['city_target_simple'] = df['city'].map(target_encoding)

print("\nSimple Target Encoding:")
print(df[['city', 'price', 'city_target_simple']].head(10))

# Better: K-fold target encoding (reduces overfitting)
def kfold_target_encoding(df, column, target, n_folds=5):
    """Target encode using K-fold to prevent overfitting"""
    df['target_encoded'] = 0
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(df):
        train = df.iloc[train_idx]
        val = df.iloc[val_idx]

        # Calculate mean target for each category on training fold
        target_means = train.groupby(column)[target].mean()

        # Apply to validation fold
        df.loc[val_idx, 'target_encoded'] = val[column].map(target_means)

    return df

df = kfold_target_encoding(df, 'city', 'price')

print("\nK-Fold Target Encoding (better):")
print(df[['city', 'price', 'city_target_simple', 'target_encoded']].head(10))

# Visualize
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Simple target encoding
for city in df['city'].unique():
    city_data = df[df['city'] == city]
    axes[0].scatter(city_data['city_target_simple'], city_data['price'], label=city, alpha=0.6)
axes[0].plot([150, 350], [150, 350], 'r--', label='Perfect encoding')
axes[0].set_xlabel('Target Encoded Value')
axes[0].set_ylabel('Actual Price')
axes[0].set_title('Simple Target Encoding')
axes[0].legend()

# K-fold target encoding
for city in df['city'].unique():
    city_data = df[df['city'] == city]
    axes[1].scatter(city_data['target_encoded'], city_data['price'], label=city, alpha=0.6)
axes[1].plot([150, 350], [150, 350], 'r--', label='Perfect encoding')
axes[1].set_xlabel('Target Encoded Value (K-fold)')
axes[1].set_ylabel('Actual Price')
axes[1].set_title('K-Fold Target Encoding')
axes[1].legend()

plt.tight_layout()
plt.show()
```

**When to use:**
- **High cardinality** categorical variables
- Strong relationship between category and target
- Regression or classification tasks

**Pros:**
- Single column
- Handles high cardinality well
- Captures category-target relationship

**Cons:**
- Risk of overfitting (use K-fold)
- Needs target variable (supervised only)
- Can cause data leakage if not careful
- Requires more data per category

### 6. Binary Encoding

Converts categories to binary digits, then splits digits into columns.

```python
import pandas as pd
import numpy as np

# High cardinality data
cities = ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix',
          'Philadelphia', 'San Antonio', 'San Diego']
df = pd.DataFrame({'city': cities})

print("Original data:")
print(df)

# Binary encoding
# Step 1: Label encode
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels = le.fit_transform(df['city'])

# Step 2: Convert to binary
n_bits = int(np.ceil(np.log2(len(cities))))
print(f"\nNeed {n_bits} bits to encode {len(cities)} categories")

binary_encoded = np.array([list(f'{label:0{n_bits}b}') for label in labels], dtype=int)

# Create binary columns
binary_df = pd.DataFrame(
    binary_encoded,
    columns=[f'city_bin_{i}' for i in range(n_bits)]
)

df_encoded = pd.concat([df, binary_df], axis=1)

print("\nBinary Encoded:")
print(df_encoded)

# Compare with one-hot
print(f"\nComparison:")
print(f"One-hot encoding: {len(cities)} columns")
print(f"Binary encoding: {n_bits} columns")
print(f"Reduction: {len(cities) - n_bits} fewer columns")
```

**When to use:**
- **High cardinality** categorical variables
- Want fewer columns than one-hot
- Tree-based algorithms

**Pros:**
- Fewer columns than one-hot
- Still captures categorical information
- Good for high cardinality

**Cons:**
- Less interpretable than one-hot
- Not as natural for linear models
- Requires understanding of binary representation

### 7. Hashing (Feature Hashing)

Maps categories to fixed number of columns using hash function.

```python
from sklearn.feature_extraction import FeatureHasher
import pandas as pd

# High cardinality data
data = {
    'product_id': [f'PROD_{i:04d}' for i in range(1000)]
}
df = pd.DataFrame(data)

print(f"Original data: {len(df)} products")
print(df.head())

# Hash to fixed number of features
hasher = FeatureHasher(n_features=10, input_type='string')
hashed = hasher.transform(df['product_id']).toarray()

df_hashed = pd.DataFrame(
    hashed,
    columns=[f'hash_{i}' for i in range(10)]
)

print("\nHashed to 10 features:")
print(df_hashed.head())

# Note: Different products can hash to same values (collisions)
print("\nNote: Hash collisions are possible")
print("Multiple products may map to same hash value")
```

**When to use:**
- **Very high cardinality** (thousands of categories)
- Streaming data
- Memory constraints
- Online learning

**Pros:**
- Fixed number of features
- Memory efficient
- Fast
- Handles unknown categories naturally

**Cons:**
- Hash collisions
- Loss of interpretability
- Need to tune number of features
- Can hurt performance with small datasets

## Handling High Cardinality

```python
import pandas as pd
import numpy as np

# Simulate high cardinality data
np.random.seed(42)
n_samples = 10000
n_unique_zipcodes = 500

data = {
    'zipcode': np.random.choice([f'ZIP_{i:05d}' for i in range(n_unique_zipcodes)],
                                size=n_samples),
    'price': np.random.normal(100, 25, n_samples)
}
df = pd.DataFrame(data)

print(f"Data shape: {df.shape}")
print(f"Unique zipcodes: {df['zipcode'].nunique()}")

# Strategy 1: Group rare categories
def group_rare_categories(df, column, threshold=100):
    """Group categories with < threshold occurrences"""
    counts = df[column].value_counts()
    rare_categories = counts[counts < threshold].index
    df[f'{column}_grouped'] = df[column].apply(
        lambda x: 'RARE' if x in rare_categories else x
    )
    return df

df = group_rare_categories(df, 'zipcode', threshold=50)
print(f"\nAfter grouping rare zipcodes: {df['zipcode_grouped'].nunique()} unique")

# Strategy 2: Keep top N categories
def keep_top_n(df, column, n=50):
    """Keep only top N most frequent categories"""
    top_n = df[column].value_counts().nlargest(n).index
    df[f'{column}_top{n}'] = df[column].apply(
        lambda x: x if x in top_n else 'OTHER'
    )
    return df

df = keep_top_n(df, 'zipcode', n=50)
print(f"After keeping top 50: {df['zipcode_top50'].nunique()} unique")

# Strategy 3: Use target encoding instead of one-hot
target_means = df.groupby('zipcode')['price'].mean()
df['zipcode_target'] = df['zipcode'].map(target_means)
print(f"\nTarget encoding: 1 column (vs {df['zipcode'].nunique()} for one-hot)")

# Comparison
print("\nEncoding comparison for high cardinality:")
print(f"Original unique values: {df['zipcode'].nunique()}")
print(f"One-hot encoding would create: {df['zipcode'].nunique()} columns")
print(f"After grouping rare: {df['zipcode_grouped'].nunique()} columns")
print(f"After top 50: {df['zipcode_top50'].nunique()} columns")
print(f"Target encoding: 1 column")
print(f"Binary encoding: ~{int(np.ceil(np.log2(df['zipcode'].nunique())))} columns")
print(f"Hashing: user-defined (e.g., 20 columns)")
```

## Encoding Decision Guide

```python
def recommend_encoding(df, column, target=None):
    """Recommend encoding method based on variable characteristics"""

    n_unique = df[column].nunique()
    is_ordinal = input(f"Is '{column}' ordinal (has natural order)? (y/n): ").lower() == 'y'

    print(f"\n=== Encoding Recommendations for '{column}' ===")
    print(f"Unique values: {n_unique}")
    print(f"Type: {'Ordinal' if is_ordinal else 'Nominal'}")

    if is_ordinal:
        print("\n✓ RECOMMENDED: Ordinal Encoding")
        print("  Reason: Preserves natural ordering")
        print("  Implementation: OrdinalEncoder with specified categories")

    elif n_unique == 2:
        print("\n✓ RECOMMENDED: Label Encoding or One-Hot")
        print("  Reason: Binary variable, either works")
        print("  Implementation: LabelEncoder (0/1) or get_dummies")

    elif n_unique <= 10:
        print("\n✓ RECOMMENDED: One-Hot Encoding")
        print("  Reason: Low cardinality, creates manageable number of features")
        print("  Implementation: pd.get_dummies() or OneHotEncoder")

    elif n_unique <= 50:
        print("\n✓ RECOMMENDED: Consider multiple options:")
        print("  1. One-Hot Encoding (if you have enough data)")
        print("  2. Binary Encoding (reduces dimensionality)")
        print("  3. Target Encoding (if target available)")

    else:  # High cardinality
        print("\n✓ RECOMMENDED: High Cardinality Methods")
        print("  Options:")
        print("  1. Target Encoding (if target available) - Captures category importance")
        print("  2. Frequency Encoding - Simple and effective")
        print("  3. Group rare categories + One-Hot - Reduces cardinality")
        print("  4. Binary Encoding - Fewer features than one-hot")
        print("  5. Hashing - Very high cardinality (1000+)")

    # Additional considerations
    print("\n=== Additional Considerations ===")

    if target is not None:
        correlation = abs(pd.get_dummies(df[[column]]).corrwith(df[target]).max())
        print(f"Max correlation with target: {correlation:.3f}")
        if correlation > 0.3:
            print("  → Strong relationship with target: consider Target Encoding")

    # Check distribution
    freq = df[column].value_counts()
    imbalance = freq.max() / freq.min()
    if imbalance > 100:
        print(f"  → Highly imbalanced (ratio: {imbalance:.0f}:1)")
        print("  → Consider grouping rare categories")

# Example usage
# recommend_encoding(df, 'city', target='price')
```

## Complete Example: Multiple Encoding Strategies

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Create sample dataset
np.random.seed(42)
n_samples = 1000

data = {
    # Nominal, low cardinality
    'color': np.random.choice(['red', 'blue', 'green'], n_samples),

    # Nominal, medium cardinality
    'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix',
                              'Philadelphia', 'San Antonio'], n_samples),

    # Ordinal
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),

    # Nominal, high cardinality
    'zipcode': np.random.choice([f'ZIP_{i:05d}' for i in range(200)], n_samples),

    # Target
    'purchased': np.random.choice([0, 1], n_samples)
}
df = pd.DataFrame(data)

print("Dataset overview:")
print(df.head())
print(f"\nShape: {df.shape}")
print("\nUnique values per column:")
for col in df.columns:
    if col != 'purchased':
        print(f"  {col}: {df[col].nunique()}")

# Split data
X = df.drop('purchased', axis=1)
y = df['purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encoding strategy
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define transformations
preprocessor = ColumnTransformer(
    transformers=[
        # One-hot encode low cardinality nominal
        ('onehot', OneHotEncoder(drop='first', sparse_output=False),
         ['color']),

        # One-hot encode medium cardinality
        ('onehot_city', OneHotEncoder(drop='first', sparse_output=False),
         ['city']),

        # Ordinal encode ordered categories
        ('ordinal', OrdinalEncoder(
            categories=[['High School', 'Bachelor', 'Master', 'PhD']]),
         ['education']),

        # Target encode high cardinality (using training target)
        # Note: Would use K-fold in production
        ('pass', 'passthrough', ['zipcode'])  # Handle separately
    ])

# For target encoding of zipcode (simplified)
target_encoding = X_train.groupby('zipcode')['purchased'].mean().to_dict() if 'purchased' in X_train.columns else {}

# Handle zipcode separately
X_train_zipcode = X_train['zipcode'].map(target_encoding)
X_test_zipcode = X_test['zipcode'].map(target_encoding)

# Transform
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

# Add target-encoded zipcode
X_train_final = np.column_stack([X_train_encoded[:, :-1], X_train_zipcode])
X_test_final = np.column_stack([X_test_encoded[:, :-1], X_test_zipcode])

print(f"\nOriginal features: {X_train.shape[1]}")
print(f"Encoded features: {X_train_final.shape[1]}")

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_final, y_train)

score = model.score(X_test_final, y_test)
print(f"\nModel accuracy: {score:.4f}")
```

## Common Pitfalls

### 1. Label Encoding Nominal Variables

```python
# WRONG: Label encoding nominal data for linear models
colors = ['red', 'blue', 'green']
le = LabelEncoder()
encoded = le.fit_transform(colors)  # red=2, green=1, blue=0
# Linear model thinks: red > green > blue (meaningless!)

# CORRECT: One-hot encode nominal data
encoded = pd.get_dummies(colors)
```

### 2. Data Leakage in Target Encoding

```python
# WRONG: Target encode on entire dataset
target_means = df.groupby('category')['target'].mean()
df['category_encoded'] = df['category'].map(target_means)
# Uses future information!

# CORRECT: Use K-fold or fit only on training data
```

### 3. Not Handling Unknown Categories

```python
# WRONG: OneHotEncoder without handling unknown
encoder = OneHotEncoder()
encoder.fit(X_train[['city']])
# Fails on new cities in test set!

# CORRECT: Handle unknown categories
encoder = OneHotEncoder(handle_unknown='ignore')
```

### 4. Creating Too Many Features

```python
# WRONG: One-hot encoding high cardinality without thinking
df_encoded = pd.get_dummies(df, columns=['zipcode'])
# Creates 10,000 columns for 10,000 zipcodes!

# CORRECT: Use appropriate method for high cardinality
# Group rare, target encode, or use binary encoding
```

## Best Practices

1. **Understand Variable Type**: Identify if nominal or ordinal
2. **Check Cardinality**: Choose method based on number of unique values
3. **Prevent Data Leakage**: Fit encoders only on training data
4. **Handle Unknown**: Use `handle_unknown='ignore'` for production
5. **Use Pipelines**: Integrate encoding into sklearn pipelines
6. **Test Multiple Methods**: Compare performance with different encodings
7. **Document Choices**: Record why you chose each encoding method

## Related Topics

- [Feature Scaling](./feature-scaling.md) - Scale numerical features after encoding
- [Feature Creation](./feature-creation.md) - Create new features from encoded variables
- [Feature Selection](./feature-selection.md) - Select best encoded features

## Summary

Encoding categorical variables is essential for machine learning. Choose one-hot encoding for low-cardinality nominal variables, ordinal encoding for ordered categories, and target/binary encoding for high-cardinality variables. Always split data before encoding to prevent leakage, and handle unknown categories appropriately for production deployment.

**Next Step**: Learn about [feature creation](./feature-creation.md) to generate new features from existing ones.

---

*Last Updated: 2026-02-09*

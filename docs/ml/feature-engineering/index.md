# Feature Engineering

## Overview

Feature engineering is the process of transforming raw data into features that better represent the underlying problem to the predictive models, resulting in improved model accuracy. It's often considered the most important aspect of applied machine learning and can make the difference between a mediocre and an excellent model.

**Key Principle**: Good features can make simple models perform well, while poor features make even complex models struggle.

## Why Feature Engineering Matters

1. **Model Performance**: Can improve model accuracy by 10-50%
2. **Simpler Models**: Better features allow simpler, more interpretable models
3. **Faster Training**: Well-engineered features reduce training time
4. **Better Generalization**: Proper features help models generalize to new data
5. **Domain Knowledge**: Incorporates expertise into the model

## The Feature Engineering Pipeline

```
Raw Data
   ↓
1. DATA CLEANING
   ├── Remove duplicates
   ├── Fix data types
   ├── Handle missing values
   ├── Detect outliers
   └── Standardize formats
   ↓
2. FEATURE CREATION
   ├── Transform existing features
   ├── Create new features
   ├── Extract from complex data
   ├── Combine features
   └── Domain-specific features
   ↓
3. FEATURE SELECTION
   ├── Remove redundant features
   ├── Select important features
   ├── Reduce dimensionality
   └── Avoid overfitting
   ↓
Clean, Engineered Features → Model Training
```

## When Feature Engineering Matters Most

### High Impact Scenarios
- **Structured/Tabular Data**: Feature engineering is crucial
- **Small Datasets**: Good features compensate for limited data
- **Time Series**: Date/time features can be very powerful
- **Domain-Specific Problems**: Expert knowledge creates winning features
- **Competition/Production**: Every percentage point matters

### Lower Impact Scenarios
- **Deep Learning on Images/Text**: Neural networks learn features automatically
- **Very Large Datasets**: Models can learn patterns from raw data
- **Standard Problems**: Pre-built pipelines often sufficient

## Essential Tools

### Python Libraries

```python
# Data manipulation
import pandas as pd
import numpy as np

# Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# Imputation
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

# Feature selection
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif

# Feature creation
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

# Imbalanced data
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

# Specialized library
import feature_engine as fe

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
```

### Key Frameworks

1. **Pandas**: Data manipulation and basic preprocessing
2. **Scikit-learn**: Comprehensive preprocessing and feature selection
3. **Feature-engine**: Specialized feature engineering library
4. **Imbalanced-learn**: Handling imbalanced datasets
5. **Category-encoders**: Advanced categorical encoding

## Feature Engineering Topics

### 1. Data Cleaning
**Purpose**: Prepare raw data for machine learning

- Remove duplicates
- Fix data types and formats
- Handle structural errors
- Standardize text data
- Validate data ranges

[Learn more →](./data-cleaning.md)

### 2. Missing Values
**Purpose**: Handle incomplete data appropriately

- Understand missing data types (MCAR, MAR, MNAR)
- Deletion strategies
- Imputation methods (mean, median, mode, KNN, iterative)
- Missing indicator features
- Domain-specific handling

[Learn more →](./missing-values.md)

### 3. Outlier Detection
**Purpose**: Identify and handle extreme values

- Statistical methods (IQR, Z-score)
- Machine learning methods (Isolation Forest, LOF)
- Domain-based detection
- Visualization techniques
- When to remove vs. keep outliers

[Learn more →](./outlier-detection.md)

### 4. Feature Scaling
**Purpose**: Normalize feature ranges for model performance

- StandardScaler (z-score normalization)
- MinMaxScaler (range scaling)
- RobustScaler (outlier-resistant)
- When scaling is required
- Common pitfalls (data leakage)

[Learn more →](./feature-scaling.md)

### 5. Encoding Categorical Variables
**Purpose**: Convert categorical data to numerical format

- Label encoding (ordinal data)
- One-hot encoding (nominal data)
- Ordinal encoding (ordered categories)
- Target encoding (high cardinality)
- Handling unknown categories

[Learn more →](./encoding-categorical.md)

### 6. Feature Creation
**Purpose**: Generate new features from existing data

- Polynomial features
- Interaction features
- Datetime feature extraction
- Text feature engineering
- Domain-specific transformations
- Aggregations and statistics

[Learn more →](./feature-creation.md)

### 7. Feature Selection
**Purpose**: Select most relevant features

- Filter methods (correlation, mutual information)
- Wrapper methods (RFE, forward/backward selection)
- Embedded methods (L1 regularization, tree importance)
- Dimensionality reduction (PCA, t-SNE)
- Evaluation and comparison

[Learn more →](./feature-selection.md)

### 8. Imbalanced Data
**Purpose**: Handle skewed class distributions

- Resampling techniques (SMOTE, undersampling)
- Class weights adjustment
- Ensemble methods
- Evaluation metrics (precision, recall, F1)
- Cost-sensitive learning

[Learn more →](./imbalanced-data.md)

## Learning Path

### Beginner (Weeks 1-2)
1. Start with **Data Cleaning** - understand data quality issues
2. Learn **Missing Values** - essential for any dataset
3. Practice **Feature Scaling** - understand when and why
4. Master **Encoding Categorical Variables** - common requirement

**Project**: Clean and prepare a Kaggle dataset (Titanic, House Prices)

### Intermediate (Weeks 3-4)
5. Study **Outlier Detection** - identify data anomalies
6. Explore **Feature Creation** - generate new features
7. Learn **Feature Selection** - reduce dimensionality
8. Practice **Imbalanced Data** - real-world challenge

**Project**: Build end-to-end pipeline for classification problem

### Advanced (Ongoing)
- Combine multiple techniques in automated pipelines
- Learn domain-specific feature engineering
- Experiment with advanced methods (target encoding, embeddings)
- Participate in Kaggle competitions
- Read feature engineering winning solutions

## Best Practices

### 1. Understand Your Data First
```python
# Always start with exploratory data analysis
df.info()
df.describe()
df.isnull().sum()
df.head()

# Visualize distributions
import matplotlib.pyplot as plt
df.hist(figsize=(15, 10), bins=50)
plt.tight_layout()
plt.show()
```

### 2. Prevent Data Leakage
```python
# WRONG: Fit on entire dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled)

# CORRECT: Split first, then fit only on training data
X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform test
```

### 3. Use Pipelines
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Define preprocessing for different column types
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create full pipeline with model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Fit and predict (no data leakage!)
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### 4. Iterate and Validate
```python
from sklearn.model_selection import cross_val_score

# Test feature engineering impact
baseline_score = cross_val_score(model, X_original, y, cv=5).mean()
new_score = cross_val_score(model, X_engineered, y, cv=5).mean()

print(f"Baseline: {baseline_score:.3f}")
print(f"With feature engineering: {new_score:.3f}")
print(f"Improvement: {(new_score - baseline_score):.3f}")
```

### 5. Document Your Process
```python
# Keep track of transformations
feature_engineering_log = {
    'dropped_features': ['id', 'name'],
    'imputation': {'age': 'median', 'cabin': 'most_frequent'},
    'encoding': {'sex': 'label', 'embarked': 'onehot'},
    'scaling': 'standard',
    'new_features': ['family_size', 'title', 'fare_per_person']
}
```

## Common Pitfalls to Avoid

1. **Data Leakage**: Fitting transformers on entire dataset before split
2. **Overfitting**: Creating too many features for small datasets
3. **Ignoring Data Types**: Not checking dtypes before preprocessing
4. **One-Size-Fits-All**: Using same preprocessing for all features
5. **No Validation**: Not testing if features actually improve model
6. **Forgetting Test Data**: Not saving fitted transformers for test/production
7. **Outlier Removal**: Blindly removing outliers without understanding
8. **Feature Selection Bias**: Selecting features based on test performance

## Quick Reference Checklist

### Before Model Training
- [ ] Explored data distribution and relationships
- [ ] Handled missing values appropriately
- [ ] Detected and handled outliers
- [ ] Scaled numerical features if needed
- [ ] Encoded categorical variables properly
- [ ] Created relevant new features
- [ ] Selected most important features
- [ ] Handled class imbalance if present
- [ ] Split data before any fitting
- [ ] Used pipelines to prevent leakage
- [ ] Validated feature engineering improves model

## Resources

### Books
- "Feature Engineering for Machine Learning" by Alice Zheng & Amanda Casari
- "Feature Engineering and Selection" by Kuhn & Johnson

### Online Courses
- Kaggle Learn: Feature Engineering
- Fast.ai: Practical Deep Learning

### Kaggle Competitions
- Study winning solutions for feature engineering ideas
- Participate in tabular data competitions

### Libraries Documentation
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Feature-engine](https://feature-engine.readthedocs.io/)
- [Category Encoders](https://contrib.scikit-learn.org/category_encoders/)

## Next Steps

1. **Start Learning**: Begin with [Data Cleaning](./data-cleaning.md)
2. **Practice**: Work through each tutorial with examples
3. **Apply**: Use techniques on real datasets
4. **Iterate**: Continuously improve your feature engineering skills
5. **Share**: Document and share your feature engineering discoveries

Remember: Feature engineering is both art and science. It requires domain knowledge, creativity, and systematic experimentation. The best feature engineers combine technical skills with deep understanding of the problem domain.

---

*Last Updated: 2026-02-09*

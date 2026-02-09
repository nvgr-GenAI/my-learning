# Missing Values

## Overview

Missing values are inevitable in real-world datasets. How you handle them significantly impacts model performance. This tutorial covers understanding why data is missing, strategies for handling it, and implementing solutions that minimize information loss while maintaining data integrity.

**Key Principle**: The right approach to missing values depends on WHY they're missing, not just THAT they're missing.

## Why Missing Values Matter

1. **Model Compatibility**: Many ML algorithms can't handle missing values
2. **Biased Results**: Naive handling introduces bias
3. **Information Loss**: Removing data wastes valuable information
4. **Pattern Recognition**: Missingness itself can be predictive
5. **Production Reliability**: Models must handle missing values in deployment

## Types of Missing Data

### 1. MCAR (Missing Completely At Random)

Data is missing independent of any variables. The missingness is random and unrelated to any data.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate MCAR example
np.random.seed(42)
n = 1000

data = {
    'age': np.random.normal(40, 15, n),
    'income': np.random.normal(50000, 20000, n),
    'health_score': np.random.normal(70, 15, n)
}
df = pd.DataFrame(data)

# Randomly set 20% of income to missing (MCAR)
missing_mask = np.random.random(n) < 0.2
df.loc[missing_mask, 'income'] = np.nan

print("MCAR Example:")
print(f"Missing income values: {df['income'].isna().sum()}")

# Check if missingness is independent
print("\nAverage age for missing vs non-missing income:")
print(f"Missing income: {df[df['income'].isna()]['age'].mean():.2f}")
print(f"Not missing income: {df[df['income'].notna()]['age'].mean():.2f}")
# These should be similar for MCAR

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Age distribution: missing vs not missing
df['income_missing'] = df['income'].isna()
df[df['income_missing']]['age'].hist(ax=axes[0], alpha=0.5, label='Missing Income', bins=30)
df[~df['income_missing']]['age'].hist(ax=axes[0], alpha=0.5, label='Not Missing Income', bins=30)
axes[0].set_title('Age Distribution: MCAR')
axes[0].legend()

# Health score distribution: missing vs not missing
df[df['income_missing']]['health_score'].hist(ax=axes[1], alpha=0.5, label='Missing Income', bins=30)
df[~df['income_missing']]['health_score'].hist(ax=axes[1], alpha=0.5, label='Not Missing Income', bins=30)
axes[1].set_title('Health Score Distribution: MCAR')
axes[1].legend()

plt.tight_layout()
plt.show()
```

**Strategy for MCAR**: Deletion or simple imputation is safe since missingness is random.

### 2. MAR (Missing At Random)

Data is missing based on observed variables, but not the missing variable itself.

```python
# Generate MAR example
np.random.seed(42)
n = 1000

data = {
    'age': np.random.normal(40, 15, n),
    'income': np.random.normal(50000, 20000, n),
    'health_score': np.random.normal(70, 15, n)
}
df = pd.DataFrame(data)

# Income is more likely to be missing for older people (MAR)
# Probability of missing increases with age
missing_prob = 0.1 + 0.3 * (df['age'] > 50).astype(int)
missing_mask = np.random.random(n) < missing_prob
df.loc[missing_mask, 'income'] = np.nan

print("MAR Example:")
print(f"Missing income values: {df['income'].isna().sum()}")

# Check if missingness depends on age
print("\nAverage age for missing vs non-missing income:")
print(f"Missing income: {df[df['income'].isna()]['age'].mean():.2f}")
print(f"Not missing income: {df[df['income'].notna()]['age'].mean():.2f}")
# These will be different for MAR

# Missingness rate by age group
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle', 'Old'])
missing_rate = df.groupby('age_group')['income'].apply(lambda x: x.isna().sum() / len(x))
print("\nMissing rate by age group:")
print(missing_rate)
```

**Strategy for MAR**: Use observed variables to predict missing values (model-based imputation).

### 3. MNAR (Missing Not At Random)

Data is missing based on the missing value itself. People with very high or very low incomes might not report it.

```python
# Generate MNAR example
np.random.seed(42)
n = 1000

data = {
    'age': np.random.normal(40, 15, n),
    'income': np.random.normal(50000, 20000, n),
    'health_score': np.random.normal(70, 15, n)
}
df = pd.DataFrame(data)

# Very high incomes are more likely to be missing (MNAR)
# People don't want to report very high income
missing_prob = 0.05 + 0.5 * (df['income'] > 70000).astype(int)
missing_mask = np.random.random(n) < missing_prob
df.loc[missing_mask, 'income'] = np.nan

print("MNAR Example:")
print(f"Missing income values: {df['income'].isna().sum()}")

# Average of observed incomes will be biased low
print(f"\nTrue mean income: {data['income'].mean():.2f}")
print(f"Observed mean income: {df['income'].mean():.2f}")
print("Observed mean is biased because high earners don't report!")
```

**Strategy for MNAR**: Most challenging! Need domain expertise or auxiliary data. Consider modeling missingness explicitly.

## Detecting Missing Data Patterns

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample dataset
np.random.seed(42)
n = 500
df = pd.DataFrame({
    'age': np.random.normal(40, 15, n),
    'income': np.random.normal(50000, 20000, n),
    'education_years': np.random.normal(14, 3, n),
    'health_score': np.random.normal(70, 15, n)
})

# Introduce different missing patterns
# Age: MCAR
df.loc[np.random.random(n) < 0.1, 'age'] = np.nan

# Income: MAR (depends on education)
missing_prob = 0.05 + 0.3 * (df['education_years'] < 12).astype(int)
df.loc[np.random.random(n) < missing_prob, 'income'] = np.nan

# Health: MNAR (low scores more likely missing)
missing_prob = 0.05 + 0.4 * (df['health_score'] < 60).astype(int)
df.loc[np.random.random(n) < missing_prob, 'health_score'] = np.nan

# Analyze missing patterns
def analyze_missingness(df):
    """Comprehensive missing value analysis"""

    print("=" * 60)
    print("MISSING VALUE ANALYSIS")
    print("=" * 60)

    # 1. Basic counts
    print("\n1. Missing Value Counts:")
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing Count': missing_counts,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0])

    # 2. Missing value patterns
    print("\n2. Missing Value Patterns:")
    # Create binary matrix of missingness
    missing_matrix = df.isnull().astype(int)
    # Find most common patterns
    pattern_counts = missing_matrix.value_counts()
    print(f"Number of unique missing patterns: {len(pattern_counts)}")
    print("\nTop 5 patterns:")
    print(pattern_counts.head())

    # 3. Correlation between missingness
    print("\n3. Correlation Between Missing Values:")
    missing_corr = missing_matrix.corr()
    print(missing_corr)

    # 4. Visualize missing patterns
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Missing value heatmap
    ax = axes[0, 0]
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, ax=ax, cmap='viridis')
    ax.set_title('Missing Value Pattern (Yellow = Missing)')

    # Missing percentage by column
    ax = axes[0, 1]
    missing_pct.plot(kind='bar', ax=ax)
    ax.set_title('Missing Percentage by Column')
    ax.set_ylabel('Percentage Missing')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Correlation heatmap of missingness
    ax = axes[1, 0]
    sns.heatmap(missing_corr, annot=True, fmt='.2f', ax=ax, cmap='coolwarm', center=0)
    ax.set_title('Correlation Between Missing Values')

    # Distribution comparison for one variable
    ax = axes[1, 1]
    col_to_check = 'income'
    if col_to_check in df.columns:
        df[df['health_score'].notna()]['income'].hist(ax=ax, alpha=0.5,
                                                        label='Health Score Present', bins=30)
        df[df['health_score'].isna()]['income'].hist(ax=ax, alpha=0.5,
                                                       label='Health Score Missing', bins=30)
        ax.set_title('Income Distribution by Health Score Missingness')
        ax.legend()

    plt.tight_layout()
    plt.show()

    # 5. Statistical tests for MAR
    print("\n4. Testing for MAR (Chi-square tests):")
    for col_missing in df.columns:
        if df[col_missing].isnull().sum() > 0:
            for col_predictor in df.columns:
                if col_missing != col_predictor and df[col_predictor].dtype in [np.float64, np.int64]:
                    # Compare means
                    present_mean = df[df[col_missing].notna()][col_predictor].mean()
                    missing_mean = df[df[col_missing].isna()][col_predictor].mean()
                    diff = abs(present_mean - missing_mean)

                    if diff / present_mean > 0.1:  # More than 10% difference
                        print(f"\n{col_missing} missingness may depend on {col_predictor}:")
                        print(f"  Mean when present: {present_mean:.2f}")
                        print(f"  Mean when missing: {missing_mean:.2f}")
                        print(f"  Difference: {diff:.2f} ({diff/present_mean*100:.1f}%)")

# Run analysis
analyze_missingness(df)
```

## Handling Strategies

### 1. Deletion Methods

#### Listwise Deletion (Complete Case Analysis)

```python
# Remove rows with any missing values
df_complete = df.dropna()

print(f"Original rows: {len(df)}")
print(f"After listwise deletion: {len(df_complete)}")
print(f"Rows removed: {len(df) - len(df_complete)} ({(len(df) - len(df_complete))/len(df)*100:.1f}%)")

# Pros: Simple, no imputation bias
# Cons: Loses data, can introduce bias if not MCAR
```

#### Pairwise Deletion

```python
# Use all available data for each calculation
# Example: correlation matrix with pairwise deletion
corr_pairwise = df.corr()  # pandas default is pairwise deletion

print("Correlation with pairwise deletion:")
print(corr_pairwise)

# Pros: Uses more data than listwise
# Cons: Different n for different calculations
```

#### Column Deletion

```python
# Remove columns with too many missing values
threshold = 0.5  # 50% missing threshold
missing_pct = df.isnull().sum() / len(df)
cols_to_keep = missing_pct[missing_pct < threshold].index
df_col_dropped = df[cols_to_keep]

print(f"Original columns: {len(df.columns)}")
print(f"After column deletion: {len(df_col_dropped.columns)}")

# Pros: Removes unreliable features
# Cons: Loses potentially useful information
```

### 2. Simple Imputation

```python
from sklearn.impute import SimpleImputer

# Create sample data with missing values
np.random.seed(42)
df = pd.DataFrame({
    'age': [25, np.nan, 35, 40, np.nan, 30, 45],
    'income': [50000, 60000, np.nan, 80000, 90000, np.nan, 70000],
    'score': [85, 90, 78, np.nan, 95, 88, np.nan]
})

print("Original data:")
print(df)
print(f"\nMissing values:\n{df.isnull().sum()}")

# Mean imputation
imputer_mean = SimpleImputer(strategy='mean')
df_mean = pd.DataFrame(
    imputer_mean.fit_transform(df),
    columns=df.columns
)
print("\nMean imputation:")
print(df_mean)

# Median imputation (robust to outliers)
imputer_median = SimpleImputer(strategy='median')
df_median = pd.DataFrame(
    imputer_median.fit_transform(df),
    columns=df.columns
)
print("\nMedian imputation:")
print(df_median)

# Mode imputation (for categorical)
imputer_mode = SimpleImputer(strategy='most_frequent')
df_mode = pd.DataFrame(
    imputer_mode.fit_transform(df),
    columns=df.columns
)
print("\nMode imputation:")
print(df_mode)

# Constant value imputation
imputer_const = SimpleImputer(strategy='constant', fill_value=0)
df_const = pd.DataFrame(
    imputer_const.fit_transform(df),
    columns=df.columns
)
print("\nConstant (0) imputation:")
print(df_const)
```

**When to use:**
- **Mean**: Numeric data, approximately normal distribution, MCAR
- **Median**: Numeric data with outliers, skewed distributions
- **Mode**: Categorical data
- **Constant**: When missing has specific meaning (e.g., 0 for "none")

### 3. Advanced Imputation

#### K-Nearest Neighbors (KNN) Imputation

```python
from sklearn.impute import KNNImputer
import numpy as np
import pandas as pd

# Create sample data
np.random.seed(42)
df = pd.DataFrame({
    'age': [25, np.nan, 35, 40, 26, 30, 45, np.nan, 34],
    'income': [50000, 60000, np.nan, 80000, 55000, 65000, 90000, 78000, np.nan],
    'years_exp': [2, 5, 8, 15, 3, 6, 20, 12, 7],
    'score': [85, 90, 78, 95, 88, 92, 98, 94, 80]
})

print("Original data:")
print(df)

# KNN imputation (k=3)
imputer_knn = KNNImputer(n_neighbors=3)
df_knn = pd.DataFrame(
    imputer_knn.fit_transform(df),
    columns=df.columns
)

print("\nKNN imputation (k=3):")
print(df_knn)

# Show which values were imputed
print("\nImputed values:")
for col in df.columns:
    missing_mask = df[col].isna()
    if missing_mask.any():
        print(f"\n{col}:")
        for idx in df[missing_mask].index:
            print(f"  Row {idx}: {df_knn.loc[idx, col]:.2f}")

# Pros: Uses feature relationships, better than mean
# Cons: Computationally expensive, sensitive to k choice
```

#### Iterative Imputation (MICE - Multivariate Imputation by Chained Equations)

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

# Iterative imputation
imputer_iterative = IterativeImputer(
    estimator=BayesianRidge(),
    max_iter=10,
    random_state=42
)

df_iterative = pd.DataFrame(
    imputer_iterative.fit_transform(df),
    columns=df.columns
)

print("\nIterative imputation (MICE):")
print(df_iterative)

# Compare all methods
comparison = pd.DataFrame({
    'Original': df['age'],
    'Mean': df_mean['age'],
    'Median': df_median['age'],
    'KNN': df_knn['age'],
    'Iterative': df_iterative['age']
})
print("\nComparison of imputation methods for 'age':")
print(comparison)

# Pros: Most sophisticated, uses all features
# Cons: Slow, complex, requires careful tuning
```

### 4. Missing Indicator Strategy

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Create sample data
np.random.seed(42)
df = pd.DataFrame({
    'age': [25, np.nan, 35, 40, np.nan, 30],
    'income': [50000, 60000, np.nan, 80000, 90000, np.nan],
    'score': [85, 90, 78, np.nan, 95, 88]
})

print("Original data:")
print(df)

# Strategy: Impute + add missing indicator
df_enhanced = df.copy()

# Add missing indicators
for col in df.columns:
    if df[col].isna().any():
        df_enhanced[f'{col}_missing'] = df[col].isna().astype(int)

# Impute original columns
imputer = SimpleImputer(strategy='median')
original_cols = df.columns
df_enhanced[original_cols] = imputer.fit_transform(df_enhanced[original_cols])

print("\nData with missing indicators:")
print(df_enhanced)

# Pros: Preserves information about missingness
# Cons: Increases feature count
```

### 5. Domain-Specific Imputation

```python
# Example: E-commerce data
df = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5],
    'age': [25, np.nan, 35, 40, np.nan],
    'previous_purchases': [0, 5, np.nan, 10, 2],
    'cart_value': [100, np.nan, 200, np.nan, 150],
    'email_opened': [1, 0, 1, np.nan, 1]
})

# Domain knowledge-based imputation
df_domain = df.copy()

# If no previous purchases, missing age might be young first-time customer
df_domain.loc[
    (df_domain['age'].isna()) & (df_domain['previous_purchases'] == 0),
    'age'
] = 22  # Young first-time customer

# If email not opened, cart value missing might mean no purchase intent
df_domain.loc[
    (df_domain['cart_value'].isna()) & (df_domain['email_opened'] == 0),
    'cart_value'
] = 0

# Missing previous purchases for new customers
df_domain['previous_purchases'].fillna(0, inplace=True)

print("Domain-specific imputation:")
print(df_domain)

# Pros: Uses business logic, makes sense
# Cons: Requires domain expertise, not generalizable
```

## Choosing the Right Strategy

```python
def recommend_imputation_strategy(df, col):
    """Recommend imputation strategy based on data characteristics"""

    missing_pct = df[col].isna().sum() / len(df)

    print(f"\nAnalyzing column: {col}")
    print(f"Missing percentage: {missing_pct*100:.1f}%")

    # Too much missing
    if missing_pct > 0.5:
        print("Recommendation: DROP COLUMN (>50% missing)")
        return "drop"

    # Check data type
    dtype = df[col].dtype

    if dtype == 'object':
        print("Recommendation: MODE IMPUTATION (categorical)")
        return "mode"

    # Numeric column
    # Check distribution
    if df[col].notna().sum() > 0:
        skewness = df[col].skew()
        print(f"Skewness: {skewness:.2f}")

        if abs(skewness) > 1:
            print("Recommendation: MEDIAN IMPUTATION (skewed)")
            return "median"
        else:
            # Check for correlations with other features
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                correlations = df[numeric_cols].corr()[col].drop(col)
                max_corr = correlations.abs().max()
                print(f"Max correlation with other features: {max_corr:.2f}")

                if max_corr > 0.5:
                    print("Recommendation: KNN or ITERATIVE IMPUTATION (high correlation)")
                    return "knn"
                else:
                    print("Recommendation: MEAN IMPUTATION (low correlation)")
                    return "mean"

# Example usage
for col in df.columns:
    if df[col].isna().any():
        recommend_imputation_strategy(df, col)
```

## Common Pitfalls

### 1. Data Leakage

```python
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# WRONG: Fit imputer on entire dataset
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)  # Uses info from test set!
X_train, X_test = train_test_split(X_imputed)

# CORRECT: Split first, fit only on training data
X_train, X_test = train_test_split(X)
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)  # Use training statistics
```

### 2. Ignoring Missingness Type

```python
# Don't blindly use mean imputation for all missing data
# Check the missingness pattern first!

def check_missingness_type(df, col_with_missing, col_predictor):
    """Simple check for MAR"""
    present_mean = df[df[col_with_missing].notna()][col_predictor].mean()
    missing_mean = df[df[col_with_missing].isna()][col_predictor].mean()

    if abs(present_mean - missing_mean) / present_mean > 0.1:
        print(f"WARNING: {col_with_missing} missingness may depend on {col_predictor}")
        print("Consider model-based imputation instead of mean/median")
```

### 3. Not Documenting Imputation

```python
# Always document what was imputed
imputation_log = {}

for col in df.columns:
    if df[col].isna().any():
        n_missing = df[col].isna().sum()
        imputation_log[col] = {
            'n_missing': n_missing,
            'pct_missing': n_missing / len(df) * 100,
            'strategy': 'median',  # or whatever you used
            'fill_value': df[col].median()  # if applicable
        }

print("Imputation log:")
for col, info in imputation_log.items():
    print(f"{col}: {info}")
```

## Evaluation

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import numpy as np

def compare_imputation_methods(X, y):
    """Compare different imputation methods"""

    methods = {
        'Listwise Deletion': None,
        'Mean': SimpleImputer(strategy='mean'),
        'Median': SimpleImputer(strategy='median'),
        'KNN (k=5)': KNNImputer(n_neighbors=5),
        'Iterative': IterativeImputer(random_state=42)
    }

    results = {}

    for name, imputer in methods.items():
        if name == 'Listwise Deletion':
            # Remove rows with missing values
            mask = X.notna().all(axis=1)
            X_clean = X[mask]
            y_clean = y[mask]
        else:
            X_clean = pd.DataFrame(
                imputer.fit_transform(X),
                columns=X.columns
            )
            y_clean = y

        # Evaluate
        model = RandomForestClassifier(random_state=42)
        scores = cross_val_score(model, X_clean, y_clean, cv=5)

        results[name] = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'n_samples': len(X_clean)
        }

    # Display results
    results_df = pd.DataFrame(results).T
    print("\nImputation Method Comparison:")
    print(results_df.sort_values('mean_score', ascending=False))

    return results_df

# Example usage
# compare_imputation_methods(X, y)
```

## Best Practices Summary

1. **Understand Why Data is Missing**: Analyze MCAR, MAR, or MNAR
2. **Visualize Missing Patterns**: Use heatmaps and correlation analysis
3. **Choose Method Based on Type**: Match strategy to missingness type
4. **Avoid Data Leakage**: Split before imputation
5. **Consider Missing Indicators**: Preserve missingness information
6. **Validate Impact**: Test if imputation improves model performance
7. **Document Everything**: Record what was imputed and how

## Related Topics

- [Data Cleaning](./data-cleaning.md) - Prepare data before handling missing values
- [Outlier Detection](./outlier-detection.md) - Identify extreme values
- [Feature Creation](./feature-creation.md) - Create features from imputed data

## Summary

Missing values are a common challenge in machine learning. The key is understanding WHY data is missing and choosing an appropriate strategy. Simple methods like mean/median imputation work for MCAR data, while MAR and MNAR require more sophisticated approaches. Always validate your imputation strategy's impact on model performance.

**Next Step**: Learn about [outlier detection](./outlier-detection.md) to identify extreme values in your data.

---

*Last Updated: 2026-02-09*

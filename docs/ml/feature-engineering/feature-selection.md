# Feature Selection

## Overview

Feature selection is the process of identifying and selecting the most relevant features for model training. It reduces dimensionality, improves model performance, decreases training time, and enhances interpretability by removing redundant or irrelevant features.

**Key Principle**: More features ≠ better model. Select features that contribute meaningfully to predictions while avoiding overfitting.

## Why Feature Selection Matters

1. **Reduce Overfitting**: Fewer features = less chance to memorize noise
2. **Improve Performance**: Remove irrelevant features that confuse the model
3. **Faster Training**: Fewer features = faster computation
4. **Better Interpretability**: Simpler models are easier to understand
5. **Reduced Storage**: Less data to store and process
6. **Curse of Dimensionality**: High-dimensional spaces need exponentially more data

## Feature Selection vs Dimensionality Reduction

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# Generate high-dimensional data
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=5,
    n_redundant=10,
    n_repeated=0,
    random_state=42
)

print(f"Original data: {X.shape}")

# Feature Selection (selects subset of original features)
selector = SelectKBest(f_classif, k=5)
X_selected = selector.fit_transform(X, y)
selected_features = selector.get_support(indices=True)

print(f"\nFeature Selection: {X_selected.shape}")
print(f"Selected feature indices: {selected_features}")

# Dimensionality Reduction (creates new features)
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

print(f"\nDimensionality Reduction (PCA): {X_pca.shape}")
print("PCA creates 5 NEW features from 20 original features")

print("\n=== Key Difference ===")
print("Feature Selection: Keeps original features (interpretable)")
print("Dimensionality Reduction: Creates new features (less interpretable)")
```

## Feature Selection Methods

### 1. Filter Methods

Select features based on statistical properties, independent of any machine learning algorithm.

#### Correlation-Based Selection

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Generate correlated features
np.random.seed(42)
n = 1000

data = {
    'feature1': np.random.randn(n),
    'feature2': np.random.randn(n),
    'feature3': np.random.randn(n),
}

# Add target
data['target'] = (
    2 * data['feature1'] +
    0.1 * data['feature2'] +
    np.random.randn(n) * 0.5
)

# Add redundant features (highly correlated with existing)
data['feature4'] = data['feature1'] + np.random.randn(n) * 0.1  # Redundant with feature1
data['feature5'] = np.random.randn(n)  # Independent noise

df = pd.DataFrame(data)

print("Dataset shape:", df.shape)

# Calculate correlations
corr_matrix = df.corr()

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Select features based on correlation with target
target_corr = corr_matrix['target'].drop('target').abs().sort_values(ascending=False)

print("\nCorrelation with target:")
print(target_corr)

# Threshold-based selection
threshold = 0.3
selected_features = target_corr[target_corr > threshold].index.tolist()

print(f"\nSelected features (correlation > {threshold}):")
print(selected_features)

# Remove highly correlated features (multicollinearity)
def remove_collinear_features(df, target_var, threshold=0.9):
    """Remove features that are highly correlated with each other"""

    # Calculate correlation matrix
    corr_matrix = df.drop(target_var, axis=1).corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    return to_drop

collinear_features = remove_collinear_features(df, 'target', threshold=0.9)
print(f"\nHighly correlated features to remove: {collinear_features}")

# Final feature set
final_features = [f for f in selected_features if f not in collinear_features]
print(f"\nFinal feature set: {final_features}")
```

#### Statistical Tests

```python
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.datasets import make_classification
import pandas as pd

# Generate classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=3,
    random_state=42
)

# Create DataFrame
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print("Dataset shape:", df.shape)

# ANOVA F-statistic (for classification with continuous features)
f_scores, p_values = f_classif(X, y)

f_scores_df = pd.DataFrame({
    'feature': feature_names,
    'f_score': f_scores,
    'p_value': p_values
}).sort_values('f_score', ascending=False)

print("\nANOVA F-scores:")
print(f_scores_df)

# Mutual Information (captures non-linear relationships)
mi_scores = mutual_info_classif(X, y, random_state=42)

mi_df = pd.DataFrame({
    'feature': feature_names,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("\nMutual Information scores:")
print(mi_df)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# F-scores
axes[0].barh(f_scores_df['feature'], f_scores_df['f_score'])
axes[0].set_xlabel('F-score')
axes[0].set_title('ANOVA F-scores')
axes[0].invert_yaxis()

# MI scores
axes[1].barh(mi_df['feature'], mi_df['mi_score'])
axes[1].set_xlabel('Mutual Information')
axes[1].set_title('Mutual Information Scores')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()

# Select top k features
from sklearn.feature_selection import SelectKBest

# Using F-scores
selector_f = SelectKBest(f_classif, k=5)
X_selected_f = selector_f.fit_transform(X, y)
selected_features_f = [feature_names[i] for i in selector_f.get_support(indices=True)]

print(f"\nTop 5 features (F-score): {selected_features_f}")

# Using MI
selector_mi = SelectKBest(mutual_info_classif, k=5)
X_selected_mi = selector_mi.fit_transform(X, y)
selected_features_mi = [feature_names[i] for i in selector_mi.get_support(indices=True)]

print(f"Top 5 features (MI): {selected_features_mi}")
```

#### Variance Threshold

```python
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import pandas as pd

# Generate data with low-variance features
np.random.seed(42)
data = {
    'high_var': np.random.randn(1000),
    'low_var': np.random.randn(1000) * 0.01,
    'constant': np.ones(1000),
    'quasi_constant': np.random.choice([0, 1], 1000, p=[0.99, 0.01])
}
df = pd.DataFrame(data)

print("Feature variances:")
print(df.var())

# Remove low variance features
selector = VarianceThreshold(threshold=0.01)  # Remove features with variance < 0.01
X_high_var = selector.fit_transform(df)

kept_features = df.columns[selector.get_support()].tolist()
removed_features = df.columns[~selector.get_support()].tolist()

print(f"\nKept features: {kept_features}")
print(f"Removed features: {removed_features}")
```

**When to use Filter Methods:**
- **Quick preprocessing** step before more complex selection
- **Large datasets** where computational efficiency matters
- **Initial exploration** to understand feature relevance
- **Independent of model** type

**Pros:**
- Fast and scalable
- Model-independent
- Good for removing obviously irrelevant features

**Cons:**
- Doesn't consider feature interactions
- May remove features useful in combination
- Ignores model performance

### 2. Wrapper Methods

Select features by iteratively training models with different feature subsets.

#### Recursive Feature Elimination (RFE)

```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt

# Generate dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    random_state=42
)

feature_names = [f'feature_{i}' for i in range(X.shape[1])]

print("Original features:", X.shape[1])

# RFE with fixed number of features
estimator = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=estimator, n_features_to_select=10)
rfe.fit(X, y)

# Get selected features
selected_features = [feature_names[i] for i in range(len(feature_names)) if rfe.support_[i]]
print(f"\nRFE selected {len(selected_features)} features:")
print(selected_features)

# Feature ranking
ranking_df = pd.DataFrame({
    'feature': feature_names,
    'ranking': rfe.ranking_,
    'selected': rfe.support_
}).sort_values('ranking')

print("\nFeature rankings (1 = selected first):")
print(ranking_df)

# RFECV: Automatically find optimal number of features using CV
rfecv = RFECV(
    estimator=estimator,
    step=1,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
rfecv.fit(X, y)

print(f"\nOptimal number of features: {rfecv.n_features_}")

selected_features_cv = [feature_names[i] for i in range(len(feature_names)) if rfecv.support_[i]]
print(f"RFECV selected features: {selected_features_cv}")

# Plot number of features vs cross-validation score
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
         rfecv.cv_results_['mean_test_score'])
plt.xlabel('Number of Features Selected')
plt.ylabel('Cross-Validation Score')
plt.title('RFECV: Optimal Number of Features')
plt.axvline(x=rfecv.n_features_, color='r', linestyle='--',
            label=f'Optimal: {rfecv.n_features_} features')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Compare performance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# All features
model_all = RandomForestClassifier(n_estimators=100, random_state=42)
model_all.fit(X_train, y_train)
score_all = model_all.score(X_test, y_test)

# RFE features
X_train_rfe = rfecv.transform(X_train)
X_test_rfe = rfecv.transform(X_test)
model_rfe = RandomForestClassifier(n_estimators=100, random_state=42)
model_rfe.fit(X_train_rfe, y_train)
score_rfe = model_rfe.score(X_test_rfe, y_test)

print(f"\nAccuracy with all features ({X.shape[1]}): {score_all:.4f}")
print(f"Accuracy with RFE features ({rfecv.n_features_}): {score_rfe:.4f}")
```

#### Forward/Backward Selection

```python
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import pandas as pd

# Generate dataset
X, y = make_classification(
    n_samples=500,
    n_features=10,
    n_informative=5,
    random_state=42
)

feature_names = [f'feature_{i}' for i in range(X.shape[1])]

# Forward Selection
print("=== Forward Selection ===")
estimator = LogisticRegression(max_iter=1000)

sfs_forward = SequentialFeatureSelector(
    estimator,
    k_features=5,
    forward=True,
    scoring='accuracy',
    cv=5
)

sfs_forward.fit(X, y)

print(f"Selected features: {[feature_names[i] for i in sfs_forward.k_feature_idx_]}")
print(f"CV Score: {sfs_forward.k_score_:.4f}")

# Backward Selection
print("\n=== Backward Selection ===")
sfs_backward = SequentialFeatureSelector(
    estimator,
    k_features=5,
    forward=False,
    scoring='accuracy',
    cv=5
)

sfs_backward.fit(X, y)

print(f"Selected features: {[feature_names[i] for i in sfs_backward.k_feature_idx_]}")
print(f"CV Score: {sfs_backward.k_score_:.4f}")
```

**When to use Wrapper Methods:**
- **Small to medium datasets** (computationally expensive)
- **Want optimal feature subset** for specific model
- **Model performance** is the primary goal
- **Feature interactions** matter

**Pros:**
- Considers feature interactions
- Optimizes for specific model
- Usually better performance than filter methods

**Cons:**
- Computationally expensive
- Risk of overfitting
- Model-specific (not generalizable)

### 3. Embedded Methods

Feature selection built into the model training process.

#### L1 Regularization (Lasso)

```python
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate regression dataset
X, y = make_regression(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    noise=10,
    random_state=42
)

feature_names = [f'feature_{i}' for i in range(X.shape[1])]

# Scale features (important for regularization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lasso with different alpha values
alphas = [0.001, 0.01, 0.1, 1, 10]

fig, axes = plt.subplots(1, len(alphas), figsize=(20, 4))

for idx, alpha in enumerate(alphas):
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_scaled, y)

    # Count non-zero coefficients
    n_nonzero = np.sum(lasso.coef_ != 0)

    # Plot coefficients
    axes[idx].bar(range(len(lasso.coef_)), lasso.coef_)
    axes[idx].set_title(f'Alpha={alpha}\n{n_nonzero} features selected')
    axes[idx].set_xlabel('Feature Index')
    axes[idx].set_ylabel('Coefficient')
    axes[idx].axhline(y=0, color='r', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()

# Use LassoCV to find optimal alpha
lasso_cv = LassoCV(cv=5, random_state=42)
lasso_cv.fit(X_scaled, y)

print(f"Optimal alpha: {lasso_cv.alpha_:.4f}")

# Get feature importances
coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': lasso_cv.coef_,
    'abs_coefficient': np.abs(lasso_cv.coef_)
}).sort_values('abs_coefficient', ascending=False)

print("\nFeature coefficients (sorted by magnitude):")
print(coef_df)

# Select features with non-zero coefficients
selected_features = coef_df[coef_df['coefficient'] != 0]['feature'].tolist()
print(f"\nSelected features ({len(selected_features)}):")
print(selected_features)
```

#### Tree-Based Feature Importance

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
import pandas as pd
import matplotlib.pyplot as plt

# Generate dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    random_state=42
)

feature_names = [f'feature_{i}' for i in range(X.shape[1])]

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X, y)

# Feature importances
rf_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

gb_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': gb.feature_importances_
}).sort_values('importance', ascending=False)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].barh(rf_importance['feature'].head(10), rf_importance['importance'].head(10))
axes[0].set_xlabel('Importance')
axes[0].set_title('Random Forest: Top 10 Features')
axes[0].invert_yaxis()

axes[1].barh(gb_importance['feature'].head(10), gb_importance['importance'].head(10))
axes[1].set_xlabel('Importance')
axes[1].set_title('Gradient Boosting: Top 10 Features')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()

# Select features above threshold
threshold = 0.05  # Select features with importance > 5%

rf_selected = rf_importance[rf_importance['importance'] > threshold]['feature'].tolist()
print(f"\nRandom Forest selected {len(rf_selected)} features (threshold={threshold}):")
print(rf_selected)

# SelectFromModel (automated threshold)
from sklearn.feature_selection import SelectFromModel

selector = SelectFromModel(rf, threshold='median')
selector.fit(X, y)

selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
print(f"\nSelectFromModel (median threshold) selected {len(selected_features)} features:")
print(selected_features)
```

#### Permutation Importance

```python
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# Generate and split data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

feature_names = [f'feature_{i}' for i in range(X.shape[1])]

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Calculate permutation importance
perm_importance = permutation_importance(
    rf, X_test, y_test,
    n_repeats=10,
    random_state=42
)

# Create DataFrame
perm_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

print("Permutation Importance:")
print(perm_df.head(10))

# Plot
plt.figure(figsize=(10, 8))
plt.barh(perm_df['feature'].head(15), perm_df['importance_mean'].head(15))
plt.xlabel('Permutation Importance')
plt.title('Top 15 Features by Permutation Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Compare: Built-in importance vs Permutation importance
comparison = pd.DataFrame({
    'feature': feature_names,
    'builtin_importance': rf.feature_importances_,
    'permutation_importance': perm_importance.importances_mean
}).sort_values('permutation_importance', ascending=False)

print("\nComparison: Built-in vs Permutation Importance")
print(comparison.head(10))
```

**When to use Embedded Methods:**
- **During model training** (efficient)
- **Want model-specific importance**
- **Regularization** naturally handles feature selection
- **Interpretability** matters

**Pros:**
- Less computationally expensive than wrapper methods
- Considers feature interactions
- Integrated with model training

**Cons:**
- Model-specific
- May require hyperparameter tuning
- Different methods give different results

## Comparison: All Methods Together

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    SelectKBest, f_classif,
    RFE,
    SelectFromModel
)
from sklearn.linear_model import Lasso
import pandas as pd
import time

# Generate dataset
X, y = make_classification(
    n_samples=1000,
    n_features=50,
    n_informative=15,
    n_redundant=20,
    random_state=42
)

print(f"Original features: {X.shape[1]}")

# Base model
base_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Baseline: All features
start = time.time()
baseline_score = cross_val_score(base_model, X, y, cv=5).mean()
baseline_time = time.time() - start

results = [{
    'Method': 'Baseline (All Features)',
    'n_features': X.shape[1],
    'CV Score': baseline_score,
    'Time (s)': baseline_time
}]

# 1. Filter: SelectKBest
start = time.time()
selector_filter = SelectKBest(f_classif, k=15)
X_filter = selector_filter.fit_transform(X, y)
filter_score = cross_val_score(base_model, X_filter, y, cv=5).mean()
filter_time = time.time() - start

results.append({
    'Method': 'Filter (SelectKBest)',
    'n_features': X_filter.shape[1],
    'CV Score': filter_score,
    'Time (s)': filter_time
})

# 2. Wrapper: RFE
start = time.time()
selector_rfe = RFE(base_model, n_features_to_select=15)
X_rfe = selector_rfe.fit_transform(X, y)
rfe_score = cross_val_score(base_model, X_rfe, y, cv=5).mean()
rfe_time = time.time() - start

results.append({
    'Method': 'Wrapper (RFE)',
    'n_features': X_rfe.shape[1],
    'CV Score': rfe_score,
    'Time (s)': rfe_time
})

# 3. Embedded: Tree-based
start = time.time()
selector_tree = SelectFromModel(base_model, threshold='median')
selector_tree.fit(X, y)
X_tree = selector_tree.transform(X)
tree_score = cross_val_score(base_model, X_tree, y, cv=5).mean()
tree_time = time.time() - start

results.append({
    'Method': 'Embedded (Tree)',
    'n_features': X_tree.shape[1],
    'CV Score': tree_score,
    'Time (s)': tree_time
})

# Create comparison table
comparison_df = pd.DataFrame(results)
print("\n=== Feature Selection Method Comparison ===")
print(comparison_df.to_string(index=False))

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Accuracy comparison
axes[0].barh(comparison_df['Method'], comparison_df['CV Score'])
axes[0].set_xlabel('CV Score')
axes[0].set_title('Model Performance')
axes[0].axvline(x=baseline_score, color='r', linestyle='--', alpha=0.5, label='Baseline')
axes[0].legend()

# Time comparison
axes[1].barh(comparison_df['Method'], comparison_df['Time (s)'])
axes[1].set_xlabel('Time (seconds)')
axes[1].set_title('Computation Time')

# Feature count
axes[2].barh(comparison_df['Method'], comparison_df['n_features'])
axes[2].set_xlabel('Number of Features')
axes[2].set_title('Features Selected')
axes[2].axvline(x=X.shape[1], color='r', linestyle='--', alpha=0.5, label='Original')
axes[2].legend()

plt.tight_layout()
plt.show()
```

## Best Practices

1. **Start Simple**: Begin with filter methods for quick insights
2. **Consider Computation**: Wrapper methods are expensive for large datasets
3. **Use Multiple Methods**: Compare results from different approaches
4. **Cross-Validate**: Always use CV to validate feature selection
5. **Domain Knowledge**: Don't blindly remove features experts consider important
6. **Monitor Performance**: Ensure feature selection improves model
7. **Feature Stability**: Check if selected features are consistent across folds
8. **Document Decisions**: Record why features were selected/removed

## Common Pitfalls

1. **Data Leakage**: Selecting features using entire dataset before splitting
2. **Overfitting**: Selecting features based on test performance
3. **Ignoring Domain**: Removing features that are business-critical
4. **Wrong Metric**: Using different metric for selection vs evaluation
5. **Not Validating**: Not checking if selection improves performance
6. **Removing Correlated Features**: Both might be important
7. **One-Time Selection**: Features should be re-evaluated periodically

## Related Topics

- [Feature Creation](./feature-creation.md) - Create features before selecting
- [Feature Scaling](./feature-scaling.md) - Scale features appropriately
- [Model Evaluation](../evaluation/index.md) - Evaluate selected features

## Summary

Feature selection is crucial for building efficient, interpretable, and accurate models. Use filter methods for quick preprocessing, wrapper methods for optimal subsets (if computationally feasible), and embedded methods for efficiency. Always validate that feature selection improves model performance and prevents overfitting. Remember: the goal is not to minimize features, but to maximize model performance and interpretability.

**Next Step**: Learn about [handling imbalanced data](./imbalanced-data.md) for classification problems with skewed class distributions.

---

*Last Updated: 2026-02-09*

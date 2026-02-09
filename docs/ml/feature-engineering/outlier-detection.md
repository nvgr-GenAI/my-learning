# Outlier Detection

## Overview

Outliers are data points that significantly differ from other observations. They can represent errors, rare events, or genuine extreme values. Proper outlier detection and handling is crucial for building robust machine learning models.

**Key Principle**: Not all outliers should be removed. Understanding their nature and impact is more important than blindly eliminating them.

## Why Outlier Detection Matters

1. **Model Performance**: Outliers can skew model training
2. **Statistical Accuracy**: Affect mean, standard deviation, and correlation
3. **Algorithm Sensitivity**: Some algorithms (linear regression, k-means) are very sensitive
4. **Data Quality**: Can indicate measurement errors or data entry mistakes
5. **Domain Insights**: Outliers might represent important rare events

## Types of Outliers

### 1. Point Outliers (Global)

Single data points far from the rest of the data.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Generate data with outliers
np.random.seed(42)
normal_data = np.random.normal(100, 15, 95)
outliers = np.array([200, 210, 220, 10, 5])
data = np.concatenate([normal_data, outliers])

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(data, bins=30, edgecolor='black')
plt.title('Distribution with Outliers')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.boxplot(data, vert=True)
plt.title('Box Plot showing Outliers')
plt.ylabel('Value')

plt.tight_layout()
plt.show()

print(f"Data range: {data.min():.2f} to {data.max():.2f}")
print(f"Mean: {data.mean():.2f}")
print(f"Median: {np.median(data):.2f}")
print(f"Std Dev: {data.std():.2f}")
```

### 2. Contextual Outliers

Values that are outliers in a specific context.

```python
# Temperature data: 90°F is normal in summer, outlier in winter
dates = pd.date_range('2023-01-01', periods=365, freq='D')
temperatures = []

for date in dates:
    if date.month in [12, 1, 2]:  # Winter
        temp = np.random.normal(40, 10)
    elif date.month in [6, 7, 8]:  # Summer
        temp = np.random.normal(85, 10)
    else:  # Spring/Fall
        temp = np.random.normal(65, 15)
    temperatures.append(temp)

df = pd.DataFrame({'date': dates, 'temperature': temperatures})
df['month'] = df['date'].dt.month

# Add contextual outliers: warm day in winter, cold day in summer
df.loc[15, 'temperature'] = 85  # Warm day in January
df.loc[200, 'temperature'] = 45  # Cold day in July

plt.figure(figsize=(15, 4))
plt.plot(df['date'], df['temperature'])
plt.scatter(df['date'].iloc[15], df['temperature'].iloc[15], color='red', s=100, label='Winter outlier')
plt.scatter(df['date'].iloc[200], df['temperature'].iloc[200], color='blue', s=100, label='Summer outlier')
plt.title('Temperature Over Year (Contextual Outliers)')
plt.xlabel('Date')
plt.ylabel('Temperature (°F)')
plt.legend()
plt.show()

print("Winter temperatures (Dec, Jan, Feb):")
winter = df[df['month'].isin([12, 1, 2])]
print(f"Mean: {winter['temperature'].mean():.2f}, Max: {winter['temperature'].max():.2f}")
```

### 3. Collective Outliers

Collection of data points that are outliers together.

```python
# Network traffic: sudden spike indicating DDoS attack
hours = np.arange(0, 48)
normal_traffic = np.random.poisson(100, 48)

# Simulate DDoS attack for hours 20-24
ddos_traffic = normal_traffic.copy()
ddos_traffic[20:25] = np.random.poisson(1000, 5)

plt.figure(figsize=(12, 4))
plt.plot(hours, ddos_traffic, marker='o')
plt.axvspan(20, 24, alpha=0.3, color='red', label='Collective Outlier (DDoS)')
plt.title('Network Traffic Over Time')
plt.xlabel('Hour')
plt.ylabel('Requests per Second')
plt.legend()
plt.show()
```

## Detection Methods

### 1. Statistical Methods

#### Z-Score Method

```python
import numpy as np
import pandas as pd
from scipy import stats

# Generate sample data
np.random.seed(42)
data = np.random.normal(100, 15, 100)
data = np.append(data, [200, 210, 10, 5])  # Add outliers

df = pd.DataFrame({'value': data})

# Calculate Z-scores
df['z_score'] = np.abs(stats.zscore(df['value']))

# Identify outliers (|z| > 3)
threshold = 3
df['is_outlier_z'] = df['z_score'] > threshold

print("Z-Score Method:")
print(f"Outliers detected: {df['is_outlier_z'].sum()}")
print("\nOutlier values:")
print(df[df['is_outlier_z']][['value', 'z_score']])

# Visualize
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(df.index, df['value'], c=df['is_outlier_z'], cmap='coolwarm')
plt.title('Data Points (Red = Outliers)')
plt.xlabel('Index')
plt.ylabel('Value')

plt.subplot(1, 2, 2)
plt.scatter(df.index, df['z_score'], c=df['is_outlier_z'], cmap='coolwarm')
plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold = {threshold}')
plt.axhline(y=-threshold, color='r', linestyle='--')
plt.title('Z-Scores')
plt.xlabel('Index')
plt.ylabel('Z-Score')
plt.legend()

plt.tight_layout()
plt.show()

# Pros: Simple, works well for normal distributions
# Cons: Sensitive to extreme outliers (affects mean and std), assumes normality
```

#### IQR (Interquartile Range) Method

```python
# More robust than Z-score
def detect_outliers_iqr(data, multiplier=1.5):
    """Detect outliers using IQR method"""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    outliers = (data < lower_bound) | (data > upper_bound)

    return outliers, lower_bound, upper_bound

df['is_outlier_iqr'], lower, upper = detect_outliers_iqr(df['value'])

print("\nIQR Method:")
print(f"Lower bound: {lower:.2f}")
print(f"Upper bound: {upper:.2f}")
print(f"Outliers detected: {df['is_outlier_iqr'].sum()}")
print("\nOutlier values:")
print(df[df['is_outlier_iqr']]['value'])

# Visualize with box plot
plt.figure(figsize=(10, 6))
box = plt.boxplot(df['value'], vert=False, patch_artist=True)
plt.axvline(lower, color='r', linestyle='--', label='Bounds')
plt.axvline(upper, color='r', linestyle='--')
plt.title('Box Plot with IQR Bounds')
plt.xlabel('Value')
plt.legend()
plt.show()

# Pros: Robust to extreme values, doesn't assume normal distribution
# Cons: May flag too many points in skewed distributions
```

#### Modified Z-Score (MAD - Median Absolute Deviation)

```python
def detect_outliers_mad(data, threshold=3.5):
    """Detect outliers using Modified Z-score (MAD)"""
    median = np.median(data)
    mad = np.median(np.abs(data - median))

    if mad == 0:
        mad = np.mean(np.abs(data - median))

    modified_z_scores = 0.6745 * (data - median) / mad
    outliers = np.abs(modified_z_scores) > threshold

    return outliers, modified_z_scores

df['is_outlier_mad'], df['mad_score'] = detect_outliers_mad(df['value'].values)

print("\nModified Z-Score (MAD) Method:")
print(f"Outliers detected: {df['is_outlier_mad'].sum()}")
print("\nOutlier values:")
print(df[df['is_outlier_mad']][['value', 'mad_score']])

# Pros: More robust than Z-score, uses median instead of mean
# Cons: Still parametric, may not work well for multimodal distributions
```

### 2. Machine Learning Methods

#### Isolation Forest

```python
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd

# Generate multivariate data with outliers
np.random.seed(42)
n_samples = 300
n_outliers = 20

# Normal data
X_normal = np.random.randn(n_samples, 2)

# Outliers
X_outliers = np.random.uniform(low=-4, high=4, size=(n_outliers, 2))

X = np.vstack([X_normal, X_outliers])
df = pd.DataFrame(X, columns=['feature1', 'feature2'])

# Fit Isolation Forest
iso_forest = IsolationForest(
    contamination=0.1,  # Expected proportion of outliers
    random_state=42
)

df['outlier_iso'] = iso_forest.fit_predict(df[['feature1', 'feature2']])
# -1 for outliers, 1 for inliers

df['is_outlier_iso'] = df['outlier_iso'] == -1

print("Isolation Forest:")
print(f"Outliers detected: {df['is_outlier_iso'].sum()}")

# Visualize
plt.figure(figsize=(10, 6))
inliers = df[~df['is_outlier_iso']]
outliers = df[df['is_outlier_iso']]

plt.scatter(inliers['feature1'], inliers['feature2'], c='blue', label='Inliers', alpha=0.6)
plt.scatter(outliers['feature1'], outliers['feature2'], c='red', label='Outliers', s=100, alpha=0.8)
plt.title('Isolation Forest Outlier Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Pros: Works well for high-dimensional data, no assumptions about distribution
# Cons: Requires tuning contamination parameter, can be slow for large datasets
```

#### Local Outlier Factor (LOF)

```python
from sklearn.neighbors import LocalOutlierFactor

# Fit LOF
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
df['outlier_lof'] = lof.fit_predict(df[['feature1', 'feature2']])
df['is_outlier_lof'] = df['outlier_lof'] == -1

# LOF scores (more negative = more outlier)
df['lof_score'] = lof.negative_outlier_factor_

print("\nLocal Outlier Factor (LOF):")
print(f"Outliers detected: {df['is_outlier_lof'].sum()}")

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
inliers = df[~df['is_outlier_lof']]
outliers = df[df['is_outlier_lof']]
plt.scatter(inliers['feature1'], inliers['feature2'], c='blue', label='Inliers', alpha=0.6)
plt.scatter(outliers['feature1'], outliers['feature2'], c='red', label='Outliers', s=100, alpha=0.8)
plt.title('LOF Outlier Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(df['feature1'], df['feature2'], c=df['lof_score'], cmap='coolwarm', s=50)
plt.colorbar(label='LOF Score (more negative = outlier)')
plt.title('LOF Scores')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

# Pros: Detects local outliers, good for varying densities
# Cons: Sensitive to k parameter, computationally expensive
```

#### DBSCAN (Density-Based Clustering)

```python
from sklearn.cluster import DBSCAN

# Fit DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['cluster'] = dbscan.fit_predict(df[['feature1', 'feature2']])

# Points with cluster label -1 are outliers
df['is_outlier_dbscan'] = df['cluster'] == -1

print("\nDBSCAN:")
print(f"Outliers detected: {df['is_outlier_dbscan'].sum()}")
print(f"Clusters found: {df['cluster'].max() + 1}")

# Visualize
plt.figure(figsize=(10, 6))
for cluster in df['cluster'].unique():
    if cluster == -1:
        cluster_data = df[df['cluster'] == cluster]
        plt.scatter(cluster_data['feature1'], cluster_data['feature2'],
                   c='red', label='Outliers', s=100, alpha=0.8, marker='x')
    else:
        cluster_data = df[df['cluster'] == cluster]
        plt.scatter(cluster_data['feature1'], cluster_data['feature2'],
                   label=f'Cluster {cluster}', alpha=0.6)

plt.title('DBSCAN Clustering (outliers in red)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Pros: Finds outliers naturally, no contamination parameter needed
# Cons: Sensitive to eps and min_samples parameters
```

### 3. Domain-Specific Methods

```python
# Example: Credit card transaction fraud detection
def detect_transaction_outliers(df):
    """Domain-specific outlier detection for transactions"""

    df['is_outlier'] = False

    # 1. Unusually large transaction
    avg_transaction = df.groupby('customer_id')['amount'].transform('mean')
    std_transaction = df.groupby('customer_id')['amount'].transform('std')
    df.loc[df['amount'] > avg_transaction + 3 * std_transaction, 'is_outlier'] = True

    # 2. Transaction at unusual time (3 AM - 6 AM)
    df.loc[df['hour'].between(3, 6), 'is_outlier'] = True

    # 3. Foreign transaction (if customer hasn't traveled)
    domestic_customers = df.groupby('customer_id')['foreign'].transform('mean') < 0.1
    df.loc[domestic_customers & df['foreign'], 'is_outlier'] = True

    # 4. Multiple transactions in short time
    df['time_diff'] = df.groupby('customer_id')['timestamp'].diff()
    df.loc[df['time_diff'] < pd.Timedelta(minutes=5), 'is_outlier'] = True

    return df

# Pros: Highly accurate, uses domain knowledge
# Cons: Not generalizable, requires expertise
```

## Handling Outliers

### 1. Remove Outliers

```python
# Remove rows with outliers
df_clean = df[~df['is_outlier_iqr']].copy()

print(f"Original rows: {len(df)}")
print(f"After removing outliers: {len(df_clean)}")
print(f"Removed: {len(df) - len(df_clean)} rows")

# When to use:
# - Outliers are clear errors
# - Small percentage of data
# - Not important for prediction
```

### 2. Cap/Winsorize Outliers

```python
def winsorize_outliers(data, lower_percentile=5, upper_percentile=95):
    """Cap outliers at percentiles"""
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)

    data_winsorized = data.copy()
    data_winsorized[data_winsorized < lower_bound] = lower_bound
    data_winsorized[data_winsorized > upper_bound] = upper_bound

    return data_winsorized

df['value_winsorized'] = winsorize_outliers(df['value'])

print("\nWinsorization:")
print(f"Original range: {df['value'].min():.2f} to {df['value'].max():.2f}")
print(f"Winsorized range: {df['value_winsorized'].min():.2f} to {df['value_winsorized'].max():.2f}")

# When to use:
# - Want to reduce outlier impact without losing data
# - Outliers might be legitimate extreme values
```

### 3. Transform Data

```python
# Log transformation
df['value_log'] = np.log1p(df['value'].clip(lower=0))

# Square root transformation
df['value_sqrt'] = np.sqrt(df['value'].clip(lower=0))

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(df['value'], bins=30, edgecolor='black')
axes[0].set_title('Original Data')

axes[1].hist(df['value_log'], bins=30, edgecolor='black')
axes[1].set_title('Log Transformed')

axes[2].hist(df['value_sqrt'], bins=30, edgecolor='black')
axes[2].set_title('Square Root Transformed')

plt.tight_layout()
plt.show()

# When to use:
# - Reduce impact of outliers on model
# - Normalize skewed distributions
```

### 4. Use Robust Models

```python
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.ensemble import RandomForestRegressor

# Models robust to outliers:
# - Huber Regression (robust loss function)
# - Tree-based models (split-based, not distance-based)
# - RANSAC (Random Sample Consensus)

# Example with Huber Regression
model_huber = HuberRegressor()
model_ridge = Ridge()

# Huber is more robust to outliers than Ridge
```

### 5. Separate Model for Outliers

```python
# Train separate models for normal data and outliers
normal_data = df[~df['is_outlier']]
outlier_data = df[df['is_outlier']]

# Model 1: For normal data
model_normal = train_model(normal_data)

# Model 2: For outliers (if they represent valid rare events)
model_outlier = train_model(outlier_data)

# Prediction: Route data to appropriate model
def predict(x):
    if is_outlier(x):
        return model_outlier.predict(x)
    else:
        return model_normal.predict(x)
```

## Comprehensive Example

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

class OutlierDetector:
    """Comprehensive outlier detection pipeline"""

    def __init__(self):
        self.outlier_report = {}

    def detect(self, df, methods=['iqr', 'isolation_forest']):
        """Detect outliers using multiple methods"""

        df_analysis = df.copy()
        outlier_cols = []

        for col in df.select_dtypes(include=[np.number]).columns:
            print(f"\nAnalyzing {col}...")

            # IQR method
            if 'iqr' in methods:
                outliers_iqr, _, _ = self._detect_iqr(df_analysis[col])
                df_analysis[f'{col}_outlier_iqr'] = outliers_iqr
                print(f"  IQR: {outliers_iqr.sum()} outliers")

            # Z-score method
            if 'zscore' in methods:
                outliers_z = self._detect_zscore(df_analysis[col])
                df_analysis[f'{col}_outlier_zscore'] = outliers_z
                print(f"  Z-score: {outliers_z.sum()} outliers")

        # Isolation Forest (multivariate)
        if 'isolation_forest' in methods:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            iso = IsolationForest(contamination=0.1, random_state=42)
            outliers_iso = iso.fit_predict(df_analysis[numeric_cols]) == -1
            df_analysis['outlier_isolation_forest'] = outliers_iso
            print(f"\n  Isolation Forest: {outliers_iso.sum()} outliers")

        # Consensus: outlier if flagged by majority of methods
        outlier_flags = [col for col in df_analysis.columns if 'outlier' in col]
        df_analysis['outlier_consensus'] = df_analysis[outlier_flags].sum(axis=1) >= len(outlier_flags) / 2

        self.outlier_report = {
            'total_rows': len(df_analysis),
            'outliers_found': df_analysis['outlier_consensus'].sum(),
            'outlier_percentage': df_analysis['outlier_consensus'].mean() * 100
        }

        return df_analysis

    def _detect_iqr(self, data, multiplier=1.5):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        return (data < lower) | (data > upper), lower, upper

    def _detect_zscore(self, data, threshold=3):
        from scipy import stats
        z_scores = np.abs(stats.zscore(data))
        return z_scores > threshold

    def visualize_outliers(self, df, original_df):
        """Visualize outliers"""

        numeric_cols = original_df.select_dtypes(include=[np.number]).columns[:4]  # First 4 numeric
        n_cols = len(numeric_cols)

        fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 8))

        for idx, col in enumerate(numeric_cols):
            # Box plot
            ax = axes[0, idx] if n_cols > 1 else axes[0]
            outliers = df['outlier_consensus']
            ax.boxplot([original_df[~outliers][col], original_df[outliers][col]],
                      labels=['Normal', 'Outlier'])
            ax.set_title(f'{col}')
            ax.set_ylabel('Value')

            # Histogram
            ax = axes[1, idx] if n_cols > 1 else axes[1]
            original_df[~outliers][col].hist(ax=ax, alpha=0.6, label='Normal', bins=30)
            original_df[outliers][col].hist(ax=ax, alpha=0.6, label='Outlier', bins=30)
            ax.set_title(f'{col} Distribution')
            ax.legend()

        plt.tight_layout()
        plt.show()

    def get_report(self):
        """Get outlier detection report"""
        return self.outlier_report

# Example usage
np.random.seed(42)
df = pd.DataFrame({
    'feature1': np.concatenate([np.random.normal(100, 15, 95), [200, 210, 220, 10, 5]]),
    'feature2': np.concatenate([np.random.normal(50, 10, 95), [150, 160, -50, -60, 0]]),
    'feature3': np.random.normal(75, 20, 100)
})

detector = OutlierDetector()
df_with_outliers = detector.detect(df, methods=['iqr', 'zscore', 'isolation_forest'])

print("\nOutlier Detection Report:")
print(detector.get_report())

detector.visualize_outliers(df_with_outliers, df)
```

## Best Practices

1. **Understand Your Data**: Know the domain and what constitutes an outlier
2. **Use Multiple Methods**: Combine statistical and ML approaches
3. **Visualize**: Always plot data before and after outlier handling
4. **Don't Auto-Remove**: Investigate outliers before removing them
5. **Document Decisions**: Record which outliers were removed and why
6. **Consider Impact**: Test model performance with and without outliers
7. **Context Matters**: What's an outlier in one context may be normal in another

## Common Pitfalls

1. **Blindly Removing Outliers**: May remove important information
2. **Wrong Method for Data Type**: Using z-score on non-normal data
3. **Not Checking Multivariate Outliers**: Point may be normal in each dimension but outlier in combination
4. **Removing Too Much Data**: Over-aggressive outlier removal
5. **Data Leakage**: Detecting outliers on entire dataset before train/test split

## Related Topics

- [Data Cleaning](./data-cleaning.md) - Clean data before outlier detection
- [Feature Scaling](./feature-scaling.md) - Scale features after handling outliers
- [Missing Values](./missing-values.md) - Handle missing data

## Summary

Outlier detection is an essential step in data preprocessing. Choose detection methods based on your data distribution and use case. Remember that not all outliers should be removed - they might represent important rare events or valuable insights. Always validate the impact of outlier handling on your model's performance.

**Next Step**: Learn about [feature scaling](./feature-scaling.md) to normalize your data.

---

*Last Updated: 2026-02-09*

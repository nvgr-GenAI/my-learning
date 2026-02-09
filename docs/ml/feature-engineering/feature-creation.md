# Feature Creation

## Overview

Feature creation (or feature engineering) is the art of generating new features from existing ones to better represent the underlying problem. Creative feature engineering often makes the difference between a mediocre and an outstanding model.

**Key Principle**: Good features should be informative, non-redundant, and capture domain knowledge that helps the model make better predictions.

## Why Feature Creation Matters

1. **Model Performance**: New features can dramatically improve accuracy
2. **Domain Knowledge**: Incorporates expert understanding into the model
3. **Non-Linear Relationships**: Capture complex patterns linear models miss
4. **Simpler Models**: Better features allow simpler, more interpretable models
5. **Competitive Edge**: Creative features win competitions

## Types of Feature Creation

### 1. Mathematical Transformations

#### Basic Operations

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample data
data = {
    'length': [10, 15, 20, 25, 30],
    'width': [5, 7, 10, 12, 15],
    'height': [3, 4, 5, 6, 7],
    'price': [100, 150, 200, 250, 300]
}
df = pd.DataFrame(data)

print("Original features:")
print(df)

# Create new features through mathematical operations
df['area'] = df['length'] * df['width']  # Product
df['volume'] = df['length'] * df['width'] * df['height']  # Product of three
df['perimeter'] = 2 * (df['length'] + df['width'])  # Linear combination
df['aspect_ratio'] = df['length'] / df['width']  # Ratio
df['diagonal'] = np.sqrt(df['length']**2 + df['width']**2)  # Pythagorean
df['price_per_unit_area'] = df['price'] / df['area']  # Rate

print("\nWith created features:")
print(df)

# Visualize relationships
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original feature vs price
axes[0, 0].scatter(df['length'], df['price'])
axes[0, 0].set_xlabel('Length')
axes[0, 0].set_ylabel('Price')
axes[0, 0].set_title('Original Feature: Length')

# Engineered features vs price
axes[0, 1].scatter(df['area'], df['price'])
axes[0, 1].set_xlabel('Area (length × width)')
axes[0, 1].set_ylabel('Price')
axes[0, 1].set_title('Engineered: Area')

axes[0, 2].scatter(df['volume'], df['price'])
axes[0, 2].set_xlabel('Volume')
axes[0, 2].set_ylabel('Price')
axes[0, 2].set_title('Engineered: Volume')

axes[1, 0].scatter(df['aspect_ratio'], df['price'])
axes[1, 0].set_xlabel('Aspect Ratio')
axes[1, 0].set_ylabel('Price')
axes[1, 0].set_title('Engineered: Aspect Ratio')

axes[1, 1].scatter(df['diagonal'], df['price'])
axes[1, 1].set_xlabel('Diagonal')
axes[1, 1].set_ylabel('Price')
axes[1, 1].set_title('Engineered: Diagonal')

axes[1, 2].scatter(df['price_per_unit_area'], df['price'])
axes[1, 2].set_xlabel('Price per Unit Area')
axes[1, 2].set_ylabel('Price')
axes[1, 2].set_title('Engineered: Price Rate')

plt.tight_layout()
plt.show()
```

#### Statistical Transformations

```python
# Log transformation (for skewed data)
df['log_price'] = np.log1p(df['price'])

# Square root (for moderate skewness)
df['sqrt_area'] = np.sqrt(df['area'])

# Power transformations
df['price_squared'] = df['price'] ** 2
df['price_cubed'] = df['price'] ** 3

# Reciprocal (inverse)
df['inverse_volume'] = 1 / (df['volume'] + 1)  # +1 to avoid division by zero

print("Statistical transformations:")
print(df[['price', 'log_price', 'price_squared']].head())
```

### 2. Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd

# Simple features
X = np.array([[2, 3],
              [3, 4],
              [4, 5]])

df = pd.DataFrame(X, columns=['x1', 'x2'])
print("Original features:")
print(df)

# Generate polynomial features (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

feature_names = poly.get_feature_names_out(['x1', 'x2'])
df_poly = pd.DataFrame(X_poly, columns=feature_names)

print("\nPolynomial features (degree 2):")
print(df_poly)
print("\nFeatures created:")
print("x1, x2 (original)")
print("x1², x1×x2, x2² (degree 2)")

# Degree 3 polynomial
poly3 = PolynomialFeatures(degree=3, include_bias=False)
X_poly3 = poly3.fit_transform(X)
print(f"\nDegree 3 creates {X_poly3.shape[1]} features from {X.shape[1]} original features")

# Visualize impact
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Generate synthetic data with non-linear relationship
np.random.seed(42)
X_demo = np.linspace(0, 10, 100).reshape(-1, 1)
y_demo = 2 + 3*X_demo + 0.5*X_demo**2 + np.random.normal(0, 5, X_demo.shape)

# Linear model
model_linear = LinearRegression()
model_linear.fit(X_demo, y_demo)
y_pred_linear = model_linear.predict(X_demo)
r2_linear = r2_score(y_demo, y_pred_linear)

# Polynomial model (degree 2)
X_demo_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_demo)
model_poly = LinearRegression()
model_poly.fit(X_demo_poly, y_demo)
y_pred_poly = model_poly.predict(X_demo_poly)
r2_poly = r2_score(y_demo, y_pred_poly)

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_demo, y_demo, alpha=0.5, label='Data')
plt.plot(X_demo, y_pred_linear, 'r-', linewidth=2, label=f'Linear (R²={r2_linear:.3f})')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Model')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_demo, y_demo, alpha=0.5, label='Data')
plt.plot(X_demo, y_pred_poly, 'g-', linewidth=2, label=f'Polynomial (R²={r2_poly:.3f})')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Model (degree 2)')
plt.legend()

plt.tight_layout()
plt.show()

print(f"\nLinear model R²: {r2_linear:.4f}")
print(f"Polynomial model R²: {r2_poly:.4f}")
print(f"Improvement: {(r2_poly - r2_linear):.4f}")
```

### 3. Interaction Features

```python
import pandas as pd
import numpy as np

# Real estate example
data = {
    'bedrooms': [2, 3, 4, 2, 5],
    'bathrooms': [1, 2, 2, 1, 3],
    'size_sqft': [1000, 1500, 2000, 900, 2500],
    'age_years': [10, 5, 2, 15, 1],
    'price': [200000, 350000, 450000, 180000, 600000]
}
df = pd.DataFrame(data)

print("Original features:")
print(df)

# Create interaction features
# Multiplicative interactions
df['bed_bath_interaction'] = df['bedrooms'] * df['bathrooms']
df['size_age_interaction'] = df['size_sqft'] * df['age_years']

# Ratio interactions
df['sqft_per_bedroom'] = df['size_sqft'] / df['bedrooms']
df['bath_per_bedroom'] = df['bathrooms'] / df['bedrooms']

# Complex interactions
df['luxury_score'] = (df['bedrooms'] + df['bathrooms']) * df['size_sqft'] / (df['age_years'] + 1)

print("\nWith interaction features:")
print(df)

# Correlation analysis
import seaborn as sns

correlations = df.corr()['price'].sort_values(ascending=False)
print("\nCorrelation with price:")
print(correlations)

# Visualize
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.scatter(df['bedrooms'], df['price'])
plt.xlabel('Bedrooms')
plt.ylabel('Price')
plt.title(f'Bedrooms (r={correlations["bedrooms"]:.3f})')

plt.subplot(1, 3, 2)
plt.scatter(df['bed_bath_interaction'], df['price'])
plt.xlabel('Bedrooms × Bathrooms')
plt.ylabel('Price')
plt.title(f'Interaction (r={correlations["bed_bath_interaction"]:.3f})')

plt.subplot(1, 3, 3)
plt.scatter(df['luxury_score'], df['price'])
plt.xlabel('Luxury Score')
plt.ylabel('Price')
plt.title(f'Luxury Score (r={correlations["luxury_score"]:.3f})')

plt.tight_layout()
plt.show()
```

### 4. Datetime Features

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sales data with timestamps
dates = pd.date_range('2023-01-01', periods=365, freq='D')
np.random.seed(42)

# Simulate sales with patterns
sales = []
for date in dates:
    base_sales = 1000
    # Weekend effect
    if date.dayofweek >= 5:
        base_sales *= 1.5
    # Month-end effect
    if date.day >= 28:
        base_sales *= 1.3
    # Holiday season effect
    if date.month == 12:
        base_sales *= 1.8
    # Add noise
    sales.append(base_sales + np.random.normal(0, 100))

df = pd.DataFrame({'date': dates, 'sales': sales})

print("Original data:")
print(df.head())

# Extract datetime features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
df['day_of_year'] = df['date'].dt.dayofyear
df['week_of_year'] = df['date'].dt.isocalendar().week
df['quarter'] = df['date'].dt.quarter

# Binary flags
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)

# Cyclical encoding (preserves cyclical nature)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Custom business logic features
df['is_holiday_season'] = (df['month'] == 12).astype(int)
df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)

print("\nWith datetime features:")
print(df.head(10))

# Visualize patterns
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Day of week effect
day_of_week_sales = df.groupby('day_of_week')['sales'].mean()
axes[0, 0].bar(range(7), day_of_week_sales)
axes[0, 0].set_xticks(range(7))
axes[0, 0].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
axes[0, 0].set_title('Average Sales by Day of Week')
axes[0, 0].set_ylabel('Sales')

# Month effect
month_sales = df.groupby('month')['sales'].mean()
axes[0, 1].bar(range(1, 13), month_sales)
axes[0, 1].set_title('Average Sales by Month')
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('Sales')

# Weekend vs weekday
weekend_sales = df.groupby('is_weekend')['sales'].mean()
axes[1, 0].bar(['Weekday', 'Weekend'], weekend_sales)
axes[1, 0].set_title('Average Sales: Weekday vs Weekend')
axes[1, 0].set_ylabel('Sales')

# Cyclical encoding visualization (month)
axes[1, 1].scatter(df['month_sin'], df['month_cos'], c=df['month'], cmap='viridis')
axes[1, 1].set_xlabel('Month Sin')
axes[1, 1].set_ylabel('Month Cos')
axes[1, 1].set_title('Cyclical Encoding of Month')
axes[1, 1].set_aspect('equal')

# Time series
axes[2, 0].plot(df['date'], df['sales'], alpha=0.5)
axes[2, 0].set_title('Sales Over Time')
axes[2, 0].set_xlabel('Date')
axes[2, 0].set_ylabel('Sales')

# Quarter effect
quarter_sales = df.groupby('quarter')['sales'].mean()
axes[2, 1].bar([f'Q{i}' for i in range(1, 5)], quarter_sales)
axes[2, 1].set_title('Average Sales by Quarter')
axes[2, 1].set_ylabel('Sales')

plt.tight_layout()
plt.show()

# Feature importance
from sklearn.ensemble import RandomForestRegressor

# Select datetime features
datetime_features = ['month', 'day_of_week', 'day', 'is_weekend',
                     'is_month_end', 'is_holiday_season', 'week_of_year']

X = df[datetime_features]
y = df['sales']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Feature importances
importances = pd.DataFrame({
    'feature': datetime_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nDatetime Feature Importances:")
print(importances)

plt.figure(figsize=(10, 6))
plt.barh(importances['feature'], importances['importance'])
plt.xlabel('Importance')
plt.title('Datetime Feature Importances for Sales Prediction')
plt.tight_layout()
plt.show()
```

### 5. Text Features

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Sample product descriptions
data = {
    'product': ['Product A', 'Product B', 'Product C', 'Product D'],
    'description': [
        'High quality laptop with fast processor',
        'Budget smartphone with good camera',
        'Premium laptop with excellent display',
        'Cheap smartphone with basic features'
    ],
    'price': [1200, 300, 1500, 150]
}
df = pd.DataFrame(data)

print("Original data:")
print(df)

# Basic text features
df['description_length'] = df['description'].str.len()
df['word_count'] = df['description'].str.split().str.len()
df['avg_word_length'] = df['description'].apply(
    lambda x: np.mean([len(word) for word in x.split()])
)

# Keyword presence (domain-specific)
keywords = ['premium', 'quality', 'high', 'excellent']
for keyword in keywords:
    df[f'has_{keyword}'] = df['description'].str.contains(keyword, case=False).astype(int)

print("\nWith basic text features:")
print(df)

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=10)
tfidf_matrix = tfidf.fit_transform(df['description'])
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=[f'tfidf_{word}' for word in tfidf.get_feature_names_out()]
)

df_with_tfidf = pd.concat([df, tfidf_df], axis=1)

print("\nTF-IDF features:")
print(tfidf_df)

# Count vectorization
count_vec = CountVectorizer()
count_matrix = count_vec.fit_transform(df['description'])

print("\nWord counts:")
print(pd.DataFrame(
    count_matrix.toarray(),
    columns=count_vec.get_feature_names_out()
))

# Sentiment/emotion features (simplified)
positive_words = ['good', 'excellent', 'quality', 'premium']
negative_words = ['cheap', 'budget', 'basic']

df['positive_word_count'] = df['description'].apply(
    lambda x: sum(word in x.lower() for word in positive_words)
)
df['negative_word_count'] = df['description'].apply(
    lambda x: sum(word in x.lower() for word in negative_words)
)
df['sentiment_score'] = df['positive_word_count'] - df['negative_word_count']

print("\nSentiment features:")
print(df[['description', 'positive_word_count', 'negative_word_count', 'sentiment_score']])
```

### 6. Aggregation Features

```python
import pandas as pd
import numpy as np

# Transactional data
np.random.seed(42)
transactions = pd.DataFrame({
    'customer_id': np.repeat([1, 2, 3, 4, 5], [5, 3, 7, 2, 4]),
    'transaction_date': pd.date_range('2023-01-01', periods=21, freq='D'),
    'amount': np.random.uniform(10, 200, 21),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Food'], 21)
})

print("Transaction data:")
print(transactions.head(10))

# Aggregate features per customer
customer_features = transactions.groupby('customer_id').agg({
    'amount': ['sum', 'mean', 'median', 'std', 'min', 'max', 'count'],
    'transaction_date': ['min', 'max']
}).reset_index()

# Flatten column names
customer_features.columns = ['_'.join(col).strip('_') for col in customer_features.columns.values]
customer_features.columns = ['customer_id'] + list(customer_features.columns[1:])

print("\nAggregated customer features:")
print(customer_features)

# Time-based aggregations
customer_features['days_active'] = (
    pd.to_datetime(customer_features['transaction_date_max']) -
    pd.to_datetime(customer_features['transaction_date_min'])
).dt.days

customer_features['avg_transaction_per_day'] = (
    customer_features['amount_count'] /
    (customer_features['days_active'] + 1)
)

# Categorical aggregations
category_features = transactions.groupby(['customer_id', 'product_category']).size().unstack(fill_value=0)
category_features.columns = [f'transactions_{cat}' for cat in category_features.columns]

customer_features = customer_features.merge(category_features, on='customer_id', how='left')

# Ratio features
customer_features['electronics_ratio'] = (
    customer_features['transactions_Electronics'] /
    customer_features['amount_count']
)

print("\nEnriched customer features:")
print(customer_features)

# Recency, Frequency, Monetary (RFM) features
current_date = transactions['transaction_date'].max()
rfm = transactions.groupby('customer_id').agg({
    'transaction_date': lambda x: (current_date - x.max()).days,  # Recency
    'customer_id': 'count',  # Frequency
    'amount': 'sum'  # Monetary
}).rename(columns={
    'transaction_date': 'recency_days',
    'customer_id': 'frequency',
    'amount': 'monetary_value'
})

print("\nRFM features:")
print(rfm)
```

### 7. Domain-Specific Features

```python
# E-commerce example
ecommerce_data = {
    'user_id': [1, 2, 3, 4, 5],
    'page_views': [50, 20, 100, 30, 80],
    'time_on_site_minutes': [45, 10, 120, 15, 90],
    'cart_additions': [5, 2, 15, 1, 10],
    'purchases': [1, 0, 3, 0, 2],
    'total_spent': [100, 0, 500, 0, 250]
}
df = pd.DataFrame(ecommerce_data)

print("E-commerce data:")
print(df)

# Domain-specific features
df['conversion_rate'] = df['purchases'] / df['page_views']
df['cart_to_purchase_rate'] = df['purchases'] / (df['cart_additions'] + 1)
df['avg_time_per_page'] = df['time_on_site_minutes'] / df['page_views']
df['avg_order_value'] = df['total_spent'] / (df['purchases'] + 1)
df['engagement_score'] = (
    df['page_views'] * 0.3 +
    df['time_on_site_minutes'] * 0.4 +
    df['cart_additions'] * 0.3
)

# Behavioral segments
df['is_browser'] = ((df['page_views'] > 30) & (df['purchases'] == 0)).astype(int)
df['is_buyer'] = (df['purchases'] > 0).astype(int)
df['is_high_value'] = (df['total_spent'] > 200).astype(int)

print("\nWith domain-specific features:")
print(df)
```

## Creative Feature Engineering Strategies

### 1. Target-Based Features (Be Careful of Leakage!)

```python
# Example: Customer lifetime value prediction
customer_data = pd.DataFrame({
    'customer_id': range(1, 101),
    'first_purchase_amount': np.random.uniform(20, 200, 100),
    'days_since_first_purchase': np.random.uniform(1, 365, 100),
    'total_purchases': np.random.poisson(5, 100) + 1,
    'avg_purchase_amount': np.random.uniform(30, 150, 100)
})

# Create features based on historical patterns (no leakage)
customer_data['purchase_frequency'] = (
    customer_data['total_purchases'] /
    (customer_data['days_since_first_purchase'] + 1) * 30  # Per month
)

customer_data['purchase_acceleration'] = (
    customer_data['avg_purchase_amount'] /
    customer_data['first_purchase_amount']
)

customer_data['estimated_lifetime_value'] = (
    customer_data['avg_purchase_amount'] *
    customer_data['purchase_frequency'] *
    365  # Projected annually
)

print("Target-based features:")
print(customer_data.head())
```

### 2. Clustering-Based Features

```python
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

# Original features
np.random.seed(42)
data = pd.DataFrame({
    'age': np.random.normal(40, 15, 200),
    'income': np.random.normal(60000, 20000, 200),
    'spending_score': np.random.normal(50, 25, 200)
})

print("Original data:")
print(data.head())

# Create clusters
kmeans = KMeans(n_clusters=5, random_state=42)
data['customer_segment'] = kmeans.fit_predict(data)

# Distance to cluster centers as features
distances = kmeans.transform(data[['age', 'income', 'spending_score']])
for i in range(5):
    data[f'distance_to_cluster_{i}'] = distances[:, i]

# Cluster membership features
data_with_clusters = pd.get_dummies(data, columns=['customer_segment'], prefix='segment')

print("\nWith clustering features:")
print(data_with_clusters.head())

# Visualize
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(data['age'], data['income'], c=data['customer_segment'], cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Customer Segments')
plt.colorbar(label='Segment')

plt.subplot(1, 2, 2)
plt.scatter(data['spending_score'], data['income'], c=data['customer_segment'], cmap='viridis')
plt.xlabel('Spending Score')
plt.ylabel('Income')
plt.title('Customer Segments')
plt.colorbar(label='Segment')

plt.tight_layout()
plt.show()
```

### 3. Feature Crosses

```python
# Combining categorical features
location_data = pd.DataFrame({
    'city': ['NYC', 'NYC', 'LA', 'LA', 'Chicago'],
    'time_of_day': ['morning', 'evening', 'morning', 'evening', 'morning'],
    'day_type': ['weekday', 'weekend', 'weekday', 'weekend', 'weekday'],
    'demand': [100, 150, 80, 120, 90]
})

# Create feature crosses
location_data['city_x_time'] = (
    location_data['city'] + '_' + location_data['time_of_day']
)

location_data['city_x_day_type'] = (
    location_data['city'] + '_' + location_data['day_type']
)

location_data['time_x_day_type'] = (
    location_data['time_of_day'] + '_' + location_data['day_type']
)

location_data['city_x_time_x_day'] = (
    location_data['city'] + '_' +
    location_data['time_of_day'] + '_' +
    location_data['day_type']
)

print("Feature crosses:")
print(location_data)

# One-hot encode the crosses
location_encoded = pd.get_dummies(
    location_data,
    columns=['city_x_time', 'city_x_day_type']
)

print("\nEncoded feature crosses:")
print(location_encoded)
```

## Best Practices

1. **Start with Domain Knowledge**: Understand the problem and data
2. **Explore Relationships**: Visualize data to identify patterns
3. **Create Hypotheses**: Think about what features might be predictive
4. **Test Features**: Validate if new features improve model performance
5. **Avoid Leakage**: Never use future information
6. **Keep It Simple**: Start with simple features before complex ones
7. **Document Logic**: Explain why each feature was created
8. **Monitor Performance**: Check if features help or hurt the model

## Feature Creation Checklist

```python
def feature_creation_checklist(df):
    """Checklist for comprehensive feature creation"""

    print("=== FEATURE CREATION CHECKLIST ===\n")

    # 1. Mathematical transformations
    print("[ ] Created ratio features (A/B)")
    print("[ ] Created product features (A*B)")
    print("[ ] Created difference features (A-B)")
    print("[ ] Applied log/sqrt transformations for skewed features")

    # 2. Polynomial features
    print("\n[ ] Created polynomial features for non-linear relationships")
    print("[ ] Created interaction terms for important feature pairs")

    # 3. Datetime features
    if df.select_dtypes(include=['datetime64']).shape[1] > 0:
        print("\n[ ] Extracted year, month, day, day_of_week")
        print("[ ] Created is_weekend, is_month_end flags")
        print("[ ] Applied cyclical encoding for periodic features")
        print("[ ] Calculated time differences")

    # 4. Text features
    text_cols = df.select_dtypes(include=['object']).columns
    if len(text_cols) > 0:
        print("\n[ ] Calculated text length, word count")
        print("[ ] Created keyword presence indicators")
        print("[ ] Applied TF-IDF or word embeddings")

    # 5. Aggregations
    print("\n[ ] Created group-wise statistics (mean, sum, count)")
    print("[ ] Created rolling window features")
    print("[ ] Calculated ratios of aggregations")

    # 6. Domain-specific
    print("\n[ ] Applied domain expertise")
    print("[ ] Created business logic features")
    print("[ ] Validated features with domain experts")

    # 7. Advanced
    print("\n[ ] Tried clustering-based features")
    print("[ ] Created feature crosses")
    print("[ ] Experimented with embeddings")

    print("\n=== VALIDATION ===")
    print("[ ] Checked for data leakage")
    print("[ ] Tested feature importance")
    print("[ ] Validated improvement in model performance")
    print("[ ] Documented all created features")

# Example usage
# feature_creation_checklist(df)
```

## Common Pitfalls

1. **Data Leakage**: Using future information or target in features
2. **Overfitting**: Creating too many features for small datasets
3. **Not Testing**: Creating features without validating impact
4. **Ignoring Domain**: Pure mathematical features without business logic
5. **Computational Cost**: Creating features that are too slow to compute
6. **Redundancy**: Creating highly correlated features

## Related Topics

- [Feature Selection](./feature-selection.md) - Select best created features
- [Feature Scaling](./feature-scaling.md) - Scale created features appropriately
- [Missing Values](./missing-values.md) - Handle missing values before feature creation

## Summary

Feature creation is where creativity meets data science. Use mathematical transformations, domain knowledge, and exploratory analysis to generate features that capture important patterns. Always validate that new features improve model performance and watch out for data leakage. The best features often come from deep understanding of the problem domain.

**Next Step**: Learn about [feature selection](./feature-selection.md) to choose the most important created features.

---

*Last Updated: 2026-02-09*

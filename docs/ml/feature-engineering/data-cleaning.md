# Data Cleaning

## Overview

Data cleaning is the foundational step in feature engineering where you prepare raw data for machine learning by identifying and correcting errors, inconsistencies, and structural issues. Clean data is essential for building reliable models - "garbage in, garbage out" applies strongly in machine learning.

**Key Principle**: Before creating features or training models, ensure your data accurately represents reality.

## Why Data Cleaning Matters

1. **Model Accuracy**: Dirty data leads to incorrect predictions
2. **Training Stability**: Errors can cause training failures or convergence issues
3. **Feature Quality**: Clean data enables better feature engineering
4. **Production Reliability**: Models trained on clean data generalize better
5. **Business Impact**: Decisions based on bad data can be costly

## Common Data Quality Issues

### 1. Duplicate Records

```python
import pandas as pd
import numpy as np

# Sample data with duplicates
data = {
    'customer_id': [1, 2, 2, 3, 4, 4, 5],
    'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David', 'Eve'],
    'email': ['alice@email.com', 'bob@email.com', 'bob@email.com',
              'charlie@email.com', 'david@email.com', 'david@email.com', 'eve@email.com'],
    'purchase_amount': [100, 200, 200, 150, 300, 300, 250]
}
df = pd.DataFrame(data)

# Check for duplicates
print("Total rows:", len(df))
print("Duplicate rows:", df.duplicated().sum())
print("\nDuplicate rows:")
print(df[df.duplicated(keep=False)])

# Remove exact duplicates
df_clean = df.drop_duplicates()
print(f"\nAfter removing duplicates: {len(df_clean)} rows")

# Remove duplicates based on specific columns
df_clean = df.drop_duplicates(subset=['customer_id', 'email'])
print(f"After removing duplicates by customer_id and email: {len(df_clean)} rows")
```

**Output:**
```
Total rows: 7
Duplicate rows: 2

Duplicate rows:
   customer_id    name            email  purchase_amount
1            2     Bob    bob@email.com              200
2            2     Bob    bob@email.com              200
4            4   David  david@email.com              300
5            4   David  david@email.com              300

After removing duplicates: 5 rows
After removing duplicates by customer_id and email: 5 rows
```

### 2. Inconsistent Formatting

```python
# Sample data with formatting issues
data = {
    'product_name': ['iPhone 13', 'iphone 13', 'IPHONE 13', 'iPhone  13'],
    'category': ['Electronics', 'electronics', 'ELECTRONICS', 'Electronics '],
    'price': ['$999', '999', '$999.00', '999.0'],
    'date': ['2023-01-15', '01/15/2023', '15-Jan-2023', '2023.01.15']
}
df = pd.DataFrame(data)

print("Original data:")
print(df)

# Clean text: lowercase, strip whitespace, remove extra spaces
df['product_name_clean'] = df['product_name'].str.lower().str.strip().str.replace('\s+', ' ', regex=True)
df['category_clean'] = df['category'].str.lower().str.strip()

# Clean price: remove $, convert to float
df['price_clean'] = df['price'].str.replace('$', '', regex=False).astype(float)

# Standardize date format
df['date_clean'] = pd.to_datetime(df['date'], infer_datetime_format=True)

print("\nCleaned data:")
print(df[['product_name_clean', 'category_clean', 'price_clean', 'date_clean']])
```

**Output:**
```
Original data:
  product_name      category    price         date
0     iPhone 13   Electronics     $999   2023-01-15
1     iphone 13   electronics      999   01/15/2023
2    IPHONE 13  ELECTRONICS  $999.00  15-Jan-2023
3   iPhone  13  Electronics     999.0   2023.01.15

Cleaned data:
  product_name_clean category_clean  price_clean date_clean
0           iphone 13    electronics        999.0 2023-01-15
1           iphone 13    electronics        999.0 2023-01-15
2           iphone 13    electronics        999.0 2023-01-15
3           iphone 13    electronics        999.0 2023-01-15
```

### 3. Invalid Values

```python
# Sample data with invalid values
data = {
    'age': [25, -5, 150, 30, 999, 28, 0],
    'temperature': [98.6, 102.5, -50, 99.1, 500, 98.2, 97.8],
    'email': ['valid@email.com', 'invalid.email', 'test@test.com',
              'missing', 'another@email.com', '', 'final@email.com'],
    'phone': ['555-1234', '123', '555-5678', 'N/A', '555-9999', '000-0000', '555-1111']
}
df = pd.DataFrame(data)

print("Original data:")
print(df)

# Validate age (should be between 0-120)
df['age_valid'] = df['age'].apply(lambda x: x if 0 < x < 120 else np.nan)

# Validate temperature (should be between 95-106°F)
df['temp_valid'] = df['temperature'].apply(lambda x: x if 95 < x < 106 else np.nan)

# Validate email (simple regex check)
import re
email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
df['email_valid'] = df['email'].apply(
    lambda x: x if pd.notna(x) and re.match(email_pattern, str(x)) else np.nan
)

# Validate phone (should match pattern XXX-XXXX)
phone_pattern = r'^\d{3}-\d{4}$'
df['phone_valid'] = df['phone'].apply(
    lambda x: x if pd.notna(x) and re.match(phone_pattern, str(x)) and x != '000-0000' else np.nan
)

print("\nValidated data:")
print(df[['age_valid', 'temp_valid', 'email_valid', 'phone_valid']])
print(f"\nInvalid values found:")
print(f"Age: {df['age_valid'].isna().sum()}")
print(f"Temperature: {df['temp_valid'].isna().sum()}")
print(f"Email: {df['email_valid'].isna().sum()}")
print(f"Phone: {df['phone_valid'].isna().sum()}")
```

### 4. Data Type Issues

```python
# Sample data with type issues
data = {
    'order_id': ['1001', '1002', '1003', '1004', '1005'],
    'quantity': ['5', '10', 'N/A', '7', '12'],
    'is_shipped': ['True', 'False', '1', '0', 'Yes'],
    'price': [99.99, '149.99', 'FREE', 79.99, '199.99']
}
df = pd.DataFrame(data)

print("Original dtypes:")
print(df.dtypes)
print("\nOriginal data:")
print(df)

# Convert order_id to integer
df['order_id'] = pd.to_numeric(df['order_id'])

# Convert quantity to numeric, invalid values become NaN
df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')

# Convert boolean strings to actual boolean
boolean_map = {'True': True, 'False': False, '1': True, '0': False, 'Yes': True, 'No': False}
df['is_shipped'] = df['is_shipped'].map(boolean_map)

# Convert price to numeric, handling special cases
df['price'] = pd.to_numeric(df['price'], errors='coerce')

print("\nCleaned dtypes:")
print(df.dtypes)
print("\nCleaned data:")
print(df)
```

### 5. Typos and Spelling Errors

```python
from difflib import get_close_matches

# Sample data with typos
data = {
    'city': ['New York', 'New Yrok', 'Los Angeles', 'Los Angele',
             'Chicago', 'Chicgo', 'Houston', 'Huston', 'Phoenix', 'Phoneix']
}
df = pd.DataFrame(data)

print("Original cities:")
print(df['city'].value_counts())

# Define known correct spellings
correct_cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']

# Function to fix typos
def fix_typo(text, correct_list, threshold=0.8):
    matches = get_close_matches(text, correct_list, n=1, cutoff=threshold)
    return matches[0] if matches else text

# Apply correction
df['city_corrected'] = df['city'].apply(lambda x: fix_typo(x, correct_cities))

print("\nCorrected cities:")
print(df['city_corrected'].value_counts())

print("\nBefore and after:")
print(df)
```

## Data Cleaning Pipeline

### Complete Example

```python
import pandas as pd
import numpy as np
import re

class DataCleaner:
    """Comprehensive data cleaning pipeline"""

    def __init__(self):
        self.cleaning_log = []

    def clean(self, df):
        """Main cleaning method"""
        df_clean = df.copy()

        # 1. Remove duplicates
        df_clean = self._remove_duplicates(df_clean)

        # 2. Fix data types
        df_clean = self._fix_data_types(df_clean)

        # 3. Standardize text
        df_clean = self._standardize_text(df_clean)

        # 4. Validate values
        df_clean = self._validate_values(df_clean)

        # 5. Handle structural issues
        df_clean = self._fix_structural_issues(df_clean)

        return df_clean

    def _remove_duplicates(self, df):
        """Remove duplicate rows"""
        n_before = len(df)
        df_clean = df.drop_duplicates()
        n_after = len(df_clean)

        if n_before != n_after:
            self.cleaning_log.append(f"Removed {n_before - n_after} duplicate rows")

        return df_clean

    def _fix_data_types(self, df):
        """Fix incorrect data types"""
        df_clean = df.copy()

        for col in df_clean.columns:
            # Try to convert to numeric if mostly numeric
            if df_clean[col].dtype == 'object':
                numeric_converted = pd.to_numeric(df_clean[col], errors='coerce')
                if numeric_converted.notna().sum() / len(df_clean) > 0.8:
                    df_clean[col] = numeric_converted
                    self.cleaning_log.append(f"Converted {col} to numeric")

        return df_clean

    def _standardize_text(self, df):
        """Standardize text columns"""
        df_clean = df.copy()

        for col in df_clean.select_dtypes(include=['object']).columns:
            if df_clean[col].dtype == 'object':
                # Strip whitespace
                df_clean[col] = df_clean[col].str.strip()
                # Remove extra spaces
                df_clean[col] = df_clean[col].str.replace('\s+', ' ', regex=True)
                # Convert to lowercase (optional, depends on use case)
                # df_clean[col] = df_clean[col].str.lower()

        return df_clean

    def _validate_values(self, df):
        """Validate values based on rules"""
        df_clean = df.copy()

        # Example: Validate age
        if 'age' in df_clean.columns:
            invalid_ages = (df_clean['age'] < 0) | (df_clean['age'] > 120)
            if invalid_ages.sum() > 0:
                df_clean.loc[invalid_ages, 'age'] = np.nan
                self.cleaning_log.append(f"Set {invalid_ages.sum()} invalid ages to NaN")

        # Example: Validate email
        if 'email' in df_clean.columns:
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            invalid_emails = ~df_clean['email'].str.match(email_pattern, na=False)
            if invalid_emails.sum() > 0:
                df_clean.loc[invalid_emails, 'email'] = np.nan
                self.cleaning_log.append(f"Set {invalid_emails.sum()} invalid emails to NaN")

        return df_clean

    def _fix_structural_issues(self, df):
        """Fix structural data issues"""
        df_clean = df.copy()

        # Remove columns with all NaN
        null_cols = df_clean.columns[df_clean.isna().all()].tolist()
        if null_cols:
            df_clean = df_clean.drop(columns=null_cols)
            self.cleaning_log.append(f"Removed columns with all NaN: {null_cols}")

        # Remove rows with all NaN
        n_before = len(df_clean)
        df_clean = df_clean.dropna(how='all')
        n_after = len(df_clean)
        if n_before != n_after:
            self.cleaning_log.append(f"Removed {n_before - n_after} rows with all NaN")

        return df_clean

    def get_cleaning_report(self):
        """Get cleaning report"""
        return "\n".join(self.cleaning_log)

# Example usage
data = {
    'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David'],
    'age': [25, -5, 30, 150, 28],
    'email': ['alice@email.com', 'invalid', 'bob@email.com', 'charlie@email.com', 'david@email.com'],
    'salary': ['50000', '60000', '60000', 'N/A', '75000']
}
df = pd.DataFrame(data)

print("Original data:")
print(df)
print("\nOriginal shape:", df.shape)

# Clean data
cleaner = DataCleaner()
df_clean = cleaner.clean(df)

print("\nCleaned data:")
print(df_clean)
print("\nCleaned shape:", df_clean.shape)
print("\nCleaning report:")
print(cleaner.get_cleaning_report())
```

## Best Practices

### 1. Always Inspect First

```python
# Before cleaning, understand your data
def inspect_data(df):
    """Comprehensive data inspection"""
    print("=" * 50)
    print("DATA INSPECTION REPORT")
    print("=" * 50)

    # Basic info
    print(f"\nShape: {df.shape}")
    print(f"\nData types:\n{df.dtypes}")

    # Missing values
    print(f"\nMissing values:\n{df.isnull().sum()}")

    # Duplicates
    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    # Numeric columns statistics
    print(f"\nNumeric columns statistics:")
    print(df.describe())

    # Categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    print(f"\nCategorical columns: {list(cat_cols)}")
    for col in cat_cols:
        print(f"\n{col} - Unique values: {df[col].nunique()}")
        print(df[col].value_counts().head())

# Use it
inspect_data(df)
```

### 2. Document Your Cleaning Steps

```python
# Keep a cleaning log
cleaning_steps = {
    'date': '2026-02-09',
    'dataset': 'customer_data.csv',
    'steps': [
        '1. Removed 15 duplicate rows',
        '2. Converted price to numeric (3 invalid values → NaN)',
        '3. Standardized city names (fixed 8 typos)',
        '4. Validated emails (marked 5 as invalid)',
        '5. Fixed date formats to YYYY-MM-DD'
    ],
    'rows_before': 1000,
    'rows_after': 985
}

# Save for reproducibility
import json
with open('cleaning_log.json', 'w') as f:
    json.dump(cleaning_steps, f, indent=2)
```

### 3. Create Reusable Functions

```python
# Build a library of cleaning functions
def remove_special_chars(text):
    """Remove special characters from text"""
    return re.sub(r'[^a-zA-Z0-9\s]', '', str(text))

def standardize_phone(phone):
    """Standardize phone numbers to XXX-XXX-XXXX"""
    digits = re.sub(r'\D', '', str(phone))
    if len(digits) == 10:
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    return np.nan

def cap_outliers(series, lower_percentile=1, upper_percentile=99):
    """Cap outliers at percentiles"""
    lower_bound = series.quantile(lower_percentile / 100)
    upper_bound = series.quantile(upper_percentile / 100)
    return series.clip(lower_bound, upper_bound)

# Apply to dataframe
df['phone_clean'] = df['phone'].apply(standardize_phone)
df['price_capped'] = cap_outliers(df['price'])
```

### 4. Validate After Cleaning

```python
def validate_cleaned_data(df):
    """Validate data after cleaning"""
    issues = []

    # Check for duplicates
    if df.duplicated().sum() > 0:
        issues.append(f"Still has {df.duplicated().sum()} duplicates")

    # Check data types
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if should be numeric
            try:
                pd.to_numeric(df[col])
                issues.append(f"{col} should be numeric but is object")
            except:
                pass

    # Check for suspicious values
    for col in df.select_dtypes(include=[np.number]).columns:
        if (df[col] < 0).any():
            issues.append(f"{col} has negative values")

    if issues:
        print("Validation issues found:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("All validation checks passed!")

    return len(issues) == 0

# Use it
validate_cleaned_data(df_clean)
```

## Common Pitfalls

1. **Over-cleaning**: Removing too much data
2. **Under-cleaning**: Missing obvious issues
3. **Irreversible operations**: Not keeping original data
4. **Inconsistent rules**: Different cleaning logic for train/test
5. **Ignoring domain knowledge**: Automated cleaning without context
6. **No validation**: Not checking results after cleaning

## Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_cleaning_impact(df_before, df_after):
    """Visualize before/after cleaning"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Missing values comparison
    missing_before = df_before.isnull().sum()
    missing_after = df_after.isnull().sum()

    ax = axes[0, 0]
    x = range(len(missing_before))
    ax.bar([i - 0.2 for i in x], missing_before, 0.4, label='Before', alpha=0.7)
    ax.bar([i + 0.2 for i in x], missing_after, 0.4, label='After', alpha=0.7)
    ax.set_xlabel('Columns')
    ax.set_ylabel('Missing Values')
    ax.set_title('Missing Values: Before vs After')
    ax.set_xticks(x)
    ax.set_xticklabels(missing_before.index, rotation=45)
    ax.legend()

    # 2. Data types comparison
    types_before = df_before.dtypes.value_counts()
    types_after = df_after.dtypes.value_counts()

    ax = axes[0, 1]
    ax.bar(range(len(types_before)), types_before.values, alpha=0.7, label='Before')
    ax.bar(range(len(types_after)), types_after.values, alpha=0.7, label='After')
    ax.set_xlabel('Data Type')
    ax.set_ylabel('Count')
    ax.set_title('Data Types Distribution')
    ax.legend()

    # 3. Row count
    ax = axes[1, 0]
    ax.bar(['Before', 'After'], [len(df_before), len(df_after)])
    ax.set_ylabel('Number of Rows')
    ax.set_title('Row Count Comparison')

    # 4. Summary stats
    ax = axes[1, 1]
    summary = pd.DataFrame({
        'Before': [
            len(df_before),
            df_before.isnull().sum().sum(),
            df_before.duplicated().sum()
        ],
        'After': [
            len(df_after),
            df_after.isnull().sum().sum(),
            df_after.duplicated().sum()
        ]
    }, index=['Total Rows', 'Missing Values', 'Duplicates'])

    summary.plot(kind='bar', ax=ax, rot=0)
    ax.set_title('Overall Comparison')
    ax.legend()

    plt.tight_layout()
    plt.show()

# Use it
visualize_cleaning_impact(df, df_clean)
```

## Related Topics

- [Missing Values](./missing-values.md) - Handle incomplete data
- [Outlier Detection](./outlier-detection.md) - Identify extreme values
- [Feature Scaling](./feature-scaling.md) - Normalize feature ranges

## Summary

Data cleaning is the essential first step in any machine learning project. By systematically identifying and correcting data quality issues, you ensure that your models are trained on reliable data. Remember to always inspect your data first, document your cleaning steps, and validate the results.

**Next Step**: Learn about [handling missing values](./missing-values.md) after cleaning your data.

---

*Last Updated: 2026-02-09*

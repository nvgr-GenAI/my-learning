# Machine Learning Algorithms

## Overview

This section covers the fundamental algorithms used in machine learning, from basic linear models to advanced ensemble methods.

## Categories

### Supervised Learning Algorithms

#### Regression
- Linear Regression
- Polynomial Regression
- Ridge and Lasso Regression
- Elastic Net

#### Classification
- Logistic Regression
- Decision Trees
- Random Forest
- Support Vector Machines
- Naive Bayes
- k-Nearest Neighbors

### Unsupervised Learning Algorithms

#### Clustering
- K-Means
- Hierarchical Clustering
- DBSCAN
- Gaussian Mixture Models

#### Dimensionality Reduction
- Principal Component Analysis (PCA)
- t-SNE
- UMAP
- Linear Discriminant Analysis (LDA)

### Ensemble Methods
- Bagging
- Boosting (AdaBoost, Gradient Boosting, XGBoost)
- Stacking
- Voting Classifiers

## Algorithm Selection

### Factors to Consider
- Data size and dimensionality
- Problem type (classification vs regression)
- Interpretability requirements
- Performance requirements
- Training time constraints

### Decision Tree
```
Is the problem supervised?
├── Yes
│   ├── Is the target continuous?
│   │   ├── Yes → Regression algorithms
│   │   └── No → Classification algorithms
└── No → Unsupervised algorithms
```

## Implementation Examples

### Python with Scikit-learn
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load and split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
```

## Next Steps

- [Deep Learning](../deep-learning/index.md)
- [MLOps](../mlops/index.md)
- [ML Fundamentals](../fundamentals/index.md)

## Types & Methods

Core machine learning problem types and algorithmic approaches.

---

### Classification

**Definition:** A supervised learning task where the goal is to predict discrete categories or classes from input features.

**Explanation:** Classification assigns inputs to predefined categories. Unlike regression (which predicts continuous numbers), classification outputs are discrete labels. The model learns decision boundaries that separate different classes in the feature space.

**Types:**

**Binary Classification:**
- Two classes: yes/no, spam/not-spam, fraud/legitimate
- Output: Single probability or binary label
- Examples: Email spam detection, disease diagnosis, credit default prediction

**Multi-class Classification:**
- Multiple mutually exclusive classes
- Output: One class from N options
- Examples: Digit recognition (0-9), animal species classification, product categorization

**Multi-label Classification:**
- Multiple non-exclusive labels per instance
- Output: Set of applicable labels
- Examples: Movie genre tagging, medical condition diagnosis, image tagging

**Example: Email Classification**

**Input:** Email text, metadata
**Classes:** Spam, Promotions, Social, Primary
**Model learns:** Patterns like "free money" → spam, "meeting invite" → primary

**Common Algorithms:**
- Logistic Regression (linear boundary)
- Decision Trees (rule-based)
- Random Forest (ensemble)
- Support Vector Machines (optimal boundary)
- Neural Networks (complex boundaries)

**When to Use:**
- Output is categorical (not numeric)
- Have labeled training examples
- Need to assign items to predefined groups

**Key Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix

---

### Regression

**Definition:** A supervised learning task where the goal is to predict continuous numerical values from input features.

**Explanation:** Regression models the relationship between features and a continuous target variable. The output can be any real number, making regression suitable for predicting quantities, prices, temperatures, probabilities as continuous values, etc.

**Types:**

**Linear Regression:**
- Assumes linear relationship: y = w₁x₁ + w₂x₂ + ... + b
- Fast, interpretable
- Works well when relationship is approximately linear

**Polynomial Regression:**
- Fits curves: y = w₁x + w₂x² + w₃x³ + ...
- Captures non-linear patterns
- Risk of overfitting with high degrees

**Non-linear Regression:**
- Decision Trees, Random Forests, Neural Networks
- No linearity assumption
- Can model complex relationships

**Example: House Price Prediction**

**Input Features:** Size, location, bedrooms, age, quality
**Output:** Price (continuous number, e.g., $345,000)
**Model learns:** Larger size → higher price, better location → higher price

**Common Algorithms:**
- Linear Regression (baseline)
- Ridge/Lasso Regression (regularized)
- Decision Trees (non-linear)
- Random Forest (robust)
- Gradient Boosting (high performance)
- Neural Networks (complex patterns)

**When to Use:**
- Output is continuous numerical value
- Need to predict quantities, prices, scores, probabilities
- Have labeled training data with numerical targets

**Key Metrics:** MSE, RMSE, MAE, R²

**Regression vs Classification:**

| Aspect | Regression | Classification |
|--------|------------|----------------|
| Output | Continuous number | Discrete category |
| Example | Predict $350k | Predict "expensive" |
| Loss | MSE, MAE | Cross-entropy |
| Evaluation | RMSE, R² | Accuracy, F1 |

---

### Clustering

**Definition:** An unsupervised learning task that groups similar data points together without predefined labels.

**Explanation:** Clustering discovers natural groupings in data by measuring similarity. Unlike classification (which uses labeled data), clustering works with unlabeled data to find hidden structure. The algorithm decides how many groups exist and which points belong together based on feature similarity.

**How It Works:**

1. **Measure similarity:** Define what "similar" means (distance metric)
2. **Group points:** Assign similar points to same cluster
3. **Evaluate:** Check if clusters are meaningful

**Example: Customer Segmentation**

**Input:** Customer data (age, income, purchase history)
**No labels given**
**Algorithm discovers:**
- Cluster 1: Young, low income, frequent small purchases → "Budget Shoppers"
- Cluster 2: Middle-aged, high income, occasional large purchases → "Premium Buyers"
- Cluster 3: Seniors, medium income, consistent purchases → "Loyal Customers"

**Common Algorithms:**

**K-Means:**
- Partitions data into K spherical clusters
- Fast, scalable
- Need to specify K in advance
- Works well for globular clusters

**Hierarchical Clustering:**
- Builds tree of clusters
- Don't need to specify K
- Can visualize with dendrogram
- Slower for large data

**DBSCAN:**
- Density-based clustering
- Automatically finds number of clusters
- Handles arbitrary shapes
- Identifies outliers

**Gaussian Mixture Models (GMM):**
- Probabilistic clustering
- Soft assignments (membership probabilities)
- Assumes Gaussian distributions

**Use Cases:**

1. **Customer segmentation:** Group customers by behavior
2. **Document organization:** Cluster similar documents
3. **Image segmentation:** Group pixels into regions
4. **Anomaly detection:** Points not belonging to any cluster
5. **Data exploration:** Discover hidden patterns

**When to Use:**
- No labeled data available
- Want to discover natural groupings
- Need to organize or summarize data
- Exploratory data analysis

**Evaluation Metrics:**
- Silhouette Score (how well-separated clusters are)
- Davies-Bouldin Index (cluster compactness vs separation)
- Elbow method (optimal K selection)

**Key Insight:** Clustering is exploratory—it finds patterns you didn't know existed. Unlike classification which learns from labels, clustering creates its own groupings based purely on data similarity.

---

### Ensemble Methods

**Definition:** Machine learning techniques that combine multiple models to produce better predictions than any single model alone.

**Explanation:** The core idea: "wisdom of crowds." Just as asking multiple experts often gives better answers than asking one, combining multiple models reduces errors and improves robustness. Different models make different mistakes, so averaging their predictions cancels out individual errors.

**Why Ensembles Work:**

1. **Reduce variance:** Average out individual model quirks
2. **Reduce bias:** Combine models that underfit differently
3. **Improve robustness:** Less sensitive to data variations
4. **Capture diverse patterns:** Different models see different aspects

**Example: Medical Diagnosis**

**Single Doctor (Model):**
- Accuracy: 85%
- May have blind spots in certain conditions

**Panel of 5 Doctors (Ensemble):**
- Each has 85% accuracy individually
- Majority vote accuracy: 92%
- Different specialties catch different issues

**Types of Ensemble Methods:**

### Bagging (Bootstrap Aggregating)

**How It Works:**
1. Create multiple bootstrap samples (random sampling with replacement)
2. Train same model type on each sample
3. Aggregate predictions (vote for classification, average for regression)

**Goal:** Reduce variance (overfitting)

**Example: Random Forest**
- Train many decision trees on different data samples
- Each tree sees slightly different data
- Average their predictions

**When to Use:**
- Model is high variance (overfitting)
- Have unstable models (decision trees)
- Want to reduce overfitting without regularization

### Boosting

**How It Works:**
1. Train model on data
2. Identify misclassified examples
3. Train next model focusing on mistakes
4. Combine models (weighted sum)

**Goal:** Reduce bias (underfitting) by sequentially correcting errors

**Example: Gradient Boosting**
- Model 1: Predicts, makes errors
- Model 2: Learns to correct Model 1's errors
- Model 3: Corrects remaining errors
- Final: Sum of all models

**When to Use:**
- Model is high bias (underfitting)
- Need maximum performance
- Willing to risk overfitting (tune carefully)

**Popular Algorithms:**
- AdaBoost (Adaptive Boosting)
- Gradient Boosting
- XGBoost (extreme gradient boosting)
- LightGBM (fast gradient boosting)
- CatBoost (categorical boosting)

### Stacking

**How It Works:**
1. Train diverse base models (Level 0)
2. Use their predictions as features
3. Train meta-model (Level 1) on these features
4. Meta-model learns how to combine base models

**Goal:** Combine diverse model types optimally

**Example:**
- Base models: Logistic Regression, SVM, Random Forest, Neural Network
- Meta-model: Logistic Regression learns optimal weights for each base model

**When to Use:**
- Have computational resources
- Need maximum performance (competitions)
- Base models are diverse (different algorithms)

**Comparison:**

| Method | Strategy | Goal | Speed |
|--------|----------|------|-------|
| Bagging | Parallel training, average | Reduce variance | Fast |
| Boosting | Sequential training, correct errors | Reduce bias | Medium |
| Stacking | Train meta-learner | Optimal combination | Slow |

**Trade-offs:**

**Advantages:**
- Higher accuracy than single models
- More robust and stable
- Reduce overfitting (bagging) or underfitting (boosting)

**Disadvantages:**
- More complex (harder to interpret)
- Slower training and inference
- Risk of overfitting (especially boosting)
- More hyperparameters to tune

**When to Use Ensembles:**

1. **Performance matters:** Competitions, production systems
2. **Have computational resources:** Can train multiple models
3. **Single models plateau:** Individual models not good enough
4. **Robustness needed:** Want stable predictions

**When NOT to Use:**

1. **Interpretability critical:** Medical diagnosis, loan decisions
2. **Speed matters:** Real-time systems, mobile apps
3. **Limited resources:** Can't train/store multiple models
4. **Simple problem:** Single model already works well

**Key Insight:** Ensembles trade simplicity for performance. Like asking a committee instead of one person, you get better decisions but at the cost of complexity and speed. Use when accuracy justifies the overhead.

---

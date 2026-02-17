## Data

### Dataset

**Definition:** A collection of examples used to train and evaluate a model. Each example consists of features (inputs) and, for supervised learning, labels (correct outputs).

**Explanation:** The dataset is the foundation of machine learning—without data, there's nothing to learn from. Quality and quantity both matter. More data usually (but not always) leads to better models, but only if the data is relevant and representative.

**Example Structure:**

**House Price Dataset (5 examples):**

| Size (sq ft) | Bedrooms | Age (years) | Price ($) |
|--------------|----------|-------------|-----------|
| 1500         | 3        | 10          | 300,000   |
| 2000         | 4        | 5           | 450,000   |
| 1200         | 2        | 15          | 250,000   |

- **Features:** Size, Bedrooms, Age (what we know)
- **Label:** Price (what we want to predict)
- **Observations:** Each row is one house (one example)

**Size Matters:** Small dataset (100s): simple models. Medium (1000s-10000s): most algorithms work. Large (millions+): deep learning shines.

---

### Features (Independent Variables)

**Definition:** The input variables or attributes used to make predictions.

**Explanation:** Features are the characteristics or properties that describe each example. Good feature selection is crucial—relevant features help the model learn, irrelevant features add noise. Feature engineering (creating better features) is often more impactful than algorithm choice.

**Examples Across Domains:**

**Medical diagnosis:**

- Features: age, blood pressure, cholesterol, symptoms, family history
- Target: disease present/absent

**Credit scoring:**

- Features: income, employment length, debt-to-income ratio, payment history, credit age
- Target: loan default risk

**Email spam detection:**

- Features: word frequencies, sender domain, has attachments, link count, ALL CAPS ratio
- Target: spam/not-spam

**Types of Features:**

- **Numerical:** Continuous (temperature: 72.5°F) or discrete (number of bedrooms: 3)
- **Categorical:** Unordered (color: red, blue, green) or ordered (rating: poor, fair, good, excellent)
- **Text:** Raw text requiring preprocessing
- **Images:** Pixels or extracted features
- **Time:** Timestamps, durations

**Quality Over Quantity:** 5 relevant features often beat 100 irrelevant ones.

---

### Labels (Dependent Variables, Targets)

**Definition:** The output or value we want to predict.

**Explanation:** Labels are the "answers" in supervised learning. They represent what you're trying to predict or classify. The model learns the relationship between features and labels, then applies that knowledge to unlabeled examples.

**Examples:**

**Classification Labels (Discrete categories):**

- **Binary:** Spam (yes/no), Fraud (yes/no), Diagnosis (positive/negative)
- **Multi-class:** Animal type (dog/cat/bird), Product category, Sentiment (positive/neutral/negative)
- **Multi-label:** Movie genres (action AND comedy), Article tags (politics, economy, technology)

**Regression Labels (Continuous numbers):**

- **Physical measurements:** Temperature, pressure, speed
- **Financial:** Stock price, revenue, cost
- **Predictions:** Sales forecast, demand estimate, risk score

**Label Quality Matters:**

- **Noisy labels:** Mislabeled examples hurt model performance. If 10% of spam emails are labeled as legitimate, the model learns incorrect patterns.
- **Class imbalance:** 1000 negative examples, 10 positive examples creates challenges. The model might just always predict "negative" for 99% accuracy.

---

### Observations (Examples, Instances, Samples)

**Definition:** Individual rows in a dataset. Each observation contains a set of feature values and (in supervised learning) a label.

**Explanation:** An observation is one complete example or data point. It represents a single entity (person, transaction, image, event) with all its associated features and label.

**Example:** In a customer dataset:

- **Observation 1:** Customer ID 12345, Age: 28, Income: $65,000, Purchases: 12, Churned: No
- **Observation 2:** Customer ID 12346, Age: 45, Income: $95,000, Purchases: 3, Churned: Yes

Each row is one customer observation.

**Terminology Note:** "Example," "instance," "sample," "observation," and "data point" all mean the same thing—one row of data. Different fields use different terms (statistics: observation, computer science: instance).

**Dataset Size:** The number of observations determines how much the model can learn. General rule: need at least 10-20 observations per feature, but more is better (especially for complex models like neural networks).

---

### Structured Data

**Definition:** Data organized in a predefined format with clear relationships, typically in tables with rows and columns.

**Explanation:** Structured data is neat, organized, and easy for both humans and machines to understand. Each field has a specific data type and meaning. Think databases and spreadsheets.

**Examples:**

**Customer Database (Structured):**

| Customer_ID | Name      | Age | Income   | City        | Purchase_Date |
|-------------|-----------|-----|----------|-------------|---------------|
| 001         | Alice     | 32  | $75,000  | Seattle     | 2024-03-15    |
| 002         | Bob       | 28  | $62,000  | Portland    | 2024-03-16    |

**Sensor Readings (Structured):**

| Timestamp           | Temperature | Humidity | Pressure | Sensor_ID |
|---------------------|-------------|----------|----------|-----------|
| 2024-03-15 10:00:00 | 72.3°F      | 45%      | 1013 hPa | SENSOR_A  |

**Advantages:**

- Easy to query (SQL)
- Fast to process
- Clear feature definition
- Most ML algorithms designed for it

**Real-World:** Sales records, financial data, sensor data, user demographics, transaction logs.

---

### Unstructured Data

**Definition:** Data without a predefined structure or organization.

**Explanation:** Unstructured data doesn't fit neatly into tables. It requires preprocessing to extract meaningful features before most ML algorithms can use it. However, unstructured data is extremely common (80-90% of enterprise data) and contains rich information.

**Examples:**

**Text:** Emails, social media posts, articles, reviews, customer support tickets

- "This product is amazing! Best purchase ever. Five stars!"
- No columns or fields, must extract features like sentiment, topics, entities

**Images:** Photos, medical scans, satellite imagery

- 1920×1080 image = 2,073,600 pixels (dimensions), must extract features like edges, shapes, objects

**Audio:** Voice recordings, music, sound effects

- Sound wave, must extract features like frequency, pitch, rhythm

**Video:** Combines images (frames) and audio over time

- Even more complex, 30 frames per second of images plus audio

**Preprocessing Needed:**

- **Text:** Tokenization, word embeddings, TF-IDF
- **Images:** Resize, normalize pixels, extract features (or use CNNs)
- **Audio:** Convert to spectrograms, extract mel-frequency coefficients

**Modern Approach:** Deep learning (CNNs for images, RNNs/Transformers for text) can learn features automatically from raw unstructured data.

---

### Training Set

**Definition:** The portion of data used to train the model. The model learns patterns from this data by adjusting its parameters.

**Explanation:** The training set is where actual learning happens. The model sees these examples repeatedly, makes predictions, gets feedback on errors, and adjusts. Think of it as the practice problems a student uses to learn.

**Example Split:** Total dataset: 10,000 examples

- **Training:** 7,000 examples (70%)
- **Validation:** 1,500 examples (15%)
- **Test:** 1,500 examples (15%)

**Purpose:** Optimize model parameters (weights, biases) to minimize error on these examples.

**Key Rules:**

- **Should be representative:** Must reflect the diversity of real-world data
- **Should be large enough:** More data generally → better learning
- **Never test on training data:** Would give overly optimistic performance estimates

**Analogy:** Practice problems for a student learning math. Working through them builds understanding and skills.

---

### Validation Set

**Definition:** A separate portion of data used during training to tune hyperparameters and monitor performance.

**Explanation:** The validation set provides unbiased feedback during model development. You use it to make decisions: which model architecture to use? What learning rate? When to stop training? It acts as a "reality check" to prevent overfitting to the training set.

**Use Cases:**

**Hyperparameter Tuning:**

- Try learning rate = 0.001 → validation accuracy 82%
- Try learning rate = 0.01 → validation accuracy 85%
- Try learning rate = 0.1 → validation accuracy 79%
- Choose 0.01 (best validation performance)

**Early Stopping:**

- **Epoch 1:** Training loss ↓, Validation loss ↓ (good, keep training)
- **Epoch 10:** Training loss ↓, Validation loss ↓ (good)
- **Epoch 15:** Training loss ↓, Validation loss ↑ (overfitting! stop now)

**Model Selection:**

- Train Decision Tree → validation accuracy 78%
- Train Random Forest → validation accuracy 84%
- Train Neural Network → validation accuracy 82%
- Choose Random Forest

**Why Separate from Test?** Making decisions based on validation set "uses it up" slightly. You need a truly untouched test set for final evaluation.

---

### Test Set

**Definition:** Data completely held out until final evaluation. Used only once to assess the model's real-world performance on truly unseen data.

**Explanation:** The test set is the final exam. You touch it only once, at the very end, to get an honest estimate of how the model will perform in production. It must never influence any training decisions—if you peek and adjust based on test results, it's no longer unbiased.

**The Golden Rule:** Lock the test set in a vault. Don't look until model development is complete.

**Example Workflow:**

1. Split data: 70% train, 15% validation, 15% test (SET ASIDE TEST, don't peek!)
2. Train models, tune hyperparameters using training + validation
3. Select final model based on validation performance
4. **NOW and only now:** Evaluate on test set
5. Report test set performance as expected real-world performance

**What Test Set Tells You:**

- **Test accuracy close to validation accuracy:** Good! Model generalizes well.
- **Test accuracy much worse than validation:** Overfit to validation set through hyperparameter tuning.
- **Test accuracy better than validation:** Lucky split (rare) or data leakage (investigate!)

**Real-World:** Sometimes called "holdout set." In competitions (Kaggle), organizers keep test set completely secret.

---

### Cross-Validation

**Definition:** A technique where the dataset is split into multiple folds, and the model is trained and validated on different combinations of these folds.

**Explanation:** Cross-validation provides more reliable performance estimates than a single train/validation split. Instead of one random split, you train and validate multiple times with different splits, then average the results. This reduces the impact of lucky/unlucky splits.

**Why It Helps:** A single split might by chance put all easy examples in training and hard ones in validation (or vice versa), giving misleading results. Averaging across multiple splits gives truer performance.

**Use Cases:**

- **Small datasets:** When you can't afford to set aside much validation data
- **Model comparison:** More reliable than single split when choosing between algorithms
- **Performance estimation:** More accurate estimate of real-world performance

**Trade-off:** More reliable but computationally expensive (train model K times instead of once).

---

### K-Fold Cross-Validation

**Definition:** A specific cross-validation method where data is divided into K equal parts (folds). The model trains K times, each time using K-1 folds for training and 1 fold for validation.

**Explanation:** K-fold is the most popular cross-validation approach. Common choices: K=5 or K=10.

**Example with K=5:**

**Original dataset:** 1000 examples

**Fold split:**

- Fold 1: examples 1-200
- Fold 2: examples 201-400
- Fold 3: examples 401-600
- Fold 4: examples 601-800
- Fold 5: examples 801-1000

**Training rounds:**

1. **Train on folds 2,3,4,5** (800 examples), validate on fold 1 (200 examples) → accuracy 85%
2. **Train on folds 1,3,4,5**, validate on fold 2 → accuracy 83%
3. **Train on folds 1,2,4,5**, validate on fold 3 → accuracy 87%
4. **Train on folds 1,2,3,5**, validate on fold 4 → accuracy 84%
5. **Train on folds 1,2,3,4**, validate on fold 5 → accuracy 86%

**Final performance estimate:** Average = (85 + 83 + 87 + 84 + 86) / 5 = 85%

**Advantages:** Every example is used for both training and validation. More data used for training than simple train/validation split.

**Choosing K:**

- **K=5 or 10:** Most common, good balance
- **K=N (Leave-One-Out):** Maximum data for training, but very slow
- **Smaller K (3):** Faster but higher variance in estimates

---

### Data Leakage

**Definition:** When information from outside the training dataset inadvertently influences the model, leading to unrealistically good performance on training/validation but poor real-world performance.

**Explanation:** Data leakage is one of the most insidious ML mistakes. The model learns from information it shouldn't have access to, creating the illusion of high performance. When deployed, this "leaked" information isn't available, and performance collapses.

**Types & Examples:**

**1. Target Leakage (most common):**
Feature contains information about the target that wouldn't be available at prediction time.

- **Example:** Predicting hospital readmission using "total hospital cost." But cost includes treatment for complications that caused readmission—it's determined AFTER the outcome! Model scores 98% in validation, 60% in production.

**2. Train-Test Contamination:**
Test set information influences training.

- **Example:** Scaling all data together: `scaler.fit(all_data)`. The training set now knows the mean/std of test examples. Should be: `scaler.fit(train_data)`, then apply to train and test separately.

**3. Temporal Leakage:**
Using future information to predict the past.

- **Example:** Predicting stock price at close using features calculated from daily high/low. But high/low happen DURING the day—you can't know them before close when you need to trade.

**4. Preprocessing Leakage:**

- Filling missing values using global statistics including test set
- Feature selection based on correlations with target across entire dataset

**How to Prevent:**

- Carefully think about what information is available at prediction time
- Perform preprocessing separately on train/test (fit on train, transform both)
- Use time-aware splits for temporal data
- Question any "too good to be true" performance

---

### Target Leakage

**Definition:** A type of data leakage where features contain information about the target that wouldn't be available at prediction time.

**Explanation:** Target leakage happens when a feature is calculated using the target or is a consequence of the target. The model learns spurious relationships that don't exist in the real world.

**Classic Example: Credit Card Fraud**

**Leaked Feature:**

- **Feature:** "fraud investigation initiated"
- **Target:** "transaction is fraudulent"
- **Problem:** Investigations are initiated BECAUSE fraud was detected (after the fact). At prediction time (real-time transaction), you don't know if investigation will happen. Model achieves 99% accuracy but is useless in production.

**More Examples:**

**Medical Diagnosis:**

- **Feature:** "medication prescribed"
- **Target:** "disease present"
- **Problem:** Medication prescribed after diagnosis. At prediction time (pre-diagnosis), this isn't known.

**Loan Default:**

- **Feature:** "number of collection calls"
- **Target:** "defaulted on loan"
- **Problem:** Collection calls happen after missed payments (default), not before.

**How to Detect:**

- Suspiciously high feature importance for seemingly irrelevant features
- Performance much better than domain experts expect
- Think about timeline: Is this feature known BEFORE you need to make the prediction?

---

### Train-Test Contamination

**Definition:** A type of data leakage where information from the test set influences training.

**Explanation:** The test set should represent truly unseen real-world data. If any test set information leaks into training, your performance estimates will be overly optimistic.

**Common Mistake: Scaling on Full Dataset**

**Wrong way:**

```text
1. Combine train + test data (10,000 examples)
2. Calculate mean = 50, std = 10 (using all 10,000)
3. Standardize all data: (x - 50) / 10
4. Split into train (7,000) and test (3,000)
5. Train model
6. Evaluate on test → 92% accuracy
```

**Problem:** Training data knows the mean/std of test data! In production, you won't know future data's statistics.

**Correct way:**

```text
1. Split into train (7,000) and test (3,000) FIRST
2. Calculate mean = 49.8, std = 10.2 (using only 7,000 train examples)
3. Standardize train: (x - 49.8) / 10.2
4. Standardize test: (x - 49.8) / 10.2 (use train statistics!)
5. Train model
6. Evaluate on test → 89% accuracy (more honest)
```

**Other Sources:**

- Feature selection using correlations from full dataset
- Hyperparameter tuning using test set (should use validation)
- Removing duplicates after split (some might span train/test)

---

### Sampling Bias

**Definition:** When the training data isn't representative of the real-world population the model will encounter.

**Explanation:** If your training data is skewed toward certain subgroups or scenarios, the model learns patterns specific to that skewed sample rather than general patterns. Performance will be good on similar data but poor on the broader population.

**Examples:**

**Medical AI trained only on patients from wealthy urban hospitals:**

- **Training data:** Well-documented cases, latest equipment, mostly insured patients
- **Deployment:** Rural clinic with older equipment, different demographics
- **Result:** Poor performance, misdiagnoses

**Facial recognition trained on certain demographics:**

- **Training data:** 90% light-skinned faces, 10% dark-skinned
- **Result:** High accuracy for light-skinned individuals, poor for dark-skinned (well-documented real-world problem)

**Loan approval model trained on historical data:**

- **Training data:** Past approvals (which were biased by human prejudices)
- **Result:** Model perpetuates historical discrimination

**Survey Response Bias:**

- **Training data:** Online survey responses
- **Bias:** Only people with internet access and willingness to respond (misses elderly, poor, busy)

**How to Mitigate:**

- Collect diverse, representative data
- Analyze training data demographics
- Test model performance across subgroups
- Oversample underrepresented groups
- Use stratified sampling

---

### Noise

**Definition:** Random errors or meaningless variations in data that don't represent true underlying patterns.

**Explanation:** Noise is the unavoidable randomness and errors in data. Even with perfect collection, real-world measurements have inherent variability. Models should learn patterns while ignoring noise—this is part of generalization.

**Sources:**

**Measurement Error:**

- Sensor inaccuracy (thermometer reads 72.3°F, true temperature is 72.5°F)
- Human error (data entry mistakes, typos)
- Equipment malfunction

**Natural Variability:**

- Two identical products have slightly different measurements due to manufacturing variations
- Person's weight fluctuates throughout the day

**Random Factors:**

- Customer bought product due to random mood, not predictable pattern
- One-off events (won lottery, had emergency)

**Example: House Prices**

**True pattern:** Price = 100 × Size + 50,000 × Bedrooms + ...

**Observed with noise:**

| Size | Bedrooms | True Price | Observed Price | Noise |
|------|----------|------------|----------------|-------|
| 2000 | 3        | $400,000   | $398,500       | -$1,500 |
| 2000 | 3        | $400,000   | $405,000       | +$5,000 |

Same house features, different prices due to negotiation, timing, luck.

**Impact:** Irreducible error—even perfect models can't predict noise. Goal is to learn signal (pattern) while ignoring noise (randomness).

**Dealing with Noise:**

- More data helps (noise averages out)
- Regularization prevents fitting noise
- Outlier removal for extreme noise
- Feature engineering to extract signal

---

### Missing Values

**Definition:** Absence of data for certain features in some observations, represented as null, NaN, or blank entries.

**Explanation:** Real-world datasets often have incomplete information. A customer might not provide their age, a sensor might fail to record a measurement, or a survey respondent might skip questions. Models typically cannot process missing values directly, requiring strategies to handle them.

**Common Causes:**

- Data collection failure (sensor malfunction, network error)
- User opt-out (privacy concerns, optional fields)
- Data integration issues (joining tables with mismatched records)
- Intentionally missing (question not applicable to this respondent)

**Example: Customer Dataset**

| Customer ID | Age | Income | Purchase |
|-------------|-----|---------|----------|
| 001         | 25  | $50,000 | Yes      |
| 002         | ?   | $75,000 | No       |
| 003         | 35  | ?       | Yes      |
| 004         | ?   | ?       | No       |

Customer 002's age is missing, Customer 003's income is missing.

**Handling Strategies:**

1. **Deletion:** Remove rows/columns with missing values (loses information)
2. **Imputation:** Fill with mean/median/mode (numerical) or most frequent (categorical)
3. **Indicator Variables:** Add binary flag "Age_missing" to preserve missingness signal
4. **Model-based:** Use other features to predict missing values
5. **Leave as-is:** Some algorithms (tree-based) handle missing values naturally

**Key Decision:** Choose strategy based on amount of missing data, whether missingness is random or informative, and model requirements.

---

### Outliers

**Definition:** Data points that differ significantly from other observations, lying far outside the typical range.

**Explanation:** Outliers are extreme values that don't follow the pattern of the majority. They can be genuine rare events (billionaire in income data) or errors (age recorded as 200). Outliers can severely skew model learning, especially for algorithms sensitive to scale.

**Types:**

1. **Univariate:** Extreme in single feature (height = 8 feet)
2. **Multivariate:** Extreme combination (low income + expensive house)
3. **Legitimate:** True rare values (Olympic athlete's performance)
4. **Errors:** Data entry mistakes, measurement failures

**Example: House Prices**

**Typical range:** $200,000 - $500,000

**Dataset:**

- House 1: $300,000 ✓ Normal
- House 2: $450,000 ✓ Normal
- House 3: $5,000 ⚠️ Outlier (typo? should be $500,000?)
- House 4: $15,000,000 ⚠️ Outlier (legitimate mansion? or error?)
- House 5: $350,000 ✓ Normal

**Detection Methods:**

- Statistical: Values beyond mean ± 3 standard deviations
- IQR Method: Below Q1 - 1.5×IQR or above Q3 + 1.5×IQR
- Visual: Box plots, scatter plots

**Handling:**

1. **Investigate:** Determine if error or legitimate
2. **Remove:** If confirmed error
3. **Cap:** Replace with maximum reasonable value (winsorization)
4. **Transform:** Log transformation to reduce impact
5. **Separate Model:** Build different model for outlier segment
6. **Keep:** Use robust algorithms (tree-based, certain losses)

**Impact:** Can dramatically affect linear models and gradient-based algorithms. Tree-based models are naturally more robust.

---

### Data Augmentation

**Definition:** Artificially expanding training data by creating modified versions of existing samples, increasing dataset size and diversity without collecting new data.

**Explanation:** More diverse training data helps models generalize better. Data augmentation applies transformations that preserve the label while changing the input. A rotated image of a cat is still a cat. This technique is especially powerful when labeled data is scarce or expensive.

**Purpose:**

- Increase effective dataset size (helps overfitting)
- Introduce variations model will see in real world
- Improve generalization and robustness
- Balance underrepresented classes

**Example: Image Classification**

**Original image:** Photo of a dog facing right

**Augmented versions (all still labeled "dog"):**

- Horizontal flip: Dog facing left
- Rotation: Dog tilted 15 degrees
- Brightness adjustment: Darker/lighter dog
- Crop: Zoomed portion showing dog's face
- Noise addition: Grainy version
- Color jitter: Slightly different hue

**Result:** 1 labeled image → 6+ training samples

**Domain-Specific Techniques:**

**Images:**

- Geometric: rotation, flipping, scaling, cropping, shearing
- Color: brightness, contrast, saturation, hue adjustment
- Filters: blur, sharpen, noise addition
- Advanced: cutout (random masking), mixup (blend images)

**Text:**

- Synonym replacement: "happy" → "joyful"
- Random insertion/deletion: add/remove words
- Back-translation: English → French → English (paraphrase)
- Contextual word embeddings: BERT-based substitutions

**Time Series:**

- Time warping: stretch/compress timeline
- Magnitude warping: scale values
- Window slicing: different time windows
- Adding noise: Gaussian noise injection

**Tabular Data:**

- SMOTE: synthetic minority oversampling
- Gaussian noise: small random perturbations
- Mixup: linear combinations of samples

**Key Principle:** Augmentations must preserve label correctness. Horizontally flipping "left turn sign" to create "right turn sign" would be wrong—label would no longer match.

**Trade-off:** Computational cost during training vs. improved model performance and generalization.

---


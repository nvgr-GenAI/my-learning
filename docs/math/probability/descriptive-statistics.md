# Chapter 4.11: Descriptive Statistics

## Prerequisites

- **Chapter 4.9**: Expectation and Moments
- **Chapter 4.5**: Random Variables and Distributions
- Basic understanding of probability theory
- Familiarity with summation notation

## Introduction

Descriptive statistics provide tools for summarizing and characterizing empirical data. Unlike theoretical probability distributions, descriptive statistics deal with observed samples and form the foundation of exploratory data analysis (EDA). These techniques bridge the gap between raw data and statistical inference, enabling us to understand data structure before modeling.

In machine learning, descriptive statistics are crucial for:
- Data quality assessment and preprocessing
- Feature engineering and selection
- Model validation and interpretation
- Anomaly detection and outlier handling

---

## 4.11.1 Measures of Central Tendency

Central tendency measures identify the "typical" or "center" value of a dataset.

### Sample Mean

**Definition 4.11.1** (Sample Mean)
Given a sample $x_1, x_2, \ldots, x_n$, the **sample mean** (or arithmetic mean) is:

$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

**Properties:**
1. **Sensitivity to outliers**: Single extreme values significantly affect $\bar{x}$
2. **Optimal for symmetric distributions**: Minimizes sum of squared deviations
3. **Not robust**: Vulnerable to contamination from erroneous measurements

**Example 4.11.1**
Consider model training losses: $\{2.3, 2.1, 2.4, 15.8, 2.2\}$

$$\bar{x} = \frac{2.3 + 2.1 + 2.4 + 15.8 + 2.2}{5} = \frac{24.8}{5} = 4.96$$

The outlier (15.8) pulls the mean far from typical values, making it misleading.

### Sample Median

**Definition 4.11.2** (Sample Median)
The **sample median** $\tilde{x}$ is the middle value when data is ordered:

$$\tilde{x} = \begin{cases}
x_{(\frac{n+1}{2})} & \text{if } n \text{ is odd} \\
\frac{1}{2}(x_{(\frac{n}{2})} + x_{(\frac{n}{2}+1)}) & \text{if } n \text{ is even}
\end{cases}$$

where $x_{(i)}$ denotes the $i$-th order statistic (sorted values).

**Properties:**
1. **Robust to outliers**: Extreme values don't affect median significantly
2. **50th percentile**: Half the data lies below, half above
3. **Optimal for skewed distributions**: Better represents "typical" value

For the training loss example: Sorted values $\{2.1, 2.2, 2.3, 2.4, 15.8\}$, median = 2.3.

### Sample Mode

**Definition 4.11.3** (Sample Mode)
The **mode** is the most frequently occurring value in the dataset.

**Characteristics:**
- Can have multiple modes (bimodal, multimodal distributions)
- Most appropriate for categorical or discrete data
- May not exist for continuous data without binning

**Example 4.11.2** (Computing Mean, Median, and Mode)
Consider the dataset: $\{4, 7, 7, 3, 9, 7, 2\}$ ($n = 7$).

**Mean:** $\bar{x} = \frac{4+7+7+3+9+7+2}{7} = \frac{39}{7} \approx 5.57$

**Median:** Sort the data: $\{2, 3, 4, 7, 7, 7, 9\}$. Since $n = 7$ (odd), the median is the 4th value: $\tilde{x} = 7$.

**Mode:** The value 7 appears 3 times (more than any other), so $\text{mode} = 7$.

Here mean $< $ median $=$ mode, indicating a left-skewed distribution where the low values (2, 3) pull the mean downward.

### Choosing the Right Measure

```
Data Distribution Type → Recommended Measure
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Symmetric, no outliers ────→ Mean (most efficient)
                      │
Skewed distribution ───────→ Median (more representative)
                      │
Heavy-tailed/outliers ─────→ Median or trimmed mean
                      │
Categorical data ──────────→ Mode
                      │
Bimodal distribution ──────→ Report both modes + median
```

**ML/AI Connection: Feature Imputation**
When handling missing data, the choice of central tendency affects model performance:
- Mean imputation: Simple but sensitive to outliers
- Median imputation: Robust, preferred for skewed features
- Mode imputation: Appropriate for categorical features

---

## 4.11.2 Measures of Dispersion

Dispersion measures quantify the spread or variability of data around the center.

### Sample Variance and Standard Deviation

**Definition 4.11.4** (Sample Variance)
The **sample variance** is:

$$s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

The factor $n-1$ (instead of $n$) provides an unbiased estimate of population variance (Bessel's correction).

**Definition 4.11.5** (Sample Standard Deviation)
$$s = \sqrt{s^2} = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2}$$

**Properties:**
- Same units as original data (unlike variance)
- Measures average deviation from mean
- Sensitive to outliers (squared deviations amplify extremes)

**Computational Formula:**
$$s^2 = \frac{1}{n-1}\left(\sum_{i=1}^{n} x_i^2 - n\bar{x}^2\right)$$

This form is numerically more stable for computation.

**Example 4.11.3** (Variance and Standard Deviation Step-by-Step)
Dataset: $\{2, 4, 6, 8, 10\}$ ($n = 5$).

**Step 1 -- Mean:** $\bar{x} = \frac{2+4+6+8+10}{5} = \frac{30}{5} = 6$

**Step 2 -- Squared deviations:**

| $x_i$ | $x_i - \bar{x}$ | $(x_i - \bar{x})^2$ |
|--------|-----------------|---------------------|
| 2      | $-4$           | 16                  |
| 4      | $-2$           | 4                   |
| 6      | $0$            | 0                   |
| 8      | $2$            | 4                   |
| 10     | $4$            | 16                  |

**Step 3 -- Variance:** $s^2 = \frac{16+4+0+4+16}{5-1} = \frac{40}{4} = 10$

**Step 4 -- Standard deviation:** $s = \sqrt{10} \approx 3.16$

On average, data points deviate about 3.16 units from the mean.

### Interquartile Range (IQR)

**Definition 4.11.6** (Quartiles and IQR)
- **First quartile** $Q_1$: 25th percentile
- **Third quartile** $Q_3$: 75th percentile
- **Interquartile range**: $\text{IQR} = Q_3 - Q_1$

The IQR contains the middle 50% of the data and is robust to outliers.

**Example 4.11.4** (Computing Quartiles and IQR)
Dataset: $\{3, 5, 7, 8, 12, 14, 16, 18, 21, 25\}$ ($n = 10$, already sorted).

**$Q_2$ (Median):** $n$ is even, so $Q_2 = \frac{x_{(5)} + x_{(6)}}{2} = \frac{12+14}{2} = 13$

**$Q_1$ (Lower quartile):** Median of the lower half $\{3, 5, 7, 8, 12\}$ $\Rightarrow Q_1 = 7$

**$Q_3$ (Upper quartile):** Median of the upper half $\{14, 16, 18, 21, 25\}$ $\Rightarrow Q_3 = 18$

**IQR:** $Q_3 - Q_1 = 18 - 7 = 11$

The middle 50% of the data spans an interval of width 11.

**Box Plot Representation (ASCII):**
```
              Outlier
                 ×
                          Outlier
                             ○
      |                            |
  ────┼────────┬───┬───┬───────────┼────
      │        │   │   │           │
    Min      Q1  Median Q3        Max
             └───┴───┘
               IQR
      ├────────────────────────────┤
            Whiskers (1.5×IQR)
```

### Range and Mean Absolute Deviation

**Definition 4.11.7** (Range)
$$\text{Range} = x_{\max} - x_{\min}$$

Extremely sensitive to outliers; useful only for quick checks.

**Definition 4.11.8** (Mean Absolute Deviation)
$$\text{MAD} = \frac{1}{n} \sum_{i=1}^{n} |x_i - \bar{x}|$$

More robust than standard deviation, but less commonly used in practice.

**ML/AI Connection: Feature Scaling**

Standardization (z-score normalization) uses mean and standard deviation:

$$z_i = \frac{x_i - \bar{x}}{s}$$

This transforms features to have mean 0 and variance 1, crucial for:
- Gradient descent convergence (prevents feature domination)
- Distance-based algorithms (KNN, K-means)
- Neural networks (batch normalization layers)

Robust scaling uses median and IQR for outlier-prone data:

$$z_i = \frac{x_i - \tilde{x}}{\text{IQR}}$$

---

## 4.11.3 Moments and Shape Descriptors

Higher-order moments characterize distribution shape beyond location and scale.

### Skewness

**Definition 4.11.9** (Sample Skewness)
$$\text{Skewness} = \frac{\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^3}{s^3} = \frac{m_3}{m_2^{3/2}}$$

where $m_k$ is the $k$-th central moment.

**Interpretation:**
- **Skewness = 0**: Symmetric distribution (Gaussian ideal)
- **Skewness > 0**: Right-skewed (long right tail, mean > median)
- **Skewness < 0**: Left-skewed (long left tail, mean < median)

**Visual Representation:**
```
Left-Skewed (Negative)     Symmetric          Right-Skewed (Positive)
       │                      │                        │
   ╱╲  │                    ╱─╲                       │  ╱╲
  ╱  ╲ │                   ╱   ╲                      │ ╱  ╲___
 ╱____╲│                  ╱     ╲                     │╱       ╲
───────┴──────────    ────┴───────┴────         ─────┴─────────┴──
mean < median        mean = median            median < mean
```

**Rule of Thumb:**
- $|\text{Skewness}| < 0.5$: Approximately symmetric
- $0.5 < |\text{Skewness}| < 1$: Moderately skewed
- $|\text{Skewness}| > 1$: Highly skewed

**Example 4.11.6** (Computing and Interpreting Skewness)
Dataset: $\{1, 2, 3, 4, 15\}$ ($n = 5$).

**Step 1 -- Mean and std dev:** $\bar{x} = \frac{1+2+3+4+15}{5} = 5$, $\;s^2 = \frac{16+9+4+1+100}{4} = \frac{130}{4} = 32.5$, $\;s = 5.70$

**Step 2 -- Cubed deviations:** $(1-5)^3 + (2-5)^3 + (3-5)^3 + (4-5)^3 + (15-5)^3 = -64 -27 -8 -1 + 1000 = 900$

**Step 3 -- Skewness:**

$$\text{Skewness} = \frac{\frac{1}{n}\sum(x_i-\bar{x})^3}{s^3} = \frac{900/5}{5.70^3} = \frac{180}{185.19} \approx 0.97$$

**Interpretation:** Skewness $\approx 0.97 > 0$, indicating a moderately right-skewed distribution. The single large value (15) creates a long right tail. A log transform would reduce this skewness before fitting a linear model.

### Kurtosis

**Definition 4.11.10** (Sample Excess Kurtosis)
$$\text{Kurtosis} = \frac{\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^4}{s^4} - 3$$

The subtraction of 3 gives "excess kurtosis" (Gaussian has kurtosis = 0).

**Interpretation:**
- **Kurtosis = 0**: Mesokurtic (normal-like tails)
- **Kurtosis > 0**: Leptokurtic (heavy tails, more outliers)
- **Kurtosis < 0**: Platykurtic (light tails, fewer outliers)

**Visual Representation:**
```
Platykurtic (K<0)      Mesokurtic (K=0)      Leptokurtic (K>0)
     ╱───╲                 ╱─╲                    │
    ╱     ╲               ╱   ╲                  ╱│╲
   ╱       ╲             ╱     ╲                ╱ │ ╲
  ╱         ╲           ╱       ╲              ╱  │  ╲___
 ╱           ╲         ╱         ╲            ╱   │      ╲
─────────────────  ───────────────────  ────────────────────
 Flat, light tails  Normal tails        Peaked, heavy tails
```

**ML/AI Connection: Distribution Validation**

Before applying algorithms assuming Gaussian data (linear regression, LDA):
1. Check skewness and kurtosis
2. If $|\text{Skewness}| > 1$: Consider log transformation or Box-Cox
3. High kurtosis: Expect outliers, use robust methods or outlier removal
4. For neural networks: Batch normalization adapts to distribution shifts during training

---

## 4.11.4 Covariance and Correlation

These measures quantify relationships between two variables.

### Covariance

**Definition 4.11.11** (Sample Covariance)
For paired observations $(x_1, y_1), \ldots, (x_n, y_n)$:

$$\text{cov}(x, y) = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})$$

**Properties:**
- **Positive covariance**: Variables tend to increase together
- **Negative covariance**: One increases as the other decreases
- **Zero covariance**: No linear relationship
- **Scale-dependent**: Units are product of variable units

**Example 4.11.9** (Computing Sample Covariance)
Study hours ($x$) vs. exam score ($y$) for 5 students:

| Student | $x_i$ | $y_i$ | $x_i - \bar{x}$ | $y_i - \bar{y}$ | Product |
|---------|--------|--------|-----------------|-----------------|---------|
| 1       | 1      | 50     | $-2$            | $-20$           | 40      |
| 2       | 2      | 60     | $-1$            | $-10$           | 10      |
| 3       | 3      | 65     | $0$             | $-5$            | 0       |
| 4       | 4      | 80     | $1$             | $10$            | 10      |
| 5       | 5      | 95     | $2$             | $25$            | 50      |

Means: $\bar{x} = 3$, $\bar{y} = 70$.

$$\text{cov}(x,y) = \frac{40+10+0+10+50}{5-1} = \frac{110}{4} = 27.5$$

Positive covariance confirms: more study hours associate with higher scores.

### Pearson Correlation Coefficient

**Definition 4.11.12** (Pearson Correlation)
$$r_{xy} = \frac{\text{cov}(x, y)}{s_x s_y} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

**Properties:**
- **Range**: $-1 \leq r_{xy} \leq 1$
- **$r = 1$**: Perfect positive linear relationship
- **$r = -1$**: Perfect negative linear relationship
- **$r = 0$**: No linear relationship (may have nonlinear relationship!)
- **Scale-invariant**: Standardized measure

**Example 4.11.10** (Computing Pearson Correlation)
Continuing Example 4.11.9 ($\bar{x}=3$, $\bar{y}=70$, $\text{cov}(x,y)=27.5$):

$$s_x = \sqrt{\frac{4+1+0+1+4}{4}} = \sqrt{2.5} = 1.58, \quad s_y = \sqrt{\frac{400+100+25+100+625}{4}} = \sqrt{312.5} = 17.68$$

$$r_{xy} = \frac{\text{cov}(x,y)}{s_x \cdot s_y} = \frac{27.5}{1.58 \times 17.68} = \frac{27.5}{27.93} \approx 0.98$$

Since $r \approx 0.98$, there is a near-perfect positive linear relationship between study hours and exam scores.

**Theorem 4.11.1** (Cauchy-Schwarz for Correlation)
For any two random variables $X$ and $Y$:
$$-1 \leq r_{XY} \leq 1$$

*Proof*: Follows from Cauchy-Schwarz inequality applied to covariance.

### Spearman Rank Correlation

**Definition 4.11.13** (Spearman's Rho)
Pearson correlation applied to rank-transformed data:
$$r_s = 1 - \frac{6\sum_{i=1}^{n} d_i^2}{n(n^2 - 1)}$$

where $d_i$ is the difference between ranks of $x_i$ and $y_i$.

**Advantages:**
- Robust to outliers
- Captures monotonic (not just linear) relationships
- Appropriate for ordinal data

### Correlation ≠ Causation

**Critical Warning:**
Strong correlation does NOT imply causation. Possible explanations:
1. $X$ causes $Y$
2. $Y$ causes $X$
3. Common cause $Z$ affects both (confounding)
4. Spurious correlation (coincidence)

**Example**: Ice cream sales correlate with drowning deaths. Cause: Summer temperature (confounding variable).

**ML/AI Connection: Feature Selection**

Correlation analysis guides feature engineering:

```python
# High correlation with target: Potentially useful features
|r| > 0.7  → Strong candidate
|r| > 0.5  → Moderate candidate

# High inter-feature correlation: Multicollinearity issues
|r| > 0.9  → Consider removing one feature
```

Applications:
- Remove redundant features (reduce model complexity)
- Detect feature leakage (suspiciously high correlations)
- Guide feature creation (combine negatively correlated features)

---

## 4.11.5 Covariance and Correlation Matrices

### Covariance Matrix

**Definition 4.11.14** (Sample Covariance Matrix)
For $p$ features $\mathbf{x} = (x^{(1)}, \ldots, x^{(p)})^T$:

$$\mathbf{S} = \frac{1}{n-1} \sum_{i=1}^{n} (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T$$

$$\mathbf{S} = \begin{bmatrix}
s_1^2 & s_{12} & \cdots & s_{1p} \\
s_{21} & s_2^2 & \cdots & s_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
s_{p1} & s_{p2} & \cdots & s_p^2
\end{bmatrix}$$

where $s_{ij} = \text{cov}(x^{(i)}, x^{(j)})$.

**Properties:**
- Symmetric: $\mathbf{S} = \mathbf{S}^T$
- Positive semi-definite: $\mathbf{v}^T \mathbf{S} \mathbf{v} \geq 0$ for all $\mathbf{v}$
- Diagonal elements are variances

### Correlation Matrix

**Definition 4.11.15** (Sample Correlation Matrix)
$$\mathbf{R} = \begin{bmatrix}
1 & r_{12} & \cdots & r_{1p} \\
r_{21} & 1 & \cdots & r_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
r_{p1} & r_{p2} & \cdots & 1
\end{bmatrix}$$

where $r_{ij} = \frac{s_{ij}}{s_i s_j}$.

**ML/AI Connection: Principal Component Analysis (PCA)**

PCA performs eigendecomposition of the covariance matrix:

$$\mathbf{S} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^T$$

where $\mathbf{\Lambda}$ contains variances along principal components. This enables:
- Dimensionality reduction (retain high-variance components)
- Data visualization (project to 2D/3D)
- Noise reduction (discard low-variance components)

---

## 4.11.6 Data Visualization Concepts

### Histogram

A histogram displays frequency distribution by binning continuous data.

**ASCII Example:**
```
Frequency
    │
 15 │     ┌───┐
    │     │   │
 10 │ ┌───┤   ├───┐
    │ │   │   │   │
  5 │ │   │   │   │   ┌───┐
    │ │   │   │   │   │   │
  0 └─┴───┴───┴───┴───┴───┴───→
      0   10  20  30  40  50  Value
```

**Key Considerations:**
- Bin width affects distribution appearance
- Too few bins: Loss of detail
- Too many bins: Noisy, obscures pattern
- Common rules: Sturges' formula $k = \lceil \log_2 n \rceil + 1$, Scott's rule

### Box Plot (Box-and-Whisker)

Displays five-number summary: $\min, Q_1, \tilde{x}, Q_3, \max$.

**Comparison Across Groups:**
```
Group A:    ├──┬─┬─┬──┤       (symmetric)
Group B:  ├──┬─┬──┬────┤      (right-skewed)
Group C:     ├┬┬┬┤             (low variance)
         ────┴─────────────→
             Value
```

**Example 4.11.5** (Five-Number Summary)
Dataset: $\{1, 3, 5, 6, 8, 11, 14, 17, 20\}$ ($n = 9$, sorted).

| Statistic | Value | How |
|-----------|-------|-----|
| Minimum   | 1     | Smallest value |
| $Q_1$     | 4     | Median of $\{1,3,5,6\}$ = $\frac{3+5}{2}$ |
| Median    | 8     | Middle value (5th of 9) |
| $Q_3$     | 15.5  | Median of $\{11,14,17,20\}$ = $\frac{14+17}{2}$ |
| Maximum   | 20    | Largest value |

$$\text{Five-number summary: } (1,\; 4,\; 8,\; 15.5,\; 20)$$

The IQR $= 15.5 - 4 = 11.5$, and the data is right-skewed since $Q_3 - Q_2 = 7.5 > Q_2 - Q_1 = 4$.

### Q-Q Plot (Quantile-Quantile Plot)

Compares sample quantiles against theoretical distribution quantiles.

**Interpretation:**
```
Normal Q-Q Plot

Theoretical │     /
Quantiles   │   /    ← Points on diagonal:
            │ /         data is normally distributed
            │/
            └────────────→
              Sample Quantiles

Deviations from diagonal:
  - S-curve: Heavy tails
  - Inverted S: Light tails
  - Shift up/down: Location shift
```

Used to validate distributional assumptions before parametric tests.

---

## 4.11.7 Outlier Detection

Outliers are observations significantly deviating from the pattern. Causes:
- Measurement errors
- Data entry mistakes
- Rare but valid extreme events
- Different population

### Z-Score Method

**Definition 4.11.16** (Z-Score)
$$z_i = \frac{x_i - \bar{x}}{s}$$

**Outlier Rule:** Flag observations with $|z_i| > 3$ (assuming normality).

**Example 4.11.7** (Z-Score Standardization)
Dataset: $\{50, 60, 70, 80, 200\}$. We have $\bar{x} = 92$ and $s = 61.24$.

| $x_i$ | $z_i = \frac{x_i - 92}{61.24}$ | Interpretation |
|--------|--------------------------------|----------------|
| 50     | $-0.69$                        | Below average  |
| 60     | $-0.52$                        | Below average  |
| 70     | $-0.36$                        | Near average   |
| 80     | $-0.20$                        | Near average   |
| 200    | $+1.76$                        | Above average  |

No value exceeds $|z| > 3$, so no outlier is flagged -- yet 200 is clearly anomalous. This illustrates the limitation: the outlier inflates $\bar{x}$ and $s$, masking itself.

**Limitation:** Mean and standard deviation themselves are affected by outliers (not robust).

### IQR Method (Tukey's Fences)

**Definition 4.11.17** (Outlier Fences)
- **Lower fence**: $Q_1 - 1.5 \times \text{IQR}$
- **Upper fence**: $Q_3 + 1.5 \times \text{IQR}$

Observations outside fences are flagged as outliers.

**Example 4.11.8** (IQR Outlier Detection)
Dataset: $\{2, 5, 7, 8, 9, 10, 12, 50\}$ ($n = 8$, sorted).

**Step 1 -- Quartiles:** $Q_1 = \frac{5+7}{2} = 6$, $\;Q_3 = \frac{10+12}{2} = 11$

**Step 2 -- IQR:** $11 - 6 = 5$

**Step 3 -- Fences:**

$$\text{Lower fence} = 6 - 1.5 \times 5 = -1.5$$
$$\text{Upper fence} = 11 + 1.5 \times 5 = 18.5$$

**Step 4 -- Flag outliers:** Any $x_i < -1.5$ or $x_i > 18.5$ is an outlier. The value $50 > 18.5$, so **50 is an outlier**. All other values fall within the fences. Unlike the z-score method, IQR correctly identifies 50 because quartiles are not distorted by the extreme value.

**Advantages:**
- Robust to outliers (uses quartiles)
- No distributional assumptions
- Standard in exploratory analysis

### Mahalanobis Distance

**Definition 4.11.18** (Mahalanobis Distance)
For multivariate data $\mathbf{x}$ with mean $\bar{\mathbf{x}}$ and covariance matrix $\mathbf{S}$:

$$D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \bar{\mathbf{x}})^T \mathbf{S}^{-1} (\mathbf{x} - \bar{\mathbf{x}})}$$

**Properties:**
- Accounts for variable scales and correlations
- Reduces to Euclidean distance when variables are uncorrelated and standardized
- Under normality, $D_M^2 \sim \chi^2_p$ (chi-squared with $p$ degrees of freedom)

**Outlier Rule:** Flag observations with $D_M^2 > \chi^2_{p, 0.975}$ (97.5th percentile).

**ML/AI Connection: Anomaly Detection**

Outlier detection techniques form the basis of anomaly detection systems:
- **One-Class SVM**: Learns boundary around normal data in high-dimensional space
- **Isolation Forest**: Exploits that anomalies are easier to isolate in random partitions
- **Autoencoders**: Reconstruct normal data well, fail on anomalies (high reconstruction error)

Applications: Fraud detection, network intrusion, manufacturing defects, medical diagnosis.

---

## Summary and Connections to Machine Learning

Descriptive statistics provide the foundation for understanding data before modeling:

1. **Data Preprocessing Pipeline:**
   - Check central tendency (identify typical values)
   - Measure dispersion (detect high-variance features)
   - Assess skewness/kurtosis (validate distributional assumptions)
   - Detect outliers (clean or flag anomalous data)

2. **Feature Engineering:**
   - Correlation analysis reveals redundant features
   - Covariance matrices inform dimensionality reduction (PCA)
   - Standardization/normalization based on mean, std, median, IQR

3. **Model Selection:**
   - Heavy-tailed distributions → Use robust methods (Huber loss, quantile regression)
   - High-dimensional data → Regularization (L1/L2) or dimensionality reduction
   - Skewed data → Transformations (log, Box-Cox) before linear models

4. **Model Evaluation:**
   - Distribution of residuals (should be Gaussian with mean 0)
   - Correlation between predictions and actuals
   - Outlier analysis in prediction errors

**Key Theorem (Central Limit Theorem Connection):**
As sample size increases, sample mean $\bar{x}$ approaches normality regardless of underlying distribution, with mean $\mu$ and variance $\sigma^2/n$. This justifies using normal-based inference even for non-Gaussian data.

---

## Exercises

### Basic Exercises (★)

**Exercise 4.11.1**
Calculate mean, median, mode, variance, and standard deviation for: $\{3, 7, 7, 2, 8, 5, 10, 7\}$.

**Exercise 4.11.2**
Dataset A: $\{10, 12, 11, 9, 13\}$, Dataset B: $\{5, 15, 10, 2, 18\}$. Both have mean = 11. Calculate standard deviations and explain the difference.

**Exercise 4.11.3**
For data $\{1, 3, 5, 7, 100\}$, compare mean and median. Which better represents central tendency? Why?

### Intermediate Exercises (★★)

**Exercise 4.11.4**
Given paired data $(x, y)$: $(1,2), (3,5), (5,8), (7,10)$, compute:
(a) Covariance, (b) Pearson correlation, (c) Interpret the relationship.

**Exercise 4.11.5**
Dataset: $\{15, 18, 20, 22, 25, 28, 30, 150\}$.
(a) Identify outliers using z-score method ($|z| > 3$).
(b) Identify outliers using IQR method.
(c) Compare results and explain which method is more appropriate.

**Exercise 4.11.6**
A feature has skewness = 2.5 and kurtosis = 8. What does this tell you about the distribution? What preprocessing would you apply before using it in a linear regression model?

### Challenging Exercises (★★★)

**Exercise 4.11.7**
Prove that the sample mean minimizes the sum of squared deviations:
$$\bar{x} = \arg\min_{c} \sum_{i=1}^{n} (x_i - c)^2$$

**Exercise 4.11.8**
For standardized variables ($\bar{x} = 0, s_x = 1$), show that the Pearson correlation equals:
$$r_{xy} = \frac{1}{n-1} \sum_{i=1}^{n} x_i y_i$$

**Exercise 4.11.9**
Given covariance matrix:
$$\mathbf{S} = \begin{bmatrix} 4 & 2 \\ 2 & 9 \end{bmatrix}$$
(a) Compute the correlation matrix.
(b) Calculate Mahalanobis distance for point $(3, 4)$ with mean $(1, 2)$.
(c) Interpret the result assuming bivariate normality.

**Exercise 4.11.10** (ML Application)
You have 1000 features with pairwise correlations. The maximum correlation is 0.95.
(a) Explain why this is problematic for linear regression.
(b) Propose two methods to address this issue.
(c) How would correlation analysis differ for tree-based models (e.g., Random Forest)?

---

## Related Topics

- **Chapter 4.9**: Expectation and Moments (theoretical foundations)
- **Chapter 4.12**: Sampling Distributions (connecting samples to populations)
- **Chapter 5.1**: Hypothesis Testing (using sample statistics for inference)
- **Chapter 6.3**: Linear Regression (covariance and correlation in modeling)
- **Chapter 8.5**: Principal Component Analysis (eigendecomposition of covariance matrix)
- **Chapter 9.2**: Batch Normalization (running mean/variance in deep learning)

---

## Further Reading

1. *The Elements of Statistical Learning* by Hastie, Tibshirani, and Friedman (Chapter 2: Overview of Supervised Learning)
2. *Pattern Recognition and Machine Learning* by Bishop (Chapter 1: Introduction, Section 1.2: Probability Theory)
3. *All of Statistics* by Wasserman (Chapter 1: Probability and Chapter 3: Random Variables)

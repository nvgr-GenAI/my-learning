# 4.12 Hypothesis Testing

## Prerequisites

- **Chapter 4.9**: Probability Distributions (Normal, t, chi-squared, F distributions)
- **Chapter 4.10**: Expectation and Moments (variance, standard error)
- **Chapter 4.11**: Maximum Likelihood Estimation (likelihood functions, parameter estimation)

## Introduction

Hypothesis testing provides a formal framework for making decisions about populations based on sample data. Rather than simply estimating parameters, we test specific claims about those parameters using probabilistic reasoning.

The fundamental question: Given observed data, should we reject a claim about the population, or do the data provide insufficient evidence against it?

```
Scientific Method → Statistical Hypothesis Testing
┌─────────────────────────────────────────────┐
│ 1. Formulate hypothesis                     │
│ 2. Design experiment                        │
│ 3. Collect data                             │
│ 4. Analyze with statistics  ← We are here   │
│ 5. Draw conclusions                         │
└─────────────────────────────────────────────┘
```

---

## 4.12.1 The Hypothesis Testing Framework

**Definition 4.12.1** (Statistical Hypothesis)
A *statistical hypothesis* is a statement about a population parameter. Hypothesis testing evaluates whether sample data provide sufficient evidence to reject this statement.

**Definition 4.12.2** (Null and Alternative Hypotheses)
- **Null hypothesis** ($H_0$): The default claim to be tested, typically representing "no effect" or "no difference"
- **Alternative hypothesis** ($H_1$ or $H_a$): The claim we seek evidence for, contradicting $H_0$

**Example 4.12.1** (Drug Efficacy)
Testing whether a new drug reduces blood pressure:
- $H_0$: $\mu_{\text{drug}} = \mu_{\text{placebo}}$ (drug has no effect)
- $H_1$: $\mu_{\text{drug}} < \mu_{\text{placebo}}$ (drug reduces blood pressure)

### The Testing Procedure

```
Hypothesis Testing Pipeline
────────────────────────────────────────────────────

Data → Test Statistic → Sampling Distribution → p-value → Decision
 X₁,...,Xₙ      T(X)         Under H₀           P(T ≥ t₀)   Reject/Fail to Reject

Example: Testing μ = μ₀
├─ Sample mean: x̄ = 52.3
├─ Test statistic: z = (x̄ - μ₀)/(σ/√n) = 2.41
├─ Distribution: z ~ N(0,1) under H₀
├─ p-value: P(Z ≥ 2.41) = 0.008
└─ Decision: Reject H₀ at α = 0.05
```

**Definition 4.12.3** (Test Statistic)
A *test statistic* $T(X_1, \ldots, X_n)$ is a function of the sample data whose distribution under $H_0$ is known, used to quantify evidence against $H_0$.

**Definition 4.12.4** (p-value)
The *p-value* is the probability, under $H_0$, of observing a test statistic at least as extreme as the one computed from the data:
$$p\text{-value} = P_{H_0}(T \geq t_{\text{obs}})$$
where $t_{\text{obs}}$ is the observed test statistic value.

**Example 4.12.9** (Computing a p-value)
A factory claims mean widget weight is $\mu_0 = 50$g. A sample of $n = 36$ widgets gives $\bar{x} = 51.2$g with known $\sigma = 3$g. Test $H_0: \mu = 50$ vs $H_1: \mu > 50$ (upper-tailed).
$$z = \frac{51.2 - 50}{3/\sqrt{36}} = \frac{1.2}{0.5} = 2.40$$
$$p\text{-value} = P(Z \geq 2.40) = 1 - \Phi(2.40) = 1 - 0.9918 = 0.0082$$
Since $p = 0.0082 < 0.05$, we reject $H_0$. There is strong evidence the mean weight exceeds 50g.

**Definition 4.12.5** (Significance Level)
The *significance level* $\alpha$ is the probability threshold for rejecting $H_0$. Common values: $\alpha = 0.05, 0.01, 0.001$.

**Decision Rule**: Reject $H_0$ if $p\text{-value} < \alpha$.

### One-Sided vs Two-Sided Tests

**Table 4.12.1**: Types of Alternative Hypotheses

| Type | $H_1$ | p-value Calculation | Use Case |
|------|-------|---------------------|----------|
| **Two-sided** | $\theta \neq \theta_0$ | $P(\|T\| \geq \|t_{\text{obs}}\|)$ | Detect any difference |
| **Upper-tailed** | $\theta > \theta_0$ | $P(T \geq t_{\text{obs}})$ | Detect increase only |
| **Lower-tailed** | $\theta < \theta_0$ | $P(T \leq t_{\text{obs}})$ | Detect decrease only |

> **ML/AI Connection**: **Model Comparison A/B Tests**
> When comparing two ML models, we typically use two-sided tests because we want to detect if model B is either better or worse than model A. One-sided tests are appropriate when we only deploy if B is definitively better (e.g., new recommendation algorithm must increase engagement).

---

## 4.12.2 Type I and Type II Errors

Hypothesis testing involves two types of errors:

**Table 4.12.2**: Error Types in Hypothesis Testing

|  | $H_0$ True | $H_0$ False |
|---|------------|-------------|
| **Reject $H_0$** | Type I Error (α) | Correct ✓ |
| **Fail to Reject $H_0$** | Correct ✓ | Type II Error (β) |

**Definition 4.12.6** (Type I Error)
A *Type I error* (false positive) occurs when we reject $H_0$ when it is actually true.
Probability: $\alpha = P(\text{Reject } H_0 \mid H_0 \text{ true})$

**Definition 4.12.7** (Type II Error)
A *Type II error* (false negative) occurs when we fail to reject $H_0$ when it is actually false.
Probability: $\beta = P(\text{Fail to reject } H_0 \mid H_1 \text{ true})$

**Example 4.12.10** (Type I and Type II Error Probabilities)
A medical screening test rejects $H_0$: "patient is healthy" if blood marker $> 15$ units. Under $H_0$, the marker $\sim N(10, 4)$ (mean 10, $\sigma = 2$). Under $H_1$ (disease present), the marker $\sim N(18, 4)$.

**Type I error** ($\alpha$): Healthy patient flagged as diseased.
$$\alpha = P(X > 15 \mid \mu = 10) = P\!\left(Z > \frac{15 - 10}{2}\right) = P(Z > 2.5) = 0.0062$$

**Type II error** ($\beta$): Diseased patient missed.
$$\beta = P(X \leq 15 \mid \mu = 18) = P\!\left(Z \leq \frac{15 - 18}{2}\right) = P(Z \leq -1.5) = 0.0668$$

So $\alpha = 0.62\%$ and $\beta = 6.68\%$. The test rarely misdiagnoses healthy patients but misses about 1 in 15 diseased patients.

**Definition 4.12.8** (Statistical Power)
The *power* of a test is the probability of correctly rejecting a false null hypothesis:
$$\text{Power} = 1 - \beta = P(\text{Reject } H_0 \mid H_1 \text{ true})$$

### Visualizing Errors

```
Distribution Overlap: Type I and Type II Errors
────────────────────────────────────────────────────────────

      Distribution under H₀        Distribution under H₁
           (μ = μ₀)                     (μ = μ₁ > μ₀)

              │                              │
        ░░░░░░│░░                          ░░│░░░░░░
      ░░░░░░░░│░░░░                      ░░░│░░░░░░░░
    ░░░░░░░░░░│░░░░░░                  ░░░░░│░░░░░░░░░░
   ░░░░░░░░░░░│░░░░░░░░              ░░░░░░░│░░░░░░░░░░░
  ░░░░░░░░░░░░│░░░░░░░░░░          ░░░░░░░░░│░░░░░░░░░░░░
 ░░░░░░░░░░░░░│░░░░░░░░░░░░      ░░░░░░░░░░░│░░░░░░░░░░░░░
───────────────┼────────────┬────────────────┼──────────────→
              μ₀          Critical      μ₁ (true mean)
                          Value (c)

    ▓▓▓▓▓ = Type I Error (α)    = P(Reject H₀ | H₀ true)
    ░░░░░ = Type II Error (β)   = P(Fail to reject H₀ | H₁ true)
    ▒▒▒▒▒ = Power (1 - β)       = P(Reject H₀ | H₁ true)

Reject H₀ if test statistic > critical value c
```

**Trade-off**: Decreasing $\alpha$ (more conservative) increases $\beta$ (less power) for fixed sample size.

**Theorem 4.12.1** (Factors Affecting Power)
Statistical power increases with:
1. Larger sample size $n$
2. Larger effect size $|\theta_1 - \theta_0|$
3. Larger significance level $\alpha$
4. Smaller population variance $\sigma^2$

**Example 4.12.11** (Computing Power)
Test $H_0: \mu = 100$ vs $H_1: \mu > 100$ at $\alpha = 0.05$ with $\sigma = 10$ and $n = 25$. What is the power if the true mean is $\mu_1 = 104$?

Reject $H_0$ when $Z = \frac{\bar{X} - 100}{10/\sqrt{25}} > z_{0.05} = 1.645$, i.e., when $\bar{X} > 100 + 1.645 \cdot 2 = 103.29$.

Under $H_1$ ($\mu_1 = 104$):
$$\text{Power} = P(\bar{X} > 103.29 \mid \mu = 104) = P\!\left(Z > \frac{103.29 - 104}{2}\right) = P(Z > -0.355) = 0.6387$$

Power is only 63.9%. To increase power to 80%, we need $n = \left(\frac{(1.645 + 0.842) \cdot 10}{4}\right)^2 \approx 39$ observations.

> **ML/AI Connection**: **Statistical Power in ML Experiments**
> When comparing model performance, insufficient power leads to Type II errors—failing to detect genuine improvements. Calculate required sample size before A/B testing:
> $$n \approx \frac{2(z_{\alpha/2} + z_{\beta})^2 \sigma^2}{\delta^2}$$
> where $\delta$ is the minimum detectable effect size. For detecting a 1% improvement in accuracy with 80% power at $\alpha=0.05$, you might need 10,000+ samples.

---

## 4.12.3 Common Hypothesis Tests

### z-Test (Known Variance)

**Theorem 4.12.2** (One-Sample z-Test)
For $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$ with known $\sigma^2$, testing $H_0: \mu = \mu_0$:
$$Z = \frac{\bar{X} - \mu_0}{\sigma/\sqrt{n}} \sim N(0, 1) \text{ under } H_0$$

**Example 4.12.2** (Quality Control)
Battery lifetimes are known to have $\sigma = 5$ hours. A sample of $n=25$ has $\bar{x} = 102$ hours. Test $H_0: \mu = 100$ vs $H_1: \mu \neq 100$ at $\alpha = 0.05$:
$$z = \frac{102 - 100}{5/\sqrt{25}} = \frac{2}{1} = 2.0$$
Critical values: $z_{\alpha/2} = \pm 1.96$. Since $|z| > 1.96$, reject $H_0$.

### t-Test (Unknown Variance)

**Theorem 4.12.3** (One-Sample t-Test)
For $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$ with unknown $\sigma^2$, testing $H_0: \mu = \mu_0$:
$$T = \frac{\bar{X} - \mu_0}{S/\sqrt{n}} \sim t_{n-1} \text{ under } H_0$$
where $S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2$.

**Example 4.12.12** (One-Sample t-Test: Quality Control)
A cereal box claims 500g of content. A sample of $n = 12$ boxes gives $\bar{x} = 492$g and $s = 15$g. Test $H_0: \mu = 500$ vs $H_1: \mu \neq 500$ at $\alpha = 0.05$.
$$t = \frac{492 - 500}{15/\sqrt{12}} = \frac{-8}{4.33} = -1.848$$
Degrees of freedom: $\nu = 12 - 1 = 11$. Critical value: $t_{0.025, 11} = 2.201$.
Since $|t| = 1.848 < 2.201$, we fail to reject $H_0$. The p-value $\approx 0.091$ (two-tailed).
Conclusion: Insufficient evidence that the mean fill weight differs from 500g.

**Theorem 4.12.4** (Two-Sample t-Test)
For independent samples $X_1, \ldots, X_{n_1} \sim N(\mu_1, \sigma^2)$ and $Y_1, \ldots, Y_{n_2} \sim N(\mu_2, \sigma^2)$ with equal variances, testing $H_0: \mu_1 = \mu_2$:
$$T = \frac{\bar{X} - \bar{Y}}{S_p\sqrt{1/n_1 + 1/n_2}} \sim t_{n_1+n_2-2}$$
where $S_p^2 = \frac{(n_1-1)S_1^2 + (n_2-1)S_2^2}{n_1 + n_2 - 2}$ is the pooled variance.

**Example 4.12.13** (Two-Sample t-Test: Drug Trial)
Two groups receive different treatments. Group 1 ($n_1 = 10$): $\bar{x}_1 = 78.5$, $s_1 = 6.2$. Group 2 ($n_2 = 12$): $\bar{x}_2 = 84.1$, $s_2 = 5.8$. Test $H_0: \mu_1 = \mu_2$ vs $H_1: \mu_1 \neq \mu_2$ at $\alpha = 0.05$.

Pooled variance: $S_p^2 = \frac{9(6.2)^2 + 11(5.8)^2}{10 + 12 - 2} = \frac{345.96 + 370.04}{20} = 35.80$, so $S_p = 5.983$.

$$t = \frac{78.5 - 84.1}{5.983\sqrt{1/10 + 1/12}} = \frac{-5.6}{5.983 \times 0.4264} = \frac{-5.6}{2.551} = -2.195$$

Degrees of freedom: $\nu = 20$. Critical value: $t_{0.025, 20} = 2.086$. Since $|t| = 2.195 > 2.086$, reject $H_0$.
Conclusion: The treatments produce significantly different mean outcomes ($p \approx 0.040$).

### Chi-Squared Test

**Theorem 4.12.5** (Chi-Squared Goodness-of-Fit Test)
For observed counts $O_1, \ldots, O_k$ and expected counts $E_1, \ldots, E_k$ under $H_0$:
$$\chi^2 = \sum_{i=1}^k \frac{(O_i - E_i)^2}{E_i} \sim \chi^2_{k-1-p}$$
where $p$ is the number of estimated parameters.

**Example 4.12.3** (Dice Fairness)
Rolling a die 60 times: observed counts $(8, 12, 11, 9, 10, 10)$. Expected under fair die: all 10.
$$\chi^2 = \frac{(8-10)^2}{10} + \frac{(12-10)^2}{10} + \cdots = 1.4$$
Critical value: $\chi^2_{0.05, 5} = 11.07$. Fail to reject $H_0$ (die appears fair).

**Example 4.12.14** (Chi-Squared Test of Independence: 2x2 Table)
A study tests whether a vaccine is associated with infection outcome.

|                    | Infected | Not Infected | **Total** |
| ------------------ | -------- | ------------ | --------- |
| **Vaccinated**     | 15       | 135          | 150       |
| **Unvaccinated**   | 30       | 120          | 150       |
| **Total**          | 45       | 255          | 300       |

$H_0$: Vaccination status and infection are independent. Expected counts under $H_0$: $E_{ij} = \frac{(\text{row total})(\text{col total})}{n}$.

$E_{11} = \frac{150 \times 45}{300} = 22.5$, $E_{12} = \frac{150 \times 255}{300} = 127.5$ (same for row 2).

$$\chi^2 = \frac{(15-22.5)^2}{22.5} + \frac{(135-127.5)^2}{127.5} + \frac{(30-22.5)^2}{22.5} + \frac{(120-127.5)^2}{127.5}$$
$$= 2.50 + 0.441 + 2.50 + 0.441 = 5.882$$

Degrees of freedom: $(2-1)(2-1) = 1$. Critical value: $\chi^2_{0.05, 1} = 3.841$. Since $5.882 > 3.841$, reject $H_0$.
Conclusion: There is a significant association between vaccination and infection ($p \approx 0.015$).

### F-Test

**Theorem 4.12.6** (F-Test for Variance Equality)
For independent samples from $N(\mu_1, \sigma_1^2)$ and $N(\mu_2, \sigma_2^2)$, testing $H_0: \sigma_1^2 = \sigma_2^2$:
$$F = \frac{S_1^2}{S_2^2} \sim F_{n_1-1, n_2-1} \text{ under } H_0$$

**Table 4.12.3**: Test Selection Guide

| Scenario | Sample Size | Variance Known? | Test |
|----------|-------------|-----------------|------|
| One mean | Any | Yes | z-test |
| One mean | $n \geq 30$ | No | z-test (CLT) |
| One mean | $n < 30$ | No | t-test |
| Two means (paired) | Any | No | Paired t-test |
| Two means (independent) | Any | No | Two-sample t-test |
| Two variances | Any | N/A | F-test |
| Categorical data | Any | N/A | Chi-squared test |

> **ML/AI Connection**: **Model Performance Testing**
> - t-test: Compare mean accuracy of two models on same dataset (paired)
> - F-test: Test if model variance differs (stability check)
> - Chi-squared: Test if classification errors are uniformly distributed across classes
> - Caution: Violations of normality assumption common with accuracy metrics bounded in [0,1]

---

## 4.12.4 Confidence Intervals

**Definition 4.12.9** (Confidence Interval)
A $(1-\alpha) \times 100\%$ *confidence interval* for parameter $\theta$ is a random interval $[L(X), U(X)]$ such that:
$$P(L(X) \leq \theta \leq U(X)) = 1 - \alpha$$

**Theorem 4.12.7** (CI for Mean with Known Variance)
For $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$ with known $\sigma^2$:
$$\bar{X} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

**Theorem 4.12.8** (CI for Mean with Unknown Variance)
For $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$ with unknown $\sigma^2$:
$$\bar{X} \pm t_{\alpha/2, n-1} \cdot \frac{S}{\sqrt{n}}$$

**Example 4.12.15** (95% Confidence Interval for a Mean)
A sample of $n = 20$ patients has mean systolic blood pressure $\bar{x} = 128$ mmHg and $s = 14$ mmHg. Construct a 95% CI for the population mean.

Since $\sigma$ is unknown and $n < 30$, use the $t$-interval. Critical value: $t_{0.025, 19} = 2.093$.

$$\text{CI} = 128 \pm 2.093 \cdot \frac{14}{\sqrt{20}} = 128 \pm 2.093 \cdot 3.130 = 128 \pm 6.55$$
$$\text{CI} = [121.45, \; 134.55]$$

We are 95% confident the true mean systolic BP lies between 121.5 and 134.6 mmHg. Since $130$ (a clinical threshold) falls inside this interval, we would fail to reject $H_0: \mu = 130$ at $\alpha = 0.05$.

### Interpretation

**Correct**: "If we repeat this procedure many times, 95% of constructed intervals will contain the true parameter."

**Incorrect**: "There is a 95% probability that $\mu$ lies in this specific interval." (Parameter is fixed, interval is random)

### Relationship to Hypothesis Testing

**Theorem 4.12.9** (Duality of CIs and Tests)
A two-sided hypothesis test at significance level $\alpha$ rejects $H_0: \theta = \theta_0$ if and only if $\theta_0$ lies outside the $(1-\alpha)$ confidence interval for $\theta$.

```
Confidence Intervals vs Hypothesis Tests
────────────────────────────────────────

                    95% CI: [48.3, 53.7]
              ├───────────────────────────────┤
──────────────┼───────────────┼───────────────┼────────→
             40              50              60   μ

H₀: μ = 45    → Reject (outside CI)
H₀: μ = 50    → Fail to reject (inside CI)
H₀: μ = 58    → Reject (outside CI)
```

**Example 4.12.4** (Model Performance CI)
A classifier achieves 82% accuracy on $n=1000$ test samples. Construct 95% CI:
$$\hat{p} = 0.82, \quad SE = \sqrt{\frac{0.82 \cdot 0.18}{1000}} = 0.0121$$
$$\text{CI} = 0.82 \pm 1.96 \cdot 0.0121 = [0.796, 0.844]$$

> **ML/AI Connection**: **Reporting Model Performance**
> Always report confidence intervals, not just point estimates. "Model A: 85% ± 2%" is more informative than "Model A: 85%". CIs reveal whether observed differences are statistically meaningful or within noise margins.

---

## 4.12.5 Multiple Testing Problem

When conducting $m$ hypothesis tests simultaneously, the probability of at least one Type I error increases:
$$P(\text{at least one Type I error}) = 1 - (1-\alpha)^m$$

For $m=20$ tests at $\alpha=0.05$: probability $\approx 0.64$ of at least one false positive!

**Definition 4.12.10** (Family-Wise Error Rate)
The *family-wise error rate* (FWER) is the probability of making at least one Type I error among all $m$ tests:
$$\text{FWER} = P(\text{reject at least one true } H_0)$$

### Bonferroni Correction

**Theorem 4.12.10** (Bonferroni Method)
To control FWER at level $\alpha$, test each hypothesis at level $\alpha/m$:
$$\text{FWER} \leq \sum_{i=1}^m P(\text{reject true } H_{0,i}) = m \cdot \frac{\alpha}{m} = \alpha$$

**Example 4.12.17** (Bonferroni Correction in Practice)
A clinical study tests $m = 5$ biomarkers for association with a disease at $\alpha = 0.05$. The raw p-values are: $p_1 = 0.003$, $p_2 = 0.012$, $p_3 = 0.035$, $p_4 = 0.048$, $p_5 = 0.22$.

Without correction, 4 of 5 biomarkers are "significant" ($p < 0.05$).
Bonferroni threshold: $\alpha/m = 0.05/5 = 0.01$. Only $p_1 = 0.003 < 0.01$ survives.

Probability of at least one false positive without correction: $1 - (1 - 0.05)^5 = 0.226$ (22.6%). With Bonferroni, FWER is controlled at 5%.

**Limitation**: Overly conservative; loses power as $m$ increases.

### False Discovery Rate

**Definition 4.12.11** (False Discovery Rate)
The *false discovery rate* (FDR) is the expected proportion of false positives among rejected hypotheses:
$$\text{FDR} = E\left[\frac{\text{# false positives}}{\text{# rejections}}\right]$$

**Theorem 4.12.11** (Benjamini-Hochberg Procedure)
To control FDR at level $\alpha$:
1. Order p-values: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$
2. Find largest $k$ such that $p_{(k)} \leq \frac{k}{m} \alpha$
3. Reject hypotheses $H_{(1)}, \ldots, H_{(k)}$

**Example 4.12.5** (Multiple Feature Testing)
Testing 10 features with p-values: $(0.001, 0.01, 0.02, 0.04, 0.06, 0.10, 0.15, 0.30, 0.50, 0.80)$.

At $\alpha = 0.05$:
- **Bonferroni**: Reject if $p < 0.05/10 = 0.005$ → Reject 1 feature
- **BH**: Largest $k$ where $p_{(k)} \leq k \cdot 0.05/10$:
  - $p_{(4)} = 0.04 \leq 4 \cdot 0.005 = 0.02$? No
  - $p_{(3)} = 0.02 \leq 3 \cdot 0.005 = 0.015$? No
  - $p_{(2)} = 0.01 \leq 2 \cdot 0.005 = 0.01$? Yes! → Reject 2 features

> **ML/AI Connection**: **Hyperparameter Tuning and Multiple Testing**
> When testing 100 hyperparameter configurations, expect ~5 to appear "significantly better" by chance at $\alpha=0.05$. Use Bonferroni or FDR correction when reporting results. Better: use nested cross-validation to get unbiased performance estimates.

---

## 4.12.6 A/B Testing

A/B testing applies hypothesis testing to compare two variants (A: control, B: treatment).

### Design Principles

**Table 4.12.4**: A/B Test Design Checklist

| Component | Description | Best Practice |
|-----------|-------------|---------------|
| **Randomization** | Assign users randomly to A or B | Use hashing for consistency |
| **Sample Size** | Determine before experiment | Power analysis (80-90% power) |
| **Metric** | Primary evaluation criterion | One primary, multiple secondary |
| **Duration** | Test run time | Capture weekly cycles, holidays |
| **Significance** | Statistical threshold | $\alpha = 0.05$ or stricter |

### Sample Size Calculation

**Theorem 4.12.12** (Sample Size for Proportions)
To detect difference $\delta = p_B - p_A$ with power $1-\beta$ at significance $\alpha$ (two-sided):
$$n = \frac{(z_{\alpha/2} + z_{\beta})^2 \cdot 2\bar{p}(1-\bar{p})}{\delta^2}$$
where $\bar{p} = (p_A + p_B)/2$.

**Example 4.12.6** (Button Color Test)
Current click-through rate: $p_A = 0.10$. Minimum detectable lift: $\delta = 0.02$ (to $p_B = 0.12$).
Power: $1-\beta = 0.80$, significance: $\alpha = 0.05$.
$$n = \frac{(1.96 + 0.84)^2 \cdot 2(0.11)(0.89)}{(0.02)^2} = \frac{7.84 \cdot 0.196}{0.0004} \approx 3841$$
Need ~3,841 users per variant (7,682 total).

**Example 4.12.16** (A/B Test: Is the Conversion Rate Difference Significant?)
An e-commerce site runs an A/B test. Group A (control): 500 conversions out of 5000 visitors ($\hat{p}_A = 0.100$). Group B (new design): 560 conversions out of 5000 visitors ($\hat{p}_B = 0.112$). Test $H_0: p_A = p_B$ vs $H_1: p_A \neq p_B$ at $\alpha = 0.05$.

Pooled proportion: $\hat{p} = \frac{500 + 560}{5000 + 5000} = 0.106$.

$$z = \frac{0.112 - 0.100}{\sqrt{0.106 \cdot 0.894 \left(\frac{1}{5000} + \frac{1}{5000}\right)}} = \frac{0.012}{\sqrt{0.106 \cdot 0.894 \cdot 0.0004}} = \frac{0.012}{0.00616} = 1.948$$

Critical value: $z_{0.025} = 1.96$. Since $|z| = 1.948 < 1.96$, we barely fail to reject $H_0$ ($p \approx 0.051$).
Conclusion: The 1.2 percentage point lift is not statistically significant at the 5% level -- more data is needed.

### Statistical Significance vs Practical Significance

A result can be statistically significant but not practically meaningful:
- Statistical: $p < 0.05$
- Practical: Effect size large enough to matter (e.g., 5% revenue increase vs 0.1%)

**Definition 4.12.12** (Effect Size Measures)
- **Cohen's d**: $d = \frac{\mu_1 - \mu_2}{\sigma}$ (standardized mean difference)
- **Relative lift**: $\frac{p_B - p_A}{p_A} \times 100\%$

**Example 4.12.18** (Effect Size: Cohen's d)
Two teaching methods are compared. Method A: $\bar{x}_A = 72$, $s_A = 10$, $n_A = 50$. Method B: $\bar{x}_B = 76$, $s_B = 12$, $n_B = 50$.

Pooled standard deviation: $s_p = \sqrt{\frac{49(10)^2 + 49(12)^2}{98}} = \sqrt{\frac{4900 + 7056}{98}} = \sqrt{122} = 11.05$.

$$d = \frac{76 - 72}{11.05} = \frac{4}{11.05} = 0.362$$

By Cohen's conventions ($d = 0.2$: small, $0.5$: medium, $0.8$: large), this is a small-to-medium effect. Even if statistically significant with a large enough sample, the practical impact is modest -- a 4-point gain on a 100-point scale.

> **ML/AI Connection**: **Model A/B Testing in Production**
> Netflix case study: Testing recommendation algorithms
> - Randomization: User ID hashing ensures consistency
> - Metrics: Streaming hours (primary), engagement rate (secondary)
> - Duration: 2 weeks minimum to capture viewing patterns
> - Multiple testing: Bonferroni across 5 simultaneous experiments
> - Result: 10% increase in streaming hours with $p = 0.003$ → Deploy model B

---

## 4.12.7 Bayesian vs Frequentist Testing

### Philosophical Differences

**Table 4.12.5**: Paradigm Comparison

| Aspect | Frequentist | Bayesian |
|--------|-------------|----------|
| **Parameter** | Fixed unknown constant | Random variable with prior |
| **Probability** | Long-run frequency | Degree of belief |
| **Inference** | p-value, confidence interval | Posterior distribution, credible interval |
| **Prior Info** | Not incorporated | Explicit prior distribution |
| **Interpretation** | Probability of data given H₀ | Probability of hypothesis given data |

### Likelihood Ratio

**Definition 4.12.13** (Likelihood Ratio)
The likelihood ratio compares evidence for two hypotheses:
$$\text{LR} = \frac{P(D \mid H_1)}{P(D \mid H_0)}$$

### Bayes Factor

**Definition 4.12.14** (Bayes Factor)
The *Bayes factor* $BF_{10}$ quantifies evidence for $H_1$ over $H_0$:
$$BF_{10} = \frac{P(D \mid H_1)}{P(D \mid H_0)} = \frac{P(H_1 \mid D) / P(H_1)}{P(H_0 \mid D) / P(H_0)}$$

**Interpretation** (Jeffreys' Scale):
- $BF_{10} > 10$: Strong evidence for $H_1$
- $BF_{10} = 1$: No evidence either way
- $BF_{10} < 0.1$: Strong evidence for $H_0$

**Example 4.12.7** (Bayesian A/B Test)
Prior: $p_A, p_B \sim \text{Beta}(1, 1)$ (uniform).
Data: $n_A = 100, x_A = 10$; $n_B = 100, x_B = 15$.
Posterior: $p_A \sim \text{Beta}(11, 91)$, $p_B \sim \text{Beta}(16, 86)$.
$P(p_B > p_A) = 0.89$ (89% probability B is better).

Contrast frequentist: $p\text{-value} = 0.18$ (fail to reject $H_0: p_A = p_B$).

> **ML/AI Connection**: **Bayesian Methods in ML**
> Bayesian A/B testing naturally handles:
> - Early stopping: Can monitor and stop when $P(p_B > p_A) > 0.95$
> - Prior knowledge: Incorporate historical model performance
> - Uncertainty: Full posterior distribution, not just point estimate
> - Common in reinforcement learning (Thompson sampling) and AutoML

---

## 4.12.8 Non-Parametric Tests

When normality assumptions fail, use distribution-free tests.

### Mann-Whitney U Test

**Theorem 4.12.13** (Mann-Whitney Test)
For independent samples from continuous distributions, tests $H_0$: distributions are identical.
1. Rank all observations from both groups
2. Compute $U = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1$ where $R_1$ is sum of ranks in group 1
3. For large $n$: $U \approx N\left(\frac{n_1 n_2}{2}, \frac{n_1 n_2(n_1+n_2+1)}{12}\right)$ under $H_0$

**Use case**: Compare medians when data is ordinal or skewed.

### Wilcoxon Signed-Rank Test

**Theorem 4.12.14** (Wilcoxon Test)
For paired samples, tests $H_0$: median difference is zero.
1. Compute differences $D_i = X_i - Y_i$
2. Rank absolute differences $|D_i|$
3. Sum ranks of positive differences
4. Compare to theoretical distribution under $H_0$

**Use case**: Paired comparisons (before/after) with non-normal data.

### Permutation Tests

**Theorem 4.12.15** (Permutation Test)
Exact test without distributional assumptions:
1. Compute test statistic $T_{\text{obs}}$ on original data
2. Randomly permute group labels $B$ times (e.g., $B=10000$)
3. Compute $T_b$ for each permutation
4. p-value = proportion of $T_b$ at least as extreme as $T_{\text{obs}}$

**Advantages**:
- Exact finite-sample validity
- Works for any test statistic
- No distributional assumptions

**Example 4.12.8** (Permutation Test for Feature Importance)
To test if feature $X_j$ is important:
1. Compute model accuracy $A_{\text{obs}}$ with all features
2. Permute feature $X_j$ (breaks relationship with $Y$)
3. Recompute accuracy $A_{\text{perm}}$
4. Repeat $B$ times
5. p-value = proportion where $A_{\text{perm}} \geq A_{\text{obs}}$

> **ML/AI Connection**: **Permutation Tests for Feature Importance**
> Random forests and gradient boosting use permutation importance:
> - Permute feature values, measure performance drop
> - Significant drop → feature is important
> - No parametric assumptions about model or data
> - Works for any black-box model

---

## Common Pitfalls and Best Practices

### p-value Misinterpretations

**What p-value IS**: $P(\text{data or more extreme} \mid H_0)$
**What p-value IS NOT**:
- Probability that $H_0$ is true
- Probability that results are due to chance
- Probability of replication failure

### Statistical Significance vs Importance

- $p < 0.05$ with $n=1,000,000$: Tiny effects become "significant"
- Always report effect sizes and confidence intervals

### Publication Bias and p-Hacking

**p-hacking**: Manipulating analysis until $p < 0.05$
- Try different subgroups
- Try different statistical tests
- Stop collecting data when $p < 0.05$

**Solution**: Pre-registration, multiple testing correction, replication

> **ML/AI Connection**: **p-value Pitfalls in ML Research**
> Common issues:
> - Testing multiple models/hyperparameters without correction
> - Cherry-picking best results across random seeds
> - Reporting accuracy without confidence intervals
> - "Significant" improvements of 0.1% on huge datasets
>
> Best practices:
> - Pre-specify evaluation protocol
> - Report all experiments, not just "significant" ones
> - Use nested cross-validation for hyperparameter selection
> - Consider Bayesian methods for model comparison

---

## Key Takeaways

1. **Framework**: Hypothesis testing formalizes decision-making under uncertainty using p-values and significance levels
2. **Errors**: Balance Type I (false positive) and Type II (false negative) errors; power increases with sample size
3. **Test Selection**: Choose z-test, t-test, chi-squared, or F-test based on data type and sample size
4. **Confidence Intervals**: Provide richer information than p-values; dual relationship with hypothesis tests
5. **Multiple Testing**: Correct for inflated Type I error using Bonferroni or FDR methods
6. **A/B Testing**: Requires careful design, sample size calculation, and distinction between statistical and practical significance
7. **Non-parametric Tests**: Use when parametric assumptions fail; permutation tests work for any statistic

---

## Exercises

### Basic (★)

**Exercise 4.12.1**
A sample of $n=25$ observations has $\bar{x} = 52$ and $s = 8$. Test $H_0: \mu = 50$ vs $H_1: \mu \neq 50$ at $\alpha = 0.05$.

**Exercise 4.12.2**
If you conduct 50 independent hypothesis tests at $\alpha = 0.05$, what is the probability of at least one Type I error? What Bonferroni-corrected $\alpha$ should you use?

**Exercise 4.12.3**
Construct a 95% confidence interval for the mean of a sample with $n=16$, $\bar{x} = 100$, $s = 12$.

### Intermediate (★★)

**Exercise 4.12.4**
An A/B test compares two webpage designs. Design A: 120 clicks out of 1000 visits. Design B: 140 clicks out of 1000 visits. Perform a two-proportion z-test at $\alpha = 0.05$. What is your conclusion?

**Exercise 4.12.5**
You test 20 features for association with the target variable, obtaining p-values ranging from 0.001 to 0.8. Apply the Benjamini-Hochberg procedure at FDR level $\alpha = 0.10$. Which features are significant?

**Exercise 4.12.6**
A classifier achieves 85% accuracy on a test set of $n=200$. Construct a 99% confidence interval for the true accuracy. Would you reject $H_0: p = 0.80$ at $\alpha = 0.01$?

### Challenging (★★★)

**Exercise 4.12.7**
Derive the sample size formula (Theorem 4.12.12) for detecting a difference in proportions. Show all steps involving the normal approximation.

**Exercise 4.12.8**
In a permutation test comparing two groups ($n_1=n_2=10$), how many possible permutations exist? If you use $B=10000$ permutations, what is the minimum achievable p-value? What does this imply for very small p-values?

**Exercise 4.12.9**
Consider a Bayesian A/B test with Beta priors. Show that the posterior probability $P(p_B > p_A \mid \text{data})$ can be computed as:
$$P(p_B > p_A) = \sum_{i=0}^{\alpha_B-1} \frac{B(\alpha_A+i, \beta_B+\beta_A)}{(\beta_B+i) \cdot B(1+i, \beta_B) \cdot B(\alpha_A, \beta_A)}$$
where $\alpha_A, \beta_A, \alpha_B, \beta_B$ are posterior Beta parameters.

**Exercise 4.12.10** (ML Application)
You train a model and test on 1000 samples, achieving accuracy 0.87. Your colleague trains a different model on the same data, achieving 0.89. Design a hypothesis test to determine if the second model is significantly better. What assumptions must you check? What test would you use?

---

## Related Topics

- **Chapter 4.13**: Bayesian Inference (posterior distributions, credible intervals)
- **Chapter 4.14**: Bootstrap and Resampling Methods (computational alternatives to parametric tests)
- **Chapter 5.7**: Cross-Validation (model evaluation and selection)
- **Chapter 8.4**: Experimental Design (randomized controlled trials, causal inference)
- **Chapter 9.2**: Statistical Learning Theory (generalization bounds, PAC learning)

---

## Further Reading

1. **Lehmann & Romano** (2005): *Testing Statistical Hypotheses* - Comprehensive theoretical treatment
2. **Wasserman** (2004): *All of Statistics* - Clear modern introduction
3. **Efron & Hastie** (2016): *Computer Age Statistical Inference* - Modern perspective with ML connections
4. **Kohavi et al.** (2020): *Trustworthy Online Controlled Experiments* - A/B testing at scale
5. **Deng et al.** (2013): "Improving the Sensitivity of Online Controlled Experiments" - Netflix/Microsoft practices

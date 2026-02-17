# Part 4: Probability and Statistics

Machine learning is applied probability theory. Every prediction carries uncertainty. Every model is a probabilistic statement about data. Bayesian inference, maximum likelihood estimation, generative models, and diffusion models are all built directly on probability theory. If linear algebra is the language of data representation, probability is the language of learning from data.

---

## Chapters

| Chapter | Topic | Key Results |
|---------|-------|-------------|
| 4.1 | [Probability Foundations](probability-foundations.md) | Kolmogorov axioms, $\sigma$-algebras, probability spaces |
| 4.2 | [Conditional Probability](conditional-probability.md) | Bayes' theorem, chain rule, independence |
| 4.3 | [Random Variables](random-variables.md) | PMF, PDF, CDF, transformations, reparameterization |
| 4.4 | [Distributions](distributions.md) | Bernoulli, Gaussian, Poisson, Exponential, Beta, Dirichlet |
| 4.5 | [Expectation & Moments](expectation-and-moments.md) | $\mathbb{E}[X]$, variance, covariance, MGFs |
| 4.6 | [Joint Distributions](joint-distributions.md) | Marginal, conditional, multivariate Gaussian |
| 4.7 | [Limit Theorems](limit-theorems.md) | LLN, CLT, concentration inequalities |
| 4.8 | [Bayesian Inference](bayesian-inference.md) | Prior, posterior, MAP, conjugate priors, MCMC |
| 4.9 | [Maximum Likelihood](maximum-likelihood.md) | MLE, Fisher information, Cramér-Rao bound |
| 4.10 | [Stochastic Processes](stochastic-processes.md) | Markov chains, random walks, Brownian motion |
| 4.11 | [Descriptive Statistics](descriptive-statistics.md) | Mean, median, variance, correlation, outlier detection |
| 4.12 | [Hypothesis Testing](hypothesis-testing.md) | p-values, confidence intervals, A/B testing, multiple testing |

## Prerequisites

- [Part 1: Foundations](../foundations/index.md) — sets and logic
- [Part 2: Linear Algebra](../linear-algebra/index.md) — for multivariate distributions
- [Part 3: Calculus](../calculus/index.md) — integration and differentiation

## Why Every ML Engineer Needs Probability

| ML Concept | Probabilistic Foundation |
|-----------|------------------------|
| Training a classifier | Maximizing $P(\text{class} \mid \text{features})$ |
| Loss function (cross-entropy) | $-\sum p \log q$ = KL divergence + entropy |
| Dropout | Approximate Bayesian inference |
| Batch normalization | Standardizing to match $\mathcal{N}(0,1)$ |
| GANs | Minimizing divergence between distributions |
| Diffusion models | Reverse stochastic process |
| Uncertainty estimation | Posterior predictive distribution |

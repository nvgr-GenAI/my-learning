# Joint Distributions

When we model real-world systems, variables rarely exist in isolation. A patient's blood pressure and cholesterol are correlated. An image's pixels are spatially dependent. A language model's token probabilities depend on the entire preceding context. Joint distributions formalize how multiple random variables behave *together* — their combined probabilities, their individual behaviors recovered via marginalization, and the fundamental question of whether knowing one variable tells you anything about another. The multivariate Gaussian, the most important joint distribution in ML, provides closed-form answers to all these questions through elegant linear algebra.

---

## Prerequisites

- [Random Variables](random-variables.md) — PMF, PDF, CDF, transformations
- [Distributions](distributions.md) — Gaussian, Bernoulli, and other univariate families
- [Linear Algebra - Matrices](../linear-algebra/matrices.md) — matrix inverse, determinant, positive definiteness

---

## 1. Joint Probability Mass Function

**Definition 4.6.1 (Joint PMF).** For discrete random variables $X$ and $Y$, the *joint probability mass function* is:

$$p_{X,Y}(x, y) = P(X = x, Y = y)$$

**Properties:**

1. $p_{X,Y}(x, y) \geq 0$ for all $x, y$
2. $\sum_{x}\sum_{y} p_{X,Y}(x, y) = 1$
3. $P((X,Y) \in A) = \sum_{(x,y) \in A} p_{X,Y}(x, y)$ for any event $A$

```
JOINT PMF AS A TABLE

              Y = 0    Y = 1    Y = 2    Row sum
           ┌─────────┬─────────┬─────────┬─────────┐
  X = 0    │  0.10   │  0.15   │  0.05   │  0.30   │ ← P(X=0)
           ├─────────┼─────────┼─────────┼─────────┤
  X = 1    │  0.20   │  0.25   │  0.10   │  0.55   │ ← P(X=1)
           ├─────────┼─────────┼─────────┼─────────┤
  X = 2    │  0.05   │  0.05   │  0.05   │  0.15   │ ← P(X=2)
           └─────────┴─────────┴─────────┴─────────┘
  Col sum:   0.35      0.45      0.20      1.00
             ↑ P(Y=0)  ↑ P(Y=1)  ↑ P(Y=2)

  Each cell:  joint probability p(x, y)
  Row sums:   marginal of X
  Col sums:   marginal of Y
```

**Example 4.6.1:** Consider two discrete r.v.s $X \in \{1, 2, 3\}$ and $Y \in \{1, 2, 3\}$ with the following joint PMF table:

|       | $Y=1$ | $Y=2$ | $Y=3$ |
|-------|-------|-------|-------|
| $X=1$ | 0.10  | 0.05  | 0.05  |
| $X=2$ | 0.10  | 0.20  | 0.10  |
| $X=3$ | 0.05  | 0.15  | 0.20  |

**Verify this is a valid PMF:**

$$\sum_{x}\sum_{y} p(x,y) = 0.10 + 0.05 + 0.05 + 0.10 + 0.20 + 0.10 + 0.05 + 0.15 + 0.20 = 1.00 \; \checkmark$$

All entries are $\geq 0$ $\checkmark$. So this is a valid joint PMF.

**Compute a probability over a region:** $P(X \geq 2, Y \geq 2) = p(2,2) + p(2,3) + p(3,2) + p(3,3) = 0.20 + 0.10 + 0.15 + 0.20 = 0.65$.

*ML connection:* In discrete language models, the joint PMF $P(w_1, w_2, \ldots, w_T)$ over a sequence of tokens is the fundamental object. The chain rule decomposes it as $\prod_t P(w_t \mid w_{<t})$, which is exactly what autoregressive models like GPT learn to approximate.

---

## 2. Joint Probability Density Function

**Definition 4.6.2 (Joint PDF).** For continuous random variables $X$ and $Y$, the *joint probability density function* $f_{X,Y}(x, y)$ satisfies:

$$P((X, Y) \in A) = \iint_A f_{X,Y}(x, y) \, dx \, dy$$

**Properties:**

1. $f_{X,Y}(x, y) \geq 0$ for all $x, y$
2. $\int_{-\infty}^{\infty}\int_{-\infty}^{\infty} f_{X,Y}(x, y) \, dx \, dy = 1$

Note: $f_{X,Y}(x,y)$ is a *density*, not a probability. It can exceed 1. Only integrals over regions give probabilities.

```
JOINT PDF AS A SURFACE AND CONTOUR PLOT

  Side view (3D surface):           Top view (contour plot):

  f(x,y)                              y
   ▲                                   ▲
   │    ╱╲                             │   ╭───╮
   │   ╱  ╲                            │  ╱ ╭─╮ ╲
   │  ╱    ╲                           │ │ │ ● │ │   ● = mode
   │ ╱      ╲╌╌╌╌╌╌                   │  ╲ ╰─╯ ╱
   │╱              ╲                   │   ╰───╯
   └──────────────────▶ x             └──────────────▶ x
                                       contour lines =
   Volume under surface = 1           equal-density curves
```

**Example 4.6.2:** Let $f(x, y) = \frac{3}{2}(x^2 + y^2)$ for $0 \leq x \leq 1$, $0 \leq y \leq 1$, and $0$ otherwise.

**Verify it integrates to 1:**

$$\int_0^1 \int_0^1 \frac{3}{2}(x^2 + y^2) \, dx \, dy = \frac{3}{2}\int_0^1 \left[\frac{x^3}{3} + xy^2\right]_0^1 dy = \frac{3}{2}\int_0^1 \left(\frac{1}{3} + y^2\right) dy$$

$$= \frac{3}{2}\left[\frac{y}{3} + \frac{y^3}{3}\right]_0^1 = \frac{3}{2} \cdot \frac{2}{3} = 1 \; \checkmark$$

**Compute a probability:** $P(X \leq 1/2, \, Y \leq 1/2) = \frac{3}{2}\int_0^{1/2}\int_0^{1/2}(x^2 + y^2) \, dx \, dy = \frac{3}{2} \cdot \frac{1}{24} \cdot 2 = \frac{1}{8} = 0.125$.

*ML connection:* In Gaussian mixture models (GMMs), each data point is modeled by a joint density $f(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$. The EM algorithm iterates between computing responsibilities (which component likely generated each point) and updating the joint density parameters.

---

## 3. Marginal Distributions

**Definition 4.6.3 (Marginal Distribution).** The marginal distribution of $X$ from the joint distribution of $(X, Y)$ is obtained by "summing out" or "integrating out" $Y$:

$$\text{Discrete:} \quad p_X(x) = \sum_y p_{X,Y}(x, y)$$

$$\text{Continuous:} \quad f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x, y) \, dy$$

**Theorem 4.6.1 (Marginals from joints).** The marginal distributions are uniquely determined by the joint distribution. However, the converse is false: knowing $p_X$ and $p_Y$ individually does *not* determine $p_{X,Y}$ — the dependence structure is additional information.

*Proof.* The first statement follows directly from the definitions above. For the converse, consider $X, Y \in \{0, 1\}$ with $P(X = 0) = P(X = 1) = P(Y = 0) = P(Y = 1) = 1/2$. The joint could be $P(X = Y) = 1$ (perfect correlation) or $P(X = x, Y = y) = 1/4$ (independence) — same marginals, different joints. $\square$

**Example 4.6.3:** Using the joint PMF from Example 4.6.1, compute the marginal distributions by summing rows and columns.

**Marginal of $X$** (sum each row across $Y$):

$$p_X(1) = 0.10 + 0.05 + 0.05 = 0.20$$

$$p_X(2) = 0.10 + 0.20 + 0.10 = 0.40$$

$$p_X(3) = 0.05 + 0.15 + 0.20 = 0.40$$

Check: $0.20 + 0.40 + 0.40 = 1.00$ $\checkmark$

**Marginal of $Y$** (sum each column across $X$):

$$p_Y(1) = 0.10 + 0.10 + 0.05 = 0.25, \quad p_Y(2) = 0.05 + 0.20 + 0.15 = 0.40, \quad p_Y(3) = 0.05 + 0.10 + 0.20 = 0.35$$

Check: $0.25 + 0.40 + 0.35 = 1.00$ $\checkmark$

```
MARGINALIZATION AS PROJECTION

   Joint density f(x,y):             Marginal f_X(x):

       y ▲                            f_X(x)
         │  ╭───╮                      ▲
         │ ╱ ╭─╮ ╲                     │   ╱╲
         ││ │ ● │ │                    │  ╱  ╲
         │ ╲ ╰─╯ ╱                    │ ╱    ╲
         │  ╰───╯                      │╱      ╲
         └──────────▶ x                └──────────▶ x

     "Project" (integrate)            Shadow on the x-axis
     onto the x-axis                  = marginal density
```

*ML connection:* Marginalizing latent variables is the core operation in probabilistic ML. The marginal likelihood (or evidence) is:

$$p(\mathbf{x}) = \int p(\mathbf{x} \mid \mathbf{z}) \, p(\mathbf{z}) \, d\mathbf{z}$$

This integral is typically intractable for complex models, which motivates variational inference (VAEs) and MCMC methods.

---

## 4. Conditional Distributions

**Definition 4.6.4 (Conditional PMF).** For discrete random variables with $P(Y = y) > 0$:

$$p_{X \mid Y}(x \mid y) = \frac{p_{X,Y}(x, y)}{p_Y(y)}$$

**Example 4.6.4:** Using the joint PMF from Example 4.6.1, compute the conditional distribution $P(Y \mid X = 2)$.

From Example 4.6.3, $p_X(2) = 0.40$. Apply the definition:

$$p_{Y \mid X}(1 \mid 2) = \frac{p(2, 1)}{p_X(2)} = \frac{0.10}{0.40} = 0.25$$

$$p_{Y \mid X}(2 \mid 2) = \frac{p(2, 2)}{p_X(2)} = \frac{0.20}{0.40} = 0.50$$

$$p_{Y \mid X}(3 \mid 2) = \frac{p(2, 3)}{p_X(2)} = \frac{0.10}{0.40} = 0.25$$

Check: $0.25 + 0.50 + 0.25 = 1.00$ $\checkmark$ — this is a valid PMF over $Y$.

**Interpretation:** Given $X = 2$, the value $Y = 2$ is most likely (probability $0.50$), while $Y = 1$ and $Y = 3$ are equally likely at $0.25$ each.

**Definition 4.6.5 (Conditional PDF).** For continuous random variables with $f_Y(y) > 0$:

$$f_{X \mid Y}(x \mid y) = \frac{f_{X,Y}(x, y)}{f_Y(y)}$$

**Example 4.6.5:** Using the joint PDF from Example 4.6.2, $f(x,y) = \frac{3}{2}(x^2 + y^2)$ on $[0,1]^2$, compute $f_{X \mid Y}(x \mid y = 1/2)$.

First, the marginal of $Y$: $f_Y(y) = \int_0^1 \frac{3}{2}(x^2 + y^2) \, dx = \frac{3}{2}\left[\frac{x^3}{3} + xy^2\right]_0^1 = \frac{3}{2}\left(\frac{1}{3} + y^2\right)$

At $y = 1/2$: $f_Y(1/2) = \frac{3}{2}\left(\frac{1}{3} + \frac{1}{4}\right) = \frac{3}{2} \cdot \frac{7}{12} = \frac{7}{8}$

The conditional density: $f_{X \mid Y}\!\left(x \mid \tfrac{1}{2}\right) = \frac{\frac{3}{2}(x^2 + 1/4)}{7/8} = \frac{12(x^2 + 1/4)}{7} = \frac{12x^2 + 3}{7}$ for $0 \leq x \leq 1$.

Check: $\int_0^1 \frac{12x^2 + 3}{7} \, dx = \frac{1}{7}[4x^3 + 3x]_0^1 = \frac{7}{7} = 1$ $\checkmark$

**Theorem 4.6.2 (Properties of conditional distributions).**

1. $p_{X \mid Y}(\cdot \mid y)$ is a valid PMF/PDF (sums/integrates to 1)
2. **Chain rule:** $p_{X,Y}(x, y) = p_{X \mid Y}(x \mid y) \, p_Y(y) = p_{Y \mid X}(y \mid x) \, p_X(x)$
3. **Bayes' theorem:** $p_{X \mid Y}(x \mid y) = \frac{p_{Y \mid X}(y \mid x) \, p_X(x)}{p_Y(y)}$

*Proof of (1).* $\sum_x p_{X \mid Y}(x \mid y) = \sum_x \frac{p_{X,Y}(x,y)}{p_Y(y)} = \frac{1}{p_Y(y)} \sum_x p_{X,Y}(x,y) = \frac{p_Y(y)}{p_Y(y)} = 1$. $\square$

*ML connection:* All discriminative models learn a conditional distribution $p(y \mid \mathbf{x})$ — the probability of a label given features. All generative models learn either the joint $p(\mathbf{x}, y)$ or use conditional generation $p(\mathbf{x} \mid \text{prompt})$. The distinction between these two paradigms is precisely the distinction between joint and conditional distributions.

---

## 5. Independence of Random Variables

**Definition 4.6.6 (Independence).** Random variables $X$ and $Y$ are *independent* (written $X \perp Y$) if:

$$p_{X,Y}(x, y) = p_X(x) \, p_Y(y) \quad \text{for all } x, y$$

Equivalently: $f_{X,Y}(x, y) = f_X(x) \, f_Y(y)$ in the continuous case.

**Theorem 4.6.3 (Equivalent characterizations of independence).**

The following are equivalent:

1. $p_{X,Y}(x,y) = p_X(x) \, p_Y(y)$ for all $x, y$
2. $p_{X \mid Y}(x \mid y) = p_X(x)$ for all $x$ and all $y$ with $p_Y(y) > 0$
3. $\mathbb{E}[g(X)h(Y)] = \mathbb{E}[g(X)] \, \mathbb{E}[h(Y)]$ for all measurable $g, h$

*Proof of $(1) \Rightarrow (2)$.* $p_{X \mid Y}(x \mid y) = \frac{p_{X,Y}(x,y)}{p_Y(y)} = \frac{p_X(x) \, p_Y(y)}{p_Y(y)} = p_X(x)$. $\square$

**Example 4.6.6:** Test whether $X$ and $Y$ are independent using the joint PMF from Example 4.6.1.

Independence requires $p(x,y) = p_X(x) \cdot p_Y(y)$ for **every** cell. Check the $(1,1)$ cell:

$$p_X(1) \cdot p_Y(1) = 0.20 \times 0.25 = 0.050$$

But $p(1,1) = 0.10 \neq 0.050$. **Independence fails** at the very first cell, so $X$ and $Y$ are **not independent**.

For comparison, an independent table with the same marginals would have $p(1,1) = 0.050$, $p(1,2) = 0.080$, $p(2,1) = 0.100$, etc. The deviations from these products encode the dependence structure.

**Theorem 4.6.4 (Independence implies zero covariance).** If $X \perp Y$, then $\text{Cov}(X, Y) = 0$.

*Proof.* $\text{Cov}(X,Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$. By property (3) with $g(x) = x$, $h(y) = y$: $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$, so $\text{Cov}(X,Y) = 0$. $\square$

**Warning:** The converse is false. Zero covariance does not imply independence. Consider $X \sim \text{Uniform}(-1,1)$ and $Y = X^2$. Then $\text{Cov}(X,Y) = \mathbb{E}[X^3] - \mathbb{E}[X]\mathbb{E}[X^2] = 0 - 0 = 0$, yet $Y$ is completely determined by $X$.

**Example 4.6.7:** Compute $\text{Cov}(X, Y)$ from the joint PMF of Example 4.6.1.

Using $\text{Cov}(X,Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$:

**Step 1 — Compute $\mathbb{E}[X]$:** $\mathbb{E}[X] = 1(0.20) + 2(0.40) + 3(0.40) = 0.20 + 0.80 + 1.20 = 2.20$

**Step 2 — Compute $\mathbb{E}[Y]$:** $\mathbb{E}[Y] = 1(0.25) + 2(0.40) + 3(0.35) = 0.25 + 0.80 + 1.05 = 2.10$

**Step 3 — Compute $\mathbb{E}[XY]$:** Sum $xy \cdot p(x,y)$ over all cells:

$$\mathbb{E}[XY] = 1{\cdot}1(0.10) + 1{\cdot}2(0.05) + 1{\cdot}3(0.05) + 2{\cdot}1(0.10) + 2{\cdot}2(0.20) + 2{\cdot}3(0.10) + 3{\cdot}1(0.05) + 3{\cdot}2(0.15) + 3{\cdot}3(0.20)$$

$$= 0.10 + 0.10 + 0.15 + 0.20 + 0.80 + 0.60 + 0.15 + 0.90 + 1.80 = 4.80$$

**Step 4:** $\text{Cov}(X,Y) = 4.80 - (2.20)(2.10) = 4.80 - 4.62 = 0.18 > 0$

The positive covariance confirms that $X$ and $Y$ tend to increase together — consistent with the large mass on the diagonal of the joint table.

*ML connection:* The i.i.d. assumption (independent and identically distributed) is foundational in ML. Training data $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ are assumed to be independent draws from a joint distribution $p(\mathbf{x}, y)$. When this assumption fails (time series, spatial data, social networks), specialized architectures (RNNs, graph neural networks) are needed.

---

## 6. The Multivariate Gaussian Distribution

**Definition 4.6.7 (Multivariate Gaussian).** A random vector $\mathbf{X} = (X_1, \ldots, X_d)^T$ has a *multivariate Gaussian* (or normal) distribution $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ if its PDF is:

$$f_{\mathbf{X}}(\mathbf{x}) = \frac{1}{(2\pi)^{d/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\!\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

where:

- $\boldsymbol{\mu} \in \mathbb{R}^d$ is the **mean vector**: $\boldsymbol{\mu} = \mathbb{E}[\mathbf{X}]$
- $\boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}$ is the **covariance matrix**: $\boldsymbol{\Sigma} = \mathbb{E}[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^T]$, symmetric and positive definite
- $|\boldsymbol{\Sigma}| = \det(\boldsymbol{\Sigma})$ is the determinant
- The quantity $(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})$ is the **Mahalanobis distance** squared

### 6.1 Geometric Interpretation

The contours of constant density are **ellipsoids** defined by:

$$(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) = c^2$$

The eigenvectors of $\boldsymbol{\Sigma}$ give the axes of the ellipsoid; the eigenvalues give the squared semi-axis lengths.

```
MULTIVARIATE GAUSSIAN CONTOURS (2D)

  σ₁² = 4, σ₂² = 1, ρ = 0         σ₁² = 4, σ₂² = 1, ρ = 0.8

       y                                  y
       ▲                                  ▲
       │   ┌─────────┐                    │       ╱╱
       │   │ ╭─────╮ │                    │     ╱╱
       │   │ │  ●  │ │                    │   ╱╱ ●
       │   │ ╰─────╯ │                    │  ╱╱
       │   └─────────┘                    │╱╱
       └──────────────▶ x                └──────────────▶ x

     Axes aligned with                 Axes rotated — tilted
     coordinate axes                   ellipses indicate
     (uncorrelated)                    positive correlation

  Eigenvalues of Σ = semi-axis lengths²
  Eigenvectors of Σ = axis directions
```

### 6.2 Marginals of the Multivariate Gaussian

**Theorem 4.6.5 (Marginals are Gaussian).** If $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ and we partition:

$$\mathbf{X} = \begin{pmatrix} \mathbf{X}_1 \\ \mathbf{X}_2 \end{pmatrix}, \quad \boldsymbol{\mu} = \begin{pmatrix} \boldsymbol{\mu}_1 \\ \boldsymbol{\mu}_2 \end{pmatrix}, \quad \boldsymbol{\Sigma} = \begin{pmatrix} \boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\ \boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_{22} \end{pmatrix}$$

then the marginal distribution of $\mathbf{X}_1$ is:

$$\mathbf{X}_1 \sim \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_{11})$$

*Proof sketch.* Integrate out $\mathbf{X}_2$ from the joint density. The exponent is quadratic in $\mathbf{X}_2$, so the integral over $\mathbf{X}_2$ is a Gaussian integral that evaluates in closed form, leaving a Gaussian in $\mathbf{X}_1$ with the stated parameters. $\square$

**Example 4.6.8:** Let $(X_1, X_2)^T \sim \mathcal{N}\!\left(\begin{pmatrix}3\\-1\end{pmatrix}, \begin{pmatrix}4 & 2\\2 & 9\end{pmatrix}\right)$.

Find the marginal distribution of each component.

By Theorem 4.6.5, just read off the diagonal blocks:

$$X_1 \sim \mathcal{N}(\mu_1, \Sigma_{11}) = \mathcal{N}(3, \, 4) \quad \Rightarrow \quad \text{std dev} = 2$$

$$X_2 \sim \mathcal{N}(\mu_2, \Sigma_{22}) = \mathcal{N}(-1, \, 9) \quad \Rightarrow \quad \text{std dev} = 3$$

Since $\Sigma_{12} = 2 \neq 0$, the variables are correlated. The correlation coefficient is $\rho = \frac{2}{\sqrt{4 \cdot 9}} = \frac{2}{6} = \frac{1}{3} \approx 0.33$.

### 6.3 Conditionals of the Multivariate Gaussian

**Theorem 4.6.6 (Conditionals are Gaussian).** Using the same partition as above, the conditional distribution of $\mathbf{X}_1$ given $\mathbf{X}_2 = \mathbf{x}_2$ is:

$$\mathbf{X}_1 \mid \mathbf{X}_2 = \mathbf{x}_2 \sim \mathcal{N}\!\left(\boldsymbol{\mu}_{1|2}, \, \boldsymbol{\Sigma}_{1|2}\right)$$

where:

$$\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$$

$$\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$$

Key observations:

- The conditional **mean** is a *linear function* of the observed value $\mathbf{x}_2$
- The conditional **covariance** does *not depend* on $\mathbf{x}_2$ — only on the covariance structure
- $\boldsymbol{\Sigma}_{1|2}$ is the Schur complement of $\boldsymbol{\Sigma}_{22}$ in $\boldsymbol{\Sigma}$

**Example 4.6.9:** Using the bivariate Gaussian from Example 4.6.8, compute $X_1 \mid X_2 = 2$.

Identify the blocks: $\mu_1 = 3$, $\mu_2 = -1$, $\Sigma_{11} = 4$, $\Sigma_{12} = 2$, $\Sigma_{22} = 9$.

**Conditional mean:**

$$\mu_{1|2} = \mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2 - \mu_2) = 3 + \frac{2}{9}(2 - (-1)) = 3 + \frac{2}{9} \cdot 3 = 3 + \frac{2}{3} = \frac{11}{3} \approx 3.67$$

**Conditional variance:**

$$\Sigma_{1|2} = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21} = 4 - \frac{2 \cdot 2}{9} = 4 - \frac{4}{9} = \frac{32}{9} \approx 3.56$$

So $X_1 \mid X_2 = 2 \sim \mathcal{N}\!\left(\frac{11}{3}, \, \frac{32}{9}\right)$.

**Interpretation:** Observing $X_2 = 2$ (above its mean of $-1$) shifts $X_1$'s mean upward from $3$ to $3.67$ due to the positive correlation. The conditional variance $32/9 \approx 3.56$ is smaller than the marginal variance $4$ — knowing $X_2$ reduces uncertainty about $X_1$.

*ML connection:* These formulas are the engine behind Gaussian processes. Given observed data points, the GP posterior (conditional distribution over function values at new inputs) is computed exactly using these conditional Gaussian formulas — no iterative optimization needed.

### 6.4 Independence in the Multivariate Gaussian

**Theorem 4.6.7 (Gaussian independence = zero correlation).** For jointly Gaussian random variables:

$$X \perp Y \iff \text{Cov}(X, Y) = 0 \iff \boldsymbol{\Sigma}_{12} = \mathbf{0}$$

*Proof.* ($\Rightarrow$) Independence implies zero covariance (Theorem 4.6.4). ($\Leftarrow$) If $\boldsymbol{\Sigma}_{12} = \mathbf{0}$, the covariance matrix is block diagonal: $\boldsymbol{\Sigma} = \begin{pmatrix} \boldsymbol{\Sigma}_{11} & \mathbf{0} \\ \mathbf{0} & \boldsymbol{\Sigma}_{22} \end{pmatrix}$. Then $\boldsymbol{\Sigma}^{-1}$ is also block diagonal, and the joint PDF factors: $f(\mathbf{x}_1, \mathbf{x}_2) = f(\mathbf{x}_1) \cdot f(\mathbf{x}_2)$. $\square$

This is a special property of the Gaussian: for general distributions, zero covariance does not imply independence (recall the counterexample in Section 5).

---

## 7. Transformations of Joint Distributions

**Theorem 4.6.8 (Change of variables, multivariate).** Let $\mathbf{X}$ be a continuous random vector with density $f_{\mathbf{X}}(\mathbf{x})$, and let $\mathbf{Y} = g(\mathbf{X})$ where $g : \mathbb{R}^d \to \mathbb{R}^d$ is a differentiable bijection. Then:

$$f_{\mathbf{Y}}(\mathbf{y}) = f_{\mathbf{X}}\!\left(g^{-1}(\mathbf{y})\right) \left|\det\!\left(\frac{\partial g^{-1}}{\partial \mathbf{y}}\right)\right|$$

where $\frac{\partial g^{-1}}{\partial \mathbf{y}}$ is the **Jacobian matrix** of the inverse transformation, and $|\det(\cdot)|$ is the absolute value of its determinant.

Equivalently, using $\mathbf{x} = g^{-1}(\mathbf{y})$:

$$f_{\mathbf{Y}}(\mathbf{y}) = f_{\mathbf{X}}(\mathbf{x}) \left|\det\!\left(\frac{\partial g}{\partial \mathbf{x}}\right)\right|^{-1}$$

*Proof sketch.* For an infinitesimal volume element $d\mathbf{x}$, the transformation $g$ maps it to a volume element $d\mathbf{y} = |\det(J_g)| \, d\mathbf{x}$. Probability is preserved: $f_{\mathbf{Y}}(\mathbf{y}) \, d\mathbf{y} = f_{\mathbf{X}}(\mathbf{x}) \, d\mathbf{x}$, so $f_{\mathbf{Y}}(\mathbf{y}) = f_{\mathbf{X}}(\mathbf{x}) / |\det(J_g)|$. $\square$

```
CHANGE OF VARIABLES — VOLUME DISTORTION

  Original space (X):                Transformed space (Y = g(X)):

  ┌──┐                                ╱╲
  │  │ unit square                    ╱  ╲  parallelogram
  │  │ area = 1                      ╱    ╲ area = |det(J)|
  └──┘                              ╱──────╲

  Probability is conserved:
  f_Y(y) · |det(J)| = f_X(x)
  f_Y(y) = f_X(g⁻¹(y)) · |det(J⁻¹)|

  The Jacobian determinant measures how much g
  stretches or compresses local volumes
```

**Example 4.6.10:** Let $(X_1, X_2)$ be uniform on $[0,1]^2$, so $f_{X_1,X_2}(x_1, x_2) = 1$. Define $Y_1 = X_1 + X_2$, $Y_2 = X_1 - X_2$.

**Step 1 — Inverse:** $X_1 = (Y_1 + Y_2)/2$, $X_2 = (Y_1 - Y_2)/2$.

**Step 2 — Jacobian of the inverse:**

$$\frac{\partial(x_1, x_2)}{\partial(y_1, y_2)} = \begin{pmatrix} 1/2 & 1/2 \\ 1/2 & -1/2 \end{pmatrix}, \quad \left|\det(J^{-1})\right| = |{-1/4} - {1/4}| = \frac{1}{2}$$

**Step 3 — Apply the formula:**

$$f_{Y_1, Y_2}(y_1, y_2) = f_{X_1, X_2}\!\left(\frac{y_1+y_2}{2}, \frac{y_1-y_2}{2}\right) \cdot \frac{1}{2} = 1 \cdot \frac{1}{2} = \frac{1}{2}$$

on the diamond-shaped region $\{(y_1, y_2) : 0 < y_1 + y_2 < 2, \, 0 < y_1 - y_2 < 2\}$. The area of this diamond is $2$, so $\frac{1}{2} \times 2 = 1$ $\checkmark$.

*ML connection:* The change-of-variables formula is the mathematical foundation of **normalizing flows**. Starting from a simple base distribution $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I)$, a normalizing flow applies a sequence of invertible transformations $\mathbf{x} = f_K \circ f_{K-1} \circ \cdots \circ f_1(\mathbf{z})$ to model complex data distributions. The log-likelihood is:

$$\log p(\mathbf{x}) = \log p(\mathbf{z}) - \sum_{k=1}^K \log\!\left|\det\!\left(\frac{\partial f_k}{\partial \mathbf{h}_{k-1}}\right)\right|$$

Architectures like RealNVP and GLOW design transformations with efficiently computable Jacobian determinants (e.g., triangular Jacobians where $\det(J) = \prod_i J_{ii}$).

---

## 8. Applications in ML

### 8.1 Multivariate Gaussian in Latent Spaces (VAE)

A Variational Autoencoder (VAE) models data generation as:

$$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I) \quad \text{(prior in latent space)}$$

$$\mathbf{x} \mid \mathbf{z} \sim p_\theta(\mathbf{x} \mid \mathbf{z}) \quad \text{(decoder)}$$

The choice of $\mathcal{N}(\mathbf{0}, I)$ as the prior makes the latent space smooth and continuous — nearby points in latent space decode to similar outputs. The KL divergence term in the VAE loss:

$$D_{\text{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \| \mathcal{N}(\mathbf{0}, I))$$

has a closed-form expression precisely because both distributions are Gaussian.

### 8.2 Gaussian Processes

A Gaussian process (GP) is an infinite-dimensional generalization of the multivariate Gaussian: any finite collection of function values $[f(\mathbf{x}_1), \ldots, f(\mathbf{x}_n)]^T$ is jointly Gaussian. The GP is fully specified by a mean function $m(\mathbf{x})$ and a kernel $k(\mathbf{x}, \mathbf{x}')$:

$$\begin{pmatrix} \mathbf{f} \\ \mathbf{f}_* \end{pmatrix} \sim \mathcal{N}\!\left(\begin{pmatrix} \mathbf{m} \\ \mathbf{m}_* \end{pmatrix}, \begin{pmatrix} K & K_* \\ K_*^T & K_{**} \end{pmatrix}\right)$$

Prediction at new points uses the conditional Gaussian formula (Theorem 4.6.6):

$$\mathbf{f}_* \mid \mathbf{f} \sim \mathcal{N}(K_*^T K^{-1}\mathbf{f}, \; K_{**} - K_*^T K^{-1} K_*)$$

### 8.3 Marginalizing Latent Variables

Many generative models define a joint distribution $p(\mathbf{x}, \mathbf{z})$ and require the marginal:

$$p(\mathbf{x}) = \int p(\mathbf{x} \mid \mathbf{z}) \, p(\mathbf{z}) \, d\mathbf{z}$$

| Model | Latent $\mathbf{z}$ | $p(\mathbf{z})$ | Marginalization |
|-------|---------------------|------------------|-----------------|
| GMM | Cluster assignment | Categorical | Finite sum (tractable) |
| Factor analysis | Latent factors | $\mathcal{N}(\mathbf{0}, I)$ | Gaussian integral (tractable) |
| VAE | Continuous latent code | $\mathcal{N}(\mathbf{0}, I)$ | Intractable, use ELBO |
| Diffusion model | Noise trajectory | Markov chain | Intractable, use score matching |

### 8.4 Conditional Generation

Modern generative AI is fundamentally about conditional distributions:

- **Image generation:** $p(\text{image} \mid \text{text prompt})$
- **Machine translation:** $p(\text{target sentence} \mid \text{source sentence})$
- **Speech synthesis:** $p(\text{audio} \mid \text{text})$

Each of these learns a conditional from the joint distribution of paired data.

---

## 9. Summary of Key Results

| Result | Statement |
|--------|-----------|
| Joint determines marginals | $f_X(x) = \int f_{X,Y}(x,y) \, dy$ |
| Marginals don't determine joint | Same marginals, different dependence structures |
| Conditional from joint | $f_{X \mid Y}(x \mid y) = f_{X,Y}(x,y) / f_Y(y)$ |
| Independence criterion | $X \perp Y \iff f_{X,Y} = f_X \cdot f_Y$ |
| Gaussian marginals | Marginals of a Gaussian are Gaussian |
| Gaussian conditionals | Conditionals of a Gaussian are Gaussian |
| Gaussian independence | $X \perp Y \iff \text{Cov}(X,Y) = 0$ (Gaussian only) |
| Change of variables | $f_{\mathbf{Y}}(\mathbf{y}) = f_{\mathbf{X}}(g^{-1}(\mathbf{y})) \, |\det(J^{-1})|$ |

---

## Exercises

**★ Basic**

1. Given the joint PMF table below for $(X, Y)$, compute the marginals $p_X$ and $p_Y$, the conditional $P(X = 1 \mid Y = 0)$, and check whether $X$ and $Y$ are independent.

    |       | $Y=0$ | $Y=1$ |
    |-------|-------|-------|
    | $X=0$ | 0.2   | 0.3   |
    | $X=1$ | 0.1   | 0.4   |

2. Let $(X, Y)$ have joint density $f(x,y) = 6x$ for $0 < x < y < 1$ and $0$ otherwise. Compute the marginal densities $f_X(x)$ and $f_Y(y)$.

3. If $\mathbf{X} \sim \mathcal{N}\!\left(\begin{pmatrix}1\\2\end{pmatrix}, \begin{pmatrix}4 & 0\\0 & 9\end{pmatrix}\right)$, what are the marginal distributions of $X_1$ and $X_2$? Are they independent?

**★★ Intermediate**

4. For the bivariate Gaussian $\mathbf{X} \sim \mathcal{N}\!\left(\begin{pmatrix}0\\0\end{pmatrix}, \begin{pmatrix}1 & \rho\\\rho & 1\end{pmatrix}\right)$, compute the conditional distribution $X_1 \mid X_2 = x_2$ using Theorem 4.6.6. Show that the conditional mean is $\rho x_2$ and the conditional variance is $1 - \rho^2$.

5. Let $X$ and $Y$ be independent $\text{Exp}(\lambda)$ random variables. Find the joint density of $(U, V) = (X + Y, X/(X+Y))$ using the change-of-variables formula. Show that $U$ and $V$ are independent.

6. Prove that for a multivariate Gaussian $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, any linear transformation $\mathbf{Y} = A\mathbf{X} + \mathbf{b}$ is also Gaussian with $\mathbf{Y} \sim \mathcal{N}(A\boldsymbol{\mu} + \mathbf{b}, A\boldsymbol{\Sigma}A^T)$.

7. Consider a Gaussian process with kernel $k(x, x') = \exp(-\frac{(x-x')^2}{2})$. You observe $f(0) = 1$. Using the conditional Gaussian formula, compute the posterior mean and variance of $f(1)$.

**★★★ Challenging**

8. **(Normalizing flows)** Let $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I_2)$ and define $\mathbf{x} = g(\mathbf{z})$ where $g(z_1, z_2) = (z_1, z_2 + z_1^2)$. Compute the Jacobian, its determinant, and the density $f_{\mathbf{X}}(\mathbf{x})$. Is $g$ a valid normalizing flow transformation?

9. **(Gaussian conditioning cascade)** For three jointly Gaussian variables $(X_1, X_2, X_3)$ with known covariance $\boldsymbol{\Sigma}$, show that conditioning sequentially — first on $X_3$, then on $X_2$ — gives the same result as conditioning on $(X_2, X_3)$ simultaneously. Verify using $\boldsymbol{\Sigma} = \begin{pmatrix} 1 & 0.5 & 0.3 \\ 0.5 & 1 & 0.4 \\ 0.3 & 0.4 & 1 \end{pmatrix}$ for $X_1 \mid X_2 = 1, X_3 = 0$.

10. **(Copulas and dependence beyond correlation)** Construct two bivariate distributions with standard normal marginals and $\text{Cov}(X,Y) = 0$ where: (a) $X \perp Y$, and (b) $X$ and $Y$ are dependent. This demonstrates that even for normal *marginals*, zero correlation does not imply independence unless the *joint* is Gaussian.

---

## Related Topics

- [Random Variables](random-variables.md) — univariate PMF, PDF, CDF foundations
- [Distributions](distributions.md) — the univariate families that serve as marginals
- [Expectation & Moments](expectation-and-moments.md) — covariance, correlation, and moment generating functions
- [Bayesian Inference](bayesian-inference.md) — posterior as a conditional distribution
- [Linear Algebra - Matrices](../linear-algebra/matrices.md) — positive definite matrices, determinants, Schur complements
- [Linear Algebra - Eigenvalues](../linear-algebra/eigenvalues.md) — spectral decomposition of covariance matrices

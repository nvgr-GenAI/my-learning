# Mathematical Notation

Reading machine learning papers requires fluency in mathematical notation the way reading code requires fluency in a programming language. This chapter is a reference for the notation used throughout this text and in the ML research literature. Bookmark it — you will return here often.

---

## Prerequisites

- [Sets & Logic](sets-and-logic.md)
- [Number Systems](number-systems.md)

---

## 1. Summation and Product Notation

### 1.1 Sigma Notation

$$\sum_{i=1}^{n} a_i = a_1 + a_2 + \cdots + a_n$$

**Properties:**

$$\sum_{i=1}^n (a_i + b_i) = \sum_{i=1}^n a_i + \sum_{i=1}^n b_i \qquad \text{(linearity)}$$

$$\sum_{i=1}^n c \cdot a_i = c \sum_{i=1}^n a_i \qquad \text{(scalar factoring)}$$

$$\sum_{i=1}^n c = nc \qquad \text{(constant sum)}$$

**Example 1.1:** Compute $\sum_{i=1}^{4} i^2$.

$$\sum_{i=1}^{4} i^2 = 1^2 + 2^2 + 3^2 + 4^2 = 1 + 4 + 9 + 16 = 30$$

Verify with the closed form: $\frac{4 \cdot 5 \cdot 9}{6} = \frac{180}{6} = 30$ ✓

**Common sums:**

| Sum | Closed Form | Where It Appears |
|-----|-------------|------------------|
| $\sum_{i=1}^n i$ | $\frac{n(n+1)}{2}$ | Complexity analysis |
| $\sum_{i=0}^n r^i$ | $\frac{1 - r^{n+1}}{1 - r}$ ($r \neq 1$) | Geometric series, discount factors in RL |
| $\sum_{i=1}^n \frac{1}{i}$ | $\approx \ln n + \gamma$ | Harmonic series, algorithm analysis |
| $\sum_{i=0}^\infty \frac{x^i}{i!}$ | $e^x$ | Exponential function, softmax |

**Double sums:**

$$\sum_{i=1}^m \sum_{j=1}^n a_{ij} = \sum_{j=1}^n \sum_{i=1}^m a_{ij}$$

**Example 1.2:** Compute $\sum_{i=1}^{2} \sum_{j=1}^{3} (i + j)$.

$$\sum_{i=1}^{2} \sum_{j=1}^{3} (i+j) = \underbrace{(1{+}1)+(1{+}2)+(1{+}3)}_{i=1} + \underbrace{(2{+}1)+(2{+}2)+(2{+}3)}_{i=2} = (2+3+4) + (3+4+5) = 21$$

*ML usage:* The empirical risk (training loss) is a sum:

$$\hat{R}(h) = \frac{1}{n} \sum_{i=1}^n L(h(x_i), y_i)$$

### 1.2 Product Notation

$$\prod_{i=1}^{n} a_i = a_1 \cdot a_2 \cdots a_n$$

*ML usage:* The likelihood of independent observations is a product:

$$L(\theta) = \prod_{i=1}^n p(x_i \mid \theta)$$

The log-likelihood converts this to a sum (numerically more stable):

$$\ell(\theta) = \sum_{i=1}^n \log p(x_i \mid \theta)$$

**Example 1.3:** Compute $\prod_{i=1}^{4} i$ and then its log-sum equivalent.

$$\prod_{i=1}^{4} i = 1 \cdot 2 \cdot 3 \cdot 4 = 24$$

$$\sum_{i=1}^{4} \ln(i) = \ln 1 + \ln 2 + \ln 3 + \ln 4 = 0 + 0.693 + 1.099 + 1.386 = 3.178 \approx \ln 24$$

This illustrates why ML code uses log-likelihoods: sums are easier to work with than products.

---

## 2. Set Notation

| Symbol | Meaning | Example |
|--------|---------|---------|
| $\in$ / $\notin$ | Element of / not element of | $3 \in \mathbb{Z}$, $\pi \notin \mathbb{Q}$ |
| $\subseteq$ / $\subsetneq$ | Subset / proper subset | $\mathbb{Q} \subsetneq \mathbb{R}$ |
| $\cup$ / $\cap$ | Union / intersection | $A \cup B$, $\bigcap_{i=1}^n A_i$ |
| $\setminus$ | Set difference | $\mathbb{R} \setminus \mathbb{Q}$ = irrationals |
| $A^c$ | Complement | $A^c = U \setminus A$ |
| $A \times B$ | Cartesian product | $\mathbb{R} \times \mathbb{R} = \mathbb{R}^2$ |
| $\|A\|$ or $\text{card}(A)$ | Cardinality | $\|\{1,2,3\}\| = 3$ |
| $\emptyset$ | Empty set | |
| $\mathcal{P}(A)$ | Power set | $\|\mathcal{P}(A)\| = 2^{\|A\|}$ |

**Example 2.1:** Build a set using set-builder notation.

Let $A = \{x \in \mathbb{Z} : 1 \leq x \leq 10 \text{ and } x \text{ is even}\}$. Enumerating:

$$A = \{2, 4, 6, 8, 10\}, \qquad |A| = 5$$

Now compute the power set of a small set $B = \{1, 2\}$:

$$\mathcal{P}(B) = \{\emptyset,\; \{1\},\; \{2\},\; \{1,2\}\}, \qquad |\mathcal{P}(B)| = 2^2 = 4$$

---

## 3. Logic and Quantifiers

| Symbol | Meaning | Read As |
|--------|---------|---------|
| $\forall$ | Universal quantifier | "for all" |
| $\exists$ | Existential quantifier | "there exists" |
| $\exists!$ | Unique existence | "there exists a unique" |
| $\Rightarrow$ | Implies | "if ... then" |
| $\Leftrightarrow$ | If and only if (iff) | "is equivalent to" |
| $\neg$ | Negation | "not" |
| $\land$ / $\lor$ | And / or | |
| $:$ or $\mid$ | "such that" (in set-builder) | $\{x \in \mathbb{R} : x > 0\}$ |
| $\therefore$ | Therefore | |

**Example 3.1:** Quantifiers over a small domain.

Let $S = \{1, 2, 3, 4, 5\}$.

- $\forall x \in S,\; x > 0$ — **True**, since every element is positive.
- $\exists x \in S,\; x > 4$ — **True**, since $5 > 4$.
- $\forall x \in S,\; x \text{ is even}$ — **False**, since $1 \in S$ is odd.
- $\exists! x \in S,\; x > 4$ — **True**, since $5$ is the *unique* element greater than 4.

Negation example: $\neg(\forall x \in S,\; x < 6) \Leftrightarrow \exists x \in S,\; x \geq 6$ — this is **False** (no such element), so the original $\forall$ statement is **True**.

---

## 4. Linear Algebra Notation

| Symbol | Meaning | Dimension |
|--------|---------|-----------|
| $x, y, v, w$ | Vectors (lowercase bold or arrow) | Column vectors by default |
| $x^T$ | Transpose | Row vector if $x$ is column |
| $A, B, W$ | Matrices (uppercase) | |
| $A^T$ | Matrix transpose | $(A^T)_{ij} = A_{ji}$ |
| $A^{-1}$ | Matrix inverse | $AA^{-1} = I$ |
| $A^\dagger$ or $A^*$ | Conjugate transpose (adjoint) | $(\bar{A})^T$ |
| $I$ or $I_n$ | Identity matrix | $n \times n$ |
| $\text{det}(A)$ or $|A|$ | Determinant | Scalar |
| $\text{tr}(A)$ | Trace | $\sum_i A_{ii}$ |
| $\text{rank}(A)$ | Rank | |
| $\text{diag}(a_1, \ldots, a_n)$ | Diagonal matrix | $n \times n$ |
| $\|x\|$ or $\|x\|_2$ | Euclidean norm | $\sqrt{\sum x_i^2}$ |
| $\|x\|_p$ | $p$-norm | $\left(\sum |x_i|^p\right)^{1/p}$ |
| $\langle x, y \rangle$ or $x \cdot y$ | Inner product | $\sum x_i y_i$ (for $\mathbb{R}^n$) |
| $x \otimes y$ | Outer product or tensor product | |
| $\lambda$ | Eigenvalue | |
| $\sigma_i$ | Singular value | |

*Convention in this text:* Vectors are column vectors unless stated otherwise. We often write $x \in \mathbb{R}^n$ to mean $x$ is an $n$-dimensional column vector.

**Example 4.1:** Let $x = (1, 2, 3)^T$ and $y = (4, 5, 6)^T$.

- *Inner product:* $\langle x, y \rangle = 1 \cdot 4 + 2 \cdot 5 + 3 \cdot 6 = 4 + 10 + 18 = 32$
- *Euclidean norm:* $\|x\|_2 = \sqrt{1^2 + 2^2 + 3^2} = \sqrt{14} \approx 3.74$
- *$L_1$ norm:* $\|x\|_1 = |1| + |2| + |3| = 6$
- *Trace:* For $A = \begin{pmatrix} 2 & 1 \\ 3 & 5 \end{pmatrix}$, $\text{tr}(A) = 2 + 5 = 7$

---

## 5. Calculus Notation

| Symbol | Meaning |
|--------|---------|
| $\frac{df}{dx}$ or $f'(x)$ | Derivative of $f$ w.r.t. $x$ |
| $\frac{\partial f}{\partial x_i}$ | Partial derivative |
| $\nabla f$ | Gradient vector $\left(\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}\right)^T$ |
| $\nabla^2 f$ or $H_f$ | Hessian matrix $\left[\frac{\partial^2 f}{\partial x_i \partial x_j}\right]$ |
| $J_f$ | Jacobian matrix $\left[\frac{\partial f_i}{\partial x_j}\right]$ |
| $\int_a^b f(x)\,dx$ | Definite integral |
| $\oint$ | Contour/line integral |
| $\frac{d}{dt}$ | Differential operator |
| $\Delta x$ | Finite difference |
| $dx$ | Infinitesimal |
| $O(f(n))$ | Big-O (asymptotic upper bound) |
| $o(f(n))$ | Little-o (strictly smaller growth) |

**Example 5.1:** Compute the gradient of $f(x_1, x_2) = x_1^2 + 3x_1 x_2 + x_2^2$.

$$\frac{\partial f}{\partial x_1} = 2x_1 + 3x_2, \qquad \frac{\partial f}{\partial x_2} = 3x_1 + 2x_2$$

$$\nabla f = \begin{pmatrix} 2x_1 + 3x_2 \\ 3x_1 + 2x_2 \end{pmatrix}$$

At the point $(1, 2)$: $\nabla f(1,2) = (2 \cdot 1 + 3 \cdot 2,\; 3 \cdot 1 + 2 \cdot 2)^T = (8, 7)^T$.

---

## 6. Probability Notation

| Symbol | Meaning |
|--------|---------|
| $P(A)$ | Probability of event $A$ |
| $P(A \mid B)$ | Conditional probability of $A$ given $B$ |
| $p(x)$ or $f_X(x)$ | Probability density (continuous) or mass (discrete) function |
| $F_X(x) = P(X \leq x)$ | Cumulative distribution function |
| $X \sim \mathcal{N}(\mu, \sigma^2)$ | $X$ follows normal distribution |
| $\mathbb{E}[X]$ or $\mu$ | Expected value |
| $\text{Var}(X)$ or $\sigma^2$ | Variance |
| $\text{Cov}(X, Y)$ | Covariance |
| $X \perp Y$ | $X$ and $Y$ are independent |
| $X \perp Y \mid Z$ | Conditional independence |
| $\hat{\theta}$ | Estimator of $\theta$ |
| $\theta^*$ | True/optimal parameter |
| $\mathcal{L}(\theta)$ or $L(\theta)$ | Likelihood function |
| $\ell(\theta)$ | Log-likelihood |
| $\text{KL}(p \| q)$ | KL divergence from $q$ to $p$ |
| $H(X)$ | Shannon entropy |
| $I(X; Y)$ | Mutual information |

**Example 6.1:** A discrete random variable $X$ takes values $\{1, 2, 3\}$ with $P(X=1)=0.2$, $P(X=2)=0.5$, $P(X=3)=0.3$.

- *Expected value:* $\mathbb{E}[X] = 1(0.2) + 2(0.5) + 3(0.3) = 0.2 + 1.0 + 0.9 = 2.1$
- *Variance:* $\mathbb{E}[X^2] = 1(0.2) + 4(0.5) + 9(0.3) = 0.2 + 2.0 + 2.7 = 4.9$
  $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = 4.9 - 4.41 = 0.49$
- *Entropy:* $H(X) = -\sum p_i \log_2 p_i = -(0.2 \log_2 0.2 + 0.5 \log_2 0.5 + 0.3 \log_2 0.3) \approx 1.49$ bits

---

## 7. Optimization Notation

| Symbol | Meaning |
|--------|---------|
| $\arg\min_x f(x)$ | Value of $x$ that minimizes $f$ |
| $\arg\max_x f(x)$ | Value of $x$ that maximizes $f$ |
| $\min_x f(x)$ | Minimum value of $f$ |
| $\inf$ / $\sup$ | Infimum / supremum (generalized min/max) |
| $\text{s.t.}$ | Subject to (constraint) |
| $\nabla f(\theta) = 0$ | First-order optimality condition |
| $\theta_{t+1} = \theta_t - \eta \nabla f(\theta_t)$ | Gradient descent update |
| $\eta$ or $\alpha$ | Learning rate |

**Example 7.1:** One step of gradient descent on $f(\theta) = \theta^2 - 4\theta + 5$.

We have $f'(\theta) = 2\theta - 4$. Starting at $\theta_0 = 0$ with learning rate $\eta = 0.1$:

$$\theta_1 = \theta_0 - \eta \cdot f'(\theta_0) = 0 - 0.1(2 \cdot 0 - 4) = 0 - 0.1(-4) = 0.4$$

The true minimum is at $\arg\min_\theta f(\theta) = 2$ (where $f'(\theta)=0$). After one step, $\theta$ moved from $0$ toward $2$.

---

## 8. Asymptotic Notation

**Definition 4.1 (Big-O).** $f(n) = O(g(n))$ if there exist constants $c > 0$ and $n_0$ such that $|f(n)| \leq c \cdot |g(n)|$ for all $n \geq n_0$.

**Definition 4.2 (Big-Omega).** $f(n) = \Omega(g(n))$ if $g(n) = O(f(n))$.

**Definition 4.3 (Big-Theta).** $f(n) = \Theta(g(n))$ if $f(n) = O(g(n))$ and $f(n) = \Omega(g(n))$.

**Definition 4.4 (Little-o).** $f(n) = o(g(n))$ if $\lim_{n \to \infty} f(n)/g(n) = 0$.

**Example 8.1:** Show that $f(n) = 3n^2 + 5n + 2$ is $O(n^2)$.

We need constants $c$ and $n_0$ such that $3n^2 + 5n + 2 \leq c \cdot n^2$ for all $n \geq n_0$.

For $n \geq 1$: $5n \leq 5n^2$ and $2 \leq 2n^2$, so $3n^2 + 5n + 2 \leq 10n^2$.

Choose $c = 10$, $n_0 = 1$. Therefore $f(n) = O(n^2)$. Verify at $n=3$: $f(3) = 27 + 15 + 2 = 44 \leq 10 \cdot 9 = 90$ ✓

Note: $f(n) = \Theta(n^2)$ since the $3n^2$ term also gives a lower bound.

| Notation | Meaning | Example |
|----------|---------|---------|
| $O(1)$ | Constant | Hash table lookup |
| $O(\log n)$ | Logarithmic | Binary search |
| $O(n)$ | Linear | Single pass over data |
| $O(n \log n)$ | Linearithmic | Merge sort |
| $O(n^2)$ | Quadratic | Attention mechanism (sequence length) |
| $O(n^3)$ | Cubic | Matrix multiplication (naive) |
| $O(2^n)$ | Exponential | Brute-force subset search |

*ML connection:* Transformer self-attention is $O(n^2 d)$ where $n$ is sequence length and $d$ is embedding dimension. This quadratic dependence on $n$ motivates efficient attention mechanisms (linear attention, sparse attention).

---

## 9. Quantum Computing Notation (Dirac)

| Symbol | Meaning |
|--------|---------|
| $\|0\rangle$, $\|1\rangle$ | Basis states (kets) |
| $\langle 0\|$, $\langle 1\|$ | Dual states (bras) |
| $\langle \psi \| \phi \rangle$ | Inner product (bra-ket) |
| $\|\psi\rangle \otimes \|\phi\rangle$ | Tensor product |
| $\|\psi\rangle\langle\phi\|$ | Outer product (operator) |
| $\hat{H}$, $\hat{U}$ | Operators (hat notation) |
| $\rho$ | Density matrix |
| $\text{Tr}(\rho)$ | Trace of density matrix |

See [Complex Vector Spaces](../quantum/complex-vector-spaces.md) for full treatment.

---

## 10. Common Abbreviations in ML Papers

| Abbreviation | Meaning |
|-------------|---------|
| i.i.d. | Independent and identically distributed |
| w.r.t. | With respect to |
| s.t. | Subject to, or such that (context-dependent) |
| w.l.o.g. | Without loss of generality |
| a.s. | Almost surely |
| a.e. | Almost everywhere |
| iff | If and only if |
| LHS / RHS | Left-hand side / right-hand side |
| QED or $\square$ | End of proof |

---

## 11. Reading ML Papers: A Guide

### Common Patterns

**"Let $f \in \mathcal{F}$ be..."** — $f$ is a function from hypothesis class $\mathcal{F}$.

**"For all $\varepsilon > 0$, there exists $N$ such that..."** — An $\varepsilon$-$\delta$ style argument. The claim is that something can be made arbitrarily small.

**"$\tilde{O}(n)$"** — Big-O ignoring logarithmic factors: $O(n \cdot \text{polylog}(n))$.

**"$X = O_p(1)$"** — $X$ is bounded in probability (stochastic Big-O).

**"w.h.p."** — With high probability (typically $\geq 1 - 1/\text{poly}(n)$).

### Symbol Overloading

Be aware that symbols are reused across contexts:

- $\sigma$: standard deviation, sigmoid function, singular value, Pauli matrix
- $\lambda$: eigenvalue, regularization coefficient, Lagrange multiplier, wavelength
- $\alpha$: learning rate, significance level, mixture weight, attention weight
- $p$: probability, polynomial, prime, $p$-norm
- $H$: entropy, Hessian, Hamiltonian, hypothesis class

Context determines meaning. When reading a paper, identify the notation conventions in the first few pages.

---

## Exercises

**★ Basic**

1. Expand $\sum_{i=0}^{3} 2^i$ and compute the result. Verify using the geometric series formula.

2. Write the following in mathematical notation: "The average squared error over $n$ data points."

3. What is the difference between $\min$ and $\inf$? Give an example where $\inf$ exists but $\min$ does not.

**★★ Intermediate**

4. Show that $\sum_{i=1}^n i^2 = \frac{n(n+1)(2n+1)}{6}$ by induction.

5. Prove that $\log(\prod_{i=1}^n a_i) = \sum_{i=1}^n \log(a_i)$ for $a_i > 0$. Why is this identity essential for maximum likelihood estimation?

6. The softmax function is $\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^k e^{z_j}}$. Show that $\text{softmax}(z + c\mathbf{1}) = \text{softmax}(z)$ for any constant $c$. This is the "log-sum-exp trick" for numerical stability.

**★★★ Challenging**

7. Show that if $f(n) = O(g(n))$ and $g(n) = O(h(n))$, then $f(n) = O(h(n))$ (transitivity of Big-O).

8. Prove Stirling's approximation: $n! \approx \sqrt{2\pi n} \left(\frac{n}{e}\right)^n$. (Hint: use the integral approximation $\ln(n!) = \sum_{k=1}^n \ln k \approx \int_1^n \ln x\,dx$.)

---

## Related Topics

- [Sets & Logic](sets-and-logic.md) — formal definitions of symbols
- [Vectors](../linear-algebra/vectors.md) — vector notation in practice
- [Differentiation](../calculus/differentiation.md) — calculus notation in depth
- [Probability Foundations](../probability/probability-foundations.md) — probability notation in depth

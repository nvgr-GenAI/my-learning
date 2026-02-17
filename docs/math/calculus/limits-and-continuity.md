# Limits & Continuity

Every training loop in machine learning asks the same question: does this sequence of losses converge? Gradient descent produces a sequence $L(\theta_0), L(\theta_1), L(\theta_2), \ldots$ and we need mathematical machinery to say precisely what it means for this sequence to approach a minimum. Limits formalize the notion of "getting arbitrarily close." Continuity ensures that small changes in inputs produce small changes in outputs — a requirement for gradient-based optimization to work at all. Series underpin Taylor approximations, Fourier features, and the infinite sums that appear throughout probability theory. This chapter builds the rigorous foundations of analysis that the rest of calculus depends on.

---

## Prerequisites

- [Foundations](../foundations/index.md) — sets, logic, proof techniques, functions, and the real number system

---

## 1. Sequences and Their Limits

### 1.1 Sequences

**Definition 3.1.1 (Sequence).** A *sequence* in $\mathbb{R}$ is a function $a: \mathbb{N} \to \mathbb{R}$. We write $(a_n)_{n=1}^{\infty}$ or simply $(a_n)$ for the sequence $a_1, a_2, a_3, \ldots$

**Definition 3.1.2 (Convergence of a Sequence).** A sequence $(a_n)$ *converges* to a limit $L \in \mathbb{R}$, written $\lim_{n \to \infty} a_n = L$ or $a_n \to L$, if:

$$\forall\, \varepsilon > 0,\; \exists\, N \in \mathbb{N} \text{ such that } n \geq N \Rightarrow |a_n - L| < \varepsilon$$

A sequence that does not converge is *divergent*.

```
CONVERGENCE VISUALIZED

 aₙ
  │    ●
  │  ●   ●
  │        ●  ● ● ● ● ● ● ●   ← within ε of L
  │───────────────────────────── L + ε
  │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  L
  │───────────────────────────── L - ε
  │●
  │
  └──┬──┬──┬──┬──┬──┬──┬──┬──→ n
     1  2  3  4  5     N

After index N, ALL terms stay within the ε-band around L.
```

*ML connection:* When training a neural network, the loss sequence $(L_t)$ should converge. "The model converged after 50 epochs" means that beyond epoch 50, the loss values remain within a negligible band of the final value. Divergent loss sequences signal exploding gradients or a learning rate that is too large.

**Example 3.1.1 (Converging vs. Diverging Sequences).**

*Converging:* Let $a_n = \frac{1}{n}$. Claim: $a_n \to 0$.
Given $\varepsilon > 0$, choose $N > 1/\varepsilon$ (Archimedean property). Then for $n \geq N$:

$$|a_n - 0| = \frac{1}{n} \leq \frac{1}{N} < \varepsilon \quad \checkmark$$

*Diverging:* Let $b_n = (-1)^n$. The terms alternate $-1, 1, -1, 1, \ldots$ and never settle within any $\varepsilon$-band around a single value. For instance, with $\varepsilon = 0.5$, no matter how large $N$ is, consecutive terms are distance 2 apart. This sequence diverges.

### 1.2 Properties of Convergent Sequences

**Theorem 3.1.1 (Uniqueness of Limits).** If $a_n \to L$ and $a_n \to M$, then $L = M$.

*Proof.* Suppose $L \neq M$. Let $\varepsilon = |L - M|/2 > 0$. By convergence, there exist $N_1, N_2$ such that $|a_n - L| < \varepsilon$ for $n \geq N_1$ and $|a_n - M| < \varepsilon$ for $n \geq N_2$. For $n \geq \max(N_1, N_2)$: $|L - M| \leq |L - a_n| + |a_n - M| < 2\varepsilon = |L - M|$, a contradiction. $\square$

**Definition 3.1.3 (Bounded Sequence).** A sequence $(a_n)$ is *bounded* if there exists $M > 0$ such that $|a_n| \leq M$ for all $n$.

**Theorem 3.1.2.** Every convergent sequence is bounded.

*Proof.* Let $a_n \to L$. Choose $\varepsilon = 1$. There exists $N$ such that $|a_n - L| < 1$ for all $n \geq N$, so $|a_n| < |L| + 1$ for $n \geq N$. Set $M = \max(|a_1|, \ldots, |a_{N-1}|, |L|+1)$. Then $|a_n| \leq M$ for all $n$. $\square$

**Definition 3.1.4 (Monotone Sequence).** A sequence is *monotonically increasing* if $a_{n+1} \geq a_n$ for all $n$, and *monotonically decreasing* if $a_{n+1} \leq a_n$ for all $n$.

**Theorem 3.1.3 (Monotone Convergence Theorem).** Every bounded monotone sequence converges.

*ML connection:* In convex optimization, gradient descent with a sufficiently small learning rate produces a monotonically decreasing, bounded-below loss sequence. The Monotone Convergence Theorem guarantees this sequence converges — providing a theoretical foundation for why gradient descent works on convex problems.

**Example 3.1.2 (Monotone Convergence).** Let $a_n = \frac{n}{n+1}$. This sequence is:

- *Monotonically increasing:* $a_{n+1} - a_n = \frac{n+1}{n+2} - \frac{n}{n+1} = \frac{1}{(n+1)(n+2)} > 0$
- *Bounded above:* $a_n = \frac{n}{n+1} = 1 - \frac{1}{n+1} < 1$ for all $n$

By the Monotone Convergence Theorem, $(a_n)$ converges. Computing the limit: $\lim_{n \to \infty} \frac{n}{n+1} = \lim_{n \to \infty} \frac{1}{1 + 1/n} = 1$.

---

## 2. Limits of Functions

### 2.1 The $\varepsilon$-$\delta$ Definition

**Definition 3.1.5 (Limit of a Function).** Let $f: D \to \mathbb{R}$ and let $c$ be a limit point of $D$. We say $\lim_{x \to c} f(x) = L$ if:

$$\forall\, \varepsilon > 0,\; \exists\, \delta > 0 \text{ such that } 0 < |x - c| < \delta \Rightarrow |f(x) - L| < \varepsilon$$

```
EPSILON-DELTA DEFINITION

 f(x)
  │
  │          ┌─────────────┐
 L+ε ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─│─ ─ ─
  │          │    ╱        │
  L ─ ─ ─ ─ │─ ╱─ ─ ─ ─ ─│─ ─ ─
  │          │╱            │
 L-ε ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─│─ ─ ─
  │          └─────────────┘
  └──────────┼──┼──────────┼──→ x
            c-δ  c        c+δ

For every ε-band around L, there exists a δ-band
around c so that f maps (c-δ, c+δ) into (L-ε, L+ε).
```

**Example 3.1.3 (Evaluating a Limit by Factoring).** Compute $\lim_{x \to 2} \frac{x^2 - 4}{x - 2}$.

Direct substitution gives $\frac{0}{0}$ — an indeterminate form. Factor the numerator:

$$\frac{x^2 - 4}{x - 2} = \frac{(x-2)(x+2)}{x-2} = x + 2 \quad \text{for } x \neq 2$$

Since limits only care about values *near* $c$, not *at* $c$:

$$\lim_{x \to 2} \frac{x^2 - 4}{x - 2} = \lim_{x \to 2} (x + 2) = 4$$

**Example 3.1.4 ($\varepsilon$-$\delta$ Proof).** Prove $\lim_{x \to 3} (2x + 1) = 7$.

We need: for every $\varepsilon > 0$, find $\delta > 0$ such that $0 < |x - 3| < \delta \Rightarrow |(2x+1) - 7| < \varepsilon$.

*Scratch work:* $|(2x+1) - 7| = |2x - 6| = 2|x - 3|$. So we need $2|x-3| < \varepsilon$, i.e., $|x-3| < \varepsilon/2$.

*Proof:* Choose $\delta = \varepsilon/2$. Then $0 < |x - 3| < \delta$ implies:

$$|(2x+1) - 7| = 2|x - 3| < 2\delta = 2 \cdot \frac{\varepsilon}{2} = \varepsilon \quad \square$$

### 2.2 Limit Laws

**Theorem 3.1.4 (Limit Laws).** If $\lim_{x \to c} f(x) = L$ and $\lim_{x \to c} g(x) = M$, then:

| Law | Statement |
|-----|-----------|
| Sum | $\lim_{x \to c} [f(x) + g(x)] = L + M$ |
| Product | $\lim_{x \to c} [f(x) \cdot g(x)] = L \cdot M$ |
| Scalar | $\lim_{x \to c} [k \cdot f(x)] = k \cdot L$ |
| Quotient | $\lim_{x \to c} \frac{f(x)}{g(x)} = \frac{L}{M}$, provided $M \neq 0$ |
| Power | $\lim_{x \to c} [f(x)]^n = L^n$ for $n \in \mathbb{N}$ |

**Theorem 3.1.5 (Squeeze Theorem).** If $g(x) \leq f(x) \leq h(x)$ near $c$ (except possibly at $c$) and $\lim_{x \to c} g(x) = \lim_{x \to c} h(x) = L$, then $\lim_{x \to c} f(x) = L$.

*Proof.* Let $\varepsilon > 0$. There exist $\delta_1, \delta_2 > 0$ such that $|g(x) - L| < \varepsilon$ for $0 < |x - c| < \delta_1$ and $|h(x) - L| < \varepsilon$ for $0 < |x - c| < \delta_2$. Set $\delta = \min(\delta_1, \delta_2)$. For $0 < |x - c| < \delta$: $L - \varepsilon < g(x) \leq f(x) \leq h(x) < L + \varepsilon$, so $|f(x) - L| < \varepsilon$. $\square$

*ML connection:* The Squeeze Theorem is used to analyze activation functions at their limits. For instance, showing that the softmax output for the largest logit approaches 1 as a temperature parameter $\tau \to 0$ involves bounding the softmax between two simpler expressions that both converge to 1.

**Example 3.1.5 (Squeeze Theorem).** Show $\lim_{x \to 0} x^2 \sin\!\left(\frac{1}{x}\right) = 0$.

Since $-1 \leq \sin(1/x) \leq 1$ for all $x \neq 0$, multiplying by $x^2 \geq 0$:

$$-x^2 \leq x^2 \sin\!\left(\frac{1}{x}\right) \leq x^2$$

We know $\lim_{x \to 0}(-x^2) = 0$ and $\lim_{x \to 0} x^2 = 0$. By the Squeeze Theorem:

$$\lim_{x \to 0} x^2 \sin\!\left(\frac{1}{x}\right) = 0$$

**Example 3.1.6 (One-Sided Limits).** Consider $f(x) = \frac{|x|}{x}$. Compute the left and right limits at $x = 0$.

- *Right limit:* For $x > 0$, $|x| = x$, so $f(x) = x/x = 1$. Thus $\lim_{x \to 0^+} f(x) = 1$.
- *Left limit:* For $x < 0$, $|x| = -x$, so $f(x) = -x/x = -1$. Thus $\lim_{x \to 0^-} f(x) = -1$.

Since the one-sided limits differ ($1 \neq -1$), $\lim_{x \to 0} f(x)$ **does not exist**. This is a jump discontinuity.

**Example 3.1.7 (Limit at Infinity).** Compute $\lim_{x \to \infty} \frac{3x^2 + 1}{x^2 - 5}$.

Divide numerator and denominator by the highest power $x^2$:

$$\lim_{x \to \infty} \frac{3x^2 + 1}{x^2 - 5} = \lim_{x \to \infty} \frac{3 + 1/x^2}{1 - 5/x^2} = \frac{3 + 0}{1 - 0} = 3$$

*Rule of thumb:* When the degrees of numerator and denominator are equal, the limit at infinity is the ratio of leading coefficients.

---

## 3. Continuity

### 3.1 Definition and Characterization

**Definition 3.1.6 (Continuity at a Point).** A function $f: D \to \mathbb{R}$ is *continuous at $c \in D$* if:

$$\lim_{x \to c} f(x) = f(c)$$

Equivalently (in $\varepsilon$-$\delta$ form): for every $\varepsilon > 0$, there exists $\delta > 0$ such that $|x - c| < \delta \Rightarrow |f(x) - f(c)| < \varepsilon$.

**Definition 3.1.7 (Continuous Function).** $f$ is *continuous on $D$* if it is continuous at every point of $D$.

**Theorem 3.1.6 (Sequential Characterization of Continuity).** $f$ is continuous at $c$ if and only if for every sequence $(x_n)$ with $x_n \to c$, we have $f(x_n) \to f(c)$.

**Example 3.1.8 (Continuity Check — Three Conditions).** Is $f(x) = \frac{x^2 - 1}{x - 1}$ continuous at $x = 1$?

Check the three conditions for continuity at $c = 1$:

1. **$f(c)$ is defined?** $f(1) = \frac{1 - 1}{1 - 1} = \frac{0}{0}$ — undefined. $\boldsymbol{\times}$

Since condition 1 fails, $f$ is **not continuous** at $x = 1$. (The limit $\lim_{x \to 1} \frac{(x-1)(x+1)}{x-1} = 2$ exists, so this is a *removable* discontinuity — we could define $f(1) = 2$ to make it continuous.)

*Compare:* Let $g(x) = x^2$. At $c = 1$: (1) $g(1) = 1$ defined $\checkmark$, (2) $\lim_{x \to 1} x^2 = 1$ exists $\checkmark$, (3) limit equals $g(1)$ $\checkmark$. So $g$ is continuous at $x = 1$.

### 3.2 Types of Discontinuities

| Type | Condition | Example | Visual |
|------|-----------|---------|--------|
| Removable | $\lim_{x \to c} f(x)$ exists but $\neq f(c)$ | $f(x) = \frac{x^2 - 1}{x-1}$ at $x=1$ | Hole in graph |
| Jump | Left and right limits exist but differ | Step function at $x=0$ | Vertical gap |
| Essential | Limit does not exist | $\sin(1/x)$ at $x=0$ | Wild oscillation |

```
TYPES OF DISCONTINUITY

Removable            Jump                 Essential
    │   ╱                │  ●──────          │ ╱╲╱╲╱╲╱╲
    │  ╱                 │  │                │╱╲╱╲╱╲╱╲╱╲
    │ ○                  │  │                │ (oscillates
    │╱                   │──○                │  infinitely)
────┼──── x          ────┼──── x         ────┼──── x
    c                    c                   c

○ = point removed      Gap between          No limit exists
● or value differs     one-sided limits
```

*ML connection:* Gradient-based optimization requires the loss function to be continuous (and usually differentiable). The ReLU activation $f(x) = \max(0, x)$ is continuous everywhere but not differentiable at $x = 0$. The step function used in the original perceptron has a jump discontinuity at 0, which is why it cannot be trained with gradient descent — motivating smooth activations like sigmoid and tanh.

**Example 3.1.9 (Classifying Discontinuities).** Classify the discontinuity of each function at $x = 0$:

**(a) Removable:** $f(x) = \frac{\sin x}{x}$ at $x = 0$. The function is undefined at 0, but $\lim_{x \to 0}\frac{\sin x}{x} = 1$ exists. Defining $f(0) = 1$ removes the discontinuity.

**(b) Jump:** $f(x) = \begin{cases} 1 & x > 0 \\ -1 & x < 0 \end{cases}$ (the sign function). $\lim_{x \to 0^+} f(x) = 1$ and $\lim_{x \to 0^-} f(x) = -1$. The one-sided limits exist but differ, so the discontinuity is a jump.

**(c) Essential:** $f(x) = \sin(1/x)$ at $x = 0$. As $x \to 0$, $1/x$ oscillates through all values, so $\sin(1/x)$ oscillates between $-1$ and $1$ without settling. No limit exists — essential discontinuity.

### 3.3 Properties of Continuous Functions

**Theorem 3.1.7 (Algebra of Continuous Functions).** If $f$ and $g$ are continuous at $c$, then $f + g$, $f \cdot g$, $kf$ (for $k \in \mathbb{R}$), and $f/g$ (when $g(c) \neq 0$) are continuous at $c$. The composition $f \circ g$ is continuous if $g$ is continuous at $c$ and $f$ is continuous at $g(c)$.

*ML connection:* Since polynomials, exponentials, and compositions of continuous functions are continuous, neural networks with continuous activations (sigmoid, tanh, softplus, GELU) define continuous functions from input to output. This composition property is what makes deep networks well-behaved for optimization.

---

## 4. The Intermediate Value Theorem

**Theorem 3.1.8 (Intermediate Value Theorem).** If $f: [a, b] \to \mathbb{R}$ is continuous and $f(a) < v < f(b)$ (or $f(b) < v < f(a)$), then there exists $c \in (a, b)$ such that $f(c) = v$.

```
IVT VISUALIZED

 f(x)
  │
f(b)─ ─ ─ ─ ─ ─ ─ ─ ─●
  │                 ╱
  v ─ ─ ─ ─ ─ ─ ● ─ ─ ─ ─   ← guaranteed to hit v
  │           ╱
f(a)●─ ─ ─ ╱
  │
  └────┼────┼────┼────→ x
       a    c    b

If f is continuous and v is between f(a) and f(b),
then f(c) = v for some c between a and b.
```

*ML connection:* The IVT guarantees that a continuous loss function that starts above a target value and reaches below it must pass through that target value at some parameter setting. This is used in bisection-based hyperparameter search: if validation accuracy is 0.6 at learning rate $10^{-4}$ and 0.9 at learning rate $10^{-2}$, and accuracy varies continuously with the learning rate, then every accuracy between 0.6 and 0.9 is achieved at some intermediate learning rate.

**Corollary 3.1.1 (Bolzano's Theorem).** If $f$ is continuous on $[a,b]$ with $f(a)$ and $f(b)$ of opposite signs, then $f$ has a root in $(a, b)$.

---

## 5. Series

### 5.1 Definitions and Basic Series

**Definition 3.1.8 (Infinite Series).** Given a sequence $(a_n)$, the *infinite series* $\sum_{n=1}^{\infty} a_n$ is defined as the limit of partial sums:

$$\sum_{n=1}^{\infty} a_n = \lim_{N \to \infty} S_N \qquad \text{where } S_N = \sum_{n=1}^{N} a_n$$

The series *converges* if this limit exists and is finite; otherwise it *diverges*.

**Theorem 3.1.9 (Divergence Test).** If $\sum a_n$ converges, then $a_n \to 0$. Equivalently, if $a_n \not\to 0$, then $\sum a_n$ diverges.

**Warning:** The converse is false. $a_n \to 0$ does NOT guarantee convergence (see harmonic series below).

### 5.2 Important Series

| Series | Formula | Convergence | ML Context |
|--------|---------|-------------|------------|
| Geometric | $\sum_{n=0}^{\infty} r^n = \frac{1}{1-r}$ | $\|r\| < 1$ | Discount factors in RL, exponential decay |
| Harmonic | $\sum_{n=1}^{\infty} \frac{1}{n}$ | Diverges | Harmonic learning rate schedules |
| $p$-series | $\sum_{n=1}^{\infty} \frac{1}{n^p}$ | $p > 1$ | Heavy-tailed distributions |
| Exponential | $\sum_{n=0}^{\infty} \frac{x^n}{n!} = e^x$ | All $x$ | Softmax, log-sum-exp |

*ML connection:* The geometric series $\sum_{t=0}^{\infty} \gamma^t r_{t}$ with discount factor $\gamma \in [0,1)$ is the discounted return in reinforcement learning. Convergence requires $|\gamma| < 1$, which is why the discount factor must be strictly less than 1.

**Example 3.1.10 (Geometric and $p$-Series).**

*Geometric:* $\sum_{n=0}^{\infty} \left(\frac{1}{2}\right)^n$. Here $r = 1/2$, and $|r| < 1$, so it converges:

$$\sum_{n=0}^{\infty} \left(\frac{1}{2}\right)^n = \frac{1}{1 - 1/2} = 2$$

Partial sums: $S_0 = 1,\; S_1 = 1.5,\; S_2 = 1.75,\; S_3 = 1.875, \ldots \to 2$.

*$p$-series:* $\sum_{n=1}^{\infty} \frac{1}{n^2}$ has $p = 2 > 1$, so it converges (to $\pi^2/6 \approx 1.645$). But $\sum_{n=1}^{\infty} \frac{1}{n}$ has $p = 1$, so it **diverges** (harmonic series).

### 5.3 Convergence Tests

**Theorem 3.1.10 (Comparison Test).** If $0 \leq a_n \leq b_n$ for all $n$:

- If $\sum b_n$ converges, then $\sum a_n$ converges.
- If $\sum a_n$ diverges, then $\sum b_n$ diverges.

**Theorem 3.1.11 (Ratio Test).** Let $L = \lim_{n \to \infty} \left|\frac{a_{n+1}}{a_n}\right|$:

- $L < 1$: the series converges absolutely
- $L > 1$: the series diverges
- $L = 1$: inconclusive

**Example 3.1.11 (Ratio Test).** Does $\sum_{n=0}^{\infty} \frac{3^n}{n!}$ converge?

Let $a_n = \frac{3^n}{n!}$. Compute the ratio:

$$L = \lim_{n \to \infty} \left|\frac{a_{n+1}}{a_n}\right| = \lim_{n \to \infty} \frac{3^{n+1}}{(n+1)!} \cdot \frac{n!}{3^n} = \lim_{n \to \infty} \frac{3}{n+1} = 0$$

Since $L = 0 < 1$, the series **converges** (in fact, to $e^3 \approx 20.09$, the exponential series evaluated at $x = 3$).

**Theorem 3.1.12 (Root Test).** Let $L = \lim_{n \to \infty} |a_n|^{1/n}$:

- $L < 1$: converges absolutely
- $L > 1$: diverges
- $L = 1$: inconclusive

**Definition 3.1.9 (Absolute Convergence).** $\sum a_n$ *converges absolutely* if $\sum |a_n|$ converges. Absolute convergence implies convergence (but not conversely).

*ML connection:* Convergence tests matter when analyzing whether infinite-width neural network expansions or kernel series converge. The ratio test is particularly useful for determining the radius of convergence of Taylor series used in activation function approximations.

---

## 6. Uniform Convergence

### 6.1 Pointwise vs. Uniform Convergence

**Definition 3.1.10 (Pointwise Convergence).** A sequence of functions $(f_n)$ *converges pointwise* to $f$ on $D$ if for each fixed $x \in D$:

$$\lim_{n \to \infty} f_n(x) = f(x)$$

That is, for each $x$ and each $\varepsilon > 0$, there exists $N(x, \varepsilon)$ such that $n \geq N \Rightarrow |f_n(x) - f(x)| < \varepsilon$.

**Definition 3.1.11 (Uniform Convergence).** $(f_n)$ *converges uniformly* to $f$ on $D$ if:

$$\forall\, \varepsilon > 0,\; \exists\, N(\varepsilon) \text{ such that } n \geq N \Rightarrow |f_n(x) - f(x)| < \varepsilon \;\; \forall\, x \in D$$

The key distinction: in uniform convergence, $N$ depends only on $\varepsilon$, not on $x$.

```
POINTWISE vs. UNIFORM CONVERGENCE

Pointwise (N depends on x):        Uniform (one N works for all x):

 f(x)   f₁  f₂  f₃  f                f(x)   f₁  f₂  f₃  f
  │    │  │  │  │                       │    │  │  │  │
  │    │  │  ╲╱                         │    │  ╲ ╲╱
  │    │  ╲ ╱╲                          │    ╲  ╲╱╲
  │    ╲ ╱╲╱ ╲                          │     ╲╱╲╱ ╲
  │     ╳╱    ╲                         │     ╱╲    ╲
  │    ╱╲      ╲                        │    ╱  ╲    ╲
  └────────────────→ x                  └────────────────→ x

  Convergence speed varies             All points converge at
  across different x values             the same rate
```

### 6.2 Why Uniform Convergence Matters

**Theorem 3.1.13 (Uniform Limit of Continuous Functions).** If each $f_n$ is continuous on $D$ and $f_n \to f$ uniformly on $D$, then $f$ is continuous on $D$.

*Proof.* Fix $c \in D$ and $\varepsilon > 0$. By uniform convergence, choose $N$ such that $|f_N(x) - f(x)| < \varepsilon/3$ for all $x$. Since $f_N$ is continuous at $c$, choose $\delta > 0$ such that $|x - c| < \delta \Rightarrow |f_N(x) - f_N(c)| < \varepsilon/3$. Then for $|x - c| < \delta$:

$$|f(x) - f(c)| \leq |f(x) - f_N(x)| + |f_N(x) - f_N(c)| + |f_N(c) - f(c)| < \frac{\varepsilon}{3} + \frac{\varepsilon}{3} + \frac{\varepsilon}{3} = \varepsilon \quad \square$$

**Warning:** Pointwise convergence does NOT preserve continuity. A sequence of continuous functions can converge pointwise to a discontinuous function.

*ML connection:* The universal approximation theorem states that neural networks can approximate any continuous function on a compact set. The proof relies on uniform convergence: the neural network approximations converge uniformly to the target function, ensuring the limit inherits continuity and other analytic properties.

### 6.3 Weierstrass M-Test

**Theorem 3.1.14 (Weierstrass M-Test).** If $|f_n(x)| \leq M_n$ for all $x \in D$ and $\sum M_n$ converges, then $\sum f_n$ converges uniformly on $D$.

*ML connection:* The M-test is used to justify term-by-term differentiation of series representations of neural network outputs — critical for computing gradients of models defined by infinite series (e.g., neural tangent kernel expansions).

---

## 7. Applications in ML/AI

### 7.1 Training Convergence

The training loop produces a loss sequence:

$$L_0, L_1, L_2, \ldots \quad \text{where } L_t = \mathcal{L}(\theta_t)$$

| Convergence Concept | Training Interpretation |
|---------------------|----------------------|
| $L_t \to L^*$ | Loss converges to optimal value |
| Monotone decreasing $L_t$ | Loss decreases every step (guaranteed for small enough learning rate on convex problems) |
| Bounded below by 0 | Loss is non-negative |
| Cauchy criterion | $|L_t - L_s| < \varepsilon$ for large $t, s$ — used as stopping criterion |

### 7.2 Series in Taylor Approximations

Many ML operations use truncated Taylor series. The exponential function in softmax:

$$e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!}$$

The log-sum-exp trick, sigmoid approximations, and GELU activation all rely on Taylor series for efficient computation or theoretical analysis.

### 7.3 Continuity and Gradient-Based Optimization

Gradient descent requires the loss landscape to be continuous (and differentiable). The chain of requirements:

```
Continuous loss       Differentiable loss      Lipschitz gradients
      │                      │                        │
      ▼                      ▼                        ▼
  Small input           Gradients exist         Gradient descent
  changes →             everywhere →            converges at
  small output          backpropagation         known rate
  changes               works
```

If continuity fails at any layer, gradients become undefined or meaningless. This is why:

- **ReLU** replaced the step function (continuous, though not differentiable at 0)
- **Smooth activations** (GELU, SiLU, Softplus) are gaining popularity (continuous and infinitely differentiable)
- **Straight-through estimators** are needed for discrete/discontinuous operations during training

---

## Exercises

**★ Basic**

1. Prove from the $\varepsilon$-$\delta$ definition that $\lim_{n \to \infty} \frac{1}{n} = 0$.

2. Compute $\lim_{x \to 2} \frac{x^2 - 4}{x - 2}$ using limit laws.

3. Determine whether the geometric series $\sum_{n=0}^{\infty} (0.99)^n$ converges. If so, find its sum. Interpret this as a discounted return in RL with $\gamma = 0.99$.

4. Classify the discontinuity (if any) of $f(x) = \frac{\sin x}{x}$ at $x = 0$.

**★★ Intermediate**

5. Prove the sum law for limits: if $a_n \to L$ and $b_n \to M$, then $a_n + b_n \to L + M$.

6. Use the Squeeze Theorem to show $\lim_{n \to \infty} \frac{\sin n}{n} = 0$. Explain why this is relevant to oscillating loss curves that still converge.

7. Determine whether $\sum_{n=1}^{\infty} \frac{n}{2^n}$ converges using the ratio test. This series arises in the expected number of coin flips to see heads.

8. Give an example of a sequence of continuous functions on $[0, 1]$ that converges pointwise to a discontinuous function.

**★★★ Challenging**

9. Prove that if $f$ is continuous on $[a, b]$, then $f$ is bounded on $[a, b]$ and attains its maximum and minimum (Extreme Value Theorem). Why does this guarantee that a continuous loss function over a compact parameter space has a global minimum?

10. Prove that the harmonic series $\sum 1/n$ diverges by grouping terms: $1 + \frac{1}{2} + (\frac{1}{3} + \frac{1}{4}) + (\frac{1}{5} + \frac{1}{6} + \frac{1}{7} + \frac{1}{8}) + \cdots$ and showing each group sums to at least $\frac{1}{2}$.

11. Let $f_n(x) = x^n$ on $[0, 1]$. Show that $f_n \to f$ pointwise where $f(x) = 0$ for $x \in [0,1)$ and $f(1) = 1$. Prove that this convergence is NOT uniform. Relate this to the behavior of softmax with increasing temperature.

---

## Related Topics

- [Differentiation](differentiation.md) — limits of difference quotients define derivatives
- [Taylor Series](taylor-series.md) — infinite series representations of functions
- [Integration](integration.md) — limits of Riemann sums
- [Foundations](../foundations/index.md) — sets, real numbers, and proof techniques
- [Probability Foundations](../probability/index.md) — limits in the law of large numbers and central limit theorem

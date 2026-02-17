# Probability Foundations

Probability theory is the mathematical language of uncertainty, and machine learning is fundamentally the science of making decisions under uncertainty. Every classifier outputs a probability distribution over classes. Every generative model learns a probability distribution over data. Loss functions like cross-entropy are derived from probability axioms. Bayesian inference, maximum likelihood estimation, variational autoencoders, and diffusion models are all direct applications of the framework established in this chapter: the Kolmogorov axioms and the concept of a probability space $(\Omega, \mathcal{F}, P)$.

---

## Prerequisites

- [Foundations — Sets & Logic](../foundations/sets-and-logic.md) — set operations, $\sigma$-algebras, partitions, De Morgan's laws

---

## 1. Sample Spaces and Events

### 1.1 The Sample Space

**Definition 4.1.1 (Sample Space).** The *sample space* $\Omega$ is the set of all possible outcomes of a random experiment. Each element $\omega \in \Omega$ is called a *sample point* or *outcome*.

| Experiment | Sample Space $\Omega$ | Type |
|-----------|----------------------|------|
| Coin flip | $\{H, T\}$ | Finite |
| Roll a die | $\{1, 2, 3, 4, 5, 6\}$ | Finite |
| Count website visits | $\{0, 1, 2, \ldots\}$ | Countably infinite |
| Measure temperature | $\mathbb{R}_{\geq 0}$ | Uncountable |
| Generate an image ($n$ pixels) | $[0,1]^n$ | Uncountable |

**Example 4.1.1:** *List the sample space for flipping two coins.*

Each coin can land $H$ or $T$, so by the counting principle $|\Omega| = 2 \times 2 = 4$:

$$\Omega = \{HH,\; HT,\; TH,\; TT\}$$

If both coins are fair and flips are independent, each outcome has probability $1/4$. The event "at least one head" is $A = \{HH, HT, TH\}$, so $P(A) = 3/4$.

```
SAMPLE SPACE VISUALIZATION

Finite Ω (die roll)                 Continuous Ω (temperature)
┌─────────────────────┐             ┌──────────────────────────┐
│  ①  ②  ③  ④  ⑤  ⑥  │             │ ←─────────────────────→  │
│                     │             │ 0        50       100    │
│  Each outcome is    │             │ Every real value is      │
│  an isolated point  │             │  a possible outcome      │
└─────────────────────┘             └──────────────────────────┘
```

### 1.2 Events

**Definition 4.1.2 (Event).** An *event* is a subset $A \subseteq \Omega$. An event $A$ *occurs* if the outcome $\omega$ of the experiment satisfies $\omega \in A$.

**Definition 4.1.3 (Elementary Event).** A *simple* or *elementary event* is a singleton set $\{\omega\}$ for some $\omega \in \Omega$.

**Set operations on events:**

| Set Operation | Event Interpretation |
|--------------|---------------------|
| $A \cup B$ | $A$ or $B$ occurs (or both) |
| $A \cap B$ | Both $A$ and $B$ occur |
| $A^c = \Omega \setminus A$ | $A$ does not occur |
| $A \setminus B$ | $A$ occurs but $B$ does not |
| $A \cap B = \emptyset$ | $A$ and $B$ are mutually exclusive |

**Example 4.1.2:** *Roll a fair die. Let $A = \{2, 4, 6\}$ (even) and $B = \{1, 2, 3\}$ (at most 3). Compute $P(A \cup B)$ and $P(A \cap B)$.*

$$A \cap B = \{2\}, \quad P(A \cap B) = \tfrac{1}{6}$$

$$A \cup B = \{1, 2, 3, 4, 6\}, \quad P(A \cup B) = \tfrac{5}{6}$$

Verification via inclusion-exclusion: $P(A) + P(B) - P(A \cap B) = \tfrac{3}{6} + \tfrac{3}{6} - \tfrac{1}{6} = \tfrac{5}{6}$. $\checkmark$

```
EVENTS AS SUBSETS OF Ω

             Ω (sample space)
    ┌─────────────────────────────┐
    │         ┌───────┐           │
    │     A   │ A ∩ B │   B      │
    │  ┌──────┤       ├──────┐   │
    │  │      │       │      │   │
    │  │      └───────┘      │   │
    │  └─────────────────────┘   │
    │                             │
    │      (A ∪ B)^c              │
    └─────────────────────────────┘

P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
```

*ML connection:* In binary classification, the sample space is the feature space $\Omega = \mathbb{R}^d$. The event "class 1" is the region $A = \{\mathbf{x} \in \mathbb{R}^d : f(\mathbf{x}) > 0\}$ where $f$ is the decision function. The classifier partitions $\Omega$ into the decision regions $A$ and $A^c$.

---

## 2. $\sigma$-Algebras

### 2.1 Why We Need $\sigma$-Algebras

For finite sample spaces, we can assign probabilities to every subset of $\Omega$. For uncountable spaces like $\mathbb{R}$, the Banach-Tarski paradox and related results show that assigning a consistent probability to *every* subset is impossible. We must restrict attention to a well-behaved collection of "measurable" events.

### 2.2 Definition and Properties

**Definition 4.1.4 ($\sigma$-Algebra).** A *$\sigma$-algebra* (or *$\sigma$-field*) $\mathcal{F}$ on $\Omega$ is a collection of subsets of $\Omega$ satisfying:

1. $\Omega \in \mathcal{F}$ (the certain event is measurable)
2. If $A \in \mathcal{F}$, then $A^c \in \mathcal{F}$ (closure under complement)
3. If $A_1, A_2, \ldots \in \mathcal{F}$, then $\bigcup_{i=1}^{\infty} A_i \in \mathcal{F}$ (closure under countable union)

**Consequences:** From these three axioms it follows that $\emptyset \in \mathcal{F}$ (since $\emptyset = \Omega^c$), and $\mathcal{F}$ is closed under countable intersection (by De Morgan: $\bigcap A_i = (\bigcup A_i^c)^c$).

**Example:** For $\Omega = \{H, T\}$, the possible $\sigma$-algebras are:

| $\sigma$-Algebra | Events | Interpretation |
|------------------|--------|---------------|
| $\{\emptyset, \Omega\}$ | Trivial — no information | Cannot distinguish $H$ from $T$ |
| $\mathcal{P}(\Omega) = \{\emptyset, \{H\}, \{T\}, \{H,T\}\}$ | Full power set | Can observe outcome completely |

### 2.3 The Borel $\sigma$-Algebra

**Definition 4.1.5 (Borel $\sigma$-Algebra).** The *Borel $\sigma$-algebra* $\mathcal{B}(\mathbb{R})$ is the smallest $\sigma$-algebra on $\mathbb{R}$ containing all open intervals $(a, b)$. It also contains all closed sets, all countable sets, and virtually every subset of $\mathbb{R}$ encountered in practice.

*ML connection:* When we write $P(X \leq x)$ for a continuous random variable, we are computing $P(\{\omega : X(\omega) \in (-\infty, x]\})$. This requires $(-\infty, x] \in \mathcal{B}(\mathbb{R})$, which the Borel $\sigma$-algebra guarantees. Every CDF, PDF, and distribution used in machine learning relies on this construction.

---

## 3. Kolmogorov Axioms

### 3.1 The Axioms

**Definition 4.1.6 (Probability Measure).** A *probability measure* $P$ on $(\Omega, \mathcal{F})$ is a function $P: \mathcal{F} \to \mathbb{R}$ satisfying:

| Axiom | Statement | Meaning |
|-------|-----------|---------|
| **(K1)** Non-negativity | $P(A) \geq 0$ for all $A \in \mathcal{F}$ | Probabilities are never negative |
| **(K2)** Normalization | $P(\Omega) = 1$ | Something must happen |
| **(K3)** Countable additivity | If $A_1, A_2, \ldots$ are pairwise disjoint, then $P\!\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)$ | Probabilities of mutually exclusive events add |

These three axioms, proposed by Andrey Kolmogorov in 1933, are the foundation of all modern probability theory. Everything else — Bayes' theorem, the law of large numbers, stochastic processes — is derived from (K1)-(K3).

**Example 4.1.3:** *Verify the Kolmogorov axioms for a fair die with $P(\{i\}) = 1/6$.*

- **(K1)** $P(\{i\}) = 1/6 \geq 0$ for every outcome, and $P(A) = |A|/6 \geq 0$ for every event. $\checkmark$
- **(K2)** $P(\Omega) = P(\{1,2,3,4,5,6\}) = 6 \times \tfrac{1}{6} = 1$. $\checkmark$
- **(K3)** Take disjoint events $A = \{1,2\}$ and $B = \{5,6\}$. Then $P(A \cup B) = 4/6$ and $P(A) + P(B) = 2/6 + 2/6 = 4/6$. $\checkmark$

### 3.2 Why Countable Additivity?

Finite additivity ($P(A \cup B) = P(A) + P(B)$ for disjoint $A, B$) is insufficient because it does not allow us to take limits. Countable additivity bridges the gap between finite and infinite, enabling:

- Convergence theorems (law of large numbers)
- Integration theory (expected values)
- Continuity of probability: if $A_n \uparrow A$, then $P(A_n) \to P(A)$

*ML connection:* Training a neural network involves infinite sums and limits — expected loss over a continuous data distribution, convergence of SGD, asymptotic properties of estimators. Countable additivity is what makes all of these mathematically well-defined.

---

## 4. Probability Spaces

### 4.1 The Triple $(\Omega, \mathcal{F}, P)$

**Definition 4.1.7 (Probability Space).** A *probability space* is a triple $(\Omega, \mathcal{F}, P)$ where:

- $\Omega$ is a sample space
- $\mathcal{F}$ is a $\sigma$-algebra on $\Omega$
- $P$ is a probability measure on $(\Omega, \mathcal{F})$

| Component | Role | ML Analogy |
|-----------|------|------------|
| $\Omega$ | All possible outcomes | All possible datasets / inputs |
| $\mathcal{F}$ | Observable events | Questions we can ask about data |
| $P$ | Assigns probabilities | The model's beliefs about data |

**Example (Biased coin).** $\Omega = \{H, T\}$, $\mathcal{F} = \mathcal{P}(\Omega)$, $P(\{H\}) = p$, $P(\{T\}) = 1 - p$.

**Example (Uniform on $[0,1]$).** $\Omega = [0,1]$, $\mathcal{F} = \mathcal{B}([0,1])$, $P([a,b]) = b - a$ for $0 \leq a \leq b \leq 1$.

*ML connection:* A probabilistic model defines a probability space. A generative model like a VAE or diffusion model specifies $P$ over the space of images $\Omega = [0,1]^{n}$. Training adjusts $P$ (parameterized by $\theta$) to match the empirical distribution of the training data.

---

## 5. Basic Probability Rules

All of the following are derived from the Kolmogorov axioms.

### 5.1 Complement Rule

**Theorem 4.1.1 (Complement Rule).**

$$P(A^c) = 1 - P(A)$$

*Proof.* $\Omega = A \cup A^c$ and $A \cap A^c = \emptyset$. By (K3): $P(\Omega) = P(A) + P(A^c)$. By (K2): $P(\Omega) = 1$. Therefore $P(A^c) = 1 - P(A)$. $\square$

**Example 4.1.4:** *Roll two fair dice. What is the probability of getting at least one six?*

Let $A$ = "at least one six." The complement $A^c$ = "no sixes at all."

$$P(A^c) = \frac{5}{6} \times \frac{5}{6} = \frac{25}{36}$$

$$P(A) = 1 - P(A^c) = 1 - \frac{25}{36} = \frac{11}{36} \approx 0.306$$

Direct enumeration confirms: out of 36 equally likely pairs, exactly 11 contain at least one six.

### 5.2 Monotonicity

**Theorem 4.1.2 (Monotonicity).** If $A \subseteq B$, then $P(A) \leq P(B)$.

*Proof.* $B = A \cup (B \setminus A)$ where $A \cap (B \setminus A) = \emptyset$. So $P(B) = P(A) + P(B \setminus A) \geq P(A)$ by (K1). $\square$

**Example 4.1.5:** *Fair die. Let $A = \{6\}$ and $B = \{4, 5, 6\}$. Since $A \subseteq B$, monotonicity gives $P(A) \leq P(B)$.*

$$P(A) = \frac{1}{6} \leq \frac{3}{6} = P(B) \quad \checkmark$$

### 5.3 Inclusion-Exclusion

**Theorem 4.1.3 (Inclusion-Exclusion for Two Events).**

$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

*Proof.* Write $A \cup B = A \cup (B \setminus A)$, a disjoint union. Then $P(A \cup B) = P(A) + P(B \setminus A)$. Now $B = (A \cap B) \cup (B \setminus A)$, also disjoint, so $P(B) = P(A \cap B) + P(B \setminus A)$, giving $P(B \setminus A) = P(B) - P(A \cap B)$. Substituting yields the result. $\square$

**Example 4.1.6:** *Draw one card from a standard 52-card deck. Let $A$ = "heart" (13 cards) and $B$ = "face card" (12 cards: J, Q, K in each suit). What is $P(A \cup B)$?*

$$A \cap B = \{\text{J♥, Q♥, K♥}\}, \quad P(A \cap B) = \frac{3}{52}$$

$$P(A \cup B) = P(A) + P(B) - P(A \cap B) = \frac{13}{52} + \frac{12}{52} - \frac{3}{52} = \frac{22}{52} = \frac{11}{26} \approx 0.423$$

**Theorem 4.1.4 (General Inclusion-Exclusion).** For events $A_1, \ldots, A_n$:

$$P\!\left(\bigcup_{i=1}^n A_i\right) = \sum_{i} P(A_i) - \sum_{i < j} P(A_i \cap A_j) + \sum_{i < j < k} P(A_i \cap A_j \cap A_k) - \cdots + (-1)^{n+1} P(A_1 \cap \cdots \cap A_n)$$

### 5.4 Boole's Inequality (Union Bound)

**Theorem 4.1.5 (Union Bound).**

$$P\!\left(\bigcup_{i=1}^n A_i\right) \leq \sum_{i=1}^n P(A_i)$$

**Example 4.1.7:** *A system has 3 components. The probability each fails in a year is $P(A_1) = 0.05$, $P(A_2) = 0.03$, $P(A_3) = 0.02$. Bound $P(\text{any failure})$.*

$$P(A_1 \cup A_2 \cup A_3) \leq 0.05 + 0.03 + 0.02 = 0.10$$

The true probability is at most 10%. The bound is tight when failures are mutually exclusive; if failures overlap (e.g., correlated), the true probability is strictly less than 0.10.

*ML connection:* The union bound is used extensively in learning theory. If each hypothesis $h_i$ has probability at most $\delta$ of failing on unseen data, then the probability that *any* hypothesis in a class of $n$ hypotheses fails is at most $n\delta$. This is the starting point of PAC learning and uniform convergence bounds.

### 5.5 Independence (Preview)

Two events $A$ and $B$ are *independent* if $P(A \cap B) = P(A) \cdot P(B)$.

**Example 4.1.11:** *Roll a fair die. Let $A = \{2, 4, 6\}$ (even) and $B = \{1, 2, 3\}$ (at most 3). Are $A$ and $B$ independent?*

$$P(A) = \tfrac{3}{6} = \tfrac{1}{2}, \quad P(B) = \tfrac{3}{6} = \tfrac{1}{2}, \quad P(A) \cdot P(B) = \tfrac{1}{4}$$

$$A \cap B = \{2\}, \quad P(A \cap B) = \tfrac{1}{6}$$

Since $\tfrac{1}{6} \neq \tfrac{1}{4}$, the events are **not independent**. Knowing the die is $\leq 3$ changes the probability of "even" from $1/2$ to $1/3$.

**Example 4.1.12:** *Now let $A = \{2, 4, 6\}$ (even) and $C = \{1, 2, 3, 4\}$ (at most 4). Are $A$ and $C$ independent?*

$$P(A) = \tfrac{1}{2}, \quad P(C) = \tfrac{4}{6} = \tfrac{2}{3}, \quad P(A) \cdot P(C) = \tfrac{1}{3}$$

$$A \cap C = \{2, 4\}, \quad P(A \cap C) = \tfrac{2}{6} = \tfrac{1}{3}$$

Since $\tfrac{1}{3} = \tfrac{1}{3}$, the events **are independent**. Knowing the die is $\leq 4$ does not change the probability of "even" (it remains $1/2$: two out of $\{1,2,3,4\}$ are even).

---

## 6. Counting Methods

When $\Omega$ is finite and all outcomes are equally likely, $P(A) = |A|/|\Omega|$. Counting the sizes of sets is therefore essential.

### 6.1 Fundamental Counting Principle

If experiment 1 has $n_1$ outcomes and experiment 2 has $n_2$ outcomes, the combined experiment has $n_1 \times n_2$ outcomes.

### 6.2 Permutations

**Definition 4.1.8 (Permutation).** A *permutation* of $k$ items chosen from $n$ is an ordered arrangement:

$$P(n, k) = \frac{n!}{(n-k)!}$$

**Example 4.1.8:** *How many ways can a president and vice-president be chosen from 5 candidates?*

Order matters (president $\neq$ vice-president role), so this is a permutation:

$$P(5, 2) = \frac{5!}{(5-2)!} = \frac{120}{6} = 20$$

### 6.3 Combinations

**Definition 4.1.9 (Combination).** A *combination* of $k$ items chosen from $n$ is an unordered selection:

$$\binom{n}{k} = \frac{n!}{k!(n-k)!}$$

**Example 4.1.9:** *Compute $\binom{5}{2}$ and use it: from 5 marbles (3 red, 2 blue), what is the probability of drawing 2 red marbles?*

$$\binom{5}{2} = \frac{5!}{2! \cdot 3!} = \frac{120}{2 \cdot 6} = 10$$

Note $P(5,2) = 20$ but $\binom{5}{2} = 10$ — removing order halves the count (divides by $2!$).

$$P(\text{2 red}) = \frac{\binom{3}{2}}{\binom{5}{2}} = \frac{3}{10} = 0.3$$

**Key identity (Pascal's Rule):**

$$\binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k}$$

### 6.4 Multinomial Coefficient

**Definition 4.1.10 (Multinomial Coefficient).** The number of ways to partition $n$ items into groups of sizes $k_1, k_2, \ldots, k_r$ (where $\sum k_i = n$):

$$\binom{n}{k_1, k_2, \ldots, k_r} = \frac{n!}{k_1! k_2! \cdots k_r!}$$

**Example 4.1.10:** *How many distinct arrangements of the letters in "MISSISSIPPI" exist?*

There are 11 letters: M(1), I(4), S(4), P(2). Using the multinomial coefficient:

$$\binom{11}{1, 4, 4, 2} = \frac{11!}{1! \cdot 4! \cdot 4! \cdot 2!} = \frac{39916800}{1 \cdot 24 \cdot 24 \cdot 2} = 34650$$

| Counting Method | Formula | ML Context |
|----------------|---------|------------|
| Permutations | $n!/(n-k)!$ | Ranking / sequence prediction |
| Combinations | $\binom{n}{k}$ | Feature subset selection |
| Multinomial | $n! / \prod k_i!$ | Multiclass confusion matrix configurations |
| Stars and bars | $\binom{n+k-1}{k-1}$ | Distributing samples across classes |

*ML connection:* Feature selection from $n$ features means choosing a subset of size $k$, giving $\binom{n}{k}$ possibilities. For $n = 100$ and $k = 10$, there are $\binom{100}{10} \approx 1.7 \times 10^{13}$ subsets — far too many for brute-force search, motivating greedy and regularization-based approaches.

---

## 7. Frequentist vs. Bayesian Interpretation

The Kolmogorov axioms define how probabilities *behave* (the mathematics), but not what they *mean* (the interpretation). Two major schools of thought coexist:

| Aspect | Frequentist | Bayesian |
|--------|-------------|----------|
| **$P(A)$ means** | Long-run relative frequency of $A$ | Degree of belief that $A$ is true |
| **Parameters** | Fixed but unknown constants | Random variables with distributions |
| **Inference** | $P(\text{data} \mid \theta)$ — likelihood | $P(\theta \mid \text{data})$ — posterior |
| **Example** | "95% CI means: in repeated experiments, 95% of intervals contain $\theta$" | "There is 95% probability $\theta$ lies in this interval" |
| **ML methods** | MLE, hypothesis testing, cross-validation | MAP, MCMC, variational inference |

**Frequentist view:** $P(A)$ is the limit of $n_A / n$ as the number of trials $n \to \infty$. Probabilities are objective properties of repeatable experiments. Parameters $\theta$ are fixed; only data is random.

**Bayesian view:** $P(A)$ quantifies rational belief about $A$ given available evidence. We start with a *prior* $P(\theta)$, observe data $D$, and update to the *posterior*:

$$P(\theta \mid D) = \frac{P(D \mid \theta) \, P(\theta)}{P(D)}$$

*ML connection:* Most deep learning uses the frequentist approach (MLE via SGD). Bayesian methods appear in Gaussian processes, Bayesian neural networks, and uncertainty quantification. Modern approaches like MC Dropout bridge the two: training is frequentist, but inference approximates a Bayesian posterior.

---

## 8. Applications in ML

### 8.1 Classification as Probability

A classifier learns $P(\text{class} \mid \text{features})$. The softmax output of a neural network:

$$P(y = k \mid \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$

defines a valid probability distribution over $K$ classes (satisfies non-negativity and normalization).

### 8.2 Loss Functions as Negative Log-Probability

Cross-entropy loss for classification:

$$\mathcal{L} = -\sum_{k=1}^K y_k \log P(y = k \mid \mathbf{x})$$

This is the *negative log-likelihood*. Minimizing cross-entropy is equivalent to maximizing the probability the model assigns to the correct class — a direct consequence of probability axioms.

### 8.3 Probabilistic Graphical Models

A graphical model factorizes a joint probability distribution using the structure of a graph:

$$P(X_1, X_2, \ldots, X_n) = \prod_{i=1}^n P(X_i \mid \text{parents}(X_i))$$

Each factor satisfies the Kolmogorov axioms. The graph structure encodes conditional independence assumptions, reducing the number of parameters from exponential to manageable.

### 8.4 Training as Likelihood Maximization

Given data $D = \{x_1, \ldots, x_N\}$ drawn i.i.d. from unknown $P_{\text{true}}$, we find parameters $\theta$ that maximize:

$$P(D \mid \theta) = \prod_{i=1}^N P(x_i \mid \theta)$$

This product is valid because (K3) and independence allow us to multiply probabilities of disjoint outcomes. Taking the log converts the product to a sum — the foundation of gradient-based optimization in ML.

---

## Exercises

**★ Basic**

1. A standard deck has 52 cards. What is the probability of drawing a heart or a face card? Use inclusion-exclusion.

2. Let $\Omega = \{1, 2, 3, 4, 5, 6\}$ (fair die). Verify the Kolmogorov axioms for $P(\{i\}) = 1/6$. Compute $P(\text{even})$ and $P(\text{even}^c)$.

3. List all $\sigma$-algebras on $\Omega = \{a, b, c\}$. How many are there?

**★★ Intermediate**

4. Prove that $P(\emptyset) = 0$ directly from the Kolmogorov axioms (hint: write $\Omega = \Omega \cup \emptyset \cup \emptyset \cup \cdots$).

5. A committee of 5 is chosen from 8 men and 6 women. What is the probability that the committee has at least 2 women? Express your answer using combinatorics and simplify.

6. A neural network outputs softmax probabilities $(0.7, 0.2, 0.1)$ for three classes. Compute the cross-entropy loss when the true class is class 1. Then compute it when the true class is class 3. Why is the second loss much higher?

**★★★ Challenging**

7. Prove the continuity of probability from below: if $A_1 \subseteq A_2 \subseteq \cdots$ and $A = \bigcup_{i=1}^{\infty} A_i$, then $P(A_n) \to P(A)$ as $n \to \infty$. (Hint: write $A$ as a disjoint union of the "increments" $B_1 = A_1$, $B_n = A_n \setminus A_{n-1}$ for $n \geq 2$, and apply countable additivity.)

8. The *Borel-Cantelli lemma* states: if $\sum_{n=1}^{\infty} P(A_n) < \infty$, then $P(\limsup_{n} A_n) = 0$, where $\limsup_n A_n = \bigcap_{n=1}^{\infty} \bigcup_{k=n}^{\infty} A_k$. Prove this using continuity from above and the union bound. Then explain its relevance to convergence guarantees in online learning.

---

## Related Topics

- [Sets & Logic](../foundations/sets-and-logic.md) — set operations underlying all probability
- [Conditional Probability](conditional-probability.md) — Bayes' theorem, independence, chain rule
- [Random Variables](random-variables.md) — functions on probability spaces
- [Distributions](distributions.md) — Gaussian, Bernoulli, and other named distributions
- [Maximum Likelihood](maximum-likelihood.md) — parameter estimation from probability axioms

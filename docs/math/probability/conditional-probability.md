# Conditional Probability

Conditioning is how we update beliefs with evidence. When a medical test returns positive, we do not ask "what is the probability of disease?" but rather "what is the probability of disease *given a positive test*?" This shift from $P(A)$ to $P(A \mid B)$ is the heart of statistical reasoning, the engine of Bayesian machine learning, and the mathematical foundation for every classifier that outputs $P(\text{class} \mid \text{features})$. Bayes' theorem, derived in this chapter, is arguably the single most important equation in machine learning.

---

## Prerequisites

- [Probability Foundations](probability-foundations.md) -- sample spaces, axioms, $\sigma$-algebras

---

## 1. Conditional Probability

**Definition 4.2.1 (Conditional Probability).** Let $(\Omega, \mathcal{F}, P)$ be a probability space and $B \in \mathcal{F}$ with $P(B) > 0$. The *conditional probability* of $A$ given $B$ is:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

**Intuition:** We restrict the universe from $\Omega$ to $B$, then ask how much of $B$ is also in $A$.

```
FULL SAMPLE SPACE Ω               CONDITIONING ON B
┌────────────────────────┐         ┌────────────────────────┐
│          Ω             │         │          Ω (faded)     │
│   ┌─────────┐          │         │   ┌─────────┐          │
│   │    A    ┌┼────┐    │         │   │ (faded) ┌┼════╗    │
│   │         │█████│ B  │   ──►   │   │         │█████║ B  │
│   │         ││████│    │         │   │         │█████║    │
│   └─────────┘┼────┘    │         │   └─────────┘╚════╝    │
│              │         │         │   New universe = B      │
└────────────────────────┘         └────────────────────────┘

P(A) = |A|/|Ω|                    P(A|B) = |A∩B| / |B|
```

**Example 4.2.1:** Roll two fair dice. Let $A$ = "sum is 8" and $B$ = "first die shows 3." What is $P(A \mid B)$?

- The sample space has $|\Omega| = 36$ equally likely outcomes.
- $B = \{(3,1),(3,2),(3,3),(3,4),(3,5),(3,6)\}$, so $P(B) = 6/36 = 1/6$.
- $A \cap B = \{(3,5)\}$ (the only way to get sum 8 with first die 3), so $P(A \cap B) = 1/36$.
- $P(A \mid B) = \dfrac{P(A \cap B)}{P(B)} = \dfrac{1/36}{1/6} = \dfrac{1}{6}$.

Intuitively: given the first die is 3, the second die must be 5 for the sum to be 8 -- one outcome out of six.

**Theorem 4.2.1.** For fixed $B$ with $P(B) > 0$, the function $Q(A) = P(A \mid B)$ is itself a valid probability measure on $(\Omega, \mathcal{F})$.

*Proof.* We verify Kolmogorov's axioms:

1. $Q(A) = P(A \cap B) / P(B) \geq 0$ since $P(A \cap B) \geq 0$.
2. $Q(\Omega) = P(\Omega \cap B) / P(B) = P(B) / P(B) = 1$.
3. For disjoint $A_1, A_2, \ldots$: $Q(\bigcup_i A_i) = P((\bigcup_i A_i) \cap B) / P(B) = P(\bigcup_i (A_i \cap B)) / P(B) = \sum_i P(A_i \cap B) / P(B) = \sum_i Q(A_i)$.

Thus $Q$ satisfies all three axioms. $\square$

*ML connection:* This theorem is why conditioning is so powerful -- after observing evidence $B$, we obtain a complete new probability distribution. Every Bayesian update produces a valid probability measure over hypotheses.

---

## 2. Multiplication Rule and Chain Rule

**Theorem 4.2.2 (Multiplication Rule).** For events $A, B$ with $P(B) > 0$:

$$P(A \cap B) = P(A \mid B)\, P(B)$$

By symmetry (when $P(A) > 0$): $P(A \cap B) = P(B \mid A)\, P(A)$.

**Example 4.2.2:** A standard deck has 52 cards. You draw one card. Let $A$ = "card is a King" and $B$ = "card is a face card" (J, Q, K). Find $P(A \cap B)$ using the multiplication rule.

- There are 12 face cards, so $P(B) = 12/52 = 3/13$.
- Every King is a face card, so $P(A \mid B) = 4/12 = 1/3$.
- $P(A \cap B) = P(A \mid B) \cdot P(B) = \dfrac{1}{3} \cdot \dfrac{3}{13} = \dfrac{1}{13} = \dfrac{4}{52}$.

This confirms $A \cap B = A$ (all Kings are face cards), and indeed $P(A) = 4/52 = 1/13$.

**Theorem 4.2.3 (Chain Rule of Probability).** For events $A_1, A_2, \ldots, A_n$ with $P(A_1 \cap \cdots \cap A_{n-1}) > 0$:

$$P(A_1 \cap A_2 \cap \cdots \cap A_n) = P(A_1)\, P(A_2 \mid A_1)\, P(A_3 \mid A_1 \cap A_2) \cdots P(A_n \mid A_1 \cap \cdots \cap A_{n-1})$$

*Proof.* By induction. Base case: the multiplication rule. Inductive step: assume the result holds for $n-1$ events. Then:

$$P(A_1 \cap \cdots \cap A_n) = P(A_n \mid A_1 \cap \cdots \cap A_{n-1})\, P(A_1 \cap \cdots \cap A_{n-1})$$

Applying the inductive hypothesis to $P(A_1 \cap \cdots \cap A_{n-1})$ yields the desired product. $\square$

**Example 4.2.3:** Draw 3 cards without replacement from a standard 52-card deck. Let $A$ = "1st card is an Ace," $B$ = "2nd card is an Ace," $C$ = "3rd card is an Ace." Compute $P(A \cap B \cap C)$.

- $P(A) = 4/52$.
- $P(B \mid A) = 3/51$ (3 Aces remain among 51 cards).
- $P(C \mid A \cap B) = 2/50$ (2 Aces remain among 50 cards).
- By the chain rule: $P(A \cap B \cap C) = \dfrac{4}{52} \cdot \dfrac{3}{51} \cdot \dfrac{2}{50} = \dfrac{24}{132600} = \dfrac{1}{5525} \approx 0.000181$.

The probability of drawing three Aces in a row is roughly 1 in 5,525.

*ML connection:* Autoregressive language models factor the joint probability of a sentence exactly via the chain rule:

$$P(w_1, w_2, \ldots, w_T) = P(w_1)\, P(w_2 \mid w_1)\, P(w_3 \mid w_1, w_2) \cdots P(w_T \mid w_1, \ldots, w_{T-1})$$

GPT generates text by sampling each token from $P(w_t \mid w_1, \ldots, w_{t-1})$.

---

## 3. Law of Total Probability

**Theorem 4.2.4 (Law of Total Probability).** Let $B_1, B_2, \ldots, B_n$ be a partition of $\Omega$ (mutually exclusive, exhaustive) with $P(B_i) > 0$ for all $i$. Then for any event $A$:

$$P(A) = \sum_{i=1}^n P(A \mid B_i)\, P(B_i)$$

*Proof.* Since $\{B_i\}$ partitions $\Omega$: $A = A \cap \Omega = A \cap (\bigcup_i B_i) = \bigcup_i (A \cap B_i)$. These are disjoint, so $P(A) = \sum_i P(A \cap B_i) = \sum_i P(A \mid B_i)\, P(B_i)$. $\square$

```
PARTITION OF Ω INTO B₁, B₂, B₃

┌──────────┬──────────┬──────────┐
│          │          │          │
│    B₁    │    B₂    │    B₃    │
│      ┌───┼──────┐   │          │
│      │ A∩│B₁    │   │          │
│      │   │  A∩B₂│   │          │
│      └───┼──────┘   │          │
│          │          │          │
└──────────┴──────────┴──────────┘

P(A) = P(A|B₁)P(B₁) + P(A|B₂)P(B₂) + P(A|B₃)P(B₃)
```

**Example 4.2.4:** A factory has three machines $M_1, M_2, M_3$ producing 50%, 30%, and 20% of total output respectively. Their defect rates are 2%, 3%, and 5%. What is the probability that a randomly selected item is defective?

- Partition: $P(M_1)=0.50$, $P(M_2)=0.30$, $P(M_3)=0.20$.
- Likelihoods: $P(D \mid M_1)=0.02$, $P(D \mid M_2)=0.03$, $P(D \mid M_3)=0.05$.
- By the law of total probability:

$$P(D) = (0.02)(0.50) + (0.03)(0.30) + (0.05)(0.20) = 0.010 + 0.009 + 0.010 = 0.029$$

So 2.9% of all items are defective. Note that machine $M_3$ contributes disproportionately: it makes only 20% of output but accounts for $0.010/0.029 \approx 34\%$ of defects.

*ML connection:* Marginalizing over latent variables uses the law of total probability. In a Gaussian Mixture Model with $K$ clusters: $P(\mathbf{x}) = \sum_{k=1}^K P(\mathbf{x} \mid z=k)\, P(z=k)$, where $z$ is the latent cluster assignment.

---

## 4. Bayes' Theorem

**Theorem 4.2.5 (Bayes' Theorem).** For events $A, B$ with $P(B) > 0$:

$$P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}$$

*Proof.* From the multiplication rule: $P(A \cap B) = P(B \mid A)\,P(A) = P(A \mid B)\,P(B)$. Dividing both sides by $P(B)$ yields the result. $\square$

**Example 4.2.5 (Medical Testing):** A rare disease affects 1% of the population. A diagnostic test has sensitivity (true positive rate) 99% and specificity (true negative rate) 95%. A patient tests positive. What is $P(\text{disease} \mid +)$?

- Let $D$ = has disease, $+$ = positive test.
- Prior: $P(D) = 0.01$, so $P(D^c) = 0.99$.
- Likelihood: $P(+ \mid D) = 0.99$ (sensitivity), $P(+ \mid D^c) = 0.05$ (false positive rate $= 1 - 0.95$).
- Total probability of a positive test: $P(+) = P(+ \mid D)\,P(D) + P(+ \mid D^c)\,P(D^c) = (0.99)(0.01) + (0.05)(0.99) = 0.0099 + 0.0495 = 0.0594$.
- Bayes' theorem: $P(D \mid +) = \dfrac{(0.99)(0.01)}{0.0594} = \dfrac{0.0099}{0.0594} \approx 0.167$.

Despite a 99% sensitive test, a positive result means only a 16.7% chance of disease. Why? The low prevalence (1%) means that false positives from the 99% healthy population vastly outnumber true positives: for every 10,000 people, roughly 100 are sick (99 test positive) while 9,900 are healthy (495 test positive). Only 99 out of $99+495 = 594$ positive results are truly sick.

**Theorem 4.2.6 (Bayes' Theorem, Extended Form).** Let $\{H_1, \ldots, H_n\}$ partition $\Omega$. Given evidence $E$:

$$P(H_i \mid E) = \frac{P(E \mid H_i)\, P(H_i)}{\sum_{j=1}^n P(E \mid H_j)\, P(H_j)}$$

**Example 4.2.6 (Prosecutor's Fallacy):** A DNA profile found at a crime scene matches the defendant. The match probability is 1 in 1,000,000. A prosecutor argues: "There is only a 1 in a million chance the defendant is innocent." This is a fallacy -- it confuses $P(\text{evidence} \mid \text{innocent})$ with $P(\text{innocent} \mid \text{evidence})$.

- Let $G$ = defendant is guilty, $E$ = DNA match. Suppose the city has 5,000,000 adults.
- Prior (random citizen): $P(G) = 1/5{,}000{,}000$, $P(G^c) = 4{,}999{,}999/5{,}000{,}000$.
- $P(E \mid G) = 1$ (guilty person's DNA matches), $P(E \mid G^c) = 1/1{,}000{,}000$.
- By Bayes' extended form:

$$P(G \mid E) = \frac{1 \cdot \frac{1}{5{,}000{,}000}}{1 \cdot \frac{1}{5{,}000{,}000} + \frac{1}{1{,}000{,}000} \cdot \frac{4{,}999{,}999}{5{,}000{,}000}} = \frac{1}{1 + 4.999999} \approx \frac{1}{6} \approx 0.167$$

Far from "1 in a million chance of innocence," the posterior probability of guilt is only about 16.7%. About 5 people in the city would match, and the defendant is just one of them. The fallacy lies in ignoring the prior -- the base rate of guilt before the evidence.

### 4.1 The Bayesian Vocabulary

| Term | Symbol | Meaning |
|------|--------|---------|
| **Prior** | $P(H)$ | Belief about $H$ *before* seeing evidence |
| **Likelihood** | $P(E \mid H)$ | How probable the evidence is *if* $H$ is true |
| **Posterior** | $P(H \mid E)$ | Updated belief about $H$ *after* seeing evidence |
| **Evidence** | $P(E)$ | Total probability of the evidence (normalizer) |

$$\underbrace{P(H \mid E)}_{\text{posterior}} = \frac{\overbrace{P(E \mid H)}^{\text{likelihood}} \cdot \overbrace{P(H)}^{\text{prior}}}{\underbrace{P(E)}_{\text{evidence}}}$$

Or compactly: **posterior $\propto$ likelihood $\times$ prior**.

### 4.2 Prior-to-Posterior Update

```
BAYESIAN UPDATING

Prior P(θ)          Likelihood P(D|θ)        Posterior P(θ|D)

    ╱╲                   ╱╲                     ╱╲
   ╱  ╲                 ╱  ╲                   ╱  ╲
  ╱    ╲          ×    ╱    ╲          =      ╱    ╲
 ╱      ╲            ╱  ╲   ╲              ╱╲╱
╱ (broad) ╲         ╱ (peaked)╲           ╱(narrower)╲
───────────────   ───────────────      ───────────────
       θ                 θ                    θ

     Vague              Data speaks          Compromise:
    belief              strongly             sharper, shifted
                                             toward data
```

Each new observation $D$ tightens the posterior. With enough data, the posterior concentrates around the true parameter regardless of the prior (Bernstein-von Mises theorem).

*ML connection:* Bayes' theorem is the foundation of all Bayesian machine learning. Training a model means computing (or approximating) $P(\theta \mid \mathcal{D}) \propto P(\mathcal{D} \mid \theta)\, P(\theta)$, where $\theta$ are model parameters and $\mathcal{D}$ is the training data. L2 regularization corresponds to a Gaussian prior on weights. L1 regularization corresponds to a Laplace prior.

---

## 5. Independence

**Definition 4.2.2 (Independence).** Events $A$ and $B$ are *independent* (written $A \perp\!\!\!\perp B$) if:

$$P(A \cap B) = P(A)\, P(B)$$

Equivalently (when $P(B) > 0$): $P(A \mid B) = P(A)$. Knowing $B$ occurred tells us nothing about $A$.

**Example 4.2.7:** Roll a fair die. Let $A$ = "result is even" $= \{2,4,6\}$ and $B$ = "result $\geq 4$" $= \{4,5,6\}$.

- $P(A) = 3/6 = 1/2$, $P(B) = 3/6 = 1/2$.
- $A \cap B = \{4,6\}$, so $P(A \cap B) = 2/6 = 1/3$.
- Check: $P(A) \cdot P(B) = 1/2 \cdot 1/2 = 1/4 \neq 1/3 = P(A \cap B)$.
- Therefore $A$ and $B$ are **not** independent. Knowing the die is $\geq 4$ raises the probability of "even" from $1/2$ to $P(A \mid B) = \frac{2/6}{3/6} = 2/3$.

Now contrast with a pair that *is* independent: let $A$ = "result is even" $= \{2,4,6\}$ and $D$ = "result $\leq 4$" $= \{1,2,3,4\}$. Then $P(A) = 1/2$, $P(D) = 4/6 = 2/3$, and $A \cap D = \{2,4\}$, so $P(A \cap D) = 2/6 = 1/3 = (1/2)(2/3) = P(A) \cdot P(D)$. So $A$ and $D$ **are** independent.

**Definition 4.2.3 (Pairwise vs. Mutual Independence).** Events $A_1, \ldots, A_n$ are:

- *Pairwise independent* if $P(A_i \cap A_j) = P(A_i)\,P(A_j)$ for all $i \neq j$.
- *Mutually independent* if for every subset $S \subseteq \{1, \ldots, n\}$:

$$P\!\left(\bigcap_{i \in S} A_i\right) = \prod_{i \in S} P(A_i)$$

**Theorem 4.2.7.** Mutual independence implies pairwise independence. The converse is *false*.

*Counterexample.* Toss two fair coins. Let $A$ = "coin 1 is heads," $B$ = "coin 2 is heads," $C$ = "both coins show the same face." Then $A, B, C$ are pairwise independent but not mutually independent: $P(A \cap B \cap C) = 1/4 \neq 1/8 = P(A)\,P(B)\,P(C)$.

---

## 6. Conditional Independence

**Definition 4.2.4 (Conditional Independence).** $A$ and $B$ are *conditionally independent* given $C$ (written $A \perp\!\!\!\perp B \mid C$) if:

$$P(A \cap B \mid C) = P(A \mid C)\, P(B \mid C)$$

Equivalently: $P(A \mid B, C) = P(A \mid C)$. Once we know $C$, learning $B$ gives no additional information about $A$.

**Theorem 4.2.8.** Independence and conditional independence are distinct concepts:

- $A \perp\!\!\!\perp B$ does **not** imply $A \perp\!\!\!\perp B \mid C$.
- $A \perp\!\!\!\perp B \mid C$ does **not** imply $A \perp\!\!\!\perp B$.

*Example (explaining away).* Let $C$ = "alarm sounds," $A$ = "burglary," $B$ = "earthquake." Marginally, $A \perp\!\!\!\perp B$ (burglaries and earthquakes are unrelated). But given the alarm sounded, learning there was an earthquake *decreases* the probability of burglary: $A \not\perp\!\!\!\perp B \mid C$.

**Example 4.2.8 (Independence $\not\Leftrightarrow$ Conditional Independence):** Consider a population where 50% are male ($M$) and 50% female ($M^c$). Let $T$ = "is tall" (top 20% of height) and $A$ = "is an athlete" (10% of population). Suppose being tall and being an athlete are independent overall: $P(T \cap A) = (0.20)(0.10) = 0.02 = P(T)\,P(A)$.

Now condition on gender. Among males: $P(T \mid M) = 0.28$ and $P(A \mid M) = 0.14$, but $P(T \cap A \mid M) = 0.06$. Check: $P(T \mid M) \cdot P(A \mid M) = (0.28)(0.14) = 0.0392 \neq 0.06$. So $T \not\perp\!\!\!\perp A \mid M$ -- among males, tallness and athleticism are positively correlated (tall men are disproportionately athletes), even though they are independent in the full population. This demonstrates that $A \perp\!\!\!\perp B$ does **not** imply $A \perp\!\!\!\perp B \mid C$.

---

## 7. Bayesian Networks

Conditional independence enables compact factorization of joint distributions through directed acyclic graphs (DAGs).

```
BAYESIAN NETWORK (NAIVE BAYES STRUCTURE)

              ┌───┐
              │ C │   C = Class (e.g., spam / not spam)
              └─┬─┘
           ╱    │    ╲
          ╱     │     ╲
      ┌──▼─┐ ┌─▼──┐ ┌─▼──┐
      │ X₁ │ │ X₂ │ │ X₃ │   Xᵢ = Features (words)
      └────┘ └────┘ └────┘

Factorization (naive Bayes assumption):
  P(C, X₁, X₂, X₃) = P(C) · P(X₁|C) · P(X₂|C) · P(X₃|C)

Conditional independence: Xᵢ ⊥⊥ Xⱼ | C  for all i ≠ j


GENERAL BAYESIAN NETWORK

    ┌───┐     ┌───┐
    │ A │     │ B │         Joint factorization:
    └─┬─┘     └─┬─┘
      │    ╲  ╱  │          P(A,B,C,D,E) = P(A) · P(B)
      │     ╲╱   │                        · P(C|A,B)
      │     ╱╲   │                        · P(D|C)
      ▼    ╱  ╲  ▼                        · P(E|C)
    ┌───┐     ┌───┐
    │ C │─────│   │         Each node conditioned only
    └─┬─┘     └───┘         on its parents in the DAG
      │    ╲
      ▼     ▼
    ┌───┐ ┌───┐
    │ D │ │ E │
    └───┘ └───┘
```

The key insight: a Bayesian network with $n$ variables factorizes the joint as $P(X_1, \ldots, X_n) = \prod_{i=1}^n P(X_i \mid \text{parents}(X_i))$, dramatically reducing the number of parameters.

| Full joint over $n$ binary variables | $2^n - 1$ parameters |
|--------------------------------------|----------------------|
| Bayesian network (sparse DAG) | $O(n \cdot 2^k)$ where $k$ = max parents |

*ML connection:* Bayesian networks are the theoretical framework behind:

- **Naive Bayes classifiers** -- assume features are conditionally independent given the class, reducing parameters from exponential to linear in the number of features.
- **Hidden Markov Models** -- the hidden state at time $t$ makes past and future observations conditionally independent (Markov property).
- **Variational autoencoders** -- the generative model is a Bayesian network: $z \to x$, with the encoder approximating $P(z \mid x)$.

---

## 8. Conditional Independence in Markov Chains

**Definition 4.2.5 (Markov Property).** A sequence of random variables $X_1, X_2, \ldots$ satisfies the *Markov property* if:

$$P(X_{t+1} \mid X_1, X_2, \ldots, X_t) = P(X_{t+1} \mid X_t)$$

Equivalently: $X_{t+1} \perp\!\!\!\perp (X_1, \ldots, X_{t-1}) \mid X_t$. The future depends on the past only through the present.

```
MARKOV CHAIN

  X₁ ──► X₂ ──► X₃ ──► X₄ ──► X₅

  Given X₃, the past {X₁,X₂} and future {X₄,X₅} are independent:
    {X₁, X₂} ⊥⊥ {X₄, X₅} | X₃

  Chain rule simplifies:
    P(X₁,...,X₅) = P(X₁) · P(X₂|X₁) · P(X₃|X₂) · P(X₄|X₃) · P(X₅|X₄)
```

**Example 4.2.9:** A simple weather model: each day is Sunny ($S$) or Rainy ($R$). Transition probabilities: $P(S \mid S) = 0.8$, $P(R \mid S) = 0.2$, $P(S \mid R) = 0.4$, $P(R \mid R) = 0.6$. Suppose Monday is Sunny. Compute $P(\text{Mon}=S, \text{Tue}=R, \text{Wed}=S)$.

- By the Markov property and chain rule:

$$P(S, R, S) = P(X_1=S) \cdot P(X_2=R \mid X_1=S) \cdot P(X_3=S \mid X_2=R)$$

$$= 1.0 \times 0.2 \times 0.4 = 0.08$$

Note that $P(X_3=S \mid X_2=R)$ does not depend on $X_1$ -- Wednesday's weather depends only on Tuesday, not Monday. This is the Markov property in action.

*ML connection:* The Markov assumption is foundational in reinforcement learning (MDP: Markov Decision Process), HMMs for sequence labeling, and MCMC sampling methods. Diffusion models also exploit a Markov chain structure: the forward noising process $x_0 \to x_1 \to \cdots \to x_T$ is Markovian.

---

## Exercises

**★ Basic**

1. A bag contains 3 red and 7 blue balls. You draw two balls without replacement. What is $P(\text{2nd is red} \mid \text{1st is red})$?

2. Verify that $P(A \mid B) + P(A^c \mid B) = 1$ directly from Definition 4.2.1.

3. A disease has prevalence 1%. A test has sensitivity (true positive rate) 95% and specificity (true negative rate) 90%. Using Bayes' theorem, compute $P(\text{disease} \mid \text{positive test})$. Explain why the result is surprisingly low.

**★★ Intermediate**

4. Prove: if $A \perp\!\!\!\perp B$, then $A \perp\!\!\!\perp B^c$, $A^c \perp\!\!\!\perp B$, and $A^c \perp\!\!\!\perp B^c$.

5. A spam filter uses Naive Bayes with vocabulary $\{w_1, w_2, w_3\}$. Given $P(\text{spam}) = 0.3$ and the likelihoods:

    | Word | $P(w_i \mid \text{spam})$ | $P(w_i \mid \text{ham})$ |
    |------|--------------------------|--------------------------|
    | $w_1$ | 0.8 | 0.1 |
    | $w_2$ | 0.6 | 0.3 |
    | $w_3$ | 0.2 | 0.7 |

    Classify an email containing $\{w_1, w_2\}$ but not $w_3$.

6. Let $X_1, X_2, X_3$ form a Markov chain. Show that $P(X_1, X_2, X_3) = P(X_1) P(X_2 \mid X_1) P(X_3 \mid X_2)$ follows from the Markov property and the chain rule.

**★★★ Challenging**

7. Prove that for a Bayesian network on $n$ variables with DAG structure $G$, the factorization $P(X_1, \ldots, X_n) = \prod_i P(X_i \mid \text{parents}_G(X_i))$ is consistent with any topological ordering of $G$.

8. (Monty Hall) There are 3 doors: one hides a car, two hide goats. You pick door 1. The host (who knows what is behind each door) opens door 3, revealing a goat. Use Bayes' theorem to show that $P(\text{car behind door 2} \mid \text{host opens door 3}) = 2/3$, and explain why switching is optimal.

9. Show that conditional independence is not transitive: construct events $A, B, C, D$ such that $A \perp\!\!\!\perp B \mid C$ and $B \perp\!\!\!\perp D \mid C$ but $A \not\perp\!\!\!\perp D \mid C$.

---

## Related Topics

- [Probability Foundations](probability-foundations.md) -- sample spaces, axioms, counting
- [Random Variables](random-variables.md) -- extending conditioning to random variables
- [Distributions](distributions.md) -- parameterized families and conjugate priors
- [Bayesian Inference](bayesian-inference.md) -- full treatment of prior-posterior computation
- [Stochastic Processes](stochastic-processes.md) -- Markov chains in depth

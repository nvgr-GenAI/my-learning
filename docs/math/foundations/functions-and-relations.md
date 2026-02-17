# Functions and Relations

A neural network is a function. A loss function is a function. An activation function is a function. The entire enterprise of machine learning is: given data, find a function that maps inputs to outputs. To reason precisely about what neural networks can represent, what they cannot, and how compositions of simple functions yield complex behavior, we need the formal theory of functions and relations.

---

## Prerequisites

- [Sets & Logic](sets-and-logic.md)

---

## 1. Relations

### 1.1 Binary Relations

**Definition 3.1 (Binary Relation).** A *binary relation* $R$ from set $A$ to set $B$ is a subset $R \subseteq A \times B$. We write $aRb$ or $(a,b) \in R$ to mean "$a$ is related to $b$."

**Example:** The "less than" relation on $\mathbb{R}$: $R = \{(a,b) \in \mathbb{R}^2 : a < b\}$.

### 1.2 Properties of Relations on a Set

For a relation $R$ on $A$ (i.e., $R \subseteq A \times A$):

| Property | Definition | Example |
|----------|-----------|---------|
| **Reflexive** | $\forall a \in A,\; aRa$ | $=$ (equality) |
| **Symmetric** | $aRb \Rightarrow bRa$ | "same cluster as" |
| **Antisymmetric** | $aRb \land bRa \Rightarrow a = b$ | $\leq$ |
| **Transitive** | $aRb \land bRc \Rightarrow aRc$ | $<$, $=$ |

**Example 1.2:** Let $A = \{1, 2, 3\}$ and $R = \{(1,1), (2,2), (3,3), (1,2), (2,1)\}$.

- *Reflexive?* Yes: $(1,1), (2,2), (3,3) \in R$.
- *Symmetric?* Yes: $(1,2) \in R$ and $(2,1) \in R$; all pairs are matched.
- *Antisymmetric?* No: $(1,2) \in R$ and $(2,1) \in R$ but $1 \neq 2$.
- *Transitive?* Yes: $(1,2)$ and $(2,1) \in R$, and $(1,1) \in R$. All chains close. $\checkmark$

This relation is reflexive, symmetric, transitive (so it is an equivalence relation) but not antisymmetric.

### 1.3 Equivalence Relations

**Definition 3.2 (Equivalence Relation).** A relation that is reflexive, symmetric, and transitive.

**Example 1.3a:** Let $A = \{1, 2, 3, 4, 5, 6\}$ and define $R$ as "has the same remainder when divided by 3." Check the three properties:

- *Reflexive:* $1 R 1$ since $1 \mod 3 = 1 \mod 3$. Same for all elements. $\checkmark$
- *Symmetric:* $1 R 4$ since both give remainder 1; and $4 R 1$. $\checkmark$
- *Transitive:* $1 R 4$ and $4 R 7$ would give $1 R 7$ (same remainder). $\checkmark$

So $R$ is an equivalence relation.

**Definition 3.3 (Equivalence Class).** For $a \in A$, the equivalence class is $[a] = \{b \in A : aRb\}$.

**Example 1.3b:** Continuing the relation above on $A = \{1, 2, 3, 4, 5, 6\}$ with "same remainder mod 3":

$$[1] = \{1, 4\}, \quad [2] = \{2, 5\}, \quad [3] = \{3, 6\}$$

Note: $[4] = \{1, 4\} = [1]$, confirming overlapping classes are identical. These three classes $\{1,4\}, \{2,5\}, \{3,6\}$ partition $A$.

**Theorem 3.1.** The equivalence classes of an equivalence relation on $A$ form a *partition* of $A$.

*Proof.* Each $a \in A$ belongs to $[a]$ (reflexivity), so classes cover $A$. If $[a] \cap [b] \neq \emptyset$, take $c \in [a] \cap [b]$. Then $aRc$ and $bRc$, so $aRb$ (by symmetry and transitivity), which gives $[a] = [b]$. Hence distinct classes are disjoint. $\square$

*ML connection:* Clustering defines an equivalence relation on data: "$x_i$ is in the same cluster as $x_j$." The clusters are the equivalence classes, forming a partition of the dataset.

### 1.4 Partial Orders

**Definition 3.4 (Partial Order).** A relation that is reflexive, antisymmetric, and transitive. A set with a partial order is a *partially ordered set* (poset).

**Example 1.4:** Let $A = \{1, 2, 3, 6\}$ with the relation "divides" ($a \mid b$). Check:

- *Reflexive:* $1\mid 1$, $2\mid 2$, $3\mid 3$, $6\mid 6$. $\checkmark$
- *Antisymmetric:* $2\mid 6$ but $6 \nmid 2$, so no conflict. If $a\mid b$ and $b\mid a$, then $a = b$. $\checkmark$
- *Transitive:* $2\mid 6$ and $1\mid 2$ gives $1\mid 6$. $\checkmark$

Note: $2$ and $3$ are *incomparable* ($2 \nmid 3$ and $3 \nmid 2$), which is why this is a *partial* (not total) order.

*ML connection:* The subset ordering on feature sets ($A \subseteq B$) is a partial order. Feature selection algorithms search this poset for the "best" feature subset. Topological sort of a DAG (used in computational graphs) is a linearization of a partial order.

---

## 2. Functions

### 2.1 Definition

**Definition 3.5 (Function).** A *function* $f: A \to B$ is a relation $f \subseteq A \times B$ such that for every $a \in A$, there exists exactly one $b \in B$ with $(a, b) \in f$. We write $f(a) = b$.

- $A$ is the **domain**
- $B$ is the **codomain**
- $f(A) = \{f(a) : a \in A\} \subseteq B$ is the **image** (or range)

**Example 2.1:** Let $A = \{1, 2, 3\}$, $B = \mathbb{R}$, and $f(x) = x^2$. Then:

- Domain $= \{1, 2, 3\}$, Codomain $= \mathbb{R}$
- Image $= f(A) = \{1, 4, 9\}$
- Note: the image is much smaller than the codomain

**Definition 3.6 (Graph of a Function).** The graph of $f: A \to B$ is the set $\{(a, f(a)) : a \in A\} \subseteq A \times B$.

### 2.2 Injectivity, Surjectivity, Bijectivity

**Definition 3.7 (Injection / One-to-one).** $f: A \to B$ is *injective* if $f(a_1) = f(a_2) \Rightarrow a_1 = a_2$.

**Example 2.2a:** Let $A = \{1, 2, 3\}$, $B = \{a, b, c, d\}$, and $f(1)=a, f(2)=c, f(3)=d$.
$f$ is injective: each element of $A$ maps to a distinct element of $B$. But $f$ is *not* surjective since $b \in B$ is not hit.

**Definition 3.8 (Surjection / Onto).** $f: A \to B$ is *surjective* if for every $b \in B$, there exists $a \in A$ with $f(a) = b$. Equivalently, $f(A) = B$.

**Example 2.2b:** Let $A = \{1, 2, 3\}$, $B = \{x, y\}$, and $f(1)=x, f(2)=y, f(3)=x$.
$f$ is surjective: every element of $B$ is hit ($x$ by 1 and 3, $y$ by 2). But $f$ is *not* injective since $f(1) = f(3) = x$.

**Definition 3.9 (Bijection).** $f$ is *bijective* if it is both injective and surjective.

**Example 2.2c:** Let $A = \{1, 2, 3\}$, $B = \{a, b, c\}$, and $f(1)=b, f(2)=c, f(3)=a$.
$f$ is bijective: every element of $B$ is hit exactly once. The inverse is $f^{-1}(a)=3, f^{-1}(b)=1, f^{-1}(c)=2$.

```
INJECTION, SURJECTION, BIJECTION

Injection (one-to-one):       Surjection (onto):         Bijection (both):
A       B                     A       B                  A       B
a₁ ───→ b₁                   a₁ ───→ b₁                a₁ ───→ b₁
a₂ ───→ b₂                   a₂ ─┬─→ b₂                a₂ ───→ b₂
         b₃ (not hit)        a₃ ─┘                      a₃ ───→ b₃
No two inputs share           Every output is hit.       Perfect pairing.
an output.                    Some outputs hit twice.    Invertible.
```

**Theorem 3.2.** $f: A \to B$ is bijective if and only if $f$ has an inverse function $f^{-1}: B \to A$.

*ML connections:*

- **Injective:** An encoder that preserves all information about the input (no information loss). Autoencoders with bottleneck are NOT injective — they lose information.
- **Surjective:** A decoder that can generate any output in the target space. If the decoder is not surjective, some outputs are unreachable.
- **Bijective:** Normalizing flows use bijective transformations so that the change-of-variables formula applies: $p_Y(y) = p_X(f^{-1}(y)) \cdot |\det J_{f^{-1}}(y)|$.

### 2.3 Composition

**Definition 3.10 (Composition).** Given $f: A \to B$ and $g: B \to C$, the composition $g \circ f: A \to C$ is defined by $(g \circ f)(a) = g(f(a))$.

**Example 2.3:** Let $f(x) = 2x + 1$ and $g(x) = x^2$. Compute $g \circ f$ and $f \circ g$:

$$\begin{aligned}
(g \circ f)(3) &= g(f(3)) = g(2 \cdot 3 + 1) = g(7) = 49 \\
(f \circ g)(3) &= f(g(3)) = f(3^2) = f(9) = 2 \cdot 9 + 1 = 19
\end{aligned}$$

In general: $(g \circ f)(x) = (2x+1)^2$ while $(f \circ g)(x) = 2x^2 + 1$. Note $g \circ f \neq f \circ g$ — composition is not commutative.

**Theorem 3.3.** Composition preserves injectivity and surjectivity:

- If $f$ and $g$ are injective, then $g \circ f$ is injective.
- If $f$ and $g$ are surjective, then $g \circ f$ is surjective.

*Proof (injectivity).* Suppose $(g \circ f)(a_1) = (g \circ f)(a_2)$. Then $g(f(a_1)) = g(f(a_2))$. Since $g$ is injective, $f(a_1) = f(a_2)$. Since $f$ is injective, $a_1 = a_2$. $\square$

*ML connection:* A deep neural network is a composition of functions:

$$f_{\text{net}} = f_L \circ f_{L-1} \circ \cdots \circ f_2 \circ f_1$$

where $f_l(x) = \sigma(W_l x + b_l)$. The *depth* of the network is the number of compositions. The universal approximation theorem states that with enough width, even a single composition (one hidden layer) can approximate any continuous function on a compact set.

### 2.4 Inverse Functions

**Definition 3.11 (Left Inverse).** $g: B \to A$ is a *left inverse* of $f: A \to B$ if $g \circ f = \text{id}_A$.

**Definition 3.12 (Right Inverse).** $g: B \to A$ is a *right inverse* of $f$ if $f \circ g = \text{id}_B$.

**Theorem 3.4.** $f$ has a left inverse iff $f$ is injective. $f$ has a right inverse iff $f$ is surjective.

**Example 2.4:** Let $f(x) = 2x + 1$. Find $f^{-1}$:

$$y = 2x + 1 \;\Longrightarrow\; x = \frac{y - 1}{2} \;\Longrightarrow\; f^{-1}(y) = \frac{y-1}{2}$$

Verify: $f^{-1}(f(3)) = f^{-1}(7) = \frac{7-1}{2} = 3$ $\checkmark$ and $f(f^{-1}(5)) = f(2) = 2\cdot 2+1 = 5$ $\checkmark$.

*ML connection:* An encoder $f: \mathcal{X} \to \mathcal{Z}$ with a decoder $g: \mathcal{Z} \to \mathcal{X}$ satisfies $g \circ f \approx \text{id}_\mathcal{X}$ (reconstruction). If $f$ is injective, an exact left inverse exists in principle. Autoencoder training minimizes $\|g(f(x)) - x\|^2$, seeking an approximate left inverse.

---

## 3. Important Function Types

### 3.1 Linear Functions

**Definition 3.13.** $f: V \to W$ between vector spaces is *linear* if $f(\alpha u + \beta v) = \alpha f(u) + \beta f(v)$.

**Example 3.1:** Let $f(x) = 3x$ (from $\mathbb{R} \to \mathbb{R}$). Check linearity:

$$f(2 \cdot 4 + 5 \cdot 1) = f(13) = 39 = 2 \cdot 12 + 5 \cdot 3 = 2 \cdot f(4) + 5 \cdot f(1) \;\checkmark$$

Now consider $g(x) = 3x + 1$. Then $g(4 + 1) = g(5) = 16$, but $g(4) + g(1) = 13 + 4 = 17 \neq 16$. So $g$ is *affine*, not linear (the bias term $+1$ breaks linearity).

Every linear map from $\mathbb{R}^n$ to $\mathbb{R}^m$ can be represented as multiplication by an $m \times n$ matrix. See [Linear Transformations](../linear-algebra/linear-transformations.md).

### 3.2 Continuous Functions

**Definition 3.14 ($\varepsilon$-$\delta$ Continuity).** $f: \mathbb{R} \to \mathbb{R}$ is *continuous at $a$* if for every $\varepsilon > 0$, there exists $\delta > 0$ such that $|x - a| < \delta \Rightarrow |f(x) - f(a)| < \varepsilon$.

**Example 3.2:** Show $f(x) = 2x + 1$ is continuous at $a = 3$. Given $\varepsilon > 0$, choose $\delta = \varepsilon / 2$. Then:

$$|x - 3| < \delta \;\Longrightarrow\; |f(x) - f(3)| = |(2x+1) - 7| = 2|x - 3| < 2\delta = \varepsilon \;\;\checkmark$$

### 3.3 Measurable Functions

**Definition 3.15.** $f: (X, \mathcal{F}) \to (Y, \mathcal{G})$ is *measurable* if $f^{-1}(B) \in \mathcal{F}$ for all $B \in \mathcal{G}$.

*ML connection:* Random variables are measurable functions from the sample space to $\mathbb{R}$. The measurability requirement ensures that probabilities of events like $\{X \leq x\}$ are well-defined.

### 3.4 Convex Functions

**Definition 3.16.** $f: \mathbb{R}^n \to \mathbb{R}$ is *convex* if $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$ for all $x, y$ and $\lambda \in [0,1]$.

**Example 3.4:** Let $f(x) = x^2$. Check convexity with $x = 1$, $y = 3$, $\lambda = 0.5$:

$$\begin{aligned}
\text{LHS:} &\quad f(0.5 \cdot 1 + 0.5 \cdot 3) = f(2) = 4 \\
\text{RHS:} &\quad 0.5 \cdot f(1) + 0.5 \cdot f(3) = 0.5 \cdot 1 + 0.5 \cdot 9 = 5
\end{aligned}$$

Since $4 \leq 5$, the inequality holds. Geometrically: the chord from $(1,1)$ to $(3,9)$ lies above the curve at the midpoint.

Convex loss functions guarantee that gradient descent finds the global minimum. See [Convexity](../optimization/convexity.md).

---

## 4. Applications in ML/AI

### 4.1 Activation Functions

Neural network activation functions are specific real-valued functions with properties chosen for gradient flow:

| Function | Formula | Injective? | Surjective onto $\mathbb{R}$? | Differentiable? |
|----------|---------|-----------|-------------------------------|-----------------|
| ReLU | $\max(0, x)$ | No ($f(x) = 0$ for all $x \leq 0$) | No (range $= [0, \infty)$) | No (at $x=0$) |
| Sigmoid | $1/(1+e^{-x})$ | Yes | No (range $= (0,1)$) | Yes |
| Tanh | $(e^x - e^{-x})/(e^x + e^{-x})$ | Yes | No (range $= (-1,1)$) | Yes |
| Softmax | $e^{x_i} / \sum_j e^{x_j}$ | No | No (range $= \Delta^{k-1}$, simplex) | Yes |

**Example 4.1:** Verify ReLU's non-injectivity and sigmoid's injectivity with concrete values:

- *ReLU:* $\text{ReLU}(-3) = 0$ and $\text{ReLU}(-7) = 0$. Two distinct inputs ($-3 \neq -7$) give the same output, so ReLU is not injective. Its image on $\{-2, -1, 0, 1, 2\}$ is $\{0, 0, 0, 1, 2\} = \{0, 1, 2\}$.
- *Sigmoid:* $\sigma(0) = 0.5$, $\sigma(1) \approx 0.731$, $\sigma(-1) \approx 0.269$. All outputs are distinct, consistent with injectivity. Note all outputs lie in $(0, 1)$, so sigmoid is not surjective onto $\mathbb{R}$.

ReLU's non-injectivity (collapsing all negative inputs to zero) causes "dead neurons" — neurons that output zero for all inputs and receive zero gradient.

### 4.2 The Universal Approximation Theorem (Informal)

**Theorem 3.5 (Cybenko, 1989; Hornik, 1991).** Let $\sigma$ be any continuous, non-constant, bounded activation function. Then for any continuous function $f: [0,1]^n \to \mathbb{R}$ and any $\varepsilon > 0$, there exists a single-hidden-layer network $g(x) = \sum_{i=1}^N \alpha_i \sigma(w_i^T x + b_i)$ with $\|f - g\|_\infty < \varepsilon$.

This is an existence theorem about the surjectivity (in an approximate sense) of the neural network function class onto the space of continuous functions.

---

## Exercises

**★ Basic**

1. Let $f(x) = x^2$ on $\mathbb{R}$. Is $f$ injective? Surjective? Restrict the domain to make it bijective.

2. Let $f(x) = e^x$ and $g(x) = \ln(x)$. Verify that $g = f^{-1}$ by computing $g \circ f$ and $f \circ g$.

3. Let $R$ be the relation "trains on the same dataset as" on ML models. Is $R$ an equivalence relation?

**★★ Intermediate**

4. Prove that the composition of two bijections is a bijection.

5. Show that the sigmoid function $\sigma(x) = 1/(1+e^{-x})$ is injective and find its inverse.

6. Define the softmax function $\text{softmax}: \mathbb{R}^k \to \Delta^{k-1}$ (the probability simplex). Show it is not injective by finding two distinct inputs with the same output.

**★★★ Challenging**

7. A normalizing flow uses a bijection $f: \mathbb{R}^n \to \mathbb{R}^n$. Using the change-of-variables formula $p_Y(y) = p_X(f^{-1}(y)) \cdot |\det J_{f^{-1}}(y)|$, show that if $f$ is a composition of bijections $f = f_K \circ \cdots \circ f_1$, then $|\det J_f(x)| = \prod_{k=1}^K |\det J_{f_k}(z_{k-1})|$ where $z_0 = x$ and $z_k = f_k(z_{k-1})$.

8. Prove that the set of bijections $f: A \to A$ forms a group under composition (the symmetric group $S_A$).

---

## Related Topics

- [Sets & Logic](sets-and-logic.md) — the foundation of relations and functions
- [Number Systems](number-systems.md) — the domains and codomains we use
- [Linear Transformations](../linear-algebra/linear-transformations.md) — linear functions between vector spaces
- [Convexity](../optimization/convexity.md) — convex functions and optimization

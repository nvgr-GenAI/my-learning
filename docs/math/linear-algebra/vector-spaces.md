# Vector Spaces

A vector space is the central abstraction of linear algebra. Rather than thinking only of arrows in $\mathbb{R}^2$ or columns of numbers in $\mathbb{R}^n$, we define an algebraic structure — a set equipped with addition and scalar multiplication satisfying eight axioms — and then *prove theorems once* that apply everywhere: to polynomials, to matrices, to functions, to quantum states, and to the latent spaces of variational autoencoders. Every embedding layer in a transformer produces vectors that live in a vector space. Understanding this abstraction is essential for understanding why linear algebra is the universal language of machine learning.

---

## Prerequisites

- [Foundations](../foundations/index.md) — sets, functions, number systems (especially fields)
- [Vectors](vectors.md) — concrete vector operations, norms, inner products
- [Matrices](matrices.md) — matrix arithmetic, row operations
- [Systems of Equations](systems-of-equations.md) — Gaussian elimination, solution sets

---

## 1. Fields (Review)

Before defining vector spaces, we recall the scalar ground:

**Definition 2.4.1 (Field).** A *field* $\mathbb{F}$ is a set with two operations $+$ and $\cdot$ satisfying the usual arithmetic axioms (commutativity, associativity, distributivity, existence of $0, 1$, additive inverses, and multiplicative inverses for every element $\neq 0$).

| Field | Notation | Where It Appears |
|-------|----------|------------------|
| Real numbers | $\mathbb{R}$ | Neural network weights, feature vectors, loss functions |
| Complex numbers | $\mathbb{C}$ | Quantum computing (amplitudes), Fourier transforms |
| Rational numbers | $\mathbb{Q}$ | Number theory, exact arithmetic |
| Finite field $\text{GF}(2)$ | $\mathbb{F}_2 = \{0, 1\}$ | Error-correcting codes, Boolean circuits |

Throughout this chapter, $\mathbb{F}$ denotes an arbitrary field. When we need concreteness we use $\mathbb{F} = \mathbb{R}$.

---

## 2. Definition of a Vector Space

**Definition 2.4.2 (Vector Space).** A *vector space* over a field $\mathbb{F}$ is a set $V$ together with two operations:

- **Vector addition:** $+: V \times V \to V$
- **Scalar multiplication:** $\cdot\,: \mathbb{F} \times V \to V$

satisfying the following eight axioms for all $\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$ and all $\alpha, \beta \in \mathbb{F}$:

| # | Axiom | Statement |
|---|-------|-----------|
| A1 | Commutativity of addition | $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$ |
| A2 | Associativity of addition | $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$ |
| A3 | Additive identity | $\exists\, \mathbf{0} \in V$ s.t. $\mathbf{v} + \mathbf{0} = \mathbf{v}$ |
| A4 | Additive inverse | $\forall\, \mathbf{v} \in V,\; \exists\, (-\mathbf{v})$ s.t. $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$ |
| S1 | Multiplicative identity | $1 \cdot \mathbf{v} = \mathbf{v}$ |
| S2 | Compatibility | $\alpha(\beta\mathbf{v}) = (\alpha\beta)\mathbf{v}$ |
| S3 | Distributivity over vector addition | $\alpha(\mathbf{u} + \mathbf{v}) = \alpha\mathbf{u} + \alpha\mathbf{v}$ |
| S4 | Distributivity over scalar addition | $(\alpha + \beta)\mathbf{v} = \alpha\mathbf{v} + \beta\mathbf{v}$ |

Elements of $V$ are called *vectors*. Elements of $\mathbb{F}$ are called *scalars*.

**Example 2.4.1 (A set that IS a vector space).** Let $V = \{(x, y) : x, y \in \mathbb{R}\} = \mathbb{R}^2$ with standard addition and scalar multiplication. Verify a few axioms:

- **A1 (commutativity):** $(1, 2) + (3, 4) = (4, 6) = (3, 4) + (1, 2)$. $\checkmark$
- **A3 (additive identity):** $\mathbf{0} = (0, 0)$ and $(x, y) + (0, 0) = (x, y)$. $\checkmark$
- **A4 (additive inverse):** For $(3, -5)$, the inverse is $(-3, 5)$ since $(3, -5) + (-3, 5) = (0, 0)$. $\checkmark$
- **S3 (distributivity):** $2 \cdot ((1, 3) + (2, -1)) = 2 \cdot (3, 2) = (6, 4)$, and $2(1, 3) + 2(2, -1) = (2, 6) + (4, -2) = (6, 4)$. $\checkmark$

All eight axioms hold, so $\mathbb{R}^2$ is a vector space over $\mathbb{R}$.

**Example 2.4.2 (A set that is NOT a vector space).** Let $V = \{(x, y) : x, y \in \mathbb{R}\}$ with the *modified* operations: addition is standard, but define scalar multiplication as $\alpha \cdot (x, y) = (\alpha^2 x,\; \alpha^2 y)$. Check axiom S1:

$$1 \cdot (3, 4) = (1^2 \cdot 3,\; 1^2 \cdot 4) = (3, 4) \;\checkmark$$

But check S2 (compatibility): $2(3 \cdot (1, 0)) = 2 \cdot (9, 0) = (4 \cdot 9, 0) = (36, 0)$, while $(2 \cdot 3)(1, 0) = 6 \cdot (1, 0) = (36, 0)$. Looks fine? Try $\alpha = 2, \beta = 1, \mathbf{v} = (1, 0)$: $2(1 \cdot (1,0)) = 2 \cdot (1, 0) = (4, 0)$, but $(2 \cdot 1)(1, 0) = 2 \cdot (1, 0) = (4, 0)$. Now try S4: $(\alpha + \beta)\mathbf{v} = (2 + 3)(1, 0) = 5 \cdot (1, 0) = (25, 0)$, but $2 \cdot (1, 0) + 3 \cdot (1, 0) = (4, 0) + (9, 0) = (13, 0)$. Since $25 \neq 13$, axiom S4 fails. **Not a vector space.**

**Theorem 2.4.1 (Uniqueness of Identity and Inverses).**

(a) The zero vector $\mathbf{0}$ is unique.
(b) For each $\mathbf{v}$, the additive inverse $-\mathbf{v}$ is unique.
(c) $0 \cdot \mathbf{v} = \mathbf{0}$ for all $\mathbf{v} \in V$.
(d) $\alpha \cdot \mathbf{0} = \mathbf{0}$ for all $\alpha \in \mathbb{F}$.
(e) $(-1)\mathbf{v} = -\mathbf{v}$.

*Proof of (a).* Suppose $\mathbf{0}'$ is also an additive identity. Then $\mathbf{0} = \mathbf{0} + \mathbf{0}' = \mathbf{0}'$, using A3 with each identity. $\square$

*Proof of (c).* $0\mathbf{v} = (0+0)\mathbf{v} = 0\mathbf{v} + 0\mathbf{v}$ by S4. Adding $-(0\mathbf{v})$ to both sides gives $\mathbf{0} = 0\mathbf{v}$. $\square$

*Proof of (e).* $\mathbf{v} + (-1)\mathbf{v} = 1\mathbf{v} + (-1)\mathbf{v} = (1 + (-1))\mathbf{v} = 0\mathbf{v} = \mathbf{0}$ by S1, S4, and (c). By uniqueness of inverses, $(-1)\mathbf{v} = -\mathbf{v}$. $\square$

---

## 3. Examples of Vector Spaces

### 3.1 The Coordinate Spaces $\mathbb{F}^n$

The most concrete example. $\mathbb{R}^n$ with component-wise addition and scalar multiplication is a vector space over $\mathbb{R}$.

```
EXAMPLES OF R^n

R¹: ───●─────────────→        (the real line)

R²:    ↑                       (the plane)
       │  ● (2,3)
       │ ╱
       │╱
       ●─────────→

R³:    ↑ z                     (3D space)
       │  ╱ y
       │ ╱
       │╱
       ●─────────→ x
```

*ML connection:* Every dataset is a collection of vectors in $\mathbb{R}^n$. A dataset of $m$ images, each with $n$ pixels, lives in $\mathbb{R}^n$. A batch of $m$ token embeddings of dimension $d$ lives in $\mathbb{R}^d$. The entire parameter vector $\boldsymbol{\theta}$ of a neural network lives in $\mathbb{R}^p$ where $p$ is the parameter count (e.g., $p \approx 175 \times 10^9$ for GPT-3).

### 3.2 Function Spaces

**Example.** Let $C[a,b]$ be the set of all continuous functions $f: [a,b] \to \mathbb{R}$, with:

- $(f + g)(x) = f(x) + g(x)$
- $(\alpha f)(x) = \alpha \cdot f(x)$

This is a vector space over $\mathbb{R}$. The "vectors" are *functions*.

*ML connection:* In kernel methods and Gaussian processes, we work in function spaces. The reproducing kernel Hilbert space (RKHS) $\mathcal{H}_k$ is an infinite-dimensional vector space of functions where the kernel $k(x, x')$ defines the inner product. Neural networks with infinite width converge to Gaussian processes — functions in a specific RKHS.

### 3.3 Polynomial Spaces

**Example.** $\mathcal{P}_n(\mathbb{R})$ = set of all polynomials of degree $\leq n$ with real coefficients:

$$p(x) = a_0 + a_1 x + a_2 x^2 + \cdots + a_n x^n$$

with the usual polynomial addition and scalar multiplication. This is a vector space of dimension $n+1$.

### 3.4 Matrix Spaces

**Example.** $\mathbb{R}^{m \times n}$ = set of all $m \times n$ real matrices, with entry-wise addition and scalar multiplication, is a vector space of dimension $mn$.

### 3.5 Solution Spaces

**Example.** The set of all solutions to a homogeneous system $A\mathbf{x} = \mathbf{0}$ forms a vector space (the *null space* of $A$). The set of solutions to $A\mathbf{x} = \mathbf{b}$ for $\mathbf{b} \neq \mathbf{0}$ does *not* — it fails axiom A3 because $\mathbf{0}$ is not a solution.

### 3.6 Quantum State Space

**Example.** The state space of an $n$-qubit quantum system is $\mathbb{C}^{2^n}$ — a complex vector space of dimension $2^n$. The exponential growth of this dimension is the source of quantum computing's power (and the source of the difficulty of classical simulation).

```
QUANTUM STATE SPACES

1 qubit:   C² (dim 2)       ── simulable on a laptop
2 qubits:  C⁴ (dim 4)       ── trivial
10 qubits: C¹⁰²⁴ (dim 1024) ── still easy
40 qubits: C^(10¹²)          ── pushes classical limits
300 qubits: C^(2³⁰⁰)         ── more dimensions than atoms in universe
```

---

## 4. Subspaces

**Definition 2.4.3 (Subspace).** A subset $W \subseteq V$ is a *subspace* of $V$ if $W$ is itself a vector space under the same operations. We write $W \leq V$.

**Theorem 2.4.2 (Subspace Test).** A nonempty subset $W \subseteq V$ is a subspace if and only if for all $\mathbf{u}, \mathbf{v} \in W$ and all $\alpha \in \mathbb{F}$:

1. $\mathbf{u} + \mathbf{v} \in W$ (closure under addition)
2. $\alpha\mathbf{u} \in W$ (closure under scalar multiplication)

Equivalently (one-step test): $W \neq \emptyset$ and $\alpha\mathbf{u} + \beta\mathbf{v} \in W$ for all $\mathbf{u}, \mathbf{v} \in W$ and $\alpha, \beta \in \mathbb{F}$.

*Proof.* ($\Rightarrow$) If $W$ is a subspace, closure follows from the definition of vector space operations.

($\Leftarrow$) Since $W \neq \emptyset$, pick any $\mathbf{w} \in W$. By (2), $0 \cdot \mathbf{w} = \mathbf{0} \in W$, so A3 holds. By (2), $(-1)\mathbf{w} = -\mathbf{w} \in W$, so A4 holds. The remaining axioms are inherited from $V$ since every element of $W$ is also in $V$. $\square$

**Example 2.4.3 (Subspace test — passes).** Let $W = \{(x, y, z) \in \mathbb{R}^3 : x + 2y - z = 0\}$. Verify $W$ is a subspace of $\mathbb{R}^3$:

1. **Non-empty:** $(0, 0, 0) \in W$ since $0 + 2(0) - 0 = 0$. $\checkmark$
2. **Closure under addition:** Take $\mathbf{u} = (1, 1, 3)$ and $\mathbf{v} = (2, -1, 0)$ (both in $W$: $1+2-3=0$, $2-2-0=0$). Then $\mathbf{u} + \mathbf{v} = (3, 0, 3)$ and $3 + 0 - 3 = 0 \in W$. $\checkmark$
   In general: if $u_1 + 2u_2 - u_3 = 0$ and $v_1 + 2v_2 - v_3 = 0$, then $(u_1+v_1) + 2(u_2+v_2) - (u_3+v_3) = 0$.
3. **Closure under scalar multiplication:** $5\mathbf{u} = (5, 5, 15)$ and $5 + 10 - 15 = 0 \in W$. $\checkmark$

All three conditions hold, so $W \leq \mathbb{R}^3$.

**Example 2.4.4 (Subspace test — fails).** Let $S = \{(x, y) \in \mathbb{R}^2 : xy \geq 0\}$ (vectors where both components have the same sign or are zero). Check closure under addition:

- $(1, 2) \in S$ and $(-3, -1) \in S$ (both products $\geq 0$).
- $(1, 2) + (-3, -1) = (-2, 1)$ and $(-2)(1) = -2 < 0$. So $(-2, 1) \notin S$.

Closure under addition fails. **Not a subspace.**

**Key examples of subspaces:**

| Subspace | Ambient Space | Definition |
|----------|---------------|------------|
| Lines through origin | $\mathbb{R}^2$ | $\{t\mathbf{v} : t \in \mathbb{R}\}$ for fixed $\mathbf{v}$ |
| Planes through origin | $\mathbb{R}^3$ | $\{s\mathbf{u} + t\mathbf{v} : s, t \in \mathbb{R}\}$ |
| Null space $\text{Null}(A)$ | $\mathbb{R}^n$ | $\{\mathbf{x} : A\mathbf{x} = \mathbf{0}\}$ |
| Column space $\text{Col}(A)$ | $\mathbb{R}^m$ | $\{A\mathbf{x} : \mathbf{x} \in \mathbb{R}^n\}$ |
| Row space $\text{Row}(A)$ | $\mathbb{R}^n$ | $\text{Col}(A^T)$ |
| Polynomials of degree $\leq k$ | $\mathcal{P}_n$ | $\mathcal{P}_k$ for $k \leq n$ |

**Non-examples (common mistakes):**

- A line *not* through the origin is NOT a subspace ($\mathbf{0}$ is not in it)
- $\{(x,y) : x \geq 0\}$ is NOT a subspace (not closed under scalar multiplication by $-1$)
- The unit sphere $\{\mathbf{v} : \|\mathbf{v}\| = 1\}$ is NOT a subspace ($\mathbf{0}$ is not on it)

*ML connection:* The **null space** $\text{Null}(A)$ characterizes directions in input space that produce zero output. If a weight matrix $W$ has a nontrivial null space, then there exist input perturbations $\boldsymbol{\delta} \in \text{Null}(W)$ that are invisible to the layer: $W(\mathbf{x} + \boldsymbol{\delta}) = W\mathbf{x}$. This is precisely the information that the layer discards. In autoencoders, the encoder's null space contains the information *not* preserved in the bottleneck.

---

## 5. Span

**Definition 2.4.4 (Span).** Given vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k \in V$, their *span* is:

$$\text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\} = \left\{\sum_{i=1}^k \alpha_i \mathbf{v}_i : \alpha_i \in \mathbb{F}\right\}$$

The span is the smallest subspace containing $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$.

**Theorem 2.4.3.** $\text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ is a subspace of $V$.

*Proof.* Let $\mathbf{u} = \sum \alpha_i \mathbf{v}_i$ and $\mathbf{w} = \sum \beta_i \mathbf{v}_i$ be in the span.

- $\mathbf{u} + \mathbf{w} = \sum (\alpha_i + \beta_i)\mathbf{v}_i \in \text{span}$ (closure under addition)
- $c\mathbf{u} = \sum (c\alpha_i)\mathbf{v}_i \in \text{span}$ (closure under scalar multiplication)
- The span is nonempty since $\mathbf{0} = \sum 0 \cdot \mathbf{v}_i$ is in it.

By the subspace test, $\text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ is a subspace. $\square$

**Example 2.4.5 (Computing a span).** Let $\mathbf{v}_1 = (1, 0, 1)$ and $\mathbf{v}_2 = (0, 1, 1)$ in $\mathbb{R}^3$. Then:

$$\text{span}\{\mathbf{v}_1, \mathbf{v}_2\} = \{\alpha(1,0,1) + \beta(0,1,1) : \alpha, \beta \in \mathbb{R}\} = \{(\alpha,\; \beta,\; \alpha + \beta) : \alpha, \beta \in \mathbb{R}\}$$

This is the plane $z = x + y$ through the origin in $\mathbb{R}^3$. Some sample vectors in this span:

- $\alpha=1, \beta=0$: $(1, 0, 1)$
- $\alpha=0, \beta=2$: $(0, 2, 2)$
- $\alpha=3, \beta=-1$: $(3, -1, 2)$

Is $(2, 1, 4) \in \text{span}\{\mathbf{v}_1, \mathbf{v}_2\}$? We need $\alpha = 2, \beta = 1$, but $\alpha + \beta = 3 \neq 4$. So **no** — $(2, 1, 4)$ is not in the span.

```
VISUALIZING SPAN IN R³

span{v₁}:                     span{v₁, v₂}:
         ↑                              ↑  ╱
         │                              │ ╱  (a plane
─────────●─────────            ─────────●───── through
         │  (a line                     │      the origin)
         ↓  through                     │
            the origin)

span{v₁, v₂, v₃} = R³  (if the three vectors are linearly independent)
```

*ML connection:* An **embedding space** (e.g., the output of an embedding layer) lives in $\mathbb{R}^d$, but the actual data typically occupies a lower-dimensional subspace — the span of the learned embedding vectors. If your vocabulary has 50,000 tokens embedded in $\mathbb{R}^{768}$, the embeddings span at most a 768-dimensional subspace of the ambient space. The **manifold hypothesis** in deep learning states that high-dimensional data lies near a low-dimensional manifold, which locally resembles the span of a few vectors.

---

## 6. Linear Independence

**Definition 2.4.5 (Linear Independence).** Vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k \in V$ are *linearly independent* if the only solution to:

$$\alpha_1\mathbf{v}_1 + \alpha_2\mathbf{v}_2 + \cdots + \alpha_k\mathbf{v}_k = \mathbf{0}$$

is $\alpha_1 = \alpha_2 = \cdots = \alpha_k = 0$.

If a nontrivial solution exists (some $\alpha_i \neq 0$), the vectors are *linearly dependent*.

**Theorem 2.4.4 (Dependence Characterization).** $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ is linearly dependent if and only if at least one $\mathbf{v}_j$ can be written as a linear combination of the others.

*Proof.* ($\Rightarrow$) Suppose $\sum \alpha_i \mathbf{v}_i = \mathbf{0}$ with some $\alpha_j \neq 0$. Then $\mathbf{v}_j = -\frac{1}{\alpha_j}\sum_{i \neq j} \alpha_i \mathbf{v}_i$.

($\Leftarrow$) If $\mathbf{v}_j = \sum_{i \neq j} \beta_i \mathbf{v}_i$, then $\sum_{i \neq j} \beta_i \mathbf{v}_i + (-1)\mathbf{v}_j = \mathbf{0}$, a nontrivial relation. $\square$

**Example 2.4.6 (Linear independence — yes).** Test whether $\mathbf{v}_1 = (1, 0, 2)$, $\mathbf{v}_2 = (0, 1, 1)$, $\mathbf{v}_3 = (0, 0, 1)$ are linearly independent. Solve $\alpha_1(1,0,2) + \alpha_2(0,1,1) + \alpha_3(0,0,1) = (0,0,0)$:

$$\alpha_1 = 0, \quad \alpha_2 = 0, \quad 2\alpha_1 + \alpha_2 + \alpha_3 = 0 \implies \alpha_3 = 0$$

Only the trivial solution exists. **Linearly independent.**

**Example 2.4.7 (Linear dependence — find the relation).** Test $\mathbf{v}_1 = (1, 2, 3)$, $\mathbf{v}_2 = (4, 5, 6)$, $\mathbf{v}_3 = (2, 1, 0)$. Solve $\alpha_1(1,2,3) + \alpha_2(4,5,6) + \alpha_3(2,1,0) = (0,0,0)$:

$$\alpha_1 + 4\alpha_2 + 2\alpha_3 = 0, \quad 2\alpha_1 + 5\alpha_2 + \alpha_3 = 0, \quad 3\alpha_1 + 6\alpha_2 = 0$$

From the third equation: $\alpha_1 = -2\alpha_2$. Substituting into the first: $-2\alpha_2 + 4\alpha_2 + 2\alpha_3 = 0 \implies \alpha_3 = -\alpha_2$. Take $\alpha_2 = 1$: $\alpha_1 = -2,\; \alpha_2 = 1,\; \alpha_3 = -1$, giving:

$$-2\mathbf{v}_1 + \mathbf{v}_2 - \mathbf{v}_3 = \mathbf{0}, \quad\text{i.e., } \mathbf{v}_2 = 2\mathbf{v}_1 + \mathbf{v}_3$$

**Linearly dependent** — $\mathbf{v}_2$ is a combination of the other two.

**Theorem 2.4.5 (Steinitz Exchange Lemma).** If $\{\mathbf{u}_1, \ldots, \mathbf{u}_m\}$ is linearly independent and $\text{span}\{\mathbf{u}_1, \ldots, \mathbf{u}_m\} \subseteq \text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$, then $m \leq k$.

This fundamental lemma implies that any two bases of a finite-dimensional vector space have the same number of elements.

*ML connection:* **Multicollinearity** in regression is precisely linear dependence among feature vectors. If one feature is a linear combination of others, the design matrix $X$ is rank-deficient and ordinary least squares has infinitely many solutions. Regularization (L1/L2) resolves this by breaking the degeneracy, and PCA explicitly identifies and removes dependent directions.

---

## 7. Basis and Dimension

**Definition 2.4.6 (Basis).** A set $\mathcal{B} = \{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$ is a *basis* for $V$ if:

1. $\mathcal{B}$ is linearly independent
2. $\text{span}(\mathcal{B}) = V$

Equivalently, $\mathcal{B}$ is a basis if every $\mathbf{v} \in V$ can be written *uniquely* as $\mathbf{v} = \sum \alpha_i \mathbf{v}_i$.

**Theorem 2.4.6 (Uniqueness of Representation).** If $\mathcal{B} = \{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$ is a basis for $V$, then every $\mathbf{v} \in V$ has a *unique* representation as a linear combination of the $\mathbf{v}_i$.

*Proof.* Existence follows from $\text{span}(\mathcal{B}) = V$. For uniqueness, suppose $\mathbf{v} = \sum \alpha_i \mathbf{v}_i = \sum \beta_i \mathbf{v}_i$. Then $\sum (\alpha_i - \beta_i)\mathbf{v}_i = \mathbf{0}$. By linear independence, $\alpha_i - \beta_i = 0$ for all $i$, so $\alpha_i = \beta_i$. $\square$

**Definition 2.4.7 (Dimension).** The *dimension* of a vector space $V$, written $\dim(V)$, is the number of elements in any basis. (The Steinitz exchange lemma guarantees this is well-defined.)

We write $\dim(V) = \infty$ if no finite basis exists.

**Theorem 2.4.7 (Dimension Theorem).** All bases of a finite-dimensional vector space have the same number of elements.

*Proof.* Suppose $\mathcal{B}_1$ and $\mathcal{B}_2$ are bases with $|\mathcal{B}_1| = m$ and $|\mathcal{B}_2| = n$. Since $\mathcal{B}_1$ is independent and spans $\subseteq \text{span}(\mathcal{B}_2) = V$, the Steinitz lemma gives $m \leq n$. Symmetrically, $n \leq m$. Hence $m = n$. $\square$

### 7.1 Standard Examples

| Vector Space | Standard Basis | Dimension |
|-------------|----------------|-----------|
| $\mathbb{R}^n$ | $\{\mathbf{e}_1, \ldots, \mathbf{e}_n\}$ (standard unit vectors) | $n$ |
| $\mathcal{P}_n(\mathbb{R})$ | $\{1, x, x^2, \ldots, x^n\}$ | $n + 1$ |
| $\mathbb{R}^{m \times n}$ | $\{E_{ij}\}$ (matrices with $1$ in position $(i,j)$, $0$ elsewhere) | $mn$ |
| $\mathbb{C}^{2^n}$ (n-qubit space) | Computational basis $\{|0\cdots0\rangle, \ldots, |1\cdots1\rangle\}$ | $2^n$ |
| $C[a,b]$ | No finite basis | $\infty$ |

**Example 2.4.8 (Finding a basis).** Find a basis for $W = \{(x, y, z) \in \mathbb{R}^3 : x - 2y + z = 0\}$.

Solve $x = 2y - z$, so every vector in $W$ has the form $(2y - z,\; y,\; z) = y(2, 1, 0) + z(-1, 0, 1)$. Thus $W = \text{span}\{(2,1,0),\; (-1,0,1)\}$.

**Independence check:** $\alpha(2,1,0) + \beta(-1,0,1) = (0,0,0) \implies \alpha = 0, \beta = 0$. $\checkmark$

So $\{(2, 1, 0),\; (-1, 0, 1)\}$ is a basis for $W$, and $\dim(W) = 2$. (Geometrically, $W$ is a plane through the origin.)

**Example 2.4.9 (Dimension).** The space of $2 \times 2$ symmetric matrices $\text{Sym}_2(\mathbb{R}) = \left\{ \begin{pmatrix} a & b \\ b & c \end{pmatrix} : a, b, c \in \mathbb{R} \right\}$ has basis:

$$\left\{ \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix},\; \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix},\; \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix} \right\}$$

since $\begin{pmatrix} a & b \\ b & c \end{pmatrix} = a\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} + b\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} + c\begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}$. So $\dim(\text{Sym}_2) = 3$.

### 7.2 Coordinate Vectors

**Definition 2.4.8 (Coordinate Vector).** If $\mathcal{B} = \{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$ is a basis for $V$ and $\mathbf{v} = \alpha_1\mathbf{v}_1 + \cdots + \alpha_n\mathbf{v}_n$, then the *coordinate vector* of $\mathbf{v}$ relative to $\mathcal{B}$ is:

$$[\mathbf{v}]_\mathcal{B} = \begin{pmatrix} \alpha_1 \\ \alpha_2 \\ \vdots \\ \alpha_n \end{pmatrix} \in \mathbb{F}^n$$

This establishes an *isomorphism* between any $n$-dimensional vector space and $\mathbb{F}^n$. Every finite-dimensional vector space "looks like" $\mathbb{F}^n$ once you choose a basis.

*ML connection:* **Latent spaces in VAEs** provide exactly this structure. The encoder maps high-dimensional data (images in $\mathbb{R}^{784}$) to coordinate vectors in a low-dimensional latent space $\mathbb{R}^d$ (typically $d = 2$ to $256$). The decoder maps coordinates back. The basis of the latent space is *learned* — each latent dimension captures a meaningful variation (e.g., digit style, stroke thickness). The reparameterization trick $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$ generates new coordinate vectors for the decoder.

```
VAE AS A CHANGE OF REPRESENTATION

   Image Space (R^784)              Latent Space (R^d)

   ┌─────────────────┐   Encoder   ┌───────────┐
   │ ░░▓▓░░ (image)  │ ─────────→  │ (z₁, z₂)  │  ← coordinate vector
   │ ░▓▓▓░░          │             │  in the    │     in learned basis
   │ ░░▓▓░░          │   Decoder   │  latent    │
   │                 │ ←─────────  │  basis     │
   └─────────────────┘             └───────────┘

   dim = 784                       dim = d ≪ 784
```

---

## 8. Key Subspace Theorems

**Theorem 2.4.8 (Basis Extension).** Every linearly independent set in a finite-dimensional vector space can be extended to a basis.

*Proof sketch.* Let $S = \{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ be linearly independent in $V$ with $\dim(V) = n$. If $\text{span}(S) = V$, we are done. Otherwise, there exists $\mathbf{v}_{k+1} \notin \text{span}(S)$, and $\{\mathbf{v}_1, \ldots, \mathbf{v}_{k+1}\}$ is linearly independent. Repeat until we have $n$ vectors. The process terminates because independent sets have at most $n$ elements. $\square$

**Theorem 2.4.9 (Dimension of Subspaces).** If $W \leq V$ and $\dim(V) = n$, then:

1. $\dim(W) \leq n$
2. $\dim(W) = n$ implies $W = V$

**Theorem 2.4.10 (Rank-Nullity Preview).** For $A \in \mathbb{R}^{m \times n}$:

$$\dim(\text{Col}(A)) + \dim(\text{Null}(A)) = n$$

This is a special case of the rank-nullity theorem, proved in full generality in [Linear Transformations](linear-transformations.md).

---

## 9. Sum and Direct Sum

**Definition 2.4.9 (Sum of Subspaces).** If $W_1, W_2 \leq V$, their *sum* is:

$$W_1 + W_2 = \{\mathbf{w}_1 + \mathbf{w}_2 : \mathbf{w}_1 \in W_1,\; \mathbf{w}_2 \in W_2\}$$

**Theorem 2.4.11 (Dimension of Sum).**

$$\dim(W_1 + W_2) = \dim(W_1) + \dim(W_2) - \dim(W_1 \cap W_2)$$

*Proof sketch.* Choose a basis for $W_1 \cap W_2$, extend it to a basis for $W_1$ and separately to a basis for $W_2$. The union of these extensions, together with the basis for the intersection, forms a basis for $W_1 + W_2$. Counting gives the formula. $\square$

**Definition 2.4.10 (Direct Sum).** The sum $W_1 + W_2$ is a *direct sum*, written $W_1 \oplus W_2$, if $W_1 \cap W_2 = \{\mathbf{0}\}$.

In a direct sum, every vector $\mathbf{v} \in W_1 \oplus W_2$ has a *unique* decomposition $\mathbf{v} = \mathbf{w}_1 + \mathbf{w}_2$ with $\mathbf{w}_i \in W_i$.

**Theorem 2.4.12.** $V = W_1 \oplus W_2$ if and only if:

1. $V = W_1 + W_2$
2. $W_1 \cap W_2 = \{\mathbf{0}\}$

When this holds, $\dim(V) = \dim(W_1) + \dim(W_2)$.

```
DIRECT SUM VISUALIZATION IN R³

V = W₁ ⊕ W₂

    ↑  W₂ (z-axis, dim 1)
    │
    │     Every v ∈ R³ uniquely decomposes as:
    │     v = w₁ + w₂
    │     where w₁ ∈ xy-plane, w₂ ∈ z-axis
    │
────┼────────→  W₁ (xy-plane, dim 2)
   ╱│
  ╱ │
 ╱
dim(R³) = dim(W₁) + dim(W₂) = 2 + 1 = 3
```

*ML connection:* **Skip connections** in ResNets implement a direct-sum-like decomposition. At each residual block, the output is $\mathbf{y} = \mathbf{x} + F(\mathbf{x})$, decomposing the representation into the identity (from $\mathbf{x}$) and the learned residual (from $F(\mathbf{x})$). More formally, **multi-head attention** computes $d_k$-dimensional heads independently in subspaces $W_1, \ldots, W_h$ and concatenates — this is a direct sum $W_1 \oplus \cdots \oplus W_h \cong \mathbb{R}^d$ where $d = h \cdot d_k$.

---

## 10. Change of Basis

**Definition 2.4.11 (Change-of-Basis Matrix).** Let $\mathcal{B} = \{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$ and $\mathcal{B}' = \{\mathbf{w}_1, \ldots, \mathbf{w}_n\}$ be two bases for $V$. The *change-of-basis matrix* from $\mathcal{B}$ to $\mathcal{B}'$ is the matrix $P$ whose columns are the coordinate vectors of the old basis vectors in the new basis:

$$P = \bigl[[\mathbf{v}_1]_{\mathcal{B}'} \;\; [\mathbf{v}_2]_{\mathcal{B}'} \;\; \cdots \;\; [\mathbf{v}_n]_{\mathcal{B}'}\bigr]$$

Then for any $\mathbf{v} \in V$:

$$[\mathbf{v}]_{\mathcal{B}'} = P^{-1} [\mathbf{v}]_{\mathcal{B}}$$

**Theorem 2.4.13 (Change of Basis).** The change-of-basis matrix $P$ is invertible, and $P^{-1}$ is the change-of-basis matrix from $\mathcal{B}'$ to $\mathcal{B}$.

*Proof.* Each column of $P$ is a coordinate vector in a basis, so $P$ has $n$ linearly independent columns (since the $\mathbf{v}_i$ are independent and their coordinate representations in any basis are independent). Therefore $P$ is invertible. The inverse maps coordinate vectors in the opposite direction by construction. $\square$

**Example 2.4.12 (Change of basis in $\mathbb{R}^2$).** Let $\mathcal{B} = \{\mathbf{e}_1, \mathbf{e}_2\}$ (standard basis) and $\mathcal{B}' = \{(1, 1),\; (1, -1)\}$. Find the coordinates of $\mathbf{v} = (5, 3)$ in basis $\mathcal{B}'$.

The change-of-basis matrix from $\mathcal{B}'$ to $\mathcal{B}$ has the $\mathcal{B}'$ vectors as columns:

$$P = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}, \quad P^{-1} = \frac{1}{-2}\begin{pmatrix} -1 & -1 \\ -1 & 1 \end{pmatrix} = \begin{pmatrix} 1/2 & 1/2 \\ 1/2 & -1/2 \end{pmatrix}$$

$$[\mathbf{v}]_{\mathcal{B}'} = P^{-1}\begin{pmatrix} 5 \\ 3 \end{pmatrix} = \begin{pmatrix} 1/2 & 1/2 \\ 1/2 & -1/2 \end{pmatrix}\begin{pmatrix} 5 \\ 3 \end{pmatrix} = \begin{pmatrix} 4 \\ 1 \end{pmatrix}$$

**Verify:** $4(1, 1) + 1(1, -1) = (4, 4) + (1, -1) = (5, 3) = \mathbf{v}$. $\checkmark$

```
CHANGE OF BASIS: THE SAME VECTOR IN TWO COORDINATE SYSTEMS

Standard basis B = {e₁, e₂}:        Rotated basis B' = {w₁, w₂}:

    ↑ e₂                                 ╱ w₂
    │                                    ╱
    │   ● v = (3,2) in B               ╱   ● v = (α₁, α₂) in B'
    │  ╱                               ╱  ╱
    │ ╱                               ╱ ╱
    ●─────→ e₁                       ●─────→ w₁

    [v]_B = (3, 2)                   [v]_B' = P⁻¹(3, 2)

The vector v is THE SAME — only the coordinates change.
```

*ML connection:* **PCA is a change of basis.** The principal components are eigenvectors of the covariance matrix, forming a new basis aligned with the directions of maximum variance. Projecting data onto the top $k$ principal components is equivalent to: (1) changing to the eigenvector basis, (2) keeping only the first $k$ coordinates, and (3) discarding the rest. The change-of-basis matrix is the matrix of eigenvectors.

**Transformer positional encodings** also represent a change of basis: the sinusoidal encoding maps position integers into a vector space where relative positions become easy to compute via linear operations — effectively choosing a basis (of sine/cosine functions at different frequencies) that makes position arithmetic natural.

---

## 11. The Four Fundamental Subspaces

For any matrix $A \in \mathbb{R}^{m \times n}$, there are four fundamental subspaces:

| Subspace | Space | Definition | Dimension |
|----------|-------|------------|-----------|
| Column space $\text{Col}(A)$ | $\mathbb{R}^m$ | $\{A\mathbf{x} : \mathbf{x} \in \mathbb{R}^n\}$ | $r$ |
| Null space $\text{Null}(A)$ | $\mathbb{R}^n$ | $\{\mathbf{x} : A\mathbf{x} = \mathbf{0}\}$ | $n - r$ |
| Row space $\text{Row}(A)$ | $\mathbb{R}^n$ | $\text{Col}(A^T)$ | $r$ |
| Left null space $\text{Null}(A^T)$ | $\mathbb{R}^m$ | $\{\mathbf{y} : A^T\mathbf{y} = \mathbf{0}\}$ | $m - r$ |

where $r = \text{rank}(A)$.

**Example 2.4.10 (Null space computation).** Let $A = \begin{pmatrix} 1 & 2 & 1 \\ 2 & 4 & 3 \end{pmatrix}$. Find $\text{Null}(A)$.

Row reduce: $R_2 \leftarrow R_2 - 2R_1$: $\begin{pmatrix} 1 & 2 & 1 \\ 0 & 0 & 1 \end{pmatrix}$. Back-substitute: $x_3 = 0$, $x_1 = -2x_2$. With $x_2 = t$:

$$\text{Null}(A) = \left\{ t(-2, 1, 0) : t \in \mathbb{R} \right\} = \text{span}\{(-2, 1, 0)\}$$

So $\dim(\text{Null}(A)) = 1$. Verify: $A(-2, 1, 0)^T = (-2+2+0,\; -4+4+0)^T = (0, 0)^T$. $\checkmark$

**Example 2.4.11 (Column space).** For the same $A = \begin{pmatrix} 1 & 2 & 1 \\ 2 & 4 & 3 \end{pmatrix}$, identify $\text{Col}(A)$.

The columns are $\mathbf{a}_1 = (1, 2)$, $\mathbf{a}_2 = (2, 4)$, $\mathbf{a}_3 = (1, 3)$. Note $\mathbf{a}_2 = 2\mathbf{a}_1$, so column 2 is redundant. From the row echelon form, pivot columns are 1 and 3, so:

$$\text{Col}(A) = \text{span}\{(1, 2),\; (1, 3)\}$$

These two vectors are independent (neither is a scalar multiple of the other), so $\dim(\text{Col}(A)) = 2 = \text{rank}(A)$. Since $\text{Col}(A) \subseteq \mathbb{R}^2$ and $\dim(\text{Col}(A)) = 2$, we have $\text{Col}(A) = \mathbb{R}^2$ — every 2D vector is a possible output.

**Rank-nullity check:** $\text{rank}(A) + \dim(\text{Null}(A)) = 2 + 1 = 3 = n$. $\checkmark$

```
THE FOUR FUNDAMENTAL SUBSPACES (after Gilbert Strang)

                  R^n                                R^m
        ┌───────────────────────┐          ┌───────────────────────┐
        │                       │          │                       │
        │   Row(A)    Null(A)   │    A     │  Col(A)   Null(A^T)  │
        │   dim r     dim n-r   │ ───────→ │  dim r    dim m-r    │
        │     ⊥                 │          │    ⊥                 │
        │  (orthogonal                     │  (orthogonal          │
        │   complements)                   │   complements)        │
        └───────────────────────┘          └───────────────────────┘

Key relationships:
  Row(A)  ⊕ Null(A)   = R^n
  Col(A)  ⊕ Null(A^T) = R^m
  A maps Row(A) → Col(A) bijectively
  A maps Null(A) → {0}
```

*ML connection:* In a linear layer $\mathbf{y} = W\mathbf{x}$:

- **Col(W)** is the subspace of all possible outputs — the "expressible" space of the layer
- **Null(W)** is the set of input perturbations the layer ignores — the "invisible" directions
- **Row(W)** is the set of input directions the layer actually uses
- **Null(W^T)** is the set of output directions the layer can never produce

Understanding these subspaces explains **information bottlenecks** in neural networks. A layer with rank $r < \min(m,n)$ compresses $n$-dimensional input to an $r$-dimensional representation and cannot produce outputs outside an $r$-dimensional subspace.

---

## Exercises

**★ Basic**

1. Verify that $\mathcal{P}_2(\mathbb{R})$ (polynomials of degree $\leq 2$) is a vector space by checking all eight axioms.

2. Determine which of the following are subspaces of $\mathbb{R}^3$:
    - (a) $\{(x, y, z) : x + 2y - z = 0\}$
    - (b) $\{(x, y, z) : x + 2y - z = 1\}$
    - (c) $\{(x, y, z) : xy = 0\}$
    - (d) $\{(x, y, z) : x = 2z\}$

3. Find a basis and the dimension of the subspace $W = \{(x, y, z) : x - 2y + z = 0\}$ of $\mathbb{R}^3$.

4. Show that $\{1, x-1, (x-1)^2\}$ is a basis for $\mathcal{P}_2(\mathbb{R})$ and find the coordinate vector of $p(x) = 3x^2 + 2x + 1$ in this basis.

**★★ Intermediate**

5. Let $W_1 = \text{span}\{(1,0,1), (0,1,1)\}$ and $W_2 = \text{span}\{(1,1,0), (0,1,1)\}$ in $\mathbb{R}^3$. Find $\dim(W_1 \cap W_2)$ and $\dim(W_1 + W_2)$. Is the sum direct?

6. Prove that the set of $n \times n$ symmetric matrices ($A = A^T$) is a subspace of $\mathbb{R}^{n \times n}$. Find its dimension and a basis. Do the same for skew-symmetric matrices ($A = -A^T$). Show that $\mathbb{R}^{n \times n} = \text{Sym}_n \oplus \text{Skew}_n$.

7. Let $V = C[0, 2\pi]$ (continuous functions on $[0, 2\pi]$). Show that $W = \text{span}\{\sin(x), \sin(2x), \ldots, \sin(nx)\}$ is an $n$-dimensional subspace. *Hint:* Use the orthogonality of the sine functions under the inner product $\langle f, g \rangle = \int_0^{2\pi} f(x)g(x)\,dx$.

8. (VAE Geometry) A variational autoencoder maps images in $\mathbb{R}^{784}$ to a latent space $\mathbb{R}^{10}$. The encoder is a composition of two linear maps $L_2 \circ L_1$ where $L_1: \mathbb{R}^{784} \to \mathbb{R}^{256}$ and $L_2: \mathbb{R}^{256} \to \mathbb{R}^{10}$.
    - (a) What is the maximum possible dimension of $\text{Col}(L_2 L_1)$?
    - (b) What is the minimum possible dimension of $\text{Null}(L_2 L_1)$?
    - (c) Interpret parts (a) and (b) in terms of information preservation.

**★★★ Challenging**

9. Prove that if $V$ is finite-dimensional and $W_1, W_2, W_3$ are subspaces with $V = W_1 \oplus W_2 = W_1 \oplus W_3$, it does NOT follow that $W_2 = W_3$. Give a counterexample in $\mathbb{R}^2$.

10. (Infinite dimension) Show that $\mathcal{P}(\mathbb{R})$ — the space of *all* polynomials (no degree bound) — is infinite-dimensional by proving that $\{1, x, x^2, \ldots\}$ is linearly independent. Then show that this space is *not* isomorphic to $C[0,1]$ as vector spaces despite both being infinite-dimensional. *Hint:* Consider the dimension type (countable vs. uncountable basis).

11. (Quantum Computing) A system of $n$ qubits has state space $\mathbb{C}^{2^n}$. The *product states* are vectors of the form $|\psi_1\rangle \otimes |\psi_2\rangle \otimes \cdots \otimes |\psi_n\rangle$ where each $|\psi_i\rangle \in \mathbb{C}^2$. Show that the set of product states is NOT a subspace of $\mathbb{C}^{2^n}$ (this is why entanglement is interesting). How many real parameters describe an arbitrary $n$-qubit state vs. a product state?

---

## Related Topics

- [Vectors](vectors.md) — concrete vector arithmetic and geometry in $\mathbb{R}^n$
- [Systems of Equations](systems-of-equations.md) — null space and column space via row reduction
- [Linear Transformations](linear-transformations.md) — maps between vector spaces, kernel and image
- [Inner Product Spaces](inner-product-spaces.md) — adding geometry (angles, orthogonality) to vector spaces
- [Eigenvalues & Eigenvectors](eigenvalues.md) — invariant subspaces and diagonalization
- [Matrix Decompositions](matrix-decompositions.md) — SVD as the ultimate subspace decomposition

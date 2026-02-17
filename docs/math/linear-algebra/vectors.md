# Vectors

Every data point in machine learning is a vector. A house described by square footage, bedrooms, and age is a vector in $\mathbb{R}^3$. A word embedding from a transformer is a vector in $\mathbb{R}^{768}$. A quantum state is a vector in a complex Hilbert space. Understanding vectors — their algebra, geometry, and the distances between them — is the first step in the mathematics of machine learning.

---

## Prerequisites

- [Foundations](../foundations/index.md)

---

## 1. Definitions

**Definition 2.1.1 (Vector).** A *vector* in $\mathbb{R}^n$ (or $\mathbb{F}^n$ for a general field $\mathbb{F}$) is an ordered $n$-tuple:

$$\mathbf{v} = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix} \in \mathbb{R}^n$$

We use bold lowercase ($\mathbf{v}$) or arrow notation ($\vec{v}$) for vectors and italic ($v_i$) for components.

**Convention:** Vectors are column vectors unless stated otherwise. A row vector is written $\mathbf{v}^T$.

### 1.1 Vector Operations

**Definition 2.1.2 (Vector Addition).** For $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$:

$$\mathbf{u} + \mathbf{v} = \begin{pmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{pmatrix}$$

**Example 2.1.1:** Let $\mathbf{u} = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$ and $\mathbf{v} = \begin{pmatrix} 4 \\ -1 \\ 2 \end{pmatrix}$. Then:

$$\mathbf{u} + \mathbf{v} = \begin{pmatrix} 1+4 \\ 2+(-1) \\ 3+2 \end{pmatrix} = \begin{pmatrix} 5 \\ 1 \\ 5 \end{pmatrix}$$

**Definition 2.1.3 (Scalar Multiplication).** For $c \in \mathbb{R}$, $\mathbf{v} \in \mathbb{R}^n$:

$$c\mathbf{v} = \begin{pmatrix} cv_1 \\ cv_2 \\ \vdots \\ cv_n \end{pmatrix}$$

**Example 2.1.2:** Let $c = 3$ and $\mathbf{v} = \begin{pmatrix} 4 \\ -1 \\ 2 \end{pmatrix}$. Then:

$$3\mathbf{v} = \begin{pmatrix} 3 \cdot 4 \\ 3 \cdot (-1) \\ 3 \cdot 2 \end{pmatrix} = \begin{pmatrix} 12 \\ -3 \\ 6 \end{pmatrix}$$

The result points in the same direction as $\mathbf{v}$ but is 3 times as long. If $c = -2$, the result $(-8, 2, -4)$ reverses direction.

**Theorem 2.1.1.** $(\mathbb{R}^n, +, \cdot)$ satisfies the vector space axioms (see [Vector Spaces](vector-spaces.md)):

1. $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$ (commutativity)
2. $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$ (associativity)
3. $\exists\, \mathbf{0}$ such that $\mathbf{v} + \mathbf{0} = \mathbf{v}$ (additive identity)
4. $\forall\, \mathbf{v}, \exists\, (-\mathbf{v})$ with $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$ (additive inverse)
5. $1 \cdot \mathbf{v} = \mathbf{v}$ (multiplicative identity)
6. $c(d\mathbf{v}) = (cd)\mathbf{v}$ (compatibility)
7. $c(\mathbf{u} + \mathbf{v}) = c\mathbf{u} + c\mathbf{v}$ (distributivity)
8. $(c + d)\mathbf{v} = c\mathbf{v} + d\mathbf{v}$ (distributivity)

---

## 2. Inner Product (Dot Product)

**Definition 2.1.4 (Dot Product).** For $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$:

$$\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u} \cdot \mathbf{v} = \mathbf{u}^T \mathbf{v} = \sum_{i=1}^n u_i v_i$$

**Example 2.1.3:** Let $\mathbf{u} = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$ and $\mathbf{v} = \begin{pmatrix} 4 \\ -1 \\ 2 \end{pmatrix}$. Then:

$$\langle \mathbf{u}, \mathbf{v} \rangle = (1)(4) + (2)(-1) + (3)(2) = 4 - 2 + 6 = 8$$

Since $\langle \mathbf{u}, \mathbf{v} \rangle = 8 > 0$, the angle between them is less than $90°$.

**Theorem 2.1.2 (Properties of the Dot Product).**

1. $\langle \mathbf{u}, \mathbf{v} \rangle = \langle \mathbf{v}, \mathbf{u} \rangle$ (symmetry)
2. $\langle \alpha\mathbf{u} + \beta\mathbf{w}, \mathbf{v} \rangle = \alpha \langle \mathbf{u}, \mathbf{v} \rangle + \beta \langle \mathbf{w}, \mathbf{v} \rangle$ (linearity)
3. $\langle \mathbf{v}, \mathbf{v} \rangle \geq 0$ with equality iff $\mathbf{v} = \mathbf{0}$ (positive definiteness)

*Proof of (3).* $\langle \mathbf{v}, \mathbf{v} \rangle = \sum_{i=1}^n v_i^2 \geq 0$ since each $v_i^2 \geq 0$. Equality holds iff every $v_i = 0$, i.e., $\mathbf{v} = \mathbf{0}$. $\square$

**Geometric interpretation:**

$$\langle \mathbf{u}, \mathbf{v} \rangle = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta$$

where $\theta$ is the angle between $\mathbf{u}$ and $\mathbf{v}$.

```
         v
        ╱
       ╱ θ
      ╱───── u
     ╱

⟨u,v⟩ > 0 : θ < 90°  (vectors point "same direction")
⟨u,v⟩ = 0 : θ = 90°  (orthogonal)
⟨u,v⟩ < 0 : θ > 90°  (vectors point "opposite directions")
```

*ML connection:* **Cosine similarity** between embeddings:

$$\text{cos\_sim}(\mathbf{u}, \mathbf{v}) = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{u}\| \|\mathbf{v}\|}$$

This is the core similarity measure in word embeddings (Word2Vec, GloVe), recommendation systems, and retrieval-augmented generation (RAG). Two documents are "similar" if their embedding vectors point in nearly the same direction.

**Example 2.1.4:** Find the angle between $\mathbf{u} = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$ and $\mathbf{v} = \begin{pmatrix} 4 \\ -1 \\ 2 \end{pmatrix}$.

$$\|\mathbf{u}\| = \sqrt{1^2 + 2^2 + 3^2} = \sqrt{14}, \quad \|\mathbf{v}\| = \sqrt{4^2 + (-1)^2 + 2^2} = \sqrt{21}$$

$$\cos\theta = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{u}\| \|\mathbf{v}\|} = \frac{8}{\sqrt{14}\sqrt{21}} = \frac{8}{\sqrt{294}} \approx 0.467$$

$$\theta = \arccos(0.467) \approx 62.2°$$

The vectors are neither nearly parallel nor orthogonal.

---

## 3. Norms

**Definition 2.1.5 (Norm).** A *norm* on $\mathbb{R}^n$ is a function $\|\cdot\|: \mathbb{R}^n \to \mathbb{R}_{\geq 0}$ satisfying:

1. $\|\mathbf{v}\| \geq 0$ with $\|\mathbf{v}\| = 0 \iff \mathbf{v} = \mathbf{0}$ (positive definiteness)
2. $\|c\mathbf{v}\| = |c| \|\mathbf{v}\|$ (absolute homogeneity)
3. $\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$ (triangle inequality)

### 3.1 The $p$-Norms

**Definition 2.1.6 ($\ell^p$ Norm).**

$$\|\mathbf{v}\|_p = \left(\sum_{i=1}^n |v_i|^p\right)^{1/p} \qquad p \geq 1$$

| Norm | Formula | Geometric Meaning | ML Usage |
|------|---------|-------------------|----------|
| $\ell^1$ (Manhattan) | $\sum_i \|v_i\|$ | Diamond-shaped ball | Lasso regularization (sparsity) |
| $\ell^2$ (Euclidean) | $\sqrt{\sum_i v_i^2}$ | Circular ball | Ridge regularization, distance metrics |
| $\ell^\infty$ (Chebyshev) | $\max_i \|v_i\|$ | Square ball | Adversarial robustness ($\ell^\infty$ attacks) |
| $\ell^0$ (pseudo-norm) | $\#\{i : v_i \neq 0\}$ | Count of nonzeros | Feature selection (NP-hard to optimize) |

**Example 2.1.5:** Compute all three norms for $\mathbf{v} = \begin{pmatrix} 3 \\ -4 \\ 2 \end{pmatrix}$:

$$\|\mathbf{v}\|_1 = |3| + |-4| + |2| = 3 + 4 + 2 = 9$$

$$\|\mathbf{v}\|_2 = \sqrt{3^2 + (-4)^2 + 2^2} = \sqrt{9 + 16 + 4} = \sqrt{29} \approx 5.39$$

$$\|\mathbf{v}\|_\infty = \max(|3|, |-4|, |2|) = 4$$

Note the ordering: $\|\mathbf{v}\|_\infty \leq \|\mathbf{v}\|_2 \leq \|\mathbf{v}\|_1$, which always holds.

**Example 2.1.6 (Unit Vector):** Normalize $\mathbf{v} = \begin{pmatrix} 3 \\ -4 \\ 2 \end{pmatrix}$ to a unit vector:

$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2} = \frac{1}{\sqrt{29}} \begin{pmatrix} 3 \\ -4 \\ 2 \end{pmatrix} \approx \begin{pmatrix} 0.557 \\ -0.743 \\ 0.371 \end{pmatrix}$$

Verify: $\|\hat{\mathbf{v}}\|_2 = \sqrt{(3/\sqrt{29})^2 + (-4/\sqrt{29})^2 + (2/\sqrt{29})^2} = \sqrt{29/29} = 1$ ✓

```
UNIT BALLS IN 2D

     ℓ¹              ℓ²              ℓ∞
     ◇               ○               □
   ╱   ╲           ╱   ╲           ┌───┐
  ╱     ╲         │     │          │   │
 ◇       ◇        ○     ○          │   │
  ╲     ╱         │     │          └───┘
   ╲   ╱           ╲   ╱
     ◇               ○

{v : ‖v‖₁ ≤ 1}  {v : ‖v‖₂ ≤ 1}  {v : ‖v‖∞ ≤ 1}
```

*ML connection:* L1 regularization ($\|\mathbf{w}\|_1$) in Lasso produces sparse solutions because the $\ell^1$ ball has corners on the axes — the optimal point is more likely to lie at a corner where some coordinates are exactly zero. L2 regularization ($\|\mathbf{w}\|_2^2$) in Ridge produces small but nonzero weights because the $\ell^2$ ball is smooth.

### 3.2 Key Inequalities

**Theorem 2.1.3 (Cauchy-Schwarz Inequality).**

$$|\langle \mathbf{u}, \mathbf{v} \rangle| \leq \|\mathbf{u}\| \cdot \|\mathbf{v}\|$$

with equality iff $\mathbf{u} = c\mathbf{v}$ for some scalar $c$ (i.e., they are parallel).

*Proof.* For any $t \in \mathbb{R}$, $\|\mathbf{u} - t\mathbf{v}\|^2 \geq 0$. Expanding: $\|\mathbf{u}\|^2 - 2t\langle \mathbf{u}, \mathbf{v} \rangle + t^2\|\mathbf{v}\|^2 \geq 0$. This quadratic in $t$ is non-negative, so its discriminant is non-positive: $4\langle \mathbf{u}, \mathbf{v} \rangle^2 - 4\|\mathbf{u}\|^2\|\mathbf{v}\|^2 \leq 0$. $\square$

**Example 2.1.7 (Cauchy-Schwarz Verification):** Let $\mathbf{u} = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$ and $\mathbf{v} = \begin{pmatrix} 4 \\ -1 \\ 2 \end{pmatrix}$.

- Left side: $|\langle \mathbf{u}, \mathbf{v} \rangle| = |8| = 8$
- Right side: $\|\mathbf{u}\| \cdot \|\mathbf{v}\| = \sqrt{14} \cdot \sqrt{21} = \sqrt{294} \approx 17.15$
- Check: $8 \leq 17.15$ ✓

Equality would hold only if $\mathbf{u} = c\mathbf{v}$, which it does not (no single $c$ satisfies $1 = 4c$, $2 = -c$, $3 = 2c$).

**Theorem 2.1.4 (Triangle Inequality).**

$$\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$$

*Proof.* $\|\mathbf{u}+\mathbf{v}\|^2 = \|\mathbf{u}\|^2 + 2\langle \mathbf{u},\mathbf{v}\rangle + \|\mathbf{v}\|^2 \leq \|\mathbf{u}\|^2 + 2\|\mathbf{u}\|\|\mathbf{v}\| + \|\mathbf{v}\|^2 = (\|\mathbf{u}\| + \|\mathbf{v}\|)^2$ by Cauchy-Schwarz. $\square$

**Example 2.1.13 (Triangle Inequality Verification):** Let $\mathbf{u} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$ and $\mathbf{v} = \begin{pmatrix} 3 \\ -1 \end{pmatrix}$.

$$\|\mathbf{u} + \mathbf{v}\| = \left\|\begin{pmatrix} 4 \\ 1 \end{pmatrix}\right\| = \sqrt{17} \approx 4.12$$

$$\|\mathbf{u}\| + \|\mathbf{v}\| = \sqrt{5} + \sqrt{10} \approx 2.24 + 3.16 = 5.40$$

Check: $4.12 \leq 5.40$ ✓ The "shortcut" (direct path $\mathbf{u}+\mathbf{v}$) is shorter than going via $\mathbf{u}$ then $\mathbf{v}$.

*ML connection:* The triangle inequality guarantees that distances in feature space behave intuitively: the distance between $A$ and $C$ is at most the sum of distances $A$-to-$B$ and $B$-to-$C$. This property is required for metric learning, $k$-NN, and clustering algorithms.

---

## 4. Linear Combinations and Span

**Definition 2.1.7 (Linear Combination).** A *linear combination* of vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$ is:

$$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k \qquad c_i \in \mathbb{R}$$

**Example 2.1.8:** Let $\mathbf{v}_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$ and $\mathbf{v}_2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$. Then the vector $\begin{pmatrix} 3 \\ 5 \end{pmatrix}$ is a linear combination:

$$3\mathbf{v}_1 + 5\mathbf{v}_2 = 3\begin{pmatrix} 1 \\ 0 \end{pmatrix} + 5\begin{pmatrix} 0 \\ 1 \end{pmatrix} = \begin{pmatrix} 3 \\ 0 \end{pmatrix} + \begin{pmatrix} 0 \\ 5 \end{pmatrix} = \begin{pmatrix} 3 \\ 5 \end{pmatrix}$$

**Definition 2.1.8 (Span).** The *span* of $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ is the set of all linear combinations:

$$\text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\} = \left\{\sum_{i=1}^k c_i \mathbf{v}_i : c_i \in \mathbb{R}\right\}$$

**Definition 2.1.9 (Linear Independence).** Vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$ are *linearly independent* if the only solution to $\sum c_i \mathbf{v}_i = \mathbf{0}$ is $c_1 = \cdots = c_k = 0$.

**Example 2.1.9:** Test whether $\mathbf{v}_1 = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$ and $\mathbf{v}_2 = \begin{pmatrix} 3 \\ 4 \end{pmatrix}$ are linearly independent.

Solve $c_1\mathbf{v}_1 + c_2\mathbf{v}_2 = \mathbf{0}$:

$$c_1 + 3c_2 = 0, \quad 2c_1 + 4c_2 = 0$$

From the first equation: $c_1 = -3c_2$. Substituting: $2(-3c_2) + 4c_2 = -2c_2 = 0$, so $c_2 = 0$ and $c_1 = 0$.
The only solution is the trivial one, so $\mathbf{v}_1, \mathbf{v}_2$ are **linearly independent**.

Compare with $\mathbf{v}_1 = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$, $\mathbf{v}_2 = \begin{pmatrix} 2 \\ 4 \end{pmatrix}$: here $2\mathbf{v}_1 - \mathbf{v}_2 = \mathbf{0}$, so they are **linearly dependent** ($\mathbf{v}_2$ is just $2\mathbf{v}_1$).

*ML connection:* Features that are linearly dependent carry redundant information. If feature 3 = 2×(feature 1) + 3×(feature 2), it adds no information to a linear model. This is why we check for multicollinearity in regression and why PCA removes redundant dimensions.

---

## 5. Orthogonality and Projection

**Definition 2.1.10 (Orthogonal Vectors).** $\mathbf{u} \perp \mathbf{v}$ if $\langle \mathbf{u}, \mathbf{v} \rangle = 0$.

**Example 2.1.10:** Let $\mathbf{u} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$ and $\mathbf{v} = \begin{pmatrix} -2 \\ 1 \end{pmatrix}$. Then:

$$\langle \mathbf{u}, \mathbf{v} \rangle = (1)(-2) + (2)(1) = -2 + 2 = 0$$

So $\mathbf{u} \perp \mathbf{v}$. Note that rotating any 2D vector by $90°$ (swapping components and negating one) always produces an orthogonal vector.

**Theorem 2.1.5 (Pythagorean Theorem).** If $\mathbf{u} \perp \mathbf{v}$, then $\|\mathbf{u} + \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2$.

*Proof.* $\|\mathbf{u}+\mathbf{v}\|^2 = \langle \mathbf{u}+\mathbf{v}, \mathbf{u}+\mathbf{v}\rangle = \|\mathbf{u}\|^2 + 2\langle \mathbf{u},\mathbf{v}\rangle + \|\mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2$. $\square$

**Definition 2.1.11 (Projection).** The *orthogonal projection* of $\mathbf{u}$ onto $\mathbf{v}$:

$$\text{proj}_{\mathbf{v}} \mathbf{u} = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\langle \mathbf{v}, \mathbf{v} \rangle} \mathbf{v}$$

```
          u
         ╱│
        ╱ │ (u - proj_v u)  ← orthogonal to v
       ╱  │
      ╱   │
─────●────●──────── v
     proj_v u
```

**Example 2.1.11:** Project $\mathbf{u} = \begin{pmatrix} 3 \\ 4 \end{pmatrix}$ onto $\mathbf{v} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$.

$$\langle \mathbf{u}, \mathbf{v} \rangle = (3)(1) + (4)(2) = 11, \quad \langle \mathbf{v}, \mathbf{v} \rangle = 1^2 + 2^2 = 5$$

$$\text{proj}_{\mathbf{v}} \mathbf{u} = \frac{11}{5} \begin{pmatrix} 1 \\ 2 \end{pmatrix} = \begin{pmatrix} 2.2 \\ 4.4 \end{pmatrix}$$

Verify orthogonality of the residual: $\mathbf{u} - \text{proj}_{\mathbf{v}}\mathbf{u} = \begin{pmatrix} 0.8 \\ -0.4 \end{pmatrix}$, and $\langle \begin{pmatrix} 0.8 \\ -0.4 \end{pmatrix}, \begin{pmatrix} 1 \\ 2 \end{pmatrix} \rangle = 0.8 - 0.8 = 0$ ✓

*ML connection:* **Attention mechanisms** compute a form of projection. The query vector $\mathbf{q}$ computes similarity with key vectors $\mathbf{k}_i$ via $\langle \mathbf{q}, \mathbf{k}_i \rangle$, then uses these scores to take a weighted combination of value vectors — geometrically, this is a soft projection onto the subspace spanned by the values.

---

## 6. Cross Product (in $\mathbb{R}^3$)

**Definition 2.1.12.** For $\mathbf{u}, \mathbf{v} \in \mathbb{R}^3$:

$$\mathbf{u} \times \mathbf{v} = \begin{pmatrix} u_2 v_3 - u_3 v_2 \\ u_3 v_1 - u_1 v_3 \\ u_1 v_2 - u_2 v_1 \end{pmatrix}$$

$\|\mathbf{u} \times \mathbf{v}\| = \|\mathbf{u}\| \|\mathbf{v}\| \sin\theta$ = area of the parallelogram spanned by $\mathbf{u}$ and $\mathbf{v}$.

**Example 2.1.12:** Let $\mathbf{u} = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$ and $\mathbf{v} = \begin{pmatrix} 4 \\ -1 \\ 2 \end{pmatrix}$. Then:

$$\mathbf{u} \times \mathbf{v} = \begin{pmatrix} (2)(2) - (3)(-1) \\ (3)(4) - (1)(2) \\ (1)(-1) - (2)(4) \end{pmatrix} = \begin{pmatrix} 4 + 3 \\ 12 - 2 \\ -1 - 8 \end{pmatrix} = \begin{pmatrix} 7 \\ 10 \\ -9 \end{pmatrix}$$

Verify perpendicularity: $\langle \mathbf{u}, \mathbf{u} \times \mathbf{v} \rangle = (1)(7) + (2)(10) + (3)(-9) = 7 + 20 - 27 = 0$ ✓

---

## 7. Applications in ML/AI

### 7.1 Word Embeddings

In Word2Vec and GloVe, each word is a vector in $\mathbb{R}^d$ (typically $d = 300$). The famous relationship:

$$\text{vec}(\text{king}) - \text{vec}(\text{man}) + \text{vec}(\text{woman}) \approx \text{vec}(\text{queen})$$

is vector arithmetic. Semantic relationships are encoded as vector directions.

### 7.2 Neural Network Forward Pass

A single dense layer computes $\mathbf{y} = \sigma(W\mathbf{x} + \mathbf{b})$ where:

- $\mathbf{x} \in \mathbb{R}^n$ is the input vector
- $W \in \mathbb{R}^{m \times n}$ is the weight matrix
- $\mathbf{b} \in \mathbb{R}^m$ is the bias vector
- $\sigma$ is the activation function applied element-wise

The matrix-vector product $W\mathbf{x}$ computes $m$ inner products simultaneously — each row of $W$ dotted with $\mathbf{x}$ produces one component of the output.

### 7.3 Quantum State Vectors

A single qubit state is a unit vector in $\mathbb{C}^2$:

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle = \begin{pmatrix} \alpha \\ \beta \end{pmatrix} \qquad |\alpha|^2 + |\beta|^2 = 1$$

The inner product $\langle \phi | \psi \rangle$ gives the probability amplitude of measuring state $|\psi\rangle$ in state $|\phi\rangle$.

---

## Exercises

**★ Basic**

1. Compute $\langle (1,2,3), (4,-1,2) \rangle$ and the angle between these vectors.

2. Compute $\|(3, 4)\|_1$, $\|(3, 4)\|_2$, and $\|(3, 4)\|_\infty$.

3. Find the projection of $\mathbf{u} = (3, 4)$ onto $\mathbf{v} = (1, 0)$.

**★★ Intermediate**

4. Prove that for any norm $\|\cdot\|$, the *reverse triangle inequality* holds: $|\|\mathbf{u}\| - \|\mathbf{v}\|| \leq \|\mathbf{u} - \mathbf{v}\|$.

5. Show that $\|\mathbf{v}\|_\infty \leq \|\mathbf{v}\|_2 \leq \|\mathbf{v}\|_1 \leq n\|\mathbf{v}\|_\infty$ for $\mathbf{v} \in \mathbb{R}^n$.

6. Given word vectors $\mathbf{w}_1, \mathbf{w}_2, \mathbf{w}_3 \in \mathbb{R}^d$, express the analogy "A is to B as C is to ?" as a vector operation and explain why cosine similarity is used to find the answer.

**★★★ Challenging**

7. Prove the *parallelogram law*: $\|\mathbf{u}+\mathbf{v}\|^2 + \|\mathbf{u}-\mathbf{v}\|^2 = 2(\|\mathbf{u}\|^2 + \|\mathbf{v}\|^2)$ for the $\ell^2$ norm. Show that this fails for $\ell^1$, proving that $\ell^1$ is not an inner product space.

8. Define the $\ell^p$ norm for $0 < p < 1$. Show that it violates the triangle inequality and is therefore not a true norm. Despite this, $\ell^p$ "quasi-norms" are used in compressed sensing. Why?

---

## Related Topics

- [Matrices](matrices.md) — matrix-vector multiplication and beyond
- [Vector Spaces](vector-spaces.md) — the abstract algebraic structure
- [Inner Product Spaces](inner-product-spaces.md) — generalized inner products
- [Eigenvalues](eigenvalues.md) — the vectors that matrices don't change direction

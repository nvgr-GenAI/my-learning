# Eigenvalues and Eigenvectors

Eigenvalues and eigenvectors are the most important concept in linear algebra for machine learning. When a matrix acts on an eigenvector, it merely scales it — the direction is unchanged. This simple idea unlocks PCA (find the directions of maximum variance), PageRank (find the steady-state of web surfing), spectral clustering (find the smoothest partitions of a graph), and the analysis of neural network training dynamics (Hessian eigenvalues reveal the curvature of the loss landscape).

---

## Prerequisites

- [Vectors](vectors.md) — inner products, norms, linear independence
- [Matrices](matrices.md) — matrix operations, transpose, inverse
- [Systems of Equations](systems-of-equations.md) — solving $A\mathbf{x} = \mathbf{b}$
- [Vector Spaces](vector-spaces.md) — basis, dimension, subspaces
- [Linear Transformations](linear-transformations.md) — kernel, image
- [Determinants](determinants.md) — properties, computation, $\det(A) = 0 \iff A$ singular

---

## 1. Definition

**Definition 2.7.1 (Eigenvalue and Eigenvector).** Let $A \in \mathbb{R}^{n \times n}$ (or $\mathbb{C}^{n \times n}$). A scalar $\lambda$ is an *eigenvalue* of $A$ if there exists a nonzero vector $\mathbf{v} \neq \mathbf{0}$ such that:

$$A\mathbf{v} = \lambda\mathbf{v}$$

The vector $\mathbf{v}$ is called an *eigenvector* corresponding to $\lambda$.

**Example 2.7.1:** Let $A = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$ and $\mathbf{v} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$. Verify that $\mathbf{v}$ is an eigenvector.

$$A\mathbf{v} = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}\begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 3 \\ 0 \end{pmatrix} = 3\begin{pmatrix} 1 \\ 0 \end{pmatrix} = 3\mathbf{v}$$

So $\lambda = 3$ is an eigenvalue and $\mathbf{v} = (1, 0)^T$ is a corresponding eigenvector. The matrix scales $\mathbf{v}$ by factor $3$ without changing its direction.

**Equivalent formulation:** $(A - \lambda I)\mathbf{v} = \mathbf{0}$ has a nontrivial solution, which happens if and only if:

$$\det(A - \lambda I) = 0$$

```
GEOMETRIC MEANING

    Regular vector:                   Eigenvector:
    A changes both direction          A only scales — direction unchanged
    and magnitude                     (or reversed if λ < 0)

         Av                                 Av = λv
        ╱                                   ▲
       ╱                                    │  (stretched by λ)
      ╱                                     │
     ▶ v                                    ▶ v
    Original                            Original

    When λ > 1: stretching along v
    When 0 < λ < 1: compression along v
    When λ < 0: reversal and scaling
    When λ = 0: collapse (v maps to zero)
```

*ML connection:* In PCA, the eigenvectors of the covariance matrix $\Sigma = \frac{1}{n}X^TX$ are the *principal components* — the directions along which the data has maximum variance. The corresponding eigenvalues are the variances along those directions. Projecting data onto the top $k$ eigenvectors gives the best $k$-dimensional approximation (minimizing reconstruction error).

---

## 2. The Characteristic Polynomial

**Definition 2.7.2 (Characteristic Polynomial).** The *characteristic polynomial* of $A \in \mathbb{R}^{n \times n}$ is:

$$p_A(\lambda) = \det(A - \lambda I)$$

This is a polynomial of degree $n$ in $\lambda$. Its roots are the eigenvalues of $A$.

**Example 2.7.2:** Find the eigenvalues of $A = \begin{pmatrix} 5 & 2 \\ 2 & 1 \end{pmatrix}$ using the characteristic polynomial.

$$p_A(\lambda) = \det\begin{pmatrix} 5 - \lambda & 2 \\ 2 & 1 - \lambda \end{pmatrix} = (5 - \lambda)(1 - \lambda) - 4 = \lambda^2 - 6\lambda + 1$$

$$\lambda = \frac{6 \pm \sqrt{36 - 4}}{2} = \frac{6 \pm \sqrt{32}}{2} = 3 \pm 2\sqrt{2}$$

So $\lambda_1 = 3 + 2\sqrt{2} \approx 5.83$ and $\lambda_2 = 3 - 2\sqrt{2} \approx 0.17$. Both are positive (as expected for a positive definite symmetric matrix).

**Theorem 2.7.1.** The characteristic polynomial can be written as:

$$p_A(\lambda) = (-1)^n \lambda^n + (-1)^{n-1}\text{tr}(A)\lambda^{n-1} + \cdots + \det(A)$$

In particular:

- The sum of eigenvalues equals the trace: $\sum_{i=1}^n \lambda_i = \text{tr}(A)$
- The product of eigenvalues equals the determinant: $\prod_{i=1}^n \lambda_i = \det(A)$

*Proof of the trace relation.* The coefficient of $(-\lambda)^{n-1}$ in $\det(A - \lambda I)$ comes from choosing $n-1$ diagonal entries $-\lambda$ and one diagonal entry $a_{ii}$, giving $\sum_i a_{ii} = \text{tr}(A)$. Since each eigenvalue contributes to $\prod_i (\lambda_i - \lambda)$, expanding and matching coefficients gives $\sum \lambda_i = \text{tr}(A)$. $\square$

*Proof of the determinant relation.* Setting $\lambda = 0$: $p_A(0) = \det(A - 0 \cdot I) = \det(A)$. Also $p_A(0) = \prod_i (0 - \lambda_i) \cdot (-1)^n$... more carefully, if $p_A(\lambda) = (-1)^n \prod_i (\lambda - \lambda_i)$, then $p_A(0) = (-1)^n \prod_i (-\lambda_i) = \prod_i \lambda_i = \det(A)$. $\square$

**Example 2.7.3:** For $A = \begin{pmatrix} 4 & 2 \\ 1 & 3 \end{pmatrix}$ with eigenvalues $\lambda_1 = 5$, $\lambda_2 = 2$ (from Section 3.1), verify:

- **Trace:** $\text{tr}(A) = 4 + 3 = 7$ and $\lambda_1 + \lambda_2 = 5 + 2 = 7$ $\checkmark$
- **Determinant:** $\det(A) = 4 \cdot 3 - 2 \cdot 1 = 10$ and $\lambda_1 \cdot \lambda_2 = 5 \cdot 2 = 10$ $\checkmark$

These identities hold for any square matrix: the trace equals the sum of eigenvalues and the determinant equals their product.

*ML connection:* The trace-eigenvalue relation explains why the trace of the Hessian $\nabla^2 \mathcal{L}$ is used as a cheap proxy for the full eigenvalue spectrum in neural network training diagnostics. Computing $\text{tr}(H)$ requires only $n$ Hessian-vector products (via Hutchinson's trace estimator), while computing all eigenvalues requires $O(n^3)$.

---

## 3. Computing Eigenvalues

### 3.1 The 2x2 Case

For $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$:

$$p_A(\lambda) = \lambda^2 - (a+d)\lambda + (ad - bc) = \lambda^2 - \text{tr}(A)\lambda + \det(A)$$

By the quadratic formula:

$$\lambda = \frac{\text{tr}(A) \pm \sqrt{\text{tr}(A)^2 - 4\det(A)}}{2}$$

**Example.** $A = \begin{pmatrix} 4 & 2 \\ 1 & 3 \end{pmatrix}$:

$$\text{tr}(A) = 7, \quad \det(A) = 10, \quad \lambda = \frac{7 \pm \sqrt{49 - 40}}{2} = \frac{7 \pm 3}{2}$$

So $\lambda_1 = 5$, $\lambda_2 = 2$.

**Eigenvector for $\lambda_1 = 5$:** Solve $(A - 5I)\mathbf{v} = \mathbf{0}$:

$$\begin{pmatrix} -1 & 2 \\ 1 & -2 \end{pmatrix}\mathbf{v} = \mathbf{0} \implies \mathbf{v}_1 = \begin{pmatrix} 2 \\ 1 \end{pmatrix}$$

**Eigenvector for $\lambda_2 = 2$:** Solve $(A - 2I)\mathbf{v} = \mathbf{0}$:

$$\begin{pmatrix} 2 & 2 \\ 1 & 1 \end{pmatrix}\mathbf{v} = \mathbf{0} \implies \mathbf{v}_2 = \begin{pmatrix} -1 \\ 1 \end{pmatrix}$$

### 3.2 The 3x3 Case

**Example.** $A = \begin{pmatrix} 2 & 1 & 0 \\ 0 & 3 & 0 \\ 0 & 0 & 5 \end{pmatrix}$ (upper triangular).

For triangular matrices, the eigenvalues are the diagonal entries: $\lambda_1 = 2, \lambda_2 = 3, \lambda_3 = 5$.

*Proof.* $\det(A - \lambda I)$ is the determinant of a triangular matrix, which equals $\prod_i (a_{ii} - \lambda)$. This is zero iff $\lambda = a_{ii}$ for some $i$. $\square$

**General 3x3 example.** $A = \begin{pmatrix} 1 & 2 & 0 \\ 0 & 3 & 0 \\ 2 & 0 & 4 \end{pmatrix}$:

$$p_A(\lambda) = \det\begin{pmatrix} 1-\lambda & 2 & 0 \\ 0 & 3-\lambda & 0 \\ 2 & 0 & 4-\lambda \end{pmatrix}$$

Expanding along the second row (which has two zeros):

$$= (3 - \lambda)\det\begin{pmatrix} 1-\lambda & 0 \\ 2 & 4-\lambda \end{pmatrix} = (3-\lambda)(1-\lambda)(4-\lambda)$$

Eigenvalues: $\lambda = 1, 3, 4$.

### 3.3 Complex Eigenvalues

**Theorem 2.7.2 (Fundamental Theorem of Algebra).** Over $\mathbb{C}$, every $n \times n$ matrix has exactly $n$ eigenvalues (counted with multiplicity).

Over $\mathbb{R}$, eigenvalues may be complex. They always occur in conjugate pairs for real matrices: if $\lambda = a + bi$ is an eigenvalue of a real matrix, so is $\bar{\lambda} = a - bi$.

**Example.** Rotation by $90°$: $A = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$.

$$p_A(\lambda) = \lambda^2 + 1, \quad \lambda = \pm i$$

No real eigenvectors — rotation has no invariant direction in $\mathbb{R}^2$, which makes geometric sense.

*ML connection:* Complex eigenvalues arise in recurrent neural networks. The eigenvalues of the recurrence matrix $W$ in $\mathbf{h}_t = \sigma(W\mathbf{h}_{t-1} + U\mathbf{x}_t)$ determine the network's long-term memory behavior. Eigenvalues with $|\lambda| > 1$ cause exploding gradients; $|\lambda| < 1$ cause vanishing gradients. Unitary RNNs constrain $W$ to have $|\lambda| = 1$ for all eigenvalues, preventing both problems.

---

## 4. Eigenspaces and Multiplicity

**Definition 2.7.3 (Eigenspace).** The *eigenspace* of $A$ corresponding to eigenvalue $\lambda$ is:

$$E_\lambda = \ker(A - \lambda I) = \{\mathbf{v} \in \mathbb{R}^n : A\mathbf{v} = \lambda\mathbf{v}\}$$

This is a subspace of $\mathbb{R}^n$.

**Example 2.7.4:** For $A = \begin{pmatrix} 4 & 2 \\ 1 & 3 \end{pmatrix}$ with $\lambda_1 = 5$, find the eigenspace $E_5$.

$$A - 5I = \begin{pmatrix} -1 & 2 \\ 1 & -2 \end{pmatrix} \implies \text{row reduce} \implies \begin{pmatrix} 1 & -2 \\ 0 & 0 \end{pmatrix}$$

The free variable is $v_2 = t$, so $v_1 = 2t$. Thus $E_5 = \text{span}\left\{\begin{pmatrix} 2 \\ 1 \end{pmatrix}\right\}$, a 1-dimensional subspace (a line through the origin).

**Definition 2.7.4 (Algebraic Multiplicity).** The *algebraic multiplicity* $\text{alg}(\lambda)$ is the multiplicity of $\lambda$ as a root of the characteristic polynomial $p_A(\lambda)$.

**Definition 2.7.5 (Geometric Multiplicity).** The *geometric multiplicity* $\text{geo}(\lambda)$ is the dimension of the eigenspace: $\text{geo}(\lambda) = \dim(E_\lambda) = n - \text{rank}(A - \lambda I)$.

**Theorem 2.7.3.** For every eigenvalue $\lambda$:

$$1 \leq \text{geo}(\lambda) \leq \text{alg}(\lambda)$$

*Proof.* $\text{geo}(\lambda) \geq 1$ because eigenvalues have at least one eigenvector. For $\text{geo} \leq \text{alg}$: extend an eigenbasis of $E_\lambda$ to a basis of $\mathbb{R}^n$. In this basis, $A$ has a block structure where the first $\text{geo}(\lambda)$ diagonal entries are $\lambda$, contributing $(\lambda - t)^{\text{geo}(\lambda)}$ to the characteristic polynomial. The remaining block may contribute additional factors of $(\lambda - t)$, so $\text{alg}(\lambda) \geq \text{geo}(\lambda)$. $\square$

**Example where they differ.** $A = \begin{pmatrix} 2 & 1 \\ 0 & 2 \end{pmatrix}$ has $p_A(\lambda) = (2 - \lambda)^2$, so $\lambda = 2$ with $\text{alg}(2) = 2$.

But $A - 2I = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$ has rank $1$, so $\text{geo}(2) = 2 - 1 = 1$. Only one linearly independent eigenvector: $\mathbf{v} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$.

```
MULTIPLICITY COMPARISON

  Matrix A = [2 1; 0 2]              Matrix B = [2 0; 0 2] = 2I

  Eigenvalue λ = 2:                  Eigenvalue λ = 2:
  alg = 2, geo = 1                   alg = 2, geo = 2
  (defective — cannot diagonalize)   (every vector is an eigenvector!)

  Eigenspace: a line                 Eigenspace: all of R²
       ▲                                  ▲───────▶
       │                                  │      ╱
       │ v = (1,0)                        │    ╱
  ─────●─────▶                       ─────●──╱──▶
                                          │╱
```

---

## 5. Diagonalization

**Definition 2.7.6 (Diagonalizable Matrix).** $A$ is *diagonalizable* if there exists an invertible matrix $P$ and a diagonal matrix $D$ such that:

$$A = PDP^{-1}$$

equivalently $D = P^{-1}AP$. The columns of $P$ are eigenvectors, and the diagonal entries of $D$ are the corresponding eigenvalues.

**Theorem 2.7.4 (Diagonalization Criterion).** $A \in \mathbb{R}^{n \times n}$ is diagonalizable if and only if:

$$\text{geo}(\lambda_i) = \text{alg}(\lambda_i) \quad \text{for every eigenvalue } \lambda_i$$

Equivalently, $A$ has $n$ linearly independent eigenvectors.

*Proof.* ($\Rightarrow$) If $A = PDP^{-1}$, the columns of $P$ are $n$ linearly independent eigenvectors. For each eigenvalue, the number of columns with that eigenvalue on the diagonal equals $\text{alg}(\lambda_i)$, and these columns are independent eigenvectors in $E_{\lambda_i}$, so $\text{geo}(\lambda_i) \geq \text{alg}(\lambda_i)$. Combined with $\text{geo} \leq \text{alg}$, we get equality.

($\Leftarrow$) If $\text{geo}(\lambda_i) = \text{alg}(\lambda_i)$ for all $i$, the total number of linearly independent eigenvectors is $\sum \text{geo}(\lambda_i) = \sum \text{alg}(\lambda_i) = n$ (eigenvectors from different eigenspaces are independent by the next theorem). Form $P$ from these $n$ eigenvectors. $\square$

**Theorem 2.7.5 (Independence of Eigenvectors).** Eigenvectors corresponding to distinct eigenvalues are linearly independent.

*Proof.* By induction. Base case: one eigenvector is nonzero, hence independent. Inductive step: suppose $\mathbf{v}_1, \ldots, \mathbf{v}_k$ are independent eigenvectors for distinct $\lambda_1, \ldots, \lambda_k$. If $c_1\mathbf{v}_1 + \cdots + c_{k+1}\mathbf{v}_{k+1} = \mathbf{0}$, apply $A$ to get $c_1\lambda_1\mathbf{v}_1 + \cdots + c_{k+1}\lambda_{k+1}\mathbf{v}_{k+1} = \mathbf{0}$. Subtract $\lambda_{k+1}$ times the first equation: $c_1(\lambda_1 - \lambda_{k+1})\mathbf{v}_1 + \cdots + c_k(\lambda_k - \lambda_{k+1})\mathbf{v}_k = \mathbf{0}$. By the inductive hypothesis and distinctness of eigenvalues, $c_1 = \cdots = c_k = 0$, hence $c_{k+1} = 0$. $\square$

**Corollary 2.7.1.** If $A$ has $n$ distinct eigenvalues, it is diagonalizable.

**Example 2.7.5:** Diagonalize $A = \begin{pmatrix} 4 & 2 \\ 1 & 3 \end{pmatrix}$.

From Section 3.1: $\lambda_1 = 5$ with $\mathbf{v}_1 = \begin{pmatrix} 2 \\ 1 \end{pmatrix}$, $\lambda_2 = 2$ with $\mathbf{v}_2 = \begin{pmatrix} -1 \\ 1 \end{pmatrix}$.

$$P = \begin{pmatrix} 2 & -1 \\ 1 & 1 \end{pmatrix}, \quad D = \begin{pmatrix} 5 & 0 \\ 0 & 2 \end{pmatrix}, \quad P^{-1} = \frac{1}{3}\begin{pmatrix} 1 & 1 \\ -1 & 2 \end{pmatrix}$$

**Verify:** $PDP^{-1} = \begin{pmatrix} 2 & -1 \\ 1 & 1 \end{pmatrix}\begin{pmatrix} 5 & 0 \\ 0 & 2 \end{pmatrix}\frac{1}{3}\begin{pmatrix} 1 & 1 \\ -1 & 2 \end{pmatrix} = \begin{pmatrix} 10 & -2 \\ 5 & 2 \end{pmatrix}\frac{1}{3}\begin{pmatrix} 1 & 1 \\ -1 & 2 \end{pmatrix} = \frac{1}{3}\begin{pmatrix} 12 & 6 \\ 3 & 9 \end{pmatrix} = \begin{pmatrix} 4 & 2 \\ 1 & 3 \end{pmatrix} = A$ $\checkmark$

### 5.1 Why Diagonalization Matters

If $A = PDP^{-1}$, then $A^k = PD^kP^{-1}$, where $D^k = \text{diag}(\lambda_1^k, \ldots, \lambda_n^k)$.

```
POWER OF DIAGONALIZATION

Computing A^k directly: O(n³ · k) matrix multiplications

With diagonalization:
  A^k = P D^k P⁻¹

                    ┌ λ₁^k  0    0  ┐
  = P              │  0   λ₂^k  0  │  P⁻¹
                    └  0    0   λ₃^k┘

  Cost: O(n³) once for eigendecomposition + O(n) for diagonal powers
```

*ML connection:* Diagonalization is the mathematical foundation of PCA. The covariance matrix $\Sigma = PDP^T$ (where $P$ is orthogonal for symmetric $\Sigma$). The columns of $P$ are the principal directions, and $D$ contains the variances. Projecting data $\mathbf{x}$ onto the top $k$ columns of $P$ gives the optimal $k$-dimensional representation:

$$\text{PCA}_k(\mathbf{x}) = P_k^T(\mathbf{x} - \boldsymbol{\mu}) \quad \text{where } P_k = [\mathbf{v}_1 | \cdots | \mathbf{v}_k]$$

**Example 2.7.6:** Compute $A^3$ for $A = \begin{pmatrix} 4 & 2 \\ 1 & 3 \end{pmatrix}$ using diagonalization.

From Example 2.7.5: $A = PDP^{-1}$, so $A^3 = PD^3P^{-1}$.

$$D^3 = \begin{pmatrix} 5^3 & 0 \\ 0 & 2^3 \end{pmatrix} = \begin{pmatrix} 125 & 0 \\ 0 & 8 \end{pmatrix}$$

$$A^3 = \begin{pmatrix} 2 & -1 \\ 1 & 1 \end{pmatrix}\begin{pmatrix} 125 & 0 \\ 0 & 8 \end{pmatrix}\frac{1}{3}\begin{pmatrix} 1 & 1 \\ -1 & 2 \end{pmatrix} = \frac{1}{3}\begin{pmatrix} 250 & -8 \\ 125 & 8 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ -1 & 2 \end{pmatrix} = \frac{1}{3}\begin{pmatrix} 258 & 234 \\ 117 & 141 \end{pmatrix} = \begin{pmatrix} 86 & 78 \\ 39 & 47 \end{pmatrix}$$

This required only diagonalizing once and raising scalars to a power -- far cheaper than multiplying $A \cdot A \cdot A$ for large matrices or large exponents.

---

## 6. The Spectral Theorem

**Theorem 2.7.6 (Spectral Theorem for Real Symmetric Matrices).** If $A \in \mathbb{R}^{n \times n}$ is symmetric ($A = A^T$), then:

1. All eigenvalues of $A$ are real
2. Eigenvectors corresponding to distinct eigenvalues are orthogonal
3. $A$ is orthogonally diagonalizable: $A = Q\Lambda Q^T$ where $Q$ is orthogonal ($Q^TQ = I$) and $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$

*Proof of (1).* Let $\lambda$ be an eigenvalue with eigenvector $\mathbf{v}$ (possibly complex). Then $\lambda \bar{\mathbf{v}}^T \mathbf{v} = \bar{\mathbf{v}}^T A\mathbf{v}$. Since $A$ is real and symmetric, $\bar{\mathbf{v}}^T A \mathbf{v} = (A\bar{\mathbf{v}})^T\mathbf{v} = (\bar{\lambda}\bar{\mathbf{v}})^T\mathbf{v} = \bar{\lambda}\bar{\mathbf{v}}^T\mathbf{v}$. Since $\bar{\mathbf{v}}^T\mathbf{v} = \|\mathbf{v}\|^2 > 0$, we get $\lambda = \bar{\lambda}$, so $\lambda$ is real. $\square$

*Proof of (2).* Let $A\mathbf{v}_1 = \lambda_1\mathbf{v}_1$ and $A\mathbf{v}_2 = \lambda_2\mathbf{v}_2$ with $\lambda_1 \neq \lambda_2$. Then $\lambda_1 \langle \mathbf{v}_1, \mathbf{v}_2 \rangle = \langle A\mathbf{v}_1, \mathbf{v}_2 \rangle = \langle \mathbf{v}_1, A^T\mathbf{v}_2 \rangle = \langle \mathbf{v}_1, A\mathbf{v}_2 \rangle = \lambda_2 \langle \mathbf{v}_1, \mathbf{v}_2 \rangle$. So $(\lambda_1 - \lambda_2)\langle \mathbf{v}_1, \mathbf{v}_2 \rangle = 0$. Since $\lambda_1 \neq \lambda_2$, $\langle \mathbf{v}_1, \mathbf{v}_2 \rangle = 0$. $\square$

**Example 2.7.7:** Let $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$ (symmetric). Find eigenvalues and verify orthogonality.

$$p_A(\lambda) = (2 - \lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = (\lambda - 3)(\lambda - 1)$$

Eigenvalues $\lambda_1 = 3$, $\lambda_2 = 1$ -- both **real** (as the theorem guarantees).

- $\lambda_1 = 3$: $(A - 3I)\mathbf{v} = \begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix}\mathbf{v} = \mathbf{0}$ gives $\mathbf{v}_1 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$
- $\lambda_2 = 1$: $(A - I)\mathbf{v} = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}\mathbf{v} = \mathbf{0}$ gives $\mathbf{v}_2 = \begin{pmatrix} -1 \\ 1 \end{pmatrix}$

**Orthogonality check:** $\mathbf{v}_1 \cdot \mathbf{v}_2 = (1)(-1) + (1)(1) = 0$ $\checkmark$ -- eigenvectors are orthogonal, as guaranteed for symmetric matrices.

The spectral decomposition can be written as a sum of rank-1 projections:

$$A = \sum_{i=1}^n \lambda_i \mathbf{q}_i \mathbf{q}_i^T$$

where $\mathbf{q}_i$ are the orthonormal eigenvectors. Each term $\lambda_i \mathbf{q}_i \mathbf{q}_i^T$ is the projection onto the $i$-th eigendirection, scaled by $\lambda_i$.

**Example 2.7.8:** Decompose $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$ spectrally (continuing Example 2.7.7).

Normalize the eigenvectors: $\mathbf{q}_1 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$, $\mathbf{q}_2 = \frac{1}{\sqrt{2}}\begin{pmatrix} -1 \\ 1 \end{pmatrix}$.

$$A = 3 \cdot \mathbf{q}_1\mathbf{q}_1^T + 1 \cdot \mathbf{q}_2\mathbf{q}_2^T = 3 \cdot \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} + 1 \cdot \frac{1}{2}\begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}$$

$$= \begin{pmatrix} 3/2 & 3/2 \\ 3/2 & 3/2 \end{pmatrix} + \begin{pmatrix} 1/2 & -1/2 \\ -1/2 & 1/2 \end{pmatrix} = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix} = A \quad \checkmark$$

Each rank-1 term $\lambda_i \mathbf{q}_i\mathbf{q}_i^T$ captures the contribution of one eigendirection.

```
SPECTRAL DECOMPOSITION (2D symmetric matrix)

A = λ₁ q₁q₁ᵀ + λ₂ q₂q₂ᵀ

         q₂ (λ₂ = 1)
         ▲
         │
    ─────●─────▶ q₁ (λ₁ = 3)
         │

A stretches by factor 3 along q₁ and factor 1 along q₂

The ellipse Ax·x = 1 has axes along q₁, q₂
with semi-axis lengths 1/√λ₁, 1/√λ₂
```

*ML connection:* **Spectral clustering** uses the eigenvectors of the graph Laplacian $L = D - W$ (where $W$ is the adjacency matrix and $D$ is the degree matrix). The Laplacian is symmetric and positive semi-definite. Its smallest eigenvalues correspond to the "smoothest" functions on the graph — eigenvectors that change slowly across edges. The second-smallest eigenvector (the Fiedler vector) gives the optimal bipartition. Using $k$ eigenvectors for $k$-means gives spectral clustering.

---

## 7. Positive Definite Matrices

**Definition 2.7.7 (Positive Definite).** A symmetric matrix $A$ is *positive definite* (PD) if:

$$\mathbf{x}^T A \mathbf{x} > 0 \quad \text{for all } \mathbf{x} \neq \mathbf{0}$$

It is *positive semi-definite* (PSD) if $\mathbf{x}^T A \mathbf{x} \geq 0$ for all $\mathbf{x}$.

**Theorem 2.7.7 (Characterizations of Positive Definiteness).** For a symmetric matrix $A$, the following are equivalent:

1. $A$ is positive definite
2. All eigenvalues of $A$ are positive: $\lambda_i > 0$ for all $i$
3. All leading principal minors are positive (Sylvester's criterion)
4. There exists an invertible matrix $L$ such that $A = L^TL$ (Cholesky-like)
5. All pivots in Gaussian elimination (without row swaps) are positive

*Proof of $(1) \Leftrightarrow (2)$.* ($\Rightarrow$) If $A\mathbf{v} = \lambda\mathbf{v}$ with $\mathbf{v} \neq \mathbf{0}$, then $\lambda\|\mathbf{v}\|^2 = \mathbf{v}^TA\mathbf{v} > 0$, so $\lambda > 0$. ($\Leftarrow$) Write $A = Q\Lambda Q^T$. Then $\mathbf{x}^TA\mathbf{x} = \mathbf{y}^T\Lambda\mathbf{y} = \sum_i \lambda_i y_i^2 > 0$ where $\mathbf{y} = Q^T\mathbf{x} \neq \mathbf{0}$ (since $Q$ is invertible). $\square$

**Example 2.7.9:** Is $A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$ positive definite?

$$p_A(\lambda) = (3 - \lambda)^2 - 1 = \lambda^2 - 6\lambda + 8 = (\lambda - 4)(\lambda - 2)$$

Eigenvalues: $\lambda_1 = 4 > 0$, $\lambda_2 = 2 > 0$. Both positive, so $A$ is **positive definite**.

**Cross-check with Sylvester's criterion:** Leading minors are $a_{11} = 3 > 0$ and $\det(A) = 9 - 1 = 8 > 0$. $\checkmark$

**Cross-check with the definition:** For $\mathbf{x} = \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$: $\mathbf{x}^TA\mathbf{x} = 3x_1^2 + 2x_1 x_2 + 3x_2^2 = 2x_1^2 + (x_1 + x_2)^2 + 2x_2^2 > 0$ for all $\mathbf{x} \neq \mathbf{0}$. $\checkmark$

| Property | Positive Definite | Positive Semi-Definite | Indefinite |
|----------|------------------|----------------------|------------|
| Eigenvalues | All $> 0$ | All $\geq 0$ | Mixed signs |
| $\mathbf{x}^TA\mathbf{x}$ | $> 0$ ($\mathbf{x} \neq 0$) | $\geq 0$ | Can be $+$ or $-$ |
| Determinant | $> 0$ | $\geq 0$ | Can be $+$ or $-$ |
| Invertible? | Yes | Not necessarily | Depends |
| Example | $\begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$ | $\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$ | $\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ |

*ML connection:* **Covariance matrices are always positive semi-definite** because $\mathbf{x}^T\Sigma\mathbf{x} = \mathbf{x}^T \mathbb{E}[(\mathbf{z}-\boldsymbol{\mu})(\mathbf{z}-\boldsymbol{\mu})^T]\mathbf{x} = \mathbb{E}[(\mathbf{x}^T(\mathbf{z}-\boldsymbol{\mu}))^2] \geq 0$. The Hessian $\nabla^2 \mathcal{L}$ of a loss function at a local minimum is PSD. If it is PD, the minimum is *strict* (isolated). If it has zero eigenvalues, there are flat directions — common in overparameterized neural networks where many parameter configurations achieve the same loss.

---

## 8. The Rayleigh Quotient

**Definition 2.7.8 (Rayleigh Quotient).** For symmetric $A$ and nonzero $\mathbf{x}$:

$$R(\mathbf{x}) = \frac{\mathbf{x}^T A \mathbf{x}}{\mathbf{x}^T \mathbf{x}}$$

**Theorem 2.7.8 (Min-Max Characterization).** For symmetric $A$ with eigenvalues $\lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$:

$$\lambda_1 = \min_{\mathbf{x} \neq \mathbf{0}} R(\mathbf{x}), \quad \lambda_n = \max_{\mathbf{x} \neq \mathbf{0}} R(\mathbf{x})$$

The minimum is attained at the eigenvector for $\lambda_1$, and the maximum at the eigenvector for $\lambda_n$.

*Proof.* Write $\mathbf{x} = \sum c_i \mathbf{q}_i$ in the orthonormal eigenbasis. Then $R(\mathbf{x}) = \frac{\sum \lambda_i c_i^2}{\sum c_i^2}$. This is a weighted average of the $\lambda_i$ with weights $c_i^2/\sum c_j^2$, so $\lambda_1 \leq R(\mathbf{x}) \leq \lambda_n$, with equality when $\mathbf{x}$ is the corresponding eigenvector. $\square$

**Example 2.7.10:** For $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$ (eigenvalues $\lambda_1 = 1$, $\lambda_2 = 3$ from Example 2.7.7), compute the Rayleigh quotient at several points.

- At eigenvector $\mathbf{x} = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$: $R = \frac{(1,1)\begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}\begin{pmatrix} 1 \\ 1 \end{pmatrix}}{(1,1)\begin{pmatrix} 1 \\ 1 \end{pmatrix}} = \frac{6}{2} = 3 = \lambda_{\max}$
- At eigenvector $\mathbf{x} = \begin{pmatrix} -1 \\ 1 \end{pmatrix}$: $R = \frac{(-1,1)\begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}\begin{pmatrix} -1 \\ 1 \end{pmatrix}}{(-1,1)\begin{pmatrix} -1 \\ 1 \end{pmatrix}} = \frac{2}{2} = 1 = \lambda_{\min}$
- At non-eigenvector $\mathbf{x} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$: $R = \frac{(1,0)\begin{pmatrix} 2 \\ 1 \end{pmatrix}}{1} = 2$, which satisfies $1 \leq 2 \leq 3$ $\checkmark$

*ML connection:* PCA is exactly the Rayleigh quotient optimization. Finding the direction of maximum variance is:

$$\mathbf{v}_1 = \arg\max_{\|\mathbf{v}\| = 1} \mathbf{v}^T \Sigma \mathbf{v} = \arg\max_{\mathbf{v} \neq 0} R_\Sigma(\mathbf{v})$$

The second principal component maximizes $R_\Sigma(\mathbf{v})$ subject to $\mathbf{v} \perp \mathbf{v}_1$, and so on. This is the *Courant-Fischer min-max theorem* in action.

---

## 9. Power Iteration

**Algorithm 2.7.1 (Power Iteration).** To find the largest eigenvalue $\lambda_1$ (in absolute value) and its eigenvector:

1. Start with a random vector $\mathbf{b}_0$
2. Iterate: $\mathbf{b}_{k+1} = \frac{A\mathbf{b}_k}{\|A\mathbf{b}_k\|}$
3. $\lambda_1 \approx \mathbf{b}_k^T A \mathbf{b}_k$ (Rayleigh quotient)

**Theorem 2.7.9 (Convergence of Power Iteration).** If $|\lambda_1| > |\lambda_2| \geq \cdots \geq |\lambda_n|$ and $\mathbf{b}_0$ has a nonzero component along $\mathbf{v}_1$, then $\mathbf{b}_k$ converges to the eigenvector $\mathbf{v}_1$ at rate $|\lambda_2/\lambda_1|^k$.

*Proof sketch.* Write $\mathbf{b}_0 = c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_n\mathbf{v}_n$. Then $A^k \mathbf{b}_0 = c_1\lambda_1^k\mathbf{v}_1 + c_2\lambda_2^k\mathbf{v}_2 + \cdots$. After normalization:

$$\mathbf{b}_k \approx \frac{c_1\lambda_1^k\mathbf{v}_1 + c_2\lambda_2^k\mathbf{v}_2 + \cdots}{|c_1||\lambda_1|^k\|\mathbf{v}_1\|}$$

The ratio $|\lambda_2/\lambda_1|^k \to 0$ as $k \to \infty$, so the $\mathbf{v}_2, \ldots, \mathbf{v}_n$ components decay exponentially. $\square$

**Example 2.7.11:** Apply power iteration to $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$ (eigenvalues $3$ and $1$) starting from $\mathbf{b}_0 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$.

- **Step 1:** $A\mathbf{b}_0 = \begin{pmatrix} 2 \\ 1 \end{pmatrix}$, $\|A\mathbf{b}_0\| = \sqrt{5}$, $\mathbf{b}_1 = \frac{1}{\sqrt{5}}\begin{pmatrix} 2 \\ 1 \end{pmatrix} \approx \begin{pmatrix} 0.894 \\ 0.447 \end{pmatrix}$
- **Step 2:** $A\mathbf{b}_1 = \frac{1}{\sqrt{5}}\begin{pmatrix} 5 \\ 4 \end{pmatrix}$, $\mathbf{b}_2 \approx \begin{pmatrix} 0.781 \\ 0.625 \end{pmatrix}$
- **Step 3:** $A\mathbf{b}_2 \approx \begin{pmatrix} 2.187 \\ 2.031 \end{pmatrix}$, $\mathbf{b}_3 \approx \begin{pmatrix} 0.733 \\ 0.680 \end{pmatrix}$

After more iterations, $\mathbf{b}_k \to \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$ (the eigenvector for $\lambda = 3$). Convergence rate: $|\lambda_2/\lambda_1| = 1/3$, so error decays by factor $1/3$ per step.

```
POWER ITERATION — CONVERGENCE

   b₀ (random)        b₁ = Ab₀/‖Ab₀‖      b₂ = Ab₁/‖Ab₁‖     bₖ → v₁

     ↗                   ↗                    ↗                   ▲
    ╱  (random          ╱  (closer to         ╱  (even closer)    │  (dominant
   ╱   direction)      ╱   eigenvector)      ╱                    │  eigenvector)
  ●                   ●                     ●                     ●

  Convergence rate: |λ₂/λ₁|^k
  Fast when λ₁ ≫ λ₂ (dominant eigenvalue well-separated)
  Slow when λ₁ ≈ λ₂ (use shift-and-invert or Lanczos instead)
```

*ML connection:* **PageRank** is power iteration applied to the web graph. Google's original algorithm computes the dominant eigenvector of the matrix $M = \alpha S + (1-\alpha)\frac{1}{n}\mathbf{1}\mathbf{1}^T$, where $S$ is the column-stochastic link matrix and $\alpha \approx 0.85$ is the damping factor. The dominant eigenvalue is $\lambda_1 = 1$ (guaranteed by the Perron-Frobenius theorem for positive stochastic matrices), and the corresponding eigenvector gives the PageRank scores. With billions of web pages, power iteration is attractive because each step is a sparse matrix-vector multiply — $O(\text{nnz})$ where $\text{nnz}$ is the number of hyperlinks.

---

## 10. Applications in ML and AI

### 10.1 PCA (Principal Component Analysis)

Given data $\mathbf{x}_1, \ldots, \mathbf{x}_m \in \mathbb{R}^n$, PCA finds the directions of maximum variance.

1. Center: $\bar{\mathbf{x}} = \frac{1}{m}\sum \mathbf{x}_i$, $\tilde{\mathbf{x}}_i = \mathbf{x}_i - \bar{\mathbf{x}}$
2. Covariance: $\Sigma = \frac{1}{m}\sum \tilde{\mathbf{x}}_i \tilde{\mathbf{x}}_i^T = \frac{1}{m}\tilde{X}^T\tilde{X}$
3. Eigendecompose: $\Sigma = Q\Lambda Q^T$, sort $\lambda_1 \geq \lambda_2 \geq \cdots$
4. Project: $\mathbf{z}_i = Q_k^T \tilde{\mathbf{x}}_i$ where $Q_k$ has the top $k$ eigenvectors

**Proportion of variance retained:** $\frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^n \lambda_i}$

```
PCA — FINDING THE BEST 1D PROJECTION

  Original data in 2D              Projected onto 1st PC

       ·  ·   ·                          ·
    ·   ·   ·   ·                       · ·
   ·  ·  ·  ·  ·  ·    ───▶     ·  · · · · · · ·  ·
    ·   ·   ·   ·               ────────────────────────
       ·  ·   ·                   v₁ (max variance direction)

  λ₁ = variance along v₁          λ₁/(λ₁+λ₂) = fraction of
  λ₂ = variance along v₂          variance retained
```

### 10.2 Graph Neural Networks and the Graph Laplacian

The graph Laplacian $L = D - W$ is real symmetric and PSD. Its eigendecomposition $L = U\Lambda U^T$ provides:

- **Graph Fourier Transform:** Project a signal $\mathbf{f}$ on the graph to $\hat{\mathbf{f}} = U^T\mathbf{f}$
- **Spectral convolution:** Graph convolution in the spectral domain is pointwise multiplication: $\mathbf{g} *_G \mathbf{f} = U(U^T\mathbf{g} \odot U^T\mathbf{f})$
- **ChebNet / GCN:** Approximate spectral filters with polynomials of $L$ to avoid explicit eigendecomposition: $g_\theta(L)\mathbf{f} \approx \sum_{k=0}^K \theta_k T_k(\tilde{L})\mathbf{f}$ (Chebyshev polynomials)

The eigenvalues of $L$ are the graph's "frequencies": $\lambda = 0$ corresponds to constant signals, and large $\lambda$ correspond to high-frequency (rapidly oscillating) signals.

### 10.3 Hessian Eigenvalues and the Loss Landscape

The Hessian $H = \nabla^2 \mathcal{L}(\theta)$ at a critical point of the loss function reveals the local geometry:

| Hessian Eigenvalue Spectrum | Interpretation |
|---------------------------|----------------|
| All $\lambda_i > 0$ | Strict local minimum (convex bowl) |
| All $\lambda_i < 0$ | Strict local maximum |
| Mixed signs | Saddle point |
| Some $\lambda_i = 0$ | Flat directions (degenerate critical point) |
| $\lambda_{\max} / \lambda_{\min} \gg 1$ | Ill-conditioned (elongated valley) |

In modern deep learning, the Hessian is too large ($n = 10^8$ parameters) to compute explicitly. Instead, researchers use:

- **Lanczos algorithm** to find the top-$k$ and bottom-$k$ eigenvalues
- **Hessian-vector products** $H\mathbf{v}$ (computed via autodiff in $O(n)$) as the primitive
- **Stochastic trace estimation** for $\text{tr}(H) = \sum \lambda_i$

### 10.4 Spectral Clustering

```
SPECTRAL CLUSTERING PIPELINE

  1. Build similarity       2. Compute graph        3. Eigendecompose
     graph W                   Laplacian L = D-W       L = UΛUᵀ

  ●───●     ●               ┌─────────────┐        Take k smallest
  │ ╲ │    ╱│               │             │        eigenvectors
  │  ╲│   ╱ │               │  L = D - W  │        (skip λ₁ = 0)
  ●───●──●  │               │             │
       ╲   ╱                └─────────────┘
        ● ●                                         u₂, u₃, ..., uₖ

  4. Embed in Rᵏ            5. Run k-means in
     using eigenvectors        the new space

     ·  · ·                    Cluster 1: ● ● ●
     · ·                       Cluster 2: ■ ■ ■
         ■ ■ ■                 (well-separated in
         ■ ■                    spectral embedding)
```

### 10.5 Quantum Computing: Eigenvalues of Hamiltonians

In quantum mechanics, the eigenvalues of the Hamiltonian operator $H$ are the allowed energy levels of the system. Quantum phase estimation (QPE) is an algorithm that estimates eigenvalues of a unitary operator — it is the key subroutine in Shor's factoring algorithm and quantum simulation.

The variational quantum eigensolver (VQE) uses a parameterized quantum circuit to find the ground state energy (smallest eigenvalue) of a molecular Hamiltonian, with applications to drug discovery and materials science.

---

## 11. Important Special Cases

### 11.1 Stochastic Matrices

A *column-stochastic* matrix has non-negative entries and columns summing to $1$. By the **Perron-Frobenius theorem**, the dominant eigenvalue is $\lambda_1 = 1$, and the corresponding eigenvector (the *stationary distribution*) has non-negative entries.

### 11.2 Normal Matrices

A matrix $A$ is *normal* if $A^TA = AA^T$ (or $A^*A = AA^*$ over $\mathbb{C}$). Normal matrices include symmetric, skew-symmetric, orthogonal, and unitary matrices. The spectral theorem generalizes: normal matrices are unitarily diagonalizable.

### 11.3 Relationship Summary

| Matrix Type | Eigenvalue Properties | Diagonalizable? |
|-------------|----------------------|-----------------|
| Symmetric ($A = A^T$) | All real | Yes (orthogonally) |
| Skew-symmetric ($A = -A^T$) | Purely imaginary or $0$ | Yes (unitarily) |
| Orthogonal ($Q^TQ = I$) | $|\lambda| = 1$ | Yes (unitarily) |
| Positive definite | All real and $> 0$ | Yes (orthogonally) |
| Stochastic | $|\lambda| \leq 1$, $\lambda_1 = 1$ | Usually |
| Nilpotent ($A^k = 0$) | All $\lambda = 0$ | Only if $A = 0$ |
| Defective | $\text{geo} < \text{alg}$ for some $\lambda$ | No |

---

## 12. Summary of Key Results

| Result | Statement |
|--------|-----------|
| Eigenvalue equation | $A\mathbf{v} = \lambda\mathbf{v}$, $\mathbf{v} \neq \mathbf{0}$ |
| Characteristic polynomial | $p_A(\lambda) = \det(A - \lambda I) = 0$ |
| Trace-eigenvalue | $\text{tr}(A) = \sum \lambda_i$ |
| Determinant-eigenvalue | $\det(A) = \prod \lambda_i$ |
| Independence | Eigenvectors for distinct eigenvalues are independent |
| Diagonalization | $A = PDP^{-1}$ iff $\text{geo} = \text{alg}$ for all eigenvalues |
| Spectral theorem | Symmetric $A = Q\Lambda Q^T$ with orthogonal $Q$ |
| Positive definite | PD $\iff$ all eigenvalues positive |
| Rayleigh quotient | $\lambda_{\min} \leq R(\mathbf{x}) \leq \lambda_{\max}$ |
| Power iteration | Converges to dominant eigenvector at rate $|\lambda_2/\lambda_1|^k$ |

---

## Exercises

**★ Basic**

1. Find all eigenvalues and eigenvectors of $A = \begin{pmatrix} 5 & 4 \\ 1 & 2 \end{pmatrix}$.

2. The matrix $A = \begin{pmatrix} 3 & 0 & 0 \\ 0 & 3 & 0 \\ 0 & 0 & 7 \end{pmatrix}$ is diagonal. Write down its eigenvalues and eigenvectors without computation.

3. Verify that $\text{tr}(A) = \lambda_1 + \lambda_2$ and $\det(A) = \lambda_1\lambda_2$ for $A = \begin{pmatrix} 1 & 3 \\ 4 & 2 \end{pmatrix}$.

4. Is the matrix $B = \begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix}$ positive definite? Find its eigenvalues and check.

**★★ Intermediate**

5. Show that if $\lambda$ is an eigenvalue of $A$, then $\lambda^2$ is an eigenvalue of $A^2$, and $\lambda^{-1}$ is an eigenvalue of $A^{-1}$ (assuming $A$ invertible).

6. Prove that a real matrix $A$ and its transpose $A^T$ have the same eigenvalues. Do they have the same eigenvectors?

7. The matrix $A = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 1 & 0 & 0 \end{pmatrix}$ is a permutation matrix. Find its eigenvalues. (Hint: what is $A^3$?)

8. Let $A$ be symmetric with eigenvalues $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n > 0$. Show that the condition number $\kappa(A) = \lambda_1/\lambda_n$ controls the convergence rate of gradient descent on $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^TA\mathbf{x} - \mathbf{b}^T\mathbf{x}$.

9. **(PCA)** Given data points $(1,2), (3,4), (5,6)$ in $\mathbb{R}^2$: center the data, compute the covariance matrix, find its eigenvalues and eigenvectors, and identify the first principal component. What fraction of variance does it capture?

**★★★ Challenging**

10. **(PageRank)** Consider a web with 3 pages and link matrix $S = \begin{pmatrix} 0 & 0 & 1 \\ 1 & 0 & 0 \\ 0 & 1 & 0 \end{pmatrix}$. With damping factor $\alpha = 0.85$, compute the PageRank matrix $M = \alpha S + (1-\alpha)\frac{1}{3}\mathbf{1}\mathbf{1}^T$ and find its dominant eigenvector by power iteration (3 iterations, starting from $\mathbf{b}_0 = (1,0,0)^T$).

11. **(Spectral gap and mixing time)** For a Markov chain with transition matrix $P$, prove that the mixing time to stationarity is $O\left(\frac{1}{1-|\lambda_2|}\log\frac{1}{\epsilon}\right)$ where $\lambda_2$ is the second-largest eigenvalue in absolute value. Explain why a large spectral gap ($1 - |\lambda_2|$) means fast mixing.

12. **(Hessian analysis)** Consider the loss $\mathcal{L}(\theta_1, \theta_2) = \theta_1^2 + 100\theta_2^2$. Compute the Hessian, find its eigenvalues, determine the condition number, and explain why gradient descent with a fixed learning rate would be slow. How does this relate to the "elongated valley" problem in neural network optimization?

13. Prove that for a symmetric matrix $A$, if $\|\mathbf{x}\| = 1$ and $A\mathbf{x} = \lambda\mathbf{x} + \mathbf{r}$ where $\|\mathbf{r}\|$ is small, then $A$ has an eigenvalue within $\|\mathbf{r}\|$ of $\lambda$. This is the *Bauer-Fike theorem* for symmetric matrices and explains why approximate eigenvectors give good eigenvalue estimates.

---

## Related Topics

- [Vectors](vectors.md) — the objects that eigenvectors are
- [Determinants](determinants.md) — $\det(A) = \prod \lambda_i$ and the characteristic polynomial
- [Inner Product Spaces](inner-product-spaces.md) — orthogonality, Gram-Schmidt for eigenvector computation
- [Matrix Decompositions](matrix-decompositions.md) — SVD, which generalizes eigendecomposition to rectangular matrices
- [Multivariable Calculus](../calculus/multivariable-calculus.md) — Hessian matrices and second-order analysis
- [Optimization](../optimization/index.md) — eigenvalues of the Hessian control convergence rates

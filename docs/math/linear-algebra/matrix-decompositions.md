# Matrix Decompositions

Every major algorithm in machine learning decomposes a matrix. PCA decomposes the covariance matrix via eigenvalues. Recommender systems decompose user-item matrices via SVD. Solving linear systems uses LU or QR. Training neural networks requires solving least squares subproblems, often via Cholesky. This chapter develops the four fundamental decompositions --- LU, QR, Cholesky, and SVD --- with complete proofs, then derives the pseudoinverse and its central role in machine learning.

---

## Prerequisites

- [Vectors](vectors.md) --- norms, inner products
- [Eigenvalues & Eigenvectors](eigenvalues.md) --- diagonalisation, spectral theorem
- [Inner Product Spaces](inner-product-spaces.md) --- Gram-Schmidt, projections, least squares

---

## 1. LU Decomposition

**Definition 2.9.1 (LU Decomposition).** A matrix $A \in \mathbb{R}^{n \times n}$ has an *LU decomposition* if it can be written as:

$$A = LU$$

where $L$ is lower triangular with ones on the diagonal (unit lower triangular) and $U$ is upper triangular.

```
LU STRUCTURE

    ┌             ┐   ┌             ┐ ┌             ┐
    │ a₁₁ a₁₂ a₁₃│   │  1   0   0 │ │ u₁₁ u₁₂ u₁₃│
    │ a₂₁ a₂₂ a₂₃│ = │ l₂₁  1   0 │ │  0  u₂₂ u₂₃│
    │ a₃₁ a₃₂ a₃₃│   │ l₃₁ l₃₂  1 │ │  0   0  u₃₃│
    └             ┘   └             ┘ └             ┘
          A                 L               U
```

**Theorem 2.9.1 (Existence of LU).** If all leading principal minors of $A$ are nonzero (i.e., $\det(A_{1:k, 1:k}) \neq 0$ for $k = 1, \dots, n$), then the LU decomposition exists and is unique.

*Proof sketch.* Gaussian elimination without row swaps produces $U$, and the multipliers $l_{ij}$ used during elimination form $L$. The condition on leading minors ensures no zero pivot is encountered. Uniqueness: if $A = L_1 U_1 = L_2 U_2$, then $L_2^{-1}L_1 = U_2 U_1^{-1}$. The left side is unit lower triangular and the right side is upper triangular; a matrix that is both must be $I$. $\square$

**Example 2.9.1:** Factor $A = \begin{pmatrix} 2 & 1 & 1 \\ 4 & 3 & 3 \\ 8 & 7 & 9 \end{pmatrix}$ into $A = LU$.

**Step 1 --- Eliminate column 1.** Multipliers: $l_{21} = 4/2 = 2$, $l_{31} = 8/2 = 4$.

$$R_2 \leftarrow R_2 - 2R_1: \quad (4,3,3) - 2(2,1,1) = (0,1,1)$$

$$R_3 \leftarrow R_3 - 4R_1: \quad (8,7,9) - 4(2,1,1) = (0,3,5)$$

**Step 2 --- Eliminate column 2.** Multiplier: $l_{32} = 3/1 = 3$.

$$R_3 \leftarrow R_3 - 3R_2: \quad (0,3,5) - 3(0,1,1) = (0,0,2)$$

**Result:**

$$L = \begin{pmatrix} 1 & 0 & 0 \\ 2 & 1 & 0 \\ 4 & 3 & 1 \end{pmatrix}, \quad U = \begin{pmatrix} 2 & 1 & 1 \\ 0 & 1 & 1 \\ 0 & 0 & 2 \end{pmatrix}$$

**Verification:** $LU = \begin{pmatrix} 2 & 1 & 1 \\ 4 & 3 & 3 \\ 8 & 7 & 9 \end{pmatrix} = A$. $\checkmark$

**Definition 2.9.2 (PLU Decomposition).** For *any* invertible matrix $A$, there exists a permutation matrix $P$ such that:

$$PA = LU$$

The permutation matrix $P$ encodes the row swaps (partial pivoting) needed during Gaussian elimination.

### 1.1 Solving Linear Systems with LU

To solve $A\mathbf{x} = \mathbf{b}$:

1. Decompose: $A = LU$ (cost: $\frac{2}{3}n^3$ flops)
2. Forward substitution: solve $L\mathbf{y} = \mathbf{b}$ (cost: $n^2$ flops)
3. Back substitution: solve $U\mathbf{x} = \mathbf{y}$ (cost: $n^2$ flops)

**Advantage:** Once $LU$ is computed, solving for *additional* right-hand sides $\mathbf{b}'$ costs only $O(n^2)$.

**Example 2.9.2:** Using the LU factors from Example 2.9.1, solve $A\mathbf{x} = \mathbf{b}$ where $\mathbf{b} = (4, 10, 34)^T$.

**Forward substitution** --- solve $L\mathbf{y} = \mathbf{b}$:

$$y_1 = 4, \quad y_2 = 10 - 2(4) = 2, \quad y_3 = 34 - 4(4) - 3(2) = 12$$

So $\mathbf{y} = (4, 2, 12)^T$.

**Back substitution** --- solve $U\mathbf{x} = \mathbf{y}$:

$$x_3 = 12/2 = 6, \quad x_2 = (2 - 1 \cdot 6)/1 = -4, \quad x_1 = (4 - 1(-4) - 1(6))/2 = 1$$

**Solution:** $\mathbf{x} = (1, -4, 6)^T$. **Verify:** $A\mathbf{x} = \begin{pmatrix} 2+(-4)+6 \\ 4+(-12)+18 \\ 8+(-28)+54 \end{pmatrix} = \begin{pmatrix} 4 \\ 10 \\ 34 \end{pmatrix} = \mathbf{b}$. $\checkmark$

| Method | Cost | When to use |
|--------|------|-------------|
| Direct inversion $A^{-1}\mathbf{b}$ | $O(n^3)$ per solve | Never (numerically unstable) |
| LU decomposition | $O(n^3)$ once + $O(n^2)$ per solve | Multiple right-hand sides |
| Gaussian elimination | $O(n^3)$ per solve | Single system |

*ML connection:* LU decomposition is used internally by **NumPy/SciPy** when you call `np.linalg.solve()`. In Gaussian processes, solving $(K + \sigma^2 I)\boldsymbol{\alpha} = \mathbf{y}$ for predictions uses Cholesky (a specialised LU for positive definite matrices; see Section 3).

---

## 2. QR Decomposition

**Definition 2.9.3 (QR Decomposition).** Any matrix $A \in \mathbb{R}^{m \times n}$ with $m \geq n$ and linearly independent columns can be factored as:

$$A = QR$$

where $Q \in \mathbb{R}^{m \times n}$ has orthonormal columns ($Q^TQ = I_n$) and $R \in \mathbb{R}^{n \times n}$ is upper triangular with positive diagonal entries.

### 2.1 Construction via Gram-Schmidt

*Proof (constructive).* Let $\mathbf{a}_1, \dots, \mathbf{a}_n$ be the columns of $A$. Apply Gram-Schmidt (Theorem 2.8.5) to produce orthonormal vectors $\mathbf{q}_1, \dots, \mathbf{q}_n$.

By the Gram-Schmidt construction:

$$\mathbf{a}_j = \sum_{i=1}^{j} \langle \mathbf{a}_j, \mathbf{q}_i \rangle \, \mathbf{q}_i$$

Define $r_{ij} = \langle \mathbf{a}_j, \mathbf{q}_i \rangle$ for $i \leq j$ and $r_{ij} = 0$ for $i > j$. Then $\mathbf{a}_j = \sum_{i=1}^j r_{ij} \mathbf{q}_i$, which in matrix form is $A = QR$.

$R$ is upper triangular by construction. The diagonal entries $r_{jj} = \|\mathbf{u}_j\| > 0$ (where $\mathbf{u}_j$ are the unnormalised Gram-Schmidt vectors) since the columns are independent. $\square$

**Example 2.9.3:** Compute the QR decomposition of $A = \begin{pmatrix} 1 & 1 \\ 1 & 0 \\ 0 & 1 \end{pmatrix}$.

**Step 1 --- First column.** $\mathbf{a}_1 = (1, 1, 0)^T$. Normalise: $\|\mathbf{a}_1\| = \sqrt{2}$, so $\mathbf{q}_1 = \frac{1}{\sqrt{2}}(1, 1, 0)^T$.

**Step 2 --- Second column.** $\mathbf{a}_2 = (1, 0, 1)^T$. Project out $\mathbf{q}_1$:

$$\langle \mathbf{a}_2, \mathbf{q}_1 \rangle = \frac{1}{\sqrt{2}}(1 + 0 + 0) = \frac{1}{\sqrt{2}}$$

$$\mathbf{u}_2 = \mathbf{a}_2 - \frac{1}{\sqrt{2}}\mathbf{q}_1 = \begin{pmatrix} 1 \\ 0 \\ 1 \end{pmatrix} - \frac{1}{2}\begin{pmatrix} 1 \\ 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 1/2 \\ -1/2 \\ 1 \end{pmatrix}$$

Normalise: $\|\mathbf{u}_2\| = \sqrt{1/4 + 1/4 + 1} = \sqrt{3/2}$, so $\mathbf{q}_2 = \frac{1}{\sqrt{6}}(1, -1, 2)^T$.

**Result:**

$$Q = \begin{pmatrix} 1/\sqrt{2} & 1/\sqrt{6} \\ 1/\sqrt{2} & -1/\sqrt{6} \\ 0 & 2/\sqrt{6} \end{pmatrix}, \quad R = \begin{pmatrix} \sqrt{2} & 1/\sqrt{2} \\ 0 & \sqrt{3/2} \end{pmatrix}$$

**Verification:** $QR = \begin{pmatrix} \frac{\sqrt{2}}{\sqrt{2}} & \frac{1}{2} + \frac{1}{\sqrt{6}} \cdot \sqrt{\frac{3}{2}} \\ \frac{\sqrt{2}}{\sqrt{2}} & \frac{1}{2} - \frac{1}{\sqrt{6}} \cdot \sqrt{\frac{3}{2}} \\ 0 & \frac{2}{\sqrt{6}} \cdot \sqrt{\frac{3}{2}} \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 1 & 0 \\ 0 & 1 \end{pmatrix} = A$. $\checkmark$

```
QR GEOMETRY IN R³

    Columns of A          Columns of Q
    (independent)          (orthonormal)

       a₂                    q₂
      ╱                      │
     ╱   a₃         q₃      │
    ╱   ╱             ╲      │
   ╱   ╱      →→→      ╲    │
  ╱   ╱     Gram-        ╲   │
 ╱   ╱      Schmidt       ╲──│── q₁
────── a₁
                     All angles = 90°, all lengths = 1
```

### 2.2 Full vs. Thin QR

**Definition 2.9.4 (Full QR).** The *full* QR decomposition writes $A = \hat{Q}\hat{R}$ where $\hat{Q} \in \mathbb{R}^{m \times m}$ is a square orthogonal matrix and $\hat{R} \in \mathbb{R}^{m \times n}$ is upper triangular (with zero rows below row $n$).

```
THIN QR vs FULL QR   (m = 5, n = 3)

   Thin:  A   =   Q    R
         [5×3] [5×3] [3×3]

   Full:  A   =   Q̂     R̂
         [5×3] [5×5] [5×3]
                       ┌───┐
                       │ R │ ← n rows (same R)
                       ├───┤
                       │ 0 │ ← (m−n) zero rows
                       └───┘
```

### 2.3 Solving Least Squares with QR

For the least squares problem $\min_\mathbf{x} \|A\mathbf{x} - \mathbf{b}\|^2$:

1. Compute $A = QR$
2. Multiply: $Q^T\mathbf{b} = \mathbf{c}$
3. Solve: $R\mathbf{x} = \mathbf{c}$ by back substitution

This is numerically superior to the normal equations because $Q$ has condition number 1.

| Method | Numerical stability | Cost |
|--------|-------------------|------|
| Normal equations $(A^TA)^{-1}A^T\mathbf{b}$ | $\kappa(A^TA) = \kappa(A)^2$ (squares the condition number) | $mn^2 + \frac{1}{3}n^3$ |
| QR decomposition | $\kappa(R) = \kappa(A)$ (preserves condition number) | $2mn^2$ |
| SVD | Most stable, handles rank deficiency | $2mn^2 + 11n^3$ |

*ML connection:* **Numerical stability** matters enormously in practice. When features are correlated (multicollinearity), $A^TA$ is ill-conditioned and the normal equations produce garbage. Libraries like scikit-learn use QR or SVD internally for `LinearRegression.fit()`, never the normal equations directly.

---

## 3. Cholesky Decomposition

**Definition 2.9.5 (Positive Definite Matrix).** A symmetric matrix $A \in \mathbb{R}^{n \times n}$ is *positive definite* (written $A \succ 0$) if:

$$\mathbf{x}^T A \mathbf{x} > 0 \quad \text{for all } \mathbf{x} \neq \mathbf{0}$$

**Definition 2.9.6 (Cholesky Decomposition).** If $A \succ 0$, the *Cholesky decomposition* is:

$$A = LL^T$$

where $L$ is lower triangular with positive diagonal entries.

**Theorem 2.9.2 (Existence and Uniqueness of Cholesky).** Every positive definite matrix has a unique Cholesky decomposition.

*Proof.* By induction on $n$. Base case $n = 1$: $A = (a)$ with $a > 0$, so $L = (\sqrt{a})$.

Inductive step: partition $A$ as

$$A = \begin{pmatrix} A_{11} & \mathbf{a} \\ \mathbf{a}^T & \alpha \end{pmatrix}$$

where $A_{11} \in \mathbb{R}^{(n-1) \times (n-1)}$ is positive definite (as a principal submatrix of a positive definite matrix). By the induction hypothesis, $A_{11} = L_{11}L_{11}^T$. Set:

$$L = \begin{pmatrix} L_{11} & \mathbf{0} \\ \mathbf{l}^T & l_{nn} \end{pmatrix}$$

Matching blocks in $A = LL^T$:

- $\mathbf{a} = L_{11}\mathbf{l}$, so $\mathbf{l} = L_{11}^{-1}\mathbf{a}$ (exists since $L_{11}$ is invertible)
- $\alpha = \mathbf{l}^T\mathbf{l} + l_{nn}^2$, so $l_{nn} = \sqrt{\alpha - \mathbf{l}^T\mathbf{l}}$

We need $\alpha - \mathbf{l}^T\mathbf{l} > 0$. This follows from $A \succ 0$: the Schur complement $\alpha - \mathbf{a}^T A_{11}^{-1}\mathbf{a} > 0$, and $\mathbf{l}^T\mathbf{l} = \mathbf{a}^T(L_{11}L_{11}^T)^{-1}\mathbf{a} = \mathbf{a}^T A_{11}^{-1}\mathbf{a}$.

Uniqueness follows from the uniqueness of each $\mathbf{l}$ and $l_{nn}$ (since we require $l_{nn} > 0$). $\square$

```
CHOLESKY = "SQUARE ROOT" OF A MATRIX

    Symmetric positive definite:     Cholesky factor:

    ┌                   ┐            ┌             ┐
    │  4    2    2      │            │  2   0   0  │
    │  2    5    3      │ = L Lᵀ =   │  1   2   0  │ × Lᵀ
    │  2    3   14      │            │  1   1   3  │
    └                   ┘            └             ┘

    Cost: n³/6 flops  (half of LU!)
```

**Example 2.9.4:** Compute the Cholesky decomposition of $A = \begin{pmatrix} 4 & 2 & 2 \\ 2 & 5 & 3 \\ 2 & 3 & 14 \end{pmatrix}$.

First verify $A \succ 0$: leading minors are $4 > 0$, $\det\begin{pmatrix}4&2\\2&5\end{pmatrix} = 16 > 0$, and $\det(A) = 4(70-9) - 2(28-6) + 2(6-10) = 244 - 44 - 8 = 192 > 0$.

**Row 1:** $l_{11} = \sqrt{4} = 2$.

**Row 2:** $l_{21} = a_{21}/l_{11} = 2/2 = 1$. Then $l_{22} = \sqrt{a_{22} - l_{21}^2} = \sqrt{5 - 1} = 2$.

**Row 3:** $l_{31} = a_{31}/l_{11} = 2/2 = 1$. Then $l_{32} = (a_{32} - l_{31}l_{21})/l_{22} = (3 - 1)/2 = 1$. Finally $l_{33} = \sqrt{a_{33} - l_{31}^2 - l_{32}^2} = \sqrt{14 - 1 - 1} = \sqrt{12} = 2\sqrt{3}$.

$$L = \begin{pmatrix} 2 & 0 & 0 \\ 1 & 2 & 0 \\ 1 & 1 & 2\sqrt{3} \end{pmatrix}$$

**Verification:** $LL^T = \begin{pmatrix} 4 & 2 & 2 \\ 2 & 5 & 3 \\ 2 & 3 & 14 \end{pmatrix} = A$. $\checkmark$

**Where positive definiteness arises in ML:**

| Matrix | Why positive definite | ML context |
|--------|----------------------|------------|
| $X^TX$ | $\mathbf{v}^T(X^TX)\mathbf{v} = \|X\mathbf{v}\|^2 \geq 0$ | Gram matrix in linear regression |
| $\Sigma$ (covariance) | Variance is non-negative | Gaussian distributions, Gaussian processes |
| $K$ (kernel matrix) | Mercer's theorem | SVM, Gaussian processes |
| Hessian $\nabla^2 L$ | At a local minimum | Newton's method for optimisation |

*ML connection:* **Gaussian processes** are the primary consumer of Cholesky in ML. To sample from $\mathcal{N}(\boldsymbol{\mu}, K)$ or compute the marginal likelihood $\log p(\mathbf{y}) = -\frac{1}{2}\mathbf{y}^T K^{-1}\mathbf{y} - \frac{1}{2}\log\det K - \frac{n}{2}\log 2\pi$, one computes $K = LL^T$ then solves $L\boldsymbol{\alpha} = \mathbf{y}$ by forward substitution and computes $\log\det K = 2\sum_i \log L_{ii}$. GPyTorch and JAX both use Cholesky extensively.

---

## 4. Singular Value Decomposition (SVD)

The SVD is arguably the most important matrix decomposition in all of applied mathematics.

### 4.1 Statement and Existence

**Definition 2.9.7 (Singular Value Decomposition).** For any matrix $A \in \mathbb{R}^{m \times n}$, there exist:

- An orthogonal matrix $U \in \mathbb{R}^{m \times m}$ (left singular vectors)
- An orthogonal matrix $V \in \mathbb{R}^{n \times n}$ (right singular vectors)
- A diagonal matrix $\Sigma \in \mathbb{R}^{m \times n}$ with non-negative entries $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_{\min(m,n)} \geq 0$

such that:

$$A = U \Sigma V^T$$

The values $\sigma_i$ are the *singular values* of $A$.

```
SVD STRUCTURE   (m = 4, n = 3, rank = 2)

   A     =     U          Σ         Vᵀ
 [4×3]      [4×4]      [4×3]      [3×3]

 ┌     ┐   ┌         ┐ ┌       ┐ ┌       ┐
 │ × × ×│   │ | | | | │ │ σ₁    │ │ ─ ─ ─ │
 │ × × ×│ = │ u₁u₂u₃u₄│ │   σ₂  │ │ ─ ─ ─ │
 │ × × ×│   │ | | | | │ │     0 │ │ ─ ─ ─ │
 │ × × ×│   │ | | | | │ │     0 │ └       ┘
 └     ┘   └         ┘ └       ┘  v₁ᵀv₂ᵀv₃ᵀ

    σ₁ ≥ σ₂ > σ₃ = 0  →  rank(A) = 2
```

**Theorem 2.9.3 (Existence of SVD).** Every matrix $A \in \mathbb{R}^{m \times n}$ has a singular value decomposition.

*Proof.* Consider $A^TA \in \mathbb{R}^{n \times n}$, which is symmetric and positive semidefinite (since $\mathbf{x}^T A^TA\mathbf{x} = \|A\mathbf{x}\|^2 \geq 0$). By the spectral theorem, $A^TA$ has an orthonormal eigenbasis $\{\mathbf{v}_1, \dots, \mathbf{v}_n\}$ with eigenvalues $\lambda_1 \geq \cdots \geq \lambda_n \geq 0$.

Define $\sigma_i = \sqrt{\lambda_i}$ for $i = 1, \dots, n$. Let $r = \operatorname{rank}(A)$, so $\sigma_1 \geq \cdots \geq \sigma_r > 0$ and $\sigma_{r+1} = \cdots = \sigma_n = 0$.

For $i = 1, \dots, r$, define $\mathbf{u}_i = \frac{1}{\sigma_i} A\mathbf{v}_i$. These are orthonormal:

$$\langle \mathbf{u}_i, \mathbf{u}_j \rangle = \frac{1}{\sigma_i \sigma_j} \langle A\mathbf{v}_i, A\mathbf{v}_j \rangle = \frac{1}{\sigma_i \sigma_j} \mathbf{v}_i^T A^TA\mathbf{v}_j = \frac{\lambda_j}{\sigma_i \sigma_j} \mathbf{v}_i^T\mathbf{v}_j = \frac{\sigma_j}{\sigma_i}\delta_{ij} = \delta_{ij}$$

Extend $\{\mathbf{u}_1, \dots, \mathbf{u}_r\}$ to an orthonormal basis $\{\mathbf{u}_1, \dots, \mathbf{u}_m\}$ of $\mathbb{R}^m$.

Now $A\mathbf{v}_i = \sigma_i \mathbf{u}_i$ for $i \leq r$ and $A\mathbf{v}_i = \mathbf{0}$ for $i > r$ (since $\sigma_i = 0$). In matrix form: $AV = U\Sigma$, so $A = U\Sigma V^T$. $\square$

**Example 2.9.5:** Compute the SVD of $A = \begin{pmatrix} 3 & 2 \\ 0 & 3 \end{pmatrix}$.

**Step 1 --- Compute $A^TA$:**

$$A^TA = \begin{pmatrix} 3 & 0 \\ 2 & 3 \end{pmatrix}\begin{pmatrix} 3 & 2 \\ 0 & 3 \end{pmatrix} = \begin{pmatrix} 9 & 6 \\ 6 & 13 \end{pmatrix}$$

**Step 2 --- Eigenvalues of $A^TA$.** $\det(A^TA - \lambda I) = (9-\lambda)(13-\lambda) - 36 = \lambda^2 - 22\lambda + 81 = 0$.

$$\lambda = \frac{22 \pm \sqrt{484 - 324}}{2} = \frac{22 \pm \sqrt{160}}{2} = 11 \pm 2\sqrt{10}$$

So $\sigma_1 = \sqrt{11 + 2\sqrt{10}}$, $\sigma_2 = \sqrt{11 - 2\sqrt{10}}$. Note: $11 + 2\sqrt{10} \approx 17.32$ and $11 - 2\sqrt{10} \approx 4.68$, giving $\sigma_1 \approx 4.16$, $\sigma_2 \approx 2.16$.

**Step 3 --- Right singular vectors $V$.** For $\lambda_1 = 11 + 2\sqrt{10}$: solve $(A^TA - \lambda_1 I)\mathbf{v} = 0$. The eigenvector is $\mathbf{v}_1 = \frac{1}{\sqrt{1+c^2}}(1, c)^T$ where $c = (\lambda_1 - 9)/6 = \sqrt{10}/3$.

For $\lambda_2 = 11 - 2\sqrt{10}$: $\mathbf{v}_2 = \frac{1}{\sqrt{1+c'^2}}(-c, 1)^T$ (orthogonal to $\mathbf{v}_1$), where $c' = \sqrt{10}/3$.

**Step 4 --- Left singular vectors:** $\mathbf{u}_i = \frac{1}{\sigma_i}A\mathbf{v}_i$.

**Verification:** $U\Sigma V^T = A$, and $\sigma_1^2 + \sigma_2^2 = 22 = \|A\|_F^2 = 9 + 4 + 0 + 9$. $\checkmark$

### 4.2 Geometric Interpretation

The SVD says: *every linear transformation is a rotation, followed by a scaling along coordinate axes, followed by another rotation.*

```
SVD: WHAT A MATRIX DOES TO THE UNIT SPHERE

    Unit sphere        Rotate (Vᵀ)       Scale (Σ)        Rotate (U)
    in Rⁿ              (still sphere)    (ellipsoid)       (ellipsoid)

       ○         →→→       ○       →→→      ◯      →→→     ◯
      ╱│╲              ╱│╲              ╱  │ ╲            ╱   │╲
     ○─┼─○            ○─┼─○            ○───┼──○          ○────┼─○
      ╲│╱              ╲│╱              ╲  │ ╱            ╲   │╱
       ○                ○                ◯                  ◯

                                      σ₁ = major axis
                                      σ₂ = minor axis
```

**Theorem 2.9.4 (SVD and the Four Fundamental Subspaces).** For $A = U\Sigma V^T$ with $r = \operatorname{rank}(A)$:

| Subspace | Basis | Dimension |
|----------|-------|-----------|
| $\operatorname{col}(A)$ (column space) | $\mathbf{u}_1, \dots, \mathbf{u}_r$ | $r$ |
| $\operatorname{null}(A)$ (null space) | $\mathbf{v}_{r+1}, \dots, \mathbf{v}_n$ | $n - r$ |
| $\operatorname{row}(A)$ (row space) | $\mathbf{v}_1, \dots, \mathbf{v}_r$ | $r$ |
| $\operatorname{null}(A^T)$ (left null space) | $\mathbf{u}_{r+1}, \dots, \mathbf{u}_m$ | $m - r$ |

### 4.3 Singular Values and Matrix Properties

**Theorem 2.9.4a (Singular Values Encode Everything).** The singular values reveal fundamental matrix properties:

| Property | Formula via singular values |
|----------|----------------------------|
| Rank | $r = \#\{i : \sigma_i > 0\}$ |
| Spectral norm | $\lVert A\rVert_2 = \sigma_1$ |
| Frobenius norm | $\lVert A\rVert_F = \sqrt{\sigma_1^2 + \cdots + \sigma_r^2}$ |
| Condition number | $\kappa(A) = \sigma_1 / \sigma_r$ |
| Determinant ($n \times n$) | $\lvert\det A\rvert = \prod_{i=1}^n \sigma_i$ |
| Nuclear norm | $\lVert A\rVert_* = \sum_{i=1}^r \sigma_i$ |

**Definition 2.9.7a (Condition Number).** The *condition number* of a matrix $A$ is:

$$\kappa(A) = \frac{\sigma_{\max}}{\sigma_{\min}} = \frac{\sigma_1}{\sigma_r}$$

A matrix with $\kappa(A) \gg 1$ is *ill-conditioned*: small perturbations in the input cause large changes in the output.

**Example 2.9.6:** From Example 2.9.5, $A = \begin{pmatrix} 3 & 2 \\ 0 & 3 \end{pmatrix}$ has $\sigma_1 \approx 4.16$ and $\sigma_2 \approx 2.16$. So:

$$\kappa(A) = \frac{\sigma_1}{\sigma_2} = \frac{\sqrt{11 + 2\sqrt{10}}}{\sqrt{11 - 2\sqrt{10}}} \approx \frac{4.16}{2.16} \approx 1.93$$

This is well-conditioned ($\kappa$ close to 1). Compare: $B = \begin{pmatrix} 1 & 1 \\ 1 & 1.0001 \end{pmatrix}$ has $\kappa(B) \approx 40000$ --- nearly singular and very ill-conditioned.

```
CONDITION NUMBER: SENSITIVITY TO PERTURBATION

    Well-conditioned (κ ≈ 1):        Ill-conditioned (κ ≫ 1):

    Input circle → output circle     Input circle → output needle

         ○    ──→    ○                     ○    ──→    ═══════
        ╱ ╲         ╱ ╲                   ╱ ╲          (stretched in one
       ○   ○       ○   ○                 ○   ○          direction, crushed
        ╲ ╱         ╲ ╱                   ╲ ╱           in another)
         ○           ○                     ○

    σ₁ ≈ σ₂                          σ₁ ≫ σ₂
    Small error → small error         Small error → large error
```

*ML connection:* **Batch normalisation** in deep learning improves training by reducing the condition number of each layer's Jacobian. Without normalisation, layers deep in a network can have $\kappa \sim 10^6$, making gradient descent extremely slow (it zigzags in the narrow directions of the loss landscape).

### 4.4 Relation to Eigendecomposition

**Proposition 2.9.1.** Let $A = U\Sigma V^T$. Then:

- $A^TA = V\Sigma^T\Sigma V^T = V \operatorname{diag}(\sigma_1^2, \dots, \sigma_n^2) V^T$ --- eigendecomposition of $A^TA$
- $AA^T = U\Sigma\Sigma^T U^T = U \operatorname{diag}(\sigma_1^2, \dots, \sigma_m^2) U^T$ --- eigendecomposition of $AA^T$
- If $A$ is symmetric positive semidefinite, then $U = V$ and $\sigma_i = \lambda_i$ --- SVD reduces to eigendecomposition

**Example 2.9.7:** Diagonalise the symmetric matrix $A = \begin{pmatrix} 5 & 2 \\ 2 & 2 \end{pmatrix}$ via eigendecomposition, then verify it matches the SVD.

**Eigenvalues:** $\det(A - \lambda I) = (5-\lambda)(2-\lambda) - 4 = \lambda^2 - 7\lambda + 6 = (\lambda - 6)(\lambda - 1) = 0$, so $\lambda_1 = 6$, $\lambda_2 = 1$.

**Eigenvectors:** For $\lambda_1 = 6$: $(A - 6I)\mathbf{v} = 0 \Rightarrow -v_1 + 2v_2 = 0$, so $\mathbf{v}_1 = \frac{1}{\sqrt{5}}(2, 1)^T$.
For $\lambda_2 = 1$: $(A - I)\mathbf{v} = 0 \Rightarrow 4v_1 + 2v_2 = 0$, so $\mathbf{v}_2 = \frac{1}{\sqrt{5}}(-1, 2)^T$.

**Eigendecomposition:** $A = V\Lambda V^T$ where $V = \frac{1}{\sqrt{5}}\begin{pmatrix} 2 & -1 \\ 1 & 2 \end{pmatrix}$, $\Lambda = \begin{pmatrix} 6 & 0 \\ 0 & 1 \end{pmatrix}$.

Since $A$ is symmetric positive definite, the SVD is $A = U\Sigma V^T$ with $U = V$ and $\sigma_i = \lambda_i$.

**Verification:** $V\Lambda V^T = \frac{1}{5}\begin{pmatrix} 2 & -1 \\ 1 & 2 \end{pmatrix}\begin{pmatrix} 6 & 0 \\ 0 & 1 \end{pmatrix}\begin{pmatrix} 2 & 1 \\ -1 & 2 \end{pmatrix} = \frac{1}{5}\begin{pmatrix} 12 & -1 \\ 6 & 2 \end{pmatrix}\begin{pmatrix} 2 & 1 \\ -1 & 2 \end{pmatrix} = \frac{1}{5}\begin{pmatrix} 25 & 10 \\ 10 & 10 \end{pmatrix} = \begin{pmatrix} 5 & 2 \\ 2 & 2 \end{pmatrix}$. $\checkmark$

### 4.4 Compact and Outer Product Forms

**Definition 2.9.8 (Compact SVD).** Keeping only the $r$ nonzero singular values:

$$A = U_r \Sigma_r V_r^T$$

where $U_r \in \mathbb{R}^{m \times r}$, $\Sigma_r \in \mathbb{R}^{r \times r}$, $V_r \in \mathbb{R}^{n \times r}$.

**Definition 2.9.9 (Outer Product Form).** The SVD can be written as a sum of rank-1 matrices:

$$A = \sum_{i=1}^{r} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

Each $\sigma_i \mathbf{u}_i \mathbf{v}_i^T$ is a rank-1 matrix capturing one "layer" of the transformation.

**Example 2.9.8:** Write $A = \begin{pmatrix} 3 & 0 \\ 0 & 2 \end{pmatrix}$ in outer product form.

This diagonal matrix has SVD with $U = V = I$, $\sigma_1 = 3$, $\sigma_2 = 2$. The outer product form is:

$$A = 3 \begin{pmatrix} 1 \\ 0 \end{pmatrix}\begin{pmatrix} 1 & 0 \end{pmatrix} + 2 \begin{pmatrix} 0 \\ 1 \end{pmatrix}\begin{pmatrix} 0 & 1 \end{pmatrix} = 3\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} + 2\begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 3 & 0 \\ 0 & 2 \end{pmatrix}$$

Each rank-1 term captures one "axis" of the transformation: the first scales the $x$-direction by 3, the second scales the $y$-direction by 2.

---

## 5. Truncated SVD and Low-Rank Approximation

**Definition 2.9.10 (Truncated SVD).** The *rank-$k$ truncated SVD* retains only the $k$ largest singular values:

$$A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T = U_k \Sigma_k V_k^T$$

**Theorem 2.9.5 (Eckart-Young-Mirsky).** The best rank-$k$ approximation to $A$ (in both Frobenius and spectral norms) is $A_k$:

$$A_k = \arg\min_{\operatorname{rank}(B) \leq k} \|A - B\|_F$$

The approximation error is:

$$\|A - A_k\|_F = \sqrt{\sigma_{k+1}^2 + \cdots + \sigma_r^2}$$

**Example 2.9.9:** Let $A = \begin{pmatrix} 3 & 0 \\ 0 & 2 \\ 0 & 0 \end{pmatrix}$ with singular values $\sigma_1 = 3$, $\sigma_2 = 2$. Compute the best rank-1 approximation.

From the outer product form, $A = 3\mathbf{u}_1\mathbf{v}_1^T + 2\mathbf{u}_2\mathbf{v}_2^T$ where $\mathbf{u}_1 = (1,0,0)^T$, $\mathbf{v}_1 = (1,0)^T$, $\mathbf{u}_2 = (0,1,0)^T$, $\mathbf{v}_2 = (0,1)^T$.

**Rank-1 approximation** (keep only $\sigma_1$):

$$A_1 = 3\mathbf{u}_1\mathbf{v}_1^T = 3\begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix}(1 \;\; 0) = \begin{pmatrix} 3 & 0 \\ 0 & 0 \\ 0 & 0 \end{pmatrix}$$

**Error:** $\|A - A_1\|_F = \sqrt{\sigma_2^2} = 2$. **Energy captured:** $\sigma_1^2 / (\sigma_1^2 + \sigma_2^2) = 9/13 \approx 69.2\%$.

**Verification:** $A - A_1 = \begin{pmatrix} 0 & 0 \\ 0 & 2 \\ 0 & 0 \end{pmatrix}$ has Frobenius norm $\sqrt{0+4+0} = 2$. $\checkmark$

*Proof sketch.* Any rank-$k$ matrix $B$ can eliminate at most $k$ singular values. The minimum residual is achieved by eliminating the $k$ largest, leaving $\sigma_{k+1}, \dots, \sigma_r$. The Frobenius norm of the residual equals the root-sum-of-squares of the remaining singular values. $\square$

```
TRUNCATED SVD: KEEPING ONLY k COMPONENTS

  Full SVD (rank r):   A  = σ₁u₁v₁ᵀ + σ₂u₂v₂ᵀ + ... + σᵣuᵣvᵣᵀ

  Truncated (rank k):  Aₖ = σ₁u₁v₁ᵀ + σ₂u₂v₂ᵀ + ... + σₖuₖvₖᵀ

  Singular value spectrum:

  σ
  │ ██
  │ ██ ██
  │ ██ ██ ██
  │ ██ ██ ██ ▒▒
  │ ██ ██ ██ ▒▒ ▒▒
  │ ██ ██ ██ ▒▒ ▒▒ ░░ ░░ ░░
  └──────────────────────────→ i
    ←─ keep ──→←── discard ──→
       (k)           (r-k)

  ██ = kept components    ▒▒/░░ = discarded (small σᵢ)
```

**Fraction of "energy" captured:**

$$\text{explained variance ratio} = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^r \sigma_i^2}$$

*ML connection:* **Principal Component Analysis (PCA)** is truncated SVD of the centred data matrix. If $\tilde{X} = X - \bar{X}$ is the centred $m \times n$ data matrix:

1. Compute $\tilde{X} = U\Sigma V^T$
2. The principal components are the columns of $V_k$ (top $k$ right singular vectors)
3. The projected data is $\tilde{X}V_k = U_k\Sigma_k$
4. The variance explained by component $i$ is $\sigma_i^2 / (m - 1)$

**Recommender systems** use truncated SVD for collaborative filtering. The Netflix Prize-winning approach approximated the $m \times n$ user-item rating matrix $R \approx U_k \Sigma_k V_k^T$, where $k \sim 20\text{-}200$ captures latent features like genre preference.

**Image compression:** An $m \times n$ grayscale image stored naively needs $mn$ values. The rank-$k$ approximation needs $k(m + n + 1)$ values, giving compression ratio $mn / (k(m+n+1))$.

---

## 6. The Moore-Penrose Pseudoinverse

**Definition 2.9.11 (Moore-Penrose Pseudoinverse).** For any matrix $A \in \mathbb{R}^{m \times n}$, the *pseudoinverse* $A^+ \in \mathbb{R}^{n \times m}$ is the unique matrix satisfying:

1. $AA^+A = A$
2. $A^+AA^+ = A^+$
3. $(AA^+)^T = AA^+$
4. $(A^+A)^T = A^+A$

**Theorem 2.9.6 (Pseudoinverse via SVD).** If $A = U\Sigma V^T$, then:

$$A^+ = V\Sigma^+ U^T$$

where $\Sigma^+$ is obtained by transposing $\Sigma$ and inverting each nonzero diagonal entry:

$$\sigma_i^+ = \begin{cases} 1/\sigma_i & \sigma_i > 0 \\ 0 & \sigma_i = 0 \end{cases}$$

*Proof.* Verify the four Moore-Penrose conditions directly. For example:

$$AA^+A = (U\Sigma V^T)(V\Sigma^+ U^T)(U\Sigma V^T) = U\Sigma\Sigma^+\Sigma V^T = U\Sigma V^T = A$$

since $\Sigma\Sigma^+\Sigma = \Sigma$ (each diagonal entry satisfies $\sigma_i \cdot \sigma_i^+ \cdot \sigma_i = \sigma_i$). The remaining conditions follow similarly. $\square$

**Example 2.9.10:** Compute the pseudoinverse of $A = \begin{pmatrix} 3 & 0 \\ 0 & 2 \\ 0 & 0 \end{pmatrix}$.

The SVD is $A = U\Sigma V^T$ with $U = I_3$, $\Sigma = \begin{pmatrix} 3 & 0 \\ 0 & 2 \\ 0 & 0 \end{pmatrix}$, $V = I_2$. Invert the nonzero singular values and transpose:

$$\Sigma^+ = \begin{pmatrix} 1/3 & 0 & 0 \\ 0 & 1/2 & 0 \end{pmatrix}, \quad A^+ = V\Sigma^+ U^T = \begin{pmatrix} 1/3 & 0 & 0 \\ 0 & 1/2 & 0 \end{pmatrix}$$

**Verification** (condition 1: $AA^+A = A$):

$$AA^+ = \begin{pmatrix} 3 & 0 \\ 0 & 2 \\ 0 & 0 \end{pmatrix}\begin{pmatrix} 1/3 & 0 & 0 \\ 0 & 1/2 & 0 \end{pmatrix} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix}, \quad AA^+A = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix}\begin{pmatrix} 3 & 0 \\ 0 & 2 \\ 0 & 0 \end{pmatrix} = \begin{pmatrix} 3 & 0 \\ 0 & 2 \\ 0 & 0 \end{pmatrix} = A. \; \checkmark$$

```
PSEUDOINVERSE: WHAT IT DOES

    Case 1: m > n (overdetermined, e.g., regression)
    A⁺ = (AᵀA)⁻¹Aᵀ     ← left inverse (least squares solution)
    A⁺b = argmin ‖Ax − b‖²

    Case 2: m < n (underdetermined, e.g., compressed sensing)
    A⁺ = Aᵀ(AAᵀ)⁻¹     ← right inverse (minimum norm solution)
    A⁺b = argmin ‖x‖²  subject to Ax = b

    Case 3: m = n, A invertible
    A⁺ = A⁻¹             ← ordinary inverse
```

**Theorem 2.9.7 (Pseudoinverse and Least Squares).** The minimum-norm least squares solution to $A\mathbf{x} = \mathbf{b}$ is:

$$\hat{\mathbf{x}} = A^+\mathbf{b}$$

This simultaneously:
- Minimises $\|A\mathbf{x} - \mathbf{b}\|$ (least squares)
- Among all minimisers, selects the one with smallest $\|\mathbf{x}\|$ (minimum norm)

*ML connection:* The pseudoinverse unifies several ML concepts:

- **Linear regression** with redundant features: $\hat{\boldsymbol{\beta}} = X^+\mathbf{y}$
- **Neural networks:** Gradient descent on overparameterised networks converges to the minimum-norm solution, which is exactly $X^+\mathbf{y}$. This connects to the implicit bias of gradient descent.
- **`numpy.linalg.lstsq`** computes the pseudoinverse via SVD internally.

---

## 7. Summary: Choosing the Right Decomposition

| Decomposition | Input requirement | Cost | When to use |
|--------------|-------------------|------|-------------|
| **LU** | Square (or with pivoting) | $\frac{2}{3}n^3$ | Solving $Ax = b$, determinants |
| **QR** | Any $m \times n$, $m \geq n$ | $2mn^2$ | Least squares (numerically stable) |
| **Cholesky** | Symmetric positive definite | $\frac{1}{3}n^3$ | Gaussian processes, covariance matrices |
| **Eigendecomposition** | Square (symmetric for real eigenvalues) | $O(n^3)$ | Spectral analysis, PCA (via covariance) |
| **SVD** | Any $m \times n$ | $2mn^2 + 11n^3$ | PCA, low-rank approx, pseudoinverse |

```
DECISION TREE: WHICH DECOMPOSITION?

                         Is A square?
                        ╱           ╲
                      yes             no
                      ╱                 ╲
              Is A symmetric?          Use QR or SVD
              ╱          ╲             (least squares,
            yes           no            low-rank approx)
            ╱               ╲
    Is A pos. definite?    Use LU
    ╱         ╲            (general systems)
  yes          no
  ╱              ╲
Cholesky     Eigendecomposition
(fastest)    (spectral analysis)
```

---

## Exercises

### Foundations (★)

**Exercise 2.9.1.** Compute the LU decomposition of $A = \begin{pmatrix} 2 & 1 \\ 6 & 4 \end{pmatrix}$. Use it to solve $A\mathbf{x} = (3, 11)^T$.

**Exercise 2.9.2.** Apply Gram-Schmidt to the columns of $A = \begin{pmatrix} 1 & 1 \\ 1 & 0 \\ 0 & 1 \end{pmatrix}$ to find the QR decomposition.

**Exercise 2.9.3.** Verify that $A = \begin{pmatrix} 4 & 2 \\ 2 & 3 \end{pmatrix}$ is positive definite and compute its Cholesky factor $L$.

**Exercise 2.9.4.** Compute the SVD of $A = \begin{pmatrix} 3 & 0 \\ 0 & 2 \\ 0 & 0 \end{pmatrix}$. Identify $U, \Sigma, V^T$, the four fundamental subspaces, and the pseudoinverse $A^+$.

### Core Theory (★★)

**Exercise 2.9.5.** Prove that the singular values of $A$ are the square roots of the eigenvalues of $A^TA$. Show that $\|A\|_2 = \sigma_1$ (the spectral norm equals the largest singular value).

**Exercise 2.9.6.** Show that the Frobenius norm satisfies $\|A\|_F = \sqrt{\sigma_1^2 + \cdots + \sigma_r^2} = \sqrt{\operatorname{tr}(A^TA)}$.

**Exercise 2.9.7.** Let $A$ have SVD $A = U\Sigma V^T$. Prove that the best rank-1 approximation to $A$ is $\sigma_1 \mathbf{u}_1 \mathbf{v}_1^T$ by showing that any rank-1 matrix $B = \mathbf{a}\mathbf{b}^T$ satisfies $\|A - B\|_F \geq \sqrt{\sigma_2^2 + \cdots + \sigma_r^2}$.

**Exercise 2.9.8.** Show that for a symmetric positive definite matrix $A$, the Cholesky decomposition $A = LL^T$ can be obtained from the LU decomposition $A = \tilde{L}\tilde{U}$ by writing $\tilde{U} = D\tilde{L}^T$ where $D = \operatorname{diag}(\tilde{u}_{11}, \dots, \tilde{u}_{nn})$, then $L = \tilde{L}\sqrt{D}$.

### Advanced / ML Applications (★★★)

**Exercise 2.9.9.** **(PCA by hand)** Given the data matrix $X = \begin{pmatrix} 2 & 1 \\ 4 & 3 \\ 6 & 5 \\ 8 & 7 \end{pmatrix}$, centre $X$, compute the covariance matrix, find its eigendecomposition, and determine the first principal component. Verify by computing the SVD of the centred matrix.

**Exercise 2.9.10.** **(Image compression)** A $1000 \times 1000$ image has singular values $\sigma_1 = 500, \sigma_2 = 200, \sigma_3 = 100$, and remaining $\sigma_i < 10$. (a) How much memory does a rank-3 approximation use compared to the original? (b) What fraction of the Frobenius-norm "energy" do the first 3 singular values capture if $\sum_{i=1}^{1000} \sigma_i^2 = 310000$?

**Exercise 2.9.11.** **(Pseudoinverse and gradient descent)** Consider an overparameterised linear model $y = X\boldsymbol{\beta}$ where $X \in \mathbb{R}^{10 \times 100}$ (more parameters than data). Show that gradient descent on $\|\mathbf{y} - X\boldsymbol{\beta}\|^2$ starting from $\boldsymbol{\beta}_0 = \mathbf{0}$ converges to $X^+\mathbf{y}$, the minimum-norm interpolating solution. Why does this relate to the generalisation properties of overparameterised neural networks?

**Exercise 2.9.12.** **(Netflix Prize)** A user-movie rating matrix $R \in \mathbb{R}^{500000 \times 20000}$ has approximately $1\%$ observed entries. Explain why direct SVD is impractical and describe how alternating least squares (ALS) computes an approximate low-rank factorisation $R \approx UV^T$ where $U \in \mathbb{R}^{500000 \times k}$ and $V \in \mathbb{R}^{20000 \times k}$.

---

## Related Topics

- **Previous:** [Inner Product Spaces](inner-product-spaces.md) --- Gram-Schmidt (used in QR), projections (used in least squares)
- **Next:** [Tensors](tensors.md) --- tensor decompositions generalise SVD to higher dimensions
- **Eigenvalues:** [Eigenvalues & Eigenvectors](eigenvalues.md) --- SVD extends eigendecomposition to non-square matrices
- **Optimisation:** Gradient descent, Newton's method (use Cholesky for Hessian)
- **Probability:** Covariance matrices (Cholesky for sampling), PCA (SVD of data matrix)

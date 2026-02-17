# Determinants

The determinant is a single number that encodes an extraordinary amount of information about a square matrix: whether it is invertible, how it scales volumes, and whether it preserves or reverses orientation. In machine learning, determinants appear whenever we need to reason about volume change under transformations — the Jacobian determinant in normalizing flows, the normalization constant of a multivariate Gaussian, and the change-of-variables formula in probability.

---

## Prerequisites

- [Vectors](vectors.md) — inner products, cross product
- [Matrices](matrices.md) — matrix operations, transpose, inverse
- [Systems of Equations](systems-of-equations.md) — row operations, row echelon form
- [Linear Transformations](linear-transformations.md) — kernel, image, rank-nullity

---

## 1. Motivation: Why Determinants?

Before the formal definition, consider what we want: a function $\det: \mathbb{R}^{n \times n} \to \mathbb{R}$ that tells us:

| Question | Answer via Determinant |
|----------|----------------------|
| Is $A$ invertible? | $\det(A) \neq 0$ |
| How does $A$ scale volumes? | $|\det(A)|$ = volume scaling factor |
| Does $A$ preserve orientation? | $\det(A) > 0$ preserves, $< 0$ reverses |
| What is $\det(AB)$? | $\det(A)\det(B)$ (multiplicative) |

```
GEOMETRIC INTUITION (2D)

Original unit square          After applying A = [2 1; 0 3]

    (0,1)──────(1,1)          (1,3)──────────(3,4)
      │          │             ╱             ╱
      │  Area=1  │            ╱  Area = 6   ╱
      │          │           ╱             ╱
    (0,0)──────(1,0)       (0,0)──────(2,1)

    det(A) = 2·3 - 1·0 = 6
    The unit square maps to a parallelogram of area |det(A)| = 6
```

---

## 2. Formal Definition via Permutations

**Definition 2.6.1 (Permutation).** A *permutation* of $\{1, 2, \ldots, n\}$ is a bijection $\sigma: \{1, \ldots, n\} \to \{1, \ldots, n\}$. The set of all such permutations is the *symmetric group* $S_n$, which has $n!$ elements.

**Definition 2.6.2 (Sign of a Permutation).** The *sign* (or *parity*) of a permutation $\sigma$ is:

$$\text{sgn}(\sigma) = (-1)^{\text{number of inversions}}$$

where an *inversion* is a pair $(i, j)$ with $i < j$ but $\sigma(i) > \sigma(j)$. Equivalently, $\text{sgn}(\sigma) = (-1)^k$ where $k$ is the number of transpositions (swaps of two elements) needed to express $\sigma$.

**Example.** For $\sigma = (2, 3, 1)$ in $S_3$: inversions are $(1,3)$ since $\sigma(1) = 2 > 1 = \sigma(3)$ and $(2,3)$ since $\sigma(2) = 3 > 1 = \sigma(3)$. So $\text{sgn}(\sigma) = (-1)^2 = +1$.

**Definition 2.6.3 (Determinant — Leibniz Formula).** For $A \in \mathbb{R}^{n \times n}$:

$$\det(A) = \sum_{\sigma \in S_n} \text{sgn}(\sigma) \prod_{i=1}^n a_{i,\sigma(i)}$$

This sums over all $n!$ permutations. Each term picks exactly one entry from each row and each column, weighted by the sign of the permutation.

**For $n = 2$:**

$$\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc$$

There are $2! = 2$ permutations: identity $(1,2)$ with sign $+1$ giving $a_{11}a_{22} = ad$, and swap $(2,1)$ with sign $-1$ giving $-a_{12}a_{21} = -bc$.

**Example 2.3.1:** Compute the determinant of $A = \begin{pmatrix} 3 & 4 \\ 2 & 5 \end{pmatrix}$.

$$\det(A) = ad - bc = (3)(5) - (4)(2) = 15 - 8 = 7$$

Since $\det(A) = 7 \neq 0$, the matrix $A$ is invertible.

**For $n = 3$:**

$$\det\begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{pmatrix} = a_{11}a_{22}a_{33} + a_{12}a_{23}a_{31} + a_{13}a_{21}a_{32} - a_{13}a_{22}a_{31} - a_{12}a_{21}a_{33} - a_{11}a_{23}a_{32}$$

This is the *Sarrus' rule* (valid only for $3 \times 3$).

```
SARRUS' RULE — VISUAL MNEMONIC

     a₁₁  a₁₂  a₁₃ │ a₁₁  a₁₂
     a₂₁  a₂₂  a₂₃ │ a₂₁  a₂₂
     a₃₁  a₃₂  a₃₃ │ a₃₁  a₃₂

     ╲    ╲    ╲        ╱    ╱    ╱
      +    +    +      -    -    -

  Positive diagonals:           Negative diagonals:
  a₁₁·a₂₂·a₃₃                 a₁₃·a₂₂·a₃₁
  a₁₂·a₂₃·a₃₁                 a₁₂·a₂₁·a₃₃
  a₁₃·a₂₁·a₃₂                 a₁₁·a₂₃·a₃₂
```

---

## 3. Cofactor Expansion (Recursive Definition)

**Definition 2.6.4 (Minor).** The $(i,j)$-*minor* $M_{ij}$ is the determinant of the $(n-1) \times (n-1)$ submatrix obtained by deleting row $i$ and column $j$ from $A$.

**Definition 2.6.5 (Cofactor).** The $(i,j)$-*cofactor* is:

$$C_{ij} = (-1)^{i+j} M_{ij}$$

**Theorem 2.6.1 (Cofactor Expansion).** For any fixed row $i$:

$$\det(A) = \sum_{j=1}^n a_{ij} C_{ij} = \sum_{j=1}^n (-1)^{i+j} a_{ij} M_{ij}$$

Similarly, for any fixed column $j$:

$$\det(A) = \sum_{i=1}^n a_{ij} C_{ij}$$

The result is the same regardless of which row or column is chosen.

*Proof sketch.* Each term $a_{ij} C_{ij}$ collects exactly those permutations in the Leibniz formula where $\sigma(i) = j$. The sign $(-1)^{i+j}$ accounts for the positional offset, and $M_{ij}$ sums over the remaining $(n-1)!$ permutations of the reduced matrix. Since every permutation has exactly one value of $\sigma(i)$, the cofactor expansion along row $i$ partitions the Leibniz sum into $n$ groups, one for each $j$. $\square$

**Example.** Expand along the first row of a $3 \times 3$ matrix:

$$\det(A) = a_{11}\det\begin{pmatrix} a_{22} & a_{23} \\ a_{32} & a_{33} \end{pmatrix} - a_{12}\det\begin{pmatrix} a_{21} & a_{23} \\ a_{31} & a_{33} \end{pmatrix} + a_{13}\det\begin{pmatrix} a_{21} & a_{22} \\ a_{31} & a_{32} \end{pmatrix}$$

**Example 2.3.2:** Compute $\det\begin{pmatrix} 1 & 3 & 2 \\ 0 & -1 & 4 \\ 2 & 5 & 1 \end{pmatrix}$ by cofactor expansion along the first row.

$$\det(A) = 1 \cdot \det\begin{pmatrix} -1 & 4 \\ 5 & 1 \end{pmatrix} - 3 \cdot \det\begin{pmatrix} 0 & 4 \\ 2 & 1 \end{pmatrix} + 2 \cdot \det\begin{pmatrix} 0 & -1 \\ 2 & 5 \end{pmatrix}$$

$$= 1 \cdot ((-1)(1) - (4)(5)) - 3 \cdot ((0)(1) - (4)(2)) + 2 \cdot ((0)(5) - (-1)(2))$$

$$= 1 \cdot (-21) - 3 \cdot (-8) + 2 \cdot (2) = -21 + 24 + 4 = 7$$

**Computational complexity.** Cofactor expansion is $O(n!)$ — impractical for large $n$. In practice, we use row reduction to upper triangular form ($O(n^3)$) and multiply the diagonal entries.

*ML connection:* In deep learning, we never compute determinants of large matrices by cofactor expansion. When normalizing flows require $\det(J)$ for a Jacobian $J \in \mathbb{R}^{d \times d}$ with $d = 784$ (MNIST), architectures are specifically designed so $J$ is triangular (autoregressive flows) or has low-rank structure (residual flows), making the determinant $O(d)$ or $O(d \cdot k^2)$ instead of $O(d^3)$.

---

## 4. Properties of Determinants

**Theorem 2.6.2 (Fundamental Properties).** Let $A, B \in \mathbb{R}^{n \times n}$.

**(a) Multiplicativity:**

$$\det(AB) = \det(A)\det(B)$$

*Proof.* If $A$ is singular, then $AB$ is singular, and both sides are $0$. If $A$ is invertible, $A$ is a product of elementary matrices $E_1 \cdots E_k$. By induction, it suffices to show $\det(EA) = \det(E)\det(A)$ for elementary matrices $E$, which follows from properties (b)–(d) below and the known determinants of elementary matrices. $\square$

**Example 2.3.3:** Verify $\det(AB) = \det(A)\det(B)$ for $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$, $B = \begin{pmatrix} 2 & 0 \\ 1 & 3 \end{pmatrix}$.

$$\det(A) = (1)(4) - (2)(3) = -2, \qquad \det(B) = (2)(3) - (0)(1) = 6$$

$$AB = \begin{pmatrix} 4 & 6 \\ 10 & 12 \end{pmatrix}, \qquad \det(AB) = (4)(12) - (6)(10) = 48 - 60 = -12$$

$$\det(A)\det(B) = (-2)(6) = -12 \; \checkmark$$

**(b) Row swap:** Swapping two rows multiplies the determinant by $-1$.

$$\det(\text{swap rows } i, j \text{ of } A) = -\det(A)$$

**Example 2.3.4:** Let $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$ with $\det(A) = 4 - 6 = -2$. Swapping the two rows:

$$A' = \begin{pmatrix} 3 & 4 \\ 1 & 2 \end{pmatrix}, \qquad \det(A') = (3)(2) - (4)(1) = 2 = -(-2) = -\det(A) \; \checkmark$$

**(c) Row scaling:** Multiplying a row by scalar $c$ multiplies the determinant by $c$.

$$\det(\text{multiply row } i \text{ by } c) = c \cdot \det(A)$$

**(d) Row addition:** Adding a multiple of one row to another does not change the determinant.

$$\det(\text{add } c \cdot \text{row } j \text{ to row } i) = \det(A)$$

**Example 2.3.5:** Let $A = \begin{pmatrix} 2 & 1 \\ 3 & 5 \end{pmatrix}$ with $\det(A) = 10 - 3 = 7$. Add $(-2) \times \text{row 1}$ to row 2:

$$A' = \begin{pmatrix} 2 & 1 \\ -1 & 3 \end{pmatrix}, \qquad \det(A') = (2)(3) - (1)(-1) = 7 = \det(A) \; \checkmark$$

The determinant is unchanged, which is why row addition is the "safe" operation in Gaussian elimination.

**(e) Transpose:**

$$\det(A^T) = \det(A)$$

*Proof.* In the Leibniz formula, $\det(A^T) = \sum_{\sigma} \text{sgn}(\sigma) \prod_i a_{\sigma(i),i}$. Re-indexing by $\tau = \sigma^{-1}$ (which has the same sign as $\sigma$), this equals $\sum_{\tau} \text{sgn}(\tau) \prod_j a_{j,\tau(j)} = \det(A)$. $\square$

**(f) Singular matrices:** $\det(A) = 0$ if and only if $A$ is singular (not invertible).

**Example 2.3.6:** The matrix $A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 5 & 7 & 9 \end{pmatrix}$ has row 3 = row 1 + row 2 (linearly dependent rows).

$$\det(A) = 1(45-42) - 2(36-30) + 3(28-25) = 3 - 12 + 9 = 0$$

Since $\det(A) = 0$, the matrix is singular and has no inverse.

**(g) Inverse:**

$$\det(A^{-1}) = \frac{1}{\det(A)}$$

*Proof.* $\det(A)\det(A^{-1}) = \det(AA^{-1}) = \det(I) = 1$. $\square$

**(h) Scalar multiple:**

$$\det(cA) = c^n \det(A)$$

*Proof.* Scaling each of the $n$ rows by $c$ multiplies the determinant by $c$ each time. $\square$

**Example 2.3.7:** Let $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$ with $\det(A) = -2$. Then for $c = 3$ and $n = 2$:

$$3A = \begin{pmatrix} 3 & 6 \\ 9 & 12 \end{pmatrix}, \qquad \det(3A) = (3)(12) - (6)(9) = 36 - 54 = -18$$

$$c^n \det(A) = 3^2 \cdot (-2) = 9 \cdot (-2) = -18 \; \checkmark$$

**Corollary 2.6.1.** For triangular matrices (upper or lower), $\det(A) = \prod_{i=1}^n a_{ii}$ (product of diagonal entries).

*Proof.* By cofactor expansion along the first column (for lower triangular) or first row (for upper triangular), only one term survives at each level of recursion. $\square$

*ML connection:* The property $\det(cA) = c^n \det(A)$ explains why numerical issues arise in high dimensions. If each entry of a $1000 \times 1000$ matrix is scaled by $2$, the determinant is multiplied by $2^{1000}$ — a number with over 300 digits. This is why we work with $\log|\det(A)|$ in normalizing flows and Gaussian processes.

---

## 5. Computing Determinants Efficiently

### 5.1 Via Row Reduction

**Algorithm.** To compute $\det(A)$:

1. Apply row operations to reduce $A$ to upper triangular form $U$
2. Track sign changes from row swaps: $s = (-1)^{\text{number of swaps}}$
3. Track scaling factors from row scaling: $\prod c_i$
4. $\det(A) = s \cdot \frac{\prod_{i} u_{ii}}{\prod c_i}$

**Complexity:** $O(n^3)$ via Gaussian elimination.

**Example.** Compute $\det\begin{pmatrix} 2 & 1 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix}$:

$$\xrightarrow{R_2 - 2R_1} \begin{pmatrix} 2 & 1 & 3 \\ 0 & 3 & 0 \\ 7 & 8 & 9 \end{pmatrix} \xrightarrow{R_3 - \frac{7}{2}R_1} \begin{pmatrix} 2 & 1 & 3 \\ 0 & 3 & 0 \\ 0 & \frac{9}{2} & -\frac{3}{2} \end{pmatrix} \xrightarrow{R_3 - \frac{3}{2}R_2} \begin{pmatrix} 2 & 1 & 3 \\ 0 & 3 & 0 \\ 0 & 0 & -\frac{3}{2} \end{pmatrix}$$

No row swaps, no row scaling. $\det(A) = 2 \cdot 3 \cdot (-\frac{3}{2}) = -9$.

### 5.2 Block Matrices

**Theorem 2.6.3 (Block Triangular Determinant).** If $A$ is block triangular:

$$\det\begin{pmatrix} A_{11} & A_{12} \\ 0 & A_{22} \end{pmatrix} = \det(A_{11}) \cdot \det(A_{22})$$

**Theorem 2.6.4 (Schur Complement).** For a block matrix with $A_{11}$ invertible:

$$\det\begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix} = \det(A_{11}) \cdot \det(A_{22} - A_{21}A_{11}^{-1}A_{12})$$

where $A_{22} - A_{21}A_{11}^{-1}A_{12}$ is the *Schur complement* of $A_{11}$.

*ML connection:* The Schur complement appears in Gaussian process regression. When partitioning a joint Gaussian over observed and unobserved variables, the conditional distribution's covariance is exactly the Schur complement of the observed block — enabling efficient posterior computation.

---

## 6. Geometric Interpretation

### 6.1 Volume Scaling

**Theorem 2.6.5 (Determinant as Volume Factor).** If $T: \mathbb{R}^n \to \mathbb{R}^n$ is a linear transformation with matrix $A$, and $S \subseteq \mathbb{R}^n$ is a measurable region, then:

$$\text{Vol}(T(S)) = |\det(A)| \cdot \text{Vol}(S)$$

*Proof (for $n = 2$).* The columns of $A$ map the standard basis vectors $\mathbf{e}_1, \mathbf{e}_2$ to $\mathbf{a}_1, \mathbf{a}_2$. The unit square maps to the parallelogram spanned by $\mathbf{a}_1$ and $\mathbf{a}_2$, whose area is $|\mathbf{a}_1 \times \mathbf{a}_2| = |a_{11}a_{22} - a_{12}a_{21}| = |\det(A)|$. For general regions, approximate by small squares and take limits. $\square$

**Example 2.3.8:** The columns of $A = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$ are $\mathbf{a}_1 = (3,0)$ and $\mathbf{a}_2 = (1,2)$.

$$\det(A) = (3)(2) - (1)(0) = 6$$

The parallelogram spanned by $(3,0)$ and $(1,2)$ has area $|\det(A)| = 6$. Since $\det(A) > 0$, the transformation preserves orientation (the vectors form a counterclockwise pair).

**Example 2.3.9 (3D Volume):** The columns of $B = \begin{pmatrix} 1 & 0 & 1 \\ 0 & 2 & 0 \\ 0 & 0 & 3 \end{pmatrix}$ span a parallelepiped.

$$\det(B) = 1 \cdot (2 \cdot 3 - 0) - 0 + 1 \cdot (0 - 0) = 6$$

The parallelepiped has volume $|\det(B)| = 6$, which equals $1 \times 2 \times 3$ since the columns are orthogonal.

```
DETERMINANT AS SIGNED AREA (2D)

     det > 0                    det < 0
     (preserves orientation)    (reverses orientation)

        ▲ a₂                      a₁ ▲
       ╱│                           │╲
      ╱ │                           │ ╲
     ╱  │                           │  ╲
    ╱   │                           │   ╲
   O────┼──▶ a₁                ◀───┼────O
                                    ▼ a₂

     Counterclockwise:            Clockwise:
     positive area                negative area
```

### 6.2 Orientation

**Definition 2.6.6 (Orientation).** A linear transformation $A$ is *orientation-preserving* if $\det(A) > 0$ and *orientation-reversing* if $\det(A) < 0$.

**Example.** Reflection across the $x$-axis: $\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ has $\det = -1$ (reverses orientation). Rotation by $\theta$: $\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$ has $\det = 1$ (preserves orientation and volume).

---

## 7. Determinant and Invertibility

**Theorem 2.6.6 (The Invertibility Theorem — Determinant Conditions).** The following are equivalent for $A \in \mathbb{R}^{n \times n}$:

1. $\det(A) \neq 0$
2. $A$ is invertible
3. $\text{rank}(A) = n$
4. The columns of $A$ are linearly independent
5. The rows of $A$ are linearly independent
6. $A\mathbf{x} = \mathbf{0}$ has only the trivial solution
7. $A\mathbf{x} = \mathbf{b}$ has a unique solution for every $\mathbf{b}$
8. $0$ is not an eigenvalue of $A$
9. The row echelon form of $A$ has $n$ pivots

*Proof of $(1) \Leftrightarrow (2)$.* If $A$ is invertible, $\det(A)\det(A^{-1}) = \det(I) = 1$, so $\det(A) \neq 0$. Conversely, if $\det(A) \neq 0$, define $A^{-1} = \frac{1}{\det(A)} \text{adj}(A)$ where $\text{adj}(A)$ is the adjugate (matrix of cofactors, transposed). One verifies $A \cdot \frac{1}{\det(A)}\text{adj}(A) = I$ using the cofactor expansion and the fact that expanding along a "wrong" row gives $0$. $\square$

**Definition 2.6.7 (Adjugate Matrix).** The *adjugate* (or *classical adjoint*) of $A$ is:

$$\text{adj}(A) = (C_{ji})_{n \times n}$$

where $C_{ji}$ is the $(j,i)$-cofactor. Note the transposition: entry $(i,j)$ of $\text{adj}(A)$ is $C_{ji}$.

**Theorem 2.6.7.** $A \cdot \text{adj}(A) = \det(A) \cdot I$, so $A^{-1} = \frac{1}{\det(A)}\text{adj}(A)$ when $\det(A) \neq 0$.

---

## 8. Cramer's Rule

**Theorem 2.6.8 (Cramer's Rule).** If $A \in \mathbb{R}^{n \times n}$ with $\det(A) \neq 0$, the unique solution of $A\mathbf{x} = \mathbf{b}$ is:

$$x_j = \frac{\det(A_j)}{\det(A)} \qquad j = 1, \ldots, n$$

where $A_j$ is the matrix $A$ with column $j$ replaced by $\mathbf{b}$.

*Proof.* From $A^{-1} = \frac{1}{\det(A)}\text{adj}(A)$, the $j$-th component of $\mathbf{x} = A^{-1}\mathbf{b}$ is:

$$x_j = \frac{1}{\det(A)} \sum_{i=1}^n C_{ji} b_i$$

The sum $\sum_{i} C_{ji} b_i$ is precisely the cofactor expansion of $\det(A_j)$ along column $j$ (where column $j$ has been replaced by $\mathbf{b}$). $\square$

**Example.** Solve $\begin{pmatrix} 2 & 1 \\ 5 & 3 \end{pmatrix}\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} 4 \\ 7 \end{pmatrix}$:

$$\det(A) = 6 - 5 = 1$$

$$x = \frac{\det\begin{pmatrix} 4 & 1 \\ 7 & 3 \end{pmatrix}}{\det(A)} = \frac{12 - 7}{1} = 5, \qquad y = \frac{\det\begin{pmatrix} 2 & 4 \\ 5 & 7 \end{pmatrix}}{\det(A)} = \frac{14 - 20}{1} = -6$$

**Practical note.** Cramer's rule is $O(n \cdot n!) = O((n+1)!)$ with cofactor expansion — theoretically elegant but computationally impractical for $n > 3$. Gaussian elimination ($O(n^3)$) is always preferred in practice.

*ML connection:* While Cramer's rule is never used in ML for solving systems, the formula $x_j = \det(A_j)/\det(A)$ provides theoretical insight: each variable depends on *all* entries of $A$ and $\mathbf{b}$ through a ratio of polynomials. This perspective connects to the theory of linear regression coefficients and their sensitivity to individual data points.

---

## 9. Special Determinant Formulas

### 9.1 Vandermonde Determinant

**Theorem 2.6.9.** The Vandermonde determinant:

$$\det\begin{pmatrix} 1 & x_1 & x_1^2 & \cdots & x_1^{n-1} \\ 1 & x_2 & x_2^2 & \cdots & x_2^{n-1} \\ \vdots & & & & \vdots \\ 1 & x_n & x_n^2 & \cdots & x_n^{n-1} \end{pmatrix} = \prod_{1 \leq i < j \leq n} (x_j - x_i)$$

This is nonzero iff all $x_i$ are distinct — confirming that polynomial interpolation through $n$ distinct points has a unique solution.

**Example 2.3.10:** For $x_1 = 1, x_2 = 2, x_3 = 3$, the $3 \times 3$ Vandermonde matrix is:

$$V = \begin{pmatrix} 1 & 1 & 1 \\ 1 & 2 & 4 \\ 1 & 3 & 9 \end{pmatrix}$$

By the product formula: $\det(V) = (x_2 - x_1)(x_3 - x_1)(x_3 - x_2) = (2-1)(3-1)(3-2) = 1 \cdot 2 \cdot 1 = 2$.

Verify by cofactor expansion along row 1: $\det(V) = 1(18-12) - 1(9-4) + 1(3-2) = 6 - 5 + 1 = 2 \; \checkmark$

### 9.2 Matrix Determinant Lemma

**Theorem 2.6.10 (Matrix Determinant Lemma).** If $A$ is invertible and $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$:

$$\det(A + \mathbf{u}\mathbf{v}^T) = (1 + \mathbf{v}^T A^{-1} \mathbf{u}) \det(A)$$

*ML connection:* This lemma is used in Gaussian process inference when rank-1 updates are made to the kernel matrix. Instead of recomputing the full determinant ($O(n^3)$), a rank-1 update costs only $O(n^2)$. It also appears in the Woodbury identity for efficient online learning.

### 9.3 Determinant of Block Diagonal Matrices

**Theorem 2.6.11.** For block diagonal $A = \text{diag}(A_1, A_2, \ldots, A_k)$:

$$\det(A) = \prod_{i=1}^k \det(A_i)$$

---

## 10. Determinants in ML and AI

### 10.1 Jacobian Determinant in Normalizing Flows

A normalizing flow transforms a simple distribution $p_z(\mathbf{z})$ (e.g., Gaussian) through an invertible function $f$ to produce a complex distribution. By the change of variables formula:

$$p_x(\mathbf{x}) = p_z(f^{-1}(\mathbf{x})) \left|\det\left(\frac{\partial f^{-1}}{\partial \mathbf{x}}\right)\right|$$

```
NORMALIZING FLOW

   Simple distribution          Invertible          Complex distribution
                               transformation
   p_z(z)                         f                    p_x(x)

      ╱╲                                                 ╱╲
     ╱  ╲                     ─────────▶              ╱    ╲
    ╱    ╲            f                              ╱  ╱╲  ╲
   ╱      ╲         det(J)                          ╱  ╱  ╲  ╲
  ╱────────╲      adjusts for              ────────╱──╱────╲──╲────
              volume change

   log p_x(x) = log p_z(f⁻¹(x)) + log |det(J_{f⁻¹}(x))|
```

The $\log|\det(J)|$ term corrects for how $f$ stretches or compresses probability density — regions where $f$ expands volume have lower density, and vice versa. Modern flow architectures (RealNVP, Glow, Neural Spline Flows) design $f$ so that $J$ is triangular, making $\det(J) = \prod_i J_{ii}$ computable in $O(d)$.

### 10.2 Multivariate Gaussian Density

The density of a multivariate Gaussian $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$ is:

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{n/2} |\det(\Sigma)|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1}(\mathbf{x} - \boldsymbol{\mu})\right)$$

The determinant $\det(\Sigma)$ appears in the normalization constant. Geometrically, $|\det(\Sigma)|^{1/2}$ is proportional to the volume of the confidence ellipsoid — a larger determinant means more spread-out data.

**Log-likelihood:** $\log p(\mathbf{x}) = -\frac{n}{2}\log(2\pi) - \frac{1}{2}\log\det(\Sigma) - \frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x} - \boldsymbol{\mu})$

Computing $\log\det(\Sigma)$ efficiently: use the Cholesky decomposition $\Sigma = LL^T$ (since $\Sigma$ is positive definite), then $\log\det(\Sigma) = 2\sum_i \log L_{ii}$.

### 10.3 Fisher Information and Model Geometry

The Fisher information matrix $F = \mathbb{E}\left[\nabla_\theta \log p(x|\theta) \nabla_\theta \log p(x|\theta)^T\right]$ plays a central role in statistics and natural gradient descent. Its determinant, $\det(F)$, measures the "volume" of distinguishable models in a neighborhood of $\theta$ — larger determinants mean the model is more sensitive to parameter changes (more "informative" data).

---

## 11. Determinants in Quantum Computing

In quantum mechanics, the Slater determinant describes the wave function of a system of identical fermions (particles obeying the Pauli exclusion principle):

$$\Psi(x_1, \ldots, x_n) = \frac{1}{\sqrt{n!}} \det\begin{pmatrix} \phi_1(x_1) & \cdots & \phi_n(x_1) \\ \vdots & \ddots & \vdots \\ \phi_1(x_n) & \cdots & \phi_n(x_n) \end{pmatrix}$$

The determinant ensures antisymmetry: swapping two particles flips the sign of $\Psi$, enforcing the Pauli exclusion principle. If two particles occupy the same state ($\phi_i = \phi_j$), the determinant is $0$ — the state is forbidden.

---

## 12. Eigenvalues and Determinants

A key connection (developed fully in [Eigenvalues & Eigenvectors](eigenvalues.md)) is that the determinant equals the product of eigenvalues:

$$\det(A) = \prod_{i=1}^{n} \lambda_i$$

**Example 2.3.11:** Let $A = \begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix}$. First, $\det(A) = (4)(3) - (1)(2) = 10$.

Now find eigenvalues from $\det(A - \lambda I) = 0$:

$$(4 - \lambda)(3 - \lambda) - 2 = \lambda^2 - 7\lambda + 10 = (\lambda - 5)(\lambda - 2) = 0$$

So $\lambda_1 = 5$, $\lambda_2 = 2$, and $\lambda_1 \cdot \lambda_2 = 5 \cdot 2 = 10 = \det(A) \; \checkmark$

**Example 2.3.12:** Let $B = \begin{pmatrix} 2 & 0 & 0 \\ 0 & 3 & 0 \\ 0 & 0 & -1 \end{pmatrix}$ (diagonal, so eigenvalues are the diagonal entries).

$$\det(B) = 2 \cdot 3 \cdot (-1) = -6 = \lambda_1 \lambda_2 \lambda_3 \; \checkmark$$

The negative determinant tells us $B$ reverses orientation (due to the $\lambda_3 = -1$ eigenvalue).

---

## 13. Summary of Key Results

| Result | Statement |
|--------|-----------|
| Leibniz formula | $\det(A) = \sum_{\sigma \in S_n} \text{sgn}(\sigma) \prod_i a_{i,\sigma(i)}$ |
| Cofactor expansion | $\det(A) = \sum_j (-1)^{i+j} a_{ij} M_{ij}$ (along any row $i$) |
| Multiplicativity | $\det(AB) = \det(A)\det(B)$ |
| Transpose | $\det(A^T) = \det(A)$ |
| Invertibility | $A$ invertible $\iff$ $\det(A) \neq 0$ |
| Triangular matrices | $\det = \prod a_{ii}$ |
| Scalar multiple | $\det(cA) = c^n\det(A)$ |
| Volume scaling | $\text{Vol}(A(S)) = |\det(A)| \cdot \text{Vol}(S)$ |
| Cramer's rule | $x_j = \det(A_j)/\det(A)$ |
| Matrix det. lemma | $\det(A + \mathbf{u}\mathbf{v}^T) = (1 + \mathbf{v}^T A^{-1}\mathbf{u})\det(A)$ |

---

## Exercises

**★ Basic**

1. Compute $\det\begin{pmatrix} 3 & 7 \\ 1 & -4 \end{pmatrix}$ and verify that it equals the area of the parallelogram spanned by the column vectors.

2. Use cofactor expansion along the first row to compute $\det\begin{pmatrix} 1 & 2 & 3 \\ 0 & 4 & 5 \\ 1 & 0 & 6 \end{pmatrix}$.

3. If $A$ is $5 \times 5$ with $\det(A) = 3$, compute $\det(2A)$ and $\det(A^{-1})$.

4. True or false: $\det(A + B) = \det(A) + \det(B)$. If false, give a counterexample.

**★★ Intermediate**

5. Prove that if $A$ has two identical rows, then $\det(A) = 0$. (Hint: swap those rows and use property (b).)

6. Use row reduction to compute the determinant of $\begin{pmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \\ 9 & 10 & 11 & 12 \\ 13 & 14 & 15 & 16 \end{pmatrix}$. Explain geometrically why the result makes sense.

7. Prove the matrix determinant lemma (Theorem 2.6.10) using the identity $\begin{pmatrix} A & \mathbf{u} \\ \mathbf{v}^T & 1 \end{pmatrix} = \begin{pmatrix} I & \mathbf{u} \\ 0 & 1 \end{pmatrix}\begin{pmatrix} A & 0 \\ \mathbf{v}^T & 1 - \mathbf{v}^T A^{-1}\mathbf{u} \end{pmatrix}$ after a Schur complement argument.

8. Show that for orthogonal matrices ($Q^TQ = I$), $\det(Q) = \pm 1$. Which orthogonal matrices have $\det = +1$, and which have $\det = -1$?

**★★★ Challenging**

9. **(Normalizing flows)** A flow layer transforms $\mathbf{z} \in \mathbb{R}^2$ via $f(z_1, z_2) = (z_1, z_2 + \alpha \cdot \tanh(z_1))$. Compute the Jacobian $J_f$, verify that $\det(J_f) = 1$ for all $\mathbf{z}$ and $\alpha$, and explain what this implies about the density transformation.

10. **(Hadamard's inequality)** Prove that for a positive definite matrix $A$ with columns $\mathbf{a}_1, \ldots, \mathbf{a}_n$: $|\det(A)| \leq \prod_{i=1}^n \|\mathbf{a}_i\|$. Interpret this as: the volume of a parallelepiped is at most the product of its edge lengths, with equality iff the edges are orthogonal.

11. Compute the $4 \times 4$ Vandermonde determinant for $x_1 = 1, x_2 = 2, x_3 = 3, x_4 = 4$ both directly (cofactor expansion) and via the product formula. Verify they agree.

---

## Related Topics

- [Matrices](matrices.md) — matrix operations, types, and properties
- [Systems of Equations](systems-of-equations.md) — row reduction and solution methods
- [Linear Transformations](linear-transformations.md) — the functions whose volume-scaling factor is the determinant
- [Eigenvalues & Eigenvectors](eigenvalues.md) — $\det(A) = \prod \lambda_i$ and the characteristic polynomial
- [Matrix Decompositions](matrix-decompositions.md) — LU decomposition computes determinants efficiently
- [Multivariable Calculus](../calculus/multivariable-calculus.md) — Jacobians and the change of variables formula

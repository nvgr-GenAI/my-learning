# Systems of Linear Equations

Solving systems of linear equations is one of the oldest problems in mathematics, and it remains one of the most practically important. When you fit a linear regression model, you solve a system of equations. When you compute the PageRank of a web graph, you solve a system of equations. When you find the equilibrium of a Markov chain, you solve a system of equations. The question "$A\mathbf{x} = \mathbf{b}$: does a solution exist, is it unique, and how do we find it?" underpins much of computational science and machine learning.

---

## Prerequisites

- [Vectors](vectors.md) — vector operations, linear independence
- [Matrices](matrices.md) — matrix operations, rank, inverse

---

## 1. The Fundamental Problem

**Definition 2.3.1 (System of Linear Equations).** A *system of $m$ linear equations in $n$ unknowns* is:

$$\begin{aligned}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n &= b_1 \\
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n &= b_2 \\
&\;\vdots \\
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n &= b_m
\end{aligned}$$

In matrix form: $A\mathbf{x} = \mathbf{b}$ where $A \in \mathbb{R}^{m \times n}$, $\mathbf{x} \in \mathbb{R}^n$, $\mathbf{b} \in \mathbb{R}^m$.

**Definition 2.3.2 (Augmented Matrix).** The *augmented matrix* of the system $A\mathbf{x} = \mathbf{b}$ is:

$$[A \mid \mathbf{b}] = \left(\begin{array}{cccc|c} a_{11} & a_{12} & \cdots & a_{1n} & b_1 \\ a_{21} & a_{22} & \cdots & a_{2n} & b_2 \\ \vdots & \vdots & \ddots & \vdots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} & b_m \end{array}\right)$$

**Example 2.5.1:** Write the system $2x + 3y = 7$, $x - y = 1$ in matrix form $A\mathbf{x} = \mathbf{b}$ and as an augmented matrix.

$$A = \begin{pmatrix} 2 & 3 \\ 1 & -1 \end{pmatrix}, \quad \mathbf{x} = \begin{pmatrix} x \\ y \end{pmatrix}, \quad \mathbf{b} = \begin{pmatrix} 7 \\ 1 \end{pmatrix} \qquad \Rightarrow \qquad [A \mid \mathbf{b}] = \left(\begin{array}{cc|c} 2 & 3 & 7 \\ 1 & -1 & 1 \end{array}\right)$$

```
THREE GEOMETRIC POSSIBILITIES (2 equations, 2 unknowns)

  Unique solution         No solution          Infinitely many
  (lines intersect)       (parallel lines)     (same line)

      y                      y                     y
      │   ╱                  │  ╱  ╱               │  ╱
      │  ╱                   │ ╱  ╱                │ ╱
      │ ╱                    │╱  ╱                 │╱
  ────●──── x            ───╱──╱── x           ───╱──── x
     ╱│                    ╱  ╱                  ╱ │
    ╱ │                   ╱  ╱                  ╱  │

  rank(A) = rank[A|b]    rank(A) < rank[A|b]   rank(A) = rank[A|b]
          = n                                          < n
```

*ML connection:* In linear regression with $m$ data points and $n$ features, the normal equations $X^T X \boldsymbol{\theta} = X^T \mathbf{y}$ form a system of $n$ equations in $n$ unknowns. If $m > n$ (more data than features, the typical case), the original system $X\boldsymbol{\theta} = \mathbf{y}$ is overdetermined and generally has no exact solution — we seek the least-squares approximate solution instead.

---

## 2. Elementary Row Operations

**Definition 2.3.3 (Elementary Row Operations).** The three *elementary row operations* on a matrix are:

1. **Swap** two rows: $R_i \leftrightarrow R_j$
2. **Scale** a row by a nonzero constant: $R_i \leftarrow cR_i$ where $c \neq 0$
3. **Add** a multiple of one row to another: $R_i \leftarrow R_i + cR_j$

**Theorem 2.3.1.** Elementary row operations do not change the solution set of a linear system. That is, if $[A|\mathbf{b}]$ is transformed to $[A'|\mathbf{b}']$ by elementary row operations, then $A\mathbf{x} = \mathbf{b}$ and $A'\mathbf{x} = \mathbf{b}'$ have the same solutions.

*Proof.* Each elementary row operation corresponds to left-multiplication by an invertible elementary matrix $E$. The system $EA\mathbf{x} = E\mathbf{b}$ has the same solutions as $A\mathbf{x} = \mathbf{b}$ because $E$ is invertible: if $A\mathbf{x}_0 = \mathbf{b}$ then $EA\mathbf{x}_0 = E\mathbf{b}$, and conversely if $EA\mathbf{x}_0 = E\mathbf{b}$ then $A\mathbf{x}_0 = E^{-1}(E\mathbf{b}) = \mathbf{b}$. $\square$

**Definition 2.3.4 (Elementary Matrices).** Each elementary row operation corresponds to an *elementary matrix* $E$ obtained by performing the operation on $I$:

$$E_{\text{swap}} = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \quad E_{\text{scale}} = \begin{pmatrix} c & 0 \\ 0 & 1 \end{pmatrix} \quad E_{\text{add}} = \begin{pmatrix} 1 & c \\ 0 & 1 \end{pmatrix}$$

Left-multiplying $A$ by $E$ performs the corresponding row operation on $A$.

**Example 2.5.2:** Apply each elementary row operation to $A = \begin{pmatrix} 1 & 4 \\ 2 & 3 \end{pmatrix}$.

$$\text{Swap } R_1 \leftrightarrow R_2: \quad \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}\begin{pmatrix} 1 & 4 \\ 2 & 3 \end{pmatrix} = \begin{pmatrix} 2 & 3 \\ 1 & 4 \end{pmatrix}$$

$$\text{Scale } R_1 \leftarrow 3R_1: \quad \begin{pmatrix} 3 & 0 \\ 0 & 1 \end{pmatrix}\begin{pmatrix} 1 & 4 \\ 2 & 3 \end{pmatrix} = \begin{pmatrix} 3 & 12 \\ 2 & 3 \end{pmatrix}$$

$$\text{Add } R_2 \leftarrow R_2 - 2R_1: \quad \begin{pmatrix} 1 & 0 \\ -2 & 1 \end{pmatrix}\begin{pmatrix} 1 & 4 \\ 2 & 3 \end{pmatrix} = \begin{pmatrix} 1 & 4 \\ 0 & -5 \end{pmatrix}$$

---

## 3. Gaussian Elimination

### 3.1 Row Echelon Form

**Definition 2.3.5 (Row Echelon Form).** A matrix is in *row echelon form* (REF) if:

1. All zero rows are at the bottom.
2. The leading entry (pivot) of each nonzero row is to the right of the pivot in the row above.
3. All entries below a pivot are zero.

$$\begin{pmatrix} \boxed{2} & 3 & 1 & 5 \\ 0 & \boxed{1} & -2 & 3 \\ 0 & 0 & 0 & \boxed{4} \\ 0 & 0 & 0 & 0 \end{pmatrix} \quad \text{(REF: pivots boxed)}$$

**Definition 2.3.6 (Reduced Row Echelon Form).** A matrix is in *reduced row echelon form* (RREF) if it is in REF and additionally:

4. Each pivot is 1.
5. Each pivot is the only nonzero entry in its column.

$$\begin{pmatrix} \boxed{1} & 0 & 5 & 0 \\ 0 & \boxed{1} & -2 & 0 \\ 0 & 0 & 0 & \boxed{1} \\ 0 & 0 & 0 & 0 \end{pmatrix} \quad \text{(RREF)}$$

**Example 2.5.3:** Reduce the augmented matrix to REF.

$$\left(\begin{array}{ccc|c} 1 & 3 & 1 & 9 \\ 1 & 1 & -1 & 1 \\ 3 & 11 & 5 & 35 \end{array}\right) \xrightarrow{R_2 \leftarrow R_2 - R_1} \left(\begin{array}{ccc|c} 1 & 3 & 1 & 9 \\ 0 & -2 & -2 & -8 \\ 3 & 11 & 5 & 35 \end{array}\right) \xrightarrow{R_3 \leftarrow R_3 - 3R_1} \left(\begin{array}{ccc|c} 1 & 3 & 1 & 9 \\ 0 & -2 & -2 & -8 \\ 0 & 2 & 2 & 8 \end{array}\right)$$

$$\xrightarrow{R_3 \leftarrow R_3 + R_2} \left(\begin{array}{ccc|c} 1 & 3 & 1 & 9 \\ 0 & -2 & -2 & -8 \\ 0 & 0 & 0 & 0 \end{array}\right) \quad \text{(REF)}$$

The zero row signals $\text{rank}(A) = 2 < 3 = n$, so the system has infinitely many solutions with one free variable.

**Theorem 2.3.2 (Uniqueness of RREF).** Every matrix has a unique reduced row echelon form.

### 3.2 The Algorithm

**Gaussian Elimination** transforms $[A|\mathbf{b}]$ to REF. **Gauss-Jordan Elimination** continues to RREF.

```
GAUSSIAN ELIMINATION: STEP BY STEP

Example: solve   x + 2y - z = 3
                2x + 5y + z = 8
                 x + 3y + 4z = 7

Step 1: Form augmented matrix

    ┌ 1  2  -1 │ 3 ┐
    │ 2  5   1 │ 8 │
    └ 1  3   4 │ 7 ┘

Step 2: Eliminate below pivot (column 1)

    R₂ ← R₂ - 2R₁        ┌ 1  2  -1 │  3 ┐
    R₃ ← R₃ - R₁          │ 0  1   3 │  2 │
                           └ 0  1   5 │  4 ┘

Step 3: Eliminate below pivot (column 2)

    R₃ ← R₃ - R₂          ┌ 1  2  -1 │  3 ┐
                           │ 0  1   3 │  2 │   ← REF
                           └ 0  0   2 │  2 ┘

Step 4: Back substitution

    From R₃:  2z = 2     →  z = 1
    From R₂:  y + 3(1) = 2  →  y = -1
    From R₁:  x + 2(-1) - 1 = 3  →  x = 6

    Solution: (x, y, z) = (6, -1, 1)
```

**Example 2.5.4 (Back Substitution).** Given the REF augmented matrix, recover the solution bottom-up:

$$\left(\begin{array}{ccc|c} 2 & 1 & -1 & 8 \\ 0 & 3 & 2 & 1 \\ 0 & 0 & 5 & 10 \end{array}\right)$$

- **Row 3:** $5z = 10 \;\Rightarrow\; z = 2$
- **Row 2:** $3y + 2(2) = 1 \;\Rightarrow\; 3y = -3 \;\Rightarrow\; y = -1$
- **Row 1:** $2x + (-1) - 2 = 8 \;\Rightarrow\; 2x = 11 \;\Rightarrow\; x = 11/2$

Solution: $(x, y, z) = (11/2,\; -1,\; 2)$.

**Theorem 2.3.3 (Complexity).** Gaussian elimination on an $n \times n$ system requires $\frac{2}{3}n^3 + O(n^2)$ floating-point operations.

*ML connection:* Although Gaussian elimination has $O(n^3)$ complexity, linear regression via the normal equations $X^T X \boldsymbol{\theta} = X^T \mathbf{y}$ uses this (via Cholesky or LU decomposition) when $n$ (number of features) is moderate. For $n$ in the thousands, iterative methods like conjugate gradient become preferable. For the millions of parameters in neural networks, we abandon exact solutions entirely in favor of stochastic gradient descent.

---

## 4. Pivot Variables and Free Variables

**Definition 2.3.7 (Pivot and Free Variables).** After reducing $[A|\mathbf{b}]$ to REF:

- **Pivot variables** correspond to columns containing pivots.
- **Free variables** correspond to columns without pivots.

Free variables can take any value; pivot variables are determined by the free variables via back substitution.

```
PIVOT vs FREE VARIABLES

    RREF of [A|b]:

    ┌ 1  0  3  0 │ 5 ┐
    │ 0  1 -1  0 │ 2 │
    └ 0  0  0  1 │ 4 ┘

    Pivot columns: 1, 2, 4  →  x₁, x₂, x₄ are pivot variables
    Non-pivot column: 3     →  x₃ is a FREE variable (let x₃ = t)

    Solution:  x₁ = 5 - 3t
               x₂ = 2 + t
               x₃ = t          (free)
               x₄ = 4

    In vector form:

    x = ┌ 5 ┐     ┌ -3 ┐
        │ 2 │ + t │  1 │     t ∈ ℝ
        │ 0 │     │  1 │
        └ 4 ┘     └  0 ┘

      particular    null space
      solution      direction
```

---

## 5. Existence and Uniqueness of Solutions

**Theorem 2.3.4 (Rouche-Capelli Theorem).** The system $A\mathbf{x} = \mathbf{b}$ is consistent (has at least one solution) if and only if:

$$\text{rank}(A) = \text{rank}([A \mid \mathbf{b}])$$

**Theorem 2.3.5 (Complete Classification).** For $A \in \mathbb{R}^{m \times n}$ with $r = \text{rank}(A)$:

| Condition | Solutions | Free Variables |
|-----------|-----------|----------------|
| $\text{rank}(A) < \text{rank}([A\|\mathbf{b}])$ | None (inconsistent) | N/A |
| $\text{rank}(A) = \text{rank}([A\|\mathbf{b}]) = n$ | Exactly one | 0 |
| $\text{rank}(A) = \text{rank}([A\|\mathbf{b}]) = r < n$ | Infinitely many | $n - r$ |

*Proof.* If $\text{rank}(A) < \text{rank}([A|\mathbf{b}])$, then the REF of $[A|\mathbf{b}]$ contains a row of the form $(0 \; 0 \; \cdots \; 0 \mid c)$ with $c \neq 0$, giving the contradictory equation $0 = c$. If ranks are equal, the system is consistent. The number of free variables is $n - r$: if $r = n$, every variable is a pivot variable and the solution is unique; if $r < n$, the $n - r$ free variables each range over $\mathbb{R}$, giving infinitely many solutions. $\square$

```
DECISION TREE FOR Ax = b

                    Compute rank(A) and rank([A|b])
                              │
                    ┌─────────┴─────────┐
                    │                   │
            rank(A) = rank([A|b])   rank(A) < rank([A|b])
                    │                   │
                    │               NO SOLUTION
            ┌───────┴───────┐
            │               │
        rank = n        rank < n
            │               │
      UNIQUE SOLUTION   INFINITELY MANY
                        (n - rank free vars)
```

**Example 2.5.5 (No Solution).** Show that the system $x + 2y = 3$, $2x + 4y = 5$ is inconsistent.

$$\left(\begin{array}{cc|c} 1 & 2 & 3 \\ 2 & 4 & 5 \end{array}\right) \xrightarrow{R_2 \leftarrow R_2 - 2R_1} \left(\begin{array}{cc|c} 1 & 2 & 3 \\ 0 & 0 & -1 \end{array}\right)$$

Row 2 reads $0x + 0y = -1$, which is impossible. Here $\text{rank}(A) = 1$ but $\text{rank}([A|\mathbf{b}]) = 2$, confirming no solution exists.

**Example 2.5.6 (Infinitely Many Solutions).** Solve $x + 2y + 3z = 4$, $2x + 4y + 6z = 8$.

$$\left(\begin{array}{ccc|c} 1 & 2 & 3 & 4 \\ 2 & 4 & 6 & 8 \end{array}\right) \xrightarrow{R_2 \leftarrow R_2 - 2R_1} \left(\begin{array}{ccc|c} 1 & 2 & 3 & 4 \\ 0 & 0 & 0 & 0 \end{array}\right)$$

$\text{rank}(A) = \text{rank}([A|\mathbf{b}]) = 1$ with $n = 3$, giving $3 - 1 = 2$ free variables. Let $y = s$, $z = t$:

$$x = 4 - 2s - 3t, \quad y = s, \quad z = t \qquad (s, t \in \mathbb{R})$$

*ML connection:* In linear regression with $m$ observations and $n$ features: if $m < n$ (underdetermined, more features than data), the system has infinitely many solutions and we need regularization to pick one. If $m = n$ and $X$ is invertible, the solution is unique but likely overfits. If $m > n$ (typical), the system is overdetermined and we find the least-squares solution. This is precisely the bias-variance tradeoff in action.

---

## 6. Homogeneous Systems

**Definition 2.3.8 (Homogeneous System).** A system $A\mathbf{x} = \mathbf{0}$ is called *homogeneous*. It always has the trivial solution $\mathbf{x} = \mathbf{0}$.

**Theorem 2.3.6.** The set of solutions to $A\mathbf{x} = \mathbf{0}$ forms a subspace of $\mathbb{R}^n$, called the *null space* (or *kernel*) of $A$:

$$\text{Null}(A) = \ker(A) = \{\mathbf{x} \in \mathbb{R}^n : A\mathbf{x} = \mathbf{0}\}$$

*Proof.* (i) $A\mathbf{0} = \mathbf{0}$, so $\mathbf{0} \in \ker(A)$. (ii) If $A\mathbf{x} = \mathbf{0}$ and $A\mathbf{y} = \mathbf{0}$, then $A(\mathbf{x} + \mathbf{y}) = A\mathbf{x} + A\mathbf{y} = \mathbf{0}$. (iii) If $A\mathbf{x} = \mathbf{0}$, then $A(c\mathbf{x}) = cA\mathbf{x} = \mathbf{0}$. $\square$

**Theorem 2.3.7 (Rank-Nullity Theorem).** For $A \in \mathbb{R}^{m \times n}$:

$$\text{rank}(A) + \text{nullity}(A) = n$$

where $\text{nullity}(A) = \dim(\ker(A))$.

**Corollary.** $A\mathbf{x} = \mathbf{0}$ has a nontrivial solution if and only if $\text{rank}(A) < n$.

**Corollary.** If $A$ is $m \times n$ with $m < n$ (more unknowns than equations), then $A\mathbf{x} = \mathbf{0}$ always has a nontrivial solution.

*Proof.* $\text{rank}(A) \leq m < n$, so $\text{nullity}(A) = n - \text{rank}(A) \geq n - m > 0$. $\square$

**Example 2.5.7 (Null Space).** Find all solutions to $A\mathbf{x} = \mathbf{0}$ where $A = \begin{pmatrix} 1 & 2 & 1 \\ 2 & 4 & 0 \end{pmatrix}$.

$$\left(\begin{array}{ccc} 1 & 2 & 1 \\ 2 & 4 & 0 \end{array}\right) \xrightarrow{R_2 \leftarrow R_2 - 2R_1} \begin{pmatrix} 1 & 2 & 1 \\ 0 & 0 & -2 \end{pmatrix}$$

$\text{rank}(A) = 2$, $n = 3$, so $\text{nullity} = 1$. With $y = t$ (free), from Row 2: $z = 0$; from Row 1: $x = -2t$. Thus:

$$\ker(A) = \text{span}\left\{\begin{pmatrix} -2 \\ 1 \\ 0 \end{pmatrix}\right\}$$

### 6.1 General Solution Structure

**Theorem 2.3.8.** If $\mathbf{x}_p$ is any particular solution to $A\mathbf{x} = \mathbf{b}$, then the general solution is:

$$\mathbf{x} = \mathbf{x}_p + \mathbf{x}_h$$

where $\mathbf{x}_h \in \ker(A)$ is any solution to the homogeneous system.

*Proof.* If $A\mathbf{x}_p = \mathbf{b}$ and $A\mathbf{x}_h = \mathbf{0}$, then $A(\mathbf{x}_p + \mathbf{x}_h) = \mathbf{b}$. Conversely, if $A\mathbf{x}' = \mathbf{b}$, then $A(\mathbf{x}' - \mathbf{x}_p) = \mathbf{0}$, so $\mathbf{x}' - \mathbf{x}_p \in \ker(A)$. $\square$

**Example 2.5.8:** Solve $\begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\mathbf{x} = \begin{pmatrix} 3 \\ 6 \end{pmatrix}$.

Row reduce: $R_2 \leftarrow R_2 - 2R_1$ gives $\begin{pmatrix} 1 & 2 & | & 3 \\ 0 & 0 & | & 0\end{pmatrix}$. One particular solution: $\mathbf{x}_p = (3, 0)^T$.

The homogeneous system $A\mathbf{x} = \mathbf{0}$ gives $x_1 = -2t$, $x_2 = t$, so $\ker(A) = \text{span}\{(-2, 1)^T\}$.

General solution: $\mathbf{x} = \begin{pmatrix} 3 \\ 0 \end{pmatrix} + t\begin{pmatrix} -2 \\ 1 \end{pmatrix}$, $t \in \mathbb{R}$ — a line through $(3, 0)$ in the direction of the null space.

```
SOLUTION STRUCTURE: general = particular + homogeneous

                        ↗ x_p + x_h₁
                      ╱
    ─────── x_p ────●─── x_p + x_h₂     (affine subspace)
                      ╲
                        ↘ x_p + x_h₃

    The solution set is a "shifted" copy of the null space,
    passing through x_p instead of through the origin.

    If null space is {0}  → single point (unique solution)
    If null space is 1D   → line through x_p
    If null space is 2D   → plane through x_p
```

*ML connection:* In neural networks, overparameterized models ($n \gg m$) have many weight configurations that achieve zero training loss — the solution set is a high-dimensional manifold. The specific solution found depends on the optimizer and initialization, which is why different training runs with different random seeds converge to different (but equally good) solutions. Understanding this null space structure is key to the theory of implicit regularization.

---

## 7. The Four Fundamental Subspaces

For $A \in \mathbb{R}^{m \times n}$ with $r = \text{rank}(A)$, there are four fundamental subspaces:

| Subspace | Definition | Dimension | Lives in |
|----------|-----------|-----------|----------|
| Column space $\text{Col}(A)$ | $\{A\mathbf{x} : \mathbf{x} \in \mathbb{R}^n\}$ | $r$ | $\mathbb{R}^m$ |
| Null space $\text{Null}(A)$ | $\{\mathbf{x} : A\mathbf{x} = \mathbf{0}\}$ | $n - r$ | $\mathbb{R}^n$ |
| Row space $\text{Row}(A)$ | $\text{Col}(A^T)$ | $r$ | $\mathbb{R}^n$ |
| Left null space $\text{Null}(A^T)$ | $\{\mathbf{y} : A^T\mathbf{y} = \mathbf{0}\}$ | $m - r$ | $\mathbb{R}^m$ |

**Theorem 2.3.9 (Orthogonal Complements).** The four subspaces pair into orthogonal complements:

$$\mathbb{R}^n = \text{Row}(A) \oplus \text{Null}(A) \qquad \mathbb{R}^m = \text{Col}(A) \oplus \text{Null}(A^T)$$

```
THE FOUR FUNDAMENTAL SUBSPACES

           ℝⁿ (domain)                    ℝᵐ (codomain)
    ┌────────────────────┐          ┌────────────────────┐
    │                    │          │                    │
    │   Row(A)     ─────────A────→    Col(A)           │
    │   dim = r          │          │   dim = r          │
    │         ⊥          │          │         ⊥          │
    │   Null(A)          │          │   Null(Aᵀ)         │
    │   dim = n-r        │          │   dim = m-r        │
    │                    │          │                    │
    └────────────────────┘          └────────────────────┘

    A maps Row(A) → Col(A) bijectively (rank r to rank r)
    A maps Null(A) → {0}   (everything in the null space dies)
```

**Example 2.5.9:** Find the four fundamental subspaces of $A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}$.

RREF: $\begin{pmatrix} 1 & 2 \\ 0 & 0 \end{pmatrix}$, so $r = \text{rank}(A) = 1$ with $m = n = 2$.

| Subspace | Basis | Dimension |
|----------|-------|-----------|
| $\text{Col}(A) \subseteq \mathbb{R}^2$ | $\{(1, 2)^T\}$ | 1 |
| $\text{Null}(A) \subseteq \mathbb{R}^2$ | $\{(-2, 1)^T\}$ | 1 |
| $\text{Row}(A) \subseteq \mathbb{R}^2$ | $\{(1, 2)^T\}$ | 1 |
| $\text{Null}(A^T) \subseteq \mathbb{R}^2$ | $\{(-2, 1)^T\}$ | 1 |

Verify orthogonality: $(1, 2) \cdot (-2, 1) = -2 + 2 = 0$. The row space and null space are perpendicular lines in $\mathbb{R}^2$.

*ML connection:* The four fundamental subspaces explain why $A\mathbf{x} = \mathbf{b}$ has no solution when $\mathbf{b} \notin \text{Col}(A)$. The least-squares solution $\hat{\mathbf{x}}$ projects $\mathbf{b}$ onto $\text{Col}(A)$, making the residual $\mathbf{b} - A\hat{\mathbf{x}}$ lie in $\text{Null}(A^T)$ — orthogonal to the column space. This is precisely the geometric interpretation of linear regression.

---

## 8. Least Squares

When $A\mathbf{x} = \mathbf{b}$ has no exact solution (the overdetermined case), we seek the $\mathbf{x}$ that minimizes the squared error.

**Definition 2.3.9 (Least Squares Problem).** Given $A \in \mathbb{R}^{m \times n}$ and $\mathbf{b} \in \mathbb{R}^m$ with $m > n$, find:

$$\hat{\mathbf{x}} = \arg\min_{\mathbf{x} \in \mathbb{R}^n} \|A\mathbf{x} - \mathbf{b}\|_2^2$$

**Theorem 2.3.10 (Normal Equations).** The least-squares solution satisfies:

$$A^T A \hat{\mathbf{x}} = A^T \mathbf{b}$$

If $A$ has full column rank ($\text{rank}(A) = n$), then $A^T A$ is invertible and:

$$\hat{\mathbf{x}} = (A^T A)^{-1} A^T \mathbf{b}$$

*Proof.* Let $f(\mathbf{x}) = \|A\mathbf{x} - \mathbf{b}\|^2 = (A\mathbf{x} - \mathbf{b})^T(A\mathbf{x} - \mathbf{b}) = \mathbf{x}^T A^T A \mathbf{x} - 2\mathbf{b}^T A\mathbf{x} + \mathbf{b}^T\mathbf{b}$. Setting $\nabla_\mathbf{x} f = 2A^T A\mathbf{x} - 2A^T\mathbf{b} = \mathbf{0}$ gives $A^T A\hat{\mathbf{x}} = A^T\mathbf{b}$.

If $\text{rank}(A) = n$, then $A^T A \in \mathbb{R}^{n \times n}$ is invertible (since $\text{rank}(A^T A) = \text{rank}(A) = n$), giving the closed-form solution. $\square$

*Geometric proof.* The residual $\mathbf{r} = \mathbf{b} - A\hat{\mathbf{x}}$ must be orthogonal to $\text{Col}(A)$: $A^T(\mathbf{b} - A\hat{\mathbf{x}}) = \mathbf{0}$, which rearranges to $A^T A\hat{\mathbf{x}} = A^T\mathbf{b}$. $\square$

**Example 2.5.10:** Find the least-squares solution to the overdetermined system $x = 1$, $x = 2$, $x = 4$ (i.e., $A = \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix}$, $\mathbf{b} = \begin{pmatrix} 1 \\ 2 \\ 4 \end{pmatrix}$).

$$A^T A = (1\;1\;1)\begin{pmatrix}1\\1\\1\end{pmatrix} = 3, \qquad A^T \mathbf{b} = (1\;1\;1)\begin{pmatrix}1\\2\\4\end{pmatrix} = 7$$

$$\hat{x} = (A^T A)^{-1} A^T \mathbf{b} = \frac{7}{3}$$

This is simply the mean of $\{1, 2, 4\}$ -- least squares with a constant model is just averaging.

```
LEAST SQUARES GEOMETRY

    b ∈ ℝᵐ
    │
    │  r = b - Ax̂  (residual, ⊥ to Col(A))
    │ ╱
    │╱
    ●─────────── Col(A) (column space of A)
   Ax̂ = proj of b
        onto Col(A)

    Ax̂ is the closest point in Col(A) to b.
    The residual r ⊥ Col(A), i.e., Aᵀr = 0.
```

*ML connection:* **Linear regression** is exactly the least-squares problem. Given a design matrix $X \in \mathbb{R}^{m \times n}$ (data) and response $\mathbf{y} \in \mathbb{R}^m$ (labels), the optimal weights are $\hat{\boldsymbol{\theta}} = (X^T X)^{-1} X^T \mathbf{y}$. Ridge regression adds regularization: $\hat{\boldsymbol{\theta}} = (X^T X + \lambda I)^{-1} X^T \mathbf{y}$. The term $\lambda I$ ensures invertibility even when $X^T X$ is singular (multicollinear features) and shrinks the solution toward zero, reducing overfitting.

---

## 9. Computational Considerations

### 9.1 LU Decomposition

Rather than performing Gaussian elimination from scratch for each new $\mathbf{b}$, we factor $A = LU$ once:

$$A = LU \qquad \text{then solve} \qquad L\mathbf{y} = \mathbf{b}, \quad U\mathbf{x} = \mathbf{y}$$

where $L$ is lower triangular and $U$ is upper triangular. Each triangular solve is $O(n^2)$.

**Example 2.5.11 (LU Decomposition).** Factor $A = \begin{pmatrix} 2 & 1 \\ 6 & 4 \end{pmatrix}$.

Eliminate below the pivot: $R_2 \leftarrow R_2 - 3R_1$ gives $U = \begin{pmatrix} 2 & 1 \\ 0 & 1 \end{pmatrix}$. The multiplier was $3$, so:

$$L = \begin{pmatrix} 1 & 0 \\ 3 & 1 \end{pmatrix}, \quad U = \begin{pmatrix} 2 & 1 \\ 0 & 1 \end{pmatrix} \qquad \text{Verify: } LU = \begin{pmatrix} 2 & 1 \\ 6 & 4 \end{pmatrix} = A \;\checkmark$$

To solve $A\mathbf{x} = \begin{pmatrix}5 \\ 17\end{pmatrix}$: first $L\mathbf{y} = \begin{pmatrix}5\\17\end{pmatrix}$ gives $y_1 = 5$, $y_2 = 17 - 3(5) = 2$. Then $U\mathbf{x} = \begin{pmatrix}5\\2\end{pmatrix}$ gives $x_2 = 2$, $x_1 = (5-2)/2 = 3/2$.

### 9.2 Numerical Stability: Pivoting

**Definition 2.3.10 (Partial Pivoting).** At each step of Gaussian elimination, swap the current row with the row below it that has the largest absolute value in the pivot column. This prevents dividing by small numbers that amplify rounding errors.

### 9.3 Complexity Summary

| Method | Cost | When to Use |
|--------|------|-------------|
| Gaussian elimination | $O(n^3)$ | Single system, moderate $n$ |
| LU decomposition | $O(n^3)$ factorization + $O(n^2)$ per solve | Multiple systems with same $A$ |
| Cholesky ($A = LL^T$) | $O(n^3/3)$ | Symmetric positive definite $A$ |
| QR decomposition | $O(2mn^2)$ | Least squares (more stable than normal equations) |
| Conjugate gradient | $O(n \sqrt{\kappa})$ iterations | Large sparse systems |
| Direct inverse $A^{-1}\mathbf{b}$ | $O(n^3)$ | Almost never — use factorization instead |

*ML connection:* Gaussian processes require solving $K\boldsymbol{\alpha} = \mathbf{y}$ where $K$ is the $m \times m$ kernel matrix. With $m$ training points, this costs $O(m^3)$ via Cholesky, which is why standard GPs do not scale beyond ~10,000 data points. Sparse GP approximations reduce this to $O(m k^2)$ where $k \ll m$ is the number of inducing points.

---

## 10. Worked Example: Linear Regression

**Problem.** Fit $y = \theta_0 + \theta_1 x$ to data: $(1, 2), (2, 3), (3, 5), (4, 4)$.

**Step 1.** Form the design matrix and response vector:

$$X = \begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \\ 1 & 4 \end{pmatrix}, \qquad \mathbf{y} = \begin{pmatrix} 2 \\ 3 \\ 5 \\ 4 \end{pmatrix}$$

**Step 2.** Compute $X^T X$ and $X^T \mathbf{y}$:

$$X^T X = \begin{pmatrix} 4 & 10 \\ 10 & 30 \end{pmatrix}, \qquad X^T \mathbf{y} = \begin{pmatrix} 14 \\ 41 \end{pmatrix}$$

**Step 3.** Solve the normal equations $X^T X \boldsymbol{\theta} = X^T \mathbf{y}$:

$$\begin{pmatrix} 4 & 10 \\ 10 & 30 \end{pmatrix} \begin{pmatrix} \theta_0 \\ \theta_1 \end{pmatrix} = \begin{pmatrix} 14 \\ 41 \end{pmatrix}$$

Using Gaussian elimination on the augmented matrix:

$$\left(\begin{array}{cc|c} 4 & 10 & 14 \\ 10 & 30 & 41 \end{array}\right) \xrightarrow{R_2 \leftarrow R_2 - 2.5R_1} \left(\begin{array}{cc|c} 4 & 10 & 14 \\ 0 & 5 & 6 \end{array}\right)$$

Back substitution: $\theta_1 = 6/5 = 1.2$ is not exact; let us redo: $5\theta_1 = 6$, so $\theta_1 = 6/5$. Then $4\theta_0 + 10(6/5) = 14$, so $4\theta_0 = 14 - 12 = 2$, giving $\theta_0 = 1/2$.

**Result:** $\hat{y} = 0.5 + 1.2x$

```
LEAST SQUARES FIT

  y
  5 │         ·
    │       ╱
  4 │     ╱   ·
    │   ╱
  3 │ ╱ ·
    │╱
  2 ·
    │
  1 │
    └───┬───┬───┬───┬── x
        1   2   3   4

  Line: ŷ = 0.5 + 1.2x
  Residuals: -0.2, -0.9, 0.9, -0.3
  Sum of squared residuals: 1.7
```

---

## Exercises

**★ Basic**

1. Solve using Gaussian elimination:
   $x + y + z = 6$, $2x + 3y + z = 14$, $x + 2y + 3z = 14$.

2. Reduce the following to RREF and identify pivot and free variables:
   $$\begin{pmatrix} 1 & 2 & 1 & 3 \\ 2 & 4 & 3 & 8 \\ 3 & 6 & 4 & 11 \end{pmatrix}$$

3. Determine whether $\mathbf{b} = (1, 2, 3)^T$ is in the column space of $A = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}$.

**★★ Intermediate**

4. Let $A \in \mathbb{R}^{3 \times 5}$ with $\text{rank}(A) = 3$. How many free variables does $A\mathbf{x} = \mathbf{0}$ have? Describe the null space geometrically.

5. Prove that if $A\mathbf{x} = \mathbf{b}$ has a solution and $A\mathbf{x} = \mathbf{c}$ has a solution, then $A\mathbf{x} = s\mathbf{b} + t\mathbf{c}$ has a solution for any $s, t \in \mathbb{R}$. Is this true for $A\mathbf{x} = \mathbf{0}$ replaced by $A\mathbf{x} = \mathbf{b}$?

6. Set up and solve the normal equations for fitting $y = \theta_0 + \theta_1 x + \theta_2 x^2$ to data $(0, 1), (1, 2), (2, 5), (3, 10)$.

7. Show that the normal equations $A^T A \hat{\mathbf{x}} = A^T \mathbf{b}$ are always consistent, even when $A\mathbf{x} = \mathbf{b}$ is not.

**★★★ Challenging**

8. Prove that among all solutions to the underdetermined system $A\mathbf{x} = \mathbf{b}$ (with infinitely many solutions), the pseudoinverse solution $\hat{\mathbf{x}} = A^T(AA^T)^{-1}\mathbf{b}$ has the smallest $\ell^2$ norm. *Hint:* decompose any solution as $\hat{\mathbf{x}} + \mathbf{x}_h$ with $\mathbf{x}_h \in \ker(A)$ and use the Pythagorean theorem.

9. Let $A \in \mathbb{R}^{m \times n}$. Show that the ridge regression solution $\hat{\mathbf{x}}_\lambda = (A^T A + \lambda I)^{-1} A^T \mathbf{b}$ converges to the ordinary least squares solution as $\lambda \to 0^+$ when $A$ has full column rank, and converges to $\mathbf{0}$ as $\lambda \to \infty$.

10. (Condition number) The condition number $\kappa(A) = \|A\| \cdot \|A^{-1}\|$ measures sensitivity to perturbations. If $A\mathbf{x} = \mathbf{b}$ and $(A + \delta A)(\mathbf{x} + \delta\mathbf{x}) = \mathbf{b} + \delta\mathbf{b}$, prove that $\frac{\|\delta\mathbf{x}\|}{\|\mathbf{x}\|} \leq \kappa(A) \left(\frac{\|\delta A\|}{\|A\|} + \frac{\|\delta\mathbf{b}\|}{\|\mathbf{b}\|}\right) + O(\|\delta A\|^2)$. Explain why this matters for training neural networks with float16.

---

## Related Topics

- [Vectors](vectors.md) -- linear independence and span
- [Matrices](matrices.md) -- matrix operations, rank, and inverse
- [Vector Spaces](vector-spaces.md) -- null space, column space, and the four fundamental subspaces
- [Determinants](determinants.md) -- Cramer's rule for small systems
- [Matrix Decompositions](matrix-decompositions.md) -- LU, QR, and SVD for solving linear systems
- [Inner Product Spaces](inner-product-spaces.md) -- projection and least squares in general inner product spaces

# Matrices

A matrix is a rectangular array of numbers, but thinking of it as "just an array" misses the point. A matrix is a **linear transformation** — it takes vectors as input and produces vectors as output. Every neural network layer, every graph adjacency structure, every quantum gate, and every dataset can be represented as a matrix. The weight matrix $W$ in a dense layer, the attention matrix $QK^T / \sqrt{d_k}$ in a transformer, the adjacency matrix $A$ of a social network — these are all matrices, and understanding their properties is essential to understanding modern AI.

---

## Prerequisites

- [Vectors](vectors.md) — vector operations, inner products, norms

---

## 1. Definition and Basic Notation

**Definition 2.2.1 (Matrix).** An $m \times n$ *matrix* over a field $\mathbb{F}$ (typically $\mathbb{R}$ or $\mathbb{C}$) is a rectangular array of elements:

$$A = \begin{pmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{pmatrix} \in \mathbb{F}^{m \times n}$$

We write $A = (a_{ij})$ where $a_{ij}$ denotes the entry in row $i$, column $j$. The set of all $m \times n$ real matrices is denoted $\mathbb{R}^{m \times n}$.

**Convention:** We use bold uppercase ($\mathbf{A}$, $\mathbf{B}$) or plain uppercase ($A$, $B$) for matrices, bold lowercase ($\mathbf{v}$) for vectors, and lowercase italic ($a_{ij}$) for scalar entries.

```
MATRIX AS A COLLECTION OF VECTORS

        Column vectors                    Row vectors

    ┌─ c₁  c₂  c₃ ─┐              ┌─── r₁ ───┐
    │  ↓   ↓   ↓   │              │─── r₂ ───│
A = │  │   │   │   │    or    A = │─── r₃ ───│
    │  │   │   │   │              │─── r₄ ───│
    └───────────────┘              └───────────┘

  A ∈ ℝ⁴ˣ³ has 3 column            same A has 4 row
  vectors in ℝ⁴                    vectors in ℝ³
```

*ML connection:* A dataset with $m$ samples and $n$ features is stored as a matrix $X \in \mathbb{R}^{m \times n}$. Each row is a data point; each column is a feature. In a neural network, the weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ has each row representing the weights connecting all inputs to a single output neuron.

---

## 2. Matrix Operations

### 2.1 Addition and Scalar Multiplication

**Definition 2.2.2 (Matrix Addition).** For $A, B \in \mathbb{R}^{m \times n}$:

$$(A + B)_{ij} = a_{ij} + b_{ij}$$

Matrices must have the same dimensions for addition to be defined.

**Definition 2.2.3 (Scalar Multiplication).** For $c \in \mathbb{R}$, $A \in \mathbb{R}^{m \times n}$:

$$(cA)_{ij} = c \cdot a_{ij}$$

**Example 2.2.1 (Addition and Scalar Multiplication).**

$$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, \quad B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$$

$$A + B = \begin{pmatrix} 1+5 & 2+6 \\ 3+7 & 4+8 \end{pmatrix} = \begin{pmatrix} 6 & 8 \\ 10 & 12 \end{pmatrix}$$

$$3A = \begin{pmatrix} 3 & 6 \\ 9 & 12 \end{pmatrix}$$

*ML connection:* In gradient descent, the weight update $W \leftarrow W - \eta \nabla W$ is matrix addition (subtracting a scaled gradient matrix from the weight matrix).

**Theorem 2.2.1.** The set $\mathbb{R}^{m \times n}$ with matrix addition and scalar multiplication forms a vector space of dimension $mn$.

*Proof (sketch).* Verify the eight vector space axioms. The zero element is the $m \times n$ zero matrix, and the additive inverse of $A$ is $(-1)A$. The standard basis consists of $mn$ matrices $E_{ij}$ with a 1 in position $(i,j)$ and 0 elsewhere. $\square$

### 2.2 Matrix Multiplication

**Definition 2.2.4 (Matrix Multiplication).** For $A \in \mathbb{R}^{m \times p}$ and $B \in \mathbb{R}^{p \times n}$, the product $C = AB \in \mathbb{R}^{m \times n}$ is:

$$c_{ij} = \sum_{k=1}^p a_{ik} b_{kj} = (\text{row } i \text{ of } A) \cdot (\text{column } j \text{ of } B)$$

```
MATRIX MULTIPLICATION: (m×p) · (p×n) = (m×n)

A           B           C = AB
┌─────┐   ┌───┐       ┌───┐
│ · · ·│   │ · │       │   │
│ · · ·│ × │ · │   =   │ · │
│ · · ·│   │ · │       │   │
│ · · ·│   └───┘       └───┘
└─────┘
 (4×3)     (3×2)       (4×2)

Inner dimensions must match: ─┘ └─
Outer dimensions give result size: 4×2

Entry c₂₁ = (row 2 of A) · (col 1 of B)
           = a₂₁b₁₁ + a₂₂b₂₁ + a₂₃b₃₁
```

**Example 2.2.2 (Matrix Multiplication — step by step).**

$$A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}, \quad B = \begin{pmatrix} 7 & 8 \\ 9 & 10 \\ 11 & 12 \end{pmatrix}$$

$A$ is $2 \times 3$, $B$ is $3 \times 2$, so $AB$ is $2 \times 2$:

$$c_{11} = (1)(7) + (2)(9) + (3)(11) = 7 + 18 + 33 = 58$$

$$c_{12} = (1)(8) + (2)(10) + (3)(12) = 8 + 20 + 36 = 64$$

$$c_{21} = (4)(7) + (5)(9) + (6)(11) = 28 + 45 + 66 = 139$$

$$c_{22} = (4)(8) + (5)(10) + (6)(12) = 32 + 50 + 72 = 154$$

$$AB = \begin{pmatrix} 58 & 64 \\ 139 & 154 \end{pmatrix}$$

Note: $BA$ would be $3 \times 3$ (different shape!) — matrix multiplication is not commutative.

**Theorem 2.2.2 (Properties of Matrix Multiplication).**

1. **Associativity:** $(AB)C = A(BC)$
2. **Distributivity:** $A(B + C) = AB + AC$ and $(A + B)C = AC + BC$
3. **Scalar compatibility:** $c(AB) = (cA)B = A(cB)$
4. **NOT commutative:** $AB \neq BA$ in general

*Proof of non-commutativity (by counterexample).* Let $A = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$, $B = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$. Then $AB = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$ but $BA = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$. $\square$

*ML connection:* The forward pass of a multi-layer perceptron computes $\mathbf{y} = \sigma_L(W_L \cdots \sigma_2(W_2 \sigma_1(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2) \cdots + \mathbf{b}_L)$. Without activations, this collapses to a single matrix $W = W_L \cdots W_2 W_1$ by associativity — which is why nonlinear activations are essential.

### 2.3 Matrix-Vector Product

**Definition 2.2.5 (Matrix-Vector Product).** For $A \in \mathbb{R}^{m \times n}$, $\mathbf{x} \in \mathbb{R}^n$:

$$A\mathbf{x} = \sum_{j=1}^n x_j \mathbf{a}_j$$

where $\mathbf{a}_j$ is the $j$-th column of $A$. This reveals a key insight: $A\mathbf{x}$ is a **linear combination of the columns of $A$**, with the entries of $\mathbf{x}$ as coefficients.

```
MATRIX-VECTOR PRODUCT AS LINEAR COMBINATION

              ┌ 1  3 ┐   ┌ 2 ┐       ┌ 1 ┐       ┌ 3 ┐     ┌ 11 ┐
A·x =         │ 2  0 │ · │   │ = 2 · │ 2 │ + 3 · │ 0 │  =  │  4 │
              └ 4  1 ┘   └ 3 ┘       └ 4 ┘       └ 1 ┘     └ 11 ┘

         (3×2 matrix)  (2×1)    x₁ · col₁   x₂ · col₂    (3×1)
```

*ML connection:* In graph neural networks, the aggregation step computes $A\mathbf{h}$ where $A$ is the (normalized) adjacency matrix and $\mathbf{h}$ is the node feature vector. Each node's new representation is a linear combination of its neighbors' features — the matrix encodes the graph structure.

### 2.4 Outer Product

**Definition 2.2.5b (Outer Product).** For $\mathbf{u} \in \mathbb{R}^m$, $\mathbf{v} \in \mathbb{R}^n$, the *outer product* is the $m \times n$ matrix:

$$\mathbf{u}\mathbf{v}^T = \begin{pmatrix} u_1 v_1 & u_1 v_2 & \cdots & u_1 v_n \\ u_2 v_1 & u_2 v_2 & \cdots & u_2 v_n \\ \vdots & \vdots & \ddots & \vdots \\ u_m v_1 & u_m v_2 & \cdots & u_m v_n \end{pmatrix}$$

Note that $\text{rank}(\mathbf{u}\mathbf{v}^T) = 1$ (assuming $\mathbf{u}, \mathbf{v} \neq \mathbf{0}$). Every rank-1 matrix is an outer product.

**Example 2.2.3 (Outer Product).**

$$\mathbf{u} = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}, \quad \mathbf{v} = \begin{pmatrix} 4 \\ 5 \end{pmatrix} \implies \mathbf{u}\mathbf{v}^T = \begin{pmatrix} 1 \cdot 4 & 1 \cdot 5 \\ 2 \cdot 4 & 2 \cdot 5 \\ 3 \cdot 4 & 3 \cdot 5 \end{pmatrix} = \begin{pmatrix} 4 & 5 \\ 8 & 10 \\ 12 & 15 \end{pmatrix}$$

Notice every row is a scaled version of $\mathbf{v}^T$ — this is why the result has rank 1.

*ML connection:* Matrix multiplication can be viewed as a sum of outer products: $AB = \sum_{k=1}^p \mathbf{a}_k \mathbf{b}_k^T$ where $\mathbf{a}_k$ is column $k$ of $A$ and $\mathbf{b}_k^T$ is row $k$ of $B$. The SVD decomposes any matrix as a sum of rank-1 outer products: $A = \sum_i \sigma_i \mathbf{u}_i \mathbf{v}_i^T$. In Hebbian learning, the weight update $\Delta W = \eta \mathbf{y}\mathbf{x}^T$ is an outer product — "neurons that fire together, wire together."

### 2.5 Block Matrices

**Definition 2.2.5c (Block Matrix).** A *block matrix* (or partitioned matrix) is a matrix viewed as composed of submatrices:

$$M = \begin{pmatrix} A & B \\ C & D \end{pmatrix}$$

where $A \in \mathbb{R}^{p \times q}$, $B \in \mathbb{R}^{p \times s}$, $C \in \mathbb{R}^{r \times q}$, $D \in \mathbb{R}^{r \times s}$.

Block multiplication follows the same rules as scalar multiplication: if the blocks have compatible dimensions, $\begin{pmatrix} A & B \\ C & D \end{pmatrix}\begin{pmatrix} E \\ F \end{pmatrix} = \begin{pmatrix} AE + BF \\ CE + DF \end{pmatrix}$.

*ML connection:* Multi-head attention in transformers uses block structure. Each head operates on a different block of the embedding dimension: $\text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)$. The concatenation of heads and projection can be seen as a block matrix multiplication. Mixture-of-experts (MoE) layers in models like GPT-4 route tokens to different expert blocks — the effective weight matrix is a sparse block matrix.

---

## 3. Transpose

**Definition 2.2.6 (Transpose).** The *transpose* of $A \in \mathbb{R}^{m \times n}$ is $A^T \in \mathbb{R}^{n \times m}$ defined by $(A^T)_{ij} = a_{ji}$.

**Example 2.2.4 (Transpose).** Rows become columns, columns become rows:

$$A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix} \implies A^T = \begin{pmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{pmatrix}$$

Note: $A$ is $2 \times 3$, but $A^T$ is $3 \times 2$.

**Theorem 2.2.3 (Properties of Transpose).**

1. $(A^T)^T = A$
2. $(A + B)^T = A^T + B^T$
3. $(cA)^T = cA^T$
4. $(AB)^T = B^T A^T$ (reversal of order)

*Proof of (4).* $((AB)^T)_{ij} = (AB)_{ji} = \sum_k a_{jk} b_{ki} = \sum_k (B^T)_{ik} (A^T)_{kj} = (B^T A^T)_{ij}$. $\square$

*ML connection:* In backpropagation, if the forward pass computes $\mathbf{z} = W\mathbf{x}$, the gradient with respect to $\mathbf{x}$ is $\frac{\partial L}{\partial \mathbf{x}} = W^T \frac{\partial L}{\partial \mathbf{z}}$. The transpose naturally appears when propagating gradients backward through layers.

---

## 4. Special Matrix Types

### 4.1 Square Matrices and the Identity

**Definition 2.2.7 (Identity Matrix).** The $n \times n$ *identity matrix* $I_n$ has:

$$(I_n)_{ij} = \delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

For any $A \in \mathbb{R}^{m \times n}$: $I_m A = A I_n = A$.

### 4.2 Diagonal Matrices

**Definition 2.2.8 (Diagonal Matrix).** $D = \text{diag}(d_1, d_2, \ldots, d_n)$ has $d_{ij} = 0$ for $i \neq j$.

Diagonal matrices scale each coordinate independently: $D\mathbf{x} = (d_1 x_1, d_2 x_2, \ldots, d_n x_n)^T$.

*ML connection:* In the spectral theorem $A = Q\Lambda Q^T$, the diagonal matrix $\Lambda$ contains eigenvalues. PCA retains the top-$k$ eigenvalues and zeros out the rest — this is dimensionality reduction via a modified diagonal matrix.

### 4.3 Symmetric Matrices

**Definition 2.2.9 (Symmetric Matrix).** $A$ is *symmetric* if $A^T = A$, i.e., $a_{ij} = a_{ji}$ for all $i, j$.

**Theorem 2.2.4 (Spectral Theorem for Real Symmetric Matrices).** If $A \in \mathbb{R}^{n \times n}$ is symmetric, then:

1. All eigenvalues of $A$ are real.
2. Eigenvectors corresponding to distinct eigenvalues are orthogonal.
3. $A$ is orthogonally diagonalizable: $A = Q\Lambda Q^T$ where $Q$ is orthogonal and $\Lambda$ is diagonal.

*ML connection:* The covariance matrix $\Sigma = \frac{1}{n-1}(X - \bar{X})^T(X - \bar{X})$ is always symmetric and positive semi-definite. PCA exploits the spectral theorem to decompose $\Sigma$ into orthogonal principal components — the eigenvectors of $\Sigma$ are the directions of maximum variance.

### 4.4 Orthogonal Matrices

**Definition 2.2.10 (Orthogonal Matrix).** $Q \in \mathbb{R}^{n \times n}$ is *orthogonal* if $Q^T Q = QQ^T = I$, equivalently $Q^{-1} = Q^T$.

**Theorem 2.2.5.** If $Q$ is orthogonal, then:

1. $\|Q\mathbf{x}\| = \|\mathbf{x}\|$ for all $\mathbf{x}$ (preserves lengths)
2. $\langle Q\mathbf{x}, Q\mathbf{y} \rangle = \langle \mathbf{x}, \mathbf{y} \rangle$ (preserves angles)
3. $|\det(Q)| = 1$

*Proof of (1).* $\|Q\mathbf{x}\|^2 = (Q\mathbf{x})^T(Q\mathbf{x}) = \mathbf{x}^T Q^T Q \mathbf{x} = \mathbf{x}^T I \mathbf{x} = \|\mathbf{x}\|^2$. $\square$

*ML connection:* Orthogonal weight initialization (used in RNNs) preserves gradient magnitudes during backpropagation, mitigating the vanishing/exploding gradient problem. In quantum computing, all quantum gates are unitary matrices (the complex analogue of orthogonal), preserving the norm of the quantum state vector.

### 4.5 Triangular Matrices

**Definition 2.2.11 (Triangular Matrix).** $A$ is *upper triangular* if $a_{ij} = 0$ for $i > j$, and *lower triangular* if $a_{ij} = 0$ for $i < j$.

$$U = \begin{pmatrix} u_{11} & u_{12} & u_{13} \\ 0 & u_{22} & u_{23} \\ 0 & 0 & u_{33} \end{pmatrix} \qquad L = \begin{pmatrix} l_{11} & 0 & 0 \\ l_{21} & l_{22} & 0 \\ l_{31} & l_{32} & l_{33} \end{pmatrix}$$

**Theorem 2.2.6.** The determinant of a triangular matrix is the product of its diagonal entries: $\det(T) = \prod_{i=1}^n t_{ii}$.

*ML connection:* In the transformer's masked self-attention, the attention weight matrix is masked to be lower triangular — position $i$ can only attend to positions $\leq i$. This enforces the autoregressive property in language models like GPT.

### 4.6 Positive Definite Matrices

**Definition 2.2.12 (Positive Definite).** A symmetric matrix $A \in \mathbb{R}^{n \times n}$ is:

- *Positive definite* (PD) if $\mathbf{x}^T A \mathbf{x} > 0$ for all $\mathbf{x} \neq \mathbf{0}$
- *Positive semi-definite* (PSD) if $\mathbf{x}^T A \mathbf{x} \geq 0$ for all $\mathbf{x}$

**Theorem 2.2.7 (Characterizations of Positive Definiteness).** For a symmetric matrix $A$, the following are equivalent:

1. $A$ is positive definite
2. All eigenvalues of $A$ are positive
3. All leading principal minors of $A$ are positive (Sylvester's criterion)
4. There exists an invertible matrix $B$ such that $A = B^T B$

**Example 2.2.8 (Testing Positive Definiteness).**

$$A = \begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix}$$

**Method 1 (eigenvalues):** $\det(A - \lambda I) = (2-\lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = (\lambda - 1)(\lambda - 3) = 0$. Eigenvalues: $\lambda_1 = 1 > 0$, $\lambda_2 = 3 > 0$ → PD ✓

**Method 2 (leading minors):** $a_{11} = 2 > 0$, $\det(A) = 4 - 1 = 3 > 0$ → PD ✓

**Method 3 (direct test):** $\mathbf{x}^T A \mathbf{x} = 2x_1^2 - 2x_1 x_2 + 2x_2^2 = x_1^2 + (x_1 - x_2)^2 + x_2^2 > 0$ for $\mathbf{x} \neq \mathbf{0}$ ✓

*ML connection:* The Hessian matrix $H = \nabla^2 L(\theta)$ of the loss function is positive definite at a local minimum. In Gaussian processes, the kernel matrix $K$ (with $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$) must be positive semi-definite to define a valid covariance. The Cholesky decomposition $K = LL^T$ (which requires PD) is used for efficient sampling.

---

## 5. Trace

**Definition 2.2.13 (Trace).** The *trace* of a square matrix $A \in \mathbb{R}^{n \times n}$ is the sum of diagonal entries:

$$\text{tr}(A) = \sum_{i=1}^n a_{ii}$$

**Example 2.2.5 (Trace).**

$$A = \begin{pmatrix} 3 & 1 & 0 \\ 1 & 2 & 1 \\ 0 & 1 & 5 \end{pmatrix} \implies \text{tr}(A) = 3 + 2 + 5 = 10$$

Verify cyclic property: for $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$, $B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$:

$AB = \begin{pmatrix} 19 & 22 \\ 43 & 50 \end{pmatrix}$, $BA = \begin{pmatrix} 23 & 34 \\ 31 & 46 \end{pmatrix}$. Both have $\text{tr} = 19 + 50 = 23 + 46 = 69$. ✓

**Theorem 2.2.8 (Properties of Trace).**

1. $\text{tr}(A + B) = \text{tr}(A) + \text{tr}(B)$ (linearity)
2. $\text{tr}(cA) = c \cdot \text{tr}(A)$
3. $\text{tr}(AB) = \text{tr}(BA)$ (cyclic property)
4. $\text{tr}(A^T) = \text{tr}(A)$
5. $\text{tr}(A) = \sum_{i=1}^n \lambda_i$ (sum of eigenvalues)

*Proof of (3).* $\text{tr}(AB) = \sum_i (AB)_{ii} = \sum_i \sum_k a_{ik} b_{ki} = \sum_k \sum_i b_{ki} a_{ik} = \sum_k (BA)_{kk} = \text{tr}(BA)$. $\square$

**Corollary (Cyclic permutation).** $\text{tr}(ABC) = \text{tr}(CAB) = \text{tr}(BCA)$ (but $\neq \text{tr}(ACB)$ in general).

*ML connection:* The Frobenius norm of a matrix is $\|A\|_F = \sqrt{\text{tr}(A^T A)} = \sqrt{\sum_{ij} a_{ij}^2}$. This appears in weight regularization: $\|W\|_F^2 = \text{tr}(W^T W)$. The trace also appears in the nuclear norm (trace norm) $\|A\|_* = \text{tr}(\sqrt{A^T A})$, which is used in matrix completion for recommender systems.

---

## 6. Rank

**Definition 2.2.14 (Rank).** The *rank* of $A \in \mathbb{R}^{m \times n}$ is:

$$\text{rank}(A) = \dim(\text{Col}(A)) = \dim(\text{Row}(A))$$

where $\text{Col}(A)$ is the column space (span of columns) and $\text{Row}(A)$ is the row space.

**Example 2.2.6 (Rank).**

$$A = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 0 & 1 & 1 \end{pmatrix}$$

Row 2 = 2 × Row 1, so only 2 rows are independent → $\text{rank}(A) = 2$ (rank-deficient, since $\min(3,3) = 3$).

$$B = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}$$

Both columns are independent → $\text{rank}(B) = 2$ (full column rank).

**Theorem 2.2.9 (Rank Equivalences).** For $A \in \mathbb{R}^{m \times n}$, the following are equal:

1. Dimension of the column space of $A$
2. Dimension of the row space of $A$
3. Number of pivot positions in the row echelon form of $A$
4. Number of nonzero singular values of $A$
5. Size of the largest nonzero minor of $A$

**Theorem 2.2.10 (Rank Inequalities).**

1. $\text{rank}(A) \leq \min(m, n)$
2. $\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$
3. $\text{rank}(A + B) \leq \text{rank}(A) + \text{rank}(B)$
4. $\text{rank}(A^T A) = \text{rank}(A)$

**Definition 2.2.15 (Full Rank).** $A$ has *full rank* if $\text{rank}(A) = \min(m, n)$.

- Full column rank ($\text{rank}(A) = n$): columns are linearly independent, $A\mathbf{x} = \mathbf{b}$ has at most one solution
- Full row rank ($\text{rank}(A) = m$): $A\mathbf{x} = \mathbf{b}$ has at least one solution for every $\mathbf{b}$

```
RANK AND MATRIX DIMENSIONS

Full rank (rank = min(m,n)):         Rank-deficient (rank < min(m,n)):

    ┌───────┐                            ┌───────┐
    │ × × × │ rank 3                     │ × × × │ rank 2
    │ × × × │ (= n)                      │ × × × │ (< min(4,3) = 3)
    │ × × × │                            │ × × × │ ← row 3 is a linear
    │ × × × │                            │ × × × │   combination of rows 1,2
    └───────┘                            └───────┘
     4 × 3                                4 × 3
```

*ML connection:* Low-rank approximation is the principle behind many ML techniques. In recommender systems, the rating matrix $R \approx UV^T$ where $U \in \mathbb{R}^{m \times k}$, $V \in \mathbb{R}^{n \times k}$, and $k \ll \min(m, n)$. LoRA (Low-Rank Adaptation) fine-tunes large language models by adding low-rank updates $\Delta W = BA$ where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$ with rank $r \ll d$, reducing trainable parameters from millions to thousands.

---

## 7. Matrix Inverse

**Definition 2.2.16 (Inverse).** A square matrix $A \in \mathbb{R}^{n \times n}$ is *invertible* (nonsingular) if there exists $A^{-1} \in \mathbb{R}^{n \times n}$ such that:

$$AA^{-1} = A^{-1}A = I$$

**Example 2.2.7 (2×2 Inverse — formula).**

For a $2 \times 2$ matrix, the inverse has a closed form:

$$A = \begin{pmatrix} a & b \\ c & d \end{pmatrix} \implies A^{-1} = \frac{1}{ad - bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$$

Concrete example:

$$A = \begin{pmatrix} 4 & 7 \\ 2 & 6 \end{pmatrix}, \quad \det(A) = 4(6) - 7(2) = 10$$

$$A^{-1} = \frac{1}{10}\begin{pmatrix} 6 & -7 \\ -2 & 4 \end{pmatrix} = \begin{pmatrix} 0.6 & -0.7 \\ -0.2 & 0.4 \end{pmatrix}$$

Verify: $AA^{-1} = \begin{pmatrix} 4(0.6)+7(-0.2) & 4(-0.7)+7(0.4) \\ 2(0.6)+6(-0.2) & 2(-0.7)+6(0.4) \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ ✓

**Theorem 2.2.11 (Invertibility Equivalences).** For $A \in \mathbb{R}^{n \times n}$, the following are equivalent:

1. $A$ is invertible
2. $\det(A) \neq 0$
3. $\text{rank}(A) = n$
4. The columns of $A$ are linearly independent
5. The null space of $A$ is $\{\mathbf{0}\}$
6. $A\mathbf{x} = \mathbf{b}$ has a unique solution for every $\mathbf{b}$
7. $0$ is not an eigenvalue of $A$

**Theorem 2.2.12 (Properties of the Inverse).**

1. $(A^{-1})^{-1} = A$
2. $(AB)^{-1} = B^{-1}A^{-1}$ (reversal of order)
3. $(A^T)^{-1} = (A^{-1})^T$
4. $\det(A^{-1}) = 1 / \det(A)$

*Proof of (2).* $(AB)(B^{-1}A^{-1}) = A(BB^{-1})A^{-1} = AIA^{-1} = AA^{-1} = I$. $\square$

**Definition 2.2.17 (Moore-Penrose Pseudoinverse).** For any $A \in \mathbb{R}^{m \times n}$ (not necessarily square or invertible), the *pseudoinverse* $A^+ \in \mathbb{R}^{n \times m}$ is the unique matrix satisfying:

1. $AA^+A = A$
2. $A^+AA^+ = A^+$
3. $(AA^+)^T = AA^+$
4. $(A^+A)^T = A^+A$

For full column rank $A$: $A^+ = (A^T A)^{-1} A^T$.

*ML connection:* The ordinary least squares solution to linear regression is $\hat{\boldsymbol{\theta}} = (X^T X)^{-1} X^T \mathbf{y} = X^+ \mathbf{y}$. When $X^T X$ is singular or ill-conditioned (multicollinear features), we use regularization: $\hat{\boldsymbol{\theta}} = (X^T X + \lambda I)^{-1} X^T \mathbf{y}$ (Ridge regression), which guarantees invertibility.

---

## 8. The Attention Matrix

One of the most important matrices in modern AI is the attention matrix in transformers.

Given queries $Q \in \mathbb{R}^{n \times d_k}$, keys $K \in \mathbb{R}^{n \times d_k}$, and values $V \in \mathbb{R}^{n \times d_v}$:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

```
THE ATTENTION MECHANISM (matrix view)

    Q · Kᵀ              softmax              · V
  ┌──────────┐       ┌──────────┐       ┌─────────┐
  │          │       │ .1 .7 .2 │       │         │
  │  n × n   │  ──→  │ .3 .3 .4 │  ──→  │  n × dᵥ │
  │ (scores) │       │ .8 .1 .1 │       │ (output) │
  └──────────┘       └──────────┘       └─────────┘
                     rows sum to 1
                     each row = attention
                     weights for one query
```

The matrix $S = QK^T / \sqrt{d_k} \in \mathbb{R}^{n \times n}$ has entry $s_{ij}$ measuring the compatibility between query $i$ and key $j$. After softmax, each row becomes a probability distribution over positions. Multiplying by $V$ computes a weighted average of value vectors.

---

## 9. Summary of Matrix Types

| Type | Condition | Key Property | ML Application |
|------|-----------|--------------|----------------|
| Diagonal | $a_{ij} = 0$ for $i \neq j$ | Scales axes independently | Eigenvalue matrix $\Lambda$ in PCA |
| Symmetric | $A^T = A$ | Real eigenvalues, orthogonal eigenvectors | Covariance matrices, kernel matrices |
| Orthogonal | $Q^T Q = I$ | Preserves lengths and angles | Orthogonal initialization, QR decomposition |
| Upper triangular | $a_{ij} = 0$ for $i > j$ | $\det = \prod a_{ii}$ | Causal attention mask, LU decomposition |
| Positive definite | $\mathbf{x}^T A \mathbf{x} > 0$ | All eigenvalues $> 0$ | Covariance, kernel matrices, Hessian at minimum |
| Sparse | Most entries are 0 | Efficient storage $O(\text{nnz})$ | Adjacency matrices, attention (sparse transformers) |
| Low-rank | $\text{rank} \ll \min(m,n)$ | $A \approx UV^T$ | LoRA, matrix factorization, recommender systems |
| Stochastic | Rows sum to 1, entries $\geq 0$ | Represents transition probabilities | Markov chains, PageRank, attention weights |

---

## Exercises

**★ Basic**

1. Let $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$, $B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$. Compute $AB$ and $BA$. Verify that $AB \neq BA$.

2. Show that for any matrix $A$, both $A^T A$ and $AA^T$ are symmetric.

3. Compute the trace and determinant of $A = \begin{pmatrix} 3 & 1 & 0 \\ 1 & 2 & 1 \\ 0 & 1 & 3 \end{pmatrix}$.

4. Verify that $Q = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}$ is orthogonal and compute $Q^{-1}$.

**★★ Intermediate**

5. Prove that if $A$ is symmetric and $B$ is any matrix of compatible dimensions, then $B^T A B$ is symmetric.

6. Show that $\text{rank}(A^T A) = \text{rank}(A)$ by proving that $A\mathbf{x} = \mathbf{0}$ and $A^T A \mathbf{x} = \mathbf{0}$ have the same solution set.

7. Let $A$ be an $n \times n$ matrix with $A^2 = A$ (idempotent). Prove that $\text{rank}(A) = \text{tr}(A)$.

8. For the attention matrix, explain why dividing by $\sqrt{d_k}$ is necessary. *Hint:* consider the variance of entries in $QK^T$ when $Q$ and $K$ have entries drawn from $\mathcal{N}(0, 1)$.

**★★★ Challenging**

9. Prove that the set of $n \times n$ orthogonal matrices forms a group under matrix multiplication (verify closure, associativity, identity, inverse). This group is denoted $O(n)$.

10. Let $A \in \mathbb{R}^{m \times n}$ with $m > n$ and $\text{rank}(A) = n$. Prove that the pseudoinverse $A^+ = (A^T A)^{-1}A^T$ satisfies the four Moore-Penrose conditions. Show that $A^+ \mathbf{b}$ minimizes $\|A\mathbf{x} - \mathbf{b}\|_2$.

11. (LoRA) A pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times d}$ is updated to $W_0 + BA$ where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$ with $r \ll d$. How many parameters does this save compared to fine-tuning $W_0$ directly? If $d = 4096$ and $r = 8$, compute the ratio of trainable parameters.

---

## Related Topics

- [Vectors](vectors.md) -- vector operations, norms, and inner products
- [Systems of Equations](systems-of-equations.md) -- solving $A\mathbf{x} = \mathbf{b}$ via Gaussian elimination
- [Vector Spaces](vector-spaces.md) -- column space, null space, and the four fundamental subspaces
- [Linear Transformations](linear-transformations.md) -- matrices as representations of linear maps
- [Determinants](determinants.md) -- the scalar that encodes volume scaling
- [Eigenvalues](eigenvalues.md) -- the spectral decomposition of matrices
- [Matrix Decompositions](matrix-decompositions.md) -- SVD, LU, QR, and Cholesky

# Tensors

When a neural network processes an image, the input is not a vector or a matrix --- it is a three-dimensional array (height $\times$ width $\times$ channels). When a transformer computes attention, it operates on a four-dimensional structure (batch $\times$ heads $\times$ sequence $\times$ features). When a quantum computer represents $n$ entangled qubits, the state lives in a $2^n$-dimensional space naturally described by an order-$n$ tensor. This chapter develops tensors rigorously as multilinear maps, connects them to the multidimensional arrays of deep learning frameworks, and introduces decompositions that generalise SVD to higher dimensions.

---

## Prerequisites

- [Vector Spaces](vector-spaces.md) --- basis, dimension, dual spaces
- [Linear Transformations](linear-transformations.md) --- linear maps, matrix representation
- [Inner Product Spaces](inner-product-spaces.md) --- inner products, orthogonality
- [Matrix Decompositions](matrix-decompositions.md) --- SVD, low-rank approximation

---

## 1. Tensors as Multilinear Maps

### 1.1 Motivation: Beyond Matrices

Scalars, vectors, and matrices form a hierarchy:

```
ORDER OF TENSORS

    Order 0:  scalar         a ∈ R                    1 number
    Order 1:  vector         vᵢ ∈ Rⁿ                  n numbers
    Order 2:  matrix         Aᵢⱼ ∈ Rᵐˣⁿ               m×n numbers
    Order 3:  3-tensor       Tᵢⱼₖ ∈ Rᵐˣⁿˣᵖ            m×n×p numbers
    Order N:  N-tensor       Tᵢ₁ᵢ₂...ᵢₙ               ∏ᵢ dᵢ numbers

    Order 0      Order 1       Order 2        Order 3
      ●          ──────        ┌────┐        ┌────┐╱│
                               │    │        │    │ │
                               │    │        │    │╱
                               └────┘        └────┘
    scalar       vector        matrix         cube
```

**Example 2.10.3 (A rank-3 tensor with numbers).** A tensor $\mathcal{T} \in \mathbb{R}^{2 \times 3 \times 2}$ is a "box" of 12 numbers. Writing the two $2 \times 3$ slices along the third axis:

$$\mathcal{T}(:,:,1) = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}, \quad \mathcal{T}(:,:,2) = \begin{pmatrix} 7 & 8 & 9 \\ 10 & 11 & 12 \end{pmatrix}$$

So $T_{1,2,1} = 2$, $T_{2,3,2} = 12$, $T_{1,1,2} = 7$, etc. This tensor has order 3, shape $(2,3,2)$, and contains $2 \times 3 \times 2 = 12$ entries.

But the mathematical definition of a tensor is not simply "a multidimensional array." A tensor is a *multilinear map* --- an object whose transformation rules under change of basis are precisely defined.

### 1.2 The Dual Space

**Definition 2.10.1 (Dual Space).** Let $V$ be a finite-dimensional vector space over $\mathbb{F}$. The *dual space* $V^*$ is the set of all linear functionals $f: V \to \mathbb{F}$:

$$V^* = \mathcal{L}(V, \mathbb{F})$$

**Theorem 2.10.1.** $\dim V^* = \dim V$. If $\{\mathbf{e}_1, \dots, \mathbf{e}_n\}$ is a basis for $V$, the *dual basis* $\{\mathbf{e}^1, \dots, \mathbf{e}^n\}$ defined by $\mathbf{e}^i(\mathbf{e}_j) = \delta^i_j$ is a basis for $V^*$.

### 1.3 The Formal Definition

**Definition 2.10.2 (Tensor).** A *tensor of type $(r, s)$* on a vector space $V$ is a multilinear map:

$$T: \underbrace{V^* \times \cdots \times V^*}_{r \text{ copies}} \times \underbrace{V \times \cdots \times V}_{s \text{ copies}} \longrightarrow \mathbb{F}$$

The integer $r + s$ is the *order* (or rank in the physicist's sense, not to be confused with tensor rank defined later).

- $r$ is the *contravariant* order (number of $V^*$ arguments)
- $s$ is the *covariant* order (number of $V$ arguments)

**Example 2.10.1 (Familiar tensors as multilinear maps).**

| Object | Type | Multilinear map |
|--------|------|----------------|
| Scalar $a$ | $(0, 0)$ | Constant function $\to \mathbb{F}$ |
| Vector $\mathbf{v}$ | $(1, 0)$ | $f \mapsto f(\mathbf{v})$, acts on one covector |
| Covector $f$ | $(0, 1)$ | $\mathbf{v} \mapsto f(\mathbf{v})$, acts on one vector |
| Matrix / linear map | $(1, 1)$ | $(f, \mathbf{v}) \mapsto f(A\mathbf{v})$ |
| Inner product | $(0, 2)$ | $(\mathbf{u}, \mathbf{v}) \mapsto \langle \mathbf{u}, \mathbf{v} \rangle$ |
| 3D array | $(0, 3)$ or other | Trilinear map $V \times V \times V \to \mathbb{F}$ |

**Multilinearity** means linearity in each argument separately:

$$T(\dots, \alpha\mathbf{u} + \beta\mathbf{w}, \dots) = \alpha\, T(\dots, \mathbf{u}, \dots) + \beta\, T(\dots, \mathbf{w}, \dots)$$

*ML connection:* When PyTorch or TensorFlow stores a "tensor," it stores a multidimensional array of floating-point numbers. This array is the *component representation* of a tensor with respect to the standard basis. The framework handles basis changes (reshaping, permuting axes) behind the scenes. The mathematical definition ensures that operations like contraction and the Einstein convention are well-defined regardless of basis.

---

## 2. Tensor Product of Vector Spaces

**Definition 2.10.3 (Tensor Product of Vectors).** Given $\mathbf{u} \in V$ and $\mathbf{w} \in W$, their *tensor product* $\mathbf{u} \otimes \mathbf{w}$ is the bilinear map:

$$(\mathbf{u} \otimes \mathbf{w})(f, g) = f(\mathbf{u}) \cdot g(\mathbf{w}) \quad \text{for } f \in V^*, g \in W^*$$

**Example 2.10.4 (Tensor product of two vectors).** Let $\mathbf{u} = \begin{pmatrix} 1 \\ 2 \end{pmatrix} \in \mathbb{R}^2$ and $\mathbf{w} = \begin{pmatrix} 3 \\ 4 \\ 5 \end{pmatrix} \in \mathbb{R}^3$. Their tensor product $\mathbf{u} \otimes \mathbf{w}$ is the $2 \times 3$ matrix:

$$\mathbf{u} \otimes \mathbf{w} = \begin{pmatrix} 1 \\ 2 \end{pmatrix} \begin{pmatrix} 3 & 4 & 5 \end{pmatrix} = \begin{pmatrix} 1 \cdot 3 & 1 \cdot 4 & 1 \cdot 5 \\ 2 \cdot 3 & 2 \cdot 4 & 2 \cdot 5 \end{pmatrix} = \begin{pmatrix} 3 & 4 & 5 \\ 6 & 8 & 10 \end{pmatrix}$$

This lives in $\mathbb{R}^2 \otimes \mathbb{R}^3 \cong \mathbb{R}^{2 \times 3}$ and has rank 1, as expected for a simple tensor product.

**Definition 2.10.4 (Tensor Product of Spaces).** The *tensor product* $V \otimes W$ is the vector space spanned by all elements $\mathbf{u} \otimes \mathbf{w}$:

$$V \otimes W = \operatorname{span}\{\mathbf{u} \otimes \mathbf{w} : \mathbf{u} \in V, \mathbf{w} \in W\}$$

**Theorem 2.10.2.** If $\dim V = m$ and $\dim W = n$, then $\dim(V \otimes W) = mn$.

*Proof.* Let $\{\mathbf{e}_i\}_{i=1}^m$ be a basis for $V$ and $\{\mathbf{f}_j\}_{j=1}^n$ a basis for $W$. We claim $\{\mathbf{e}_i \otimes \mathbf{f}_j\}_{1 \leq i \leq m, 1 \leq j \leq n}$ is a basis for $V \otimes W$.

**Spanning:** Any $\mathbf{u} \otimes \mathbf{w} = \left(\sum_i u_i \mathbf{e}_i\right) \otimes \left(\sum_j w_j \mathbf{f}_j\right) = \sum_{i,j} u_i w_j\, (\mathbf{e}_i \otimes \mathbf{f}_j)$ by bilinearity.

**Independence:** Suppose $\sum_{i,j} c_{ij}(\mathbf{e}_i \otimes \mathbf{f}_j) = 0$. Evaluating on $(\mathbf{e}^k, \mathbf{f}^l)$:

$$\sum_{i,j} c_{ij} \mathbf{e}^k(\mathbf{e}_i) \mathbf{f}^l(\mathbf{f}_j) = \sum_{i,j} c_{ij} \delta_{ki}\delta_{lj} = c_{kl} = 0$$

for all $k, l$. Hence the $mn$ elements are independent. $\square$

### 2.1 Properties of the Tensor Product

**Theorem 2.10.3.** The tensor product satisfies:

1. **Bilinearity:** $(\alpha\mathbf{u}_1 + \beta\mathbf{u}_2) \otimes \mathbf{w} = \alpha(\mathbf{u}_1 \otimes \mathbf{w}) + \beta(\mathbf{u}_2 \otimes \mathbf{w})$
2. **Associativity:** $(U \otimes V) \otimes W \cong U \otimes (V \otimes W)$
3. **Distributivity:** $U \otimes (V \oplus W) \cong (U \otimes V) \oplus (U \otimes W)$
4. **Not commutative** in general, but $V \otimes W \cong W \otimes V$ (canonically isomorphic)

### 2.2 Components in a Basis

A tensor $T \in V \otimes W$ has components:

$$T = \sum_{i=1}^m \sum_{j=1}^n T_{ij}\, \mathbf{e}_i \otimes \mathbf{f}_j$$

The array of numbers $(T_{ij})$ is an $m \times n$ matrix --- the component representation.

For higher-order tensor products $V_1 \otimes V_2 \otimes \cdots \otimes V_N$:

$$T = \sum_{i_1, i_2, \dots, i_N} T_{i_1 i_2 \cdots i_N}\, \mathbf{e}^{(1)}_{i_1} \otimes \mathbf{e}^{(2)}_{i_2} \otimes \cdots \otimes \mathbf{e}^{(N)}_{i_N}$$

The components $T_{i_1 i_2 \cdots i_N}$ form an $N$-dimensional array of shape $d_1 \times d_2 \times \cdots \times d_N$.

**Example 2.10.5 (Higher-order tensor product in components).** Let $\mathbf{a} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$, $\mathbf{b} = \begin{pmatrix} 3 \\ 4 \end{pmatrix}$, $\mathbf{c} = \begin{pmatrix} 5 \\ 6 \end{pmatrix}$. The order-3 tensor $\mathcal{T} = \mathbf{a} \otimes \mathbf{b} \otimes \mathbf{c} \in \mathbb{R}^{2 \times 2 \times 2}$ has components $T_{ijk} = a_i \cdot b_j \cdot c_k$:

$$\mathcal{T}(:,:,1) = 5 \begin{pmatrix} 1 \cdot 3 & 1 \cdot 4 \\ 2 \cdot 3 & 2 \cdot 4 \end{pmatrix} = \begin{pmatrix} 15 & 20 \\ 30 & 40 \end{pmatrix}, \quad \mathcal{T}(:,:,2) = 6 \begin{pmatrix} 3 & 4 \\ 6 & 8 \end{pmatrix} = \begin{pmatrix} 18 & 24 \\ 36 & 48 \end{pmatrix}$$

For instance, $T_{2,1,2} = a_2 \cdot b_1 \cdot c_2 = 2 \cdot 3 \cdot 6 = 36$. This is a rank-1 tensor since it is a single outer product.

*ML connection:* **Deep learning tensors** are exactly these component arrays:

| ML tensor | Mathematical tensor product | Shape |
|-----------|---------------------------|-------|
| Batch of images | $\mathbb{R}^B \otimes \mathbb{R}^C \otimes \mathbb{R}^H \otimes \mathbb{R}^W$ | $(B, C, H, W)$ |
| Word embedding matrix | $\mathbb{R}^V \otimes \mathbb{R}^d$ | $(V, d)$ |
| Attention scores | $\mathbb{R}^B \otimes \mathbb{R}^H \otimes \mathbb{R}^L \otimes \mathbb{R}^L$ | $(B, H, L, L)$ |
| Conv filter | $\mathbb{R}^{C_\text{out}} \otimes \mathbb{R}^{C_\text{in}} \otimes \mathbb{R}^{k_h} \otimes \mathbb{R}^{k_w}$ | $(C_\text{out}, C_\text{in}, k_h, k_w)$ |

---

## 3. Einstein Summation Notation

**Definition 2.10.5 (Einstein Convention).** When an index appears once as a subscript and once as a superscript (or twice in a summation context), summation over that index is implied:

$$A^i{}_j v^j \equiv \sum_{j} A^i{}_j v^j$$

The repeated index $j$ is called a *dummy index* or *contracted index*.

### 3.1 Common Operations in Einstein Notation

| Operation | Standard | Einstein | NumPy `einsum` |
|-----------|----------|----------|----------------|
| Inner product | $\sum_i u_i v_i$ | $u_i v^i$ | `'i,i->'` |
| Matrix-vector | $\sum_j A_{ij} v_j$ | $A_{ij} v^j$ | `'ij,j->i'` |
| Matrix multiply | $\sum_k A_{ik} B_{kj}$ | $A_{ik} B^k{}_j$ | `'ik,kj->ij'` |
| Trace | $\sum_i A_{ii}$ | $A^i{}_i$ | `'ii->'` |
| Outer product | $u_i v_j$ | $u_i v_j$ | `'i,j->ij'` |
| Batch matmul | $\sum_k A_{bik} B_{bkj}$ | $A_{bik} B_b{}^k{}_j$ | `'bik,bkj->bij'` |
| Attention | $\sum_k Q_{bhik} K_{bhjk}$ | --- | `'bhik,bhjk->bhij'` |

**Example 2.10.6 (Einstein summation with numbers).** Let $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$ and $\mathbf{v} = \begin{pmatrix} 5 \\ 6 \end{pmatrix}$. Computing $w_i = A_{ij}v^j$ (matrix-vector product):

$$w_1 = A_{11}v^1 + A_{12}v^2 = 1 \cdot 5 + 2 \cdot 6 = 17$$

$$w_2 = A_{21}v^1 + A_{22}v^2 = 3 \cdot 5 + 4 \cdot 6 = 39$$

Computing the trace $A^i{}_i = A_{11} + A_{22} = 1 + 4 = 5$. For the outer product $u_i v_j$ with $\mathbf{u} = \begin{pmatrix} 2 \\ 3 \end{pmatrix}$: the result is $\begin{pmatrix} 2 \cdot 5 & 2 \cdot 6 \\ 3 \cdot 5 & 3 \cdot 6 \end{pmatrix} = \begin{pmatrix} 10 & 12 \\ 15 & 18 \end{pmatrix}$.

**Theorem 2.10.4.** Einstein summation is well-defined: the result is independent of the choice of basis, provided the transformation laws are respected (contravariant indices transform with the inverse of the change-of-basis matrix, covariant indices transform with the change-of-basis matrix itself).

*ML connection:* **`torch.einsum` and `np.einsum`** implement Einstein notation directly. This is the most general and readable way to express tensor operations in ML code:

```python
# Attention: scores = softmax(QK^T / sqrt(d_k))
scores = torch.einsum('bhid,bhjd->bhij', Q, K) / sqrt(d_k)

# Bilinear form: y = x^T A z
y = torch.einsum('bi,ijk,bk->bj', x, A, z)
```

**Example 2.10.10 (Batch matrix multiply with einsum).** Let $\mathcal{A} \in \mathbb{R}^{2 \times 2 \times 2}$ be a batch of two $2 \times 2$ matrices and $\mathcal{B} \in \mathbb{R}^{2 \times 2 \times 2}$ another batch:

$$A_1 = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix},\; A_2 = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}, \quad B_1 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix},\; B_2 = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$$

The batch matmul $C_{bij} = \sum_k A_{bik} B_{bkj}$ (einsum `'bik,bkj->bij'`) computes each product independently:

$$C_1 = A_1 B_1 = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, \quad C_2 = A_2 B_2 = \begin{pmatrix} 16 & 17 \\ 22 & 23 \end{pmatrix}$$

The batch index $b$ is *free* (not summed over), so each batch element is processed in parallel.

---

## 4. Contraction

**Definition 2.10.6 (Contraction).** Given a type $(r, s)$ tensor $T$, *contraction* on the $p$-th contravariant and $q$-th covariant indices produces a type $(r-1, s-1)$ tensor:

$$(\operatorname{tr}_{p,q} T)(\dots) = \sum_{k=1}^n T(\dots, \underset{p\text{-th}}{\mathbf{e}^k}, \dots, \underset{q\text{-th}}{\mathbf{e}_k}, \dots)$$

In components:

$$(C^{i_1 \cdots \hat{i}_p \cdots}{}_{j_1 \cdots \hat{j}_q \cdots}) = \sum_{k} T^{i_1 \cdots k \cdots}{}_{j_1 \cdots k \cdots}$$

**Familiar examples of contraction:**

| Contraction | Result |
|-------------|--------|
| Contract a $(1,1)$ tensor (matrix) | Trace: $A^i{}_i = \operatorname{tr}(A)$ |
| Contract $(1,0) \otimes (0,1)$ | Inner product: $u^i v_i = \langle \mathbf{u}, \mathbf{v} \rangle$ |
| Contract $(1,1) \otimes (1,0)$ | Matrix-vector product: $A^i{}_j v^j$ |

```
CONTRACTION VISUALISED

    Order-3 tensor Tᵢⱼₖ          Contract over j,k
                                   (sum over shared index)
    ┌──────────┐                  ┌──────────┐
    │  j       │                  │          │
    │ ┌──┐     │     contract     │          │
    │ │T │──k  │    ─────────→    │   vᵢ     │   (order-1 vector)
    │ └──┘     │     j = k        │          │
    │  i       │                  │          │
    └──────────┘                  └──────────┘
    (3 indices)                   (1 index)
```

**Example 2.10.7 (Contraction of a rank-3 tensor).** Let $\mathcal{T} \in \mathbb{R}^{2 \times 2 \times 2}$ with slices:

$$\mathcal{T}(:,:,1) = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, \quad \mathcal{T}(:,:,2) = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$$

Contracting over the 2nd and 3rd indices (setting $j = k$) gives a vector $v_i = \sum_j T_{ijj}$:

$$v_1 = T_{111} + T_{122} = 1 + 6 = 7, \quad v_2 = T_{211} + T_{222} = 3 + 8 = 11$$

So the contraction produces $\mathbf{v} = \begin{pmatrix} 7 \\ 11 \end{pmatrix}$. We went from an order-3 tensor (8 numbers) to an order-1 tensor (2 numbers).

---

## 5. Tensor Rank

**Definition 2.10.7 (Rank-1 Tensor).** A tensor $T \in V_1 \otimes V_2 \otimes \cdots \otimes V_N$ is *rank-1* if it can be written as:

$$T = \mathbf{a}^{(1)} \otimes \mathbf{a}^{(2)} \otimes \cdots \otimes \mathbf{a}^{(N)}$$

In components: $T_{i_1 i_2 \cdots i_N} = a^{(1)}_{i_1} a^{(2)}_{i_2} \cdots a^{(N)}_{i_N}$.

**Example 2.10.8 (Rank-1 vs. higher-rank tensor).** The tensor $\mathcal{S} \in \mathbb{R}^{2 \times 2}$ with $S = \begin{pmatrix} 2 & 3 \\ 4 & 6 \end{pmatrix}$ is rank-1 because $S = \begin{pmatrix} 1 \\ 2 \end{pmatrix} \otimes \begin{pmatrix} 2 & 3 \end{pmatrix}$. We can verify: every $2 \times 2$ submatrix has determinant zero ($2 \cdot 6 - 3 \cdot 4 = 0$).

In contrast, $I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ has rank 2, since $\det(I) = 1 \neq 0$, so it cannot be $\mathbf{a} \otimes \mathbf{b}$. Its decomposition requires two terms: $I = \begin{pmatrix} 1 \\ 0 \end{pmatrix} \otimes \begin{pmatrix} 1 \\ 0 \end{pmatrix}^T + \begin{pmatrix} 0 \\ 1 \end{pmatrix} \otimes \begin{pmatrix} 0 \\ 1 \end{pmatrix}^T$.

**Definition 2.10.8 (Tensor Rank / CP Rank).** The *rank* of a tensor $T$ is the minimum number of rank-1 tensors needed to express it:

$$\operatorname{rank}(T) = \min\left\{R : T = \sum_{r=1}^R \mathbf{a}_r^{(1)} \otimes \mathbf{a}_r^{(2)} \otimes \cdots \otimes \mathbf{a}_r^{(N)}\right\}$$

**Key differences from matrix rank:**

| Property | Matrix rank | Tensor rank ($N \geq 3$) |
|----------|-------------|--------------------------|
| Computation | Polynomial time (SVD) | NP-hard in general |
| Maximum rank | $\min(m, n)$ | Can exceed any single dimension |
| Typical rank | Well-defined | Depends on field ($\mathbb{R}$ vs $\mathbb{C}$) |
| Best rank-$k$ approx | Always exists (Eckart-Young) | May not exist (not closed) |

**Theorem 2.10.5.** For order-2 tensors (matrices), tensor rank equals matrix rank.

*Proof.* A rank-1 matrix is $\mathbf{u}\mathbf{v}^T = \mathbf{u} \otimes \mathbf{v}$. The minimal number of such terms is the matrix rank by the SVD outer product form $A = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^T$. $\square$

---

## 6. Tensor Decompositions

### 6.1 CP Decomposition (CANDECOMP/PARAFAC)

**Definition 2.10.9 (CP Decomposition).** The *CP decomposition* expresses a tensor $\mathcal{T} \in \mathbb{R}^{d_1 \times d_2 \times \cdots \times d_N}$ as a sum of rank-1 tensors:

$$\mathcal{T} = \sum_{r=1}^{R} \lambda_r\, \mathbf{a}_r^{(1)} \otimes \mathbf{a}_r^{(2)} \otimes \cdots \otimes \mathbf{a}_r^{(N)}$$

In components:

$$T_{i_1 i_2 \cdots i_N} = \sum_{r=1}^{R} \lambda_r\, a^{(1)}_{r,i_1}\, a^{(2)}_{r,i_2} \cdots a^{(N)}_{r,i_N}$$

```
CP DECOMPOSITION OF AN ORDER-3 TENSOR

                                R
  ┌─────────┐                 ___
  │         │╲               ╲
  │    T    │ │    =          ╱  λᵣ  aᵣ⁽¹⁾ ⊗ aᵣ⁽²⁾ ⊗ aᵣ⁽³⁾
  │         │ │              ╱
  │         │╱               ‾‾‾
  └─────────┘               r = 1

  d₁×d₂×d₃             = λ₁ [|] ⊗ [──] ⊗ [|]  +  λ₂ [|] ⊗ [──] ⊗ [|]  + ...
   tensor                     a₁    b₁    c₁         a₂    b₂    c₂
```

**Storage:** A CP decomposition with rank $R$ of a tensor of shape $d_1 \times \cdots \times d_N$ requires $R \sum_{n=1}^N d_n$ parameters, compared to $\prod_{n=1}^N d_n$ for the full tensor. For large $N$, this is an exponential reduction.

### 6.2 Tucker Decomposition

**Definition 2.10.10 (Tucker Decomposition).** The *Tucker decomposition* expresses a tensor as a small *core tensor* multiplied by factor matrices along each mode:

$$\mathcal{T} = \mathcal{G} \times_1 U^{(1)} \times_2 U^{(2)} \times_3 \cdots \times_N U^{(N)}$$

where $\mathcal{G} \in \mathbb{R}^{r_1 \times r_2 \times \cdots \times r_N}$ is the core tensor and $U^{(n)} \in \mathbb{R}^{d_n \times r_n}$ are the factor matrices. Here $\times_n$ denotes the *mode-$n$ product*.

In components:

$$T_{i_1 i_2 \cdots i_N} = \sum_{j_1=1}^{r_1} \sum_{j_2=1}^{r_2} \cdots \sum_{j_N=1}^{r_N} G_{j_1 j_2 \cdots j_N}\, U^{(1)}_{i_1 j_1}\, U^{(2)}_{i_2 j_2} \cdots U^{(N)}_{i_N j_N}$$

```
TUCKER DECOMPOSITION OF AN ORDER-3 TENSOR

  ┌─────────┐           ┌─────┐
  │         │╲          │     │╲          U⁽¹⁾         U⁽²⁾        U⁽³⁾
  │    T    │ │   =     │  G  │ │    ×₁  [d₁×r₁]  ×₂  [d₂×r₂]  ×₃ [d₃×r₃]
  │         │ │         │     │ │
  │         │╱          │     │╱
  └─────────┘           └─────┘
  d₁×d₂×d₃            r₁×r₂×r₃
  (large)              (small core)
```

**Storage:** $\prod_n r_n + \sum_n d_n r_n$ parameters.

### 6.3 Comparison of Decompositions

| Property | CP | Tucker |
|----------|-----|--------|
| Core tensor | None (diagonal $\lambda_r$) | Full $\mathcal{G}$ (interactions between modes) |
| Parameters | $R \sum_n d_n$ | $\prod_n r_n + \sum_n d_n r_n$ |
| Uniqueness | Often unique (under mild conditions) | Not unique (rotation ambiguity in $U^{(n)}$) |
| Generalises | SVD outer product form | Truncated SVD |
| Computation | Alternating least squares (ALS) | Higher-order SVD (HOSVD) |

**Theorem 2.10.6 (Kruskal's Uniqueness).** The CP decomposition is essentially unique (up to permutation and scaling of components) if the factor matrices satisfy $\sum_{n=1}^N k_{\text{rank}}(A^{(n)}) \geq 2R + N - 1$, where $k_{\text{rank}}$ is the Kruskal rank (the largest $k$ such that every set of $k$ columns is linearly independent).

*ML connection:* **Tensor decompositions** have become central in modern deep learning:

- **Model compression:** A convolutional filter $\mathcal{W} \in \mathbb{R}^{C_\text{out} \times C_\text{in} \times k \times k}$ can be decomposed via CP or Tucker to reduce parameters and FLOPs by 3-10x with minimal accuracy loss.
- **Tensor networks in NLP:** Decomposing embedding tensors in language models reduces memory from $O(V \cdot d)$ to $O(V^{1/N} \cdot d \cdot R)$.
- **Tensor completion:** Filling in missing entries of a tensor (generalising matrix completion in recommender systems) uses low-rank tensor assumptions.

---

## 7. Tensors and Multidimensional Arrays

### 7.1 The Correspondence

**Proposition 2.10.1.** After choosing bases for each vector space, there is a one-to-one correspondence:

$$\text{Order-}N\text{ tensors over } V_1 \times \cdots \times V_N \quad \longleftrightarrow \quad \text{Arrays in } \mathbb{R}^{d_1 \times \cdots \times d_N}$$

The tensor is the *basis-independent* object; the array is its *representation* in chosen coordinates.

**Example 2.10.9 (Reshape operations).** Consider the $2 \times 3$ matrix $A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}$. Reshaping reinterprets the same 6 numbers with a different index structure (row-major order):

$$\text{reshape}(A, (3,2)) = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix}, \quad \text{reshape}(A, (6,1)) = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \\ 6 \end{pmatrix}, \quad \text{reshape}(A, (2,1,3)) = \text{a } 2 \times 1 \times 3 \text{ tensor}$$

Note: reshaping does not change the data, only how indices map to entries. In contrast, *transposing* $A$ to get $A^T = \begin{pmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{pmatrix}$ actually rearranges data, since $(A^T)_{ij} = A_{ji}$.

### 7.2 Mode-$n$ Unfolding (Matricisation)

**Definition 2.10.11 (Mode-$n$ Unfolding).** The *mode-$n$ unfolding* of a tensor $\mathcal{T} \in \mathbb{R}^{d_1 \times \cdots \times d_N}$ is the matrix $T_{(n)} \in \mathbb{R}^{d_n \times (d_1 \cdots d_{n-1} d_{n+1} \cdots d_N)}$ obtained by arranging the mode-$n$ fibres as columns.

```
MODE UNFOLDING OF A 3×4×2 TENSOR

  Original tensor T:           Mode-1 unfolding T₍₁₎ ∈ R³ˣ⁸:

  "Slice" k=1:  "Slice" k=2:  ┌                         ┐
  ┌─────────┐   ┌─────────┐   │ t₁₁₁ t₁₂₁ t₁₃₁ t₁₄₁ t₁₁₂ t₁₂₂ t₁₃₂ t₁₄₂ │
  │ t₁₁₁ ...│   │ t₁₁₂ ...│   │ t₂₁₁ t₂₂₁ t₂₃₁ t₂₄₁ t₂₁₂ t₂₂₂ t₂₃₂ t₂₄₂ │
  │ t₂₁₁ ...│   │ t₂₁₂ ...│   │ t₃₁₁ t₃₂₁ t₃₃₁ t₃₄₁ t₃₁₂ t₃₂₂ t₃₃₂ t₃₄₂ │
  │ t₃₁₁ ...│   │ t₃₁₂ ...│   └                         ┘
  └─────────┘   └─────────┘
```

**Example 2.10.11 (Mode unfolding with numbers).** Let $\mathcal{T} \in \mathbb{R}^{2 \times 3 \times 2}$ with slices:

$$\mathcal{T}(:,:,1) = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}, \quad \mathcal{T}(:,:,2) = \begin{pmatrix} 7 & 8 & 9 \\ 10 & 11 & 12 \end{pmatrix}$$

The mode-1 unfolding $T_{(1)} \in \mathbb{R}^{2 \times 6}$ arranges mode-1 fibres as rows:

$$T_{(1)} = \begin{pmatrix} 1 & 2 & 3 & 7 & 8 & 9 \\ 4 & 5 & 6 & 10 & 11 & 12 \end{pmatrix}$$

The mode-2 unfolding $T_{(2)} \in \mathbb{R}^{3 \times 4}$ arranges mode-2 fibres as rows:

$$T_{(2)} = \begin{pmatrix} 1 & 4 & 7 & 10 \\ 2 & 5 & 8 & 11 \\ 3 & 6 & 9 & 12 \end{pmatrix}$$

Each unfolding reshapes the same 12 numbers into a different matrix, exposing a different mode's structure.

The Tucker decomposition can be computed via SVDs of the mode-$n$ unfoldings: this is the **Higher-Order SVD (HOSVD)**.

---

## 8. Quantum State Tensors

**Definition 2.10.12 (Quantum State as Tensor).** A system of $N$ qubits has a state $|\psi\rangle \in (\mathbb{C}^2)^{\otimes N} = \mathbb{C}^2 \otimes \mathbb{C}^2 \otimes \cdots \otimes \mathbb{C}^2$, which is an order-$N$ tensor with $2^N$ components:

$$|\psi\rangle = \sum_{i_1, i_2, \dots, i_N \in \{0,1\}} c_{i_1 i_2 \cdots i_N}\, |i_1\rangle \otimes |i_2\rangle \otimes \cdots \otimes |i_N\rangle$$

**Example 2.10.2 (Two-qubit states).**

| State | Tensor form | Rank | Entanglement |
|-------|-------------|------|-------------|
| $\|00\rangle$ | $\|0\rangle \otimes \|0\rangle$ | 1 | None (product state) |
| $\frac{1}{\sqrt{2}}(\|00\rangle + \|11\rangle)$ | $\frac{1}{\sqrt{2}}(c_{00} + c_{11})$ | 2 | Maximally entangled (Bell state) |
| $\frac{1}{\sqrt{2}}(\|0\rangle + \|1\rangle) \otimes \|0\rangle$ | Rank-1 | 1 | None |

**Theorem 2.10.7.** A pure quantum state $|\psi\rangle$ of a bipartite system $\mathcal{H}_A \otimes \mathcal{H}_B$ is *entangled* if and only if $\operatorname{rank}(\psi) > 1$ as a tensor, i.e., it cannot be written as $|\alpha\rangle \otimes |\beta\rangle$.

*The Schmidt decomposition* of a bipartite state is exactly the SVD of the coefficient matrix $C_{ij}$ when we reshape $|\psi\rangle$ into a matrix indexed by $(i, j)$.

```
TENSOR PRODUCT IN QUANTUM COMPUTING

    Single qubit:  |ψ⟩ = α|0⟩ + β|1⟩   ∈ C²

    Two qubits:    |ψ⟩ ⊗ |φ⟩ = (α|0⟩ + β|1⟩) ⊗ (γ|0⟩ + δ|1⟩)
                              = αγ|00⟩ + αδ|01⟩ + βγ|10⟩ + βδ|11⟩

    As a matrix:   C = ┌         ┐
                       │ αγ   αδ │   ← rank 1 (product state)
                       │ βγ   βδ │
                       └         ┘

    Bell state:    C = ┌              ┐
                       │ 1/√2    0    │   ← rank 2 (entangled!)
                       │   0   1/√2   │
                       └              ┘
```

*Quantum-ML connection:* **Tensor networks** (MPS, PEPS, MERA) were originally developed in quantum many-body physics to represent $2^N$-dimensional quantum states efficiently. They have since been imported into ML for:

- Compressing neural network weight tensors
- Expressiveness analysis of deep networks (depth vs. width trade-offs)
- Quantum machine learning circuits as parameterised tensor networks

---

## 9. Key Identities and Properties

**Theorem 2.10.8 (Mixed-Product Property).** For matrices $A, B, C, D$ of compatible sizes:

$$(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$$

**Theorem 2.10.9 (Kronecker Product Properties).** The Kronecker product $A \otimes B$ (the matrix representation of the tensor product of linear maps) satisfies:

1. $(A \otimes B)^T = A^T \otimes B^T$
2. $(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}$ (when inverses exist)
3. $\operatorname{tr}(A \otimes B) = \operatorname{tr}(A) \cdot \operatorname{tr}(B)$
4. $\det(A \otimes B) = (\det A)^n (\det B)^m$ for $A \in \mathbb{R}^{m \times m}, B \in \mathbb{R}^{n \times n}$
5. Eigenvalues of $A \otimes B$ are $\{\lambda_i \mu_j\}$ where $\lambda_i, \mu_j$ are eigenvalues of $A, B$

**Example 2.10.12 (Kronecker product).** Let $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$ and $B = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$. The Kronecker product $A \otimes B \in \mathbb{R}^{4 \times 4}$:

$$A \otimes B = \begin{pmatrix} 1 \cdot B & 2 \cdot B \\ 3 \cdot B & 4 \cdot B \end{pmatrix} = \begin{pmatrix} 0 & 1 & 0 & 2 \\ 1 & 0 & 2 & 0 \\ 0 & 3 & 0 & 4 \\ 3 & 0 & 4 & 0 \end{pmatrix}$$

We can verify: $\operatorname{tr}(A \otimes B) = \operatorname{tr}(A) \cdot \operatorname{tr}(B) = (1+4)(0+0) = 0$, which matches the trace of the $4 \times 4$ matrix ($0+0+0+0 = 0$). Also, $\det(A \otimes B) = (\det A)^2 (\det B)^2 = (-2)^2(-1)^2 = 4$.

*ML connection:* The mixed-product property is why **separable convolutions** work. A 2D convolution with a rank-1 filter $\mathbf{h}\mathbf{g}^T = \mathbf{h} \otimes \mathbf{g}$ can be decomposed into a 1D convolution along rows (with $\mathbf{g}$) followed by a 1D convolution along columns (with $\mathbf{h}$), reducing cost from $O(k^2)$ to $O(2k)$ per pixel. MobileNets exploit this via depthwise separable convolutions.

---

## Exercises

### Foundations (★)

**Exercise 2.10.1.** Let $\mathbf{u} = (1, 2)^T \in \mathbb{R}^2$ and $\mathbf{v} = (3, 4, 5)^T \in \mathbb{R}^3$. Compute $\mathbf{u} \otimes \mathbf{v}$ as a $2 \times 3$ matrix. Verify that $\operatorname{rank}(\mathbf{u} \otimes \mathbf{v}) = 1$.

**Exercise 2.10.2.** Using Einstein notation, write out (a) the matrix product $C_{ij} = A_{ik}B_{kj}$, (b) the trace $\operatorname{tr}(AB) = A_{ij}B_{ji}$, and (c) the quadratic form $\mathbf{x}^TA\mathbf{x} = x_i A_{ij} x_j$. Verify each by expanding for $2 \times 2$ matrices.

**Exercise 2.10.3.** Express the following NumPy operation in Einstein notation: `np.einsum('bij,bjk->bik', A, B)`. What is this operation called?

### Core Theory (★★)

**Exercise 2.10.4.** Prove that the tensor $T \in \mathbb{R}^2 \otimes \mathbb{R}^2$ with components $T_{ij} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ has tensor rank 2. (Hint: show it cannot be written as $\mathbf{a} \otimes \mathbf{b}$ for any $\mathbf{a}, \mathbf{b} \in \mathbb{R}^2$.)

**Exercise 2.10.5.** Let $\mathcal{T} \in \mathbb{R}^{3 \times 4 \times 5}$. (a) What are the shapes of the three mode unfoldings $T_{(1)}, T_{(2)}, T_{(3)}$? (b) If $\mathcal{T}$ has Tucker decomposition with core $\mathcal{G} \in \mathbb{R}^{2 \times 3 \times 2}$, how many parameters are needed compared to storing $\mathcal{T}$ directly?

**Exercise 2.10.6.** Prove the mixed-product property $(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$ by verifying it in components using Einstein notation.

### Advanced / ML Applications (★★★)

**Exercise 2.10.7.** **(Convolutional filter compression)** A convolutional layer has filter tensor $\mathcal{W} \in \mathbb{R}^{64 \times 64 \times 3 \times 3}$. (a) How many parameters does the full tensor have? (b) A CP decomposition with rank $R = 16$ represents $\mathcal{W} \approx \sum_{r=1}^{16} \mathbf{a}_r \otimes \mathbf{b}_r \otimes \mathbf{c}_r \otimes \mathbf{d}_r$. How many parameters? (c) What is the compression ratio?

**Exercise 2.10.8.** **(Quantum entanglement)** Consider the 3-qubit state $|\psi\rangle = \frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)$ (the GHZ state). Write out the coefficient tensor $c_{i_1 i_2 i_3}$ and show that the tensor rank is 2. Find the CP decomposition. Is there a bipartition for which the Schmidt rank is 2?

**Exercise 2.10.9.** **(Attention as tensor contraction)** In multi-head attention, $Q, K, V \in \mathbb{R}^{B \times H \times L \times d}$. Write the attention output $\text{Attn}(Q, K, V)_{bhid} = \sum_j \operatorname{softmax}_j\!\left(\frac{Q_{bhik}K_{bhjk}}{\sqrt{d}}\right) V_{bhjd}$ as a sequence of tensor operations. Identify which operations are contractions and which are element-wise.

**Exercise 2.10.10.** **(Tensor network for MNIST)** A $28 \times 28$ MNIST image can be reshaped into a tensor of shape $4 \times 7 \times 4 \times 7$, then further into a tensor train (TT) of order 4. If each TT core has shape $r_{n-1} \times d_n \times r_n$ with ranks $(1, r, r, r, 1)$ and $r = 5$, how many parameters describe the image? Compare to the original $784$ pixels.

---

## Related Topics

- **Previous:** [Matrix Decompositions](matrix-decompositions.md) --- SVD as order-2 tensor decomposition
- **Quantum Computing:** Tensor networks (MPS, PEPS), entanglement measures
- **Deep Learning:** PyTorch/TensorFlow tensor operations, `torch.einsum`
- **Multilinear Algebra:** Symmetric tensors, alternating tensors (differential forms)
- **Probability:** Moment tensors, cumulant tensors (independent component analysis)

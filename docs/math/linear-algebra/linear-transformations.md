# Linear Transformations

A linear transformation is a function between vector spaces that preserves the two fundamental operations: addition and scalar multiplication. Every matrix multiplication $\mathbf{y} = A\mathbf{x}$ is a linear transformation, and conversely, every linear transformation between finite-dimensional spaces can be represented as matrix multiplication once bases are chosen. This correspondence is at the heart of all computational linear algebra. In machine learning, every dense layer (without its activation function) is a linear transformation, every attention projection is a linear transformation, and dimensionality reduction techniques like PCA are linear transformations that optimally preserve variance.

---

## Prerequisites

- [Foundations](../foundations/index.md) — sets, functions (especially injections, surjections, bijections)
- [Vectors](vectors.md) — vector operations, linear combinations
- [Matrices](matrices.md) — matrix multiplication, rank, inverse
- [Systems of Equations](systems-of-equations.md) — null space, solution structure
- [Vector Spaces](vector-spaces.md) — subspaces, basis, dimension, coordinate vectors

---

## 1. Definition and Basic Properties

**Definition 2.5.1 (Linear Transformation).** Let $V$ and $W$ be vector spaces over the same field $\mathbb{F}$. A function $T: V \to W$ is a *linear transformation* (or *linear map*) if for all $\mathbf{u}, \mathbf{v} \in V$ and all $\alpha \in \mathbb{F}$:

1. $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$ (preserves addition)
2. $T(\alpha\mathbf{v}) = \alpha T(\mathbf{v})$ (preserves scalar multiplication)

Equivalently (combined form): $T(\alpha\mathbf{u} + \beta\mathbf{v}) = \alpha T(\mathbf{u}) + \beta T(\mathbf{v})$ for all $\alpha, \beta \in \mathbb{F}$.

**Example 2.8.1 (Verifying Linearity).** Let $T: \mathbb{R}^2 \to \mathbb{R}^2$ be defined by $T(x, y) = (2x - y,\; x + 3y)$. Verify linearity with $\mathbf{u} = (1, 2)$, $\mathbf{v} = (3, -1)$, $\alpha = 2$, $\beta = -1$.

$$\alpha\mathbf{u} + \beta\mathbf{v} = 2(1,2) + (-1)(3,-1) = (2,4) + (-3,1) = (-1, 5)$$

$$T(-1, 5) = (2(-1) - 5,\; -1 + 3(5)) = (-7, 14)$$

$$\alpha T(\mathbf{u}) + \beta T(\mathbf{v}) = 2\,T(1,2) + (-1)\,T(3,-1) = 2(0, 7) + (-1)(7, 0) = (0,14) + (-7,0) = (-7, 14) \;\checkmark$$

**Terminology:**

| Term | Meaning |
|------|---------|
| Linear transformation | General term for $T: V \to W$ |
| Linear operator | $T: V \to V$ (same domain and codomain) |
| Linear functional | $T: V \to \mathbb{F}$ (maps to the scalar field) |
| Homomorphism | Synonym for linear transformation (in algebraic language) |

**Theorem 2.5.1 (Elementary Properties).** Let $T: V \to W$ be linear. Then:

(a) $T(\mathbf{0}_V) = \mathbf{0}_W$
(b) $T(-\mathbf{v}) = -T(\mathbf{v})$
(c) $T\left(\sum_{i=1}^k \alpha_i \mathbf{v}_i\right) = \sum_{i=1}^k \alpha_i T(\mathbf{v}_i)$

*Proof of (a).* $T(\mathbf{0}) = T(\mathbf{0} + \mathbf{0}) = T(\mathbf{0}) + T(\mathbf{0})$. Subtracting $T(\mathbf{0})$ from both sides gives $\mathbf{0}_W = T(\mathbf{0}_V)$. $\square$

*Proof of (c).* By induction on $k$, applying both linearity conditions at each step. $\square$

**Theorem 2.5.2 (Determination by Basis).** A linear transformation $T: V \to W$ is completely determined by its action on a basis. If $\mathcal{B} = \{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$ is a basis for $V$, then specifying $T(\mathbf{v}_1), \ldots, T(\mathbf{v}_n) \in W$ uniquely determines $T$.

*Proof.* Any $\mathbf{v} \in V$ can be written uniquely as $\mathbf{v} = \sum \alpha_i \mathbf{v}_i$. Then $T(\mathbf{v}) = \sum \alpha_i T(\mathbf{v}_i)$ is forced by linearity. Conversely, defining $T$ this way on all of $V$ gives a well-defined linear map: the unique expansion guarantees well-definedness, and the formula ensures linearity. $\square$

*ML connection:* Theorem 2.5.2 is why a **dense layer** $\mathbf{y} = W\mathbf{x}$ is fully described by its weight matrix. The columns of $W$ are exactly $T(\mathbf{e}_1), \ldots, T(\mathbf{e}_n)$ — the images of the standard basis vectors. The $mn$ entries of $W$ are the *only* parameters needed. The activation function $\sigma$ applied afterward is what makes neural networks nonlinear and thus more expressive than a single matrix.

---

## 2. Examples of Linear Transformations

### 2.1 Geometric Transformations in $\mathbb{R}^2$

```
ROTATION BY θ                   SCALING BY (sx, sy)

    ↑ y'                            ↑ y
    │  ╱ rotated v                  │
    │ ╱  θ                          │   ┌─────┐ scaled
    │╱ ╱ original v                 │   │     │
    ●─────→ x'                      │   │  ●  │
                                    │   │     │
    R_θ(v) = rotation matrix        ●───┴─────┴───→ x
              applied to v
```

| Transformation | Formula | Matrix | Geometric Effect |
|---------------|---------|--------|-----------------|
| Rotation by $\theta$ | $R_\theta(\mathbf{v})$ | $\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$ | Rotates by $\theta$ counterclockwise |
| Scaling | $S(\mathbf{v})$ | $\begin{pmatrix} s_x & 0 \\ 0 & s_y \end{pmatrix}$ | Stretches along axes |
| Reflection (x-axis) | $F(\mathbf{v})$ | $\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ | Flips across x-axis |
| Shear (horizontal) | $H(\mathbf{v})$ | $\begin{pmatrix} 1 & k \\ 0 & 1 \end{pmatrix}$ | Slants horizontally |
| Projection onto x-axis | $P(\mathbf{v})$ | $\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$ | Drops y-component |

**Example 2.8.2 (Rotation by 90°).** Rotate $\mathbf{v} = (3, 1)$ counterclockwise by $\theta = 90°$.

$$R_{90°} = \begin{pmatrix} \cos 90° & -\sin 90° \\ \sin 90° & \cos 90° \end{pmatrix} = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$$

$$R_{90°}\begin{pmatrix} 3 \\ 1 \end{pmatrix} = \begin{pmatrix} 0(3) + (-1)(1) \\ 1(3) + 0(1) \end{pmatrix} = \begin{pmatrix} -1 \\ 3 \end{pmatrix}$$

The vector $(3,1)$ maps to $(-1, 3)$: same length ($\sqrt{10}$), rotated 90° counterclockwise.

**Example 2.8.3 (Reflection Across x-axis).** Reflect $\mathbf{v} = (2, 5)$ across the x-axis.

$$F = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}, \qquad F\begin{pmatrix} 2 \\ 5 \end{pmatrix} = \begin{pmatrix} 2 \\ -5 \end{pmatrix}$$

The x-component is unchanged; the y-component flips sign — the point "mirrors" across the x-axis.

**Example 2.8.4 (Horizontal Shear).** Apply a horizontal shear with $k = 2$ to $\mathbf{v} = (1, 3)$.

$$H = \begin{pmatrix} 1 & 2 \\ 0 & 1 \end{pmatrix}, \qquad H\begin{pmatrix} 1 \\ 3 \end{pmatrix} = \begin{pmatrix} 1 + 2(3) \\ 3 \end{pmatrix} = \begin{pmatrix} 7 \\ 3 \end{pmatrix}$$

The y-component stays at 3, but x shifts by $k \cdot y = 6$. Points higher up are pushed further right.

**Example 2.8.5a (Non-uniform Scaling).** Scale $\mathbf{v} = (2, 3)$ by $s_x = 3$ horizontally and $s_y = 0.5$ vertically.

$$S = \begin{pmatrix} 3 & 0 \\ 0 & 0.5 \end{pmatrix}, \qquad S\begin{pmatrix} 2 \\ 3 \end{pmatrix} = \begin{pmatrix} 6 \\ 1.5 \end{pmatrix}$$

The point stretches to 3x its width but compresses to half its height — commonly used in image rescaling.

*ML connection:* **Data augmentation** in computer vision uses these transformations: random rotations, flips, scaling, and shearing create new training samples. Since these are linear, they compose efficiently: applying rotation then scaling is a single matrix multiplication $S \cdot R_\theta$. **Spatial transformer networks** learn these transformation matrices as part of the network, enabling geometric invariance.

### 2.2 Projection

**Definition 2.5.2 (Orthogonal Projection).** The orthogonal projection onto a subspace $W \leq V$ (in an inner product space) maps each vector to its closest point in $W$:

$$\text{proj}_W(\mathbf{v}) = \sum_{i=1}^k \frac{\langle \mathbf{v}, \mathbf{w}_i \rangle}{\langle \mathbf{w}_i, \mathbf{w}_i \rangle} \mathbf{w}_i$$

where $\{\mathbf{w}_1, \ldots, \mathbf{w}_k\}$ is an orthogonal basis for $W$.

Properties: $\text{proj}_W$ is linear, $\text{proj}_W \circ \text{proj}_W = \text{proj}_W$ (idempotent), and $\text{proj}_W^T = \text{proj}_W$ (symmetric).

**Example 2.8.5 (Projection onto a Line).** Project $\mathbf{v} = (3, 4)$ onto $W = \text{span}\{(1, 1)\}$.

$$\text{proj}_W(\mathbf{v}) = \frac{\langle (3,4),\, (1,1) \rangle}{\langle (1,1),\, (1,1) \rangle}(1,1) = \frac{3 + 4}{1 + 1}(1,1) = \frac{7}{2}(1,1) = \left(\frac{7}{2},\, \frac{7}{2}\right)$$

Verify idempotence: $\text{proj}_W\!\left(\frac{7}{2}, \frac{7}{2}\right) = \frac{7}{2}(1,1) = \left(\frac{7}{2}, \frac{7}{2}\right)$. Projecting again gives the same result.

*ML connection:* **PCA** projects data onto the top-$k$ eigenvectors of the covariance matrix — this is an orthogonal projection from $\mathbb{R}^n$ to a $k$-dimensional subspace. **Attention mechanisms** compute a soft, data-dependent projection: the query-key scores determine which subspace of the value space to project onto.

### 2.3 Differentiation as a Linear Map

**Example.** $D: \mathcal{P}_n(\mathbb{R}) \to \mathcal{P}_{n-1}(\mathbb{R})$ defined by $D(p) = p'$ (the derivative) is linear:

- $D(p + q) = (p+q)' = p' + q' = D(p) + D(q)$
- $D(\alpha p) = (\alpha p)' = \alpha p' = \alpha D(p)$

In the standard basis $\{1, x, x^2, x^3\}$ for $\mathcal{P}_3$:

$$D(1) = 0, \quad D(x) = 1, \quad D(x^2) = 2x, \quad D(x^3) = 3x^2$$

The matrix representation is:

$$[D] = \begin{pmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & 2 & 0 \\ 0 & 0 & 0 & 3 \end{pmatrix}$$

### 2.4 Integration as a Linear Map

**Example.** $\mathcal{I}: C[0,1] \to \mathbb{R}$ defined by $\mathcal{I}(f) = \int_0^1 f(x)\, dx$ is a linear functional. It maps a vector space of functions to scalars. Linearity follows from the linearity of the integral.

### 2.5 The Zero and Identity Maps

**Example.** The *zero map* $O: V \to W$, $O(\mathbf{v}) = \mathbf{0}$ for all $\mathbf{v}$, and the *identity map* $I: V \to V$, $I(\mathbf{v}) = \mathbf{v}$, are both linear. These are the extremes: $O$ loses all information; $I$ preserves everything.

---

## 3. Matrix Representation

**Theorem 2.5.3 (Matrix Representation).** Let $T: V \to W$ be linear, $\mathcal{B} = \{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$ a basis for $V$, and $\mathcal{C} = \{\mathbf{w}_1, \ldots, \mathbf{w}_m\}$ a basis for $W$. Then there exists a unique matrix $A \in \mathbb{F}^{m \times n}$ such that:

$$[T(\mathbf{v})]_\mathcal{C} = A [\mathbf{v}]_\mathcal{B}$$

The $j$-th column of $A$ is $[T(\mathbf{v}_j)]_\mathcal{C}$ — the coordinate vector of the image of the $j$-th basis vector.

*Proof.* Write $T(\mathbf{v}_j) = \sum_{i=1}^m a_{ij}\mathbf{w}_i$. The scalars $a_{ij}$ form the matrix $A$. For any $\mathbf{v} = \sum_j \alpha_j \mathbf{v}_j$:

$$T(\mathbf{v}) = \sum_j \alpha_j T(\mathbf{v}_j) = \sum_j \alpha_j \sum_i a_{ij}\mathbf{w}_i = \sum_i \left(\sum_j a_{ij}\alpha_j\right)\mathbf{w}_i$$

The coordinates in basis $\mathcal{C}$ are $\sum_j a_{ij}\alpha_j$, which is exactly the matrix-vector product $A[\mathbf{v}]_\mathcal{B}$. $\square$

**Example 2.8.6 (Finding a Matrix Representation).** Let $T: \mathbb{R}^2 \to \mathbb{R}^2$ be defined by $T(x, y) = (2x - y,\; x + 3y)$. Find its matrix in the standard basis.

Apply $T$ to each standard basis vector:

$$T(\mathbf{e}_1) = T(1, 0) = (2, 1), \qquad T(\mathbf{e}_2) = T(0, 1) = (-1, 3)$$

The columns of $A$ are $T(\mathbf{e}_1)$ and $T(\mathbf{e}_2)$:

$$A = \begin{pmatrix} 2 & -1 \\ 1 & 3 \end{pmatrix}$$

Verify: $A\begin{pmatrix}3\\2\end{pmatrix} = \begin{pmatrix}2(3)+(-1)(2)\\1(3)+3(2)\end{pmatrix} = \begin{pmatrix}4\\9\end{pmatrix} = T(3,2)$. $\checkmark$

```
THE FUNDAMENTAL CORRESPONDENCE

                     T
    V ──────────────────────────→ W
    │                              │
    │ choose         choose        │
    │ basis B        basis C       │
    │                              │
    ▼         matrix mult.         ▼
   F^n ──────────────────────────→ F^m
           [v]_B ├──→ A[v]_B = [T(v)]_C

Every linear map T ←→ a matrix A (once bases are chosen)
Different bases ←→ different matrices for THE SAME map
```

*ML connection:* A **neural network layer** $\mathbf{y} = W\mathbf{x} + \mathbf{b}$ is the matrix representation of an affine transformation in the standard basis. The weight matrix $W$ *is* the linear map (with respect to standard bases). When we perform **weight tying** in encoder-decoder models (using $W^T$ for the decoder), we are using the adjoint of the same linear map. When we apply **LoRA** (Low-Rank Adaptation), we decompose the weight update as $\Delta W = BA$ where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, and $r \ll d$ — constraining the update to a low-rank linear transformation.

---

## 4. Kernel and Image

**Definition 2.5.3 (Kernel / Null Space).** The *kernel* (or *null space*) of a linear map $T: V \to W$ is:

$$\ker(T) = \{\mathbf{v} \in V : T(\mathbf{v}) = \mathbf{0}\}$$

**Definition 2.5.4 (Image / Range / Column Space).** The *image* (or *range*) of $T$ is:

$$\text{im}(T) = \{T(\mathbf{v}) : \mathbf{v} \in V\} = \{w \in W : w = T(\mathbf{v}) \text{ for some } \mathbf{v} \in V\}$$

**Example 2.8.7 (Computing Kernel and Image).** Let $T: \mathbb{R}^2 \to \mathbb{R}^2$ have matrix $A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}$.

*Kernel:* Solve $A\mathbf{x} = \mathbf{0}$. Row-reduce: $R_2 \leftarrow R_2 - 2R_1$ gives $\begin{pmatrix} 1 & 2 \\ 0 & 0 \end{pmatrix}$, so $x_1 = -2x_2$ with $x_2$ free.

$$\ker(T) = \text{span}\left\{\begin{pmatrix} -2 \\ 1 \end{pmatrix}\right\}, \quad \dim(\ker(T)) = 1$$

*Image:* The column space of $A$ is $\text{span}\left\{\begin{pmatrix} 1 \\ 2 \end{pmatrix}\right\}$ (second column is $2 \times$ first), so $\dim(\text{im}(T)) = 1$.

**Theorem 2.5.4.** $\ker(T)$ is a subspace of $V$ and $\text{im}(T)$ is a subspace of $W$.

*Proof.* (Kernel) Let $\mathbf{u}, \mathbf{v} \in \ker(T)$ and $\alpha \in \mathbb{F}$.
- $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v}) = \mathbf{0} + \mathbf{0} = \mathbf{0}$, so $\mathbf{u} + \mathbf{v} \in \ker(T)$.
- $T(\alpha\mathbf{u}) = \alpha T(\mathbf{u}) = \alpha\mathbf{0} = \mathbf{0}$, so $\alpha\mathbf{u} \in \ker(T)$.
- $\ker(T) \neq \emptyset$ since $T(\mathbf{0}) = \mathbf{0}$.

By the subspace test, $\ker(T)$ is a subspace.

(Image) Let $\mathbf{w}_1, \mathbf{w}_2 \in \text{im}(T)$, so $\mathbf{w}_i = T(\mathbf{v}_i)$ for some $\mathbf{v}_i$.
- $\mathbf{w}_1 + \mathbf{w}_2 = T(\mathbf{v}_1) + T(\mathbf{v}_2) = T(\mathbf{v}_1 + \mathbf{v}_2) \in \text{im}(T)$.
- $\alpha\mathbf{w}_1 = \alpha T(\mathbf{v}_1) = T(\alpha\mathbf{v}_1) \in \text{im}(T)$.
- $\text{im}(T) \neq \emptyset$ since $\mathbf{0}_W = T(\mathbf{0}_V)$.

By the subspace test, $\text{im}(T)$ is a subspace. $\square$

**Definition 2.5.5 (Rank and Nullity).**

$$\text{rank}(T) = \dim(\text{im}(T)) \qquad \text{nullity}(T) = \dim(\ker(T))$$

```
KERNEL AND IMAGE VISUALIZED

    V (domain)                           W (codomain)
┌─────────────────┐               ┌─────────────────┐
│                 │               │                 │
│  ┌───────────┐  │      T       │  ┌───────────┐  │
│  │  ker(T)   │──│─── maps ───→ │  │   {0}     │  │
│  │  (nullity │  │  to zero     │  └───────────┘  │
│  │  = n - r) │  │               │                 │
│  └───────────┘  │               │  ┌───────────┐  │
│                 │  ─── maps ──→ │  │  im(T)    │  │
│  everything    │  onto image   │  │  (rank     │  │
│  else          │               │  │  = r)      │  │
│                 │               │  └───────────┘  │
└─────────────────┘               └─────────────────┘

dim(V) = dim(ker(T)) + dim(im(T))
   n    =   (n - r)   +     r
```

**Theorem 2.5.5 (Injectivity Criterion).** $T$ is injective (one-to-one) if and only if $\ker(T) = \{\mathbf{0}\}$.

*Proof.* ($\Rightarrow$) If $T$ is injective and $T(\mathbf{v}) = \mathbf{0} = T(\mathbf{0})$, then $\mathbf{v} = \mathbf{0}$.

($\Leftarrow$) If $T(\mathbf{u}) = T(\mathbf{v})$, then $T(\mathbf{u} - \mathbf{v}) = \mathbf{0}$, so $\mathbf{u} - \mathbf{v} \in \ker(T) = \{\mathbf{0}\}$, giving $\mathbf{u} = \mathbf{v}$. $\square$

**Example 2.8.11 (Injectivity Check).** Is $T$ with matrix $A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}$ injective? From Example 2.8.7, $\ker(T) = \text{span}\{(-2, 1)\} \neq \{\mathbf{0}\}$, so $T$ is **not** injective. For instance, $T(2, 0) = (2, 4) = T(0, 1)$ — distinct inputs produce the same output.

Contrast: rotation $R_{90°}$ has $\ker(R_{90°}) = \{\mathbf{0}\}$ (no nonzero vector maps to zero), so it **is** injective.

*ML connection:* Consider an **encoder** $E: \mathbb{R}^n \to \mathbb{R}^d$ with $d < n$. Since $\dim(\ker(E)) \geq n - d > 0$, the encoder *cannot* be injective — it must lose information. The kernel $\ker(E)$ contains the directions in input space that map to the same latent vector. Two inputs $\mathbf{x}_1$ and $\mathbf{x}_2$ that differ only by an element of $\ker(E)$ are indistinguishable in the latent space. This is the **information bottleneck** principle: the encoder is forced to learn which information to preserve and which to discard (the kernel).

---

## 5. The Rank-Nullity Theorem

This is one of the most important theorems in linear algebra.

**Theorem 2.5.6 (Rank-Nullity Theorem).** Let $T: V \to W$ be a linear map and $\dim(V) = n$ (finite). Then:

$$\dim(\ker(T)) + \dim(\text{im}(T)) = \dim(V)$$

$$\text{nullity}(T) + \text{rank}(T) = n$$

*Proof.* Let $\dim(\ker(T)) = k$ and let $\{\mathbf{u}_1, \ldots, \mathbf{u}_k\}$ be a basis for $\ker(T)$.

**Step 1.** Extend this to a basis for $V$: $\{\mathbf{u}_1, \ldots, \mathbf{u}_k, \mathbf{v}_1, \ldots, \mathbf{v}_r\}$ where $k + r = n$ (by the basis extension theorem).

**Step 2.** Claim: $\{T(\mathbf{v}_1), \ldots, T(\mathbf{v}_r)\}$ is a basis for $\text{im}(T)$.

*Spanning:* Any $\mathbf{w} \in \text{im}(T)$ has the form $\mathbf{w} = T(\mathbf{x})$ for some $\mathbf{x} \in V$. Write $\mathbf{x} = \sum_{i=1}^k \alpha_i\mathbf{u}_i + \sum_{j=1}^r \beta_j\mathbf{v}_j$. Then:

$$\mathbf{w} = T(\mathbf{x}) = \sum_{i=1}^k \alpha_i \underbrace{T(\mathbf{u}_i)}_{= \mathbf{0}} + \sum_{j=1}^r \beta_j T(\mathbf{v}_j) = \sum_{j=1}^r \beta_j T(\mathbf{v}_j)$$

So $\text{im}(T) = \text{span}\{T(\mathbf{v}_1), \ldots, T(\mathbf{v}_r)\}$.

*Independence:* Suppose $\sum_{j=1}^r \gamma_j T(\mathbf{v}_j) = \mathbf{0}$. By linearity, $T\left(\sum_{j=1}^r \gamma_j\mathbf{v}_j\right) = \mathbf{0}$, so $\sum \gamma_j \mathbf{v}_j \in \ker(T)$. Therefore $\sum \gamma_j\mathbf{v}_j = \sum_{i=1}^k \delta_i\mathbf{u}_i$ for some scalars $\delta_i$. This gives:

$$\sum_{j=1}^r \gamma_j\mathbf{v}_j - \sum_{i=1}^k \delta_i\mathbf{u}_i = \mathbf{0}$$

Since $\{\mathbf{u}_1, \ldots, \mathbf{u}_k, \mathbf{v}_1, \ldots, \mathbf{v}_r\}$ is a basis for $V$ (linearly independent), all coefficients are zero: $\gamma_j = 0$ for all $j$ and $\delta_i = 0$ for all $i$.

**Step 3.** Therefore $\dim(\text{im}(T)) = r$ and $\dim(V) = k + r = \dim(\ker(T)) + \dim(\text{im}(T))$. $\square$

**Corollary 2.5.1.** For $A \in \mathbb{R}^{m \times n}$: $\text{rank}(A) + \text{nullity}(A) = n$.

**Example 2.8.8 (Rank-Nullity Verification).** For the matrix from Example 2.8.7, $A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}$ with domain $\mathbb{R}^2$ ($n = 2$):

$$\underbrace{\dim(\ker(T))}_{= 1} + \underbrace{\dim(\text{im}(T))}_{= 1} = 2 = \dim(\mathbb{R}^2) \;\checkmark$$

One dimension of information is lost (the kernel direction $(-2, 1)$); one is kept (the image direction $(1, 2)$).

**Corollary 2.5.2.** If $\dim(V) = \dim(W) = n$, then for $T: V \to W$:

$$T \text{ is injective} \iff T \text{ is surjective} \iff T \text{ is bijective}$$

*Proof.* $T$ injective $\iff \text{nullity} = 0 \iff \text{rank} = n = \dim(W) \iff T$ surjective. $\square$

```
RANK-NULLITY BALANCE

dim(V) = n
├── nullity(T) = dim(ker(T))     ← information LOST
└── rank(T)    = dim(im(T))      ← information KEPT

n = nullity + rank

Examples with n = 10:

  rank 10, nullity 0:  bijective (no information lost)
  rank 7,  nullity 3:  3 dimensions collapsed
  rank 1,  nullity 9:  severe compression (maps to a line)
  rank 0,  nullity 10: zero map (all information lost)
```

*ML connection:* The rank-nullity theorem quantifies the **information bottleneck** in every linear layer.

- A linear encoder $E: \mathbb{R}^{784} \to \mathbb{R}^{64}$ has rank $\leq 64$, so nullity $\geq 720$. At least 720 dimensions of input information are discarded.
- **Matrix rank** of a weight matrix $W$ equals the effective dimensionality of the layer's output. If $W$ has rank $r < \min(m,n)$, the layer is equivalent to a two-layer network $\mathbb{R}^n \to \mathbb{R}^r \to \mathbb{R}^m$ — this is the theoretical basis for **LoRA**, which explicitly parameterizes weight updates as rank-$r$ matrices $\Delta W = BA$.
- In **dropout** (at training time), randomly zeroing neurons reduces the effective rank of the transformation, which can be viewed as a rank-nullity rebalancing that prevents overfitting.

---

## 6. Composition of Linear Maps

**Definition 2.5.6 (Composition).** If $T_1: U \to V$ and $T_2: V \to W$ are linear, their *composition* $T_2 \circ T_1: U \to W$ is defined by $(T_2 \circ T_1)(\mathbf{u}) = T_2(T_1(\mathbf{u}))$.

**Theorem 2.5.7.** The composition of linear maps is linear, and its matrix representation is the matrix product:

$$[T_2 \circ T_1] = [T_2][T_1]$$

*Proof.* $(T_2 \circ T_1)(\alpha\mathbf{u} + \beta\mathbf{v}) = T_2(\alpha T_1(\mathbf{u}) + \beta T_1(\mathbf{v})) = \alpha T_2(T_1(\mathbf{u})) + \beta T_2(T_1(\mathbf{v}))$. For the matrix product: $[T_2 \circ T_1][\mathbf{u}]_\mathcal{B} = [T_2]([T_1][\mathbf{u}]_\mathcal{B}) = ([T_2][T_1])[\mathbf{u}]_\mathcal{B}$. $\square$

**Example 2.8.9 (Composition = Matrix Product).** Let $T_1$ = rotation by 90° and $T_2$ = reflection across x-axis. Apply $T_2 \circ T_1$ to $\mathbf{v} = (1, 0)$.

$$[T_1] = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}, \quad [T_2] = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

*Step-by-step:* $T_1(1,0) = (0, 1)$, then $T_2(0, 1) = (0, -1)$.

*Via matrix product:* $[T_2][T_1] = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}\begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix} = \begin{pmatrix} 0 & -1 \\ -1 & 0 \end{pmatrix}$

$$\begin{pmatrix} 0 & -1 \\ -1 & 0 \end{pmatrix}\begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 0 \\ -1 \end{pmatrix} \;\checkmark$$

The combined matrix $\begin{pmatrix} 0 & -1 \\ -1 & 0 \end{pmatrix}$ is a reflection across the line $y = -x$.

**Theorem 2.5.8 (Rank Inequality for Composition).**

$$\text{rank}(T_2 \circ T_1) \leq \min(\text{rank}(T_1), \text{rank}(T_2))$$

*Proof.* $\text{im}(T_2 \circ T_1) = T_2(\text{im}(T_1)) \subseteq \text{im}(T_2)$, so $\text{rank}(T_2 \circ T_1) \leq \text{rank}(T_2)$. Also, $\text{im}(T_2 \circ T_1) = T_2(\text{im}(T_1))$, and a linear map on a space of dimension $\text{rank}(T_1)$ has image dimension $\leq \text{rank}(T_1)$. $\square$

```
COMPOSITION = MATRIX MULTIPLICATION

    U ──T₁──→ V ──T₂──→ W

       [T₁]        [T₂]
  R^p ──────→ R^n ──────→ R^m

  Composition: [T₂ ∘ T₁] = [T₂] · [T₁]

  Size: (m×n) · (n×p) = (m×p)
```

*ML connection:* A **deep neural network** (without activations) is a composition of linear maps:

$$T = T_L \circ T_{L-1} \circ \cdots \circ T_1$$

The corresponding matrix is $W = W_L W_{L-1} \cdots W_1$. By the rank inequality, $\text{rank}(W) \leq \min_i \text{rank}(W_i)$. This means that a deep linear network with a bottleneck layer of dimension $r$ can express at most a rank-$r$ transformation — *exactly the same* as a single rank-$r$ matrix. This is why nonlinear activations are essential: without them, depth provides no additional expressiveness. With activations, each layer can reshape the space before the next linear map, enabling the network to approximate arbitrary functions.

---

## 7. Isomorphisms

**Definition 2.5.7 (Isomorphism).** A linear map $T: V \to W$ is an *isomorphism* if it is bijective (both injective and surjective). We write $V \cong W$.

**Theorem 2.5.9 (Isomorphism Criterion).** For finite-dimensional spaces, $V \cong W$ if and only if $\dim(V) = \dim(W)$.

*Proof.* ($\Rightarrow$) If $T: V \to W$ is an isomorphism, then $\ker(T) = \{\mathbf{0}\}$ and $\text{im}(T) = W$. By rank-nullity: $\dim(V) = 0 + \dim(W)$.

($\Leftarrow$) If $\dim(V) = \dim(W) = n$, choose bases $\mathcal{B}_V$ and $\mathcal{B}_W$. Define $T(\mathbf{v}_i) = \mathbf{w}_i$. This $T$ maps basis to basis, is clearly bijective, and is linear by Theorem 2.5.2. $\square$

**Corollary 2.5.3.** Every $n$-dimensional vector space over $\mathbb{F}$ is isomorphic to $\mathbb{F}^n$.

This is why we can always "work in coordinates" — choosing a basis reduces any abstract vector space problem to a concrete problem in $\mathbb{F}^n$.

**Theorem 2.5.10 (Inverse is Linear).** If $T: V \to W$ is an isomorphism, then $T^{-1}: W \to V$ is also linear.

*Proof.* Let $\mathbf{w}_1, \mathbf{w}_2 \in W$ with $T^{-1}(\mathbf{w}_i) = \mathbf{v}_i$. Then $T(\alpha\mathbf{v}_1 + \beta\mathbf{v}_2) = \alpha\mathbf{w}_1 + \beta\mathbf{w}_2$, so $T^{-1}(\alpha\mathbf{w}_1 + \beta\mathbf{w}_2) = \alpha\mathbf{v}_1 + \beta\mathbf{v}_2 = \alpha T^{-1}(\mathbf{w}_1) + \beta T^{-1}(\mathbf{w}_2)$. $\square$

*ML connection:* An **autoencoder** with encoder $E: \mathbb{R}^n \to \mathbb{R}^d$ and decoder $D: \mathbb{R}^d \to \mathbb{R}^n$ tries to learn $D \circ E \approx I$. If $d < n$, this cannot be an isomorphism (rank deficiency). If $d = n$ and both $E$ and $D$ are linear and invertible, then $D = E^{-1}$ is an isomorphism and the autoencoder is trivial (it perfectly reconstructs everything). The interesting case is $d < n$, where the encoder-decoder pair learns the "best" rank-$d$ approximation. The nonlinearities in deep autoencoders allow them to capture nonlinear structure that purely linear maps (like PCA) miss.

---

## 8. Change of Basis for Linear Maps

**Theorem 2.5.11 (Change of Basis for Transformations).** Let $T: V \to V$ be a linear operator with matrix $A$ relative to basis $\mathcal{B}$, and let $P$ be the change-of-basis matrix from $\mathcal{B}$ to $\mathcal{B}'$. Then the matrix of $T$ relative to $\mathcal{B}'$ is:

$$A' = P^{-1}AP$$

*Proof.* For any $\mathbf{v} \in V$:

$$[T(\mathbf{v})]_{\mathcal{B}'} = P^{-1}[T(\mathbf{v})]_\mathcal{B} = P^{-1}A[\mathbf{v}]_\mathcal{B} = P^{-1}A P[\mathbf{v}]_{\mathcal{B}'}$$

Therefore the matrix of $T$ in basis $\mathcal{B}'$ is $P^{-1}AP$. $\square$

**Example 2.8.10 (Change of Basis).** Let $T: \mathbb{R}^2 \to \mathbb{R}^2$ have matrix $A = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$ in the standard basis $\mathcal{B} = \{\mathbf{e}_1, \mathbf{e}_2\}$. Find the matrix of $T$ in the basis $\mathcal{B}' = \{(1, 0),\, (1, 1)\}$.

The change-of-basis matrix $P$ has columns $= $ new basis vectors: $P = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$, $P^{-1} = \begin{pmatrix} 1 & -1 \\ 0 & 1 \end{pmatrix}$.

$$A' = P^{-1}AP = \begin{pmatrix} 1 & -1 \\ 0 & 1 \end{pmatrix}\begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 3 & -1 \\ 0 & 2 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 3 & 2 \\ 0 & 2 \end{pmatrix}$$

Both $A$ and $A'$ represent the same transformation $T$; they share trace $= 5$, determinant $= 6$, and eigenvalues $\{3, 2\}$.

**Definition 2.5.8 (Similar Matrices).** Matrices $A$ and $B$ are *similar* if $B = P^{-1}AP$ for some invertible $P$. Similar matrices represent the same linear operator in different bases.

**Properties preserved by similarity:**

| Property | Preserved? | Why It Matters |
|----------|-----------|----------------|
| Rank | Yes | Information capacity is basis-independent |
| Determinant | Yes | $\det(P^{-1}AP) = \det(A)$ |
| Trace | Yes | $\text{tr}(P^{-1}AP) = \text{tr}(A)$ |
| Eigenvalues | Yes | Characteristic polynomial is invariant |
| Characteristic polynomial | Yes | $\det(P^{-1}AP - \lambda I) = \det(A - \lambda I)$ |
| Entries of the matrix | No | Entries are coordinate-dependent |

```
CHANGE OF BASIS FOR LINEAR MAPS

Basis B:                     Basis B':
    [v]_B ──── A ────→ [Tv]_B      [v]_B' ── A' ──→ [Tv]_B'

       │                  ↑            │                 ↑
       │ P                │ P⁻¹        │                 │
       ↓                  │            ↓                 │
    [v]_B'               [Tv]_B'     Same computation but
                                     A' = P⁻¹AP

The KEY INSIGHT: A and A' represent the SAME transformation T.
Diagonalization = finding a basis where A' is diagonal.
```

*ML connection:* **Diagonalization** is the art of finding a basis where a linear operator has the simplest matrix (a diagonal matrix). In ML:

- **PCA** diagonalizes the covariance matrix: in the eigenvector basis, the covariance becomes diagonal, meaning the new features are uncorrelated.
- **Batch normalization** implicitly performs a change of basis (whitening) to make optimization easier.
- **Spectral graph convolutions** work in the eigenvector basis of the graph Laplacian, where the convolution operator is diagonal.

---

## 9. The Space of Linear Maps

**Definition 2.5.9.** The set of all linear maps from $V$ to $W$ is denoted $\mathcal{L}(V, W)$ or $\text{Hom}(V, W)$.

**Theorem 2.5.12.** $\mathcal{L}(V, W)$ is itself a vector space under:

- $(T_1 + T_2)(\mathbf{v}) = T_1(\mathbf{v}) + T_2(\mathbf{v})$
- $(\alpha T)(\mathbf{v}) = \alpha T(\mathbf{v})$

with $\dim(\mathcal{L}(V, W)) = \dim(V) \cdot \dim(W)$.

*Proof sketch.* The correspondence $T \leftrightarrow A$ (matrix representation) gives an isomorphism $\mathcal{L}(V, W) \cong \mathbb{F}^{m \times n}$, which has dimension $mn$. $\square$

*ML connection:* The **parameter space** of a single linear layer mapping $\mathbb{R}^n \to \mathbb{R}^m$ is $\mathcal{L}(\mathbb{R}^n, \mathbb{R}^m) \cong \mathbb{R}^{mn}$. This is why a dense layer has $mn$ parameters (plus $m$ bias terms). Understanding the structure of this space — its dimension, its geometry under different loss functions — is the foundation of **neural architecture search** and **parameter efficiency** research.

---

## 10. Summary of Key Correspondences

| Linear Algebra Concept | Matrix Language | ML / Neural Network Interpretation |
|------------------------|----------------|-------------------------------------|
| Linear map $T: V \to W$ | Matrix $A \in \mathbb{R}^{m \times n}$ | Dense layer (without activation) |
| $\ker(T)$ | $\text{Null}(A)$ | Input directions the layer ignores |
| $\text{im}(T)$ | $\text{Col}(A)$ | Set of possible layer outputs |
| $\text{rank}(T)$ | $\text{rank}(A)$ | Effective output dimensionality |
| $T$ injective | $\text{Null}(A) = \{\mathbf{0}\}$ | No information loss |
| $T$ surjective | $\text{Col}(A) = \mathbb{R}^m$ | Can produce any output |
| $T$ bijective (isomorphism) | $A$ invertible | Perfectly reversible layer |
| Composition $T_2 \circ T_1$ | Matrix product $A_2 A_1$ | Stacking layers |
| Similar matrices | $A' = P^{-1}AP$ | Same layer, different coordinate systems |
| Rank-nullity | $r + (n-r) = n$ | Information kept + lost = total |

---

## Exercises

**★ Basic**

1. Determine which of the following are linear transformations:
    - (a) $T: \mathbb{R}^2 \to \mathbb{R}$, $T(x, y) = x + y$
    - (b) $T: \mathbb{R}^2 \to \mathbb{R}$, $T(x, y) = x^2 + y$
    - (c) $T: \mathbb{R}^2 \to \mathbb{R}^2$, $T(x, y) = (2x - y, x + 3y)$
    - (d) $T: \mathbb{R}^2 \to \mathbb{R}^2$, $T(x, y) = (x + 1, y)$

2. Find the matrix representation of the linear map $T: \mathbb{R}^3 \to \mathbb{R}^2$ defined by $T(x, y, z) = (x + 2y, 3y - z)$. Find $\ker(T)$ and $\text{im}(T)$ and verify the rank-nullity theorem.

3. Let $R_\theta$ be rotation by $\theta$ in $\mathbb{R}^2$. Compute the matrix of $R_{\pi/4}$ and verify that $R_{\pi/4} \circ R_{\pi/4} = R_{\pi/2}$ using matrix multiplication.

4. Find the matrix of the projection $P: \mathbb{R}^3 \to \mathbb{R}^3$ onto the $xy$-plane. Compute $\ker(P)$, $\text{im}(P)$, and verify $P^2 = P$.

**★★ Intermediate**

5. Let $T: \mathcal{P}_3(\mathbb{R}) \to \mathcal{P}_3(\mathbb{R})$ be defined by $T(p) = p' + p$ (derivative plus identity). Find the matrix of $T$ in the basis $\{1, x, x^2, x^3\}$. Find $\ker(T)$ and $\text{im}(T)$.

6. (Neural Network Layers) Consider two linear layers: $T_1: \mathbb{R}^5 \to \mathbb{R}^3$ (weight matrix $W_1$) and $T_2: \mathbb{R}^3 \to \mathbb{R}^4$ (weight matrix $W_2$).
    - (a) What is the maximum rank of $T_2 \circ T_1$?
    - (b) If $\text{rank}(W_1) = 2$, what is the maximum rank of $W_2 W_1$?
    - (c) Explain why this shows that a bottleneck layer of dimension 3 limits the expressiveness of all subsequent layers.

7. Prove that $T: V \to V$ is an isomorphism if and only if $T$ maps every basis of $V$ to a basis of $V$.

8. Let $A$ and $B$ be similar matrices. Prove that $A^k$ and $B^k$ are similar for all $k \geq 1$. Why does this matter for computing $A^{1000}$ in practice?

**★★★ Challenging**

9. (Rank-Nullity Applications) Let $T: V \to V$ be a linear operator on a finite-dimensional space with $T^2 = T$ (idempotent).
    - (a) Prove that $V = \ker(T) \oplus \text{im}(T)$.
    - (b) Prove that the only eigenvalues of $T$ are $0$ and $1$.
    - (c) Give an ML example of an idempotent transformation and interpret parts (a) and (b) in that context.

10. (Dimensionality Reduction) Let $E: \mathbb{R}^n \to \mathbb{R}^d$ and $D: \mathbb{R}^d \to \mathbb{R}^n$ be linear maps with $d < n$. Prove that $D \circ E$ cannot be the identity on $\mathbb{R}^n$. What is the maximum rank of $D \circ E$? If we want to minimize $\|x - DE(x)\|^2$ averaged over a dataset, show that the optimal solution involves the top-$d$ eigenvectors of the data covariance matrix (this is PCA).

11. (Quantum Gates) A quantum gate on $n$ qubits is a unitary linear transformation $U: \mathbb{C}^{2^n} \to \mathbb{C}^{2^n}$ (i.e., $U^\dagger U = I$). Prove that every unitary map is an isomorphism. Show that the kernel of a unitary map is trivial and the rank is maximal. Why does this mean quantum computation is inherently reversible (unlike classical computation where AND gates lose information)?

---

## Related Topics

- [Vectors](vectors.md) — the concrete objects that linear transformations act on
- [Matrices](matrices.md) — the coordinate representation of linear maps
- [Vector Spaces](vector-spaces.md) — the domains and codomains of linear maps
- [Determinants](determinants.md) — $\det(T) \neq 0$ iff $T$ is an isomorphism
- [Eigenvalues & Eigenvectors](eigenvalues.md) — the directions preserved by a linear operator
- [Inner Product Spaces](inner-product-spaces.md) — adjoints, orthogonal projections, unitary maps
- [Matrix Decompositions](matrix-decompositions.md) — factoring transformations (SVD, spectral decomposition)

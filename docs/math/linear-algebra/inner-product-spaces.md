# Inner Product Spaces

Every notion of "similarity" in machine learning ultimately rests on an inner product. Cosine similarity between word embeddings, the attention scores in a transformer, kernel functions in SVMs, and the loss function in least squares regression --- all are inner products or derived from them. This chapter generalises the dot product from $\mathbb{R}^n$ (Definition 2.1.4) to arbitrary vector spaces over $\mathbb{R}$ and $\mathbb{C}$, then develops the rich geometric theory that follows.

---

## Prerequisites

- [Vectors](vectors.md) --- dot product, norms
- [Vector Spaces](vector-spaces.md) --- subspaces, basis, dimension
- [Linear Transformations](linear-transformations.md) --- kernel, image

---

## 1. Inner Products

### 1.1 Definition Over $\mathbb{R}$ and $\mathbb{C}$

**Definition 2.8.1 (Inner Product).** Let $V$ be a vector space over a field $\mathbb{F}$ where $\mathbb{F} = \mathbb{R}$ or $\mathbb{F} = \mathbb{C}$. An *inner product* on $V$ is a function $\langle \cdot, \cdot \rangle : V \times V \to \mathbb{F}$ satisfying, for all $\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$ and $\alpha, \beta \in \mathbb{F}$:

1. **Conjugate symmetry:** $\langle \mathbf{u}, \mathbf{v} \rangle = \overline{\langle \mathbf{v}, \mathbf{u} \rangle}$
2. **Linearity in the first argument:** $\langle \alpha\mathbf{u} + \beta\mathbf{w}, \mathbf{v} \rangle = \alpha\langle \mathbf{u}, \mathbf{v} \rangle + \beta\langle \mathbf{w}, \mathbf{v} \rangle$
3. **Positive definiteness:** $\langle \mathbf{v}, \mathbf{v} \rangle \geq 0$ with equality if and only if $\mathbf{v} = \mathbf{0}$

A vector space equipped with an inner product is called an *inner product space*. A complete inner product space is a *Hilbert space*.

**Convention (Physics vs. Mathematics).** Physicists use linearity in the *second* argument (Dirac convention). We follow the mathematics/ML convention: linearity in the *first* argument.

> **Note:** When $\mathbb{F} = \mathbb{R}$, conjugate symmetry reduces to plain symmetry: $\langle \mathbf{u}, \mathbf{v} \rangle = \langle \mathbf{v}, \mathbf{u} \rangle$.

**Example 2.8.1 (Standard inner products).**

| Space | Inner Product | Usage |
|-------|-------------|-------|
| $\mathbb{R}^n$ | $\langle \mathbf{u}, \mathbf{v} \rangle = \sum_{i=1}^n u_i v_i = \mathbf{u}^T\mathbf{v}$ | Euclidean geometry, basic ML |
| $\mathbb{C}^n$ | $\langle \mathbf{u}, \mathbf{v} \rangle = \sum_{i=1}^n \overline{u_i} v_i = \mathbf{u}^*\mathbf{v}$ | Quantum computing |
| $L^2([a,b])$ | $\langle f, g \rangle = \int_a^b \overline{f(x)} g(x)\, dx$ | Fourier analysis, signal processing |
| $\mathbb{R}^{m \times n}$ | $\langle A, B \rangle = \operatorname{tr}(A^T B)$ | Matrix completion, Frobenius norm |

**Example 2.6.1 (Verifying inner product axioms).** Define $\langle \mathbf{u}, \mathbf{v} \rangle = 2u_1v_1 + 3u_2v_2$ on $\mathbb{R}^2$. Let $\mathbf{u} = (1, 2)^T$, $\mathbf{v} = (3, -1)^T$, $\mathbf{w} = (0, 4)^T$.

- **Symmetry:** $\langle \mathbf{u}, \mathbf{v} \rangle = 2(1)(3) + 3(2)(-1) = 0$ and $\langle \mathbf{v}, \mathbf{u} \rangle = 2(3)(1) + 3(-1)(2) = 0$. $\checkmark$
- **Linearity:** $\langle \mathbf{u} + \mathbf{w}, \mathbf{v} \rangle = 2(1)(3) + 3(6)(-1) = -12$ and $\langle \mathbf{u}, \mathbf{v} \rangle + \langle \mathbf{w}, \mathbf{v} \rangle = 0 + (2(0)(3) + 3(4)(-1)) = -12$. $\checkmark$
- **Positive definiteness:** $\langle \mathbf{u}, \mathbf{u} \rangle = 2(1)^2 + 3(2)^2 = 14 > 0$. Equals zero only when $\mathbf{u} = \mathbf{0}$ (since coefficients $2, 3 > 0$). $\checkmark$

This is a valid **weighted inner product**, common in ML when features have different scales.

*ML connection:* **Kernel functions** define inner products in high-dimensional feature spaces *without* explicitly computing the feature map. The kernel $k(\mathbf{x}, \mathbf{y}) = \langle \phi(\mathbf{x}), \phi(\mathbf{y}) \rangle_{\mathcal{H}}$ turns a non-linear classification problem into a linear one in a reproducing kernel Hilbert space (RKHS). The Gaussian RBF kernel $k(\mathbf{x}, \mathbf{y}) = \exp(-\|\mathbf{x} - \mathbf{y}\|^2 / 2\sigma^2)$ implicitly maps into an infinite-dimensional inner product space.

---

## 2. The Norm Induced by an Inner Product

**Definition 2.8.2 (Induced Norm).** Given an inner product space $(V, \langle \cdot, \cdot \rangle)$, the *induced norm* is:

$$\|\mathbf{v}\| = \sqrt{\langle \mathbf{v}, \mathbf{v} \rangle}$$

**Theorem 2.8.1.** The induced norm is indeed a norm (satisfies Definition 2.1.5).

*Proof.* Positive definiteness and homogeneity follow directly from the inner product axioms. The triangle inequality follows from Cauchy-Schwarz (Theorem 2.8.2 below). $\square$

**Example 2.6.2 (Computing the induced norm).** In $\mathbb{R}^3$ with the standard inner product, let $\mathbf{v} = (3, -4, 0)^T$.

$$\|\mathbf{v}\| = \sqrt{\langle \mathbf{v}, \mathbf{v} \rangle} = \sqrt{3^2 + (-4)^2 + 0^2} = \sqrt{9 + 16} = \sqrt{25} = 5$$

With the weighted inner product $\langle \mathbf{u}, \mathbf{v} \rangle = 2u_1v_1 + u_2v_2 + 3u_3v_3$, the same vector has a different norm:

$$\|\mathbf{v}\|_w = \sqrt{2(9) + 1(16) + 3(0)} = \sqrt{34} \approx 5.83$$

Different inner products induce different geometries on the same space.

**Proposition 2.8.1 (Parallelogram Law).** A norm $\|\cdot\|$ arises from an inner product if and only if it satisfies the *parallelogram law*:

$$\|\mathbf{u} + \mathbf{v}\|^2 + \|\mathbf{u} - \mathbf{v}\|^2 = 2\|\mathbf{u}\|^2 + 2\|\mathbf{v}\|^2$$

*Proof ($\Rightarrow$).* Expand each term using $\|\mathbf{w}\|^2 = \langle \mathbf{w}, \mathbf{w} \rangle$:

$$\|\mathbf{u} + \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + 2\operatorname{Re}\langle \mathbf{u}, \mathbf{v}\rangle + \|\mathbf{v}\|^2$$
$$\|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 - 2\operatorname{Re}\langle \mathbf{u}, \mathbf{v}\rangle + \|\mathbf{v}\|^2$$

Adding these yields $2\|\mathbf{u}\|^2 + 2\|\mathbf{v}\|^2$. The converse follows from the *polarization identity* which recovers the inner product from the norm. $\square$

**Example 2.6.3 (Parallelogram law verification).** Let $\mathbf{u} = (1, 2)^T$ and $\mathbf{v} = (3, 0)^T$ in $\mathbb{R}^2$ with the standard inner product.

- $\|\mathbf{u} + \mathbf{v}\|^2 = \|(4, 2)\|^2 = 16 + 4 = 20$
- $\|\mathbf{u} - \mathbf{v}\|^2 = \|(-2, 2)\|^2 = 4 + 4 = 8$
- $2\|\mathbf{u}\|^2 + 2\|\mathbf{v}\|^2 = 2(1 + 4) + 2(9 + 0) = 10 + 18 = 28$

Check: $20 + 8 = 28 = 2(5) + 2(9)$. $\checkmark$

> **Consequence:** The $\ell^1$ and $\ell^\infty$ norms (from Definition 2.1.6) do *not* come from any inner product, since they violate the parallelogram law.

---

## 3. The Cauchy-Schwarz Inequality

**Theorem 2.8.2 (Cauchy-Schwarz Inequality).** For all $\mathbf{u}, \mathbf{v}$ in an inner product space:

$$|\langle \mathbf{u}, \mathbf{v} \rangle| \leq \|\mathbf{u}\| \cdot \|\mathbf{v}\|$$

with equality if and only if $\mathbf{u}$ and $\mathbf{v}$ are linearly dependent.

*Proof.* If $\mathbf{v} = \mathbf{0}$, both sides are zero. Assume $\mathbf{v} \neq \mathbf{0}$. For any $t \in \mathbb{F}$, positive definiteness gives:

$$0 \leq \|\mathbf{u} - t\mathbf{v}\|^2 = \langle \mathbf{u} - t\mathbf{v},\, \mathbf{u} - t\mathbf{v} \rangle = \|\mathbf{u}\|^2 - t\langle \mathbf{v}, \mathbf{u} \rangle - \bar{t}\langle \mathbf{u}, \mathbf{v} \rangle + |t|^2\|\mathbf{v}\|^2$$

Choose $t = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{v}\|^2}$ (the value that minimises the right side). Substituting:

$$0 \leq \|\mathbf{u}\|^2 - \frac{|\langle \mathbf{u}, \mathbf{v} \rangle|^2}{\|\mathbf{v}\|^2}$$

Rearranging: $|\langle \mathbf{u}, \mathbf{v} \rangle|^2 \leq \|\mathbf{u}\|^2 \|\mathbf{v}\|^2$. Taking square roots gives the result.

*Equality:* The inequality is tight iff $\|\mathbf{u} - t\mathbf{v}\|^2 = 0$, i.e., $\mathbf{u} = t\mathbf{v}$. $\square$

```
CAUCHY-SCHWARZ GEOMETRICALLY

    The projection of u onto v has length ≤ ‖u‖:

         u
        ╱|
       ╱ |
      ╱  |         |⟨u,v⟩|
     ╱   |         ─────── ≤ ‖u‖
    ╱θ___|______   ‖v‖
    ─────────── v
    ←─ proj ──→

    Equality ⟺ θ = 0 or π  ⟺  u ∥ v
```

**Example 2.6.4 (Cauchy-Schwarz with numbers).** Let $\mathbf{u} = (1, 2, 3)^T$ and $\mathbf{v} = (4, -1, 2)^T$ in $\mathbb{R}^3$.

- $\langle \mathbf{u}, \mathbf{v} \rangle = 1(4) + 2(-1) + 3(2) = 4 - 2 + 6 = 8$
- $\|\mathbf{u}\| = \sqrt{1 + 4 + 9} = \sqrt{14}$, $\quad \|\mathbf{v}\| = \sqrt{16 + 1 + 4} = \sqrt{21}$
- $\|\mathbf{u}\|\|\mathbf{v}\| = \sqrt{14 \cdot 21} = \sqrt{294} \approx 17.15$

Check: $|8| = 8 \leq 17.15$. $\checkmark$ Equality would require $\mathbf{u} = t\mathbf{v}$, which is not the case here.

**Example 2.6.5 (Triangle inequality verification).** With the same vectors, verify $\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$.

- $\mathbf{u} + \mathbf{v} = (5, 1, 5)^T$, $\quad \|\mathbf{u} + \mathbf{v}\| = \sqrt{25 + 1 + 25} = \sqrt{51} \approx 7.14$
- $\|\mathbf{u}\| + \|\mathbf{v}\| = \sqrt{14} + \sqrt{21} \approx 3.74 + 4.58 = 8.32$

Check: $7.14 \leq 8.32$. $\checkmark$ The gap reflects that $\mathbf{u}$ and $\mathbf{v}$ are not parallel.

*ML connection:* Cauchy-Schwarz guarantees that **cosine similarity** always lies in $[-1, 1]$:

$$-1 \leq \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{u}\|\|\mathbf{v}\|} \leq 1$$

This is why cosine similarity is a well-defined similarity measure for embeddings. In **attention mechanisms** (transformers), the softmax over scaled dot products $\frac{\mathbf{q}^T \mathbf{k}}{\sqrt{d_k}}$ relies on Cauchy-Schwarz to keep values bounded before exponentiation.

---

## 4. Orthogonality

**Definition 2.8.3 (Orthogonality).** Two vectors $\mathbf{u}, \mathbf{v} \in V$ are *orthogonal*, written $\mathbf{u} \perp \mathbf{v}$, if $\langle \mathbf{u}, \mathbf{v} \rangle = 0$.

**Definition 2.8.4 (Orthogonal Set).** A set $\{\mathbf{e}_1, \dots, \mathbf{e}_k\}$ is *orthogonal* if $\langle \mathbf{e}_i, \mathbf{e}_j \rangle = 0$ for all $i \neq j$. It is *orthonormal* if additionally $\|\mathbf{e}_i\| = 1$ for all $i$:

$$\langle \mathbf{e}_i, \mathbf{e}_j \rangle = \delta_{ij} = \begin{cases} 1 & i = j \\ 0 & i \neq j \end{cases}$$

**Example 2.6.6 (Orthonormal set verification).** Consider $\mathbf{e}_1 = \frac{1}{\sqrt{2}}(1, 1, 0)^T$ and $\mathbf{e}_2 = \frac{1}{\sqrt{2}}(1, -1, 0)^T$ in $\mathbb{R}^3$.

- $\langle \mathbf{e}_1, \mathbf{e}_2 \rangle = \frac{1}{2}(1 \cdot 1 + 1 \cdot (-1) + 0 \cdot 0) = \frac{1}{2}(1 - 1) = 0$ $\quad$ (orthogonal $\checkmark$)
- $\|\mathbf{e}_1\| = \sqrt{\frac{1}{2}(1 + 1 + 0)} = 1$ $\quad$ (unit length $\checkmark$)
- $\|\mathbf{e}_2\| = \sqrt{\frac{1}{2}(1 + 1 + 0)} = 1$ $\quad$ (unit length $\checkmark$)

So $\{\mathbf{e}_1, \mathbf{e}_2\}$ is an orthonormal set. It is not a basis for $\mathbb{R}^3$ (only spans a 2D subspace).

**Theorem 2.8.3.** An orthogonal set of nonzero vectors is linearly independent.

*Proof.* Suppose $\sum_{i=1}^k c_i \mathbf{e}_i = \mathbf{0}$. Taking the inner product with $\mathbf{e}_j$:

$$0 = \Big\langle \sum_{i=1}^k c_i \mathbf{e}_i,\, \mathbf{e}_j \Big\rangle = \sum_{i=1}^k c_i \langle \mathbf{e}_i, \mathbf{e}_j \rangle = c_j \langle \mathbf{e}_j, \mathbf{e}_j \rangle = c_j \|\mathbf{e}_j\|^2$$

Since $\mathbf{e}_j \neq \mathbf{0}$, we have $c_j = 0$ for every $j$. $\square$

**Theorem 2.8.4 (Generalised Pythagoras).** If $\mathbf{u} \perp \mathbf{v}$, then $\|\mathbf{u} + \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2$.

More generally, for mutually orthogonal $\mathbf{v}_1, \dots, \mathbf{v}_k$:

$$\left\|\sum_{i=1}^k \mathbf{v}_i\right\|^2 = \sum_{i=1}^k \|\mathbf{v}_i\|^2$$

---

## 5. Gram-Schmidt Orthogonalisation

**Theorem 2.8.5 (Gram-Schmidt Process).** Let $\{\mathbf{v}_1, \dots, \mathbf{v}_k\}$ be a linearly independent set in an inner product space $V$. There exists an orthonormal set $\{\mathbf{e}_1, \dots, \mathbf{e}_k\}$ such that:

$$\operatorname{span}(\mathbf{e}_1, \dots, \mathbf{e}_j) = \operatorname{span}(\mathbf{v}_1, \dots, \mathbf{v}_j) \quad \text{for each } j = 1, \dots, k$$

*Proof (Constructive).* Define recursively:

$$\mathbf{u}_1 = \mathbf{v}_1, \qquad \mathbf{e}_1 = \frac{\mathbf{u}_1}{\|\mathbf{u}_1\|}$$

For $j = 2, \dots, k$:

$$\mathbf{u}_j = \mathbf{v}_j - \sum_{i=1}^{j-1} \langle \mathbf{v}_j, \mathbf{e}_i \rangle \, \mathbf{e}_i, \qquad \mathbf{e}_j = \frac{\mathbf{u}_j}{\|\mathbf{u}_j\|}$$

**Orthogonality:** For $m < j$, we verify $\langle \mathbf{u}_j, \mathbf{e}_m \rangle = 0$:

$$\langle \mathbf{u}_j, \mathbf{e}_m \rangle = \langle \mathbf{v}_j, \mathbf{e}_m \rangle - \sum_{i=1}^{j-1} \langle \mathbf{v}_j, \mathbf{e}_i \rangle \langle \mathbf{e}_i, \mathbf{e}_m \rangle = \langle \mathbf{v}_j, \mathbf{e}_m \rangle - \langle \mathbf{v}_j, \mathbf{e}_m \rangle = 0$$

**Well-defined:** $\mathbf{u}_j \neq \mathbf{0}$ because $\mathbf{v}_j \notin \operatorname{span}(\mathbf{v}_1, \dots, \mathbf{v}_{j-1}) = \operatorname{span}(\mathbf{e}_1, \dots, \mathbf{e}_{j-1})$ (linear independence).

**Span preservation:** By construction, $\mathbf{e}_j \in \operatorname{span}(\mathbf{v}_1, \dots, \mathbf{v}_j)$ and $\mathbf{v}_j \in \operatorname{span}(\mathbf{e}_1, \dots, \mathbf{e}_j)$, so the spans agree at each step. $\square$

**Example 2.6.7 (Gram-Schmidt on two vectors in $\mathbb{R}^3$).** Orthogonalise $\mathbf{v}_1 = (1, 1, 0)^T$ and $\mathbf{v}_2 = (1, 0, 1)^T$.

**Step 1:** $\mathbf{u}_1 = \mathbf{v}_1 = (1, 1, 0)^T$, $\quad \|\mathbf{u}_1\| = \sqrt{2}$, $\quad \mathbf{e}_1 = \frac{1}{\sqrt{2}}(1, 1, 0)^T$

**Step 2:** Compute $\langle \mathbf{v}_2, \mathbf{e}_1 \rangle = \frac{1}{\sqrt{2}}(1 + 0 + 0) = \frac{1}{\sqrt{2}}$

$$\mathbf{u}_2 = \mathbf{v}_2 - \langle \mathbf{v}_2, \mathbf{e}_1 \rangle \mathbf{e}_1 = (1, 0, 1)^T - \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}}(1, 1, 0)^T = (1, 0, 1)^T - \tfrac{1}{2}(1, 1, 0)^T = \tfrac{1}{2}(1, -1, 2)^T$$

$\|\mathbf{u}_2\| = \frac{1}{2}\sqrt{1 + 1 + 4} = \frac{\sqrt{6}}{2}$, $\quad \mathbf{e}_2 = \frac{1}{\sqrt{6}}(1, -1, 2)^T$

**Verify:** $\langle \mathbf{e}_1, \mathbf{e}_2 \rangle = \frac{1}{\sqrt{12}}(1 - 1 + 0) = 0$ $\checkmark$

```
GRAM-SCHMIDT VISUALISED IN R³

  Step 1: e₁ = v₁/‖v₁‖

              v₂
             ╱
            ╱
           ╱
    ──────●──────── e₁
          ↑
       proj of v₂ onto e₁

  Step 2: u₂ = v₂ − ⟨v₂,e₁⟩e₁    (subtract projection)

              v₂       e₂ = u₂/‖u₂‖
             ╱         ↑
            ╱          │ u₂
           ╱           │
    ──────●──────── e₁

  Step 3: u₃ = v₃ − ⟨v₃,e₁⟩e₁ − ⟨v₃,e₂⟩e₂

    Result: {e₁, e₂, e₃} orthonormal basis
```

*ML connection:* Gram-Schmidt is the engine behind **QR decomposition** (see [Matrix Decompositions](matrix-decompositions.md)), which is used in numerical linear algebra throughout ML. It also appears in **Natural Language Processing** when constructing orthogonal concept directions in embedding spaces (e.g., removing gender bias by projecting onto the orthogonal complement of a gender direction).

---

## 6. Orthogonal Complements

**Definition 2.8.5 (Orthogonal Complement).** Let $W$ be a subspace of an inner product space $V$. The *orthogonal complement* of $W$ is:

$$W^\perp = \{\mathbf{v} \in V : \langle \mathbf{v}, \mathbf{w} \rangle = 0 \text{ for all } \mathbf{w} \in W\}$$

**Example 2.6.8 (Orthogonal complement in $\mathbb{R}^3$).** Let $W = \operatorname{span}\{(1, 0, 1)^T, (0, 1, 1)^T\}$ in $\mathbb{R}^3$. Find $W^\perp$.

A vector $\mathbf{v} = (a, b, c)^T \in W^\perp$ must satisfy $\langle \mathbf{v}, \mathbf{w} \rangle = 0$ for all $\mathbf{w} \in W$:

- $\langle \mathbf{v}, (1, 0, 1)^T \rangle = a + c = 0$
- $\langle \mathbf{v}, (0, 1, 1)^T \rangle = b + c = 0$

From these: $a = -c$ and $b = -c$. Setting $c = 1$: $W^\perp = \operatorname{span}\{(-1, -1, 1)^T\}$.

**Check:** $\dim W + \dim W^\perp = 2 + 1 = 3 = \dim \mathbb{R}^3$. $\checkmark$

**Theorem 2.8.6 (Orthogonal Decomposition).** Let $W$ be a finite-dimensional subspace of an inner product space $V$. Then:

$$V = W \oplus W^\perp$$

That is, every $\mathbf{v} \in V$ can be written uniquely as $\mathbf{v} = \mathbf{w} + \mathbf{w}^\perp$ where $\mathbf{w} \in W$ and $\mathbf{w}^\perp \in W^\perp$.

*Proof.* Let $\{\mathbf{e}_1, \dots, \mathbf{e}_k\}$ be an orthonormal basis for $W$ (obtained via Gram-Schmidt). Define:

$$\mathbf{w} = \sum_{i=1}^k \langle \mathbf{v}, \mathbf{e}_i \rangle \, \mathbf{e}_i, \qquad \mathbf{w}^\perp = \mathbf{v} - \mathbf{w}$$

Then $\mathbf{w} \in W$ by construction. For any basis element $\mathbf{e}_j$:

$$\langle \mathbf{w}^\perp, \mathbf{e}_j \rangle = \langle \mathbf{v}, \mathbf{e}_j \rangle - \sum_{i=1}^k \langle \mathbf{v}, \mathbf{e}_i \rangle \langle \mathbf{e}_i, \mathbf{e}_j \rangle = \langle \mathbf{v}, \mathbf{e}_j \rangle - \langle \mathbf{v}, \mathbf{e}_j \rangle = 0$$

So $\mathbf{w}^\perp \in W^\perp$, giving existence. Uniqueness: if $\mathbf{v} = \mathbf{w}_1 + \mathbf{w}_1^\perp = \mathbf{w}_2 + \mathbf{w}_2^\perp$, then $\mathbf{w}_1 - \mathbf{w}_2 = \mathbf{w}_2^\perp - \mathbf{w}_1^\perp \in W \cap W^\perp$. But $\langle \mathbf{x}, \mathbf{x} \rangle = 0$ for $\mathbf{x} \in W \cap W^\perp$, so $\mathbf{x} = \mathbf{0}$. $\square$

**Corollary 2.8.1.** For a finite-dimensional inner product space: $\dim W + \dim W^\perp = \dim V$ and $(W^\perp)^\perp = W$.

```
ORTHOGONAL COMPLEMENT IN R³

    W = span of the plane
    W⊥ = normal line to the plane

            W⊥ (normal)
            ↑
            │
            │
    ────────┼──────── W (plane)
           ╱│╲
          ╱ │ ╲
         ╱  │  ╲
        ╱   │   ╲

    Every v ∈ R³ decomposes uniquely:
    v = w + w⊥,   w ∈ W,  w⊥ ∈ W⊥
```

---

## 7. Orthogonal Projections

**Definition 2.8.6 (Orthogonal Projection).** The *orthogonal projection* of $\mathbf{v}$ onto a subspace $W$ is the component $\mathbf{w}$ from the decomposition $\mathbf{v} = \mathbf{w} + \mathbf{w}^\perp$:

$$\operatorname{proj}_W(\mathbf{v}) = \sum_{i=1}^k \langle \mathbf{v}, \mathbf{e}_i \rangle \, \mathbf{e}_i$$

where $\{\mathbf{e}_1, \dots, \mathbf{e}_k\}$ is an orthonormal basis for $W$.

**Example 2.6.9 (Orthogonal projection onto a subspace).** Project $\mathbf{v} = (1, 2, 3)^T$ onto $W = \operatorname{span}\{(1, 0, 0)^T, (0, 1, 0)^T\}$ (the $xy$-plane). The standard basis vectors $\mathbf{e}_1, \mathbf{e}_2$ are already orthonormal, so:

$$\operatorname{proj}_W(\mathbf{v}) = \langle \mathbf{v}, \mathbf{e}_1 \rangle \mathbf{e}_1 + \langle \mathbf{v}, \mathbf{e}_2 \rangle \mathbf{e}_2 = 1 \cdot (1,0,0)^T + 2 \cdot (0,1,0)^T = (1, 2, 0)^T$$

The residual is $\mathbf{v} - \operatorname{proj}_W(\mathbf{v}) = (0, 0, 3)^T$, which is orthogonal to $W$. $\checkmark$

**Example 2.6.10 (Projection onto a line).** Project $\mathbf{v} = (3, 4)^T$ onto $W = \operatorname{span}\{(1, 1)^T\}$. Set $\mathbf{e}_1 = \frac{1}{\sqrt{2}}(1, 1)^T$.

$$\operatorname{proj}_W(\mathbf{v}) = \langle \mathbf{v}, \mathbf{e}_1 \rangle \mathbf{e}_1 = \frac{3 + 4}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}}(1, 1)^T = \frac{7}{2}(1, 1)^T = (3.5,\, 3.5)^T$$

The residual $\mathbf{v} - \operatorname{proj}_W(\mathbf{v}) = (-0.5, 0.5)^T$ satisfies $\langle (-0.5, 0.5), (1, 1) \rangle = 0$. $\checkmark$

**Theorem 2.8.7 (Best Approximation).** The projection $\operatorname{proj}_W(\mathbf{v})$ is the *closest point in $W$ to $\mathbf{v}$*:

$$\|\mathbf{v} - \operatorname{proj}_W(\mathbf{v})\| \leq \|\mathbf{v} - \mathbf{w}\| \quad \text{for all } \mathbf{w} \in W$$

with equality iff $\mathbf{w} = \operatorname{proj}_W(\mathbf{v})$.

*Proof.* Let $\hat{\mathbf{v}} = \operatorname{proj}_W(\mathbf{v})$. For any $\mathbf{w} \in W$:

$$\|\mathbf{v} - \mathbf{w}\|^2 = \|\mathbf{v} - \hat{\mathbf{v}} + \hat{\mathbf{v}} - \mathbf{w}\|^2$$

Since $(\mathbf{v} - \hat{\mathbf{v}}) \in W^\perp$ and $(\hat{\mathbf{v}} - \mathbf{w}) \in W$, they are orthogonal, so by Pythagoras:

$$= \|\mathbf{v} - \hat{\mathbf{v}}\|^2 + \|\hat{\mathbf{v}} - \mathbf{w}\|^2 \geq \|\mathbf{v} - \hat{\mathbf{v}}\|^2$$

Equality holds iff $\hat{\mathbf{v}} = \mathbf{w}$. $\square$

### 7.1 Projection Matrices

When $W = \operatorname{col}(A)$ for a matrix $A$ with linearly independent columns, the projection matrix is:

$$P = A(A^T A)^{-1} A^T$$

**Properties of projection matrices:**

| Property | Meaning |
|----------|---------|
| $P^2 = P$ | Idempotent: projecting twice gives the same result |
| $P^T = P$ | Symmetric: projection is self-adjoint |
| $\operatorname{rank}(P) = \dim W$ | Rank equals dimension of target subspace |
| $I - P$ | Projects onto $W^\perp$ |

*ML connection:* **Attention mechanisms** in transformers compute a projection. The attention output for a query $\mathbf{q}$ is a weighted combination of values, where the weights come from inner products $\mathbf{q}^T\mathbf{k}_i$. This is geometrically a soft projection of the query onto the subspace spanned by keys, with values as the coordinate representation.

---

## 8. Least Squares via Projection

**Problem.** Given $A \in \mathbb{R}^{m \times n}$ with $m > n$ (overdetermined system), find $\hat{\mathbf{x}}$ that minimises $\|A\mathbf{x} - \mathbf{b}\|^2$.

**Theorem 2.8.8 (Normal Equations).** The least squares solution satisfies:

$$A^T A \hat{\mathbf{x}} = A^T \mathbf{b}$$

If $A$ has linearly independent columns, the unique solution is:

$$\hat{\mathbf{x}} = (A^T A)^{-1} A^T \mathbf{b}$$

*Proof.* The residual $\mathbf{r} = \mathbf{b} - A\hat{\mathbf{x}}$ is minimised when $A\hat{\mathbf{x}} = \operatorname{proj}_{\operatorname{col}(A)}(\mathbf{b})$, i.e., when $\mathbf{r} \perp \operatorname{col}(A)$. This means:

$$A^T(\mathbf{b} - A\hat{\mathbf{x}}) = \mathbf{0} \implies A^T A \hat{\mathbf{x}} = A^T \mathbf{b}$$

When $A$ has independent columns, $A^T A$ is invertible (positive definite), giving the unique solution. $\square$

```
LEAST SQUARES GEOMETRY

    b ∈ Rᵐ
    │╲
    │  ╲ residual r = b − Ax̂
    │    ╲           (r ⊥ col(A))
    │     ╲
    │──────● Ax̂ = proj_{col(A)}(b)
    │      │
    │  col(A)
    │

    We can't solve Ax = b exactly (b ∉ col(A)),
    so we find the closest point in col(A) to b.
```

*ML connection:* **Linear regression** is exactly this problem. Given features $X \in \mathbb{R}^{m \times n}$ and targets $\mathbf{y} \in \mathbb{R}^m$, the ordinary least squares (OLS) estimate is $\hat{\boldsymbol{\beta}} = (X^TX)^{-1}X^T\mathbf{y}$. Adding $L^2$ regularisation gives **ridge regression**: $\hat{\boldsymbol{\beta}} = (X^TX + \lambda I)^{-1}X^T\mathbf{y}$, which ensures $(X^TX + \lambda I)$ is always invertible, even when $X^TX$ is singular or ill-conditioned.

---

## 9. Adjoint Operators

**Definition 2.8.7 (Adjoint).** Let $T: V \to W$ be a linear map between inner product spaces. The *adjoint* of $T$ is the unique linear map $T^*: W \to V$ satisfying:

$$\langle T\mathbf{v}, \mathbf{w} \rangle_W = \langle \mathbf{v}, T^*\mathbf{w} \rangle_V \quad \text{for all } \mathbf{v} \in V, \mathbf{w} \in W$$

**Theorem 2.8.9 (Existence of the Adjoint).** If $V$ and $W$ are finite-dimensional inner product spaces, then $T^*$ exists and is unique.

*Proof.* Fix $\mathbf{w} \in W$. The map $\mathbf{v} \mapsto \langle T\mathbf{v}, \mathbf{w} \rangle$ is a linear functional on $V$. By the Riesz representation theorem, there exists a unique $\mathbf{z} \in V$ such that $\langle T\mathbf{v}, \mathbf{w} \rangle = \langle \mathbf{v}, \mathbf{z} \rangle$ for all $\mathbf{v}$. Define $T^*\mathbf{w} = \mathbf{z}$. Linearity of $T^*$ follows from uniqueness and linearity of the inner product. $\square$

**Theorem 2.8.10 (Matrix Representation).** If $T$ is represented by the matrix $A$ with respect to orthonormal bases, then $T^*$ is represented by:

- $A^T$ (transpose) when $\mathbb{F} = \mathbb{R}$
- $A^* = \overline{A}^T$ (conjugate transpose) when $\mathbb{F} = \mathbb{C}$

### 9.1 Self-Adjoint and Unitary Operators

**Definition 2.8.8 (Self-Adjoint / Hermitian).** An operator $T$ is *self-adjoint* if $T = T^*$.

In matrix form: $A = A^T$ (real symmetric) or $A = A^*$ (Hermitian).

**Definition 2.8.9 (Unitary / Orthogonal).** An operator $U$ is *unitary* if $U^*U = UU^* = I$, i.e., $U^{-1} = U^*$.

When $\mathbb{F} = \mathbb{R}$, we say *orthogonal* instead of unitary: $Q^TQ = QQ^T = I$.

| Property | Self-adjoint ($A = A^*$) | Unitary ($U^*U = I$) |
|----------|--------------------------|----------------------|
| Eigenvalues | All real | All on unit circle ($|\lambda| = 1$) |
| Eigenvectors | Orthogonal (Spectral Theorem) | Orthogonal |
| Preserves | Inner product structure | Norms and inner products |
| ML role | Covariance matrices, Hessians | Rotations, orthogonal initialisations |

**Theorem 2.8.11 (Properties of the Adjoint).**

1. $(S + T)^* = S^* + T^*$
2. $(\alpha T)^* = \bar{\alpha} T^*$
3. $(ST)^* = T^* S^*$
4. $(T^*)^* = T$
5. $\ker(T^*) = (\operatorname{im} T)^\perp$
6. $\operatorname{im}(T^*) = (\ker T)^\perp$

*ML connection:* Property 5 is the **fundamental theorem of linear algebra** in inner product form. In deep learning, the transpose (adjoint) appears during **backpropagation**: if the forward pass through a linear layer computes $\mathbf{y} = W\mathbf{x}$, the backward pass propagates gradients as $\frac{\partial L}{\partial \mathbf{x}} = W^T \frac{\partial L}{\partial \mathbf{y}}$. The adjoint $W^T$ maps gradients backward through the network.

---

## 10. Bessel's Inequality and Parseval's Identity

**Theorem 2.8.12 (Bessel's Inequality).** Let $\{\mathbf{e}_1, \dots, \mathbf{e}_k\}$ be an orthonormal set in $V$. For any $\mathbf{v} \in V$:

$$\sum_{i=1}^k |\langle \mathbf{v}, \mathbf{e}_i \rangle|^2 \leq \|\mathbf{v}\|^2$$

*Proof.* Let $\hat{\mathbf{v}} = \sum_{i=1}^k \langle \mathbf{v}, \mathbf{e}_i \rangle \mathbf{e}_i$. Then $\mathbf{v} - \hat{\mathbf{v}} \perp \hat{\mathbf{v}}$, so:

$$\|\mathbf{v}\|^2 = \|\hat{\mathbf{v}}\|^2 + \|\mathbf{v} - \hat{\mathbf{v}}\|^2 \geq \|\hat{\mathbf{v}}\|^2 = \sum_{i=1}^k |\langle \mathbf{v}, \mathbf{e}_i \rangle|^2 \quad \square$$

**Example 2.6.11 (Bessel's inequality and Fourier coefficients).** Let $\mathbf{v} = (3, 4, 1)^T$ in $\mathbb{R}^3$ and take the orthonormal set $\{\mathbf{e}_1, \mathbf{e}_2\}$ where $\mathbf{e}_1 = (1, 0, 0)^T$ and $\mathbf{e}_2 = (0, 1, 0)^T$ (not a full basis).

The Fourier coefficients are $\langle \mathbf{v}, \mathbf{e}_1 \rangle = 3$ and $\langle \mathbf{v}, \mathbf{e}_2 \rangle = 4$.

$$\sum_{i=1}^{2} |\langle \mathbf{v}, \mathbf{e}_i \rangle|^2 = 9 + 16 = 25 \leq 26 = 9 + 16 + 1 = \|\mathbf{v}\|^2 \quad \checkmark$$

The gap $\|\mathbf{v}\|^2 - 25 = 1 = \|\mathbf{v} - \operatorname{proj}(\mathbf{v})\|^2$ measures the energy in the component $(0, 0, 1)^T$ not captured by $\{\mathbf{e}_1, \mathbf{e}_2\}$.

**Corollary 2.8.2 (Parseval's Identity).** If $\{\mathbf{e}_1, \dots, \mathbf{e}_n\}$ is an orthonormal *basis*, equality holds:

$$\|\mathbf{v}\|^2 = \sum_{i=1}^n |\langle \mathbf{v}, \mathbf{e}_i \rangle|^2$$

**Example 2.6.12 (Parseval's identity verification).** Let $\mathbf{v} = (3, 4, 1)^T$ and use the full standard orthonormal basis $\{\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3\}$ of $\mathbb{R}^3$.

$$\sum_{i=1}^{3} |\langle \mathbf{v}, \mathbf{e}_i \rangle|^2 = 3^2 + 4^2 + 1^2 = 9 + 16 + 1 = 26 = \|\mathbf{v}\|^2 \quad \checkmark$$

With a full orthonormal basis, Bessel's inequality becomes an equality: no energy is lost.

*ML connection:* Parseval's identity underlies **energy preservation** in transforms. When you compute the Fourier transform of a signal, the total energy (sum of squared magnitudes) is preserved. In dimensionality reduction, Bessel's inequality quantifies the **information loss** when projecting onto a lower-dimensional subspace: the squared coefficients you keep must be less than or equal to the total energy.

---

## Exercises

### Foundations (★)

**Exercise 2.8.1.** Verify that $\langle f, g \rangle = \int_0^1 f(x)g(x)\,dx$ defines an inner product on $C([0,1])$. Compute $\langle x, x^2 \rangle$ and $\|x\|$.

**Exercise 2.8.2.** Let $\mathbf{u} = (1, 2, 3)^T$ and $\mathbf{v} = (4, -1, 2)^T$ in $\mathbb{R}^3$. Compute $\langle \mathbf{u}, \mathbf{v} \rangle$, $\|\mathbf{u}\|$, $\|\mathbf{v}\|$, and verify Cauchy-Schwarz.

**Exercise 2.8.3.** Show that the $\ell^1$ norm on $\mathbb{R}^2$ does not satisfy the parallelogram law, and hence does not arise from any inner product.

### Core Theory (★★)

**Exercise 2.8.4.** Apply Gram-Schmidt to $\{(1, 1, 0)^T, (1, 0, 1)^T, (0, 1, 1)^T\}$ in $\mathbb{R}^3$ with the standard inner product.

**Exercise 2.8.5.** Let $W = \operatorname{span}\{(1, 1, 1, 1)^T, (1, -1, 0, 0)^T\}$ in $\mathbb{R}^4$. Find $\operatorname{proj}_W(\mathbf{b})$ where $\mathbf{b} = (3, 1, 5, 7)^T$. Verify $\mathbf{b} - \operatorname{proj}_W(\mathbf{b}) \in W^\perp$.

**Exercise 2.8.6.** Prove that if $P$ is an orthogonal projection matrix, then $I - P$ is also an orthogonal projection matrix. What does it project onto?

**Exercise 2.8.7.** Find the least squares line $y = \beta_0 + \beta_1 x$ for the data points $(0, 1), (1, 3), (2, 4), (3, 8)$ by setting up and solving the normal equations.

### Advanced / ML Applications (★★★)

**Exercise 2.8.8.** Let $k(\mathbf{x}, \mathbf{y}) = (\mathbf{x}^T \mathbf{y} + 1)^2$ for $\mathbf{x}, \mathbf{y} \in \mathbb{R}^2$. Find the explicit feature map $\phi: \mathbb{R}^2 \to \mathbb{R}^d$ such that $k(\mathbf{x}, \mathbf{y}) = \langle \phi(\mathbf{x}), \phi(\mathbf{y}) \rangle$. What is $d$?

**Exercise 2.8.9.** In a transformer with $d_k = 64$, the attention weight for query $\mathbf{q}$ and key $\mathbf{k}$ is $\operatorname{softmax}(\mathbf{q}^T\mathbf{k} / \sqrt{d_k})$. Use Cauchy-Schwarz to derive an upper bound on the unscaled dot product $\mathbf{q}^T\mathbf{k}$ when $\|\mathbf{q}\| = \|\mathbf{k}\| = \sqrt{d_k}$. Why is the $\sqrt{d_k}$ scaling necessary?

**Exercise 2.8.10.** Prove that the ridge regression estimator $\hat{\boldsymbol{\beta}}_\lambda = (X^TX + \lambda I)^{-1}X^T\mathbf{y}$ can be interpreted as the orthogonal projection of $\mathbf{y}$ onto $\operatorname{col}(X)$ in a *modified* inner product space. What is the modification?

---

## Related Topics

- **Previous:** [Eigenvalues & Eigenvectors](eigenvalues.md) --- spectral theorem uses inner products
- **Next:** [Matrix Decompositions](matrix-decompositions.md) --- QR via Gram-Schmidt, SVD uses inner product structure
- **Applications:** [Tensors](tensors.md) --- inner products on tensor spaces
- **Probability:** Covariance as an inner product on random variables
- **Quantum:** Hilbert spaces and bra-ket notation ($\langle \psi | \phi \rangle$) are complex inner products

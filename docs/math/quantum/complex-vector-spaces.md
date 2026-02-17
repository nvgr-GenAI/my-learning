# 8.1 Complex Vector Spaces and Hilbert Spaces

## Prerequisites

- **Linear Algebra**: Vector spaces, linear independence, inner products, eigenvalues/eigenvectors
- **Complex Numbers**: $\mathbb{C}$, complex conjugation, polar form $re^{i\theta}$
- **Basic Analysis**: Sequences, limits, completeness (for infinite-dimensional spaces)

**Recommended Reading**: Chapter 2 (Linear Algebra Foundations), Chapter 3 (Complex Numbers)

---

## Overview

Classical machine learning operates in real vector spaces ($\mathbb{R}^n$), but quantum computing requires the richer structure of **complex vector spaces**. The phase information encoded in complex numbers is essential for interference effects that power quantum algorithms.

**Why Complex Numbers Matter:**
- **Quantum states** are vectors in $\mathbb{C}^n$ (not $\mathbb{R}^n$)
- **Interference** requires phase: $e^{i\theta}$ enables constructive/destructive interference
- **Unitary evolution** preserves complex inner products

```
Classical ML              Quantum Computing
───────────────────────   ──────────────────────────
ℝⁿ (real vectors)    →    ℂⁿ (complex vectors)
Dot product          →    Hermitian inner product
Orthogonal matrices  →    Unitary matrices
Feature vectors      →    Quantum states (kets)
```

---

## 8.1.1 Complex Vector Spaces

### Definition 8.1.1: Complex Vector Space

A **complex vector space** $V$ over $\mathbb{C}$ is a set with:
1. **Vector addition**: $u + v \in V$ for all $u, v \in V$
2. **Scalar multiplication**: $\alpha v \in V$ for all $\alpha \in \mathbb{C}, v \in V$
3. Satisfies 8 axioms (associativity, commutativity, distributivity, identity, inverse)

**Example 8.1.1**: $\mathbb{C}^n$ with componentwise operations:
$$v = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix}, \quad v_i \in \mathbb{C}$$

$$v + w = \begin{pmatrix} v_1 + w_1 \\ v_2 + w_2 \\ \vdots \\ v_n + w_n \end{pmatrix}, \quad \alpha v = \begin{pmatrix} \alpha v_1 \\ \alpha v_2 \\ \vdots \\ \alpha v_n \end{pmatrix}$$

**Example 8.1.1a** (Concrete vector addition and scalar multiplication):
Let $|u\rangle = \begin{pmatrix} 1+i \\ 2 \end{pmatrix}$ and $|v\rangle = \begin{pmatrix} 3-i \\ i \end{pmatrix}$ in $\mathbb{C}^2$. Then:

$$|u\rangle + |v\rangle = \begin{pmatrix} (1+i)+(3-i) \\ 2 + i \end{pmatrix} = \begin{pmatrix} 4 \\ 2+i \end{pmatrix}$$

$$\alpha = (1+i): \quad \alpha|u\rangle = (1+i)\begin{pmatrix} 1+i \\ 2 \end{pmatrix} = \begin{pmatrix} (1+i)(1+i) \\ (1+i)(2) \end{pmatrix} = \begin{pmatrix} 2i \\ 2+2i \end{pmatrix}$$

since $(1+i)(1+i) = 1 + 2i + i^2 = 1 + 2i - 1 = 2i$.

### Definition 8.1.2: Hermitian Inner Product

For $u, v \in \mathbb{C}^n$, the **Hermitian inner product** is:
$$\langle u | v \rangle = \sum_{i=1}^n \bar{u}_i v_i$$

where $\bar{u}_i$ is the complex conjugate of $u_i$.

**Properties:**
1. **Conjugate symmetry**: $\langle u | v \rangle = \overline{\langle v | u \rangle}$
2. **Linearity in 2nd argument**: $\langle u | \alpha v + \beta w \rangle = \alpha \langle u | v \rangle + \beta \langle u | w \rangle$
3. **Positive definiteness**: $\langle v | v \rangle \geq 0$, with equality iff $v = 0$

**Example 8.1.2**: In $\mathbb{C}^2$:
$$u = \begin{pmatrix} 1+i \\ 2 \end{pmatrix}, \quad v = \begin{pmatrix} i \\ 1-i \end{pmatrix}$$

$$\langle u | v \rangle = (1-i)(i) + (2)(1-i) = (1-i)i + 2 - 2i = i + 1 + 2 - 2i = 3 - i$$

**Norm**: $\|v\| = \sqrt{\langle v | v \rangle} = \sqrt{\sum_i |v_i|^2}$

**Example 8.1.2a** (Norm of a complex vector):
For $|v\rangle = \begin{pmatrix} 1+i \\ 2-i \end{pmatrix}$:

$$\|v\| = \sqrt{|1+i|^2 + |2-i|^2} = \sqrt{(1^2+1^2) + (2^2+(-1)^2)} = \sqrt{2 + 5} = \sqrt{7}$$

The normalized state is $|\hat{v}\rangle = \frac{1}{\sqrt{7}}\begin{pmatrix} 1+i \\ 2-i \end{pmatrix}$, satisfying $\langle \hat{v} | \hat{v} \rangle = 1$.

**Example 8.1.2b** (Orthogonality check):
Are $|u\rangle = \begin{pmatrix} 1 \\ i \end{pmatrix}$ and $|v\rangle = \begin{pmatrix} i \\ 1 \end{pmatrix}$ orthogonal?

$$\langle u | v \rangle = \bar{u}_1 v_1 + \bar{u}_2 v_2 = (1)(i) + (-i)(1) = i - i = 0 \checkmark$$

Yes, $\langle u | v \rangle = 0$, so $|u\rangle$ and $|v\rangle$ are orthogonal. Note the complex conjugation of $u_2 = i$ to $\bar{u}_2 = -i$ is essential.

> **ML/Quantum Connection**: In quantum computing, $|\langle u | v \rangle|^2$ is the **probability amplitude** of transitioning from state $|v\rangle$ to $|u\rangle$. The Hermitian inner product ensures probabilities are real and non-negative.

---

## 8.1.2 Hilbert Spaces

### Definition 8.1.3: Hilbert Space

A **Hilbert space** $\mathcal{H}$ is a complete complex inner product space. "Complete" means every Cauchy sequence converges to an element in $\mathcal{H}$.

**Finite-Dimensional Hilbert Spaces:**
- $\mathbb{C}^n$ with Hermitian inner product
- Automatically complete (all finite-dimensional spaces are)
- **Quantum states** for $n$-level systems (qudits)

**Infinite-Dimensional Hilbert Spaces:**
- $L^2(\mathbb{R})$ (square-integrable functions): $\langle f | g \rangle = \int_{-\infty}^\infty \overline{f(x)} g(x) dx$
- **Quantum states** for continuous variables (position, momentum)
- Requires functional analysis for rigor

**Theorem 8.1.1**: Every finite-dimensional inner product space over $\mathbb{C}$ is a Hilbert space.

*Proof sketch*: Finite-dimensional $\Rightarrow$ bounded $\Rightarrow$ Cauchy sequences converge. $\square$

**Example 8.1.3**: The space of 2-qubit states is $\mathbb{C}^4$:
$$|\psi\rangle = \alpha_{00}|00\rangle + \alpha_{01}|01\rangle + \alpha_{10}|10\rangle + \alpha_{11}|11\rangle$$
where $|\alpha_{00}|^2 + |\alpha_{01}|^2 + |\alpha_{10}|^2 + |\alpha_{11}|^2 = 1$ (normalization).

**Example 8.1.3a** (Normalization verification in $\mathbb{C}^2$):
Is $|\psi\rangle = \begin{pmatrix} \frac{1+i}{2} \\ \frac{1}{2} \end{pmatrix}$ a valid quantum state?

$$\langle \psi | \psi \rangle = \left|\frac{1+i}{2}\right|^2 + \left|\frac{1}{2}\right|^2 = \frac{|1+i|^2}{4} + \frac{1}{4} = \frac{2}{4} + \frac{1}{4} = \frac{3}{4} \neq 1$$

Not normalized. The normalized version is $|\hat{\psi}\rangle = \frac{2}{\sqrt{3}}|\psi\rangle = \begin{pmatrix} \frac{1+i}{\sqrt{3}} \\ \frac{1}{\sqrt{3}} \end{pmatrix}$.

> **ML/Quantum Connection**: **Every quantum system lives in a Hilbert space**. Finite-dimensional Hilbert spaces model discrete systems (qubits, spin), while infinite-dimensional spaces model continuous systems (photon fields). The mathematical structure ensures probabilistic interpretation.

---

## 8.1.3 Dirac Notation (Bra-Ket Formalism)

### Definition 8.1.4: Kets and Bras

Paul Dirac introduced a notation that elegantly separates vectors from dual vectors:

**Ket** (column vector): $|v\rangle \in \mathcal{H}$
$$|v\rangle = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix}$$

**Bra** (row vector, dual): $\langle v| \in \mathcal{H}^*$
$$\langle v| = \begin{pmatrix} \bar{v}_1 & \bar{v}_2 & \cdots & \bar{v}_n \end{pmatrix} = |v\rangle^\dagger$$

where $\dagger$ denotes **conjugate transpose** (Hermitian adjoint).

**Example 8.1.4a** (Constructing a bra from a ket):
Given $|v\rangle = \begin{pmatrix} 2+i \\ 1-3i \end{pmatrix}$, the corresponding bra is:

$$\langle v| = |v\rangle^\dagger = \overline{\begin{pmatrix} 2+i \\ 1-3i \end{pmatrix}}^T = \begin{pmatrix} 2-i & 1+3i \end{pmatrix}$$

Each component is complex conjugated: $\overline{2+i} = 2-i$ and $\overline{1-3i} = 1+3i$.

### Definition 8.1.5: Bracket (Inner Product)

$$\langle u | v \rangle = \text{bra} \cdot \text{ket} = \text{inner product}$$

**Example 8.1.4**:
$$|u\rangle = \begin{pmatrix} 1 \\ i \end{pmatrix}, \quad |v\rangle = \begin{pmatrix} 1+i \\ 2 \end{pmatrix}$$

$$\langle u| = \begin{pmatrix} 1 & -i \end{pmatrix}$$

$$\langle u | v \rangle = \begin{pmatrix} 1 & -i \end{pmatrix} \begin{pmatrix} 1+i \\ 2 \end{pmatrix} = (1)(1+i) + (-i)(2) = 1 + i - 2i = 1 - i$$

### Definition 8.1.6: Outer Product

$$|u\rangle \langle v| = \text{ket} \cdot \text{bra} = \text{operator (matrix)}$$

**Example 8.1.5**:
$$|u\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad |v\rangle = \begin{pmatrix} \alpha \\ \beta \end{pmatrix}$$

$$|u\rangle \langle v| = \begin{pmatrix} 1 \\ 0 \end{pmatrix} \begin{pmatrix} \bar{\alpha} & \bar{\beta} \end{pmatrix} = \begin{pmatrix} \bar{\alpha} & \bar{\beta} \\ 0 & 0 \end{pmatrix}$$

This is a **projection operator** onto the subspace spanned by $|u\rangle$.

**Example 8.1.5a** (Concrete outer product):
Let $|u\rangle = \begin{pmatrix} 1 \\ i \end{pmatrix}$ and $|v\rangle = \begin{pmatrix} 1+i \\ 2 \end{pmatrix}$. Then $\langle v| = \begin{pmatrix} 1-i & 2 \end{pmatrix}$ and:

$$|u\rangle\langle v| = \begin{pmatrix} 1 \\ i \end{pmatrix}\begin{pmatrix} 1-i & 2 \end{pmatrix} = \begin{pmatrix} 1(1-i) & 1(2) \\ i(1-i) & i(2) \end{pmatrix} = \begin{pmatrix} 1-i & 2 \\ i+1 & 2i \end{pmatrix}$$

since $i(1-i) = i - i^2 = i + 1$. Note that $|u\rangle\langle v|$ is a $2 \times 2$ matrix (an operator), not a scalar.

**Dirac Notation Cheat Sheet:**
```
|v⟩           Quantum state (ket)
⟨v|           Dual state (bra)
⟨u|v⟩         Inner product (complex number)
|u⟩⟨v|        Outer product (operator)
A|v⟩          Apply operator A to |v⟩
⟨u|A|v⟩       Matrix element of A
```

> **ML/Quantum Connection**: Dirac notation is the **universal language of quantum computing**. It cleanly separates states (kets), measurements (bras), and operators. Writing $|\psi\rangle$ instead of $\vec{\psi}$ immediately signals "quantum state."

---

## 8.1.4 Hermitian and Unitary Operators

### Definition 8.1.7: Hermitian Operator

An operator $A$ is **Hermitian** (self-adjoint) if:
$$A = A^\dagger$$

Equivalently: $\langle u | A | v \rangle = \overline{\langle v | A | u \rangle}$ for all $u, v$.

**Properties of Hermitian Operators:**
1. **Real eigenvalues**: If $A|v\rangle = \lambda |v\rangle$, then $\lambda \in \mathbb{R}$
2. **Orthogonal eigenvectors**: Eigenvectors for distinct eigenvalues are orthogonal
3. **Spectral decomposition**: $A = \sum_i \lambda_i |v_i\rangle \langle v_i|$

**Example 8.1.6**: Pauli matrices (observables in quantum computing):
$$\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}, \quad \sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad \sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$$

Check $\sigma_x = \sigma_x^\dagger$:
$$\sigma_x^\dagger = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}^T \text{ (transpose then conjugate)} = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} = \sigma_x \checkmark$$

**Theorem 8.1.2: Spectral Theorem (Finite-Dimensional)**

Every Hermitian operator $A$ on a finite-dimensional Hilbert space has:
1. Real eigenvalues $\lambda_1, \lambda_2, \ldots, \lambda_n$
2. Orthonormal eigenbasis $\{|v_1\rangle, |v_2\rangle, \ldots, |v_n\rangle\}$
3. Spectral decomposition: $A = \sum_{i=1}^n \lambda_i |v_i\rangle \langle v_i|$

**Example 8.1.6a** (Hermitian eigenvalues and spectral decomposition):
Consider the Hermitian matrix $A = \begin{pmatrix} 1 & -i \\ i & 1 \end{pmatrix}$. Verify $A = A^\dagger$:

$$A^\dagger = (\overline{A})^T: \quad \overline{A} = \begin{pmatrix} 1 & i \\ -i & 1 \end{pmatrix}, \quad (\overline{A})^T = \begin{pmatrix} 1 & -i \\ i & 1 \end{pmatrix} = A \checkmark$$

Eigenvalues: $\det(A - \lambda I) = (1-\lambda)^2 - (-i)(i) = (1-\lambda)^2 - 1 = 0$, giving $\lambda_1 = 0, \lambda_2 = 2$ (both real, as guaranteed).

Eigenvectors: For $\lambda_1 = 0$: $|v_1\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} i \\ 1 \end{pmatrix}$. For $\lambda_2 = 2$: $|v_2\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} -i \\ 1 \end{pmatrix}$.

Spectral decomposition: $A = 0 \cdot |v_1\rangle\langle v_1| + 2 \cdot |v_2\rangle\langle v_2|$.

### Definition 8.1.8: Unitary Operator

An operator $U$ is **unitary** if:
$$U^\dagger U = U U^\dagger = I$$

Equivalently: $U$ preserves inner products: $\langle U u | U v \rangle = \langle u | v \rangle$.

**Properties of Unitary Operators:**
1. **Preserve norms**: $\|U|v\rangle\| = \||v\rangle\|$ (probabilities conserved)
2. **Eigenvalues on unit circle**: If $U|v\rangle = \lambda |v\rangle$, then $|\lambda| = 1$
3. **Form a group**: $U_1 U_2$ is unitary if $U_1, U_2$ are

**Example 8.1.7**: Hadamard gate (quantum):
$$H = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

Check $H^\dagger H = I$:
$$H^\dagger = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} = H \quad \text{(H is real and symmetric)}$$

$$H^2 = \frac{1}{2} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} = \frac{1}{2} \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix} = I \checkmark$$

**Example 8.1.7a** (Complex unitary matrix verification):
The phase gate $S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}$ is unitary. Verify $S^\dagger S = I$:

$$S^\dagger = \overline{S}^T = \begin{pmatrix} 1 & 0 \\ 0 & -i \end{pmatrix}$$

$$S^\dagger S = \begin{pmatrix} 1 & 0 \\ 0 & -i \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & (-i)(i) \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = I \checkmark$$

Eigenvalues of $S$ are $1$ and $i$, both satisfying $|\lambda| = 1$ (on the unit circle), as expected for unitary operators.

**Hermitian vs Unitary:**
```
Hermitian (A = A†)          Unitary (U†U = I)
─────────────────────────   ─────────────────────────
Observables                 Time evolution, gates
Real eigenvalues            Eigenvalues on |λ| = 1
Measurement outcomes        Preserve probabilities
Example: σz, σx, σy         Example: Hadamard, CNOT
```

> **Quantum Connection**:
> - **Observables are Hermitian operators** (real eigenvalues = measurement outcomes)
> - **Time evolution is unitary** (Schrödinger equation $i\hbar \partial_t |\psi\rangle = H|\psi\rangle$ gives $U(t) = e^{-iHt/\hbar}$)
> - All quantum gates (NOT, CNOT, Toffoli) are unitary matrices

---

## 8.1.5 Tensor Products

### Definition 8.1.9: Tensor Product of Hilbert Spaces

Given Hilbert spaces $\mathcal{H}_A$ (dim $n$) and $\mathcal{H}_B$ (dim $m$), their **tensor product** is:
$$\mathcal{H}_A \otimes \mathcal{H}_B$$
with dimension $nm$.

For basis states $|i\rangle \in \mathcal{H}_A$ and $|j\rangle \in \mathcal{H}_B$:
$$|i\rangle \otimes |j\rangle = |i\rangle|j\rangle = |ij\rangle$$

**General state**:
$$|\psi\rangle = \sum_{i,j} \alpha_{ij} |i\rangle \otimes |j\rangle$$

**Example 8.1.8**: Two qubits ($\mathbb{C}^2 \otimes \mathbb{C}^2 = \mathbb{C}^4$):
$$|0\rangle \otimes |1\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix} \otimes \begin{pmatrix} 0 \\ 1 \end{pmatrix} = \begin{pmatrix} 1 \cdot 0 \\ 1 \cdot 1 \\ 0 \cdot 0 \\ 0 \cdot 1 \end{pmatrix} = \begin{pmatrix} 0 \\ 1 \\ 0 \\ 0 \end{pmatrix} = |01\rangle$$

**Computational basis for 2 qubits**: $\{|00\rangle, |01\rangle, |10\rangle, |11\rangle\}$

**Example 8.1.8a** (Tensor product with complex entries):
Let $|\psi\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ i \end{pmatrix}$ and $|\phi\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix} = |1\rangle$. Then:

$$|\psi\rangle \otimes |\phi\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ i \end{pmatrix} \otimes \begin{pmatrix} 0 \\ 1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \cdot 0 \\ 1 \cdot 1 \\ i \cdot 0 \\ i \cdot 1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 0 \\ 1 \\ 0 \\ i \end{pmatrix}$$

Verification: $\||\psi\rangle \otimes |\phi\rangle\|^2 = \frac{1}{2}(0 + 1 + 0 + |i|^2) = \frac{1}{2}(1+1) = 1$. Normalized as expected.

### Tensor Product of Operators

If $A: \mathcal{H}_A \to \mathcal{H}_A$ and $B: \mathcal{H}_B \to \mathcal{H}_B$:
$$(A \otimes B)(|u\rangle \otimes |v\rangle) = (A|u\rangle) \otimes (B|v\rangle)$$

**Example 8.1.9**: Apply $\sigma_x$ to first qubit, $I$ to second:
$$(\sigma_x \otimes I)|01\rangle = (\sigma_x |0\rangle) \otimes (I|1\rangle) = |1\rangle \otimes |1\rangle = |11\rangle$$

**Example 8.1.10**: Bell state (maximally entangled):
$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

**Cannot** be written as $|a\rangle \otimes |b\rangle$ for any single-qubit states $|a\rangle, |b\rangle$ (proof: try and fail).

> **Quantum Connection**:
> - **Tensor products build multi-qubit systems from single qubits**
> - $n$ qubits live in $(\mathbb{C}^2)^{\otimes n} = \mathbb{C}^{2^n}$ (exponential scaling!)
> - Entanglement arises when state is not factorizable: $|\psi\rangle \neq |a\rangle \otimes |b\rangle$

---

## 8.1.6 Orthonormal Bases

### Definition 8.1.10: Orthonormal Basis

A set $\{|e_i\rangle\}_{i=1}^n$ is an **orthonormal basis** if:
1. **Orthonormality**: $\langle e_i | e_j \rangle = \delta_{ij}$ (Kronecker delta)
2. **Completeness**: $\sum_{i=1}^n |e_i\rangle \langle e_i| = I$ (identity resolution)

**Example 8.1.11**: Computational basis for single qubit:
$$|0\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad |1\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$

Check: $\langle 0 | 1 \rangle = 0$, $\langle 0 | 0 \rangle = 1$, $|0\rangle\langle 0| + |1\rangle\langle 1| = I$. ✓

**Example 8.1.12**: Hadamard basis (diagonal basis):
$$|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle), \quad |-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$$

Verify orthonormality:
$$\langle + | - \rangle = \frac{1}{2}(\langle 0| + \langle 1|)(|0\rangle - |1\rangle) = \frac{1}{2}(1 - 1) = 0 \checkmark$$

### Change of Basis

Express $|\psi\rangle$ in basis $\{|e_i\rangle\}$:
$$|\psi\rangle = \sum_i \langle e_i | \psi \rangle |e_i\rangle = \sum_i \psi_i |e_i\rangle$$

**Example 8.1.13**: Convert $|+\rangle$ to computational basis:
$$|+\rangle = \frac{1}{\sqrt{2}}|0\rangle + \frac{1}{\sqrt{2}}|1\rangle$$

Coefficients: $\psi_0 = \langle 0 | + \rangle = \frac{1}{\sqrt{2}}$, $\psi_1 = \langle 1 | + \rangle = \frac{1}{\sqrt{2}}$.

> **Quantum Connection**:
> - **Computational basis** $\{|0\rangle, |1\rangle\}$ is the standard measurement basis
> - Measuring in **Hadamard basis** $\{|+\rangle, |-\rangle\}$ requires rotating first: $H|\psi\rangle$
> - Change of basis = unitary transformation

---

## 8.1.7 Trace and Partial Trace

### Definition 8.1.11: Trace

For operator $A$ with matrix elements $A_{ij}$ in orthonormal basis:
$$\text{Tr}(A) = \sum_i \langle e_i | A | e_i \rangle = \sum_i A_{ii}$$

**Properties:**
1. **Linearity**: $\text{Tr}(A + B) = \text{Tr}(A) + \text{Tr}(B)$
2. **Cyclic**: $\text{Tr}(AB) = \text{Tr}(BA)$
3. **Invariant under basis change**: $\text{Tr}(UAU^\dagger) = \text{Tr}(A)$ for unitary $U$
4. **Trace of projection**: $\text{Tr}(|v\rangle \langle v|) = \langle v | v \rangle = 1$ (if $|v\rangle$ normalized)

**Example 8.1.14**:
$$A = \begin{pmatrix} 2 & i \\ -i & 3 \end{pmatrix}$$
$$\text{Tr}(A) = 2 + 3 = 5$$

**Example 8.1.14a** (Trace cyclic property verification):
Let $A = \begin{pmatrix} 1 & i \\ 0 & 2 \end{pmatrix}$ and $B = \begin{pmatrix} 0 & 1 \\ -i & 1 \end{pmatrix}$. Verify $\text{Tr}(AB) = \text{Tr}(BA)$:

$$AB = \begin{pmatrix} 1(0)+i(-i) & 1(1)+i(1) \\ 0(0)+2(-i) & 0(1)+2(1) \end{pmatrix} = \begin{pmatrix} 1 & 1+i \\ -2i & 2 \end{pmatrix}$$

$$\text{Tr}(AB) = 1 + 2 = 3$$

$$BA = \begin{pmatrix} 0(1)+1(0) & 0(i)+1(2) \\ -i(1)+1(0) & -i(i)+1(2) \end{pmatrix} = \begin{pmatrix} 0 & 2 \\ -i & 3 \end{pmatrix}$$

$$\text{Tr}(BA) = 0 + 3 = 3 \checkmark$$

### Definition 8.1.12: Partial Trace

For composite system $\mathcal{H}_A \otimes \mathcal{H}_B$, the **partial trace** over $\mathcal{H}_B$ is:
$$\text{Tr}_B(|\psi\rangle \langle \psi|) = \sum_j (I_A \otimes \langle j|_B)(|\psi\rangle \langle \psi|)(I_A \otimes |j\rangle_B)$$

where $\{|j\rangle_B\}$ is orthonormal basis for $\mathcal{H}_B$.

**Example 8.1.15**: Bell state $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$:
$$\rho = |\Phi^+\rangle \langle \Phi^+| = \frac{1}{2}(|00\rangle\langle 00| + |00\rangle\langle 11| + |11\rangle\langle 00| + |11\rangle\langle 11|)$$

Trace out qubit B:
$$\rho_A = \text{Tr}_B(\rho) = \frac{1}{2}(|0\rangle\langle 0| \cdot 1 + |1\rangle\langle 1| \cdot 1) = \frac{1}{2}I$$

This is a **maximally mixed state** (50% $|0\rangle$, 50% $|1\rangle$).

**Example 8.1.16**: Product state $|\psi\rangle = |0\rangle \otimes |+\rangle$:
$$\rho = |\psi\rangle \langle \psi| = (|0\rangle\langle 0|) \otimes (|+\rangle\langle +|)$$

$$\rho_A = \text{Tr}_B(\rho) = |0\rangle\langle 0| \cdot \text{Tr}(|+\rangle\langle +|) = |0\rangle\langle 0|$$

Pure state remains pure (no entanglement).

> **Quantum Connection**:
> - **Partial trace extracts reduced density matrix** for subsystem
> - Entanglement detection: $\text{Tr}_B(|\psi\rangle\langle\psi|)$ is mixed iff system is entangled
> - Used to analyze quantum correlations in many-body systems

---

## Summary: Complex Vector Spaces Hierarchy

```
Complex Vector Space (ℂⁿ)
    ↓ + Hermitian inner product
Inner Product Space
    ↓ + Completeness
Hilbert Space (ℋ)
    ↓
Quantum State Space
```

**Key Operator Types:**
- **Hermitian** ($A = A^\dagger$): Observables, measurement operators
- **Unitary** ($U^\dagger U = I$): Time evolution, quantum gates
- **Projection** ($P^2 = P = P^\dagger$): Measurement outcomes

**Dirac Notation Hierarchy:**
```
|v⟩         State (ket)
⟨v|         Dual (bra)
⟨u|v⟩       Probability amplitude
|u⟩⟨v|      Operator (outer product)
⟨u|A|v⟩     Matrix element
```

---

## Exercises

### Basic (★)

**Exercise 8.1.1**: Compute $\langle u | v \rangle$ and $\|v\|$ for:
$$u = \begin{pmatrix} 1 \\ i \\ 0 \end{pmatrix}, \quad v = \begin{pmatrix} 1-i \\ 1+i \\ 2 \end{pmatrix}$$

**Exercise 8.1.2**: Verify that $\sigma_y$ (Pauli Y) is Hermitian and find its eigenvalues.

**Exercise 8.1.3**: Show that the rotation gate $R_z(\theta) = e^{-i\theta \sigma_z/2} = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$ is unitary.

**Exercise 8.1.4**: Compute $|0\rangle \otimes |+\rangle$ in the computational basis for 2 qubits.

### Intermediate (★★)

**Exercise 8.1.5**: Show that if $A$ is Hermitian with eigenvalues $\lambda_i$ and orthonormal eigenvectors $|v_i\rangle$, then $A = \sum_i \lambda_i |v_i\rangle \langle v_i|$.

**Exercise 8.1.6**: Prove that the composition of two unitary operators is unitary.

**Exercise 8.1.7**: Find the matrix representation of $\sigma_x \otimes \sigma_z$ in the computational basis for 2 qubits.

**Exercise 8.1.8**: Express $|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$ (Bell state) in the Hadamard basis $\{|++\rangle, |+-\rangle, |-+\rangle, |--\rangle\}$.

**Exercise 8.1.9**: Compute $\text{Tr}(\sigma_x \sigma_y)$ and verify the cyclic property with $\text{Tr}(\sigma_y \sigma_x)$.

### Challenging (★★★)

**Exercise 8.1.10**: Prove that for any Hermitian operator $A$ with eigenvalues $\lambda_i$:
$$e^{iA} = \sum_i e^{i\lambda_i} |v_i\rangle \langle v_i|$$
is unitary (this is how time evolution works in quantum mechanics).

**Exercise 8.1.11**: Show that the partial trace of $\rho = |\Psi^-\rangle \langle \Psi^-|$ over either subsystem yields $\frac{1}{2}I$ (maximally mixed state).

**Exercise 8.1.12**: Prove that a pure state $|\psi\rangle \in \mathcal{H}_A \otimes \mathcal{H}_B$ is **entangled** iff:
$$\text{Tr}(\rho_A^2) < 1 \quad \text{where} \quad \rho_A = \text{Tr}_B(|\psi\rangle \langle \psi|)$$

**Exercise 8.1.13**: Show that the **commutator** $[A, B] = AB - BA$ satisfies:
$$\text{Tr}([A, B]) = 0$$
for any operators $A, B$. (Hint: Use cyclic property.)

**Exercise 8.1.14**: Prove the **Cauchy-Schwarz inequality** for Hilbert spaces:
$$|\langle u | v \rangle|^2 \leq \langle u | u \rangle \cdot \langle v | v \rangle$$

---

## Related Topics

**Previous Chapters:**
- Chapter 2: Linear Algebra Foundations
- Chapter 3: Complex Numbers and Euler's Formula
- Chapter 7: Eigenvalues and Spectral Theory

**Next Chapters:**
- Chapter 8.2: Quantum States and Measurements
- Chapter 8.3: Quantum Gates and Circuits
- Chapter 8.4: Entanglement and Quantum Information

**Advanced Topics:**
- Density matrices and mixed states
- Quantum channels and CPTP maps
- Quantum entanglement measures (entanglement entropy)
- Infinite-dimensional Hilbert spaces ($L^2$ spaces)

**Applications:**
- Quantum algorithms (Grover, Shor)
- Quantum error correction
- Quantum machine learning (QML)
- Variational quantum eigensolvers (VQE)

---

## Further Reading

1. Nielsen & Chuang, *Quantum Computation and Quantum Information* (2010) - Chapter 2
2. Sakurai & Napolitano, *Modern Quantum Mechanics* (2020) - Chapter 1
3. Watrous, *The Theory of Quantum Information* (2018) - Chapters 1-2
4. Preskill, *Quantum Computation Lecture Notes* - Chapter 2 (online)
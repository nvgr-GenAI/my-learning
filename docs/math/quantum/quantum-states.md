# 8.2 Quantum States

Quantum states are the fundamental mathematical objects describing quantum systems. Unlike classical bits that are definitively 0 or 1, quantum bits (qubits) exist in **superposition** of multiple states simultaneously. This chapter develops the mathematical framework for representing and manipulating quantum states, which forms the foundation for quantum computing and quantum machine learning.

---

## Prerequisites

Before studying this chapter, you should be familiar with:
- [Linear algebra](../linear-algebra/index.md): complex vector spaces, inner products, tensor products
- [Probability theory](../probability/index.md): normalization, random variables
- Dirac notation: kets, bras, inner products (Section 8.1)

---

## 1. Single Qubit States

**Definition 8.2.1 (Qubit State).** A single qubit state is represented by a unit vector in a 2-dimensional complex Hilbert space $\mathbb{C}^2$:

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$

where $\alpha, \beta \in \mathbb{C}$ are complex amplitudes satisfying the **normalization constraint**:

$$|\alpha|^2 + |\beta|^2 = 1$$

The computational basis states are:

$$|0\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad |1\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$

**Physical Interpretation:**
- $|\alpha|^2$ = probability of measuring outcome 0
- $|\beta|^2$ = probability of measuring outcome 1
- The phase relationship between $\alpha$ and $\beta$ affects quantum interference

**Example 8.2.1 (Valid Qubit States).**

1. $|0\rangle$ - classical state, 100% probability of measuring 0
2. $\frac{1}{\sqrt{2}}|0\rangle + \frac{1}{\sqrt{2}}|1\rangle$ - equal superposition
3. $\frac{1}{2}|0\rangle + \frac{\sqrt{3}}{2}|1\rangle$ - 25% chance of 0, 75% chance of 1
4. $\frac{1}{\sqrt{2}}|0\rangle + \frac{i}{\sqrt{2}}|1\rangle$ - equal probabilities, $\pi/2$ phase

**Example 8.2.6 (Normalization Verification).** Consider $|\psi\rangle = \frac{1+i}{2}|0\rangle + \frac{1}{\sqrt{2}}|1\rangle$. Verify this is a valid qubit state.

Compute $|\alpha|^2$: $\alpha = \frac{1+i}{2}$, so $|\alpha|^2 = \left|\frac{1+i}{2}\right|^2 = \frac{(1)^2 + (1)^2}{4} = \frac{2}{4} = \frac{1}{2}$

Compute $|\beta|^2$: $\beta = \frac{1}{\sqrt{2}}$, so $|\beta|^2 = \frac{1}{2}$

Check: $|\alpha|^2 + |\beta|^2 = \frac{1}{2} + \frac{1}{2} = 1$ $\checkmark$

Measurement probabilities: $P(0) = \frac{1}{2} = 50\%$, $P(1) = \frac{1}{2} = 50\%$.

**Example 8.2.7 (Invalid State Detection).** Is $|\psi\rangle = \frac{1}{2}|0\rangle + \frac{1}{2}|1\rangle$ a valid qubit state?

Check: $|\frac{1}{2}|^2 + |\frac{1}{2}|^2 = \frac{1}{4} + \frac{1}{4} = \frac{1}{2} \neq 1$. **Not valid** -- not normalized.

The correctly normalized version would be $|\psi\rangle = \frac{1}{\sqrt{2}}|0\rangle + \frac{1}{\sqrt{2}}|1\rangle$.

---

## 2. The Bloch Sphere

**Definition 8.2.2 (Bloch Sphere Representation).** Any single qubit pure state can be parameterized by two real angles $\theta \in [0, \pi]$ and $\phi \in [0, 2\pi)$:

$$|\psi\rangle = \cos\left(\frac{\theta}{2}\right)|0\rangle + e^{i\phi}\sin\left(\frac{\theta}{2}\right)|1\rangle$$

This maps qubit states to points on the unit sphere in $\mathbb{R}^3$.

**Bloch Sphere Visualization:**

```
                    z (|0⟩)
                    ↑
                    |
                    •  θ = 0
                   /|\
                  / | \
                 /  |  \
      y         /   |   \
      ↑        /    |    \
      |       /     |     \
      |      •------+------•─────→ x (|+⟩)
      |     |ψ⟩    |
      |            \|/
      |             •  θ = π (|1⟩)
      |
      +────→

Spherical coordinates:
x = sin(θ)cos(φ)
y = sin(θ)sin(φ)
z = cos(θ)

Bloch vector: r⃗ = (x, y, z)
```

**Cross-section view:**

```
         |0⟩
          •
          |
          |
  |-⟩ •---+---• |+⟩
         /|
        / |
       •  |
     |i⟩ •
        |1⟩
```

**Theorem 8.2.1 (Bloch Sphere Properties).**

1. Antipodal points represent orthogonal states: $\langle\psi|\psi^\perp\rangle = 0$
2. Global phase $e^{i\gamma}|\psi\rangle$ represents the same physical state
3. Any single-qubit unitary corresponds to a rotation on the Bloch sphere

**Example 8.2.2 (Standard Bloch Sphere States).**

| State | $\theta$ | $\phi$ | Location | Coordinates |
|-------|----------|--------|----------|-------------|
| $\|0\rangle$ | 0 | - | North pole | (0, 0, 1) |
| $\|1\rangle$ | $\pi$ | - | South pole | (0, 0, -1) |
| $\|+\rangle$ | $\pi/2$ | 0 | +x axis | (1, 0, 0) |
| $\|-\rangle$ | $\pi/2$ | $\pi$ | -x axis | (-1, 0, 0) |
| $\|i\rangle$ | $\pi/2$ | $\pi/2$ | +y axis | (0, 1, 0) |
| $\|-i\rangle$ | $\pi/2$ | $3\pi/2$ | -y axis | (0, -1, 0) |

**Example 8.2.8 (Bloch Sphere Coordinate Computation).** Find the Bloch sphere angles $(\theta, \phi)$ and Cartesian coordinates for $|\psi\rangle = \frac{\sqrt{3}}{2}|0\rangle + \frac{1}{2}|1\rangle$.

Compare with the Bloch form $|\psi\rangle = \cos(\theta/2)|0\rangle + e^{i\phi}\sin(\theta/2)|1\rangle$:

- $\cos(\theta/2) = \frac{\sqrt{3}}{2} \implies \theta/2 = \pi/6 \implies \theta = \pi/3$
- $e^{i\phi}\sin(\theta/2) = \frac{1}{2}$. Since $\sin(\pi/6) = \frac{1}{2}$, we get $e^{i\phi} = 1 \implies \phi = 0$.

Cartesian coordinates on the Bloch sphere:

$$x = \sin(\pi/3)\cos(0) = \frac{\sqrt{3}}{2}, \quad y = \sin(\pi/3)\sin(0) = 0, \quad z = \cos(\pi/3) = \frac{1}{2}$$

The state sits on the $xz$-plane, tilted $60°$ from the north pole ($|0\rangle$) toward the equator.

---

## 3. Common Single-Qubit States

**Definition 8.2.3 (Hadamard Basis).** The eigenstates of the Pauli-X operator:

$$|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$$

$$|-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle) = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix}$$

**Definition 8.2.4 (Y-Basis States).** The eigenstates of the Pauli-Y operator:

$$|i\rangle = \frac{1}{\sqrt{2}}(|0\rangle + i|1\rangle) = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ i \end{pmatrix}$$

$$|-i\rangle = \frac{1}{\sqrt{2}}(|0\rangle - i|1\rangle) = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -i \end{pmatrix}$$

**Theorem 8.2.2 (Basis Completeness).**
These form orthonormal bases:
- $\langle +|-\rangle = 0$, $\{|+\rangle, |-\rangle\}$ spans $\mathbb{C}^2$
- $\langle i|-i\rangle = 0$, $\{|i\rangle, |-i\rangle\}$ spans $\mathbb{C}^2$
- Any qubit state can be expressed in any basis

Different bases correspond to measuring different physical observables.

**Example 8.2.9 (Hadamard Basis Orthonormality Verification).** Verify that $|+\rangle$ and $|-\rangle$ are orthonormal.

**Normalization of** $|+\rangle$:

$$\langle +|+\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix}1 & 1\end{pmatrix} \cdot \frac{1}{\sqrt{2}}\begin{pmatrix}1 \\ 1\end{pmatrix} = \frac{1}{2}(1 \cdot 1 + 1 \cdot 1) = \frac{2}{2} = 1 \;\checkmark$$

**Normalization of** $|-\rangle$:

$$\langle -|-\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix}1 & -1\end{pmatrix} \cdot \frac{1}{\sqrt{2}}\begin{pmatrix}1 \\ -1\end{pmatrix} = \frac{1}{2}(1 + 1) = 1 \;\checkmark$$

**Orthogonality:**

$$\langle +|-\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix}1 & 1\end{pmatrix} \cdot \frac{1}{\sqrt{2}}\begin{pmatrix}1 \\ -1\end{pmatrix} = \frac{1}{2}(1 - 1) = 0 \;\checkmark$$

Thus $\{|+\rangle, |-\rangle\}$ is an orthonormal basis for $\mathbb{C}^2$.

**Example 8.2.10 (Measurement Probabilities in Hadamard Basis).** Given $|\psi\rangle = \frac{\sqrt{3}}{2}|0\rangle + \frac{1}{2}|1\rangle$, find the probability of measuring $|+\rangle$ and $|-\rangle$.

Express $|\psi\rangle$ in the Hadamard basis using $|0\rangle = \frac{1}{\sqrt{2}}(|+\rangle + |-\rangle)$ and $|1\rangle = \frac{1}{\sqrt{2}}(|+\rangle - |-\rangle)$:

$$|\psi\rangle = \frac{\sqrt{3}}{2}\cdot\frac{|+\rangle + |-\rangle}{\sqrt{2}} + \frac{1}{2}\cdot\frac{|+\rangle - |-\rangle}{\sqrt{2}} = \frac{\sqrt{3}+1}{2\sqrt{2}}|+\rangle + \frac{\sqrt{3}-1}{2\sqrt{2}}|-\rangle$$

$$P(+) = \left|\frac{\sqrt{3}+1}{2\sqrt{2}}\right|^2 = \frac{(\sqrt{3}+1)^2}{8} = \frac{4 + 2\sqrt{3}}{8} = \frac{2+\sqrt{3}}{4} \approx 93.3\%$$

$$P(-) = \left|\frac{\sqrt{3}-1}{2\sqrt{2}}\right|^2 = \frac{(\sqrt{3}-1)^2}{8} = \frac{4 - 2\sqrt{3}}{8} = \frac{2-\sqrt{3}}{4} \approx 6.7\%$$

Verify: $P(+) + P(-) = \frac{2+\sqrt{3}+2-\sqrt{3}}{4} = \frac{4}{4} = 1$ $\checkmark$

---

## 4. Multi-Qubit States

**Definition 8.2.5 (n-Qubit State Space).** An $n$-qubit system lives in the Hilbert space $(\mathbb{C}^2)^{\otimes n} \cong \mathbb{C}^{2^n}$. A general state is:

$$|\psi\rangle = \sum_{x \in \{0,1\}^n} \alpha_x |x\rangle$$

where $\sum_{x} |\alpha_x|^2 = 1$ and $|x\rangle = |x_1x_2\cdots x_n\rangle$.

### 4.1 Two-Qubit Computational Basis

The basis states for 2 qubits are formed by tensor products:

$$|00\rangle = |0\rangle \otimes |0\rangle = \begin{pmatrix} 1 \\ 0 \\ 0 \\ 0 \end{pmatrix}, \quad |01\rangle = |0\rangle \otimes |1\rangle = \begin{pmatrix} 0 \\ 1 \\ 0 \\ 0 \end{pmatrix}$$

$$|10\rangle = |1\rangle \otimes |0\rangle = \begin{pmatrix} 0 \\ 0 \\ 1 \\ 0 \end{pmatrix}, \quad |11\rangle = |1\rangle \otimes |1\rangle = \begin{pmatrix} 0 \\ 0 \\ 0 \\ 1 \end{pmatrix}$$

**General two-qubit state:**

$$|\psi\rangle = \alpha_{00}|00\rangle + \alpha_{01}|01\rangle + \alpha_{10}|10\rangle + \alpha_{11}|11\rangle$$

with $|\alpha_{00}|^2 + |\alpha_{01}|^2 + |\alpha_{10}|^2 + |\alpha_{11}|^2 = 1$.

**Example 8.2.11 (Two-Qubit State Probabilities).** Consider the 2-qubit state:

$$|\psi\rangle = \frac{1}{2}|00\rangle + \frac{i}{2}|01\rangle + \frac{1}{2}|10\rangle + \frac{1}{2}|11\rangle$$

**Verify normalization:** $|\frac{1}{2}|^2 + |\frac{i}{2}|^2 + |\frac{1}{2}|^2 + |\frac{1}{2}|^2 = \frac{1}{4}+\frac{1}{4}+\frac{1}{4}+\frac{1}{4} = 1$ $\checkmark$

**Individual outcome probabilities:** $P(00) = \frac{1}{4}$, $P(01) = \frac{1}{4}$, $P(10) = \frac{1}{4}$, $P(11) = \frac{1}{4}$.

**Marginal probability of first qubit being** $|0\rangle$: $P(\text{1st} = 0) = P(00) + P(01) = \frac{1}{2}$.

**Marginal probability of first qubit being** $|1\rangle$: $P(\text{1st} = 1) = P(10) + P(11) = \frac{1}{2}$.

### 4.2 Product States vs Entangled States

**Definition 8.2.6 (Product State).** A state $|\psi\rangle_{AB}$ is a **product state** if it can be written as:

$$|\psi\rangle_{AB} = |\psi\rangle_A \otimes |\psi\rangle_B$$

for some single-qubit states $|\psi\rangle_A$ and $|\psi\rangle_B$.

**Example 8.2.3.**

**Product state** (separable):
$$|\psi_1\rangle = (|0\rangle + |1\rangle) \otimes |0\rangle = |00\rangle + |10\rangle$$

Can be written as $|\psi_A\rangle \otimes |\psi_B\rangle$.

**Entangled state** (non-separable):
$$|\psi_2\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

Cannot be written as a tensor product of single-qubit states.

**Example 8.2.12 (Proving Entanglement by Contradiction).** Show that $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$ cannot be factored.

Suppose $|\Phi^+\rangle = (a|0\rangle + b|1\rangle) \otimes (c|0\rangle + d|1\rangle)$ for some $a,b,c,d \in \mathbb{C}$.

Expanding the tensor product:

$$ac|00\rangle + ad|01\rangle + bc|10\rangle + bd|11\rangle = \frac{1}{\sqrt{2}}|00\rangle + 0|01\rangle + 0|10\rangle + \frac{1}{\sqrt{2}}|11\rangle$$

Matching coefficients: $ac = \frac{1}{\sqrt{2}}$, $ad = 0$, $bc = 0$, $bd = \frac{1}{\sqrt{2}}$.

From $ad = 0$: either $a = 0$ or $d = 0$.

- If $a = 0$: then $ac = 0 \neq \frac{1}{\sqrt{2}}$ -- contradiction.
- If $d = 0$: then $bd = 0 \neq \frac{1}{\sqrt{2}}$ -- contradiction.

No valid factorization exists, so $|\Phi^+\rangle$ is **entangled**. $\square$

**State space scaling:**

```
Qubits  Basis States  Complex Amplitudes  Classical Bits Equivalent
  1         2               2                      1
  2         4               4                      2
  3         8               8                      3
  n        2^n             2^n                     n

Exponential scaling: 2^n is the source of quantum computational power
```

---

## 5. Superposition vs Classical Probability

**Definition 8.2.7 (Quantum Superposition).** A quantum state is in superposition when it is a linear combination of basis states with complex amplitudes. The system simultaneously "exists" in all basis states until measurement.

**Example 8.2.13 (Superposition Measurement Probabilities).** The state $|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$ is a superposition. Compute measurement outcomes in two bases.

**Computational basis** $\{|0\rangle, |1\rangle\}$:

$$P(0) = \left|\frac{1}{\sqrt{2}}\right|^2 = \frac{1}{2}, \quad P(1) = \left|\frac{1}{\sqrt{2}}\right|^2 = \frac{1}{2}$$

Outcomes are uniformly random -- looks like a fair coin flip.

**Hadamard basis** $\{|+\rangle, |-\rangle\}$:

Since $|+\rangle = 1 \cdot |+\rangle + 0 \cdot |-\rangle$:

$$P(+) = |1|^2 = 1, \quad P(-) = |0|^2 = 0$$

Outcome is **deterministic** -- always get $|+\rangle$. This is what distinguishes quantum superposition from classical randomness: the result depends on which basis you measure in.

### 5.1 Key Differences from Classical Probability

| Aspect | Classical Mixture | Quantum Superposition |
|--------|------------------|----------------------|
| Mathematical | Probability distribution $p_i$ | Complex amplitudes $\alpha_i$ |
| Combination | Convex sum $\sum p_i = 1$ | Normalized vector $\sum \|\alpha_i\|^2 = 1$ |
| Interference | No interference | Amplitudes interfere |
| Measurement | Reveals pre-existing value | Collapses superposition |
| Information | $\log_2(n)$ bits | $2n$ real parameters |

**Example 8.2.4 (Interference Demonstration).**

Consider the state $|\psi\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$.

Measuring in computational basis: 50% probability each outcome (like classical).

But measuring in Hadamard basis $\{|+\rangle, |-\rangle\}$:

$$|\psi\rangle = |+\rangle$$

Result: 100% probability of $|+\rangle$, 0% for $|-\rangle$.

This demonstrates **constructive interference** - amplitudes combine coherently.

**Contrast:** A classical system with 50/50 probability of 0 or 1 would give 50/50 in any measurement basis.

### 5.2 Visualization: Superposition vs Mixture

```
Classical Mixture (50% |0⟩, 50% |1⟩):
  Definitely in |0⟩ OR definitely in |1⟩
  Don't know which until measurement

  [0] 50%  OR  [1] 50%

Quantum Superposition (1/√2)|0⟩ + (1/√2)|1⟩:
  Simultaneously in |0⟩ AND |1⟩
  Creates interference patterns

  [0]──┬──[1]
       └──── amplitudes interfere
```

---

## 6. The No-Cloning Theorem

**Theorem 8.2.3 (No-Cloning Theorem).** There exists no unitary operation that can copy an arbitrary unknown quantum state:

$$U|\psi\rangle|0\rangle = |\psi\rangle|\psi\rangle \quad \text{(impossible for all $|\psi\rangle$)}$$

**Proof.** Suppose a unitary $U$ clones states: $U|\psi\rangle|0\rangle = |\psi\rangle|\psi\rangle$.

For two different states $|\phi\rangle$ and $|\chi\rangle$:
$$U|\phi\rangle|0\rangle = |\phi\rangle|\phi\rangle$$
$$U|\chi\rangle|0\rangle = |\chi\rangle|\chi\rangle$$

Take inner product of both sides:
$$\langle\phi|U^\dagger U|\chi\rangle \langle 0|0\rangle = \langle\phi|\chi\rangle\langle\phi|\chi\rangle$$

Since $U$ is unitary ($U^\dagger U = I$):
$$\langle\phi|\chi\rangle = \langle\phi|\chi\rangle^2$$

This implies $\langle\phi|\chi\rangle \in \{0, 1\}$ (states must be orthogonal or identical).

Since this must hold for arbitrary $|\phi\rangle, |\chi\rangle$, no universal cloning operation exists. $\square$

**Corollary 8.2.4.** Orthogonal states CAN be cloned (e.g., computational basis states), but not superpositions of them.

### 6.1 Implications

1. **Quantum security:** Eavesdropping on quantum communication disturbs the state
2. **Information theory:** Quantum information is fundamentally different from classical
3. **Error correction:** Must use sophisticated entanglement-based codes
4. **ML applications:** Cannot "back up" quantum data during training

---

## 7. Bell States

**Definition 8.2.8 (Bell Basis).** The four maximally entangled two-qubit states:

$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

$$|\Phi^-\rangle = \frac{1}{\sqrt{2}}(|00\rangle - |11\rangle)$$

$$|\Psi^+\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$$

$$|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$$

**Theorem 8.2.5 (Bell Basis Properties).**
1. Form an orthonormal basis for 2-qubit space
2. Maximally entangled: measuring one qubit fully determines the other
3. Used in quantum teleportation, superdense coding, and quantum cryptography

**Example 8.2.14 (Bell State Inner Products).** Verify that $|\Phi^+\rangle$ and $|\Psi^+\rangle$ are orthogonal and normalized.

**Normalization of** $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$:

$$\langle\Phi^+|\Phi^+\rangle = \frac{1}{2}(\langle 00| + \langle 11|)(|00\rangle + |11\rangle) = \frac{1}{2}(\langle 00|00\rangle + \langle 00|11\rangle + \langle 11|00\rangle + \langle 11|11\rangle)$$

$$= \frac{1}{2}(1 + 0 + 0 + 1) = 1 \;\checkmark$$

**Orthogonality** with $|\Psi^+\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$:

$$\langle\Phi^+|\Psi^+\rangle = \frac{1}{2}(\langle 00| + \langle 11|)(|01\rangle + |10\rangle) = \frac{1}{2}(\langle 00|01\rangle + \langle 00|10\rangle + \langle 11|01\rangle + \langle 11|10\rangle)$$

$$= \frac{1}{2}(0 + 0 + 0 + 0) = 0 \;\checkmark$$

**Bell State Correlations:**

```
|Φ⁺⟩: Measure qubit 1 → 0  ⟹  Qubit 2 collapses to |0⟩ (same)
                        → 1  ⟹  Qubit 2 collapses to |1⟩ (same)

|Φ⁻⟩: Measure qubit 1 → 0  ⟹  Qubit 2 collapses to |0⟩ (same, phase flip)
                        → 1  ⟹  Qubit 2 collapses to |1⟩ (same, phase flip)

|Ψ⁺⟩: Measure qubit 1 → 0  ⟹  Qubit 2 collapses to |1⟩ (opposite)
                        → 1  ⟹  Qubit 2 collapses to |0⟩ (opposite)

|Ψ⁻⟩: Measure qubit 1 → 0  ⟹  Qubit 2 collapses to |1⟩ (opposite, phase)
                        → 1  ⟹  Qubit 2 collapses to |0⟩ (opposite, phase)
```

---

## 8. Pure States vs Mixed States

**Definition 8.2.9 (Pure State).** A quantum system is in a **pure state** if it can be described by a single state vector $|\psi\rangle$. This represents maximal knowledge about the system.

**Definition 8.2.10 (Mixed State).** A **mixed state** represents statistical uncertainty about which pure state the system is in. It is described by a density matrix (Chapter 8.6):

$$\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$$

where $p_i$ are classical probabilities and $\sum_i p_i = 1$.

### 8.1 Comparison

| Property | Pure State | Mixed State |
|----------|-----------|-------------|
| Representation | State vector $\|\psi\rangle$ | Density matrix $\rho$ |
| Uncertainty | Quantum only | Quantum + classical |
| Rank | Rank-1 projection | Rank $> 1$ |
| Purity | $\text{Tr}(\rho^2) = 1$ | $\text{Tr}(\rho^2) < 1$ |

**Example 8.2.5.**

Pure state: $|\psi\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$ - coherent superposition

Density matrix: $\rho_{pure} = |\psi\rangle\langle\psi| = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$

Mixed state: 50% chance of $|0\rangle$ or $|1\rangle$ - classical mixture

$$\rho_{mixed} = \frac{1}{2}|0\rangle\langle 0| + \frac{1}{2}|1\rangle\langle 1| = \frac{1}{2}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

These are **fundamentally different** despite identical measurement probabilities in the computational basis.

Check purity: $\text{Tr}(\rho_{pure}^2) = 1$, $\text{Tr}(\rho_{mixed}^2) = 1/2 < 1$.

---

## 9. Quantum Data Encoding

*ML/AI Connection: Feature encoding for quantum machine learning*

Classical data must be encoded into quantum states for quantum machine learning. Common encoding schemes:

### 9.1 Basis Encoding

$$x \in \{0,1\}^n \quad \rightarrow \quad |x\rangle$$

Example: Binary string 101 $\rightarrow$ $|101\rangle$

- Uses $n$ qubits for $n$ bits
- Simple but not space-efficient

### 9.2 Amplitude Encoding

$$\mathbf{x} \in \mathbb{R}^{2^n} \quad \rightarrow \quad |\psi\rangle = \sum_{i=0}^{2^n-1} \alpha_i |i\rangle, \quad \alpha_i = \frac{x_i}{\|\mathbf{x}\|}$$

- Encodes $2^n$ values in $n$ qubits (exponential compression!)
- Requires normalization: $\sum |\alpha_i|^2 = 1$
- Loading data is non-trivial (state preparation problem)

**Example:** Vector $(1, 0, 0, 1)$ normalized to $(\frac{1}{\sqrt{2}}, 0, 0, \frac{1}{\sqrt{2}})$:

$$|\psi\rangle = \frac{1}{\sqrt{2}}|00\rangle + \frac{1}{\sqrt{2}}|11\rangle$$

This is a Bell state!

### 9.3 Angle Encoding

$$x_i \in \mathbb{R} \quad \rightarrow \quad |\psi_i\rangle = \cos(x_i)|0\rangle + \sin(x_i)|1\rangle$$

Combined: $|\psi\rangle = |\psi_1\rangle \otimes |\psi_2\rangle \otimes \cdots \otimes |\psi_n\rangle$

**Example 8.2.15 (Angle Encoding Computation).** Encode the classical data vector $\mathbf{x} = (\frac{\pi}{4}, \frac{\pi}{3})$ into 2 qubits using angle encoding.

**Qubit 1** ($x_1 = \pi/4$):

$$|\psi_1\rangle = \cos(\pi/4)|0\rangle + \sin(\pi/4)|1\rangle = \frac{\sqrt{2}}{2}|0\rangle + \frac{\sqrt{2}}{2}|1\rangle$$

**Qubit 2** ($x_2 = \pi/3$):

$$|\psi_2\rangle = \cos(\pi/3)|0\rangle + \sin(\pi/3)|1\rangle = \frac{1}{2}|0\rangle + \frac{\sqrt{3}}{2}|1\rangle$$

**Combined 2-qubit state** (tensor product):

$$|\psi\rangle = |\psi_1\rangle \otimes |\psi_2\rangle = \frac{\sqrt{2}}{4}|00\rangle + \frac{\sqrt{6}}{4}|01\rangle + \frac{\sqrt{2}}{4}|10\rangle + \frac{\sqrt{6}}{4}|11\rangle$$

Verify: $\frac{2}{16} + \frac{6}{16} + \frac{2}{16} + \frac{6}{16} = \frac{16}{16} = 1$ $\checkmark$

### 9.4 Trade-offs

| Encoding | Qubits | Data Capacity | Preparation Complexity |
|----------|--------|---------------|----------------------|
| Basis | $n$ | $n$ bits | $O(1)$ |
| Amplitude | $n$ | $2^n$ values | $O(2^n)$ |
| Angle | $n$ | $n$ values | $O(n)$ |

The state preparation bottleneck is a major challenge in quantum machine learning.

---

## 10. Quantum Advantage from Superposition

**Why superposition matters for quantum computing:**

### 10.1 Quantum Parallelism

**Classical parallel evaluation:**
- Evaluate $f(x)$ for $N$ inputs: requires $N$ function calls
- Time complexity: $O(N)$

**Quantum parallel evaluation:**
- Prepare superposition: $|\psi\rangle = \frac{1}{\sqrt{N}}\sum_{x=0}^{N-1} |x\rangle$
- Apply quantum function: $U_f|\psi\rangle = \frac{1}{\sqrt{N}}\sum_{x=0}^{N-1} |x\rangle|f(x)\rangle$
- All $f(x)$ computed in **one** operation!

**The catch:** Measurement collapses to single $(x, f(x))$ pair.

**Solution:** Quantum algorithms use interference to amplify correct answer amplitudes.

### 10.2 Amplitude Amplification

```
Start:    Equal superposition
          ||||||||||||||||

Compute:  f(x) in superposition
          ||||||||||||||||

Amplify:  Interference increases
          ______|______   target amplitude
         /      |      \
        /       |       \

Measure:  High probability of
          correct answer
```

This enables algorithms like:
- **Grover's search:** $O(\sqrt{N})$ vs classical $O(N)$
- **Shor's factoring:** Polynomial vs exponential
- **Quantum simulation:** Exponential advantage for certain systems

---

## 11. Quantum Randomness vs Classical

### 11.1 True Randomness

**Classical computing:** Pseudorandom (deterministic algorithms with long periods)

**Quantum mechanics:** Fundamentally random outcomes upon measurement

**Example:**
$$|\psi\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$$

Measuring this state gives 0 or 1 with exactly 50% probability each. No hidden variables, no deterministic mechanism - this is true randomness certified by quantum mechanics.

### 11.2 Applications

- **Quantum random number generators (QRNGs):** Hardware-based true randomness
- **Cryptographic key generation:** Security relies on unpredictability
- **Monte Carlo simulations:** Quantum ML sampling
- **Secure protocols:** Randomness cannot be predicted by adversary

Bell's theorem and experimental violations of Bell inequalities confirm this randomness is fundamental to nature, not due to incomplete information.

---

## Summary

Quantum states are normalized vectors in complex Hilbert spaces:

- **Single qubit:** $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ with $|\alpha|^2 + |\beta|^2 = 1$
- **Bloch sphere:** Geometric visualization on unit sphere, $(\theta, \phi)$ parameterization
- **Common states:** $|+\rangle, |-\rangle, |i\rangle, |-i\rangle$ form alternative bases
- **Multi-qubit states:** Live in exponentially large spaces ($2^n$ dimensions)
- **Superposition:** Enables quantum interference, unlike classical probability
- **No-cloning theorem:** Cannot copy unknown quantum states
- **Bell states:** Maximally entangled 2-qubit states
- **Pure vs mixed:** Vectors vs density matrices represent different types of uncertainty
- **Data encoding:** Trade-offs between qubit count and preparation complexity
- **Quantum advantage:** Superposition enables parallelism when combined with interference
- **True randomness:** Quantum measurements are fundamentally probabilistic

---

## Exercises

### Basic (★)

**Exercise 8.2.1.** Verify that $|\psi\rangle = \frac{3}{5}|0\rangle + \frac{4}{5}|1\rangle$ is a valid qubit state. What are the measurement probabilities?

**Exercise 8.2.2.** Express $|+\rangle$ and $|-\rangle$ in terms of Bloch sphere angles $(\theta, \phi)$.

**Exercise 8.2.3.** Write the 3-qubit computational basis states $|000\rangle, |001\rangle, \ldots, |111\rangle$ as column vectors in $\mathbb{C}^8$.

**Exercise 8.2.4.** Show that $\langle +|0\rangle = \frac{1}{\sqrt{2}}$ and interpret this result in terms of measurement probabilities.

**Exercise 8.2.5.** For the state $|\psi\rangle = \frac{1}{\sqrt{2}}(|0\rangle + i|1\rangle)$, find the Bloch sphere angles $\theta$ and $\phi$.

### Intermediate (★★)

**Exercise 8.2.6.** Prove that the Bell states $|\Phi^+\rangle, |\Phi^-\rangle, |\Psi^+\rangle, |\Psi^-\rangle$ form an orthonormal basis for $\mathbb{C}^4$.

**Exercise 8.2.7.** A qubit is prepared in state $|\psi\rangle = \cos(30°)|0\rangle + \sin(30°)|1\rangle$. What is the probability of measuring $|+\rangle$ if we measure in the Hadamard basis?

**Exercise 8.2.8.** Show that the state $\frac{1}{\sqrt{3}}(|00\rangle + |01\rangle + |10\rangle)$ cannot be written as a tensor product of two single-qubit states.

**Exercise 8.2.9.** Given a 2-qubit state $|\psi\rangle = \alpha_{00}|00\rangle + \alpha_{01}|01\rangle + \alpha_{10}|10\rangle + \alpha_{11}|11\rangle$, express it in the Bell basis.

**Exercise 8.2.10.** Compare the density matrices for (a) pure state $\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$ and (b) mixed state with 50% $|0\rangle$, 50% $|1\rangle$. Compute their purities.

### Advanced (★★★)

**Exercise 8.2.11.** Complete the no-cloning proof: if a cloning operation works for two non-orthogonal states $|\psi\rangle$ and $|\phi\rangle$, prove that $\langle\psi|\phi\rangle \in \{0, 1\}$.

**Exercise 8.2.12.** How many real parameters are needed to specify an arbitrary $n$-qubit pure state? Justify your answer considering normalization and global phase.

**Exercise 8.2.13.** Design an amplitude encoding scheme for the vector $\mathbf{x} = (1, 2, 3, 4, 5, 6, 7, 8)$. How many qubits are required? What are the normalized amplitudes?

**Exercise 8.2.14.** Show that measuring the first qubit of $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$ in the computational basis collapses the second qubit to the same state (calculate the post-measurement state explicitly).

**Exercise 8.2.15.** Prove that any single-qubit unitary $U$ can be written as $U = e^{i\alpha}R_{\hat{n}}(\theta)$ where $R_{\hat{n}}(\theta)$ is a rotation by angle $\theta$ around axis $\hat{n}$ on the Bloch sphere (Hint: use eigenvalue decomposition).

---

## Related Topics

- **[Section 8.3: Quantum Gates](quantum-gates.md)** - Operations that transform quantum states
- **[Section 8.4: Entanglement](entanglement.md)** - Non-classical correlations between qubits
- **[Section 8.5: Measurement](measurement.md)** - Extracting information from quantum states
- **[Section 8.6: Density Matrices](density-matrices.md)** - Complete formalism for mixed states
- **[Chapter 8.7: Quantum Algorithms](quantum-algorithms.md)** - Leveraging superposition and interference
- **[Chapter 8.10: Quantum Machine Learning](quantum-machine-learning.md)** - Quantum data processing for ML tasks

---

## References

- Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and Quantum Information*. Cambridge University Press.
- Wilde, M. M. (2013). *Quantum Information Theory*. Cambridge University Press.
- Schuld, M., & Petruccione, F. (2018). *Supervised Learning with Quantum Computers*. Springer.

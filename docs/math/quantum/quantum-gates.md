# 8.3 Quantum Gates

## Prerequisites

- **Linear Algebra**: Unitary matrices, eigenvalues, tensor products (§2.4, §2.5)
- **Complex Numbers**: Complex exponentials, conjugation (§1.3)
- **Quantum States**: Qubits, superposition, measurement (§8.1, §8.2)
- **Matrix Operations**: Matrix multiplication, conjugate transpose (§2.1)

## Overview

Quantum gates are the fundamental building blocks of quantum computation, analogous to classical logic gates but operating on quantum states. Unlike classical gates, quantum gates must be **reversible** and preserve quantum information, leading to the requirement that they are represented by **unitary matrices**.

This chapter establishes quantum gates as the operational foundation of quantum computing, connecting abstract linear algebra to practical quantum circuit design.

---

## 8.3.1 Quantum Gates as Unitary Matrices

**Definition 8.3.1** (Quantum Gate)
A **quantum gate** acting on $n$ qubits is a linear operator $U: \mathbb{C}^{2^n} \to \mathbb{C}^{2^n}$ that is **unitary**, satisfying:

$$U U^\dagger = U^\dagger U = I$$

where $U^\dagger$ is the conjugate transpose of $U$ and $I$ is the identity matrix.

**Theorem 8.3.1** (Unitarity Preserves Normalization)
If $U$ is unitary and $|\psi\rangle$ is a normalized quantum state ($\langle\psi|\psi\rangle = 1$), then $U|\psi\rangle$ is also normalized.

*Proof:*
$$\langle U\psi | U\psi \rangle = \langle \psi | U^\dagger U | \psi \rangle = \langle \psi | I | \psi \rangle = \langle \psi | \psi \rangle = 1$$

**Physical Interpretation:**
- **Reversibility**: Since $U^{-1} = U^\dagger$ exists, quantum operations can be reversed
- **Information Preservation**: No information is lost during gate operations
- **Probability Conservation**: Total probability remains 1 (unitarity preserves norms)

**Connection to Classical Computing:**
Classical irreversible gates (like AND, OR) cannot be directly implemented in quantum computing. Classical reversible gates (like Toffoli) have quantum analogues.

> **ML Connection**: Just as neural network layers are parameterized transformations, quantum gates are parameterized unitary transformations. Variational quantum algorithms optimize gate parameters analogously to training neural networks.

---

## 8.3.2 Single-Qubit Gates

Single-qubit gates operate on the two-dimensional Hilbert space $\mathbb{C}^2$, represented by $2 \times 2$ unitary matrices.

### Pauli Gates

**Definition 8.3.2** (Pauli Matrices)
The **Pauli matrices** are:

$$X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

**Properties:**
1. **Hermitian**: $X^\dagger = X$, $Y^\dagger = Y$, $Z^\dagger = Z$
2. **Unitary**: $X^2 = Y^2 = Z^2 = I$
3. **Traceless**: $\text{tr}(X) = \text{tr}(Y) = \text{tr}(Z) = 0$
4. **Anticommutation**: $\{X, Y\} = XY + YX = 0$, etc.

**Actions on Basis States:**

$$X|0\rangle = |1\rangle, \quad X|1\rangle = |0\rangle \quad \text{(bit flip)}$$

$$Z|0\rangle = |0\rangle, \quad Z|1\rangle = -|1\rangle \quad \text{(phase flip)}$$

$$Y|0\rangle = i|1\rangle, \quad Y|1\rangle = -i|0\rangle \quad \text{(bit + phase flip)}$$

**Example 8.3.1:** (Pauli-X gate action by matrix multiplication)
Apply the Pauli-X gate to $|0\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$ and $|1\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$:

$$X|0\rangle = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 0 \cdot 1 + 1 \cdot 0 \\ 1 \cdot 1 + 0 \cdot 0 \end{pmatrix} = \begin{pmatrix} 0 \\ 1 \end{pmatrix} = |1\rangle$$

$$X|1\rangle = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 0 \\ 1 \end{pmatrix} = \begin{pmatrix} 0 \cdot 0 + 1 \cdot 1 \\ 1 \cdot 0 + 0 \cdot 1 \end{pmatrix} = \begin{pmatrix} 1 \\ 0 \end{pmatrix} = |0\rangle$$

This confirms $X$ acts as a bit flip, swapping $|0\rangle \leftrightarrow |1\rangle$.

**Geometric Interpretation on Bloch Sphere:**
- $X$: rotation by $\pi$ around $x$-axis
- $Y$: rotation by $\pi$ around $y$-axis
- $Z$: rotation by $\pi$ around $z$-axis

### Hadamard Gate

**Definition 8.3.3** (Hadamard Gate)
The **Hadamard gate** is:

$$H = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

**Actions:**

$$H|0\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}} = |+\rangle$$

$$H|1\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}} = |-\rangle$$

**Properties:**
- Self-inverse: $H^2 = I$
- Creates equal superposition from basis states
- Changes between computational basis $\{|0\rangle, |1\rangle\}$ and Hadamard basis $\{|+\rangle, |-\rangle\}$

**Example 8.3.2:** (Verify unitarity of the Hadamard gate)
Since $H$ is real and symmetric, $H^\dagger = H$. We verify $H^\dagger H = H^2 = I$:

$$H^\dagger H = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} \cdot \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} = \frac{1}{2} \begin{pmatrix} 1 \cdot 1 + 1 \cdot 1 & 1 \cdot 1 + 1 \cdot (-1) \\ 1 \cdot 1 + (-1) \cdot 1 & 1 \cdot 1 + (-1)(-1) \end{pmatrix} = \frac{1}{2} \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix} = I \; \checkmark$$

This confirms $H$ is unitary and self-inverse ($H = H^{-1}$).

**Example 8.3.3:** (Hadamard gate applied to $|0\rangle$, step by step)
Compute $H|0\rangle$ by explicit matrix-vector multiplication:

$$H|0\rangle = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \cdot 1 + 1 \cdot 0 \\ 1 \cdot 1 + (-1) \cdot 0 \end{pmatrix} = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 1 \end{pmatrix} = \frac{1}{\sqrt{2}}|0\rangle + \frac{1}{\sqrt{2}}|1\rangle = |+\rangle$$

The resulting state $|+\rangle$ gives measurement probabilities $P(0) = |1/\sqrt{2}|^2 = 1/2$ and $P(1) = |1/\sqrt{2}|^2 = 1/2$, confirming equal superposition.

**Importance**: The Hadamard gate is the primary tool for creating superposition, essential for quantum parallelism.

### Phase Gates

**Definition 8.3.4** (Phase Gates)
The **S gate** (phase gate) and **T gate** ($\pi/8$ gate) are:

$$S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}, \quad T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$$

**Properties:**
- $S^2 = Z$, $T^2 = S$, $T^4 = Z$
- Leave $|0\rangle$ unchanged, add phase to $|1\rangle$
- $S$ and $T$ are essential for universal quantum computation

**Example 8.3.4:** (S gate applied to $|1\rangle$)
Apply the S gate to $|1\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$:

$$S|1\rangle = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix} \begin{pmatrix} 0 \\ 1 \end{pmatrix} = \begin{pmatrix} 0 \\ i \end{pmatrix} = i|1\rangle$$

The S gate adds a phase of $i = e^{i\pi/2}$ to $|1\rangle$ while leaving the measurement probabilities unchanged: $P(1) = |i|^2 = 1$. Note that $S|0\rangle = |0\rangle$ (no effect on $|0\rangle$).

**Example 8.3.5:** (T gate applied to $|+\rangle$)
Apply the T gate to the superposition state $|+\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$:

$$T|+\rangle = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix} \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 1 \end{pmatrix} = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ e^{i\pi/4} \end{pmatrix} = \frac{1}{\sqrt{2}}\left(|0\rangle + e^{i\pi/4}|1\rangle\right)$$

The measurement probabilities are still $P(0) = P(1) = 1/2$ (the T gate only changes the relative phase), but interference patterns in subsequent gates will differ. This is the key role of phase gates: they alter *how* states interfere without changing individual measurement probabilities.

### Rotation Gates

**Definition 8.3.5** (Rotation Gates)
The **rotation gates** around axes $x$, $y$, $z$ by angle $\theta$ are:

$$R_x(\theta) = e^{-i\theta X/2} = \begin{pmatrix} \cos(\theta/2) & -i\sin(\theta/2) \\ -i\sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$$

$$R_y(\theta) = e^{-i\theta Y/2} = \begin{pmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$$

$$R_z(\theta) = e^{-i\theta Z/2} = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$$

**Key Relations:**
- $R_z(\theta) = e^{-i\theta/2} \begin{pmatrix} 1 & 0 \\ 0 & e^{i\theta} \end{pmatrix}$ (up to global phase)
- $R_x(\pi) = -iX$, $R_y(\pi) = -iY$, $R_z(\pi) = -iZ$
- Any single-qubit unitary can be decomposed as rotations

> **ML Connection**: Rotation gates with trainable angles $\theta$ are the quantum analogue of parameterized layers in neural networks. Variational quantum eigensolvers (VQE) and quantum approximate optimization algorithm (QAOA) optimize these rotation angles.

---

## 8.3.3 Circuit Notation

Quantum circuits provide a visual representation of quantum algorithms as sequences of gates applied to qubits.

### Basic Circuit Elements

```
Quantum wire (represents a qubit over time):
|ψ⟩ ────────────

Single-qubit gate:
    ┌───┐
────┤ H ├────
    └───┘

Measurement:
    ┌─┐
────┤M├──── → classical bit
    └╥┘
     ║

Complete single-qubit circuit:
    ┌───┐┌───┐┌───┐┌─┐
|0⟩─┤ H ├┤ T ├┤ H ├┤M├
    └───┘└───┘└───┘└╥┘
                     ║
                    c: 1-bit classical register
```

**Convention**: Time flows left to right, qubits are horizontal wires.

### Example: Hadamard-Measure Circuit

```
    ┌───┐┌─┐
|0⟩─┤ H ├┤M├
    └───┘└╥┘
          ║
         50% |0⟩, 50% |1⟩
```

This circuit demonstrates quantum superposition and measurement collapse.

---

## 8.3.4 Multi-Qubit Gates

Multi-qubit gates create and manipulate **entanglement**, the defining feature of quantum computation that enables exponential speedups.

### CNOT Gate

**Definition 8.3.6** (Controlled-NOT Gate)
The **CNOT gate** (CX) acts on two qubits with matrix:

$$\text{CNOT} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$$

In basis $\{|00\rangle, |01\rangle, |10\rangle, |11\rangle\}$.

**Action**: $\text{CNOT}|a,b\rangle = |a, b \oplus a\rangle$ where $\oplus$ is XOR.

**Circuit Symbol:**
```
Control ────●────
            │
Target  ────⊕────
```

**Truth Table:**

| Input | Output |
|-------|--------|
| $\|00\rangle$ | $\|00\rangle$ |
| $\|01\rangle$ | $\|01\rangle$ |
| $\|10\rangle$ | $\|11\rangle$ |
| $\|11\rangle$ | $\|10\rangle$ |

**Example 8.3.6:** (CNOT gate applied to $|10\rangle$ and $|11\rangle$ via matrix multiplication)
Express the two-qubit states as 4-vectors: $|10\rangle = \begin{pmatrix} 0 \\ 0 \\ 1 \\ 0 \end{pmatrix}$ and $|11\rangle = \begin{pmatrix} 0 \\ 0 \\ 0 \\ 1 \end{pmatrix}$ (ordering: $|00\rangle, |01\rangle, |10\rangle, |11\rangle$).

$$\text{CNOT}|10\rangle = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix} \begin{pmatrix} 0 \\ 0 \\ 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ 0 \\ 1 \end{pmatrix} = |11\rangle$$

$$\text{CNOT}|11\rangle = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix} \begin{pmatrix} 0 \\ 0 \\ 0 \\ 1 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ 1 \\ 0 \end{pmatrix} = |10\rangle$$

When the control qubit is $|1\rangle$, the target qubit is flipped: $|10\rangle \to |11\rangle$ (target $0 \to 1$) and $|11\rangle \to |10\rangle$ (target $1 \to 0$), matching the XOR rule $|a, b \oplus a\rangle$.

**Creating Entanglement:**
```
    ┌───┐
|0⟩─┤ H ├───●──── |Φ⁺⟩ = (|00⟩ + |11⟩)/√2  (Bell state)
    └───┘   │
|0⟩─────────⊕────
```

**Theorem 8.3.2** (CNOT Creates Entanglement)
Applying CNOT to $(|0\rangle + |1\rangle)/\sqrt{2} \otimes |0\rangle$ produces the entangled Bell state:

$$\text{CNOT} \left( \frac{|0\rangle + |1\rangle}{\sqrt{2}} \otimes |0\rangle \right) = \frac{|00\rangle + |11\rangle}{\sqrt{2}}$$

This state cannot be written as a tensor product of single-qubit states.

**Example 8.3.9:** (Bell state $|\Phi^+\rangle$ creation, step by step)
Starting from $|00\rangle$, apply $H$ to the first qubit, then CNOT.

Step 1 -- Apply $H \otimes I$ to $|00\rangle$:

$$(H \otimes I)|00\rangle = H|0\rangle \otimes I|0\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}} \otimes |0\rangle = \frac{1}{\sqrt{2}}|00\rangle + \frac{1}{\sqrt{2}}|10\rangle$$

Step 2 -- Apply CNOT (control = first qubit, target = second):

$$\text{CNOT}\left(\frac{1}{\sqrt{2}}|00\rangle + \frac{1}{\sqrt{2}}|10\rangle\right) = \frac{1}{\sqrt{2}}\text{CNOT}|00\rangle + \frac{1}{\sqrt{2}}\text{CNOT}|10\rangle = \frac{1}{\sqrt{2}}|00\rangle + \frac{1}{\sqrt{2}}|11\rangle$$

The result $|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}$ is maximally entangled: measuring the first qubit as $|0\rangle$ instantly determines the second qubit is also $|0\rangle$, and vice versa.

### SWAP Gate

**Definition 8.3.7** (SWAP Gate)
The **SWAP gate** exchanges two qubits:

$$\text{SWAP} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

**Action**: $\text{SWAP}|a,b\rangle = |b,a\rangle$

**Circuit Symbol:**
```
────×────
    │
────×────
```

**Decomposition**: SWAP can be constructed from three CNOTs:
```
────●────⊕────●────
    │    │    │
────⊕────●────⊕────
```

### Toffoli Gate

**Definition 8.3.8** (Toffoli Gate)
The **Toffoli gate** (CCNOT) is a 3-qubit controlled-controlled-NOT:

$$\text{Toffoli}|a,b,c\rangle = |a, b, c \oplus (a \land b)\rangle$$

**Circuit Symbol:**
```
Control1 ────●────
             │
Control2 ────●────
             │
Target   ────⊕────
```

**Properties:**
- Classical universal gate (can implement NAND, hence all classical logic)
- Quantum universal with Hadamard
- Reversible version of classical AND gate (with ancilla)

### Controlled-U Gates

**Definition 8.3.9** (Controlled-U Gate)
For any single-qubit unitary $U$, the **controlled-U gate** is:

$$\text{C-U} = \begin{pmatrix} I & 0 \\ 0 & U \end{pmatrix} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & u_{00} & u_{01} \\ 0 & 0 & u_{10} & u_{11} \end{pmatrix}$$

**Circuit Symbol:**
```
Control ────●────
            │
         ┌──┴──┐
Target   ┤  U  ├
         └─────┘
```

**Action**: Applies $U$ to target if control is $|1\rangle$, identity otherwise.

**Example 8.3.10:** (Controlled-Z gate applied to $|1+\rangle$)
Let $U = Z$. The Controlled-Z (CZ) gate matrix is:

$$\text{CZ} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \end{pmatrix}$$

Apply CZ to $|1\rangle \otimes |+\rangle = |1\rangle \otimes \frac{|0\rangle + |1\rangle}{\sqrt{2}} = \frac{1}{\sqrt{2}}(|10\rangle + |11\rangle) = \frac{1}{\sqrt{2}}\begin{pmatrix} 0 \\ 0 \\ 1 \\ 1 \end{pmatrix}$:

$$\text{CZ}\frac{1}{\sqrt{2}}\begin{pmatrix} 0 \\ 0 \\ 1 \\ 1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 0 \\ 0 \\ 1 \\ -1 \end{pmatrix} = \frac{1}{\sqrt{2}}(|10\rangle - |11\rangle) = |1\rangle \otimes \frac{|0\rangle - |1\rangle}{\sqrt{2}} = |1\rangle \otimes |-\rangle$$

Since the control qubit is $|1\rangle$, $Z$ acts on the target, flipping $|+\rangle$ to $|-\rangle$. Note that CZ is symmetric: it does not matter which qubit is called "control."

---

## 8.3.5 Universal Gate Sets

**Definition 8.3.10** (Universal Gate Set)
A set of quantum gates is **universal** if any unitary operation on $n$ qubits can be approximated to arbitrary accuracy by a finite sequence of gates from the set.

**Theorem 8.3.3** (Universality of {H, T, CNOT})
The set $\{H, T, \text{CNOT}\}$ is universal for quantum computation. That is, any $n$-qubit unitary can be approximated to precision $\epsilon$ using $O(\text{poly}(n, \log(1/\epsilon)))$ gates from this set.

**Proof Sketch:**
1. Any single-qubit unitary can be approximated by products of $H$ and $T$ gates (Solovay-Kitaev)
2. CNOT generates entanglement between qubits
3. Combination of single-qubit and CNOT gates can approximate any multi-qubit unitary

**Alternative Universal Sets:**
- $\{H, S, T, \text{CNOT}\}$
- $\{R_y(\theta), R_z(\theta), \text{CNOT}\}$ for continuous $\theta$
- $\{\text{Toffoli}, H\}$
- Any entangling gate + single-qubit rotations

**Solovay-Kitaev Theorem** (Conceptual)

**Theorem 8.3.4** (Solovay-Kitaev)
Let $U$ be a single-qubit unitary and $\mathcal{G}$ a finite universal gate set. Then $U$ can be approximated to precision $\epsilon$ using $O(\log^c(1/\epsilon))$ gates from $\mathcal{G}$, where $c \approx 2$.

**Significance**:
- Efficient compilation from continuous rotations to discrete gates
- Polynomial overhead for approximation (not exponential)
- Enables fault-tolerant quantum computing with discrete gates

**Practical Implication**: To achieve error $\epsilon = 10^{-10}$, need $\sim 10^3$ gates, not $10^{10}$.

---

## 8.3.6 Gate Decomposition

Any multi-qubit unitary can be systematically decomposed into single-qubit gates and CNOTs.

**Theorem 8.3.5** (Two-Qubit Gate Decomposition)
Any two-qubit unitary $U$ can be decomposed as:

$$U = (A_1 \otimes B_1) \cdot \text{CNOT} \cdot (A_2 \otimes B_2) \cdot \text{CNOT} \cdot (A_3 \otimes B_3) \cdot \text{CNOT}$$

where $A_i, B_i$ are single-qubit unitaries.

**Circuit Form:**
```
    ┌────┐       ┌────┐       ┌────┐
────┤ A₁ ├───●───┤ A₂ ├───●───┤ A₃ ├────
    └────┘   │   └────┘   │   └────┘
    ┌────┐   │   ┌────┐   │   ┌────┐
────┤ B₁ ├───⊕───┤ B₂ ├───⊕───┤ B₃ ├────
    └────┘       └────┘       └────┘
```

**Corollary**: Any two-qubit gate requires at most 3 CNOTs (this is optimal for generic gates).

**Example 8.3.8:** (Compute $(H \otimes I)|00\rangle$ step by step)
First, build the $4 \times 4$ matrix $H \otimes I$ using the tensor (Kronecker) product:

$$H \otimes I = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} \otimes \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \cdot I & 1 \cdot I \\ 1 \cdot I & -1 \cdot I \end{pmatrix} = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 0 & -1 & 0 \\ 0 & 1 & 0 & -1 \end{pmatrix}$$

Now apply to $|00\rangle = \begin{pmatrix} 1 \\ 0 \\ 0 \\ 0 \end{pmatrix}$:

$$(H \otimes I)|00\rangle = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 0 \\ 1 \\ 0 \end{pmatrix} = \frac{1}{\sqrt{2}}|00\rangle + \frac{1}{\sqrt{2}}|10\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}} \otimes |0\rangle = |+\rangle \otimes |0\rangle$$

The Hadamard acts only on the first qubit while the second qubit remains $|0\rangle$, as expected from the tensor product structure.

### Single-Qubit Decomposition

**Theorem 8.3.6** (Euler Decomposition)
Any single-qubit unitary $U$ can be written as:

$$U = e^{i\alpha} R_z(\beta) R_y(\gamma) R_z(\delta)$$

for some real numbers $\alpha, \beta, \gamma, \delta$.

**Alternative Form (ZYZ Decomposition):**

$$U = \begin{pmatrix} e^{i\alpha} & 0 \\ 0 & e^{-i\alpha} \end{pmatrix} \begin{pmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{pmatrix} \begin{pmatrix} e^{i\phi} & 0 \\ 0 & e^{-i\phi} \end{pmatrix}$$

**Circuit:**
```
    ┌─────────┐┌─────────┐┌─────────┐
────┤ Rz(β)   ├┤ Ry(γ)   ├┤ Rz(δ)   ├──── (up to global phase)
    └─────────┘└─────────┘└─────────┘
```

---

## 8.3.7 Quantum Circuit Model

The **quantum circuit model** is the standard framework for quantum algorithm design, analogous to digital logic circuits in classical computing.

**Definition 8.3.11** (Quantum Circuit)
A **quantum circuit** on $n$ qubits is a sequence of quantum gates $U_1, U_2, \ldots, U_L$ applied to an initial state $|\psi_0\rangle$, producing final state:

$$|\psi_f\rangle = U_L \cdots U_2 U_1 |\psi_0\rangle$$

### Circuit Complexity Measures

**Definition 8.3.12** (Circuit Parameters)
For a quantum circuit:
- **Width**: Number of qubits $n$
- **Depth**: Length of longest path from input to output (time complexity)
- **Size**: Total number of gates (space-time complexity)

**Example Circuit with Annotations:**
```
        Depth = 4
    ┌───┐┌───┐   ┌───┐
q₀: ┤ H ├┤ T ├─●─┤ H ├──── Width = 3
    └───┘└───┘ │ └───┘
    ┌───┐      │
q₁: ┤ X ├──────┼────●──── Size = 7 gates
    └───┘      │    │
               │ ┌──┴──┐
q₂: ───────────⊕─┤  T  ├──
                 └─────┘
```

**Theorem 8.3.7** (Circuit Depth Lower Bounds)
For certain problems, circuit depth provides lower bounds on parallel quantum time complexity. For example, searching an unsorted database of size $N$ requires depth $\Omega(\sqrt{N})$ even with unlimited qubits.

### Parameterized Quantum Circuits

**Definition 8.3.13** (Parameterized Quantum Circuit)
A **parameterized quantum circuit** (PQC) is a circuit where gate parameters (typically rotation angles) are variables:

$$U(\boldsymbol{\theta}) = U_L(\theta_L) \cdots U_2(\theta_2) U_1(\theta_1)$$

**Example PQC:**
```
    ┌─────────┐       ┌─────────┐
|0⟩─┤ Ry(θ₁)  ├───●───┤ Ry(θ₃)  ├──
    └─────────┘   │   └─────────┘
    ┌─────────┐   │   ┌─────────┐
|0⟩─┤ Ry(θ₂)  ├───⊕───┤ Ry(θ₄)  ├──
    └─────────┘       └─────────┘
```

**Applications:**
- **Variational Quantum Eigensolver (VQE)**: Find ground state energy by optimizing $\langle \psi(\boldsymbol{\theta}) | H | \psi(\boldsymbol{\theta}) \rangle$
- **Quantum Approximate Optimization Algorithm (QAOA)**: Solve combinatorial optimization
- **Quantum Machine Learning**: Train quantum neural networks

> **ML Connection**: PQCs are the quantum analogue of neural networks. The parameters $\boldsymbol{\theta}$ are trained using classical optimization (gradient descent) based on measurement outcomes, similar to backpropagation in classical ML. The expressivity of PQCs and barren plateau problems mirror challenges in deep learning architecture design.

### Circuit Equivalence

**Definition 8.3.14** (Circuit Equivalence)
Two circuits are **equivalent** if they implement the same unitary transformation (up to global phase).

**Example Equivalence:**
```
H-gate decomposition:

    ┌───┐              ┌─────────┐┌───┐┌─────────┐
────┤ H ├────  ≡  ────┤ Ry(π/2) ├┤ Z ├┤ Ry(π/2) ├────
    └───┘              └─────────┘└───┘└─────────┘
```

**Example 8.3.7:** (Verify $HXH = Z$ by matrix multiplication)
This identity shows that conjugating $X$ by $H$ produces $Z$, reflecting the Hadamard's role as a basis-change gate.

Step 1 -- compute $XH$:

$$XH = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}$$

Step 2 -- compute $H(XH)$:

$$HXH = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} \cdot \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix} = \frac{1}{2} \begin{pmatrix} 2 & 0 \\ 0 & -2 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} = Z \; \checkmark$$

This means an $X$-gate sandwiched between two Hadamards is equivalent to a $Z$-gate: the Hadamard swaps the $X$ and $Z$ bases.

**Practical Importance**: Circuit optimization algorithms search for equivalent circuits with:
- Fewer gates (reduce errors)
- Shorter depth (reduce decoherence)
- Only native gates available on hardware

---

## Quantum-Classical Connection

**Quantum Circuits as Quantum "Neural Networks":**

| Classical Neural Networks | Quantum Circuits |
|---------------------------|------------------|
| Layers with weights | Gates with parameters $\theta$ |
| Activation functions | Fixed unitaries (H, CNOT) |
| Forward pass | Circuit execution |
| Backpropagation | Parameter shift rule for gradients |
| Training data | Measurement outcomes |
| Loss function | Expectation value $\langle O \rangle$ |

**Key Difference**: Quantum circuits are **linear** (unitary), while neural networks have **nonlinear** activations. Quantum advantage comes from exponential Hilbert space dimension and entanglement, not nonlinearity.

---

## Summary

**Key Concepts:**
1. Quantum gates are unitary matrices preserving quantum information
2. Single-qubit gates manipulate superposition and phase
3. Multi-qubit gates (especially CNOT) create entanglement
4. Universal gate sets enable arbitrary quantum computation
5. Quantum circuits organize gates into algorithms
6. Parameterized circuits bridge quantum computing and machine learning

**Essential Gates:**
- **Creating superposition**: Hadamard $H$
- **Creating entanglement**: CNOT
- **Universal computation**: $\{H, T, \text{CNOT}\}$
- **Variational algorithms**: Rotation gates $R_x, R_y, R_z$

**Circuit Design Principles:**
- Minimize depth (reduce decoherence time)
- Use native gate set (reduce compilation overhead)
- Balance expressivity vs. trainability (avoid barren plateaus)

---

## Exercises

**Exercise 8.3.1** (★): Verify that the Pauli $X$ gate is unitary by computing $XX^\dagger$.

**Exercise 8.3.2** (★): Show that $H = \frac{1}{\sqrt{2}}(X + Z)$ by explicit matrix multiplication.

**Exercise 8.3.3** (★★): Prove that $HXH = Z$ and $HZH = X$. What does this tell you about the Hadamard's role in changing bases?

**Exercise 8.3.4** (★★): Write the CNOT gate in the form $\text{CNOT} = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes X$ and verify equivalence to the standard matrix.

**Exercise 8.3.5** (★★): Show that applying $H$ to both qubits before and after CNOT produces a "controlled-Z" gate: $(H \otimes H) \cdot \text{CNOT} \cdot (H \otimes H) = \text{CZ}$.

**Exercise 8.3.6** (★★): Decompose the SWAP gate into three CNOT gates and verify by matrix multiplication.

**Exercise 8.3.7** (★★): For the rotation gate $R_z(\theta)$, verify that $R_z(\theta)R_z(\phi) = R_z(\theta + \phi)$ (up to global phase).

**Exercise 8.3.8** (★★★): Prove that any unitary $U \in SU(2)$ can be written as $U = e^{i\alpha} R_z(\beta)R_y(\gamma)R_z(\delta)$ by:
   1. Showing $U$ has trace $2\cos(\theta/2)e^{i\phi}$ for some $\theta, \phi$
   2. Constructing the decomposition from eigenvalues

**Exercise 8.3.9** (★★★): Show that the Toffoli gate can implement the classical NAND gate: $\text{NAND}(a,b) = \neg(a \land b)$. Why does this prove Toffoli is classically universal?

**Exercise 8.3.10** (★★★): **Bell State Creation**: Draw the quantum circuit to create all four Bell states:
   - $|\Phi^+\rangle = (|00\rangle + |11\rangle)/\sqrt{2}$
   - $|\Phi^-\rangle = (|00\rangle - |11\rangle)/\sqrt{2}$
   - $|\Psi^+\rangle = (|01\rangle + |10\rangle)/\sqrt{2}$
   - $|\Psi^-\rangle = (|01\rangle - |10\rangle)/\sqrt{2}$

**Exercise 8.3.11** (★★★): **Circuit Depth Analysis**: Consider the quantum Fourier transform on $n$ qubits, which has $O(n^2)$ gates. What is its circuit depth? Can it be parallelized?

**Exercise 8.3.12** (★★★): **Parameterized Circuit**: Design a 2-qubit parameterized circuit $U(\theta_1, \theta_2)$ that can represent any state in the form $\cos\theta|00\rangle + \sin\theta e^{i\phi}|11\rangle$. How many parameters are needed?

---

## Related Topics

- **§8.1 Quantum States**: Foundation for understanding gate actions
- **§8.2 Quantum Measurement**: Outcome of circuit execution
- **§8.4 Quantum Algorithms**: Applications of circuits (Grover, Shor, VQE)
- **§8.5 Quantum Entanglement**: Created by multi-qubit gates
- **§8.6 Quantum Error Correction**: Protecting circuits from noise
- **§9.3 Variational Quantum Algorithms**: Optimizing parameterized circuits for ML
- **§2.5 Tensor Products**: Mathematical structure of multi-qubit gates

---

## Further Reading

1. Nielsen & Chuang, *Quantum Computation and Quantum Information* (2010) — Chapter 4
2. Barenco et al., "Elementary gates for quantum computation," *Phys. Rev. A* (1995) — Universal gates
3. Dawson & Nielsen, "The Solovay-Kitaev algorithm," *Quantum Inf. Comput.* (2006)
4. Schuld & Petruccione, *Machine Learning with Quantum Computers* (2021) — Chapter 6
5. Cerezo et al., "Variational quantum algorithms," *Nat. Rev. Phys.* (2021) — PQCs in ML

---

*Last updated: 2026-02-16*
# Chapter 8.6: Density Matrices

## Prerequisites

- **Linear Algebra:** Hermitian operators, trace, eigenvalues/eigenvectors (Chapter 2)
- **Quantum States:** Pure states, ket notation, tensor products (Chapter 8.1-8.3)
- **Probability Theory:** Classical probability distributions, entropy (Chapter 5)
- **Operator Theory:** Positive semi-definite operators, spectral theorem (Chapter 3.4)

## Introduction

In the pure state formalism, quantum systems are described by normalized state vectors $|\psi\rangle \in \mathcal{H}$. However, this framework is insufficient for describing:

1. **Statistical Mixtures:** When we have classical uncertainty about which quantum state the system is in
2. **Open Quantum Systems:** Subsystems entangled with an environment
3. **Incomplete Knowledge:** Situations where we only have partial information about preparation
4. **Measurement Statistics:** Post-measurement ensembles with classical probabilities

The density matrix formalism extends quantum mechanics to handle these scenarios, providing a complete description of both pure and mixed quantum states.

---

## 8.6.1 Motivation: Beyond Pure States

### Classical vs Quantum Uncertainty

```
Pure State (Quantum Only)              Mixed State (Classical + Quantum)
=====================                  ================================

     |ψ⟩                               p₁ · |ψ₁⟩  (with probability p₁)
      ↓                                p₂ · |ψ₂⟩  (with probability p₂)
Coherent superposition                 p₃ · |ψ₃⟩  (with probability p₃)
α|0⟩ + β|1⟩                                  ↓
                                       Statistical mixture
Example:
  |+⟩ = (|0⟩ + |1⟩)/√2                Example:
  Definite quantum state                50% |0⟩ + 50% |1⟩ (classical mix)
  Interference possible                 ≠ |+⟩ (no interference)
```

**Key Distinction:** The superposition $|\psi\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$ exhibits quantum interference, while the classical mixture "prepare $|0\rangle$ with 50% probability or $|1\rangle$ with 50% probability" does not.

### Why Pure States Are Insufficient

**Example 8.6.1** (Thermal State)
A quantum system at temperature $T$ doesn't occupy a single energy eigenstate. Instead, it's a statistical mixture of eigenstates $|E_n\rangle$ with Boltzmann probabilities:

$$p_n = \frac{e^{-E_n/k_B T}}{Z}, \quad Z = \sum_n e^{-E_n/k_B T}$$

No single pure state $|\psi\rangle$ can represent this situation.

**Example 8.6.2** (Entanglement and Subsystems)
Consider the Bell state $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$ shared between Alice and Bob. If Alice only has access to her qubit, what is *her* state?

- Not $|0\rangle$ (she has 50% chance of $|0\rangle$ or $|1\rangle$)
- Not $|+\rangle$ (no coherent superposition from her perspective)
- Requires a *mixed state* description

---

## 8.6.2 Definition and Properties

**Definition 8.6.1** (Density Matrix/Density Operator)
A **density matrix** $\rho$ on a Hilbert space $\mathcal{H}$ is a linear operator satisfying:

1. **Hermiticity:** $\rho = \rho^\dagger$
2. **Positive Semi-Definiteness:** $\rho \geq 0$ (i.e., $\langle\psi|\rho|\psi\rangle \geq 0$ for all $|\psi\rangle$)
3. **Trace Normalization:** $\text{tr}(\rho) = 1$

**Physical Interpretation:** The operator $\rho$ encodes all measurable properties of the quantum state, whether pure or mixed.

**Worked Example 8.6.1:** (Pure State Density Matrix for $|+\rangle$)
Compute $\rho = |\psi\rangle\langle\psi|$ for $|\psi\rangle = |+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$:

$$\rho = |+\rangle\langle +| = \frac{1}{2}\begin{pmatrix}1\\1\end{pmatrix}\begin{pmatrix}1&1\end{pmatrix} = \frac{1}{2}\begin{pmatrix}1&1\\1&1\end{pmatrix}$$

**Verify the three axioms:**

- Hermiticity: $\rho^\dagger = \frac{1}{2}\begin{pmatrix}1&1\\1&1\end{pmatrix} = \rho$ ✓
- Positive semi-definite: eigenvalues are $\lambda = 1, 0$ (both $\geq 0$) ✓
- Trace: $\text{tr}(\rho) = \frac{1}{2}(1 + 1) = 1$ ✓

### Ensemble Representation

**Definition 8.6.2** (Quantum Ensemble)
A density matrix can be expressed as a **convex combination** of pure states:

$$\rho = \sum_{i=1}^{n} p_i |\psi_i\rangle\langle\psi_i|$$

where $p_i \geq 0$ and $\sum_i p_i = 1$ (classical probabilities), and $\langle\psi_i|\psi_i\rangle = 1$ (normalized quantum states).

**Remark:** This decomposition is **not unique**. The same $\rho$ can be realized by infinitely many different ensembles $\{(p_i, |\psi_i\rangle)\}$.

**Worked Example 8.6.2:** (Constructing a Mixed State)
Build the equal mixture $\rho = \frac{1}{2}|0\rangle\langle 0| + \frac{1}{2}|1\rangle\langle 1|$:

$$\rho = \frac{1}{2}\begin{pmatrix}1&0\\0&0\end{pmatrix} + \frac{1}{2}\begin{pmatrix}0&0\\0&1\end{pmatrix} = \begin{pmatrix}\frac{1}{2}&0\\0&\frac{1}{2}\end{pmatrix} = \frac{1}{2}I$$

**Verify:** $\text{tr}(\rho) = \frac{1}{2} + \frac{1}{2} = 1$ ✓, eigenvalues $= \frac{1}{2}, \frac{1}{2}$ (both $\geq 0$) ✓.

Note: this is $\frac{1}{2}I$, the maximally mixed qubit state -- contrast with $|+\rangle\langle+|$ from Worked Example 8.6.1, which has off-diagonal elements. Same measurement probabilities for $Z$-basis, but completely different coherence properties.

**Example 8.6.3** (Non-Uniqueness)
The maximally mixed qubit state $\rho = \frac{1}{2}I$ can be written as:

$$\begin{align}
\rho &= \frac{1}{2}|0\rangle\langle 0| + \frac{1}{2}|1\rangle\langle 1| \\
&= \frac{1}{2}|+\rangle\langle +| + \frac{1}{2}|-\rangle\langle -| \\
&= \frac{1}{2}|+i\rangle\langle +i| + \frac{1}{2}|-i\rangle\langle -i|
\end{align}$$

where $|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$, $|-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$, etc. All represent the same physics.

### Properties of Density Matrices

**Theorem 8.6.1** (Spectral Decomposition)
Every density matrix $\rho$ admits a spectral decomposition:

$$\rho = \sum_{i=1}^{d} \lambda_i |e_i\rangle\langle e_i|$$

where $\lambda_i \geq 0$ are eigenvalues, $\sum_i \lambda_i = 1$, and $\{|e_i\rangle\}$ is an orthonormal basis.

**Proof:** By the spectral theorem for Hermitian operators, $\rho$ is diagonalizable with real eigenvalues. Positive semi-definiteness ensures $\lambda_i \geq 0$, and trace normalization gives $\sum_i \lambda_i = 1$. □

**Theorem 8.6.2** (Expectation Values)
For an observable $A$, the expectation value in state $\rho$ is:

$$\langle A \rangle = \text{tr}(A\rho)$$

**Proof:** For an ensemble $\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$:

$$\text{tr}(A\rho) = \sum_i p_i \text{tr}(A|\psi_i\rangle\langle\psi_i|) = \sum_i p_i \langle\psi_i|A|\psi_i\rangle$$

This is the classical average of quantum expectation values, as expected. □

**Theorem 8.6.3** (Born Rule for Density Matrices)
The probability of measuring outcome $m$ (projector $P_m$) is:

$$\text{Pr}(m) = \text{tr}(P_m \rho)$$

---

## 8.6.3 Pure vs Mixed States

**Definition 8.6.3** (Pure and Mixed States)
- A state is **pure** if $\rho = |\psi\rangle\langle\psi|$ for some $|\psi\rangle$
- A state is **mixed** if it is not pure (genuine statistical mixture)

### Purity Criteria

**Theorem 8.6.4** (Purity Characterization)
The following are equivalent:

1. $\rho$ is pure
2. $\rho^2 = \rho$ (idempotent)
3. $\text{tr}(\rho^2) = 1$
4. $\rho$ has rank 1 (only one nonzero eigenvalue)
5. $S(\rho) = -\text{tr}(\rho \log \rho) = 0$ (zero entropy)

**Proof Sketch:**
- $(1 \Rightarrow 2)$: If $\rho = |\psi\rangle\langle\psi|$, then $\rho^2 = |\psi\rangle\langle\psi|\psi\rangle\langle\psi| = |\psi\rangle\langle\psi| = \rho$
- $(2 \Rightarrow 3)$: $\text{tr}(\rho^2) = \text{tr}(\rho) = 1$
- $(3 \Rightarrow 1)$: If $\rho = \sum_i \lambda_i |e_i\rangle\langle e_i|$, then $\text{tr}(\rho^2) = \sum_i \lambda_i^2 = 1$. Since $\sum_i \lambda_i = 1$ and $\lambda_i \geq 0$, Cauchy-Schwarz gives equality iff exactly one $\lambda_i = 1$. □

**Definition 8.6.4** (Purity)
The **purity** of a state is:

$$\mathcal{P}(\rho) = \text{tr}(\rho^2) \in [\frac{1}{d}, 1]$$

where $d = \dim(\mathcal{H})$. Bounds: $\mathcal{P} = 1$ (pure), $\mathcal{P} = 1/d$ (maximally mixed).

**Worked Example 8.6.3:** (Purity: Pure vs Mixed)
Compare purity for the pure state $\rho_{\text{pure}} = |+\rangle\langle+|$ and the mixed state $\rho_{\text{mix}} = \frac{1}{2}I$:

*Pure state:* $\rho_{\text{pure}}^2 = \frac{1}{4}\begin{pmatrix}1&1\\1&1\end{pmatrix}\begin{pmatrix}1&1\\1&1\end{pmatrix} = \frac{1}{4}\begin{pmatrix}2&2\\2&2\end{pmatrix} = \frac{1}{2}\begin{pmatrix}1&1\\1&1\end{pmatrix} = \rho_{\text{pure}}$

$$\mathcal{P}(\rho_{\text{pure}}) = \text{tr}(\rho_{\text{pure}}^2) = \text{tr}(\rho_{\text{pure}}) = 1 \quad \text{(pure)} ✓$$

*Mixed state:* $\rho_{\text{mix}}^2 = \frac{1}{4}I$

$$\mathcal{P}(\rho_{\text{mix}}) = \text{tr}\!\left(\frac{1}{4}I\right) = \frac{1}{4}\cdot 2 = \frac{1}{2} = \frac{1}{d} \quad \text{(maximally mixed)} ✓$$

### Bloch Ball Representation (Qubits)

For a single qubit ($d=2$), any density matrix can be written:

$$\rho = \frac{1}{2}(I + \vec{r} \cdot \vec{\sigma})$$

where $\vec{r} = (r_x, r_y, r_z) \in \mathbb{R}^3$ is the **Bloch vector** and $\vec{\sigma} = (\sigma_x, \sigma_y, \sigma_z)$ are Pauli matrices.

**Theorem 8.6.5** (Bloch Ball)
The Bloch vector $\vec{r}$ satisfies $|\vec{r}| \leq 1$:

- $|\vec{r}| = 1$: Pure state (surface of Bloch sphere)
- $|\vec{r}| < 1$: Mixed state (interior of Bloch ball)
- $\vec{r} = 0$: Maximally mixed state $\rho = \frac{1}{2}I$ (center)

```
Bloch Ball (Qubit Density Matrices)
====================================

           z (|0⟩)
           ↑
           |     • |ψ⟩ (pure, |r|=1)
           |    /|
           |   / |
           |  /  |
           | /   |
           |/____|____→ y
          /|     |
         / |  •  |  ρ (mixed, |r|<0.5)
        /  |     |
       ↙   |     |
      x    ↓
         |1⟩

Pure states:     Surface (Bloch sphere)
Mixed states:    Interior (Bloch ball)
Maximally mixed: Origin (r = 0)
```

**Purity in Bloch representation:**

$$\mathcal{P}(\rho) = \text{tr}(\rho^2) = \frac{1}{2}(1 + |\vec{r}|^2)$$

**Worked Example 8.6.4:** (Bloch Sphere for a Mixed State)
Consider the state with Bloch vector $\vec{r} = (0, 0, \frac{1}{2})$ (pointing halfway to $|0\rangle$):

$$\rho = \frac{1}{2}(I + \tfrac{1}{2}\sigma_z) = \frac{1}{2}\left[\begin{pmatrix}1&0\\0&1\end{pmatrix} + \frac{1}{2}\begin{pmatrix}1&0\\0&-1\end{pmatrix}\right] = \begin{pmatrix}\frac{3}{4}&0\\0&\frac{1}{4}\end{pmatrix}$$

This is a **mixed state** biased toward $|0\rangle$: measuring in the $Z$-basis gives $|0\rangle$ with probability $\frac{3}{4}$.

**Purity:** $\mathcal{P} = \frac{1}{2}(1 + |\vec{r}|^2) = \frac{1}{2}(1 + \frac{1}{4}) = \frac{5}{8}$. Since $\frac{1}{2} < \frac{5}{8} < 1$, the state is mixed but not maximally mixed.

---

## 8.6.4 Partial Trace and Reduced Density Matrices

### Motivation: Subsystem States

For composite systems $\mathcal{H} = \mathcal{H}_A \otimes \mathcal{H}_B$, we often need to describe subsystem $A$ alone when only local measurements are possible.

**Definition 8.6.5** (Partial Trace)
The **partial trace** over subsystem $B$ is a linear map $\text{tr}_B: \mathcal{L}(\mathcal{H}_A \otimes \mathcal{H}_B) \to \mathcal{L}(\mathcal{H}_A)$ defined by:

$$\text{tr}_B(|a_1\rangle\langle a_2| \otimes |b_1\rangle\langle b_2|) = |a_1\rangle\langle a_2| \cdot \langle b_2|b_1\rangle$$

Extended by linearity to all operators.

**Operational Definition:** For any operator $M_A$ on subsystem $A$ and density matrix $\rho_{AB}$:

$$\text{tr}(M_A \rho_A) = \text{tr}((M_A \otimes I_B) \rho_{AB})$$

where $\rho_A = \text{tr}_B(\rho_{AB})$ is the **reduced density matrix**.

### Computing Partial Traces

**Theorem 8.6.6** (Partial Trace in Basis)
If $\{|i\rangle_A\}$ and $\{|j\rangle_B\}$ are orthonormal bases, then:

$$\text{tr}_B(\rho_{AB}) = \sum_{j} (I_A \otimes \langle j|_B) \rho_{AB} (I_A \otimes |j\rangle_B)$$

**Example 8.6.4** (Bell State Reduced Density Matrix)
For the Bell state $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$:

$$\rho_{AB} = |\Phi^+\rangle\langle\Phi^+| = \frac{1}{2}(|00\rangle\langle 00| + |00\rangle\langle 11| + |11\rangle\langle 00| + |11\rangle\langle 11|)$$

Tracing out $B$:

$$\begin{align}
\rho_A &= \text{tr}_B(\rho_{AB}) \\
&= \frac{1}{2}\left[\langle 0|0\rangle \cdot |0\rangle\langle 0| + \langle 1|1\rangle \cdot |0\rangle\langle 0| + \langle 0|0\rangle \cdot |1\rangle\langle 1| + \langle 1|1\rangle \cdot |1\rangle\langle 1|\right] \\
&= \frac{1}{2}(|0\rangle\langle 0| + |1\rangle\langle 1|) = \frac{1}{2}I
\end{align}$$

**Interpretation:** Although the joint state is pure ($\mathcal{P}(\rho_{AB}) = 1$), Alice's local state is maximally mixed ($\mathcal{P}(\rho_A) = 1/2$). This is the hallmark of entanglement.

**Worked Example 8.6.5:** (Partial Trace for a Product State)
Consider the separable 2-qubit state $|\psi\rangle = |0\rangle \otimes |+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |01\rangle)$:

$$\rho_{AB} = |\psi\rangle\langle\psi| = \frac{1}{2}(|00\rangle + |01\rangle)(\langle 00| + \langle 01|)$$

$$= \frac{1}{2}(|00\rangle\langle 00| + |00\rangle\langle 01| + |01\rangle\langle 00| + |01\rangle\langle 01|)$$

Tracing out $B$: apply $\sum_j (I_A \otimes \langle j|_B) \rho_{AB} (I_A \otimes |j\rangle_B)$ with $j \in \{0,1\}$:

$$\rho_A = \frac{1}{2}(\langle 0|0\rangle|0\rangle\langle 0| + \langle 1|0\rangle|0\rangle\langle 0| + \langle 0|1\rangle|0\rangle\langle 0| + \langle 1|1\rangle|0\rangle\langle 0|) = \frac{1}{2}\cdot 2 \cdot |0\rangle\langle 0| = |0\rangle\langle 0|$$

Result: $\rho_A = |0\rangle\langle 0|$ is **pure** -- as expected for a product state, the subsystem retains its purity. Compare with Example 8.6.4 where entanglement caused $\rho_A$ to be maximally mixed.

**Theorem 8.6.7** (Properties of Partial Trace)

1. **Linearity:** $\text{tr}_B(\alpha \rho + \beta \sigma) = \alpha \text{tr}_B(\rho) + \beta \text{tr}_B(\sigma)$
2. **Trace Preservation:** $\text{tr}(\text{tr}_B(\rho)) = \text{tr}(\rho)$
3. **Product States:** $\text{tr}_B(\rho_A \otimes \rho_B) = \rho_A \cdot \text{tr}(\rho_B)$
4. **Purity Decrease:** $\mathcal{P}(\text{tr}_B(\rho)) \leq \mathcal{P}(\rho)$ (tracing out can only mix)

---

## 8.6.5 Von Neumann Entropy

### Definition and Properties

**Definition 8.6.6** (Von Neumann Entropy)
The **von Neumann entropy** of a density matrix $\rho$ is:

$$S(\rho) = -\text{tr}(\rho \log \rho) = -\sum_i \lambda_i \log \lambda_i$$

where $\lambda_i$ are eigenvalues of $\rho$, and $\log$ is base 2 (measured in bits or qubits).

**Convention:** $0 \log 0 = 0$ (by continuity).

**Worked Example 8.6.6:** (Von Neumann Entropy of Maximally Mixed Qubit)
Compute $S(\rho)$ for the maximally mixed qubit $\rho = \frac{1}{2}I$ with eigenvalues $\lambda_1 = \lambda_2 = \frac{1}{2}$:

$$S(\rho) = -\sum_i \lambda_i \log_2 \lambda_i = -\frac{1}{2}\log_2\frac{1}{2} - \frac{1}{2}\log_2\frac{1}{2} = -\frac{1}{2}(-1) - \frac{1}{2}(-1) = 1 \text{ bit}$$

This matches the upper bound $S \leq \log_2 d = \log_2 2 = 1$, confirming maximal uncertainty. Compare with a pure state $\rho = |0\rangle\langle 0|$ which has eigenvalues $1, 0$, giving $S = -1 \cdot \log_2 1 - 0 \cdot \log_2 0 = 0$ (zero entropy, complete knowledge).

**Theorem 8.6.8** (Entropy Properties)

1. **Non-negativity:** $S(\rho) \geq 0$, with equality iff $\rho$ is pure
2. **Upper Bound:** $S(\rho) \leq \log d$, with equality iff $\rho = \frac{1}{d}I$ (maximally mixed)
3. **Concavity:** $S(\sum_i p_i \rho_i) \geq \sum_i p_i S(\rho_i)$ (mixing increases entropy)
4. **Unitary Invariance:** $S(U\rho U^\dagger) = S(\rho)$ for unitary $U$
5. **Subadditivity:** $S(\rho_{AB}) \leq S(\rho_A) + S(\rho_B)$ (correlations reduce total entropy)

**Connection to Shannon Entropy:** For a classical probability distribution embedded as $\rho = \sum_i p_i |i\rangle\langle i|$ (diagonal):

$$S(\rho) = H(p) = -\sum_i p_i \log p_i$$

Thus von Neumann entropy generalizes Shannon entropy to quantum states.

### Information-Theoretic Interpretation

```
Entropy Hierarchy
==================

Classical System              Quantum System
─────────────────            ─────────────────
Shannon Entropy H(X)  ──→    Von Neumann Entropy S(ρ)

Measures:                     Measures:
- Classical uncertainty       - Quantum + classical uncertainty
- Information content         - Mixedness of state
                              - Entanglement (indirectly)

Pure state:    H = 0          Pure state:    S = 0
Max mixed:     H = log d      Max mixed:     S = log d
```

**Example 8.6.5** (Entropy of Qubit)
For $\rho = \frac{1}{2}(I + \vec{r} \cdot \vec{\sigma})$ with $|\vec{r}| = r$:

$$S(\rho) = h\left(\frac{1+r}{2}\right)$$

where $h(p) = -p \log p - (1-p)\log(1-p)$ is the binary entropy function.

- $r = 1$ (pure): $S = 0$
- $r = 0$ (maximally mixed): $S = 1$ bit

### Entanglement and Entropy

**Theorem 8.6.9** (Entropy of Entangled States)
For a pure state $|\psi\rangle_{AB}$ on $\mathcal{H}_A \otimes \mathcal{H}_B$:

$$S(\rho_A) = S(\rho_B)$$

where $\rho_A = \text{tr}_B(|\psi\rangle\langle\psi|)$ and $\rho_B = \text{tr}_A(|\psi\rangle\langle\psi|)$.

**Proof:** Schmidt decomposition ensures both reduced density matrices share the same eigenvalues. □

**Definition 8.6.7** (Entanglement Entropy)
For a pure bipartite state, the **entanglement entropy** is $S(\rho_A) = S(\rho_B)$, quantifying entanglement between $A$ and $B$.

**Example 8.6.6** (Bell State Entanglement)
For $|\Phi^+\rangle$, we computed $\rho_A = \frac{1}{2}I$, so:

$$S(\rho_A) = 1 \text{ qubit (maximally entangled)}$$

---

## 8.6.6 Quantum Channels

### Motivation: Evolution of Open Systems

Closed systems evolve unitarily: $\rho(t) = U(t) \rho(0) U(t)^\dagger$. But real quantum systems interact with environments (decoherence, noise), requiring a broader framework.

**Definition 8.6.8** (Quantum Channel)
A **quantum channel** is a linear map $\mathcal{E}: \mathcal{L}(\mathcal{H}_{\text{in}}) \to \mathcal{L}(\mathcal{H}_{\text{out}})$ that is:

1. **Trace-Preserving (TP):** $\text{tr}(\mathcal{E}(\rho)) = \text{tr}(\rho)$ for all $\rho$
2. **Completely Positive (CP):** $(\mathcal{E} \otimes \mathcal{I}_n)(\rho) \geq 0$ for all $n$ and all $\rho \geq 0$

**Interpretation:** CP ensures physical validity when the system is entangled with an ancilla. TP ensures probability conservation.

### Kraus Representation

**Theorem 8.6.10** (Kraus Representation Theorem)
A map $\mathcal{E}$ is a quantum channel iff there exist operators $\{E_k\}$ (Kraus operators) such that:

$$\mathcal{E}(\rho) = \sum_{k} E_k \rho E_k^\dagger$$

with the completeness relation:

$$\sum_k E_k^\dagger E_k = I$$

**Remark:** The decomposition $\{E_k\}$ is not unique; different Kraus representations can describe the same channel.

**Example 8.6.7** (Bit Flip Channel)
With probability $p$, apply $X$ (bit flip); otherwise, do nothing:

$$\mathcal{E}_{\text{BF}}(\rho) = (1-p)\rho + p X\rho X$$

Kraus operators: $E_0 = \sqrt{1-p} \cdot I$, $E_1 = \sqrt{p} \cdot X$.

Check: $E_0^\dagger E_0 + E_1^\dagger E_1 = (1-p)I + pI = I$ ✓

**Worked Example 8.6.7:** (Applying the Bit-Flip Channel)
Apply the bit-flip channel with $p = 0.3$ to the pure state $\rho = |0\rangle\langle 0| = \begin{pmatrix}1&0\\0&0\end{pmatrix}$:

$$\mathcal{E}(\rho) = 0.7 \cdot \rho + 0.3 \cdot X\rho X = 0.7\begin{pmatrix}1&0\\0&0\end{pmatrix} + 0.3\begin{pmatrix}0&1\\1&0\end{pmatrix}\begin{pmatrix}1&0\\0&0\end{pmatrix}\begin{pmatrix}0&1\\1&0\end{pmatrix}$$

$$= 0.7\begin{pmatrix}1&0\\0&0\end{pmatrix} + 0.3\begin{pmatrix}0&0\\0&1\end{pmatrix} = \begin{pmatrix}0.7&0\\0&0.3\end{pmatrix}$$

The output is a **mixed state** with $\text{tr}(\rho'^2) = 0.7^2 + 0.3^2 = 0.58 < 1$. The channel introduced classical uncertainty: 70% chance the qubit is still $|0\rangle$, 30% chance it flipped to $|1\rangle$.

**Example 8.6.8** (Phase Damping)
Random phase flips (dephasing):

$$\mathcal{E}_{\text{PD}}(\rho) = (1-p)\rho + p Z\rho Z$$

This destroys off-diagonal coherences in the $Z$-basis:

$$\rho = \begin{pmatrix} \rho_{00} & \rho_{01} \\ \rho_{10} & \rho_{11} \end{pmatrix} \quad \longrightarrow \quad \mathcal{E}_{\text{PD}}(\rho) = \begin{pmatrix} \rho_{00} & (1-2p)\rho_{01} \\ (1-2p)\rho_{10} & \rho_{11} \end{pmatrix}$$

**Example 8.6.9** (Amplitude Damping)
Models spontaneous emission ($|1\rangle \to |0\rangle$ with rate $\gamma$):

$$E_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad E_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$

This is non-unital: $\mathcal{E}(I) \neq I$ (breaks time-reversal symmetry).

### Stinespring Dilation

**Theorem 8.6.11** (Stinespring Dilation)
Every quantum channel $\mathcal{E}$ can be represented as:

$$\mathcal{E}(\rho) = \text{tr}_E(U (\rho \otimes |e_0\rangle\langle e_0|_E) U^\dagger)$$

for some environment Hilbert space $\mathcal{H}_E$, initial state $|e_0\rangle_E$, and unitary $U$ on $\mathcal{H} \otimes \mathcal{H}_E$.

**Interpretation:** Every noisy evolution is equivalent to unitary evolution on a larger system followed by tracing out the environment.

```
Stinespring Dilation Picture
==============================

System + Environment (pure evolution)
─────────────────────────────────────
    ρ ⊗ |e₀⟩⟨e₀|
         ↓  U (unitary)
    ρ'(system+env)
         ↓  Trace out environment
      ℰ(ρ) (mixed state on system)

Key Insight: Noise = Entanglement with environment
```

---

## 8.6.7 Decoherence in the Density Matrix Formalism

### Mechanism of Decoherence

Decoherence describes how quantum superpositions evolve into classical mixtures due to environmental entanglement.

**Setup:** System starts in superposition $|\psi\rangle_S = \alpha|0\rangle + \beta|1\rangle$, environment in $|E_0\rangle$.

**Evolution:** Entangling interaction creates:

$$|\Psi\rangle_{SE} = \alpha|0\rangle|E_0\rangle + \beta|1\rangle|E_1\rangle$$

where $|E_0\rangle$ and $|E_1\rangle$ are (nearly) orthogonal environment states.

**Reduced Density Matrix:**

$$\begin{align}
\rho_S &= \text{tr}_E(|\Psi\rangle\langle\Psi|) \\
&= |\alpha|^2 |0\rangle\langle 0| + |\beta|^2 |1\rangle\langle 1| + \alpha\beta^* \langle E_1|E_0\rangle |0\rangle\langle 1| + \alpha^*\beta \langle E_0|E_1\rangle |1\rangle\langle 0|
\end{align}$$

**Key Effect:** Off-diagonal terms (coherences) are suppressed by the overlap $\langle E_1|E_0\rangle \approx 0$.

### Decoherence Dynamics

For continuous dephasing, the density matrix evolves as:

$$\rho(t) = \begin{pmatrix} \rho_{00} & e^{-\Gamma t} \rho_{01}(0) \\ e^{-\Gamma t} \rho_{10}(0) & \rho_{11} \end{pmatrix}$$

where $\Gamma$ is the decoherence rate.

```
Decoherence Evolution
======================

t = 0                  t = τ_decoherence        t → ∞
─────                  ─────────────────        ──────

ρ₀₀  ρ₀₁               ρ₀₀  ρ₀₁·e^(-1)          ρ₀₀   0
                 →                         →
ρ₁₀  ρ₁₁               ρ₁₀·e^(-1)  ρ₁₁           0   ρ₁₁

Coherent state         Partially decohered       Classical mixture
(pure/entangled)       (mixed)                   (diagonal, incoherent)
```

**Timescale:** Decoherence time $\tau_{\text{dec}} = 1/\Gamma$ is typically much shorter than relaxation times, explaining the classical world's emergence.

**Example 8.6.10** (Pointer States)
States that commute with the system-environment interaction Hamiltonian (e.g., energy eigenstates) don't decohere:

$$[H_{\text{int}}, |n\rangle\langle n|] = 0 \implies \text{no decoherence}$$

These "pointer states" form the preferred basis of classical observation.

### Quantum-to-Classical Transition

**Theorem 8.6.12** (Classicalization by Decoherence)
Under Markovian decoherence in the $Z$-basis:

$$\lim_{t \to \infty} \rho(t) = \sum_n \langle n|\rho(0)|n\rangle |n\rangle\langle n|$$

The asymptotic state is diagonal (classical) in the pointer basis $\{|n\rangle\}$.

**Quantum ML Connection:** Decoherence limits coherent quantum advantage. Quantum machine learning algorithms must complete before $\tau_{\text{dec}}$ or employ error correction.

---

## Key Takeaways

1. **Density matrices** unify pure and mixed states, essential for open quantum systems
2. **Purity** $\text{tr}(\rho^2)$ distinguishes pure ($=1$) from mixed ($<1$) states
3. **Partial trace** gives reduced density matrices for subsystems, revealing entanglement
4. **Von Neumann entropy** quantifies mixedness and entanglement
5. **Quantum channels** (CPTP maps) model noisy evolution via Kraus operators
6. **Decoherence** explains quantum-to-classical transition as off-diagonal decay
7. **Bloch ball** (not just sphere) visualizes all qubit states, pure and mixed

---

## Exercises

### ★ Basic Understanding

**Exercise 8.6.1**
Show that the maximally mixed state $\rho = \frac{1}{d}I$ satisfies all density matrix axioms.

**Exercise 8.6.2**
Compute $\text{tr}(\rho^2)$ for $\rho = \frac{1}{2}|0\rangle\langle 0| + \frac{1}{2}|1\rangle\langle 1|$ and $\rho = |+\rangle\langle +|$. Which is pure?

**Exercise 8.6.3**
For the Werner state $\rho_W(p) = p|\Phi^+\rangle\langle\Phi^+| + (1-p)\frac{I}{4}$, find the range of $p$ for which $\rho_W$ is pure.

**Exercise 8.6.4**
Verify that $E_0 = \sqrt{1-p} \cdot I$ and $E_1 = \sqrt{p} \cdot X$ satisfy the completeness relation for a quantum channel.

### ★★ Intermediate Problems

**Exercise 8.6.5**
Compute the reduced density matrix $\rho_A$ for the state $|\psi\rangle = \frac{1}{\sqrt{3}}(|00\rangle + |01\rangle + |11\rangle)$ and determine its purity.

**Exercise 8.6.6**
Show that the von Neumann entropy satisfies $S(U\rho U^\dagger) = S(\rho)$ for any unitary $U$.

**Exercise 8.6.7**
For the depolarizing channel $\mathcal{E}_{\text{dep}}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$, find Kraus operators.

**Exercise 8.6.8**
Prove that if $\rho_A = \text{tr}_B(|\psi\rangle\langle\psi|)$ is pure, then $|\psi\rangle = |\phi\rangle_A \otimes |\chi\rangle_B$ (product state).

**Exercise 8.6.9**
Show that the partial trace is the unique map satisfying $\text{tr}(M_A \rho_A) = \text{tr}((M_A \otimes I_B)\rho_{AB})$ for all $M_A$.

### ★★★ Advanced Challenges

**Exercise 8.6.10**
Prove the **Araki-Lieb inequality**: For a bipartite pure state $|\psi\rangle_{AB}$,

$$|S(\rho_A) - S(\rho_B)| \leq S(\rho_{AB}) \leq S(\rho_A) + S(\rho_B)$$

**Exercise 8.6.11**
Derive the **Lindblad master equation** for Markovian open system dynamics:

$$\frac{d\rho}{dt} = -i[H, \rho] + \sum_k \left(L_k \rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\}\right)$$

where $L_k$ are Lindblad operators.

**Exercise 8.6.12**
Show that the amplitude damping channel (Example 8.6.9) drives any initial state toward $|0\rangle\langle 0|$ as $\gamma \to 1$.

**Exercise 8.6.13** (Quantum Data Processing Inequality)
Prove that for any channel $\mathcal{E}$ and states $\rho, \sigma$:

$$S(\mathcal{E}(\rho) \| \mathcal{E}(\sigma)) \leq S(\rho \| \sigma)$$

where $S(\rho\|\sigma) = \text{tr}(\rho \log \rho - \rho \log \sigma)$ is the quantum relative entropy.

**Exercise 8.6.14**
For a qubit undergoing dephasing $\rho(t)$ with rate $\Gamma$, compute the **coherence time** $\tau_{\text{coh}}$ defined by $|\rho_{01}(\tau_{\text{coh}})| = e^{-1}|\rho_{01}(0)|$.

---

## Related Topics

- **Chapter 8.7:** Quantum Entanglement and Bell Inequalities
- **Chapter 8.8:** Quantum Measurement Theory (POVMs, generalized measurements)
- **Chapter 9.3:** Quantum Error Correction (stabilizer codes, surface codes)
- **Chapter 10.5:** Quantum Machine Learning (variational quantum algorithms, noisy circuits)
- **Chapter 11.2:** Open Quantum Systems (master equations, non-Markovian dynamics)
- **Appendix D:** Matrix Functions and Operator Calculus

---

## Further Reading

1. **Nielsen & Chuang** (2010), *Quantum Computation and Quantum Information*, Chapter 8
2. **Preskill** (1998), *Lecture Notes on Quantum Computation*, Chapter 3
3. **Breuer & Petruccione** (2002), *The Theory of Open Quantum Systems*
4. **Wilde** (2013), *Quantum Information Theory*, Chapters 3-4

---

> **Quantum ML Insight:** Density matrices are essential for describing noisy intermediate-scale quantum (NISQ) devices. Variational quantum algorithms must account for decoherence via effective $\rho(t)$ modeling, and quantum generative models (like quantum Boltzmann machines) inherently use mixed states for thermal distributions.
# Chapter 8.5: Quantum Measurement

## Prerequisites

- **Required:**
  - Linear algebra: inner products, orthonormal bases, projection operators (Chapter 2)
  - Quantum states and superposition (Chapter 8.1)
  - Quantum operators and observables (Chapter 8.3)
  - Hilbert spaces and tensor products (Chapter 8.2)

- **Recommended:**
  - Probability theory fundamentals (Chapter 3)
  - Eigenvalues and spectral decomposition (Chapter 2.4)

## Overview

Measurement is the fundamental process that bridges the quantum and classical worlds. While quantum states evolve unitarily according to the Schrödinger equation, measurement introduces fundamentally probabilistic outcomes and irreversible state changes. This chapter explores the mathematical framework of quantum measurement, from the foundational Born rule to generalized measurements and their role in quantum computing.

**Key Insight:** Measurement is not merely observation—it is an active process that disturbs the quantum system, extracting classical information at the cost of destroying quantum superposition.

---

## 8.5.1 The Born Rule

The Born rule, formulated by Max Born in 1926, is a fundamental postulate of quantum mechanics that prescribes how to compute probabilities of measurement outcomes.

### Definition 8.5.1 (Born Rule)

Let $|\psi\rangle$ be a quantum state and $\{|m\rangle\}$ be an orthonormal basis of measurement outcomes. The **probability** of obtaining outcome $m$ when measuring $|\psi\rangle$ is:

$$P(m) = |\langle m|\psi\rangle|^2$$

where $\langle m|\psi\rangle$ is the probability amplitude.

**Properties:**

1. **Normalization:** $\sum_m P(m) = \sum_m |\langle m|\psi\rangle|^2 = \langle\psi|\psi\rangle = 1$
2. **Non-negativity:** $P(m) \geq 0$ for all $m$
3. **Probability amplitude:** Complex number $\langle m|\psi\rangle$ whose squared magnitude gives probability

### Example 8.5.1 (Qubit Measurement)

Consider a qubit in state:

$$|\psi\rangle = \frac{1}{\sqrt{3}}|0\rangle + \sqrt{\frac{2}{3}}|1\rangle$$

Measuring in the computational basis $\{|0\rangle, |1\rangle\}$:

$$P(0) = \left|\frac{1}{\sqrt{3}}\right|^2 = \frac{1}{3}, \quad P(1) = \left|\sqrt{\frac{2}{3}}\right|^2 = \frac{2}{3}$$

Note: The phase information in amplitudes is lost in measurement—$\frac{1}{\sqrt{3}}$ and $\frac{e^{i\phi}}{\sqrt{3}}$ yield the same probability.

### Example 8.5.1a (Born Rule — Two-Qubit State)

Consider the two-qubit state:

$$|\psi\rangle = \frac{1}{2}|00\rangle + \frac{1}{2}|01\rangle + \frac{1}{\sqrt{2}}|10\rangle$$

Measuring **both qubits** in the computational basis $\{|00\rangle, |01\rangle, |10\rangle, |11\rangle\}$:

$$P(00) = \left|\frac{1}{2}\right|^2 = \frac{1}{4}, \quad P(01) = \left|\frac{1}{2}\right|^2 = \frac{1}{4}, \quad P(10) = \left|\frac{1}{\sqrt{2}}\right|^2 = \frac{1}{2}, \quad P(11) = 0$$

Measuring **only the first qubit**: group amplitudes by first-qubit value.

- $P(\text{1st qubit}=0) = |1/2|^2 + |1/2|^2 = 1/4 + 1/4 = 1/2$
- $P(\text{1st qubit}=1) = |1/\sqrt{2}|^2 + 0 = 1/2$

Post-measurement state if first qubit yields $|1\rangle$: $\;|\psi'\rangle = |1\rangle \otimes |0\rangle = |10\rangle$.

### Quantum Connection: Probability Amplitudes vs Classical Probabilities

Unlike classical probability distributions, quantum amplitudes can interfere:
- **Classical:** $P(A \text{ or } B) = P(A) + P(B)$
- **Quantum:** $P(\text{outcome}) = |\alpha_A + \alpha_B|^2 = |\alpha_A|^2 + |\alpha_B|^2 + 2\text{Re}(\alpha_A^*\alpha_B)$

The interference term $2\text{Re}(\alpha_A^*\alpha_B)$ has no classical analog.

---

## 8.5.2 Projective Measurement

Projective measurements are the standard measurement model in quantum mechanics, associated with Hermitian operators.

### Definition 8.5.2 (Projective Measurement)

A **projective measurement** is described by a Hermitian observable $A$ with spectral decomposition:

$$A = \sum_m \lambda_m P_m$$

where $\lambda_m$ are eigenvalues and $P_m = |m\rangle\langle m|$ are projection operators onto the eigenspace for $\lambda_m$.

**Measurement Process:**

1. **Outcome probability:** $P(m) = \langle\psi|P_m|\psi\rangle = \|\,P_m|\psi\rangle\|^2$
2. **Post-measurement state:** $|\psi'\rangle = \frac{P_m|\psi\rangle}{\sqrt{\langle\psi|P_m|\psi\rangle}}$
3. **Measured value:** Eigenvalue $\lambda_m$

### Example 8.5.2a (Projective Measurement — Worked Computation)

Measure $|\psi\rangle = \frac{3}{5}|0\rangle + \frac{4}{5}|1\rangle$ in the Z-basis. The projectors are $P_0 = |0\rangle\langle 0|$ and $P_1 = |1\rangle\langle 1|$.

**Step 1 — Probabilities:**

$$P(0) = \langle\psi|P_0|\psi\rangle = \left|\langle 0|\psi\rangle\right|^2 = \left(\frac{3}{5}\right)^2 = \frac{9}{25}, \quad P(1) = \left(\frac{4}{5}\right)^2 = \frac{16}{25}$$

Check: $9/25 + 16/25 = 1$. $\checkmark$

**Step 2 — Post-measurement states:**

$$\text{If outcome } 0: \quad |\psi'\rangle = \frac{P_0|\psi\rangle}{\sqrt{P(0)}} = \frac{(3/5)|0\rangle}{3/5} = |0\rangle$$

$$\text{If outcome } 1: \quad |\psi'\rangle = \frac{P_1|\psi\rangle}{\sqrt{P(1)}} = \frac{(4/5)|1\rangle}{4/5} = |1\rangle$$

**Step 3 — Measured values:** Outcome 0 yields eigenvalue $+1$; outcome 1 yields eigenvalue $-1$.

### ASCII Diagram: Measurement Process

```
Before Measurement                  Measurement              After Measurement

     |ψ⟩ = α|0⟩ + β|1⟩             P(0) = |α|²            |ψ'⟩ = |0⟩
           │                           ↓                     │
           │  Superposition         Measure                 │  Definite
           │                           ↓                     │
   Quantum ├─────────────────────> Classical ──────────> Quantum
   State   │                        Outcome                 State
           │                           ↓                     │
           │                        P(1) = |β|²            |ψ'⟩ = |1⟩

Interference                    Information               Destroyed
Preserved                       Extraction                Superposition
```

### Theorem 8.5.1 (Properties of Projective Measurements)

For projection operators $\{P_m\}$:

1. **Hermitian:** $P_m^\dagger = P_m$
2. **Idempotent:** $P_m^2 = P_m$
3. **Orthogonality:** $P_m P_n = \delta_{mn} P_m$
4. **Completeness:** $\sum_m P_m = I$

**Proof sketch:** These follow directly from the properties of eigenvectors of Hermitian operators forming an orthonormal basis. ∎

### Example 8.5.2 (Pauli Z Measurement)

The Pauli-Z operator:

$$Z = |0\rangle\langle 0| - |1\rangle\langle 1| = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

has projection operators $P_0 = |0\rangle\langle 0|$, $P_1 = |1\rangle\langle 1|$ with eigenvalues $+1, -1$.

For state $|\psi\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$:

- $P(+1) = |\langle 0|\psi\rangle|^2 = \frac{1}{2}$, post-measurement: $|0\rangle$
- $P(-1) = |\langle 1|\psi\rangle|^2 = \frac{1}{2}$, post-measurement: $|1\rangle$

---

## 8.5.3 Measurement in Different Bases

The choice of measurement basis fundamentally affects outcomes—measurements in non-commuting bases are complementary.

### Definition 8.5.3 (Computational Basis Measurement)

The **computational basis** (or **Z-basis**) measurement uses projectors:

$$P_0 = |0\rangle\langle 0|, \quad P_1 = |1\rangle\langle 1|$$

This is the standard "read out" measurement in quantum computing.

### Definition 8.5.4 (Hadamard Basis Measurement)

The **Hadamard basis** (or **X-basis**) uses states:

$$|+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}, \quad |-\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}$$

with projectors $P_+ = |+\rangle\langle +|$, $P_- = |-\rangle\langle -|$.

### Example 8.5.3a (X-Basis Measurement of $|0\rangle$)

Measure $|\psi\rangle = |0\rangle$ in the X-basis $\{|+\rangle, |-\rangle\}$.

**Step 1 — Re-express in X-basis:** Since $|0\rangle = \frac{1}{\sqrt{2}}|+\rangle + \frac{1}{\sqrt{2}}|-\rangle$:

**Step 2 — Compute amplitudes and probabilities:**

$$\langle +|0\rangle = \frac{1}{\sqrt{2}}, \quad \langle -|0\rangle = \frac{1}{\sqrt{2}}$$

$$P(+) = \left|\frac{1}{\sqrt{2}}\right|^2 = \frac{1}{2}, \quad P(-) = \left|\frac{1}{\sqrt{2}}\right|^2 = \frac{1}{2}$$

**Step 3 — Post-measurement states:**

- If outcome $+$: state collapses to $|+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$
- If outcome $-$: state collapses to $|-\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}$

**Observation:** A state perfectly determined in Z-basis becomes maximally random in X-basis.

### Example 8.5.3 (Basis Comparison)

Measure $|\psi\rangle = |0\rangle$ in different bases:

| Basis | State | P(outcome 1) | P(outcome 2) |
|-------|-------|--------------|--------------|
| Z-basis | $|0\rangle$ | $P(0) = 1$ | $P(1) = 0$ |
| X-basis | $\frac{\|+\rangle + \|-\rangle}{\sqrt{2}}$ | $P(+) = \frac{1}{2}$ | $P(-) = \frac{1}{2}$ |
| Y-basis | $\frac{\|i+\rangle + \|i-\rangle}{\sqrt{2}}$ | $P(i+) = \frac{1}{2}$ | $P(i-) = \frac{1}{2}$ |

**Key Observation:** $|0\rangle$ is definite in Z-basis but maximally uncertain in X and Y bases.

### Theorem 8.5.2 (Complementarity)

Two observables $A$ and $B$ with eigenbases $\{|a_i\rangle\}$ and $\{|b_j\rangle\}$ are **complementary** if:

$$|\langle a_i | b_j \rangle|^2 = \frac{1}{d}$$

for all $i, j$, where $d$ is the dimension. Measuring one gives maximal uncertainty about the other.

**Example:** Z and X measurements are complementary for qubits.

### Example 8.5.3b (Sequential Measurements — Z then X)

Start with $|\psi\rangle = \frac{3}{5}|0\rangle + \frac{4}{5}|1\rangle$ and perform Z-measurement followed by X-measurement.

**Stage 1 — Z-measurement:**

$$P_Z(0) = (3/5)^2 = 9/25, \quad P_Z(1) = (4/5)^2 = 16/25$$

Suppose we obtain outcome $0$. Post-measurement state: $|\psi_1\rangle = |0\rangle$.

**Stage 2 — X-measurement on $|0\rangle$:**

Since $|0\rangle = \frac{1}{\sqrt{2}}(|+\rangle + |-\rangle)$:

$$P_X(+) = 1/2, \quad P_X(-) = 1/2$$

**Joint probabilities** (outcome $0$ then $+$):

$$P(Z{=}0 \text{ then } X{=}+) = P_Z(0) \cdot P_X(+|Z{=}0) = \frac{9}{25} \cdot \frac{1}{2} = \frac{9}{50}$$

**Key point:** After the Z-measurement collapses $|\psi\rangle$ to $|0\rangle$, the original amplitude information $(3/5, 4/5)$ is lost — the X-measurement sees only $|0\rangle$.

### ASCII Diagram: Bloch Sphere Measurements

```
           Z-axis
            |
            |0⟩
            |
            |
    ──|+⟩───●───|−⟩── X-axis
           /|
          / |
       Y /  |
            |1⟩

Measurement extracts classical bit:
- Z-measurement: projects to poles (|0⟩ or |1⟩)
- X-measurement: projects to equator (|+⟩ or |−⟩)
- Y-measurement: projects to Y-axis (|i+⟩ or |i−⟩)
```

---

## 8.5.4 Generalized Measurements (POVM)

Not all measurements are projective. Positive Operator-Valued Measures (POVMs) describe the most general quantum measurements.

### Definition 8.5.5 (POVM)

A **POVM** is a set of positive operators $\{E_m\}$ satisfying:

1. **Positivity:** $E_m \geq 0$ (positive semidefinite)
2. **Completeness:** $\sum_m E_m = I$

The probability of outcome $m$ is:

$$P(m) = \langle\psi|E_m|\psi\rangle = \text{Tr}(E_m|\psi\rangle\langle\psi|)$$

**Note:** POVMs describe measurement statistics but not post-measurement states (need Kraus operators for that).

### Example 8.5.4a (Three-Element Qubit POVM — Trine Measurement)

Define three directions equally spaced at $120°$ on the Bloch sphere equator. The POVM elements are:

$$E_k = \frac{2}{3}|\psi_k\rangle\langle\psi_k|, \quad k = 0, 1, 2$$

where $|\psi_0\rangle = |0\rangle$, $\;|\psi_1\rangle = \frac{1}{2}|0\rangle + \frac{\sqrt{3}}{2}|1\rangle$, $\;|\psi_2\rangle = \frac{1}{2}|0\rangle - \frac{\sqrt{3}}{2}|1\rangle$.

**Verify completeness** ($\sum_k E_k = I$): Computing each $|\psi_k\rangle\langle\psi_k|$ and summing:

$$\sum_{k=0}^{2}|\psi_k\rangle\langle\psi_k| = \begin{pmatrix}1 & 0\\0 & 0\end{pmatrix} + \begin{pmatrix}1/4 & \sqrt{3}/4\\\sqrt{3}/4 & 3/4\end{pmatrix} + \begin{pmatrix}1/4 & -\sqrt{3}/4\\-\sqrt{3}/4 & 3/4\end{pmatrix} = \begin{pmatrix}3/2 & 0\\0 & 3/2\end{pmatrix} = \frac{3}{2}I$$

Thus $\sum_k E_k = \frac{2}{3} \cdot \frac{3}{2}I = I$. $\checkmark$

**Apply to** $|\psi\rangle = |1\rangle$: $\;P(0) = \frac{2}{3}|\langle\psi_0|1\rangle|^2 = 0$, $\;P(1) = \frac{2}{3}\cdot\frac{3}{4} = \frac{1}{2}$, $\;P(2) = \frac{1}{2}$.

This is **not projective** since $E_k^2 = \frac{4}{9}|\psi_k\rangle\langle\psi_k| \neq E_k$.

### Proposition 8.5.1 (POVM vs Projective Measurement)

Every projective measurement is a POVM (with $E_m = P_m$), but not every POVM is projective.

**Example:** $E_1 = \frac{1}{2}I$, $E_2 = \frac{1}{2}I$ is a POVM (uniform random outcome) but not projective since $E_1^2 = \frac{1}{4}I \neq E_1$.

### Example 8.5.4 (SIC-POVM)

A **Symmetric Informationally Complete** (SIC) POVM for a qubit consists of 4 operators:

$$E_k = \frac{1}{2}|\psi_k\rangle\langle\psi_k|, \quad k = 0, 1, 2, 3$$

where $|\langle\psi_j|\psi_k\rangle|^2 = \frac{1}{3}$ for $j \neq k$.

**Application:** Quantum state tomography—4 measurements suffice to reconstruct any qubit state.

### Theorem 8.5.3 (Naimark Dilation)

Every POVM on system $\mathcal{H}_A$ can be realized as a projective measurement on an extended system $\mathcal{H}_A \otimes \mathcal{H}_B$.

**Intuition:** Generalized measurements are projective measurements with extra ancilla qubits that we don't observe.

---

## 8.5.5 Expectation Values

Measurement statistics are often summarized by expectation values—the average outcome over many measurements.

### Definition 8.5.6 (Expectation Value)

For observable $A = \sum_m \lambda_m P_m$ and state $|\psi\rangle$, the **expectation value** is:

$$\langle A \rangle = \langle\psi|A|\psi\rangle = \sum_m \lambda_m \langle\psi|P_m|\psi\rangle = \sum_m \lambda_m P(m)$$

**Interpretation:** Average of measurement outcomes $\lambda_m$ weighted by probabilities $P(m)$.

### Example 8.5.5a (Expectation Value — Explicit Computation)

Compute $\langle Z \rangle$ for $|\psi\rangle = \frac{3}{5}|0\rangle + \frac{4}{5}|1\rangle$.

**Method 1 — Direct matrix computation:**

$$\langle Z \rangle = \langle\psi|Z|\psi\rangle = \begin{pmatrix}3/5 & 4/5\end{pmatrix}\begin{pmatrix}1&0\\0&-1\end{pmatrix}\begin{pmatrix}3/5\\4/5\end{pmatrix} = \begin{pmatrix}3/5 & 4/5\end{pmatrix}\begin{pmatrix}3/5\\-4/5\end{pmatrix} = \frac{9}{25} - \frac{16}{25} = -\frac{7}{25}$$

**Method 2 — Weighted eigenvalues:**

$$\langle Z \rangle = (+1)\cdot P(0) + (-1)\cdot P(1) = (+1)\frac{9}{25} + (-1)\frac{16}{25} = -\frac{7}{25}$$

**Interpretation:** The negative value $-7/25$ indicates the state is biased toward $|1\rangle$ (eigenvalue $-1$), consistent with the larger amplitude $4/5 > 3/5$.

### Example 8.5.5 (Pauli Expectation Values)

For state $|\psi\rangle = \cos\frac{\theta}{2}|0\rangle + e^{i\phi}\sin\frac{\theta}{2}|1\rangle$:

$$\langle Z \rangle = \cos\theta, \quad \langle X \rangle = \sin\theta\cos\phi, \quad \langle Y \rangle = \sin\theta\sin\phi$$

These correspond to the Bloch vector coordinates: $\vec{r} = (\langle X \rangle, \langle Y \rangle, \langle Z \rangle)$.

### Theorem 8.5.4 (Properties of Expectation Values)

1. **Linearity:** $\langle aA + bB \rangle = a\langle A \rangle + b\langle B \rangle$
2. **Bounds:** $\lambda_{\min} \leq \langle A \rangle \leq \lambda_{\max}$
3. **Variance:** $(\Delta A)^2 = \langle A^2 \rangle - \langle A \rangle^2$

### Quantum Connection: Variational Quantum Algorithms

Variational algorithms (VQE, QAOA) optimize parameters $\theta$ to minimize:

$$E(\theta) = \langle\psi(\theta)|H|\psi(\theta)\rangle$$

where $H$ is a problem Hamiltonian. Measuring $\langle H \rangle$ requires decomposing $H$ into measurable Pauli terms:

$$H = \sum_i c_i P_i \implies \langle H \rangle = \sum_i c_i \langle P_i \rangle$$

---

## 8.5.6 Measurement and Decoherence

Measurement destroys quantum coherence—but why? The answer lies in entanglement with the environment.

### Definition 8.5.7 (Decoherence)

**Decoherence** is the loss of quantum coherence due to entanglement with an environment, transforming pure states to mixed states.

**Measurement-induced decoherence:** Interaction with measurement apparatus entangles system and device, causing apparent wave function collapse.

### Example 8.5.6a (Measurement Collapse — Density Matrix Trace-Through)

Track the density matrix of $|\psi\rangle = \frac{3}{5}|0\rangle + \frac{4}{5}|1\rangle$ through a Z-measurement.

**Before measurement** — pure state density matrix:

$$\rho_{\text{before}} = |\psi\rangle\langle\psi| = \begin{pmatrix}9/25 & 12/25\\12/25 & 16/25\end{pmatrix}$$

The off-diagonal terms $12/25$ represent quantum coherence (superposition).

**After measurement** — average over outcomes (no record of result):

$$\rho_{\text{after}} = P(0)|0\rangle\langle 0| + P(1)|1\rangle\langle 1| = \frac{9}{25}\begin{pmatrix}1&0\\0&0\end{pmatrix} + \frac{16}{25}\begin{pmatrix}0&0\\0&1\end{pmatrix} = \begin{pmatrix}9/25 & 0\\0 & 16/25\end{pmatrix}$$

**What changed:** The off-diagonal coherence terms $12/25 \to 0$. The state went from a pure superposition to a classical probability mixture. This is irreversible — the relative phase between $|0\rangle$ and $|1\rangle$ is permanently lost.

### Example 8.5.6 (Measurement as Entanglement)

Consider measuring qubit $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ with measurement device initially in $|M_0\rangle$:

**Before measurement:**
$$|\psi\rangle \otimes |M_0\rangle = (\alpha|0\rangle + \beta|1\rangle) \otimes |M_0\rangle$$

**After interaction (unitary):**
$$|\Psi\rangle = \alpha|0\rangle|M_0\rangle + \beta|1\rangle|M_1\rangle$$

**Reduced density matrix of qubit:**
$$\rho = \text{Tr}_{\text{device}}(|\Psi\rangle\langle\Psi|) = |\alpha|^2|0\rangle\langle 0| + |\beta|^2|1\rangle\langle 1|$$

**Result:** Superposition $\alpha|0\rangle + \beta|1\rangle$ becomes classical mixture with off-diagonal terms (coherence) destroyed.

### ASCII Diagram: Decoherence Process

```
Pure State                Entanglement           Mixed State
                          with Environment

|ψ⟩ = α|0⟩ + β|1⟩        System ⊗ Environment   ρ = |α|²|0⟩⟨0| + |β|²|1⟩⟨1|

Density Matrix:                                  Density Matrix:
┌─────────┐                                     ┌─────────┐
│ |α|²  αβ*│             Decoherence            │ |α|²   0 │
│ α*β  |β|²│  ─────────────────────────────>    │  0   |β|²│
└─────────┘             (trace out env)         └─────────┘

Off-diagonal             Information leaks       Off-diagonal
coherence terms          to environment          terms → 0
```

### Theorem 8.5.5 (Decoherence Time Scales)

System coherence decays on time scale $T_2$ (transverse relaxation), while energy relaxation occurs on scale $T_1$ (longitudinal relaxation):

$$T_2 \leq 2T_1$$

**Typical values (superconducting qubits):** $T_1 \sim 50\text{-}100\,\mu\text{s}$, $T_2 \sim 20\text{-}100\,\mu\text{s}$

---

## 8.5.7 Quantum Zeno Effect

Frequent measurement can freeze quantum evolution—a counterintuitive consequence of projective measurement.

### Definition 8.5.8 (Quantum Zeno Effect)

If a quantum system is repeatedly measured with time interval $\Delta t \to 0$, the system remains in its initial state despite Hamiltonian evolution attempting to change it.

**Mathematical formulation:** For initial state $|\psi_0\rangle$ and Hamiltonian $H$:

$$P(\text{stay in } |\psi_0\rangle) \approx 1 - \frac{(\Delta E)^2(\Delta t)^2}{\hbar^2}$$

where $(\Delta E)^2 = \langle H^2 \rangle - \langle H \rangle^2$.

**Result:** For $N$ measurements over time $T$:
- Measurement interval: $\Delta t = T/N$
- Survival probability: $P_{\text{survive}} \approx \left(1 - \frac{(\Delta E)^2 T^2}{\hbar^2 N^2}\right)^N \xrightarrow{N \to \infty} 1$

### Example 8.5.7 (Two-Level System)

Consider a qubit initialized in $|0\rangle$ evolving under $H = \frac{\omega}{2}X$:

**Without measurement:** $|\psi(t)\rangle = \cos(\omega t/2)|0\rangle - i\sin(\omega t/2)|1\rangle$
- At $t = \pi/\omega$: complete transition to $|1\rangle$

**With continuous measurement of Z:** System frozen in $|0\rangle$
- Each measurement projects back to $|0\rangle$
- Prevents accumulation of $|1\rangle$ amplitude

### Quantum Connection: Measurement-Based Stabilization

Quantum error correction uses frequent syndrome measurements to detect errors without collapsing encoded information—a controlled application of the Zeno effect.

---

## 8.5.8 Measurement in Quantum Computing

Measurement is not just the final readout—modern quantum algorithms use measurements as computational resources.

### Definition 8.5.9 (Mid-Circuit Measurement)

**Mid-circuit measurement** measures qubits during computation, using outcomes for classical control flow:

```
|ψ⟩ ──U── [M] ─┬─ if 0: apply U_0
               └─ if 1: apply U_1
```

**Applications:**
- Quantum teleportation (measure Bell pair, apply corrections)
- Syndrome extraction in error correction (measure stabilizers)
- Adaptive circuits (measurement-based quantum computing)

### Example 8.5.8 (Quantum Teleportation)

Alice sends qubit $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ to Bob using shared EPR pair:

1. **Entangle:** Alice and Bob share $\frac{|00\rangle + |11\rangle}{\sqrt{2}}$
2. **Measure:** Alice performs Bell measurement on $|\psi\rangle$ and her EPR qubit → 2 classical bits
3. **Correct:** Bob applies corrections based on Alice's measurement results
4. **Result:** Bob's qubit becomes $|\psi\rangle$

**Key insight:** Measurement extracts 2 classical bits that enable quantum state transfer.

### Definition 8.5.10 (Measurement-Based Quantum Computing)

**MBQC** uses measurements as the primary computational primitive:

1. Prepare large entangled resource state (cluster state)
2. Perform single-qubit measurements in adaptive bases
3. Classical feedforward determines subsequent measurement bases
4. Output read from final measurement pattern

**Equivalence:** MBQC is equivalent in power to circuit-based quantum computing (gate model).

### Quantum Connection: Quantum Machine Learning Readout

QML algorithms measure expectation values to extract classical information:

**Classification:** Measure $\langle\psi|O_{\text{class}}|\psi\rangle$ where eigenstates correspond to classes
**Regression:** Measure $\langle H_{\text{output}} \rangle$ encoding continuous prediction
**Sampling:** Measure computational basis for generative models (quantum GANs)

### Example 8.5.9 (Variational Classifier)

Quantum neural network for binary classification:

```
Input x ──U(x,θ)── |ψ(x,θ)⟩ ── Measure Z ── ⟨Z⟩ ── Sign(⟨Z⟩) ── Class
```

1. **Encoding:** Map $x$ to quantum state via $U(x, \theta)$
2. **Measurement:** Measure Pauli-Z expectation $\langle Z \rangle$
3. **Classification:** $\langle Z \rangle > 0 \implies \text{Class } +1$, else $-1$
4. **Training:** Optimize $\theta$ to minimize classification error

**Challenge:** Need many shots to estimate $\langle Z \rangle$ accurately (shot noise $\sim 1/\sqrt{N_{\text{shots}}}$).

---

## Summary

This chapter covered the mathematical framework of quantum measurement:

1. **Born Rule:** Probability amplitudes $\langle m|\psi\rangle$ determine outcome probabilities $|\langle m|\psi\rangle|^2$
2. **Projective Measurements:** Standard measurements via Hermitian operators and projection operators
3. **Measurement Bases:** Choice of basis fundamentally affects outcomes (complementarity principle)
4. **POVMs:** Most general measurement framework, includes non-projective measurements
5. **Expectation Values:** Statistical averages central to variational quantum algorithms
6. **Decoherence:** Entanglement with environment explains apparent wave function collapse
7. **Quantum Zeno Effect:** Frequent measurement freezes quantum evolution
8. **Quantum Computing:** Measurements as computational resource (mid-circuit, MBQC, QML readout)

**Key Takeaway:** Measurement is the interface between quantum and classical worlds, enabling information extraction at the cost of disturbing quantum states—a trade-off fundamental to quantum computing and machine learning.

---

## Exercises

### ★ Basic Exercises

**Exercise 8.5.1:** A qubit is in state $|\psi\rangle = \frac{3}{5}|0\rangle + \frac{4}{5}|1\rangle$.
(a) Compute probabilities of measuring 0 and 1 in computational basis.
(b) What is the post-measurement state in each case?

**Exercise 8.5.2:** Verify that $P_0 = |0\rangle\langle 0|$ and $P_1 = |1\rangle\langle 1|$ satisfy the four properties of projection operators in Theorem 8.5.1.

**Exercise 8.5.3:** A qubit in state $|+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$ is measured in the Z-basis. Compute the expectation value $\langle Z \rangle$ before measurement and after obtaining outcome 0.

**Exercise 8.5.4:** Show that the set $\{E_1, E_2, E_3\} = \{\frac{1}{3}I, \frac{1}{3}I, \frac{1}{3}I\}$ is a valid POVM. Is it projective?

### ★★ Intermediate Exercises

**Exercise 8.5.5:** Prove that for any projective measurement, the expectation value $\langle A \rangle = \sum_m \lambda_m P(m)$ equals $\langle\psi|A|\psi\rangle$.

**Exercise 8.5.6:** Consider measuring $|\psi\rangle = \cos\theta|0\rangle + \sin\theta|1\rangle$ in the X-basis.
(a) Express $|\psi\rangle$ in the $\{|+\rangle, |-\rangle\}$ basis.
(b) Compute $P(+)$ and $P(-)$.
(c) For what angle $\theta$ are these probabilities equal?

**Exercise 8.5.7:** Two qubits are in Bell state $|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}$.
(a) Alice measures her qubit in Z-basis. What are the possible outcomes and probabilities?
(b) After Alice measures and obtains 0, what is Bob's state?
(c) Repeat for Alice measuring in X-basis.

**Exercise 8.5.8:** The variance of an observable is $(\Delta A)^2 = \langle A^2 \rangle - \langle A \rangle^2$.
(a) Compute $\Delta Z$ for state $|+\rangle$.
(b) Find a state where $\Delta Z = 0$.
(c) Prove $\Delta A = 0$ if and only if $|\psi\rangle$ is an eigenstate of $A$.

**Exercise 8.5.9:** Construct a POVM for a qubit with 3 outcomes, where each $E_m$ is a positive multiple of a rank-1 projector. Verify completeness.

### ★★★ Advanced Exercises

**Exercise 8.5.10:** Prove the quantum Zeno effect result: for $N$ measurements over time $T$, the survival probability approaches 1 as $N \to \infty$. (Hint: Use $P = \left(1 - \frac{c}{N^2}\right)^N$ with $c = (\Delta E)^2 T^2/\hbar^2$.)

**Exercise 8.5.11:** Show that any POVM $\{E_m\}$ on $\mathbb{C}^d$ can be realized as a projective measurement on $\mathbb{C}^d \otimes \mathbb{C}^k$ for sufficiently large $k$. (Naimark dilation theorem—provide explicit construction.)

**Exercise 8.5.12:** A Hamiltonian $H = \omega X$ evolves state $|0\rangle$ for time $t$.
(a) Compute $|\psi(t)\rangle$ without measurement.
(b) With $N$ measurements uniformly spaced in $[0, t]$, compute the probability of remaining in $|0\rangle$.
(c) Plot $P_{\text{survive}}$ vs $N$ for $\omega t = \pi$.

**Exercise 8.5.13:** In measurement-based quantum computing, a cluster state on a line is:
$$|C\rangle = \frac{1}{2^{n/2}}\bigotimes_{j=1}^n(|0\rangle + |1\rangle)_j \cdot \prod_{\langle j,k \rangle} CZ_{jk}$$
Show that single-qubit measurements in bases $\{e^{i\theta_j Z}|+\rangle, e^{i\theta_j Z}|-\rangle\}$ can implement arbitrary single-qubit rotations.

**Exercise 8.5.14:** For a variational classifier using ansatz $U(x, \theta)$, the classification loss is:
$$L(\theta) = \sum_{i=1}^N \left(y_i - \text{sign}(\langle Z \rangle_i)\right)^2$$
where $\langle Z \rangle_i = \langle\psi(x_i, \theta)|Z|\psi(x_i, \theta)\rangle$.
(a) Compute the gradient $\frac{\partial L}{\partial \theta}$.
(b) Show that measuring expectation values introduces shot noise scaling as $O(1/\sqrt{N_{\text{shots}}})$.
(c) How many shots are needed for gradient estimation with error $\epsilon$?

---

## Related Topics

- **Chapter 8.1:** Quantum States and Superposition (foundational prerequisite)
- **Chapter 8.3:** Quantum Operators and Observables (Hermitian operators, spectral theorem)
- **Chapter 8.4:** Quantum Entanglement (measurement on entangled states, Bell measurements)
- **Chapter 8.6:** Quantum Dynamics (unitary evolution vs measurement-induced collapse)
- **Chapter 8.7:** Quantum Error Correction (syndrome measurements, stabilizer measurements)
- **Chapter 9.2:** Variational Quantum Algorithms (expectation value estimation, VQE, QAOA)
- **Chapter 9.4:** Quantum Machine Learning (measurement-based readout, quantum neural networks)
- **Chapter 10.3:** Quantum Tomography (state reconstruction from measurement statistics)

---

## Further Reading

- **Nielsen & Chuang** (2010): *Quantum Computation and Quantum Information* — Chapter 2.2-2.3 (measurement postulates, POVM)
- **Peres** (1995): *Quantum Theory: Concepts and Methods* — Chapter 9 (measurement theory, foundations)
- **Wiseman & Milburn** (2010): *Quantum Measurement and Control* — Comprehensive treatment of measurement theory
- **Raussendorf & Briegel** (2001): "A One-Way Quantum Computer" — Measurement-based quantum computing
- **Facchi & Pascazio** (2008): "Quantum Zeno phenomena" — Rev. Mod. Phys. (quantum Zeno effect review)
- **Schuld & Petruccione** (2018): *Supervised Learning with Quantum Computers* — Chapter 4 (measurement in QML)

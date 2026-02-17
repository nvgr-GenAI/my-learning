# Chapter 8.10: Quantum Machine Learning

## Prerequisites

- **Chapter 8.1–8.3:** Quantum states, gates, measurement (qubit formalism, superposition, entanglement)
- **Chapter 8.7:** Quantum algorithms (variational quantum eigensolver, quantum phase estimation)
- **Chapter 8.8:** Quantum error correction (noise models, NISQ limitations)
- **Linear Algebra:** Kernel methods, feature maps, Hilbert spaces
- **Machine Learning:** Supervised learning, gradient descent, neural networks

---

## 8.10.1 The Quantum Machine Learning Landscape

Quantum machine learning (QML) sits at the intersection of quantum computing and machine learning. The field is organized along two axes: the **type of data** (quantum vs. classical) and the **type of model** (quantum vs. classical).

### Definition 8.10.1 (QML Taxonomy)

The quantum machine learning landscape partitions into four quadrants:

| Data Type | Classical Model | Quantum Model |
|-----------|----------------|---------------|
| **Classical Data** | Classical ML (baseline) | **Quantum-enhanced ML** (parameterized circuits, quantum kernels) |
| **Quantum Data** | Quantum tomography → classical ML | **Quantum-native ML** (classify quantum states directly) |

**Table 8.10.1: Quantum Machine Learning Taxonomy**

**Quantum Connection:** Most near-term QML research focuses on the **quantum-enhanced ML** quadrant: encoding classical data into quantum states, processing with parameterized quantum circuits, and using classical optimizers. This is the NISQ-era workhorse approach.

---

### Comparison of Approaches

| Approach | Data Source | Model | Primary Goal | Current Status |
|----------|-------------|-------|--------------|----------------|
| Classical ML | Real-world datasets | Neural networks, SVMs | Pattern recognition | Mature, production-ready |
| Quantum-enhanced ML | Classical data → quantum encoding | Parameterized circuits | Potential speedup on specific tasks | Research/NISQ |
| Quantum-native ML | Quantum sensors, simulations | Quantum classifier | Process quantum data efficiently | Research (quantum chemistry) |
| Hybrid ML | Classical + quantum | Classical NN + quantum layer | Leverage quantum subroutines | Experimental |

**Table 8.10.2: QML Approach Comparison**

---

## 8.10.2 Data Encoding into Quantum States

To use quantum computers for classical ML tasks, we must encode classical data $\mathbf{x} \in \mathbb{R}^d$ into quantum states $|\psi(\mathbf{x})\rangle$. This encoding is non-trivial and impacts expressibility, depth, and measurement complexity.

### Definition 8.10.2 (Basis Encoding)

For binary data $\mathbf{x} = (x_1, \ldots, x_n) \in \{0,1\}^n$, **basis encoding** maps:
$$
\mathbf{x} \mapsto |x_1 x_2 \cdots x_n\rangle
$$
Each bit directly corresponds to a qubit's computational basis state.

**Example:** $\mathbf{x} = (1,0,1) \mapsto |101\rangle$ (3 qubits).

**Properties:**
- Requires $n$ qubits for $n$ bits
- Simple to implement (no gates needed, just initialization)
- Does **not** exploit superposition or entanglement
- Useful for quantum RAM and database search

---

### Definition 8.10.3 (Amplitude Encoding)

For normalized data $\mathbf{x} \in \mathbb{R}^N$ (with $||\mathbf{x}|| = 1$ and $N = 2^n$), **amplitude encoding** maps:
$$
\mathbf{x} = (x_0, x_1, \ldots, x_{N-1}) \mapsto |\psi\rangle = \sum_{i=0}^{N-1} x_i |i\rangle
$$
Each amplitude encodes a feature value.

**Example:** $\mathbf{x} = (0.5, 0.5, 0.5, 0.5) \mapsto \frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)$ (2 qubits encode 4 features).

**Properties:**
- **Exponentially efficient:** $n$ qubits encode $2^n$ features
- Requires state preparation circuits (depth grows with complexity)
- Extracting all amplitudes requires $O(2^n)$ measurements (measurement bottleneck)

---

### Definition 8.10.4 (Angle Encoding)

For data $\mathbf{x} = (x_1, \ldots, x_n)$, **angle encoding** applies rotations:
$$
|\psi(\mathbf{x})\rangle = \bigotimes_{j=1}^{n} R(\theta_j) |0\rangle, \quad \text{where } \theta_j = f(x_j)
$$
Typically $R(\theta) = R_Y(\theta)$ or $R_Z(\theta)$, and $\theta_j = \pi x_j$ or $2\pi x_j$.

**Example:** For $\mathbf{x} = (x_1, x_2)$, apply $R_Y(\pi x_1) \otimes R_Y(\pi x_2)$ to $|00\rangle$.

**Properties:**
- Requires $n$ qubits for $n$ features
- Easy to implement (one rotation per feature)
- Natural for variational circuits (angles as parameters)
- Exploits quantum superposition

**Example 8.10.1 (Feature Map — Angle Encoding a 2D Data Point):**
Encode $\mathbf{x} = (0.5, 0.25)$ into a 2-qubit state using $R_Y$ rotations with $\theta_j = \pi x_j$.
Compute rotation angles: $\theta_1 = 0.5\pi$, $\theta_2 = 0.25\pi$.
Apply to each qubit independently:
$$
R_Y(\theta)|0\rangle = \cos\!\left(\frac{\theta}{2}\right)|0\rangle + \sin\!\left(\frac{\theta}{2}\right)|1\rangle
$$
Qubit 1: $R_Y(0.5\pi)|0\rangle = \cos(0.25\pi)|0\rangle + \sin(0.25\pi)|1\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$.
Qubit 2: $R_Y(0.25\pi)|0\rangle = \cos(0.125\pi)|0\rangle + \sin(0.125\pi)|1\rangle \approx 0.924|0\rangle + 0.383|1\rangle$.
The full encoded state is:
$$
|\phi(\mathbf{x})\rangle = \tfrac{1}{\sqrt{2}}(|0\rangle + |1\rangle) \otimes (0.924|0\rangle + 0.383|1\rangle) \approx 0.654|00\rangle + 0.271|01\rangle + 0.654|10\rangle + 0.271|11\rangle
$$

---

### Encoding Trade-Offs

| Encoding Type | Qubits for $N$ Features | Circuit Depth | Superposition | Measurement Cost | Use Case |
|---------------|------------------------|---------------|---------------|------------------|----------|
| **Basis** | $N$ | $O(1)$ | No | $O(1)$ | Quantum search, small datasets |
| **Amplitude** | $\log_2 N$ | $O(\text{poly}(n))$ | Yes | $O(N)$ | Large feature spaces (if state prep efficient) |
| **Angle** | $N$ | $O(1)$ | Yes | $O(1)$ per qubit | Variational algorithms, NISQ |

**Table 8.10.3: Data Encoding Comparison**

**Quantum Connection:** Amplitude encoding promises exponential compression but suffers from the **measurement bottleneck**: extracting full state information requires exponentially many measurements. Angle encoding is NISQ-friendly.

---

## 8.10.3 Parameterized Quantum Circuits (PQCs)

Parameterized quantum circuits are the core building block of variational quantum algorithms. They act as differentiable quantum models whose parameters are optimized classically.

### Definition 8.10.5 (Parameterized Quantum Circuit)

A **PQC** $U(\boldsymbol{\theta})$ is a unitary operator depending on classical parameters $\boldsymbol{\theta} = (\theta_1, \ldots, \theta_p) \in \mathbb{R}^p$:
$$
U(\boldsymbol{\theta}) = \prod_{\ell=1}^{L} U_\ell(\theta_\ell)
$$
where each $U_\ell(\theta_\ell)$ is a parameterized gate (e.g., $R_Y(\theta_\ell)$, $R_Z(\theta_\ell)$) or entangling layer.

**Ansatz:** The structure (gate sequence, connectivity) of the PQC. Common ansatzes:
- **Hardware-efficient ansatz:** Matches qubit connectivity of hardware
- **Problem-inspired ansatz:** Designed for specific Hamiltonians (chemistry, optimization)
- **Expressibility-optimized ansatz:** Maximizes coverage of Hilbert space

---

### ASCII Diagram: Example PQC (3 qubits, 2 layers)

```
Layer 1:              Layer 2:
q₀ ──RY(θ₁)──■────────RY(θ₄)──■──────
             │                 │
q₁ ──RY(θ₂)──⊕──■─────RY(θ₅)──⊕──■───
                 │                 │
q₂ ──RY(θ₃)─────⊕─────RY(θ₆)─────⊕───

Encoding: |000⟩ → U_enc(x) → [PQC layers] → Measure
Parameters: θ = (θ₁, θ₂, θ₃, θ₄, θ₅, θ₆)
Entanglers: CNOT gates (■──⊕) between adjacent qubits
```

**Figure 8.10.1:** A two-layer hardware-efficient ansatz. Each layer alternates single-qubit rotations $R_Y(\theta_i)$ with entangling CNOT gates.

**Example 8.10.2 (Variational Ansatz — 2-Qubit PQC):**
Consider the simplest hardware-efficient ansatz on 2 qubits with 1 layer:
$$
U(\theta_1, \theta_2) = \text{CNOT}_{01} \cdot \bigl(R_Y(\theta_1) \otimes R_Y(\theta_2)\bigr)
$$
Starting from $|00\rangle$ with $\theta_1 = \pi/2, \; \theta_2 = 0$:
Step 1 — Apply rotations: $R_Y(\pi/2)|0\rangle \otimes R_Y(0)|0\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) \otimes |0\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |10\rangle)$.
Step 2 — Apply CNOT (control=q₀, target=q₁): $\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$.
The result is a **Bell state** $|\Phi^+\rangle$ — a maximally entangled state produced by just 2 parameters and 3 gates.
With $\theta_1 = 0, \theta_2 = 0$, the output is simply $|00\rangle$ (no entanglement). This shows how parameter values control the degree of entanglement.

---

### Definition 8.10.6 (Expressibility)

The **expressibility** of a PQC measures how uniformly it covers the unitary group $U(2^n)$ as parameters vary. High expressibility means the circuit can approximate a wide range of functions.

**Formal:** Let $\rho_\text{Haar}$ be the uniform distribution over pure states (Haar measure). The expressibility distance is:
$$
\mathcal{E} = \left|\left| \rho_\text{PQC} - \rho_\text{Haar} \right|\right|
$$
where $\rho_\text{PQC}$ is the distribution of states generated by the PQC with random parameters.

**Trade-off:** High expressibility → more expressive models, but deeper circuits → more noise on NISQ hardware.

---

### Definition 8.10.7 (Variational Parameter Optimization)

Given a cost function $C(\boldsymbol{\theta})$ computed via quantum measurements, variational optimization minimizes:
$$
\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} C(\boldsymbol{\theta})
$$
using classical optimizers (gradient descent, ADAM, SPSA, evolutionary algorithms).

**Gradient estimation:**
- **Parameter-shift rule:** For $U(\theta) = e^{-i\theta \hat{P}}$ (Pauli generator $\hat{P}$):
  $$
  \frac{\partial C}{\partial \theta} = \frac{C(\theta + s) - C(\theta - s)}{2\sin(2s)}
  $$
  where $s = \pi/2$ for single-parameter gates.
- **Finite differences:** Approximate $\frac{\partial C}{\partial \theta_j} \approx \frac{C(\theta_j + \epsilon) - C(\theta_j)}{\epsilon}$.

**Example 8.10.3 (Cost Function Evaluation — 1-Qubit VQE):**
Compute $C(\theta) = \langle \psi(\theta) | Z | \psi(\theta) \rangle$ for the ansatz $|\psi(\theta)\rangle = R_Y(\theta)|0\rangle$ with $\theta = \pi/3$.
The state is $|\psi(\pi/3)\rangle = \cos(\pi/6)|0\rangle + \sin(\pi/6)|1\rangle = \frac{\sqrt{3}}{2}|0\rangle + \frac{1}{2}|1\rangle$.
Since $Z|0\rangle = |0\rangle$ and $Z|1\rangle = -|1\rangle$:
$$
C(\pi/3) = \left(\frac{\sqrt{3}}{2}\right)^2 \cdot (+1) + \left(\frac{1}{2}\right)^2 \cdot (-1) = \frac{3}{4} - \frac{1}{4} = \frac{1}{2}
$$
The cost function equals $\cos(\theta)$. The minimum $C = -1$ occurs at $\theta = \pi$ (state $|1\rangle$), which is the ground state energy of $Z$.

**Example 8.10.4 (Parameter Shift Rule — Gradient of a Single $R_Y$ Gate):**
For the same 1-qubit circuit $C(\theta) = \langle 0 | R_Y(-\theta) \, Z \, R_Y(\theta) | 0 \rangle$, compute $\frac{\partial C}{\partial \theta}$ at $\theta = \pi/3$ using the parameter-shift rule with $s = \pi/2$:
$$
\frac{\partial C}{\partial \theta} = \frac{C(\theta + \pi/2) - C(\theta - \pi/2)}{2}
$$
Evaluate: $C(\pi/3 + \pi/2) = C(5\pi/6) = \cos(5\pi/6) = -\frac{\sqrt{3}}{2} \approx -0.866$.
$C(\pi/3 - \pi/2) = C(-\pi/6) = \cos(-\pi/6) = \frac{\sqrt{3}}{2} \approx 0.866$.
$$
\frac{\partial C}{\partial \theta}\bigg|_{\theta=\pi/3} = \frac{-\frac{\sqrt{3}}{2} - \frac{\sqrt{3}}{2}}{2} = -\frac{\sqrt{3}}{2} \approx -0.866
$$
This matches the exact derivative $-\sin(\pi/3) = -\frac{\sqrt{3}}{2}$. The parameter-shift rule gives the **exact** gradient, not an approximation.

---

## 8.10.4 Quantum Kernels and Feature Maps

Kernel methods (SVMs) compute inner products in high-dimensional feature spaces. Quantum computers can implement exponentially large feature maps efficiently.

### Definition 8.10.8 (Quantum Feature Map)

A **quantum feature map** $\phi: \mathcal{X} \to \mathcal{H}$ embeds classical data $\mathbf{x}$ into a quantum Hilbert space:
$$
\mathbf{x} \mapsto |\phi(\mathbf{x})\rangle = U_\phi(\mathbf{x}) |0\rangle^{\otimes n}
$$
where $U_\phi(\mathbf{x})$ is a data-encoding unitary (e.g., angle encoding followed by entangling layers).

**Example (ZZ feature map):**
$$
U_\phi(\mathbf{x}) = \prod_{j=1}^{n} R_Z(x_j) \prod_{(j,k)} e^{-i (x_j x_k) Z_j Z_k}
$$
This creates entanglement proportional to data correlations.

---

### Definition 8.10.9 (Quantum Kernel)

The **quantum kernel** between data points $\mathbf{x}, \mathbf{x}'$ is:
$$
k_Q(\mathbf{x}, \mathbf{x}') = |\langle \phi(\mathbf{x}') | \phi(\mathbf{x}) \rangle|^2
$$

**Estimation on quantum hardware:**
1. Prepare $|\phi(\mathbf{x})\rangle = U_\phi(\mathbf{x}) |0\rangle^{\otimes n}$
2. Apply $U_\phi^\dagger(\mathbf{x}')$ to implement overlap
3. Measure in computational basis; $k_Q(\mathbf{x}, \mathbf{x}') = P(|0\rangle^{\otimes n})$

**Quantum Connection:** The kernel trick implicitly computes inner products in an exponentially large space ($2^n$ dimensions) using only $n$ qubits. Classical computation of the same feature map would require storing $2^n$ amplitudes.

**Example 8.10.5 (Quantum Kernel — 1-Qubit Feature Map):**
Use the feature map $|\phi(x)\rangle = R_Y(2x)|0\rangle = \cos(x)|0\rangle + \sin(x)|1\rangle$.
Compute the quantum kernel for $x_1 = \pi/4$ and $x_2 = \pi/3$:
$$
|\phi(x_1)\rangle = \cos(\pi/4)|0\rangle + \sin(\pi/4)|1\rangle = \tfrac{1}{\sqrt{2}}|0\rangle + \tfrac{1}{\sqrt{2}}|1\rangle
$$
$$
|\phi(x_2)\rangle = \cos(\pi/3)|0\rangle + \sin(\pi/3)|1\rangle = \tfrac{1}{2}|0\rangle + \tfrac{\sqrt{3}}{2}|1\rangle
$$
Inner product: $\langle \phi(x_2) | \phi(x_1) \rangle = \frac{1}{2} \cdot \frac{1}{\sqrt{2}} + \frac{\sqrt{3}}{2} \cdot \frac{1}{\sqrt{2}} = \frac{1 + \sqrt{3}}{2\sqrt{2}} \approx 0.966$.
$$
k_Q(x_1, x_2) = |\langle \phi(x_2) | \phi(x_1) \rangle|^2 = \left(\frac{1+\sqrt{3}}{2\sqrt{2}}\right)^2 = \frac{4 + 2\sqrt{3}}{8} \approx 0.933
$$
Note: $k_Q(x, x) = 1$ always (self-similarity), and $k_Q = \cos^2(x_1 - x_2)$ for this feature map, confirming $\cos^2(\pi/12) \approx 0.933$.

---

### Quantum Kernel SVM Pipeline

```
Classical Data
     ↓
1. Encode each x_i into |ϕ(x_i)⟩ via quantum circuit U_ϕ(x_i)
     ↓
2. Compute kernel matrix K_ij = k_Q(x_i, x_j) via quantum measurements
     ↓
3. Train classical SVM with quantum kernel matrix K
     ↓
4. Classify new points using quantum kernel evaluations
```

**Figure 8.10.2:** Quantum kernel method pipeline. The quantum computer acts as a kernel evaluator; training is classical.

---

## 8.10.5 Variational Quantum Classifier (VQC)

The VQC is a supervised learning model combining data encoding, parameterized circuits, and classical optimization.

### Definition 8.10.10 (Variational Quantum Classifier)

A **VQC** consists of:
1. **Encoding layer:** Map data $\mathbf{x} \to |\psi(\mathbf{x})\rangle$ (e.g., angle encoding)
2. **Variational layer:** Apply $U(\boldsymbol{\theta})$ (PQC with trainable parameters)
3. **Measurement:** Measure observable $\hat{M}$ (e.g., Pauli-$Z$ on first qubit)
4. **Prediction:** Output $f(\mathbf{x}; \boldsymbol{\theta}) = \langle \psi(\mathbf{x}) | U^\dagger(\boldsymbol{\theta}) \hat{M} U(\boldsymbol{\theta}) | \psi(\mathbf{x}) \rangle$

**Loss function (binary classification):**
$$
\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^{N} \left( y_i - f(\mathbf{x}_i; \boldsymbol{\theta}) \right)^2
$$
where $y_i \in \{-1, +1\}$ are labels.

---

### ASCII Flow Diagram: VQC Training Loop

```
┌─────────────────────────────────────────────────────────┐
│  Classical Data: {(x₁,y₁), (x₂,y₂), ..., (xₙ,yₙ)}     │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────▼──────────┐
         │  Initialize θ        │
         └───────────┬──────────┘
                     │
         ┌───────────▼──────────────────────────┐
         │  FOR each training example (xᵢ, yᵢ)  │
         └───────────┬──────────────────────────┘
                     │
    ┌────────────────▼────────────────┐
    │  1. Encode: |0⟩ → |ψ(xᵢ)⟩       │  (Quantum)
    │  2. Apply: U(θ)|ψ(xᵢ)⟩          │  (Quantum)
    │  3. Measure: ⟨M̂⟩ → prediction   │  (Quantum)
    └────────────────┬────────────────┘
                     │
         ┌───────────▼──────────────────┐
         │  4. Compute loss: ℒ(θ)       │  (Classical)
         │  5. Gradient: ∂ℒ/∂θ          │  (Classical)
         │     (via parameter-shift)    │
         └───────────┬──────────────────┘
                     │
         ┌───────────▼──────────────┐
         │  6. Update: θ ← θ - η∇ℒ  │  (Classical)
         └───────────┬──────────────┘
                     │
              ┌──────▼──────┐
              │  Converged? │
              └──────┬──────┘
                 No  │  Yes
         ┌───────────┘
         │           │
         └───────────┼─────────────────────────┐
                     │                         │
             ┌───────▼──────────┐   ┌──────────▼─────────┐
             │  Return to step 1│   │  Output θ* (trained)│
             └──────────────────┘   └────────────────────┘
```

**Figure 8.10.3:** Variational quantum classifier training loop. The quantum computer evaluates the cost function and gradients; the classical computer updates parameters.

---

## 8.10.6 Barren Plateaus in Quantum ML

Barren plateaus are a fundamental challenge in training deep PQCs: gradients vanish exponentially with circuit depth or qubit count, making optimization nearly impossible.

### Definition 8.10.11 (Barren Plateau Phenomenon)

A PQC exhibits a **barren plateau** if the variance of the cost function gradient vanishes exponentially:
$$
\text{Var}\left[\frac{\partial C}{\partial \theta_j}\right] \in O\left(\frac{1}{2^{an}}\right)
$$
where $n$ is the number of qubits and $a > 0$ is a constant.

**Intuition:** Random parameter initialization leads to gradients concentrated near zero. The optimizer receives no signal to improve.

**Quantum Connection:** This is the quantum analog of the **vanishing gradient problem** in deep classical neural networks, but more severe due to exponential Hilbert space growth.

---

### Theorem 8.10.1 (Barren Plateau for Global Cost Functions)

For a global cost function $C = \langle \hat{O} \rangle$ (where $\hat{O}$ acts on all qubits) and a PQC with random layers forming a unitary 2-design, the gradient variance satisfies:
$$
\text{Var}\left[\frac{\partial C}{\partial \theta}\right] \in O\left(\frac{1}{2^n}\right)
$$

**Proof sketch:** Random unitaries create states uniformly distributed over the Bloch sphere. The overlap with a fixed observable vanishes exponentially as dimension grows. □

**Example 8.10.6 (Barren Plateau — Gradient Variance Scaling):**
Consider a global cost function $C = \langle Z_1 \otimes Z_2 \otimes \cdots \otimes Z_n \rangle$ with a random PQC. The gradient variance scales as $\text{Var}[\partial C / \partial \theta] \sim 1/2^n$. Numerically:

| Qubits $n$ | Hilbert space dim $2^n$ | $\text{Var}[\partial C / \partial \theta]$ | Required samples to resolve gradient |
|------------|-------------------------|--------------------------------------------|---------------------------------------|
| 2          | 4                       | $\sim 0.25$                                | $\sim 10$                             |
| 5          | 32                      | $\sim 0.031$                               | $\sim 100$                            |
| 10         | 1024                    | $\sim 9.8 \times 10^{-4}$                  | $\sim 3{,}000$                        |
| 20         | $10^6$                  | $\sim 9.5 \times 10^{-7}$                  | $\sim 3 \times 10^6$                  |
| 50         | $10^{15}$               | $\sim 10^{-15}$                            | $\sim 10^{15}$ (infeasible)           |

At $n = 50$ qubits, the gradient signal is indistinguishable from measurement noise. This is why training random deep circuits on $\geq 20$ qubits is practically impossible without structured ansatzes or local cost functions.

---

### Mitigation Strategies

| Strategy | Description | Trade-off |
|----------|-------------|-----------|
| **Local cost functions** | Measure observables on few qubits (e.g., single-qubit Pauli) | Reduces expressibility; may not capture global structure |
| **Problem-inspired ansatzes** | Use structure from problem Hamiltonian (e.g., QAOA) | Requires domain knowledge; less general |
| **Shallow circuits** | Limit circuit depth | Reduces expressibility; may underfit |
| **Correlated initialization** | Initialize parameters near known good solutions | Requires prior knowledge or warm start |
| **Layerwise training** | Train one layer at a time | Slower; may get stuck in local minima |

**Table 8.10.4: Barren Plateau Mitigation Strategies**

**Current Research:** Understanding precisely which ansatzes avoid barren plateaus and developing initialization schemes remain open problems.

---

## 8.10.7 Quantum Advantage for Machine Learning

When (if ever) do quantum algorithms outperform classical ML? This question is central to QML's future.

### Definition 8.10.12 (Quantum ML Advantage)

A quantum ML algorithm achieves **quantum advantage** over classical ML if:
1. **Provable speedup:** Polynomial or exponential complexity improvement (worst-case or average-case)
2. **Practical advantage:** Quantum implementation outperforms state-of-the-art classical methods on real datasets
3. **Resource efficiency:** Speedup holds accounting for quantum state preparation and measurement costs

**Example 8.10.7 (Quantum vs. Classical — Classifying Quantum States):**
Consider the task: given copies of an unknown $n$-qubit state $\rho$, determine whether $\rho$ is entangled or separable.
**Classical approach:** Perform full quantum state tomography, reconstructing all $4^n - 1$ parameters of $\rho$. For $n = 10$ qubits, this requires estimating $\sim 10^6$ parameters, each needing $\sim 1{,}000$ measurements $\Rightarrow$ total $\sim 10^9$ measurements.
**Quantum approach:** A quantum classifier directly measures entanglement witnesses $\hat{W}$ with $\text{Tr}(\hat{W}\rho) < 0$ iff $\rho$ is entangled. This requires $O(\text{poly}(n))$ measurements — for $n = 10$, roughly $\sim 10^3$ measurements.
**Comparison for $n = 10$ qubits:**

| Method                       | Measurements needed | Time (at $10^6$ meas/sec) |
|------------------------------|---------------------|---------------------------|
| Classical (tomography + SVM) | $\sim 10^9$         | $\sim 17$ minutes         |
| Quantum (direct witness)     | $\sim 10^3$         | $\sim 1$ microsecond      |

This is a genuine exponential separation: the quantum approach avoids reconstructing the full classical description of the quantum state. This advantage is specific to **quantum data** — for classical datasets, such separations are not known.

---

### Where Quantum May Help

| Task | Potential Quantum Advantage | Evidence |
|------|---------------------------|----------|
| **Quantum data classification** | Process quantum sensor data without tomography | Strong (avoids exponential reconstruction) |
| **Kernel evaluation** | Exponentially large feature spaces | Theoretical (but measurement bottleneck) |
| **Optimization** | Variational algorithms for NP-hard problems | Weak (classical heuristics competitive) |
| **Generative modeling** | Sampling from complex distributions | Theoretical (quantum GANs, Boltzmann machines) |
| **Dimensionality reduction** | Quantum PCA for exponentially large data | Theoretical (requires amplitude encoding) |

**Table 8.10.5: Potential Quantum ML Advantages**

---

### When Quantum Does NOT Help: Dequantization Results

**Theorem 8.10.2 (Dequantization of Quantum ML)**

Many quantum ML algorithms claiming exponential speedups have been **dequantized**: classical algorithms achieve similar performance by:
1. Working with **succinct data structures** (e.g., sample and query access instead of full state)
2. Using **randomized linear algebra** (e.g., random projection, sketching)

**Example:** Quantum recommendation systems (Kerenidis–Prakash) claimed exponential speedup for collaborative filtering. Tang (2018) showed a classical algorithm with polynomial complexity under the same data access assumptions.

**Lesson:** Quantum advantage depends critically on **data access model** and **problem structure**. Exponential speedups often disappear under realistic assumptions.

---

### Current Evidence

As of 2026:
- **No demonstrated quantum advantage** for classical ML tasks on real-world datasets
- **NISQ devices** (50–1,000 qubits, high error rates) insufficient for advantage
- **Theoretical advantage** for quantum-native tasks (classifying quantum states, chemistry)
- **Potential** in hybrid quantum-classical models for specific structured problems

**Quantum Connection:** Quantum ML is in its "pre-ImageNet" era. Classical deep learning showed exponential growth only after algorithmic breakthroughs (backprop, ReLU, dropout) and hardware scaling. QML awaits similar breakthroughs.

---

## 8.10.8 Quantum Neural Networks (QNNs)

Quantum neural networks generalize classical neural networks using quantum operations.

### Definition 8.10.13 (Quantum Neural Network)

A **QNN** is a quantum circuit organized into layers, each performing:
1. **Linear transformation:** Parameterized unitaries $U_\ell(\boldsymbol{\theta}_\ell)$
2. **Nonlinearity:** Entangling gates or measurement-based nonlinearity

**Comparison to classical NNs:**

| Property | Classical NN | Quantum NN |
|----------|--------------|------------|
| **State space** | $\mathbb{R}^n$ (vector) | $\mathcal{H}_{2^n}$ (exponential Hilbert space) |
| **Computation** | Matrix-vector products, nonlinear activation | Unitary evolution, entanglement |
| **Gradient flow** | Backpropagation | Parameter-shift rule |
| **Scalability** | Polynomial parameters in width × depth | Exponential Hilbert space, but polynomial parameters |
| **Hardware** | GPUs (mature) | Quantum processors (NISQ, error-prone) |

**Table 8.10.6: Classical vs. Quantum Neural Networks**

**Quantum Connection:** QNNs do NOT have a natural analog of ReLU or sigmoid activations. Nonlinearity arises from entanglement and measurement, not point-wise functions.

---

### Quantum Reservoir Computing

**Reservoir computing:** Use a fixed random quantum circuit as a "reservoir"; train only the output layer (classical linear regression).

**Advantages:**
- Avoids barren plateaus (no training of quantum parameters)
- Exploits quantum dynamics for temporal processing
- NISQ-friendly (noise acts as part of reservoir)

**Example:** Time-series forecasting using 20-qubit random circuits on IBM hardware shows competitive performance with classical echo state networks.

---

## 8.10.9 Near-Term Applications (NISQ Era)

NISQ devices (Noisy Intermediate-Scale Quantum) have 50–1,000 qubits with error rates $10^{-3}$ to $10^{-2}$ per gate. QML applications must tolerate noise.

### NISQ-Suitable Algorithms

| Application | Algorithm | Status | Example |
|-------------|-----------|--------|---------|
| **Quantum chemistry** | Variational Quantum Eigensolver (VQE) | Most mature | Drug discovery (e.g., simulating molecules) |
| **Combinatorial optimization** | Quantum Approximate Optimization Algorithm (QAOA) | Experimental | MaxCut, portfolio optimization |
| **Generative modeling** | Quantum GANs, quantum Boltzmann machines | Research | Generating synthetic data |
| **Classification** | Variational quantum classifiers | Proof-of-concept | Small datasets (Iris, MNIST subsets) |

**Table 8.10.7: NISQ-Era QML Applications**

---

### Case Study: Quantum Chemistry

**Problem:** Compute ground-state energy of molecule $H$ (Hamiltonian).

**VQE approach:**
1. Encode molecular Hamiltonian as Pauli sum: $H = \sum_i c_i \hat{P}_i$
2. Prepare ansatz $|\psi(\boldsymbol{\theta})\rangle$ (e.g., unitary coupled cluster)
3. Measure $E(\boldsymbol{\theta}) = \langle \psi(\boldsymbol{\theta}) | H | \psi(\boldsymbol{\theta}) \rangle$
4. Optimize $\boldsymbol{\theta}$ to minimize $E(\boldsymbol{\theta})$

**Quantum advantage:** Classical simulation requires exponential memory for correlated systems. Quantum computers represent states naturally.

**Current scale:** 10–20 qubit molecules (H₂, LiH, BeH₂) simulated on IBM/Google hardware. Practical advantage over classical methods (CCSD(T), DMRG) remains debated.

---

## 8.10.10 Current Hardware Landscape (2026)

### Leading Platforms

| Provider | Qubit Count | Qubit Type | Error Rate (1-qubit / 2-qubit) | Connectivity | Availability |
|----------|-------------|------------|---------------------------------|--------------|--------------|
| **IBM Quantum** | 127–433 qubits | Superconducting | $10^{-4}$ / $10^{-3}$ | Heavy-hex lattice | Cloud (free + paid) |
| **Google Quantum AI** | 70 qubits (Sycamore) | Superconducting | $10^{-3}$ / $10^{-2}$ | 2D grid | Limited access |
| **IonQ** | 32 qubits | Trapped ions | $10^{-3}$ / $10^{-2}$ | All-to-all | Cloud (AWS, Azure) |
| **Rigetti** | 80 qubits | Superconducting | $10^{-3}$ / $10^{-2}$ | 2D lattice | Cloud |
| **Xanadu** | 216 modes | Photonic | N/A (continuous-variable) | Programmable interferometer | Cloud |

**Table 8.10.8: Quantum Hardware Landscape (2026)**

---

### Key Metrics for QML

- **Qubit count:** More qubits → larger feature spaces, but exponentially harder to control
- **Gate fidelity:** High error rates limit circuit depth (~100–200 gates before noise dominates)
- **Connectivity:** All-to-all (trapped ions) vs. nearest-neighbor (superconducting) affects circuit compilation
- **Coherence time:** $T_2 \sim 100~\mu\text{s}$ (superconducting) limits algorithm duration

**Bottleneck:** Error rates remain $\sim 100\times$ too high for fault-tolerant quantum computing. NISQ algorithms must complete in $<1~\text{ms}$ before decoherence.

---

## Summary

Quantum machine learning bridges quantum computing and classical ML, with four main approaches:
1. **Quantum-enhanced ML:** Encode classical data, process with PQCs, optimize classically (NISQ focus)
2. **Quantum kernels:** Exploit exponentially large feature spaces via quantum feature maps
3. **Variational algorithms:** VQC, VQE combine quantum circuits with classical optimizers
4. **Quantum-native ML:** Process quantum data directly (chemistry, sensors)

**Key challenges:**
- **Barren plateaus:** Vanishing gradients in deep circuits limit trainability
- **Dequantization:** Many claimed speedups have efficient classical analogs
- **NISQ constraints:** Current hardware too noisy for large-scale advantage

**Future outlook:** QML awaits algorithmic breakthroughs and fault-tolerant hardware. Near-term value in quantum chemistry and hybrid algorithms; long-term potential in quantum data processing and generative models.

---

## Exercises

### Foundations (★)

**8.10.1.** Encode the classical vector $\mathbf{x} = (0.6, 0.8)$ using amplitude encoding into a 1-qubit state. Verify normalization.

**8.10.2.** For angle encoding with $\theta_j = 2\pi x_j$, compute the quantum state $|\psi(\mathbf{x})\rangle$ for $\mathbf{x} = (0.25, 0.75)$ on 2 qubits initialized to $|00\rangle$.

**8.10.3.** Show that basis encoding of $n$ bits requires $n$ qubits, while amplitude encoding of $2^n$ real numbers also requires $n$ qubits.

---

### Intermediate (★★)

**8.10.4.** Compute the quantum kernel $k_Q(\mathbf{x}, \mathbf{x}')$ for the feature map $|\phi(\mathbf{x})\rangle = \cos(x)|0\rangle + \sin(x)|1\rangle$ with $\mathbf{x} = \pi/4$ and $\mathbf{x}' = \pi/3$.

**8.10.5.** Design a 2-layer PQC for 3 qubits using $R_Y$ rotations and nearest-neighbor CNOTs. How many parameters does it have?

**8.10.6.** Prove that for a single-parameter gate $U(\theta) = e^{-i\theta Z/2}$, the parameter-shift rule gives:
$$
\frac{\partial \langle Z \rangle}{\partial \theta} = \frac{\langle Z \rangle_{\theta+\pi/2} - \langle Z \rangle_{\theta-\pi/2}}{2}
$$

**8.10.7.** Explain why amplitude encoding has a measurement bottleneck: extracting all amplitudes from $|\psi\rangle = \sum_i \alpha_i |i\rangle$ requires $O(2^n)$ measurements.

---

### Advanced (★★★)

**8.10.8.** Derive the gradient variance for a global cost function $C = \langle \psi(\theta) | \hat{O} | \psi(\theta) \rangle$ when $|\psi(\theta)\rangle$ is a Haar-random state. Show $\text{Var}[\partial C / \partial \theta] \in O(1/2^n)$.

**8.10.9.** Propose a NISQ-friendly ansatz for a binary classification problem on the Iris dataset (4 features, 2 classes). Specify:
- Encoding scheme
- Number of qubits
- PQC structure (depth, gates)
- Measurement observable

**8.10.10.** Research the dequantization result for quantum recommendation systems (Tang, 2018). Summarize the classical algorithm and explain what assumption enables polynomial complexity.

**8.10.11.** Design a quantum reservoir computing architecture for time-series prediction. Specify:
- Input encoding (sequential data)
- Reservoir (fixed random circuit)
- Output readout (classical regression)

---

## Related Topics

- **Chapter 8.7:** Variational Quantum Eigensolver (VQE) and Quantum Approximate Optimization Algorithm (QAOA)
- **Chapter 8.8:** Quantum error correction and noise models (error mitigation for NISQ)
- **Chapter 8.11:** Quantum complexity theory (BQP, query complexity, lower bounds)
- **Classical ML:** Support vector machines, neural network training, kernel methods
- **Optimization:** Gradient descent, stochastic optimization, evolutionary algorithms

---

## Further Reading

1. **Biamonte et al. (2017):** "Quantum machine learning," *Nature* — foundational review
2. **McClean et al. (2018):** "Barren plateaus in quantum neural network training landscapes," *Nature Communications* — barren plateau analysis
3. **Havlíček et al. (2019):** "Supervised learning with quantum-enhanced feature spaces," *Nature* — quantum kernel methods
4. **Cerezo et al. (2021):** "Variational quantum algorithms," *Nature Reviews Physics* — comprehensive VQA survey
5. **Tang (2019):** "A quantum-inspired classical algorithm for recommendation systems," *STOC* — dequantization result
6. **Schuld & Petruccione (2021):** *Machine Learning with Quantum Computers* — textbook

---

**End of Chapter 8.10**

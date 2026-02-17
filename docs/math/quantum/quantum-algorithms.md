# 8.7 Quantum Algorithms

## Prerequisites

- Quantum mechanics foundations (Chapter 8.1)
- Quantum gates and circuits (Chapter 8.5)
- Tensor products and entanglement (Chapter 8.3)
- Linear algebra (eigenvalues, unitary matrices)
- Basic computational complexity theory (P, NP)

## Introduction

Quantum algorithms exploit superposition, entanglement, and interference to solve computational problems with complexity advantages over classical algorithms. This chapter explores the foundational quantum algorithms that demonstrate exponential or polynomial speedups, as well as near-term variational approaches for quantum machine learning.

**Key Insight:** Quantum parallelism allows evaluating a function on exponentially many inputs simultaneously, but extracting useful information requires careful algorithm design to amplify correct answers through interference.

---

## 8.7.1 Quantum Parallelism

**Definition 8.7.1 (Quantum Parallelism):**
For a function $f: \{0,1\}^n \to \{0,1\}^m$, a quantum computer can evaluate $f$ on all $2^n$ possible inputs simultaneously by applying a unitary operator $U_f$ to a superposition state:

$$
U_f |x\rangle |y\rangle = |x\rangle |y \oplus f(x)\rangle
$$

where $\oplus$ denotes bitwise XOR and $|x\rangle = |x_1, \ldots, x_n\rangle$ is a computational basis state.

**Construction:**
Starting with $|\psi_0\rangle = |0\rangle^{\otimes n}$, apply Hadamard gates to create uniform superposition:

$$
H^{\otimes n}|0\rangle^{\otimes n} = \frac{1}{\sqrt{2^n}} \sum_{x \in \{0,1\}^n} |x\rangle
$$

Applying $U_f$ with ancilla $|0\rangle$:

$$
U_f \left( \frac{1}{\sqrt{2^n}} \sum_x |x\rangle |0\rangle \right) = \frac{1}{\sqrt{2^n}} \sum_x |x\rangle |f(x)\rangle
$$

This state encodes $f(x)$ for all $2^n$ inputs simultaneously.

**Measurement Challenge:**
Measuring this superposition collapses to a single $(x, f(x))$ pair. The algorithmic challenge is designing interference patterns that extract global properties of $f$ rather than individual values.

**Example 8.7.1 (Quantum Parallelism with 2 Qubits):**
Let $n = 2$ and $f:\{0,1\}^2 \to \{0,1\}$ with $f(00) = 0,\; f(01) = 1,\; f(10) = 1,\; f(11) = 0$. Starting from $|00\rangle$:

*Step 1 — Create superposition:*
$$
H^{\otimes 2}|00\rangle = \frac{1}{2}\big(|00\rangle + |01\rangle + |10\rangle + |11\rangle\big)
$$

*Step 2 — Append ancilla $|0\rangle$ and apply $U_f$:*
$$
U_f \left[\frac{1}{2}\sum_x |x\rangle|0\rangle\right] = \frac{1}{2}\big(|00\rangle|0\rangle + |01\rangle|1\rangle + |10\rangle|1\rangle + |11\rangle|0\rangle\big)
$$

All four values of $f$ are encoded simultaneously in a single quantum state using just one application of $U_f$. Classically, four separate evaluations would be required.

> **Connection to ML:** Quantum parallelism underlies quantum speedups in optimization and sampling tasks relevant to machine learning, such as exploring large solution spaces in QAOA.

---

## 8.7.2 Deutsch-Jozsa Algorithm

**Problem Statement:**
Given a function $f: \{0,1\}^n \to \{0,1\}$ promised to be either:
- **Constant:** $f(x)$ same for all $x$
- **Balanced:** $f(x) = 0$ for exactly half of inputs, $f(x) = 1$ for other half

Determine which with minimum queries.

**Classical Complexity:** Requires $2^{n-1} + 1$ queries in worst case.

**Quantum Complexity:** **1 query** (exponential speedup).

**Algorithm:**

```
Circuit Diagram (n=3):

|0⟩ ──H── ┬──────── H ──M──
|0⟩ ──H── │         H ──M──
|0⟩ ──H── │  U_f    H ──M──
|1⟩ ──H── ┴──────── ────────

Phase Oracle: U_f|x⟩|y⟩ = |x⟩|y ⊕ f(x)⟩
With |y⟩=|−⟩, creates phase kickback: |x⟩|−⟩ → (−1)^f(x)|x⟩|−⟩
```

**Example 8.7.2 (Oracle Construction for $f(x) = x$):**
Consider the 1-qubit balanced function $f(0) = 0,\; f(1) = 1$ (identity). The oracle $U_f$ must act as:
$$
U_f|x\rangle|y\rangle = |x\rangle|y \oplus x\rangle
$$

This is exactly a CNOT gate (control = input qubit, target = ancilla):

```
Oracle circuit for f(x) = x:

|x⟩ ──●── |x⟩
      │
|y⟩ ──⊕── |y ⊕ x⟩
```

Verify: $U_f|0\rangle|0\rangle = |0\rangle|0 \oplus 0\rangle = |0\rangle|0\rangle$ and $U_f|1\rangle|0\rangle = |1\rangle|0 \oplus 1\rangle = |1\rangle|1\rangle$. The oracle correctly encodes $f(0)=0$ and $f(1)=1$.

**Definition 8.7.2 (Deutsch-Jozsa Circuit):**
1. Initialize: $|0\rangle^{\otimes n} |1\rangle$
2. Apply $H^{\otimes (n+1)}$: creates superposition
3. Apply phase oracle $U_f$: encodes $f$ as phase
4. Apply $H^{\otimes n}$ to first $n$ qubits: interference
5. Measure first $n$ qubits

**Mathematical Analysis:**

After step 2:
$$
|\psi_1\rangle = \frac{1}{\sqrt{2^n}} \sum_{x \in \{0,1\}^n} |x\rangle \otimes \frac{|0\rangle - |1\rangle}{\sqrt{2}}
$$

After phase oracle (step 3):
$$
|\psi_2\rangle = \frac{1}{\sqrt{2^n}} \sum_x (-1)^{f(x)} |x\rangle \otimes |{-}\rangle
$$

After final Hadamard (step 4), amplitude of $|0\rangle^{\otimes n}$:
$$
\alpha_{0\ldots0} = \frac{1}{2^n} \sum_x (-1)^{f(x)}
$$

**Result:**
- **Constant $f$:** $\alpha_{0\ldots0} = \pm 1$ (all phases same) → measure $|0\rangle^{\otimes n}$ with probability 1
- **Balanced $f$:** $\alpha_{0\ldots0} = 0$ (phases cancel) → measure $|0\rangle^{\otimes n}$ with probability 0

**Example 8.7.3 (Deutsch's Algorithm for Balanced $f(0)=0, f(1)=1$):**
Trace through the $n=1$ case (Deutsch problem) step by step.

*Step 1 — Initialize:* $|\psi_0\rangle = |0\rangle|1\rangle$

*Step 2 — Apply $H \otimes H$:*
$$
|\psi_1\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}} \otimes \frac{|0\rangle - |1\rangle}{\sqrt{2}} = |{+}\rangle|{-}\rangle
$$

*Step 3 — Apply phase oracle* (using $f(0)=0, f(1)=1$, so $(-1)^{f(x)} = (-1)^x$):
$$
|\psi_2\rangle = \frac{(-1)^{f(0)}|0\rangle + (-1)^{f(1)}|1\rangle}{\sqrt{2}} \otimes |{-}\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}} \otimes |{-}\rangle = |{-}\rangle|{-}\rangle
$$

*Step 4 — Apply $H$ to first qubit:*
$$
H|{-}\rangle = |1\rangle \quad \Rightarrow \quad |\psi_3\rangle = |1\rangle|{-}\rangle
$$

*Step 5 — Measure first qubit:* Get $|1\rangle$ with certainty. Since $|1\rangle \neq |0\rangle$, conclude $f$ is **balanced**. One query sufficed, versus two classically.

**Why Exponentially Faster:**
Quantum interference allows extracting a global property (sum of phases) that would classically require examining exponentially many individual values.

---

## 8.7.3 Quantum Fourier Transform

**Definition 8.7.3 (Quantum Fourier Transform):**
The QFT on $n$ qubits is the unitary transformation:

$$
\text{QFT}_N |j\rangle = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} e^{2\pi ijk/N} |k\rangle
$$

where $N = 2^n$ and $j, k \in \{0, 1, \ldots, N-1\}$.

**Product Representation:**
For $j = j_1 j_2 \ldots j_n$ in binary:

$$
\text{QFT}_N |j_1, \ldots, j_n\rangle = \frac{1}{\sqrt{2^n}} \bigotimes_{k=1}^n \left( |0\rangle + e^{2\pi i \cdot 0.j_k j_{k+1} \ldots j_n} |1\rangle \right)
$$

where $0.j_k j_{k+1} \ldots j_n$ is binary fraction $\sum_{\ell=k}^n j_\ell 2^{-(l-k+1)}$.

**Circuit Construction:**

```
QFT Circuit (n=3):

|j₁⟩ ──H── R₂ ── R₃ ────────────── ┬── |k₃⟩
           │     │                  │
|j₂⟩ ────  ●  ── │ ── H ── R₂ ──── │ ─ |k₂⟩
                  │         │       │
|j₃⟩ ──────────── ● ─────── ● ── H ┴── |k₁⟩
                                    SWAP

R_k = [1      0    ]  controlled phase gate
      [0  e^(2πi/2^k)]
```

**Complexity:**
- **Classical FFT:** $O(N \log N) = O(2^n \cdot n)$ operations
- **Quantum QFT:** $O(n^2)$ gates (exponential speedup)

**Relation to Classical FFT:**
Both compute discrete Fourier transform, but QFT operates on quantum superposition amplitudes rather than classical data vectors. QFT outputs quantum state encoding Fourier coefficients as amplitudes.

**Key Property (Periodicity Detection):**
If input state has periodic structure with period $r$, QFT concentrates amplitude on states corresponding to multiples of $N/r$, enabling efficient period finding.

**Example 8.7.4 (QFT on 2 Qubits — Compute $\text{QFT}_4|3\rangle$):**
With $n=2$, we have $N = 2^2 = 4$ and input $|j\rangle = |3\rangle = |11\rangle$. Apply the definition:

$$
\text{QFT}_4|3\rangle = \frac{1}{\sqrt{4}} \sum_{k=0}^{3} e^{2\pi i \cdot 3k/4}\, |k\rangle = \frac{1}{2}\sum_{k=0}^{3} e^{i\pi k \cdot 3/2}\, |k\rangle
$$

Evaluate each term ($\omega = e^{i\pi/2} = i$, so $e^{2\pi i \cdot 3k/4} = i^{3k}$):

| $k$ | $i^{3k}$   | Value |
|-----|------------|-------|
| 0   | $i^0 = 1$  | $1$   |
| 1   | $i^3 = -i$ | $-i$  |
| 2   | $i^6 = -1$ | $-1$  |
| 3   | $i^9 = i$  | $i$   |

$$
\text{QFT}_4|3\rangle = \frac{1}{2}\big(|0\rangle - i|1\rangle - |2\rangle + i|3\rangle\big)
$$

*Verify via product form:* $j = 11$ in binary, so $j_1 = 1, j_2 = 1$:
$$
\frac{1}{2}\big(|0\rangle + e^{2\pi i \cdot 0.1}|1\rangle\big) \otimes \big(|0\rangle + e^{2\pi i \cdot 0.11}|1\rangle\big) = \frac{1}{2}(|0\rangle - |1\rangle) \otimes (|0\rangle - i|1\rangle)
$$
Expanding: $\frac{1}{2}(|00\rangle - i|01\rangle - |10\rangle + i|11\rangle)$, which matches.

> **Connection to Shor's Algorithm:** QFT is the key subroutine enabling efficient period finding, which reduces integer factorization to a quantum computation.

---

## 8.7.4 Shor's Algorithm

**Problem:** Factor integer $N$ into primes.

**Classical Complexity:** Best known $O(e^{1.9(\log N)^{1/3}(\log \log N)^{2/3}})$ (sub-exponential but not polynomial).

**Quantum Complexity:** $O((\log N)^3)$ (polynomial, **exponential speedup**).

**Reduction to Period Finding:**

**Theorem 8.7.1:** Factoring $N$ reduces to finding period $r$ of function $f(x) = a^x \bmod N$ for random $a < N$ coprime to $N$.

**Why:** If $a^r \equiv 1 \pmod{N}$ and $r$ even, then $a^{r/2} \pm 1$ likely share non-trivial factor with $N$:
$$
\gcd(a^{r/2} - 1, N), \quad \gcd(a^{r/2} + 1, N)
$$

**Algorithm Flow:**

```
Shor's Algorithm:

1. Classical: Choose random a < N, gcd(a,N)=1
                │
                ▼
2. Quantum Period Finding:
   ┌─────────────────────────────────────┐
   │ |0⟩⊗n ──H⊗n── U_f ── QFT⁻¹ ── M    │
   │                │                    │
   │ |0⟩⊗m ─────────┴─────────────────  │
   │                                     │
   │ U_f: |x⟩|y⟩ → |x⟩|y·a^x mod N⟩    │
   └─────────────────────────────────────┘
                │
                ▼ Measure: get s/r
                │
                ▼
3. Classical: Continued fractions to find r
                │
                ▼
4. Classical: Compute gcd(a^(r/2)±1, N)
                │
                ▼
           Factors of N
```

**Quantum Subroutine (Period Finding):**

1. **Superposition:** $H^{\otimes n}|0\rangle^{\otimes n} = \frac{1}{\sqrt{2^n}} \sum_{x=0}^{2^n-1} |x\rangle$

2. **Function Evaluation:**
$$
\frac{1}{\sqrt{2^n}} \sum_x |x\rangle |a^x \bmod N\rangle
$$

3. **Measurement of second register:** Collapses to state with periodic structure:
$$
\frac{1}{\sqrt{\lfloor 2^n/r \rfloor}} \sum_{k=0}^{\lfloor 2^n/r \rfloor - 1} |x_0 + kr\rangle
$$
where $a^{x_0} \bmod N$ was measured outcome.

4. **QFT:** Transforms periodic state to peaks at multiples of $2^n/r$:
$$
\text{QFT} \left( \sum_{k=0}^{M-1} |kr\rangle \right) \approx \sqrt{\frac{r}{2^n}} \sum_{\ell=0}^{r-1} \left| \frac{\ell \cdot 2^n}{r} \right\rangle
$$

5. **Measure:** Get $s \approx \ell \cdot 2^n / r$, then $r \approx \ell \cdot 2^n / s$. Classical continued fractions extract $r$.

**Example 8.7.5 (Period Finding for $f(x) = 2^x \bmod 15$):**
Factor $N = 15$ using $a = 2$. Compute $f(x)$ classically to see the period:

| $x$             | 0   | 1   | 2   | 3   | 4   | 5   | 6   | 7   |
|-----------------|-----|-----|-----|-----|-----|-----|-----|-----|
| $2^x \bmod 15$  | 1   | 2   | 4   | 8   | 1   | 2   | 4   | 8   |

The period is $r = 4$. In the quantum algorithm with $n = 4$ qubits ($2^n = 16$):

After QFT, amplitude peaks appear at multiples of $2^n / r = 16/4 = 4$, so measurement outcomes are $s \in \{0, 4, 8, 12\}$.

Suppose we measure $s = 12$. Then $s/2^n = 12/16 = 3/4$. Continued fraction expansion: $3/4$ is already in lowest terms, so the denominator gives $r = 4$.

*Extract factors:* $r = 4$ is even, so compute:
$$
\gcd(2^{4/2} - 1, 15) = \gcd(3, 15) = 3, \quad \gcd(2^{4/2} + 1, 15) = \gcd(5, 15) = 5
$$

We find $15 = 3 \times 5$.

**Cryptographic Impact:**
Shor's algorithm breaks RSA encryption, which relies on hardness of factoring large $N = pq$. Quantum computers with sufficient qubits ($\sim 2000$ logical qubits) would compromise current public-key infrastructure.

---

## 8.7.5 Grover's Algorithm

**Problem:** Unstructured search in database of $N$ items for marked item satisfying $f(x) = 1$.

**Classical Complexity:** $O(N)$ queries (must check $\sim N/2$ items on average).

**Quantum Complexity:** $O(\sqrt{N})$ queries (**quadratic speedup**).

**Oracle:**
Black-box operator $U_\omega$ that marks target state $|\omega\rangle$:
$$
U_\omega |x\rangle = \begin{cases} -|x\rangle & \text{if } x = \omega \\ |x\rangle & \text{otherwise} \end{cases}
$$

Equivalently: $U_\omega = I - 2|\omega\rangle\langle\omega|$ (reflection about $|\omega\rangle^\perp$).

**Grover Operator:**

**Definition 8.7.4 (Grover Iteration):**
$$
G = (2|s\rangle\langle s| - I) U_\omega = U_s U_\omega
$$

where $|s\rangle = \frac{1}{\sqrt{N}} \sum_x |x\rangle$ is uniform superposition, and $U_s = 2|s\rangle\langle s| - I$ is diffusion operator (reflection about $|s\rangle$).

**Geometric Interpretation:**
- **Subspace:** Span$\{|\omega\rangle, |s'\rangle\}$ where $|s'\rangle$ is uniform superposition over non-targets
- **Action:** $G$ is rotation by $\theta \approx 2/\sqrt{N}$ toward $|\omega\rangle$
- **Iterations:** After $\sim \frac{\pi}{4}\sqrt{N}$ iterations, state $\approx |\omega\rangle$

**Amplitude Evolution Diagram:**

```
Grover Search Amplitude Evolution:

        Amplitude
            ▲
    1.0   ──┼──────────────●         Target |ω⟩
            │            ╱ │
            │          ╱   │
    0.5   ──┼────────●─────┼
            │      ╱       │
            │    ╱         │
    0.0   ──●──●───────────┼──────   Uniform |s⟩
            │              │
   -0.5   ──┼──────────────┼
            │              │
            └──────────────┴──────▶
            0   k/4   k/2   k    Iterations

After k ≈ π√N/4 iterations:
- Target amplitude ≈ 1
- Non-target amplitudes ≈ 0
- Measure to get ω with high probability
```

**Algorithm:**

```
Grover Circuit (N = 2^n items):

|0⟩⊗n ──H⊗n── ┬─────────┬──── ... ── M
              │    G    │
              │  ╱  ╲   │  (repeat
              │ Uω  Us  │   ~√N times)
              └─────────┘

Full Grover Iteration G:
1. Oracle Uω:  mark target
2. H⊗n:        go to computational basis
3. X⊗n:        flip all bits
4. Multi-controlled-Z: mark |0⟩⊗n
5. X⊗n:        flip back
6. H⊗n:        return to superposition
```

**Optimality:**

**Theorem 8.7.2 (Bennett et al.):** Any quantum algorithm solving unstructured search requires $\Omega(\sqrt{N})$ oracle queries. Grover's algorithm is asymptotically optimal.

**Applications:**
- Database search
- Satisfiability solving (SAT)
- Constraint satisfaction
- Accelerating classical algorithms with $O(N)$ subroutines

> **Connection to ML:** Grover provides quadratic speedup for optimization problems requiring exhaustive search, applicable to hyperparameter tuning or exploring discrete solution spaces.

---

## 8.7.6 Variational Quantum Eigensolver (VQE)

**Motivation:** Near-term quantum computers have limited qubits and high noise, preventing deep circuits required for Shor/Grover. VQE uses shallow parameterized circuits with classical optimization.

**Problem:** Find ground state energy $E_0$ and state $|\psi_0\rangle$ of Hamiltonian $H$:
$$
E_0 = \min_{|\psi\rangle} \langle \psi | H | \psi \rangle
$$

**Definition 8.7.5 (Parameterized Quantum Circuit):**
A unitary $U(\boldsymbol{\theta})$ where $\boldsymbol{\theta} = (\theta_1, \ldots, \theta_p) \in \mathbb{R}^p$ are classical parameters controlling rotation gates:
$$
U(\boldsymbol{\theta}) = U_L(\theta_L) \cdots U_2(\theta_2) U_1(\theta_1)
$$

Each layer $U_\ell$ typically contains single-qubit rotations (RX, RY, RZ) and entangling gates (CNOT).

**VQE Algorithm:**

```
VQE Hybrid Loop:

Classical                 Quantum
Computer                  Computer
    │                         │
    │  1. Parameters θ        │
    ├─────────────────────────▶
    │                         │
    │         ┌──────────────┐│
    │         │ |0⟩⊗n        ││
    │         │   ↓          ││
    │         │ U(θ)         ││  Prepare
    │         │   ↓          ││  |ψ(θ)⟩
    │         │ |ψ(θ)⟩       ││
    │         │   ↓          ││
    │         │ Measure ⟨H⟩  ││
    │         └──────────────┘│
    │                         │
    │  2. Energy E(θ)         │
    │◀─────────────────────────┤
    │                         │
    │  3. Update θ            │
    │     (gradient descent,  │
    │      COBYLA, etc.)      │
    │                         │
    └─────────────────────────┘
         Repeat until
         convergence
```

**Steps:**

1. **Initialize** parameters $\boldsymbol{\theta}^{(0)}$
2. **Prepare** trial state $|\psi(\boldsymbol{\theta}^{(t)})\rangle = U(\boldsymbol{\theta}^{(t)})|0\rangle^{\otimes n}$ on quantum computer
3. **Measure** expectation $E(\boldsymbol{\theta}^{(t)}) = \langle \psi(\boldsymbol{\theta}^{(t)}) | H | \psi(\boldsymbol{\theta}^{(t)}) \rangle$
4. **Update** $\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - \alpha \nabla_{\boldsymbol{\theta}} E(\boldsymbol{\theta}^{(t)})$ using classical optimizer
5. **Repeat** until convergence

**Hamiltonian Measurement:**
Decompose $H = \sum_i c_i P_i$ where $P_i$ are Pauli strings (tensor products of $\{I, X, Y, Z\}$). Measure each $\langle P_i \rangle$ separately:
$$
\langle H \rangle = \sum_i c_i \langle P_i \rangle
$$

**Applications:**
- Molecular chemistry (finding ground state energies)
- Material science
- Optimization problems encoded as Ising models
- Quantum machine learning (training parameterized quantum models)

**Challenges:**
- Barren plateaus: gradients vanish exponentially with depth in random circuits
- Local minima: classical optimizer may get stuck
- Measurement overhead: estimating $\langle H \rangle$ requires many shots

---

## 8.7.7 Quantum Approximate Optimization Algorithm (QAOA)

**Problem:** Find approximate solutions to combinatorial optimization:
$$
\max_{z \in \{0,1\}^n} C(z)
$$

where $C: \{0,1\}^n \to \mathbb{R}$ is cost function (e.g., MaxCut, graph coloring).

**Encoding:** Map cost to Hamiltonian $H_C$ where $C(z)$ is eigenvalue of $|z\rangle$:
$$
H_C = \sum_{\text{clauses}} c_\alpha \prod_{i \in \alpha} Z_i
$$

**Definition 8.7.6 (QAOA Circuit):**
For depth $p$, alternate between cost and mixer Hamiltonians:
$$
|\psi(\boldsymbol{\beta}, \boldsymbol{\gamma})\rangle = U_B(\beta_p) U_C(\gamma_p) \cdots U_B(\beta_1) U_C(\gamma_1) |s\rangle
$$

where:
- $|s\rangle = |+\rangle^{\otimes n}$ is uniform superposition (easy-to-prepare state)
- $U_C(\gamma) = e^{-i\gamma H_C}$: encodes cost (problem-specific)
- $U_B(\beta) = e^{-i\beta H_B}$: mixer Hamiltonian $H_B = \sum_i X_i$ (explores solution space)
- $\boldsymbol{\gamma} = (\gamma_1, \ldots, \gamma_p)$, $\boldsymbol{\beta} = (\beta_1, \ldots, \beta_p)$ are variational parameters

**Algorithm:**

```
QAOA Circuit (p=2 layers):

|0⟩ ──H── Rz(γ₁)── Rz(γ₁)── Rx(β₁)── Rz(γ₂)── Rz(γ₂)── Rx(β₂)── M
          │        │                 │        │
|0⟩ ──H── │     ── CNOT ─── Rx(β₁)── │     ── CNOT ─── Rx(β₂)── M
          │                          │
|0⟩ ──H── CNOT ───────────── Rx(β₁)── CNOT ───────────── Rx(β₂)── M

    U_C(γ₁)      U_B(β₁)     U_C(γ₂)      U_B(β₂)

Cost layer U_C: implements e^(-iγH_C) for problem Hamiltonian
Mixer layer U_B: implements e^(-iβH_B) = ∏ᵢ e^(-iβXᵢ) = ∏ᵢ Rx(2β)
```

**Optimization:**
Maximize expected cost $\langle C \rangle_{\boldsymbol{\beta}, \boldsymbol{\gamma}}$ using classical optimizer (same hybrid loop as VQE).

**Performance:**
- $p=1$: typically beats random guessing
- $p \to \infty$: recovers optimal solution (becomes adiabatic quantum computing)
- Small $p$ suitable for near-term quantum devices

**Example (MaxCut):**
For graph $G=(V,E)$, maximize number of edges cut by partition:
$$
C(z) = \frac{1}{2} \sum_{(i,j) \in E} (1 - z_i z_j)
$$

Cost Hamiltonian:
$$
H_C = \sum_{(i,j) \in E} \frac{1}{2}(I - Z_i Z_j)
$$

QAOA with $p=1$ gives expected approximation ratio $> 0.6924$ for 3-regular graphs (beats $0.5$ classical random).

> **Connection to ML:** QAOA is a quantum analog of neural networks with cost/mixer layers analogous to forward propagation and parameter optimization via classical backpropagation. Used for quantum-enhanced optimization in ML pipelines.

---

## 8.7.8 Quantum Complexity Classes

**Definition 8.7.7 (BQP - Bounded-Error Quantum Polynomial Time):**
Class of decision problems solvable by quantum computer in polynomial time with error probability $\leq 1/3$:
$$
\text{BQP} = \{L : \exists \text{ poly-time quantum algorithm } Q \text{ s.t. } \Pr[Q(x) = L(x)] \geq 2/3 \}
$$

**Classical Complexity Classes:**

- **P:** Problems solvable deterministically in polynomial time
- **NP:** Problems with polynomial-time verifiable solutions
- **BPP:** Problems solvable probabilistically in polynomial time with bounded error (classical randomized algorithms)

**Relationships:**

```
Complexity Class Hierarchy:

           NP-complete
              ▲
              │ (likely not)
              │
    ┌─────────┼─────────┐
    │         │         │
    │      ┌──┴──┐      │
    │   P  │     │ NP   │
    │      │ BQP │      │
    │      │     │      │
    │      └──┬──┘      │
    │         │         │
    └─────────┼─────────┘
              │
            BPP
              │
              │
              P

Known Relations:
- P ⊆ BPP ⊆ BQP ⊆ PSPACE
- BPP ⊆ BQP (quantum subsumes classical randomized)
- BQP ⊄ NP (unlikely: would solve NP-complete in poly-time)
- NP ⊄ BQP (quantum can't verify all NP solutions efficiently)
```

**Theorem 8.7.3 (Inclusions):**
$$
\text{P} \subseteq \text{BPP} \subseteq \text{BQP} \subseteq \text{PP} \subseteq \text{PSPACE}
$$

**Key Open Questions:**
1. Is $\text{BQP} = \text{P}$? (Does quantum computing provide real advantage?)
2. Is $\text{BQP} \subseteq \text{NP}$? (Can quantum efficiently solve NP-complete problems?)

**Evidence for BQP ≠ P:**
- **Factoring** (Shor's algorithm): in BQP, believed not in P
- **Discrete log:** in BQP, believed not in P
- Oracle separation: Relative to certain oracles, BQP ≠ P proven

**Evidence for NP ⊄ BQP:**
- No quantum speedup known for NP-complete problems beyond Grover's $\sqrt{N}$ search
- Grover optimal for unstructured search $\Rightarrow$ unlikely exponential speedup for SAT

**Quantum Algorithms in Complexity:**

| Problem | Classical | Quantum | Class |
|---------|-----------|---------|-------|
| Factoring | Sub-exp | $O((\log N)^3)$ | BQP (not known in P) |
| Discrete Log | Sub-exp | $O((\log N)^3)$ | BQP |
| Unstructured Search | $O(N)$ | $O(\sqrt{N})$ | BQP |
| Graph Isomorphism | Quasi-poly | Unknown advantage | NP ∩ coNP |
| SAT (NP-complete) | Exponential | $O(2^{n/2})$ Grover | NP-complete |

**Practical Implications:**
- Quantum computers provide **proven** exponential speedups for specific problems (Shor)
- No quantum algorithm solves general NP-complete problems in polynomial time
- Quantum advantage likely **problem-specific** rather than universal

---

## Summary

| Algorithm | Problem | Classical | Quantum | Speedup Type |
|-----------|---------|-----------|---------|--------------|
| Deutsch-Jozsa | Constant vs Balanced | $O(2^n)$ | $O(1)$ | Exponential |
| Shor | Factoring $N$ | Sub-exp | $O((\log N)^3)$ | Exponential |
| Grover | Unstructured Search | $O(N)$ | $O(\sqrt{N})$ | Quadratic |
| QFT | Fourier Transform | $O(N \log N)$ | $O((\log N)^2)$ | Exponential |
| VQE | Ground State Energy | Exponential | Heuristic (poly-time) | Problem-dependent |
| QAOA | Combinatorial Opt | Exponential | Approximate (poly-time) | Problem-dependent |

**Key Principles:**
1. **Quantum Parallelism:** Compute $f(x)$ for all $x$ simultaneously
2. **Interference:** Amplify correct answers, cancel wrong answers
3. **Measurement:** Extract global properties without collapsing to individual values
4. **Near-term vs Fault-tolerant:** VQE/QAOA work with noisy qubits; Shor/Grover need error correction

**Impact on Machine Learning:**
- **Optimization:** QAOA, VQE for non-convex landscapes
- **Sampling:** Quantum speedup for Boltzmann sampling
- **Kernel Methods:** Quantum kernels for enhanced feature spaces
- **Search:** Grover for hyperparameter/architecture search

---

## Exercises

### Basic Concepts (★)

**Exercise 8.7.1:** Consider quantum parallelism with $n=2$ qubits. Write the explicit superposition state after applying $H^{\otimes 2}$ to $|00\rangle$, and describe what happens when a function oracle $U_f$ is applied.

**Exercise 8.7.2:** In the Deutsch-Jozsa algorithm for $n=1$ (Deutsch problem), verify by explicit calculation that constant and balanced functions produce orthogonal output states.

**Exercise 8.7.3:** Show that the Grover diffusion operator $U_s = 2|s\rangle\langle s| - I$ can be implemented using $H^{\otimes n}$, multi-controlled-NOT, and single-qubit gates.

**Exercise 8.7.4:** For Grover search with $N=256$ items, how many iterations are needed to maximize probability of finding the marked item? What is the success probability after this many iterations?

### Intermediate Problems (★★)

**Exercise 8.7.5:** Prove that the QFT can be written as tensor product form:
$$
\text{QFT}_N |j_1 \ldots j_n\rangle = \frac{1}{\sqrt{2^n}} \bigotimes_{k=1}^n \left( |0\rangle + e^{2\pi i \cdot 0.j_k \ldots j_n} |1\rangle \right)
$$

**Exercise 8.7.6:** In Shor's algorithm, suppose after measuring the first register you obtain $s = 12$ when $2^n = 32$. Use continued fractions to extract possible period $r$ from $s/2^n = 12/32 = 3/8$.

**Exercise 8.7.7:** Show that Grover's operator $G = U_s U_\omega$ rotates the state by angle $\theta$ in the 2D subspace spanned by $|\omega\rangle$ and the uniform superposition over non-target states. Calculate $\theta$ for $N=2^{10}$.

**Exercise 8.7.8:** For a 2-qubit VQE circuit with ansatz $U(\theta_1, \theta_2) = e^{-i\theta_2 X_1 X_2} e^{-i\theta_1 Y_1}$, compute the gradient $\frac{\partial}{\partial \theta_1} \langle 0 | U^\dagger(\boldsymbol{\theta}) H U(\boldsymbol{\theta}) |0\rangle$ for $H = Z_1 + Z_2$.

### Advanced Problems (★★★)

**Exercise 8.7.9:** Prove that any quantum algorithm for unstructured search (oracle-based decision problem) requires $\Omega(\sqrt{N})$ queries, showing Grover is optimal. (Hint: Use Ambainis' adversary method or polynomial method.)

**Exercise 8.7.10:** In QAOA with $p=1$ for MaxCut on a triangle graph $K_3$, find optimal parameters $(\beta^*, \gamma^*)$ that maximize $\langle H_C \rangle$ and compute the expected approximation ratio.

**Exercise 8.7.11:** Show that if $\text{NP} \subseteq \text{BQP}$, then problems in NP-complete (like SAT) could be solved in quantum polynomial time. Discuss why most researchers believe this is unlikely and what Grover's algorithm says about this question.

**Exercise 8.7.12:** Design a quantum circuit implementing period finding for $f(x) = 7^x \bmod 15$ with $n=8$ qubits. Trace through the algorithm and show how measuring outcome $s$ relates to the period $r=4$ using continued fractions.

---

## Related Topics

- **8.1 Quantum Mechanics Foundations:** Postulates underlying algorithm design
- **8.5 Quantum Gates and Circuits:** Building blocks for implementing algorithms
- **8.8 Quantum Error Correction:** Enabling fault-tolerant versions of Shor/Grover
- **8.9 Quantum Machine Learning:** Applications of VQE/QAOA to ML tasks
- **Computational Complexity Theory:** Deeper study of BQP and separation results
- **Adiabatic Quantum Computing:** Alternative model related to QAOA at $p \to \infty$
- **Topological Quantum Computing:** Fault-tolerant approach using anyons
- **Post-Quantum Cryptography:** Cryptosystems secure against Shor's algorithm

---

## Further Reading

1. Nielsen & Chuang, *Quantum Computation and Quantum Information* (2010) - Comprehensive textbook covering all algorithms in detail
2. Shor, "Polynomial-Time Algorithms for Prime Factorization and Discrete Logarithms on a Quantum Computer" (1997) - Original paper
3. Grover, "A fast quantum mechanical algorithm for database search" (1996) - Original Grover paper
4. Farhi et al., "A Quantum Approximate Optimization Algorithm" (2014) - Original QAOA paper
5. Peruzzo et al., "A variational eigenvalue solver on a photonic quantum processor" (2014) - First VQE implementation
6. Aaronson, "BQP and the Polynomial Hierarchy" (2010) - Complexity-theoretic perspective
7. Biamonte et al., "Quantum Machine Learning" (2017) - Survey connecting quantum algorithms to ML

**Online Resources:**
- Qiskit Textbook: https://qiskit.org/textbook - Interactive quantum algorithm tutorials
- Quantum Algorithm Zoo: http://quantumalgorithmzoo.org - Comprehensive list of quantum algorithms
- Complexity Zoo: https://complexityzoo.net - Database of complexity classes including BQP

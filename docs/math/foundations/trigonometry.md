# Trigonometry

Transformers encode position using sine and cosine. Quantum gates rotate qubits through angles. Neural networks activate with $\tanh$. Signal processing decomposes waveforms into sinusoids. These applications share a foundation: trigonometry, the mathematics of periodic phenomena. Beyond triangles, trigonometric functions are the natural language of rotation, oscillation, and wave-like behavior—ubiquitous in ML systems and quantum computing.

---

## Prerequisites

- [Number Systems](number-systems.md) (real and complex numbers)
- [Functions and Relations](functions-and-relations.md) (function composition, inverse functions)
- Basic geometry (circles, angles)

---

## 1. The Unit Circle and Trigonometric Functions

**Definition 1.6.1 (Unit Circle).** The **unit circle** is the set $\{(x, y) \in \mathbb{R}^2 : x^2 + y^2 = 1\}$.

For any angle $\theta$ (measured counterclockwise from the positive $x$-axis), we associate a unique point on the unit circle:

```
                    y
                    │
                (0,1)
                    │
        II      •   │   •      I
                    │
    (-1,0)•─────────┼─────────•(1,0)  x
                    │
        III     •   │   •      IV
                    │
                (0,-1)

Unit Circle: x² + y² = 1

For angle θ measured from positive x-axis:
    P(θ) = (cos θ, sin θ)
```

**Definition 1.6.2 (Sine and Cosine).** For any angle $\theta \in \mathbb{R}$:
- $\cos(\theta)$ = $x$-coordinate of the point at angle $\theta$ on the unit circle
- $\sin(\theta)$ = $y$-coordinate of the point at angle $\theta$ on the unit circle

**Example 1.1:** Compute $\sin$, $\cos$, and $\tan$ for $\theta = \pi/3$ (i.e., $60°$).

From the unit circle, the point at angle $\pi/3$ is $\left(\frac{1}{2}, \frac{\sqrt{3}}{2}\right)$. Therefore:

$$\cos\left(\frac{\pi}{3}\right) = \frac{1}{2}, \quad \sin\left(\frac{\pi}{3}\right) = \frac{\sqrt{3}}{2}, \quad \tan\left(\frac{\pi}{3}\right) = \frac{\sin(\pi/3)}{\cos(\pi/3)} = \frac{\sqrt{3}/2}{1/2} = \sqrt{3}$$

**Definition 1.6.3 (Other Trigonometric Functions).** From sine and cosine, we derive:

$$\tan(\theta) = \frac{\sin \theta}{\cos \theta} \quad (\cos \theta \neq 0)$$

$$\cot(\theta) = \frac{\cos \theta}{\sin \theta} \quad (\sin \theta \neq 0)$$

$$\sec(\theta) = \frac{1}{\cos \theta} \quad (\cos \theta \neq 0)$$

$$\csc(\theta) = \frac{1}{\sin \theta} \quad (\sin \theta \neq 0)$$

**Theorem 1.6.1 (Fundamental Properties).** For all $\theta \in \mathbb{R}$:

1. **Boundedness**: $-1 \leq \sin \theta \leq 1$ and $-1 \leq \cos \theta \leq 1$
2. **Periodicity**: $\sin(\theta + 2\pi) = \sin \theta$ and $\cos(\theta + 2\pi) = \cos \theta$
3. **Pythagorean Identity**: $\sin^2 \theta + \cos^2 \theta = 1$
4. **Symmetry**: $\cos(-\theta) = \cos \theta$ (even), $\sin(-\theta) = -\sin \theta$ (odd)

*Proof*. Properties (1) and (3) follow directly from the unit circle definition. Property (2) reflects that a full rotation of $2\pi$ radians returns to the same point. Property (4) follows from reflection symmetry about the $x$-axis. $\square$

**Example 1.2:** Verify the Pythagorean identity for $\theta = \pi/4$ (i.e., $45°$).

$$\sin^2\left(\frac{\pi}{4}\right) + \cos^2\left(\frac{\pi}{4}\right) = \left(\frac{\sqrt{2}}{2}\right)^2 + \left(\frac{\sqrt{2}}{2}\right)^2 = \frac{2}{4} + \frac{2}{4} = \frac{1}{2} + \frac{1}{2} = 1 \; \checkmark$$

Also verify the even/odd symmetry properties: $\cos(-\pi/4) = \cos(\pi/4) = \frac{\sqrt{2}}{2}$ and $\sin(-\pi/4) = -\sin(\pi/4) = -\frac{\sqrt{2}}{2}$.

### Key Angle Values

| $\theta$ | $0$ | $\frac{\pi}{6}$ | $\frac{\pi}{4}$ | $\frac{\pi}{3}$ | $\frac{\pi}{2}$ | $\pi$ | $\frac{3\pi}{2}$ | $2\pi$ |
|----------|-----|-----------------|-----------------|-----------------|-----------------|-------|------------------|--------|
| $\sin \theta$ | $0$ | $\frac{1}{2}$ | $\frac{\sqrt{2}}{2}$ | $\frac{\sqrt{3}}{2}$ | $1$ | $0$ | $-1$ | $0$ |
| $\cos \theta$ | $1$ | $\frac{\sqrt{3}}{2}$ | $\frac{\sqrt{2}}{2}$ | $\frac{1}{2}$ | $0$ | $-1$ | $0$ | $1$ |
| $\tan \theta$ | $0$ | $\frac{\sqrt{3}}{3}$ | $1$ | $\sqrt{3}$ | undef | $0$ | undef | $0$ |

---

## 2. Radians: The Natural Angle Measure

**Definition 1.6.4 (Radian Measure).** One **radian** is the angle subtended by an arc of length 1 on the unit circle. Since the circle's circumference is $2\pi$, a full rotation is $2\pi$ radians.

**Conversion**: $\theta_{\text{rad}} = \frac{\pi}{180} \cdot \theta_{\text{deg}}$

**Example 2.1:** Convert between degrees and radians.

- $45° = \frac{\pi}{180} \cdot 45 = \frac{\pi}{4} \approx 0.7854$ radians
- $120° = \frac{\pi}{180} \cdot 120 = \frac{2\pi}{3} \approx 2.0944$ radians
- $\frac{5\pi}{6}$ radians $= \frac{180}{\pi} \cdot \frac{5\pi}{6} = 150°$

**Why radians are essential**: When $\theta$ is in radians, the fundamental limit

$$\lim_{h \to 0} \frac{\sin h}{h} = 1$$

holds exactly. This yields the clean derivatives:

$$\frac{d}{d\theta} \sin \theta = \cos \theta, \quad \frac{d}{d\theta} \cos \theta = -\sin \theta$$

In degrees, these derivatives include a factor of $\pi/180$, making analysis clumsy. **Always use radians for calculus and computation.**

*ML connection*: Neural networks parameterize rotations (e.g., in RoPE embeddings, spatial transformer networks) using radians. Degree-based angles would introduce scaling factors throughout gradient computations.

---

## 3. Fundamental Trigonometric Identities

### 3.1 Pythagorean Identities

**Theorem 1.6.2 (Pythagorean Identities).** For all $\theta$ where the functions are defined:

1. $\sin^2 \theta + \cos^2 \theta = 1$
2. $1 + \tan^2 \theta = \sec^2 \theta$
3. $1 + \cot^2 \theta = \csc^2 \theta$

*Proof*. (1) is immediate from the unit circle definition. For (2), divide (1) by $\cos^2 \theta$:

$$\frac{\sin^2 \theta}{\cos^2 \theta} + 1 = \frac{1}{\cos^2 \theta} \implies \tan^2 \theta + 1 = \sec^2 \theta$$

Identity (3) follows by dividing (1) by $\sin^2 \theta$. $\square$

**Example 3.1:** Verify identity (2) for $\theta = \pi/4$ (i.e., $45°$).

$$1 + \tan^2\left(\frac{\pi}{4}\right) = 1 + 1^2 = 2$$

$$\sec^2\left(\frac{\pi}{4}\right) = \frac{1}{\cos^2(\pi/4)} = \frac{1}{(\sqrt{2}/2)^2} = \frac{1}{1/2} = 2 \; \checkmark$$

### 3.2 Angle Addition Formulas

**Theorem 1.6.3 (Angle Addition).** For all $\alpha, \beta \in \mathbb{R}$:

$$\sin(\alpha + \beta) = \sin \alpha \cos \beta + \cos \alpha \sin \beta$$

$$\cos(\alpha + \beta) = \cos \alpha \cos \beta - \sin \alpha \sin \beta$$

$$\tan(\alpha + \beta) = \frac{\tan \alpha + \tan \beta}{1 - \tan \alpha \tan \beta}$$

*Proof sketch*. These can be proven via rotation matrices (see exercises) or geometric arguments. The tangent formula follows from the sine and cosine formulas. $\square$

**Example 3.2:** Compute $\sin(75°)$ using $\sin(45° + 30°)$.

$$\sin(75°) = \sin 45° \cos 30° + \cos 45° \sin 30° = \frac{\sqrt{2}}{2} \cdot \frac{\sqrt{3}}{2} + \frac{\sqrt{2}}{2} \cdot \frac{1}{2}$$

$$= \frac{\sqrt{6}}{4} + \frac{\sqrt{2}}{4} = \frac{\sqrt{6} + \sqrt{2}}{4} \approx 0.9659$$

**Corollary 1.6.4 (Angle Subtraction).** Replace $\beta$ with $-\beta$ and apply symmetry:

$$\sin(\alpha - \beta) = \sin \alpha \cos \beta - \cos \alpha \sin \beta$$

$$\cos(\alpha - \beta) = \cos \alpha \cos \beta + \sin \alpha \sin \beta$$

*ML connection*: Relative position encoding in RoPE (Rotary Position Embeddings) uses the angle subtraction identity. When queries and keys are rotated by angles $m\theta$ and $n\theta$ respectively, their dot product depends on $(m-n)\theta$ via $\cos(\alpha - \beta)$.

### 3.3 Double and Half Angle Formulas

**Theorem 1.6.5 (Double Angle Formulas).** Setting $\alpha = \beta = \theta$ in angle addition:

$$\sin(2\theta) = 2 \sin \theta \cos \theta$$

$$\cos(2\theta) = \cos^2 \theta - \sin^2 \theta = 2\cos^2 \theta - 1 = 1 - 2\sin^2 \theta$$

$$\tan(2\theta) = \frac{2\tan \theta}{1 - \tan^2 \theta}$$

**Example 3.3:** Compute $\sin(60°)$ and $\cos(60°)$ using the double angle formulas with $\theta = 30°$.

$$\sin(60°) = 2\sin 30° \cos 30° = 2 \cdot \frac{1}{2} \cdot \frac{\sqrt{3}}{2} = \frac{\sqrt{3}}{2} \; \checkmark$$

$$\cos(60°) = \cos^2 30° - \sin^2 30° = \left(\frac{\sqrt{3}}{2}\right)^2 - \left(\frac{1}{2}\right)^2 = \frac{3}{4} - \frac{1}{4} = \frac{1}{2} \; \checkmark$$

**Theorem 1.6.6 (Half Angle Formulas).** Solving the double angle formulas for $\sin(\theta/2)$ and $\cos(\theta/2)$:

$$\sin^2\left(\frac{\theta}{2}\right) = \frac{1 - \cos \theta}{2}$$

$$\cos^2\left(\frac{\theta}{2}\right) = \frac{1 + \cos \theta}{2}$$

The signs of $\sin(\theta/2)$ and $\cos(\theta/2)$ depend on which quadrant $\theta/2$ lies in.

**Example 3.4:** Find $\cos(15°)$ using the half angle formula with $\theta = 30°$.

Since $15° = 30°/2$ lies in the first quadrant, $\cos(15°) > 0$:

$$\cos^2(15°) = \frac{1 + \cos 30°}{2} = \frac{1 + \frac{\sqrt{3}}{2}}{2} = \frac{2 + \sqrt{3}}{4}$$

$$\cos(15°) = \sqrt{\frac{2 + \sqrt{3}}{4}} = \frac{\sqrt{2 + \sqrt{3}}}{2} \approx 0.9659$$

---

## 4. Inverse Trigonometric Functions

Since sine, cosine, and tangent are periodic, they are not one-to-one. To define inverses, we restrict their domains.

**Definition 1.6.5 (Inverse Trigonometric Functions).**

| Function | Domain | Range | Notation |
|----------|--------|-------|----------|
| Inverse sine | $[-1, 1]$ | $\left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$ | $\arcsin x$ or $\sin^{-1} x$ |
| Inverse cosine | $[-1, 1]$ | $[0, \pi]$ | $\arccos x$ or $\cos^{-1} x$ |
| Inverse tangent | $\mathbb{R}$ | $\left(-\frac{\pi}{2}, \frac{\pi}{2}\right)$ | $\arctan x$ or $\tan^{-1} x$ |

**Properties**:
- $\sin(\arcsin x) = x$ for $x \in [-1, 1]$
- $\arcsin(\sin \theta) = \theta$ for $\theta \in [-\pi/2, \pi/2]$
- Similar identities hold for $\arccos$ and $\arctan$

**Example 4.1:** Compute the following inverse trig values.

- $\arcsin\left(\frac{1}{2}\right) = \frac{\pi}{6}$ because $\sin\left(\frac{\pi}{6}\right) = \frac{1}{2}$ and $\frac{\pi}{6} \in [-\frac{\pi}{2}, \frac{\pi}{2}]$
- $\arccos\left(-\frac{\sqrt{2}}{2}\right) = \frac{3\pi}{4}$ because $\cos\left(\frac{3\pi}{4}\right) = -\frac{\sqrt{2}}{2}$ and $\frac{3\pi}{4} \in [0, \pi]$
- $\arctan(1) = \frac{\pi}{4}$ because $\tan\left(\frac{\pi}{4}\right) = 1$ and $\frac{\pi}{4} \in (-\frac{\pi}{2}, \frac{\pi}{2})$
- $\arctan(-\sqrt{3}) = -\frac{\pi}{3}$ because $\tan\left(-\frac{\pi}{3}\right) = -\sqrt{3}$

**Theorem 1.6.7 (Derivatives of Inverse Functions).**

$$\frac{d}{dx} \arcsin x = \frac{1}{\sqrt{1 - x^2}} \quad (|x| < 1)$$

$$\frac{d}{dx} \arccos x = -\frac{1}{\sqrt{1 - x^2}} \quad (|x| < 1)$$

$$\frac{d}{dx} \arctan x = \frac{1}{1 + x^2}$$

*Proof sketch*. Use implicit differentiation on $y = \arcsin x \iff x = \sin y$ with $y \in [-\pi/2, \pi/2]$:

$$\frac{dx}{dy} = \cos y \implies \frac{dy}{dx} = \frac{1}{\cos y} = \frac{1}{\sqrt{1 - \sin^2 y}} = \frac{1}{\sqrt{1 - x^2}}$$

We take the positive square root since $\cos y \geq 0$ for $y \in [-\pi/2, \pi/2]$. $\square$

*ML connection*: The derivative of $\arctan$ appears in numerical optimization. The function $\arctan(x)$ is a smooth, bounded alternative to clipping operations.

---

## 5. Solving Trigonometric Equations

**Example 1.6.1**. Solve $\sin \theta = \frac{1}{2}$ for $\theta \in [0, 2\pi)$.

*Solution*. The sine function equals $\frac{1}{2}$ at:
- $\theta = \frac{\pi}{6}$ (Quadrant I)
- $\theta = \pi - \frac{\pi}{6} = \frac{5\pi}{6}$ (Quadrant II)

**General solution** (for all integers $k$):

$$\theta = \frac{\pi}{6} + 2\pi k \quad \text{or} \quad \theta = \frac{5\pi}{6} + 2\pi k$$

**Example 1.6.2**. Solve $2\cos^2 \theta - \cos \theta - 1 = 0$.

*Solution*. Let $u = \cos \theta$. Then $2u^2 - u - 1 = 0$ factors as $(2u + 1)(u - 1) = 0$.

Solutions: $u = -\frac{1}{2}$ or $u = 1$

- $\cos \theta = 1 \implies \theta = 2\pi k$
- $\cos \theta = -\frac{1}{2} \implies \theta = \frac{2\pi}{3} + 2\pi k$ or $\theta = \frac{4\pi}{3} + 2\pi k$

**Example 5.1 (Law of Cosines):** In triangle $ABC$, sides $a = 7$, $b = 5$, and the included angle $C = 60°$. Find side $c$.

The **Law of Cosines** states: $c^2 = a^2 + b^2 - 2ab\cos C$.

$$c^2 = 7^2 + 5^2 - 2(7)(5)\cos 60° = 49 + 25 - 70 \cdot \frac{1}{2} = 74 - 35 = 39$$

$$c = \sqrt{39} \approx 6.245$$

---

## 6. Hyperbolic Functions

**Definition 1.6.6 (Hyperbolic Sine and Cosine).** The **hyperbolic functions** are defined via exponentials:

$$\sinh x = \frac{e^x - e^{-x}}{2}, \quad \cosh x = \frac{e^x + e^{-x}}{2}$$

$$\tanh x = \frac{\sinh x}{\cosh x} = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{e^{2x} - 1}{e^{2x} + 1}$$

**Example 6.1:** Compute the hyperbolic functions at $x = 1$ (using $e \approx 2.718$).

$$\sinh(1) = \frac{e^1 - e^{-1}}{2} = \frac{2.718 - 0.368}{2} \approx 1.175$$

$$\cosh(1) = \frac{e^1 + e^{-1}}{2} = \frac{2.718 + 0.368}{2} \approx 1.543$$

$$\tanh(1) = \frac{\sinh(1)}{\cosh(1)} \approx \frac{1.175}{1.543} \approx 0.762$$

Note that $\tanh(1)$ is already close to its asymptote of $1$, illustrating its sigmoid-like saturation.

**Geometric interpretation**: Just as $(\cos \theta, \sin \theta)$ parameterizes the unit circle $x^2 + y^2 = 1$, the point $(\cosh t, \sinh t)$ parameterizes the right branch of the unit hyperbola $x^2 - y^2 = 1$.

```
Unit Hyperbola: x² - y² = 1

         y
         │     ╱│
         │   ╱  │
         │ ╱    │  Right branch: (cosh t, sinh t)
    ─────┼──────┼─────── x
         │╲     │
         │  ╲   │
         │    ╲ │
```

### Properties

**Theorem 1.6.8 (Hyperbolic Identities).**

1. **Hyperbolic Pythagorean**: $\cosh^2 x - \sinh^2 x = 1$
2. **Symmetry**: $\cosh(-x) = \cosh x$ (even), $\sinh(-x) = -\sinh x$ (odd)
3. **Addition**: $\sinh(x \pm y) = \sinh x \cosh y \pm \cosh x \sinh y$
4. **Addition**: $\cosh(x \pm y) = \cosh x \cosh y \pm \sinh x \sinh y$

**Example 6.2:** Verify the hyperbolic Pythagorean identity $\cosh^2 x - \sinh^2 x = 1$ for $x = 1$.

Using the values from Example 6.1:

$$\cosh^2(1) - \sinh^2(1) \approx (1.543)^2 - (1.175)^2 = 2.381 - 1.381 = 1.000 \; \checkmark$$

Alternatively, verify exactly: $\cosh^2(1) - \sinh^2(1) = \left(\frac{e + e^{-1}}{2}\right)^2 - \left(\frac{e - e^{-1}}{2}\right)^2 = \frac{(e + e^{-1})^2 - (e - e^{-1})^2}{4} = \frac{4e \cdot e^{-1}}{4} = 1$.

**Theorem 1.6.9 (Derivatives).**

$$\frac{d}{dx} \sinh x = \cosh x, \quad \frac{d}{dx} \cosh x = \sinh x$$

$$\frac{d}{dx} \tanh x = \operatorname{sech}^2 x = \frac{1}{\cosh^2 x} = 1 - \tanh^2 x$$

These mirror the derivatives of trigonometric functions (with sign differences).

### Connection to Complex Numbers

**Theorem 1.6.10 (Euler's Formula).**

$$e^{i\theta} = \cos \theta + i \sin \theta$$

This implies:

$$\cos \theta = \frac{e^{i\theta} + e^{-i\theta}}{2}, \quad \sin \theta = \frac{e^{i\theta} - e^{-i\theta}}{2i}$$

Comparing with hyperbolic definitions:

$$\cosh(ix) = \cos x, \quad \sinh(ix) = i \sin x$$

Thus trigonometric and hyperbolic functions are related via the complex plane: $\sin x = -i \sinh(ix)$ and $\cos x = \cosh(ix)$.

*Quantum connection*: Euler's formula is the foundation of quantum gate decomposition. Any single-qubit unitary can be written as $e^{i\theta \hat{n} \cdot \vec{\sigma}}$ where $\vec{\sigma}$ are Pauli matrices.

---

## 7. Preview: Fourier Analysis

One of the deepest results in mathematics: **any periodic function can be represented as a sum of sines and cosines**.

**Theorem 1.6.11 (Fourier Series - Informal).** A periodic function $f(x)$ with period $2\pi$ can be represented as:

$$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left( a_n \cos(nx) + b_n \sin(nx) \right)$$

where the **Fourier coefficients** are determined by:

$$a_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \cos(nx) \, dx, \quad b_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \sin(nx) \, dx$$

**Why this matters**:
- **Signal processing**: Decompose audio/images into frequency components
- **PDEs**: Solve heat equation, wave equation via separation of variables
- **Neural networks**: Fourier features encode continuous inputs for coordinate-based MLPs
- **Transformers**: Positional encodings inject frequency information into token embeddings

We will explore this rigorously in later chapters on calculus and real analysis.

---

## 8. Connections to Machine Learning

### 8.1 The tanh Activation Function

**The hyperbolic tangent** is a classic activation function:

$$\tanh(x) = \frac{e^{2x} - 1}{e^{2x} + 1}$$

**Properties**:
- Range: $(-1, 1)$ (zero-centered, unlike sigmoid's $(0,1)$)
- Derivative: $\tanh'(x) = 1 - \tanh^2(x)$ (efficient backpropagation)
- Smooth and differentiable everywhere
- Saturates for large $|x|$ (vanishing gradient problem)

Modern architectures prefer ReLU for speed, but $\tanh$ remains crucial in LSTMs (forget/input/output gates) and GRUs.

### 8.2 Positional Encoding in Transformers

Transformers (Vaswani et al., 2017) lack recurrence, so they cannot inherently encode token position. **Positional encodings** inject position information:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

where $pos$ is the token position, $i$ is the dimension index, and $d$ is the embedding dimension.

**Why sine/cosine?**
- Periodic: similar encodings for nearby positions
- Multiple frequencies: capture local (high-frequency) and global (low-frequency) patterns
- Extrapolation: model can handle sequences longer than training length
- Fixed (not learned): saves parameters, provides inductive bias

### 8.3 Rotary Position Embeddings (RoPE)

Modern LLMs (GPT-NeoX, LLaMA, Mistral) use **Rotary Position Embeddings**:

Instead of adding positional encodings, RoPE applies rotation to query/key vectors:

$$\begin{pmatrix} q_1' \\ q_2' \end{pmatrix} = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} q_1 \\ q_2 \end{pmatrix}$$

where $m$ is the token position and $\theta$ is a frequency parameter.

**Key insight**: The attention score between tokens at positions $m$ and $n$ depends on their relative distance $m - n$:

$$q_m^\top k_n = q^\top R(\theta(m-n)) k$$

using the angle subtraction identity. This naturally encodes relative positions.

### 8.4 Fourier Features for Neural Networks

**Problem**: Standard MLPs struggle to learn high-frequency functions from low-dimensional inputs (e.g., $(x, y)$ pixel coordinates).

**Solution** (Tancik et al., 2020): Map inputs through **Fourier features**:

$$\gamma(x) = \left[ \sin(2\pi \mathbf{b}_1 \cdot x), \cos(2\pi \mathbf{b}_1 \cdot x), \ldots, \sin(2\pi \mathbf{b}_m \cdot x), \cos(2\pi \mathbf{b}_m \cdot x) \right]$$

where $\mathbf{b}_i$ are random frequency vectors sampled from a Gaussian. This transforms the input into a higher-dimensional space where MLPs can fit fine details. Used in NeRF (Neural Radiance Fields) for photorealistic 3D scene rendering.

---

## 9. Connections to Quantum Computing

### 9.1 Rotation Gates

Single-qubit rotations are parameterized by angles $\theta$ (in radians):

$$R_x(\theta) = \begin{pmatrix} \cos(\theta/2) & -i\sin(\theta/2) \\ -i\sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$$

$$R_y(\theta) = \begin{pmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$$

$$R_z(\theta) = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$$

These rotate the qubit state on the **Bloch sphere**:

```
         |0⟩
          │
          │   ╱ Qubit state
          │ ╱θ
          │╱
    ──────●────── x
        ╱ │
      ╱   │
    ╱     │
          |1⟩
```

The Bloch sphere is a geometric representation where pure single-qubit states correspond to points on the unit sphere.

### 9.2 Quantum Fourier Transform

The **Quantum Fourier Transform** (QFT) maps computational basis states via roots of unity:

$$|j\rangle \mapsto \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} e^{2\pi ijk/N} |k\rangle$$

where $e^{2\pi i/N} = \cos(2\pi/N) + i\sin(2\pi/N)$ is a primitive $N$-th root of unity.

**Applications**:
- **Shor's algorithm**: Factoring integers in polynomial time (breaks RSA)
- **Phase estimation**: Estimates eigenvalues of unitary operators
- **HHL algorithm**: Solves linear systems exponentially faster (under conditions)

The QFT is exponentially faster than classical FFT in terms of gate complexity, though measurement collapses the quantum state, limiting direct readout.

---

## Exercises

### Basic (★)

**1.6.1.** Verify the Pythagorean identity $\sin^2 \theta + \cos^2 \theta = 1$ for $\theta = \pi/3$ using exact values.

**1.6.2.** Convert to radians: (a) $30°$, (b) $135°$, (c) $270°$.

**1.6.3.** Evaluate without a calculator:
- (a) $\sin(7\pi/6)$
- (b) $\cos(5\pi/4)$
- (c) $\tan(2\pi/3)$

**1.6.4.** Solve for $\theta \in [0, 2\pi)$: $\cos \theta = -\frac{\sqrt{3}}{2}$.

**1.6.5.** Compute $\sinh(0)$, $\cosh(0)$, and $\tanh(0)$.

### Intermediate (★★)

**1.6.6.** Prove the angle subtraction formula for sine using the addition formula and symmetry properties.

**1.6.7.** Derive $\cos(2\theta) = 1 - 2\sin^2 \theta$ starting from $\cos(2\theta) = \cos^2 \theta - \sin^2 \theta$.

**1.6.8.** Solve $2\sin^2 \theta + 3\sin \theta + 1 = 0$ for $\theta \in [0, 2\pi)$.

**1.6.9.** Show that $\tanh'(x) = 1 - \tanh^2(x)$ using the definition of $\tanh$.

**1.6.10.** Prove $\cosh^2 x - \sinh^2 x = 1$ directly from the exponential definitions.

**1.6.11.** Compute $\frac{d}{dx} \arctan(3x)$.

**1.6.12.** Show that $\lim_{x \to \infty} \tanh(x) = 1$ and interpret this as a sigmoid-like behavior.

### Challenging (★★★)

**1.6.13.** **(Euler's Formula)** Prove the angle addition formula for cosine using $e^{i(\alpha + \beta)} = e^{i\alpha} e^{i\beta}$ and Euler's formula.

**1.6.14.** **(Rotation Matrices)** Show that the $2 \times 2$ rotation matrix

$$R(\theta) = \begin{pmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{pmatrix}$$

satisfies $R(\alpha) R(\beta) = R(\alpha + \beta)$, corresponding to the angle addition formulas.

**1.6.15.** **(RoPE Insight)** In RoPE, prove that if vectors $\mathbf{q}$ and $\mathbf{k}$ are rotated by angles $m\theta$ and $n\theta$ respectively (for positions $m$ and $n$), their dot product depends only on the relative position $(m - n)\theta$.

**1.6.16.** **(Fourier Orthogonality)** Let $f(x) = \sin(2\pi k x)$ for integer frequency $k$. Show that:

$$\int_0^1 f(x) \, dx = 0 \quad \text{and} \quad \int_0^1 f(x)^2 \, dx = \frac{1}{2}$$

These properties underlie Fourier series decomposition.

**1.6.17.** **(Chebyshev Polynomials)** The Chebyshev polynomials satisfy $T_n(\cos \theta) = \cos(n\theta)$. Derive the recurrence relation $T_{n+1}(x) = 2xT_n(x) - T_{n-1}(x)$ using the angle addition formula.

**1.6.18.** **(Quantum Gates)** Show that $R_z(\theta) R_y(\phi) R_z(\psi)$ can represent any single-qubit unitary (up to a global phase). This is the $ZYZ$ decomposition used in quantum compilation.

---

## Summary

Trigonometry extends from triangle measurement to the foundation of periodic phenomena:

1. **Trigonometric functions** arise from the unit circle and exhibit periodicity, symmetry, and rich algebraic structure
2. **Radians** are the natural angle measure for calculus, yielding derivatives without scaling factors
3. **Identities** (Pythagorean, angle addition, double/half angle) enable algebraic manipulation and equation solving
4. **Inverse functions** require domain restriction and provide important antiderivatives
5. **Hyperbolic functions** mirror trigonometric functions via exponentials and connect through complex numbers
6. **Fourier analysis** decomposes functions into sinusoidal components—fundamental to signal processing and PDEs

**ML/AI Applications**: $\tanh$ activation, positional encodings (sinusoidal and RoPE), Fourier features for coordinate-based networks

**Quantum Computing**: Rotation gates on the Bloch sphere, QFT for phase estimation and Shor's algorithm

---

## Related Topics

- [Complex Numbers](complex-numbers.md) — Euler's formula, polar representation, roots of unity
- [Differential Calculus](../calculus/differentiation.md) — derivatives of trig functions, chain rule
- [Integration Techniques](../calculus/integration.md) — trig substitution, integrals of trig functions
- [Linear Algebra: Rotations](../linear-algebra/linear-transformations.md) — rotation matrices in higher dimensions
- [Quantum Computing](../quantum/quantum-gates.md) — single-qubit and multi-qubit gates

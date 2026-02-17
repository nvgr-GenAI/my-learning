# Chapter 1.5: Complex Numbers

## Prerequisites

This chapter assumes familiarity with:
- **Sets & Logic** (Chapter 1.1): Set notation, logical operations
- **Number Systems** (Chapter 1.3): Construction of $\mathbb{R}$, field axioms, algebraic vs transcendental numbers
- **Basic Trigonometry**: sine, cosine, their properties
- **Elementary Calculus**: Taylor series, exponential function

## Introduction

Complex numbers emerged from the need to solve polynomial equations like $x^2 + 1 = 0$, which have no real solutions. While initially viewed as "imaginary" and suspect, they are now fundamental to mathematics, physics, and engineering. In machine learning, complex numbers appear in Fourier analysis, signal processing, quantum computing, and modern attention mechanisms.

The complex numbers $\mathbb{C}$ form an algebraically closed field: every non-constant polynomial has roots in $\mathbb{C}$. This closure property makes complex analysis remarkably elegant compared to real analysis.

---

## 1.5.1 Complex Number Arithmetic

**Definition 1.5.1** (Complex Number)
A **complex number** is an ordered pair $(a, b) \in \mathbb{R}^2$, written as $z = a + bi$, where:
- $a = \text{Re}(z)$ is the **real part**
- $b = \text{Im}(z)$ is the **imaginary part**
- $i$ is the **imaginary unit** satisfying $i^2 = -1$

The set of all complex numbers is denoted $\mathbb{C} = \{a + bi : a, b \in \mathbb{R}\}$.

**Definition 1.5.2** (Complex Arithmetic)
For $z_1 = a + bi$ and $z_2 = c + di$:

1. **Addition**: $z_1 + z_2 = (a + c) + (b + d)i$
2. **Multiplication**: $z_1 \cdot z_2 = (ac - bd) + (ad + bc)i$
3. **Complex Conjugate**: $\overline{z_1} = a - bi$
4. **Modulus**: $|z_1| = \sqrt{a^2 + b^2}$

**Example 1.5.1:** Let $z_1 = 3 + 4i$ and $z_2 = 1 - 2i$.

$$z_1 + z_2 = (3 + 1) + (4 + (-2))i = 4 + 2i$$

$$z_1 \cdot z_2 = (3)(1) - (4)(-2) + ((3)(-2) + (4)(1))i = 3 + 8 + (-6 + 4)i = 11 - 2i$$

$$\overline{z_1} = 3 - 4i, \quad |z_1| = \sqrt{3^2 + 4^2} = \sqrt{25} = 5$$

**Theorem 1.5.1** (Field Structure)
$(\mathbb{C}, +, \cdot)$ is a field with additive identity $0 = 0 + 0i$ and multiplicative identity $1 = 1 + 0i$.

*Proof sketch:* Associativity, commutativity, and distributivity follow from $\mathbb{R}$. The multiplicative inverse of $z = a + bi \neq 0$ is:

$$z^{-1} = \frac{\overline{z}}{|z|^2} = \frac{a - bi}{a^2 + b^2}$$

Verification: $z \cdot z^{-1} = (a + bi) \cdot \frac{a - bi}{a^2 + b^2} = \frac{a^2 + b^2}{a^2 + b^2} = 1$. $\square$

**Example 1.5.2:** Compute $\frac{3 + 4i}{1 - 2i}$.

$$\frac{3 + 4i}{1 - 2i} = \frac{(3 + 4i)(1 + 2i)}{(1 - 2i)(1 + 2i)} = \frac{(3 - 8) + (6 + 4)i}{1 + 4} = \frac{-5 + 10i}{5} = -1 + 2i$$

We multiplied numerator and denominator by the conjugate $\overline{1 - 2i} = 1 + 2i$.

### The Argand Diagram

Complex numbers have a natural geometric interpretation as points in the plane:

```
      Imaginary axis (Im)
            |
        3i  +     * z = 2 + 3i
            |    /
        2i  +   /
            |  / |z| = √13
         i  + /
            |/ θ = arctan(3/2)
    --------+--------+----+----+---- Real axis (Re)
           0|        1    2    3
        -i  +
            |
       -2i  +
```

**Key Properties:**
- Addition: Parallelogram law (vector addition)
- Multiplication: Rotates and scales (see polar form)
- Conjugation: Reflection across real axis
- Modulus: Euclidean distance from origin

**Theorem 1.5.2** (Properties of Conjugate and Modulus)
For $z, w \in \mathbb{C}$:

1. $\overline{z + w} = \overline{z} + \overline{w}$
2. $\overline{z \cdot w} = \overline{z} \cdot \overline{w}$
3. $z \cdot \overline{z} = |z|^2$
4. $|z \cdot w| = |z| \cdot |w|$ (multiplicativity)
5. $|z + w| \leq |z| + |w|$ (triangle inequality)

**Example 1.5.3:** Verify properties (1), (3), and (4) for $z = 3 + 4i$ and $w = 1 - 2i$.

*Property (1)*: $\overline{z + w} = \overline{4 + 2i} = 4 - 2i$, and $\overline{z} + \overline{w} = (3 - 4i) + (1 + 2i) = 4 - 2i$. $\checkmark$

*Property (3)*: $z \cdot \overline{z} = (3 + 4i)(3 - 4i) = 9 + 16 = 25 = |z|^2 = 5^2$. $\checkmark$

*Property (4)*: $z \cdot w = 11 - 2i$, so $|zw| = \sqrt{121 + 4} = \sqrt{125} = 5\sqrt{5}$, and $|z| \cdot |w| = 5 \cdot \sqrt{5} = 5\sqrt{5}$. $\checkmark$

---

## 1.5.2 Polar Form and Euler's Formula

**Definition 1.5.3** (Polar Form)
Any complex number $z = a + bi$ can be written in **polar form**:

$$z = r(\cos\theta + i\sin\theta) = r \, e^{i\theta}$$

where:
- $r = |z| = \sqrt{a^2 + b^2}$ is the **modulus**
- $\theta = \arg(z) = \arctan(b/a)$ is the **argument** (angle from positive real axis)
- $\theta$ is defined modulo $2\pi$; the principal value is $\theta \in (-\pi, \pi]$

**Conversion formulas:**
- Cartesian to Polar: $r = \sqrt{a^2 + b^2}$, $\theta = \arctan_2(b, a)$
- Polar to Cartesian: $a = r\cos\theta$, $b = r\sin\theta$

**Example 1.5.4:** Find the modulus and argument of $z = -1 + \sqrt{3}\,i$.

$$r = |z| = \sqrt{(-1)^2 + (\sqrt{3})^2} = \sqrt{1 + 3} = 2$$

Since the point $(-1, \sqrt{3})$ lies in the second quadrant:

$$\theta = \arg(z) = \pi - \arctan\!\left(\frac{\sqrt{3}}{1}\right) = \pi - \frac{\pi}{3} = \frac{2\pi}{3}$$

So $z = 2\,e^{i2\pi/3}$.

**Example 1.5.5:** Convert $z = 3\,e^{i\pi/6}$ to rectangular form.

$$a = 3\cos\frac{\pi}{6} = 3 \cdot \frac{\sqrt{3}}{2} = \frac{3\sqrt{3}}{2}, \quad b = 3\sin\frac{\pi}{6} = 3 \cdot \frac{1}{2} = \frac{3}{2}$$

So $z = \frac{3\sqrt{3}}{2} + \frac{3}{2}\,i$.

### Euler's Formula

**Theorem 1.5.3** (Euler's Formula)
For any $\theta \in \mathbb{R}$:

$$e^{i\theta} = \cos\theta + i\sin\theta$$

*Proof:* Using Taylor series for $e^x$, $\cos x$, and $\sin x$:

$$e^{i\theta} = \sum_{n=0}^{\infty} \frac{(i\theta)^n}{n!} = 1 + i\theta + \frac{(i\theta)^2}{2!} + \frac{(i\theta)^3}{3!} + \frac{(i\theta)^4}{4!} + \cdots$$

Noting $i^2 = -1$, $i^3 = -i$, $i^4 = 1$, separate real and imaginary parts:

$$= \left(1 - \frac{\theta^2}{2!} + \frac{\theta^4}{4!} - \cdots\right) + i\left(\theta - \frac{\theta^3}{3!} + \frac{\theta^5}{5!} - \cdots\right)$$

$$= \cos\theta + i\sin\theta \quad \square$$

**Corollary 1.5.1** (Euler's Identity)
Setting $\theta = \pi$:

$$e^{i\pi} + 1 = 0$$

This relates the five fundamental constants: $e$, $i$, $\pi$, $1$, and $0$.

**Example 1.5.6:** Verify Euler's formula for $\theta = \pi/4$.

$$e^{i\pi/4} = \cos\frac{\pi}{4} + i\sin\frac{\pi}{4} = \frac{\sqrt{2}}{2} + \frac{\sqrt{2}}{2}\,i$$

Check the modulus: $\left|e^{i\pi/4}\right| = \sqrt{\left(\frac{\sqrt{2}}{2}\right)^2 + \left(\frac{\sqrt{2}}{2}\right)^2} = \sqrt{\frac{1}{2} + \frac{1}{2}} = 1$. $\checkmark$

This confirms $e^{i\pi/4}$ lies on the unit circle at angle $45°$.

**Geometric Interpretation:**
$e^{i\theta}$ represents a point on the unit circle at angle $\theta$. As $\theta$ increases, $e^{i\theta}$ traces the circle counterclockwise:

```
         Im
          |
      1   + e^(iπ/2) = i
          |  *
          | /|\
          |/ | \
    ------+--+--*---- Re
     -1   0  |   1
    e^(iπ)   |   e^(i0) = 1
          *  |
       e^(i3π/2) = -i
```

**Theorem 1.5.4** (Multiplication in Polar Form)
If $z_1 = r_1 e^{i\theta_1}$ and $z_2 = r_2 e^{i\theta_2}$, then:

$$z_1 \cdot z_2 = r_1 r_2 \, e^{i(\theta_1 + \theta_2)}$$

*Geometric meaning:* Multiply moduli, add arguments. Multiplication rotates and scales.

**Example 1.5.7:** Compute $z_1 \cdot z_2$ where $z_1 = 2\,e^{i\pi/6}$ and $z_2 = 3\,e^{i\pi/3}$.

$$z_1 \cdot z_2 = (2)(3)\,e^{i(\pi/6 + \pi/3)} = 6\,e^{i\pi/2} = 6(\cos\tfrac{\pi}{2} + i\sin\tfrac{\pi}{2}) = 6i$$

The moduli multiply ($2 \times 3 = 6$) and the arguments add ($30° + 60° = 90°$).

> **ML/AI Connection: Rotary Position Embeddings**
> Modern transformers (RoFormer, LLaMA) use complex exponentials for position encoding:
> $$\text{RoPE}(x, m) = x \cdot e^{im\theta}$$
> where $m$ is the position index. This encodes relative positions through angle differences, preserving rotational properties under attention operations.

---

## 1.5.3 Roots of Unity

**Definition 1.5.4** ($n$-th Root of Unity)
An **$n$-th root of unity** is a complex number $\omega$ satisfying $\omega^n = 1$.

**Theorem 1.5.5** (Roots of Unity)
The $n$-th roots of unity are:

$$\omega_k = e^{2\pi i k/n} = \cos\frac{2\pi k}{n} + i\sin\frac{2\pi k}{n}, \quad k = 0, 1, \ldots, n-1$$

There are exactly $n$ distinct $n$-th roots, evenly spaced around the unit circle.

**Example 1.5.8:** Find all cube roots of unity ($n = 3$).

$$\omega_k = e^{2\pi i k/3}, \quad k = 0, 1, 2$$

- $\omega_0 = e^{0} = 1$
- $\omega_1 = e^{2\pi i/3} = \cos\frac{2\pi}{3} + i\sin\frac{2\pi}{3} = -\frac{1}{2} + \frac{\sqrt{3}}{2}\,i$
- $\omega_2 = e^{4\pi i/3} = \cos\frac{4\pi}{3} + i\sin\frac{4\pi}{3} = -\frac{1}{2} - \frac{\sqrt{3}}{2}\,i$

Verify: $\omega_1^3 = e^{2\pi i} = 1$. $\checkmark$ Also note $\omega_0 + \omega_1 + \omega_2 = 1 + (-\tfrac{1}{2} + \tfrac{\sqrt{3}}{2}i) + (-\tfrac{1}{2} - \tfrac{\sqrt{3}}{2}i) = 0$. $\checkmark$

*Proof:* If $z^n = 1$, write $z = re^{i\theta}$. Then $r^n e^{in\theta} = 1 = e^{i \cdot 0}$, so $r^n = 1$ and $n\theta = 2\pi k$ for integer $k$. Since $r > 0$, we have $r = 1$ and $\theta = 2\pi k/n$. This gives $n$ distinct values for $k = 0, \ldots, n-1$. $\square$

**Example:** The 8th roots of unity:

```
        Im
         |
      ω₂ * ω₃
         |\ /|
         | * |     * = unit circle
    ω₁ * | | | * ω₄
         | * |
    -----+---+----- Re
         | * |
    ω₇ * | | | * ω₅
         |/ \|
      ω₆ * ω₅
```

**Definition 1.5.5** (Primitive Root of Unity)
$\omega_n = e^{2\pi i/n}$ is the **primitive $n$-th root of unity**. All other roots are powers: $\omega_n^k$ for $k = 0, \ldots, n-1$.

**Theorem 1.5.6** (Sum of Roots of Unity)
For $n \geq 2$:

$$\sum_{k=0}^{n-1} \omega_n^k = 0$$

*Proof:* Let $S = \sum_{k=0}^{n-1} \omega_n^k$. Then $\omega_n S = \sum_{k=0}^{n-1} \omega_n^{k+1} = \sum_{k=1}^{n} \omega_n^k = S$ (using $\omega_n^n = 1$). Thus $(\omega_n - 1)S = 0$. Since $\omega_n \neq 1$, we have $S = 0$. $\square$

> **ML/AI Connection: Discrete Fourier Transform**
> The DFT decomposes a signal into frequency components using roots of unity:
> $$X_k = \sum_{n=0}^{N-1} x_n \cdot e^{-2\pi i kn/N} = \sum_{n=0}^{N-1} x_n \cdot \omega_N^{-kn}$$
> The geometric distribution of roots on the unit circle provides orthogonal basis functions for frequency analysis. Fast Fourier Transform (FFT) algorithms exploit this structure for $O(N \log N)$ computation.

---

## 1.5.4 Complex Exponentials

**Definition 1.5.6** (Complex Exponential)
For $z = a + bi \in \mathbb{C}$:

$$e^z = e^{a+bi} = e^a \cdot e^{bi} = e^a(\cos b + i\sin b)$$

**Example 1.5.9:** Compute $e^z$ for $z = 2 + i\pi/3$.

$$e^{2 + i\pi/3} = e^2 \cdot e^{i\pi/3} = e^2\!\left(\cos\frac{\pi}{3} + i\sin\frac{\pi}{3}\right) = e^2\!\left(\frac{1}{2} + \frac{\sqrt{3}}{2}\,i\right) \approx 3.695 + 6.398\,i$$

Modulus: $|e^z| = e^{\text{Re}(z)} = e^2 \approx 7.389$. Argument: $\arg(e^z) = \text{Im}(z) = \pi/3$.

**Key Properties:**
1. $|e^{i\theta}| = 1$ for real $\theta$ (unit circle)
2. $|e^z| = e^{\text{Re}(z)}$ (modulus depends only on real part)
3. $\arg(e^z) = \text{Im}(z) \pmod{2\pi}$
4. $e^{z+w} = e^z \cdot e^w$ (functional equation)
5. $e^z$ is $2\pi i$-periodic: $e^{z + 2\pi i} = e^z$

### Oscillations and Damping

Complex exponentials elegantly describe damped oscillations:

$$z(t) = A e^{(\sigma + i\omega)t} = A e^{\sigma t} e^{i\omega t} = A e^{\sigma t}(\cos\omega t + i\sin\omega t)$$

where:
- $\sigma < 0$: damping factor (exponential decay)
- $\omega$: angular frequency (oscillation rate)
- $A$: amplitude

**Physical Interpretation:**
- $\text{Re}(z(t)) = A e^{\sigma t} \cos\omega t$: damped cosine wave
- $|\text{Im}(z(t))| = |A e^{\sigma t} \sin\omega t|$: damped sine wave

```
Real Part: Damped Oscillation
    |
  A +     ___
    |    /   \___
    |___/        \___
    +---+---+---+---+---+---> t
    |           envelope = e^(σt)
    |
```

> **Quantum Connection: Probability Amplitudes**
> In quantum mechanics, state amplitudes are complex:
> $$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle, \quad \alpha, \beta \in \mathbb{C}$$
> The normalization condition $|\alpha|^2 + |\beta|^2 = 1$ ensures probabilities sum to 1. Time evolution uses complex exponentials:
> $$|\psi(t)\rangle = e^{-iHt/\hbar}|\psi(0)\rangle$$
> where $H$ is the Hamiltonian. Phase information (argument of $\alpha$, $\beta$) is crucial for interference effects.

---

## 1.5.5 De Moivre's Theorem

**Theorem 1.5.7** (De Moivre's Theorem)
For any real $\theta$ and integer $n$:

$$(\cos\theta + i\sin\theta)^n = \cos(n\theta) + i\sin(n\theta)$$

Equivalently: $(e^{i\theta})^n = e^{in\theta}$

*Proof:* By induction on $n \geq 0$. Base case $n=1$ is trivial. Assume true for $n$, then:

$$(\cos\theta + i\sin\theta)^{n+1} = (\cos(n\theta) + i\sin(n\theta))(\cos\theta + i\sin\theta)$$

Using product formulas from trigonometry:

$$= \cos(n\theta)\cos\theta - \sin(n\theta)\sin\theta + i(\sin(n\theta)\cos\theta + \cos(n\theta)\sin\theta)$$

$$= \cos((n+1)\theta) + i\sin((n+1)\theta) \quad \square$$

**Example 1.5.10:** Compute $(1 + i)^8$ using De Moivre's theorem.

First, convert to polar form: $|1 + i| = \sqrt{2}$, $\arg(1+i) = \pi/4$, so $1 + i = \sqrt{2}\,e^{i\pi/4}$.

$$(1 + i)^8 = (\sqrt{2})^8 \, e^{i \cdot 8\pi/4} = 2^4 \, e^{i \cdot 2\pi} = 16 \cdot 1 = 16$$

Verify directly: $(1+i)^2 = 2i$, $(2i)^2 = -4$, $(-4)^2 = 16$. $\checkmark$

**Application: Finding $n$-th Roots**

To solve $z^n = w$ for $w = r e^{i\theta}$:

$$z = r^{1/n} e^{i(\theta + 2\pi k)/n}, \quad k = 0, 1, \ldots, n-1$$

**Example:** Find all cube roots of $1 + i$.

*Solution:* First, convert to polar: $|1+i| = \sqrt{2}$, $\arg(1+i) = \pi/4$, so $1 + i = \sqrt{2} e^{i\pi/4}$.

The cube roots are:

$$z_k = 2^{1/6} e^{i(\pi/4 + 2\pi k)/3}, \quad k = 0, 1, 2$$

- $z_0 = 2^{1/6} e^{i\pi/12}$
- $z_1 = 2^{1/6} e^{i9\pi/12} = 2^{1/6} e^{i3\pi/4}$
- $z_2 = 2^{1/6} e^{i17\pi/12}$

**Example 1.5.11:** Find both roots of $z^2 + 2z + 5 = 0$.

Using the quadratic formula with complex arithmetic:

$$z = \frac{-2 \pm \sqrt{4 - 20}}{2} = \frac{-2 \pm \sqrt{-16}}{2} = \frac{-2 \pm 4i}{2} = -1 \pm 2i$$

Verify $z = -1 + 2i$: $(-1+2i)^2 + 2(-1+2i) + 5 = (1 - 4i - 4) + (-2 + 4i) + 5 = 0$. $\checkmark$

Note the roots are conjugate pairs, as guaranteed for polynomials with real coefficients.

---

## 1.5.6 Complex Functions and Analyticity

**Definition 1.5.7** (Complex Function)
A **complex function** is a map $f: D \subseteq \mathbb{C} \to \mathbb{C}$, where $D$ is the domain.

**Definition 1.5.8** (Complex Differentiability)
A function $f$ is **complex differentiable** at $z_0$ if the limit exists:

$$f'(z_0) = \lim_{h \to 0} \frac{f(z_0 + h) - f(z_0)}{h}, \quad h \in \mathbb{C}$$

This limit must be the same regardless of how $h \to 0$ (from any direction in $\mathbb{C}$).

**Definition 1.5.9** (Analytic Function)
A function is **analytic** (or **holomorphic**) on a domain $D$ if it is complex differentiable at every point in $D$.

### Cauchy-Riemann Equations

**Theorem 1.5.8** (Cauchy-Riemann Equations)
Let $f(z) = u(x,y) + iv(x,y)$ where $z = x + iy$. If $f$ is analytic at $z$, then:

$$\frac{\partial u}{\partial x} = \frac{\partial v}{\partial y}, \quad \frac{\partial u}{\partial y} = -\frac{\partial v}{\partial x}$$

*Geometric interpretation:* The Cauchy-Riemann equations ensure $f$ is a **conformal map** (preserves angles locally). Analyticity is far more restrictive than real differentiability.

**Example:** Show $f(z) = e^z$ is analytic.

*Solution:* Write $f(x+iy) = e^x(\cos y + i\sin y) = e^x \cos y + i e^x \sin y$.
Thus $u(x,y) = e^x \cos y$, $v(x,y) = e^x \sin y$.

Check Cauchy-Riemann:
- $\frac{\partial u}{\partial x} = e^x \cos y = \frac{\partial v}{\partial y}$ ✓
- $\frac{\partial u}{\partial y} = -e^x \sin y = -\frac{\partial v}{\partial x}$ ✓

Since partial derivatives are continuous everywhere, $e^z$ is analytic on $\mathbb{C}$.

> **ML/AI Connection: Signal Processing & Phase**
> In signal processing, complex representation captures both amplitude and phase:
> $$s(t) = A(t) e^{i\phi(t)}$$
> where $A(t) = |s(t)|$ is the envelope and $\phi(t) = \arg(s(t))$ is the instantaneous phase. Analytic signals (complex extensions of real signals) allow clean separation of amplitude and frequency modulation, crucial for feature extraction in audio/speech ML models.

---

## 1.5.7 The Complex Plane as $\mathbb{R}^2$

**Theorem 1.5.9** (Isomorphism with $\mathbb{R}^2$)
As a real vector space, $\mathbb{C} \cong \mathbb{R}^2$ under the map:

$$\phi: a + bi \mapsto \begin{pmatrix} a \\ b \end{pmatrix}$$

This is an isomorphism of real vector spaces (preserves addition and scalar multiplication by reals).

### Multiplication as Linear Transformation

**Theorem 1.5.10** (Complex Multiplication as Matrix)
Multiplication by $z = a + bi$ corresponds to the linear transformation on $\mathbb{R}^2$:

$$\begin{pmatrix} a & -b \\ b & a \end{pmatrix}$$

*Proof:* Let $w = c + di$. Then $zw = (ac - bd) + (ad + bc)i$. In matrix form:

$$\begin{pmatrix} a & -b \\ b & a \end{pmatrix} \begin{pmatrix} c \\ d \end{pmatrix} = \begin{pmatrix} ac - bd \\ ad + bc \end{pmatrix}$$

which corresponds to $\text{Re}(zw)$ and $\text{Im}(zw)$. $\square$

**Example 1.5.12:** Represent multiplication by $z = 1 + 2i$ as a matrix, and apply it to $w = 3 - i$.

$$M_z = \begin{pmatrix} 1 & -2 \\ 2 & 1 \end{pmatrix}, \quad M_z \begin{pmatrix} 3 \\ -1 \end{pmatrix} = \begin{pmatrix} (1)(3) + (-2)(-1) \\ (2)(3) + (1)(-1) \end{pmatrix} = \begin{pmatrix} 5 \\ 5 \end{pmatrix}$$

This gives $5 + 5i$. Verify: $(1 + 2i)(3 - i) = 3 - i + 6i - 2i^2 = 3 + 5i + 2 = 5 + 5i$. $\checkmark$

**Corollary 1.5.2** (Rotation and Scaling)
Multiplication by $e^{i\theta} = \cos\theta + i\sin\theta$ is a rotation by angle $\theta$:

$$\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

This is the standard 2D rotation matrix. Multiplication by $r e^{i\theta}$ is rotation by $\theta$ followed by scaling by $r$.

### Complex Eigenvalues

**Theorem 1.5.11** (Complex Eigenvalues of Real Matrices)
If $A$ is a real $n \times n$ matrix and $\lambda = a + bi$ (with $b \neq 0$) is an eigenvalue with eigenvector $v = u + iw$ (where $u, w \in \mathbb{R}^n$), then $\overline{\lambda} = a - bi$ is also an eigenvalue with eigenvector $\overline{v} = u - iw$.

*Geometric interpretation:* Complex eigenvalues come in conjugate pairs, representing rotations with scaling in 2D invariant subspaces.

**Example:** The matrix $A = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$ has eigenvalues $\lambda = \pm i$.

This matrix rotates vectors by $90°$ counterclockwise. The eigenvector for $\lambda = i$ is $\begin{pmatrix} 1 \\ -i \end{pmatrix}$:

$$A \begin{pmatrix} 1 \\ -i \end{pmatrix} = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 1 \\ -i \end{pmatrix} = \begin{pmatrix} i \\ 1 \end{pmatrix} = i \begin{pmatrix} 1 \\ -i \end{pmatrix}$$

> **ML Connection: Eigenvalues in Neural Dynamics**
> Recurrent neural networks (RNNs) have dynamics governed by weight matrix $W$. If $W$ has complex eigenvalues $\lambda = re^{i\theta}$ with $|\lambda| > 1$, the network exhibits oscillatory exploding gradients. Spectral normalization techniques constrain $\max_i |\lambda_i| \leq 1$ to ensure stable training.

---

## Summary

Complex numbers extend $\mathbb{R}$ to an algebraically closed field with rich geometric structure:

1. **Arithmetic**: Addition (vector), multiplication (rotate-scale), conjugation (reflection)
2. **Polar Form**: $z = re^{i\theta}$ reveals geometric nature via Euler's formula
3. **Roots of Unity**: Evenly spaced on unit circle, fundamental to Fourier analysis
4. **Complex Exponentials**: Unify oscillations and damping, $e^{i\theta}$ traces unit circle
5. **Analyticity**: Cauchy-Riemann equations, far stronger than real differentiability
6. **Geometric Structure**: $\mathbb{C} \cong \mathbb{R}^2$, multiplication as rotation-scaling matrix

**Key Insights:**
- Complex numbers are not "imaginary" but geometric (points in plane with special multiplication)
- Euler's formula $e^{i\theta} = \cos\theta + i\sin\theta$ bridges exponentials and trigonometry
- Roots of unity provide orthogonal bases for Fourier decomposition
- Complex analysis powers signal processing, quantum mechanics, and modern ML architectures

---

## Exercises

### Basic (★)

1. Compute $(3 + 4i)(1 - 2i)$ and express in the form $a + bi$.

2. Find the modulus and argument of $z = -1 + \sqrt{3}i$.

3. Convert $z = 2e^{i\pi/3}$ to Cartesian form.

4. Find all 4th roots of $i$.

5. Show that $|z_1 z_2| = |z_1| |z_2|$ using the polar form.

### Intermediate (★★)

6. Prove that $\cos(3\theta) = 4\cos^3\theta - 3\cos\theta$ using De Moivre's theorem.

7. Let $\omega = e^{2\pi i/5}$ be a primitive 5th root of unity. Compute $1 + \omega + \omega^2 + \omega^3 + \omega^4$.

8. Show that $f(z) = z^2$ satisfies the Cauchy-Riemann equations. Find $f'(z)$.

9. Express $\sin\theta$ in terms of $e^{i\theta}$ and $e^{-i\theta}$. (Hint: Use Euler's formula for both $\theta$ and $-\theta$.)

10. Find the matrix representation of multiplication by $2 + 3i$ as a linear transformation on $\mathbb{R}^2$.

### Challenging (★★★)

11. **Quantum Rotation Gates**: In quantum computing, the rotation gate $R_\theta = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\theta} \end{pmatrix}$ applies phase $e^{i\theta}$ to $|1\rangle$. Show that $R_\theta |+\rangle = \cos(\theta/2)|+\rangle + \sin(\theta/2)e^{i\phi}|-\rangle$ where $|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$.

12. **DFT Matrix**: Show that the $N \times N$ DFT matrix $F$ with entries $F_{jk} = \frac{1}{\sqrt{N}} \omega_N^{-jk}$ (where $\omega_N = e^{2\pi i/N}$) is unitary: $F^* F = I$. (Hint: Use orthogonality of roots of unity.)

13. **Complex Logarithm**: Define $\log z$ for $z \neq 0$ by $\log z = \ln|z| + i\arg(z)$. Show that $\log z$ is multi-valued due to periodicity of $\arg(z)$. What is $\log(-1)$?

14. Let $f(z) = \frac{1}{z}$. Show $f$ is analytic on $\mathbb{C} \setminus \{0\}$ by verifying Cauchy-Riemann equations.

15. **Mandelbrot Set**: For $c \in \mathbb{C}$, define the sequence $z_0 = 0$, $z_{n+1} = z_n^2 + c$. The Mandelbrot set is $M = \{c : |z_n| \not\to \infty\}$. Show that $c = 0 \in M$ but $c = 1 \notin M$.

---

## Related Topics

- **Chapter 2.2: Linear Transformations**: Rotation matrices, eigenvalue decomposition
- **Chapter 3.4: Fourier Series**: Orthogonality of $e^{inx}$, frequency decomposition
- **Chapter 4.1: Differential Equations**: Complex characteristic equations, oscillatory solutions
- **Chapter 6.3: Quantum Computing**: State vectors, unitary evolution, phase estimation
- **Chapter 7.2: Signal Processing**: Analytic signals, Hilbert transform, time-frequency analysis
- **Chapter 8.5: Attention Mechanisms**: Rotary embeddings, complex-valued neural networks

---

## Further Reading

- **Ahlfors, L.** *Complex Analysis* (3rd ed., 1979) - Classic rigorous treatment
- **Tristan Needham**, *Visual Complex Analysis* (1997) - Geometric intuition
- **Stein & Shakarchi**, *Complex Analysis* (2003) - Modern with applications
- **ML/Quantum**: Nielsen & Chuang, *Quantum Computation and Quantum Information* (2010)
- **RoPE Paper**: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
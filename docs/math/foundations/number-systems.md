# Number Systems

The numbers we use in machine learning — real-valued weights, integer indices, complex quantum amplitudes — are not a monolith. They form a hierarchy of increasingly rich algebraic structures, each extending the previous one to solve equations that couldn't be solved before. Understanding this hierarchy clarifies why we use $\mathbb{R}^n$ for feature spaces, $\mathbb{C}$ for quantum states, and why floating-point arithmetic can silently corrupt your gradients.

---

## Prerequisites

- [Sets & Logic](sets-and-logic.md)

---

## 1. The Number Hierarchy

```
ℕ ⊂ ℤ ⊂ ℚ ⊂ ℝ ⊂ ℂ

ℕ (Natural)     {0, 1, 2, 3, ...}        Counting, indexing
  │  Added: negative numbers (additive inverses)
  ▼
ℤ (Integers)    {..., -2, -1, 0, 1, 2, ...}  Differences, signed quantities
  │  Added: division (multiplicative inverses)
  ▼
ℚ (Rationals)   {p/q : p,q ∈ ℤ, q ≠ 0}     Exact fractions, ratios
  │  Added: limits of sequences (completeness)
  ▼
ℝ (Reals)       The continuum               Weights, features, probabilities
  │  Added: √(-1) (algebraic closure)
  ▼
ℂ (Complex)     {a + bi : a,b ∈ ℝ}         Quantum amplitudes, Fourier
```

Each extension solves a specific algebraic deficiency and introduces new structure.

---

## 2. Natural Numbers $\mathbb{N}$

**Definition 2.1 (Peano Axioms).** The natural numbers are defined by:

1. $0 \in \mathbb{N}$
2. If $n \in \mathbb{N}$, then $S(n) \in \mathbb{N}$ (successor function)
3. There is no $n$ with $S(n) = 0$
4. $S(m) = S(n) \Rightarrow m = n$ (injectivity)
5. **Induction:** If $P(0)$ and $P(k) \Rightarrow P(S(k))$ for all $k$, then $P(n)$ for all $n$

From these axioms, we can define addition ($m + 0 = m$, $m + S(n) = S(m + n)$) and multiplication ($m \cdot 0 = 0$, $m \cdot S(n) = m \cdot n + m$).

**Example 2.1:** Computing $2 + 3$ from the axioms. Here $2 = S(S(0))$ and $3 = S(S(S(0)))$:

$$2 + 3 = 2 + S(2) = S(2 + 2) = S(2 + S(1)) = S(S(2 + 1)) = S(S(S(2 + 0))) = S(S(S(2))) = 5$$

Each step applies $m + S(n) = S(m + n)$, bottoming out at $m + 0 = m$.

*ML connection:* $\mathbb{N}$ indexes training examples ($x_1, x_2, \ldots, x_n$), counts classes, and enumerates layers. Induction is the primary tool for proving properties of recursive algorithms and neural network depth.

**Algebraic structure:** $(\mathbb{N}, +, \cdot)$ is a commutative semiring. It lacks additive inverses: $3 + x = 0$ has no solution in $\mathbb{N}$.

---

## 3. Integers $\mathbb{Z}$

**Definition 2.2.** $\mathbb{Z} = \{\ldots, -2, -1, 0, 1, 2, \ldots\}$, formally constructed as equivalence classes of pairs $(a, b) \in \mathbb{N} \times \mathbb{N}$ under $(a,b) \sim (c,d) \iff a + d = b + c$, where $(a,b)$ represents $a - b$.

**Algebraic structure:** $(\mathbb{Z}, +, \cdot)$ is a commutative ring. Every element has an additive inverse: $n + (-n) = 0$. But it lacks multiplicative inverses: $2x = 1$ has no solution in $\mathbb{Z}$.

**Example 3.1:** In the formal construction, the integer $-3$ is the equivalence class of the pair $(0, 3)$ since it represents $0 - 3$. But $(1, 4)$, $(2, 5)$, and $(7, 10)$ all represent the same integer, because:

$$(0, 3) \sim (1, 4) \iff 0 + 4 = 3 + 1 = 4 \quad \checkmark$$

To add $-3 + 5$: we use representatives $(0,3)$ and $(5,0)$, giving $(0+5,\; 3+0) = (5, 3)$, which represents $5 - 3 = 2$.

*ML connection:* Integer quantization maps floating-point weights to $\mathbb{Z}$ (typically int8) for efficient inference. The quantization function $Q(w) = \text{round}(w / s)$ where $s$ is a scale factor trades precision for speed.

---

## 4. Rational Numbers $\mathbb{Q}$

**Definition 2.3.** $\mathbb{Q} = \{p/q : p \in \mathbb{Z}, q \in \mathbb{Z} \setminus \{0\}\}$, with $(p,q) \sim (p', q') \iff pq' = p'q$.

**Algebraic structure:** $(\mathbb{Q}, +, \cdot)$ is a field — a commutative ring where every nonzero element has a multiplicative inverse. This is the smallest field containing $\mathbb{Z}$.

**Example 4.1:** The fractions $\frac{2}{3}$ and $\frac{6}{9}$ represent the same rational number because $2 \times 9 = 6 \times 3 = 18$. Arithmetic example:

$$\frac{2}{3} + \frac{3}{4} = \frac{2 \cdot 4 + 3 \cdot 3}{3 \cdot 4} = \frac{8 + 9}{12} = \frac{17}{12}$$

The multiplicative inverse of $\frac{2}{3}$ is $\frac{3}{2}$, since $\frac{2}{3} \cdot \frac{3}{2} = \frac{6}{6} = 1$. This is why $\mathbb{Q}$ is a field but $\mathbb{Z}$ is not.

**The gap in $\mathbb{Q}$:** $\mathbb{Q}$ is *not complete*. There exist Cauchy sequences of rationals that do not converge to a rational. Example: the sequence $1, 1.4, 1.41, 1.414, \ldots$ converges to $\sqrt{2} \notin \mathbb{Q}$.

**Theorem 2.1 (Density of $\mathbb{Q}$).** Between any two real numbers, there exists a rational number.

*Proof.* Let $a < b$ with $a, b \in \mathbb{R}$. Choose $n \in \mathbb{N}$ with $n > 1/(b-a)$ (Archimedean property). Then there exists $m \in \mathbb{Z}$ with $a < m/n < b$. $\square$

**Example 4.2:** Find a rational between $\sqrt{2} \approx 1.4142$ and $\sqrt{3} \approx 1.7320$.

We need $n > 1/(1.7320 - 1.4142) \approx 3.14$, so take $n = 4$. Now find $m$ with $\sqrt{2} < m/4 < \sqrt{3}$, i.e., $5.66 < m < 6.93$. Take $m = 6$: then $\frac{6}{4} = \frac{3}{2} = 1.5$, and indeed $\sqrt{2} < 1.5 < \sqrt{3}$. $\checkmark$

*ML connection:* Despite $\mathbb{Q}$ being dense in $\mathbb{R}$, computations on digital hardware use a finite subset of $\mathbb{Q}$ (floating-point numbers). This introduces rounding errors that can accumulate during training.

---

## 5. Real Numbers $\mathbb{R}$

### 5.1 Construction and Completeness

**Definition 2.4 (Dedekind Cuts).** A *Dedekind cut* is a partition of $\mathbb{Q}$ into two nonempty sets $(L, R)$ such that every element of $L$ is less than every element of $R$, and $L$ has no greatest element. Each cut defines a real number.

**Axiom 2.1 (Completeness).** Every nonempty subset of $\mathbb{R}$ that is bounded above has a least upper bound (supremum).

**Example 5.1:** The Dedekind cut for $\sqrt{2}$: let $L = \{q \in \mathbb{Q} : q < 0 \text{ or } q^2 < 2\}$ and $R = \{q \in \mathbb{Q} : q \geq 0 \text{ and } q^2 \geq 2\}$.

- $1.4 \in L$ because $1.4^2 = 1.96 < 2$
- $1.5 \in R$ because $1.5^2 = 2.25 \geq 2$
- $L$ has no greatest element (we can always find a closer rational), so this cut defines the real number $\sqrt{2}$.

**Example 5.2:** Let $S = \{1 - 1/n : n \in \mathbb{N}, n \geq 1\} = \{0,\; 1/2,\; 2/3,\; 3/4,\; \ldots\}$. Then $\sup(S) = 1$, because:

- $1 - 1/n < 1$ for all $n$, so $1$ is an upper bound
- For any $\varepsilon > 0$, choose $n > 1/\varepsilon$; then $1 - 1/n > 1 - \varepsilon$, so no smaller upper bound works

In $\mathbb{Q}$ this supremum exists (it is $1 \in \mathbb{Q}$), but for $S' = \{q \in \mathbb{Q} : q^2 < 2\}$, $\sup(S') = \sqrt{2} \notin \mathbb{Q}$ — completeness fails.

This is the property that distinguishes $\mathbb{R}$ from $\mathbb{Q}$. It guarantees that limits of bounded sequences exist, which is essential for:

- **Convergence of gradient descent** — the loss sequence $\{L(\theta_t)\}$ converges if bounded below
- **Existence of optima** — continuous functions on compact sets attain their minimum (Extreme Value Theorem)
- **Integration** — the Riemann integral is defined via suprema and infima of sums

### 5.2 Key Properties

**Theorem 2.2 (Archimedean Property).** For any $x \in \mathbb{R}$, there exists $n \in \mathbb{N}$ with $n > x$.

**Theorem 2.3 (Bolzano-Weierstrass).** Every bounded sequence in $\mathbb{R}$ has a convergent subsequence.

*ML connection:* The Bolzano-Weierstrass theorem is used to prove that gradient descent on compact parameter spaces has convergent subsequences, which is key to showing that training procedures find local minima.

### 5.3 Floating-Point Representation

In practice, we work with a finite subset of $\mathbb{R}$: IEEE 754 floating-point numbers.

$$x = (-1)^s \times m \times 2^e$$

where $s$ is the sign bit, $m$ is the mantissa, and $e$ is the exponent.

| Format | Bits | Mantissa | Exponent | Range | Precision |
|--------|------|----------|----------|-------|-----------|
| float16 (half) | 16 | 10 bits | 5 bits | $\pm 6.5 \times 10^4$ | ~3 decimal digits |
| bfloat16 | 16 | 7 bits | 8 bits | $\pm 3.4 \times 10^{38}$ | ~2 decimal digits |
| float32 (single) | 32 | 23 bits | 8 bits | $\pm 3.4 \times 10^{38}$ | ~7 decimal digits |
| float64 (double) | 64 | 52 bits | 11 bits | $\pm 1.8 \times 10^{308}$ | ~15 decimal digits |

*ML connection:* Mixed-precision training uses float16 for forward/backward passes (speed) and float32 for weight updates (precision). bfloat16 was designed by Google specifically for deep learning — it sacrifices mantissa bits for exponent range, since neural networks care more about dynamic range than precision.

**Example 5.3:** How is $-6.5$ stored in float32? First, $-6.5 = -1 \times 1.625 \times 2^2$.

- Sign bit: $s = 1$ (negative)
- Exponent: $e = 2$, stored as $2 + 127 = 129 = 10000001_2$ (bias-127 encoding)
- Mantissa: $1.625 = 1.101_2$ (the leading $1.$ is implicit), so we store $10100\ldots0$ (23 bits)

$$\underbrace{1}_{s}\;\underbrace{10000001}_{e=129}\;\underbrace{10100000000000000000000}_{m=1.625}$$

This is an exact representation. But $0.1_{10} = 0.0\overline{0011}_2$ repeats forever, so it cannot be stored exactly.

**Machine epsilon** $\varepsilon_{\text{mach}}$: the smallest $\varepsilon > 0$ such that $1.0 + \varepsilon \neq 1.0$ in floating point. For float32, $\varepsilon_{\text{mach}} \approx 1.19 \times 10^{-7}$. Gradients smaller than this are effectively zero.

**Example 5.4:** In float32, the mantissa has 23 bits, so $\varepsilon_{\text{mach}} = 2^{-23} \approx 1.19 \times 10^{-7}$. This means:

$$1.0 + 5 \times 10^{-8} = 1.0 \quad \text{(in float32!)}$$

A practical consequence: if your learning rate is $10^{-7}$ and a weight is $1.0$, the update $w \leftarrow w - \eta \cdot g$ does nothing when $|g| \leq 1$. This is why loss scaling is essential in float16 training, where $\varepsilon_{\text{mach}} = 2^{-10} \approx 9.77 \times 10^{-4}$.

---

## 6. Complex Numbers $\mathbb{C}$

### 6.1 Definition and Arithmetic

**Definition 2.5.** $\mathbb{C} = \{a + bi : a, b \in \mathbb{R}\}$ where $i^2 = -1$.

- **Real part:** $\text{Re}(a + bi) = a$
- **Imaginary part:** $\text{Im}(a + bi) = b$
- **Complex conjugate:** $\overline{a + bi} = a - bi$
- **Modulus:** $|a + bi| = \sqrt{a^2 + b^2}$
- **Multiplication:** $(a+bi)(c+di) = (ac-bd) + (ad+bc)i$

**Example 6.1:** Let $z_1 = 3 + 4i$ and $z_2 = 1 - 2i$.

- **Conjugate:** $\overline{z_1} = 3 - 4i$
- **Modulus:** $|z_1| = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = 5$
- **Multiplication:** $z_1 \cdot z_2 = (3+4i)(1-2i) = (3 \cdot 1 - 4 \cdot (-2)) + (3 \cdot (-2) + 4 \cdot 1)i = 11 - 2i$
- **Modulus check:** $|z_1 \cdot z_2| = |z_1| \cdot |z_2| = 5 \cdot \sqrt{5} = 5\sqrt{5} \approx 11.18$, and indeed $\sqrt{11^2 + 2^2} = \sqrt{125} = 5\sqrt{5}$ $\checkmark$

### 6.2 Euler's Formula

**Theorem 2.4 (Euler's Formula).** $e^{i\theta} = \cos\theta + i\sin\theta$

This connects exponentials, trigonometry, and complex numbers. The special case $e^{i\pi} + 1 = 0$ (Euler's identity) unites five fundamental constants.

**Example 6.2:** Verify Euler's formula for $\theta = \pi/3$:

$$e^{i\pi/3} = \cos(\pi/3) + i\sin(\pi/3) = \frac{1}{2} + i\frac{\sqrt{3}}{2}$$

This is a point on the unit circle at $60°$, with modulus $|e^{i\pi/3}| = \sqrt{(1/2)^2 + (\sqrt{3}/2)^2} = \sqrt{1/4 + 3/4} = 1$. $\checkmark$

### 6.3 Polar Form

Every nonzero complex number can be written as $z = re^{i\theta}$ where $r = |z|$ and $\theta = \arg(z)$.

```
COMPLEX PLANE (Argand diagram)

  Im
   │        z = a + bi = r·e^(iθ)
   │       ╱│
   │ r   ╱  │ b
   │   ╱    │
   │ ╱ θ    │
   └────────────── Re
         a
```

**Example 6.3:** Convert $z = -1 + i$ to polar form.

- $r = |z| = \sqrt{(-1)^2 + 1^2} = \sqrt{2}$
- $\theta = \arg(z) = \arctan(1 / -1) = 3\pi/4$ (second quadrant, since $\text{Re} < 0$, $\text{Im} > 0$)
- Polar form: $z = \sqrt{2}\, e^{i \cdot 3\pi/4}$

Multiplying two polar numbers is easy: $\sqrt{2}\,e^{i\cdot 3\pi/4} \times \sqrt{2}\,e^{i\cdot 3\pi/4} = 2\,e^{i\cdot 3\pi/2} = -2i$.
Check: $(-1+i)^2 = 1 - 2i + i^2 = 1 - 2i - 1 = -2i$. $\checkmark$

### 6.4 The Fundamental Theorem of Algebra

**Theorem 2.5.** Every non-constant polynomial with complex coefficients has at least one root in $\mathbb{C}$.

This means $\mathbb{C}$ is *algebraically closed* — we never need to extend it further to solve polynomial equations.

*ML/Quantum connection:*

- **Fourier transforms** map signals to the complex plane: $\hat{f}(\omega) = \int f(t)e^{-i\omega t}\,dt$. Every frequency component is a complex number encoding amplitude and phase.
- **Quantum states** are unit vectors in complex Hilbert spaces: $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ where $\alpha, \beta \in \mathbb{C}$ and $|\alpha|^2 + |\beta|^2 = 1$.
- **Eigenvalues** of real matrices can be complex. The complex eigenvalues of the Hessian matrix determine the nature of critical points in the loss landscape.

---

## 7. Field Axioms (Summary)

A **field** $(F, +, \cdot)$ satisfies:

| Axiom | Addition | Multiplication |
|-------|----------|----------------|
| Closure | $a + b \in F$ | $a \cdot b \in F$ |
| Associativity | $(a+b)+c = a+(b+c)$ | $(ab)c = a(bc)$ |
| Commutativity | $a+b = b+a$ | $ab = ba$ |
| Identity | $a + 0 = a$ | $a \cdot 1 = a$ |
| Inverse | $a + (-a) = 0$ | $a \cdot a^{-1} = 1$ ($a \neq 0$) |
| Distributivity | $a(b+c) = ab + ac$ | |

$\mathbb{Q}, \mathbb{R}, \mathbb{C}$ are all fields. $\mathbb{Z}$ is not (no multiplicative inverses). $\mathbb{N}$ is not (no additive inverses).

**Example 7.1:** Why $\mathbb{Z}$ fails the field axioms but $\mathbb{Q}$ passes. Consider the element $3$:

- In $\mathbb{Z}$: we need $3 \cdot x = 1$ for some $x \in \mathbb{Z}$. But $x = 1/3 \notin \mathbb{Z}$. No multiplicative inverse exists.
- In $\mathbb{Q}$: $3 \cdot (1/3) = 1$. The inverse $1/3 \in \mathbb{Q}$. $\checkmark$

Contrast with $\mathbb{Z}_5$ (integers mod 5): $3 \cdot 2 = 6 \equiv 1 \pmod{5}$, so $3^{-1} = 2$ in $\mathbb{Z}_5$. This works because $5$ is prime, making $\mathbb{Z}_5$ a finite field with 5 elements.

*ML connection:* Linear algebra over a field $F$ means we can divide by scalars, invert matrices, and solve systems of equations. Neural networks operate over $\mathbb{R}$ (or $\mathbb{C}$ in some architectures), which is why the field axioms matter.

---

## Exercises

**★ Basic**

1. Prove that $\sqrt{3}$ is irrational.
2. Compute $(2 + 3i)(1 - i)$ and find its modulus.
3. Verify Euler's formula for $\theta = \pi/4$ by computing both sides.

**★★ Intermediate**

4. Prove that between any two irrational numbers, there exists a rational number.
5. Show that $(\mathbb{Z}_p, +, \cdot)$ is a field when $p$ is prime (integers modulo $p$). This is the foundation of finite-field cryptography.
6. Demonstrate with a concrete example how float32 addition is not associative: find $a, b, c$ where $(a + b) + c \neq a + (b + c)$.

**★★★ Challenging**

7. Prove the Fundamental Theorem of Algebra using Liouville's theorem (a bounded entire function is constant).
8. The ring of Gaussian integers $\mathbb{Z}[i] = \{a + bi : a, b \in \mathbb{Z}\}$ is not a field. Find a Gaussian integer that has no multiplicative inverse. Then show that $\mathbb{Z}[i]$ is a unique factorization domain.

---

## Related Topics

- [Sets & Logic](sets-and-logic.md) — the framework in which number systems are constructed
- [Functions & Relations](functions-and-relations.md) — functions between number systems
- [Vectors](../linear-algebra/vectors.md) — vectors over $\mathbb{R}$ and $\mathbb{C}$
- [Complex Vector Spaces](../quantum/complex-vector-spaces.md) — extending to quantum Hilbert spaces

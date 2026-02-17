# Taylor Series

Taylor series are the bridge between complicated functions and simple polynomials. In machine learning, nearly every optimization algorithm relies on local polynomial approximations: gradient descent uses a first-order Taylor expansion, Newton's method uses a second-order expansion, and the entire theory of loss landscape analysis rests on approximating the loss function near critical points. In variational inference, Taylor-like bounds (via Jensen's inequality) underpin the ELBO. In numerical computing, Taylor expansions explain why the log-sum-exp trick works and how finite difference gradient checking achieves its accuracy. Mastering Taylor series means understanding *why* these approximations are valid and *when* they break down.

---

## Prerequisites

- [Differentiation](differentiation.md) --- derivatives, chain rule, higher-order derivatives
- [Integration](integration.md) --- integral remainder form
- [Multivariable Calculus](multivariable-calculus.md) --- gradient, Hessian (for Section 5)

---

## 1. Taylor's Theorem (Single Variable)

**Definition 3.6.1 (Taylor Polynomial).** The *$n$-th degree Taylor polynomial* of $f$ centered at $a$ is:

$$T_n(x) = \sum_{k=0}^{n} \frac{f^{(k)}(a)}{k!}(x - a)^k = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots + \frac{f^{(n)}(a)}{n!}(x-a)^n$$

**Example 3.6.1:** Compute $T_3(x)$ for $f(x) = e^x$ centered at $a = 0$.

Since every derivative of $e^x$ is $e^x$, we have $f^{(k)}(0) = e^0 = 1$ for all $k$. Therefore:

$$T_3(x) = \frac{1}{0!}x^0 + \frac{1}{1!}x^1 + \frac{1}{2!}x^2 + \frac{1}{3!}x^3 = 1 + x + \frac{x^2}{2} + \frac{x^3}{6}$$

At $x = 1$: $T_3(1) = 1 + 1 + 0.5 + 0.1\overline{6} = 2.6\overline{6}$, compared to $e^1 \approx 2.71828$ (error $\approx 0.05$).

**Theorem 3.6.1 (Taylor's Theorem with Remainder).** Let $f$ be $(n+1)$-times differentiable on an interval containing $a$ and $x$. Then:

$$f(x) = T_n(x) + R_n(x)$$

where the remainder $R_n(x)$ can be expressed in several forms:

| Remainder Form | Expression |
|----------------|------------|
| **Lagrange** | $R_n(x) = \frac{f^{(n+1)}(c)}{(n+1)!}(x-a)^{n+1}$ for some $c$ between $a$ and $x$ |
| **Integral** | $R_n(x) = \int_a^x \frac{f^{(n+1)}(t)}{n!}(x-t)^n \, dt$ |
| **Cauchy** | $R_n(x) = \frac{f^{(n+1)}(c)}{n!}(x-c)^n(x-a)$ for some $c$ between $a$ and $x$ |

*Proof (Lagrange remainder).* Define $F(t) = f(x) - T_n(x;a\to t) - K(x-t)^{n+1}$ where $T_n(x; a \to t)$ is the Taylor polynomial of $f$ at $t$ evaluated at $x$, and $K$ is chosen so that $F(a) = 0$. We have $F(x) = 0$ trivially (the expansion at $x$ is exact). By Rolle's theorem, $F'(c) = 0$ for some $c$ between $a$ and $x$. Computing $F'(t)$ and using the telescoping cancellation of derivative terms:

$$F'(t) = -\frac{f^{(n+1)}(t)}{n!}(x-t)^n + K(n+1)(x-t)^n$$

Setting $F'(c) = 0$ gives $K = \frac{f^{(n+1)}(c)}{(n+1)!}$. Since $F(a) = 0$:

$$f(x) - T_n(x) - \frac{f^{(n+1)}(c)}{(n+1)!}(x-a)^{n+1} = 0$$

which yields $R_n(x) = \frac{f^{(n+1)}(c)}{(n+1)!}(x-a)^{n+1}$. $\square$

**Example 3.6.2:** Bound the error when approximating $e^x$ by $T_3(x)$ at $x = 0.5$ (centered at $a = 0$).

By the Lagrange remainder: $|R_3(0.5)| = \frac{|f^{(4)}(c)|}{4!}|0.5|^4$ for some $c \in (0, 0.5)$.

Since $f^{(4)}(x) = e^x$ is increasing, $|f^{(4)}(c)| \leq e^{0.5} < 2$. Therefore:

$$|R_3(0.5)| \leq \frac{2}{24} \cdot \frac{1}{16} = \frac{2}{384} \approx 0.0052$$

Actual error: $|e^{0.5} - T_3(0.5)| = |1.64872 - 1.64583| = 0.00289$, well within the bound.

```
TAYLOR POLYNOMIALS APPROACHING f(x) = eˣ NEAR a = 0

   f(x) = eˣ
   │
 3 ┤                                          ╱ eˣ (true function)
   │                                        ╱
   │                                      ╱╱
 2 ┤                                  ╱╱╱    ← T₃ (cubic)
   │                              ╱╱╱   ╱
   │                          ╱╱╱╱  ╱╱╱      ← T₂ (quadratic)
 1 ┤─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ●╱╱╱╱╱╱╱╱
   │                ╱╱╱╱╱╱╱╱  ╱              ← T₁ (linear: 1 + x)
   │           ╱╱╱╱╱╱    ╱
 0 ┤──────╱╱╱╱──────╱─────────────────────
   │  ╱╱╱╱      ╱                            ← T₀ (constant: 1)
   │╱        ╱
   ┼────┼────●────┼────┼────┼─────
  -2   -1    0    1    2    3     x

  T₀ = 1
  T₁ = 1 + x
  T₂ = 1 + x + x²/2
  T₃ = 1 + x + x²/2 + x³/6

  Each degree adds a correction term, improving the
  approximation near a = 0. Farther from 0, more terms needed.
```

*ML connection:* Gradient descent uses the first-order Taylor expansion $f(\theta - \eta \nabla f) \approx f(\theta) - \eta \|\nabla f\|^2$, which is valid when the step size $\eta$ is small enough that higher-order terms are negligible. The learning rate must be chosen so that $R_1$ (the remainder) does not dominate --- this is precisely the condition that the Lipschitz constant of the gradient controls.

---

## 2. Maclaurin Series

**Definition 3.6.2 (Maclaurin Series).** The *Maclaurin series* is the Taylor series centered at $a = 0$:

$$f(x) = \sum_{k=0}^{\infty} \frac{f^{(k)}(0)}{k!} x^k$$

**Example 3.6.3:** Derive the first 4 non-zero terms of the Maclaurin series for $f(x) = \sin x$.

Compute successive derivatives at $x = 0$: $f(0) = 0$, $f'(0) = \cos 0 = 1$, $f''(0) = -\sin 0 = 0$, $f'''(0) = -\cos 0 = -1$, $f^{(4)}(0) = \sin 0 = 0$, $f^{(5)}(0) = \cos 0 = 1$, $f^{(6)}(0) = 0$, $f^{(7)}(0) = -1$.

Only odd derivatives are non-zero, giving:

$$\sin x = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \frac{x^7}{7!} + \cdots = x - \frac{x^3}{6} + \frac{x^5}{120} - \frac{x^7}{5040} + \cdots$$

Check: $\sin(\pi/6) = 0.5$. Using 2 terms: $\frac{\pi}{6} - \frac{(\pi/6)^3}{6} = 0.5236 - 0.0239 = 0.4997 \approx 0.5$. $\checkmark$

**Definition 3.6.3 (Taylor Series).** More generally, the *Taylor series* of $f$ centered at $a$ is:

$$f(x) = \sum_{k=0}^{\infty} \frac{f^{(k)}(a)}{k!}(x - a)^k$$

provided the series converges to $f(x)$.

**Important subtlety:** A function can be infinitely differentiable and yet its Taylor series may not converge to the function. The classic example is $f(x) = e^{-1/x^2}$ (with $f(0) = 0$), which has $f^{(k)}(0) = 0$ for all $k$, so its Maclaurin series is identically zero --- but $f(x) \neq 0$ for $x \neq 0$.

---

## 3. Common Taylor Expansions

These series appear constantly in ML derivations. Every practitioner should know them:

| Function | Maclaurin Series | Radius of Convergence |
|----------|------------------|-----------------------|
| $e^x$ | $\displaystyle\sum_{k=0}^{\infty} \frac{x^k}{k!} = 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \cdots$ | $R = \infty$ |
| $\sin x$ | $\displaystyle\sum_{k=0}^{\infty} \frac{(-1)^k x^{2k+1}}{(2k+1)!} = x - \frac{x^3}{6} + \frac{x^5}{120} - \cdots$ | $R = \infty$ |
| $\cos x$ | $\displaystyle\sum_{k=0}^{\infty} \frac{(-1)^k x^{2k}}{(2k)!} = 1 - \frac{x^2}{2} + \frac{x^4}{24} - \cdots$ | $R = \infty$ |
| $\ln(1+x)$ | $\displaystyle\sum_{k=1}^{\infty} \frac{(-1)^{k+1} x^k}{k} = x - \frac{x^2}{2} + \frac{x^3}{3} - \cdots$ | $R = 1$ |
| $\frac{1}{1-x}$ | $\displaystyle\sum_{k=0}^{\infty} x^k = 1 + x + x^2 + x^3 + \cdots$ | $R = 1$ |
| $(1+x)^\alpha$ | $\displaystyle\sum_{k=0}^{\infty} \binom{\alpha}{k} x^k$ where $\binom{\alpha}{k} = \frac{\alpha(\alpha-1)\cdots(\alpha-k+1)}{k!}$ | $R = 1$ (for non-integer $\alpha$) |

**Example 3.6.4:** Estimate $e^{0.1}$ using the Maclaurin series and measure the accuracy.

$$e^{0.1} \approx 1 + 0.1 + \frac{(0.1)^2}{2} + \frac{(0.1)^3}{6} + \frac{(0.1)^4}{24} = 1 + 0.1 + 0.005 + 0.000\overline{16} + 0.0000041\overline{6}$$

$$= 1.10517083\ldots$$

True value: $e^{0.1} = 1.10517092\ldots$. The 4-term approximation already has error $< 10^{-7}$. This rapid convergence occurs because the remainder $|R_4(0.1)| \leq \frac{e^{0.1}}{5!}(0.1)^5 \approx 9.2 \times 10^{-8}$.

**Example 3.6.5:** Approximate $\ln(1.2)$ using the Maclaurin series for $\ln(1+x)$ with $x = 0.2$.

$$\ln(1.2) \approx 0.2 - \frac{(0.2)^2}{2} + \frac{(0.2)^3}{3} - \frac{(0.2)^4}{4} = 0.2 - 0.02 + 0.002\overline{6} - 0.0004 = 0.1822\overline{6}$$

True value: $\ln(1.2) = 0.18232\ldots$. Error $\approx 5 \times 10^{-5}$. Convergence is slower than $e^x$ because $R = 1$ (not $\infty$) and $|x| = 0.2$ is a non-trivial fraction of the radius.

**Theorem 3.6.2 (Euler's Formula).** Combining the series for $e^x$, $\sin x$, and $\cos x$ with $x = i\theta$:

$$e^{i\theta} = \cos\theta + i\sin\theta$$

*ML connection:* The expansion $\ln(1+x) \approx x - x^2/2$ for small $x$ is used throughout probabilistic ML. For instance, the KL divergence $D_{\text{KL}}(p \| q)$ for distributions close to each other can be approximated using this expansion. The geometric series $1/(1-x) = \sum x^k$ appears in discount factors for reinforcement learning: $\sum_{t=0}^\infty \gamma^t r_t$ converges when $|\gamma| < 1$.

---

## 4. Convergence of Taylor Series

**Definition 3.6.4 (Radius of Convergence).** For the power series $\sum_{k=0}^\infty a_k (x-a)^k$, the *radius of convergence* $R$ is:

$$R = \frac{1}{\limsup_{k \to \infty} |a_k|^{1/k}}$$

The series converges absolutely for $|x - a| < R$ and diverges for $|x - a| > R$.

**Theorem 3.6.3 (Ratio Test for Radius).** If $\lim_{k \to \infty} \left|\frac{a_{k+1}}{a_k}\right|$ exists, then:

$$R = \lim_{k \to \infty} \left|\frac{a_k}{a_{k+1}}\right|$$

*Proof.* Apply the ratio test: $\sum |a_k(x-a)^k|$ converges when $\lim \frac{|a_{k+1}||x-a|^{k+1}}{|a_k||x-a|^k} = |x-a| \lim \frac{|a_{k+1}|}{|a_k|} < 1$, giving $|x - a| < \lim |a_k/a_{k+1}| = R$. $\square$

**Example.** For $e^x$: $a_k = 1/k!$, so $|a_k/a_{k+1}| = (k+1)! / k! = k+1 \to \infty$, giving $R = \infty$.

For $\ln(1+x)$: $a_k = (-1)^{k+1}/k$, so $|a_k/a_{k+1}| = (k+1)/k \to 1$, giving $R = 1$.

**Example 3.6.6:** Find the radius of convergence of $\displaystyle\sum_{k=0}^{\infty} \frac{k!}{3^k} x^k$.

Here $a_k = k!/3^k$. Apply the ratio test:

$$\left|\frac{a_{k+1}}{a_k}\right| = \frac{(k+1)!}{3^{k+1}} \cdot \frac{3^k}{k!} = \frac{k+1}{3} \to \infty$$

So $R = \lim |a_k / a_{k+1}| = \lim \frac{3}{k+1} = 0$. The series converges only at $x = 0$ --- the factorials grow too fast. This contrasts with $e^x = \sum x^k/k!$, where the factorial is in the denominator, giving $R = \infty$.

```
CONVERGENCE VISUALIZATION

  Taylor series for ln(1+x), centered at 0, converges for |x| < 1

                         Converges           Diverges
                    ◄─────────────────►  ◄──────────►
     ───────────────●═══════════════════●───────────────
                   -1         0          1
                    │                    │
                  Diverges        Conditional convergence
                                  (converges at x=1 to ln 2,
                                   diverges at x=-1)

  Taylor series for eˣ converges EVERYWHERE (R = ∞)

     ═══════════════════════●═══════════════════════════
                            0
                   R = ∞: converges for all x ∈ R
```

**Theorem 3.6.4 (Convergence to the Function).** If $f$ is analytic on $(a - R, a + R)$ --- meaning $R_n(x) \to 0$ as $n \to \infty$ for each $x$ in this interval --- then the Taylor series converges to $f(x)$:

$$f(x) = \sum_{k=0}^{\infty} \frac{f^{(k)}(a)}{k!}(x-a)^k \quad \text{for } |x - a| < R$$

A sufficient condition: if $|f^{(n)}(x)| \leq M$ for all $n$ and all $x$ in an interval, then $|R_n(x)| \leq \frac{M|x-a|^{n+1}}{(n+1)!} \to 0$ because factorials dominate powers.

---

## 5. Multivariable Taylor Expansion

**Theorem 3.6.5 (Multivariable Taylor Expansion to Second Order).** Let $f : \mathbb{R}^n \to \mathbb{R}$ be twice continuously differentiable. Then for a perturbation $\mathbf{h} \in \mathbb{R}^n$:

$$f(\mathbf{x} + \mathbf{h}) = f(\mathbf{x}) + \nabla f(\mathbf{x})^T \mathbf{h} + \frac{1}{2}\mathbf{h}^T H_f(\mathbf{x})\, \mathbf{h} + O(\|\mathbf{h}\|^3)$$

where $\nabla f(\mathbf{x}) \in \mathbb{R}^n$ is the gradient and $H_f(\mathbf{x}) \in \mathbb{R}^{n \times n}$ is the Hessian:

$$[\nabla f]_i = \frac{\partial f}{\partial x_i}, \qquad [H_f]_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

*Proof.* Define $g(t) = f(\mathbf{x} + t\mathbf{h})$. By the single-variable Taylor theorem:

$$g(1) = g(0) + g'(0) + \frac{1}{2}g''(0) + O(1)$$

Using the chain rule: $g'(t) = \nabla f(\mathbf{x} + t\mathbf{h})^T \mathbf{h}$, so $g'(0) = \nabla f(\mathbf{x})^T \mathbf{h}$. Differentiating again: $g''(t) = \mathbf{h}^T H_f(\mathbf{x} + t\mathbf{h})\, \mathbf{h}$, so $g''(0) = \mathbf{h}^T H_f(\mathbf{x})\, \mathbf{h}$. Substituting $t = 1$ yields the result. $\square$

**Written component-wise (2D case):**

$$f(x + h_1, y + h_2) \approx f(x,y) + f_x h_1 + f_y h_2 + \frac{1}{2}\bigl(f_{xx} h_1^2 + 2f_{xy}h_1 h_2 + f_{yy}h_2^2\bigr)$$

**Example 3.6.7:** Expand $f(x,y) = e^{x+y}$ to second order around $(0, 0)$.

All partial derivatives of $e^{x+y}$ equal $e^{x+y}$, so at $(0,0)$: $f = 1$, $f_x = f_y = 1$, $f_{xx} = f_{yy} = f_{xy} = 1$.

$$e^{x+y} \approx 1 + x + y + \frac{1}{2}(x^2 + 2xy + y^2) = 1 + (x+y) + \frac{(x+y)^2}{2}$$

This matches the single-variable result $e^u \approx 1 + u + u^2/2$ with $u = x + y$, as expected.

Check: $f(0.1, 0.2) \approx 1 + 0.3 + 0.045 = 1.345$ vs. $e^{0.3} = 1.34986$ (error $\approx 0.005$).

```
QUADRATIC APPROXIMATION OF A LOSS SURFACE

   Loss L(θ)                        Quadratic approximation
                                    L(θ*) + ½(θ - θ*)ᵀ H (θ - θ*)
     │  ╲      ╱                      │  ╲      ╱
     │   ╲    ╱                       │   ╲    ╱
     │    ╲  ╱                        │    ╲  ╱
     │     ╲╱  ← true minimum         │     ╲╱  ← same minimum
     │     θ*                         │     θ*
     │                                │
     │  (may have flat regions,       │  (perfect parabola —
     │   asymmetry, etc.)             │   symmetric bowl)
     └──────────────────              └──────────────────
              θ                                θ

   The approximation is excellent near θ*, poor far away.
   Newton's method trusts this approximation exactly.
```

*ML connection:* This is the foundation of second-order optimization. Newton's method sets the gradient of the quadratic approximation to zero: $\nabla f(\mathbf{x}) + H_f(\mathbf{x})\mathbf{h} = \mathbf{0}$, giving the update $\mathbf{h} = -H_f^{-1}\nabla f$. This converges quadratically near a minimum (vs. linearly for gradient descent) but requires computing and inverting the $n \times n$ Hessian --- infeasible for neural networks with millions of parameters. Approximate methods (L-BFGS, natural gradient, K-FAC) use structured approximations to $H$.

---

## 6. Applications in Machine Learning

### 6.1 Newton's Method via Quadratic Approximation

At the current parameters $\theta$, approximate the loss:

$$\mathcal{L}(\theta + \Delta\theta) \approx \mathcal{L}(\theta) + \nabla\mathcal{L}^T \Delta\theta + \frac{1}{2}\Delta\theta^T H\, \Delta\theta$$

Minimizing over $\Delta\theta$ (set gradient of RHS to zero):

$$\Delta\theta^* = -H^{-1}\nabla\mathcal{L}$$

| Method | Update Rule | Taylor Order | Cost per Step | Convergence |
|--------|-------------|-------------|---------------|-------------|
| Gradient descent | $-\eta \nabla \mathcal{L}$ | 1st | $O(n)$ | Linear |
| Newton's method | $-H^{-1}\nabla\mathcal{L}$ | 2nd | $O(n^3)$ | Quadratic |
| L-BFGS | $-B_k^{-1}\nabla\mathcal{L}$ | ~2nd | $O(mn)$ | Super-linear |

**Example 3.6.8:** Use one iteration of Newton's method to find $\sqrt{2}$ (solve $f(x) = x^2 - 2 = 0$).

Newton's update: $x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)} = x_n - \frac{x_n^2 - 2}{2x_n}$.

Starting from $x_0 = 1$:

$$x_1 = 1 - \frac{1 - 2}{2} = 1 + \frac{1}{2} = 1.5$$

$$x_2 = 1.5 - \frac{2.25 - 2}{3} = 1.5 - \frac{0.25}{3} = 1.41\overline{6}$$

After just 2 iterations: $x_2 = 1.4167$ vs. $\sqrt{2} = 1.41421\ldots$ (error $\approx 0.002$). This illustrates quadratic convergence --- the number of correct digits roughly doubles each step.

### 6.2 Log-Sum-Exp and Numerical Stability

The log-sum-exp function $\text{LSE}(\mathbf{z}) = \log\sum_i e^{z_i}$ appears in softmax cross-entropy. For numerical stability, use the identity:

$$\text{LSE}(\mathbf{z}) = z_{\max} + \log\sum_i e^{z_i - z_{\max}}$$

**Why this works (Taylor perspective):** When one $z_i$ dominates (say $z_1 \gg z_j$ for $j \neq 1$), the terms $e^{z_j - z_1}$ are small, and $\log(1 + \epsilon) \approx \epsilon$ (first-order Taylor of $\ln(1+x)$) gives:

$$\text{LSE}(\mathbf{z}) \approx z_1 + \sum_{j \neq 1} e^{z_j - z_1}$$

Without the shift by $z_{\max}$, computing $e^{z_i}$ directly overflows for large $z_i$. The Taylor expansion reveals that the correction terms are exponentially small.

### 6.3 Activation Function Approximations

The sigmoid function $\sigma(x) = 1/(1 + e^{-x})$ has the Taylor expansion around $x = 0$:

$$\sigma(x) = \frac{1}{2} + \frac{x}{4} - \frac{x^3}{48} + O(x^5)$$

**Key observations:**

- Near $x = 0$: $\sigma(x) \approx \frac{1}{2} + \frac{x}{4}$ --- approximately linear
- This means sigmoid-activated networks behave like linear models for small pre-activations
- The vanishing gradient problem: $\sigma'(x) = \sigma(x)(1 - \sigma(x)) \leq 1/4$, and for large $|x|$, $\sigma'(x) \approx 0$ exponentially fast (from the $e^{-|x|}$ terms)

Similarly, $\tanh(x) = x - x^3/3 + O(x^5)$, which is linear near zero with slope $1$ (better than sigmoid's $1/4$).

### 6.4 The ELBO and Jensen's Inequality

In variational inference, we need $\log p(\mathbf{x}) = \log \int p(\mathbf{x}, \mathbf{z}) \, d\mathbf{z}$, which is intractable. The ELBO (Evidence Lower Bound) uses Jensen's inequality, which can be understood through the lens of Taylor expansions.

Since $\log$ is concave, its Taylor expansion satisfies $\log(x) \leq \log(a) + \frac{1}{a}(x - a)$ (the tangent line lies above the curve). Applied to the expectation:

$$\log \mathbb{E}_q\!\left[\frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z})}\right] \geq \mathbb{E}_q\!\left[\log \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z})}\right] = \text{ELBO}$$

The gap between $\log p(\mathbf{x})$ and the ELBO is exactly $D_{\text{KL}}(q \| p(\mathbf{z}|\mathbf{x}))$, which is the second-order Taylor remainder of $\log$ around $q = p(\mathbf{z}|\mathbf{x})$.

### 6.5 Finite Difference Gradient Checking

Taylor expansions give the error analysis for numerical gradient approximation:

**Forward difference:** $f'(x) \approx \frac{f(x+h) - f(x)}{h}$

$$f(x+h) = f(x) + f'(x)h + \frac{f''(x)}{2}h^2 + O(h^3)$$

$$\implies \frac{f(x+h) - f(x)}{h} = f'(x) + O(h) \qquad \text{(first-order accurate)}$$

**Central difference:** $f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$

$$f(x+h) = f(x) + f'(x)h + \frac{f''(x)}{2}h^2 + \frac{f'''(x)}{6}h^3 + O(h^4)$$

$$f(x-h) = f(x) - f'(x)h + \frac{f''(x)}{2}h^2 - \frac{f'''(x)}{6}h^3 + O(h^4)$$

$$\implies \frac{f(x+h) - f(x-h)}{2h} = f'(x) + O(h^2) \qquad \text{(second-order accurate)}$$

The even-order terms cancel in the central difference, giving an extra order of accuracy for free.

| Method | Formula | Error | Typical $h$ |
|--------|---------|-------|-------------|
| Forward | $\frac{f(x+h) - f(x)}{h}$ | $O(h)$ | $10^{-5}$ |
| Central | $\frac{f(x+h) - f(x-h)}{2h}$ | $O(h^2)$ | $10^{-4}$ |

**Example 3.6.9:** Compare forward and central differences for $f(x) = e^x$ at $x = 1$ with $h = 0.01$.

True derivative: $f'(1) = e^1 = 2.71828\ldots$

Forward: $\frac{e^{1.01} - e^1}{0.01} = \frac{2.74560 - 2.71828}{0.01} = 2.73192$, error $= 0.01364$ ($O(h)$).

Central: $\frac{e^{1.01} - e^{0.99}}{0.02} = \frac{2.74560 - 2.69123}{0.02} = 2.71828$, error $= 0.00005$ ($O(h^2)$).

The central difference is $\sim 270 \times$ more accurate, demonstrating the $O(h)$ vs. $O(h^2)$ difference.

*ML connection:* Central differences are the standard way to verify backpropagation implementations. With $h \approx 10^{-4}$ to $10^{-5}$, the relative error between the analytic gradient and the numerical gradient should be $< 10^{-5}$ for a correct implementation. If using forward differences, you need smaller $h$ for the same accuracy, increasing floating-point cancellation errors.

---

## 7. Key Manipulations with Taylor Series

**Theorem 3.6.6 (Differentiation and Integration of Power Series).** Within the radius of convergence, a power series can be differentiated and integrated term-by-term:

$$\text{If } f(x) = \sum_{k=0}^\infty a_k x^k, \text{ then } f'(x) = \sum_{k=1}^\infty k\, a_k x^{k-1} \quad \text{and} \quad \int_0^x f(t)\,dt = \sum_{k=0}^\infty \frac{a_k}{k+1} x^{k+1}$$

with the same radius of convergence.

**Example 3.6.10:** Derive the Maclaurin series for $\cos x$ by differentiating the $\sin x$ series term-by-term.

Starting from $\sin x = x - \frac{x^3}{6} + \frac{x^5}{120} - \frac{x^7}{5040} + \cdots$, differentiate each term:

$$\cos x = \frac{d}{dx}\sin x = 1 - \frac{3x^2}{6} + \frac{5x^4}{120} - \frac{7x^6}{5040} + \cdots = 1 - \frac{x^2}{2} + \frac{x^4}{24} - \frac{x^6}{720} + \cdots$$

This matches the known series $\cos x = \sum_{k=0}^{\infty} \frac{(-1)^k x^{2k}}{(2k)!}$. $\checkmark$

**Example (deriving the $\ln(1+x)$ series).** Start with the geometric series:

$$\frac{1}{1+x} = \sum_{k=0}^\infty (-x)^k = 1 - x + x^2 - x^3 + \cdots \qquad |x| < 1$$

Integrate both sides from $0$ to $x$:

$$\ln(1+x) = x - \frac{x^2}{2} + \frac{x^3}{3} - \frac{x^4}{4} + \cdots$$

*ML connection:* Term-by-term differentiation of power series is exactly what happens symbolically in automatic differentiation when applied to functions defined by convergent series. The radius of convergence determines where the computed gradients are valid.

---

## Exercises

**★ Basic**

1. Compute the Maclaurin polynomial $T_3(x)$ for $f(x) = e^{-x^2}$. Use this to approximate $\int_0^1 e^{-x^2} dx$.

2. Using the Taylor expansion of $\sigma(x) = 1/(1+e^{-x})$, show that $\sigma(x) \approx 1/2 + x/4$ near $x = 0$. What is the maximum error of this linear approximation on $[-1, 1]$?

3. Verify the central difference formula: compute $f'(1)$ for $f(x) = x^3$ using $h = 0.1$ with both forward and central differences. Compare errors.

4. Write the second-order Taylor expansion of $f(x,y) = e^{x+y}$ around $(0, 0)$.

**★★ Intermediate**

5. Using Taylor's theorem, prove that Newton's method $x_{n+1} = x_n - f(x_n)/f'(x_n)$ has quadratic convergence: if $x^*$ is a simple root of $f$, then $|x_{n+1} - x^*| \leq C|x_n - x^*|^2$ for some constant $C$.

6. The softplus function $\text{sp}(x) = \ln(1 + e^x)$ is a smooth approximation to ReLU. Find its Maclaurin series up to $O(x^4)$ and show that $\text{sp}(x) \approx \ln 2 + x/2 + x^2/8$ for small $x$. At what value of $x$ does this become a poor approximation?

7. Given $f(\theta) = \frac{1}{2}\theta^T A \theta - b^T\theta$ for symmetric positive definite $A$, show that the multivariable Taylor expansion is *exact* at second order (i.e., $R_2 = 0$). Why does this mean Newton's method converges in one step for quadratic objectives?

8. Derive the error bound for finite difference gradient checking: if $f$ has bounded third derivative $|f'''| \leq M$, show that the central difference error satisfies $|f'(x) - \frac{f(x+h)-f(x-h)}{2h}| \leq \frac{M}{6}h^2$.

**★★★ Challenging**

9. **(Loss landscape analysis)** For a loss function $\mathcal{L}(\theta)$ with Hessian eigenvalues $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n > 0$ at a minimum $\theta^*$, use the second-order Taylor expansion to show that gradient descent with learning rate $\eta < 2/\lambda_1$ converges, and that the convergence rate in the worst eigendirection is $|1 - \eta\lambda|$ (maximized at $\lambda_1$ or $\lambda_n$). Derive the optimal learning rate $\eta^* = 2/(\lambda_1 + \lambda_n)$.

10. **(Convergence radius in practice)** The function $f(x) = 1/(1+25x^2)$ (Runge's function) has singularities at $x = \pm i/5$ in the complex plane. Explain why its Maclaurin series has radius of convergence $R = 1/5$ even though $f$ is infinitely smooth on all of $\mathbb{R}$. What does this imply about polynomial approximation of smooth activation functions?

11. **(Quantum connection)** The matrix exponential $e^{-iHt} = \sum_{k=0}^\infty \frac{(-iHt)^k}{k!}$ governs time evolution in quantum mechanics. Show that if $H$ is Hermitian with eigenvalues $\{E_j\}$, then $e^{-iHt}$ is unitary with eigenvalues $\{e^{-iE_j t}\}$. The Suzuki-Trotter decomposition $e^{(A+B)t} \approx (e^{At/n}e^{Bt/n})^n$ has error $O(t^2/n)$ --- derive this from the Baker-Campbell-Hausdorff formula (a Taylor expansion for matrix exponentials).

---

## Related Topics

- [Differentiation](differentiation.md) --- derivatives that Taylor series are built from
- [Integration](integration.md) --- integral form of the remainder, term-by-term integration
- [Multivariable Calculus](multivariable-calculus.md) --- gradient and Hessian in the multivariable expansion
- [Vector Calculus](vector-calculus.md) --- Taylor-like expansions in vector fields
- [Limits & Continuity](limits-and-continuity.md) --- convergence, radius of convergence
- [Optimization](../optimization/index.md) --- Newton's method, second-order methods
- [Eigenvalues](../linear-algebra/eigenvalues.md) --- Hessian eigenvalues control Taylor remainder

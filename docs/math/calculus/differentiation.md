# Differentiation

Differentiation is the mathematical engine behind all of modern deep learning. Every time a neural network learns, it computes derivatives of a loss function with respect to millions of parameters — the chain rule, applied systematically through a computational graph, is exactly the backpropagation algorithm. Understanding derivatives is not optional for ML: gradient descent requires the gradient, Newton's method requires the Hessian, and diagnosing training failures (vanishing gradients, exploding gradients, saddle points) requires understanding where derivatives are large, small, zero, or undefined.

---

## Prerequisites

- [Foundations](../foundations/index.md) — functions, notation, proof techniques
- [Limits & Continuity](limits-and-continuity.md) — $\varepsilon$-$\delta$ limits, continuity, sequences

---

## 1. The Derivative

**Definition 3.2.1 (Derivative).** The *derivative* of $f: \mathbb{R} \to \mathbb{R}$ at a point $a$ is:

$$f'(a) = \lim_{h \to 0} \frac{f(a + h) - f(a)}{h}$$

provided the limit exists. If $f'(a)$ exists, we say $f$ is *differentiable* at $a$.

**Example 3.2.1 (Derivative from the definition).** Compute $f'(x)$ for $f(x) = x^2$.

$$f'(x) = \lim_{h \to 0} \frac{(x+h)^2 - x^2}{h} = \lim_{h \to 0} \frac{x^2 + 2xh + h^2 - x^2}{h} = \lim_{h \to 0} \frac{2xh + h^2}{h} = \lim_{h \to 0}(2x + h) = 2x$$

At $x = 3$: $f'(3) = 6$, so the tangent line to $y = x^2$ at $(3, 9)$ has slope $6$.

Equivalent formulation using $x \to a$:

$$f'(a) = \lim_{x \to a} \frac{f(x) - f(a)}{x - a}$$

**Definition 3.2.2 (Notation).** Common notations for the derivative of $y = f(x)$:

| Notation | Name | Usage |
|----------|------|-------|
| $f'(x)$ | Lagrange | Functions, general calculus |
| $\frac{dy}{dx}$ | Leibniz | Chain rule, physics, differential equations |
| $\frac{df}{dx}$ | Leibniz (variant) | Explicit function naming |
| $\dot{y}$ | Newton | Derivatives with respect to time |
| $Df(x)$ | Operator | Functional analysis, abstract settings |

### 1.1 Geometric Interpretation

The derivative $f'(a)$ is the slope of the tangent line to $y = f(x)$ at the point $(a, f(a))$.

```
TANGENT LINE AS LIMIT OF SECANT LINES

  y
  │          · f(x)
  │        ·╱
  │      ·╱         Secant line: slope = [f(a+h) - f(a)] / h
  │    ·╱·                                  │
  │  ·╱ · ·                                 │  As h → 0, secant → tangent
  │·╱·     ·                                ▼
  ╱·────────·──── Tangent line: slope = f'(a)
  │         a  a+h
  └──────────────── x

  The tangent line equation: y = f(a) + f'(a)(x - a)
  This is the best LINEAR approximation to f near a.
```

*ML connection:* The tangent line approximation $f(a + h) \approx f(a) + f'(a)h$ is the foundation of gradient descent. When we update a parameter $\theta \leftarrow \theta - \eta \nabla \mathcal{L}(\theta)$, we are using the linear approximation to predict which direction decreases the loss.

### 1.2 Differentiability Implies Continuity

**Theorem 3.2.1.** If $f$ is differentiable at $a$, then $f$ is continuous at $a$.

*Proof.* $\lim_{x \to a} [f(x) - f(a)] = \lim_{x \to a} \frac{f(x) - f(a)}{x - a} \cdot (x - a) = f'(a) \cdot 0 = 0$. So $\lim_{x \to a} f(x) = f(a)$. $\square$

The converse is false: $f(x) = |x|$ is continuous at $0$ but not differentiable there (the left and right limits of the difference quotient differ).

*ML connection:* The ReLU function $\text{ReLU}(x) = \max(0, x)$ is continuous everywhere but not differentiable at $x = 0$. This does not prevent its use in neural networks — we use *subgradients* at the kink (see Section 7).

---

## 2. Differentiation Rules

**Theorem 3.2.2 (Basic Rules).** Let $f$ and $g$ be differentiable at $x$, and let $c \in \mathbb{R}$.

| Rule | Formula | ML Context |
|------|---------|------------|
| Constant | $(c)' = 0$ | Bias terms held fixed in some layers |
| Constant multiple | $(cf)' = cf'$ | Scaling the learning rate scales the gradient |
| Sum | $(f + g)' = f' + g'$ | Gradients of summed losses add linearly |
| Product | $(fg)' = f'g + fg'$ | Weight decay regularization term |
| Quotient | $\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}$ | Normalizing by batch statistics |
| Power | $(x^n)' = nx^{n-1}$ | Polynomial features, polynomial activations |

**Example 3.2.2 (Power rule).** Differentiate $f(x) = x^5$ and $g(x) = x^{-2}$.

$$f'(x) = 5x^4 \qquad\qquad g'(x) = -2x^{-3} = -\frac{2}{x^3}$$

At $x = 2$: $f'(2) = 5 \cdot 16 = 80$. At $x = 3$: $g'(3) = -2/27$.

**Example 3.2.3 (Product rule).** Differentiate $h(x) = x^2 \sin(x)$.

$$h'(x) = \underbrace{(x^2)'}_{2x}\sin(x) + x^2\underbrace{(\sin x)'}_{cos x} = 2x\sin(x) + x^2\cos(x)$$

At $x = \pi$: $h'(\pi) = 2\pi\sin(\pi) + \pi^2\cos(\pi) = 0 + \pi^2(-1) = -\pi^2 \approx -9.87$.

**Example 3.2.4 (Quotient rule).** Differentiate $q(x) = \dfrac{\sin(x)}{x^2}$.

$$q'(x) = \frac{(\sin x)' \cdot x^2 - \sin(x) \cdot (x^2)'}{(x^2)^2} = \frac{x^2\cos(x) - 2x\sin(x)}{x^4} = \frac{x\cos(x) - 2\sin(x)}{x^3}$$

At $x = \pi$: $q'(\pi) = \dfrac{\pi(-1) - 2(0)}{\pi^3} = \dfrac{-1}{\pi^2} \approx -0.101$.

*Proof of the Product Rule.*

$$\begin{aligned}
(fg)'(x) &= \lim_{h \to 0} \frac{f(x+h)g(x+h) - f(x)g(x)}{h} \\
&= \lim_{h \to 0} \frac{f(x+h)g(x+h) - f(x)g(x+h) + f(x)g(x+h) - f(x)g(x)}{h} \\
&= \lim_{h \to 0} \left[\frac{f(x+h) - f(x)}{h} \cdot g(x+h) + f(x) \cdot \frac{g(x+h) - g(x)}{h}\right] \\
&= f'(x)g(x) + f(x)g'(x)
\end{aligned}$$

using continuity of $g$ (since $g$ is differentiable, $g(x+h) \to g(x)$). $\square$

---

## 3. The Chain Rule

The chain rule is the single most important differentiation rule for machine learning. It is, in a precise sense, the backpropagation algorithm.

**Theorem 3.2.3 (Chain Rule).** If $g$ is differentiable at $x$ and $f$ is differentiable at $g(x)$, then the composition $f \circ g$ is differentiable at $x$ and:

$$(f \circ g)'(x) = f'(g(x)) \cdot g'(x)$$

In Leibniz notation, if $y = f(u)$ and $u = g(x)$:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

*Proof sketch.* Define $\phi(k) = \frac{f(g(x) + k) - f(g(x))}{k}$ for $k \neq 0$ and $\phi(0) = f'(g(x))$. Then $\phi$ is continuous at $0$ (by differentiability of $f$), and $f(g(x+h)) - f(g(x)) = \phi(g(x+h) - g(x)) \cdot (g(x+h) - g(x))$. Dividing by $h$:

$$\frac{f(g(x+h)) - f(g(x))}{h} = \phi(g(x+h) - g(x)) \cdot \frac{g(x+h) - g(x)}{h}$$

As $h \to 0$, $g(x+h) - g(x) \to 0$ (by continuity), so $\phi(\cdot) \to f'(g(x))$ and $\frac{g(x+h)-g(x)}{h} \to g'(x)$. $\square$

**Example 3.2.5 (Chain rule).** Differentiate $f(x) = \sin(x^2)$.

Let $u = x^2$ (inner) and $y = \sin(u)$ (outer). Then:

$$f'(x) = \cos(u) \cdot 2x = 2x\cos(x^2)$$

At $x = \sqrt{\pi}$: $f'(\sqrt{\pi}) = 2\sqrt{\pi}\cos(\pi) = -2\sqrt{\pi} \approx -3.54$.

**Example 3.2.6 (Chain rule).** Differentiate $f(x) = e^{3x}$.

Let $u = 3x$, $y = e^u$. Then:

$$f'(x) = e^{u} \cdot 3 = 3e^{3x}$$

At $x = 0$: $f'(0) = 3e^0 = 3$. At $x = 1$: $f'(1) = 3e^3 \approx 60.26$.

**Example 3.2.7 (Chain rule).** Differentiate $f(x) = \ln(\cos(x))$.

Let $u = \cos(x)$, $y = \ln(u)$. Then:

$$f'(x) = \frac{1}{\cos(x)} \cdot (-\sin(x)) = -\frac{\sin(x)}{\cos(x)} = -\tan(x)$$

At $x = \pi/4$: $f'(\pi/4) = -\tan(\pi/4) = -1$.

### 3.1 Chain Rule for Multiple Compositions

For $y = f_1(f_2(\cdots f_n(x)\cdots))$:

$$\frac{dy}{dx} = f_1'(f_2(\cdots f_n(x)\cdots)) \cdot f_2'(f_3(\cdots f_n(x)\cdots)) \cdot \;\cdots\; \cdot f_n'(x)$$

This is a product of *local derivatives*, each evaluated at the appropriate intermediate value.

**Example 3.2.14 (Nested chain rule).** Differentiate $f(x) = e^{\sin(x^2)}$.

Identify the composition: $f_1(u) = e^u$, $f_2(v) = \sin(v)$, $f_3(x) = x^2$. Then:

$$f'(x) = e^{\sin(x^2)} \cdot \cos(x^2) \cdot 2x$$

At $x = 0$: $f'(0) = e^{\sin(0)} \cdot \cos(0) \cdot 0 = e^0 \cdot 1 \cdot 0 = 0$.

### 3.2 Chain Rule as Backpropagation

Consider a simple neural network with one hidden layer:

$$\hat{y} = \sigma(w_2 \cdot \sigma(w_1 x + b_1) + b_2)$$

The loss is $\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2$. Let us trace the computation:

```
COMPUTATIONAL GRAPH (Forward Pass)

  x ──→ [×w₁+b₁] ──→ z₁ ──→ [σ] ──→ a₁ ──→ [×w₂+b₂] ──→ z₂ ──→ [σ] ──→ ŷ ──→ [L]
                                                                                    │
  Each arrow is a function. The chain rule multiplies                                ▼
  the local derivatives along any path from L back to                               L
  a parameter.

BACKWARD PASS (Chain Rule Applied)

  ∂L     ∂L   ∂ŷ   ∂z₂   ∂a₁   ∂z₁
  ── = ── · ── · ── · ── · ──
  ∂w₁   ∂ŷ   ∂z₂   ∂a₁   ∂z₁   ∂w₁
       └──┘ └──┘ └──┘ └──┘ └──┘
       each factor is a LOCAL derivative
       evaluated at the value from the forward pass
```

*ML connection:* Backpropagation is nothing more than the chain rule applied to a computational graph, computed efficiently by caching intermediate values (the forward pass) and then multiplying local derivatives in reverse order (the backward pass). The time complexity is $O(n)$ where $n$ is the number of operations — the same as the forward pass. This efficiency is what makes training deep networks with millions of parameters feasible.

### 3.3 Implicit Differentiation

When a relationship between $x$ and $y$ is defined implicitly (not solved for $y$), we differentiate both sides with respect to $x$, treating $y$ as a function of $x$ and applying the chain rule.

**Example 3.2.13 (Implicit differentiation).** Find $\dfrac{dy}{dx}$ for the circle $x^2 + y^2 = 25$.

Differentiate both sides with respect to $x$:

$$2x + 2y\frac{dy}{dx} = 0$$

Solve for $\dfrac{dy}{dx}$:

$$\frac{dy}{dx} = -\frac{x}{y}$$

At the point $(3, 4)$ on the circle: $\dfrac{dy}{dx} = -\dfrac{3}{4}$. The tangent line is $y - 4 = -\frac{3}{4}(x - 3)$.

*ML connection:* Implicit differentiation underlies implicit layers in deep learning, where the output $y$ is defined as the solution to an equation $F(x, y) = 0$ rather than by an explicit formula. The implicit function theorem guarantees $\frac{dy}{dx} = -\frac{\partial F/\partial x}{\partial F/\partial y}$, enabling gradient computation through equilibrium models and deep equilibrium networks (DEQs).

---

## 4. Higher-Order Derivatives

**Definition 3.2.3 (Higher-Order Derivatives).** The $n$-th derivative of $f$ is defined recursively:

$$f^{(n)}(x) = \frac{d}{dx}\left[f^{(n-1)}(x)\right]$$

with $f^{(0)} = f$, $f^{(1)} = f'$, $f^{(2)} = f''$.

| Order | Notation | Meaning | ML Context |
|-------|----------|---------|------------|
| 1st | $f'(x)$ | Rate of change, slope | Gradient — direction of steepest descent |
| 2nd | $f''(x)$ | Curvature, concavity | Hessian — curvature of loss landscape |
| $n$-th | $f^{(n)}(x)$ | Higher-order variation | Taylor expansion, higher-order optimizers |

**Example 3.2.8 (Higher-order derivatives).** Compute $f'$, $f''$, and $f'''$ for $f(x) = x^4 - 3x^2 + 2$.

$$f'(x) = 4x^3 - 6x$$

$$f''(x) = 12x^2 - 6$$

$$f'''(x) = 24x$$

At $x = 1$: $f'(1) = -2$, $f''(1) = 6 > 0$ (concave up), $f'''(1) = 24$.

**Theorem 3.2.4 (Second Derivative Test).** If $f'(c) = 0$ (critical point), then:

- $f''(c) > 0 \implies c$ is a local minimum
- $f''(c) < 0 \implies c$ is a local maximum
- $f''(c) = 0 \implies$ test is inconclusive

**Example 3.2.9 (Critical points and classification).** Find and classify all critical points of $f(x) = x^3 - 3x + 1$.

$$f'(x) = 3x^2 - 3 = 3(x^2 - 1) = 0 \implies x = \pm 1$$

$$f''(x) = 6x$$

- At $x = -1$: $f''(-1) = -6 < 0 \implies$ **local maximum**, $f(-1) = 3$
- At $x = 1$: $f''(1) = 6 > 0 \implies$ **local minimum**, $f(1) = -1$

*ML connection:* In optimization, the second derivative tells us the curvature. Newton's method uses the update $\theta \leftarrow \theta - \frac{f'(\theta)}{f''(\theta)}$, which accounts for curvature and converges quadratically near a minimum. In high dimensions, this generalizes to $\theta \leftarrow \theta - H^{-1}\nabla\mathcal{L}$, where $H$ is the Hessian matrix (see [Multivariable Calculus](multivariable-calculus.md)).

---

## 5. The Mean Value Theorem

**Theorem 3.2.5 (Rolle's Theorem).** If $f$ is continuous on $[a, b]$, differentiable on $(a, b)$, and $f(a) = f(b)$, then there exists $c \in (a, b)$ such that $f'(c) = 0$.

*Proof.* By the extreme value theorem, $f$ attains its maximum and minimum on $[a, b]$. If both occur at the endpoints, then $f$ is constant and $f'(c) = 0$ for all $c \in (a, b)$. Otherwise, an extremum occurs at some interior point $c$. At an interior extremum, $f'(c) = 0$ (the left and right difference quotients have opposite signs, so both must be zero for the limit to exist). $\square$

**Theorem 3.2.6 (Mean Value Theorem).** If $f$ is continuous on $[a, b]$ and differentiable on $(a, b)$, then there exists $c \in (a, b)$ such that:

$$f'(c) = \frac{f(b) - f(a)}{b - a}$$

*Proof.* Apply Rolle's theorem to $g(x) = f(x) - f(a) - \frac{f(b) - f(a)}{b - a}(x - a)$. We have $g(a) = g(b) = 0$, so there exists $c$ with $g'(c) = 0$, giving $f'(c) = \frac{f(b) - f(a)}{b - a}$. $\square$

**Example 3.2.10 (MVT verification).** Verify the Mean Value Theorem for $f(x) = x^2$ on $[1, 3]$.

Secant slope: $\dfrac{f(3) - f(1)}{3 - 1} = \dfrac{9 - 1}{2} = 4$

We need $f'(c) = 2c = 4$, so $c = 2$. Since $2 \in (1, 3)$, the MVT is verified.

Geometrically: the tangent line to $y = x^2$ at $x = 2$ is parallel to the secant line from $(1, 1)$ to $(3, 9)$.

```
MEAN VALUE THEOREM — GEOMETRIC PICTURE

  y
  │       · f(x)
  │     ·╱ ·
  │   · ╱    ·  ← tangent at c has the same slope
  │  ·╱·      ·    as the secant from a to b
  │·╱  ·       ·
  ╱ ·   c       ·
  │a─────────────b── x
  │
  │  Secant slope = [f(b) - f(a)] / (b - a)
  │  Tangent slope at c = f'(c)
  │  MVT guarantees they are equal for some c ∈ (a,b)
```

*ML connection:* The MVT is the theoretical backbone of Lipschitz continuity. A function $f$ is $L$-Lipschitz if $|f(x) - f(y)| \leq L|x - y|$. By the MVT, this holds if and only if $|f'(x)| \leq L$ everywhere. Lipschitz constraints on discriminator networks are central to Wasserstein GANs (spectral normalization enforces $L = 1$).

---

## 6. L'Hopital's Rule

**Theorem 3.2.7 (L'Hopital's Rule).** If $\lim_{x \to a} f(x) = \lim_{x \to a} g(x) = 0$ (or both $\to \pm\infty$), and $g'(x) \neq 0$ near $a$, then:

$$\lim_{x \to a} \frac{f(x)}{g(x)} = \lim_{x \to a} \frac{f'(x)}{g'(x)}$$

provided the right-hand limit exists (or is $\pm\infty$).

**Example 3.2.11 (L'Hopital's rule — 0/0 form).** Evaluate $\displaystyle\lim_{x \to 0} \frac{\sin(x)}{x}$.

Both $\sin(0) = 0$ and $x = 0$, so this is a $0/0$ form. Applying L'Hopital's rule:

$$\lim_{x \to 0} \frac{\sin(x)}{x} \overset{L'H}{=} \lim_{x \to 0} \frac{\cos(x)}{1} = \cos(0) = 1$$

**Example 3.2.12 (L'Hopital's rule — repeated application).** Evaluate $\displaystyle\lim_{x \to 0} \frac{e^x - 1 - x}{x^2}$.

Both numerator and denominator $\to 0$, so apply L'Hopital's once:

$$\lim_{x \to 0} \frac{e^x - 1}{2x} \quad \text{(still } 0/0\text{)}$$

Apply L'Hopital's a second time:

$$\lim_{x \to 0} \frac{e^x}{2} = \frac{1}{2}$$

**Example (Cross-entropy at boundary).**

$$\lim_{p \to 0^+} p \log p = \lim_{p \to 0^+} \frac{\log p}{1/p} \overset{L'H}{=} \lim_{p \to 0^+} \frac{1/p}{-1/p^2} = \lim_{p \to 0^+} (-p) = 0$$

*ML connection:* This shows that the cross-entropy loss $H(y, \hat{y}) = -\sum y_i \log \hat{y}_i$ is well-defined in the limit: the convention $0 \log 0 = 0$ is justified by L'Hopital's rule. This matters when computing KL divergence for sparse distributions.

---

## 7. Common Derivatives Table

| Function $f(x)$ | Derivative $f'(x)$ | ML Context |
|------------------|---------------------|------------|
| $x^n$ | $nx^{n-1}$ | Polynomial regression, feature engineering |
| $e^x$ | $e^x$ | Exponential growth/decay, softmax numerics |
| $\ln x$ | $\frac{1}{x}$ | Log-likelihood, cross-entropy loss |
| $a^x$ | $a^x \ln a$ | Learning rate schedules |
| $\sin x$ | $\cos x$ | Positional encodings (Transformers) |
| $\cos x$ | $-\sin x$ | Fourier features |
| $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1 - \sigma(x))$ | Sigmoid activation, logistic regression |
| $\tanh(x)$ | $1 - \tanh^2(x)$ | RNN activation, bounded output |
| $\text{ReLU}(x) = \max(0, x)$ | $\begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases}$ | Most common activation in deep learning |
| $\text{softplus}(x) = \ln(1 + e^x)$ | $\sigma(x)$ | Smooth approximation to ReLU |

### 7.1 The Sigmoid Derivative

**Definition 3.2.4 (Sigmoid Function).** $\sigma(x) = \frac{1}{1 + e^{-x}}$

**Theorem 3.2.8.** $\sigma'(x) = \sigma(x)(1 - \sigma(x))$.

*Proof.* Using the quotient rule:

$$\sigma'(x) = \frac{e^{-x}}{(1 + e^{-x})^2} = \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}} = \sigma(x)(1 - \sigma(x))$$

$\square$

```
SIGMOID AND ITS DERIVATIVE

  σ(x)                          σ'(x) = σ(x)(1 - σ(x))
  1 ─────────────·····                    ·····
  │           ···             0.25 ─────··     ··─────
  │         ··                     │   ·         ·
  │       ·                        │  ·           ·
  0.5 ──·─────────             │ ·             ·
  │   ·                        │·               ·
  │  ·                         ·─────────────────·─
  │··                          │
  0 ····─────────── x          0 ──────────────────── x
    -6  -3   0   3   6           -6  -3   0   3   6

  Maximum of σ'(x) is 0.25 at x = 0.
  At x = ±6, σ'(x) ≈ 0.002 — nearly zero!
```

*ML connection:* **Sigmoid saturation** is a major cause of vanishing gradients. When $|x|$ is large, $\sigma'(x) \approx 0$, which means the gradient signal essentially dies. In a deep network with $L$ sigmoid layers, the gradient is multiplied by $\sigma'$ at each layer: if each factor is at most $0.25$, the gradient shrinks as $(0.25)^L$. With $L = 10$ layers, this is $\approx 10^{-6}$. This is why ReLU replaced sigmoid as the default activation — ReLU has gradient exactly $1$ for positive inputs.

### 7.2 ReLU and Subgradients

The ReLU function $\text{ReLU}(x) = \max(0, x)$ is not differentiable at $x = 0$. In practice, we define:

$$\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$$

This is a *subgradient* — a generalization of the derivative for convex but non-smooth functions.

**Definition 3.2.5 (Subgradient).** A vector $g$ is a *subgradient* of a convex function $f$ at $x$ if:

$$f(y) \geq f(x) + g(y - x) \quad \text{for all } y$$

At $x = 0$ for ReLU, any value $g \in [0, 1]$ satisfies this condition. Frameworks like PyTorch use $g = 0$ by convention.

*ML connection:* Despite non-differentiability at a single point, ReLU works perfectly in practice because: (1) the set of inputs exactly equal to zero has measure zero during training, (2) subgradient methods still converge for convex problems, and (3) even for non-convex neural networks, the empirical success is overwhelming. The key advantage is that $\text{ReLU}'(x) = 1$ for $x > 0$ — the gradient does not shrink, solving the vanishing gradient problem.

### 7.3 The Softmax Derivative

**Definition 3.2.6 (Softmax).** For $\mathbf{z} \in \mathbb{R}^K$, the softmax function is:

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$

**Theorem 3.2.9.** The Jacobian of softmax is:

$$\frac{\partial \text{softmax}(\mathbf{z})_i}{\partial z_j} = \text{softmax}(\mathbf{z})_i \left(\delta_{ij} - \text{softmax}(\mathbf{z})_j\right)$$

where $\delta_{ij}$ is the Kronecker delta.

*ML connection:* The softmax Jacobian has a compact form $\text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T$ where $\mathbf{p} = \text{softmax}(\mathbf{z})$. When combined with cross-entropy loss $\mathcal{L} = -\log p_y$, the gradient simplifies beautifully to $\nabla_{\mathbf{z}} \mathcal{L} = \mathbf{p} - \mathbf{e}_y$, where $\mathbf{e}_y$ is the one-hot target vector. This elegant cancellation is why cross-entropy + softmax is the standard classification setup.

---

## 8. Applications in Machine Learning

### 8.1 Gradient of Loss Functions

For a model with parameters $\theta$ and loss $\mathcal{L}(\theta)$, training requires computing $\nabla_\theta \mathcal{L}$.

**Example (MSE loss for linear regression):**

$$\mathcal{L}(\mathbf{w}) = \frac{1}{2n}\sum_{i=1}^n (y_i - \mathbf{w}^T\mathbf{x}_i)^2$$

$$\frac{\partial \mathcal{L}}{\partial w_j} = -\frac{1}{n}\sum_{i=1}^n (y_i - \mathbf{w}^T\mathbf{x}_i) \cdot x_{ij}$$

Each step: outer derivative of $(\cdot)^2$ times inner derivative of the linear predictor — the chain rule.

### 8.2 Gradient Descent Update

The parameter update $\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t)$ moves in the direction of steepest decrease. The gradient $\nabla \mathcal{L}$ points in the direction of steepest *increase*, so the negative gradient points downhill.

```
GRADIENT DESCENT ON A LOSS SURFACE

  L(θ)
  │╲
  │ ╲        ·
  │  ╲     ·   ·
  │   ╲  ·       ·
  │    ·←─ -η∇L    ·
  │    θ₁             ·
  │      ╲              ·
  │       ·←─ -η∇L       ·
  │       θ₂                ·
  │         ╲                 ·
  │          · θ* (minimum)
  └────────────────────────────── θ

  Step size η (learning rate) controls how far we move.
  Too large: overshoot. Too small: slow convergence.
```

### 8.3 Why Deep Networks Need Careful Derivative Analysis

In a network with $L$ layers, the gradient of the loss with respect to the first layer's weights involves a product of $L$ Jacobian matrices (chain rule):

$$\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \prod_{\ell=2}^{L} \frac{\partial \mathbf{a}_\ell}{\partial \mathbf{a}_{\ell-1}} \cdot \frac{\partial \mathbf{a}_1}{\partial W_1}$$

| Problem | Cause | Derivative Signature | Solution |
|---------|-------|---------------------|----------|
| Vanishing gradient | $\sigma'(x) \ll 1$ repeatedly | $\prod \sigma'(z_\ell) \to 0$ | ReLU, skip connections (ResNets) |
| Exploding gradient | Weight matrices with $\|W_\ell\| > 1$ | $\prod \|W_\ell\| \to \infty$ | Gradient clipping, normalization |
| Dead neurons | ReLU input always negative | $\text{ReLU}'(x) = 0$ permanently | Leaky ReLU, careful initialization |
| Saturation | Sigmoid/tanh in flat regions | $f'(x) \approx 0$ for $|x| \gg 0$ | Batch normalization, gradient-friendly activations |

---

## Exercises

**★ Basic**

1. Using the limit definition, compute $f'(x)$ for $f(x) = x^2 + 3x$.

2. Differentiate $f(x) = (3x^2 + 1)(e^x)$ using the product rule.

3. Apply the chain rule to find $\frac{d}{dx}\sigma(2x + 1)$ where $\sigma$ is the sigmoid function.

4. Verify that $\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)$ using the fact that $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$.

**★★ Intermediate**

5. Show that the derivative of the softplus function $\text{sp}(x) = \ln(1 + e^x)$ is the sigmoid $\sigma(x)$. Explain why softplus is called a "smooth ReLU."

6. For the logistic regression loss $\mathcal{L}(w) = -[y\log\sigma(wx) + (1-y)\log(1-\sigma(wx))]$, compute $\frac{\partial \mathcal{L}}{\partial w}$ and show it simplifies to $(\sigma(wx) - y)x$.

7. Use L'Hopital's rule to evaluate $\lim_{x \to 0} \frac{e^x - 1 - x}{x^2}$. Relate this to the second-order Taylor approximation of $e^x$.

8. A neural network has layers $z_1 = w_1 x$, $a_1 = \text{ReLU}(z_1)$, $z_2 = w_2 a_1$, $a_2 = \sigma(z_2)$, $\mathcal{L} = -\log(a_2)$. Write out $\frac{\partial \mathcal{L}}{\partial w_1}$ as a product of local derivatives using the chain rule. Identify each factor.

**★★★ Challenging**

9. Prove the general Leibniz rule: $(fg)^{(n)} = \sum_{k=0}^{n} \binom{n}{k} f^{(k)} g^{(n-k)}$. (Hint: induction on $n$.)

10. Consider a deep sigmoid network with $L$ layers, all weights equal to $w$, and input $x = 0$ (so all pre-activations are at the origin). Show that $\left|\frac{\partial \mathcal{L}}{\partial w_1}\right| \leq \left(\frac{|w|}{4}\right)^{L-1} \cdot C$ for some constant $C$, and conclude that training becomes exponentially harder as $L$ grows when $|w| < 4$.

11. The Swish activation is $f(x) = x \cdot \sigma(\beta x)$ where $\sigma$ is sigmoid and $\beta$ is a learnable parameter. Compute $f'(x)$ and $\frac{\partial f}{\partial \beta}$. Show that Swish interpolates between a linear function ($\beta \to 0$) and ReLU ($\beta \to \infty$).

12. Prove that for the softmax-cross-entropy composition $\mathcal{L} = -\log(\text{softmax}(\mathbf{z})_y)$, the gradient is $\frac{\partial \mathcal{L}}{\partial z_i} = \text{softmax}(\mathbf{z})_i - \delta_{iy}$. (This requires combining the softmax Jacobian from Theorem 3.2.9 with the derivative of $-\log$.)

---

## Related Topics

- [Limits & Continuity](limits-and-continuity.md) — the $\varepsilon$-$\delta$ foundations underlying the derivative
- [Integration](integration.md) — the fundamental theorem of calculus connects differentiation and integration
- [Multivariable Calculus](multivariable-calculus.md) — partial derivatives, gradient, Jacobian, Hessian
- [Taylor Series](taylor-series.md) — higher-order derivatives and polynomial approximation
- [Optimization](../optimization/index.md) — gradient descent and beyond
- [Linear Algebra: Eigenvalues](../linear-algebra/eigenvalues.md) — Hessian eigenvalues and the loss landscape

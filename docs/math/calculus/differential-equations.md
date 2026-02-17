# Chapter 3.8: Differential Equations

## Prerequisites

- **Required:**
  - Chapter 3.1-3.7: Derivatives, integration, multivariable calculus
  - Chapter 2.4: Eigenvalues and eigenvectors
  - Chapter 2.2: Matrix exponential and linear systems
  - Chapter 1.4: Vector spaces and linear transformations

- **Recommended:**
  - Numerical analysis fundamentals
  - Basic probability theory (for SDEs)
  - Understanding of neural network architectures

## Overview

Differential equations describe how quantities change over time or space. In machine learning, they provide a continuous framework for understanding deep networks (Neural ODEs), generative models (diffusion models), and optimization dynamics (gradient flow).

**Core Concepts:**
- ODEs model temporal evolution of systems
- Systems of ODEs describe coupled dynamics
- Numerical methods connect to discrete optimization
- SDEs incorporate randomness and uncertainty
- PDEs extend to spatial domains

---

## 3.8.1 Ordinary Differential Equations (ODEs)

**Definition 3.8.1 (Ordinary Differential Equation)**
An *ordinary differential equation* of order $n$ is an equation relating a function $y(t)$ and its derivatives:

$$F\left(t, y, \frac{dy}{dt}, \frac{d^2y}{dt^2}, \ldots, \frac{d^ny}{dt^n}\right) = 0$$

A *first-order ODE* has the standard form:

$$\frac{dy}{dt} = f(t, y)$$

where $f: \mathbb{R} \times \mathbb{R} \to \mathbb{R}$ defines the rate of change.

**Definition 3.8.2 (Initial Value Problem)**
An *initial value problem* (IVP) consists of an ODE together with an initial condition:

$$\begin{cases}
\frac{dy}{dt} = f(t, y) \\
y(t_0) = y_0
\end{cases}$$

The goal is to find a function $y(t)$ satisfying both conditions.

**Example 3.8.4:** Solve the IVP: $\frac{dy}{dt} = 3y$, $y(0) = 2$.

This is separable: $\frac{dy}{y} = 3\,dt \implies \ln|y| = 3t + C$.

General solution: $y(t) = Ae^{3t}$.

Apply initial condition: $y(0) = A = 2$, so $y(t) = 2e^{3t}$.

Verify: $y'(t) = 6e^{3t} = 3 \cdot 2e^{3t} = 3y(t)$ and $y(0) = 2$. ✓

**Theorem 3.8.1 (Picard-Lindelöf Existence and Uniqueness)**
Let $f(t, y)$ be continuous in a rectangle $R = \{(t, y) : |t - t_0| \leq a, |y - y_0| \leq b\}$ and satisfy a Lipschitz condition in $y$:

$$|f(t, y_1) - f(t, y_2)| \leq L|y_1 - y_2|$$

for some constant $L > 0$. Then the IVP has a unique solution on $[t_0 - \delta, t_0 + \delta]$ for some $\delta > 0$.

**Geometric Interpretation:**

```
y
│     ╱╱╱╱╱  Solution curves never intersect
│    ╱ ╱ ╱   (uniqueness)
│   ╱ ╱ ╱
│  ╱ ╱ ╱     Each point (t₀, y₀) determines
│ • ╱ ╱      exactly one trajectory
│  ╱ ╱
└────────────> t
```

**ML Connection:** Neural ODE layers must satisfy Lipschitz conditions for well-defined forward passes. The existence theorem guarantees the output exists for any input.

---

## 3.8.2 First-Order ODEs: Solution Methods

### Separable Equations

**Definition 3.8.3 (Separable ODE)**
An ODE is *separable* if it can be written as:

$$\frac{dy}{dt} = g(t)h(y)$$

**Solution Method:** Separate variables and integrate:

$$\int \frac{dy}{h(y)} = \int g(t)\,dt + C$$

**Example 3.8.1:** Exponential growth/decay: $\frac{dy}{dt} = ky$

$$\frac{dy}{y} = k\,dt \implies \ln|y| = kt + C \implies y(t) = y_0 e^{kt}$$

**Example 3.8.3:** Solve $\frac{dy}{dx} = 2xy$.

Separate variables: $\frac{dy}{y} = 2x\,dx$

Integrate both sides:

$$\int \frac{dy}{y} = \int 2x\,dx \implies \ln|y| = x^2 + C$$

Exponentiate: $y(x) = Ae^{x^2}$ where $A = \pm e^C$.

If $y(0) = 3$, then $A = 3$, so $y(x) = 3e^{x^2}$.

Check: $y' = 3 \cdot 2x \cdot e^{x^2} = 2x \cdot 3e^{x^2} = 2xy$. ✓

### Linear First-Order ODEs

**Definition 3.8.4 (Linear First-Order ODE)**
A *linear first-order ODE* has the form:

$$\frac{dy}{dt} + p(t)y = q(t)$$

**Solution Method (Integrating Factor):**

1. Compute integrating factor: $\mu(t) = e^{\int p(t)\,dt}$
2. Multiply both sides by $\mu(t)$
3. Left side becomes: $\frac{d}{dt}[\mu(t)y]$
4. Integrate: $y(t) = \frac{1}{\mu(t)}\left(\int \mu(t)q(t)\,dt + C\right)$

**Example 3.8.2:** $\frac{dy}{dt} + 2y = e^{-t}$

Integrating factor: $\mu(t) = e^{2t}$

$$e^{2t}\frac{dy}{dt} + 2e^{2t}y = e^t \implies \frac{d}{dt}[e^{2t}y] = e^t$$

$$e^{2t}y = e^t + C \implies y(t) = e^{-t} + Ce^{-2t}$$

**Example 3.8.5:** Solve $\frac{dy}{dt} - \frac{y}{t} = t^2$ for $t > 0$.

Here $p(t) = -1/t$ and $q(t) = t^2$.

Integrating factor: $\mu(t) = e^{\int -1/t\,dt} = e^{-\ln t} = 1/t$.

Multiply through: $\frac{1}{t}\frac{dy}{dt} - \frac{y}{t^2} = t$, i.e., $\frac{d}{dt}\!\left[\frac{y}{t}\right] = t$.

Integrate: $\frac{y}{t} = \frac{t^2}{2} + C \implies y(t) = \frac{t^3}{2} + Ct$.

If $y(1) = 3$: $3 = \frac{1}{2} + C$, so $C = \frac{5}{2}$ and $y(t) = \frac{t^3}{2} + \frac{5t}{2}$.

### Second-Order Linear ODEs with Constant Coefficients

**Example 3.8.6:** Solve $y'' + 3y' + 2y = 0$.

Write the characteristic equation: $r^2 + 3r + 2 = 0$.

Factor: $(r + 1)(r + 2) = 0$, so $r_1 = -1$, $r_2 = -2$.

General solution: $y(t) = C_1 e^{-t} + C_2 e^{-2t}$.

If $y(0) = 1$ and $y'(0) = 0$:

- $y(0) = C_1 + C_2 = 1$
- $y'(0) = -C_1 - 2C_2 = 0 \implies C_1 = -2C_2$

Solving: $C_2 = -1$, $C_1 = 2$, so $y(t) = 2e^{-t} - e^{-2t}$.

Both roots are negative, so $y(t) \to 0$ as $t \to \infty$ (stable).

---

## 3.8.3 Systems of ODEs

**Definition 3.8.5 (System of Linear ODEs)**
A *linear system* of first-order ODEs in matrix form:

$$\frac{d\mathbf{y}}{dt} = A\mathbf{y}$$

where $\mathbf{y}(t) \in \mathbb{R}^n$ and $A \in \mathbb{R}^{n \times n}$ is constant.

**Theorem 3.8.2 (Matrix Exponential Solution)**
The solution to the IVP $\frac{d\mathbf{y}}{dt} = A\mathbf{y}$, $\mathbf{y}(0) = \mathbf{y}_0$ is:

$$\mathbf{y}(t) = e^{At}\mathbf{y}_0$$

where the matrix exponential is:

$$e^{At} = I + At + \frac{A^2t^2}{2!} + \frac{A^3t^3}{3!} + \cdots = \sum_{k=0}^{\infty}\frac{A^kt^k}{k!}$$

### Eigenvalue Analysis

**Theorem 3.8.3 (Diagonalizable Case)**
If $A = PDP^{-1}$ where $D = \text{diag}(\lambda_1, \ldots, \lambda_n)$, then:

$$e^{At} = Pe^{Dt}P^{-1} = P\begin{bmatrix}e^{\lambda_1 t} & & \\ & \ddots & \\ & & e^{\lambda_n t}\end{bmatrix}P^{-1}$$

**Example 3.8.7:** Solve the system $\frac{d\mathbf{y}}{dt} = A\mathbf{y}$ where $A = \begin{bmatrix}-1 & 0 \\ 2 & -3\end{bmatrix}$, $\mathbf{y}(0) = \begin{bmatrix}4 \\ 1\end{bmatrix}$.

Find eigenvalues: $\det(A - \lambda I) = (-1-\lambda)(-3-\lambda) = 0$, so $\lambda_1 = -1$, $\lambda_2 = -3$.

Eigenvectors: For $\lambda_1 = -1$: $(A + I)\mathbf{v} = 0 \implies \begin{bmatrix}0 & 0 \\ 2 & -2\end{bmatrix}\mathbf{v} = 0$, so $\mathbf{v}_1 = \begin{bmatrix}1 \\ 1\end{bmatrix}$.

For $\lambda_2 = -3$: $(A + 3I)\mathbf{v} = 0 \implies \begin{bmatrix}2 & 0 \\ 2 & 0\end{bmatrix}\mathbf{v} = 0$, so $\mathbf{v}_2 = \begin{bmatrix}0 \\ 1\end{bmatrix}$.

General solution: $\mathbf{y}(t) = c_1 e^{-t}\begin{bmatrix}1 \\ 1\end{bmatrix} + c_2 e^{-3t}\begin{bmatrix}0 \\ 1\end{bmatrix}$.

Apply $\mathbf{y}(0)$: $c_1 = 4$, $c_1 + c_2 = 1 \implies c_2 = -3$.

$$\mathbf{y}(t) = 4e^{-t}\begin{bmatrix}1 \\ 1\end{bmatrix} - 3e^{-3t}\begin{bmatrix}0 \\ 1\end{bmatrix} = \begin{bmatrix}4e^{-t} \\ 4e^{-t} - 3e^{-3t}\end{bmatrix}$$

Both eigenvalues are negative, so $\mathbf{y}(t) \to \mathbf{0}$ (stable node).

**Stability Analysis:**

```
Phase Portrait Classification by Eigenvalues λ₁, λ₂

λ₁, λ₂ < 0           λ₁, λ₂ > 0           λ₁ < 0 < λ₂
(Stable Node)        (Unstable Node)      (Saddle Point)
    ↘ ↓ ↙               ↗ ↑ ↖               ↗ ← ↙
     ↓ • ↓              ↑ • ↑              ↑ • ↓
    ↙ ↓ ↘               ↖ ↑ ↗              ↖ → ↘

λ = α ± iβ, α < 0    λ = α ± iβ, α > 0   λ = ±iβ (α = 0)
(Stable Spiral)      (Unstable Spiral)   (Center)
      ⤸ ↑ ⤹              ⤸ ↑ ⤹              ↺ ↑ ↻
    ← ← • → →          ← ← • → →          ← • →
      ⤷ ↓ ⤶              ⤷ ↓ ⤶              ↻ ↓ ↺
```

**Example 3.8.8:** Classify the equilibrium at the origin for $\frac{d\mathbf{y}}{dt} = \begin{bmatrix}1 & -5 \\ 1 & -3\end{bmatrix}\mathbf{y}$.

Characteristic equation: $(1-\lambda)(-3-\lambda)+5 = \lambda^2 + 2\lambda + 2 = 0$.

Roots: $\lambda = \frac{-2 \pm \sqrt{4-8}}{2} = -1 \pm i$.

Since $\lambda = \alpha \pm i\beta$ with $\alpha = -1 < 0$ and $\beta = 1$:

- The origin is a **stable spiral** (trajectories spiral inward)
- Solutions oscillate with frequency $\beta = 1$ and decay with rate $|\alpha| = 1$
- All trajectories approach the origin as $t \to \infty$

**ML Connection:** Residual network stability depends on eigenvalues of residual transformation matrices. Stable eigenvalues (negative real parts) ensure gradients don't explode.

---

## 3.8.4 Numerical Methods for ODEs

### Euler's Method

**Definition 3.8.6 (Euler's Method)**
To solve $\frac{dy}{dt} = f(t, y)$ with $y(t_0) = y_0$, discretize time: $t_n = t_0 + nh$ for step size $h$.

**Forward Euler iteration:**

$$y_{n+1} = y_n + h \cdot f(t_n, y_n)$$

**Geometric Interpretation:**

```
True Solution vs Euler's Method

y  True y(t)
│   ╱╱╱╱╱╱
│  ╱      ╱
│ ╱   ╱╱╱    Euler follows tangent
│╱ ╱╱        line at each step
•──•──•──•   (linear approximation)
t₀ t₁ t₂ t₃

Error accumulates:
Local error: O(h²)
Global error: O(h)
```

**Example 3.8.9:** Apply Euler's method to $\frac{dy}{dt} = -2y$, $y(0) = 1$, with step size $h = 0.5$ for two steps.

Recall: $y_{n+1} = y_n + h \cdot f(t_n, y_n)$ where $f(t, y) = -2y$.

**Step 1** ($t_0 = 0$): $y_1 = y_0 + 0.5 \cdot (-2)(1) = 1 - 1 = 0$

**Step 2** ($t_1 = 0.5$): $y_2 = y_1 + 0.5 \cdot (-2)(0) = 0$

Exact solution: $y(t) = e^{-2t}$, so $y(0.5) = e^{-1} \approx 0.368$ and $y(1) = e^{-2} \approx 0.135$.

Euler gives $y_1 = 0$ and $y_2 = 0$ -- the large step size $h = 0.5$ causes significant error. With $h = 0.1$, the approximation would be much closer to the true solution.

**Connection to Gradient Descent:**

Minimize $\mathcal{L}(\theta)$: gradient flow ODE is $\frac{d\theta}{dt} = -\nabla\mathcal{L}(\theta)$

Euler discretization: $\theta_{n+1} = \theta_n - h\nabla\mathcal{L}(\theta_n)$

This is gradient descent with learning rate $h$.

### Higher-Order Methods

**Definition 3.8.7 (Runge-Kutta Methods)**
The *4th-order Runge-Kutta method* (RK4):

$$\begin{align}
k_1 &= f(t_n, y_n) \\
k_2 &= f(t_n + h/2, y_n + hk_1/2) \\
k_3 &= f(t_n + h/2, y_n + hk_2/2) \\
k_4 &= f(t_n + h, y_n + hk_3) \\
y_{n+1} &= y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\end{align}$$

**Accuracy:** Global error is $O(h^4)$, much better than Euler's $O(h)$.

**ML Connection:** Adaptive solvers (e.g., Dormand-Prince) are used in Neural ODE implementations to balance accuracy and computation.

---

## 3.8.5 Neural Ordinary Differential Equations

**Definition 3.8.8 (Neural ODE Layer)**
A *Neural ODE* layer defines a continuous transformation via an ODE:

$$\frac{d\mathbf{h}}{dt} = f_\theta(\mathbf{h}(t), t)$$

where:
- $\mathbf{h}(t) \in \mathbb{R}^d$ is the hidden state at "time" $t$
- $f_\theta$ is a neural network (e.g., MLP) with parameters $\theta$
- Input $\mathbf{h}(0) = \mathbf{x}$, output $\mathbf{h}(1) = \mathbf{z}$

**Architecture Diagram:**

```
Standard ResNet         Neural ODE (Continuous)

x                       x = h(0)
│                       │
├─[Block]─┬→            │  dh/dt = f_θ(h,t)
│    ↓    +             │  ↓
├─[Block]─┬→            │  Solve ODE
│    ↓    +      →      │  ↓
├─[Block]─┬→            │  ↓
│    ↓    +             │  ↓
├─[Block]─┬→            │
│    ↓    +             z = h(1)
z

Discrete layers         Continuous depth
O(L) memory             O(1) memory
Fixed computation       Adaptive computation
```

**Theorem 3.8.4 (ResNet as Euler Discretization)**
A residual block $\mathbf{h}_{l+1} = \mathbf{h}_l + f_l(\mathbf{h}_l)$ is the Euler discretization of:

$$\frac{d\mathbf{h}}{dt} = f(\mathbf{h})$$

with step size $h = 1$ and $f_l \approx f$.

### Adjoint Method for Backpropagation

**Problem:** Backpropagating through ODE solver requires storing all intermediate states (memory-intensive).

**Solution (Adjoint Method):** Define adjoint state $\mathbf{a}(t) = \frac{\partial L}{\partial \mathbf{h}(t)}$.

**Theorem 3.8.5 (Adjoint Equation)**
The gradient $\frac{\partial L}{\partial \theta}$ can be computed by solving the *adjoint ODE* backward in time:

$$\frac{d\mathbf{a}}{dt} = -\mathbf{a}^T \frac{\partial f_\theta}{\partial \mathbf{h}}, \quad \mathbf{a}(1) = \frac{\partial L}{\partial \mathbf{z}}$$

$$\frac{dL}{d\theta} = -\int_1^0 \mathbf{a}^T \frac{\partial f_\theta}{\partial \theta} dt$$

**Advantage:** Constant memory (O(1)) regardless of network depth.

**ML Applications:**
- Continuous normalizing flows
- Time-series modeling with irregular observations
- Parameter-efficient deep networks

---

## 3.8.6 Stochastic Differential Equations (SDEs)

**Definition 3.8.9 (Stochastic Differential Equation)**
An *SDE* incorporates random noise via Brownian motion $W_t$:

$$dX_t = \mu(X_t, t)\,dt + \sigma(X_t, t)\,dW_t$$

where:
- $\mu$ is the *drift* coefficient (deterministic trend)
- $\sigma$ is the *diffusion* coefficient (noise magnitude)
- $W_t$ is standard Brownian motion

**Brownian Motion Properties:**

1. $W_0 = 0$
2. Continuous paths
3. Independent increments: $W_t - W_s \sim \mathcal{N}(0, t-s)$ for $t > s$
4. $\mathbb{E}[dW_t] = 0$, $\mathbb{E}[dW_t^2] = dt$

**Itô's Lemma (Chain Rule for SDEs):**

For $Y_t = g(X_t, t)$, where $dX_t = \mu\,dt + \sigma\,dW_t$:

$$dY_t = \left(\frac{\partial g}{\partial t} + \mu\frac{\partial g}{\partial x} + \frac{1}{2}\sigma^2\frac{\partial^2 g}{\partial x^2}\right)dt + \sigma\frac{\partial g}{\partial x}dW_t$$

**Note:** The second-order term $\frac{1}{2}\sigma^2\frac{\partial^2 g}{\partial x^2}$ arises from $(dW_t)^2 = dt$.

### Euler-Maruyama Method

**Numerical discretization for SDEs:**

$$X_{n+1} = X_n + \mu(X_n, t_n)\Delta t + \sigma(X_n, t_n)\sqrt{\Delta t}\,\xi_n$$

where $\xi_n \sim \mathcal{N}(0, 1)$ are independent standard normals.

**Example 3.8.10:** Apply Euler-Maruyama to $dX_t = -X_t\,dt + 0.5\,dW_t$ with $X_0 = 1$, $\Delta t = 0.1$.

Here $\mu(X, t) = -X$ and $\sigma(X, t) = 0.5$.

**Step 1** ($t = 0$): Suppose $\xi_0 = 0.3$ (drawn from $\mathcal{N}(0,1)$).

$$X_1 = 1 + (-1)(0.1) + 0.5 \cdot \sqrt{0.1} \cdot 0.3 = 0.9 + 0.5(0.316)(0.3) = 0.9 + 0.047 = 0.947$$

**Step 2** ($t = 0.1$): Suppose $\xi_1 = -0.8$.

$$X_2 = 0.947 + (-0.947)(0.1) + 0.5 \cdot \sqrt{0.1} \cdot (-0.8) = 0.852 - 0.126 = 0.726$$

The drift $-X_t$ pulls toward zero while the noise $0.5\,dW_t$ adds randomness. Each simulation run produces a different trajectory.

**ML Connection:** Score-based generative models use SDE dynamics to generate samples.

---

## 3.8.7 Diffusion Models

**Definition 3.8.10 (Diffusion Process)**
A *diffusion model* consists of two SDEs:

1. **Forward Process (Adding Noise):**

$$dx = -\frac{1}{2}\beta(t)x\,dt + \sqrt{\beta(t)}\,dW_t$$

Starting from data $x_0 \sim p_{\text{data}}$, gradually adds noise until $x_T \sim \mathcal{N}(0, I)$.

2. **Reverse Process (Denoising):**

$$dx = \left[\frac{1}{2}\beta(t)x + \beta(t)\nabla_x\log p_t(x)\right]dt + \sqrt{\beta(t)}\,d\bar{W}_t$$

Starting from noise $x_T$, removes noise to generate data.

**Key Insight:** The score function $\nabla_x\log p_t(x)$ is learned by a neural network $s_\theta(x, t)$.

**Diffusion Process Diagram:**

```
Forward Process (Data → Noise)

t=0         t=0.25      t=0.5       t=0.75      t=1
x₀          x₀.₂₅       x₀.₅        x₀.₇₅       xₜ
🖼️   →      📷    →     ⬜    →     ⬛    →     🌫️
(data)      (slight     (noisy)     (very       (pure
            noise)                  noisy)      noise)

Reverse Process (Noise → Data)

t=1         t=0.75      t=0.5       t=0.25      t=0
xₜ          x₀.₇₅       x₀.₅        x₀.₂₅       x₀
🌫️   →      ⬛    →     ⬜    →     📷    →     🖼️
(noise)     (denoise)   (denoise)   (refine)    (sample)

Score Network: s_θ(x,t) ≈ ∇ₓ log p_t(x)
```

**Theorem 3.8.6 (DDPM Objective)**
Training diffusion models minimizes the score matching loss:

$$\mathcal{L}(\theta) = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - s_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t)\|^2\right]$$

where $\epsilon \sim \mathcal{N}(0, I)$ is the noise added.

**ML Applications:**
- Image generation (DALL-E 2, Stable Diffusion)
- Molecule design
- Audio synthesis

---

## 3.8.8 Partial Differential Equations (Brief Overview)

**Definition 3.8.11 (Partial Differential Equation)**
A *PDE* involves partial derivatives with respect to multiple variables (e.g., space and time).

### Heat Equation

**One-dimensional heat equation:**

$$\frac{\partial u}{\partial t} = \alpha\frac{\partial^2 u}{\partial x^2}$$

where $u(x, t)$ is temperature at position $x$ and time $t$.

**Interpretation:** Temperature diffuses from hot to cold regions.

### Wave Equation

$$\frac{\partial^2 u}{\partial t^2} = c^2\frac{\partial^2 u}{\partial x^2}$$

Models vibrations, sound waves, electromagnetic waves.

### Laplacian Operator

**Definition 3.8.12 (Laplacian)**
In $\mathbb{R}^n$:

$$\Delta u = \nabla^2 u = \sum_{i=1}^n\frac{\partial^2 u}{\partial x_i^2}$$

**Graph Laplacian:** For graph $G = (V, E)$ with adjacency matrix $A$ and degree matrix $D$:

$$L = D - A$$

**ML Connection:** Graph Neural Networks use the graph Laplacian for message passing:

$$\mathbf{h}^{(l+1)} = \sigma\left(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}\mathbf{h}^{(l)}W^{(l)}\right)$$

where $\tilde{A} = A + I$ includes self-loops.

### Physics-Informed Neural Networks (PINNs)

**Idea:** Solve PDEs by training neural networks $u_\theta(x, t)$ to satisfy:

1. **PDE constraint:** $\mathcal{N}[u] = 0$ (e.g., heat equation)
2. **Boundary conditions:** $u(x, 0) = u_0(x)$
3. **Initial conditions:** $u(x_b, t) = g(t)$

**Loss function:**

$$\mathcal{L}(\theta) = \|\mathcal{N}[u_\theta]\|^2 + \|u_\theta - u_0\|^2_{\text{IC}} + \|u_\theta - g\|^2_{\text{BC}}$$

**Advantage:** Meshless, can incorporate noisy data, solve inverse problems.

---

## Key Formulas Summary

| **Concept** | **Formula** |
|-------------|-------------|
| First-order ODE | $\frac{dy}{dt} = f(t, y)$ |
| Matrix exponential solution | $\mathbf{y}(t) = e^{At}\mathbf{y}_0$ |
| Euler's method | $y_{n+1} = y_n + h \cdot f(t_n, y_n)$ |
| Neural ODE | $\frac{d\mathbf{h}}{dt} = f_\theta(\mathbf{h}(t), t)$ |
| Adjoint equation | $\frac{d\mathbf{a}}{dt} = -\mathbf{a}^T \frac{\partial f_\theta}{\partial \mathbf{h}}$ |
| SDE | $dX_t = \mu\,dt + \sigma\,dW_t$ |
| Itô's lemma | $dY_t = \left(\frac{\partial g}{\partial t} + \mu\frac{\partial g}{\partial x} + \frac{1}{2}\sigma^2\frac{\partial^2 g}{\partial x^2}\right)dt + \sigma\frac{\partial g}{\partial x}dW_t$ |
| Diffusion forward | $dx = -\frac{1}{2}\beta(t)x\,dt + \sqrt{\beta(t)}\,dW_t$ |
| Diffusion reverse | $dx = \left[\frac{1}{2}\beta(t)x + \beta(t)\nabla_x\log p_t(x)\right]dt + \sqrt{\beta(t)}\,d\bar{W}_t$ |
| Laplacian | $\Delta u = \sum_{i=1}^n\frac{\partial^2 u}{\partial x_i^2}$ |

---

## ML Applications Across Topics

### Neural ODEs
- **Continuous-depth networks:** Replace discrete layers with ODE solvers
- **Normalizing flows:** Continuous change of variables for generative modeling
- **Time-series:** Model irregular time observations naturally

### Diffusion Models
- **Image generation:** DDPM, Stable Diffusion, DALL-E 2
- **Score-based models:** Learn energy-based models via SDE denoising
- **Conditional generation:** Text-to-image, inpainting, super-resolution

### ResNets as Discretized ODEs
- **Insight:** ResNet $\mathbf{h}_{l+1} = \mathbf{h}_l + f_l(\mathbf{h}_l)$ approximates $\frac{d\mathbf{h}}{dt} = f(\mathbf{h})$
- **Implications:** Deeper networks = finer ODE discretization
- **Stability:** Eigenvalue analysis predicts gradient behavior

### Optimization as Gradient Flow
- **Gradient descent:** Discretized $\frac{d\theta}{dt} = -\nabla\mathcal{L}(\theta)$
- **Momentum:** Second-order ODE $\frac{d^2\theta}{dt^2} + \gamma\frac{d\theta}{dt} = -\nabla\mathcal{L}(\theta)$
- **Continuous-time analysis:** Understand convergence via Lyapunov functions

### Graph Neural Networks
- **Graph diffusion:** $\frac{\partial \mathbf{H}}{\partial t} = -L\mathbf{H}$ where $L$ is graph Laplacian
- **Spectral methods:** Eigendecomposition of Laplacian defines graph filters
- **Attention as diffusion:** Attention weights control information flow

---

## Exercises

### ★ Basic Understanding

**Exercise 3.8.1:** Solve the separable ODE $\frac{dy}{dt} = ty$ with $y(0) = 1$.

**Exercise 3.8.2:** For the system $\frac{d\mathbf{y}}{dt} = \begin{bmatrix}-1 & 0 \\ 0 & -2\end{bmatrix}\mathbf{y}$, compute $e^{At}$ and describe the stability.

**Exercise 3.8.3:** Implement Euler's method to solve $\frac{dy}{dt} = -2y$, $y(0) = 1$ with $h = 0.1$ for $t \in [0, 1]$. Compare with exact solution $y(t) = e^{-2t}$.

**Exercise 3.8.4:** Explain why the diffusion reverse process requires learning $\nabla_x\log p_t(x)$ rather than $p_t(x)$ directly.

### ★★ Intermediate Application

**Exercise 3.8.5:** A 2D system has matrix $A = \begin{bmatrix}0 & -1 \\ 1 & 0\end{bmatrix}$. Find eigenvalues and sketch the phase portrait. What type of equilibrium is the origin?

**Exercise 3.8.6:** Show that a ResNet block $\mathbf{h}_{l+1} = \mathbf{h}_l + f(\mathbf{h}_l)$ with $N$ layers approximates the ODE $\frac{d\mathbf{h}}{dt} = f(\mathbf{h})$ integrated from $t=0$ to $t=N$ with step size 1.

**Exercise 3.8.7:** For the SDE $dX_t = -X_t\,dt + dW_t$, find the mean $\mathbb{E}[X_t]$ and variance $\text{Var}(X_t)$ given $X_0 = x_0$. (Hint: Use Itô's lemma on $Y_t = e^t X_t$.)

**Exercise 3.8.8:** Implement RK4 for the Neural ODE $\frac{d\mathbf{h}}{dt} = \tanh(W\mathbf{h} + \mathbf{b})$ where $W = \begin{bmatrix}0.5 & 0.2 \\ -0.3 & 0.4\end{bmatrix}$, $\mathbf{b} = \begin{bmatrix}0.1 \\ -0.1\end{bmatrix}$. Compare memory usage with storing all intermediate steps.

### ★★★ Advanced Problems

**Exercise 3.8.9:** Derive the adjoint equation for computing $\frac{\partial L}{\partial \theta}$ in a Neural ODE. Prove that it requires only O(1) memory.

**Exercise 3.8.10:** For the heat equation $\frac{\partial u}{\partial t} = \alpha\frac{\partial^2 u}{\partial x^2}$ on $[0, L]$ with boundary conditions $u(0, t) = u(L, t) = 0$, solve via separation of variables $u(x, t) = X(x)T(t)$. Relate eigenfunctions to Fourier modes.

**Exercise 3.8.11:** Prove that the forward diffusion SDE $dx = -\frac{1}{2}\beta(t)x\,dt + \sqrt{\beta(t)}\,dW_t$ converges to $\mathcal{N}(0, I)$ as $t \to \infty$ for any initial $x_0$.

**Exercise 3.8.12:** Implement a physics-informed neural network to solve the 1D heat equation with initial condition $u(x, 0) = \sin(\pi x)$ on $[0, 1]$. Use automatic differentiation to compute PDE residuals.

**Exercise 3.8.13:** Design a continuous normalizing flow using Neural ODEs. Show that the log-likelihood can be computed using the instantaneous change of variables formula:

$$\log p(\mathbf{x}) = \log p(\mathbf{z}) - \int_0^1 \text{Tr}\left(\frac{\partial f_\theta}{\partial \mathbf{h}(t)}\right)dt$$

where $\mathbf{z} = \mathbf{h}(1)$ and $\mathbf{x} = \mathbf{h}(0)$.

---

## Related Topics

### Within This Text
- **Chapter 2.4:** Eigenvalues and eigenvectors (for ODE system stability)
- **Chapter 3.5:** Optimization and gradient descent (gradient flow connection)
- **Chapter 4.2:** Probability and stochastic processes (foundation for SDEs)
- **Chapter 5.1:** Deep learning architectures (ResNets, Neural ODEs)
- **Chapter 5.6:** Generative models (VAEs, GANs, diffusion models)

### Advanced Extensions
- **Dynamical systems theory:** Bifurcations, chaos, attractors
- **Stochastic calculus:** Martingales, Itô integrals, Stratonovich calculus
- **Numerical analysis:** Stiff ODEs, adaptive solvers, symplectic integrators
- **Optimal control theory:** Pontryagin's maximum principle, Hamilton-Jacobi-Bellman equation
- **Partial differential equations:** Finite element methods, spectral methods
- **Continuous optimization:** Gradient flow, Wasserstein gradient descent
- **Score-based generative models:** Denoising diffusion probabilistic models, score matching

### Research Frontiers
- **Neural controlled differential equations** for time-series with irregular sampling
- **Stochastic normalizing flows** combining SDEs and normalizing flows
- **SDE-based optimization** (e.g., stochastic gradient Langevin dynamics)
- **Graph Neural ODEs** for continuous-time graph learning
- **Latent SDEs** for stochastic modeling of time-series
- **Riemannian ODEs/SDEs** on manifolds for constrained learning

---

## Further Reading

**Textbooks:**
- Tenenbaum & Pollard, *Ordinary Differential Equations* (rigorous ODE theory)
- Oksendal, *Stochastic Differential Equations* (SDE fundamentals)
- Evans, *Partial Differential Equations* (comprehensive PDE treatment)

**ML-Specific:**
- Chen et al., "Neural Ordinary Differential Equations" (NeurIPS 2018)
- Song et al., "Score-Based Generative Modeling through SDEs" (ICLR 2021)
- Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
- Kidger, "On Neural Differential Equations" (PhD thesis, Oxford 2021)

**Implementation:**
- `torchdiffeq`: PyTorch library for differentiable ODE solvers
- `diffrax`: JAX library for differential equation solvers
- `torchsde`: PyTorch SDE solver with GPU support

---

*Navigation:*
← [Chapter 3.7: Integration and Measure Theory](integration.md) | [Chapter 4: Probability Theory](../probability/index.md) →

# Policy Gradient Methods

**Learn policies directly by optimizing parameters to maximize expected return.**

Policy gradient methods parameterize the policy and update it using gradient ascent on expected rewards.

---

## 🎯 Why Policy Gradient?

### Value-Based vs. Policy-Based

**Value-Based (Q-Learning, DQN):**
- Learn Q(s,a) or V(s)
- Policy derived implicitly: π(s) = argmax_a Q(s,a)
- ✅ Sample efficient
- ❌ Only deterministic policies
- ❌ Can't handle continuous actions easily

**Policy-Based (Policy Gradient):**
- Learn policy π(a|s; θ) directly
- Parameterized by θ (neural network weights)
- ✅ Can learn stochastic policies
- ✅ Handles continuous action spaces naturally
- ✅ Better convergence properties
- ❌ High variance
- ❌ Sample inefficient

---

## 🧠 Policy Parameterization

### Stochastic Policy

**Discrete Actions:**
```python
# Softmax policy
π(a|s; θ) = exp(f_θ(s,a)) / Σ_a' exp(f_θ(s,a'))

# Neural network outputs logits → softmax → probabilities
logits = network(state)  # [Q(s,a₀), Q(s,a₁), ..., Q(s,aₙ)]
probs = softmax(logits)  # [π(a₀|s), π(a₁|s), ..., π(aₙ|s)]
action = sample(probs)
```

**Continuous Actions:**
```python
# Gaussian policy
π(a|s; θ) = N(μ_θ(s), σ_θ(s))

# Network outputs mean and std
mean, std = network(state)
action = mean + std * noise  # noise ~ N(0,1)
```

### Example: PyTorch Policy Network

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Policy network for discrete actions

        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_dim: Hidden layer size
        """
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """
        Forward pass

        Args:
            state: State tensor

        Returns:
            action_probs: Probability distribution over actions
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        action_probs = F.softmax(logits, dim=-1)
        return action_probs

    def select_action(self, state):
        """
        Sample action from policy

        Args:
            state: Current state

        Returns:
            action, log_prob
        """
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
```

---

## ⚡ Policy Gradient Theorem

### The Objective

**Goal:** Maximize expected return

```
J(θ) = E_π[G_t] = E_π[Σ_{t=0}^T γ^t r_t]
```

**Gradient:**
```
∇_θ J(θ) = E_π[∇_θ log π(a_t|s_t; θ) · G_t]
```

**Intuition:**
- If return G_t is high → increase probability of actions taken
- If return G_t is low → decrease probability of actions taken

### Derivation (Simplified)

```
J(θ) = E_π[G_t]
     = Σ_τ P(τ; θ) G(τ)     (τ = trajectory)

∇_θ J(θ) = Σ_τ ∇_θ P(τ; θ) G(τ)
         = Σ_τ P(τ; θ) ∇_θ log P(τ; θ) G(τ)     (log-derivative trick)
         = E_π[∇_θ log P(τ; θ) G(τ)]

P(τ; θ) = P(s_0) Π_t π(a_t|s_t; θ) P(s_{t+1}|s_t,a_t)

∇_θ log P(τ; θ) = Σ_t ∇_θ log π(a_t|s_t; θ)     (only policy depends on θ)

Therefore:
∇_θ J(θ) = E_π[Σ_t ∇_θ log π(a_t|s_t; θ) · G_t]
```

---

## 🎮 REINFORCE Algorithm

### Monte Carlo Policy Gradient

**Algorithm:**
```
Initialize policy parameters θ randomly

For each episode:
    Generate episode: s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T

    For each time step t:
        Calculate return: G_t = Σ_{k=t}^T γ^{k-t} r_k

        Update policy:
            θ ← θ + α ∇_θ log π(a_t|s_t; θ) · G_t
```

### Implementation

```python
class REINFORCE:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        """
        REINFORCE agent

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            lr: Learning rate
            gamma: Discount factor
        """
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

        # Storage
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        """Select action and store log probability"""
        state_t = torch.FloatTensor(state)
        action, log_prob = self.policy.select_action(state_t)

        self.log_probs.append(log_prob)

        return action

    def store_reward(self, reward):
        """Store reward"""
        self.rewards.append(reward)

    def update(self):
        """Update policy at end of episode"""
        # Calculate returns
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)

        # Normalize returns (reduce variance)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Calculate policy gradient
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)  # Negative for gradient ascent

        # Update policy
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        # Clear storage
        self.log_probs = []
        self.rewards = []

        return policy_loss.item()

    def train(self, env, num_episodes=1000):
        """Train REINFORCE agent"""
        episode_rewards = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False

            # Generate episode
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                self.store_reward(reward)

                state = next_state
                episode_reward += reward

            # Update policy
            loss = self.update()

            episode_rewards.append(episode_reward)

            # Log
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode+1}, Avg Reward: {avg_reward:.2f}")

        return episode_rewards
```

---

## 📊 Variance Reduction Techniques

### 1. Baseline Subtraction

**Problem:** High variance in gradient estimates

**Solution:** Subtract baseline b(s) from return

```
∇_θ J(θ) = E_π[∇_θ log π(a_t|s_t; θ) · (G_t - b(s_t))]
```

**Common Baseline:** State-value function V(s)

```python
def update_with_baseline(self):
    """Update with baseline (value function)"""
    returns = self.calculate_returns()

    # Baseline: average return
    baseline = returns.mean()

    # Advantages
    advantages = returns - baseline

    # Normalize
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy gradient with baseline
    policy_loss = []
    for log_prob, advantage in zip(self.log_probs, advantages):
        policy_loss.append(-log_prob * advantage)

    # Update
    loss = torch.stack(policy_loss).sum()
    loss.backward()
    self.optimizer.step()
```

### 2. Reward-to-Go

**Idea:** Use future rewards from time t, not all episode rewards

```python
def calculate_returns_to_go(self):
    """Calculate reward-to-go (more accurate)"""
    returns = []
    G = 0

    # Backward pass
    for r in reversed(self.rewards):
        G = r + self.gamma * G
        returns.insert(0, G)

    return torch.FloatTensor(returns)
```

**Comparison:**
```
Full episode return: G_t = r_0 + r_1 + r_2 + ... + r_T
Reward-to-go:        G_t = r_t + r_{t+1} + ... + r_T

Reward-to-go has lower variance!
```

### 3. Generalized Advantage Estimation (GAE)

**Idea:** Exponentially weighted average of n-step advantages

```
A^{GAE}(s_t,a_t) = Σ_{l=0}^∞ (γλ)^l δ_{t+l}

Where δ_t = r_t + γV(s_{t+1}) - V(s_t)  (TD error)
```

**Implementation:**
```python
def compute_gae(rewards, values, next_value, gamma=0.99, lambda_=0.95):
    """
    Compute Generalized Advantage Estimation

    Args:
        rewards: List of rewards
        values: List of value estimates
        next_value: Value of next state
        gamma: Discount factor
        lambda_: GAE parameter

    Returns:
        advantages
    """
    advantages = []
    gae = 0

    # Backward pass
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        # TD error
        delta = rewards[t] + gamma * next_val - values[t]

        # GAE
        gae = delta + gamma * lambda_ * gae
        advantages.insert(0, gae)

    return torch.FloatTensor(advantages)
```

---

## 🔬 Variants and Improvements

### 1. Natural Policy Gradient (NPG)

**Idea:** Use natural gradient (Fisher information matrix)

```
θ_{new} = θ + α F^{-1} ∇_θ J(θ)

Where F = E[∇_θ log π(a|s) ∇_θ log π(a|s)^T]  (Fisher information)
```

**Benefits:**
- More stable updates
- Invariant to parameterization

### 2. Trust Region Policy Optimization (TRPO)

**Idea:** Constrain policy update to trust region

```
maximize_θ E[π_θ/π_θ_old · A]
subject to E[KL(π_θ_old || π_θ)] ≤ δ
```

**Benefits:**
- Monotonic improvement guarantee
- More stable than vanilla policy gradient

### 3. Proximal Policy Optimization (PPO)

**Idea:** Clip objective to prevent large updates

```
L^{CLIP}(θ) = E[min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)]

Where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (probability ratio)
```

**Implementation:**
```python
def ppo_loss(old_log_probs, new_log_probs, advantages, epsilon=0.2):
    """
    PPO clipped objective

    Args:
        old_log_probs: Log probs from old policy
        new_log_probs: Log probs from new policy
        advantages: Advantage estimates
        epsilon: Clipping parameter

    Returns:
        PPO loss
    """
    # Probability ratio
    ratio = torch.exp(new_log_probs - old_log_probs)

    # Clipped surrogate objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages

    # Take minimum
    policy_loss = -torch.min(surr1, surr2).mean()

    return policy_loss
```

**Benefits:**
- Simpler than TRPO (no constraint, just clipping)
- State-of-the-art performance
- Most popular policy gradient method

---

## 🎯 Continuous Action Spaces

### Gaussian Policy

```python
class ContinuousPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ContinuousPolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Mean and log std
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)
        std = torch.exp(self.log_std)

        return mean, std

    def select_action(self, state):
        mean, std = self.forward(state)

        # Sample from Gaussian
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob
```

---

## 🎓 Example: CartPole with REINFORCE

```python
import gym

# Environment
env = gym.make('CartPole-v1')

# Agent
agent = REINFORCE(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    lr=0.01,
    gamma=0.99
)

# Train
rewards = agent.train(env, num_episodes=1000)

# Plot results
import matplotlib.pyplot as plt
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('REINFORCE on CartPole-v1')
plt.show()
```

---

## 🎯 Best Practices

### 1. Normalize Advantages

```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

### 2. Use Entropy Regularization

```python
# Encourage exploration
entropy = dist.entropy().mean()
loss = policy_loss - entropy_coef * entropy
```

### 3. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
```

### 4. Learning Rate Schedule

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
```

---

## 🚀 Next Steps

[→ Actor-Critic Methods](actor-critic.md){ .md-button .md-button--primary }

**Key Takeaway:** Policy gradient methods optimize policies directly using gradient ascent on expected returns. They handle continuous actions naturally and can learn stochastic policies, but require careful variance reduction techniques for stable training.

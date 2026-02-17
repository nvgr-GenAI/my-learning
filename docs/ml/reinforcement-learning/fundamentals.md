# RL Fundamentals

**Master the core concepts: MDPs, value functions, policies, and Bellman equations.**

Understanding these fundamentals is essential for all reinforcement learning algorithms.

---

## 🎯 The Markov Decision Process (MDP)

### Formal Definition

An MDP is defined by a tuple (S, A, P, R, γ):

- **S:** Set of states (all possible situations)
- **A:** Set of actions (all possible decisions)
- **P:** Transition probability P(s'|s,a) = probability of reaching state s' from state s taking action a
- **R:** Reward function R(s,a) = immediate reward for taking action a in state s
- **γ:** Discount factor (0 ≤ γ ≤ 1)

### The Markov Property

**Definition:** The future is independent of the past given the present

```
P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ..., s_0, a_0)
= P(s_{t+1} | s_t, a_t)
```

**Intuition:** Current state contains all relevant information needed to predict the future.

**Example: Chess**
- **Markovian:** Current board position is sufficient (don't need move history)
- **Non-Markovian:** If we only saw last 3 moves (missing board state)

---

## 🎓 Example: Grid World MDP

### Environment Setup

```
┌───┬───┬───┬───┐
│ S │   │   │ G │  S = Start (0,0)
├───┼───┼───┼───┤  G = Goal (0,3)
│   │ X │   │   │  X = Obstacle (1,1)
├───┼───┼───┼───┤  Actions: UP, DOWN, LEFT, RIGHT
│   │   │   │   │
└───┴───┴───┴───┘

Rewards:
- Step: -1 (incentivize shortest path)
- Goal: +10
- Obstacle: -5
```

### MDP Components

**States (S):**
```python
S = {(0,0), (0,1), (0,2), (0,3),  # Row 0
     (1,0), (1,1), (1,2), (1,3),  # Row 1
     (2,0), (2,1), (2,2), (2,3)}  # Row 2
```

**Actions (A):**
```python
A = {UP, DOWN, LEFT, RIGHT}
```

**Transition Dynamics P(s'|s,a):**
```python
# Deterministic transitions
P((1,0) | (0,0), DOWN) = 1.0  # From (0,0), moving DOWN goes to (1,0)
P((0,1) | (0,0), RIGHT) = 1.0  # From (0,0), moving RIGHT goes to (0,1)

# Stochastic transitions (slippery grid)
P((1,0) | (0,0), DOWN) = 0.8   # Intended direction
P((0,1) | (0,0), DOWN) = 0.1   # Slip right
P((0,0) | (0,0), DOWN) = 0.1   # Slip left (stay)
```

**Reward Function R(s,a):**
```python
R((0,3), *) = +10   # Reaching goal
R((1,1), *) = -5    # Hitting obstacle
R(s, a) = -1        # Every other step
```

---

## 🎯 Policy π

### Definition

A policy π is a mapping from states to actions:

**Deterministic Policy:**
```
π : S → A
π(s) = a
```

**Stochastic Policy:**
```
π : S × A → [0,1]
π(a|s) = probability of taking action a in state s
```

### Example Policies

**Random Policy:**
```python
π_random(a|s) = 1/|A| = 0.25  # Equal probability for each action
```

**Greedy Policy (always move toward goal):**
```python
π_greedy((0,0)) = RIGHT   # From start, go right toward goal
π_greedy((0,1)) = RIGHT   # Keep going right
π_greedy((1,0)) = UP      # Go up if below
```

**Optimal Policy π\*:**
- The policy that maximizes expected cumulative reward
- What we're trying to find!

---

## 💰 Value Functions

### State-Value Function V^π(s)

**Definition:** Expected return starting from state s and following policy π

```
V^π(s) = E_π[G_t | s_t = s]
       = E_π[r_t + γr_{t+1} + γ²r_{t+2} + ... | s_t = s]
```

**Intuition:** "How good is it to be in state s under policy π?"

**Example: Grid World**
```
Policy: Always move RIGHT

V^π(0,0) = -1 + γ(-1) + γ²(-1) + γ³(10)  ≈ 6.3  (γ=0.9)
V^π(0,1) = -1 + γ(-1) + γ²(10)           ≈ 7.1
V^π(0,2) = -1 + γ(10)                     ≈ 8.0
V^π(0,3) = 10                              = 10.0  (at goal)
```

### Action-Value Function Q^π(s,a)

**Definition:** Expected return starting from state s, taking action a, then following policy π

```
Q^π(s,a) = E_π[G_t | s_t = s, a_t = a]
         = E_π[r_t + γV^π(s_{t+1}) | s_t = s, a_t = a]
```

**Intuition:** "How good is it to take action a in state s, then follow π?"

**Relationship:**
```
V^π(s) = Σ_a π(a|s) · Q^π(s,a)

For deterministic policy:
V^π(s) = Q^π(s, π(s))
```

**Example: Grid World at (0,0)**
```
Q^π((0,0), RIGHT) = -1 + γ·V^π(0,1) = -1 + 0.9×7.1 ≈ 5.4
Q^π((0,0), DOWN)  = -1 + γ·V^π(1,0) = -1 + 0.9×4.2 ≈ 2.8
Q^π((0,0), LEFT)  = -1 + γ·V^π(0,0) = -1 + 0.9×6.3 ≈ 4.7  (wall)
Q^π((0,0), UP)    = -1 + γ·V^π(0,0) = -1 + 0.9×6.3 ≈ 4.7  (wall)

Best action: RIGHT (highest Q-value)
```

### Optimal Value Functions

**Optimal State-Value Function V\*(s):**
```
V*(s) = max_π V^π(s)
```

**Optimal Action-Value Function Q\*(s,a):**
```
Q*(s,a) = max_π Q^π(s,a)
```

**Relationship:**
```
V*(s) = max_a Q*(s,a)

Q*(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) V*(s')
```

**Optimal Policy:**
```
π*(s) = argmax_a Q*(s,a)
```

Once we have Q\*(s,a), we can act optimally by always choosing the action with highest Q-value!

---

## ⚡ Bellman Equations

### Bellman Expectation Equation (for policy π)

**For V^π:**
```
V^π(s) = Σ_a π(a|s) [R(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')]
```

**For Q^π:**
```
Q^π(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) Σ_{a'} π(a'|s') Q^π(s',a')
```

**Intuition:** The value of a state is the immediate reward plus the discounted value of the next state.

**Example: Computing V^π(0,0)**
```
Policy: π(RIGHT|(0,0)) = 1.0  (always go right)

V^π(0,0) = R((0,0), RIGHT) + γ · P((0,1)|(0,0), RIGHT) · V^π(0,1)
         = -1 + 0.9 × 1.0 × 7.1
         = -1 + 6.39
         = 5.39
```

### Bellman Optimality Equation (for optimal policy)

**For V\*:**
```
V*(s) = max_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V*(s')]
```

**For Q\*:**
```
Q*(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) max_{a'} Q*(s',a')
```

**Key Difference:** max_a instead of Σ_a π(a|s)

**Example: Computing V\*(0,0)**
```
V*(0,0) = max_a [R((0,0), a) + γ Σ_{s'} P(s'|(0,0), a) V*(s')]

Try RIGHT: -1 + 0.9 × V*(0,1) = -1 + 0.9 × 8.0 = 6.2
Try DOWN:  -1 + 0.9 × V*(1,0) = -1 + 0.9 × 5.5 = 3.95
Try LEFT:  -1 + 0.9 × V*(0,0) = (stays at wall)
Try UP:    -1 + 0.9 × V*(0,0) = (stays at wall)

V*(0,0) = max(6.2, 3.95, ...) = 6.2
π*(0,0) = RIGHT
```

---

## 🔄 Solving MDPs

### 1. Dynamic Programming (Model-Based)

**Requires:** Full knowledge of P(s'|s,a) and R(s,a)

**Methods:**

**Policy Iteration:**
```
1. Initialize policy π randomly
2. Repeat:
     a. Policy Evaluation: Compute V^π using Bellman expectation equation
     b. Policy Improvement: Update π(s) = argmax_a Q^π(s,a)
3. Until policy converges
```

**Value Iteration:**
```
1. Initialize V(s) = 0 for all s
2. Repeat:
     V(s) ← max_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V(s')]
3. Until V converges
4. Extract policy: π(s) = argmax_a Q(s,a)
```

**Implementation: Value Iteration**
```python
def value_iteration(env, gamma=0.9, theta=1e-6):
    """
    Value Iteration algorithm

    Args:
        env: Environment with P (transitions) and R (rewards)
        gamma: Discount factor
        theta: Convergence threshold

    Returns:
        V: Optimal value function
        policy: Optimal policy
    """
    V = np.zeros(env.num_states)

    while True:
        delta = 0
        for s in range(env.num_states):
            v = V[s]
            # Bellman optimality backup
            V[s] = max([
                env.R(s, a) + gamma * sum([
                    env.P(s_prime, s, a) * V[s_prime]
                    for s_prime in range(env.num_states)
                ])
                for a in range(env.num_actions)
            ])
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    # Extract optimal policy
    policy = np.zeros(env.num_states, dtype=int)
    for s in range(env.num_states):
        policy[s] = np.argmax([
            env.R(s, a) + gamma * sum([
                env.P(s_prime, s, a) * V[s_prime]
                for s_prime in range(env.num_states)
            ])
            for a in range(env.num_actions)
        ])

    return V, policy
```

### 2. Monte Carlo Methods (Model-Free)

**Idea:** Learn from complete episodes

```
1. Generate episode using policy π
2. For each state s visited:
     - Calculate return G from that point
     - Update V(s) ← average of all returns from s
3. Improve policy based on V
```

**First-Visit MC:**
```python
def first_visit_mc(env, num_episodes=10000, gamma=0.9):
    """
    First-visit Monte Carlo for value estimation

    Args:
        env: Environment
        num_episodes: Number of episodes to sample
        gamma: Discount factor

    Returns:
        V: Estimated value function
    """
    V = np.zeros(env.num_states)
    returns = {s: [] for s in range(env.num_states)}

    for _ in range(num_episodes):
        # Generate episode
        episode = generate_episode(env, policy)

        # Calculate returns
        G = 0
        visited = set()

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = reward + gamma * G

            # First-visit: only count first occurrence
            if state not in visited:
                returns[state].append(G)
                V[state] = np.mean(returns[state])
                visited.add(state)

    return V
```

### 3. Temporal Difference Learning (Model-Free)

**Idea:** Learn from incomplete episodes (bootstrapping)

**TD(0) Update:**
```
V(s_t) ← V(s_t) + α[r_t + γV(s_{t+1}) - V(s_t)]
                    └────────┬──────────┘
                         TD Target

Where:
  α = learning rate
  r_t + γV(s_{t+1}) = TD target (estimate of true value)
  r_t + γV(s_{t+1}) - V(s_t) = TD error
```

**Advantages over MC:**
- Can learn from incomplete episodes
- Can learn online (every step)
- Lower variance (but biased)

**Implementation:**
```python
def td_zero(env, num_episodes=10000, alpha=0.1, gamma=0.9):
    """
    TD(0) for value estimation

    Args:
        env: Environment
        num_episodes: Number of episodes
        alpha: Learning rate
        gamma: Discount factor

    Returns:
        V: Estimated value function
    """
    V = np.zeros(env.num_states)

    for _ in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            # Take action according to policy
            action = policy(state)
            next_state, reward, done = env.step(action)

            # TD(0) update
            td_target = reward + gamma * V[next_state]
            td_error = td_target - V[state]
            V[state] += alpha * td_error

            state = next_state

    return V
```

---

## 📊 Comparison: DP vs. MC vs. TD

| Aspect | Dynamic Programming | Monte Carlo | Temporal Difference |
|--------|-------------------|-------------|---------------------|
| **Model** | Requires | Not required | Not required |
| **Bootstrap** | Yes | No | Yes |
| **Learns from** | Model | Complete episodes | Incomplete episodes |
| **Bias** | None | None | Biased (bootstrap) |
| **Variance** | Low | High | Medium |
| **Online** | ✅ Yes | ❌ No | ✅ Yes |
| **Converge** | ✅ Fast | ⚠️ Slow | ✅ Fast |

**Definitions:**
- **Bootstrap:** Update estimate using other estimates
- **Bias:** Error from approximation
- **Variance:** Error from sampling

---

## 🎯 Key Concepts Summary

### 1. Return (G_t)

**Definition:**
```
G_t = r_t + γr_{t+1} + γ²r_{t+2} + ...
    = r_t + γG_{t+1}
```

**Discount Factor Effects:**
```
γ = 0:  G_t = r_t  (only immediate reward)
γ = 0.9: G_t = r_t + 0.9r_{t+1} + 0.81r_{t+2} + ...  (typical)
γ = 1:  G_t = r_t + r_{t+1} + r_{t+2} + ...  (all rewards equal)
```

### 2. Policy Evaluation vs. Improvement

**Policy Evaluation:** Given policy π, compute V^π
```
V^π(s) = Σ_a π(a|s) [R(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')]
```

**Policy Improvement:** Given V^π, find better policy
```
π'(s) = argmax_a Q^π(s,a)
π'(s) = argmax_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')]
```

**Policy Iteration:**
```
π_0 → V^π_0 → π_1 → V^π_1 → π_2 → ... → π*
```

### 3. Prediction vs. Control

**Prediction Problem:** Evaluate a given policy
- Given: Policy π
- Find: V^π or Q^π
- Methods: Policy evaluation, MC, TD

**Control Problem:** Find optimal policy
- Given: MDP
- Find: Optimal policy π*
- Methods: Policy iteration, value iteration, Q-learning

---

## 💡 Practical Tips

### 1. Choosing γ (Discount Factor)

**Guidelines:**
- **γ = 0.9-0.95:** Short-term planning (games, robotics)
- **γ = 0.99:** Long-term planning (finance, strategic games)
- **γ = 1.0:** Episodic tasks with guaranteed termination

**Trade-offs:**
- Higher γ → Considers distant future, but slower convergence
- Lower γ → Faster convergence, but myopic behavior

### 2. Learning Rate α

**Guidelines:**
- **α = 0.1:** Good starting point
- **α = 1/n:** Decreasing over time (where n = visit count)
- **α = 0.01:** Fine-tuning, near convergence

**Effects:**
- Too high → Unstable, oscillation
- Too low → Slow convergence

### 3. Exploration Strategies

**ε-greedy:**
```python
def epsilon_greedy(Q, state, epsilon=0.1):
    if random.random() < epsilon:
        return random.choice(actions)  # Explore
    else:
        return argmax(Q[state])  # Exploit
```

**Decaying ε:**
```python
epsilon = max(0.01, epsilon * 0.995)  # Decay over time
```

---

## 🚀 Next Steps

Now that you understand the fundamentals, you're ready to implement RL algorithms!

**Recommended Learning Path:**

1. **Value-Based Methods:**
   - [Q-Learning](q-learning.md) - Classic tabular algorithm
   - [Deep Q-Networks](deep-q-networks.md) - Scaling with neural networks

2. **Policy-Based Methods:**
   - [Policy Gradient](policy-gradient.md) - Direct policy optimization

3. **Actor-Critic:**
   - [Actor-Critic Methods](actor-critic.md) - Combining value and policy

---

**Key Takeaway:** All RL algorithms build on these fundamentals: MDPs model the problem, value functions estimate long-term rewards, Bellman equations relate current and future values, and policies map states to actions. Master these, and you'll understand all of RL!

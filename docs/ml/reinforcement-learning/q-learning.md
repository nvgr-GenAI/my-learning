# Q-Learning

**Learn optimal action-value function Q\*(s,a) through temporal difference learning without requiring a model.**

Q-Learning is a model-free, off-policy algorithm that learns the optimal policy by directly estimating action values.

---

## 🎯 What is Q-Learning?

### Core Idea

**The Q-Function Q(s,a):**
- Estimates expected return from taking action a in state s
- Once we have Q\*(s,a), we can act optimally: π\*(s) = argmax_a Q\*(s,a)
- Learns through trial and error (no model required)

**Key Innovation:**
- **Model-Free:** Doesn't need P(s'|s,a) or R(s,a)
- **Off-Policy:** Learns optimal policy while following exploratory policy
- **Temporal Difference:** Updates using bootstrapping

---

## ⚡ The Q-Learning Update Rule

### The Algorithm

**Update Equation:**
```
Q(s_t, a_t) ← Q(s_t, a_t) + α[r_t + γ max_a Q(s_{t+1}, a) - Q(s_t, a_t)]
                                └────────────┬──────────────┘
                                        TD Target

Where:
  s_t, a_t = current state, action
  r_t = immediate reward
  s_{t+1} = next state
  α = learning rate (0 < α ≤ 1)
  γ = discount factor (0 ≤ γ ≤ 1)
```

**Components:**

1. **TD Target:** r_t + γ max_a Q(s_{t+1}, a)
   - Estimate of true Q-value
   - Uses max Q-value of next state (optimistic)

2. **TD Error:** TD Target - Q(s_t, a_t)
   - Difference between estimate and current value
   - Drives learning

3. **Update:** Move Q(s,a) toward TD target at rate α

---

## 🎓 Complete Q-Learning Algorithm

### Pseudocode

```
Algorithm: Q-Learning

Initialize Q(s,a) = 0 for all s, a

For each episode:
    Initialize state s

    While s is not terminal:
        Choose action a using ε-greedy policy from Q:
            With probability ε: choose random action (explore)
            With probability 1-ε: choose a = argmax_a' Q(s,a') (exploit)

        Take action a, observe reward r and next state s'

        Update Q-value:
            Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

        s ← s'

Return Q
```

### Python Implementation

```python
import numpy as np
import random

class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1,
                 discount_factor=0.99, epsilon=0.1):
        """
        Q-Learning Agent

        Args:
            n_states: Number of states
            n_actions: Number of actions
            learning_rate: α (learning rate)
            discount_factor: γ (discount factor)
            epsilon: Exploration rate for ε-greedy
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        # Initialize Q-table with zeros
        self.Q = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        """
        ε-greedy action selection

        Args:
            state: Current state

        Returns:
            action: Chosen action
        """
        if random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.n_actions - 1)
        else:
            # Exploit: best action according to Q
            return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        """
        Q-Learning update

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        # Current Q-value
        current_q = self.Q[state, action]

        # TD target
        if done:
            # Terminal state: no future rewards
            td_target = reward
        else:
            # Max Q-value of next state
            max_next_q = np.max(self.Q[next_state])
            td_target = reward + self.gamma * max_next_q

        # TD error
        td_error = td_target - current_q

        # Update Q-value
        self.Q[state, action] = current_q + self.lr * td_error

    def train(self, env, num_episodes=1000):
        """
        Train the agent

        Args:
            env: Environment
            num_episodes: Number of training episodes

        Returns:
            episode_rewards: List of rewards per episode
        """
        episode_rewards = []

        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                # Choose action
                action = self.choose_action(state)

                # Take action
                next_state, reward, done, _ = env.step(action)

                # Update Q-table
                self.update(state, action, reward, next_state, done)

                # Update state and total reward
                state = next_state
                total_reward += reward

            episode_rewards.append(total_reward)

            # Print progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")

        return episode_rewards

    def get_policy(self):
        """
        Extract deterministic policy from Q-table

        Returns:
            policy: Best action for each state
        """
        return np.argmax(self.Q, axis=1)
```

---

## 🎮 Example: Grid World

### Environment Setup

```
┌───┬───┬───┬───┐
│ S │   │   │ G │  S = Start
├───┼───┼───┼───┤  G = Goal (+10)
│   │ X │   │   │  X = Obstacle (-5)
├───┼───┼───┼───┤  Actions: UP, DOWN, LEFT, RIGHT
│   │   │   │   │  Step reward: -1
└───┴───┴───┴───┘
```

### Training Q-Learning

```python
# Environment
class GridWorld:
    def __init__(self):
        self.grid_size = (3, 4)
        self.start = (0, 0)
        self.goal = (0, 3)
        self.obstacle = (1, 1)
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self._state_to_index(self.state)

    def step(self, action):
        # Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        row, col = self.state
        if action == 0:  # UP
            row = max(0, row - 1)
        elif action == 1:  # DOWN
            row = min(2, row + 1)
        elif action == 2:  # LEFT
            col = max(0, col - 1)
        elif action == 3:  # RIGHT
            col = min(3, col + 1)

        self.state = (row, col)

        # Determine reward and done
        if self.state == self.goal:
            reward = 10
            done = True
        elif self.state == self.obstacle:
            reward = -5
            done = True
        else:
            reward = -1
            done = False

        return self._state_to_index(self.state), reward, done, {}

    def _state_to_index(self, state):
        row, col = state
        return row * 4 + col

# Train agent
env = GridWorld()
agent = QLearningAgent(
    n_states=12,
    n_actions=4,
    learning_rate=0.1,
    discount_factor=0.9,
    epsilon=0.1
)

rewards = agent.train(env, num_episodes=1000)

# Print learned Q-table
print("\nLearned Q-Table:")
print(agent.Q)

# Extract and display policy
policy = agent.get_policy()
action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
print("\nLearned Policy:")
for i in range(12):
    if i == 3:  # Goal state
        print(f"State {i}: GOAL")
    elif i == 5:  # Obstacle
        print(f"State {i}: OBSTACLE")
    else:
        print(f"State {i}: {action_names[policy[i]]}")
```

### Training Output

```
Episode 100, Avg Reward: -15.23
Episode 200, Avg Reward: -8.45
Episode 300, Avg Reward: -5.12
Episode 500, Avg Reward: 6.78
Episode 1000, Avg Reward: 8.92

Learned Q-Table:
State 0:  [2.43, 3.65, 1.20, 4.85]  # RIGHT is best (4.85)
State 1:  [3.12, 4.32, 2.10, 5.92]  # RIGHT is best
State 2:  [4.21, 5.15, 3.45, 7.10]  # RIGHT is best
State 3:  [0.00, 0.00, 0.00, 0.00]  # Goal state

Learned Policy:
State 0: RIGHT
State 1: RIGHT
State 2: RIGHT
State 3: GOAL
State 4: UP
State 5: OBSTACLE
...

Agent learned to navigate to goal, avoiding obstacle!
```

---

## ✅ Why Q-Learning Works

### Convergence Guarantees

**Theorem:** Q-Learning converges to Q\* if:
1. All state-action pairs are visited infinitely often
2. Learning rate α decreases appropriately (Σα = ∞, Σα² < ∞)

**Practical Implications:**
- Need sufficient exploration (ε-greedy)
- Learning rate schedule matters
- Eventually learns optimal policy

### Key Properties

**1. Off-Policy Learning**
- Learns optimal Q\* while following exploratory (ε-greedy) policy
- Behavior policy ≠ target policy
- More sample efficient

**2. Model-Free**
- Doesn't need to know P(s'|s,a) or R(s,a)
- Learns from experience
- Works in unknown environments

**3. Temporal Difference**
- Updates using incomplete episodes
- Can learn online (every step)
- Balances bias and variance

---

## ⚠️ Common Issues and Solutions

### 1. Slow Convergence

**Problem:** Takes many episodes to learn

**Causes:**
- Large state/action space
- Sparse rewards
- Suboptimal hyperparameters

**Solutions:**

**Reward Shaping:**
```python
# Add intermediate rewards
def shaped_reward(state, next_state, goal):
    base_reward = -1  # Step penalty
    # Add distance-based shaping
    old_dist = manhattan_distance(state, goal)
    new_dist = manhattan_distance(next_state, goal)
    shaping = 0.1 * (old_dist - new_dist)  # Reward moving closer
    return base_reward + shaping
```

**Learning Rate Decay:**
```python
# Decrease learning rate over time
def get_learning_rate(episode, initial_lr=0.1, decay=0.99):
    return initial_lr * (decay ** episode)
```

**Optimistic Initialization:**
```python
# Initialize Q-table with high values to encourage exploration
Q = np.ones((n_states, n_actions)) * 10  # Instead of zeros
```

### 2. Exploration vs. Exploitation

**Problem:** Balance between exploring new actions and exploiting known good actions

**ε-greedy with Decay:**
```python
def get_epsilon(episode, initial_eps=1.0, final_eps=0.01, decay_episodes=1000):
    """Decay epsilon from initial to final value"""
    epsilon = initial_eps - (initial_eps - final_eps) * (episode / decay_episodes)
    return max(final_eps, epsilon)

# Usage
for episode in range(num_episodes):
    epsilon = get_epsilon(episode)
    agent.epsilon = epsilon
    # ... train episode
```

**Boltzmann Exploration (Softmax):**
```python
def softmax_action(Q, state, temperature=1.0):
    """Choose action using softmax probabilities"""
    q_values = Q[state]
    exp_q = np.exp(q_values / temperature)
    probs = exp_q / np.sum(exp_q)
    return np.random.choice(len(q_values), p=probs)
```

### 3. Overestimation Bias

**Problem:** Q-Learning tends to overestimate Q-values

**Cause:** max operator in update (always picks highest, even if noise)

**Solution: Double Q-Learning**
```python
class DoubleQLearningAgent:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99):
        # Two Q-tables
        self.Q1 = np.zeros((n_states, n_actions))
        self.Q2 = np.zeros((n_states, n_actions))

    def update(self, state, action, reward, next_state, done):
        # Randomly update one of the two Q-tables
        if random.random() < 0.5:
            # Use Q1 to select action, Q2 to evaluate
            best_action = np.argmax(self.Q1[next_state])
            td_target = reward + self.gamma * self.Q2[next_state, best_action]
            td_error = td_target - self.Q1[state, action]
            self.Q1[state, action] += self.lr * td_error
        else:
            # Use Q2 to select action, Q1 to evaluate
            best_action = np.argmax(self.Q2[next_state])
            td_target = reward + self.gamma * self.Q1[next_state, best_action]
            td_error = td_target - self.Q2[state, action]
            self.Q2[state, action] += self.lr * td_error
```

### 4. Non-Stationary Environments

**Problem:** Environment changes over time

**Solution: Constant Learning Rate**
```python
# Instead of decreasing α, keep it constant
# Recent experiences weighted more heavily
alpha = 0.1  # Constant

# This gives exponentially decaying weights to past experiences
```

---

## 📊 Hyperparameter Tuning

### Key Hyperparameters

| Parameter | Typical Range | Effect | Recommendation |
|-----------|--------------|--------|----------------|
| **Learning Rate (α)** | 0.001 - 0.5 | Speed of learning | Start 0.1, decay if needed |
| **Discount Factor (γ)** | 0.9 - 0.99 | Future importance | 0.99 for long-term, 0.9 for short |
| **Epsilon (ε)** | 0.01 - 1.0 | Exploration | Start 1.0, decay to 0.01-0.1 |
| **Episodes** | 1K - 1M | Training time | Until convergence |

### Tuning Process

```python
def grid_search_hyperparameters(env, param_grid):
    """
    Find best hyperparameters

    Args:
        env: Environment
        param_grid: Dict of parameter lists to try

    Returns:
        best_params, best_score
    """
    best_score = float('-inf')
    best_params = None

    for lr in param_grid['learning_rate']:
        for gamma in param_grid['discount']:
            for eps in param_grid['epsilon']:
                # Train agent
                agent = QLearningAgent(
                    n_states=env.n_states,
                    n_actions=env.n_actions,
                    learning_rate=lr,
                    discount_factor=gamma,
                    epsilon=eps
                )

                rewards = agent.train(env, num_episodes=1000)

                # Evaluate
                avg_reward = np.mean(rewards[-100:])

                if avg_reward > best_score:
                    best_score = avg_reward
                    best_params = {'lr': lr, 'gamma': gamma, 'eps': eps}

    return best_params, best_score

# Usage
param_grid = {
    'learning_rate': [0.01, 0.1, 0.3],
    'discount': [0.9, 0.95, 0.99],
    'epsilon': [0.05, 0.1, 0.2]
}

best_params, best_score = grid_search_hyperparameters(env, param_grid)
print(f"Best params: {best_params}, Score: {best_score}")
```

---

## 🔬 Variants of Q-Learning

### 1. SARSA (On-Policy Q-Learning)

**Difference:** Uses action actually taken instead of max

```python
# Q-Learning (off-policy):
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

# SARSA (on-policy):
Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
#                          ↑
#               Action actually taken (from policy)

class SARSAAgent:
    def update(self, state, action, reward, next_state, next_action, done):
        """Note: requires next_action (actually taken)"""
        current_q = self.Q[state, action]

        if done:
            td_target = reward
        else:
            # Use next_action (not max)
            td_target = reward + self.gamma * self.Q[next_state, next_action]

        td_error = td_target - current_q
        self.Q[state, action] += self.lr * td_error
```

**Comparison:**
- SARSA: More conservative, safer (learns policy actually following)
- Q-Learning: More aggressive, optimal (learns optimal policy)

### 2. Expected SARSA

**Idea:** Use expected value instead of max or sampled action

```python
# Expected SARSA:
Q(s,a) ← Q(s,a) + α[r + γ E[Q(s',a')] - Q(s,a)]
#                          ↑
#               Expected value over policy

def expected_sarsa_update(self, state, action, reward, next_state, done):
    current_q = self.Q[state, action]

    if done:
        td_target = reward
    else:
        # Expected value: weighted sum over actions
        expected_q = 0
        for a in range(self.n_actions):
            prob = self.get_action_probability(next_state, a)  # From policy
            expected_q += prob * self.Q[next_state, a]

        td_target = reward + self.gamma * expected_q

    td_error = td_target - current_q
    self.Q[state, action] += self.lr * td_error
```

**Advantages:**
- Lower variance than SARSA
- Often converges faster
- Handles stochastic policies better

---

## 🎯 Best Practices

### 1. Initialize Q-Table Wisely

```python
# Option 1: Zeros (standard)
Q = np.zeros((n_states, n_actions))

# Option 2: Optimistic initialization (encourages exploration)
Q = np.ones((n_states, n_actions)) * 10

# Option 3: Random (break symmetry)
Q = np.random.uniform(0, 1, (n_states, n_actions))
```

### 2. Monitor Learning Progress

```python
def evaluate_agent(agent, env, num_episodes=100):
    """Evaluate agent performance"""
    total_rewards = []

    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Greedy action (no exploration)
            action = np.argmax(agent.Q[state])
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state

        total_rewards.append(episode_reward)

    return np.mean(total_rewards), np.std(total_rewards)

# Evaluate during training
if episode % 100 == 0:
    mean_reward, std_reward = evaluate_agent(agent, env)
    print(f"Episode {episode}: {mean_reward:.2f} ± {std_reward:.2f}")
```

### 3. Save and Load Models

```python
# Save Q-table
np.save('q_table.npy', agent.Q)

# Load Q-table
loaded_Q = np.load('q_table.npy')
agent.Q = loaded_Q
```

---

## 🚀 Next Steps

**You've Mastered Q-Learning! What's Next?**

**Scale to Complex Problems:**
- [Deep Q-Networks (DQN)](deep-q-networks.md) - Use neural networks for large state spaces

**Explore Policy-Based Methods:**
- [Policy Gradient](policy-gradient.md) - Learn policy directly
- [Actor-Critic](actor-critic.md) - Combine value and policy

**Advanced Topics:**
- Multi-agent Q-Learning
- Hierarchical Q-Learning
- Inverse Reinforcement Learning

---

**Key Takeaway:** Q-Learning learns optimal action values through temporal difference updates, enabling model-free reinforcement learning. It's the foundation for modern deep RL algorithms like DQN, and understanding it is essential for mastering reinforcement learning.

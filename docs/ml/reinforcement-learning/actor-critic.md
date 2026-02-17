# Actor-Critic Methods

**Combine value-based and policy-based approaches for stable and efficient learning.**

Actor-Critic methods use two networks: an actor that learns the policy and a critic that evaluates actions.

---

## 🎯 The Actor-Critic Architecture

### Two Components

**Actor (Policy Network):**
- Learns policy π(a|s; θ)
- Selects actions
- Updated using policy gradient

**Critic (Value Network):**
- Learns value function V(s; w) or Q(s,a; w)
- Evaluates actions
- Updated using TD learning

**Why Both?**
- **Actor alone:** High variance (REINFORCE)
- **Critic alone:** Biased, can't handle continuous actions
- **Together:** Lower variance + unbiased + continuous actions ✅

---

## 🧠 Basic Actor-Critic Algorithm

### Algorithm

```
Initialize actor π(a|s; θ) and critic V(s; w)

For each episode:
    Initialize state s

    For each time step:
        # Actor: Select action
        a ~ π(·|s; θ)

        # Environment: Take action
        Take action a, observe r, s'

        # Critic: Evaluate
        δ = r + γV(s'; w) - V(s; w)  # TD error

        # Update critic
        w ← w + α_w δ ∇_w V(s; w)

        # Update actor
        θ ← θ + α_θ δ ∇_θ log π(a|s; θ)
```

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Actor network (policy)

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dim: Hidden layer size
        """
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """
        Forward pass

        Args:
            state: State tensor

        Returns:
            Action probabilities
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        """
        Critic network (value function)

        Args:
            state_dim: State space dimension
            hidden_dim: Hidden layer size
        """
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        Forward pass

        Args:
            state: State tensor

        Returns:
            State value
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class ActorCritic:
    def __init__(self, state_dim, action_dim,
                 actor_lr=0.001, critic_lr=0.001, gamma=0.99):
        """
        Actor-Critic agent

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            gamma: Discount factor
        """
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma

    def select_action(self, state):
        """
        Select action using actor

        Args:
            state: Current state

        Returns:
            action, log_prob
        """
        state_t = torch.FloatTensor(state)
        probs = self.actor(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def update(self, state, action, reward, next_state, done, log_prob):
        """
        Update actor and critic

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            log_prob: Log probability of action
        """
        state_t = torch.FloatTensor(state)
        next_state_t = torch.FloatTensor(next_state)

        # Critic: Calculate TD error
        value = self.critic(state_t)
        next_value = self.critic(next_state_t)

        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * next_value.item()

        td_error = td_target - value.item()

        # Update critic
        critic_loss = F.mse_loss(value, torch.FloatTensor([td_target]))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -log_prob * td_error  # Policy gradient

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def train(self, env, num_episodes=1000):
        """Train actor-critic agent"""
        episode_rewards = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Select action
                action, log_prob = self.select_action(state)

                # Take action
                next_state, reward, done, _ = env.step(action)

                # Update
                actor_loss, critic_loss = self.update(
                    state, action, reward, next_state, done, log_prob
                )

                state = next_state
                episode_reward += reward

            episode_rewards.append(episode_reward)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode+1}, Avg Reward: {avg_reward:.2f}")

        return episode_rewards
```

---

## 🎓 Advantage Actor-Critic (A2C)

### Advantage Function

**Instead of TD error, use advantage:**

```
Advantage: A(s,a) = Q(s,a) - V(s)
                  ≈ r + γV(s') - V(s)  (TD estimate)
```

**Why Advantage?**
- Reduces variance
- Tells us how much better action a is compared to average

### A2C Implementation

```python
class A2C:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )

        self.gamma = gamma

    def update(self, states, actions, rewards, next_states, dones, log_probs):
        """
        Update using batch of transitions

        Args:
            states, actions, rewards, next_states, dones: Batch
            log_probs: Log probabilities of actions
        """
        states_t = torch.FloatTensor(states)
        next_states_t = torch.FloatTensor(next_states)
        rewards_t = torch.FloatTensor(rewards)
        dones_t = torch.FloatTensor(dones)

        # Critic: Compute values
        values = self.critic(states_t).squeeze()
        next_values = self.critic(next_states_t).squeeze()

        # Compute advantages
        td_targets = rewards_t + self.gamma * next_values * (1 - dones_t)
        advantages = td_targets - values

        # Actor loss (policy gradient with advantage)
        actor_loss = -(torch.stack(log_probs) * advantages.detach()).mean()

        # Critic loss (TD error)
        critic_loss = F.mse_loss(values, td_targets.detach())

        # Total loss
        loss = actor_loss + 0.5 * critic_loss

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            max_norm=0.5
        )
        self.optimizer.step()

        return actor_loss.item(), critic_loss.item()
```

---

## 🔥 Advanced: A3C (Asynchronous Advantage Actor-Critic)

### Key Idea

**Parallel Training:**
- Multiple agents explore environment in parallel
- Each agent updates global network asynchronously
- Decorrelates experiences (like experience replay)

### Architecture

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Agent 1 │     │ Agent 2 │ ... │ Agent N │
└────┬────┘     └────┬────┘     └────┬────┘
     │               │               │
     └───────────────┼───────────────┘
                     ↓
             ┌──────────────┐
             │ Global Model │
             └──────────────┘
```

### Benefits

- **Parallelism:** Faster training
- **Exploration:** Different agents explore differently
- **Stability:** Decorrelates samples
- **No replay buffer:** Saves memory

---

## 🚀 PPO (Proximal Policy Optimization)

### The Gold Standard

**Why PPO?**
- State-of-the-art performance
- Simple to implement
- Stable training
- Works across many domains

### Key Ideas

**1. Clipped Objective:**
```
L^{CLIP}(θ) = E[min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)]

Where:
  r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (probability ratio)
  ε = 0.2 (typical clipping range)
```

**2. Multiple Epochs:**
- Reuse each batch multiple times (e.g., 4-10 epochs)
- More sample efficient

**3. Value Function Loss:**
```
L^{VF} = (V_θ(s_t) - V^{target}_t)²
```

### PPO Implementation

```python
class PPO:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99,
                 epsilon=0.2, epochs=10, batch_size=64):
        """
        PPO agent

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            lr: Learning rate
            gamma: Discount factor
            epsilon: Clipping parameter
            epochs: Update epochs per batch
            batch_size: Batch size
        """
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )

        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.batch_size = batch_size

        # Storage
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def select_action(self, state):
        """Select action and store data"""
        state_t = torch.FloatTensor(state)

        with torch.no_grad():
            probs = self.actor(state_t)
            value = self.critic(state_t)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        self.states.append(state)
        self.actions.append(action.item())
        self.log_probs.append(log_prob)
        self.values.append(value.item())

        return action.item()

    def store_reward(self, reward, done):
        """Store reward and done"""
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_gae(self, next_value):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        lambda_ = 0.95

        values = self.values + [next_value]

        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t+1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * lambda_ * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(self.values)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(self, next_state):
        """PPO update"""
        # Compute advantages
        next_value = self.critic(torch.FloatTensor(next_state)).item()
        advantages, returns = self.compute_gae(next_value)

        # Convert to tensors
        states = torch.FloatTensor(self.states)
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.stack(self.log_probs).detach()

        # PPO epochs
        for _ in range(self.epochs):
            # Current policy
            probs = self.actor(states)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # Probability ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            values = self.critic(states).squeeze()
            critic_loss = F.mse_loss(values, returns)

            # Total loss
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                max_norm=0.5
            )
            self.optimizer.step()

        # Clear storage
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

        return loss.item()
```

---

## 🎯 Continuous Actions: DDPG & SAC

### DDPG (Deep Deterministic Policy Gradient)

**For continuous action spaces:**

```python
class DDPGActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(DDPGActor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x * self.max_action  # Scale to action range

class DDPGCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DDPGCritic, self).__init__()

        # Q(s,a)
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value
```

### SAC (Soft Actor-Critic)

**State-of-the-art for continuous control:**

**Key Features:**
- Maximum entropy objective (encourages exploration)
- Off-policy (sample efficient)
- Stable training
- Automatic temperature tuning

---

## 📊 Comparison of Actor-Critic Methods

| Algorithm | On/Off-Policy | Continuous | Parallel | Difficulty | Performance |
|-----------|---------------|------------|----------|------------|-------------|
| **A2C** | On-policy | ✅ | ❌ | 🟡 Medium | Good |
| **A3C** | On-policy | ✅ | ✅ | 🔴 Hard | Good |
| **PPO** | On-policy | ✅ | ✅ | 🟡 Medium | Excellent |
| **DDPG** | Off-policy | ✅ | ❌ | 🔴 Hard | Good |
| **SAC** | Off-policy | ✅ | ❌ | 🔴 Hard | Excellent |

**Recommendations:**
- **Start with:** A2C (simple, on-policy)
- **For production:** PPO (best balance)
- **For continuous:** SAC (state-of-the-art)

---

## 🎯 Best Practices

### 1. Hyperparameters

```python
# PPO (good defaults)
config = {
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'value_loss_coef': 0.5,
    'entropy_coef': 0.01,
    'max_grad_norm': 0.5,
    'ppo_epochs': 10,
    'batch_size': 64,
}
```

### 2. Normalize Observations

```python
class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean += delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)
```

### 3. Monitor Training

```python
# Track metrics
wandb.log({
    'episode_reward': episode_reward,
    'actor_loss': actor_loss,
    'critic_loss': critic_loss,
    'entropy': entropy,
    'value_estimate': value.mean(),
})
```

---

## 🚀 Summary

**Actor-Critic combines best of both worlds:**
- **Actor:** Learns policy (like policy gradient)
- **Critic:** Learns value (like Q-learning)
- **Result:** Lower variance, more stable, handles continuous actions

**Progression:**
1. Basic Actor-Critic
2. A2C (with advantages)
3. PPO (clipped objective, gold standard)
4. SAC (continuous control, maximum entropy)

---

**Key Takeaway:** Actor-Critic methods unite value-based and policy-based RL, using a critic to reduce variance while the actor learns optimal policies. PPO is the current gold standard for most applications.

# Deep Q-Networks (DQN)

**Scale Q-Learning to complex problems using neural networks to approximate Q-values.**

DQN extends Q-Learning to handle large state spaces (like images) by using deep neural networks as function approximators.

---

## 🎯 The Problem with Tabular Q-Learning

### Limitation: State Space Explosion

**Tabular Q-Learning:**
- Stores Q(s,a) in a table
- One entry per state-action pair
- Works for small discrete state spaces

**Problems:**
| Problem | States | Q-Table Size |
|---------|--------|--------------|
| Grid World (4x4) | 16 | 16 × 4 = 64 |
| Chess | ~10^43 | Impossible! |
| Atari (210×160 RGB) | ~256^(210×160×3) | Astronomical! |
| Continuous (robot arm) | ∞ | Infinite |

**Solution:** Approximate Q(s,a) with a neural network!

---

## 🧠 Deep Q-Network Architecture

### Function Approximation

**Instead of Q-table:**
```python
Q = np.zeros((n_states, n_actions))  # Table
Q[state, action]  # Lookup
```

**Use neural network:**
```python
Q_network = NeuralNetwork()  # Function approximator
Q_values = Q_network(state)  # Forward pass
Q_value = Q_values[action]  # Select action
```

### Network Architecture

**Input → Hidden Layers → Output**

```
State (e.g., image)
       ↓
   Convolution Layer 1
       ↓
   Convolution Layer 2
       ↓
   Convolution Layer 3
       ↓
   Fully Connected Layer
       ↓
   Output Layer
   [Q(s,a₀), Q(s,a₁), ..., Q(s,aₙ)]
```

**Example: Atari DQN**
```python
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        """
        Deep Q-Network for Atari games

        Args:
            input_shape: (C, H, W) - e.g., (4, 84, 84) for 4 stacked frames
            n_actions: Number of possible actions
        """
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            # Conv layer 1: (4, 84, 84) -> (32, 20, 20)
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # Conv layer 2: (32, 20, 20) -> (64, 9, 9)
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Conv layer 3: (64, 9, 9) -> (64, 7, 7)
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        """Calculate flattened size after conv layers"""
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """
        Forward pass

        Args:
            x: State tensor (batch, C, H, W)

        Returns:
            Q-values for all actions (batch, n_actions)
        """
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
```

---

## ⚡ Key Innovations in DQN

### 1. Experience Replay

**Problem:** Correlated samples lead to instability

**Example:**
```
Sequential experience:
  s₁ → s₂ → s₃ → s₄  (highly correlated!)
  Training on this sequence → unstable learning
```

**Solution: Experience Replay Buffer**
```python
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=100000):
        """
        Store and sample past experiences

        Args:
            capacity: Maximum buffer size
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store transition"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample random batch"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions),
                np.array(rewards), np.array(next_states),
                np.array(dones))

    def __len__(self):
        return len(self.buffer)
```

**Benefits:**
- Breaks correlation between consecutive samples
- Improves sample efficiency (reuse experiences)
- More stable training

### 2. Target Network

**Problem:** Moving target issue

**Q-Learning Update:**
```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
                      └──────┬──────┘
                          Target
```

**Issue:** Both Q(s,a) and target use same network
- Update Q → changes target → unstable!

**Solution: Separate Target Network**
```python
class DQNAgent:
    def __init__(self, state_shape, n_actions):
        # Online network (updated every step)
        self.q_network = DQN(state_shape, n_actions)

        # Target network (updated periodically)
        self.target_network = DQN(state_shape, n_actions)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters())

    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

# During training:
if step % TARGET_UPDATE_FREQUENCY == 0:
    agent.update_target_network()
```

**Update Rule with Target Network:**
```
Q(s,a; θ) ← Q(s,a; θ) + α[r + γ max_a' Q(s',a'; θ⁻) - Q(s,a; θ)]
                                              └─┬─┘
                                          Target network
                                         (frozen weights)
```

---

## 🎮 Complete DQN Algorithm

### Pseudocode

```
Initialize replay buffer D with capacity N
Initialize Q-network with random weights θ
Initialize target network with weights θ⁻ = θ

For episode = 1 to M:
    Initialize state s₁

    For t = 1 to T:
        With probability ε: select random action aₜ
        Otherwise: aₜ = argmax_a Q(sₜ, a; θ)

        Execute aₜ, observe reward rₜ and next state sₜ₊₁

        Store transition (sₜ, aₜ, rₜ, sₜ₊₁) in D

        Sample random minibatch from D: {(sⱼ, aⱼ, rⱼ, sⱼ₊₁)}

        For each transition j:
            yⱼ = rⱼ                                    if sⱼ₊₁ is terminal
            yⱼ = rⱼ + γ max_a' Q(sⱼ₊₁, a'; θ⁻)       otherwise

        Perform gradient descent on (yⱼ - Q(sⱼ, aⱼ; θ))²

        Every C steps: θ⁻ ← θ
```

### PyTorch Implementation

```python
import torch
import torch.nn.functional as F

class DQNAgent:
    def __init__(self, state_shape, n_actions, lr=0.0001, gamma=0.99,
                 epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=10000,
                 buffer_size=100000, batch_size=32, target_update_freq=1000):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Networks
        self.q_network = DQN(state_shape, n_actions).cuda()
        self.target_network = DQN(state_shape, n_actions).cuda()
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)

        # Epsilon schedule
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

    def get_epsilon(self):
        """Linearly decay epsilon"""
        epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                  np.exp(-self.steps_done / self.epsilon_decay)
        return epsilon

    def select_action(self, state):
        """ε-greedy action selection"""
        epsilon = self.get_epsilon()

        if random.random() > epsilon:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).cuda()
                q_values = self.q_network(state_t)
                action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.n_actions)

        self.steps_done += 1
        return action

    def train_step(self):
        """Perform one training step"""
        if len(self.buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).cuda()
        actions = torch.LongTensor(actions).cuda()
        rewards = torch.FloatTensor(rewards).cuda()
        next_states = torch.FloatTensor(next_states).cuda()
        dones = torch.FloatTensor(dones).cuda()

        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Target Q-values
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()

        # Update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def train(self, env, num_episodes=1000):
        """Train DQN agent"""
        episode_rewards = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Select and perform action
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)

                # Store transition
                self.buffer.push(state, action, reward, next_state, done)

                # Train
                loss = self.train_step()

                # Update state
                state = next_state
                episode_reward += reward

            episode_rewards.append(episode_reward)

            # Log progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                epsilon = self.get_epsilon()
                print(f"Episode {episode + 1}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {epsilon:.3f}")

        return episode_rewards
```

---

## 🔬 DQN Improvements

### 1. Double DQN (DDQN)

**Problem:** DQN overestimates Q-values

**Cause:** max operator (selects highest, even if noise)

**Solution:** Decouple action selection and evaluation
```python
# DQN:
target = r + γ max_a' Q(s', a'; θ⁻)

# Double DQN:
a* = argmax_a' Q(s', a'; θ)      # Select using online network
target = r + γ Q(s', a*; θ⁻)      # Evaluate using target network
```

**Implementation:**
```python
# In train_step():
with torch.no_grad():
    # Select actions using online network
    next_actions = self.q_network(next_states).max(1)[1].unsqueeze(1)
    # Evaluate using target network
    max_next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze()
    target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
```

### 2. Dueling DQN

**Idea:** Separate value and advantage streams

**Architecture:**
```
State → Conv Layers → Fully Connected
                            ↓
                    ┌───────┴────────┐
                    ↓                ↓
               Value V(s)    Advantage A(s,a)
                    ↓                ↓
                    └───────┬────────┘
                           Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
```

**Implementation:**
```python
class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()

        self.conv = nn.Sequential(...)  # Convolutional layers

        conv_out_size = self._get_conv_out(input_shape)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # V(s)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)  # A(s,a)
        )

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)

        value = self.value_stream(conv_out)
        advantage = self.advantage_stream(conv_out)

        # Combine: Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values
```

### 3. Prioritized Experience Replay

**Idea:** Sample important transitions more frequently

**Priority:** Based on TD error (how surprising the transition is)
```python
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        """
        Args:
            capacity: Buffer size
            alpha: Priority exponent (0 = uniform, 1 = fully prioritized)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def push(self, transition):
        """Store transition with max priority"""
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        Sample batch based on priorities

        Args:
            batch_size: Number of samples
            beta: Importance sampling exponent

        Returns:
            batch, indices, weights
        """
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize

        batch = [self.buffer[idx] for idx in indices]

        return batch, indices, weights

    def update_priorities(self, indices, priorities):
        """Update priorities based on TD errors"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Small constant to avoid zero
```

---

## 🎯 Best Practices

### 1. Preprocessing (Atari)

```python
import cv2

def preprocess_frame(frame):
    """
    Preprocess Atari frame

    Args:
        frame: RGB frame (210, 160, 3)

    Returns:
        Processed frame (84, 84, 1)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Resize
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized / 255.0  # Normalize to [0, 1]

def stack_frames(stacked_frames, new_frame, is_new_episode):
    """
    Stack frames for temporal information

    Args:
        stacked_frames: Deque of past frames
        new_frame: New frame to add
        is_new_episode: Whether starting new episode

    Returns:
        Stacked frames (4, 84, 84)
    """
    frame = preprocess_frame(new_frame)

    if is_new_episode:
        # Stack same frame 4 times
        stacked_frames = deque([frame] * 4, maxlen=4)
    else:
        # Append new frame
        stacked_frames.append(frame)

    return np.stack(stacked_frames, axis=0)
```

### 2. Hyperparameters

**Recommended Values (Atari):**
```python
config = {
    'learning_rate': 0.00025,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_final': 0.01,
    'epsilon_decay_steps': 1000000,
    'batch_size': 32,
    'buffer_size': 100000,
    'target_update_freq': 10000,
    'learning_starts': 50000,  # Steps before training
    'train_freq': 4,  # Train every 4 steps
}
```

### 3. Monitoring

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def log_metrics(episode, reward, loss, epsilon):
    writer.add_scalar('Reward/episode', reward, episode)
    writer.add_scalar('Loss/episode', loss, episode)
    writer.add_scalar('Epsilon', epsilon, episode)
```

---

## 🚀 Next Steps

[→ Policy Gradient Methods](policy-gradient.md){ .md-button }
[→ Actor-Critic Algorithms](actor-critic.md){ .md-button }

**Key Takeaway:** DQN scales Q-Learning to complex problems using neural networks, experience replay, and target networks, enabling agents to learn directly from high-dimensional inputs like images.

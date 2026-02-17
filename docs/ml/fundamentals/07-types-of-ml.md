# Chapter 7: Types of Machine Learning

Machine learning isn't a monolithic field with a single approach to learning from data. Instead, it encompasses various paradigms, each suited to different types of problems and data scenarios. Understanding these different types helps you choose the right approach for your specific challenge.

In this chapter, we explore the major categories of machine learning, from the well-established supervised and unsupervised learning to the more specialized reinforcement learning and emerging paradigms. We'll also examine ensemble methods that combine multiple models to achieve better performance than any single model alone.

## 7.1 Overview of Learning Paradigms

The taxonomy of machine learning is primarily organized around the type of feedback available during learning. This feedback determines how the model learns and what kind of problems it can solve.

### The Fundamental Question: What Guides Learning?

At the heart of each learning paradigm is a fundamental question about feedback:

**Supervised Learning** asks: "Given examples with correct answers, can we learn to predict the answer for new examples?" The learning is guided by explicit labels or targets that tell the model what the correct output should be.

**Unsupervised Learning** asks: "Given only the data itself, can we discover hidden patterns or structure?" There are no labels to guide learning—the model must find meaningful patterns on its own.

**Reinforcement Learning** asks: "Given feedback about outcomes rather than correct actions, can we learn a strategy to maximize rewards?" The learning is guided by delayed feedback about whether actions were good or bad.

**Semi-Supervised Learning** occupies the middle ground: "Can we use a small amount of labeled data combined with lots of unlabeled data?" This approach acknowledges that labels are often expensive while unlabeled data is abundant.

### The Learning Spectrum

We can visualize these paradigms on a spectrum based on the type and amount of supervision:

```text
No Supervision ←──────────────────────────────────────→ Full Supervision

Unsupervised          Semi-Supervised           Supervised
    |                       |                        |
    |                       |                        |
Clustering        Few labeled examples      Every example labeled
Dimensionality    + Many unlabeled         Classification
Reduction         Positive + unlabeled     Regression
Association       Self-supervision


                  Reinforcement Learning
                  (Different axis: delayed rewards)
```

Each position on this spectrum comes with different assumptions, requirements, and appropriate problem types. Let's explore each paradigm in detail.

## 7.2 Supervised Learning

Supervised learning is the most common and well-understood form of machine learning. It's called "supervised" because the learning process is guided by a supervisor (the training labels) that tells the model what the correct output should be.

### The Core Concept

In supervised learning, we have a dataset where each input example x is paired with its corresponding output y. The goal is to learn a function f that maps inputs to outputs: f(x) = ŷ, such that ŷ is as close as possible to the true y.

```python
# Supervised learning structure
training_data = [
    (x₁, y₁),  # Input x₁ paired with correct output y₁
    (x₂, y₂),
    (x₃, y₃),
    ...
    (xₙ, yₙ)
]

# Learning: Find function f such that f(xᵢ) ≈ yᵢ
model.fit(X_train, y_train)

# Prediction: Apply learned function to new inputs
y_new = model.predict(x_new)
```

The "supervision" comes from having these correct outputs (labels) during training. The model can compare its predictions to the true labels and adjust itself to reduce the difference.

### Two Flavors: Classification vs Regression

Supervised learning splits into two major categories based on the type of output being predicted:

#### Classification: Predicting Categories

Classification tasks involve predicting which category or class an example belongs to. The output y is discrete—it takes on one of a finite set of possible values.

**Binary Classification** involves two classes (yes/no, positive/negative, spam/not-spam):

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate binary classification data
X, y = make_classification(n_samples=1000, n_features=20,
                           n_classes=2, random_state=42)

# Train a binary classifier
classifier = LogisticRegression()
classifier.fit(X, y)

# Predict class (0 or 1)
prediction = classifier.predict(X_new)  # Returns: 0 or 1

# Get probability of each class
probabilities = classifier.predict_proba(X_new)  # Returns: [0.2, 0.8]
```

**Multi-class Classification** involves more than two classes:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Iris dataset: 3 classes (setosa, versicolor, virginica)
iris = load_iris()
X, y = iris.data, iris.target

# Train multi-class classifier
classifier = RandomForestClassifier()
classifier.fit(X, y)

# Predict class (0, 1, or 2)
prediction = classifier.predict([[5.1, 3.5, 1.4, 0.2]])
print(f"Predicted class: {iris.target_names[prediction[0]]}")
# Output: Predicted class: setosa
```

**Multi-label Classification** allows each example to belong to multiple classes simultaneously (e.g., a movie can be both "action" and "comedy"):

```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier

# Movie genres (multiple labels possible)
X = [[100, 50],   # [action_scenes, funny_scenes]
     [10, 90],
     [80, 70]]
y = [[1, 0],      # [is_action, is_comedy]
     [0, 1],
     [1, 1]]

# Train multi-label classifier
classifier = MultiOutputClassifier(DecisionTreeClassifier())
classifier.fit(X, y)

prediction = classifier.predict([[70, 60]])
# Output: [[1, 1]] (both action and comedy)
```

Common classification algorithms include:

- Logistic Regression (despite the name, it's for classification)
- Decision Trees and Random Forests
- Support Vector Machines (SVM)
- Naive Bayes
- k-Nearest Neighbors (k-NN)
- Neural Networks

#### Regression: Predicting Continuous Values

Regression tasks involve predicting a continuous numerical value. The output y can take on any value within a range (or even the entire real number line).

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# House price prediction (regression)
# Features: [size, bedrooms, age]
X = np.array([
    [1500, 3, 10],
    [2000, 4, 5],
    [1200, 2, 15],
    [1800, 3, 8],
])
y = np.array([300000, 450000, 250000, 380000])  # Prices (continuous)

# Train regressor
regressor = LinearRegression()
regressor.fit(X, y)

# Predict continuous value
new_house = [[1600, 3, 7]]
predicted_price = regressor.predict(new_house)
print(f"Predicted price: ${predicted_price[0]:,.0f}")
# Output: Predicted price: $340,000
```

The key difference: classification outputs are discrete categories, while regression outputs are continuous numbers. This difference affects:

1. **Evaluation metrics**: Classification uses accuracy, precision, recall, F1. Regression uses MSE, RMSE, MAE, R².
2. **Model architectures**: Some models work for both (e.g., neural networks), while others are specialized.
3. **Loss functions**: Classification often uses cross-entropy, while regression uses squared error.

Common regression algorithms include:

- Linear Regression
- Ridge/Lasso Regression (with regularization)
- Decision Trees and Random Forests
- Support Vector Regression (SVR)
- Gradient Boosting (XGBoost, LightGBM)
- Neural Networks

### When to Use Supervised Learning

Supervised learning is appropriate when:

1. **You have labeled data**: Each training example must have a known correct output.
2. **You can define the output**: You know what you want to predict (house price, spam/not-spam, customer segment).
3. **The pattern is learnable**: There exists a relationship between inputs and outputs that can be captured by a model.
4. **You have sufficient examples**: Generally, more labeled examples lead to better performance.

Supervised learning excels at tasks like:

- Image classification (recognizing objects in photos)
- Natural language processing (sentiment analysis, translation)
- Predictive analytics (sales forecasting, demand prediction)
- Medical diagnosis (disease detection from symptoms)
- Fraud detection (identifying fraudulent transactions)

The main limitation is the requirement for labeled data, which can be expensive and time-consuming to obtain. This leads us to other paradigms that require less or different types of supervision.

## 7.3 Unsupervised Learning

Unsupervised learning tackles the challenge of finding structure in data without any labels or correct answers. The model must discover patterns, groupings, or representations purely from the input data itself.

### The Core Concept

Unlike supervised learning where we have input-output pairs (x, y), unsupervised learning works only with inputs x. The goal is to discover hidden structure or patterns in the data:

```python
# Unsupervised learning structure
training_data = [
    x₁,  # Input only—no label
    x₂,
    x₃,
    ...
    xₙ
]

# Learning: Find patterns, groups, or representations
model.fit(X)  # No y provided!

# Application: Transform or cluster new data
clusters = model.predict(X_new)
transformed = model.transform(X_new)
```

The "unsupervised" nature means there's no external feedback telling the model whether its discovered patterns are correct. Instead, the model relies on internal criteria like maximizing similarity within groups or minimizing reconstruction error.

### Major Types of Unsupervised Learning

#### 1. Clustering: Finding Natural Groups

Clustering algorithms partition data into groups (clusters) such that similar examples are grouped together. Each cluster represents a natural grouping in the data.

**K-Means Clustering** is the most popular clustering algorithm:

```python
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Customer segmentation example
# Features: [annual_income, spending_score]
X = np.array([
    [15, 39], [15, 81], [16, 6], [16, 77],
    [17, 40], [17, 76], [18, 6], [18, 94],
    [19, 3], [19, 72], [20, 14], [20, 99],
    # ... more customers
])

# Cluster into 3 segments
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

print(f"Cluster centers:\n{kmeans.cluster_centers_}")
# Output:
# Cluster 0: low income, low spending
# Cluster 1: high income, high spending
# Cluster 2: low income, high spending

# Assign new customer to a cluster
new_customer = [[18, 85]]
segment = kmeans.predict(new_customer)
print(f"Customer belongs to segment: {segment[0]}")
```

**Hierarchical Clustering** builds a tree of clusters:

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Build hierarchy
hierarchical = AgglomerativeClustering(n_clusters=3)
clusters = hierarchical.fit_predict(X)

# Visualize hierarchy (dendrogram)
linkage_matrix = linkage(X, method='ward')
dendrogram(linkage_matrix)
plt.title('Customer Hierarchy')
plt.show()
```

**DBSCAN** (Density-Based Spatial Clustering) finds clusters of arbitrary shape and automatically detects outliers:

```python
from sklearn.cluster import DBSCAN

# DBSCAN doesn't require specifying number of clusters
dbscan = DBSCAN(eps=3, min_samples=2)
clusters = dbscan.fit_predict(X)

# -1 indicates outliers/noise
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = list(clusters).count(-1)
print(f"Found {n_clusters} clusters and {n_noise} outliers")
```

Common clustering applications:

- Customer segmentation (marketing)
- Document organization (grouping similar documents)
- Image segmentation (identifying regions in images)
- Anomaly detection (outliers don't fit any cluster)
- Gene sequence analysis (grouping similar genes)

#### 2. Dimensionality Reduction: Simplifying Data

Dimensionality reduction algorithms transform high-dimensional data into lower-dimensional representations while preserving important structure. This addresses the curse of dimensionality and enables visualization.

**Principal Component Analysis (PCA)** finds orthogonal directions of maximum variance:

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# Digit images: 64 dimensions (8x8 pixels)
digits = load_digits()
X, y = digits.data, digits.target

print(f"Original shape: {X.shape}")  # (1797, 64)

# Reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print(f"Reduced shape: {X_reduced.shape}")  # (1797, 2)
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# Visualize in 2D
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='tab10')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Digits in 2D')
plt.colorbar()
plt.show()
```

We can also choose the number of components based on variance explained:

```python
# Keep components that explain 95% of variance
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)

print(f"Components needed for 95% variance: {pca.n_components_}")
# Output: Components needed for 95% variance: 29
# (Reduced from 64 to 29 dimensions!)
```

**t-SNE** (t-Distributed Stochastic Neighbor Embedding) excels at visualizing high-dimensional data:

```python
from sklearn.manifold import TSNE

# t-SNE for visualization (typically reduce to 2 or 3 dimensions)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10')
plt.title('Digits visualized with t-SNE')
plt.colorbar()
plt.show()
```

**Autoencoders** (neural networks for dimensionality reduction):

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Autoencoder: 64 → 32 → 16 → 32 → 64
input_dim = 64
encoding_dim = 16

# Encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dense(encoding_dim, activation='relu')(encoded)

# Decoder
decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

# Full autoencoder
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train to reconstruct input
autoencoder.fit(X, X, epochs=50, batch_size=32, shuffle=True)

# Extract encoder for dimensionality reduction
encoder = Model(input_layer, encoded)
X_reduced = encoder.predict(X)

print(f"Reduced to {encoding_dim} dimensions")
```

Common dimensionality reduction applications:

- Data visualization (reducing to 2D or 3D for plotting)
- Feature extraction (creating better features for downstream tasks)
- Noise reduction (by removing less important dimensions)
- Compression (storing data more efficiently)
- Preprocessing (before applying supervised learning)

#### 3. Anomaly Detection: Finding Outliers

Anomaly detection identifies data points that deviate significantly from the norm. This is crucial for fraud detection, system monitoring, and quality control.

**Isolation Forest** isolates anomalies by randomly partitioning the data:

```python
from sklearn.ensemble import IsolationForest

# Normal transactions + fraudulent ones
X = np.array([
    [100, 50],    # Normal
    [110, 45],    # Normal
    [95, 55],     # Normal
    [1000, 5],    # Anomaly (unusual pattern)
    [105, 48],    # Normal
])

# Train isolation forest
iso_forest = IsolationForest(contamination=0.2, random_state=42)
predictions = iso_forest.fit_predict(X)

# -1 indicates anomaly, 1 indicates normal
print("Predictions:", predictions)
# Output: [ 1  1  1 -1  1]  (4th transaction flagged as anomaly)

# Get anomaly scores (more negative = more anomalous)
scores = iso_forest.score_samples(X)
print("Anomaly scores:", scores)
```

**One-Class SVM** learns a boundary around normal data:

```python
from sklearn.svm import OneClassSVM

# Train on normal data only
normal_X = X[predictions == 1]
oc_svm = OneClassSVM(nu=0.1)
oc_svm.fit(normal_X)

# Detect anomalies in new data
new_data = np.array([[105, 50], [900, 10]])
anomalies = oc_svm.predict(new_data)
print("Predictions:", anomalies)
# Output: [ 1 -1]  (second point is anomaly)
```

**Statistical Methods** use statistical properties to detect outliers:

```python
from scipy import stats

# Z-score method
z_scores = np.abs(stats.zscore(X))
threshold = 3

# Points with z-score > 3 are outliers
outliers = (z_scores > threshold).any(axis=1)
print("Outlier indices:", np.where(outliers)[0])
```

Common anomaly detection applications:

- Fraud detection (credit cards, insurance claims)
- Network intrusion detection (cybersecurity)
- Manufacturing quality control (defect detection)
- System health monitoring (detecting failures)
- Medical diagnosis (identifying abnormal scans)

### When to Use Unsupervised Learning

Unsupervised learning is appropriate when:

1. **You have no labels**: Obtaining labels is expensive, impossible, or doesn't make sense for your problem.
2. **You want to explore data**: You're not sure what patterns exist and want to discover them.
3. **You need data preprocessing**: Dimensionality reduction or feature extraction before supervised learning.
4. **You're looking for structure**: Natural groupings, hierarchies, or anomalies in your data.

The main challenge is evaluating results. Without ground truth labels, it's hard to know if the discovered patterns are meaningful or just artifacts of the algorithm. Domain knowledge and careful validation are essential.

## 7.4 Reinforcement Learning

Reinforcement learning (RL) takes a fundamentally different approach. Instead of learning from a fixed dataset, an RL agent learns by interacting with an environment, taking actions, and receiving feedback in the form of rewards or penalties.

### The Core Concept

Reinforcement learning is inspired by behavioral psychology: learning through trial and error with feedback about outcomes. An agent exists in an environment and must learn a policy (strategy) that maximizes cumulative rewards over time.

The key components:

- **Agent**: The learner or decision-maker
- **Environment**: The world the agent interacts with
- **State**: The current situation the agent finds itself in
- **Action**: A choice the agent can make
- **Reward**: Immediate feedback (positive or negative) from taking an action
- **Policy**: The agent's strategy (mapping from states to actions)

```text
        ┌─────────────────────────┐
        │                         │
        │        Agent            │
        │                         │
        └───────┬─────────▲───────┘
                │         │
           Action│         │State, Reward
                │         │
        ┌───────▼─────────┴───────┐
        │                         │
        │     Environment         │
        │                         │
        └─────────────────────────┘
```

### The RL Learning Loop

The agent-environment interaction follows this cycle:

1. Agent observes current **state** sₜ
2. Agent selects an **action** aₜ based on its policy
3. Environment transitions to new **state** sₜ₊₁
4. Environment provides **reward** rₜ
5. Agent updates its policy to maximize future rewards
6. Repeat

```python
# Pseudocode for RL loop
state = environment.reset()

for episode in range(num_episodes):
    done = False
    total_reward = 0

    while not done:
        # Agent chooses action based on current state
        action = agent.choose_action(state)

        # Environment responds
        next_state, reward, done, info = environment.step(action)

        # Agent learns from this experience
        agent.learn(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

    print(f"Episode {episode}: Total Reward = {total_reward}")
```

### Key Concepts in Reinforcement Learning

#### Exploration vs Exploitation

The agent faces a fundamental dilemma: should it **exploit** what it knows (take actions that have worked well in the past) or **explore** new actions (to potentially discover better strategies)?

```python
import numpy as np

def epsilon_greedy_policy(Q, state, epsilon=0.1):
    """
    Epsilon-greedy: Explore with probability epsilon,
    exploit (choose best action) otherwise
    """
    if np.random.random() < epsilon:
        # Explore: random action
        return np.random.randint(len(Q[state]))
    else:
        # Exploit: best known action
        return np.argmax(Q[state])
```

#### Credit Assignment Problem

A reward received now might be the result of actions taken many steps ago. How do we assign credit appropriately? This is solved through **discounting**:

```python
# Discounted cumulative reward
# Immediate rewards matter more than distant rewards
gamma = 0.9  # Discount factor

# Future reward at time t:
# R_t = r_t + γ*r_(t+1) + γ²*r_(t+2) + γ³*r_(t+3) + ...
# Decays exponentially with time
```

#### The Value Function

The **value function** V(s) estimates the expected cumulative reward from a state:

```python
# V(s) = Expected cumulative reward starting from state s
# Following the current policy

# Q(s, a) = Expected cumulative reward from state s, taking action a
# Then following the current policy
```

### A Simple Example: Q-Learning

Q-Learning is a classic RL algorithm that learns the value of taking each action in each state:

```python
import numpy as np

# Grid world environment (5x5 grid)
# Agent starts at (0,0), goal at (4,4)
# Actions: 0=up, 1=right, 2=down, 3=left

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        return tuple(self.agent_pos)

    def step(self, action):
        # Move agent
        if action == 0 and self.agent_pos[0] > 0:  # Up
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[1] < self.size-1:  # Right
            self.agent_pos[1] += 1
        elif action == 2 and self.agent_pos[0] < self.size-1:  # Down
            self.agent_pos[0] += 1
        elif action == 3 and self.agent_pos[1] > 0:  # Left
            self.agent_pos[1] -= 1

        state = tuple(self.agent_pos)
        reward = 1.0 if state == (4, 4) else -0.01  # Goal or small penalty
        done = (state == (4, 4))

        return state, reward, done, {}

# Q-Learning algorithm
class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1,
                 discount=0.9, epsilon=0.1):
        self.Q = np.zeros((n_states, n_states, n_actions))
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(4)  # Explore
        else:
            return np.argmax(self.Q[state])  # Exploit

    def learn(self, state, action, reward, next_state, done):
        # Q-Learning update rule
        current_q = self.Q[state][action]

        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.Q[next_state])

        # Update Q-value
        self.Q[state][action] = current_q + self.lr * (target_q - current_q)

# Training loop
env = GridWorld()
agent = QLearningAgent(n_states=5, n_actions=4)

for episode in range(500):
    state = env.reset()
    total_reward = 0

    for step in range(100):
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)

        agent.learn(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if done:
            break

    if episode % 100 == 0:
        print(f"Episode {episode}: Total Reward = {total_reward:.2f}")

# Test learned policy
state = env.reset()
path = [state]
for step in range(20):
    action = np.argmax(agent.Q[state])  # Pure exploitation
    state, reward, done, _ = env.step(action)
    path.append(state)
    if done:
        break

print(f"Learned path: {path}")
# Output: Learned path from (0,0) to (4,4)
```

### Types of RL Algorithms

RL algorithms fall into several categories:

**Value-Based Methods** learn the value function and derive policy from it:

- Q-Learning
- Deep Q-Networks (DQN)
- SARSA (State-Action-Reward-State-Action)

**Policy-Based Methods** directly learn the policy:

- REINFORCE
- Policy Gradient methods
- Actor-Critic (combines value and policy)

**Model-Based Methods** learn a model of the environment:

- Monte Carlo Tree Search (MCTS)
- Dyna-Q

### When to Use Reinforcement Learning

Reinforcement learning is appropriate when:

1. **Sequential decision-making**: The problem involves making a series of decisions over time.
2. **Delayed feedback**: The consequences of actions aren't immediately apparent.
3. **Interactive environment**: You can interact with the environment and receive feedback.
4. **No labeled examples**: You don't have examples of correct actions, just rewards.
5. **Exploration possible**: You can try different strategies without catastrophic consequences.

Common RL applications:

- Game playing (chess, Go, video games)
- Robotics (robot control, manipulation)
- Autonomous vehicles (navigation, driving)
- Resource allocation (data center management, portfolio optimization)
- Personalization (recommendation systems, adaptive interfaces)

The main challenge is sample efficiency—RL often requires many interactions with the environment to learn effective policies.

## 7.5 Other Paradigms

Beyond the main three paradigms, several specialized approaches address specific challenges or scenarios.

### Semi-Supervised Learning

Semi-supervised learning sits between supervised and unsupervised learning. It uses a small amount of labeled data combined with a large amount of unlabeled data.

**Why it matters**: Labeling data is expensive. In many real-world scenarios, you might have thousands of unlabeled examples but only dozens of labeled ones. Semi-supervised learning leverages both.

```python
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

# Mix of labeled and unlabeled data
# Use -1 to indicate unlabeled examples
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [6, 7.5],
              [8, 1], [9, 1.5], [2, 3], [5.5, 7]])
y = np.array([0, 0, 1, 1, 2, 2, -1, -1])  # -1 = unlabeled

# Label propagation: spread labels to unlabeled data
label_prop = LabelPropagation()
label_prop.fit(X, y)

# Predict labels for previously unlabeled data
predicted_labels = label_prop.predict(X)
print(f"Original labels: {y}")
print(f"After propagation: {predicted_labels}")
# Unlabeled examples now have predicted labels based on nearby labeled examples
```

**Common semi-supervised techniques**:

- **Self-training**: Train on labeled data, predict on unlabeled data, add confident predictions to training set, repeat
- **Co-training**: Train multiple models on different views of data, let them teach each other
- **Label propagation**: Spread labels through the data graph based on similarity

Applications include text classification (few labeled documents), image recognition (few labeled images), and medical diagnosis (few expert-labeled cases).

### Self-Supervised Learning

Self-supervised learning creates its own supervision signal from the data itself. It's a form of unsupervised learning that generates pseudo-labels automatically.

**The key idea**: Create a pretext task where labels can be automatically generated, then use the learned representations for downstream tasks.

```python
# Example: Self-supervised learning for images
# Pretext task: Predict image rotation

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D

# Rotate images by 0, 90, 180, 270 degrees
# Label: which rotation was applied (0, 1, 2, or 3)

def create_rotated_dataset(images):
    rotated_images = []
    labels = []

    for img in images:
        for rotation in range(4):  # 0, 90, 180, 270 degrees
            rotated = tf.image.rot90(img, k=rotation)
            rotated_images.append(rotated)
            labels.append(rotation)  # Automatic label!

    return np.array(rotated_images), np.array(labels)

# Train model to predict rotation
# The model learns useful image features in the process
model = tf.keras.Sequential([
    Conv2D(32, 3, activation='relu'),
    Conv2D(64, 3, activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')  # 4 rotations
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(rotated_images, labels, epochs=10)

# Now extract learned features (remove last layer)
# These features can be used for downstream tasks like classification
feature_extractor = tf.keras.Model(
    inputs=model.input,
    outputs=model.layers[-2].output  # Second to last layer
)
```

**Common pretext tasks**:

- Image rotation prediction
- Image colorization (predict color from grayscale)
- Image inpainting (fill in missing patches)
- Next sentence prediction (NLP)
- Masked language modeling (NLP)

Self-supervised learning powers modern foundation models like BERT, GPT, and vision transformers.

### Transfer Learning

Transfer learning applies knowledge learned from one task to a different but related task. Instead of training from scratch, you start with a pre-trained model.

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load pre-trained model (trained on ImageNet)
base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(224, 224, 3))

# Freeze base model layers (don't retrain them)
base_model.trainable = False

# Add custom layers for your specific task
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(10, activation='softmax')(x)  # 10 classes for your task

# New model: pre-trained base + your custom head
model = Model(inputs=base_model.input, outputs=output)

# Train only the custom layers on your data
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(your_data, your_labels, epochs=10)

# Optionally: Fine-tune some base layers later
base_model.trainable = True
# Freeze early layers, train later layers
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy')
model.fit(your_data, your_labels, epochs=5)
```

**Why transfer learning works**: Early layers learn general features (edges, colors, textures) that are useful across tasks. Later layers learn task-specific features.

**Common scenarios**:

- Using ImageNet-trained models for medical image classification
- Using BERT for domain-specific text tasks
- Using speech models across different languages

### Few-Shot and Zero-Shot Learning

Few-shot learning aims to learn from very few examples (typically 1-5) per class. Zero-shot learning goes further: classify classes never seen during training.

**Few-Shot Learning Example**:

```python
# Prototypical networks: Learn a metric space
# Where examples from same class are close together

# Support set: Few examples per class
support_cats = [cat_img_1, cat_img_2]  # 2 examples of cats
support_dogs = [dog_img_1, dog_img_2]  # 2 examples of dogs

# Compute class prototypes (average representation)
cat_prototype = model.encode(support_cats).mean(axis=0)
dog_prototype = model.encode(support_dogs).mean(axis=0)

# Classify new example by finding nearest prototype
new_image = unknown_animal
embedding = model.encode(new_image)

cat_distance = distance(embedding, cat_prototype)
dog_distance = distance(embedding, dog_prototype)

predicted_class = 'cat' if cat_distance < dog_distance else 'dog'
```

**Zero-Shot Learning Example**:

```python
# CLIP: Learns joint image-text embedding space
# Can classify images into categories never seen during training

import clip

model, preprocess = clip.load("ViT-B/32")

# Image never seen before
image = preprocess(Image.open("unknown_animal.jpg"))

# Text descriptions (classes not in training data)
text_descriptions = ["a photo of a quokka",
                     "a photo of a numbat",
                     "a photo of a bilby"]

# Compute similarity between image and each text description
image_features = model.encode_image(image)
text_features = model.encode_text(text_descriptions)

similarity = (image_features @ text_features.T).softmax(dim=-1)
predicted_class = text_descriptions[similarity.argmax()]
```

These approaches are crucial when data is scarce or when new classes emerge frequently.

### Online Learning vs Batch Learning

**Batch Learning** (the default): Train once on entire dataset, then deploy:

```python
# Train once on full dataset
model.fit(X_train, y_train, epochs=100)

# Deploy and use for predictions
# Model doesn't change after deployment
predictions = model.predict(X_test)
```

**Online Learning**: Update the model incrementally as new data arrives:

```python
from sklearn.linear_model import SGDClassifier

# Initialize model
model = SGDClassifier()

# Train incrementally as data arrives
for X_batch, y_batch in data_stream:
    model.partial_fit(X_batch, y_batch, classes=[0, 1])

    # Model is continuously updated
    # Adapts to changes in data distribution over time
```

**When to use online learning**:

- Data is too large to fit in memory (process in batches)
- Data distribution changes over time (concept drift)
- Need to adapt quickly to new patterns
- Real-time learning required

**Examples**: Stock market prediction, fraud detection, personalized recommendation, spam filtering.

## 7.6 Ensemble Methods

Ensemble methods combine multiple models to achieve better performance than any single model. The idea is that a group of "weak learners" can combine to form a "strong learner."

### The Wisdom of Crowds

The core principle: independent models that make different errors can correct each other when combined. This works because:

1. **Reduced variance**: Averaging predictions smooths out individual model fluctuations
2. **Reduced bias**: Different models capture different aspects of the data
3. **Robustness**: Less likely to overfit or make catastrophic errors

```python
# Simple example: Majority voting
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Create individual models
model1 = DecisionTreeClassifier()
model2 = LogisticRegression()
model3 = SVC(probability=True)

# Combine into ensemble
ensemble = VotingClassifier(
    estimators=[('dt', model1), ('lr', model2), ('svc', model3)],
    voting='soft'  # Average probabilities
)

ensemble.fit(X_train, y_train)

# Ensemble prediction combines all three models
accuracy = ensemble.score(X_test, y_test)
```

### Three Main Approaches

#### 1. Bagging (Bootstrap Aggregating)

Bagging trains multiple instances of the same model on different random subsets of the data (with replacement), then averages their predictions.

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Train 100 decision trees on different subsets
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,  # Use 80% of data for each tree
    bootstrap=True,   # Sample with replacement
    random_state=42
)

bagging.fit(X_train, y_train)
predictions = bagging.predict(X_test)

# Each tree sees different data, makes different errors
# Averaging reduces variance
```

**Random Forest** is the most popular bagging method:

```python
from sklearn.ensemble import RandomForestClassifier

# Random Forest = Bagging + Random feature subsets
rf = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    max_features='sqrt',     # Random feature subset per split
    max_depth=10,            # Limit tree depth
    min_samples_split=5,     # Prevent overfit
    random_state=42
)

rf.fit(X_train, y_train)

# Feature importance (which features matter most)
importances = rf.feature_importances_
for feature, importance in zip(feature_names, importances):
    print(f"{feature}: {importance:.3f}")
```

**Why it works**: Each tree overfits in a different way. Averaging many overfitted models cancels out the overfitting.

#### 2. Boosting

Boosting trains models sequentially, where each new model focuses on correcting the errors of previous models.

**AdaBoost** (Adaptive Boosting):

```python
from sklearn.ensemble import AdaBoostClassifier

# Train weak learners sequentially
# Each focuses on examples previous learners got wrong
adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # Weak learner (stump)
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

adaboost.fit(X_train, y_train)
predictions = adaboost.predict(X_test)
```

**Gradient Boosting**:

```python
from sklearn.ensemble import GradientBoostingClassifier

# Train trees to predict residual errors of previous trees
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb.fit(X_train, y_train)
```

**XGBoost** (Extreme Gradient Boosting) is the most popular boosting library:

```python
import xgboost as xgb

# High-performance gradient boosting
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,       # Row sampling
    colsample_bytree=0.8 # Column sampling
)

xgb_model.fit(X_train, y_train)

# XGBoost has many optimizations:
# - Parallel processing
# - Regularization
# - Handling missing values
# - Cross-validation
```

**Why it works**: Each model corrects the previous model's mistakes. The ensemble focuses on hard examples that earlier models struggled with.

#### 3. Stacking

Stacking trains a meta-model to combine predictions from multiple base models.

```python
from sklearn.ensemble import StackingClassifier

# Base models (level 0)
level0 = [
    ('rf', RandomForestClassifier(n_estimators=10)),
    ('svm', SVC(probability=True)),
    ('lr', LogisticRegression())
]

# Meta-model (level 1)
level1 = LogisticRegression()

# Stacking ensemble
stacking = StackingClassifier(
    estimators=level0,
    final_estimator=level1,
    cv=5  # Use cross-validation to generate meta-features
)

stacking.fit(X_train, y_train)

# Process:
# 1. Train base models on training data
# 2. Use base model predictions as features for meta-model
# 3. Meta-model learns optimal combination
```

**Why it works**: The meta-model learns which base models to trust in different situations.

### Comparing Ensemble Methods

| Method | Training | Strengths | Weaknesses |
|--------|----------|-----------|------------|
| **Bagging** | Parallel | Reduces variance, easy to parallelize | May not reduce bias |
| **Boosting** | Sequential | Reduces bias and variance, high accuracy | Prone to overfitting, sequential |
| **Stacking** | Two-level | Very flexible, can combine diverse models | Complex, requires more data |

### When to Use Ensemble Methods

Use ensemble methods when:

1. **You want maximum accuracy**: Ensembles often win Kaggle competitions
2. **You have computational resources**: Training multiple models is expensive
3. **Interpretability isn't critical**: Understanding ensemble decisions is harder
4. **Your base models are diverse**: More diversity leads to better ensembles

**Practical tips**:

- **Start with Random Forest**: Easy to use, works well out of the box
- **Use XGBoost for structured data**: Often best performance on tabular data
- **Combine diverse models**: Mix tree-based, linear, and neural network models
- **Watch for diminishing returns**: 10 models might be 90% as good as 100

### A Complete Example

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
import xgboost as xgb

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=20,
                           n_informative=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Individual models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)

# Evaluate individual models
print("Individual Model Performance:")
for name, model in [('Random Forest', rf), ('Gradient Boosting', gb), ('XGBoost', xgb_model)]:
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Ensemble (voting)
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('xgb', xgb_model)],
    voting='soft'
)

scores = cross_val_score(ensemble, X_train, y_train, cv=5)
print(f"\nEnsemble: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Train and test
ensemble.fit(X_train, y_train)
accuracy = ensemble.score(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.3f}")
```

Ensemble methods are a cornerstone of modern machine learning, offering a reliable path to improved performance across many problem types.

---

In this chapter, we've explored the major types of machine learning—from the well-established supervised and unsupervised learning to reinforcement learning and specialized paradigms like semi-supervised and transfer learning. We've also seen how ensemble methods combine multiple models for superior performance.

Each paradigm has its place. Supervised learning excels when you have labeled data. Unsupervised learning discovers patterns without labels. Reinforcement learning masters sequential decision-making through trial and error. Semi-supervised and transfer learning make the most of limited labels. And ensembles push performance to its limits by combining diverse models.

Understanding these different types allows you to choose the right approach for your specific problem, data availability, and constraints. In the next chapter, we'll dive into the mathematical foundations that underpin all these learning paradigms.

---

[← Previous: Chapter 6](06-generalization.md) | [Back to Index](index.md) | [Next: Chapter 8 →](08-mathematics.md)

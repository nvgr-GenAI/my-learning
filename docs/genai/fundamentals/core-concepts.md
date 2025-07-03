# Core Concepts in GenAI

!!! info "Essential Building Blocks"
    Master the fundamental concepts that underpin all generative AI systems, from embeddings to attention mechanisms.

## Embeddings and Representation Learning

### What are Embeddings?

Embeddings are dense vector representations that capture semantic relationships between discrete objects (words, images, etc.) in a continuous space.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

# Simple word embedding example
class SimpleWordEmbedding:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Initialize embeddings randomly
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.1
        
        # Word to index mapping
        self.word_to_idx = {}
        self.idx_to_word = {}
    
    def add_word(self, word):
        """Add a word to vocabulary"""
        if word not in self.word_to_idx:
            idx = len(self.word_to_idx)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
    
    def get_embedding(self, word):
        """Get embedding for a word"""
        if word in self.word_to_idx:
            idx = self.word_to_idx[word]
            return self.embeddings[idx]
        return None
    
    def similarity(self, word1, word2):
        """Calculate cosine similarity between two words"""
        emb1 = self.get_embedding(word1)
        emb2 = self.get_embedding(word2)
        
        if emb1 is None or emb2 is None:
            return 0
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        return dot_product / (norm1 * norm2)
    
    def find_similar(self, word, top_k=5):
        """Find most similar words"""
        target_embedding = self.get_embedding(word)
        if target_embedding is None:
            return []
        
        similarities = []
        for other_word in self.word_to_idx:
            if other_word != word:
                sim = self.similarity(word, other_word)
                similarities.append((other_word, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

# Create example embeddings
embedding_model = SimpleWordEmbedding(vocab_size=100, embedding_dim=50)

# Add some words
words = ['king', 'queen', 'man', 'woman', 'prince', 'princess', 
         'cat', 'dog', 'car', 'truck', 'apple', 'orange']

for word in words:
    embedding_model.add_word(word)

# Demonstrate embedding relationships
print("Word Similarities:")
for word in ['king', 'cat', 'car']:
    print(f"\nSimilar to '{word}':")
    similar = embedding_model.find_similar(word, top_k=3)
    for sim_word, sim_score in similar:
        print(f"  {sim_word}: {sim_score:.3f}")
```

### Training Embeddings: Word2Vec

```python
class Word2Vec:
    def __init__(self, vocab_size, embedding_dim, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        
        # Initialize embeddings and context weights
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.1
        self.context_weights = np.random.randn(vocab_size, embedding_dim) * 0.1
        
        self.word_to_idx = {}
        self.idx_to_word = {}
    
    def softmax(self, x):
        """Stable softmax implementation"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def train_pair(self, center_word_idx, context_word_idx):
        """Train on a single word pair (Skip-gram)"""
        # Forward pass
        center_embedding = self.embeddings[center_word_idx]
        
        # Calculate scores for all words
        scores = np.dot(self.context_weights, center_embedding)
        probs = self.softmax(scores)
        
        # Calculate loss (negative log likelihood)
        loss = -np.log(probs[context_word_idx] + 1e-8)
        
        # Backward pass
        # Gradient for context weights
        grad_context = np.outer(probs, center_embedding)
        grad_context[context_word_idx] -= center_embedding
        
        # Gradient for center embedding
        grad_center = np.sum((probs.reshape(-1, 1) * self.context_weights), axis=0)
        grad_center -= self.context_weights[context_word_idx]
        
        # Update weights
        self.context_weights -= self.learning_rate * grad_context
        self.embeddings[center_word_idx] -= self.learning_rate * grad_center
        
        return loss
    
    def generate_training_data(self, sentences, window_size=2):
        """Generate training pairs from sentences"""
        training_pairs = []
        
        for sentence in sentences:
            for i, center_word in enumerate(sentence):
                center_idx = self.word_to_idx.get(center_word)
                if center_idx is None:
                    continue
                
                # Context window
                start = max(0, i - window_size)
                end = min(len(sentence), i + window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        context_word = sentence[j]
                        context_idx = self.word_to_idx.get(context_word)
                        if context_idx is not None:
                            training_pairs.append((center_idx, context_idx))
        
        return training_pairs

# Example training data
sentences = [
    ['the', 'cat', 'sits', 'on', 'the', 'mat'],
    ['the', 'dog', 'runs', 'in', 'the', 'park'],
    ['cats', 'and', 'dogs', 'are', 'pets'],
    ['the', 'king', 'rules', 'the', 'kingdom'],
    ['the', 'queen', 'is', 'royal']
]

# Build vocabulary
vocab = set()
for sentence in sentences:
    vocab.update(sentence)

vocab = list(vocab)
print(f"Vocabulary size: {len(vocab)}")
print(f"Vocabulary: {vocab}")
```

### Modern Embeddings: Transformers

```python
def positional_encoding(seq_length, d_model):
    """Generate sinusoidal positional encodings"""
    pos_encoding = np.zeros((seq_length, d_model))
    
    for pos in range(seq_length):
        for i in range(0, d_model, 2):
            pos_encoding[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pos_encoding[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    
    return pos_encoding

def visualize_positional_encoding():
    """Visualize positional encodings"""
    seq_len = 100
    d_model = 64
    
    pos_enc = positional_encoding(seq_len, d_model)
    
    plt.figure(figsize=(12, 8))
    
    # Plot the positional encoding
    plt.subplot(2, 2, 1)
    plt.imshow(pos_enc.T, cmap='RdYlBu', aspect='auto')
    plt.colorbar()
    plt.title('Positional Encoding Heatmap')
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    
    # Plot specific dimensions
    plt.subplot(2, 2, 2)
    for dim in [0, 1, 4, 8]:
        plt.plot(pos_enc[:, dim], label=f'Dim {dim}')
    plt.title('Positional Encoding for Different Dimensions')
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot similarity between positions
    plt.subplot(2, 2, 3)
    similarity_matrix = np.dot(pos_enc, pos_enc.T)
    plt.imshow(similarity_matrix, cmap='viridis')
    plt.colorbar()
    plt.title('Position Similarity Matrix')
    plt.xlabel('Position')
    plt.ylabel('Position')
    
    # Plot encoding for a specific position
    plt.subplot(2, 2, 4)
    position = 10
    plt.plot(pos_enc[position])
    plt.title(f'Positional Encoding at Position {position}')
    plt.xlabel('Dimension')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_positional_encoding()
```

## Attention Mechanisms

### Self-Attention Explained

```python
class SimpleAttention:
    def __init__(self, d_model):
        self.d_model = d_model
        self.scale = np.sqrt(d_model)
    
    def attention(self, query, key, value, mask=None):
        """
        Compute scaled dot-product attention
        
        Args:
            query: (seq_len, d_model)
            key: (seq_len, d_model) 
            value: (seq_len, d_model)
            mask: (seq_len, seq_len) - optional
        """
        # Compute attention scores
        scores = np.dot(query, key.T) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # Apply softmax
        attention_weights = self.softmax(scores)
        
        # Apply attention to values
        output = np.dot(attention_weights, value)
        
        return output, attention_weights
    
    def softmax(self, x):
        """Softmax along the last dimension"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def demonstrate_attention():
    """Demonstrate self-attention with a simple example"""
    # Create example sequence
    seq_length = 5
    d_model = 8
    
    # Random input embeddings
    np.random.seed(42)
    embeddings = np.random.randn(seq_length, d_model)
    
    # Initialize attention
    attention = SimpleAttention(d_model)
    
    # Self-attention (Q, K, V are all the same)
    output, weights = attention.attention(embeddings, embeddings, embeddings)
    
    print("Input shape:", embeddings.shape)
    print("Output shape:", output.shape)
    print("Attention weights shape:", weights.shape)
    
    # Visualize attention weights
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(weights, cmap='Blues')
    plt.colorbar()
    plt.title('Self-Attention Weights')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    
    # Show how each position attends to others
    plt.subplot(1, 2, 2)
    positions = ['Pos 0', 'Pos 1', 'Pos 2', 'Pos 3', 'Pos 4']
    for i in range(seq_length):
        plt.plot(weights[i], label=f'Query {i}', marker='o')
    plt.title('Attention Patterns by Query Position')
    plt.xlabel('Key Position')
    plt.ylabel('Attention Weight')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return output, weights

attention_output, attention_weights = demonstrate_attention()
```

### Multi-Head Attention

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize projection matrices
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1
    
    def split_heads(self, x):
        """Split input into multiple heads"""
        seq_len, d_model = x.shape
        x = x.reshape(seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 0, 2)  # (num_heads, seq_len, d_k)
    
    def combine_heads(self, x):
        """Combine multiple heads back"""
        num_heads, seq_len, d_k = x.shape
        x = x.transpose(1, 0, 2)  # (seq_len, num_heads, d_k)
        return x.reshape(seq_len, self.d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Attention for a single head"""
        d_k = Q.shape[-1]
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
        
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        attention_weights = self.softmax(scores)
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def softmax(self, x):
        """Softmax along the last dimension"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, x, mask=None):
        """Forward pass through multi-head attention"""
        seq_len = x.shape[0]
        
        # Linear projections
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)
        
        # Split into heads
        Q_heads = self.split_heads(Q)
        K_heads = self.split_heads(K)
        V_heads = self.split_heads(V)
        
        # Apply attention to each head
        head_outputs = []
        head_weights = []
        
        for i in range(self.num_heads):
            head_output, head_weight = self.scaled_dot_product_attention(
                Q_heads[i:i+1], K_heads[i:i+1], V_heads[i:i+1], mask
            )
            head_outputs.append(head_output[0])
            head_weights.append(head_weight[0])
        
        # Combine heads
        combined_output = np.concatenate(head_outputs, axis=-1)
        
        # Final linear projection
        output = np.dot(combined_output, self.W_o)
        
        return output, head_weights

def visualize_multi_head_attention():
    """Visualize multi-head attention patterns"""
    seq_length = 8
    d_model = 64
    num_heads = 8
    
    # Create input
    np.random.seed(42)
    x = np.random.randn(seq_length, d_model)
    
    # Initialize multi-head attention
    mha = MultiHeadAttention(d_model, num_heads)
    
    # Forward pass
    output, head_weights = mha.forward(x)
    
    # Visualize attention patterns for each head
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, head_weight in enumerate(head_weights):
        im = axes[i].imshow(head_weight, cmap='Blues')
        axes[i].set_title(f'Head {i+1}')
        axes[i].set_xlabel('Key Position')
        axes[i].set_ylabel('Query Position')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.show()
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of attention heads: {len(head_weights)}")

visualize_multi_head_attention()
```

## Autoregressive Generation

### Language Modeling Basics

```python
class SimpleLanguageModel:
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.1
        
        # Simple RNN
        self.W_h = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.W_x = np.random.randn(embedding_dim, hidden_dim) * 0.1
        self.b_h = np.zeros(hidden_dim)
        
        # Output layer
        self.W_out = np.random.randn(hidden_dim, vocab_size) * 0.1
        self.b_out = np.zeros(vocab_size)
        
        # Vocabulary
        self.word_to_idx = {}
        self.idx_to_word = {}
    
    def add_word(self, word):
        """Add word to vocabulary"""
        if word not in self.word_to_idx:
            idx = len(self.word_to_idx)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
    
    def forward_step(self, word_idx, hidden_state):
        """Single forward step"""
        # Embedding lookup
        x = self.embeddings[word_idx]
        
        # RNN step
        hidden_state = np.tanh(
            np.dot(x, self.W_x) + np.dot(hidden_state, self.W_h) + self.b_h
        )
        
        # Output projection
        logits = np.dot(hidden_state, self.W_out) + self.b_out
        
        return logits, hidden_state
    
    def predict_next(self, sequence, temperature=1.0):
        """Predict next word given sequence"""
        hidden_state = np.zeros(self.hidden_dim)
        
        # Process sequence
        for word in sequence:
            if word in self.word_to_idx:
                word_idx = self.word_to_idx[word]
                logits, hidden_state = self.forward_step(word_idx, hidden_state)
        
        # Apply temperature
        logits = logits / temperature
        
        # Softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        return probs
    
    def generate_text(self, seed_sequence, max_length=20, temperature=1.0):
        """Generate text autoregressively"""
        generated = list(seed_sequence)
        
        for _ in range(max_length):
            # Predict next word probabilities
            probs = self.predict_next(generated, temperature)
            
            # Sample next word
            next_word_idx = np.random.choice(len(probs), p=probs)
            next_word = self.idx_to_word.get(next_word_idx, '<UNK>')
            
            generated.append(next_word)
            
            # Stop if we generate an end token
            if next_word == '<END>':
                break
        
        return generated

def demonstrate_autoregressive_generation():
    """Demonstrate autoregressive text generation"""
    # Create a simple language model
    lm = SimpleLanguageModel(vocab_size=1000, embedding_dim=50, hidden_dim=100)
    
    # Build vocabulary from sample text
    sample_text = [
        "the cat sits on the mat",
        "the dog runs in the park", 
        "cats and dogs are pets",
        "the sun shines bright",
        "birds fly in the sky"
    ]
    
    vocab = set()
    for sentence in sample_text:
        words = sentence.split()
        vocab.update(words)
    
    vocab.update(['<START>', '<END>', '<UNK>'])
    
    for word in vocab:
        lm.add_word(word)
    
    print(f"Vocabulary size: {len(lm.word_to_idx)}")
    print(f"Sample vocabulary: {list(lm.word_to_idx.keys())[:10]}")
    
    # Generate text with different temperatures
    seed = ['the', 'cat']
    
    print("\nText Generation Examples:")
    for temp in [0.5, 1.0, 1.5]:
        generated = lm.generate_text(seed, max_length=10, temperature=temp)
        print(f"Temperature {temp}: {' '.join(generated)}")

demonstrate_autoregressive_generation()
```

### Sampling Strategies

```python
class SamplingStrategies:
    @staticmethod
    def greedy_sampling(logits):
        """Always pick the most likely token"""
        return np.argmax(logits)
    
    @staticmethod
    def temperature_sampling(logits, temperature=1.0):
        """Sample with temperature scaling"""
        if temperature == 0:
            return SamplingStrategies.greedy_sampling(logits)
        
        scaled_logits = logits / temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        return np.random.choice(len(probs), p=probs)
    
    @staticmethod
    def top_k_sampling(logits, k=10, temperature=1.0):
        """Sample from top-k most likely tokens"""
        # Get top-k indices
        top_k_indices = np.argpartition(logits, -k)[-k:]
        top_k_logits = logits[top_k_indices]
        
        # Apply temperature
        scaled_logits = top_k_logits / temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Sample from top-k
        selected_idx = np.random.choice(len(probs), p=probs)
        return top_k_indices[selected_idx]
    
    @staticmethod
    def top_p_sampling(logits, p=0.9, temperature=1.0):
        """Nucleus sampling - sample from smallest set with cumulative prob >= p"""
        # Apply temperature
        scaled_logits = logits / temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Sort probabilities in descending order
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        
        # Find nucleus (smallest set with cumulative prob >= p)
        cumulative_probs = np.cumsum(sorted_probs)
        nucleus_size = np.argmax(cumulative_probs >= p) + 1
        
        # Sample from nucleus
        nucleus_indices = sorted_indices[:nucleus_size]
        nucleus_probs = sorted_probs[:nucleus_size]
        nucleus_probs = nucleus_probs / np.sum(nucleus_probs)  # Renormalize
        
        selected_idx = np.random.choice(len(nucleus_probs), p=nucleus_probs)
        return nucleus_indices[selected_idx]

def compare_sampling_strategies():
    """Compare different sampling strategies"""
    # Create example logits (simulating model output)
    np.random.seed(42)
    vocab_size = 50
    logits = np.random.randn(vocab_size)
    
    # Make some tokens much more likely
    logits[5] = 3.0   # High probability token
    logits[12] = 2.5  # Second highest
    logits[23] = 2.0  # Third highest
    
    strategies = {
        'Greedy': lambda: SamplingStrategies.greedy_sampling(logits),
        'Temp=0.5': lambda: SamplingStrategies.temperature_sampling(logits, 0.5),
        'Temp=1.0': lambda: SamplingStrategies.temperature_sampling(logits, 1.0),
        'Temp=1.5': lambda: SamplingStrategies.temperature_sampling(logits, 1.5),
        'Top-k=5': lambda: SamplingStrategies.top_k_sampling(logits, k=5),
        'Top-p=0.9': lambda: SamplingStrategies.top_p_sampling(logits, p=0.9)
    }
    
    # Sample multiple times from each strategy
    num_samples = 1000
    results = {}
    
    for name, sampler in strategies.items():
        samples = []
        for _ in range(num_samples):
            samples.append(sampler())
        results[name] = samples
    
    # Visualize distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (name, samples) in enumerate(results.items()):
        # Count frequency of each token
        unique, counts = np.unique(samples, return_counts=True)
        
        axes[i].bar(unique, counts / num_samples)
        axes[i].set_title(f'{name} Sampling')
        axes[i].set_xlabel('Token Index')
        axes[i].set_ylabel('Probability')
        axes[i].set_xlim(0, 30)  # Show first 30 tokens
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("Sampling Strategy Comparison:")
    for name, samples in results.items():
        unique_tokens = len(set(samples))
        most_common = max(set(samples), key=samples.count)
        frequency = samples.count(most_common) / len(samples)
        print(f"{name:12}: {unique_tokens:2d} unique tokens, most common: {most_common:2d} ({frequency:.2%})")

compare_sampling_strategies()
```

## Loss Functions and Training Objectives

### Language Modeling Loss

```python
def cross_entropy_loss(logits, targets):
    """Compute cross-entropy loss for language modeling"""
    # Apply softmax to logits
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    # Extract probabilities for target tokens
    target_probs = probs[np.arange(len(targets)), targets]
    
    # Compute negative log likelihood
    loss = -np.mean(np.log(target_probs + 1e-8))
    
    return loss

def perplexity(logits, targets):
    """Compute perplexity from logits and targets"""
    loss = cross_entropy_loss(logits, targets)
    return np.exp(loss)

# Example calculation
np.random.seed(42)
batch_size = 32
vocab_size = 10000
seq_length = 128

# Simulate model outputs and targets
logits = np.random.randn(batch_size, vocab_size)
targets = np.random.randint(0, vocab_size, batch_size)

loss = cross_entropy_loss(logits, targets)
ppl = perplexity(logits, targets)

print(f"Cross-entropy loss: {loss:.4f}")
print(f"Perplexity: {ppl:.2f}")
```

### Contrastive Learning

```python
class ContrastiveLearning:
    def __init__(self, temperature=0.07):
        self.temperature = temperature
    
    def compute_similarity(self, embeddings1, embeddings2):
        """Compute cosine similarity between embeddings"""
        # Normalize embeddings
        norm1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        
        # Compute similarity matrix
        similarity = np.dot(norm1, norm2.T)
        return similarity
    
    def contrastive_loss(self, anchor_embeddings, positive_embeddings, negative_embeddings=None):
        """Compute contrastive loss (InfoNCE)"""
        batch_size = anchor_embeddings.shape[0]
        
        # If no negative embeddings provided, use other samples in batch
        if negative_embeddings is None:
            # Concatenate positive and anchor embeddings as negatives
            all_embeddings = np.concatenate([positive_embeddings, anchor_embeddings], axis=0)
            
            # Compute similarities
            similarities = self.compute_similarity(anchor_embeddings, all_embeddings)
            
            # Scale by temperature
            similarities = similarities / self.temperature
            
            # Positive similarities are diagonal elements
            positive_similarities = similarities[:, :batch_size]
            positive_scores = np.diag(positive_similarities)
            
            # Compute softmax denominator (all similarities except self-similarity)
            mask = np.eye(batch_size * 2, dtype=bool)
            mask[:batch_size, batch_size:] = True  # Remove self-similarities
            
            exp_similarities = np.exp(similarities)
            exp_similarities[mask] = 0  # Mask out self-similarities
            
            denominator = np.sum(exp_similarities, axis=1)
            
            # Compute loss
            loss = -np.mean(positive_scores - np.log(denominator + 1e-8))
            
        return loss

def demonstrate_contrastive_learning():
    """Demonstrate contrastive learning concepts"""
    # Generate example embeddings
    np.random.seed(42)
    embedding_dim = 128
    batch_size = 64
    
    # Anchor embeddings (original sentences)
    anchor_embeddings = np.random.randn(batch_size, embedding_dim)
    
    # Positive embeddings (paraphrases or augmented versions)
    positive_embeddings = anchor_embeddings + np.random.normal(0, 0.1, (batch_size, embedding_dim))
    
    # Initialize contrastive learning
    cl = ContrastiveLearning(temperature=0.07)
    
    # Compute loss
    loss = cl.contrastive_loss(anchor_embeddings, positive_embeddings)
    
    print(f"Contrastive loss: {loss:.4f}")
    
    # Visualize similarity patterns
    similarities = cl.compute_similarity(anchor_embeddings[:20], positive_embeddings[:20])
    
    plt.figure(figsize=(10, 8))
    plt.imshow(similarities, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Similarity Matrix (Anchor vs Positive)')
    plt.xlabel('Positive Embeddings')
    plt.ylabel('Anchor Embeddings')
    plt.show()
    
    # Show distribution of similarities
    diagonal_similarities = np.diag(similarities)
    off_diagonal_similarities = similarities[~np.eye(similarities.shape[0], dtype=bool)]
    
    plt.figure(figsize=(10, 6))
    plt.hist(diagonal_similarities, bins=20, alpha=0.7, label='Positive Pairs', density=True)
    plt.hist(off_diagonal_similarities, bins=20, alpha=0.7, label='Negative Pairs', density=True)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title('Distribution of Similarities')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

demonstrate_contrastive_learning()
```

---

!!! success "Core Concepts Mastered"
    You now understand the fundamental building blocks that power modern generative AI systems!

!!! tip "Key Takeaways"
    - **Embeddings** capture semantic relationships in continuous vector spaces
    - **Attention mechanisms** allow models to focus on relevant information
    - **Autoregressive generation** enables coherent text generation
    - **Sampling strategies** control the creativity vs consistency trade-off
    - **Contrastive learning** helps models learn meaningful representations

!!! note "Next Steps"
    Ready to explore how these concepts evolved? Continue with **[Evolution of GenAI](evolution.md)** to understand the historical development of generative AI.

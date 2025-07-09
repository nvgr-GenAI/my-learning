# Attention Mechanisms in Transformers

!!! abstract "Core of Modern AI"
    Attention mechanisms are the revolutionary innovation that enables transformers to process sequences in parallel while capturing long-range dependencies.

## What is Attention?

Attention allows models to **selectively focus** on relevant parts of the input when processing each element. Think of it as a **spotlight** that can dynamically highlight the most important information.

### Key Advantages

- **Parallel Processing**: Unlike RNNs, all positions processed simultaneously
- **Long-range Dependencies**: Direct connections between any two positions
- **Interpretability**: Attention weights show what the model focuses on
- **Flexibility**: Can handle variable-length sequences

#### Basic Attention Formula

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def basic_attention(query, key, value):
    """
    Basic attention mechanism
    
    Args:
        query: [batch_size, seq_len, d_k]
        key: [batch_size, seq_len, d_k]
        value: [batch_size, seq_len, d_v]
    
    Returns:
        output: [batch_size, seq_len, d_v]
        attention_weights: [batch_size, seq_len, seq_len]
    """
    # Calculate attention scores
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # Scale by square root of dimension
    d_k = query.size(-1)
    scores = scores / math.sqrt(d_k)
    
    # Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply attention to values
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights
```

### Intuitive Understanding

Think of attention as a spotlight that can illuminate different parts of a sentence:

```python
def attention_visualization():
    """Visualize how attention works with a simple example"""
    
    sentence = "The cat sat on the mat"
    words = sentence.split()
    
    # When processing "sat", attention might focus on:
    attention_weights = {
        "The": 0.1,    # Low attention - articles are less important
        "cat": 0.6,    # High attention - subject doing the action
        "sat": 0.2,    # Some attention to the word itself
        "on": 0.05,    # Low attention - preposition
        "the": 0.02,   # Very low attention
        "mat": 0.03    # Low attention - object but not direct
    }
    
    print("When processing 'sat', attention focuses on:")
    for word, weight in attention_weights.items():
        print(f"  {word}: {weight:.2f} {'█' * int(weight * 20)}")
```

## Self-Attention Mechanism

### Mathematical Foundation

Self-attention allows each position to attend to all positions in the same sequence:

```python
class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k=None):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k or d_model
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, self.d_k, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(self.d_k, d_model, bias=False)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] (optional)
        
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = x.size()
        
        # Create Q, K, V
        Q = self.W_q(x)  # [batch_size, seq_len, d_k]
        K = self.W_k(x)  # [batch_size, seq_len, d_k]
        V = self.W_v(x)  # [batch_size, seq_len, d_k]
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Output projection
        output = self.W_o(context)
        
        return output, attention_weights
```

### Key Properties

1. **Parallelization**: All positions can be computed simultaneously
2. **Long-range dependencies**: Direct connections between any two positions
3. **Interpretability**: Attention weights show which positions the model focuses on

## Multi-Head Attention

### Concept and Implementation

Multi-head attention allows the model to attend to different types of information simultaneously:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers for all heads combined
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # 1. Linear projections and reshape for multi-head
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Apply attention
        context, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 3. Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 4. Final linear projection
        output = self.W_o(context)
        
        return output, attention_weights
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Args:
            Q, K, V: [batch_size, num_heads, seq_len, d_k]
            mask: [batch_size, 1, seq_len, seq_len] or None
        """
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply to values
        context = torch.matmul(attention_weights, V)
        
        return context, attention_weights
```

### Why Multiple Heads?

Each head can focus on different aspects:

```python
def analyze_attention_heads():
    """Example of what different heads might focus on"""
    
    heads_focus = {
        "Head 1": "Syntactic relationships (subject-verb, verb-object)",
        "Head 2": "Semantic similarity (synonyms, related concepts)",
        "Head 3": "Positional patterns (nearby words, sequence structure)",
        "Head 4": "Long-range dependencies (coreference, distant relationships)",
        "Head 5": "Compositional meaning (phrases, named entities)",
        "Head 6": "Grammatical roles (determiners, modifiers)",
        "Head 7": "Discourse structure (topic, focus)",
        "Head 8": "Task-specific patterns (question-answer, cause-effect)"
    }
    
    for head, focus in heads_focus.items():
        print(f"{head}: {focus}")
```

## Attention Variants

### Causal (Masked) Attention

Used in decoder-only models like GPT to prevent looking at future tokens:

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_length=2048):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        # Create causal mask
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_seq_length, max_seq_length)).view(
                1, 1, max_seq_length, max_seq_length
            )
        )
    
    def forward(self, x):
        seq_len = x.size(1)
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        
        return self.attention(x, x, x, mask)

def create_causal_mask(seq_len):
    """Create a causal mask for autoregressive generation"""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.view(1, 1, seq_len, seq_len)

# Visualization
def visualize_causal_mask():
    mask = create_causal_mask(6)
    print("Causal mask (1=can attend, 0=masked):")
    print(mask.squeeze().int().numpy())
    
    # Output:
    # [[1 0 0 0 0 0]
    #  [1 1 0 0 0 0]
    #  [1 1 1 0 0 0]
    #  [1 1 1 1 0 0]
    #  [1 1 1 1 1 0]
    #  [1 1 1 1 1 1]]
```

### Cross-Attention

Used in encoder-decoder models where decoder attends to encoder:

```python
class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
    
    def forward(self, decoder_hidden, encoder_outputs, encoder_mask=None):
        """
        Args:
            decoder_hidden: [batch_size, tgt_len, d_model]
            encoder_outputs: [batch_size, src_len, d_model]
            encoder_mask: [batch_size, 1, 1, src_len]
        """
        # Query from decoder, Key and Value from encoder
        output, attention_weights = self.attention(
            query=decoder_hidden,
            key=encoder_outputs,
            value=encoder_outputs,
            mask=encoder_mask
        )
        
        return output, attention_weights
```

### Sparse Attention

Reduces computational complexity by limiting attention patterns:

```python
class SparseAttention(nn.Module):
    def __init__(self, d_model, num_heads, sparsity_pattern='local'):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.sparsity_pattern = sparsity_pattern
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def create_sparse_mask(self, seq_len):
        """Create different sparse attention patterns"""
        if self.sparsity_pattern == 'local':
            return self.local_attention_mask(seq_len, window_size=128)
        elif self.sparsity_pattern == 'strided':
            return self.strided_attention_mask(seq_len, stride=64)
        elif self.sparsity_pattern == 'block':
            return self.block_attention_mask(seq_len, block_size=64)
        else:
            raise ValueError(f"Unknown sparsity pattern: {self.sparsity_pattern}")
    
    def local_attention_mask(self, seq_len, window_size):
        """Local attention: each position attends to nearby positions"""
        mask = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = 1
        return mask
    
    def strided_attention_mask(self, seq_len, stride):
        """Strided attention: attend to every stride-th position"""
        mask = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            # Local attention
            if i > 0:
                mask[i, i-1] = 1
            mask[i, i] = 1
            if i < seq_len - 1:
                mask[i, i+1] = 1
            
            # Strided attention
            for j in range(0, seq_len, stride):
                mask[i, j] = 1
        
        return mask
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Create sparse mask
        sparse_mask = self.create_sparse_mask(seq_len).to(x.device)
        sparse_mask = sparse_mask.view(1, 1, seq_len, seq_len)
        
        # Standard multi-head attention with sparse mask
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention with sparsity
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(sparse_mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        # Concatenate and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(context)
        
        return output, attention_weights
```

## Advanced Attention Mechanisms

### Relative Position Encoding

Instead of absolute positions, use relative distances:

```python
class RelativePositionAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_relative_position=128):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_relative_position = max_relative_position
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Relative position embeddings
        vocab_size = 2 * max_relative_position + 1
        self.relative_position_k = nn.Embedding(vocab_size, self.d_k)
        self.relative_position_v = nn.Embedding(vocab_size, self.d_k)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Get relative position encodings
        relative_position_matrix = self.get_relative_position_matrix(seq_len)
        relations_keys = self.relative_position_k(relative_position_matrix)
        relations_values = self.relative_position_v(relative_position_matrix)
        
        # Standard attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Add relative position bias
        relative_scores = self.relative_attention_bias(Q, relations_keys)
        scores = (scores + relative_scores) / math.sqrt(self.d_k)
        
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Add relative position to values
        relative_context = self.relative_attention_values(attention_weights, relations_values)
        context = context + relative_context
        
        # Concatenate and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(context)
        
        return output, attention_weights
    
    def get_relative_position_matrix(self, seq_len):
        """Create matrix of relative positions"""
        range_vec = torch.arange(seq_len)
        distance_mat = range_vec[None, :] - range_vec[:, None]
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position
        )
        final_mat = distance_mat_clipped + self.max_relative_position
        return final_mat
    
    def relative_attention_bias(self, query, relations_keys):
        """Calculate relative position bias for attention scores"""
        # query: [batch_size, num_heads, seq_len, d_k]
        # relations_keys: [seq_len, seq_len, d_k]
        
        # Expand for batch and heads
        relations_keys = relations_keys.unsqueeze(0).unsqueeze(0)
        
        # Calculate bias
        bias = torch.matmul(query.unsqueeze(-2), relations_keys.transpose(-2, -1))
        return bias.squeeze(-2)
    
    def relative_attention_values(self, attention_weights, relations_values):
        """Apply relative position to attention values"""
        # attention_weights: [batch_size, num_heads, seq_len, seq_len]
        # relations_values: [seq_len, seq_len, d_k]
        
        relations_values = relations_values.unsqueeze(0).unsqueeze(0)
        return torch.matmul(attention_weights.unsqueeze(-2), relations_values).squeeze(-2)
```

### Rotary Position Embedding (RoPE)

A more advanced relative position encoding:

```python
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, x, seq_len=None):
        seq_len = seq_len or x.shape[-2]
        
        # Create position tensor
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        
        # Compute frequencies
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        
        return cos, sin

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to query and key tensors"""
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

class RotaryAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.rotary_emb = RotaryPositionEmbedding(self.d_k)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply rotary position embedding
        cos, sin = self.rotary_emb(x, seq_len)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
        
        # Standard attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        # Output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(context)
        
        return output, attention_weights
```

## Attention Analysis and Visualization

### Attention Pattern Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class AttentionAnalyzer:
    def __init__(self):
        self.attention_patterns = {}
    
    def analyze_attention_heads(self, attention_weights, layer_idx, tokens=None):
        """
        Analyze attention patterns for each head
        
        Args:
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
            layer_idx: int, layer index
            tokens: list of strings (optional)
        """
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # Average across batch
        avg_attention = attention_weights.mean(dim=0)  # [num_heads, seq_len, seq_len]
        
        patterns = {}
        for head_idx in range(num_heads):
            head_attention = avg_attention[head_idx]
            
            patterns[f"head_{head_idx}"] = {
                'attention_matrix': head_attention.detach().cpu().numpy(),
                'entropy': self.calculate_attention_entropy(head_attention),
                'locality': self.calculate_locality_score(head_attention),
                'diagonal_strength': self.calculate_diagonal_strength(head_attention)
            }
        
        self.attention_patterns[f"layer_{layer_idx}"] = patterns
        return patterns
    
    def calculate_attention_entropy(self, attention_matrix):
        """Calculate entropy of attention distribution"""
        entropy = -(attention_matrix * torch.log(attention_matrix + 1e-10)).sum(dim=-1)
        return entropy.mean().item()
    
    def calculate_locality_score(self, attention_matrix):
        """Calculate how local the attention pattern is"""
        seq_len = attention_matrix.size(0)
        positions = torch.arange(seq_len).float()
        
        # Expected position for each query
        expected_pos = (attention_matrix * positions).sum(dim=-1)
        
        # Distance from actual position
        actual_pos = torch.arange(seq_len).float()
        locality = torch.abs(expected_pos - actual_pos).mean()
        
        return locality.item()
    
    def calculate_diagonal_strength(self, attention_matrix):
        """Calculate strength of diagonal pattern (self-attention)"""
        diagonal = torch.diag(attention_matrix)
        return diagonal.mean().item()
    
    def visualize_attention_head(self, layer_idx, head_idx, tokens=None, figsize=(10, 8)):
        """Visualize attention pattern for a specific head"""
        pattern = self.attention_patterns[f"layer_{layer_idx}"][f"head_{head_idx}"]
        attention_matrix = pattern['attention_matrix']
        
        plt.figure(figsize=figsize)
        
        if tokens:
            sns.heatmap(
                attention_matrix,
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='Blues',
                cbar=True
            )
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
        else:
            sns.heatmap(attention_matrix, cmap='Blues', cbar=True)
        
        plt.title(f'Attention Pattern - Layer {layer_idx}, Head {head_idx}')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        
        # Add pattern statistics
        entropy = pattern['entropy']
        locality = pattern['locality']
        diagonal = pattern['diagonal_strength']
        
        plt.figtext(0.02, 0.02, 
                   f'Entropy: {entropy:.2f} | Locality: {locality:.2f} | Diagonal: {diagonal:.2f}',
                   fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def compare_attention_heads(self, layer_idx):
        """Compare all heads in a layer"""
        patterns = self.attention_patterns[f"layer_{layer_idx}"]
        num_heads = len(patterns)
        
        fig, axes = plt.subplots(2, (num_heads + 1) // 2, figsize=(4 * num_heads, 8))
        axes = axes.flatten()
        
        for i, (head_name, pattern) in enumerate(patterns.items()):
            sns.heatmap(
                pattern['attention_matrix'],
                ax=axes[i],
                cmap='Blues',
                cbar=False,
                square=True
            )
            axes[i].set_title(f'{head_name}')
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')
        
        # Hide unused subplots
        for i in range(len(patterns), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Attention Patterns - Layer {layer_idx}')
        plt.tight_layout()
        plt.show()

# Usage example
def demo_attention_analysis():
    # Create dummy attention weights
    batch_size, num_heads, seq_len = 1, 8, 12
    attention_weights = torch.randn(batch_size, num_heads, seq_len, seq_len)
    attention_weights = F.softmax(attention_weights, dim=-1)
    
    tokens = ["The", "cat", "sat", "on", "the", "mat", "and", "looked", "at", "the", "dog", "."]
    
    analyzer = AttentionAnalyzer()
    patterns = analyzer.analyze_attention_heads(attention_weights, layer_idx=0, tokens=tokens)
    
    # Visualize first head
    analyzer.visualize_attention_head(0, 0, tokens)
    
    # Compare all heads
    analyzer.compare_attention_heads(0)
```

### Attention Interpretation

```python
class AttentionInterpreter:
    def __init__(self):
        self.interpretation_rules = {
            'syntactic': self.detect_syntactic_patterns,
            'semantic': self.detect_semantic_patterns,
            'positional': self.detect_positional_patterns,
            'compositional': self.detect_compositional_patterns
        }
    
    def interpret_attention_pattern(self, attention_matrix, tokens, pos_tags=None):
        """Interpret what linguistic phenomena the attention captures"""
        interpretations = {}
        
        for pattern_type, detector in self.interpretation_rules.items():
            score = detector(attention_matrix, tokens, pos_tags)
            interpretations[pattern_type] = score
        
        return interpretations
    
    def detect_syntactic_patterns(self, attention_matrix, tokens, pos_tags):
        """Detect syntactic relationships in attention"""
        if pos_tags is None:
            return 0.0
        
        syntactic_score = 0.0
        count = 0
        
        for i, pos_i in enumerate(pos_tags):
            for j, pos_j in enumerate(pos_tags):
                if self.is_syntactic_relation(pos_i, pos_j):
                    syntactic_score += attention_matrix[i, j]
                    count += 1
        
        return syntactic_score / max(count, 1)
    
    def detect_semantic_patterns(self, attention_matrix, tokens, pos_tags):
        """Detect semantic relationships"""
        # Simplified: look for attention between nouns and related words
        semantic_score = 0.0
        count = 0
        
        for i, token_i in enumerate(tokens):
            for j, token_j in enumerate(tokens):
                if self.is_semantic_relation(token_i, token_j):
                    semantic_score += attention_matrix[i, j]
                    count += 1
        
        return semantic_score / max(count, 1)
    
    def detect_positional_patterns(self, attention_matrix, tokens, pos_tags):
        """Detect position-based patterns"""
        seq_len = len(tokens)
        
        # Check for local attention (nearby positions)
        local_attention = 0.0
        for i in range(seq_len):
            for j in range(max(0, i-2), min(seq_len, i+3)):
                if i != j:
                    local_attention += attention_matrix[i, j]
        
        return local_attention / (seq_len * 4)  # Normalize by possible local connections
    
    def detect_compositional_patterns(self, attention_matrix, tokens, pos_tags):
        """Detect compositional/phrase-level patterns"""
        # Look for attention within potential phrases
        compositional_score = 0.0
        phrases = self.identify_phrases(tokens, pos_tags)
        
        for phrase_start, phrase_end in phrases:
            phrase_attention = 0.0
            phrase_size = phrase_end - phrase_start
            
            for i in range(phrase_start, phrase_end):
                for j in range(phrase_start, phrase_end):
                    phrase_attention += attention_matrix[i, j]
            
            compositional_score += phrase_attention / (phrase_size * phrase_size)
        
        return compositional_score / max(len(phrases), 1)
    
    def is_syntactic_relation(self, pos_i, pos_j):
        """Check if two POS tags have a syntactic relationship"""
        syntactic_pairs = {
            ('NOUN', 'VERB'), ('VERB', 'NOUN'),
            ('ADJ', 'NOUN'), ('NOUN', 'ADJ'),
            ('DET', 'NOUN'), ('NOUN', 'DET'),
            ('ADP', 'NOUN'), ('NOUN', 'ADP')
        }
        return (pos_i, pos_j) in syntactic_pairs
    
    def is_semantic_relation(self, token_i, token_j):
        """Check if two tokens are semantically related (simplified)"""
        # This would normally use embeddings or a knowledge base
        semantic_groups = [
            {'cat', 'dog', 'animal', 'pet'},
            {'sit', 'stand', 'run', 'walk'},
            {'house', 'home', 'building'},
            {'red', 'blue', 'color', 'green'}
        ]
        
        for group in semantic_groups:
            if token_i.lower() in group and token_j.lower() in group:
                return True
        
        return False
    
    def identify_phrases(self, tokens, pos_tags):
        """Identify potential phrases (simplified)"""
        if pos_tags is None:
            return []
        
        phrases = []
        i = 0
        
        while i < len(tokens):
            if pos_tags[i] == 'DET':
                # Look for determiner + adjective* + noun
                phrase_end = i + 1
                while phrase_end < len(pos_tags) and pos_tags[phrase_end] in ['ADJ', 'NOUN']:
                    phrase_end += 1
                
                if phrase_end > i + 1:  # At least det + noun
                    phrases.append((i, phrase_end))
                    i = phrase_end
                else:
                    i += 1
            else:
                i += 1
        
        return phrases
```

## Performance Considerations

### Computational Complexity

```python
def attention_complexity_analysis():
    """Analyze computational complexity of different attention variants"""
    
    complexities = {
        'Standard Attention': {
            'time': 'O(n²d)',
            'space': 'O(n²)',
            'description': 'Quadratic in sequence length'
        },
        'Sparse Attention': {
            'time': 'O(n√n·d)',
            'space': 'O(n√n)',
            'description': 'Reduced by sparsity pattern'
        },
        'Linear Attention': {
            'time': 'O(nd²)',
            'space': 'O(nd)',
            'description': 'Linear in sequence length'
        },
        'Local Attention': {
            'time': 'O(nwd)',
            'space': 'O(nw)',
            'description': 'w is window size, typically w << n'
        }
    }
    
    return complexities

def benchmark_attention_variants():
    """Benchmark different attention implementations"""
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d_model = 512
    num_heads = 8
    
    results = {}
    
    for seq_len in [128, 256, 512, 1024, 2048]:
        batch_size = 8
        x = torch.randn(batch_size, seq_len, d_model).to(device)
        
        # Standard attention
        standard_attn = MultiHeadAttention(d_model, num_heads).to(device)
        
        start_time = time.time()
        for _ in range(10):
            _ = standard_attn(x, x, x)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        standard_time = (time.time() - start_time) / 10
        
        # Sparse attention
        sparse_attn = SparseAttention(d_model, num_heads, 'local').to(device)
        
        start_time = time.time()
        for _ in range(10):
            _ = sparse_attn(x)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        sparse_time = (time.time() - start_time) / 10
        
        results[seq_len] = {
            'standard': standard_time,
            'sparse': sparse_time,
            'speedup': standard_time / sparse_time
        }
    
    return results
```

## Best Practices

### Implementation Tips

1. **Numerical Stability**

```python
def stable_attention(Q, K, V, mask=None):
    """Numerically stable attention implementation"""
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Subtract max for numerical stability
    scores_max = scores.max(dim=-1, keepdim=True)[0]
    scores = scores - scores_max
    
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights
```

2. **Memory Optimization**

```python
def memory_efficient_attention(Q, K, V, chunk_size=1024):
    """Process attention in chunks to save memory"""
    batch_size, num_heads, seq_len, d_k = Q.shape
    
    if seq_len <= chunk_size:
        return stable_attention(Q, K, V)
    
    output = torch.zeros_like(Q)
    
    for i in range(0, seq_len, chunk_size):
        end_i = min(i + chunk_size, seq_len)
        Q_chunk = Q[:, :, i:end_i, :]
        
        chunk_output, _ = stable_attention(Q_chunk, K, V)
        output[:, :, i:end_i, :] = chunk_output
    
    return output
```

3. **Gradient Checkpointing**

```python
class CheckpointedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
    
    def forward(self, x):
        if self.training:
            return torch.utils.checkpoint.checkpoint(self.attention, x, x, x)
        else:
            return self.attention(x, x, x)
```

## Conclusion

Attention mechanisms are the foundation of modern transformer architectures. Understanding their variants, implementations, and analysis techniques is crucial for:

1. **Model Design**: Choosing appropriate attention patterns for your use case
2. **Performance Optimization**: Implementing efficient attention for large sequences
3. **Interpretability**: Understanding what your model learns
4. **Debugging**: Identifying attention-related issues

The field continues to evolve with new attention variants that balance computational efficiency with modeling capability.

## Further Reading

- "Attention Is All You Need" (Vaswani et al., 2017)
- "Analyzing Multi-Head Self-Attention" (Clark et al., 2019)
- "Longformer: The Long-Document Transformer" (Beltagy et al., 2020)
- "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
- "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao et al., 2022)

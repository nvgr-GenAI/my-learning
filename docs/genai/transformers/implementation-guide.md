# Building Your First Transformer: A Step-by-Step Guide

!!! tip "🛠️ Learning by Building"
    The best way to understand transformers is to build one yourself! We'll start simple and gradually add complexity, with clear explanations at every step.

## 🎯 Our Building Plan

Think of building a transformer like constructing a house:

1. **🧱 Foundation:** Basic attention mechanism
2. **🏗️ Framework:** Multi-head attention  
3. **🔌 Utilities:** Positional encoding and embeddings
4. **🏠 Structure:** Complete transformer block
5. **🏘️ Full Model:** Stack everything together

Let's start building!

## 🧱 Step 1: Simple Attention (The Foundation)

### The Basic Idea

Remember our party analogy? Let's code that up!

=== "🎯 The Concept"

    ```python
    # Think of it like this:
    # query = "What am I looking for?"
    # key = "What information do I have?"  
    # value = "What is the actual content?"
    
    def simple_attention(query, key, value):
        # How relevant is each key to our query?
        scores = query @ key.T  # Matrix multiplication
        
        # Convert to probabilities (attention weights)
        attention_weights = softmax(scores)
        
        # Get weighted combination of values
        output = attention_weights @ value
        
        return output, attention_weights
    ```

=== "💻 Real Implementation"

    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math
    
    def scaled_dot_product_attention(query, key, value, mask=None):
        """
        The heart of transformers - simple but powerful!
        
        Args:
            query: [batch_size, seq_len, d_k] - What we're looking for
            key: [batch_size, seq_len, d_k] - What we compare against
            value: [batch_size, seq_len, d_v] - The actual content
            mask: Optional mask to hide certain positions
            
        Returns:
            output: [batch_size, seq_len, d_v] - Attention-enhanced values
            attention_weights: [batch_size, seq_len, seq_len] - Where we looked
        """
        # Get dimensions
        d_k = query.size(-1)
        
        # Step 1: Calculate how relevant each key is to each query
        # Think: "How much should I pay attention to word X when processing word Y?"
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Step 2: Apply mask if provided (e.g., hide future words in GPT)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Step 3: Convert scores to probabilities using softmax
        # Now we have attention weights that sum to 1
        attention_weights = F.softmax(scores, dim=-1)
        
        # Step 4: Use attention weights to combine values
        # This is the magic - we get a weighted mix of all information!
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights
    ```

=== "🧪 Let's Test It!"

    ```python
    # Create some simple test data
    batch_size, seq_len, d_model = 1, 4, 8
    
    # Random input (in practice, these come from embeddings)
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)  
    value = torch.randn(batch_size, seq_len, d_model)
    
    # Apply attention
    output, weights = scaled_dot_product_attention(query, key, value)
    
    print(f"Input shape: {query.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"\\nExample attention weights for first position:")
    print(weights[0, 0, :])  # How much position 0 attends to each position
    ```

### 🔍 Understanding the Output

The attention weights tell us **where the model is looking**:

    Sentence: ["The", "cat", "sat", "down"]
    
    Attention weights for "sat":
    The: 0.1, cat: 0.7, sat: 0.1, down: 0.1
    
    Interpretation: When processing "sat", the model pays 70% attention to "cat" 
    (makes sense - cat is the subject doing the sitting!)

## 🏗️ Step 2: Multi-Head Attention (Multiple Experts)

### Why Multiple Heads?

Think of having different experts analyze the same text:

- **Expert 1:** Grammar specialist
- **Expert 2:** Meaning specialist  
- **Expert 3:** Reference specialist
- **Expert 4:** Context specialist

=== "🧠 The Concept"

    ```python
    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads  # Dimension per head
            
            # Each head gets its own transformation matrices
            self.W_q = nn.Linear(d_model, d_model)  # Query projections
            self.W_k = nn.Linear(d_model, d_model)  # Key projections  
            self.W_v = nn.Linear(d_model, d_model)  # Value projections
            self.W_o = nn.Linear(d_model, d_model)  # Output projection
    ```

=== "💻 Complete Implementation"

    ```python
    class MultiHeadAttention(nn.Module):
        """
        Multi-head attention: Like having multiple experts analyze the text
        """
        def __init__(self, d_model, num_heads):
            super().__init__()
            assert d_model % num_heads == 0  # Must be evenly divisible
            
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            
            # Linear layers for Q, K, V transformations
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)
            
        def forward(self, query, key, value, mask=None):
            batch_size = query.size(0)
            
            # Step 1: Apply linear transformations and split into heads
            Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            
            # Step 2: Apply attention for each head
            attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
            
            # Step 3: Concatenate heads back together
            attn_output = attn_output.transpose(1, 2).contiguous().view(
                batch_size, -1, self.d_model
            )
            
            # Step 4: Apply final linear transformation
            output = self.W_o(attn_output)
            
            return output, attn_weights
    ```

=== "🧪 Testing Multi-Head"

    ```python
    # Test our multi-head attention
    d_model, num_heads = 512, 8
    seq_len, batch_size = 10, 2
    
    mha = MultiHeadAttention(d_model, num_heads)
    
    # Create test input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Self-attention (query, key, value are all the same)
    output, weights = mha(x, x, x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"We have {num_heads} attention heads working in parallel!")
    ```

## 🔌 Step 3: Positional Encoding (GPS for Words)

### The Problem

Attention doesn't care about word order! "Dog bites man" = "Man bites dog" without position info.

### The Solution: Positional Encoding

=== "🧭 The Concept"

    Add a unique "signature" to each position:
    
    ```
    Position 0: [1.0, 0.0, 1.0, 0.0, ...]
    Position 1: [0.0, 1.0, 0.9, 0.1, ...]  
    Position 2: [0.0, 0.0, 0.8, 0.2, ...]
    ...
    ```
    
    Each position gets a unique pattern the model can recognize!

=== "💻 Implementation"

    ```python
    class PositionalEncoding(nn.Module):
        """
        Add position information to embeddings using sine and cosine functions
        """
        def __init__(self, d_model, max_len=5000):
            super().__init__()
            
            # Create a matrix to hold positional encodings
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            
            # Calculate the div_term for sine and cosine functions
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                               -(math.log(10000.0) / d_model))
            
            # Apply sine to even indices
            pe[:, 0::2] = torch.sin(position * div_term)
            # Apply cosine to odd indices  
            pe[:, 1::2] = torch.cos(position * div_term)
            
            # Add batch dimension and register as buffer
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
            
        def forward(self, x):
            # Add positional encoding to input embeddings
            return x + self.pe[:, :x.size(1)]
    ```

=== "🎨 Visualizing Position Patterns"

    ```python
    # Let's see what positional encodings look like
    pe = PositionalEncoding(d_model=128, max_len=100)
    
    # Get encoding for first 50 positions
    pos_encoding = pe.pe[0, :50, :].numpy()
    
    # Plot the first few dimensions
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.plot(pos_encoding[:, :10])
    plt.title("Positional Encoding Patterns")
    plt.xlabel("Position")
    plt.ylabel("Encoding Value")
    plt.legend([f"Dim {i}" for i in range(10)])
    plt.show()
    
    # Each line shows a different pattern - unique for each position!
    ```

## 🏠 Step 4: Complete Transformer Block

Now let's combine everything into a complete transformer layer:

=== "🧱 Building Blocks"

    A transformer block needs:
    1. **Multi-head attention** (the star)
    2. **Feed-forward network** (the thinker)  
    3. **Residual connections** (the memory bridge)
    4. **Layer normalization** (the stabilizer)

=== "💻 Full Implementation"

    ```python
    class TransformerBlock(nn.Module):
        """
        A complete transformer block with attention and feed-forward layers
        """
        def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
            super().__init__()
            
            # Multi-head attention
            self.attention = MultiHeadAttention(d_model, num_heads)
            
            # Feed-forward network
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            )
            
            # Layer normalization (applied before each sub-layer)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            
            # Dropout for regularization
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x, mask=None):
            # Sub-layer 1: Multi-head attention with residual connection
            # Pre-normalization: norm -> attention -> residual
            attn_output, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
            x = x + self.dropout(attn_output)  # Residual connection
            
            # Sub-layer 2: Feed-forward with residual connection  
            ff_output = self.ff(self.norm2(x))
            x = x + self.dropout(ff_output)  # Residual connection
            
            return x
    ```

=== "🧪 Testing Our Block"

    ```python
    # Test the complete transformer block
    d_model, num_heads, d_ff = 512, 8, 2048
    seq_len, batch_size = 20, 2
    
    # Create transformer block
    transformer_block = TransformerBlock(d_model, num_heads, d_ff)
    
    # Create test input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Pass through transformer block
    output = transformer_block(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Success! Our transformer block works!")
    ```

## 🏘️ Step 5: Complete Transformer Model

Finally, let's stack multiple blocks to create a full transformer:

=== "🏗️ The Architecture"

    ```python
    class Transformer(nn.Module):
        """
        Complete transformer model for language modeling
        """
        def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout=0.1):
            super().__init__()
            
            # Token and positional embeddings
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoding = PositionalEncoding(d_model, max_len)
            
            # Stack of transformer blocks
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ])
            
            # Output layer for next token prediction
            self.ln_f = nn.LayerNorm(d_model)  # Final layer norm
            self.head = nn.Linear(d_model, vocab_size)  # Project to vocabulary
            
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x, mask=None):
            # Convert token IDs to embeddings
            x = self.embedding(x)
            
            # Add positional encoding
            x = self.pos_encoding(x)
            x = self.dropout(x)
            
            # Pass through transformer blocks
            for block in self.transformer_blocks:
                x = block(x, mask)
            
            # Final layer norm and projection to vocabulary
            x = self.ln_f(x)
            logits = self.head(x)
            
            return logits
    ```

=== "🎉 Let's Build GPT-Mini!"

    ```python
    # Create a small GPT-like model
    vocab_size = 50000  # Size of vocabulary
    d_model = 512      # Embedding dimension
    num_heads = 8      # Number of attention heads
    num_layers = 6     # Number of transformer blocks
    d_ff = 2048        # Feed-forward dimension
    max_len = 1024     # Maximum sequence length
    
    # Create our mini-GPT!
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model, 
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_len=max_len
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Our transformer has {total_params:,} parameters!")
    print("That's a real neural network!")
    
    # Test with random input
    batch_size, seq_len = 2, 50
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print("Success! We built a transformer from scratch! 🎉")
    ```

## 🎯 What We've Built

Congratulations! You've just built a complete transformer model that includes:

✅ **Scaled dot-product attention** - The core mechanism  
✅ **Multi-head attention** - Multiple parallel experts  
✅ **Positional encoding** - Position awareness  
✅ **Transformer blocks** - Complete processing units  
✅ **Full model** - Ready for training!

### Model Sizes Comparison

| Model | Parameters | Layers | Heads | d_model |
|-------|------------|--------|-------|---------|
| **Your Model** | ~25M | 6 | 8 | 512 |
| **GPT-2 Small** | 117M | 12 | 12 | 768 |
| **GPT-2 Medium** | 345M | 24 | 16 | 1024 |
| **GPT-3** | 175B | 96 | 96 | 12288 |

**Your model is comparable to early successful transformers!**

## 🚀 Next Steps

Now that you've built a transformer, you can:

1. **[Train Your Model](training-objectives.md)** - Learn how to actually train it
2. **[Fine-tune for Tasks](fine-tuning.md)** - Adapt it for specific applications  
3. **[Optimize Performance](optimization.md)** - Make it faster and more efficient
4. **[Explore Applications](applications.md)** - See what you can build with it

---

!!! success "🏆 Transformer Builder!"
    You've just built a complete transformer from scratch! You now understand every component and how they work together. This is the same architecture powering ChatGPT, BERT, and other modern AI systems!

# Large Language Model Architecture

## Introduction

Large Language Models (LLMs) represent the current pinnacle of natural language processing technology. Understanding their architecture is crucial for anyone working with or developing GenAI applications.

## Transformer Foundation

### Core Architecture Components

#### Self-Attention Mechanism

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.W_o(context)
```

#### Position Encoding

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

### Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

## LLM-Specific Architectures

### GPT Architecture (Decoder-Only)

```python
class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, input_ids, attention_mask=None):
        # Create causal mask
        seq_length = input_ids.size(1)
        causal_mask = torch.tril(torch.ones(seq_length, seq_length))
        
        # Embeddings
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.position_encoding(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, causal_mask)
            
        # Output projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
```

### T5 Architecture (Encoder-Decoder)

```python
class T5Model(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff):
        super().__init__()
        
        # Shared embeddings
        self.shared_embedding = nn.Embedding(vocab_size, d_model)
        
        # Encoder
        self.encoder = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        # Decoder
        self.decoder = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, encoder_input_ids, decoder_input_ids):
        # Encoder
        encoder_embeddings = self.shared_embedding(encoder_input_ids)
        encoder_output = encoder_embeddings
        
        for layer in self.encoder:
            encoder_output = layer(encoder_output)
            
        # Decoder
        decoder_embeddings = self.shared_embedding(decoder_input_ids)
        decoder_output = decoder_embeddings
        
        for layer in self.decoder:
            decoder_output = layer(decoder_output, encoder_output)
            
        # Output projection
        logits = self.head(decoder_output)
        return logits
```

## Scaling Strategies

### Model Parallelism

```python
class ModelParallelGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Distribute layers across devices
        self.layers_per_device = config.num_layers // config.num_devices
        
        for device_id in range(config.num_devices):
            start_layer = device_id * self.layers_per_device
            end_layer = start_layer + self.layers_per_device
            
            device_layers = nn.ModuleList([
                TransformerBlock(config.d_model, config.num_heads, config.d_ff)
                for _ in range(self.layers_per_device)
            ]).to(f'cuda:{device_id}')
            
            setattr(self, f'device_{device_id}_layers', device_layers)
    
    def forward(self, x):
        for device_id in range(self.config.num_devices):
            x = x.to(f'cuda:{device_id}')
            layers = getattr(self, f'device_{device_id}_layers')
            
            for layer in layers:
                x = layer(x)
                
        return x
```

### Data Parallelism

```python
def train_with_data_parallelism(model, dataloader, optimizer):
    model = nn.DataParallel(model)
    model.train()
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        # Forward pass (automatically distributed)
        outputs = model(batch['input_ids'])
        loss = compute_loss(outputs, batch['labels'])
        
        # Backward pass (gradients aggregated)
        loss.backward()
        optimizer.step()
```

## Memory Optimization Techniques

### Gradient Checkpointing

```python
class CheckpointedTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x):
        # Use gradient checkpointing to save memory
        x = checkpoint(self._attention_forward, x)
        x = checkpoint(self._ff_forward, x)
        return x
        
    def _attention_forward(self, x):
        return x + self.attention(x, x, x)
        
    def _ff_forward(self, x):
        return x + self.feed_forward(x)
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

def train_with_mixed_precision(model, dataloader, optimizer):
    scaler = GradScaler()
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(batch['input_ids'])
            loss = compute_loss(outputs, batch['labels'])
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

## Attention Variants

### Sparse Attention

```python
class SparseAttention(nn.Module):
    def __init__(self, d_model, num_heads, sparsity_pattern):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.sparsity_pattern = sparsity_pattern
        
    def forward(self, query, key, value):
        # Compute attention only for specified positions
        batch_size, seq_len = query.shape[:2]
        
        # Create sparse attention mask
        mask = self.create_sparse_mask(seq_len)
        
        # Standard attention computation with sparse mask
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        
        return output
        
    def create_sparse_mask(self, seq_len):
        # Implement specific sparsity pattern
        # e.g., local attention, strided attention, etc.
        pass
```

### Rotary Position Embedding (RoPE)

```python
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        
        return cos, sin
        
def apply_rotary_pos_emb(q, k, cos, sin):
    # Apply rotary position embedding
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
    
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)
```

## Model Configurations

### GPT-3 Style Configuration

```python
GPT3_CONFIGS = {
    'gpt3-small': {
        'vocab_size': 50257,
        'd_model': 768,
        'num_layers': 12,
        'num_heads': 12,
        'd_ff': 3072,
        'max_seq_length': 2048,
        'dropout': 0.1
    },
    'gpt3-medium': {
        'vocab_size': 50257,
        'd_model': 1024,
        'num_layers': 24,
        'num_heads': 16,
        'd_ff': 4096,
        'max_seq_length': 2048,
        'dropout': 0.1
    },
    'gpt3-large': {
        'vocab_size': 50257,
        'd_model': 1280,
        'num_layers': 36,
        'num_heads': 20,
        'd_ff': 5120,
        'max_seq_length': 2048,
        'dropout': 0.1
    },
    'gpt3-xl': {
        'vocab_size': 50257,
        'd_model': 1600,
        'num_layers': 48,
        'num_heads': 25,
        'd_ff': 6400,
        'max_seq_length': 2048,
        'dropout': 0.1
    }
}
```

### Model Initialization

```python
def initialize_model(config_name):
    config = GPT3_CONFIGS[config_name]
    
    model = GPTModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        max_seq_length=config['max_seq_length']
    )
    
    # Initialize weights
    model.apply(lambda m: init_weights(m, config))
    
    return model

def init_weights(module, config):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

## Inference Optimization

### Key-Value Caching

```python
class KVCacheGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer_blocks = nn.ModuleList([
            KVCacheTransformerBlock(config)
            for _ in range(config.num_layers)
        ])
        
    def forward(self, input_ids, past_key_values=None):
        batch_size, seq_len = input_ids.shape
        
        # Initialize cache if not provided
        if past_key_values is None:
            past_key_values = [None] * len(self.transformer_blocks)
            
        present_key_values = []
        x = self.embed(input_ids)
        
        for i, (block, past_kv) in enumerate(zip(self.transformer_blocks, past_key_values)):
            x, present_kv = block(x, past_kv)
            present_key_values.append(present_kv)
            
        return x, present_key_values
        
class KVCacheTransformerBlock(nn.Module):
    def forward(self, x, past_key_value=None):
        # Use cached keys and values for efficiency
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, self.W_k(x)], dim=-2)
            value = torch.cat([past_value, self.W_v(x)], dim=-2)
        else:
            key = self.W_k(x)
            value = self.W_v(x)
            
        query = self.W_q(x)
        
        # Attention computation
        attention_output = self.attention(query, key, value)
        
        # Return output and current key-value for caching
        present_key_value = (key, value)
        return attention_output, present_key_value
```

### Quantization

```python
def quantize_model(model, quantization_config):
    """Apply quantization to reduce model size and memory usage"""
    
    if quantization_config.method == 'int8':
        return torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
    elif quantization_config.method == 'int4':
        # Custom int4 quantization implementation
        return apply_int4_quantization(model)
    
def apply_int4_quantization(model):
    """Apply 4-bit quantization to model weights"""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # Quantize weights to 4-bit
            weight = module.weight.data
            scale = weight.abs().max() / 7  # 4-bit signed range: -8 to 7
            quantized_weight = torch.round(weight / scale).clamp(-8, 7)
            
            # Store quantized weights and scale
            module.register_buffer('quantized_weight', quantized_weight.to(torch.int8))
            module.register_buffer('weight_scale', scale)
```

## Architecture Comparison

### Model Family Characteristics

| Model Family | Architecture | Strengths | Use Cases |
|--------------|-------------|-----------|-----------|
| GPT | Decoder-only | Text generation, Few-shot learning | Chatbots, Content creation |
| BERT | Encoder-only | Understanding, Classification | Search, Analysis |
| T5 | Encoder-Decoder | Text-to-text tasks | Translation, Summarization |
| PaLM | Decoder-only | Reasoning, Code generation | Complex problem solving |
| LaMDA | Decoder-only | Dialogue, Conversation | Conversational AI |

### Performance Considerations

```python
def compare_architectures():
    metrics = {
        'gpt3': {
            'parameters': '175B',
            'training_cost': 'Very High',
            'inference_speed': 'Medium',
            'memory_usage': 'High',
            'versatility': 'High'
        },
        'bert': {
            'parameters': '340M',
            'training_cost': 'Medium',
            'inference_speed': 'Fast',
            'memory_usage': 'Medium',
            'versatility': 'Medium'
        },
        't5': {
            'parameters': '11B',
            'training_cost': 'High',
            'inference_speed': 'Medium',
            'memory_usage': 'High',
            'versatility': 'High'
        }
    }
    return metrics
```

## Best Practices

### Architecture Design

1. **Layer Normalization Placement**
   - Pre-norm vs Post-norm configurations
   - Impact on training stability

2. **Activation Functions**
   - ReLU vs GELU vs SwiGLU
   - Trade-offs in computational efficiency

3. **Attention Patterns**
   - Full attention vs sparse attention
   - Local vs global attention mechanisms

### Implementation Tips

```python
class OptimizedTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Use pre-layer normalization for better training stability
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        
        # Use GELU activation for better performance
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model)
        )
        
        # Apply dropout for regularization
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
    
    def forward(self, x):
        # Pre-norm residual connections
        x = x + self.dropout1(self.attention(self.ln1(x)))
        x = x + self.dropout2(self.feed_forward(self.ln2(x)))
        return x
```

## Conclusion

Understanding LLM architecture is essential for:

1. **Model Selection**: Choosing the right architecture for your use case
2. **Optimization**: Implementing efficient training and inference
3. **Customization**: Adapting models for specific requirements
4. **Debugging**: Identifying and resolving performance issues

The field continues to evolve with new architectural innovations, efficiency improvements, and scaling strategies that push the boundaries of what's possible with language models.

## Further Reading

- "Attention Is All You Need" - Original Transformer paper
- "Language Models are Few-Shot Learners" - GPT-3 paper
- "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" - T5 paper
- "PaLM: Scaling Language Modeling with Pathways" - PaLM paper

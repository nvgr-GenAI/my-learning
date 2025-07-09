# Core Concepts in GenAI

!!! info "Essential Building Blocks"
    Master the fundamental concepts that underpin all generative AI systems. We'll explore these concepts intuitively first, then see how they work in practice.

## Learning Objectives

By the end of this guide, you'll understand:

- **What embeddings are** and why they're crucial for AI
- **How attention mechanisms** help models focus on relevant information
- **The autoregressive generation process** that powers text generation
- **Different sampling strategies** and their trade-offs
- **Training objectives** that teach models to generate coherent content

---

## Embeddings and Representation Learning

### What are Embeddings? 🎯

Imagine you're organizing a massive library. Instead of just putting books randomly on shelves, you want to place similar books near each other. Romance novels go together, science textbooks cluster in one area, and cookbooks have their own section.

**Embeddings do exactly this for computers with words, images, or any data.**

#### The Core Idea

- **Traditional approach**: Each word is just a unique symbol (like "cat" = 1, "dog" = 2, "car" = 3)
- **Embedding approach**: Each word becomes a point in a multi-dimensional space where similar words are close together

#### Why This Matters

```
Traditional: "cat" and "dog" are completely unrelated numbers
Embedding: "cat" and "dog" are close in space (both are animals)
           "car" and "truck" are close (both are vehicles)
           "cat" and "car" are far apart (different concepts)
```

#### Real-World Analogy

Think of embeddings like a GPS coordinate system:
- **New York** and **Boston** have coordinates close to each other (both East Coast cities)
- **New York** and **Los Angeles** are far apart (different coasts)
- **Paris** and **London** are closer than **Paris** and **Tokyo**

### How Embeddings Capture Meaning

#### 1. **Semantic Similarity**
Words with similar meanings end up close together in the embedding space.

```
Animals: cat, dog, horse, elephant (clustered together)
Vehicles: car, truck, bicycle, airplane (clustered together)
Colors: red, blue, green, yellow (clustered together)
```

#### 2. **Relationships and Analogies**
Embeddings can capture relationships like:
- **King** - **Man** + **Woman** ≈ **Queen**
- **Paris** - **France** + **Germany** ≈ **Berlin**

#### 3. **Context Sensitivity**
The same word can have different embeddings in different contexts:
- "Apple" (fruit) vs "Apple" (company)
- "Bank" (financial) vs "Bank" (river)

### From Word2Vec to Modern Embeddings

#### **Word2Vec Era (2013)**
- **Core idea**: Words appearing in similar contexts should have similar embeddings
- **Training**: Predict surrounding words given a center word
- **Limitation**: Each word gets exactly one embedding

#### **BERT Era (2018)**
- **Innovation**: Same word can have different embeddings based on context
- **Training**: Predict masked words in sentences
- **Breakthrough**: Understanding context dramatically improves performance

#### **GPT Era (2018-present)**
- **Approach**: Learn embeddings while learning to generate text
- **Advantage**: Embeddings optimized for generation tasks
- **Scale**: Billions of parameters create incredibly rich representations

### Practical Applications

#### **1. Search Engines**
- Query: "fast car"
- Matches: "speedy vehicle", "quick automobile", "rapid transportation"
- **Why**: Embeddings understand synonyms and related concepts

#### **2. Recommendation Systems**
- You liked: "The Matrix"
- Recommendations: "Blade Runner", "Ghost in the Shell"
- **Why**: Sci-fi movies cluster together in embedding space

#### **3. Language Translation**
- English "cat" and French "chat" have similar embeddings
- **Why**: Cross-lingual embeddings align concepts across languages

---

## Attention Mechanisms

### What is Attention? 🧠

Imagine reading a complex sentence: "The cat that was sitting on the mat yesterday was very fluffy."

When you process "was very fluffy," your brain automatically focuses on "cat" (not "mat" or "yesterday"). This selective focus is exactly what attention mechanisms do in AI.

### The Attention Revolution

#### **Before Attention (RNN Era)**
```
Problem: "The cat ... [100 words] ... was fluffy"
         ^                           ^
         Important info gets "forgotten" by the time we reach the end
```

#### **With Attention**
```
Solution: "The cat ... [100 words] ... was fluffy"
          ^                           ^
          Direct connection! The model can "look back" to relevant parts
```

### How Attention Works: The Restaurant Analogy

Think of attention like a restaurant scene:

#### **The Players**
- **Query**: "What am I looking for?" (the customer's order)
- **Key**: "What do I have?" (items on the menu)
- **Value**: "What do I actually provide?" (the prepared dish)

#### **The Process**
1. **Customer arrives** with a specific craving (Query)
2. **Compares craving** to menu items (Query × Key)
3. **Decides how much** they want each dish (Attention weights)
4. **Receives a combination** of dishes based on their preferences (Weighted Value)

### Self-Attention: The Game Changer

#### **What is Self-Attention?**
The sentence talks to itself: each word figures out which other words are most relevant to it.

```
Sentence: "The cat sat on the mat"

For word "cat":
- "The" → Low attention (just an article)
- "cat" → Medium attention (self-reference)
- "sat" → High attention (what the cat did)
- "on" → Low attention (preposition)
- "the" → Low attention (article)
- "mat" → Medium attention (where the cat sat)
```

#### **Why This is Powerful**
- **Parallel processing**: All words attend to all others simultaneously
- **Long-range dependencies**: First word can directly influence the last word
- **Relationship understanding**: Captures who did what to whom

### Multi-Head Attention: Multiple Perspectives

#### **The Concept**
Like having multiple experts look at the same sentence, each focusing on different aspects:

- **Head 1**: Focuses on grammatical relationships (subject-verb-object)
- **Head 2**: Focuses on semantic relationships (related concepts)
- **Head 3**: Focuses on temporal relationships (sequence and timing)
- **Head 4**: Focuses on emotional tone

#### **The Benefit**
Each head learns different types of relationships, then combines insights for richer understanding.

### Attention Patterns in Practice

#### **1. Translation**
```
English: "The cat is sleeping"
French:  "Le chat dort"

Attention shows: "cat" ↔ "chat", "sleeping" ↔ "dort"
```

#### **2. Question Answering**
```
Context: "Paris is the capital of France. It has a population of 2.2 million."
Question: "What is the capital of France?"
Answer: "Paris"

Attention highlights: "capital" connects to "Paris"
```

#### **3. Creative Writing**
```
Story: "The old wizard raised his staff. Lightning crackled..."
Next: "...across the dark sky"

Attention connects: "Lightning" → "wizard", "staff" → "crackled"
```

---

## Autoregressive Generation

### What is Autoregressive Generation? 📝

Think of autoregressive generation like telling a story word by word, where each new word depends on everything you've said before.

#### **The Process**
```
1. Start with: "Once upon a time"
2. Predict next: "Once upon a time there"
3. Predict next: "Once upon a time there was"
4. Predict next: "Once upon a time there was a"
5. Continue until story is complete...
```

### The Human Analogy

#### **How Humans Write**
- You have an idea for a story
- You write the first sentence
- Each new sentence builds on what came before
- You sometimes revise, but generally move forward

#### **How AI Generates**
- Model has learned patterns from millions of texts
- Generates one token at a time
- Each token is predicted based on all previous tokens
- No "going back" - each choice affects all future choices

### Why Autoregressive Works

#### **1. Coherence**
Each new word considers the entire context, maintaining consistency.

```
Good: "The cat sat on the mat. It was very comfortable."
      (refers back to "cat" correctly)

Bad: "The cat sat on the mat. He was very comfortable."
     (gender inconsistency - autoregressive helps avoid this)
```

#### **2. Flexibility**
Can generate text of any length, adapting to context.

```
Short: "Hello!"
Medium: "Hello! How are you doing today?"
Long: "Hello! How are you doing today? I hope you're having a wonderful time..."
```

#### **3. Controllability**
You can guide generation by providing specific starting contexts.

```
Prompt: "Write a recipe for chocolate cake:"
Output: "Ingredients: 2 cups flour, 1 cup sugar..."

Prompt: "Explain quantum physics:"
Output: "Quantum physics is the study of matter and energy..."
```

### The Challenge: Exposure Bias

#### **Training vs. Generation**
- **Training**: Model sees correct previous words
- **Generation**: Model sees its own (possibly incorrect) previous words

#### **Example**
```
Training: "The cat sat on the [mat]" ← Model learns to predict "mat"
Generation: "The cat sat on the [rug]" ← Model predicted "rug" instead

Now for next word:
Training context: "The cat sat on the mat [...]"
Generation context: "The cat sat on the rug [...]"
```

This difference can lead to error accumulation during long generation.

### Applications of Autoregressive Generation

#### **1. Text Completion**
```
Input: "The weather today is"
Output: "sunny and warm, perfect for a picnic in the park."
```

#### **2. Creative Writing**
```
Input: "In a world where gravity works backwards"
Output: "everything fell upward toward the sky. People built their houses on the undersides of floating islands..."
```

#### **3. Code Generation**
```
Input: "def calculate_fibonacci(n):"
Output: "    if n <= 1:
              return n
          return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)"
```

---

## Sampling Strategies

### Why Sampling Matters 🎲

When a model generates text, it doesn't just pick the single "best" word. Instead, it considers probabilities and makes choices. Different sampling strategies create different styles of generation.

### The Probability Landscape

Imagine the model's prediction as a landscape:
- **Mountains**: High-probability words (likely choices)
- **Hills**: Medium-probability words (possible choices)
- **Valleys**: Low-probability words (unlikely choices)

### Sampling Strategies Explained

#### **1. Greedy Sampling**
**Strategy**: Always pick the highest probability word.

```
Model predictions: "The cat [sat: 60%, walked: 25%, ran: 10%, jumped: 5%]"
Greedy choice: "sat" (always picks 60%)
```

**Characteristics**:
- ✅ Deterministic and consistent
- ❌ Repetitive and boring
- ❌ Can get stuck in loops

#### **2. Temperature Sampling**
**Strategy**: Adjust the "confidence" of the model's predictions.

```
Low temperature (0.2): Makes confident predictions more likely
High temperature (1.5): Makes predictions more random
```

**Temperature Effects**:
- **Low (0.1-0.7)**: Conservative, coherent, predictable
- **Medium (0.7-1.2)**: Balanced creativity and coherence
- **High (1.2-2.0)**: Creative, surprising, potentially chaotic

#### **3. Top-k Sampling**
**Strategy**: Only consider the k most likely words.

```
Model predictions: [word1: 30%, word2: 25%, word3: 20%, word4: 15%, word5: 10%]
Top-k (k=3): Only consider word1, word2, word3
Ignore: word4, word5 (cut off less likely options)
```

**Benefits**:
- ✅ Prevents very unlikely words
- ✅ Maintains quality while allowing variety
- ❌ Fixed cutoff might be too restrictive

#### **4. Top-p (Nucleus) Sampling**
**Strategy**: Consider the smallest set of words that make up p% of the probability.

```
Model predictions: [A: 40%, B: 30%, C: 20%, D: 8%, E: 2%]
Top-p (p=0.9): Include A, B, C, D (total = 98% > 90%)
Exclude: E (adds too little probability)
```

**Advantages**:
- ✅ Adaptive cutoff based on confidence
- ✅ Includes more options when model is uncertain
- ✅ Includes fewer options when model is confident

### Choosing the Right Strategy

#### **For Creative Writing**
```
Strategy: Top-p (p=0.9) + Temperature (1.0-1.2)
Result: Creative but coherent stories
```

#### **For Technical Documentation**
```
Strategy: Top-k (k=5) + Temperature (0.3-0.7)
Result: Accurate, professional tone
```

#### **For Consistent Responses**
```
Strategy: Greedy or Low temperature (0.1-0.3)
Result: Predictable, reliable outputs
```

### The Trade-off Spectrum

```
Deterministic ←→ Creative
Coherent ←→ Surprising
Safe ←→ Risky
Repetitive ←→ Diverse
```

Every sampling strategy sits somewhere on this spectrum, and the best choice depends on your specific use case.

---

## Training Objectives and Loss Functions

### What are Training Objectives? 🎯

Training objectives are like teaching methods. They define what the model should learn and how it should be evaluated.

### Language Modeling: The Foundation

#### **The Basic Objective**
"Given some text, predict what comes next."

```
Training example:
Input: "The cat sat on the"
Target: "mat"

Model learns: When you see "The cat sat on the", "mat" is likely next
```

#### **Why This Works**
- **Massive scale**: Learn from billions of text examples
- **Universal patterns**: Grammar, facts, reasoning emerge naturally
- **Task agnostic**: Same objective works for many applications

### Cross-Entropy Loss: The Mathematics of Learning

#### **What is Cross-Entropy?**
A way to measure how "surprised" the model is by the correct answer.

```
Model is confident and correct: Low loss (good!)
Model is confident and wrong: High loss (bad!)
Model is uncertain: Medium loss (needs improvement)
```

#### **The Learning Process**
1. **Prediction**: Model predicts probabilities for next word
2. **Comparison**: Compare prediction to actual next word
3. **Adjustment**: Adjust model to be less surprised next time
4. **Repeat**: Process millions of examples

### Perplexity: Measuring Confusion

#### **What is Perplexity?**
Perplexity measures how "confused" the model is on average.

```
Low perplexity (2-10): Model is confident and usually correct
Medium perplexity (10-100): Model has reasonable understanding
High perplexity (100+): Model is confused and often wrong
```

#### **Intuitive Understanding**
- **Perplexity of 2**: Model choosing between 2 equally likely options
- **Perplexity of 10**: Model choosing between 10 equally likely options
- **Perplexity of 100**: Model is very uncertain

### Contrastive Learning: Learning by Comparison

#### **The Core Idea**
"Learn to distinguish between similar and different examples."

```
Positive pairs: "The cat is sleeping" ↔ "A cat is taking a nap"
Negative pairs: "The cat is sleeping" ↔ "The car is driving"

Goal: Make positive pairs more similar, negative pairs more different
```

#### **Why This is Powerful**
- **Semantic understanding**: Learns meaning beyond just word patterns
- **Robustness**: Works with paraphrases and variations
- **Efficiency**: Learns from comparisons, not just next-word prediction

### Modern Training Objectives

#### **1. Masked Language Modeling (BERT-style)**
```
Original: "The cat sat on the mat"
Masked: "The cat [MASK] on the mat"
Task: Predict "sat"
```

#### **2. Autoregressive Generation (GPT-style)**
```
Input: "The cat sat on the"
Task: Predict "mat"
Continue: "The cat sat on the mat and"
Task: Predict "slept"
```

#### **3. Instruction Following**
```
Instruction: "Translate this to French:"
Input: "Hello, how are you?"
Expected output: "Bonjour, comment allez-vous?"
```

#### **4. Reinforcement Learning from Human Feedback (RLHF)**
```
Generate multiple responses → Humans rank them → Train model to prefer highly-ranked responses
```

---

---

## Putting It All Together

### How These Concepts Work Together

Now that you understand each concept individually, let's see how they combine to create powerful AI systems:

#### **The Complete Generation Pipeline**

1. **Input Processing**: Convert text to embeddings
2. **Context Understanding**: Use attention to focus on relevant parts
3. **Prediction**: Generate probability distribution over next words
4. **Sampling**: Choose next word using sampling strategy
5. **Iteration**: Repeat process for each new word

#### **Real-World Example: Chatbot Response**

```
User: "What's the weather like in Paris?"

1. Embeddings: Convert words to vectors that capture meaning
2. Attention: Focus on "weather" and "Paris" as key concepts
3. Generation: Predict response words autoregressively
4. Sampling: Balance creativity with coherence
5. Output: "The weather in Paris is currently sunny and warm..."
```

### Why This Foundation Matters

Understanding these core concepts helps you:

- **Debug AI behavior**: Know why models make certain choices
- **Improve performance**: Tune parameters like temperature and sampling
- **Design better systems**: Combine techniques effectively
- **Stay current**: Understand new developments in the field

---

!!! success "🎉 Conceptual Foundation Complete!"
    You now understand the core concepts that power modern generative AI! These building blocks work together to create the intelligent systems you interact with daily.

!!! tip "💡 Key Insights"
    - **Embeddings** transform discrete symbols into meaningful numerical representations
    - **Attention** allows models to focus on relevant information across long sequences
    - **Autoregressive generation** creates coherent text by predicting one word at a time
    - **Sampling strategies** control the balance between creativity and consistency
    - **Training objectives** shape what models learn and how they behave

!!! note "🚀 Ready for More?"
    - **Next**: [Neural Networks](neural-networks.md) - Deep dive into the underlying architecture
    - **Advanced**: [Transformers](../transformers/index.md) - Modern architecture powering most GenAI
    - **Practical**: [Building Your First Model](../practical/first-model.md) - Hands-on implementation

---

## Implementation Examples

=== "📚 Explanation"
    
    ## Understanding Through Code
    
    Now that you understand the core concepts, let's see how they're implemented in practice. The code examples demonstrate:
    
    - **Embeddings**: How words become vectors and find similar words
    - **Attention**: How models focus on relevant parts of sequences
    - **Generation**: How text is produced one word at a time
    - **Sampling**: How different strategies affect output creativity
    - **Training**: How models learn from data
    
    Each implementation is simplified for clarity while maintaining the essential concepts.
    
    !!! tip "Learning Approach"
        - Start with the explanation tab to understand concepts
        - Switch to code tab to see practical implementation
        - Try running the code examples to see results
        - Experiment with different parameters

=== "💻 Code Examples"
    
    ### Embedding Implementation
    
    Here's how you might implement simple word embeddings from scratch:
    ```python
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
    
    # Example usage
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
    
    ### Attention Mechanism Implementation
    
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
    
    # Example usage
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
    ```
    
    ### Autoregressive Generation Implementation
    
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
    ```
    
    ### Sampling Strategies Implementation
    
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
    
    # Example usage
    vocab_size = 50
    logits = np.random.randn(vocab_size)
    
    # Compare different sampling strategies
    print("Sampling Strategy Comparison:")
    print(f"Greedy: {SamplingStrategies.greedy_sampling(logits)}")
    print(f"Temperature 0.5: {SamplingStrategies.temperature_sampling(logits, 0.5)}")
    print(f"Temperature 1.0: {SamplingStrategies.temperature_sampling(logits, 1.0)}")
    print(f"Top-k (k=5): {SamplingStrategies.top_k_sampling(logits, k=5)}")
    print(f"Top-p (p=0.9): {SamplingStrategies.top_p_sampling(logits, p=0.9)}")
    ```
    
    ### Training Objectives Implementation
    
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
    
    # Simulate model outputs and targets
    logits = np.random.randn(batch_size, vocab_size)
    targets = np.random.randint(0, vocab_size, batch_size)
    
    loss = cross_entropy_loss(logits, targets)
    ppl = perplexity(logits, targets)
    
    print(f"Cross-entropy loss: {loss:.4f}")
    print(f"Perplexity: {ppl:.2f}")
    ```

=== "🎯 Exercises"
    
    ## Practice Exercises
    
    Try these exercises to reinforce your understanding:
    
    ### Exercise 1: Embedding Exploration
    
    **Task**: Create embeddings for a small vocabulary and explore relationships
    
    ```python
    # Create your own word embeddings
    words = ['apple', 'orange', 'banana', 'car', 'truck', 'bicycle']
    
    # Questions to explore:
    # 1. Which words are most similar to 'apple'?
    # 2. Can you find the 'fruit' vs 'vehicle' clustering?
    # 3. How does embedding dimension affect similarity?
    ```
    
    ### Exercise 2: Attention Patterns
    
    **Task**: Visualize attention patterns for different sentences
    
    ```python
    # Try with different sentences:
    sentences = [
        "The cat sat on the mat",
        "The quick brown fox jumps over the lazy dog",
        "To be or not to be, that is the question"
    ]
    
    # Questions to explore:
    # 1. Which words attend to each other most?
    # 2. How does sentence length affect attention?
    # 3. Can you identify grammatical patterns?
    ```
    
    ### Exercise 3: Sampling Comparison
    
    **Task**: Compare different sampling strategies on the same model
    
    ```python
    # Generate text with different strategies
    strategies = ['greedy', 'temperature', 'top_k', 'top_p']
    
    # Questions to explore:
    # 1. Which strategy produces most creative text?
    # 2. Which is most consistent?
    # 3. How do parameters affect output quality?
    ```
    
    ### Exercise 4: Loss Function Analysis
    
    **Task**: Understand how loss changes during training
    
    ```python
    # Simulate training progress
    epochs = [1, 10, 50, 100, 500]
    
    # Questions to explore:
    # 1. How does loss change over time?
    # 2. What does perplexity tell us about model quality?
    # 3. When might training be complete?
    ```

=== "📚 Further Reading"
    
    ## Deep Dive Resources
    
    ### Academic Papers
    
    - **Attention Is All You Need** (Vaswani et al., 2017)
        - The seminal transformer paper
        - Introduced modern attention mechanisms
    
    - **BERT: Pre-training of Deep Bidirectional Transformers** (Devlin et al., 2018)
        - Revolutionized language understanding
        - Contextual embeddings
    
    - **Language Models are Few-Shot Learners** (Brown et al., 2020)
        - GPT-3 and emergence of few-shot learning
        - Scaling laws and emergent abilities
    
    ### Technical Tutorials
    
    - **The Illustrated Transformer** (Jay Alammar)
        - Visual explanation of transformer architecture
        - Excellent for understanding attention
    
    - **Word2Vec Tutorial** (Chris McCormick)
        - Detailed explanation of word embeddings
        - Skip-gram and CBOW methods
    
    ### Interactive Learning
    
    - **Transformer Debugger** (Anthropic)
        - Interactive transformer visualization
        - See attention patterns in real-time
    
    - **Embedding Projector** (TensorFlow)
        - Visualize high-dimensional embeddings
        - Explore word relationships
    
    ### Advanced Topics
    
    - **Attention Mechanisms Survey** (Chaudhari et al., 2021)
        - Comprehensive overview of attention variants
        - Recent developments and applications
    
    - **Sampling Methods for Language Models** (Holtzman et al., 2019)
        - Analysis of different sampling strategies
        - Quality vs diversity trade-offs
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

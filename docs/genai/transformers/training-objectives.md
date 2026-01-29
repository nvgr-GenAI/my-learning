# Training Objectives: How Transformers Learn

!!! tip "🎯 The Learning Strategies"
    Before transformers can be useful, they need to learn from data. But how do you teach a model to understand language? Let's explore the clever training objectives that make transformers so powerful!

## 🎓 The Learning Challenge

Imagine you're teaching someone to understand language, but you have limited time and resources. What would be the most efficient strategies?

**The challenge:**

- 📚 Language is complex and nuanced
- 🌍 We want models that work for many tasks
- ⚡ Training is expensive and time-consuming
- 🎯 We need general understanding, not just memorization

**The solution:** Clever training objectives that teach fundamental language skills!

## 🧠 Pre-training vs Fine-tuning: The Two-Phase Approach

Think of it like human education:

=== "📚 Pre-training (Elementary School)"

    **Goal:** Learn fundamental language skills
    
    **What we teach:**
    - 📖 How words relate to each other
    - 🧩 Grammar and syntax patterns  
    - 💭 Basic reasoning and inference
    - 🌍 World knowledge and facts
    
    **How we teach:** Self-supervised learning on massive text
    
    **Time:** Months of training on powerful computers

=== "🎯 Fine-tuning (Specialized Training)"

    **Goal:** Apply general skills to specific tasks
    
    **What we teach:**
    - 📝 Text classification
    - ❓ Question answering
    - 🌍 Translation
    - 📰 Summarization
    
    **How we teach:** Supervised learning on task-specific data
    
    **Time:** Hours to days of additional training

## 🔍 Pre-training Objectives: The Foundation

### 1. Masked Language Modeling (MLM) - BERT's Strategy

**The idea:** Hide some words and ask the model to guess them.

=== "🎭 The Masking Game"

    **Original sentence:**
    ```
    "The cat sat on the mat because it was comfortable."
    ```
    
    **After masking (15% of tokens):**
    ```
    "The [MASK] sat on the [MASK] because it was comfortable."
    ```
    
    **Model's task:**
    ```
    Predict: [MASK] = "cat" and [MASK] = "mat"
    ```
    
    **Why this works:**
    - Forces model to understand context from both directions
    - Learns relationships between words
    - Develops understanding of grammar and semantics

=== "🧮 Technical Details"

    ```python
    def create_masked_input(text, mask_prob=0.15):
        """
        Create masked language modeling training data
        """
        tokens = tokenizer.tokenize(text)
        masked_tokens = []
        labels = []
        
        for token in tokens:
            if random.random() < mask_prob:
                # 80% of time: replace with [MASK]
                if random.random() < 0.8:
                    masked_tokens.append('[MASK]')
                # 10% of time: replace with random token
                elif random.random() < 0.5:
                    masked_tokens.append(random_token())
                # 10% of time: keep original (helps with bias)
                else:
                    masked_tokens.append(token)
                labels.append(token)  # Original token is the label
            else:
                masked_tokens.append(token)
                labels.append('[IGNORE]')  # Don't compute loss
        
        return masked_tokens, labels
    
    # Example usage
    original = "The cat sat on the mat"
    masked, labels = create_masked_input(original)
    print(f"Masked: {masked}")
    print(f"Labels: {labels}")
    ```

=== "🎯 Learning Outcomes"

    **What MLM teaches:**
    - **Context understanding:** Use surrounding words to infer meaning
    - **Bidirectional reasoning:** Look both left and right
    - **Word relationships:** Similar words appear in similar contexts
    - **Grammar rules:** Syntactic patterns and constraints
    
    **Perfect for:** Classification, question answering, understanding tasks

### 2. Causal Language Modeling (CLM) - GPT's Strategy

**The idea:** Predict the next word given all previous words.

=== "📖 The Storytelling Game"

    **Given context:**
    ```
    "Once upon a time, there was a brave knight who"
    ```
    
    **Model's task:**
    ```
    Predict next word: "lived" or "fought" or "traveled" ...
    ```
    
    **Training process:**
    ```
    Input:  "Once upon a time"           → Predict: "there"
    Input:  "Once upon a time there"     → Predict: "was"  
    Input:  "Once upon a time there was" → Predict: "a"
    ...
    ```

=== "💻 Implementation"

    ```python
    def create_causal_training_data(text):
        """
        Create causal language modeling training data
        """
        tokens = tokenizer.encode(text)
        
        # Each token predicts the next one
        inputs = []
        targets = []
        
        for i in range(len(tokens) - 1):
            inputs.append(tokens[:i+1])    # All tokens up to position i
            targets.append(tokens[i+1])    # Next token
        
        return inputs, targets
    
    # Example
    text = "The cat sat on the mat"
    inputs, targets = create_causal_training_data(text)
    
    for inp, target in zip(inputs[:3], targets[:3]):
        print(f"Input: {tokenizer.decode(inp)} → Target: {tokenizer.decode([target])}")
    # Output:
    # Input: The → Target: cat
    # Input: The cat → Target: sat  
    # Input: The cat sat → Target: on
    ```

=== "🎯 Learning Outcomes"

    **What CLM teaches:**
    - **Sequential reasoning:** Understand how ideas flow
    - **Generation skills:** Create coherent continuations
    - **Probabilistic thinking:** Estimate likelihood of next words
    - **Long-range dependencies:** Connect distant concepts
    
    **Perfect for:** Text generation, chatbots, creative writing

### 3. Sequence-to-Sequence Learning - T5's Strategy

**The idea:** Convert input sequences to output sequences.

=== "🔄 The Translation Game"

    **Task format:**
    ```
    Input:  "translate English to French: Hello world"
    Output: "Bonjour le monde"
    
    Input:  "summarize: [long article text]"
    Output: "[concise summary]"
    
    Input:  "question: What is the capital of France?"
    Output: "Paris"
    ```
    
    **Universal format:** Everything becomes text-to-text!

=== "🧩 Span Corruption (T5's Secret Sauce)"

    ```python
    def span_corruption(text, corruption_rate=0.15):
        """
        T5's pre-training objective: corrupt spans and recover them
        """
        tokens = tokenizer.tokenize(text)
        
        # Identify spans to corrupt (consecutive tokens)
        corrupted_tokens = []
        target_tokens = []
        sentinel_id = 0
        
        i = 0
        while i < len(tokens):
            if random.random() < corruption_rate:
                # Start a corrupted span
                span_start = i
                span_length = random.randint(1, 5)  # Random span length
                span_end = min(span_start + span_length, len(tokens))
                
                # Replace span with sentinel token
                corrupted_tokens.append(f'<extra_id_{sentinel_id}>')
                
                # Add span to targets
                target_tokens.extend([f'<extra_id_{sentinel_id}>'])
                target_tokens.extend(tokens[span_start:span_end])
                
                sentinel_id += 1
                i = span_end
            else:
                corrupted_tokens.append(tokens[i])
                i += 1
        
        return corrupted_tokens, target_tokens
    
    # Example
    text = "The quick brown fox jumps over the lazy dog"
    corrupted, targets = span_corruption(text)
    print(f"Input:  {' '.join(corrupted)}")
    print(f"Target: {' '.join(targets)}")
    # Input:  The <extra_id_0> fox jumps <extra_id_1> lazy dog
    # Target: <extra_id_0> quick brown <extra_id_1> over the
    ```

=== "🎯 Learning Outcomes"

    **What T5's training teaches:**
    - **Flexible reasoning:** Handle any input-output format
    - **Span understanding:** Work with chunks of text
    - **Task generalization:** Apply same skills to different problems
    - **Bidirectional generation:** Use context from both sides
    
    **Perfect for:** Translation, summarization, question answering

## 🎯 Training Objectives Comparison

| Objective | Model Examples | Best For | Key Strength |
|-----------|---------------|----------|--------------|
| **MLM** | BERT, RoBERTa | Understanding tasks | Bidirectional context |
| **CLM** | GPT, ChatGPT | Generation tasks | Autoregressive fluency |
| **Seq2Seq** | T5, BART | Conversion tasks | Flexible input/output |

## 🔧 Advanced Training Techniques

### 1. Next Sentence Prediction (NSP) - BERT's Original Companion

=== "📰 The Headline Game"

    **The task:** Given two sentences, predict if the second follows the first.
    
    ```python
    # Positive example (50% of training data)
    sentence_a = "I went to the store yesterday."
    sentence_b = "I bought some milk and bread."
    label = 1  # IsNext
    
    # Negative example (50% of training data)  
    sentence_a = "I went to the store yesterday."
    sentence_b = "The weather was beautiful today."
    label = 0  # NotNext
    
    # BERT input format:
    # [CLS] sentence_a [SEP] sentence_b [SEP]
    ```

=== "🎯 What NSP Teaches"

    **Benefits:**
    - Document-level understanding
    - Coherence and flow
    - Logical connections
    
    **Limitations:**
    - Later research showed it's not as helpful as expected
    - RoBERTa removed it and performed better
    - Too easy for models to solve using topic similarity

### 2. Sentence Order Prediction (SOP) - ALBERT's Improvement

=== "🔀 The Shuffle Game"

    **Better than NSP:** Instead of random sentences, use consecutive sentences in wrong order.
    
    ```python
    # Original order (positive example)
    sentence_a = "First, I went to the store."
    sentence_b = "Then, I bought some groceries."
    label = 1  # Correct order
    
    # Swapped order (negative example)
    sentence_a = "Then, I bought some groceries."  
    sentence_b = "First, I went to the store."
    label = 0  # Wrong order
    ```
    
    **Why it's better:** Forces model to understand discourse and temporal relationships

### 3. Replaced Token Detection (RTD) - ELECTRA's Innovation

=== "🕵️ The Detective Game"

    **The idea:** Train a generator to create plausible replacements, then train a discriminator to detect them.
    
    ```python
    # Step 1: Generator creates replacements
    original = "The cat sat on the mat"
    generated = "The cat sat on the couch"  # Generator replaces "mat" with "couch"
    
    # Step 2: Discriminator detects replacements
    labels = [0, 0, 0, 0, 0, 1]  # Only "couch" is replaced
    
    # Model learns to detect subtle replacements
    # More efficient than MLM - uses all tokens, not just masked ones
    ```

## 📊 Loss Functions: Measuring Learning

### Cross-Entropy Loss: The Standard Choice

=== "🎯 The Concept"

    **For classification (MLM, CLM):**
    ```python
    import torch.nn.functional as F
    
    def compute_mlm_loss(predictions, targets, mask):
        """
        Compute masked language modeling loss
        """
        # Only compute loss for masked positions
        active_loss = mask.view(-1) == 1
        active_logits = predictions.view(-1, vocab_size)[active_loss]
        active_labels = targets.view(-1)[active_loss]
        
        # Cross-entropy loss
        loss = F.cross_entropy(active_logits, active_labels)
        return loss
    ```

=== "📈 Label Smoothing"

    **The problem:** Model becomes overconfident
    **The solution:** Make targets slightly "soft"
    
    ```python
    def label_smoothing_loss(predictions, targets, smoothing=0.1):
        """
        Apply label smoothing to reduce overconfidence
        """
        vocab_size = predictions.size(-1)
        
        # Convert hard targets to soft targets
        soft_targets = torch.zeros_like(predictions)
        soft_targets.fill_(smoothing / (vocab_size - 1))
        soft_targets.scatter_(-1, targets.unsqueeze(-1), 1.0 - smoothing)
        
        # Compute loss with soft targets
        loss = F.kl_div(F.log_softmax(predictions, dim=-1), soft_targets, reduction='batchmean')
        return loss
    ```

## 🚀 Training Best Practices

### 1. Curriculum Learning: Start Easy, Get Harder

=== "📚 The Learning Progression"

    **Like human education:**
    ```python
    # Stage 1: Short, simple sentences
    easy_data = ["The cat sat.", "I am happy.", "Birds fly."]
    
    # Stage 2: Medium complexity
    medium_data = ["The cat sat on the comfortable mat.", "I am happy because it's sunny."]
    
    # Stage 3: Complex, long sequences  
    hard_data = ["The cat that I saw yesterday was sitting on the mat..."]
    
    # Train progressively
    for stage, data in [(1, easy_data), (2, medium_data), (3, hard_data)]:
        train_model(data, epochs=stage*10)
    ```

### 2. Data Mixing: Balance is Key

=== "⚖️ Optimal Data Composition"

    **For general models:**
    ```python
    data_mix = {
        'web_text': 0.60,      # General knowledge
        'books': 0.15,         # Long-form reasoning  
        'news': 0.10,          # Current events
        'wikipedia': 0.10,     # Factual knowledge
        'code': 0.05          # Structured thinking
    }
    ```

### 3. Learning Rate Scheduling: The Warmup Dance

=== "📈 Optimal Learning Schedule"

    ```python
    def transformer_lr_schedule(step, d_model=512, warmup_steps=4000):
        """
        The famous transformer learning rate schedule
        """
        arg1 = step ** -0.5
        arg2 = step * (warmup_steps ** -1.5)
        
        return d_model ** -0.5 * min(arg1, arg2)
    
    # Why this works:
    # 1. Warmup: Gradual increase prevents instability
    # 2. Decay: Slower learning as training progresses
    # 3. Square root: Balances exploration vs exploitation
    ```

## 🎯 Putting It All Together

Here's a complete training pipeline:

=== "🏗️ Training Setup"

    ```python
    def train_transformer(model, data_loader, objective='mlm'):
        """
        Complete training loop for transformers
        """
        optimizer = AdamW(model.parameters(), lr=1e-4)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=1000,
            num_training_steps=100000
        )
        
        model.train()
        for epoch in range(num_epochs):
            for batch in data_loader:
                # Forward pass
                if objective == 'mlm':
                    outputs = model(**batch)
                    loss = outputs.loss
                elif objective == 'clm':
                    outputs = model(input_ids=batch['input_ids'], 
                                  labels=batch['labels'])
                    loss = outputs.loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Logging
                if step % 100 == 0:
                    print(f"Step {step}, Loss: {loss.item():.4f}")
    ```

## 🌟 Success Metrics

How do we know if training is working?

=== "📊 During Training"

    ```python
    metrics_to_track = {
        'loss': 'Decreasing smoothly?',
        'perplexity': 'exp(loss) - lower is better', 
        'gradient_norm': 'Stable? Not exploding?',
        'learning_rate': 'Following schedule?',
        'tokens_per_second': 'Training efficiently?'
    }
    ```

=== "🎯 After Training"

    **Intrinsic evaluation:**
    - Perplexity on held-out data
    - Masked token prediction accuracy
    - Generated text quality
    
    **Extrinsic evaluation:**
    - Performance on downstream tasks
    - GLUE/SuperGLUE benchmarks
    - Human evaluation of outputs

## 🚀 Ready for Fine-tuning?

Now you understand how transformers learn their foundational skills! This knowledge prepares you for:

- **[Fine-tuning Guide](fine-tuning.md)** - Adapt pre-trained models to specific tasks
- **[Model Zoo](model-zoo.md)** - Explore different pre-trained models
- **[Applications](applications.md)** - See training objectives in action

---

!!! success "🎓 Training Mastered!"
    You now understand the clever strategies that teach transformers to understand language. These training objectives are the foundation of all modern AI systems!

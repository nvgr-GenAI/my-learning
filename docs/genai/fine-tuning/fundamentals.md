# Fine-tuning Fundamentals

## What is Fine-tuning?

Fine-tuning is the process of adapting a pre-trained language model to perform specific tasks or exhibit particular behaviors by training it on a smaller, task-specific dataset.

## Why Fine-tune?

### Advantages

**Task Specialization**:
- Better performance on specific domains
- Improved understanding of domain terminology
- More consistent outputs for specialized tasks

**Data Efficiency**:
- Requires less training data than training from scratch
- Leverages pre-existing knowledge from base model
- Faster training times and lower computational costs

**Customization**:
- Align model behavior with specific requirements
- Incorporate proprietary or domain-specific knowledge
- Control model personality and response style

### When to Fine-tune

**Good Candidates**:
- Domain-specific tasks (medical, legal, financial)
- Consistent formatting requirements
- Specialized reasoning patterns
- Custom data that's not in training set

**Consider Alternatives**:
- Simple tasks (prompt engineering might suffice)
- Very small datasets (few-shot learning)
- Rapidly changing requirements (RAG systems)

## Fine-tuning Approaches

### 1. Full Fine-tuning

**Process**: Update all model parameters during training

**Characteristics**:
- Maximum customization potential
- Highest computational requirements
- Risk of catastrophic forgetting

```python
def full_fine_tune(model, dataset, config):
    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate
    )
    
    for epoch in range(config.num_epochs):
        for batch in dataset:
            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            optimizer.step()
    
    return model
```

### 2. Parameter-Efficient Fine-tuning (PEFT)

**Concept**: Update only a small subset of parameters

**Benefits**:
- Significantly reduced memory requirements
- Faster training and inference
- Better preservation of original capabilities
- Easier deployment and version management

#### Low-Rank Adaptation (LoRA)

**Principle**: Decompose weight updates into low-rank matrices

```python
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # Frozen pre-trained weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False
        
        # Trainable low-rank decomposition
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.randn(out_features, rank))
        
        # Initialize B to zero for stable training
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        # Original computation
        result = F.linear(x, self.weight)
        
        # LoRA adaptation
        lora_result = F.linear(F.linear(x, self.lora_A), self.lora_B)
        
        return result + (self.alpha / self.rank) * lora_result
```

#### Prefix Tuning

**Concept**: Add trainable prefix tokens to input sequences

```python
class PrefixTuning(nn.Module):
    def __init__(self, model, prefix_length=10):
        super().__init__()
        self.model = model
        self.prefix_length = prefix_length
        
        # Trainable prefix embeddings
        self.prefix_embeddings = nn.Parameter(
            torch.randn(prefix_length, model.config.hidden_size)
        )
    
    def forward(self, input_ids, attention_mask=None):
        batch_size = input_ids.size(0)
        
        # Expand prefix for batch
        prefix = self.prefix_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        
        # Get input embeddings
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        # Concatenate prefix with inputs
        inputs_embeds = torch.cat([prefix, inputs_embeds], dim=1)
        
        # Adjust attention mask
        if attention_mask is not None:
            prefix_mask = torch.ones(
                batch_size, self.prefix_length,
                device=attention_mask.device
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
```

#### Adapter Layers

**Concept**: Insert small trainable modules between existing layers

```python
class AdapterLayer(nn.Module):
    def __init__(self, hidden_size, adapter_size=64):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(adapter_size, hidden_size)
        
        # Initialize for identity mapping
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return residual + x  # Residual connection
```

### 3. Instruction Tuning

**Purpose**: Teach model to follow instructions and respond appropriately

**Dataset Format**:
```json
{
  "instruction": "Summarize the following text in one sentence.",
  "input": "Long text to be summarized...",
  "output": "Concise summary sentence."
}
```

**Training Process**:
```python
def prepare_instruction_data(examples):
    prompts = []
    responses = []
    
    for example in examples:
        # Format as instruction-following prompt
        if example.get('input'):
            prompt = f"Instruction: {example['instruction']}\nInput: {example['input']}\nResponse:"
        else:
            prompt = f"Instruction: {example['instruction']}\nResponse:"
        
        prompts.append(prompt)
        responses.append(example['output'])
    
    return prompts, responses

def instruction_tune(model, tokenizer, dataset):
    prompts, responses = prepare_instruction_data(dataset)
    
    for prompt, response in zip(prompts, responses):
        # Tokenize full sequence
        full_text = prompt + " " + response
        tokens = tokenizer(full_text, return_tensors="pt")
        
        # Create labels (mask prompt tokens)
        labels = tokens.input_ids.clone()
        prompt_length = len(tokenizer(prompt).input_ids)
        labels[:, :prompt_length] = -100  # Ignore prompt in loss
        
        # Compute loss only on response tokens
        outputs = model(input_ids=tokens.input_ids, labels=labels)
        loss = outputs.loss
        
        # Backpropagation
        loss.backward()
```

## Training Data Preparation

### Data Quality Principles

**High Quality Examples**:
- Accurate and consistent labeling
- Diverse input variations
- Representative of target use cases
- Balanced across different scenarios

**Data Formatting**:
```python
def format_training_data(raw_data, task_type):
    formatted_data = []
    
    if task_type == "qa":
        for item in raw_data:
            formatted_data.append({
                "input": f"Question: {item['question']}\nAnswer:",
                "output": item['answer']
            })
    
    elif task_type == "classification":
        for item in raw_data:
            formatted_data.append({
                "input": f"Classify the following text: {item['text']}\nCategory:",
                "output": item['label']
            })
    
    elif task_type == "summarization":
        for item in raw_data:
            formatted_data.append({
                "input": f"Summarize: {item['document']}\nSummary:",
                "output": item['summary']
            })
    
    return formatted_data
```

### Data Augmentation

**Techniques**:
- Paraphrasing existing examples
- Adding noise or variations
- Synthetic data generation
- Cross-linguistic translation

```python
def augment_training_data(original_data, augmentation_factor=2):
    augmented_data = original_data.copy()
    
    for _ in range(augmentation_factor):
        for example in original_data:
            # Paraphrase the input
            paraphrased_input = paraphrase_text(example['input'])
            
            # Create augmented example
            augmented_example = {
                'input': paraphrased_input,
                'output': example['output']
            }
            
            augmented_data.append(augmented_example)
    
    return augmented_data
```

## Training Configuration

### Hyperparameter Selection

**Learning Rate**:
- Start with 1e-5 to 1e-4 for full fine-tuning
- Use 1e-3 to 1e-2 for PEFT methods
- Consider learning rate scheduling

**Batch Size**:
- Balance between memory constraints and training stability
- Use gradient accumulation for effective larger batches
- Typical range: 4-32 for fine-tuning

**Training Duration**:
- Monitor validation loss to avoid overfitting
- Early stopping based on performance plateaus
- Typically 1-10 epochs depending on dataset size

```python
class FineTuningConfig:
    def __init__(self):
        self.learning_rate = 2e-5
        self.batch_size = 8
        self.num_epochs = 3
        self.warmup_steps = 100
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0
        self.save_steps = 500
        self.eval_steps = 100
        self.logging_steps = 10

def setup_training(model, config):
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.total_steps
    )
    
    return optimizer, scheduler
```

## Evaluation and Validation

### Evaluation Metrics

**Task-Specific Metrics**:
- **Classification**: Accuracy, F1-score, Precision, Recall
- **Generation**: BLEU, ROUGE, BERTScore
- **Question Answering**: Exact Match, F1
- **Summarization**: ROUGE-1, ROUGE-2, ROUGE-L

**General Quality Metrics**:
- Perplexity on validation set
- Human evaluation scores
- Consistency across similar inputs
- Alignment with instructions

```python
def evaluate_model(model, eval_dataset, metrics):
    model.eval()
    results = {}
    
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in eval_dataset:
            outputs = model.generate(
                batch['input_ids'],
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
            
            predictions.extend(outputs)
            references.extend(batch['labels'])
    
    # Compute metrics
    for metric_name, metric_func in metrics.items():
        results[metric_name] = metric_func(predictions, references)
    
    return results
```

### Validation Strategies

**Train/Validation/Test Split**:
- 70-80% training data
- 10-15% validation for hyperparameter tuning
- 10-15% test for final evaluation

**Cross-Validation**:
- K-fold validation for small datasets
- Stratified sampling for imbalanced data
- Time-based splits for temporal data

**Early Stopping**:
```python
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.wait = 0
    
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
            return False
        
        if val_score < self.best_score - self.min_delta:
            self.wait += 1
            if self.wait >= self.patience:
                return True
        else:
            self.best_score = val_score
            self.wait = 0
        
        return False
```

## Common Challenges and Solutions

### Overfitting

**Symptoms**:
- Training loss decreases but validation loss increases
- Perfect training accuracy but poor test performance
- Model memorizes training examples

**Solutions**:
- Regularization techniques (dropout, weight decay)
- Early stopping based on validation performance
- Data augmentation to increase dataset diversity
- Reduce model complexity or training duration

### Catastrophic Forgetting

**Problem**: Model loses previously learned capabilities

**Mitigation Strategies**:
- Use PEFT methods instead of full fine-tuning
- Mix original training data with fine-tuning data
- Implement knowledge distillation
- Regular evaluation on general benchmarks

### Data Quality Issues

**Common Problems**:
- Inconsistent labeling
- Distribution mismatch between train and test
- Insufficient data for complex tasks

**Solutions**:
- Implement data quality checks
- Use active learning for better annotation
- Augment data strategically
- Consider few-shot learning approaches

## Best Practices

### Data Preparation
1. **Quality over Quantity**: Focus on high-quality, consistent examples
2. **Diversity**: Include varied examples covering edge cases
3. **Balance**: Maintain balanced representation across categories
4. **Validation**: Hold out representative test data

### Training Process
1. **Start Small**: Begin with a subset of data for quick iteration
2. **Monitor Carefully**: Track both training and validation metrics
3. **Save Checkpoints**: Regular saving for recovery and comparison
4. **Experiment Tracking**: Log all hyperparameters and results

### Model Management
1. **Version Control**: Track model versions and training configurations
2. **Documentation**: Document training process and performance
3. **Testing**: Comprehensive evaluation before deployment
4. **Monitoring**: Continuous performance monitoring in production

## Next Steps

- [LoRA Implementation](lora.md)
- [RLHF Training](rlhf.md)
- [Custom Training Pipelines](custom-training.md)

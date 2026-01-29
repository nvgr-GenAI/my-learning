# Tokenization: From Text to Numbers

!!! tip "🔤 Breaking Down Language"
    Before transformers can understand text, they need to convert it into numbers. This process is called tokenization - think of it as teaching AI to read by breaking words into digestible pieces!

## 🧩 What is Tokenization?

Imagine you're learning a new language where the words are incredibly long and complex. How would you approach it?

**Human approach:**

- 📚 Start with simple sounds and syllables
- 🧩 Combine them to make words
- 📖 Build vocabulary gradually
- 💡 Understand that some pieces appear in many words

**Tokenization does exactly this for AI!**

### The Challenge

=== "🤔 The Problem"

    **Computers need numbers, but language is messy:**
    
    ```
    Original text: "I'm running to the supermarket"
    
    Problems:
    - "I'm" = "I" + "'m" (contraction)
    - "running" = "run" + "ning" (suffix) 
    - "supermarket" = "super" + "market" (compound)
    - Punctuation mixed with letters
    - Different forms of the same word
    ```
    
    **If we treated each unique word as one token:**
    - Vocabulary would be HUGE (millions of words)
    - New words would break the system
    - Similar words wouldn't share understanding

=== "💡 The Solution"

    **Subword tokenization - break words into meaningful pieces:**
    
    ```
    "I'm running to the supermarket"
    ↓
    ["I", "'m", "run", "ning", "to", "the", "super", "market"]
    ```
    
    **Benefits:**
    - ✅ Smaller vocabulary (thousands, not millions)
    - ✅ Handles new words by combining known pieces
    - ✅ Similar words share components
    - ✅ Captures meaning relationships

## 🔧 Popular Tokenization Methods

### 1. Byte-Pair Encoding (BPE) - The Smart Merger

**The Story:** Imagine you're organizing a library and notice that certain letter combinations appear together very frequently.

=== "📚 How BPE Works"

    **Step 1: Start with characters**
    ```
    Text: "hello hello world"
    Initial: ['h', 'e', 'l', 'l', 'o', ' ', 'h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']
    ```
    
    **Step 2: Find most frequent pairs**
    ```
    Count pairs:
    'l', 'l' → appears 4 times (most frequent!)
    'h', 'e' → appears 2 times
    'e', 'l' → appears 2 times
    ```
    
    **Step 3: Merge the most frequent pair**
    ```
    Merge 'l' + 'l' → 'll'
    Result: ['h', 'e', 'll', 'o', ' ', 'h', 'e', 'll', 'o', ' ', 'w', 'o', 'r', 'l', 'd']
    ```
    
    **Step 4: Repeat until desired vocabulary size**
    ```
    Next merge: 'e' + 'll' → 'ell'
    Then: 'h' + 'ell' → 'hell'
    Finally: 'hell' + 'o' → 'hello'
    
    Final: ['hello', ' ', 'hello', ' ', 'w', 'o', 'r', 'l', 'd']
    ```

=== "🎯 Real Example"

    ```python
    # Example with GPT tokenizer
    from transformers import GPT2Tokenizer
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    text = "The quick brown fox jumps"
    tokens = tokenizer.tokenize(text)
    print(f"Original: {text}")
    print(f"Tokens: {tokens}")
    # Output: ['The', 'Ġquick', 'Ġbrown', 'Ġfox', 'Ġjumps']
    # Note: 'Ġ' represents a space character
    ```

### 2. WordPiece - The Google Approach

**Used by:** BERT, DistilBERT

**The idea:** Instead of just frequency, consider how much information each merge gives us.

=== "🧠 How WordPiece Differs"

    **BPE says:** "Merge the most frequent pair"
    **WordPiece says:** "Merge the pair that gives us the most information"
    
    **Information formula:**
    ```
    Score = freq(pair) / (freq(first) × freq(second))
    ```
    
    **Why this matters:**
    - Merges that create meaningful units get higher scores
    - More linguistically motivated than pure frequency
    - Better handles morphological patterns

=== "📝 Example"

    ```python
    # BERT tokenizer example
    from transformers import BertTokenizer
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    text = "unhappiness"
    tokens = tokenizer.tokenize(text)
    print(f"Original: {text}")
    print(f"Tokens: {tokens}")
    # Output: ['un', '##happiness'] or ['un', '##hap', '##piness']
    # Note: '##' indicates continuation of a word
    ```

### 3. SentencePiece - The Universal Solution

**Used by:** T5, mT5, many multilingual models

**The breakthrough:** Treats text as a sequence of Unicode characters, not words.

=== "🌍 Why SentencePiece is Special"

    **Traditional tokenizers:**
    - Assume words are separated by spaces
    - Break on languages like Chinese/Japanese (no spaces!)
    - Struggle with different scripts
    
    **SentencePiece:**
    - Treats entire text as one long sequence
    - Works with ANY language
    - Handles mixed scripts naturally
    
    ```python
    import sentencepiece as spm
    
    # Train a SentencePiece model
    spm.SentencePieceTrainer.train(
        input='text.txt', 
        model_prefix='m', 
        vocab_size=8000
    )
    
    # Use the model
    sp = spm.SentencePieceProcessor(model_file='m.model')
    tokens = sp.encode('Hello world', out_type=str)
    print(tokens)
    # Output: ['▁Hello', '▁world'] 
    # Note: '▁' represents word boundary
    ```

## 🔢 From Tokens to Numbers

Once we have tokens, we need to convert them to numbers:

=== "📖 The Vocabulary"

    **Every tokenizer has a vocabulary - a mapping of tokens to IDs:**
    
    ```python
    vocabulary = {
        "[PAD]": 0,     # Padding token
        "[UNK]": 1,     # Unknown token  
        "[CLS]": 2,     # Classification token (BERT)
        "[SEP]": 3,     # Separator token (BERT)
        "the": 4,
        "a": 5,
        "an": 6,
        "hello": 7,
        "world": 8,
        "##ing": 9,     # Subword piece
        # ... thousands more
    }
    ```

=== "🔄 Encoding Process"

    ```python
    # Complete tokenization pipeline
    text = "Hello world!"
    
    # Step 1: Tokenize
    tokens = tokenizer.tokenize(text)
    # Result: ['Hello', 'world', '!']
    
    # Step 2: Convert to IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # Result: [7, 8, 15]  # Numbers from vocabulary
    
    # Step 3: Add special tokens (for BERT)
    input_ids = [2] + input_ids + [3]  # [CLS] + tokens + [SEP]
    # Result: [2, 7, 8, 15, 3]
    
    # One-liner for the whole process:
    input_ids = tokenizer.encode(text)
    ```

=== "🔄 Decoding Process"

    ```python
    # Convert back to text
    input_ids = [2, 7, 8, 15, 3]
    
    # Method 1: Step by step
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    text = tokenizer.convert_tokens_to_string(tokens)
    
    # Method 2: Direct decoding
    text = tokenizer.decode(input_ids)
    # Result: "Hello world!"
    ```

## 🎭 Special Tokens: The Control Characters

Different models use different special tokens for different purposes:

=== "🔍 BERT Tokens"

    ```python
    tokens = {
        "[PAD]": "Padding - make all sequences same length",
        "[UNK]": "Unknown - for words not in vocabulary", 
        "[CLS]": "Classification - start of sequence",
        "[SEP]": "Separator - between sentences",
        "[MASK]": "Mask - for masked language modeling"
    }
    
    # Example usage
    text1 = "Hello world"
    text2 = "How are you?"
    
    # BERT format for sentence pair:
    # [CLS] text1 [SEP] text2 [SEP]
    encoded = "[CLS] Hello world [SEP] How are you? [SEP]"
    ```

=== "🤖 GPT Tokens"

    ```python
    tokens = {
        "<|endoftext|>": "End of document/sequence",
        "<|pad|>": "Padding token",
        # GPT uses BPE, so no [MASK] or [CLS] tokens
    }
    
    # Example usage
    text = "The quick brown fox<|endoftext|>"
    ```

=== "🌍 T5 Tokens"

    ```python
    tokens = {
        "<pad>": "Padding token",
        "</s>": "End of sequence", 
        "<unk>": "Unknown token",
        "<extra_id_0>": "Special sentinel for span corruption"
    }
    ```

## 🧪 Hands-On: Exploring Tokenization

Let's experiment with different tokenizers:

=== "💻 Setup Code"

    ```python
    from transformers import (
        GPT2Tokenizer, 
        BertTokenizer, 
        T5Tokenizer,
        AutoTokenizer
    )
    
    # Load different tokenizers
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
    
    # Test text
    text = "I'm running to the supermarket"
    ```

=== "🔍 Compare Results"

    ```python
    def compare_tokenizers(text):
        tokenizers = {
            'GPT-2 (BPE)': gpt2_tokenizer,
            'BERT (WordPiece)': bert_tokenizer, 
            'T5 (SentencePiece)': t5_tokenizer
        }
        
        print(f"Original text: '{text}'")
        print("-" * 50)
        
        for name, tokenizer in tokenizers.items():
            tokens = tokenizer.tokenize(text)
            ids = tokenizer.encode(text)
            
            print(f"{name}:")
            print(f"  Tokens: {tokens}")
            print(f"  Token count: {len(tokens)}")
            print(f"  IDs: {ids}")
            print()
    
    # Try different examples
    compare_tokenizers("I'm running to the supermarket")
    compare_tokenizers("The quick brown fox jumps")
    compare_tokenizers("Transformer models are amazing!")
    ```

=== "🎯 Results Analysis"

    **Typical output:**
    ```
    Original text: 'I'm running to the supermarket'
    
    GPT-2 (BPE):
      Tokens: ['I', "'m", 'Ġrunning', 'Ġto', 'Ġthe', 'Ġsuper', 'market']
      Token count: 7
    
    BERT (WordPiece): 
      Tokens: ['i', "'", 'm', 'running', 'to', 'the', 'super', '##market']
      Token count: 8
    
    T5 (SentencePiece):
      Tokens: ['▁I', "'", 'm', '▁running', '▁to', '▁the', '▁supermarket']
      Token count: 7
    ```
    
    **Key observations:**
    - Different tokenizers split the same text differently
    - BPE tends to be more aggressive in merging
    - WordPiece preserves more linguistic boundaries
    - SentencePiece handles subwords cleanly

## 🎯 Best Practices and Tips

### Vocabulary Size Considerations

=== "📊 Size Trade-offs"

    | Vocab Size | Pros | Cons | Best For |
    |------------|------|------|----------|
    | **Small (8K-16K)** | Faster, less memory | More subword splits | Resource-constrained |
    | **Medium (32K-50K)** | Good balance | Standard choice | Most applications |
    | **Large (100K+)** | Fewer splits, more words | Slower, more memory | Rich languages |

=== "🎯 Choosing the Right Size"

    ```python
    # Analyze your text to choose vocabulary size
    def analyze_tokenization(text, vocab_sizes=[8000, 16000, 32000, 50000]):
        for vocab_size in vocab_sizes:
            # This is conceptual - actual training needs large corpus
            tokenizer = train_tokenizer(text, vocab_size=vocab_size)
            tokens = tokenizer.tokenize(text)
            
            avg_tokens_per_word = len(tokens) / len(text.split())
            print(f"Vocab {vocab_size}: {avg_tokens_per_word:.2f} tokens/word")
    
    # Rule of thumb: aim for 1.2-1.5 tokens per word
    ```

### Handling Special Cases

=== "🌍 Multilingual Considerations"

    ```python
    # For multilingual models
    multilingual_text = "Hello 世界 مرحبا Hola"
    
    # Use SentencePiece for best multilingual support
    from transformers import XLMRobertaTokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    
    tokens = tokenizer.tokenize(multilingual_text)
    print(f"Multilingual tokens: {tokens}")
    # Handles all scripts gracefully
    ```

=== "💻 Code and Technical Text"

    ```python
    # For code tokenization
    code_text = "def hello_world(): print('Hello, World!')"
    
    # Code-specific tokenizers preserve programming constructs
    from transformers import CodeBertTokenizer
    code_tokenizer = CodeBertTokenizer.from_pretrained('microsoft/codebert-base')
    
    tokens = code_tokenizer.tokenize(code_text)
    print(f"Code tokens: {tokens}")
    # Better handling of programming syntax
    ```

## 🚀 Impact on Model Performance

Understanding tokenization helps you:

### Debug Model Behavior

=== "🔍 Attention Patterns"

    ```python
    # Visualize how tokenization affects attention
    text = "unhappiness"
    
    # WordPiece splits it as: ['un', '##happiness']  
    # Attention might connect:
    # - 'un' with other negative prefixes
    # - '##happiness' with other emotion words
    
    # This explains why BERT understands:
    # "un" + "happy" + "ness" = negative emotion concept
    ```

=== "⚡ Generation Quality"

    ```python
    # For text generation
    prompt = "The artificial intelligence"
    
    # Good tokenization preserves meaningful units:
    # ['The', 'artificial', 'intelligence'] 
    # vs bad: ['The', 'art', 'ificial', 'int', 'elligence']
    
    # Better tokenization → Better generation
    ```

### Optimize for Your Domain

=== "🎯 Domain-Specific Vocabulary"

    **For medical text:**
    ```
    "pneumonia" → ['pneumonia'] (one token)
    vs generic: ['pne', 'umon', 'ia'] (three tokens)
    ```
    
    **For legal text:**
    ```
    "defendant" → ['defendant'] (one token)  
    vs generic: ['def', 'end', 'ant'] (three tokens)
    ```
    
    **Training your own tokenizer can significantly improve performance!**

## 🎉 Ready for the Next Step?

Now you understand how transformers "see" text! This knowledge will help you:

- 🧠 **Understand model behavior** - Why certain words are treated similarly
- 🎯 **Debug performance issues** - Check if tokenization makes sense
- ⚡ **Optimize for your domain** - Train custom tokenizers when needed
- 💡 **Choose the right model** - Different tokenizers for different tasks

**Next up:** [Architecture Deep Dive](architecture.md) - See how tokenized text flows through transformer layers!

---

!!! success "🔤 Tokenization Mastered!"
    You now understand how text becomes numbers that transformers can process. This foundation will help you understand everything else about how these models work!

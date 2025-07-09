# Mathematics for GenAI

!!! abstract "Mathematical Foundations"
    Master the essential mathematical concepts that power generative AI systems. This section focuses on intuitive understanding backed by practical examples.

## Why Mathematics Matters in GenAI

Understanding the mathematics behind GenAI isn't just academic—it's practical:

- **Better Prompting**: Understand how models process information
- **Efficient Implementation**: Optimize performance and memory usage
- **Debugging**: Diagnose issues when things go wrong
- **Innovation**: Build upon existing techniques for new applications

=== "📚 Explanation"

    ## Linear Algebra Essentials

    ### Vectors: The Building Blocks

    Everything in GenAI starts with vectors—lists of numbers that represent information.

    **Think of vectors as coordinates in space:**
    - A word embedding is like a GPS coordinate for a word's meaning
    - Similar words have similar coordinates (close to each other)
    - Each dimension captures a different aspect of meaning

    **Key insight**: If you can represent information as numbers, you can do math with it!

    ### Matrices: Transforming Information

    Matrices transform vectors from one representation to another.

    **Think of matrices as transformation machines:**
    - Input a vector representing a sentence
    - Output a vector representing what the AI should say next
    - Each layer in a neural network is a different transformation

    **Real-world analogy**: Like a translator that converts thoughts from one language to another.

    ### The Attention Matrix

    The heart of transformers is matrix operations that compute attention.

    **Attention is like selective focus:**
    - When reading "The cat sat on the mat", which words matter most?
    - Attention weights tell us: "cat" and "mat" are important for understanding location
    - This helps the model focus on relevant parts of long text

    ## Probability and Statistics

    ### Why Probability Matters

    AI models are fundamentally probabilistic—they predict the **likelihood** of different outcomes.

    **Key insight**: AI doesn't "know" answers—it predicts the most likely response based on training data.

    **Real-world example**: 
    - Input: "The weather is sunny, so I'll wear my..."
    - Model thinks: "sunglasses" (70%), "hat" (20%), "jacket" (10%)
    - It picks the most probable completion

    ### Distributions in GenAI

    **Softmax function**: The mathematical way to convert any numbers into probabilities.

    **Why it matters**: 
    - Raw model outputs are just numbers (logits)
    - Softmax converts them to probabilities that sum to 100%
    - Higher numbers become higher probabilities

    ## Calculus for Optimization

    ### Gradients: The Direction of Improvement

    Models learn by following gradients—the direction that reduces error.

    **Think of gradients as compass directions:**
    - You're lost on a mountain (high error)
    - Gradient points toward the valley (lower error)
    - Take small steps in that direction until you reach the bottom

    **Learning process**: Model makes mistake → calculates gradient → adjusts weights → gets better

    ### Backpropagation: Chain Rule in Action

    **The "blame assignment" problem:**
    - Model makes wrong prediction
    - Which part of the network is responsible?
    - Backpropagation traces the error backward through layers
    - Each layer gets blamed proportionally and adjusts accordingly

    **Chain rule intuition**: If A affects B, and B affects C, then changes in A indirectly affect C.

    ## Information Theory

    ### Entropy: Measuring Uncertainty

    **Entropy measures how "surprising" information is:**
    - Low entropy = predictable (weather report: "sunny in Arizona")
    - High entropy = unpredictable (lottery numbers)
    - GenAI models try to predict patterns, so they prefer low entropy

    ### Cross-Entropy Loss

    **How we measure how "wrong" a model is:**
    - True answer: "The sky is blue"
    - Model says: "The sky is green" (very wrong = high loss)
    - Model says: "The sky is blue" (correct = low loss)

    **Training goal**: Minimize cross-entropy loss across all examples.

    ## Practical Applications

    ### Attention Computation

    **Scaled dot-product attention** is the core of transformer models:

    **The process:**
    1. **Query**: What am I looking for?
    2. **Key**: What information do I have?
    3. **Value**: What should I actually use?
    4. **Attention**: How much should I focus on each piece?

    **Real example**: In "The cat sat on the mat", when processing "mat", the model pays attention to "cat" and "sat" to understand the relationship.

    ### Embedding Arithmetic

    **Famous example**: King - Man + Woman ≈ Queen

    **What this means:**
    - Word embeddings capture semantic relationships
    - You can do math with meanings!
    - Vector arithmetic reveals hidden patterns in language

    **Other examples:**
    - Paris - France + Italy ≈ Rome
    - Walking - Walk + Swimming ≈ Swim

=== "💻 Code Examples"

    ## Linear Algebra Implementation

    ### Vector Operations

    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    # A word embedding is just a vector
    word_embedding = np.array([0.2, -0.1, 0.8, 0.3, -0.5])
    print(f"Word embedding: {word_embedding}")

    # Similarity between words = similarity between vectors
    def cosine_similarity(vec1, vec2):
        """Calculate similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)
        return dot_product / (magnitude1 * magnitude2)

    # Example: Similar words have similar embeddings
    king = np.array([0.5, 0.8, 0.2, 0.9, 0.1])
    queen = np.array([0.4, 0.7, 0.3, 0.8, 0.2])
    cat = np.array([-0.3, 0.1, 0.6, -0.2, 0.8])

    print(f"King-Queen similarity: {cosine_similarity(king, queen):.3f}")
    print(f"King-Cat similarity: {cosine_similarity(king, cat):.3f}")
    ```

    ### Matrix Operations

    ```python
    # Neural network layer: matrix multiplication
    input_dim = 4
    hidden_dim = 3

    # Weight matrix transforms input to hidden representation
    W = np.random.randn(input_dim, hidden_dim)
    b = np.zeros(hidden_dim)

    # Input vector
    x = np.array([1.0, 0.5, -0.3, 0.8])

    # Linear transformation
    hidden = np.dot(x, W) + b
    print(f"Input: {x}")
    print(f"Hidden: {hidden}")
    ```

    ### Attention Mechanism

    ```python
    # Simplified attention calculation
    def simple_attention(query, key, value):
        """
        Args:
            query: What we're looking for [seq_len, d_model]
            key: What we're comparing against [seq_len, d_model]
            value: What we actually use [seq_len, d_model]
        """
        # Step 1: Calculate attention scores
        scores = np.dot(query, key.T)  # [seq_len, seq_len]
        
        # Step 2: Normalize scores
        attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
        
        # Step 3: Apply attention to values
        output = np.dot(attention_weights, value)
        
        return output, attention_weights

    # Example with 3 words
    seq_len, d_model = 3, 4
    query = np.random.randn(seq_len, d_model)
    key = np.random.randn(seq_len, d_model)
    value = np.random.randn(seq_len, d_model)

    output, weights = simple_attention(query, key, value)
    print(f"Attention weights:\n{weights}")
    ```

    ## Probability and Statistics

    ### Language Model Probabilities

    ```python
    # Language model predicts next word probability
    vocabulary = ['the', 'cat', 'sat', 'on', 'mat']
    probabilities = [0.3, 0.2, 0.25, 0.15, 0.1]

    # Higher probability = more likely next word
    print("Next word predictions:")
    for word, prob in zip(vocabulary, probabilities):
        print(f"'{word}': {prob:.1%}")
    ```

    ### Softmax Function

    ```python
    # Softmax: Convert numbers to probabilities
    def softmax(x):
        exp_x = np.exp(x - np.max(x))  # Numerical stability
        return exp_x / np.sum(exp_x)

    # Model outputs (logits)
    logits = np.array([2.0, 1.0, 0.5, 3.0])
    probs = softmax(logits)

    print(f"Logits: {logits}")
    print(f"Probabilities: {probs}")
    print(f"Sum: {np.sum(probs):.3f}")  # Should be 1.0
    ```

    ## Calculus for Optimization

    ### Gradient Descent

    ```python
    # Simple loss function
    def loss_function(w):
        return (w - 2) ** 2  # Minimum at w = 2

    # Gradient (derivative)
    def gradient(w):
        return 2 * (w - 2)

    # Gradient descent
    w = 0.0  # Starting point
    learning_rate = 0.1

    for step in range(10):
        grad = gradient(w)
        w = w - learning_rate * grad
        loss = loss_function(w)
        print(f"Step {step}: w={w:.3f}, loss={loss:.3f}")
    ```

    ### Backpropagation Example

    ```python
    # Simple 2-layer network
    def forward_pass(x, w1, w2):
        h = np.maximum(0, x * w1)  # ReLU activation
        y = h * w2
        return y, h

    def backward_pass(x, y_pred, y_true, w1, w2, h):
        # Loss gradient
        dy = 2 * (y_pred - y_true)
        
        # Gradients through layers
        dw2 = dy * h
        dh = dy * w2
        
        # ReLU gradient
        dw1 = dh * x if h > 0 else 0
        
        return dw1, dw2

    # Training example
    x, y_true = 1.0, 3.0
    w1, w2 = 0.5, 0.5

    for epoch in range(5):
        y_pred, h = forward_pass(x, w1, w2)
        dw1, dw2 = backward_pass(x, y_pred, y_true, w1, w2, h)
        
        # Update weights
        w1 -= 0.1 * dw1
        w2 -= 0.1 * dw2
        
        loss = (y_pred - y_true) ** 2
        print(f"Epoch {epoch}: loss={loss:.3f}, w1={w1:.3f}, w2={w2:.3f}")
    ```

    ## Information Theory

    ### Entropy Calculation

    ```python
    # Entropy measures information content
    def entropy(probabilities):
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    # Low entropy = predictable
    certain_probs = np.array([0.9, 0.05, 0.05])
    print(f"Certain distribution entropy: {entropy(certain_probs):.3f}")

    # High entropy = unpredictable
    uncertain_probs = np.array([0.33, 0.33, 0.34])
    print(f"Uncertain distribution entropy: {entropy(uncertain_probs):.3f}")
    ```

    ### Cross-Entropy Loss

    ```python
    # Cross-entropy: How far off are our predictions?
    def cross_entropy_loss(y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred + 1e-10))

    # True answer (one-hot encoded)
    true_label = np.array([0, 1, 0])  # Word 2 is correct

    # Model predictions
    good_prediction = np.array([0.1, 0.8, 0.1])
    bad_prediction = np.array([0.4, 0.3, 0.3])

    print(f"Good prediction loss: {cross_entropy_loss(true_label, good_prediction):.3f}")
    print(f"Bad prediction loss: {cross_entropy_loss(true_label, bad_prediction):.3f}")
    ```

    ## Practical Applications

    ### Scaled Dot-Product Attention

    ```python
    # Scaled dot-product attention (simplified)
    def scaled_dot_product_attention(Q, K, V):
        """
        Args:
            Q: Query matrix [seq_len, d_k]
            K: Key matrix [seq_len, d_k]
            V: Value matrix [seq_len, d_v]
        """
        d_k = Q.shape[-1]
        
        # Attention scores
        scores = np.dot(Q, K.T) / np.sqrt(d_k)
        
        # Attention weights
        attention_weights = softmax(scores)
        
        # Weighted values
        output = np.dot(attention_weights, V)
        
        return output, attention_weights

    # Example usage
    seq_len, d_k, d_v = 4, 8, 8
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_v)

    output, attention = scaled_dot_product_attention(Q, K, V)
    print(f"Attention pattern:\n{attention}")
    ```

    ### Embedding Arithmetic

    ```python
    # Famous example: King - Man + Woman ≈ Queen
    king = np.array([0.5, 0.8, 0.2, 0.9])
    man = np.array([0.3, 0.4, 0.1, 0.6])
    woman = np.array([0.2, 0.7, 0.3, 0.8])

    # Vector arithmetic
    result = king - man + woman
    print(f"King - Man + Woman = {result}")

    # This should be close to queen embedding
    queen = np.array([0.4, 0.9, 0.4, 0.9])
    similarity = cosine_similarity(result, queen)
    print(f"Similarity to Queen: {similarity:.3f}")
    ```

=== "🎯 Exercises"

    ## Beginner Level

    ### 1. Vector Similarity Practice
    - Create embeddings for words like "happy", "sad", "joyful"
    - Calculate cosine similarity between them
    - Which words are most similar?

    ### 2. Softmax Understanding
    - Given logits [1.0, 2.0, 3.0], calculate softmax probabilities
    - What happens when you multiply all logits by 2?
    - What happens when you add 1 to all logits?

    ### 3. Gradient Descent Visualization
    - Try different learning rates (0.01, 0.1, 0.5)
    - Plot the loss over time
    - What happens with very high learning rates?

    ## Intermediate Level

    ### 4. Attention Mechanism
    - Implement multi-head attention
    - Visualize attention weights for different sentence
    - Compare attention patterns for different tasks

    ### 5. Information Theory
    - Calculate entropy for different probability distributions
    - Compare cross-entropy loss for good vs bad predictions
    - How does temperature affect entropy in language models?

    ## Advanced Level

    ### 6. Backpropagation Implementation
    - Build a simple neural network from scratch
    - Implement forward and backward passes
    - Train on a simple classification task

    ### 7. Embedding Analysis
    - Load pre-trained word embeddings
    - Perform embedding arithmetic (King - Man + Woman)
    - Find analogies in the embedding space

    ## Project Ideas

    ### Mathematical Visualization Tool
    - Create interactive plots for attention weights
    - Visualize gradient descent optimization
    - Show how different parameters affect learning

    ### Mini Language Model
    - Build a character-level language model
    - Implement attention from scratch
    - Train on simple text data

=== "📚 Further Reading"

    ## Essential Resources

    ### Books
    - **"Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville**
        - Comprehensive mathematical foundations
        - Chapters 2-4 cover linear algebra, probability, and numerical computation
    
    - **"Mathematics for Machine Learning" by Deisenroth, Faisal, and Ong**
        - Focused specifically on ML mathematics
        - Excellent for building intuition

    ### Online Courses
    - **3Blue1Brown's Linear Algebra Series**
        - Visual explanations of linear algebra concepts
        - Perfect for building geometric intuition
    
    - **Khan Academy Statistics and Probability**
        - Solid foundation in probability theory
        - Interactive exercises and examples

    ### Research Papers
    - **"Attention Is All You Need" (Vaswani et al., 2017)**
        - Original transformer paper
        - Mathematical details of attention mechanism
    
    - **"The Illustrated Transformer" by Jay Alammar**
        - Visual explanation of transformer mathematics
        - Great for understanding attention computations

    ## Advanced Topics

    ### Optimization Theory
    - **Adam optimizer mathematics**
        - Adaptive learning rates
        - Momentum and bias correction
    
    - **Learning rate scheduling**
        - Cosine annealing
        - Warm-up strategies

    ### Information Theory Deep Dive
    - **Mutual information in neural networks**
        - Information bottleneck theory
        - Representation learning theory
    
    - **Variational inference**
        - VAE mathematics
        - KL divergence applications

    ### Geometric Deep Learning
    - **Manifold learning**
        - High-dimensional data geometry
        - Dimensionality reduction techniques
    
    - **Graph neural networks**
        - Mathematics of graph convolutions
        - Message passing frameworks

    ## Practical Tools

    ### Libraries for Mathematical Exploration
    - **NumPy**: Core numerical computing
    - **SciPy**: Advanced mathematical functions
    - **Matplotlib/Seaborn**: Visualization
    - **SymPy**: Symbolic mathematics

    ### Interactive Learning
    - **Jupyter notebooks**: Experiment with concepts
    - **Wolfram Alpha**: Symbolic computation
    - **Desmos**: Function plotting and visualization

    ## Mathematical Intuition Building

    ### Visualization Resources
    - **Seeing Theory**: Interactive probability visualizations
    - **Immersive Math**: Interactive linear algebra
    - **Distill.pub**: Research with interactive explanations

    ### Practice Problems
    - **Problem-solving strategies for each concept**
    - **Worked examples with step-by-step solutions**
    - **Common pitfalls and how to avoid them**

## Key Takeaways

!!! success "Mathematical Intuition"
    - **Vectors** represent information (words, images, concepts)
    - **Matrices** transform information between representations
    - **Attention** weights determine what information to focus on
    - **Gradients** guide learning in the right direction
    - **Probability** quantifies uncertainty and prediction confidence

## Next Steps

Now that you understand the mathematical foundations:

1. **Practice**: Try the code examples and experiment with different values
2. **Visualize**: Use matplotlib to plot attention weights and embeddings
3. **Connect**: See how these concepts appear in transformer architectures
4. **Build**: Apply these principles in your own GenAI projects

**Ready to dive deeper?** Continue with [Neural Networks](neural-networks.md) to see how these mathematical concepts combine to create intelligent systems.

def kl_divergence(p, q):
    """Calculate KL divergence"""
    p = np.array(p)


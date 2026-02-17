## Training & Optimization

### Training

**Definition:** The process of adjusting a model's parameters by iteratively exposing it to data and minimizing a loss function.

**Explanation:** Training is how models learn. The model starts with random parameters, makes predictions, calculates how wrong they are (loss), and adjusts parameters to reduce that wrongness. This cycle repeats thousands of times until the model performs well.

**The Training Loop:**

1. **Forward Pass:** Feed data through model, get predictions
2. **Calculate Loss:** Measure how wrong predictions are
3. **Backward Pass:** Compute gradients (how to adjust parameters)
4. **Update Parameters:** Adjust weights and biases
5. **Repeat:** Do this for all training data, multiple times

**Example: Learning House Prices**

**Iteration 1:**

- Random weights: w₁=5, w₂=10, bias=0
- Prediction: $15,000 (way too low!)
- Actual: $300,000
- Loss: Very high
- **Update:** Increase weights

**Iteration 1000:**

- Learned weights: w₁=300, w₂=50000, bias=100000
- Prediction: $298,000 (close!)
- Actual: $300,000
- Loss: Low
- **Result:** Model has learned!

**Training vs. Inference:**

- **Training:** Learning phase, adjusts parameters, requires labels
- **Inference:** Prediction phase, parameters frozen, no labels needed

**Duration:** Training can take seconds (simple models, small data) to weeks (large neural networks, massive datasets).

---

### Epoch

**Definition:** One complete pass through the entire training dataset during model training.

**Explanation:** Models rarely learn from seeing data just once. An epoch means the model has seen every training example exactly once. Training typically involves many epochs—the model cycles through the data repeatedly, gradually improving its parameters each time.

**Example: Training with 1000 Samples**

**1 Epoch:** Model sees all 1000 samples once
**10 Epochs:** Model sees all 1000 samples ten times
**100 Epochs:** Model sees all 1000 samples one hundred times

**Why Multiple Epochs?**

**Epoch 1:** Model learns rough patterns, high loss
**Epoch 10:** Model refines understanding, medium loss
**Epoch 50:** Model achieves good performance, low loss
**Epoch 100:** Further improvement slows, risk of overfitting

**Typical Training:**

- Small datasets: 100-1000 epochs
- Large datasets: 10-50 epochs (one pass through millions of examples teaches a lot)
- Deep learning: Often 50-200 epochs

**Monitoring:**

Track loss after each epoch:

- **Loss decreasing steadily:** Good! Model is learning
- **Loss stopped improving:** May need to stop (early stopping)
- **Training loss low, validation loss high:** Overfitting!

**Not to Confuse With:**

- **Batch:** Subset of data processed at once
- **Iteration:** One parameter update (multiple per epoch)

---

### Batch

**Definition:** A subset of the training dataset processed together in a single forward and backward pass during training.

**Explanation:** Instead of updating parameters after every single example (slow) or after seeing all data (memory intensive), we process small groups called batches. After each batch, we update parameters. This balances computational efficiency with learning stability.

**Example: Training with 10,000 Samples**

**Batch Size = 32:**

- Process 32 samples together
- Calculate average loss for these 32
- Update parameters
- Repeat with next 32 samples
- **Total batches per epoch:** 10,000 / 32 ≈ 313 batches

**Types:**

**Batch Gradient Descent (Full Batch):**

- Batch size = all training data
- Update parameters once per epoch
- **Advantage:** Stable, accurate gradients
- **Disadvantage:** Memory intensive, slow for large datasets

**Stochastic Gradient Descent (SGD):**

- Batch size = 1 (one sample at a time)
- Update parameters after each sample
- **Advantage:** Fast updates, can escape local minima
- **Disadvantage:** Noisy, unstable gradients

**Mini-Batch Gradient Descent:**

- Batch size = 16, 32, 64, 128, 256 (typical)
- Best of both worlds!
- **Advantage:** Efficient, stable, fits in memory
- **Most common** approach in practice

**Choosing Batch Size:**

**Small batches (16-32):**

- More frequent updates
- More noise (can help escape bad local minima)
- Better generalization sometimes

**Large batches (128-512):**

- More stable gradients
- Faster training (parallelization on GPUs)
- May converge to sharper minima (worse generalization)

**Memory Constraint:** Larger batches need more GPU memory!

---

### Learning Rate

**Definition:** A hyperparameter that controls how much to adjust model parameters during each update, determining the step size in gradient descent.

**Explanation:** The learning rate is arguably the most important hyperparameter. Too high, and training diverges (overshoots optimal values). Too low, and training takes forever or gets stuck. It's the "speed" at which your model learns.

**Analogy:** Imagine walking down a hill blindfolded, trying to reach the bottom:

- **High learning rate:** Take giant steps → might jump over the valley, end up climbing back up
- **Low learning rate:** Take tiny steps → safe but slow, might take years to reach bottom
- **Good learning rate:** Reasonable steps → efficiently reach the bottom

**Example: Parameter Update**

**Current weight:** w = 10
**Gradient:** ∇w = -2 (loss decreases if we reduce w)
**Learning rate:** α

**Update rule:** w_new = w - α × ∇w

**With α = 0.01 (small):**

- w_new = 10 - 0.01 × (-2) = 10 + 0.02 = 10.02
- Tiny adjustment, very safe

**With α = 1.0 (large):**

- w_new = 10 - 1.0 × (-2) = 10 + 2 = 12
- Big adjustment, might overshoot

**With α = 0.1 (reasonable):**

- w_new = 10 - 0.1 × (-2) = 10 + 0.2 = 10.2
- Balanced adjustment

**Common Values:**

- **Neural networks:** 0.001 - 0.01 (often start with 0.001)
- **SGD:** 0.01 - 0.1
- **Adam optimizer:** 0.001 (default, often works well)

**Symptoms:**

**Learning rate too high:**

- Loss increases instead of decreases
- Loss oscillates wildly
- Training diverges (loss → infinity)
- Weights become NaN (Not a Number)

**Learning rate too low:**

- Loss decreases very slowly
- Training takes too long
- May get stuck in local minima
- Plateau in loss curve

**Learning Rate Scheduling:**

Often start high and decrease over time:

- **Step decay:** Reduce by factor every N epochs (e.g., halve every 30 epochs)
- **Exponential decay:** Gradually decrease continuously
- **Cosine annealing:** Decrease following cosine curve
- **ReduceLROnPlateau:** Decrease when loss stops improving

**Key Insight:** Finding the right learning rate is critical. Too high breaks training; too low wastes time. Modern optimizers (Adam) adapt it automatically.

---

### Gradient Descent

**Definition:** An iterative optimization algorithm that adjusts model parameters by moving in the direction of steepest decrease of the loss function.

**Explanation:** Gradient descent is the engine of machine learning. It's how models learn—by repeatedly asking "which direction should I adjust parameters to reduce error?" and taking steps in that direction. The "gradient" points uphill; we go downhill (opposite direction) to minimize loss.

**Intuition:**

Imagine you're on a foggy mountain and need to reach the valley (lowest point). You can't see far, but you can feel the ground slope. Strategy: Feel which direction is steepest downward, take a step that way, repeat until you reach bottom.

**Mathematics:**

**Goal:** Minimize loss function L(w)

**Current parameters:** w
**Gradient (slope):** ∇L(w) = derivative of loss with respect to w
**Learning rate:** α

**Update rule:**
w_new = w - α × ∇L(w)

**Subtract** gradient because gradient points uphill; we want to go downhill.

**Example: Linear Model**

**Model:** y = wx + b
**Loss:** L = (y_pred - y_actual)²
**Current:** w=2, actual_output=10, x=5, predicted=10 (2×5=10)

Gradient tells us: "increase w to decrease loss"
Update: w = 2 - 0.1 × gradient = 2.1 (better!)

**Variants:**

**Batch Gradient Descent:**

- Compute gradient using all training data
- One update per epoch
- Accurate but slow

**Stochastic Gradient Descent (SGD):**

- Compute gradient using one sample
- Many updates per epoch
- Fast but noisy

**Mini-Batch Gradient Descent:**

- Compute gradient using small batch (32-256 samples)
- Best balance
- **Most commonly used**

**Challenges:**

**Local Minima:** Might get stuck in suboptimal solution
**Saddle Points:** Flat regions where gradients vanish
**Slow Convergence:** Can take many iterations

**Solution:** Use advanced optimizers (Adam, RMSprop) that adapt learning rates and add momentum.

---

### Optimizer

**Definition:** An algorithm that implements the specific strategy for adjusting model parameters during training, building on top of gradient descent with various improvements.

**Explanation:** While gradient descent is the core idea, optimizers are sophisticated implementations that add bells and whistles—momentum, adaptive learning rates, acceleration—to train faster and more reliably.

**Common Optimizers:**

**SGD (Stochastic Gradient Descent):**

- Basic gradient descent, possibly with momentum
- Update: w = w - α × gradient
- **Use case:** Simple, interpretable, sometimes best for final fine-tuning

**Momentum:**

- Adds "velocity" to smooth out updates
- Builds up speed in consistent directions
- Like rolling a ball downhill—accumulates momentum
- **Better than plain SGD**

**Adam (Adaptive Moment Estimation):**

- Adapts learning rate for each parameter
- Combines momentum + RMSprop
- **Most popular optimizer** for deep learning
- **Default choice** for most problems
- Often works well without tuning

**RMSprop:**

- Adapts learning rate based on recent gradients
- Good for non-stationary problems
- Used in recurrent neural networks

**AdaGrad:**

- Adapts learning rate per parameter
- Good for sparse features
- Learning rate decays over time (can become too small)

**Example Comparison:**

**Problem:** Train neural network on image classification

**SGD:**

- Loss after 10 epochs: 0.5
- Time: 30 minutes
- Requires careful learning rate tuning

**Adam:**

- Loss after 10 epochs: 0.3
- Time: 30 minutes
- Works well with default learning rate 0.001

**Why Adam is Popular:**

- **Adaptive:** Automatically adjusts learning rates per parameter
- **Momentum:** Accelerates in consistent directions
- **Robust:** Works well across wide range of problems
- **Less tuning:** Often works with default settings

**Choosing an Optimizer:**

**Start with Adam** (learning rate = 0.001)
↓ If not working well
Try **SGD with momentum** (learning rate = 0.01-0.1)
↓ Still issues
Tune learning rate, try other optimizers

**Key Insight:** Adam is the sensible default. It combines the best ideas from multiple optimizers and works well out-of-the-box for most deep learning tasks.

---

### Backpropagation

**Definition:** The algorithm for computing gradients in neural networks by propagating errors backward through the network layers using the chain rule of calculus.

**Explanation:** Backpropagation is how neural networks learn. After making a prediction (forward pass), backpropagation calculates how much each weight contributed to the error, working backward from output to input. This tells us exactly how to adjust each parameter to reduce error.

**The Process:**

**1. Forward Pass:** Input → Layer 1 → Layer 2 → Output → Loss

**2. Backward Pass (Backpropagation):** Loss → ∂Loss/∂Layer2 → ∂Loss/∂Layer1 → Gradients for all weights

**Simple Example: 2-Layer Network**

**Network:** Input (x) → Hidden Layer (h) → Output (y)

**Forward:**
- h = ReLU(W₁ × x + b₁)
- y = W₂ × h + b₂
- Loss = (y - target)²

**Backward (using chain rule):**
- ∂Loss/∂W₂ = ∂Loss/∂y × ∂y/∂W₂ (gradient for output layer)
- ∂Loss/∂W₁ = ∂Loss/∂y × ∂y/∂h × ∂h/∂W₁ (gradient for hidden layer)

**Why "Backward"?** Start from loss (output), compute gradients layer by layer going backward to input.

**Computational Efficiency:** Without backpropagation, computing gradients for a network with 1 million parameters would require 1 million forward passes (perturb each parameter, measure loss change). Backpropagation computes ALL gradients in just ONE backward pass using the chain rule—orders of magnitude faster!

**The Chain Rule Connection:**

For function composition f(g(x)), the derivative is: df/dx = (df/dg) × (dg/dx)

Neural networks are deep compositions of functions (many layers), so we chain multiply derivatives backward through layers.

**Challenges:**

**Vanishing Gradients:** In deep networks, gradients can become extremely small as they propagate backward, especially with sigmoid/tanh activations. Early layers barely learn. **Solution:** Use ReLU activations, batch normalization, residual connections.

**Exploding Gradients:** Gradients can grow exponentially large. **Solution:** Gradient clipping, careful weight initialization.

**Key Insight:** Backpropagation made training deep neural networks feasible. It's the workhorse algorithm behind all modern deep learning.

---

### Convergence

**Definition:** The state where a model's training loss stops decreasing significantly, indicating the optimization algorithm has found a (local) minimum.

**Explanation:** Convergence means training has stabilized—the model has learned as much as it can with the current setup. The loss might still decrease slightly, but improvements become negligible. Knowing when to stop training (convergence) prevents wasting computation and can avoid overfitting.

**Signs of Convergence:**

**Loss Curve Flattening:**
- Epoch 1-10: Loss drops from 2.5 → 1.0 (rapid learning)
- Epoch 11-50: Loss drops from 1.0 → 0.5 (steady learning)
- Epoch 51-100: Loss drops from 0.5 → 0.48 (very slow, converged!)

**Validation Loss Plateau:**
- Training loss still decreasing slightly
- Validation loss flat or increasing
- **Sign:** Stop training (early stopping)

**Types of Convergence:**

**Global Minimum:** Best possible solution (lowest loss across entire space)
- Ideal but hard to guarantee in complex models
- Convex problems (linear regression) have single global minimum

**Local Minimum:** Low point in a region, but not the absolute lowest
- Neural networks have many local minima
- Often acceptable—many local minima perform similarly

**Saddle Point:** Gradient is zero but it's not a minimum (like a mountain pass)
- Can stall training
- Modern optimizers (Adam) + momentum help escape

**Factors Affecting Convergence:**

**Learning Rate:**
- Too high: May never converge, oscillates around minimum
- Too low: Slow convergence, may get stuck
- **Adaptive optimizers** (Adam) help

**Model Capacity:**
- Sufficient capacity needed to reach low loss
- Too much capacity can overfit

**Data Quality:**
- Noisy data → higher minimum achievable loss
- More data → better convergence

**Initialization:**
- Poor initialization → stuck in bad local minima
- Good initialization (Xavier, He) → faster convergence

**Practical Convergence Criteria:**

Stop training when:
1. Validation loss hasn't improved for N epochs (early stopping)
2. Loss change < threshold (e.g., 0.001) for several epochs
3. Reached maximum epochs budget
4. Gradient norm becomes very small

**Key Insight:** Perfect convergence (zero loss) is often impossible and undesirable (overfitting). The goal is to converge to a solution that generalizes well, not memorize training data.

---


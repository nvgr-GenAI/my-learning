# Expectation and Variance

Expected values and variance are foundational concepts in probability theory and statistics that play crucial roles in algorithm design and analysis, especially for randomized algorithms and probabilistic data structures.

## Expected Value (Mean)

The expected value (or mathematical expectation) of a random variable represents the long-term average value of repetitions of the same experiment.

### Definition of Expected Value

For a discrete random variable $X$ with possible values $x_1, x_2, \ldots, x_n$ and corresponding probabilities $p_1, p_2, \ldots, p_n$, the expected value is defined as:

$$E[X] = \sum_{i=1}^{n} x_i \cdot p_i$$

For a continuous random variable $X$ with probability density function $f(x)$, the expected value is:

$$E[X] = \int_{-\infty}^{\infty} x \cdot f(x) \, dx$$

### Properties of Expected Value

1. **Linearity**: For any random variables $X$ and $Y$, and constants $a$ and $b$:
   $$E[aX + bY] = aE[X] + bE[Y]$$

2. **Independence**: If $X$ and $Y$ are independent:
   $$E[XY] = E[X] \cdot E[Y]$$

3. **Law of Large Numbers**: As the sample size increases, the average of the samples approaches the expected value.

## Variance and Standard Deviation

Variance measures how far a set of numbers are spread out from their mean value.

### Definition of Variance

The variance of a random variable $X$ is the expected value of the squared deviation from the mean:

$$\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$

The standard deviation is the square root of the variance:

$$\sigma_X = \sqrt{\text{Var}(X)}$$

### Properties of Variance

1. **Non-negativity**: $\text{Var}(X) \geq 0$

2. **Scaling**: For a constant $a$:
   $$\text{Var}(aX) = a^2 \text{Var}(X)$$

3. **For independent variables**: If $X$ and $Y$ are independent:
   $$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$$

4. **General case**: For any random variables $X$ and $Y$:
   $$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$$

   where $\text{Cov}(X,Y)$ is the covariance between $X$ and $Y$.

## Applications in Algorithmic Analysis

### Probabilistic Algorithms

Expected value analysis helps in understanding the average-case behavior of randomized algorithms:

1. **QuickSort**: Expected runtime of $O(n \log n)$ when pivot selection is random.
2. **Skip Lists**: Expected search time of $O(\log n)$.
3. **Randomized Selection**: Expected $O(n)$ time complexity.

### Example: QuickSort Analysis

```python
def quicksort(arr, low, high):
    if low < high:
        # Partition the array and get the pivot position
        pivot_index = partition(arr, low, high)
        
        # Recursively sort the subarrays
        quicksort(arr, low, pivot_index - 1)
        quicksort(arr, pivot_index + 1, high)

def partition(arr, low, high):
    # Choose a random pivot
    import random
    pivot_index = random.randint(low, high)
    arr[pivot_index], arr[high] = arr[high], arr[pivot_index]
    
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
```

With random pivot selection, the expected number of comparisons in QuickSort is $O(n \log n)$, despite its worst-case behavior of $O(n^2)$.

### Variance in Algorithm Performance

Understanding variance helps us assess the stability and reliability of algorithms:

- **Low variance** indicates consistent performance across different inputs.
- **High variance** suggests unpredictable performance, which might be problematic in time-sensitive applications.

### Concentration Bounds

These inequalities help us understand how a random variable deviates from its expected value:

1. **Markov's Inequality**: For a non-negative random variable $X$ and $a > 0$:
   $$P(X \geq a) \leq \frac{E[X]}{a}$$

2. **Chebyshev's Inequality**: For any random variable $X$ with finite expected value $\mu$ and finite non-zero variance $\sigma^2$:
   $$P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}$$

3. **Chernoff Bound**: Provides tighter bounds for sums of independent random variables.

## Probabilistic Data Structures

Many modern data structures use probability and expected value analysis:

1. **Bloom Filters**: A space-efficient probabilistic data structure for set membership testing.
2. **Count-Min Sketch**: Estimates the frequency of events in a data stream.
3. **HyperLogLog**: Approximates the number of distinct elements in a multiset.

### Example: Bloom Filter Implementation

```python
import math
import hashlib

class BloomFilter:
    def __init__(self, items_count, false_positive_probability):
        self.size = self.calculate_size(items_count, false_positive_probability)
        self.hash_count = self.calculate_hash_count(self.size, items_count)
        self.bit_array = [0] * self.size
        
    def calculate_size(self, n, p):
        '''Calculate the optimal size of bit array'''
        m = -(n * math.log(p)) / (math.log(2)**2)
        return int(m)
    
    def calculate_hash_count(self, m, n):
        '''Calculate the optimal number of hash functions'''
        k = (m / n) * math.log(2)
        return int(k)
    
    def _get_hash_values(self, item):
        hash_values = []
        for i in range(self.hash_count):
            # Create different hash functions by appending i
            hash_input = str(item) + str(i)
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16) % self.size
            hash_values.append(hash_value)
        return hash_values
    
    def add(self, item):
        for hash_value in self._get_hash_values(item):
            self.bit_array[hash_value] = 1
    
    def check(self, item):
        for hash_value in self._get_hash_values(item):
            if self.bit_array[hash_value] == 0:
                return False  # Definitely not in the set
        return True  # Probably in the set
```

The false positive probability in a Bloom filter is:

$$p \approx \left(1 - e^{-kn/m}\right)^k$$

where:

- $k$ is the number of hash functions
- $n$ is the number of elements in the set
- $m$ is the size of the bit array

## Common Probability Distributions

Understanding common distributions helps in algorithm analysis:

1. **Bernoulli Distribution**: Models a single trial with success probability $p$
   - $E[X] = p$
   - $\text{Var}(X) = p(1-p)$

2. **Binomial Distribution**: Sum of $n$ independent Bernoulli trials
   - $E[X] = np$
   - $\text{Var}(X) = np(1-p)$

3. **Geometric Distribution**: Number of trials until first success
   - $E[X] = \frac{1}{p}$
   - $\text{Var}(X) = \frac{1-p}{p^2}$

4. **Poisson Distribution**: Models rare events
   - $E[X] = \lambda$
   - $\text{Var}(X) = \lambda$

5. **Uniform Distribution**: Equal probability across a range
   - $E[X] = \frac{a+b}{2}$ (for range $[a,b]$)
   - $\text{Var}(X) = \frac{(b-a)^2}{12}$

6. **Normal Distribution**: Gaussian distribution with parameters $\mu$ and $\sigma^2$
   - $E[X] = \mu$
   - $\text{Var}(X) = \sigma^2$

## Monte Carlo Methods

Monte Carlo methods use random sampling to obtain numerical results. In algorithm design, they're used when deterministic algorithms are too complex or slow.

### Example: Estimating Pi using Monte Carlo

```python
import random
import math

def estimate_pi(num_samples):
    points_inside_circle = 0
    
    for _ in range(num_samples):
        # Generate random points in the unit square
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        
        # Check if the point lies within the unit circle
        if math.sqrt(x**2 + y**2) <= 1:
            points_inside_circle += 1
    
    # Ratio of points in circle to total points is approximately π/4
    return 4 * points_inside_circle / num_samples

# Increase the sample size for better approximation
pi_estimate = estimate_pi(1000000)
print(f"Pi estimate: {pi_estimate}")
print(f"Actual Pi: {math.pi}")
```

## Pro Tips

1. **Know when to use expected value analysis**: Not all algorithms need probabilistic analysis, but it's essential for randomized algorithms.

2. **Consider both mean and variance**: An algorithm with a good average-case behavior but high variance might still be problematic in practice.

3. **Use tail bounds**: Markov's inequality, Chebyshev's inequality, and Chernoff bounds help understand how likely an algorithm is to deviate significantly from its expected performance.

4. **Be careful with dependencies**: The independence assumption is crucial in many probability calculations. Be aware when variables are not independent.

5. **Empirical verification**: Always validate theoretical expectations with experimental measurements, especially for complex algorithms.

6. **Amortized vs. expected analysis**: Don't confuse amortized analysis (worst-case behavior averaged over a sequence of operations) with expected-case analysis (average over random choices).

## Common Interview Problems

1. **Randomized QuickSort Analysis**: Explain why the expected runtime is O(n log n).

2. **Reservoir Sampling**: Explain how to randomly select k items from a stream of unknown size with uniform probability.

3. **Random Permutations**: Generate a random permutation with equal probability for all permutations.

4. **Birthday Paradox**: Analyze the probability of collision in hash functions.

5. **Las Vegas vs. Monte Carlo Algorithms**: Explain the difference and provide examples.

## Further Resources

1. **Books**:
   - "Introduction to Algorithms" by Cormen, Leiserson, Rivest, and Stein (Chapter on Probabilistic Analysis)
   - "Randomized Algorithms" by Motwani and Raghavan
   - "Probability and Computing" by Mitzenmacher and Upfal

2. **Online Courses**:
   - Stanford's "Probabilistic Graphical Models"
   - MIT's "Mathematics for Computer Science"

3. **Papers**:
   - "Analysis of Randomized Algorithms" by Karp
   - "The Power of Two Choices in Randomized Load Balancing" by Mitzenmacher

## Conclusion

Expected value and variance are powerful tools for analyzing algorithmic performance. They help us understand the average behavior and the stability of randomized algorithms, enabling the design of efficient probabilistic data structures and algorithms.

Understanding these concepts thoroughly will give you a significant advantage in algorithm design, especially when dealing with large-scale or unpredictable data sets where deterministic approaches might be inefficient.

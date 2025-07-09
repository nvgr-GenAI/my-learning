# Random Sampling

Random sampling is a fundamental technique in computational algorithms, statistical analysis, and machine learning. It involves selecting a subset of data points from a larger population, with various methods determining how this selection occurs.

## Introduction to Random Sampling

Random sampling serves several key purposes in algorithms and data analysis:

- **Representative Subsets**: Obtain smaller, manageable datasets that still preserve the characteristics of the original population.
- **Statistical Inference**: Make conclusions about a population based on a sample.
- **Algorithmic Efficiency**: Reduce computational complexity while maintaining reasonable accuracy.
- **Cross-Validation**: Create training and testing sets for machine learning models.
- **Randomized Algorithms**: Power algorithms that use randomness to achieve efficiency or simplicity.

## Types of Random Sampling

### Simple Random Sampling

Each item in the population has an equal probability of being selected.

#### Implementation (Python)

```python
import random

def simple_random_sample(population, sample_size):
    """
    Select a simple random sample without replacement.
    
    Args:
        population: List of items to sample from
        sample_size: Number of items to select
        
    Returns:
        List containing the random sample
    """
    if sample_size > len(population):
        raise ValueError("Sample size cannot exceed population size")
    
    return random.sample(population, sample_size)

# Example usage
data = list(range(1, 101))  # Population of numbers 1-100
sample = simple_random_sample(data, 10)
print(f"Random sample: {sample}")
```

### Reservoir Sampling

A family of algorithms for randomly selecting k samples from a list of n items, where n is either very large or unknown.

#### Algorithm L (Standard Reservoir Sampling)

```python
import random

def reservoir_sample(stream, k):
    """
    Perform reservoir sampling on a stream of unknown size.
    
    Args:
        stream: Iterator over the population
        k: Sample size to select
        
    Returns:
        List containing the random sample
    """
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            # Fill the reservoir initially
            reservoir.append(item)
        else:
            # Replace elements with decreasing probability
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    
    return reservoir

# Example usage with a file stream (line by line)
def sample_from_file(filename, sample_size):
    with open(filename, 'r') as file:
        return reservoir_sample(file, sample_size)
```

### Weighted Random Sampling

Items are selected with probability proportional to their weights.

#### Implementation of Weighted Sampling

```python
import random
import bisect

def weighted_random_sample(population, weights, sample_size):
    """
    Select a weighted random sample without replacement.
    
    Args:
        population: List of items to sample from
        weights: List of weights corresponding to each item
        sample_size: Number of items to select
        
    Returns:
        List containing the weighted random sample
    """
    if len(population) != len(weights):
        raise ValueError("Population and weights must have the same length")
    
    if sample_size > len(population):
        raise ValueError("Sample size cannot exceed population size")
    
    # Create a copy of the population and weights
    items = list(zip(population, weights))
    
    selected = []
    total_weight = sum(weights)
    
    for _ in range(sample_size):
        if not items:
            break
            
        # Select an item based on weight
        r = random.uniform(0, total_weight)
        cumulative_weight = 0
        
        for i, (item, weight) in enumerate(items):
            cumulative_weight += weight
            if r <= cumulative_weight:
                selected.append(item)
                # Remove the selected item and update total weight
                total_weight -= weight
                items.pop(i)
                break
    
    return selected

# Example usage
items = ["A", "B", "C", "D", "E"]
weights = [10, 1, 3, 5, 2]
sample = weighted_random_sample(items, weights, 3)
print(f"Weighted random sample: {sample}")
```

### Stratified Sampling

The population is divided into subgroups (strata), and samples are taken from each stratum.

```python
def stratified_sample(population, strata_key, sample_size):
    """
    Perform stratified sampling.
    
    Args:
        population: List of items to sample from
        strata_key: Function that returns the stratum for each item
        sample_size: Dictionary mapping strata to sample sizes
        
    Returns:
        List containing the stratified sample
    """
    # Group items by strata
    strata = {}
    for item in population:
        key = strata_key(item)
        if key not in strata:
            strata[key] = []
        strata[key].append(item)
    
    # Sample from each stratum
    samples = []
    for key, size in sample_size.items():
        if key in strata:
            stratum_sample = simple_random_sample(strata[key], min(size, len(strata[key])))
            samples.extend(stratum_sample)
    
    return samples

# Example usage
people = [
    {"name": "Alice", "age": 25, "gender": "F"},
    {"name": "Bob", "age": 32, "gender": "M"},
    {"name": "Charlie", "age": 41, "gender": "M"},
    {"name": "Diana", "age": 29, "gender": "F"},
    # ... more people
]

# Sample 1 male and 1 female
sample = stratified_sample(
    people,
    lambda x: x["gender"],
    {"M": 1, "F": 1}
)
```

## Algorithms Utilizing Random Sampling

### Monte Carlo Methods

```python
import random
import math

def estimate_pi(num_samples):
    """
    Estimate π using Monte Carlo sampling.
    
    Points are randomly placed in a 1×1 square, and we count how many
    fall within a quarter circle of radius 1.
    """
    points_inside_circle = 0
    
    for _ in range(num_samples):
        # Generate random point
        x = random.random()
        y = random.random()
        
        # Check if point is inside quarter circle
        if math.sqrt(x**2 + y**2) <= 1:
            points_inside_circle += 1
    
    # Ratio of areas is π/4
    return 4 * points_inside_circle / num_samples

# With more samples, the estimate gets more accurate
pi_estimate = estimate_pi(1000000)
print(f"π estimate: {pi_estimate}")
print(f"Error: {abs(pi_estimate - math.pi)}")
```

### Randomized QuickSort

```python
import random

def randomized_quicksort(arr):
    """
    Sort an array using randomized QuickSort.
    """
    if len(arr) <= 1:
        return arr
        
    # Select random pivot
    pivot_idx = random.randint(0, len(arr) - 1)
    pivot = arr[pivot_idx]
    
    # Partition array
    less = [x for i, x in enumerate(arr) if x < pivot]
    equal = [x for x in arr if x == pivot]
    greater = [x for i, x in enumerate(arr) if x > pivot]
    
    # Recursive sort and combine
    return randomized_quicksort(less) + equal + randomized_quicksort(greater)
```

### Random Walk

```python
import random
import matplotlib.pyplot as plt

def random_walk_1d(steps, p=0.5):
    """
    Simulate a 1D random walk.
    
    Args:
        steps: Number of steps in the walk
        p: Probability of moving right (1-p is probability of moving left)
        
    Returns:
        List of positions at each step
    """
    position = 0
    positions = [position]
    
    for _ in range(steps):
        # Move right with probability p, left with probability 1-p
        step = 1 if random.random() < p else -1
        position += step
        positions.append(position)
    
    return positions

# Plot a random walk
walk = random_walk_1d(1000)
plt.plot(walk)
plt.title("1D Random Walk")
plt.xlabel("Step")
plt.ylabel("Position")
```

## Random Sampling in Machine Learning

### Random Sampling for Cross-Validation

```python
import numpy as np
from sklearn.model_selection import train_test_split

def k_fold_cross_validation(X, y, k=5, random_state=None):
    """
    Implement k-fold cross-validation by random sampling.
    
    Args:
        X: Feature matrix
        y: Target vector
        k: Number of folds
        random_state: Random seed for reproducibility
        
    Returns:
        List of (train_indices, test_indices) for each fold
    """
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if random_state is not None:
        np.random.seed(random_state)
    
    np.random.shuffle(indices)
    
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1
    
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_indices, test_indices))
        current = stop
        
    return folds

# Example using sklearn's built-in function
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array([0, 1, 0, 1, 0])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Random Subsampling for Imbalanced Data

```python
import numpy as np

def random_undersampling(X, y, majority_class=0):
    """
    Balance a binary classification dataset by randomly undersampling the majority class.
    
    Args:
        X: Feature matrix
        y: Target vector (binary)
        majority_class: Label of the majority class
        
    Returns:
        X_balanced, y_balanced: Balanced dataset
    """
    minority_class = 1 if majority_class == 0 else 0
    
    # Find indices of each class
    majority_indices = np.where(y == majority_class)[0]
    minority_indices = np.where(y == minority_class)[0]
    
    # Random sample from majority class, same size as minority class
    sampled_majority_indices = np.random.choice(
        majority_indices,
        size=len(minority_indices),
        replace=False
    )
    
    # Combine indices
    balanced_indices = np.concatenate([sampled_majority_indices, minority_indices])
    
    # Shuffle
    np.random.shuffle(balanced_indices)
    
    # Create balanced dataset
    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]
    
    return X_balanced, y_balanced
```

## Bootstrap Sampling

Bootstrap sampling involves sampling with replacement and is commonly used to estimate sampling distributions.

```python
import numpy as np

def bootstrap_sample(data, n_samples=None):
    """
    Create a bootstrap sample (sampling with replacement).
    
    Args:
        data: Original dataset
        n_samples: Number of samples to draw (defaults to len(data))
        
    Returns:
        Bootstrap sample of the data
    """
    if n_samples is None:
        n_samples = len(data)
        
    # Sample indices with replacement
    indices = np.random.choice(len(data), size=n_samples, replace=True)
    
    # Return the sampled data
    return [data[i] for i in indices]

def bootstrap_statistic(data, statistic_fn, n_bootstrap=1000):
    """
    Estimate the distribution of a statistic using bootstrap.
    
    Args:
        data: Original dataset
        statistic_fn: Function to compute the statistic
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        List of bootstrap statistics
    """
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        # Generate bootstrap sample
        sample = bootstrap_sample(data)
        
        # Compute statistic on the sample
        stat = statistic_fn(sample)
        bootstrap_stats.append(stat)
        
    return bootstrap_stats

# Example: Bootstrap estimate of mean and confidence interval
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
bootstrap_means = bootstrap_statistic(data, np.mean, n_bootstrap=10000)

# 95% confidence interval
confidence_interval = (
    np.percentile(bootstrap_means, 2.5),
    np.percentile(bootstrap_means, 97.5)
)

print(f"Bootstrap mean: {np.mean(bootstrap_means)}")
print(f"95% CI: {confidence_interval}")
```

## Random Sampling in Big Data

### Sampling for MapReduce

```python
def map_reduce_sampling(data, sample_rate=0.01):
    """
    Perform uniform sampling in a MapReduce context.
    
    Args:
        data: Iterator over data items
        sample_rate: Probability of including each item
        
    Returns:
        List of sampled items
    """
    sampled_data = []
    
    for item in data:
        if random.random() < sample_rate:
            sampled_data.append(item)
            
    return sampled_data
```

### Bernoulli Sampling

```python
def bernoulli_sampling(data, p=0.1):
    """
    Perform Bernoulli sampling - each item has probability p of being included.
    
    Args:
        data: Input data
        p: Probability of including each item
        
    Returns:
        List of sampled items
    """
    return [item for item in data if random.random() < p]
```

## Advanced Sampling Techniques

### Gibbs Sampling

A Markov Chain Monte Carlo (MCMC) algorithm for obtaining a sequence of observations approximated from a multivariate probability distribution.

```python
import numpy as np

def gibbs_sampler_bivariate_normal(n_samples, mu, sigma, rho, initial=None):
    """
    Gibbs sampler for a bivariate normal distribution.
    
    Args:
        n_samples: Number of samples to generate
        mu: Mean vector [mu_x, mu_y]
        sigma: Standard deviation vector [sigma_x, sigma_y]
        rho: Correlation coefficient
        initial: Initial state [x0, y0]
        
    Returns:
        Array of samples
    """
    samples = np.zeros((n_samples, 2))
    
    # Set initial state
    if initial is None:
        current = [0, 0]
    else:
        current = initial.copy()
    
    for i in range(n_samples):
        # Sample x conditional on y
        mu_x_given_y = mu[0] + rho * sigma[0] / sigma[1] * (current[1] - mu[1])
        sigma_x_given_y = sigma[0] * np.sqrt(1 - rho**2)
        current[0] = np.random.normal(mu_x_given_y, sigma_x_given_y)
        
        # Sample y conditional on x
        mu_y_given_x = mu[1] + rho * sigma[1] / sigma[0] * (current[0] - mu[0])
        sigma_y_given_x = sigma[1] * np.sqrt(1 - rho**2)
        current[1] = np.random.normal(mu_y_given_x, sigma_y_given_x)
        
        samples[i] = current
    
    return samples
```

### Metropolis-Hastings Algorithm

Another MCMC method for obtaining a sequence of random samples from a probability distribution.

```python
import numpy as np

def metropolis_hastings(target_pdf, proposal_sampler, proposal_pdf, n_samples, initial):
    """
    Metropolis-Hastings algorithm for sampling from a target distribution.
    
    Args:
        target_pdf: Target probability density function
        proposal_sampler: Function that samples from proposal distribution given current state
        proposal_pdf: Probability density function of the proposal distribution
        n_samples: Number of samples to generate
        initial: Initial state
        
    Returns:
        Array of samples
    """
    samples = [initial]
    current = initial
    
    for _ in range(n_samples - 1):
        # Propose a new sample
        proposed = proposal_sampler(current)
        
        # Calculate acceptance probability
        target_ratio = target_pdf(proposed) / target_pdf(current)
        proposal_ratio = proposal_pdf(current, proposed) / proposal_pdf(proposed, current)
        acceptance_prob = min(1, target_ratio * proposal_ratio)
        
        # Accept or reject the proposal
        if np.random.random() < acceptance_prob:
            current = proposed
            
        samples.append(current)
        
    return samples
```

## Performance Considerations

### Time Complexity

- **Simple Random Sampling**: O(n) time to scan through n elements.
- **Reservoir Sampling**: O(n) time, but constant space regardless of stream size.
- **Weighted Random Sampling**: O(n log n) for preprocessing, O(k log n) for selecting k elements.
- **Stratified Sampling**: O(n) time to categorize elements into strata.

### Space Complexity

- **Simple Random Sampling**: O(min(n, k)) space to store the sample.
- **Reservoir Sampling**: O(k) space for the reservoir.
- **Weighted Random Sampling**: O(n) space for tracking weights.
- **Stratified Sampling**: O(n) space in worst case to track strata.

## Pro Tips

1. **Choose the right sampling method**: Different problems require different sampling techniques. Simple random sampling might be sufficient for homogeneous data, but stratified sampling is better for maintaining representation across subgroups.

2. **Sample size determination**: Larger samples generally provide more accurate estimates but require more computation. Use statistical power calculations to determine the minimum sample size needed for a given level of confidence.

3. **Beware of sampling bias**: Ensure your sampling method doesn't systematically exclude certain elements of the population.

4. **Use appropriate seeds**: For reproducible results, always set random seeds when implementing random sampling algorithms.

5. **Handle edge cases**: Ensure your sampling algorithms work correctly for empty populations, very small populations, or when the requested sample size equals the population size.

6. **Parallel implementations**: For large datasets, consider parallel implementations of sampling algorithms that can distribute the work across multiple processors.

7. **Sampling without replacement vs. with replacement**: Be clear about whether sampling is done with or without replacement, as this significantly affects the probability distributions.

## Conclusion

Random sampling is a powerful technique in algorithm design, statistical analysis, and machine learning. By understanding the various sampling methods and their appropriate applications, you can efficiently process large datasets, build robust statistical models, and develop scalable algorithms.

Whether you're implementing a randomized algorithm, conducting statistical inference, or preparing data for machine learning, mastery of random sampling techniques is an essential skill in the algorithmic toolkit.

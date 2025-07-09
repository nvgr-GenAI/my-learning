# Basic Probability

Probability theory is the mathematical framework for analyzing random phenomena and forms the foundation of many algorithms, especially in machine learning, data science, and randomized algorithms.

## Fundamental Concepts

### Sample Space

The set of all possible outcomes of an experiment is called the sample space, usually denoted by Ω (omega).

**Example**: When rolling a standard six-sided die, Ω = {1, 2, 3, 4, 5, 6}.

### Events

An event is a subset of the sample space.

**Example**: When rolling a die, the event "getting an even number" = {2, 4, 6}.

### Probability Measure

For each event A, we assign a probability P(A) such that:

1. 0 ≤ P(A) ≤ 1
2. P(Ω) = 1
3. For disjoint events A and B, P(A ∪ B) = P(A) + P(B)

## Basic Probability Rules

### Complement Rule

The probability of an event not occurring is:

P(A') = 1 - P(A)

```python
def complement_probability(p_event):
    return 1 - p_event
```

### Addition Rule

For any two events A and B:

P(A ∪ B) = P(A) + P(B) - P(A ∩ B)

```python
def union_probability(p_a, p_b, p_intersection):
    return p_a + p_b - p_intersection
```

### Conditional Probability

The probability of event A given that event B has occurred:

P(A|B) = P(A ∩ B) / P(B), where P(B) > 0

```python
def conditional_probability(p_intersection, p_given):
    if p_given == 0:
        raise ValueError("P(B) must be greater than 0")
    return p_intersection / p_given
```

### Multiplication Rule

P(A ∩ B) = P(A) × P(B|A) = P(B) × P(A|B)

```python
def intersection_probability(p_a, p_b_given_a):
    return p_a * p_b_given_a
```

### Independence

Events A and B are independent if:

P(A ∩ B) = P(A) × P(B)

or equivalently:

P(A|B) = P(A) and P(B|A) = P(B)

```python
def are_events_independent(p_a, p_b, p_intersection):
    return abs(p_intersection - (p_a * p_b)) < 1e-10  # Allow for floating-point precision
```

## Bayes' Theorem

Bayes' theorem relates the conditional probabilities of events:

P(A|B) = [P(B|A) × P(A)] / P(B)

This can be expanded using the law of total probability:

P(A|B) = [P(B|A) × P(A)] / [P(B|A) × P(A) + P(B|A') × P(A')]

```python
def bayes_theorem(p_a, p_b_given_a, p_b_given_not_a=None):
    # If P(B|A') is provided, use the expanded form
    if p_b_given_not_a is not None:
        p_not_a = 1 - p_a
        p_b = p_b_given_a * p_a + p_b_given_not_a * p_not_a
    else:
        # Assume P(B) is provided directly
        p_b = p_b_given_a  # In this case, the second parameter is P(B)
    
    return (p_b_given_a * p_a) / p_b
```

## Random Variables

A random variable is a function that assigns a real number to each outcome in the sample space.

### Discrete Random Variables

A random variable with countably many possible values.

#### Probability Mass Function (PMF)

P(X = x) gives the probability that X takes the value x.

```python
def expected_value_discrete(values, probabilities):
    """
    Calculate expected value of a discrete random variable
    
    Args:
        values: List of possible values
        probabilities: Corresponding probabilities
    
    Returns:
        Expected value (mean)
    """
    return sum(x * p for x, p in zip(values, probabilities))

def variance_discrete(values, probabilities):
    """
    Calculate variance of a discrete random variable
    
    Args:
        values: List of possible values
        probabilities: Corresponding probabilities
    
    Returns:
        Variance
    """
    mean = expected_value_discrete(values, probabilities)
    return sum((x - mean)**2 * p for x, p in zip(values, probabilities))
```

### Continuous Random Variables

A random variable with uncountably many possible values.

#### Probability Density Function (PDF)

The PDF f(x) is such that the probability of X falling in an interval [a, b] is:

P(a ≤ X ≤ b) = ∫ₐᵇ f(x) dx

## Common Probability Distributions

### Discrete Distributions

#### Uniform Distribution

Each outcome has equal probability.

```python
def uniform_pmf(n):
    """PMF for discrete uniform distribution with n outcomes"""
    return [1/n] * n
```

#### Bernoulli Distribution

Models a single trial with success probability p.

```python
def bernoulli_pmf(p):
    """PMF for Bernoulli distribution"""
    return [1-p, p]  # P(X=0), P(X=1)
```

#### Binomial Distribution

Models the number of successes in n independent Bernoulli trials.

```python
from math import comb

def binomial_pmf(n, p):
    """PMF for binomial distribution"""
    return [comb(n, k) * (p**k) * ((1-p)**(n-k)) for k in range(n+1)]
```

### Continuous Distributions

#### Uniform Distribution

Equal probability density over an interval [a, b].

#### Normal (Gaussian) Distribution

The famous bell curve with parameters μ (mean) and σ² (variance).

## Applications in Computer Science

### Randomized Algorithms

Many algorithms use randomness to achieve better average-case performance.

#### Example: Monte Carlo Pi Estimation

```python
import random

def estimate_pi(num_points=1000000):
    """Estimate π using Monte Carlo method"""
    points_inside_circle = 0
    
    for _ in range(num_points):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        
        # Check if point is inside unit circle
        if x**2 + y**2 <= 1:
            points_inside_circle += 1
    
    # Area of circle / Area of square = π/4
    return (points_inside_circle / num_points) * 4
```

### Probabilistic Data Structures

#### Bloom Filter

A space-efficient probabilistic data structure for membership testing.

```python
import mmh3  # MurmurHash3 implementation

class BloomFilter:
    def __init__(self, size, num_hashes):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = [0] * size
    
    def add(self, item):
        for i in range(self.num_hashes):
            index = mmh3.hash(str(item), i) % self.size
            self.bit_array[index] = 1
    
    def contains(self, item):
        for i in range(self.num_hashes):
            index = mmh3.hash(str(item), i) % self.size
            if self.bit_array[index] == 0:
                return False  # Definitely not in set
        return True  # Might be in set
```

### Random Sampling

#### Reservoir Sampling

Algorithm for randomly selecting k items from a stream of items of unknown length.

```python
import random

def reservoir_sample(stream, k):
    """Select k random elements from a stream"""
    reservoir = []
    
    for i, item in enumerate(stream):
        if i < k:
            # Fill the reservoir until we have k items
            reservoir.append(item)
        else:
            # Replace elements with decreasing probability
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    
    return reservoir
```

## Practice Problems

1. **Birthday Paradox**: What's the probability that in a group of n people, at least two share a birthday?

2. **Coupon Collector**: How many random coupons do you need to collect on average to get all n different types?

3. **Monte Carlo Integration**: Implement a function to estimate the integral of a function using random sampling.

4. **Random Walk**: Analyze the expected distance from the origin after n steps of a random walk.

5. **Probabilistic Algorithms**: Implement a randomized quicksort and compare its performance to deterministic quicksort.

## Pro Tips

1. **Law of Large Numbers**: With increasing sample size, the sample mean converges to the expected value.

2. **Central Limit Theorem**: The sum of many independent random variables tends toward a normal distribution.

3. **Markov's Inequality**: For a non-negative random variable X and a > 0, P(X ≥ a) ≤ E[X]/a.

4. **Chebyshev's Inequality**: For a random variable X with mean μ and variance σ², P(|X - μ| ≥ kσ) ≤ 1/k².

5. **Simulation**: When analytical solutions are hard, Monte Carlo simulation can provide good approximations.

## Related Topics

- [Expectation and Variance](expectation-variance.md)
- [Random Sampling](random-sampling.md)
- [Randomized Algorithms](../randomized-algorithms.md)
- [Monte Carlo Methods](monte-carlo-methods.md)

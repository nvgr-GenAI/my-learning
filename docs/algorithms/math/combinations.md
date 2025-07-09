# Combinations

## 🎯 Overview

Combinations represent the selection of objects from a set without regard to order. Unlike permutations, where arrangement matters, combinations only consider which objects are selected, not how they are arranged. Combinations are foundational in probability, statistics, and combinatorial algorithms.

## 📋 Core Concepts

### Definition

A combination is a selection of r objects from a set of n distinct objects, where the order of selection does not matter.

### Combinations Formula

The number of ways to select r objects from n distinct objects is denoted as C(n,r), nCr, or the binomial coefficient (n choose r):

$$C(n,r) = \binom{n}{r} = \frac{n!}{r!(n-r)!}$$

### Properties of Combinations

1. C(n,r) = C(n,n-r)
2. C(n,0) = C(n,n) = 1
3. C(n,1) = n
4. C(n,r) = C(n-1,r-1) + C(n-1,r) (Pascal's identity)

## ⚙️ Algorithm Implementations

### Computing Combinations (nCr)

```python
def binomial_coefficient(n, k):
    """
    Calculate the binomial coefficient C(n,k) = n! / (k! * (n-k)!)
    Using an efficient approach to avoid overflow
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    
    # Use symmetry property
    k = min(k, n - k)
    
    result = 1
    for i in range(k):
        result *= (n - i)
        result //= (i + 1)
    
    return result
```

### Generating All Combinations

```python
def generate_combinations(elements, r):
    """
    Generate all combinations of r elements from the given list.
    
    Args:
        elements: List of elements
        r: Number of elements to select
        
    Returns:
        List of all possible combinations
    """
    n = len(elements)
    if r > n:
        return []
    
    # Recursive implementation
    if r == 0:
        return [[]]
    
    if r == n:
        return [elements.copy()]
    
    # Take first element and generate combinations with and without it
    first = elements[0]
    rest = elements[1:]
    
    # Combinations that include the first element
    with_first = generate_combinations(rest, r - 1)
    for combo in with_first:
        combo.insert(0, first)
    
    # Combinations that exclude the first element
    without_first = generate_combinations(rest, r)
    
    return with_first + without_first
```

### Iterative Combination Generation

```python
def iterative_combinations(elements, r):
    """
    Generate all combinations of r elements from the given list iteratively.
    More efficient for large inputs.
    
    Args:
        elements: List of elements
        r: Number of elements to select
        
    Returns:
        List of all possible combinations
    """
    n = len(elements)
    if r > n:
        return []
    
    result = []
    indices = list(range(r))
    
    # Add the first combination
    result.append([elements[i] for i in indices])
    
    while True:
        # Find the rightmost index that can be incremented
        i = r - 1
        while i >= 0 and indices[i] == i + n - r:
            i -= 1
        
        if i < 0:
            # No more combinations
            break
        
        # Increment the found index
        indices[i] += 1
        
        # Update all indices to the right
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        
        # Add the new combination
        result.append([elements[i] for i in indices])
    
    return result
```

### Combinations with Repetition

```python
def combinations_with_repetition(elements, r):
    """
    Generate all combinations of r elements from the given list,
    allowing elements to be repeated.
    
    Args:
        elements: List of elements
        r: Number of elements to select
        
    Returns:
        List of all possible combinations with repetition
    """
    n = len(elements)
    if n == 0 and r > 0:
        return []
    
    # Recursive implementation
    if r == 0:
        return [[]]
    
    result = []
    for i in range(n):
        # We can select elements[i] and any element after it (including itself)
        for combo in combinations_with_repetition(elements[i:], r - 1):
            result.append([elements[i]] + combo)
    
    return result
```

## 🔍 Complexity Analysis

| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| Binomial Coefficient | O(k) | O(1) |
| Recursive Combinations | O(n choose r) | O(n choose r) |
| Iterative Combinations | O(n choose r) | O(n choose r) |
| Combinations with Repetition | O((n+r-1) choose r) | O((n+r-1) choose r) |

Note: The complexity for generating combinations is proportional to the number of outputs.

## 🧩 Applications

1. **Subset Selection**: Choosing optimal subsets in various algorithms
2. **Sampling**: Random sampling for statistical analysis
3. **Lottery Systems**: Calculating odds and generating number combinations
4. **Game Theory**: Analyzing possible strategies and outcomes
5. **Machine Learning**: Feature selection and ensemble methods
6. **Network Analysis**: Graph theory problems like clique finding
7. **Bioinformatics**: DNA and protein sequence analysis

## 📝 Practice Problems

1. **Subset Generation**: Generate all possible subsets of a set (power set)
2. **k-Combination Sum**: Find all combinations of k numbers that sum to a target
3. **Binomial Expansion**: Compute coefficients in the expansion of (x + y)ⁿ
4. **Lottery Odds**: Calculate the probability of winning various lottery formats
5. **Card Combinations**: Calculate probabilities of different poker hands

## 🌟 Pro Tips

- Use the symmetry property C(n,k) = C(n,n-k) to optimize calculations
- For large values of n and k, use logarithms to avoid overflow
- When generating combinations, consider iterative approaches for better performance
- In most programming languages, there are built-in libraries for combinations:
  - Python: `itertools.combinations()`
  - C++: `std::next_permutation()` with appropriate setup
- For combinations with constraints, consider dynamic programming or backtracking
- Pascal's triangle can be used to compute binomial coefficients efficiently
- Remember that C(n,r) = P(n,r) / r!

## 🔗 Related Algorithms

- [Permutations](permutations.md)
- [Binomial Coefficients](binomial-coefficients.md)
- [Pascal's Triangle](binomial-coefficients.md#pascals-triangle)
- [Power Set Generation](../backtracking/subsets.md)
- [Dynamic Programming](../dp/fundamentals.md)

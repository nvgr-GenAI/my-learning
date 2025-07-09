# Binomial Coefficients

## 🎯 Overview

Binomial coefficients, denoted as C(n,k) or $\binom{n}{k}$, represent the number of ways to choose k objects from a set of n distinct objects without regard to order. They are central to combinatorics and appear in many mathematical contexts, including the binomial theorem, probability theory, and algorithms for counting and optimization.

## 📋 Core Concepts

### Definition

The binomial coefficient $\binom{n}{k}$ (read as "n choose k") is given by:

$$\binom{n}{k} = \frac{n!}{k!(n-k)!}$$

For example, $\binom{5}{2} = \frac{5!}{2!(5-2)!} = \frac{5 \times 4}{2 \times 1} = 10$

### Key Properties

1. $\binom{n}{k} = \binom{n}{n-k}$ (Symmetry property)
2. $\binom{n}{0} = \binom{n}{n} = 1$ (Boundary cases)
3. $\binom{n}{1} = n$ (Choosing one element)
4. $\binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k}$ (Pascal's identity)
5. $\sum_{k=0}^{n} \binom{n}{k} = 2^n$ (Sum of all binomial coefficients)

### Pascal's Triangle

Pascal's triangle is a triangular array where each number is the sum of the two numbers above it, and all binomial coefficients can be found in this triangle:

```
    1
   1 1
  1 2 1
 1 3 3 1
1 4 6 4 1
```

The entry in row n and column k is $\binom{n}{k}$.

## ⚙️ Algorithm Implementations

### Direct Computation

```python
def binomial_coefficient(n, k):
    """
    Calculate binomial coefficient C(n,k) using multiplicative formula.
    Optimized to avoid overflow.
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    
    # Use symmetry property to reduce calculations
    k = min(k, n - k)
    
    result = 1
    for i in range(k):
        result *= (n - i)
        result //= (i + 1)
    
    return result
```

### Dynamic Programming Approach

```python
def binomial_coefficient_dp(n, k):
    """
    Calculate binomial coefficient C(n,k) using dynamic programming.
    This is efficient for computing multiple binomial coefficients.
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    
    # Create a 2D table for DP
    dp = [[0 for _ in range(k + 1)] for _ in range(n + 1)]
    
    # Base case: C(i,0) = 1 for all i
    for i in range(n + 1):
        dp[i][0] = 1
    
    # Fill the table using Pascal's identity
    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
    
    return dp[n][k]
```

### Space-Optimized DP

```python
def binomial_coefficient_dp_optimized(n, k):
    """
    Calculate binomial coefficient C(n,k) using space-optimized DP.
    Uses O(k) space instead of O(n*k).
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    
    # Use symmetry property
    k = min(k, n - k)
    
    # Create a 1D array for DP
    dp = [0] * (k + 1)
    dp[0] = 1
    
    # Fill the array using Pascal's identity
    for i in range(1, n + 1):
        # Update in reverse to avoid overriding values needed for current iteration
        for j in range(min(i, k), 0, -1):
            dp[j] = dp[j] + dp[j-1]
    
    return dp[k]
```

### Lucas Theorem for Modular Binomial Coefficients

```python
def lucas_theorem(n, k, p):
    """
    Calculate C(n,k) mod p where p is a prime number.
    Uses Lucas' theorem for efficient computation.
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    
    # Base case
    if n < p and k < p:
        return binomial_coefficient(n, k) % p
    
    # Extract digits in base p
    n_digits = []
    k_digits = []
    
    n_temp, k_temp = n, k
    while n_temp > 0:
        n_digits.append(n_temp % p)
        n_temp //= p
    
    while k_temp > 0:
        k_digits.append(k_temp % p)
        k_temp //= p
    
    # Pad k_digits with zeros if needed
    while len(k_digits) < len(n_digits):
        k_digits.append(0)
    
    # Apply Lucas' theorem
    result = 1
    for i in range(len(n_digits)):
        if i < len(k_digits):
            result = (result * binomial_coefficient(n_digits[i], k_digits[i])) % p
    
    return result
```

## 🔍 Complexity Analysis

| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| Direct Computation | O(k) | O(1) |
| Dynamic Programming | O(n*k) | O(n*k) |
| Space-Optimized DP | O(n*k) | O(k) |
| Lucas Theorem | O(log n) | O(log n) |

## 🧩 Applications

1. **Combinatorial Counting**: Counting ways to select objects from a set
2. **Probability Theory**: Computing binomial probabilities
3. **Binomial Expansion**: Expanding expressions of the form (x + y)ⁿ
4. **Coding Theory**: Error-correcting codes and data compression
5. **Graph Theory**: Counting paths and structures in graphs
6. **Computational Biology**: DNA sequence analysis and alignment
7. **Statistical Sampling**: Computing sample size and distribution

## 📝 Practice Problems

1. **Pascal's Triangle**: Generate the first n rows of Pascal's triangle
2. **Large Binomial Coefficients**: Compute C(n,k) mod m for large n and k
3. **Binomial Sum**: Calculate sums involving binomial coefficients
4. **Probability Problems**: Solve problems using the binomial probability distribution
5. **Combinatorial Identities**: Prove or verify various identities involving binomial coefficients

## 🌟 Pro Tips

- Use the symmetry property C(n,k) = C(n,n-k) to optimize calculations
- For large values of n and k, use logarithms and exponentiation to avoid overflow
- When computing C(n,k) mod m, use Lucas' theorem if m is prime
- For multiple queries, precompute and store values in a table
- Remember that row n of Pascal's triangle gives all coefficients for (x+y)ⁿ
- Use factorial cancellation when possible to simplify calculations
- Be aware of the relationship with combinations: C(n,k) equals the number of k-element subsets of an n-element set

## 🔗 Related Algorithms

- [Combinations](combinations.md)
- [Permutations](permutations.md)
- [Catalan Numbers](catalan-numbers.md)
- [Modular Exponentiation](binary-exponentiation.md)
- [Dynamic Programming](../dp/fundamentals.md)

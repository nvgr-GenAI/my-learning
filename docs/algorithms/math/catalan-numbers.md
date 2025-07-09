# Catalan Numbers

## 🎯 Overview

Catalan numbers form a sequence of positive integers that appear in numerous counting problems in combinatorics. Named after Belgian mathematician Eugène Charles Catalan, these numbers describe the number of ways certain patterns can be formed and are fundamental to solving many recursive structure problems in computer science and mathematics.

## 📋 Core Concepts

### Definition

The nth Catalan number, Cₙ, can be defined by the following formulas:

1. Explicit formula: $C_n = \frac{1}{n+1}\binom{2n}{n}$

2. Recursive formula: $C_0 = 1$ and $C_{n+1} = \sum_{i=0}^{n} C_i \cdot C_{n-i}$ for n ≥ 0

3. Another recursive form: $C_{n+1} = \frac{2(2n+1)}{n+2} \cdot C_n$

### The First Few Catalan Numbers

The sequence of Catalan numbers begins:
1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796, ...

For n = 0, 1, 2, 3, 4, 5, 6, ...

### Important Properties

- Catalan numbers grow asymptotically as $\frac{4^n}{n^{3/2}\sqrt{\pi}}$
- They are always integers despite the division in the formula
- They have connections to numerous combinatorial structures

## ⚙️ Algorithm Implementations

### Computing Catalan Numbers using the Formula

```python
def catalan(n):
    """
    Calculate the nth Catalan number using the binomial coefficient formula.
    
    Args:
        n: Non-negative integer
        
    Returns:
        The nth Catalan number
    """
    if n < 0:
        return 0
    
    # Use the formula C_n = (1/(n+1)) * binomial(2n, n)
    result = 1
    
    # Calculate binomial(2n, n)
    for i in range(n):
        result *= (2 * n - i)
        result //= (i + 1)
    
    # Divide by (n+1)
    result //= (n + 1)
    
    return result
```

### Dynamic Programming Approach

```python
def catalan_dp(n):
    """
    Calculate Catalan numbers using dynamic programming.
    More efficient for computing multiple Catalan numbers.
    
    Args:
        n: Maximum Catalan number index to compute
        
    Returns:
        List of Catalan numbers from C_0 to C_n
    """
    if n < 0:
        return []
    
    # Initialize array for DP
    catalan = [0] * (n + 1)
    catalan[0] = 1
    
    # Fill using recurrence relation
    for i in range(1, n + 1):
        for j in range(i):
            catalan[i] += catalan[j] * catalan[i - j - 1]
    
    return catalan
```

### Computing Catalan Numbers with Modular Arithmetic

```python
def catalan_modular(n, mod):
    """
    Calculate nth Catalan number modulo mod.
    Useful for large n when only the remainder is needed.
    
    Args:
        n: Non-negative integer
        mod: Modulus
        
    Returns:
        The nth Catalan number modulo mod
    """
    if n < 0:
        return 0
    if n == 0:
        return 1
    
    # Calculate modular multiplicative inverse of n+1
    def mod_inverse(a, m):
        """Calculate the modular inverse of a modulo m using Fermat's Little Theorem"""
        return pow(a, m - 2, m) if is_prime(m) else pow(a, m - 1, m)
    
    def is_prime(num):
        """Check if a number is prime"""
        if num <= 1:
            return False
        if num <= 3:
            return True
        if num % 2 == 0 or num % 3 == 0:
            return False
        i = 5
        while i * i <= num:
            if num % i == 0 or num % (i + 2) == 0:
                return False
            i += 6
        return True
    
    # Use the formula C_n = (1/(n+1)) * binomial(2n, n) modulo mod
    result = 1
    
    # Calculate binomial(2n, n) modulo mod
    for i in range(n):
        result = (result * (2 * n - i) % mod) * mod_inverse(i + 1, mod) % mod
    
    # Multiply by modular inverse of (n+1)
    result = (result * mod_inverse(n + 1, mod)) % mod
    
    return result
```

## 🔍 Applications in Combinatorial Counting

Catalan numbers count various combinatorial structures, including:

1. **Valid Parentheses**: Number of ways to correctly match n pairs of parentheses
   - Example: For n=3, the valid arrangements are: `()()()`, `()(())`, `(())()`, `((()))`, `(()())`

2. **Binary Trees**: Number of different binary trees with n nodes

3. **Triangulation of a Polygon**: Ways to divide a convex polygon with (n+2) sides into triangles by connecting vertices

4. **Path Counting**: Non-crossing paths in a grid from (0,0) to (n,n) that never rise above the diagonal

5. **Mountain Ranges**: Number of ways to form a mountain range with n upstrokes and n downstrokes

## ⚙️ Complexity Analysis

| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| Formula-based | O(n) | O(1) |
| Dynamic Programming | O(n²) | O(n) |
| Modular Calculation | O(n log mod) | O(1) |

## 🧩 Applications in Computer Science

1. **Parsing**: Context-free grammar parsing and expression evaluation
2. **Data Structures**: Counting various tree and stack configurations
3. **Dynamic Programming**: Solving problems with recursive structure
4. **Graph Theory**: Counting certain types of paths in graphs
5. **Compiler Design**: Expression parsing and syntax analysis

## 📝 Practice Problems

1. **Valid Parentheses**: Count/generate all valid arrangements of n pairs of parentheses
2. **Binary Search Trees**: Count the number of distinct BSTs that can be built with n keys
3. **Polygon Triangulation**: Find the number of ways to triangulate a convex polygon
4. **Dyck Words**: Generate all Dyck words of a given length
5. **Non-Crossing Handshakes**: Count ways n people can shake hands without crossings

## 🌟 Pro Tips

- Recognize problems that have a recursive structure similar to Catalan numbers
- For large n, use the asymptotic approximation or logarithms to avoid overflow
- Many Catalan number problems can be solved with dynamic programming
- The relation to binomial coefficients provides an efficient way to compute large Catalan numbers
- Look for problems involving balanced structures, non-crossing partitions, or recursive patterns

## 🔗 Related Algorithms

- [Binomial Coefficients](binomial-coefficients.md)
- [Dynamic Programming](../dp/fundamentals.md)
- [Binary Trees](../trees/binary-trees.md)
- [Recursive Algorithms](../divide-conquer/fundamentals.md)
- [Combinatorial Generation](combinations.md)

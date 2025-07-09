# Pigeonhole Principle

## 🎯 Overview

The Pigeonhole Principle is a simple yet powerful concept in combinatorics and discrete mathematics. In its basic form, it states that if n items are placed into m containers, where n > m, then at least one container must contain more than one item. This seemingly obvious principle leads to elegant solutions for many complex problems in computer science, number theory, and combinatorics.

## 📋 Core Concepts

### Basic Principle

If n pigeons are placed into m pigeonholes and n > m, then at least one pigeonhole must contain more than one pigeon.

### Generalized Principle

If n items are placed into m containers, then at least one container must contain at least ⌈n/m⌉ items.

### Strong Form

If n items are distributed among m containers, then the maximum number of items in any container is at least ⌈n/m⌉ and the minimum number is at most ⌊n/m⌋.

## ⚙️ Applications and Examples

### 1. Duplicate Elements in Arrays

```python
def find_duplicate(arr):
    """
    Find a duplicate in an array of n+1 integers with values in range [1,n]
    using the pigeonhole principle.
    
    Args:
        arr: List of n+1 integers with values between 1 and n
        
    Returns:
        A duplicate element (there must be at least one)
    """
    n = len(arr) - 1  # Array size is n+1
    
    # Using sum approach
    expected_sum = n * (n + 1) // 2
    actual_sum = sum(arr)
    
    if actual_sum > expected_sum:
        # Use a set to find the duplicate
        seen = set()
        for num in arr:
            if num in seen:
                return num
            seen.add(num)
    
    return None  # This shouldn't happen if input follows constraints
```

### 2. Longest Increasing Subsequence in a Permutation

```python
def longest_increasing_subsequence_bound(n):
    """
    Calculate the minimum length of the longest increasing or decreasing
    subsequence in any permutation of length n.
    
    By the pigeonhole principle, this length is at least ceil(sqrt(n)).
    
    Args:
        n: Length of the permutation
        
    Returns:
        Lower bound on the length of the longest monotonic subsequence
    """
    import math
    return math.ceil(math.sqrt(n))
```

### 3. Periodic Decimal Expansions

```python
def decimal_period_bound(denominator):
    """
    Calculate the maximum possible period length of the decimal expansion
    of 1/denominator.
    
    By the pigeonhole principle, the period length is at most denominator-1.
    
    Args:
        denominator: The denominator of the fraction
        
    Returns:
        Upper bound on the period length
    """
    # Remove factors of 2 and 5 which don't contribute to periodicity
    d = denominator
    while d % 2 == 0:
        d //= 2
    while d % 5 == 0:
        d //= 5
    
    if d == 1:
        return 0  # Terminating decimal
    
    return d - 1  # Maximum possible period length
```

## 🔍 Common Proof Techniques

### Direct Application

1. Identify the items (pigeons)
2. Identify the containers (pigeonholes)
3. Verify that there are more items than containers
4. Conclude that some container must have more than one item

### Counting Argument

1. Calculate the maximum capacity if items were evenly distributed
2. Show that the actual distribution must exceed this capacity somewhere

### Contradiction Method

1. Assume the desired property doesn't hold
2. Show this would lead to a distribution violating the pigeonhole principle
3. Conclude the assumption must be false

## 🧩 Classic Problems and Solutions

### Handshake Problem

**Problem**: In a group of 6 people, show that at least 3 people must either all know each other or all be strangers.

**Solution**: Represent each person as a vertex in a graph. For any vertex, there are 5 edges connecting it to other vertices. Each edge is either colored red (know each other) or blue (strangers). By the pigeonhole principle, at least 3 edges must be the same color. These 3 edges either form a red triangle (3 people who all know each other) or a blue triangle (3 strangers).

### Socks in a Drawer

**Problem**: If a drawer contains 5 red socks, 4 blue socks, and 6 black socks, how many socks must you pull out (in the dark) to ensure you have a matching pair?

**Solution**: At most one sock can be drawn from each color without forming a pair. By the pigeonhole principle, drawing 3+1=4 socks ensures at least one matching pair.

### Periodic Functions

**Problem**: Show that any function f: {1, 2, ..., n+1} → {1, 2, ..., n} must map at least two distinct values to the same output.

**Solution**: We have n+1 inputs and only n possible outputs. By the pigeonhole principle, at least two inputs must map to the same output.

## ⚙️ Algorithmic Applications

### 1. Hash Collision Analysis

```python
def collision_probability(hash_size, data_size):
    """
    Calculate the probability of a hash collision using the birthday paradox
    (a direct application of the pigeonhole principle).
    
    Args:
        hash_size: Number of possible hash values
        data_size: Number of data items being hashed
        
    Returns:
        Approximate probability of a collision
    """
    import math
    if data_size > hash_size:
        return 1.0  # By pigeonhole principle, collision is guaranteed
    
    # Approximate formula for birthday paradox
    exponent = -data_size * (data_size - 1) / (2 * hash_size)
    probability = 1 - math.exp(exponent)
    
    return probability
```

### 2. Element Distinctness Problem

```python
def contains_duplicate(arr):
    """
    Determine if an array contains any duplicate elements.
    
    Args:
        arr: List of elements
        
    Returns:
        True if a duplicate exists, False otherwise
    """
    n = len(arr)
    m = len(set(arr))  # Number of distinct elements
    
    # By pigeonhole principle, if m < n, there must be a duplicate
    return m < n
```

## 📝 Practice Problems

1. **Birthday Paradox**: In a room of 23 people, what's the probability that at least two share a birthday?
2. **Subset Sum**: Prove that in any set of n+1 numbers selected from {1, 2, ..., 2n}, there exists a subset with a sum divisible by n+1.
3. **Coloring Problems**: Prove that if we color the points of a unit square, there must exist two points of the same color that are at most √2 apart.
4. **Repeated Digits**: Show that any 10-digit number must have at least one digit that appears more than once.
5. **Sequence Properties**: Prove that in any sequence of n integers, there exists a contiguous subsequence whose sum is divisible by n.

## 🌟 Pro Tips

- The principle is most powerful when combined with other techniques like modular arithmetic or the extreme principle
- Look for ways to map your problem to a "more items than containers" scenario
- Consider the contrapositive: if each container has at most one item, then there can be at most m items
- For generalized problems, calculate ceiling(n/m) to find the minimum number of items in the most occupied container
- In algorithm analysis, the pigeonhole principle often helps establish lower bounds
- The principle works with any mapping from a larger set to a smaller set

## 🔗 Related Concepts

- [Combinatorics](combinatorics.md)
- [Ramsey Theory](ramsey-theory.md)
- [Counting Principles](counting-principles.md)
- [Birthday Paradox](birthday-paradox.md)
- [Discrete Mathematics](discrete-mathematics.md)

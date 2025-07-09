# Permutations

## 🎯 Overview

Permutations represent the different ways to arrange a set of distinct objects in a specific order. Permutation problems are common in combinatorics, probability, and algorithm design. Understanding how to compute and generate permutations is essential for solving many computational problems.

## 📋 Core Concepts

### Definition

A permutation of n distinct objects is an ordered arrangement of these objects. The number of permutations of n distinct objects is n! (n factorial).

### Permutations of n objects taken r at a time

The number of ways to arrange r objects selected from a set of n distinct objects is denoted as P(n,r) or nPr.

$$P(n,r) = \frac{n!}{(n-r)!} = n \times (n-1) \times (n-2) \times \ldots \times (n-r+1)$$

For example, P(5,3) = 5 × 4 × 3 = 60.

### Permutations with Repetitions

If some objects are identical, the number of distinct permutations is reduced.

For a set of n objects where object type i appears n_i times, the number of distinct permutations is:

$$\frac{n!}{n_1! \times n_2! \times \ldots \times n_k!}$$

## ⚙️ Algorithm Implementations

### Generating All Permutations (Recursive)

```python
def generate_permutations(elements):
    """
    Generate all permutations of a list of elements recursively.
    
    Args:
        elements: List of elements to permute
        
    Returns:
        List of all permutations
    """
    if len(elements) <= 1:
        return [elements]
    
    result = []
    for i in range(len(elements)):
        # Take current element
        current = elements[i]
        
        # Generate permutations of remaining elements
        remaining_elements = elements[:i] + elements[i+1:]
        remaining_permutations = generate_permutations(remaining_elements)
        
        # Add current element to each permutation of remaining elements
        for perm in remaining_permutations:
            result.append([current] + perm)
    
    return result
```

### Generating All Permutations (Heap's Algorithm)

Heap's algorithm is more efficient for generating all permutations:

```python
def heaps_algorithm(elements):
    """
    Generate all permutations of a list using Heap's algorithm.
    This is more efficient than the recursive approach.
    
    Args:
        elements: List of elements to permute
        
    Returns:
        List of all permutations
    """
    n = len(elements)
    result = []
    c = [0] * n
    
    # Add the initial configuration
    result.append(elements.copy())
    
    i = 0
    while i < n:
        if c[i] < i:
            if i % 2 == 0:
                # Swap elements[0] and elements[i]
                elements[0], elements[i] = elements[i], elements[0]
            else:
                # Swap elements[c[i]] and elements[i]
                elements[c[i]], elements[i] = elements[i], elements[c[i]]
            
            # Add the new permutation
            result.append(elements.copy())
            
            # Increment counter
            c[i] += 1
            i = 0
        else:
            # Reset counter
            c[i] = 0
            i += 1
    
    return result
```

### Next Permutation Algorithm

This algorithm generates the next lexicographically greater permutation:

```python
def next_permutation(arr):
    """
    Rearranges arr in-place to the next lexicographically greater permutation.
    Returns True if such permutation exists, False otherwise.
    """
    # Find the largest index k such that a[k] < a[k + 1]
    k = len(arr) - 2
    while k >= 0:
        if arr[k] < arr[k + 1]:
            break
        k -= 1
    
    # If no such index exists, this is the last permutation
    if k < 0:
        arr.reverse()  # revert to the first permutation
        return False
    
    # Find the largest index l greater than k such that a[k] < a[l]
    l = len(arr) - 1
    while l > k:
        if arr[k] < arr[l]:
            break
        l -= 1
    
    # Swap elements at k and l
    arr[k], arr[l] = arr[l], arr[k]
    
    # Reverse the sequence from k + 1 to end
    arr[k + 1:] = arr[k + 1:][::-1]
    
    return True
```

### Computing nPr Efficiently

```python
def permutations(n, r):
    """
    Calculate the number of permutations of n things taken r at a time (nPr).
    """
    result = 1
    for i in range(n, n - r, -1):
        result *= i
    return result
```

## 🔍 Complexity Analysis

| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| Recursive Generation | O(n × n!) | O(n × n!) |
| Heap's Algorithm | O(n!) | O(n × n!) |
| Next Permutation | O(n) | O(1) |
| Computing nPr | O(r) | O(1) |

## 🧩 Applications

1. **Combinatorial Optimization**: Traveling Salesman Problem, Job Scheduling
2. **Sequence Alignment**: In bioinformatics for DNA sequence matching
3. **Cryptography**: For designing permutation-based ciphers
4. **Game Development**: For generating different level layouts or game states
5. **Statistics**: For permutation tests and sampling without replacement
6. **String Problems**: Generating all possible anagrams of a word

## 📝 Practice Problems

1. **Generate Permutations**: Write a function to generate all permutations of a string
2. **Next Permutation**: Given a permutation, find the next lexicographically greater permutation
3. **Permutation Ranking**: Find the rank of a permutation among all possible permutations
4. **Permutation Inverse**: Find the inverse of a permutation
5. **Permutation Check**: Determine if one string is a permutation of another

## 🌟 Pro Tips

- For large values of n, computing n! directly can lead to overflow; use logarithms or specialized libraries
- Use iterative algorithms for generating permutations when n is large to avoid stack overflow
- The number of permutations grows factorially, so be cautious with large inputs
- When only counting permutations (not generating them), use mathematical formulas for efficiency
- For permutation generation with specific constraints, consider backtracking algorithms
- In competitive programming, prefer the standard library functions when available:
  - Python: `itertools.permutations()`
  - C++: `std::next_permutation()`

## 🔗 Related Algorithms

- [Combinations](combinations.md)
- [Binomial Coefficients](binomial-coefficients.md)
- [Catalan Numbers](catalan-numbers.md)
- [Backtracking Algorithms](../backtracking/index.md)
- [Factorial Calculation](number-theory.md)

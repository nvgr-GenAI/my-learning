# Binary Exponentiation

Binary exponentiation (also known as exponentiation by squaring) is an algorithm to efficiently compute a^n by performing O(log n) multiplications rather than O(n) multiplications required by the naive approach.

## Problem Statement

Given a base `a` and an exponent `n`, compute `a^n` efficiently.

## Naive Approach

The straightforward approach is to multiply `a` by itself `n` times:

```python
def power_naive(a, n):
    result = 1
    for _ in range(n):
        result *= a
    return result
```

**Time Complexity**: O(n) - We perform n multiplications.
**Space Complexity**: O(1) - Constant space is used.

## Binary Exponentiation Approach

The key insight of binary exponentiation is to use the following recursive relationship:

$$
a^n = 
\begin{cases}
1 & \text{if } n = 0 \\
(a^{\lfloor n/2 \rfloor})^2 & \text{if } n \text{ is even} \\
a \times (a^{\lfloor n/2 \rfloor})^2 & \text{if } n \text{ is odd}
\end{cases}
$$

This can be implemented recursively:

```python
def binary_exponentiation_recursive(a, n):
    if n == 0:
        return 1
    
    half_pow = binary_exponentiation_recursive(a, n // 2)
    
    if n % 2 == 0:
        return half_pow * half_pow
    else:
        return a * half_pow * half_pow
```

Or iteratively, which is generally more efficient:

```python
def binary_exponentiation(a, n):
    result = 1
    
    # Process the exponent bit by bit
    while n > 0:
        # If the current bit is 1, multiply the result by a
        if n & 1:  # Equivalent to n % 2 == 1
            result *= a
        
        # Square a for the next bit
        a *= a
        
        # Move to the next bit
        n >>= 1  # Equivalent to n //= 2
    
    return result
```

**Time Complexity**: O(log n) - We perform log n multiplications.
**Space Complexity**: O(1) for the iterative version, O(log n) for the recursive version due to call stack.

## Modular Binary Exponentiation

In many applications, we need to compute (a^n) % m. We can incorporate modular arithmetic into our algorithm:

```python
def modular_binary_exponentiation(a, n, m):
    a %= m  # Ensure a is within the range [0, m-1]
    result = 1
    
    while n > 0:
        if n & 1:
            result = (result * a) % m
        
        a = (a * a) % m
        n >>= 1
    
    return result
```

**Time Complexity**: O(log n)
**Space Complexity**: O(1)

## Matrix Exponentiation

Binary exponentiation can also be applied to matrices, which is useful for solving recurrence relations:

```python
def matrix_multiply(A, B, mod=None):
    C = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]
                if mod:
                    C[i][j] %= mod
    
    return C

def matrix_power(A, n, mod=None):
    result = [[1 if i == j else 0 for j in range(len(A))] for i in range(len(A))]
    
    while n > 0:
        if n & 1:
            result = matrix_multiply(result, A, mod)
        
        A = matrix_multiply(A, A, mod)
        n >>= 1
    
    return result
```

## Applications

1. **Modular Exponentiation**: Used in cryptography (RSA), computing modular inverse, etc.
2. **Fast Fibonacci Calculation**: Using matrix exponentiation to compute Fibonacci numbers.
3. **Recurrence Relations**: Solving linear recurrence relations in O(log n) time.
4. **Powering Matrices**: Computing A^n efficiently, useful in graph algorithms.

## Examples and Practice Problems

### Computing Large Fibonacci Numbers

```python
def fibonacci(n):
    if n <= 1:
        return n
    
    # Define the matrix [[1, 1], [1, 0]]
    F = [[1, 1], [1, 0]]
    
    # Compute F^(n-1)
    power_matrix = matrix_power(F, n - 1)
    
    # The answer is the top-left element of the resulting matrix
    return power_matrix[0][0]
```

### Practice Problems:

1. [SPOJ - LASTDIG](https://www.spoj.com/problems/LASTDIG/) - Find the last digit of a^b
2. [Codeforces - 1095C](https://codeforces.com/problemset/problem/1095/C) - Powers of Two
3. [Hackerrank - Fibonacci Modified](https://www.hackerrank.com/challenges/fibonacci-modified/problem)
4. [Codechef - FEXP](https://www.codechef.com/problems/FEXP) - Fast Exponentiation

## Pro Tips

1. **Overflow Handling**: Be careful about integer overflow. When working with large numbers, use modular arithmetic or a big integer library.
2. **Edge Cases**: Pay attention to edge cases like a=0 or n=0.
3. **Optimization**: For some specific cases (like when a=2), bit manipulation can be more efficient.
4. **Pre-processing**: Sometimes, you can pre-compute and store powers of a (a^1, a^2, a^4, a^8, ...) to speed up multiple exponentiation operations.
5. **Fibonacci Optimization**: For Fibonacci numbers, there's also a closed-form formula (Binet's formula) that can be used for relatively small n.

## Related Topics

- [Modular Arithmetic](modular-arithmetic.md)
- [Fast Fourier Transform](fft.md)
- [Linear Algebra Basics](linear-algebra.md)
- [Prime Numbers](prime-numbers.md)

# Strassen Matrix Multiplication 🔢

Efficient matrix multiplication algorithm using divide-and-conquer approach.

## 🎯 Problem Statement

Multiply two n×n matrices using fewer scalar multiplications than the standard O(n³) algorithm.

**Input**: Two n×n matrices A and B
**Output**: Matrix C = A × B

## 🧠 Algorithm Approach

### Standard Matrix Multiplication

```python
def standard_multiply(A, B):
    n = len(A)
    C = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    
    return C
```

**Time Complexity**: O(n³)

### Strassen's Algorithm

Reduces 8 recursive multiplications to 7 by using clever additions and subtractions.

## 📝 Implementation

```python
import numpy as np
from typing import List

def add_matrices(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    """Add two matrices"""
    n = len(A)
    C = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            C[i][j] = A[i][j] + B[i][j]
    
    return C

def subtract_matrices(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    """Subtract two matrices"""
    n = len(A)
    C = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            C[i][j] = A[i][j] - B[i][j]
    
    return C

def split_matrix(matrix: List[List[int]]) -> tuple:
    """Split matrix into 4 quadrants"""
    n = len(matrix)
    mid = n // 2
    
    a11 = [[matrix[i][j] for j in range(mid)] for i in range(mid)]
    a12 = [[matrix[i][j] for j in range(mid, n)] for i in range(mid)]
    a21 = [[matrix[i][j] for j in range(mid)] for i in range(mid, n)]
    a22 = [[matrix[i][j] for j in range(mid, n)] for i in range(mid, n)]
    
    return a11, a12, a21, a22

def combine_matrices(c11, c12, c21, c22) -> List[List[int]]:
    """Combine 4 quadrants into single matrix"""
    n = len(c11) * 2
    C = [[0] * n for _ in range(n)]
    mid = n // 2
    
    # Copy c11
    for i in range(mid):
        for j in range(mid):
            C[i][j] = c11[i][j]
    
    # Copy c12
    for i in range(mid):
        for j in range(mid):
            C[i][j + mid] = c12[i][j]
    
    # Copy c21
    for i in range(mid):
        for j in range(mid):
            C[i + mid][j] = c21[i][j]
    
    # Copy c22
    for i in range(mid):
        for j in range(mid):
            C[i + mid][j + mid] = c22[i][j]
    
    return C

def strassen_multiply(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    """Strassen's matrix multiplication algorithm"""
    n = len(A)
    
    # Base case
    if n == 1:
        return [[A[0][0] * B[0][0]]]
    
    # If n is not power of 2, pad with zeros
    if n & (n - 1) != 0:
        next_power = 1
        while next_power < n:
            next_power <<= 1
        
        # Pad matrices
        A_padded = [[0] * next_power for _ in range(next_power)]
        B_padded = [[0] * next_power for _ in range(next_power)]
        
        for i in range(n):
            for j in range(n):
                A_padded[i][j] = A[i][j]
                B_padded[i][j] = B[i][j]
        
        C_padded = strassen_multiply(A_padded, B_padded)
        
        # Extract result
        C = [[C_padded[i][j] for j in range(n)] for i in range(n)]
        return C
    
    # Split matrices into quadrants
    a11, a12, a21, a22 = split_matrix(A)
    b11, b12, b21, b22 = split_matrix(B)
    
    # Calculate the 7 Strassen products
    m1 = strassen_multiply(add_matrices(a11, a22), add_matrices(b11, b22))
    m2 = strassen_multiply(add_matrices(a21, a22), b11)
    m3 = strassen_multiply(a11, subtract_matrices(b12, b22))
    m4 = strassen_multiply(a22, subtract_matrices(b21, b11))
    m5 = strassen_multiply(add_matrices(a11, a12), b22)
    m6 = strassen_multiply(subtract_matrices(a21, a11), add_matrices(b11, b12))
    m7 = strassen_multiply(subtract_matrices(a12, a22), add_matrices(b21, b22))
    
    # Calculate result quadrants
    c11 = add_matrices(subtract_matrices(add_matrices(m1, m4), m5), m7)
    c12 = add_matrices(m3, m5)
    c21 = add_matrices(m2, m4)
    c22 = add_matrices(subtract_matrices(add_matrices(m1, m3), m2), m6)
    
    # Combine quadrants
    return combine_matrices(c11, c12, c21, c22)

# Enhanced version with threshold
def strassen_multiply_optimized(A: List[List[int]], B: List[List[int]], threshold: int = 64) -> List[List[int]]:
    """Optimized Strassen with threshold for standard multiplication"""
    n = len(A)
    
    # Use standard multiplication for small matrices
    if n <= threshold:
        return standard_multiply(A, B)
    
    return strassen_multiply(A, B)

def standard_multiply(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    """Standard O(n³) matrix multiplication"""
    n = len(A)
    C = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    
    return C

# Example usage
if __name__ == "__main__":
    # Test with 4x4 matrices
    A = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]
    
    B = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    
    # Standard multiplication
    C_standard = standard_multiply(A, B)
    print("Standard multiplication result:")
    for row in C_standard:
        print(row)
    
    # Strassen multiplication
    C_strassen = strassen_multiply(A, B)
    print("\\nStrassen multiplication result:")
    for row in C_strassen:
        print(row)
    
    # Verify results are the same
    print(f"\\nResults match: {C_standard == C_strassen}")
```

## ⚡ Time Complexity Analysis

### Strassen's Algorithm
- **Time Complexity**: O(n^log₂7) ≈ O(n^2.807)
- **Space Complexity**: O(n²)

### Comparison
| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Standard | O(n³) | O(n²) |
| Strassen | O(n^2.807) | O(n²) |
| Coppersmith-Winograd | O(n^2.376) | O(n²) |

## 🔄 Strassen's Seven Products

```text
Given matrices divided into quadrants:
A = [a11 a12]    B = [b11 b12]
    [a21 a22]        [b21 b22]

Seven products:
m1 = (a11 + a22)(b11 + b22)
m2 = (a21 + a22)b11
m3 = a11(b12 - b22)
m4 = a22(b21 - b11)
m5 = (a11 + a12)b22
m6 = (a21 - a11)(b11 + b12)
m7 = (a12 - a22)(b21 + b22)

Result quadrants:
c11 = m1 + m4 - m5 + m7
c12 = m3 + m5
c21 = m2 + m4
c22 = m1 - m2 + m3 + m6
```

## 🎯 Key Insights

1. **Divide Strategy**: Split matrices into 2×2 blocks
2. **Clever Algebra**: Use 7 multiplications instead of 8
3. **Recursive Structure**: Apply same strategy recursively
4. **Practical Threshold**: Use standard multiplication for small matrices

## 📊 Performance Characteristics

```python
def benchmark_multiplication(n: int):
    """Benchmark different multiplication algorithms"""
    import time
    import random
    
    # Generate random matrices
    A = [[random.randint(1, 100) for _ in range(n)] for _ in range(n)]
    B = [[random.randint(1, 100) for _ in range(n)] for _ in range(n)]
    
    # Standard multiplication
    start = time.time()
    C_standard = standard_multiply(A, B)
    standard_time = time.time() - start
    
    # Strassen multiplication
    start = time.time()
    C_strassen = strassen_multiply_optimized(A, B)
    strassen_time = time.time() - start
    
    print(f"Matrix size: {n}×{n}")
    print(f"Standard time: {standard_time:.4f}s")
    print(f"Strassen time: {strassen_time:.4f}s")
    print(f"Speedup: {standard_time/strassen_time:.2f}x")
    print(f"Results match: {C_standard == C_strassen}")
```

## 🔧 Optimizations

1. **Threshold Optimization**: Use standard multiplication for small matrices
2. **Memory Management**: Reuse temporary matrices
3. **Padding Strategy**: Efficient handling of non-power-of-2 sizes
4. **Cache Optimization**: Consider memory access patterns

## 💡 Applications

- **Scientific Computing**: Large matrix operations in simulations
- **Machine Learning**: Neural network computations
- **Computer Graphics**: 3D transformations and rendering
- **Signal Processing**: Convolution operations

## 🚀 Advanced Variants

1. **Winograd's Algorithm**: Further optimization of Strassen's approach
2. **Coppersmith-Winograd**: Theoretical O(n^2.376) complexity
3. **Parallel Strassen**: Distributed implementation for large matrices
4. **Numerical Stability**: Handling floating-point precision issues

---

*Strassen's algorithm demonstrates how mathematical insight can lead to asymptotic improvements in fundamental operations.*

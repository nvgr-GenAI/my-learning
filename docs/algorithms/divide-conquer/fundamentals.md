# Divide and Conquer Fundamentals

## What is Divide and Conquer?

Divide and Conquer is an algorithmic paradigm that solves problems by breaking them down into smaller subproblems, solving each subproblem recursively, and then combining the solutions to solve the original problem.

## Core Principles

### The Three Steps

1. **Divide**: Break the problem into smaller subproblems of the same type
2. **Conquer**: Solve the subproblems recursively (base case handles trivial problems)
3. **Combine**: Merge the solutions of subproblems to get the solution to the original problem

### Generic Template

```python
def divide_and_conquer(problem):
    # Base case: problem is small enough to solve directly
    if is_base_case(problem):
        return solve_directly(problem)
    
    # Divide: split problem into subproblems
    subproblems = divide(problem)
    
    # Conquer: solve each subproblem recursively
    sub_solutions = []
    for subproblem in subproblems:
        sub_solutions.append(divide_and_conquer(subproblem))
    
    # Combine: merge solutions to get final result
    return combine(sub_solutions)
```

## Time Complexity Analysis

### Master Theorem

For recurrences of the form: `T(n) = aT(n/b) + f(n)`

Where:
- `a ≥ 1` (number of subproblems)
- `b > 1` (factor by which problem size is reduced)
- `f(n)` (cost of divide and combine steps)

**Three Cases**:

1. **Case 1**: If `f(n) = O(n^c)` where `c < log_b(a)`
   - Then `T(n) = Θ(n^(log_b(a)))`

2. **Case 2**: If `f(n) = Θ(n^c * log^k(n))` where `c = log_b(a)` and `k ≥ 0`
   - Then `T(n) = Θ(n^c * log^(k+1)(n))`

3. **Case 3**: If `f(n) = Ω(n^c)` where `c > log_b(a)`
   - Then `T(n) = Θ(f(n))` (if regularity condition holds)

### Common Patterns

```python
# Pattern 1: Binary division with linear combine
# T(n) = 2T(n/2) + O(n) → O(n log n)
# Examples: MergeSort, QuickSort

# Pattern 2: Binary division with constant combine  
# T(n) = 2T(n/2) + O(1) → O(n)
# Examples: Binary search tree operations

# Pattern 3: Multiple subproblems
# T(n) = 4T(n/2) + O(n) → O(n²)
# Examples: Matrix multiplication (naive)
```

## Classic Applications

### 1. Sorting Algorithms

#### Merge Sort
```python
def merge_sort(arr):
    """
    Divide and conquer sorting algorithm.
    
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    if len(arr) <= 1:
        return arr
    
    # Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Combine
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

#### Quick Sort
```python
def quick_sort(arr, low, high):
    """
    Divide and conquer sorting with partitioning.
    
    Average Time: O(n log n)
    Worst Time: O(n²)
    Space: O(log n) - recursion stack
    """
    if low < high:
        # Divide: partition around pivot
        pivot_index = partition(arr, low, high)
        
        # Conquer: sort subarrays
        quick_sort(arr, low, pivot_index - 1)
        quick_sort(arr, pivot_index + 1, high)

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
```

### 2. Search Algorithms

#### Binary Search
```python
def binary_search(arr, target, left, right):
    """
    Divide and conquer search in sorted array.
    
    Time Complexity: O(log n)
    Space Complexity: O(log n) - recursion
    """
    if left > right:
        return -1
    
    mid = left + (right - left) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] > target:
        return binary_search(arr, target, left, mid - 1)
    else:
        return binary_search(arr, target, mid + 1, right)
```

### 3. Mathematical Operations

#### Fast Exponentiation
```python
def power(base, exp):
    """
    Calculate base^exp using divide and conquer.
    
    Time Complexity: O(log exp)
    Space Complexity: O(log exp)
    """
    if exp == 0:
        return 1
    if exp == 1:
        return base
    
    # Divide
    half_power = power(base, exp // 2)
    
    # Combine
    if exp % 2 == 0:
        return half_power * half_power
    else:
        return half_power * half_power * base
```

#### Matrix Multiplication (Strassen's Algorithm)
```python
def strassen_multiply(A, B):
    """
    Matrix multiplication using Strassen's algorithm.
    
    Time Complexity: O(n^2.807)
    Space Complexity: O(n²)
    """
    n = len(A)
    
    # Base case: 1x1 matrices
    if n == 1:
        return [[A[0][0] * B[0][0]]]
    
    # Divide matrices into quadrants
    mid = n // 2
    
    A11 = [[A[i][j] for j in range(mid)] for i in range(mid)]
    A12 = [[A[i][j] for j in range(mid, n)] for i in range(mid)]
    A21 = [[A[i][j] for j in range(mid)] for i in range(mid, n)]
    A22 = [[A[i][j] for j in range(mid, n)] for i in range(mid, n)]
    
    B11 = [[B[i][j] for j in range(mid)] for i in range(mid)]
    B12 = [[B[i][j] for j in range(mid, n)] for i in range(mid)]
    B21 = [[B[i][j] for j in range(mid)] for i in range(mid, n)]
    B22 = [[B[i][j] for j in range(mid, n)] for i in range(mid, n)]
    
    # Compute 7 products (Strassen's optimization)
    M1 = strassen_multiply(matrix_add(A11, A22), matrix_add(B11, B22))
    M2 = strassen_multiply(matrix_add(A21, A22), B11)
    M3 = strassen_multiply(A11, matrix_subtract(B12, B22))
    M4 = strassen_multiply(A22, matrix_subtract(B21, B11))
    M5 = strassen_multiply(matrix_add(A11, A12), B22)
    M6 = strassen_multiply(matrix_subtract(A21, A11), matrix_add(B11, B12))
    M7 = strassen_multiply(matrix_subtract(A12, A22), matrix_add(B21, B22))
    
    # Combine results
    C11 = matrix_add(matrix_subtract(matrix_add(M1, M4), M5), M7)
    C12 = matrix_add(M3, M5)
    C21 = matrix_add(M2, M4)
    C22 = matrix_add(matrix_subtract(matrix_add(M1, M3), M2), M6)
    
    # Combine quadrants
    C = [[0] * n for _ in range(n)]
    for i in range(mid):
        for j in range(mid):
            C[i][j] = C11[i][j]
            C[i][j + mid] = C12[i][j]
            C[i + mid][j] = C21[i][j]
            C[i + mid][j + mid] = C22[i][j]
    
    return C
```

## Advanced Techniques

### 1. Optimization Strategies

#### Tail Recursion Optimization
```python
def binary_search_tail_recursive(arr, target, left, right):
    """Tail recursive version to avoid stack overflow."""
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] > target:
            right = mid - 1
        else:
            left = mid + 1
    
    return -1
```

#### Memoization for Overlapping Subproblems
```python
def fibonacci_divide_conquer_memo(n, memo={}):
    """
    Fibonacci using divide and conquer with memoization.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = (fibonacci_divide_conquer_memo(n-1, memo) + 
               fibonacci_divide_conquer_memo(n-2, memo))
    return memo[n]
```

### 2. Parallel Processing

```python
import multiprocessing

def parallel_merge_sort(arr):
    """
    Parallel merge sort using multiprocessing.
    """
    if len(arr) <= 1:
        return arr
    
    if len(arr) < 1000:  # Use sequential for small arrays
        return merge_sort(arr)
    
    mid = len(arr) // 2
    
    # Create processes for left and right halves
    with multiprocessing.Pool(2) as pool:
        left_future = pool.apply_async(parallel_merge_sort, (arr[:mid],))
        right_future = pool.apply_async(parallel_merge_sort, (arr[mid:],))
        
        left = left_future.get()
        right = right_future.get()
    
    return merge(left, right)
```

## Design Patterns

### 1. Binary Division Pattern
```python
def binary_divide_template(problem):
    if is_trivial(problem):
        return solve_directly(problem)
    
    left_half, right_half = split_in_half(problem)
    
    left_solution = binary_divide_template(left_half)
    right_solution = binary_divide_template(right_half)
    
    return combine_two_solutions(left_solution, right_solution)
```

### 2. Multi-way Division Pattern
```python
def multiway_divide_template(problem, k):
    if is_trivial(problem):
        return solve_directly(problem)
    
    subproblems = split_into_k_parts(problem, k)
    solutions = []
    
    for subproblem in subproblems:
        solutions.append(multiway_divide_template(subproblem, k))
    
    return combine_multiple_solutions(solutions)
```

### 3. Decrease and Conquer Pattern
```python
def decrease_and_conquer_template(problem):
    if is_trivial(problem):
        return solve_directly(problem)
    
    reduced_problem = reduce_by_constant(problem)
    sub_solution = decrease_and_conquer_template(reduced_problem)
    
    return extend_solution(sub_solution, problem)
```

## Common Pitfalls and Solutions

### 1. Stack Overflow
**Problem**: Deep recursion causes stack overflow
**Solution**: Convert to iterative or use tail recursion

### 2. Inefficient Base Cases
**Problem**: Base case too complex or not optimal
**Solution**: Choose appropriate threshold for switching to iterative

### 3. Poor Division Strategy
**Problem**: Unbalanced splits lead to poor performance
**Solution**: Use median-of-three or randomized pivoting

### 4. Expensive Combine Step
**Problem**: Combine step dominates time complexity
**Solution**: Optimize combine operation or change division strategy

## When to Use Divide and Conquer

### Good Candidates
- Problem can be broken into similar subproblems
- Subproblems are independent
- Solutions can be efficiently combined
- Recursive structure is natural

### Examples
- Sorting algorithms
- Tree and graph traversals
- Mathematical computations
- String algorithms
- Geometric algorithms

### Not Suitable When
- Subproblems overlap significantly (use DP instead)
- Problem doesn't divide naturally
- Overhead of recursion is too high
- Simple iterative solution exists

## Practice Problems

### Beginner
- Binary search
- Merge sort
- Fast exponentiation
- Finding maximum element

### Intermediate
- Quick sort with optimizations
- Closest pair of points
- Karatsuba multiplication
- Inversion counting

### Advanced
- Strassen's matrix multiplication
- Fast Fourier Transform
- Convex hull algorithms
- Advanced tree algorithms

# Divide and Conquer

## Overview

Divide and Conquer is an algorithm design paradigm that works by recursively breaking down a problem into subproblems of similar type until they become simple enough to solve directly.

## General Template

```python
def divide_and_conquer(problem):
    # Base case
    if problem.size <= threshold:
        return solve_directly(problem)
    
    # Divide
    subproblems = divide(problem)
    
    # Conquer
    solutions = []
    for subproblem in subproblems:
        solutions.append(divide_and_conquer(subproblem))
    
    # Combine
    return combine(solutions)
```

## Classic Examples

### Merge Sort

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    # Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Conquer and Combine
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

# Time Complexity: O(n log n)
# Space Complexity: O(n)
```

### Quick Sort

```python
def quick_sort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        # Partition and get pivot index
        pi = partition(arr, low, high)
        
        # Recursively sort elements before and after partition
        quick_sort(arr, low, pi - 1)
        quick_sort(arr, pi + 1, high)

def partition(arr, low, high):
    # Choose rightmost element as pivot
    pivot = arr[high]
    i = low - 1  # Index of smaller element
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# Average Time Complexity: O(n log n)
# Worst Time Complexity: O(n²)
# Space Complexity: O(log n)
```

### Binary Search

```python
def binary_search(arr, target, left=0, right=None):
    if right is None:
        right = len(arr) - 1
    
    if left > right:
        return -1
    
    mid = (left + right) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] > target:
        return binary_search(arr, target, left, mid - 1)
    else:
        return binary_search(arr, target, mid + 1, right)

# Time Complexity: O(log n)
# Space Complexity: O(log n)
```

## Advanced Applications

### Maximum Subarray (Kadane's Algorithm - D&C version)

```python
def max_subarray_dc(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    
    # Base case
    if low == high:
        return arr[low]
    
    # Divide
    mid = (low + high) // 2
    
    # Conquer
    left_sum = max_subarray_dc(arr, low, mid)
    right_sum = max_subarray_dc(arr, mid + 1, high)
    cross_sum = max_crossing_sum(arr, low, mid, high)
    
    # Combine
    return max(left_sum, right_sum, cross_sum)

def max_crossing_sum(arr, low, mid, high):
    # Find maximum sum for left side
    left_sum = float('-inf')
    current_sum = 0
    for i in range(mid, low - 1, -1):
        current_sum += arr[i]
        left_sum = max(left_sum, current_sum)
    
    # Find maximum sum for right side
    right_sum = float('-inf')
    current_sum = 0
    for i in range(mid + 1, high + 1):
        current_sum += arr[i]
        right_sum = max(right_sum, current_sum)
    
    return left_sum + right_sum

# Time Complexity: O(n log n)
# Space Complexity: O(log n)
```

### Closest Pair of Points

```python
import math

def closest_pair_points(points):
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def closest_pair_rec(px, py):
        n = len(px)
        
        # Base case for small arrays
        if n <= 3:
            min_dist = float('inf')
            for i in range(n):
                for j in range(i + 1, n):
                    min_dist = min(min_dist, distance(px[i], px[j]))
            return min_dist
        
        # Divide
        mid = n // 2
        midpoint = px[mid]
        
        pyl = [p for p in py if p[0] <= midpoint[0]]
        pyr = [p for p in py if p[0] > midpoint[0]]
        
        # Conquer
        dl = closest_pair_rec(px[:mid], pyl)
        dr = closest_pair_rec(px[mid:], pyr)
        
        d = min(dl, dr)
        
        # Combine - check points near the dividing line
        strip = [p for p in py if abs(p[0] - midpoint[0]) < d]
        
        for i in range(len(strip)):
            j = i + 1
            while j < len(strip) and (strip[j][1] - strip[i][1]) < d:
                d = min(d, distance(strip[i], strip[j]))
                j += 1
        
        return d
    
    px = sorted(points, key=lambda p: p[0])
    py = sorted(points, key=lambda p: p[1])
    
    return closest_pair_rec(px, py)

# Time Complexity: O(n log n)
# Space Complexity: O(n)
```

### Matrix Multiplication (Strassen's Algorithm)

```python
def strassen_multiply(A, B):
    n = len(A)
    
    # Base case
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
    
    # Calculate Strassen's 7 products
    M1 = strassen_multiply(add_matrix(A11, A22), add_matrix(B11, B22))
    M2 = strassen_multiply(add_matrix(A21, A22), B11)
    M3 = strassen_multiply(A11, subtract_matrix(B12, B22))
    M4 = strassen_multiply(A22, subtract_matrix(B21, B11))
    M5 = strassen_multiply(add_matrix(A11, A12), B22)
    M6 = strassen_multiply(subtract_matrix(A21, A11), add_matrix(B11, B12))
    M7 = strassen_multiply(subtract_matrix(A12, A22), add_matrix(B21, B22))
    
    # Calculate result quadrants
    C11 = add_matrix(subtract_matrix(add_matrix(M1, M4), M5), M7)
    C12 = add_matrix(M3, M5)
    C21 = add_matrix(M2, M4)
    C22 = add_matrix(subtract_matrix(add_matrix(M1, M3), M2), M6)
    
    # Combine quadrants
    C = [[0] * n for _ in range(n)]
    
    for i in range(mid):
        for j in range(mid):
            C[i][j] = C11[i][j]
            C[i][j + mid] = C12[i][j]
            C[i + mid][j] = C21[i][j]
            C[i + mid][j + mid] = C22[i][j]
    
    return C

def add_matrix(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def subtract_matrix(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

# Time Complexity: O(n^2.807)
# Space Complexity: O(n²)
```

## When to Use Divide and Conquer

1. **Problem can be broken into similar subproblems**
2. **Subproblems are independent**
3. **Combining solutions is efficient**
4. **Base case is simple to solve**

## Advantages and Disadvantages

### Advantages
- Often leads to efficient algorithms
- Natural parallelization
- Clear problem structure

### Disadvantages
- May have overhead from recursion
- Not always optimal (e.g., Fibonacci)
- Can be memory intensive

## Practice Problems

- [ ] Merge k Sorted Lists
- [ ] Maximum Subarray
- [ ] Different Ways to Add Parentheses
- [ ] Count of Smaller Numbers After Self
- [ ] Reverse Pairs
- [ ] The Skyline Problem

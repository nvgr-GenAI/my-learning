# Binary Search

Binary search is one of the most fundamental and efficient searching algorithms, operating on the divide-and-conquer principle to find elements in sorted arrays with O(log n) time complexity.

## Overview

Binary search works by repeatedly dividing the search space in half, comparing the target value with the middle element, and eliminating half of the remaining elements at each step.

**Time Complexity:** O(log n)  
**Space Complexity:** O(1) iterative, O(log n) recursive  
**Prerequisite:** Array must be sorted

## Basic Implementation

### Iterative Binary Search

```python
def binary_search(arr, target):
    """
    Standard iterative binary search implementation
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2  # Avoid overflow
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1  # Target not found

# Example usage
arr = [1, 3, 5, 7, 9, 11, 13]
target = 7
result = binary_search(arr, target)
print(f"Element {target} found at index: {result}")
```

### Recursive Binary Search

```python
def binary_search_recursive(arr, target, left=None, right=None):
    """
    Recursive binary search implementation
    """
    if left is None:
        left = 0
    if right is None:
        right = len(arr) - 1
    
    if left > right:
        return -1
    
    mid = left + (right - left) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)
```

## Binary Search Variants

### Find First Occurrence (Lower Bound)

```python
def binary_search_first(arr, target):
    """
    Find the first occurrence of target in sorted array with duplicates
    """
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result
```

### Find Last Occurrence (Upper Bound)

```python
def binary_search_last(arr, target):
    """
    Find the last occurrence of target in sorted array with duplicates
    """
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            result = mid
            left = mid + 1  # Continue searching right
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result
```

### Find Insert Position

```python
def search_insert_position(arr, target):
    """
    Find the position where target should be inserted to keep array sorted
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return left
```

## Binary Search on Answer Space

Binary search can be used to find optimal solutions in problems where:
1. The answer space is monotonic
2. We can check if a value is valid in O(n) or better

### Square Root Implementation

```python
def sqrt_binary_search(x):
    """
    Find square root using binary search
    """
    if x < 2:
        return x
    
    left, right = 2, x // 2
    
    while left <= right:
        mid = left + (right - left) // 2
        square = mid * mid
        
        if square == x:
            return mid
        elif square < x:
            left = mid + 1
        else:
            right = mid - 1
    
    return right  # Return floor value
```

### Capacity to Ship Packages Problem

```python
def ship_within_days(weights, days):
    """
    Find minimum ship capacity to ship all packages within given days
    """
    def can_ship(capacity):
        current_weight = 0
        days_needed = 1
        
        for weight in weights:
            if current_weight + weight > capacity:
                days_needed += 1
                current_weight = weight
            else:
                current_weight += weight
        
        return days_needed <= days
    
    left = max(weights)  # Minimum possible capacity
    right = sum(weights)  # Maximum possible capacity
    
    while left < right:
        mid = left + (right - left) // 2
        
        if can_ship(mid):
            right = mid
        else:
            left = mid + 1
    
    return left
```

## Binary Search Templates

### Template 1: Standard Binary Search

```python
def binary_search_template1(arr, target):
    """
    Use when you can determine the target immediately
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
```

### Template 2: Advanced Binary Search

```python
def binary_search_template2(arr, target):
    """
    Use when you need to access current index and its immediate right neighbor
    """
    left, right = 0, len(arr)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    
    return left if left < len(arr) and arr[left] == target else -1
```

### Template 3: Advanced Binary Search with Neighbors

```python
def binary_search_template3(arr, target):
    """
    Use when you need to access current index and both neighbors
    """
    left, right = 0, len(arr) - 1
    
    while left + 1 < right:
        mid = left + (right - left) // 2
        
        if arr[mid] < target:
            left = mid
        else:
            right = mid
    
    if arr[left] == target:
        return left
    if arr[right] == target:
        return right
    
    return -1
```

## Common Patterns and Problems

### 1. Peak Element Finding

```python
def find_peak_element(arr):
    """
    Find a peak element in an array (element greater than its neighbors)
    """
    left, right = 0, len(arr) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if arr[mid] > arr[mid + 1]:
            right = mid
        else:
            left = mid + 1
    
    return left
```

### 2. Search in Rotated Sorted Array

```python
def search_rotated_array(arr, target):
    """
    Search in a rotated sorted array
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        
        # Check if left half is sorted
        if arr[left] <= arr[mid]:
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1
```

## Key Points to Remember

1. **Overflow Prevention**: Use `mid = left + (right - left) // 2` instead of `mid = (left + right) // 2`

2. **Boundary Conditions**: 
   - Pay attention to `<=` vs `<` in while loop condition
   - Ensure proper handling of edge cases

3. **Template Selection**:
   - Template 1: When you can determine target immediately
   - Template 2: When you need right neighbor access
   - Template 3: When you need both neighbors

4. **Invariant Maintenance**: Always ensure the search space contains the target

5. **Termination**: Make sure the loop will eventually terminate

## Applications

- **Database Indexing**: B-trees use binary search principles
- **Library Functions**: `bisect` module in Python, `lower_bound`/`upper_bound` in C++
- **Algorithm Optimization**: Converting O(n) solutions to O(log n)
- **Competitive Programming**: Foundation for many optimization problems

Binary search is not just a searching algorithm—it's a problem-solving technique that can optimize many algorithms when the solution space has monotonic properties.

# Ternary Search

## Overview

Ternary search is a divide-and-conquer algorithm that divides the search space into three parts rather than two as in binary search. It's particularly useful for finding the maximum or minimum of a unimodal function.

## Algorithm

1. Divide the search interval into three equal parts
2. Calculate values at both partitioning points
3. Determine which third of the interval to search next
4. Repeat until the interval is small enough

```python
def ternary_search(arr, target):
    left = 0
    right = len(arr) - 1
    
    while left <= right:
        # Calculate the two mid points
        mid1 = left + (right - left) // 3
        mid2 = right - (right - left) // 3
        
        if arr[mid1] == target:
            return mid1
        if arr[mid2] == target:
            return mid2
            
        if target < arr[mid1]:
            # Target is in the first third
            right = mid1 - 1
        elif target > arr[mid2]:
            # Target is in the last third
            left = mid2 + 1
        else:
            # Target is in the middle third
            left = mid1 + 1
            right = mid2 - 1
    
    return -1  # Target not found
```

## Time and Space Complexity

- **Time Complexity**: O(log₃ n) - Which is approximately 1.58 * log₂ n
- **Space Complexity**: O(1) - Only a constant amount of extra space is used

## Advantages and Disadvantages

### Advantages

- Works well for unimodal functions where we need to find a maximum or minimum
- Has a similar time complexity to binary search (log₃ n vs log₂ n)
- Can converge faster than binary search in certain specific applications

### Disadvantages

- Despite the improved base of the logarithm, it typically performs more comparisons than binary search
- More complex to implement correctly
- Not as widely used or understood as binary search

## Use Cases

- Finding the maximum or minimum of a unimodal function
- Optimization problems where the function increases and then decreases (or vice versa)
- Numerical analysis for locating extreme points
- When the search space needs to be divided into three parts rather than two

## Implementation Details

### Python Implementation

```python
def ternary_search(arr, target):
    left = 0
    right = len(arr) - 1
    
    while left <= right:
        # Calculate the two mid points
        mid1 = left + (right - left) // 3
        mid2 = right - (right - left) // 3
        
        if arr[mid1] == target:
            return mid1
        if arr[mid2] == target:
            return mid2
            
        if target < arr[mid1]:
            # Target is in the first third
            right = mid1 - 1
        elif target > arr[mid2]:
            # Target is in the last third
            left = mid2 + 1
        else:
            # Target is in the middle third
            left = mid1 + 1
            right = mid2 - 1
    
    return -1  # Target not found

# Example usage
array = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
result = ternary_search(array, 13)
print(f"Element found at index: {result}")  # Output: Element found at index: 6
```

### Java Implementation

```java
public class TernarySearch {
    public static int ternarySearch(int[] arr, int target) {
        int left = 0;
        int right = arr.length - 1;
        
        while (left <= right) {
            // Calculate the two mid points
            int mid1 = left + (right - left) / 3;
            int mid2 = right - (right - left) / 3;
            
            if (arr[mid1] == target) {
                return mid1;
            }
            if (arr[mid2] == target) {
                return mid2;
            }
            
            if (target < arr[mid1]) {
                // Target is in the first third
                right = mid1 - 1;
            } else if (target > arr[mid2]) {
                // Target is in the last third
                left = mid2 + 1;
            } else {
                // Target is in the middle third
                left = mid1 + 1;
                right = mid2 - 1;
            }
        }
        
        return -1;  // Target not found
    }
    
    public static void main(String[] args) {
        int[] array = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29};
        int result = ternarySearch(array, 13);
        System.out.println("Element found at index: " + result);  // Output: Element found at index: 6
    }
}
```

## Ternary Search for Unimodal Functions

Ternary search is especially useful for finding the maximum or minimum of a unimodal function:

```python
def ternary_search_max(func, left, right, epsilon=1e-10):
    """
    Find the maximum value of a unimodal function using ternary search
    
    Parameters:
    - func: the function to optimize
    - left, right: the search interval boundaries
    - epsilon: the desired precision
    
    Returns:
    - x: the argument that maximizes the function
    """
    while right - left > epsilon:
        mid1 = left + (right - left) / 3
        mid2 = right - (right - left) / 3
        
        if func(mid1) < func(mid2):
            # Maximum is in the right two-thirds
            left = mid1
        else:
            # Maximum is in the left two-thirds
            right = mid2
    
    # Return the midpoint of the final interval
    return (left + right) / 2

# Example: Find the maximum of f(x) = -(x-5)^2 + 10 in [0, 10]
def f(x):
    return -(x-5)**2 + 10

max_x = ternary_search_max(f, 0, 10)
print(f"Maximum occurs at x = {max_x}, value = {f(max_x)}")
# Output: Maximum occurs at x = 5.0, value = 10.0
```

## Recursive Implementation

```python
def recursive_ternary_search(arr, target, left, right):
    if right >= left:
        # Calculate the two mid points
        mid1 = left + (right - left) // 3
        mid2 = right - (right - left) // 3
        
        if arr[mid1] == target:
            return mid1
        if arr[mid2] == target:
            return mid2
        
        if target < arr[mid1]:
            # Target is in the first third
            return recursive_ternary_search(arr, target, left, mid1 - 1)
        elif target > arr[mid2]:
            # Target is in the last third
            return recursive_ternary_search(arr, target, mid2 + 1, right)
        else:
            # Target is in the middle third
            return recursive_ternary_search(arr, target, mid1 + 1, mid2 - 1)
    
    return -1  # Target not found
```

## Interview Tips

- Explain why ternary search makes more comparisons than binary search despite having better asymptotic complexity (log₃ n vs log₂ n)
- Discuss how to optimize the algorithm by adjusting the division points
- Highlight applications where ternary search is preferred over binary search
- Be prepared to implement the algorithm for both discrete arrays and continuous functions
- Discuss the difference between finding a specific value and finding the maximum/minimum of a function

## Practice Problems

1. Find the maximum value in a bitonic array using ternary search
2. Implement ternary search for a sorted array with duplicates, returning the first occurrence
3. Use ternary search to find the minimum distance between a point and a parabola
4. Compare the performance of binary search and ternary search for different array sizes
5. Apply ternary search to find the optimal parameter in a machine learning model

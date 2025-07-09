# Binary Search

## Overview

Binary Search is a fundamental divide and conquer algorithm that efficiently finds the position of a target value within a sorted array. It works by repeatedly dividing the search interval in half, eliminating half of the remaining elements at each step.

## Algorithm

1. Compare the target value with the middle element of the array
2. If the target matches the middle element, return the middle index
3. If the target is less than the middle element, continue the search in the left half
4. If the target is greater than the middle element, continue the search in the right half
5. Repeat until the target is found or the search interval is empty

## Implementation

### Python Implementation

```python
def binary_search(arr, target):
    """
    Performs binary search to find target in a sorted array.
    
    Args:
        arr: A sorted array
        target: The value to search for
        
    Returns:
        Index of target if found, otherwise -1
    """
    left = 0
    right = len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2  # Avoid potential overflow
        
        # Check if target is at mid
        if arr[mid] == target:
            return mid
        
        # If target is greater, ignore left half
        elif arr[mid] < target:
            left = mid + 1
        
        # If target is smaller, ignore right half
        else:
            right = mid - 1
    
    # Target is not in the array
    return -1

# Recursive implementation
def binary_search_recursive(arr, target, left=None, right=None):
    """
    Recursively performs binary search to find target in a sorted array.
    
    Args:
        arr: A sorted array
        target: The value to search for
        left: Left boundary of the search
        right: Right boundary of the search
        
    Returns:
        Index of target if found, otherwise -1
    """
    # Initialize left and right for the first call
    if left is None:
        left = 0
    if right is None:
        right = len(arr) - 1
    
    # Base case: element not found
    if left > right:
        return -1
    
    # Find middle element
    mid = left + (right - left) // 2
    
    # If element is at mid
    if arr[mid] == target:
        return mid
    
    # If element is smaller than mid, search in left subarray
    if arr[mid] > target:
        return binary_search_recursive(arr, target, left, mid - 1)
    
    # Else, search in right subarray
    return binary_search_recursive(arr, target, mid + 1, right)

# Example usage
arr = [2, 3, 4, 10, 40, 50, 60, 70]
target = 10

result = binary_search(arr, target)
print(f"Iterative Binary Search: Element {target} found at index {result}")

result = binary_search_recursive(arr, target)
print(f"Recursive Binary Search: Element {target} found at index {result}")

# Test with element not in array
target = 30
result = binary_search(arr, target)
print(f"Iterative Binary Search: Element {target} found at index {result}")
```

### Java Implementation

```java
public class BinarySearch {
    
    // Iterative implementation
    public static int binarySearch(int[] arr, int target) {
        int left = 0;
        int right = arr.length - 1;
        
        while (left <= right) {
            // Find middle element (avoids integer overflow)
            int mid = left + (right - left) / 2;
            
            // Check if target is at mid
            if (arr[mid] == target) {
                return mid;
            }
            
            // If target is greater, ignore left half
            if (arr[mid] < target) {
                left = mid + 1;
            }
            // If target is smaller, ignore right half
            else {
                right = mid - 1;
            }
        }
        
        // Target is not in the array
        return -1;
    }
    
    // Recursive implementation
    public static int binarySearchRecursive(int[] arr, int target) {
        return binarySearchRecursive(arr, target, 0, arr.length - 1);
    }
    
    private static int binarySearchRecursive(int[] arr, int target, int left, int right) {
        // Base case: element not found
        if (left > right) {
            return -1;
        }
        
        // Find middle element
        int mid = left + (right - left) / 2;
        
        // If element is at mid
        if (arr[mid] == target) {
            return mid;
        }
        
        // If element is smaller than mid, search in left subarray
        if (arr[mid] > target) {
            return binarySearchRecursive(arr, target, left, mid - 1);
        }
        
        // Else, search in right subarray
        return binarySearchRecursive(arr, target, mid + 1, right);
    }
    
    public static void main(String[] args) {
        int[] arr = {2, 3, 4, 10, 40, 50, 60, 70};
        int target = 10;
        
        int result = binarySearch(arr, target);
        System.out.println("Iterative Binary Search: Element " + target + 
                           " found at index " + result);
        
        result = binarySearchRecursive(arr, target);
        System.out.println("Recursive Binary Search: Element " + target + 
                           " found at index " + result);
        
        // Test with element not in array
        target = 30;
        result = binarySearch(arr, target);
        System.out.println("Iterative Binary Search: Element " + target + 
                           " found at index " + result);
    }
}
```

## Complexity Analysis

- **Time Complexity**: O(log n) - The search space is halved at each step
- **Space Complexity**:
  - **Iterative Implementation**: O(1) - Constant extra space
  - **Recursive Implementation**: O(log n) - Due to the recursion stack

## Divide and Conquer Analysis

Binary Search follows the divide and conquer paradigm:
1. **Divide**: Compare the target with the middle element to determine which half to search
2. **Conquer**: Recursively search the appropriate half
3. **Combine**: Return the index of the target if found, or -1 otherwise

## Variations

1. **Lower Bound**: Find the first element greater than or equal to the target
2. **Upper Bound**: Find the first element strictly greater than the target
3. **Nearest Element**: Find the element closest to the target
4. **Binary Search on Answer**: Using binary search to find the answer to an optimization problem
5. **Search in Rotated Sorted Array**: Binary search modified to work on rotated arrays
6. **Search in 2D Sorted Arrays**: Binary search applied to matrices sorted in both dimensions

## Common Pitfalls

1. **Infinite Loops**: Incorrect termination conditions can lead to infinite loops
2. **Off-by-One Errors**: Incorrect handling of array boundaries
3. **Integer Overflow**: Using `(left + right) / 2` can cause overflow for large arrays
4. **Non-sorted Arrays**: Binary search only works on sorted arrays
5. **Duplicates**: Standard binary search finds any matching element, not necessarily the first or last

## Applications

1. **Searching in databases**: Finding records based on indexed keys
2. **Library search systems**: Locating books by catalog number
3. **Dictionary lookup**: Finding words in a dictionary
4. **Version control systems**: Finding changes in specific revisions
5. **Optimization problems**: Using binary search to find optimal values
6. **Computer graphics**: Intersection detection in ray tracing

## Practice Problems

1. [Binary Search](https://leetcode.com/problems/binary-search/) - Standard implementation
2. [First Bad Version](https://leetcode.com/problems/first-bad-version/) - Find the first occurrence
3. [Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/) - Modified binary search
4. [Find Peak Element](https://leetcode.com/problems/find-peak-element/) - Using binary search on unsorted array
5. [Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/) - Binary search on a 2D matrix
6. [Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/) - Advanced binary search application

## References

1. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). Introduction to Algorithms (3rd ed.). MIT Press.
2. Sedgewick, R., & Wayne, K. (2011). Algorithms (4th ed.). Addison-Wesley Professional.
3. Knuth, D. E. (1998). The Art of Computer Programming, Volume 3: Sorting and Searching (2nd ed.). Addison-Wesley Professional.

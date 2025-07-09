# Fibonacci Search

## Overview

Fibonacci Search is a divide-and-conquer algorithm that uses the Fibonacci numbers to divide the search space. Similar to binary search, it works on a sorted array but has the advantage of using only addition and subtraction operations, making it useful for systems where multiplication and division are expensive operations.

## Algorithm

1. Find the smallest Fibonacci number greater than or equal to the array length
2. Use two consecutive Fibonacci numbers to determine the partition points
3. Compare the target with the element at the partition point and adjust the search range
4. Reduce the Fibonacci numbers according to the direction of search

```python
def fibonacci_search(arr, target):
    n = len(arr)
    
    # Initialize Fibonacci numbers
    fib2 = 0  # (m-2)'th Fibonacci number
    fib1 = 1  # (m-1)'th Fibonacci number
    fib = fib1 + fib2  # m'th Fibonacci number
    
    # Find the smallest Fibonacci number >= n
    while fib < n:
        fib2 = fib1
        fib1 = fib
        fib = fib1 + fib2
    
    # Marks the eliminated range from front
    offset = -1
    
    # While there are elements to be inspected
    while fib > 1:
        # Check if fib2 is a valid index
        i = min(offset + fib2, n - 1)
        
        # If target is greater than the element at index i,
        # cut the subarray from offset to i
        if arr[i] < target:
            fib = fib1
            fib1 = fib2
            fib2 = fib - fib1
            offset = i
        
        # If target is less than the element at index i,
        # cut the subarray after i+1
        elif arr[i] > target:
            fib = fib2
            fib1 = fib1 - fib2
            fib2 = fib - fib1
        
        # Element found
        else:
            return i
    
    # Compare the last element
    if fib1 and arr[offset + 1] == target:
        return offset + 1
    
    # Element not found
    return -1
```

## Time and Space Complexity

- **Time Complexity**: O(log n) - where n is the length of the array
- **Space Complexity**: O(1) - Only a constant amount of extra space is used

## Advantages and Disadvantages

### Advantages

- Uses only addition and subtraction operations (no multiplication or division)
- Well-suited for searching on systems with limited arithmetic capabilities
- Divides the array in a more non-uniform way, which can be beneficial for certain data distributions
- The divisions are close to the golden ratio (1.618...), which is considered optimal for many problems

### Disadvantages

- More complex to implement than binary search
- Generally slower than binary search on modern hardware where multiplication/division is not a bottleneck
- Less intuitive and less commonly used compared to binary search
- Requires managing three Fibonacci numbers throughout the algorithm

## Use Cases

- Systems with limited arithmetic capabilities (e.g., embedded systems)
- When multiplication and division operations are expensive
- As an alternative to binary search when non-uniform partitioning is beneficial
- In magnetic tape or similar sequential storage devices where accessing arbitrary positions is costly

## Implementation Details

### Python Implementation

```python
def fibonacci_search(arr, target):
    n = len(arr)
    
    if n == 0:
        return -1
    
    # Initialize Fibonacci numbers
    fib2 = 0  # (m-2)'th Fibonacci number
    fib1 = 1  # (m-1)'th Fibonacci number
    fib = fib1 + fib2  # m'th Fibonacci number
    
    # Find the smallest Fibonacci number >= n
    while fib < n:
        fib2 = fib1
        fib1 = fib
        fib = fib1 + fib2
    
    # Marks the eliminated range from front
    offset = -1
    
    # While there are elements to be inspected
    while fib > 1:
        # Check if fib2 is a valid index
        i = min(offset + fib2, n - 1)
        
        # If target is greater than the element at index i,
        # cut the subarray from offset to i
        if arr[i] < target:
            fib = fib1
            fib1 = fib2
            fib2 = fib - fib1
            offset = i
        
        # If target is less than the element at index i,
        # cut the subarray after i+1
        elif arr[i] > target:
            fib = fib2
            fib1 = fib1 - fib2
            fib2 = fib - fib1
        
        # Element found
        else:
            return i
    
    # Compare the last element
    if fib1 and offset + 1 < n and arr[offset + 1] == target:
        return offset + 1
    
    # Element not found
    return -1

# Example usage
array = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
result = fibonacci_search(array, 35)
print(f"Element found at index: {result}")  # Output: Element found at index: 6
```

### Java Implementation

```java
public class FibonacciSearch {
    public static int fibonacciSearch(int[] arr, int target) {
        int n = arr.length;
        
        if (n == 0)
            return -1;
        
        // Initialize Fibonacci numbers
        int fib2 = 0;  // (m-2)'th Fibonacci number
        int fib1 = 1;  // (m-1)'th Fibonacci number
        int fib = fib1 + fib2;  // m'th Fibonacci number
        
        // Find the smallest Fibonacci number >= n
        while (fib < n) {
            fib2 = fib1;
            fib1 = fib;
            fib = fib1 + fib2;
        }
        
        // Marks the eliminated range from front
        int offset = -1;
        
        // While there are elements to be inspected
        while (fib > 1) {
            // Check if fib2 is a valid index
            int i = Math.min(offset + fib2, n - 1);
            
            // If target is greater than the element at index i,
            // cut the subarray from offset to i
            if (arr[i] < target) {
                fib = fib1;
                fib1 = fib2;
                fib2 = fib - fib1;
                offset = i;
            }
            // If target is less than the element at index i,
            // cut the subarray after i+1
            else if (arr[i] > target) {
                fib = fib2;
                fib1 = fib1 - fib2;
                fib2 = fib - fib1;
            }
            // Element found
            else {
                return i;
            }
        }
        
        // Compare the last element
        if (fib1 == 1 && offset + 1 < n && arr[offset + 1] == target) {
            return offset + 1;
        }
        
        // Element not found
        return -1;
    }
    
    public static void main(String[] args) {
        int[] array = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60};
        int result = fibonacciSearch(array, 35);
        System.out.println("Element found at index: " + result);  // Output: Element found at index: 6
    }
}
```

## Variations

### Fibonacci Search with First Occurrence

```python
def fibonacci_search_first_occurrence(arr, target):
    n = len(arr)
    
    if n == 0:
        return -1
    
    # Initialize Fibonacci numbers
    fib2 = 0
    fib1 = 1
    fib = fib1 + fib2
    
    # Find the smallest Fibonacci number >= n
    while fib < n:
        fib2 = fib1
        fib1 = fib
        fib = fib1 + fib2
    
    offset = -1
    result = -1
    
    while fib > 1:
        i = min(offset + fib2, n - 1)
        
        if arr[i] < target:
            fib = fib1
            fib1 = fib2
            fib2 = fib - fib1
            offset = i
        elif arr[i] > target:
            fib = fib2
            fib1 = fib1 - fib2
            fib2 = fib - fib1
        else:
            # Element found, but continue searching for first occurrence
            result = i
            fib = fib2
            fib1 = fib1 - fib2
            fib2 = fib - fib1
    
    # Check if we found any occurrence
    return result
```

### Recursive Fibonacci Search

```python
def recursive_fibonacci_search(arr, target, offset, fib2, fib1, fib):
    if fib <= 1:
        if fib1 and offset + 1 < len(arr) and arr[offset + 1] == target:
            return offset + 1
        return -1
    
    i = min(offset + fib2, len(arr) - 1)
    
    if arr[i] < target:
        return recursive_fibonacci_search(arr, target, i, fib1 - fib2, fib2, fib1)
    elif arr[i] > target:
        return recursive_fibonacci_search(arr, target, offset, fib2 - fib1, fib - fib1, fib2)
    else:
        return i

# Helper function to set up the initial call
def fib_search_recursive(arr, target):
    n = len(arr)
    
    if n == 0:
        return -1
    
    fib2, fib1, fib = 0, 1, 1
    while fib < n:
        fib2 = fib1
        fib1 = fib
        fib = fib1 + fib2
    
    return recursive_fibonacci_search(arr, target, -1, fib2, fib1, fib)
```

## Interview Tips

- Explain the relationship between Fibonacci search and the golden ratio (approximately 1.618)
- Compare with binary search, highlighting when Fibonacci search might be preferred
- Emphasize its advantages in systems with limited arithmetic capabilities
- Discuss the interesting property of Fibonacci numbers and how they naturally divide the search space
- Be prepared to analyze why the partitioning in Fibonacci search can be more efficient than binary search in certain scenarios

## Practice Problems

1. Implement Fibonacci search to find both the first and last occurrences of an element
2. Use Fibonacci search to find the peak element in a bitonic array
3. Apply Fibonacci search to find a target in a sorted rotated array
4. Implement a hybrid search algorithm that chooses between binary search and Fibonacci search based on array characteristics
5. Analyze and compare the average number of comparisons between binary search and Fibonacci search

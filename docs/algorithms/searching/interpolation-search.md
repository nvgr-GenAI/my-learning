# Interpolation Search

## Overview

Interpolation Search is an improved variant of binary search for uniformly distributed sorted arrays. Instead of always checking the middle element, it makes an educated guess about the probable position of the target value based on its value.

## Algorithm

The algorithm uses a formula to estimate the position of the element:

```
pos = low + ((target - arr[low]) * (high - low)) / (arr[high] - arr[low])
```

This formula is based on linear interpolation, similar to the way we might search for a name in a telephone directory.

```python
def interpolation_search(arr, target):
    low = 0
    high = len(arr) - 1
    
    while low <= high and arr[low] <= target <= arr[high]:
        if low == high:
            if arr[low] == target:
                return low
            return -1
        
        # Formula for position estimation
        pos = low + ((target - arr[low]) * (high - low)) // (arr[high] - arr[low])
        
        if arr[pos] == target:
            return pos
        
        if arr[pos] < target:
            low = pos + 1
        else:
            high = pos - 1
    
    return -1
```

## Time and Space Complexity

- **Time Complexity**: O(log log n) for uniformly distributed data, but O(n) in the worst case
- **Space Complexity**: O(1) - Only a constant amount of extra space is used

## Advantages and Disadvantages

### Advantages

- Much faster than binary search for uniformly distributed sorted arrays
- Can outperform most searching algorithms in specific scenarios
- Works well when values are uniformly distributed

### Disadvantages

- Worst-case time complexity is O(n), which is worse than binary search
- Not as effective when the data is not uniformly distributed
- The position calculation can be complex and might lead to arithmetic errors in certain implementations

## Use Cases

- Searching in large uniformly distributed sorted arrays
- When access time is very high (like remote databases)
- When binary search is not fast enough for the specific data distribution
- Dictionaries, phone books, or any alphabetically ordered list with uniform distribution

## Implementation Details

### Python Implementation

```python
def interpolation_search(arr, target):
    low = 0
    high = len(arr) - 1
    
    while low <= high and arr[low] <= target <= arr[high]:
        if low == high:
            if arr[low] == target:
                return low
            return -1
        
        # Prevent division by zero
        if arr[high] == arr[low]:
            if arr[low] == target:
                return low
            return -1
        
        # Formula for position estimation
        pos = low + int(((target - arr[low]) * (high - low)) / (arr[high] - arr[low]))
        
        # If the position is out of range (can happen due to floating-point errors)
        if pos < low or pos > high:
            break
        
        if arr[pos] == target:
            return pos
        
        if arr[pos] < target:
            low = pos + 1
        else:
            high = pos - 1
    
    return -1

# Example usage
array = [10, 12, 13, 16, 18, 19, 20, 21, 22, 23, 24, 33, 35, 42, 47]
result = interpolation_search(array, 18)
print(f"Element found at index: {result}")  # Output: Element found at index: 4
```

### Java Implementation

```java
public class InterpolationSearch {
    public static int interpolationSearch(int[] arr, int target) {
        int low = 0;
        int high = arr.length - 1;
        
        while (low <= high && target >= arr[low] && target <= arr[high]) {
            if (low == high) {
                if (arr[low] == target)
                    return low;
                return -1;
            }
            
            // Prevent division by zero
            if (arr[high] == arr[low]) {
                if (arr[low] == target)
                    return low;
                return -1;
            }
            
            // Position formula
            int pos = low + (((target - arr[low]) * (high - low)) / (arr[high] - arr[low]));
            
            // Check if position is in range
            if (pos < low || pos > high)
                break;
                
            if (arr[pos] == target)
                return pos;
                
            if (arr[pos] < target)
                low = pos + 1;
            else
                high = pos - 1;
        }
        
        return -1;
    }
    
    public static void main(String[] args) {
        int[] array = {10, 12, 13, 16, 18, 19, 20, 21, 22, 23, 24, 33, 35, 42, 47};
        int result = interpolationSearch(array, 18);
        System.out.println("Element found at index: " + result);  // Output: Element found at index: 4
    }
}
```

## Variations

### Recursive Interpolation Search

```python
def recursive_interpolation_search(arr, target, low, high):
    if low <= high and arr[low] <= target <= arr[high]:
        if low == high:
            if arr[low] == target:
                return low
            return -1
            
        # Prevent division by zero
        if arr[high] == arr[low]:
            if arr[low] == target:
                return low
            return -1
            
        # Position formula
        pos = low + int(((target - arr[low]) * (high - low)) / (arr[high] - arr[low]))
        
        # Check if position is in range
        if pos < low or pos > high:
            return -1
            
        if arr[pos] == target:
            return pos
            
        if arr[pos] < target:
            return recursive_interpolation_search(arr, target, pos + 1, high)
        else:
            return recursive_interpolation_search(arr, target, low, pos - 1)
    
    return -1

# Usage: recursive_interpolation_search(array, target, 0, len(array) - 1)
```

### Interpolation Search with Exponential Range Check

```python
def interpolation_search_with_range_check(arr, target):
    low = 0
    high = 1  # Start with a small range
    n = len(arr)
    
    # Find the range where target might be
    while high < n and arr[high] < target:
        low = high
        high = min(high * 2, n - 1)
    
    # Now apply interpolation search in this range
    return interpolation_search_on_range(arr, target, low, high)

def interpolation_search_on_range(arr, target, low, high):
    # Standard interpolation search implementation
    # ...
```

## Interview Tips

- Explain how interpolation search differs from binary search (position calculation vs middle element)
- Mention the best, average, and worst-case time complexities and when each occurs
- Discuss the importance of uniform data distribution for optimal performance
- Explain how to handle edge cases like division by zero or out-of-range position calculations
- Compare with binary search and explain when to use each

## Practice Problems

1. Implement interpolation search for a sorted array with duplicates, returning the first occurrence
2. Apply interpolation search to find a target in a sorted array with non-uniform distribution
3. Analyze the performance of interpolation search vs binary search for different data distributions
4. Implement a hybrid algorithm that switches between binary and interpolation search based on data characteristics
5. Use interpolation search to find the square root of a number with a given precision

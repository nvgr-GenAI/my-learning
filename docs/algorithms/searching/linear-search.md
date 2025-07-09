# Linear Search

## Overview

Linear search is the simplest searching algorithm that checks each element in the collection one by one until the target element is found or the whole collection has been searched.

## Algorithm

```python
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i  # Return the index of the target
    return -1  # Target not found
```

## Time and Space Complexity

- **Time Complexity**: O(n) - In the worst case, we need to check all elements
- **Space Complexity**: O(1) - Only a constant amount of extra space is used

## Advantages and Disadvantages

### Advantages
- Simple to implement and understand
- Works on both sorted and unsorted collections
- Does not require the data to be arranged in any particular way

### Disadvantages
- Inefficient for large datasets compared to other search algorithms
- Linear time complexity is not ideal when faster alternatives are available for sorted data

## Use Cases

- Small datasets where setup costs of more complex algorithms outweigh benefits
- Unsorted collections where sorting would be more expensive
- When searching for multiple occurrences of an element
- When the collection is expected to have the target near the beginning

## Implementation Details

### Python Implementation

```python
def linear_search(arr, target):
    for i, element in enumerate(arr):
        if element == target:
            return i
    return -1

# Example usage
array = [10, 24, 35, 42, 57, 68, 79, 81, 95]
result = linear_search(array, 57)
print(f"Element found at index: {result}")  # Output: Element found at index: 4
```

### Java Implementation

```java
public class LinearSearch {
    public static int linearSearch(int[] arr, int target) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == target) {
                return i;
            }
        }
        return -1;
    }
    
    public static void main(String[] args) {
        int[] array = {10, 24, 35, 42, 57, 68, 79, 81, 95};
        int result = linearSearch(array, 57);
        System.out.println("Element found at index: " + result);  // Output: Element found at index: 4
    }
}
```

## Variations

### Sentinel Linear Search

Adding a sentinel at the end of the array can eliminate the need for checking the array bounds in each iteration.

```python
def sentinel_linear_search(arr, target):
    n = len(arr)
    
    # Add sentinel at the end
    last = arr[n-1]
    arr[n-1] = target
    
    i = 0
    while arr[i] != target:
        i += 1
    
    # Restore the array
    arr[n-1] = last
    
    if i < n-1 or arr[n-1] == target:
        return i
    else:
        return -1
```

### Improved Linear Search

For better cache performance, we can search from both ends simultaneously:

```python
def improved_linear_search(arr, target):
    n = len(arr)
    left = 0
    right = n - 1
    
    while left <= right:
        if arr[left] == target:
            return left
        if arr[right] == target:
            return right
        left += 1
        right -= 1
    
    return -1
```

## Interview Tips

- Mention the simplicity but also the inefficiency for large datasets
- Discuss when linear search might be preferable to binary search
- Be ready to implement variations like sentinel search or bidirectional search
- Consider mentioning its real-world applications like searching in linked lists

## Practice Problems

1. Find the first occurrence of a number in an array
2. Count occurrences of a specific element in an array
3. Find the peak element in an array
4. Search in a rotated sorted array where binary search would be complex
5. Find missing numbers in an array with O(n) time complexity

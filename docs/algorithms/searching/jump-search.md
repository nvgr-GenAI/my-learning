# Jump Search

## Overview

Jump Search is a searching algorithm for sorted arrays that works by jumping ahead by fixed steps and then performing a linear search once we find an interval where the target might exist.

## Algorithm

1. Calculate the jump size (usually sqrt(n) where n is array length)
2. Jump ahead by the jump size until we find a value greater than the target
3. Perform a linear search within the found block

```python
import math

def jump_search(arr, target):
    n = len(arr)
    step = int(math.sqrt(n))
    
    # Finding the block where the target may be present
    prev = 0
    while arr[min(step, n) - 1] < target:
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:
            return -1
    
    # Linear search within the identified block
    while arr[prev] < target:
        prev += 1
        if prev == min(step, n):
            return -1
    
    if arr[prev] == target:
        return prev
    
    return -1
```

## Time and Space Complexity

- **Time Complexity**: O(√n) - Where n is the length of the array
- **Space Complexity**: O(1) - Only a constant amount of extra space is used

## Advantages and Disadvantages

### Advantages

- Faster than linear search (O(√n) vs O(n))
- Simpler than binary search
- Works well for arrays that are stored on external storage (like magnetic tapes)

### Disadvantages

- Slower than binary search (O(√n) vs O(log n))
- Only works on sorted arrays
- Less efficient than binary search for random access data structures

## Use Cases

- When the array is sorted
- When binary search is too complex to implement
- When jumping backward is more expensive than sequential access (like on magnetic tapes)

## Implementation Details

### Python Implementation

```python
import math

def jump_search(arr, target):
    n = len(arr)
    step = int(math.sqrt(n))
    
    # Finding the block
    prev = 0
    while prev < n and arr[min(step, n) - 1] < target:
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:
            return -1
    
    # Linear search in the block
    while prev < n and arr[prev] < target:
        prev += 1
        if prev == min(step, n) or prev == n:
            return -1
    
    if prev < n and arr[prev] == target:
        return prev
    
    return -1

# Example usage
array = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
result = jump_search(array, 55)
print(f"Element found at index: {result}")  # Output: Element found at index: 10
```

### Java Implementation

```java
public class JumpSearch {
    public static int jumpSearch(int[] arr, int target) {
        int n = arr.length;
        int step = (int)Math.floor(Math.sqrt(n));
        
        // Finding the block
        int prev = 0;
        while (prev < n && arr[Math.min(step, n) - 1] < target) {
            prev = step;
            step += (int)Math.floor(Math.sqrt(n));
            if (prev >= n)
                return -1;
        }
        
        // Linear search in the block
        while (prev < n && arr[prev] < target) {
            prev++;
            if (prev == Math.min(step, n) || prev == n)
                return -1;
        }
        
        if (prev < n && arr[prev] == target)
            return prev;
        
        return -1;
    }
    
    public static void main(String[] args) {
        int[] array = {0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610};
        int result = jumpSearch(array, 55);
        System.out.println("Element found at index: " + result);  // Output: Element found at index: 10
    }
}
```

## Variations

### Adaptive Jump Size

Instead of using sqrt(n) as the jump size, we can adapt the jump size based on the size of the array or the distribution of the data:

```python
def adaptive_jump_search(arr, target):
    n = len(arr)
    # Adaptive jump size - can be adjusted based on array characteristics
    step = int(n ** (1/3))  # Using cube root instead of square root
    
    # Rest of the algorithm remains the same
    # ...
```

### Bidirectional Jump Search

Jump search combined with bidirectional search can be more efficient for certain distributions:

```python
def bidirectional_jump_search(arr, target):
    n = len(arr)
    step = int(math.sqrt(n))
    
    # Start from both ends
    left = 0
    right = n - 1
    
    # Jump from both ends
    while left <= right:
        if arr[left] == target:
            return left
        if arr[right] == target:
            return right
            
        left_jump = min(left + step, right)
        right_jump = max(right - step, left)
        
        if left_jump < n and arr[left_jump] >= target:
            # Linear search between left and left_jump
            for i in range(left + 1, left_jump):
                if arr[i] == target:
                    return i
        
        if right_jump >= 0 and arr[right_jump] <= target:
            # Linear search between right_jump and right
            for i in range(right_jump + 1, right):
                if arr[i] == target:
                    return i
                
        left = left_jump + 1
        right = right_jump - 1
    
    return -1
```

## Interview Tips

- Mention that jump search is a middle ground between linear search and binary search
- Explain why sqrt(n) is the optimal jump size (balances the number of jumps and linear search steps)
- Discuss when jump search might be preferable over binary search (sequential access media)
- Be prepared to analyze and optimize the jump step size for different scenarios

## Practice Problems

1. Find the first occurrence of a number in a sorted array using jump search
2. Implement jump search for a circular sorted array
3. Use jump search to find the peak element in a bitonic array
4. Analyze and compare the performance of jump search vs binary search for different array sizes
5. Optimize the jump size for a specific distribution of data

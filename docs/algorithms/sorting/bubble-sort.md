# Bubble Sort

Bubble Sort is one of the simplest sorting algorithms. It repeatedly steps through the list, compares adjacent elements and swaps them if they're in the wrong order. The pass through the list is repeated until the list is sorted.

## Algorithm Overview

- **Time Complexity**: O(n²) average and worst case, O(n) best case
- **Space Complexity**: O(1)
- **Stable**: Yes
- **In-place**: Yes
- **Adaptive**: Yes (optimized version)

## How It Works

1. Compare adjacent elements
2. Swap if they're in wrong order
3. Continue through the entire array
4. Repeat until no swaps are needed
5. Optimization: Stop early if no swaps occur in a pass

## Implementation

```python
def bubble_sort(arr):
    """
    Bubble Sort implementation
    Time: O(n²), Space: O(1)
    Stable, in-place, adaptive
    """
    n = len(arr)
    
    for i in range(n):
        # Flag to optimize for already sorted arrays
        swapped = False
        
        # Last i elements are already sorted
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        
        # If no swapping occurred, array is sorted
        if not swapped:
            break
    
    return arr

def bubble_sort_basic(arr):
    """
    Basic bubble sort without optimization
    Always performs O(n²) comparisons
    """
    n = len(arr)
    
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    
    return arr

# Example usage
if __name__ == "__main__":
    # Test with random array
    import random
    
    test_arr = [64, 34, 25, 12, 22, 11, 90]
    print("Original array:", test_arr)
    
    sorted_arr = bubble_sort(test_arr.copy())
    print("Sorted array:", sorted_arr)
    
    # Performance test
    import time
    
    # Test with larger array
    large_arr = [random.randint(1, 1000) for _ in range(1000)]
    
    start_time = time.time()
    bubble_sort(large_arr.copy())
    end_time = time.time()
    
    print(f"Time taken for 1000 elements: {end_time - start_time:.4f} seconds")
```

## Visualization

```text
Initial: [64, 34, 25, 12, 22, 11, 90]

Pass 1:
[64, 34, 25, 12, 22, 11, 90] → [34, 64, 25, 12, 22, 11, 90]
[34, 64, 25, 12, 22, 11, 90] → [34, 25, 64, 12, 22, 11, 90]
[34, 25, 64, 12, 22, 11, 90] → [34, 25, 12, 64, 22, 11, 90]
[34, 25, 12, 64, 22, 11, 90] → [34, 25, 12, 22, 64, 11, 90]
[34, 25, 12, 22, 64, 11, 90] → [34, 25, 12, 22, 11, 64, 90]
[34, 25, 12, 22, 11, 64, 90] → [34, 25, 12, 22, 11, 64, 90] (no swap)

After Pass 1: [34, 25, 12, 22, 11, 64, 90] (90 is in correct position)

Pass 2:
[34, 25, 12, 22, 11, 64, 90] → [25, 34, 12, 22, 11, 64, 90]
[25, 34, 12, 22, 11, 64, 90] → [25, 12, 34, 22, 11, 64, 90]
[25, 12, 34, 22, 11, 64, 90] → [25, 12, 22, 34, 11, 64, 90]
[25, 12, 22, 34, 11, 64, 90] → [25, 12, 22, 11, 34, 64, 90]
[25, 12, 22, 11, 34, 64, 90] → [25, 12, 22, 11, 34, 64, 90] (no swap)

After Pass 2: [25, 12, 22, 11, 34, 64, 90] (64 is in correct position)

... (continues until fully sorted)
```

## When to Use

**Good for:**

- Educational purposes (easy to understand)
- Very small datasets
- Nearly sorted data (with optimization)
- When simplicity is more important than efficiency

**Not good for:**

- Large datasets
- Performance-critical applications
- Production systems

## Advantages

- Simple to understand and implement
- No additional memory space needed
- Stable sorting algorithm
- Can detect if list is already sorted (optimized version)
- Works well on small datasets

## Disadvantages

- Poor time complexity O(n²)
- More comparisons and swaps compared to other algorithms
- Not suitable for large datasets
- Generally slower than other O(n²) algorithms like insertion sort

## Variants

### Cocktail Shaker Sort (Bidirectional Bubble Sort)

```python
def cocktail_shaker_sort(arr):
    """
    Cocktail shaker sort - bubble sort that works in both directions
    Slightly better performance than regular bubble sort
    """
    n = len(arr)
    start = 0
    end = n - 1
    swapped = True
    
    while swapped:
        swapped = False
        
        # Forward pass
        for i in range(start, end):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True
        
        if not swapped:
            break
        
        end -= 1
        swapped = False
        
        # Backward pass
        for i in range(end - 1, start - 1, -1):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True
        
        start += 1
    
    return arr
```

## Practice Problems

1. **Bubble Sort Count**: Count the number of swaps needed to sort an array
2. **Bubble Sort Visualization**: Implement a step-by-step visualization
3. **Optimized Bubble Sort**: Implement various optimizations
4. **Bubble Sort Comparison**: Compare performance with other sorting algorithms

---

*Bubble sort may not be the most efficient, but it's an excellent starting point to understand sorting algorithms!*

# Selection Sort

Selection Sort divides the array into sorted and unsorted regions. It repeatedly finds the minimum element from the unsorted region and moves it to the sorted region.

## Algorithm Overview

- **Time Complexity**: O(n²) for all cases
- **Space Complexity**: O(1)
- **Stable**: No (can be made stable with modifications)
- **In-place**: Yes
- **Adaptive**: No

## How It Works

1. Find the minimum element in the unsorted array
2. Swap it with the first element of unsorted array
3. Move the boundary between sorted and unsorted regions one position right
4. Repeat until the entire array is sorted

## Implementation

```python
def selection_sort(arr):
    """
    Selection Sort implementation
    Time: O(n²), Space: O(1)
    Not stable, in-place, not adaptive
    """
    n = len(arr)
    
    for i in range(n):
        # Find minimum element in remaining unsorted array
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        
        # Swap the found minimum element with first element
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    
    return arr

def selection_sort_stable(arr):
    """
    Stable version of selection sort
    Instead of swapping, we shift elements
    """
    n = len(arr)
    
    for i in range(n):
        # Find minimum element in remaining unsorted array
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        
        # Instead of swapping, shift elements to make it stable
        if min_idx != i:
            min_val = arr[min_idx]
            # Shift elements to the right
            while min_idx > i:
                arr[min_idx] = arr[min_idx - 1]
                min_idx -= 1
            arr[i] = min_val
    
    return arr

def selection_sort_max(arr):
    """
    Selection sort that finds maximum element and places at end
    """
    n = len(arr)
    
    for i in range(n - 1, 0, -1):
        # Find maximum element in unsorted array
        max_idx = 0
        for j in range(1, i + 1):
            if arr[j] > arr[max_idx]:
                max_idx = j
        
        # Swap maximum element with last element of unsorted array
        arr[i], arr[max_idx] = arr[max_idx], arr[i]
    
    return arr

def bidirectional_selection_sort(arr):
    """
    Bidirectional selection sort - finds both min and max in each pass
    Reduces the number of passes by half
    """
    left = 0
    right = len(arr) - 1
    
    while left < right:
        min_idx = left
        max_idx = left
        
        # Find both minimum and maximum in current range
        for i in range(left, right + 1):
            if arr[i] < arr[min_idx]:
                min_idx = i
            if arr[i] > arr[max_idx]:
                max_idx = i
        
        # Place minimum at left position
        arr[left], arr[min_idx] = arr[min_idx], arr[left]
        
        # If maximum was at left position, it's now at min_idx
        if max_idx == left:
            max_idx = min_idx
        
        # Place maximum at right position
        arr[right], arr[max_idx] = arr[max_idx], arr[right]
        
        left += 1
        right -= 1
    
    return arr

# Example usage
if __name__ == "__main__":
    # Test with random array
    import random
    
    test_arr = [64, 34, 25, 12, 22, 11, 90]
    print("Original array:", test_arr)
    
    sorted_arr = selection_sort(test_arr.copy())
    print("Sorted array:", sorted_arr)
    
    # Test stability
    class Item:
        def __init__(self, value, order):
            self.value = value
            self.order = order
        
        def __lt__(self, other):
            return self.value < other.value
        
        def __repr__(self):
            return f"({self.value}, {self.order})"
    
    # Test with duplicate values to check stability
    items = [Item(3, 'a'), Item(1, 'b'), Item(3, 'c'), Item(2, 'd')]
    print("\nStability test:")
    print("Original:", items)
    
    # Regular selection sort (not stable)
    items_copy = items.copy()
    selection_sort(items_copy)
    print("Selection sort:", items_copy)
    
    # Stable selection sort
    items_copy = items.copy()
    selection_sort_stable(items_copy)
    print("Stable selection sort:", items_copy)
```

## Visualization

```text
Initial: [64, 34, 25, 12, 22, 11, 90]

Pass 1: Find minimum in [64, 34, 25, 12, 22, 11, 90]
        Minimum is 11 at index 5
        Swap with index 0: [11, 34, 25, 12, 22, 64, 90]
        Sorted: [11] | Unsorted: [34, 25, 12, 22, 64, 90]

Pass 2: Find minimum in [34, 25, 12, 22, 64, 90]
        Minimum is 12 at index 3
        Swap with index 1: [11, 12, 25, 34, 22, 64, 90]
        Sorted: [11, 12] | Unsorted: [25, 34, 22, 64, 90]

Pass 3: Find minimum in [25, 34, 22, 64, 90]
        Minimum is 22 at index 4
        Swap with index 2: [11, 12, 22, 34, 25, 64, 90]
        Sorted: [11, 12, 22] | Unsorted: [34, 25, 64, 90]

Pass 4: Find minimum in [34, 25, 64, 90]
        Minimum is 25 at index 4
        Swap with index 3: [11, 12, 22, 25, 34, 64, 90]
        Sorted: [11, 12, 22, 25] | Unsorted: [34, 64, 90]

Pass 5: Find minimum in [34, 64, 90]
        Minimum is 34 at index 4 (already in place)
        Sorted: [11, 12, 22, 25, 34] | Unsorted: [64, 90]

Pass 6: Find minimum in [64, 90]
        Minimum is 64 at index 5 (already in place)
        Sorted: [11, 12, 22, 25, 34, 64] | Unsorted: [90]

Final: [11, 12, 22, 25, 34, 64, 90]
```

## When to Use

**Good for:**

- Small datasets
- When memory is limited (in-place sorting)
- When number of swaps should be minimized
- Simple implementation is needed
- Sorting linked lists (when random access is expensive)

**Not good for:**

- Large datasets
- When stability is required
- Real-time applications (not adaptive)
- When there are many duplicate elements

## Advantages

- Simple to understand and implement
- In-place sorting (O(1) extra memory)
- Minimizes the number of swaps (O(n) swaps at most)
- Performance is not affected by initial order of elements
- Works well when cost of swapping is high

## Disadvantages

- O(n²) time complexity for all cases (not adaptive)
- Not stable (relative order of equal elements may change)
- More comparisons than necessary
- Poor performance on large datasets
- Does not perform well on nearly sorted data

## Variants and Optimizations

### Heap Sort Connection

Selection sort can be seen as a simplified version of heap sort:

```python
def heap_selection_sort(arr):
    """
    Selection sort using heap to find minimum efficiently
    This is essentially heap sort but demonstrates the connection
    """
    import heapq
    
    # Create min heap
    heap = arr.copy()
    heapq.heapify(heap)
    
    # Extract minimum elements one by one
    for i in range(len(arr)):
        arr[i] = heapq.heappop(heap)
    
    return arr
```

### Selection Sort for K Smallest Elements

```python
def k_smallest_selection(arr, k):
    """
    Find k smallest elements using selection sort approach
    Only performs k passes instead of n-1
    Time: O(k * n)
    """
    result = []
    arr_copy = arr.copy()
    
    for _ in range(k):
        min_idx = 0
        for j in range(1, len(arr_copy)):
            if arr_copy[j] < arr_copy[min_idx]:
                min_idx = j
        
        result.append(arr_copy[min_idx])
        arr_copy.pop(min_idx)
    
    return result
```

## Comparison with Other Sorting Algorithms

| Algorithm | Best Case | Average Case | Worst Case | Swaps | Stable |
|-----------|-----------|--------------|------------|-------|--------|
| **Selection Sort** | O(n²) | O(n²) | O(n²) | O(n) | ❌ |
| **Insertion Sort** | O(n) | O(n²) | O(n²) | O(n²) | ✅ |
| **Bubble Sort** | O(n) | O(n²) | O(n²) | O(n²) | ✅ |

**Key Differences:**

- **Selection Sort vs Insertion Sort**: Selection sort makes fewer swaps but more comparisons
- **Selection Sort vs Bubble Sort**: Selection sort is generally faster due to fewer swaps
- **Selection Sort vs Heap Sort**: Heap sort is selection sort with efficient minimum finding

## Use Cases in Practice

### Tournament Sort

```python
def tournament_sort(arr):
    """
    Tournament sort - efficient way to implement selection sort idea
    Uses tournament tree to find minimum efficiently
    Time: O(n log n)
    """
    import heapq
    
    # Create tournament tree (min heap)
    heap = [(val, i) for i, val in enumerate(arr)]
    heapq.heapify(heap)
    
    result = []
    while heap:
        val, idx = heapq.heappop(heap)
        result.append(val)
    
    return result
```

### Memory-Constrained Environments

```python
def external_selection_sort(file_chunks):
    """
    Selection sort for external sorting when data doesn't fit in memory
    Selects minimum from each chunk
    """
    result = []
    chunk_indices = [0] * len(file_chunks)
    
    while any(idx < len(chunk) for idx, chunk in zip(chunk_indices, file_chunks)):
        min_val = float('inf')
        min_chunk = -1
        
        # Find minimum among first elements of all chunks
        for i, (chunk, idx) in enumerate(zip(file_chunks, chunk_indices)):
            if idx < len(chunk) and chunk[idx] < min_val:
                min_val = chunk[idx]
                min_chunk = i
        
        if min_chunk != -1:
            result.append(min_val)
            chunk_indices[min_chunk] += 1
    
    return result
```

## Practice Problems

1. **Selection Sort with Custom Comparator**: Sort objects with custom comparison
2. **K Smallest Elements**: Find k smallest elements without fully sorting
3. **Stable Selection Sort**: Implement stable version of selection sort
4. **Bidirectional Selection**: Sort using both minimum and maximum finding

## Interview Questions

1. Why is selection sort not stable by default?
2. When would you choose selection sort over insertion sort?
3. How can you make selection sort stable?
4. What's the minimum number of swaps needed to sort an array?

---

*Selection sort teaches us the fundamental concept of repeatedly selecting the optimal element. While not efficient for large datasets, it forms the basis for more advanced algorithms like heap sort!*

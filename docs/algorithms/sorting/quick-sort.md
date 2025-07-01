# Quick Sort

Quick Sort is a highly efficient divide-and-conquer sorting algorithm that works by selecting a 'pivot' element and partitioning the array around it.

## Algorithm Overview

**Time Complexity:**
- Best/Average Case: O(n log n)
- Worst Case: O(n²)

**Space Complexity:** O(log n) - due to recursion stack

**Properties:**
- ❌ Not stable (relative order of equal elements may change)
- ✅ In-place (requires only O(log n) extra space)
- ❌ Not adaptive (doesn't benefit from partially sorted data)

## How It Works

1. **Choose a pivot** element from the array
2. **Partition** the array so that:
   - Elements smaller than pivot go to the left
   - Elements greater than pivot go to the right
3. **Recursively apply** the same process to the sub-arrays

## Implementation

### Basic Quick Sort

```python
def quick_sort(arr):
    """
    Simple Quick Sort implementation
    Time: O(n log n) average, O(n²) worst
    Space: O(log n) average, O(n) worst
    """
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]  # Choose middle element as pivot
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

# Example usage
data = [3, 6, 8, 10, 1, 2, 1]
print(f"Original: {data}")
print(f"Sorted: {quick_sort(data)}")
```

### In-Place Quick Sort

```python
def quick_sort_inplace(arr, low=0, high=None):
    """
    In-place Quick Sort implementation
    More memory efficient than basic version
    """
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        # Partition and get pivot index
        pivot_index = partition(arr, low, high)
        
        # Recursively sort elements before and after partition
        quick_sort_inplace(arr, low, pivot_index - 1)
        quick_sort_inplace(arr, pivot_index + 1, high)

def partition(arr, low, high):
    """
    Lomuto partition scheme
    Places pivot at correct position and returns its index
    """
    # Choose rightmost element as pivot
    pivot = arr[high]
    
    # Index of smaller element (indicates right position of pivot)
    i = low - 1
    
    for j in range(low, high):
        # If current element is smaller than or equal to pivot
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    # Place pivot at correct position
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# Example usage
data = [10, 7, 8, 9, 1, 5]
print(f"Original: {data}")
quick_sort_inplace(data)
print(f"Sorted: {data}")
```

### Hoare Partition Scheme

```python
def quick_sort_hoare(arr, low=0, high=None):
    """Quick Sort using Hoare partition scheme"""
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        pivot_index = partition_hoare(arr, low, high)
        quick_sort_hoare(arr, low, pivot_index)
        quick_sort_hoare(arr, pivot_index + 1, high)

def partition_hoare(arr, low, high):
    """
    Hoare partition scheme
    More efficient than Lomuto scheme (fewer swaps)
    """
    pivot = arr[low]  # Choose first element as pivot
    i = low - 1
    j = high + 1
    
    while True:
        # Find element on left that should be on right
        i += 1
        while arr[i] < pivot:
            i += 1
        
        # Find element on right that should be on left
        j -= 1
        while arr[j] > pivot:
            j -= 1
        
        # If elements crossed, partitioning is done
        if i >= j:
            return j
        
        # Swap elements
        arr[i], arr[j] = arr[j], arr[i]
```

## Optimization Techniques

### Random Pivot Selection

```python
import random

def quick_sort_randomized(arr, low=0, high=None):
    """
    Randomized Quick Sort to avoid worst-case O(n²) performance
    Expected time: O(n log n) even for sorted arrays
    """
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        # Randomize pivot to improve average performance
        random_index = random.randint(low, high)
        arr[random_index], arr[high] = arr[high], arr[random_index]
        
        pivot_index = partition(arr, low, high)
        quick_sort_randomized(arr, low, pivot_index - 1)
        quick_sort_randomized(arr, pivot_index + 1, high)
```

### Median-of-Three Pivot

```python
def median_of_three(arr, low, high):
    """
    Choose median of first, middle, and last elements as pivot
    Helps avoid worst-case performance on sorted/reverse-sorted arrays
    """
    mid = (low + high) // 2
    
    if arr[mid] < arr[low]:
        arr[low], arr[mid] = arr[mid], arr[low]
    if arr[high] < arr[low]:
        arr[low], arr[high] = arr[high], arr[low]
    if arr[high] < arr[mid]:
        arr[mid], arr[high] = arr[high], arr[mid]
    
    # Place median at end (will be used as pivot)
    arr[mid], arr[high] = arr[high], arr[mid]
    return arr[high]

def quick_sort_median_of_three(arr, low=0, high=None):
    """Quick Sort with median-of-three pivot selection"""
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        median_of_three(arr, low, high)
        pivot_index = partition(arr, low, high)
        quick_sort_median_of_three(arr, low, pivot_index - 1)
        quick_sort_median_of_three(arr, pivot_index + 1, high)
```

### Hybrid Quick Sort (with Insertion Sort)

```python
def quick_sort_hybrid(arr, low=0, high=None, threshold=10):
    """
    Hybrid Quick Sort that switches to Insertion Sort for small subarrays
    More efficient in practice
    """
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        if high - low + 1 < threshold:
            insertion_sort_range(arr, low, high)
        else:
            pivot_index = partition(arr, low, high)
            quick_sort_hybrid(arr, low, pivot_index - 1, threshold)
            quick_sort_hybrid(arr, pivot_index + 1, high, threshold)

def insertion_sort_range(arr, low, high):
    """Insertion sort for a specific range of array"""
    for i in range(low + 1, high + 1):
        key = arr[i]
        j = i - 1
        
        while j >= low and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        
        arr[j + 1] = key
```

## When to Use Quick Sort

### ✅ Good For:
- **General-purpose sorting** - Excellent average performance
- **Large datasets** - O(n log n) average case
- **Memory-constrained environments** - In-place sorting
- **When stability is not required**

### ❌ Avoid When:
- **Stability is required** - Use Merge Sort instead
- **Guaranteed O(n log n) is needed** - Use Heap Sort or Merge Sort
- **Already sorted data** - Without randomization, degrades to O(n²)

## Complexity Analysis

### Time Complexity
- **Best Case: O(n log n)** - When pivot divides array evenly
- **Average Case: O(n log n)** - With good pivot selection
- **Worst Case: O(n²)** - When pivot is always smallest/largest element

### Space Complexity
- **Average: O(log n)** - Due to recursion stack
- **Worst: O(n)** - In case of unbalanced partitions

## Visual Example

```
Initial: [3, 6, 8, 10, 1, 2, 1]
Pivot: 10

After Partition: [3, 6, 8, 1, 2, 1] [10] []
                      ↑              ↑    ↑
                   < pivot        pivot > pivot

Recursively sort left subarray: [3, 6, 8, 1, 2, 1]
Pivot: 1
After Partition: [] [1] [3, 6, 8, 2]

Continue until all subarrays are sorted...
Final: [1, 1, 2, 3, 6, 8, 10]
```

## Interview Questions

1. **Why does Quick Sort have O(n²) worst-case complexity?**
2. **How can you avoid the worst-case scenario?**
3. **What's the difference between Lomuto and Hoare partition schemes?**
4. **When would you choose Quick Sort over Merge Sort?**
5. **Implement Quick Sort to handle duplicate elements efficiently**

---

*Quick Sort is one of the most important algorithms to master - it's fast, elegant, and appears frequently in interviews!*

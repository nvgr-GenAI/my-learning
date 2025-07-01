# Insertion Sort

Insertion Sort builds the final sorted array one item at a time. It's much more efficient than bubble sort for small datasets and is adaptive, stable, and in-place.

## Algorithm Overview

- **Time Complexity**: O(n²) average and worst case, O(n) best case
- **Space Complexity**: O(1)
- **Stable**: Yes
- **In-place**: Yes
- **Adaptive**: Yes

## How It Works

1. Start with the second element (assume first is sorted)
2. Compare current element with previous elements
3. Shift larger elements to the right
4. Insert current element in correct position
5. Repeat for all elements

## Implementation

```python
def insertion_sort(arr):
    """
    Insertion Sort implementation
    Time: O(n²), Space: O(1)
    Stable, in-place, adaptive
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        
        # Move elements greater than key one position ahead
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        
        # Place key in correct position
        arr[j + 1] = key
    
    return arr

def insertion_sort_recursive(arr, n=None):
    """
    Recursive implementation of insertion sort
    """
    if n is None:
        n = len(arr)
    
    # Base case
    if n <= 1:
        return arr
    
    # Sort first n-1 elements
    insertion_sort_recursive(arr, n - 1)
    
    # Insert last element at correct position
    last = arr[n - 1]
    j = n - 2
    
    while j >= 0 and arr[j] > last:
        arr[j + 1] = arr[j]
        j -= 1
    
    arr[j + 1] = last
    return arr

def binary_insertion_sort(arr):
    """
    Insertion sort using binary search to find insertion point
    Reduces comparisons but not shifts
    Time: O(n log n) comparisons + O(n²) shifts
    """
    def binary_search(arr, val, start, end):
        """Find insertion point using binary search"""
        if start == end:
            return start if arr[start] > val else start + 1
        
        if start > end:
            return start
        
        mid = (start + end) // 2
        
        if arr[mid] < val:
            return binary_search(arr, val, mid + 1, end)
        elif arr[mid] > val:
            return binary_search(arr, val, start, mid - 1)
        else:
            return mid
    
    for i in range(1, len(arr)):
        key = arr[i]
        j = binary_search(arr, key, 0, i - 1)
        
        # Shift elements
        arr[j + 1:i + 1] = arr[j:i]
        arr[j] = key
    
    return arr

# Example usage
if __name__ == "__main__":
    # Test with random array
    import random
    
    test_arr = [64, 34, 25, 12, 22, 11, 90]
    print("Original array:", test_arr)
    
    sorted_arr = insertion_sort(test_arr.copy())
    print("Sorted array:", sorted_arr)
    
    # Performance comparison
    import time
    
    # Test with larger array
    large_arr = [random.randint(1, 1000) for _ in range(1000)]
    
    # Regular insertion sort
    start_time = time.time()
    insertion_sort(large_arr.copy())
    end_time = time.time()
    print(f"Regular insertion sort: {end_time - start_time:.4f} seconds")
    
    # Binary insertion sort
    start_time = time.time()
    binary_insertion_sort(large_arr.copy())
    end_time = time.time()
    print(f"Binary insertion sort: {end_time - start_time:.4f} seconds")
```

## Visualization

```text
Initial: [64, 34, 25, 12, 22, 11, 90]

Step 1: Consider 34
[64 | 34, 25, 12, 22, 11, 90]
34 < 64, so shift 64 right and insert 34
[34, 64 | 25, 12, 22, 11, 90]

Step 2: Consider 25
[34, 64 | 25, 12, 22, 11, 90]
25 < 64, shift 64 right
25 < 34, shift 34 right, insert 25
[25, 34, 64 | 12, 22, 11, 90]

Step 3: Consider 12
[25, 34, 64 | 12, 22, 11, 90]
12 < 64, shift 64 right
12 < 34, shift 34 right
12 < 25, shift 25 right, insert 12
[12, 25, 34, 64 | 22, 11, 90]

Step 4: Consider 22
[12, 25, 34, 64 | 22, 11, 90]
22 < 64, shift 64 right
22 < 34, shift 34 right
22 < 25, shift 25 right
22 > 12, insert after 12
[12, 22, 25, 34, 64 | 11, 90]

Step 5: Consider 11
[12, 22, 25, 34, 64 | 11, 90]
11 < all elements, insert at beginning
[11, 12, 22, 25, 34, 64 | 90]

Step 6: Consider 90
[11, 12, 22, 25, 34, 64 | 90]
90 > 64, no shifting needed
[11, 12, 22, 25, 34, 64, 90]
```

## When to Use

**Good for:**

- Small datasets (< 50 elements)
- Nearly sorted data
- Online algorithms (sorting data as it arrives)
- Simple implementation needed
- Stable sorting required

**Not good for:**

- Large datasets
- Worst-case performance critical applications
- When O(n log n) algorithms are available

## Advantages

- Simple implementation
- Efficient for small datasets
- Adaptive (O(n) for nearly sorted data)
- Stable sorting algorithm
- In-place (O(1) extra memory)
- Online (can sort data as it arrives)
- More efficient than bubble sort and selection sort

## Disadvantages

- O(n²) time complexity for average and worst cases
- More writes compared to selection sort
- Not suitable for large datasets
- Performance degrades with reverse-sorted data

## Variants and Optimizations

### Shell Sort (Diminishing Increment Sort)

```python
def shell_sort(arr):
    """
    Shell sort - generalization of insertion sort
    Uses gap sequence to allow exchanges of far apart elements
    """
    n = len(arr)
    gap = n // 2
    
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            
            # Perform insertion sort with gap
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            
            arr[j] = temp
        
        gap //= 2
    
    return arr
```

### Insertion Sort for Linked Lists

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def insertion_sort_list(head):
    """
    Insertion sort for linked lists
    Time: O(n²), Space: O(1)
    """
    if not head or not head.next:
        return head
    
    # Create dummy node for easier insertion
    dummy = ListNode(0)
    current = head
    
    while current:
        # Save next node
        next_node = current.next
        
        # Find insertion point
        prev = dummy
        while prev.next and prev.next.val < current.val:
            prev = prev.next
        
        # Insert current node
        current.next = prev.next
        prev.next = current
        
        # Move to next node
        current = next_node
    
    return dummy.next
```

## Comparison with Other Sorting Algorithms

| Algorithm | Best Case | Average Case | Worst Case | Space | Stable |
|-----------|-----------|--------------|------------|-------|--------|
| **Insertion Sort** | O(n) | O(n²) | O(n²) | O(1) | ✅ |
| **Bubble Sort** | O(n) | O(n²) | O(n²) | O(1) | ✅ |
| **Selection Sort** | O(n²) | O(n²) | O(n²) | O(1) | ❌ |

**Why Insertion Sort is better than Bubble Sort:**

- Fewer number of swaps/writes
- Better performance on nearly sorted data
- More intuitive and practical approach

## Practice Problems

1. **Sort Colors**: Sort array of 0s, 1s, and 2s using insertion sort
2. **Insertion Sort List**: Implement insertion sort for linked lists
3. **Sort Nearly Sorted Array**: Optimize for arrays where elements are at most k positions away
4. **Counting Inversions**: Count number of inversions while sorting

## Real-world Applications

- **Small subarrays in hybrid algorithms**: Used in Quicksort and Mergesort for small subarrays
- **Online algorithms**: Sorting data as it arrives
- **Embedded systems**: When memory is limited
- **Card sorting**: Natural way humans sort (like sorting playing cards)

---

*Insertion sort is the go-to choice for small datasets and nearly sorted data. Its simplicity and adaptability make it a fundamental algorithm to master!*

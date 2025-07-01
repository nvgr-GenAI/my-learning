# Merge Sort

Merge Sort is a stable, divide-and-conquer sorting algorithm that guarantees O(n log n) performance in all cases.

## Algorithm Overview

**Time Complexity:**
- Best/Average/Worst Case: O(n log n)

**Space Complexity:** O(n)

**Properties:**
- ✅ Stable (preserves relative order of equal elements)
- ❌ Not in-place (requires O(n) extra space)
- ❌ Not adaptive (doesn't benefit from partially sorted data)

## How It Works

1. **Divide** the array into two halves
2. **Recursively sort** both halves
3. **Merge** the sorted halves back together

## Implementation

### Basic Merge Sort

```python
def merge_sort(arr):
    """
    Stable sorting algorithm using divide and conquer
    Time: O(n log n), Space: O(n)
    """
    if len(arr) <= 1:
        return arr
    
    # Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Conquer (merge)
    return merge(left, right)

def merge(left, right):
    """Merge two sorted arrays"""
    result = []
    i = j = 0
    
    # Merge elements in sorted order
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:  # <= ensures stability
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Add remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result

# Example usage
data = [38, 27, 43, 3, 9, 82, 10]
print(f"Original: {data}")
print(f"Sorted: {merge_sort(data)}")
```

### In-Place Merge Sort

```python
def merge_sort_inplace(arr):
    """
    In-place merge sort using auxiliary array
    Space: O(n) for temporary array
    """
    temp_arr = [0] * len(arr)
    merge_sort_helper(arr, temp_arr, 0, len(arr) - 1)

def merge_sort_helper(arr, temp_arr, left, right):
    """Recursive helper for in-place merge sort"""
    if left < right:
        mid = (left + right) // 2
        
        # Recursively sort both halves
        merge_sort_helper(arr, temp_arr, left, mid)
        merge_sort_helper(arr, temp_arr, mid + 1, right)
        
        # Merge the sorted halves
        merge_inplace(arr, temp_arr, left, mid, right)

def merge_inplace(arr, temp_arr, left, mid, right):
    """Merge two sorted subarrays in-place"""
    # Copy data to temp array
    for i in range(left, right + 1):
        temp_arr[i] = arr[i]
    
    i, j, k = left, mid + 1, left
    
    # Merge temp arrays back into arr
    while i <= mid and j <= right:
        if temp_arr[i] <= temp_arr[j]:
            arr[k] = temp_arr[i]
            i += 1
        else:
            arr[k] = temp_arr[j]
            j += 1
        k += 1
    
    # Copy remaining elements
    while i <= mid:
        arr[k] = temp_arr[i]
        i += 1
        k += 1
    
    while j <= right:
        arr[k] = temp_arr[j]
        j += 1
        k += 1

# Example usage
data = [12, 11, 13, 5, 6, 7]
print(f"Original: {data}")
merge_sort_inplace(data)
print(f"Sorted: {data}")
```

### Bottom-Up Merge Sort (Iterative)

```python
def merge_sort_bottom_up(arr):
    """
    Iterative (bottom-up) merge sort
    Avoids recursion overhead
    """
    n = len(arr)
    temp_arr = [0] * n
    
    # Start with subarrays of size 1, then 2, 4, 8, ...
    size = 1
    while size < n:
        left = 0
        while left < n - 1:
            # Calculate mid and right boundaries
            mid = min(left + size - 1, n - 1)
            right = min(left + size * 2 - 1, n - 1)
            
            # Merge subarrays arr[left...mid] and arr[mid+1...right]
            if mid < right:
                merge_inplace(arr, temp_arr, left, mid, right)
            
            left += size * 2
        
        size *= 2
    
    return arr
```

### Optimized Merge Sort

```python
def merge_sort_optimized(arr, threshold=7):
    """
    Optimized merge sort with insertion sort for small arrays
    """
    def insertion_sort_range(arr, left, right):
        """Insertion sort for small ranges"""
        for i in range(left + 1, right + 1):
            key = arr[i]
            j = i - 1
            while j >= left and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
    
    def merge_sort_helper(arr, temp_arr, left, right):
        if left < right:
            # Use insertion sort for small subarrays
            if right - left + 1 <= threshold:
                insertion_sort_range(arr, left, right)
            else:
                mid = (left + right) // 2
                merge_sort_helper(arr, temp_arr, left, mid)
                merge_sort_helper(arr, temp_arr, mid + 1, right)
                
                # Skip merge if already sorted
                if arr[mid] <= arr[mid + 1]:
                    return
                
                merge_inplace(arr, temp_arr, left, mid, right)
    
    temp_arr = [0] * len(arr)
    merge_sort_helper(arr, temp_arr, 0, len(arr) - 1)
    return arr
```

## Applications

### Counting Inversions

```python
def count_inversions(arr):
    """
    Count number of inversions using merge sort
    An inversion is when arr[i] > arr[j] for i < j
    """
    def merge_and_count(arr, temp_arr, left, mid, right):
        """Merge and count split inversions"""
        # Copy to temp array
        for i in range(left, right + 1):
            temp_arr[i] = arr[i]
        
        i, j, k = left, mid + 1, left
        inv_count = 0
        
        while i <= mid and j <= right:
            if temp_arr[i] <= temp_arr[j]:
                arr[k] = temp_arr[i]
                i += 1
            else:
                arr[k] = temp_arr[j]
                # All elements from i to mid are greater than arr[j]
                inv_count += (mid - i + 1)
                j += 1
            k += 1
        
        # Copy remaining elements
        while i <= mid:
            arr[k] = temp_arr[i]
            i += 1
            k += 1
        
        while j <= right:
            arr[k] = temp_arr[j]
            j += 1
            k += 1
        
        return inv_count
    
    def merge_sort_and_count(arr, temp_arr, left, right):
        """Merge sort while counting inversions"""
        inv_count = 0
        if left < right:
            mid = (left + right) // 2
            
            inv_count += merge_sort_and_count(arr, temp_arr, left, mid)
            inv_count += merge_sort_and_count(arr, temp_arr, mid + 1, right)
            inv_count += merge_and_count(arr, temp_arr, left, mid, right)
        
        return inv_count
    
    temp_arr = [0] * len(arr)
    return merge_sort_and_count(arr[:], temp_arr, 0, len(arr) - 1)

# Example usage
data = [2, 3, 8, 6, 1]
inversions = count_inversions(data)
print(f"Number of inversions: {inversions}")  # Output: 5
```

### External Sorting

```python
def external_merge_sort(input_file, output_file, memory_limit):
    """
    External merge sort for files larger than available memory
    Used for sorting large datasets that don't fit in RAM
    """
    import tempfile
    import heapq
    
    # Phase 1: Create sorted runs
    runs = []
    with open(input_file, 'r') as f:
        while True:
            # Read chunk that fits in memory
            chunk = []
            for _ in range(memory_limit):
                line = f.readline()
                if not line:
                    break
                chunk.append(int(line.strip()))
            
            if not chunk:
                break
            
            # Sort chunk and write to temporary file
            chunk.sort()
            run_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
            for num in chunk:
                run_file.write(f"{num}\n")
            run_file.close()
            runs.append(run_file.name)
    
    # Phase 2: Merge runs
    with open(output_file, 'w') as output:
        # Use priority queue to merge k sorted runs
        heap = []
        file_handles = []
        
        # Initialize heap with first element from each run
        for i, run_file in enumerate(runs):
            f = open(run_file, 'r')
            file_handles.append(f)
            line = f.readline()
            if line:
                heapq.heappush(heap, (int(line.strip()), i))
        
        # Merge all runs
        while heap:
            value, file_idx = heapq.heappop(heap)
            output.write(f"{value}\n")
            
            # Read next element from the same file
            line = file_handles[file_idx].readline()
            if line:
                heapq.heappush(heap, (int(line.strip()), file_idx))
        
        # Close all file handles
        for f in file_handles:
            f.close()
    
    # Clean up temporary files
    import os
    for run_file in runs:
        os.unlink(run_file)
```

## When to Use Merge Sort

### ✅ Good For:
- **Stability is required** - Preserves relative order of equal elements
- **Guaranteed O(n log n)** - No worst-case degradation
- **External sorting** - Works well with large datasets
- **Linked lists** - Can be implemented to sort linked lists in O(1) space
- **Parallel processing** - Naturally parallelizable

### ❌ Avoid When:
- **Memory is limited** - Requires O(n) extra space
- **Small arrays** - Overhead may not be worth it
- **In-place sorting required** - Use Quick Sort or Heap Sort

## Complexity Analysis

- **Time Complexity: O(n log n)** in all cases
- **Space Complexity: O(n)** for the temporary arrays
- **Recurrence Relation: T(n) = 2T(n/2) + O(n)**

## Visual Example

```
Initial: [38, 27, 43, 3, 9, 82, 10]

Divide Phase:
[38, 27, 43, 3] | [9, 82, 10]
[38, 27] [43, 3] | [9, 82] [10]
[38] [27] [43] [3] | [9] [82] [10]

Merge Phase:
[27, 38] [3, 43] | [9, 82] [10]
[3, 27, 38, 43] | [9, 10, 82]
[3, 9, 10, 27, 38, 43, 82]
```

## Interview Questions

1. **Why is Merge Sort stable while Quick Sort is not?**
2. **How would you implement Merge Sort for linked lists?**
3. **What is the space complexity and can you make it in-place?**
4. **How would you use Merge Sort to count inversions in an array?**
5. **Explain external sorting and its relationship to Merge Sort**

---

*Merge Sort is the go-to algorithm when you need guaranteed O(n log n) performance and stability!*

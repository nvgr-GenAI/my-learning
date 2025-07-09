# Array Sorting

Sorting is the process of rearranging elements in an array in a specific order - typically in ascending or descending order. Sorting is a fundamental operation in computer science and a crucial building block for many algorithms and applications.

## Importance of Array Sorting

1. **Improves Search Efficiency**: Binary search requires sorted arrays
2. **Simplifies Problem-Solving**: Many algorithms require sorted input
3. **Data Organization**: Makes data more readable and analyzable
4. **Facilitates Operations**: Such as finding duplicates, merging datasets, and finding the median

## Comparison-Based Sorting Algorithms

### Bubble Sort

Bubble sort is a simple comparison-based algorithm that repeatedly steps through the array, compares adjacent elements, and swaps them if they're in the wrong order.

```python
def bubble_sort(arr):
    """
    Sort an array using bubble sort.
    
    Time Complexity:
        - Best: O(n) when array is already sorted
        - Average: O(n²)
        - Worst: O(n²)
    Space Complexity: O(1)
    Stable: Yes
    """
    n = len(arr)
    
    # Optimization: flag to detect if array is already sorted
    for i in range(n):
        swapped = False
        
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            # Swap if the element found is greater than the next element
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        
        # If no swapping occurred in this pass, array is sorted
        if not swapped:
            break
            
    return arr
```

**Pros**:
- Simple to understand and implement
- Performs well for small arrays
- In-place sorting (doesn't require extra space)

**Cons**:
- Very inefficient for large arrays
- Always makes O(n²) comparisons even if array is already sorted (without optimization)

### Selection Sort

Selection sort divides the array into a sorted and an unsorted region. It repeatedly selects the minimum element from the unsorted region and moves it to the end of the sorted region.

```python
def selection_sort(arr):
    """
    Sort an array using selection sort.
    
    Time Complexity:
        - Best: O(n²)
        - Average: O(n²)
        - Worst: O(n²)
    Space Complexity: O(1)
    Stable: No
    """
    n = len(arr)
    
    # Traverse through all array elements
    for i in range(n):
        # Find the minimum element in unsorted array
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        
        # Swap the found minimum element with the first element
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
            
    return arr
```

**Pros**:
- Simple implementation
- Performs well for small arrays
- Makes minimum number of swaps (O(n))
- In-place sorting

**Cons**:
- Always O(n²) time complexity regardless of input
- Not stable (relative order of equal elements might change)

### Insertion Sort

Insertion sort builds the sorted array one element at a time by repeatedly taking the next unsorted element and inserting it into its correct position in the sorted portion.

```python
def insertion_sort(arr):
    """
    Sort an array using insertion sort.
    
    Time Complexity:
        - Best: O(n) when array is already sorted
        - Average: O(n²)
        - Worst: O(n²)
    Space Complexity: O(1)
    Stable: Yes
    """
    n = len(arr)
    
    # Traverse through 1 to n
    for i in range(1, n):
        key = arr[i]
        
        # Move elements of arr[0..i-1], that are greater than key,
        # one position ahead of their current position
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
            
    return arr
```

**Pros**:
- Efficient for small data sets
- Works well for almost-sorted arrays
- Can sort the array as it receives input (online algorithm)
- In-place and stable

**Cons**:
- Inefficient for large arrays compared to advanced algorithms

### Merge Sort

Merge sort is a divide-and-conquer algorithm that divides the array into two halves, recursively sorts them, and then merges the sorted halves.

```python
def merge_sort(arr):
    """
    Sort an array using merge sort.
    
    Time Complexity:
        - Best: O(n log n)
        - Average: O(n log n)
        - Worst: O(n log n)
    Space Complexity: O(n)
    Stable: Yes
    """
    if len(arr) <= 1:
        return arr
        
    # Divide array into two halves
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]
    
    # Recursively sort both halves
    left_half = merge_sort(left_half)
    right_half = merge_sort(right_half)
    
    # Merge the sorted halves
    return merge(left_half, right_half)

def merge(left, right):
    """Merge two sorted arrays into one sorted array."""
    result = []
    i = j = 0
    
    # Compare elements from both arrays and add smaller one to result
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Add remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result
```

**Pros**:
- Guaranteed O(n log n) performance
- Stable sort
- Works well for large datasets

**Cons**:
- Not in-place (requires additional space)
- Overkill for small arrays
- Recursive implementation can cause stack overflow for very large arrays

### Quick Sort

Quick sort is another divide-and-conquer algorithm that selects a 'pivot' element and partitions the array around the pivot.

```python
def quick_sort(arr):
    """
    Sort an array using quick sort.
    
    Time Complexity:
        - Best: O(n log n)
        - Average: O(n log n)
        - Worst: O(n²) when poorly pivoted
    Space Complexity:
        - O(log n) for recursion stack in average case
        - O(n) in worst case
    Stable: No
    """
    if len(arr) <= 1:
        return arr
        
    return quick_sort_helper(arr, 0, len(arr) - 1)

def quick_sort_helper(arr, low, high):
    if low < high:
        # Partition the array and get pivot position
        pivot_index = partition(arr, low, high)
        
        # Sort elements before and after the pivot
        quick_sort_helper(arr, low, pivot_index - 1)
        quick_sort_helper(arr, pivot_index + 1, high)
    
    return arr

def partition(arr, low, high):
    # Choose rightmost element as pivot
    pivot = arr[high]
    
    # Pointer for greater element
    i = low - 1
    
    # Traverse all elements, compare each with pivot
    for j in range(low, high):
        # If current element is smaller than pivot
        if arr[j] <= pivot:
            # Increment index of smaller element
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    # Place pivot in its final position
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
```

**Pros**:
- Generally faster in practice than other O(n log n) algorithms
- In-place sorting
- Cache-friendly
- Tail recursion can be optimized

**Cons**:
- Not stable
- Worst case O(n²) if poorly implemented or with bad pivot selection
- Less predictable performance than merge sort

#### Randomized Quick Sort

A variant of quick sort that selects a random pivot to avoid worst-case scenarios:

```python
import random

def randomized_quick_sort(arr):
    """
    Sort an array using randomized quick sort.
    
    Time Complexity:
        - Expected: O(n log n)
        - Worst (rare): O(n²)
    Space Complexity: O(log n) on average
    Stable: No
    """
    if len(arr) <= 1:
        return arr
        
    return randomized_quick_sort_helper(arr, 0, len(arr) - 1)

def randomized_quick_sort_helper(arr, low, high):
    if low < high:
        # Choose a random pivot
        pivot_index = random.randint(low, high)
        # Swap with the last element
        arr[pivot_index], arr[high] = arr[high], arr[pivot_index]
        
        # Partition and recursive sort
        pivot_index = partition(arr, low, high)
        randomized_quick_sort_helper(arr, low, pivot_index - 1)
        randomized_quick_sort_helper(arr, pivot_index + 1, high)
    
    return arr

def partition(arr, low, high):
    # Same partition function as regular quick sort
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
```

### Heap Sort

Heap sort uses a binary heap data structure to sort elements.

```python
def heap_sort(arr):
    """
    Sort an array using heap sort.
    
    Time Complexity:
        - Best: O(n log n)
        - Average: O(n log n)
        - Worst: O(n log n)
    Space Complexity: O(1)
    Stable: No
    """
    n = len(arr)
    
    # Build a max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    
    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]  # Swap
        heapify(arr, i, 0)
            
    return arr

def heapify(arr, n, i):
    """
    Heapify a subtree rooted at index i.
    n is the size of the heap.
    """
    largest = i  # Initialize largest as root
    left = 2 * i + 1
    right = 2 * i + 2
    
    # Check if left child exists and is greater than root
    if left < n and arr[left] > arr[largest]:
        largest = left
    
    # Check if right child exists and is greater than current largest
    if right < n and arr[right] > arr[largest]:
        largest = right
    
    # Change root if needed
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # Swap
        heapify(arr, n, largest)  # Heapify the affected subtree
```

**Pros**:
- Guaranteed O(n log n) performance
- In-place sorting
- No worst case unlike quick sort

**Cons**:
- Not stable
- Typically slower than quick sort in practice
- More complex to implement

## Non-Comparison Based Sorting Algorithms

### Counting Sort

Counting sort is an integer sorting algorithm that sorts elements based on their frequency counts.

```python
def counting_sort(arr):
    """
    Sort an array of integers using counting sort.
    
    Time Complexity: O(n + k) where k is the range of input
    Space Complexity: O(k)
    Stable: Yes (with careful implementation)
    """
    if not arr:
        return []
    
    # Find range of input
    max_val = max(arr)
    min_val = min(arr)
    range_val = max_val - min_val + 1
    
    # Initialize counting array and output array
    count = [0] * range_val
    output = [0] * len(arr)
    
    # Count occurrences
    for num in arr:
        count[num - min_val] += 1
    
    # Modify count array to store positions
    for i in range(1, len(count)):
        count[i] += count[i - 1]
    
    # Build the output array
    for i in range(len(arr) - 1, -1, -1):  # Reverse order for stability
        output[count[arr[i] - min_val] - 1] = arr[i]
        count[arr[i] - min_val] -= 1
    
    return output
```

**Pros**:
- O(n) time complexity when range is small
- Stable if implemented carefully
- Works well for small integers

**Cons**:
- Inefficient when range is large compared to array size
- Limited to integers or data that can be mapped to integers

### Radix Sort

Radix sort sorts integers by processing individual digits, starting from the least significant digit.

```python
def radix_sort(arr):
    """
    Sort an array of non-negative integers using radix sort.
    
    Time Complexity: O(d * (n + k)) where d is the number of digits
                    and k is the range of each digit (usually 10)
    Space Complexity: O(n + k)
    Stable: Yes
    """
    if not arr:
        return []
    
    # Find the maximum number to know number of digits
    max_val = max(arr)
    
    # Do counting sort for every digit
    exp = 1
    while max_val // exp > 0:
        counting_sort_by_digit(arr, exp)
        exp *= 10
        
    return arr

def counting_sort_by_digit(arr, exp):
    """Sort array by digit at position exp."""
    n = len(arr)
    output = [0] * n
    count = [0] * 10  # Digits 0-9
    
    # Count occurrences of each digit
    for i in range(n):
        digit = (arr[i] // exp) % 10
        count[digit] += 1
    
    # Change count[i] to position of digit in output
    for i in range(1, 10):
        count[i] += count[i - 1]
    
    # Build output array
    for i in range(n - 1, -1, -1):
        digit = (arr[i] // exp) % 10
        output[count[digit] - 1] = arr[i]
        count[digit] -= 1
    
    # Copy output to arr
    for i in range(n):
        arr[i] = output[i]
```

**Pros**:
- Linear time complexity when number of digits is constant
- Works well for large integers
- Stable sort

**Cons**:
- Not in-place
- Less efficient for small arrays
- Limited to integers or data that can be keyed by integers

### Bucket Sort

Bucket sort divides the range into equal-size buckets and then sorts these buckets individually.

```python
def bucket_sort(arr, num_buckets=10):
    """
    Sort an array using bucket sort.
    
    Time Complexity:
        - Average: O(n + k) where k is the number of buckets
        - Worst: O(n²) if all elements are placed in a single bucket
    Space Complexity: O(n + k)
    Stable: Depends on the algorithm used to sort individual buckets
    """
    if not arr:
        return []
    
    # Find min and max values
    min_val, max_val = min(arr), max(arr)
    
    # Create buckets
    bucket_range = (max_val - min_val) / num_buckets
    buckets = [[] for _ in range(num_buckets)]
    
    # Distribute elements into buckets
    for num in arr:
        # Handle edge case for max value
        if num == max_val:
            bucket_idx = num_buckets - 1
        else:
            bucket_idx = int((num - min_val) / bucket_range)
        buckets[bucket_idx].append(num)
    
    # Sort individual buckets (using insertion sort here)
    for i in range(num_buckets):
        insertion_sort(buckets[i])
    
    # Concatenate buckets
    result = []
    for bucket in buckets:
        result.extend(bucket)
    
    return result

def insertion_sort(arr):
    """Insertion sort for sorting individual buckets."""
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
```

**Pros**:
- Good performance for uniformly distributed data
- Can be parallelized easily
- Can be very fast with the right distribution

**Cons**:
- Performance depends heavily on data distribution
- Extra space requirements
- Not in-place

## Hybrid Sorting Algorithms

### Timsort

Timsort is a hybrid sorting algorithm derived from merge sort and insertion sort, designed to perform well on many kinds of real-world data.

```python
# Python's built-in sort and sorted functions use Timsort
# Here's a simplified version

def timsort(arr):
    """
    A simplified implementation of Timsort.
    
    Time Complexity:
        - Best: O(n)
        - Average: O(n log n)
        - Worst: O(n log n)
    Space Complexity: O(n)
    Stable: Yes
    """
    min_run = 32  # Minimum size of a run
    n = len(arr)
    
    # Sort individual subarrays of size min_run
    for start in range(0, n, min_run):
        end = min(start + min_run - 1, n - 1)
        insertion_sort_range(arr, start, end)
    
    # Start merging from size min_run
    size = min_run
    while size < n:
        # Pick starting points of merges
        for start in range(0, n, size * 2):
            mid = min(n - 1, start + size - 1)
            end = min(n - 1, mid + size)
            
            # Merge if there are elements in both runs
            if mid < end:
                merge_sort_range(arr, start, mid, end)
        
        size *= 2
    
    return arr

def insertion_sort_range(arr, start, end):
    """Insertion sort a subarray."""
    for i in range(start + 1, end + 1):
        key = arr[i]
        j = i - 1
        while j >= start and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def merge_sort_range(arr, start, mid, end):
    """Merge two sorted subarrays."""
    left = arr[start:mid+1]
    right = arr[mid+1:end+1]
    
    i = j = 0
    k = start
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1
    
    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1
    
    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1
```

**Pros**:
- Efficient on real-world data
- Adapts to different data patterns
- Stable sort
- Works well with partially sorted data

**Cons**:
- Complex implementation
- More memory overhead than simpler algorithms

### Introsort

Introsort begins with quicksort and switches to heapsort when the recursion depth exceeds a certain level.

```python
import math

def introsort(arr):
    """
    Sort an array using introsort.
    
    Time Complexity:
        - Best: O(n log n)
        - Average: O(n log n)
        - Worst: O(n log n)
    Space Complexity: O(log n)
    Stable: No
    """
    max_depth = 2 * int(math.log2(len(arr)))
    introsort_helper(arr, 0, len(arr) - 1, max_depth)
    return arr

def introsort_helper(arr, start, end, max_depth):
    # Size of the array
    size = end - start + 1
    
    # Switch to insertion sort for small arrays
    if size < 16:
        insertion_sort_range(arr, start, end)
        return
    
    # Switch to heapsort if max depth reached
    if max_depth == 0:
        heapsort_range(arr, start, end)
        return
    
    # Otherwise, use quicksort
    pivot = partition(arr, start, end)
    introsort_helper(arr, start, pivot - 1, max_depth - 1)
    introsort_helper(arr, pivot + 1, end, max_depth - 1)

def heapsort_range(arr, start, end):
    """Heapsort a subarray."""
    # Build heap (rearrange array)
    size = end - start + 1
    for i in range(size // 2 - 1, -1, -1):
        heapify_range(arr, size, i, start)
    
    # Extract elements one by one
    for i in range(size - 1, 0, -1):
        arr[start], arr[start + i] = arr[start + i], arr[start]  # Swap
        heapify_range(arr, i, 0, start)

def heapify_range(arr, n, i, start):
    """Heapify a subtree with root at index i."""
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    
    if left < n and arr[start + left] > arr[start + largest]:
        largest = left
    
    if right < n and arr[start + right] > arr[start + largest]:
        largest = right
    
    if largest != i:
        arr[start + i], arr[start + largest] = arr[start + largest], arr[start + i]
        heapify_range(arr, n, largest, start)
```

**Pros**:
- Combines the best of quicksort, heapsort, and insertion sort
- Guaranteed O(n log n) worst-case
- Performs well in practice
- Works in-place

**Cons**:
- Complex implementation
- Not stable
- Less cache-friendly than Timsort

## Special Case Sorting

### Sorting Almost Sorted Arrays

When the array is already nearly sorted, insertion sort can be the most efficient choice.

```python
def sort_almost_sorted(arr, k):
    """
    Sort an almost sorted array where each element is at most k positions
    away from its sorted position.
    
    Time Complexity: O(n log k)
    Space Complexity: O(k)
    """
    import heapq
    
    # Create a min-heap with the first k+1 elements
    heap = arr[:min(k + 1, len(arr))]
    heapq.heapify(heap)
    
    # Index for the sorted part
    index = 0
    
    # Process remaining elements
    for i in range(k + 1, len(arr)):
        # Add min element to the sorted part
        arr[index] = heapq.heappop(heap)
        index += 1
        
        # Add the next element to the heap
        heapq.heappush(heap, arr[i])
    
    # Extract remaining elements from the heap
    while heap:
        arr[index] = heapq.heappop(heap)
        index += 1
    
    return arr
```

### External Sorting

External sorting is used when the data being sorted doesn't fit into memory.

```python
def external_merge_sort(filename, chunk_size=1000, temp_dir='./temp'):
    """
    External merge sort for large files that don't fit in memory.
    
    Time Complexity: O(n log n)
    Space Complexity: O(chunk_size)
    """
    import os
    import tempfile
    import shutil
    
    # Create temporary directory
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # Phase 1: Create sorted chunks
    chunk_files = []
    with open(filename, 'r') as f:
        chunk_num = 0
        while True:
            # Read a chunk of data
            chunk = []
            for _ in range(chunk_size):
                line = f.readline().strip()
                if not line:
                    break
                chunk.append(int(line))
                
            if not chunk:
                break
                
            # Sort the chunk in memory
            chunk.sort()
            
            # Write the sorted chunk to a temporary file
            chunk_filename = os.path.join(temp_dir, f'chunk_{chunk_num}.txt')
            chunk_files.append(chunk_filename)
            
            with open(chunk_filename, 'w') as chunk_file:
                for num in chunk:
                    chunk_file.write(f"{num}\n")
            
            chunk_num += 1
    
    # Phase 2: Merge the sorted chunks
    with open(f"{filename}.sorted", 'w') as out_file:
        # Open all chunk files
        files = [open(f, 'r') for f in chunk_files]
        
        # Initialize with first element from each chunk
        heap = []
        for i, f in enumerate(files):
            line = f.readline().strip()
            if line:
                heapq.heappush(heap, (int(line), i))
        
        # Merge
        while heap:
            val, file_idx = heapq.heappop(heap)
            out_file.write(f"{val}\n")
            
            line = files[file_idx].readline().strip()
            if line:
                heapq.heappush(heap, (int(line), file_idx))
        
        # Close all files
        for f in files:
            f.close()
    
    # Cleanup temporary files
    for f in chunk_files:
        os.remove(f)
    if not os.listdir(temp_dir):
        os.rmdir(temp_dir)
    
    return f"{filename}.sorted"
```

## Parallel Sorting Algorithms

### Parallel Merge Sort

A parallel version of merge sort that utilizes multiple cores/threads.

```python
def parallel_merge_sort(arr, num_threads=4):
    """
    Parallel implementation of merge sort.
    
    Time Complexity: O((n log n) / p) where p is the number of processors
    Space Complexity: O(n)
    """
    # This is a simplified example that uses Python's concurrent.futures
    # For real parallel performance, consider using a lower-level language
    
    import concurrent.futures
    
    def merge_sort_range(arr, start, end):
        if end - start <= 1:
            return arr[start:end]
        
        mid = (start + end) // 2
        left = merge_sort_range(arr, start, mid)
        right = merge_sort_range(arr, mid, end)
        return merge(left, right)
    
    def merge(left, right):
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
                
        result.extend(left[i:])
        result.extend(right[j:])
        return result
    
    # Base case for small arrays
    if len(arr) <= 1000 or num_threads <= 1:
        return merge_sort(arr)
    
    # Divide array into chunks for parallel processing
    chunk_size = len(arr) // num_threads
    chunks = [(i * chunk_size, min((i + 1) * chunk_size, len(arr))) 
              for i in range(num_threads)]
    
    # Sort chunks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        sorted_chunks = list(executor.map(
            lambda x: merge_sort_range(arr, x[0], x[1]), 
            chunks))
    
    # Merge sorted chunks
    result = sorted_chunks[0]
    for i in range(1, len(sorted_chunks)):
        result = merge(result, sorted_chunks[i])
    
    return result
```

## Performance Comparison

| Algorithm | Best | Average | Worst | Space | Stable |
|-----------|------|---------|-------|-------|--------|
| Bubble Sort | O(n) | O(n²) | O(n²) | O(1) | Yes |
| Selection Sort | O(n²) | O(n²) | O(n²) | O(1) | No |
| Insertion Sort | O(n) | O(n²) | O(n²) | O(1) | Yes |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) | No |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) | No |
| Counting Sort | O(n + k) | O(n + k) | O(n + k) | O(k) | Yes |
| Radix Sort | O(d(n + k)) | O(d(n + k)) | O(d(n + k)) | O(n + k) | Yes |
| Bucket Sort | O(n + k) | O(n + k) | O(n²) | O(n + k) | Yes |
| Timsort | O(n) | O(n log n) | O(n log n) | O(n) | Yes |
| Introsort | O(n log n) | O(n log n) | O(n log n) | O(log n) | No |

Where:
- n is the number of elements
- k is the range of elements or number of buckets
- d is the number of digits

## Choosing the Right Sorting Algorithm

The choice of sorting algorithm depends on several factors:

1. **Size of the Data**:
   - Small datasets: Insertion sort, Selection sort
   - Medium to large: Quicksort, Merge sort, Heap sort
   - Very large (external): External merge sort

2. **Data Characteristics**:
   - Nearly sorted: Insertion sort
   - Uniformly distributed: Bucket sort, Radix sort
   - Integers in small range: Counting sort
   - General purpose: Timsort, Introsort

3. **Stability Requirement**:
   - Stable sort needed: Merge sort, Insertion sort, Timsort
   - Stability not important: Quicksort, Heap sort

4. **Memory Constraints**:
   - Limited memory: In-place sorts like Heap sort or Quick sort
   - Memory available: Merge sort

5. **Parallel Processing Available**:
   - Parallelizable: Merge sort, Parallel sorting networks

## Real-world Applications

1. **Databases**: Sorting records for efficient queries and indexing
2. **Search Engines**: Ranking and sorting search results
3. **Operating Systems**: Process scheduling, memory management
4. **Analytics**: Sorting data for analysis and reporting
5. **File Systems**: Organizing files in directories
6. **Network Routing**: Sorting packets by priority
7. **Graphics**: Sorting objects by depth (z-index)

## Conclusion

Array sorting is a fundamental operation in computer science with numerous algorithms optimized for different scenarios. Understanding the strengths and weaknesses of each sorting algorithm allows you to choose the most appropriate one for your specific use case.

In practice, most programming languages provide built-in sorting functions that implement efficient hybrid algorithms (like Timsort in Python and Java). However, understanding the underlying concepts is crucial for optimizing performance in specialized scenarios.

## Practice Problems

1. Implement a sorting algorithm that works efficiently for arrays with many duplicate elements.
2. Sort an array of strings by their lengths and then lexicographically for strings of the same length.
3. Given an almost sorted array where each element is at most k positions away from its correct position, sort the array efficiently.
4. Sort an array such that all even numbers come before odd numbers, and they are in their respective relative order.
5. Implement an external sort for a large file of integers.

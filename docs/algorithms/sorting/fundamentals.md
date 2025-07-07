# Sorting Algorithms - Fundamentals

## 🎯 Overview

Sorting algorithms are fundamental to computer science, enabling efficient data organization and serving as building blocks for many other algorithms. This section covers essential sorting techniques, their properties, and optimal use cases.

=== "📋 Core Sorting Concepts"

    ## **Sorting Algorithm Properties**
    
    | Property | Definition | Importance |
    |----------|------------|------------|
    | **Stability** | Preserves relative order of equal elements | Important for multi-key sorting |
    | **In-place** | Uses O(1) extra space | Memory efficiency |
    | **Adaptive** | Performs better on partially sorted data | Real-world optimization |
    | **Comparison-based** | Only compares elements | General-purpose sorting |
    | **Online** | Can sort data as it arrives | Streaming applications |
    | **Internal/External** | Data fits in memory or not | Storage considerations |

    ## **Sorting Complexity Classes**
    
    | Class | Time Complexity | Algorithms | Use Cases |
    |-------|-----------------|------------|-----------|
    | **Simple** | O(n²) | Bubble, Selection, Insertion | Small datasets, educational |
    | **Efficient** | O(n log n) | Merge, Heap, Quick (average) | General-purpose sorting |
    | **Linear** | O(n + k) | Counting, Radix, Bucket | Specialized constraints |
    | **Hybrid** | Adaptive | Timsort, Introsort | Production systems |

=== "🔄 Simple Sorting Algorithms"

    ## **Bubble Sort**
    
    ```python
    def bubble_sort(arr):
        """
        Bubble sort - simple but inefficient
        Time: O(n²), Space: O(1)
        Stable: Yes, In-place: Yes
        """
        n = len(arr)
        
        for i in range(n):
            swapped = False
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    swapped = True
            
            # Early termination if no swaps
            if not swapped:
                break
        
        return arr
    
    def bubble_sort_optimized(arr):
        """
        Optimized bubble sort with early termination
        """
        n = len(arr)
        
        for i in range(n):
            swapped = False
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    swapped = True
            
            if not swapped:
                break
        
        return arr
    ```
    
    **Properties:**
    - Stable: Yes
    - In-place: Yes
    - Adaptive: Yes (with optimization)
    - Best case: O(n) with optimization
    
    ## **Selection Sort**
    
    ```python
    def selection_sort(arr):
        """
        Selection sort - finds minimum and places it at beginning
        Time: O(n²), Space: O(1)
        Stable: No, In-place: Yes
        """
        n = len(arr)
        
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
        
        return arr
    
    def selection_sort_stable(arr):
        """
        Stable version of selection sort
        """
        n = len(arr)
        
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            
            # Shift elements to maintain stability
            min_val = arr[min_idx]
            while min_idx > i:
                arr[min_idx] = arr[min_idx - 1]
                min_idx -= 1
            arr[i] = min_val
        
        return arr
    ```
    
    **Properties:**
    - Stable: No (stable version possible)
    - In-place: Yes
    - Adaptive: No
    - Always O(n²) comparisons
    
    ## **Insertion Sort**
    
    ```python
    def insertion_sort(arr):
        """
        Insertion sort - builds sorted portion element by element
        Time: O(n²) worst, O(n) best, Space: O(1)
        Stable: Yes, In-place: Yes, Adaptive: Yes
        """
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            
            # Move elements greater than key one position ahead
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            
            arr[j + 1] = key
        
        return arr
    
    def binary_insertion_sort(arr):
        """
        Insertion sort with binary search for position
        Reduces comparisons but same time complexity
        """
        for i in range(1, len(arr)):
            key = arr[i]
            left, right = 0, i
            
            # Binary search for insertion position
            while left < right:
                mid = (left + right) // 2
                if arr[mid] <= key:
                    left = mid + 1
                else:
                    right = mid
            
            # Shift elements and insert
            for j in range(i, left, -1):
                arr[j] = arr[j - 1]
            arr[left] = key
        
        return arr
    ```

=== "⚡ Efficient Sorting Algorithms"

    ## **Merge Sort**
    
    ```python
    def merge_sort(arr):
        """
        Merge sort - divide and conquer approach
        Time: O(n log n) always, Space: O(n)
        Stable: Yes, In-place: No
        """
        if len(arr) <= 1:
            return arr
        
        mid = len(arr) // 2
        left = merge_sort(arr[:mid])
        right = merge_sort(arr[mid:])
        
        return merge(left, right)
    
    def merge(left, right):
        """Helper function to merge two sorted arrays"""
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
    
    def merge_sort_inplace(arr, left=0, right=None):
        """
        In-place merge sort (still O(n) space due to recursion)
        """
        if right is None:
            right = len(arr) - 1
        
        if left < right:
            mid = (left + right) // 2
            merge_sort_inplace(arr, left, mid)
            merge_sort_inplace(arr, mid + 1, right)
            merge_inplace(arr, left, mid, right)
    
    def merge_inplace(arr, left, mid, right):
        """In-place merge helper"""
        left_arr = arr[left:mid + 1]
        right_arr = arr[mid + 1:right + 1]
        
        i = j = 0
        k = left
        
        while i < len(left_arr) and j < len(right_arr):
            if left_arr[i] <= right_arr[j]:
                arr[k] = left_arr[i]
                i += 1
            else:
                arr[k] = right_arr[j]
                j += 1
            k += 1
        
        while i < len(left_arr):
            arr[k] = left_arr[i]
            i += 1
            k += 1
        
        while j < len(right_arr):
            arr[k] = right_arr[j]
            j += 1
            k += 1
    ```
    
    ## **Quick Sort**
    
    ```python
    import random
    
    def quick_sort(arr, left=0, right=None):
        """
        Quick sort - divide and conquer with pivot
        Time: O(n log n) average, O(n²) worst, Space: O(log n)
        Stable: No, In-place: Yes
        """
        if right is None:
            right = len(arr) - 1
        
        if left < right:
            pivot_index = partition(arr, left, right)
            quick_sort(arr, left, pivot_index - 1)
            quick_sort(arr, pivot_index + 1, right)
        
        return arr
    
    def partition(arr, left, right):
        """Lomuto partition scheme"""
        pivot = arr[right]
        i = left - 1
        
        for j in range(left, right):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        
        arr[i + 1], arr[right] = arr[right], arr[i + 1]
        return i + 1
    
    def quick_sort_hoare(arr, left=0, right=None):
        """
        Quick sort with Hoare partition (more efficient)
        """
        if right is None:
            right = len(arr) - 1
        
        if left < right:
            pivot_index = hoare_partition(arr, left, right)
            quick_sort_hoare(arr, left, pivot_index)
            quick_sort_hoare(arr, pivot_index + 1, right)
    
    def hoare_partition(arr, left, right):
        """Hoare partition scheme"""
        pivot = arr[left]
        i = left - 1
        j = right + 1
        
        while True:
            i += 1
            while arr[i] < pivot:
                i += 1
            
            j -= 1
            while arr[j] > pivot:
                j -= 1
            
            if i >= j:
                return j
            
            arr[i], arr[j] = arr[j], arr[i]
    
    def quick_sort_randomized(arr, left=0, right=None):
        """
        Randomized quick sort to avoid worst case
        """
        if right is None:
            right = len(arr) - 1
        
        if left < right:
            # Random pivot selection
            random_index = random.randint(left, right)
            arr[random_index], arr[right] = arr[right], arr[random_index]
            
            pivot_index = partition(arr, left, right)
            quick_sort_randomized(arr, left, pivot_index - 1)
            quick_sort_randomized(arr, pivot_index + 1, right)
    ```
    
    ## **Heap Sort**
    
    ```python
    def heap_sort(arr):
        """
        Heap sort using max heap
        Time: O(n log n) always, Space: O(1)
        Stable: No, In-place: Yes
        """
        n = len(arr)
        
        # Build max heap
        for i in range(n // 2 - 1, -1, -1):
            heapify(arr, n, i)
        
        # Extract elements from heap
        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            heapify(arr, i, 0)
        
        return arr
    
    def heapify(arr, heap_size, root_index):
        """Maintain max heap property"""
        largest = root_index
        left_child = 2 * root_index + 1
        right_child = 2 * root_index + 2
        
        if left_child < heap_size and arr[left_child] > arr[largest]:
            largest = left_child
        
        if right_child < heap_size and arr[right_child] > arr[largest]:
            largest = right_child
        
        if largest != root_index:
            arr[root_index], arr[largest] = arr[largest], arr[root_index]
            heapify(arr, heap_size, largest)
    ```

=== "🎲 Non-Comparison Sorting"

    ## **Counting Sort**
    
    ```python
    def counting_sort(arr, max_val=None):
        """
        Counting sort for integers in limited range
        Time: O(n + k), Space: O(k) where k is range
        Stable: Yes, In-place: No
        """
        if not arr:
            return arr
        
        if max_val is None:
            max_val = max(arr)
        
        min_val = min(arr)
        range_size = max_val - min_val + 1
        
        # Count occurrences
        count = [0] * range_size
        for num in arr:
            count[num - min_val] += 1
        
        # Accumulate counts for stable sorting
        for i in range(1, range_size):
            count[i] += count[i - 1]
        
        # Build result array
        result = [0] * len(arr)
        for i in range(len(arr) - 1, -1, -1):
            result[count[arr[i] - min_val] - 1] = arr[i]
            count[arr[i] - min_val] -= 1
        
        return result
    
    def counting_sort_simple(arr):
        """
        Simple counting sort (not stable)
        """
        if not arr:
            return arr
        
        max_val = max(arr)
        min_val = min(arr)
        range_size = max_val - min_val + 1
        
        count = [0] * range_size
        for num in arr:
            count[num - min_val] += 1
        
        result = []
        for i in range(range_size):
            result.extend([i + min_val] * count[i])
        
        return result
    ```
    
    ## **Radix Sort**
    
    ```python
    def radix_sort(arr, base=10):
        """
        Radix sort for integers
        Time: O(d(n + k)) where d is digits, k is base
        Stable: Yes, In-place: No
        """
        if not arr:
            return arr
        
        max_num = max(arr)
        exp = 1
        
        while max_num // exp > 0:
            counting_sort_by_digit(arr, exp, base)
            exp *= base
        
        return arr
    
    def counting_sort_by_digit(arr, exp, base):
        """Counting sort by specific digit"""
        n = len(arr)
        output = [0] * n
        count = [0] * base
        
        # Count occurrences of each digit
        for num in arr:
            digit = (num // exp) % base
            count[digit] += 1
        
        # Calculate cumulative count
        for i in range(1, base):
            count[i] += count[i - 1]
        
        # Build output array
        for i in range(n - 1, -1, -1):
            digit = (arr[i] // exp) % base
            output[count[digit] - 1] = arr[i]
            count[digit] -= 1
        
        # Copy back to original array
        for i in range(n):
            arr[i] = output[i]
    
    def radix_sort_strings(arr):
        """
        Radix sort for strings of equal length
        """
        if not arr or not arr[0]:
            return arr
        
        max_len = len(arr[0])
        
        # Sort by each character position from right to left
        for i in range(max_len - 1, -1, -1):
            arr = counting_sort_strings_by_char(arr, i)
        
        return arr
    
    def counting_sort_strings_by_char(arr, char_index):
        """Counting sort by character at specific position"""
        count = [0] * 256  # ASCII characters
        output = [''] * len(arr)
        
        # Count character frequencies
        for string in arr:
            char_code = ord(string[char_index])
            count[char_code] += 1
        
        # Calculate cumulative count
        for i in range(1, 256):
            count[i] += count[i - 1]
        
        # Build output array
        for i in range(len(arr) - 1, -1, -1):
            char_code = ord(arr[i][char_index])
            output[count[char_code] - 1] = arr[i]
            count[char_code] -= 1
        
        return output
    ```
    
    ## **Bucket Sort**
    
    ```python
    def bucket_sort(arr, bucket_count=None):
        """
        Bucket sort for uniformly distributed data
        Time: O(n + k) average, O(n²) worst
        Space: O(n + k)
        """
        if not arr:
            return arr
        
        if bucket_count is None:
            bucket_count = len(arr)
        
        # Create buckets
        buckets = [[] for _ in range(bucket_count)]
        
        # Distribute elements into buckets
        max_val = max(arr)
        min_val = min(arr)
        range_size = max_val - min_val
        
        for num in arr:
            if range_size == 0:
                bucket_index = 0
            else:
                bucket_index = int((num - min_val) * (bucket_count - 1) / range_size)
            buckets[bucket_index].append(num)
        
        # Sort individual buckets and concatenate
        result = []
        for bucket in buckets:
            if bucket:
                bucket.sort()  # Can use any sorting algorithm
                result.extend(bucket)
        
        return result
    ```

=== "🔀 Hybrid Sorting Algorithms"

    ## **Introsort (Introspective Sort)**
    
    ```python
    import math
    
    def introsort(arr):
        """
        Hybrid algorithm: Quick sort with fallback to heap sort
        Time: O(n log n) worst case, Space: O(log n)
        """
        max_depth = 2 * math.floor(math.log2(len(arr)))
        introsort_helper(arr, 0, len(arr) - 1, max_depth)
        return arr
    
    def introsort_helper(arr, left, right, max_depth):
        size = right - left + 1
        
        if size <= 1:
            return
        elif size <= 16:
            # Use insertion sort for small arrays
            insertion_sort_range(arr, left, right)
        elif max_depth == 0:
            # Use heap sort when recursion depth is too deep
            heap_sort_range(arr, left, right)
        else:
            # Use quick sort
            pivot = partition(arr, left, right)
            introsort_helper(arr, left, pivot - 1, max_depth - 1)
            introsort_helper(arr, pivot + 1, right, max_depth - 1)
    
    def insertion_sort_range(arr, left, right):
        """Insertion sort for range"""
        for i in range(left + 1, right + 1):
            key = arr[i]
            j = i - 1
            while j >= left and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
    
    def heap_sort_range(arr, left, right):
        """Heap sort for range"""
        # Implementation similar to heap_sort but for specific range
        pass
    ```
    
    ## **Timsort (Python's Default)**
    
    ```python
    def timsort_simplified(arr):
        """
        Simplified version of Timsort
        Hybrid of merge sort and insertion sort
        """
        MIN_MERGE = 32
        
        def get_min_run_length(n):
            """Compute minimum run length for Timsort"""
            r = 0
            while n >= MIN_MERGE:
                r |= n & 1
                n >>= 1
            return n + r
        
        def insertion_sort_run(arr, left, right):
            """Insertion sort for small runs"""
            for i in range(left + 1, right + 1):
                key = arr[i]
                j = i - 1
                while j >= left and arr[j] > key:
                    arr[j + 1] = arr[j]
                    j -= 1
                arr[j + 1] = key
        
        n = len(arr)
        min_run = get_min_run_length(n)
        
        # Sort individual runs
        for start in range(0, n, min_run):
            end = min(start + min_run - 1, n - 1)
            insertion_sort_run(arr, start, end)
        
        # Merge runs
        size = min_run
        while size < n:
            for start in range(0, n, size * 2):
                mid = start + size - 1
                end = min(start + size * 2 - 1, n - 1)
                if mid < end:
                    merge_runs(arr, start, mid, end)
            size *= 2
        
        return arr
    
    def merge_runs(arr, left, mid, right):
        """Merge two runs"""
        left_arr = arr[left:mid + 1]
        right_arr = arr[mid + 1:right + 1]
        
        i = j = 0
        k = left
        
        while i < len(left_arr) and j < len(right_arr):
            if left_arr[i] <= right_arr[j]:
                arr[k] = left_arr[i]
                i += 1
            else:
                arr[k] = right_arr[j]
                j += 1
            k += 1
        
        while i < len(left_arr):
            arr[k] = left_arr[i]
            i += 1
            k += 1
        
        while j < len(right_arr):
            arr[k] = right_arr[j]
            j += 1
            k += 1
    ```

=== "📊 Complexity Analysis"

    ## **Time Complexities**
    
    | Algorithm | Best Case | Average Case | Worst Case | Space | Stable |
    |-----------|-----------|--------------|------------|-------|--------|
    | **Bubble Sort** | O(n) | O(n²) | O(n²) | O(1) | Yes |
    | **Selection Sort** | O(n²) | O(n²) | O(n²) | O(1) | No |
    | **Insertion Sort** | O(n) | O(n²) | O(n²) | O(1) | Yes |
    | **Merge Sort** | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes |
    | **Quick Sort** | O(n log n) | O(n log n) | O(n²) | O(log n) | No |
    | **Heap Sort** | O(n log n) | O(n log n) | O(n log n) | O(1) | No |
    | **Counting Sort** | O(n + k) | O(n + k) | O(n + k) | O(k) | Yes |
    | **Radix Sort** | O(d(n + k)) | O(d(n + k)) | O(d(n + k)) | O(n + k) | Yes |
    | **Bucket Sort** | O(n + k) | O(n + k) | O(n²) | O(n) | Yes |
    
    ## **Algorithm Selection Guide**
    
    | Scenario | Best Algorithm | Reason |
    |----------|---------------|---------|
    | **Small arrays (< 50)** | Insertion Sort | Simple, efficient for small data |
    | **Nearly sorted data** | Insertion Sort, Timsort | Adaptive algorithms perform well |
    | **Memory constrained** | Heap Sort, Quick Sort | In-place algorithms |
    | **Stability required** | Merge Sort, Timsort | Preserve relative order |
    | **Worst-case guarantee** | Merge Sort, Heap Sort | Always O(n log n) |
    | **Average case performance** | Quick Sort | Fastest average case |
    | **Integer data, limited range** | Counting Sort, Radix Sort | Linear time complexity |
    | **External sorting** | Merge Sort | Efficient for disk-based data |

=== "🎯 Problem-Solving Strategies"

    ## **Sorting Problem Patterns**
    
    | Pattern | When to Use | Example Problems |
    |---------|-------------|------------------|
    | **Custom Comparator** | Non-standard ordering | Sort by multiple criteria |
    | **Partial Sorting** | Only need k smallest/largest | Quick Select, Heap |
    | **Stability Preservation** | Maintain relative order | Multi-level sorting |
    | **In-place Requirement** | Memory constraints | Quick Sort, Heap Sort |
    | **External Sorting** | Data doesn't fit in memory | Merge Sort variants |
    | **Range-limited Data** | Known value ranges | Counting Sort, Bucket Sort |
    
    ## **Optimization Techniques**
    
    ```python
    def choose_sorting_algorithm(data_characteristics):
        """
        Algorithm selection based on data characteristics
        """
        size = data_characteristics['size']
        is_sorted = data_characteristics['partially_sorted']
        stability_needed = data_characteristics['stability_required']
        memory_limited = data_characteristics['memory_constrained']
        data_type = data_characteristics['data_type']
        
        if size < 50:
            return "Insertion Sort"
        elif is_sorted and stability_needed:
            return "Timsort"
        elif memory_limited and not stability_needed:
            return "Heap Sort"
        elif data_type == 'integer' and data_characteristics['limited_range']:
            return "Counting Sort or Radix Sort"
        elif stability_needed:
            return "Merge Sort"
        else:
            return "Quick Sort (randomized)"
    ```

---

*Master these sorting fundamentals to efficiently organize data and optimize algorithm performance!*

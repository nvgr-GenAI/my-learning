# Introsort

Introsort (Introspective Sort) is a hybrid sorting algorithm that combines the strengths of Quicksort, Heapsort, and Insertion Sort to achieve optimal performance in all scenarios.

=== "📋 Algorithm Overview"

    ## Key Characteristics
    
    - **Time Complexity**: 
      - Average and Worst Case: O(n log n)
    - **Space Complexity**: O(log n)
    - **Stable**: No
    - **In-place**: Yes
    - **Hybrid**: Combines Quicksort, Heapsort, and Insertion Sort
    - **Adaptive**: Partially, switches between algorithms based on recursion depth
    
    ## When to Use
    
    - When you need guaranteed O(n log n) worst-case performance
    - As a general-purpose sorting algorithm
    - When memory usage is a concern (requires less space than Mergesort)
    - When stability is not required
    
    ## Advantages
    
    - Combines the best aspects of multiple algorithms
    - Avoids Quicksort's worst-case O(n²) performance
    - More efficient than Heapsort in the average case
    - Better cache locality than Heapsort
    - Uses less memory than Mergesort

=== "🔄 How It Works"

    ## Algorithm Steps
    
    1. Start with Quicksort for good average-case performance
    2. Track recursion depth during the sort
    3. If recursion depth exceeds a threshold (typically 2*log₂(n)), switch to Heapsort to avoid Quicksort's worst case
    4. For small subarrays (typically less than 16 elements), use Insertion Sort
    
    ## Key Concepts
    
    ### Recursion Depth
    
    The maximum recursion depth is typically set to 2*log₂(n). This means that if Quicksort hasn't finished by this depth, the algorithm switches to Heapsort to guarantee O(n log n) worst-case performance.
    
    ### Small Subarray Threshold
    
    For small subarrays (typically less than 16 elements), Insertion Sort is used because it has less overhead than Quicksort or Heapsort for small inputs and performs well when data is nearly sorted.
    
    ### Pivot Selection
    
    Like Quicksort, the pivot selection strategy is critical. Common approaches include:
    - Median-of-three: Select the median of first, middle, and last elements
    - Random: Randomly select a pivot
    - Median-of-medians: More complex but gives better guarantees
    
    ## Visual Example
    
    Original array: `[9, 7, 5, 11, 12, 2, 14, 3, 10, 6]`
    
    **Start with Quicksort:**
    
    - Select pivot: 10 (median of 9, 11, 6)
    - Partition: `[9, 7, 5, 2, 3, 6] [10] [11, 12, 14]`
    
    **For the left partition `[9, 7, 5, 2, 3, 6]`:**
    
    - Select pivot: 7 (median of 9, 5, 6)
    - Partition: `[5, 2, 3, 6] [7] [9]`
    
    **For the subarray `[5, 2, 3, 6]`:**
    
    - Small enough for Insertion Sort
    - After sort: `[2, 3, 5, 6]`
    
    **Merge results:**
    
    - Sorted array: `[2, 3, 5, 6, 7, 9, 10, 11, 12, 14]`
    
    **Note:** If at any point the recursion depth exceeds 2*log₂(10) ≈ 6.64, the algorithm would switch to Heapsort for that subarray.

=== "💻 Implementation"

    ## Basic Introsort Implementation
    
    ```python
    def introsort(arr):
        """
        Introsort implementation
        Time: O(n log n)
        Space: O(log n)
        """
        # Calculate maximum recursion depth
        max_depth = 2 * (len(arr).bit_length())
        
        # Start the recursive introsort
        _introsort(arr, 0, len(arr) - 1, max_depth)
        
        return arr
        
    def _introsort(arr, start, end, max_depth):
        """Recursive introsort function"""
        # Size of the array
        size = end - start + 1
        
        # Use insertion sort for small arrays
        if size <= 16:
            insertion_sort(arr, start, end)
            return
            
        # Switch to heapsort if recursion too deep
        if max_depth == 0:
            heapsort(arr, start, end)
            return
            
        # Otherwise use quicksort
        pivot = median_of_three(arr, start, start + (size // 2), end)
        pivot_idx = partition(arr, start, end, pivot)
        
        # Recursively sort subarrays
        _introsort(arr, start, pivot_idx - 1, max_depth - 1)
        _introsort(arr, pivot_idx + 1, end, max_depth - 1)
        
    def median_of_three(arr, a, b, c):
        """Return the median of three elements"""
        if arr[a] < arr[b]:
            if arr[b] < arr[c]:
                return b  # a < b < c
            elif arr[a] < arr[c]:
                return c  # a < c < b
            else:
                return a  # c < a < b
        else:
            if arr[a] < arr[c]:
                return a  # b < a < c
            elif arr[b] < arr[c]:
                return c  # b < c < a
            else:
                return b  # c < b < a
        
    def partition(arr, start, end, pivot_idx):
        """Partition the array and return the pivot position"""
        pivot_value = arr[pivot_idx]
        
        # Move pivot to end
        arr[pivot_idx], arr[end] = arr[end], arr[pivot_idx]
        
        # Move all elements smaller than pivot to left side
        store_idx = start
        for i in range(start, end):
            if arr[i] < pivot_value:
                arr[store_idx], arr[i] = arr[i], arr[store_idx]
                store_idx += 1
                
        # Move pivot to its final place
        arr[store_idx], arr[end] = arr[end], arr[store_idx]
        
        return store_idx
        
    def insertion_sort(arr, start, end):
        """Insertion sort for small arrays"""
        for i in range(start + 1, end + 1):
            key = arr[i]
            j = i - 1
            while j >= start and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
            
    def heapsort(arr, start, end):
        """Heapsort algorithm"""
        # Build heap (rearrange array)
        n = end - start + 1
        
        # Build max heap
        for i in range(start + n//2 - 1, start - 1, -1):
            heapify(arr, n, i, start)
            
        # Extract elements from heap one by one
        for i in range(end, start, -1):
            arr[start], arr[i] = arr[i], arr[start]
            heapify(arr, i - start, start, start)
            
    def heapify(arr, n, i, offset):
        """
        Heapify a subtree rooted at index i
        n is the size of heap
        offset is the starting index in the array
        """
        largest = i
        left = 2 * i - offset + 1
        right = 2 * i - offset + 2
        
        # Check if left child exists and is greater than root
        if left < n + offset and arr[left] > arr[largest]:
            largest = left
            
        # Check if right child exists and is greater than root
        if right < n + offset and arr[right] > arr[largest]:
            largest = right
            
        # Change root if needed
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest, offset)
    ```
    
    ## Optimized Introsort with Dual-Pivot Quicksort
    
    ```python
    def optimized_introsort(arr):
        """
        Optimized Introsort using dual-pivot quicksort
        Time: O(n log n)
        Space: O(log n)
        """
        max_depth = 2 * (len(arr).bit_length())
        _optimized_introsort(arr, 0, len(arr) - 1, max_depth)
        return arr
        
    def _optimized_introsort(arr, start, end, max_depth):
        """Recursive optimized introsort function"""
        size = end - start + 1
        
        # Use insertion sort for small arrays
        if size <= 16:
            insertion_sort(arr, start, end)
            return
            
        # Switch to heapsort if recursion too deep
        if max_depth == 0:
            heapsort(arr, start, end)
            return
            
        # Use dual-pivot quicksort
        if size > 50:  # Only use dual-pivot for larger arrays
            dual_pivot_quicksort(arr, start, end, max_depth - 1)
        else:
            # Traditional quicksort for smaller arrays
            pivot = median_of_three(arr, start, start + (size // 2), end)
            pivot_idx = partition(arr, start, end, pivot)
            
            # Recursively sort subarrays
            _optimized_introsort(arr, start, pivot_idx - 1, max_depth - 1)
            _optimized_introsort(arr, pivot_idx + 1, end, max_depth - 1)
            
    def dual_pivot_quicksort(arr, start, end, max_depth):
        """Dual-pivot quicksort implementation"""
        if start >= end:
            return
            
        # Ensure pivot1 <= pivot2
        if arr[start] > arr[end]:
            arr[start], arr[end] = arr[end], arr[start]
            
        pivot1, pivot2 = arr[start], arr[end]
        
        # 3-way partition
        less, greater = start + 1, end - 1
        i = less
        
        while i <= greater:
            if arr[i] < pivot1:
                arr[i], arr[less] = arr[less], arr[i]
                less += 1
                i += 1
            elif arr[i] > pivot2:
                arr[i], arr[greater] = arr[greater], arr[i]
                greater -= 1
            else:
                i += 1
                
        # Move pivots to final positions
        arr[start], arr[less - 1] = arr[less - 1], arr[start]
        arr[end], arr[greater + 1] = arr[greater + 1], arr[end]
        
        # Recursively sort subarrays
        _optimized_introsort(arr, start, less - 2, max_depth)
        _optimized_introsort(arr, less, greater, max_depth)
        _optimized_introsort(arr, greater + 2, end, max_depth)
    ```

=== "📊 Performance Analysis"

    ## Time Complexity
    
    - **Best Case**: O(n log n)
    - **Average Case**: O(n log n)
    - **Worst Case**: O(n log n) - Guaranteed by switching to Heapsort
    
    ## Space Complexity
    
    - O(log n) - Stack space for recursion
    
    ## Comparative Advantages
    
    | Algorithm | Pros | Cons |
    |-----------|------|------|
    | **Introsort** | • Guaranteed O(n log n)<br>• Good average case<br>• In-place | • Not stable<br>• Complex implementation |
    | **Quicksort** | • Excellent average case<br>• Good locality | • O(n²) worst case<br>• Not stable |
    | **Heapsort** | • Guaranteed O(n log n)<br>• In-place | • Poor locality<br>• Not stable<br>• Slower than Quicksort in practice |
    | **Mergesort** | • Guaranteed O(n log n)<br>• Stable | • Not in-place<br>• O(n) extra space |
    | **Timsort** | • Adaptive<br>• Stable | • Not in-place<br>• More complex implementation |
    
    ## Cache Performance
    
    Introsort generally has good cache locality due to:
    
    1. Using Quicksort for most of the sorting, which has good locality
    2. Using Insertion Sort for small subarrays, which has excellent locality
    
    Compared to Heapsort, which has poor locality due to its tree-based structure, Introsort performs better on modern hardware.
    
    ## Switching Thresholds
    
    The performance of Introsort is influenced by two key thresholds:
    
    1. **Max Recursion Depth**: Typically set to 2*log₂(n)
    2. **Small Array Threshold**: Typically between 16-32 elements
    
    Optimal values may vary based on hardware and data characteristics.

=== "💡 Applications & Variations"

    ## Real-world Applications
    
    - **C++ Standard Library**: The `std::sort` function in the C++ STL uses Introsort
    
    - **High-performance Systems**: Used in applications where worst-case performance guarantees are critical
    
    - **Database Systems**: Some database engines use Introsort for query processing
    
    - **Compiler Optimizations**: Used in some compiler optimization phases
    
    ## C++ Implementation
    
    The C++ Standard Library's implementation of Introsort is highly optimized:
    
    ```cpp
    // Simplified version of C++ STL sort implementation
    template <typename RandomAccessIterator>
    void sort(RandomAccessIterator first, RandomAccessIterator last) {
        if (first != last) {
            // Calculate max depth
            int depth_limit = 2 * log2(last - first);
            _introsort(first, last, depth_limit);
        }
    }
    
    template <typename RandomAccessIterator>
    void _introsort(RandomAccessIterator first, 
                   RandomAccessIterator last, 
                   int depth_limit) {
        // Size of the range
        auto n = last - first;
        
        // Switch to insertion sort for small arrays
        if (n <= 16) {
            insertion_sort(first, last);
            return;
        }
        
        // Switch to heapsort if depth limit exceeded
        if (depth_limit == 0) {
            make_heap(first, last);
            sort_heap(first, last);
            return;
        }
        
        // Otherwise use quicksort
        auto pivot = median_of_three(first, first + n/2, last - 1);
        auto middle = partition(first, last, pivot);
        
        // Recursively sort subarrays
        _introsort(first, middle, depth_limit - 1);
        _introsort(middle, last, depth_limit - 1);
    }
    ```
    
    ## Variations
    
    ### 1. Introsort with Multi-Pivot Quicksort
    
    ```python
    def multi_pivot_introsort(arr):
        """Introsort using multi-pivot quicksort"""
        max_depth = 2 * (len(arr).bit_length())
        _multi_pivot_introsort(arr, 0, len(arr) - 1, max_depth)
        return arr
        
    def _multi_pivot_introsort(arr, start, end, max_depth):
        """Multi-pivot introsort recursive implementation"""
        size = end - start + 1
        
        # Use insertion sort for small arrays
        if size <= 16:
            insertion_sort(arr, start, end)
            return
            
        # Switch to heapsort if recursion too deep
        if max_depth == 0:
            heapsort(arr, start, end)
            return
            
        # Use 3-pivot quicksort for large arrays
        if size > 100:
            three_pivot_partition(arr, start, end, max_depth - 1)
        else:
            # Use regular quicksort for smaller arrays
            pivot_idx = partition(arr, start, end, median_of_three(
                arr, start, start + size//2, end))
            
            _multi_pivot_introsort(arr, start, pivot_idx - 1, max_depth - 1)
            _multi_pivot_introsort(arr, pivot_idx + 1, end, max_depth - 1)
    ```
    
    ### 2. Block-based Introsort
    
    ```python
    def block_introsort(arr):
        """
        Block-based Introsort that improves cache efficiency
        by processing elements in blocks
        """
        max_depth = 2 * (len(arr).bit_length())
        block_size = 128  # Optimize for L1 cache
        
        # Sort each block separately
        for i in range(0, len(arr), block_size):
            end_idx = min(i + block_size, len(arr))
            _introsort(arr, i, end_idx - 1, max_depth)
            
        # Merge blocks using a bottom-up approach
        curr_size = block_size
        while curr_size < len(arr):
            for start in range(0, len(arr), 2 * curr_size):
                mid = min(start + curr_size, len(arr))
                end = min(start + 2 * curr_size, len(arr))
                if mid < end:
                    merge(arr, start, mid, end)
            curr_size *= 2
            
        return arr
    ```
    
    ### 3. Parallel Introsort
    
    ```python
    def parallel_introsort(arr, num_threads=4):
        """
        Parallel implementation of Introsort using multiple threads
        """
        if len(arr) <= 1:
            return arr
            
        # For small arrays, just use sequential introsort
        if len(arr) < 10000 or num_threads <= 1:
            return introsort(arr)
            
        # Choose pivot and partition
        pivot_idx = median_of_three(arr, 0, len(arr)//2, len(arr)-1)
        pivot = arr[pivot_idx]
        
        # Partition (simplified version)
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        
        # Recursively sort in parallel
        # In practice, use proper threading library
        left_sorted = parallel_introsort(left, num_threads // 2)
        right_sorted = parallel_introsort(right, num_threads // 2)
        
        # Combine results
        return left_sorted + middle + right_sorted
    ```

=== "🎯 Practice Problems"

    ## Beginner Problems
    
    1. **Implement Introsort**
       - Create a basic implementation of Introsort
    
    2. **Compare Sorting Algorithms**
       - Compare the performance of Introsort with Quicksort, Heapsort, and Mergesort on different data distributions
    
    3. **Optimize Threshold Values**
       - Experiment with different threshold values for small arrays and maximum recursion depth
    
    ## Intermediate Problems
    
    1. **Custom Comparator Introsort**
       - Implement Introsort with a custom comparison function
    
    2. **Sort Nearly Sorted Array**
       - Use Introsort to efficiently sort an array where each element is at most k positions from its sorted position
       - [LeetCode #280: Wiggle Sort](https://leetcode.com/problems/wiggle-sort/)
    
    3. **External Introsort**
       - Adapt Introsort to work with data that doesn't fit in memory
    
    ## Advanced Problems
    
    1. **Parallel Introsort**
       - Implement a multi-threaded version of Introsort
    
    2. **Cache-Optimized Introsort**
       - Implement a version of Introsort optimized for modern cache hierarchies
    
    3. **Adaptive Introsort**
       - Create a version of Introsort that adapts its strategy based on the detected data pattern

# Radix Sort

Radix Sort is a non-comparative integer sorting algorithm that sorts data with integer keys by grouping the keys by individual digits that share the same position and value.

=== "📋 Algorithm Overview"

    ## Key Characteristics
    
    - **Time Complexity**: O(d × (n + k)) where d is number of digits, n is array size, and k is base (usually 10)
    - **Space Complexity**: O(n + k)
    - **Stable**: Yes
    - **In-place**: No
    - **Comparison-based**: No
    
    ## When to Use
    
    - Sorting integers or strings of fixed length
    - When data range is large but number of digits is small
    - When stability is required
    - When comparison-based sorts are too slow
    
    ## Limitations
    
    - Only works for digits, characters, or strings
    - Less efficient for small arrays
    - Requires extra space
    - Fixed-size alphabets work best

=== "🔄 How It Works"

    ## Algorithm Steps
    
    1. Find the maximum number to determine the number of digits
    2. For each digit position (starting from least significant digit):
       - Sort the input based on the current digit using a stable sort (typically counting sort)
    3. After sorting by each position, the array will be fully sorted
    
    ## Types of Radix Sort
    
    - **LSD (Least Significant Digit) Radix Sort**: Start from the rightmost digit (most common)
    - **MSD (Most Significant Digit) Radix Sort**: Start from the leftmost digit
    
    ## Visual Example
    
    Original array: `[170, 45, 75, 90, 802, 24, 2, 66]`
    
    **Sort by 1s place (least significant digit):**
    - After sorting: `[170, 90, 802, 2, 24, 45, 75, 66]`
    
    **Sort by 10s place:**
    - After sorting: `[802, 2, 24, 45, 66, 170, 75, 90]`
    
    **Sort by 100s place:**
    - Final sorted array: `[2, 24, 45, 66, 75, 90, 170, 802]`

=== "💻 Implementation"

    ## LSD Radix Sort Implementation
    
    ```python
    def radix_sort(arr):
        """
        LSD Radix Sort implementation
        Time: O(d * (n + k)) where d is the number of digits
        Space: O(n + k)
        """
        # Find the maximum number to determine number of digits
        max_val = max(arr)
        
        # Do counting sort for every digit
        # exp is 10^i where i is the current digit number
        exp = 1
        while max_val // exp > 0:
            counting_sort_by_digit(arr, exp)
            exp *= 10
            
        return arr
        
    def counting_sort_by_digit(arr, exp):
        """
        Counting sort by specific digit represented by exp
        exp is 10^i where i is the digit position from right
        """
        n = len(arr)
        output = [0] * n
        count = [0] * 10  # 0-9 digits
        
        # Count occurrences of each digit
        for i in range(n):
            digit = (arr[i] // exp) % 10
            count[digit] += 1
            
        # Change count[i] so that it contains the position
        # of this digit in output array
        for i in range(1, 10):
            count[i] += count[i - 1]
            
        # Build the output array
        for i in range(n-1, -1, -1):
            digit = (arr[i] // exp) % 10
            output[count[digit] - 1] = arr[i]
            count[digit] -= 1
            
        # Copy the output array to arr
        for i in range(n):
            arr[i] = output[i]
    ```
    
    ## MSD Radix Sort Implementation
    
    ```python
    def msd_radix_sort(arr):
        """
        MSD Radix Sort implementation
        Time: O(d * (n + k)) where d is the number of digits
        Space: O(n + k)
        """
        # Find the maximum number to determine number of digits
        if not arr:
            return arr
            
        max_val = max(arr)
        # Find number of digits in max_val
        num_digits = len(str(max_val))
        
        # Calculate the largest place value
        max_exp = 10 ** (num_digits - 1)
        
        return _msd_radix_sort(arr, 0, len(arr) - 1, max_exp)
        
    def _msd_radix_sort(arr, start, end, exp):
        """Recursive MSD radix sort helper"""
        if start >= end or exp <= 0:
            return arr
            
        # Use counting sort for the current digit
        counts = [0] * 10
        output = [0] * (end - start + 1)
        
        # Count occurrences
        for i in range(start, end + 1):
            digit = (arr[i] // exp) % 10
            counts[digit] += 1
            
        # Calculate positions
        for i in range(1, 10):
            counts[i] += counts[i - 1]
            
        # Build output array
        for i in range(end, start - 1, -1):
            digit = (arr[i] // exp) % 10
            output[counts[digit] - 1] = arr[i]
            counts[digit] -= 1
            
        # Copy back to original array
        for i in range(end - start + 1):
            arr[start + i] = output[i]
            
        # Recursively sort each group
        new_exp = exp // 10
        if new_exp > 0:
            index = start
            for i in range(10):
                new_start = start + (0 if i == 0 else counts[i-1])
                new_end = start + counts[i] - 1
                if new_start <= new_end:
                    _msd_radix_sort(arr, new_start, new_end, new_exp)
                    
        return arr
    ```
    
    ## String Radix Sort
    
    ```python
    def string_radix_sort(strings):
        """
        Radix sort for strings of different lengths
        """
        if not strings:
            return strings
            
        # Find maximum string length
        max_len = max(len(s) for s in strings)
        
        # Pad shorter strings with null character
        # and convert to character codes
        padded = [(s.ljust(max_len, '\0'), s) for s in strings]
        
        # Sort by each character position from right to left
        for i in range(max_len-1, -1, -1):
            # Stable sort on the current character
            padded = sorted(padded, key=lambda x: ord(x[0][i]))
            
        # Return original strings in sorted order
        return [original for _, original in padded]
    ```

=== "📊 Performance Analysis"

    ## Time Complexity
    
    - **Best Case**: O(d × (n + k)) - Always the same
    - **Average Case**: O(d × (n + k)) - Always the same
    - **Worst Case**: O(d × (n + k)) - Always the same
    
    Where:
    - d = number of digits or characters
    - n = array size
    - k = base (radix) size, typically 10 for decimal, 26 for English alphabet
    
    ## Space Complexity
    
    - O(n + k) for the output and counting arrays
    
    ## Stability
    
    Radix sort is stable if the underlying digit sort (typically counting sort) is stable. This means that elements with the same key maintain their relative order in the sorted output.
    
    ## Comparative Analysis
    
    - **vs. Quicksort/Mergesort**: Radix sort can be faster for large n when d is small
    - **vs. Counting Sort**: More versatile for large ranges of values
    - **vs. Bucket Sort**: Better for integers; bucket sort is better for floating-point
    
    **Breaking the O(n log n) Barrier**:
    
    Radix sort is one of the few sorting algorithms that can beat the O(n log n) lower bound that applies to comparison-based sorting algorithms.

=== "💡 Applications & Variations"

    ## Real-world Applications
    
    - **String sorting**: Especially useful for fixed-length strings like:
      - IP addresses
      - Dates in YYYYMMDD format
      - Fixed-length IDs or codes
    
    - **Large integer sorting**: When the range is too large for counting sort
    
    - **Suffix array construction**: Used in text processing algorithms
    
    - **External sorting**: When data doesn't fit in memory
    
    ## Variations
    
    ### 1. Parallel Radix Sort
    
    ```python
    # Pseudocode for parallel radix sort
    def parallel_radix_sort(arr, num_threads):
        # Find the maximum number to determine number of digits
        max_val = max(arr)
        
        # Calculate the number of digits
        exp = 1
        while max_val // exp > 0:
            # Partition the array for parallel processing
            chunk_size = len(arr) // num_threads
            threads = []
            
            # Create threads for each chunk
            for i in range(num_threads):
                start = i * chunk_size
                end = start + chunk_size if i < num_threads - 1 else len(arr)
                thread = Thread(target=counting_sort_by_digit, 
                               args=(arr, exp, start, end))
                threads.append(thread)
                thread.start()
                
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
                
            # Merge the sorted chunks
            merge_chunks(arr, num_threads, chunk_size)
            
            exp *= 10
    ```
    
    ### 2. American Flag Sort
    
    A more cache-efficient variant of MSD radix sort:
    
    ```python
    def american_flag_sort(arr, max_val):
        """
        American Flag Sort - a distribution-based variant of radix sort
        Performs better than traditional radix sort due to improved cache locality
        """
        # Calculate number of bits needed
        bits = max_val.bit_length()
        msb = 1 << (bits - 1)
        
        # Start the recursive sort
        _american_flag_sort(arr, 0, len(arr), msb)
        
    def _american_flag_sort(arr, start, end, bit):
        if start >= end - 1 or bit == 0:
            return
            
        # Two passes: count and distribute
        zeros = 0
        for i in range(start, end):
            if (arr[i] & bit) == 0:
                zeros += 1
                
        # Distribute
        i, j = start, start + zeros
        while i < j and j < end:
            if (arr[i] & bit) == 0:
                i += 1
            else:
                arr[i], arr[j] = arr[j], arr[i]
                j += 1
                
        # Recursively sort both partitions
        _american_flag_sort(arr, start, start + zeros, bit >> 1)
        _american_flag_sort(arr, start + zeros, end, bit >> 1)
    ```

=== "🎯 Practice Problems"

    ## Beginner Problems
    
    1. **Sort Array of Integers**
       - Implement radix sort to sort an array of non-negative integers
    
    2. **Sort Strings by Length**
       - Sort strings first by length and then lexicographically
    
    3. **Sort Dates**
       - Sort dates in YYYYMMDD format using radix sort
    
    ## Intermediate Problems
    
    1. **Maximum Gap**
       - Find the maximum difference between successive elements after sorting
       - [LeetCode #164: Maximum Gap](https://leetcode.com/problems/maximum-gap/)
    
    2. **Sort Array by Increasing Frequency**
       - Sort elements by frequency, with ties broken by the element's value
       - [LeetCode #1636: Sort Array by Increasing Frequency](https://leetcode.com/problems/sort-array-by-increasing-frequency/)
    
    3. **Sort Characters By Frequency**
       - Sort characters by decreasing frequency
       - [LeetCode #451: Sort Characters By Frequency](https://leetcode.com/problems/sort-characters-by-frequency/)
    
    ## Advanced Problems
    
    1. **Suffix Array Construction**
       - Implement a suffix array construction algorithm using radix sort
    
    2. **External Radix Sort**
       - Implement radix sort for data that doesn't fit in memory
    
    3. **Sort Very Large Numbers**
       - Sort very large numbers represented as strings using radix sort

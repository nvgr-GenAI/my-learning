# Counting Sort

Counting Sort is an integer sorting algorithm that operates by counting the number of objects that possess distinct key values, and applying arithmetic to determine their positions in the output sequence.

=== "📋 Algorithm Overview"

    ## Key Characteristics
    
    - **Time Complexity**: O(n + k) where k is the range of input
    - **Space Complexity**: O(n + k)
    - **Stable**: Yes (when implemented correctly)
    - **In-place**: No
    - **Comparison-based**: No
    
    ## When to Use
    
    - When sorting integers with a small range
    - When stability is required
    - When optimal linear time is needed
    
    ## Limitations
    
    - Only works for integers or data that can be mapped to integers
    - Inefficient when the range of values (k) is very large
    - Requires additional space proportional to the range

=== "🔄 How It Works"

    ## Algorithm Steps
    
    1. Find the range of input elements (min to max)
    2. Create a counting array of size (max - min + 1)
    3. Count the occurrence of each element in the input
    4. Modify the counting array to store cumulative counts
    5. Build the output array using the modified counting array
    6. Copy the output array back to the original array
    
    ## Visual Example
    
    Original array: `[4, 2, 2, 8, 3, 3, 1]`
    
    1. Range: 1 to 8
    2. Counting array: `[0, 0, 0, 0, 0, 0, 0, 0]`
    3. After counting: `[0, 1, 2, 2, 1, 0, 0, 1]` (indexes 0-7 for values 1-8)
    4. Cumulative counts: `[0, 1, 3, 5, 6, 6, 6, 7]`
    5. Build output array: `[1, 2, 2, 3, 3, 4, 8]`

=== "💻 Implementation"

    ## Basic Implementation
    
    ```python
    def counting_sort(arr):
        """
        Counting Sort implementation for non-negative integers
        Time: O(n + k), Space: O(n + k) where k is the range
        """
        # Find the maximum element
        max_val = max(arr)
        
        # Create a count array for 0 to max_val
        count = [0] * (max_val + 1)
        
        # Store count of each element
        for num in arr:
            count[num] += 1
        
        # Modify count array to store cumulative counts
        for i in range(1, len(count)):
            count[i] += count[i - 1]
        
        # Create output array
        output = [0] * len(arr)
        
        # Build the output array
        for i in range(len(arr) - 1, -1, -1):
            output[count[arr[i]] - 1] = arr[i]
            count[arr[i]] -= 1
        
        # Copy output array to original array
        for i in range(len(arr)):
            arr[i] = output[i]
            
        return arr
    ```
    
    ## Extended Implementation (for negative numbers)
    
    ```python
    def counting_sort_extended(arr):
        """
        Extended counting sort that handles negative integers
        Time: O(n + k), Space: O(n + k) where k is the range
        """
        if not arr:
            return arr
            
        # Find min and max values
        min_val = min(arr)
        max_val = max(arr)
        
        # Range of values
        range_of_elements = max_val - min_val + 1
        
        # Create count array and output array
        count = [0] * range_of_elements
        output = [0] * len(arr)
        
        # Store count of each element
        for num in arr:
            count[num - min_val] += 1
        
        # Modify count to store cumulative positions
        for i in range(1, len(count)):
            count[i] += count[i - 1]
        
        # Build the output array
        for i in range(len(arr) - 1, -1, -1):
            output[count[arr[i] - min_val] - 1] = arr[i]
            count[arr[i] - min_val] -= 1
        
        # Copy output array to original array
        for i in range(len(arr)):
            arr[i] = output[i]
            
        return arr
    ```

=== "📊 Performance Analysis"

    ## Time Complexity
    
    - **Best Case**: O(n + k) - Always the same regardless of input order
    - **Average Case**: O(n + k) - Always the same regardless of input order
    - **Worst Case**: O(n + k) - Always the same regardless of input order
    
    Where n is the number of elements and k is the range of values.
    
    ## Space Complexity
    
    - O(n + k) where n is for the output array and k is for the counting array
    
    ## Stability
    
    Counting sort is stable if implemented correctly (processing elements from right to left in the final placement step). This means that equal elements retain their relative order in the sorted output.
    
    ## Comparative Advantages
    
    - **vs. Quick Sort**: Faster when k is small, as it avoids comparisons
    - **vs. Radix Sort**: Simpler implementation but less efficient when range is large
    - **vs. Bucket Sort**: Better for discrete integers; bucket sort works better for floats

=== "💡 Applications & Variations"

    ## Real-world Applications
    
    - **Sorting student scores**: When scores are in a fixed range (0-100)
    - **Character frequency analysis**: For text processing and compression
    - **Radix sort subroutine**: As a stable sort within each digit pass
    - **Sorting postal codes**: When dealing with finite discrete values
    
    ## Variations
    
    ### 1. Generic Counting Sort
    
    Handle objects with integer keys:
    
    ```python
    def counting_sort_objects(objects, key_func, max_key):
        """Sort objects by their integer keys using counting sort"""
        count = [0] * (max_key + 1)
        output = [None] * len(objects)
        
        # Count occurrences of each key
        for obj in objects:
            key = key_func(obj)
            count[key] += 1
        
        # Cumulative counts
        for i in range(1, len(count)):
            count[i] += count[i - 1]
        
        # Build output array
        for i in range(len(objects) - 1, -1, -1):
            key = key_func(objects[i])
            output[count[key] - 1] = objects[i]
            count[key] -= 1
            
        return output
    ```
    
    ### 2. In-place Counting Sort
    
    A variation that attempts to minimize additional space:
    
    ```python
    def counting_sort_inplace(arr, max_val):
        """
        Partially in-place counting sort (still requires count array)
        """
        counts = [0] * (max_val + 1)
        
        # Count occurrences
        for x in arr:
            counts[x] += 1
            
        # Overwrite original array with sorted values
        i = 0
        for val in range(max_val + 1):
            for _ in range(counts[val]):
                arr[i] = val
                i += 1
                
        return arr
    ```

=== "🎯 Practice Problems"

    ## Beginner Problems
    
    1. **Sort Array of 0s, 1s, and 2s**
       - Sort an array containing only 0s, 1s, and 2s in linear time
       - [LeetCode #75: Sort Colors](https://leetcode.com/problems/sort-colors/)
    
    2. **Find the Most Frequent Element**
       - Use counting sort to find the most frequent element in an array
    
    3. **Find Missing Numbers**
       - Given an array of numbers from 1 to n with some missing, find all missing numbers
       - [LeetCode #448: Find All Numbers Disappeared in an Array](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/)
    
    ## Intermediate Problems
    
    1. **Relative Sort Array**
       - Sort an array according to the order defined by another array
       - [LeetCode #1122: Relative Sort Array](https://leetcode.com/problems/relative-sort-array/)
    
    2. **Maximum Gap**
       - Find maximum difference between successive elements in a sorted form
       - [LeetCode #164: Maximum Gap](https://leetcode.com/problems/maximum-gap/)
    
    3. **H-Index Calculation**
       - Calculate H-index of citations using counting sort
       - [LeetCode #274: H-Index](https://leetcode.com/problems/h-index/)
    
    ## Advanced Problems
    
    1. **Sorting with Custom Comparator**
       - Implement counting sort with a custom comparator function
    
    2. **K Closest Elements**
       - Find k closest elements to a given value in a sorted array
       - [LeetCode #658: Find K Closest Elements](https://leetcode.com/problems/find-k-closest-elements/)

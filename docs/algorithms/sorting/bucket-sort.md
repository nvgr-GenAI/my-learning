# Bucket Sort

Bucket Sort is a distribution-based sorting algorithm that works by distributing elements into a number of buckets, then sorting these buckets individually and finally merging them to produce a sorted array.

=== "📋 Algorithm Overview"

    ## Key Characteristics
    
    - **Time Complexity**: 
      - Average: O(n + k) where k is the number of buckets
      - Worst: O(n²) when elements are clustered in a single bucket
    - **Space Complexity**: O(n + k)
    - **Stable**: Yes (if the underlying sort is stable)
    - **In-place**: No
    - **Comparison-based**: Depends on the algorithm used to sort individual buckets
    
    ## When to Use
    
    - When input is uniformly distributed over a range
    - For floating-point numbers in a known range
    - When linear-time sorting is needed and data distribution is favorable
    - As an external sorting algorithm
    
    ## Limitations
    
    - Inefficient when data is not uniformly distributed
    - Requires knowledge about data distribution
    - Extra space overhead for buckets
    - Performance heavily dependent on the hash/bucket function

=== "🔄 How It Works"

    ## Algorithm Steps
    
    1. Create n empty buckets (or fewer based on the range)
    2. Scatter: Place each element into its corresponding bucket based on a bucket function
    3. Sort each non-empty bucket (using any sorting algorithm or recursively using bucket sort)
    4. Gather: Visit the buckets in order and put all elements back into the original array
    
    ## Visual Example
    
    Original array: `[0.42, 0.32, 0.33, 0.52, 0.37, 0.47, 0.51]`
    
    **Scatter phase (with 5 buckets):**
    - Bucket 0 (0.0-0.2): Empty
    - Bucket 1 (0.2-0.4): [0.32, 0.33, 0.37]
    - Bucket 2 (0.4-0.6): [0.42, 0.47, 0.52, 0.51]
    - Bucket 3 (0.6-0.8): Empty
    - Bucket 4 (0.8-1.0): Empty
    
    **Sort individual buckets:**
    - Bucket 1: [0.32, 0.33, 0.37]
    - Bucket 2: [0.42, 0.47, 0.51, 0.52]
    
    **Gather phase:**
    - Final sorted array: [0.32, 0.33, 0.37, 0.42, 0.47, 0.51, 0.52]

=== "💻 Implementation"

    ## Basic Bucket Sort for Floating-Point Numbers
    
    ```python
    def bucket_sort(arr):
        """
        Bucket Sort implementation for floating-point numbers in range [0,1)
        Time: Average O(n + k), Worst O(n²)
        Space: O(n + k)
        """
        if not arr:
            return arr
            
        # Create n empty buckets
        n = len(arr)
        buckets = [[] for _ in range(n)]
        
        # Put elements into buckets
        for i in range(n):
            # Use value as index for bucket
            # multiply by n to get bucket index between 0 and n-1
            index = int(n * arr[i])
            buckets[index].append(arr[i])
        
        # Sort individual buckets
        for i in range(n):
            buckets[i].sort()  # Using Python's built-in sort
        
        # Concatenate all buckets back into arr
        result = []
        for b in buckets:
            result.extend(b)
            
        return result
    ```
    
    ## Generic Bucket Sort for Integer Range
    
    ```python
    def bucket_sort_integers(arr, min_val=None, max_val=None, num_buckets=10):
        """
        Bucket Sort implementation for integers in any range
        Time: Average O(n + k), Worst O(n²)
        Space: O(n + k)
        """
        if not arr:
            return arr
            
        # Find range if not provided
        if min_val is None:
            min_val = min(arr)
        if max_val is None:
            max_val = max(arr)
            
        # Create buckets
        range_size = max(1, (max_val - min_val) // num_buckets + 1)
        buckets = [[] for _ in range(num_buckets)]
        
        # Distribute elements into buckets
        for num in arr:
            index = min(num_buckets - 1, (num - min_val) // range_size)
            buckets[index].append(num)
        
        # Sort individual buckets and concatenate
        result = []
        for bucket in buckets:
            bucket.sort()
            result.extend(bucket)
            
        return result
    ```
    
    ## Bucket Sort with Custom Bucket Function
    
    ```python
    def bucket_sort_custom(arr, bucket_function, num_buckets, sort_function=None):
        """
        Bucket Sort with custom bucket assignment function
        Time: Depends on bucket and sort functions
        Space: O(n + k)
        """
        if not arr:
            return arr
            
        if sort_function is None:
            sort_function = sorted
            
        # Create buckets
        buckets = [[] for _ in range(num_buckets)]
        
        # Place elements into buckets using custom function
        for item in arr:
            bucket_index = bucket_function(item, num_buckets)
            buckets[bucket_index].append(item)
        
        # Sort individual buckets and concatenate
        result = []
        for bucket in buckets:
            result.extend(sort_function(bucket))
            
        return result
    
    # Example usage:
    # bucket_sort_custom(data, lambda x, n: hash(x) % n, 10)
    ```

=== "📊 Performance Analysis"

    ## Time Complexity
    
    - **Best Case**: O(n) - When elements are perfectly distributed and each bucket has ≤ 1 element
    - **Average Case**: O(n + k) - When elements are uniformly distributed
    - **Worst Case**: O(n²) - When all elements are placed in a single bucket
    
    Where:
    - n = array size
    - k = number of buckets
    
    ## Space Complexity
    
    - O(n + k) for the buckets and temporary storage
    
    ## Distribution Impact
    
    The performance of bucket sort heavily depends on the distribution of the input data:
    
    | Data Distribution | Performance |
    |-------------------|-------------|
    | Uniform           | Excellent - O(n) |
    | Normal            | Good - O(n + k) |
    | Skewed/Clustered  | Poor - O(n²) |
    
    ## Bucket Size Trade-offs
    
    | Number of Buckets | Benefits | Drawbacks |
    |-------------------|----------|-----------|
    | Many small buckets | Better distribution | Higher overhead |
    | Few large buckets | Lower overhead | More work sorting each bucket |
    
    Optimal bucket number is often close to the array size (n).
    
    ## Stability
    
    Bucket sort is stable if the algorithm used to sort individual buckets is stable.

=== "💡 Applications & Variations"

    ## Real-world Applications
    
    - **Database systems**: For sorting records by a numeric key
    - **Geographic information systems**: Sorting locations by coordinates
    - **External sorting**: When data doesn't fit in memory
    - **Image processing**: Color quantization and histograms
    - **Network routing tables**: IP address sorting
    
    ## Variations
    
    ### 1. Generic Bucket Sort
    
    Uses a hash function to map elements to buckets:
    
    ```python
    def generic_bucket_sort(arr, hash_function, num_buckets):
        """Bucket sort using a custom hash function"""
        buckets = [[] for _ in range(num_buckets)]
        
        for item in arr:
            buckets[hash_function(item) % num_buckets].append(item)
            
        result = []
        for bucket in buckets:
            # Use insertion sort for each bucket (good for small arrays)
            insertion_sort(bucket)
            result.extend(bucket)
            
        return result
    ```
    
    ### 2. Recursive Bucket Sort
    
    Uses recursion for large buckets:
    
    ```python
    def recursive_bucket_sort(arr, min_val, max_val, depth=0, max_depth=2):
        """Recursive bucket sort for better handling of skewed distributions"""
        if len(arr) <= 1 or depth >= max_depth:
            return sorted(arr)
            
        num_buckets = min(len(arr), 10)  # Adjust based on array size
        range_size = (max_val - min_val) / num_buckets
        buckets = [[] for _ in range(num_buckets)]
        
        # Distribute elements
        for item in arr:
            bucket_idx = min(num_buckets - 1, 
                             int((item - min_val) / range_size))
            buckets[bucket_idx].append(item)
        
        # Recursively sort large buckets
        result = []
        for i, bucket in enumerate(buckets):
            if len(bucket) > 10:  # Threshold for recursion
                bucket_min = min_val + i * range_size
                bucket_max = bucket_min + range_size
                result.extend(recursive_bucket_sort(
                    bucket, bucket_min, bucket_max, depth + 1, max_depth))
            else:
                result.extend(sorted(bucket))
                
        return result
    ```
    
    ### 3. Parallel Bucket Sort
    
    ```python
    def parallel_bucket_sort(arr, num_buckets, num_threads):
        """Parallel implementation of bucket sort using threads"""
        # Find range
        min_val, max_val = min(arr), max(arr)
        range_size = (max_val - min_val) / num_buckets
        
        # Create buckets and distribute elements
        buckets = [[] for _ in range(num_buckets)]
        for item in arr:
            bucket_idx = min(num_buckets - 1, 
                             int((item - min_val) / range_size))
            buckets[bucket_idx].append(item)
            
        # Sort buckets in parallel
        def sort_bucket(bucket):
            bucket.sort()
            
        threads = []
        for bucket in buckets:
            if bucket:  # Only process non-empty buckets
                thread = Thread(target=sort_bucket, args=(bucket,))
                threads.append(thread)
                thread.start()
                
                # Limit active threads
                if len(threads) >= num_threads:
                    threads[0].join()
                    threads.pop(0)
                    
        # Wait for remaining threads
        for thread in threads:
            thread.join()
            
        # Gather results
        result = []
        for bucket in buckets:
            result.extend(bucket)
            
        return result
    ```

=== "🎯 Practice Problems"

    ## Beginner Problems
    
    1. **Sort Array of Floating-Point Numbers**
       - Implement bucket sort to sort an array of floating-point numbers between 0 and 1
    
    2. **Sort Strings by Length**
       - Use bucket sort where the bucket index is determined by string length
    
    3. **Sort Student Records by Score**
       - Sort student objects by their test scores (0-100) using bucket sort
    
    ## Intermediate Problems
    
    1. **Top K Frequent Elements**
       - Find the k most frequent elements in an array
       - [LeetCode #347: Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)
    
    2. **Sort Characters By Frequency**
       - Sort characters by decreasing frequency
       - [LeetCode #451: Sort Characters By Frequency](https://leetcode.com/problems/sort-characters-by-frequency/)
    
    3. **Maximum Gap**
       - Find the maximum difference between successive elements after sorting
       - [LeetCode #164: Maximum Gap](https://leetcode.com/problems/maximum-gap/)
    
    ## Advanced Problems
    
    1. **External Bucket Sort**
       - Implement bucket sort for data that doesn't fit in memory
    
    2. **Adaptive Bucket Sort**
       - Implement a bucket sort algorithm that adapts bucket sizes based on data distribution
    
    3. **2D Bucket Sort**
       - Sort points on a 2D plane using bucket sort (use grid cells as buckets)

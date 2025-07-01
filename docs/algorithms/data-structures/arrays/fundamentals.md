# Array Fundamentals

## 🔍 Core Concepts

Arrays are one of the most fundamental data structures in computer science. Understanding arrays is crucial for solving algorithmic problems effectively.

---

## 📊 What is an Array?

An **array** is a collection of elements stored in contiguous memory locations. Each element can be accessed directly using its index.

### Key Characteristics

- **Fixed Size**: Traditional arrays have a fixed size determined at creation
- **Homogeneous**: All elements are of the same data type
- **Indexed**: Elements are accessed using zero-based indexing
- **Contiguous Memory**: Elements are stored consecutively in memory

### Visual Representation

```text
Array: [10, 20, 30, 40, 50]
Index:  0   1   2   3   4

Memory Layout:
┌────┬────┬────┬────┬────┐
│ 10 │ 20 │ 30 │ 40 │ 50 │
└────┴────┴────┴────┴────┘
```

---

## ⏱️ Time Complexities

Understanding time complexities is crucial for choosing the right approach.

### Basic Operations

| Operation | Time Complexity | Description |
|-----------|-----------------|-------------|
| **Access** | O(1) | Direct access using index |
| **Search** | O(n) | Linear search through elements |
| **Insertion** | O(n) | May need to shift elements |
| **Deletion** | O(n) | May need to shift elements |

### Detailed Analysis

```python
def array_operations_analysis():
    """Demonstrate array operation complexities."""
    arr = [1, 2, 3, 4, 5]
    
    # O(1) - Direct access
    element = arr[2]  # Gets 3 instantly
    
    # O(n) - Linear search
    def search(arr, target):
        for i, val in enumerate(arr):  # May check all elements
            if val == target:
                return i
        return -1
    
    # O(n) - Insertion (worst case)
    def insert_at_beginning(arr, val):
        # Need to shift all elements right
        arr.insert(0, val)  # Expensive operation
    
    # O(1) - Insertion at end (if space available)
    def insert_at_end(arr, val):
        arr.append(val)  # Just add to end
    
    return arr
```

---

## 💾 Memory Layout

Understanding memory layout helps optimize performance.

### Memory Efficiency

```python
import sys

def memory_analysis():
    """Analyze memory usage of different array types."""
    
    # Python list (dynamic array)
    python_list = [1, 2, 3, 4, 5]
    print(f"Python list size: {sys.getsizeof(python_list)} bytes")
    
    # Array module (fixed type)
    import array
    int_array = array.array('i', [1, 2, 3, 4, 5])
    print(f"Array module size: {sys.getsizeof(int_array)} bytes")
    
    # NumPy array (scientific computing)
    import numpy as np
    numpy_array = np.array([1, 2, 3, 4, 5])
    print(f"NumPy array size: {numpy_array.nbytes} bytes")
```

### Cache Performance

Arrays have excellent **cache locality** because elements are stored contiguously:

```python
def cache_friendly_vs_unfriendly():
    """Demonstrate cache-friendly array access."""
    matrix = [[i * j for j in range(1000)] for i in range(1000)]
    
    # Cache-friendly: row-major access
    def sum_row_major(matrix):
        total = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                total += matrix[i][j]  # Accesses consecutive memory
        return total
    
    # Less cache-friendly: column-major access
    def sum_column_major(matrix):
        total = 0
        for j in range(len(matrix[0])):
            for i in range(len(matrix)):
                total += matrix[i][j]  # Jumps around in memory
        return total
    
    return sum_row_major(matrix), sum_column_major(matrix)
```

---

## 🔧 Common Operations

Let's explore fundamental array operations with implementations.

### 1. Array Creation and Initialization

```python
def array_creation_methods():
    """Different ways to create and initialize arrays."""
    
    # Method 1: List comprehension
    squares = [i**2 for i in range(10)]
    
    # Method 2: Using range and list
    zeros = [0] * 10
    
    # Method 3: Generator expression
    evens = list(i for i in range(20) if i % 2 == 0)
    
    # Method 4: Using numpy (for numerical computations)
    import numpy as np
    numpy_arr = np.zeros(10)  # Array of zeros
    numpy_arr2 = np.arange(10)  # Array [0, 1, 2, ..., 9]
    
    return squares, zeros, evens, numpy_arr, numpy_arr2
```

### 2. Array Traversal

```python
def array_traversal_methods(arr):
    """Different ways to traverse arrays."""
    
    # Method 1: Index-based traversal
    print("Index-based:")
    for i in range(len(arr)):
        print(f"arr[{i}] = {arr[i]}")
    
    # Method 2: Direct element traversal
    print("\nDirect element:")
    for element in arr:
        print(element)
    
    # Method 3: Enumerate (index + element)
    print("\nWith enumerate:")
    for i, element in enumerate(arr):
        print(f"Index {i}: {element}")
    
    # Method 4: Reverse traversal
    print("\nReverse traversal:")
    for i in range(len(arr) - 1, -1, -1):
        print(arr[i])
```

### 3. Array Searching

```python
def searching_algorithms(arr, target):
    """Implement different search algorithms."""
    
    # Linear Search - O(n)
    def linear_search(arr, target):
        for i, val in enumerate(arr):
            if val == target:
                return i
        return -1
    
    # Binary Search - O(log n) for sorted arrays
    def binary_search(arr, target):
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1
    
    # First and last occurrence
    def find_first_last(arr, target):
        first = last = -1
        
        # Find first occurrence
        for i in range(len(arr)):
            if arr[i] == target:
                first = i
                break
        
        # Find last occurrence
        for i in range(len(arr) - 1, -1, -1):
            if arr[i] == target:
                last = i
                break
        
        return first, last
    
    return (linear_search(arr, target), 
            binary_search(sorted(arr), target),
            find_first_last(arr, target))
```

---

## 🎯 Essential Patterns

Master these patterns to solve array problems efficiently.

### 1. Two Pointers Technique

```python
def two_pointers_examples():
    """Demonstrate two pointers pattern."""
    
    # Pattern 1: Opposite ends
    def is_palindrome(arr):
        left, right = 0, len(arr) - 1
        
        while left < right:
            if arr[left] != arr[right]:
                return False
            left += 1
            right -= 1
        
        return True
    
    # Pattern 2: Same direction (slow/fast)
    def remove_duplicates(arr):
        if not arr:
            return 0
        
        slow = 0
        for fast in range(1, len(arr)):
            if arr[fast] != arr[slow]:
                slow += 1
                arr[slow] = arr[fast]
        
        return slow + 1
    
    # Pattern 3: Two arrays
    def merge_sorted_arrays(arr1, arr2):
        result = []
        i = j = 0
        
        while i < len(arr1) and j < len(arr2):
            if arr1[i] <= arr2[j]:
                result.append(arr1[i])
                i += 1
            else:
                result.append(arr2[j])
                j += 1
        
        # Add remaining elements
        result.extend(arr1[i:])
        result.extend(arr2[j:])
        
        return result
    
    return is_palindrome, remove_duplicates, merge_sorted_arrays
```

### 2. Sliding Window Technique

```python
def sliding_window_examples():
    """Demonstrate sliding window pattern."""
    
    # Fixed size window
    def max_sum_subarray(arr, k):
        if len(arr) < k:
            return None
        
        # Calculate sum of first window
        window_sum = sum(arr[:k])
        max_sum = window_sum
        
        # Slide the window
        for i in range(k, len(arr)):
            window_sum = window_sum - arr[i - k] + arr[i]
            max_sum = max(max_sum, window_sum)
        
        return max_sum
    
    # Variable size window
    def longest_subarray_with_sum(arr, target_sum):
        left = 0
        current_sum = 0
        max_length = 0
        
        for right in range(len(arr)):
            current_sum += arr[right]
            
            # Shrink window if sum exceeds target
            while current_sum > target_sum and left <= right:
                current_sum -= arr[left]
                left += 1
            
            # Update max length if sum equals target
            if current_sum == target_sum:
                max_length = max(max_length, right - left + 1)
        
        return max_length
    
    return max_sum_subarray, longest_subarray_with_sum
```

### 3. Prefix Sum Technique

```python
def prefix_sum_examples():
    """Demonstrate prefix sum pattern."""
    
    def build_prefix_sum(arr):
        """Build prefix sum array."""
        prefix = [0] * (len(arr) + 1)
        
        for i in range(len(arr)):
            prefix[i + 1] = prefix[i] + arr[i]
        
        return prefix
    
    def range_sum_query(arr, queries):
        """Answer range sum queries efficiently."""
        prefix = build_prefix_sum(arr)
        results = []
        
        for left, right in queries:
            # Sum from left to right (inclusive)
            range_sum = prefix[right + 1] - prefix[left]
            results.append(range_sum)
        
        return results
    
    def subarray_sum_equals_k(arr, k):
        """Count subarrays with sum equal to k."""
        count = 0
        prefix_sum = 0
        sum_count = {0: 1}  # Handle subarrays starting from index 0
        
        for num in arr:
            prefix_sum += num
            
            # Check if (prefix_sum - k) exists
            if prefix_sum - k in sum_count:
                count += sum_count[prefix_sum - k]
            
            # Update sum_count
            sum_count[prefix_sum] = sum_count.get(prefix_sum, 0) + 1
        
        return count
    
    return build_prefix_sum, range_sum_query, subarray_sum_equals_k
```

---

## 🎪 Practice Problems

Let's apply these concepts with some fundamental problems.

### Problem 1: Array Rotation

```python
def rotate_array_examples():
    """Different ways to rotate arrays."""
    
    def rotate_right_simple(arr, k):
        """Rotate array right by k positions - O(n) space."""
        n = len(arr)
        k = k % n  # Handle k > n
        return arr[-k:] + arr[:-k]
    
    def rotate_right_inplace(arr, k):
        """Rotate array right by k positions - O(1) space."""
        n = len(arr)
        k = k % n
        
        # Reverse entire array
        arr.reverse()
        
        # Reverse first k elements
        arr[:k] = reversed(arr[:k])
        
        # Reverse remaining elements
        arr[k:] = reversed(arr[k:])
        
        return arr
    
    def rotate_left(arr, k):
        """Rotate array left by k positions."""
        n = len(arr)
        k = k % n
        return arr[k:] + arr[:k]
    
    return rotate_right_simple, rotate_right_inplace, rotate_left
```

### Problem 2: Find Missing Number

```python
def find_missing_examples():
    """Different approaches to find missing number."""
    
    def find_missing_sum(arr, n):
        """Using sum formula - O(n) time, O(1) space."""
        expected_sum = n * (n + 1) // 2
        actual_sum = sum(arr)
        return expected_sum - actual_sum
    
    def find_missing_xor(arr, n):
        """Using XOR properties - O(n) time, O(1) space."""
        xor_all = 0
        xor_arr = 0
        
        # XOR all numbers from 1 to n
        for i in range(1, n + 1):
            xor_all ^= i
        
        # XOR all numbers in array
        for num in arr:
            xor_arr ^= num
        
        # Missing number is XOR of above two
        return xor_all ^ xor_arr
    
    def find_missing_set(arr, n):
        """Using set - O(n) time, O(n) space."""
        num_set = set(arr)
        for i in range(1, n + 1):
            if i not in num_set:
                return i
        return -1
    
    return find_missing_sum, find_missing_xor, find_missing_set
```

---

## 🚀 Performance Tips

### 1. Choose the Right Data Structure

```python
def data_structure_comparison():
    """Compare different array-like structures."""
    
    # Python list - Dynamic, flexible
    python_list = [1, 2, 3, 4, 5]
    
    # Array module - Fixed type, memory efficient
    import array
    typed_array = array.array('i', [1, 2, 3, 4, 5])
    
    # NumPy array - Vectorized operations
    import numpy as np
    numpy_array = np.array([1, 2, 3, 4, 5])
    
    # Performance comparison for large arrays
    n = 1000000
    
    # Python list operations
    python_list = list(range(n))
    
    # NumPy array operations (much faster for numerical work)
    numpy_array = np.arange(n)
    
    return python_list, typed_array, numpy_array
```

### 2. Memory Optimization

```python
def memory_optimization_tips():
    """Tips for optimizing array memory usage."""
    
    # Tip 1: Use appropriate data types
    import array
    
    # For small integers, use smaller types
    small_ints = array.array('b', [1, 2, 3])  # signed char
    
    # Tip 2: Pre-allocate when size is known
    def pre_allocate_example(n):
        # Instead of repeatedly appending
        result = []
        for i in range(n):
            result.append(i)  # Causes multiple reallocations
        
        # Pre-allocate
        result = [0] * n
        for i in range(n):
            result[i] = i  # No reallocations
        
        return result
    
    # Tip 3: Use generators for large datasets
    def memory_efficient_processing():
        # Instead of loading everything in memory
        def process_large_array(arr):
            return [x * 2 for x in arr]  # Creates new list
        
        # Use generator
        def process_large_array_generator(arr):
            return (x * 2 for x in arr)  # Lazy evaluation
        
        return process_large_array, process_large_array_generator
    
    return pre_allocate_example, memory_efficient_processing
```

---

## 🎯 Next Steps

Now that you understand array fundamentals, you're ready to tackle problems:

1. **Start with Easy Problems**: Build confidence with basic patterns
2. **Master Core Patterns**: Two pointers, sliding window, prefix sum
3. **Progress to Medium**: More complex algorithms and optimizations
4. **Challenge Yourself**: Hard problems with advanced techniques

### Recommended Practice Order

1. [🟢 Easy Problems](easy-problems.md) - Master the basics
2. [🟡 Medium Problems](medium-problems.md) - Build advanced skills  
3. [🔴 Hard Problems](hard-problems.md) - Tackle complex challenges

---

*Ready to start solving problems? Begin with [Easy Problems](easy-problems.md) to build your foundation!*

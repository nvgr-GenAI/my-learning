# Search Algorithms - Fundamentals

## 🎯 Overview

Search algorithms are fundamental to computer science, enabling efficient retrieval of information from data structures. This section covers essential search techniques, from basic linear search to advanced optimization algorithms.

=== "📋 Core Search Concepts"

    ## **Search Algorithm Types**
    
    | Type | Description | Time Complexity | Best Use Case |
    |------|-------------|-----------------|---------------|
    | **Linear Search** | Check each element sequentially | O(n) | Unsorted data, small datasets |
    | **Binary Search** | Divide search space in half | O(log n) | Sorted arrays |
    | **Jump Search** | Jump by fixed steps, then linear | O(√n) | Large sorted arrays |
    | **Interpolation Search** | Estimate position based on value | O(log log n) | Uniformly distributed data |
    | **Exponential Search** | Find range, then binary search | O(log n) | Infinite/unbounded arrays |
    | **Ternary Search** | Divide into three parts | O(log₃ n) | Finding maximum/minimum |
    | **Hash-based Search** | Direct access via hash function | O(1) average | Fast lookups with keys |

    ## **Search Properties**
    
    | Property | Definition | Importance |
    |----------|------------|------------|
    | **Correctness** | Algorithm finds target if present | Must guarantee correct results |
    | **Completeness** | Algorithm terminates for all inputs | Avoids infinite loops |
    | **Optimality** | Finds best solution when multiple exist | Important for optimization |
    | **Time Complexity** | Number of operations vs input size | Performance measurement |
    | **Space Complexity** | Memory usage vs input size | Resource efficiency |
    | **Stability** | Preserves relative order of equal elements | Important for sorting-based searches |

=== "🔍 Basic Search Algorithms"

    ## **Linear Search**
    
    ```python
    def linear_search(arr, target):
        """
        Search for target in unsorted array
        Time: O(n), Space: O(1)
        """
        for i in range(len(arr)):
            if arr[i] == target:
                return i
        return -1
    
    def linear_search_all(arr, target):
        """
        Find all occurrences of target
        """
        indices = []
        for i in range(len(arr)):
            if arr[i] == target:
                indices.append(i)
        return indices
    
    def linear_search_first_occurrence(arr, condition):
        """
        Find first element satisfying condition
        """
        for i in range(len(arr)):
            if condition(arr[i]):
                return i
        return -1
    ```
    
    **Advantages:**
    - Works on unsorted data
    - Simple to implement
    - No preprocessing required
    
    **Disadvantages:**
    - Slow for large datasets
    - Not suitable for frequent searches
    
    ## **Binary Search**
    
    ```python
    def binary_search(arr, target):
        """
        Search in sorted array using binary search
        Time: O(log n), Space: O(1)
        """
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1
    
    def binary_search_recursive(arr, target, left=0, right=None):
        """
        Recursive implementation of binary search
        Time: O(log n), Space: O(log n)
        """
        if right is None:
            right = len(arr) - 1
        
        if left > right:
            return -1
        
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            return binary_search_recursive(arr, target, mid + 1, right)
        else:
            return binary_search_recursive(arr, target, left, mid - 1)
    
    def binary_search_leftmost(arr, target):
        """
        Find leftmost occurrence of target
        """
        left, right = 0, len(arr)
        
        while left < right:
            mid = left + (right - left) // 2
            if arr[mid] < target:
                left = mid + 1
            else:
                right = mid
        
        return left if left < len(arr) and arr[left] == target else -1
    
    def binary_search_rightmost(arr, target):
        """
        Find rightmost occurrence of target
        """
        left, right = 0, len(arr)
        
        while left < right:
            mid = left + (right - left) // 2
            if arr[mid] <= target:
                left = mid + 1
            else:
                right = mid
        
        return left - 1 if left > 0 and arr[left - 1] == target else -1
    ```

=== "⚡ Advanced Search Algorithms"

    ## **Jump Search**
    
    ```python
    import math
    
    def jump_search(arr, target):
        """
        Jump search for sorted arrays
        Time: O(√n), Space: O(1)
        """
        n = len(arr)
        step = int(math.sqrt(n))
        prev = 0
        
        # Jump to find block containing target
        while arr[min(step, n) - 1] < target:
            prev = step
            step += int(math.sqrt(n))
            if prev >= n:
                return -1
        
        # Linear search in the block
        while arr[prev] < target:
            prev += 1
            if prev == min(step, n):
                return -1
        
        if arr[prev] == target:
            return prev
        
        return -1
    ```
    
    ## **Interpolation Search**
    
    ```python
    def interpolation_search(arr, target):
        """
        Interpolation search for uniformly distributed sorted arrays
        Time: O(log log n) average, O(n) worst case
        """
        left, right = 0, len(arr) - 1
        
        while left <= right and target >= arr[left] and target <= arr[right]:
            if left == right:
                return left if arr[left] == target else -1
            
            # Interpolation formula
            pos = left + ((target - arr[left]) * (right - left) // 
                         (arr[right] - arr[left]))
            
            if arr[pos] == target:
                return pos
            elif arr[pos] < target:
                left = pos + 1
            else:
                right = pos - 1
        
        return -1
    ```
    
    ## **Exponential Search**
    
    ```python
    def exponential_search(arr, target):
        """
        Exponential search for unbounded/infinite arrays
        Time: O(log n), Space: O(1)
        """
        if arr[0] == target:
            return 0
        
        # Find range for binary search
        i = 1
        while i < len(arr) and arr[i] <= target:
            i *= 2
        
        # Binary search in found range
        return binary_search_range(arr, target, i // 2, 
                                 min(i, len(arr) - 1))
    
    def binary_search_range(arr, target, left, right):
        """Helper function for binary search in range"""
        while left <= right:
            mid = left + (right - left) // 2
            
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1
    ```

=== "🎯 Binary Search Variations"

    ## **Search in Rotated Array**
    
    ```python
    def search_rotated_array(arr, target):
        """
        Search in rotated sorted array
        Time: O(log n), Space: O(1)
        """
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if arr[mid] == target:
                return mid
            
            # Check which half is sorted
            if arr[left] <= arr[mid]:  # Left half is sorted
                if arr[left] <= target < arr[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:  # Right half is sorted
                if arr[mid] < target <= arr[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        
        return -1
    ```
    
    ## **Find Peak Element**
    
    ```python
    def find_peak_element(arr):
        """
        Find any peak element in array
        Time: O(log n), Space: O(1)
        """
        left, right = 0, len(arr) - 1
        
        while left < right:
            mid = left + (right - left) // 2
            
            if arr[mid] > arr[mid + 1]:
                right = mid
            else:
                left = mid + 1
        
        return left
    ```
    
    ## **Search Insert Position**
    
    ```python
    def search_insert_position(arr, target):
        """
        Find position where target should be inserted
        Time: O(log n), Space: O(1)
        """
        left, right = 0, len(arr)
        
        while left < right:
            mid = left + (right - left) // 2
            if arr[mid] < target:
                left = mid + 1
            else:
                right = mid
        
        return left
    ```

=== "🔎 Specialized Search Techniques"

    ## **Ternary Search**
    
    ```python
    def ternary_search(arr, target):
        """
        Ternary search (divide into 3 parts)
        Time: O(log₃ n), Space: O(1)
        """
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid1 = left + (right - left) // 3
            mid2 = right - (right - left) // 3
            
            if arr[mid1] == target:
                return mid1
            if arr[mid2] == target:
                return mid2
            
            if target < arr[mid1]:
                right = mid1 - 1
            elif target > arr[mid2]:
                left = mid2 + 1
            else:
                left = mid1 + 1
                right = mid2 - 1
        
        return -1
    
    def ternary_search_maximum(func, left, right, epsilon=1e-9):
        """
        Find maximum of unimodal function using ternary search
        """
        while right - left > epsilon:
            mid1 = left + (right - left) / 3
            mid2 = right - (right - left) / 3
            
            if func(mid1) < func(mid2):
                left = mid1
            else:
                right = mid2
        
        return (left + right) / 2
    ```
    
    ## **Binary Search on Answer**
    
    ```python
    def binary_search_answer(condition, left, right):
        """
        Binary search to find optimal answer
        Template for optimization problems
        """
        while left < right:
            mid = left + (right - left) // 2
            
            if condition(mid):
                right = mid
            else:
                left = mid + 1
        
        return left
    
    def find_square_root(x, precision=1e-6):
        """
        Find square root using binary search
        """
        if x < 1:
            left, right = 0, 1
        else:
            left, right = 1, x
        
        while right - left > precision:
            mid = (left + right) / 2
            if mid * mid <= x:
                left = mid
            else:
                right = mid
        
        return left
    ```

=== "🎲 Search in Data Structures"

    ## **Search in 2D Matrix**
    
    ```python
    def search_2d_matrix(matrix, target):
        """
        Search in row and column sorted matrix
        Time: O(m + n), Space: O(1)
        """
        if not matrix or not matrix[0]:
            return False
        
        rows, cols = len(matrix), len(matrix[0])
        row, col = 0, cols - 1
        
        while row < rows and col >= 0:
            if matrix[row][col] == target:
                return True
            elif matrix[row][col] > target:
                col -= 1
            else:
                row += 1
        
        return False
    
    def search_2d_matrix_binary(matrix, target):
        """
        Search in matrix where each row is sorted
        Time: O(log(mn)), Space: O(1)
        """
        if not matrix or not matrix[0]:
            return False
        
        rows, cols = len(matrix), len(matrix[0])
        left, right = 0, rows * cols - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            mid_value = matrix[mid // cols][mid % cols]
            
            if mid_value == target:
                return True
            elif mid_value < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return False
    ```
    
    ## **Search in Linked List**
    
    ```python
    class ListNode:
        def __init__(self, val=0, next=None):
            self.val = val
            self.next = next
    
    def search_linked_list(head, target):
        """
        Linear search in linked list
        Time: O(n), Space: O(1)
        """
        current = head
        position = 0
        
        while current:
            if current.val == target:
                return position
            current = current.next
            position += 1
        
        return -1
    ```

=== "📊 Complexity Analysis"

    ## **Time Complexities**
    
    | Algorithm | Best Case | Average Case | Worst Case | Space |
    |-----------|-----------|--------------|------------|-------|
    | **Linear Search** | O(1) | O(n) | O(n) | O(1) |
    | **Binary Search** | O(1) | O(log n) | O(log n) | O(1) |
    | **Jump Search** | O(1) | O(√n) | O(√n) | O(1) |
    | **Interpolation Search** | O(1) | O(log log n) | O(n) | O(1) |
    | **Exponential Search** | O(1) | O(log n) | O(log n) | O(1) |
    | **Ternary Search** | O(1) | O(log₃ n) | O(log₃ n) | O(1) |
    
    ## **When to Use Each Algorithm**
    
    | Scenario | Best Algorithm | Reason |
    |----------|---------------|---------|
    | **Small dataset** | Linear Search | Simple, no preprocessing |
    | **Sorted array** | Binary Search | Optimal O(log n) performance |
    | **Large sorted array** | Jump Search | Better than linear, simpler than binary |
    | **Uniform distribution** | Interpolation Search | Better than binary for uniform data |
    | **Unbounded array** | Exponential Search | Finds range efficiently |
    | **Finding extrema** | Ternary Search | Works on unimodal functions |
    | **Optimization problems** | Binary Search on Answer | Reduces search space efficiently |

=== "🎯 Problem-Solving Strategies"

    ## **Search Problem Patterns**
    
    | Pattern | When to Use | Example Problems |
    |---------|-------------|------------------|
    | **Direct Search** | Find exact element | Array search, string matching |
    | **Range Search** | Find elements in range | Database queries, interval problems |
    | **Optimization Search** | Find optimal value | Square root, capacity problems |
    | **Existence Search** | Check if element exists | Set membership, validation |
    | **Count Search** | Count occurrences | Frequency analysis, statistics |
    | **Position Search** | Find insertion point | Sorted insertion, ranking |
    
    ## **Search Algorithm Selection Guide**
    
    ```python
    def choose_search_algorithm(data_characteristics):
        """
        Guide for selecting appropriate search algorithm
        """
        if data_characteristics['sorted']:
            if data_characteristics['size'] == 'small':
                return "Linear Search or Binary Search"
            elif data_characteristics['uniform_distribution']:
                return "Interpolation Search"
            elif data_characteristics['size'] == 'very_large':
                return "Jump Search or Binary Search"
            else:
                return "Binary Search"
        else:
            if data_characteristics['frequent_searches']:
                return "Hash Table or Sort first then Binary Search"
            else:
                return "Linear Search"
    
    def optimize_search_performance(dataset):
        """
        Suggestions for optimizing search performance
        """
        suggestions = []
        
        if not dataset['sorted']:
            suggestions.append("Sort data for O(log n) searches")
        
        if dataset['frequent_searches']:
            suggestions.append("Use hash table for O(1) average case")
        
        if dataset['size'] > 10000 and dataset['sorted']:
            suggestions.append("Use binary search for optimal performance")
        
        if dataset['range_queries']:
            suggestions.append("Consider segment tree or binary indexed tree")
        
        return suggestions
    ```

---

*Master these search fundamentals to efficiently find information in any data structure or problem domain!*

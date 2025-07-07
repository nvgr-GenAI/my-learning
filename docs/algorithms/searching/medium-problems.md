# Searching Algorithms - Medium Problems

## 🎯 Learning Objectives

Master intermediate searching techniques and complex applications:

- Advanced binary search patterns
- Search in complex data structures
- Optimization problems using binary search
- Multi-dimensional search techniques
- Search with complex conditions

=== "📋 Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | Find First and Last Position | Binary Search Bounds | Medium | O(log n) | O(1) |
    | 2 | Search in 2D Matrix | Binary Search 2D | Medium | O(log(mn)) | O(1) |
    | 3 | Search in 2D Matrix II | Staircase Search | Medium | O(m + n) | O(1) |
    | 4 | Find Peak Element | Binary Search Peak | Medium | O(log n) | O(1) |
    | 5 | Search in Rotated Array II | Binary Search + Duplicates | Medium | O(log n) avg | O(1) |
    | 6 | Find Minimum in Rotated Array | Binary Search Min | Medium | O(log n) | O(1) |
    | 7 | Kth Smallest in Sorted Matrix | Binary Search + Count | Medium | O(n log(max-min)) | O(1) |
    | 8 | Find Right Interval | Binary Search + Sort | Medium | O(n log n) | O(n) |
    | 9 | Time Based Key-Value Store | Binary Search + HashMap | Medium | O(log n) | O(n) |
    | 10 | Online Election | Binary Search Timeline | Medium | O(log n) | O(n) |
    | 11 | Random Pick with Weight | Binary Search + Prefix Sum | Medium | O(log n) | O(n) |
    | 12 | Find K Closest Elements | Binary Search + Two Pointers | Medium | O(log n + k) | O(k) |
    | 13 | Capacity to Ship Packages | Binary Search on Answer | Medium | O(n log(sum)) | O(1) |
    | 14 | Koko Eating Bananas | Binary Search on Speed | Medium | O(n log(max)) | O(1) |
    | 15 | Minimum Time to Complete Trips | Binary Search on Time | Medium | O(n log(max*trips)) | O(1) |

=== "🎯 Advanced Patterns"

    **🔍 Binary Search Variants:**
    - Find first/last occurrence
    - Find insertion position
    - Search in infinite arrays
    - Search with unknown array size
    
    **🎛️ Multi-dimensional Search:**
    - 2D matrix search techniques
    - Staircase search algorithm
    - K-way merge problems
    - Range intersection queries
    
    **🎯 Binary Search on Answer:**
    - Minimize/maximize objective function
    - Feasibility checking with binary search
    - Optimization problems
    - Resource allocation problems
    
    **⏰ Time-based Search:**
    - Temporal data structures
    - Version control systems
    - Timeline-based queries
    - Historical data access

=== "💡 Solutions"

    === "Find First and Last Position"
        ```python
        def searchRange(nums, target):
            """
            Find first and last position of target
            """
            def findBoundary(nums, target, findFirst):
                left, right = 0, len(nums) - 1
                result = -1
                
                while left <= right:
                    mid = left + (right - left) // 2
                    
                    if nums[mid] == target:
                        result = mid
                        if findFirst:
                            right = mid - 1  # Continue searching left
                        else:
                            left = mid + 1   # Continue searching right
                    elif nums[mid] < target:
                        left = mid + 1
                    else:
                        right = mid - 1
                
                return result
            
            first = findBoundary(nums, target, True)
            if first == -1:
                return [-1, -1]
            
            last = findBoundary(nums, target, False)
            return [first, last]
        ```
    
    === "Search in 2D Matrix"
        ```python
        def searchMatrix(matrix, target):
            """
            Search in row and column sorted matrix
            Treat as 1D sorted array
            """
            if not matrix or not matrix[0]:
                return False
            
            m, n = len(matrix), len(matrix[0])
            left, right = 0, m * n - 1
            
            while left <= right:
                mid = left + (right - left) // 2
                mid_element = matrix[mid // n][mid % n]
                
                if mid_element == target:
                    return True
                elif mid_element < target:
                    left = mid + 1
                else:
                    right = mid - 1
            
            return False
        ```
    
    === "Search in 2D Matrix II"
        ```python
        def searchMatrix(matrix, target):
            """
            Search in matrix sorted by rows and columns
            Staircase search: start from top-right or bottom-left
            """
            if not matrix or not matrix[0]:
                return False
            
            row, col = 0, len(matrix[0]) - 1
            
            while row < len(matrix) and col >= 0:
                if matrix[row][col] == target:
                    return True
                elif matrix[row][col] > target:
                    col -= 1  # Move left
                else:
                    row += 1  # Move down
            
            return False
        ```
    
    === "Kth Smallest in Sorted Matrix"
        ```python
        def kthSmallest(matrix, k):
            """
            Find kth smallest element in sorted matrix
            Binary search on value range
            """
            n = len(matrix)
            left, right = matrix[0][0], matrix[n-1][n-1]
            
            def countLessEqual(matrix, target):
                count = 0
                row, col = len(matrix) - 1, 0
                
                while row >= 0 and col < len(matrix[0]):
                    if matrix[row][col] <= target:
                        count += row + 1
                        col += 1
                    else:
                        row -= 1
                
                return count
            
            while left < right:
                mid = left + (right - left) // 2
                
                if countLessEqual(matrix, mid) < k:
                    left = mid + 1
                else:
                    right = mid
            
            return left
        ```
    
    === "Capacity to Ship Packages"
        ```python
        def shipWithinDays(weights, days):
            """
            Find minimum ship capacity to ship all packages in days
            Binary search on capacity
            """
            def canShipInDays(capacity):
                current_weight = 0
                days_needed = 1
                
                for weight in weights:
                    if current_weight + weight > capacity:
                        days_needed += 1
                        current_weight = weight
                    else:
                        current_weight += weight
                
                return days_needed <= days
            
            left = max(weights)  # Minimum capacity
            right = sum(weights)  # Maximum capacity
            
            while left < right:
                mid = left + (right - left) // 2
                
                if canShipInDays(mid):
                    right = mid
                else:
                    left = mid + 1
            
            return left
        ```
    
    === "Time Based Key-Value Store"
        ```python
        from collections import defaultdict
        import bisect
        
        class TimeMap:
            def __init__(self):
                self.store = defaultdict(list)  # key -> [(timestamp, value)]
            
            def set(self, key, value, timestamp):
                self.store[key].append((timestamp, value))
            
            def get(self, key, timestamp):
                if key not in self.store:
                    return ""
                
                pairs = self.store[key]
                
                # Binary search for largest timestamp <= given timestamp
                left, right = 0, len(pairs) - 1
                result = ""
                
                while left <= right:
                    mid = left + (right - left) // 2
                    
                    if pairs[mid][0] <= timestamp:
                        result = pairs[mid][1]
                        left = mid + 1
                    else:
                        right = mid - 1
                
                return result
        ```
    
    === "Random Pick with Weight"
        ```python
        import random
        import bisect
        
        class Solution:
            def __init__(self, w):
                self.prefix_sum = []
                current_sum = 0
                
                for weight in w:
                    current_sum += weight
                    self.prefix_sum.append(current_sum)
            
            def pickIndex(self):
                target = random.randint(1, self.prefix_sum[-1])
                
                # Binary search for first element >= target
                left, right = 0, len(self.prefix_sum) - 1
                
                while left < right:
                    mid = left + (right - left) // 2
                    
                    if self.prefix_sum[mid] < target:
                        left = mid + 1
                    else:
                        right = mid
                
                return left
        ```
    
    === "Find K Closest Elements"
        ```python
        def findClosestElements(arr, k, x):
            """
            Find k elements closest to x
            Binary search + two pointers
            """
            # Binary search to find insertion point
            left, right = 0, len(arr) - k
            
            while left < right:
                mid = left + (right - left) // 2
                
                # Compare distances: arr[mid] vs arr[mid + k]
                if x - arr[mid] > arr[mid + k] - x:
                    left = mid + 1
                else:
                    right = mid
            
            return arr[left:left + k]
        ```

=== "📊 Advanced Techniques"

    **🔧 Search Space Design:**
    - **Value Range Search**: Binary search on possible answers
    - **Index Range Search**: Binary search on array indices
    - **Multi-dimensional**: Reduce to 1D problem when possible
    - **Infinite Arrays**: Handle unknown boundaries
    
    **⚡ Optimization Strategies:**
    - **Monotonic Properties**: Exploit sorted nature of search space
    - **Invariant Maintenance**: Preserve search conditions
    - **Early Termination**: Stop when answer is found
    - **Constraint Checking**: Validate feasibility efficiently
    
    **🎯 Problem Classification:**
    - **Exact Search**: Find specific element
    - **Boundary Search**: Find first/last occurrence
    - **Optimization**: Minimize/maximize objective function
    - **Existence**: Check if solution exists

=== "🚀 Expert Tips"

    **💡 Advanced Problem-Solving:**
    1. **Identify Search Space**: What are we optimizing over?
    2. **Check Monotonicity**: Is the search space sorted/monotonic?
    3. **Design Check Function**: How to verify if answer is valid?
    4. **Handle Edge Cases**: Empty arrays, single elements, duplicates
    5. **Optimize Implementation**: Choose appropriate data structures
    
    **🔍 Complex Scenarios:**
    - **Multiple Dimensions**: Reduce complexity when possible
    - **Dynamic Data**: Handle updates efficiently
    - **Approximation**: When exact answer isn't required
    - **Memory Constraints**: Space-efficient implementations
    
    **🏆 Professional Applications:**
    - **Database Systems**: Index searches and range queries
    - **Load Balancing**: Resource allocation optimization
    - **Machine Learning**: Hyperparameter optimization
    - **Game Development**: Collision detection and pathfinding
    - **Financial Systems**: Time-series data queries

## 📝 Summary

These medium searching problems demonstrate:

- **Advanced Binary Search** patterns for complex scenarios
- **Multi-dimensional Search** in matrices and higher dimensions
- **Binary Search on Answer** for optimization problems
- **Time-based Search** for temporal data structures
- **Complex Condition Handling** with custom validation

These techniques are essential for:
- **System Design** with efficient data access patterns
- **Algorithm Optimization** in competitive programming
- **Real-world Applications** requiring fast search and retrieval
- **Database Systems** and indexing strategies

Master these patterns to handle sophisticated search problems in interviews and production systems!

# Searching Algorithms - Hard Problems

## 🎯 Learning Objectives

Master advanced searching techniques and complex optimization:

- Binary search on complex search spaces
- Advanced multi-dimensional search
- Search in dynamic and streaming data
- Optimization with multiple constraints
- Real-world search applications

=== "📋 Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | Median of Two Sorted Arrays | Binary Search Partition | Hard | O(log(min(m,n))) | O(1) |
    | 2 | Split Array Largest Sum | Binary Search + Greedy | Hard | O(n log(sum)) | O(1) |
    | 3 | Minimize Max Distance to Gas Station | Binary Search + Simulation | Hard | O(n log(max_dist)) | O(1) |
    | 4 | Swim in Rising Water | Binary Search + BFS/DFS | Hard | O(n² log(max_height)) | O(n²) |
    | 5 | Find K-th Smallest Pair Distance | Binary Search + Count | Hard | O(n log n + n log(max_dist)) | O(1) |
    | 6 | Minimum Number of Days to Make Bouquets | Binary Search + Validation | Hard | O(n log(max_day)) | O(1) |
    | 7 | Magnetic Force Between Balls | Binary Search + Greedy | Hard | O(n log n + n log(max_pos)) | O(1) |
    | 8 | Count of Smaller Numbers After Self | Merge Sort + BIT | Hard | O(n log n) | O(n) |
    | 9 | Reverse Pairs | Enhanced Merge Sort | Hard | O(n log n) | O(n) |
    | 10 | Longest Duplicate Substring | Binary Search + Rolling Hash | Hard | O(n log n) | O(n) |
    | 11 | Maximum Frequency Stack | Multiple Stacks | Hard | O(1) | O(n) |
    | 12 | Range Sum Query 2D - Mutable | 2D Binary Indexed Tree | Hard | O(log m × log n) | O(mn) |
    | 13 | Count of Range Sum | Merge Sort + Count | Hard | O(n log n) | O(n) |
    | 14 | Maximum Number of Events | Binary Search + Greedy | Hard | O(n log n) | O(n) |
    | 15 | Minimum Time to Reach All Nodes | Binary Search + BFS | Hard | O(n log(max_time)) | O(n) |

=== "🎯 Expert Patterns"

    **🔍 Advanced Binary Search:**
    - Binary search on floating point values
    - Multi-dimensional binary search
    - Binary search with complex conditions
    - Parallel binary search techniques
    
    **📊 Search with Data Structures:**
    - Binary search trees and variants
    - Segment trees with binary search
    - Persistent data structures
    - Online search algorithms
    
    **🎯 Optimization Search:**
    - Minimize/maximize complex objectives
    - Multi-objective optimization
    - Constraint satisfaction problems
    - Approximation algorithms
    
    **⚡ Parallel and Distributed Search:**
    - Parallel search algorithms
    - Distributed search systems
    - Load balancing in search
    - Fault-tolerant search

=== "💡 Solutions"

    === "Median of Two Sorted Arrays"
        ```python
        def findMedianSortedArrays(nums1, nums2):
            """
            Find median of two sorted arrays in O(log(min(m,n)))
            Binary search to partition arrays correctly
            """
            # Ensure nums1 is the smaller array
            if len(nums1) > len(nums2):
                nums1, nums2 = nums2, nums1
            
            m, n = len(nums1), len(nums2)
            left, right = 0, m
            
            while left <= right:
                partition1 = (left + right) // 2
                partition2 = (m + n + 1) // 2 - partition1
                
                # Get boundary elements
                maxLeft1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
                minRight1 = float('inf') if partition1 == m else nums1[partition1]
                
                maxLeft2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
                minRight2 = float('inf') if partition2 == n else nums2[partition2]
                
                # Check if partition is correct
                if maxLeft1 <= minRight2 and maxLeft2 <= minRight1:
                    if (m + n) % 2 == 0:
                        return (max(maxLeft1, maxLeft2) + min(minRight1, minRight2)) / 2
                    else:
                        return max(maxLeft1, maxLeft2)
                elif maxLeft1 > minRight2:
                    right = partition1 - 1
                else:
                    left = partition1 + 1
        ```
    
    === "Split Array Largest Sum"
        ```python
        def splitArray(nums, m):
            """
            Split array into m subarrays to minimize largest sum
            Binary search on answer + greedy validation
            """
            def canSplit(max_sum):
                subarrays = 1
                current_sum = 0
                
                for num in nums:
                    if current_sum + num > max_sum:
                        subarrays += 1
                        current_sum = num
                        if subarrays > m:
                            return False
                    else:
                        current_sum += num
                
                return True
            
            left, right = max(nums), sum(nums)
            
            while left < right:
                mid = left + (right - left) // 2
                
                if canSplit(mid):
                    right = mid
                else:
                    left = mid + 1
            
            return left
        ```
    
    === "Find K-th Smallest Pair Distance"
        ```python
        def smallestDistancePair(nums, k):
            """
            Find k-th smallest distance between pairs
            Binary search on distance + count pairs
            """
            nums.sort()
            
            def countPairsWithDistanceLessEqual(distance):
                count = 0
                left = 0
                
                for right in range(len(nums)):
                    while nums[right] - nums[left] > distance:
                        left += 1
                    count += right - left
                
                return count
            
            left, right = 0, nums[-1] - nums[0]
            
            while left < right:
                mid = left + (right - left) // 2
                
                if countPairsWithDistanceLessEqual(mid) < k:
                    left = mid + 1
                else:
                    right = mid
            
            return left
        ```
    
    === "Magnetic Force Between Balls"
        ```python
        def maxDistance(position, m):
            """
            Place m balls to maximize minimum magnetic force
            Binary search on distance + greedy placement
            """
            position.sort()
            
            def canPlaceBalls(min_distance):
                count = 1
                last_position = position[0]
                
                for i in range(1, len(position)):
                    if position[i] - last_position >= min_distance:
                        count += 1
                        last_position = position[i]
                        if count == m:
                            return True
                
                return False
            
            left, right = 1, position[-1] - position[0]
            result = 0
            
            while left <= right:
                mid = left + (right - left) // 2
                
                if canPlaceBalls(mid):
                    result = mid
                    left = mid + 1
                else:
                    right = mid - 1
            
            return result
        ```
    
    === "Longest Duplicate Substring"
        ```python
        def longestDupSubstring(s):
            """
            Find longest duplicate substring using binary search + rolling hash
            """
            def search(length):
                """Check if there's a duplicate substring of given length"""
                if length == 0:
                    return 0
                
                base = 26
                mod = 2**32 - 1
                
                # Calculate hash for first substring
                hash_value = 0
                for i in range(length):
                    hash_value = (hash_value * base + ord(s[i]) - ord('a')) % mod
                
                seen = {hash_value}
                base_power = pow(base, length - 1, mod)
                
                # Rolling hash for remaining substrings
                for i in range(1, len(s) - length + 1):
                    # Remove first character, add new character
                    hash_value = (hash_value - (ord(s[i-1]) - ord('a')) * base_power) % mod
                    hash_value = (hash_value * base + ord(s[i + length - 1]) - ord('a')) % mod
                    
                    if hash_value in seen:
                        return i  # Return starting index
                    seen.add(hash_value)
                
                return -1
            
            # Binary search on length
            left, right = 0, len(s) - 1
            result_start = 0
            
            while left <= right:
                mid = left + (right - left) // 2
                start = search(mid)
                
                if start != -1:
                    result_start = start
                    left = mid + 1
                else:
                    right = mid - 1
            
            return s[result_start:result_start + right + 1]
        ```
    
    === "Count of Smaller Numbers After Self"
        ```python
        class BIT:
            def __init__(self, n):
                self.n = n
                self.tree = [0] * (n + 1)
            
            def update(self, i, delta):
                while i <= self.n:
                    self.tree[i] += delta
                    i += i & (-i)
            
            def query(self, i):
                result = 0
                while i > 0:
                    result += self.tree[i]
                    i -= i & (-i)
                return result
        
        def countSmaller(nums):
            """
            Count smaller numbers after self using BIT
            """
            # Coordinate compression
            sorted_nums = sorted(set(nums))
            rank = {num: i + 1 for i, num in enumerate(sorted_nums)}
            
            bit = BIT(len(sorted_nums))
            result = []
            
            # Process from right to left
            for i in range(len(nums) - 1, -1, -1):
                count = bit.query(rank[nums[i]] - 1)
                result.append(count)
                bit.update(rank[nums[i]], 1)
            
            return result[::-1]
        ```
    
    === "Range Sum Query 2D - Mutable"
        ```python
        class NumMatrix:
            def __init__(self, matrix):
                if not matrix or not matrix[0]:
                    return
                
                self.matrix = matrix
                self.m, self.n = len(matrix), len(matrix[0])
                
                # Build 2D Binary Indexed Tree
                self.bit = [[0] * (self.n + 1) for _ in range(self.m + 1)]
                
                for i in range(self.m):
                    for j in range(self.n):
                        self._update_bit(i + 1, j + 1, matrix[i][j])
            
            def _update_bit(self, row, col, delta):
                i = row
                while i <= self.m:
                    j = col
                    while j <= self.n:
                        self.bit[i][j] += delta
                        j += j & (-j)
                    i += i & (-i)
            
            def _query_bit(self, row, col):
                result = 0
                i = row
                while i > 0:
                    j = col
                    while j > 0:
                        result += self.bit[i][j]
                        j -= j & (-j)
                    i -= i & (-i)
                return result
            
            def update(self, row, col, val):
                delta = val - self.matrix[row][col]
                self.matrix[row][col] = val
                self._update_bit(row + 1, col + 1, delta)
            
            def sumRegion(self, row1, col1, row2, col2):
                return (self._query_bit(row2 + 1, col2 + 1) -
                        self._query_bit(row1, col2 + 1) -
                        self._query_bit(row2 + 1, col1) +
                        self._query_bit(row1, col1))
        ```

=== "📊 Advanced Techniques"

    **🔧 Algorithm Design:**
    - **Floating Point Binary Search**: Handle precision issues
    - **Multi-dimensional Optimization**: Reduce to 1D when possible
    - **Dynamic Search Spaces**: Adapt to changing constraints
    - **Parallel Search**: Divide search space across processors
    - **Persistent Search**: Maintain search history
    
    **⚡ Performance Optimization:**
    - **Cache-Aware Search**: Optimize for memory hierarchy
    - **Approximation Algorithms**: Trade accuracy for speed
    - **Early Termination**: Stop when good enough answer found
    - **Preprocessing**: Build indices for faster search
    
    **🎯 Real-world Applications:**
    - **Database Systems**: Query optimization and indexing
    - **Machine Learning**: Hyperparameter tuning
    - **Computer Graphics**: Ray tracing and collision detection
    - **Computational Geometry**: Spatial data structures
    - **Network Routing**: Path optimization algorithms

=== "🚀 Expert Insights"

    **💡 Advanced Problem-Solving:**
    1. **Model Search Space**: What mathematical space are we searching?
    2. **Identify Monotonicity**: How does objective function behave?
    3. **Design Validation**: How to check if solution is feasible?
    4. **Handle Precision**: Floating point vs integer search
    5. **Optimize for Scale**: How does algorithm perform on large inputs?
    
    **🔍 Complex Challenges:**
    - **Multi-objective Optimization**: Balance competing objectives
    - **Dynamic Constraints**: Handle changing problem parameters
    - **Distributed Search**: Coordinate across multiple machines
    - **Fault Tolerance**: Handle failures during search
    - **Real-time Constraints**: Search within time limits
    
    **🏆 Research Frontiers:**
    - **Quantum Search**: Leverage quantum computing advantages
    - **Machine Learning**: Learn better search strategies
    - **Approximation Theory**: Bounds on search quality
    - **Parallel Algorithms**: Exploit multi-core architectures
    - **Network Theory**: Search in graph structures

## 📝 Summary

These hard searching problems demonstrate:

- **Advanced Binary Search** on complex answer spaces
- **Multi-dimensional Search** with sophisticated constraints
- **Data Structure Integration** for dynamic search scenarios
- **Mathematical Optimization** with theoretical foundations
- **Real-world Applications** requiring high performance

These techniques are essential for:

- **System Architecture** requiring efficient search and retrieval
- **Research and Development** in algorithms and optimization
- **High-Performance Computing** applications
- **Competitive Programming** at the highest levels
- **Industry Leadership** in technical problem solving

Master these advanced patterns to tackle the most challenging search problems in computer science and push the boundaries of algorithmic efficiency!

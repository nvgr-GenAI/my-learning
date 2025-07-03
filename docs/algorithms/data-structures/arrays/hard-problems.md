# Arrays: Hard Problems

## 🔥 Advanced Array Challenges

These problems require sophisticated algorithms, complex optimizations, and deep understanding of array manipulation techniques. Master these to excel in the most challenging technical interviews.

=== "📋 Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | Median of Two Sorted Arrays | Binary Search | Hard | O(log(min(m,n))) | O(1) |
    | 2 | First Missing Positive | Cyclic Sort/Index Mapping | Hard | O(n) | O(1) |
    | 3 | Trapping Rain Water | Two Pointers/Stack | Hard | O(n) | O(1) |
    | 4 | Sliding Window Maximum | Deque/Monotonic Queue | Hard | O(n) | O(k) |
    | 5 | Longest Consecutive Sequence | Hash Set | Hard | O(n) | O(n) |
    | 6 | Minimum Window Substring | Sliding Window | Hard | O(m+n) | O(m+n) |
    | 7 | Maximum Rectangle | Stack + Histogram | Hard | O(m×n) | O(n) |
    | 8 | Largest Rectangle in Histogram | Monotonic Stack | Hard | O(n) | O(n) |
    | 9 | Merge k Sorted Arrays | Divide & Conquer | Hard | O(n log k) | O(log k) |
    | 10 | Count of Smaller Numbers After Self | Merge Sort/Fenwick Tree | Hard | O(n log n) | O(n) |
    | 11 | Best Time to Buy/Sell Stock IV | DP | Hard | O(n×k) | O(k) |
    | 12 | Maximal Rectangle | Stack + DP | Hard | O(m×n) | O(n) |
    | 13 | Russian Doll Envelopes | DP + Binary Search | Hard | O(n log n) | O(n) |
    | 14 | Count Inversions in Array | Merge Sort | Hard | O(n log n) | O(n) |
    | 15 | Maximum Subarray Product | DP | Hard | O(n) | O(1) |

=== "🎯 Advanced Patterns"

    **🔍 Binary Search on Answer:**
    - Searching in infinite spaces
    - Finding optimal values
    - Partition-based searches
    
    **🏗️ Stack-Based Algorithms:**
    - Monotonic stacks
    - Histogram problems
    - Rectangle calculations
    
    **📊 Advanced DP:**
    - Multi-dimensional state spaces
    - Optimization problems
    - Complex transitions
    
    **🔄 Divide & Conquer:**
    - Merge-based algorithms
    - Tree-like decomposition
    - Complex merging strategies

=== "⚡ Interview Strategy"

    **🎯 Pattern Recognition:**
    - Identify optimal substructure
    - Look for monotonic properties
    - Consider multiple approaches
    
    **💡 Optimization Techniques:**
    - Space-time tradeoffs
    - Auxiliary data structures
    - Mathematical insights
    
    **🚀 Implementation Tips:**
    - Handle edge cases carefully
    - Consider integer overflow
    - Validate constraints

---

## Problem 1: Median of Two Sorted Arrays

**Difficulty:** Hard  
**Pattern:** Binary Search  
**Time:** O(log(min(m,n))) | **Space:** O(1)

### Problem Statement

Given two sorted arrays `nums1` and `nums2`, return the median of the two sorted arrays combined.

**Example:**
```text
Input: nums1 = [1,3], nums2 = [2]
Output: 2.0
Explanation: merged array = [1,2,3] and median is 2
```

=== "💡 Optimal Solution"

    ```python
    def find_median_sorted_arrays(nums1, nums2):
        """
        Binary search on partitions to find median in O(log(min(m,n))) time.
        
        Key insight: Median divides array into two equal halves.
        We need to find correct partition where:
        - Left half ≤ Right half for both arrays
        - max(left_half) ≤ min(right_half)
        """
        # Ensure nums1 is smaller for optimization
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        
        m, n = len(nums1), len(nums2)
        left, right = 0, m
        
        while left <= right:
            # Partition nums1 at partition1
            partition1 = (left + right) // 2
            # Partition nums2 to maintain equal halves
            partition2 = (m + n + 1) // 2 - partition1
            
            # Handle boundary cases with infinity
            max_left1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
            min_right1 = float('inf') if partition1 == m else nums1[partition1]
            
            max_left2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
            min_right2 = float('inf') if partition2 == n else nums2[partition2]
            
            # Check if partition is correct
            if max_left1 <= min_right2 and max_left2 <= min_right1:
                # Found correct partition
                if (m + n) % 2 == 0:
                    return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2
                else:
                    return max(max_left1, max_left2)
            
            # Adjust partition
            elif max_left1 > min_right2:
                right = partition1 - 1  # Move left in nums1
            else:
                left = partition1 + 1   # Move right in nums1
        
        raise ValueError("Input arrays are not sorted")
    ```

=== "🔄 Brute Force"

    ```python
    def find_median_brute_force(nums1, nums2):
        """
        Merge arrays and find median - O((m+n)log(m+n)) time.
        """
        merged = sorted(nums1 + nums2)
        n = len(merged)
        
        if n % 2 == 0:
            return (merged[n//2 - 1] + merged[n//2]) / 2
        else:
            return merged[n//2]
    ```

=== "📝 Tips & Insights"

    **🎯 Key Insights:**
    - Binary search on partition points, not values
    - Maintain equal-sized halves using partition formula
    - Use infinity for boundary handling
    
    **💡 Interview Tips:**
    - Start with brute force, then optimize
    - Draw partition examples on whiteboard
    - Explain the invariant clearly
    
    **⚠️ Common Pitfalls:**
    - Not handling empty arrays
    - Integer overflow in median calculation
    - Wrong partition size calculation

---

## Problem 2: First Missing Positive

**Difficulty:** Hard  
**Pattern:** Cyclic Sort/Index Mapping  
**Time:** O(n) | **Space:** O(1)

### Problem Statement

Given an unsorted integer array `nums`, return the smallest missing positive integer.

**Example:**
```text
Input: nums = [3,4,-1,1]
Output: 2
Explanation: 1 is in the array but 2 is missing.
```

=== "💡 Optimal Solution"

    ```python
    def first_missing_positive(nums):
        """
        Use array indices as hash map - place each number at its correct position.
        
        Key insight: First missing positive is in range [1, n+1].
        We can use array itself as hash map by placing nums[i] at index nums[i]-1.
        """
        n = len(nums)
        
        # First pass: mark numbers out of range as 0
        for i in range(n):
            if nums[i] <= 0 or nums[i] > n:
                nums[i] = 0
        
        # Second pass: use indices as hash map
        for i in range(n):
            val = abs(nums[i])
            if val != 0:
                # Mark presence by making nums[val-1] negative
                if nums[val-1] > 0:
                    nums[val-1] = -nums[val-1]
                elif nums[val-1] == 0:
                    nums[val-1] = -(n+1)  # Special marker for 0
        
        # Third pass: find first positive number
        for i in range(n):
            if nums[i] >= 0:
                return i + 1
        
        return n + 1
    ```

=== "🔄 Alternative: Cyclic Sort"

    ```python
    def first_missing_positive_cyclic(nums):
        """
        Place each number in its correct position using cyclic sort.
        """
        n = len(nums)
        i = 0
        
        while i < n:
            # If number is in valid range and not in correct position
            if 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
                # Swap to correct position
                correct_pos = nums[i] - 1
                nums[i], nums[correct_pos] = nums[correct_pos], nums[i]
            else:
                i += 1
        
        # Find first missing positive
        for i in range(n):
            if nums[i] != i + 1:
                return i + 1
        
        return n + 1
    ```

=== "📊 Hash Set Approach"

    ```python
    def first_missing_positive_set(nums):
        """
        Using hash set - O(n) space but easier to understand.
        """
        num_set = set(nums)
        
        for i in range(1, len(nums) + 2):
            if i not in num_set:
                return i
    ```

=== "📝 Tips & Insights"

    **🎯 Key Insights:**
    - First missing positive is always in range [1, n+1]
    - Array indices can serve as hash map
    - Use sign to mark presence without extra space
    
    **💡 Interview Tips:**
    - Mention the O(n) space solution first
    - Explain why range is [1, n+1]
    - Show how to use array as hash map
    
    **⚠️ Common Pitfalls:**
    - Not handling duplicate numbers
    - Forgetting edge case of all negatives
    - Off-by-one errors in indexing

---

## Problem 3: Trapping Rain Water

**Difficulty:** Hard  
**Pattern:** Two Pointers/Stack  
**Time:** O(n) | **Space:** O(1)

### Problem Statement

Given `n` non-negative integers representing an elevation map where the width of each bar is 1, compute how much water can be trapped after raining.

**Example:**
```text
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
```

=== "💡 Optimal: Two Pointers"

    ```python
    def trap_water_two_pointers(height):
        """
        Two pointers approach - O(1) space.
        
        Key insight: Water level at position i is min(max_left, max_right).
        We can process from both ends simultaneously.
        """
        if not height or len(height) < 3:
            return 0
        
        left, right = 0, len(height) - 1
        left_max = right_max = water = 0
        
        while left < right:
            if height[left] < height[right]:
                # Process left side
                if height[left] >= left_max:
                    left_max = height[left]
                else:
                    water += left_max - height[left]
                left += 1
            else:
                # Process right side
                if height[right] >= right_max:
                    right_max = height[right]
                else:
                    water += right_max - height[right]
                right -= 1
        
        return water
    ```

=== "📊 DP Approach"

    ```python
    def trap_water_dp(height):
        """
        Dynamic programming - precompute max heights.
        """
        if not height:
            return 0
        
        n = len(height)
        left_max = [0] * n
        right_max = [0] * n
        
        # Fill left_max array
        left_max[0] = height[0]
        for i in range(1, n):
            left_max[i] = max(left_max[i-1], height[i])
        
        # Fill right_max array
        right_max[n-1] = height[n-1]
        for i in range(n-2, -1, -1):
            right_max[i] = max(right_max[i+1], height[i])
        
        # Calculate water trapped
        water = 0
        for i in range(n):
            water += min(left_max[i], right_max[i]) - height[i]
        
        return water
    ```

=== "🔄 Stack Approach"

    ```python
    def trap_water_stack(height):
        """
        Using stack to track boundaries.
        """
        stack = []
        water = 0
        
        for i, h in enumerate(height):
            while stack and height[stack[-1]] < h:
                top = stack.pop()
                if not stack:
                    break
                
                # Calculate water between current and stack top
                distance = i - stack[-1] - 1
                bounded_height = min(h, height[stack[-1]]) - height[top]
                water += distance * bounded_height
            
            stack.append(i)
        
        return water
    ```

=== "📝 Tips & Insights"

    **🎯 Key Insights:**
    - Water level = min(max_left, max_right) - current_height
    - Two pointers eliminate need for preprocessing
    - Stack approach processes water level by level
    
    **💡 Interview Tips:**
    - Draw the water levels visually
    - Start with DP, then optimize to two pointers
    - Explain the invariant clearly
    
    **⚠️ Common Pitfalls:**
    - Not handling empty arrays
    - Off-by-one in pointer movements
    - Forgetting to check stack emptiness

---

## Problem 4: Sliding Window Maximum

**Difficulty:** Hard  
**Pattern:** Deque/Monotonic Queue  
**Time:** O(n) | **Space:** O(k)

### Problem Statement

Given an array `nums` and sliding window of size `k`, return the maximum for each window position.

**Example:**
```text
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
```

=== "💡 Optimal: Deque"

    ```python
    from collections import deque
    
    def max_sliding_window(nums, k):
        """
        Monotonic deque to maintain window maximum.
        
        Key insight: Maintain decreasing deque of indices.
        Front always contains maximum of current window.
        """
        if not nums or k == 0:
            return []
        
        dq = deque()  # Store indices
        result = []
        
        for i in range(len(nums)):
            # Remove indices outside current window
            while dq and dq[0] <= i - k:
                dq.popleft()
            
            # Remove indices with smaller values (they can't be max)
            while dq and nums[dq[-1]] <= nums[i]:
                dq.pop()
            
            dq.append(i)
            
            # Add to result when window is complete
            if i >= k - 1:
                result.append(nums[dq[0]])
        
        return result
    ```

=== "🔄 Brute Force"

    ```python
    def max_sliding_window_brute(nums, k):
        """
        Check maximum for each window - O(n*k) time.
        """
        result = []
        
        for i in range(len(nums) - k + 1):
            window_max = max(nums[i:i+k])
            result.append(window_max)
        
        return result
    ```

=== "📊 Heap Approach"

    ```python
    import heapq
    
    def max_sliding_window_heap(nums, k):
        """
        Using max heap with lazy deletion.
        """
        max_heap = []
        result = []
        
        for i in range(len(nums)):
            # Add current element (negative for max heap)
            heapq.heappush(max_heap, (-nums[i], i))
            
            # Remove elements outside window
            while max_heap and max_heap[0][1] <= i - k:
                heapq.heappop(max_heap)
            
            # Add to result when window is complete
            if i >= k - 1:
                result.append(-max_heap[0][0])
        
        return result
    ```

=== "📝 Tips & Insights"

    **🎯 Key Insights:**
    - Deque maintains potential maximums in decreasing order
    - Remove elements that can never be maximum
    - Window constraint handled by index checking
    
    **💡 Interview Tips:**
    - Explain why smaller elements are removed
    - Show deque state for sample input
    - Mention heap alternative with O(n log k)
    
    **⚠️ Common Pitfalls:**
    - Using values instead of indices in deque
    - Not removing out-of-window elements
    - Confusing max heap implementation

---

## Problem 5: Longest Consecutive Sequence

**Difficulty:** Hard  
**Pattern:** Hash Set  
**Time:** O(n) | **Space:** O(n)

### Problem Statement

Given an unsorted array of integers `nums`, return the length of the longest consecutive elements sequence.

**Example:**
```text
Input: nums = [100,4,200,1,3,2]
Output: 4
Explanation: [1, 2, 3, 4] is the longest consecutive sequence.
```

=== "💡 Optimal Solution"

    ```python
    def longest_consecutive(nums):
        """
        Hash set for O(1) lookups to build sequences.
        
        Key insight: Only start counting from numbers that are 
        beginnings of sequences (no num-1 exists).
        """
        if not nums:
            return 0
        
        num_set = set(nums)
        max_length = 0
        
        for num in num_set:
            # Only start sequence from the beginning
            if num - 1 not in num_set:
                current = num
                current_length = 1
                
                # Extend sequence as far as possible
                while current + 1 in num_set:
                    current += 1
                    current_length += 1
                
                max_length = max(max_length, current_length)
        
        return max_length
    ```

=== "🔄 Union-Find Approach"

    ```python
    class UnionFind:
        def __init__(self, nums):
            self.parent = {num: num for num in nums}
            self.size = {num: 1 for num in nums}
        
        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        
        def union(self, x, y):
            px, py = self.find(x), self.find(y)
            if px != py:
                if self.size[px] < self.size[py]:
                    px, py = py, px
                self.parent[py] = px
                self.size[px] += self.size[py]
    
    def longest_consecutive_uf(nums):
        """
        Union-Find approach - connects consecutive numbers.
        """
        if not nums:
            return 0
        
        uf = UnionFind(nums)
        num_set = set(nums)
        
        for num in nums:
            if num + 1 in num_set:
                uf.union(num, num + 1)
        
        return max(uf.size.values())
    ```

=== "📊 Sorting Approach"

    ```python
    def longest_consecutive_sort(nums):
        """
        Sort first, then find consecutive sequence - O(n log n).
        """
        if not nums:
            return 0
        
        nums = sorted(set(nums))  # Remove duplicates
        max_length = current_length = 1
        
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1] + 1:
                current_length += 1
            else:
                max_length = max(max_length, current_length)
                current_length = 1
        
        return max(max_length, current_length)
    ```

=== "📝 Tips & Insights"

    **🎯 Key Insights:**
    - Start sequences only from their beginnings
    - Hash set enables O(1) consecutive checks
    - Each number visited at most twice
    
    **💡 Interview Tips:**
    - Explain why we check for sequence start
    - Show time complexity analysis
    - Mention sorting as simpler alternative
    
    **⚠️ Common Pitfalls:**
    - Starting sequences from any number
    - Not handling duplicates properly
    - Overcomplicated union-find implementation

---

## Problem 6: Minimum Window Substring

**Difficulty:** Hard  
**Pattern:** Sliding Window  
**Time:** O(m+n) | **Space:** O(m+n)

### Problem Statement

Given strings `s` and `t`, return the minimum window substring of `s` such that every character in `t` is included in the window.

**Example:**
```text
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
```

=== "💡 Optimal Solution"

    ```python
    from collections import Counter, defaultdict
    
    def min_window_substring(s, t):
        """
        Sliding window with character frequency tracking.
        
        Key insight: Expand window until valid, then contract
        while maintaining validity.
        """
        if not s or not t or len(s) < len(t):
            return ""
        
        # Character frequencies in t
        required = Counter(t)
        required_count = len(required)
        
        # Sliding window variables
        left = right = 0
        formed = 0  # Characters that have desired frequency
        window_counts = defaultdict(int)
        
        # Result tracking
        min_len = float('inf')
        min_left = 0
        
        while right < len(s):
            # Expand window
            char = s[right]
            window_counts[char] += 1
            
            # Check if frequency matches requirement
            if char in required and window_counts[char] == required[char]:
                formed += 1
            
            # Contract window if valid
            while formed == required_count:
                # Update result if current window is smaller
                if right - left + 1 < min_len:
                    min_len = right - left + 1
                    min_left = left
                
                # Remove leftmost character
                left_char = s[left]
                window_counts[left_char] -= 1
                if left_char in required and window_counts[left_char] < required[left_char]:
                    formed -= 1
                
                left += 1
            
            right += 1
        
        return "" if min_len == float('inf') else s[min_left:min_left + min_len]
    ```

=== "🔄 Optimized with Filtering"

    ```python
    def min_window_optimized(s, t):
        """
        Pre-filter string to only include relevant characters.
        """
        if not s or not t:
            return ""
        
        required = Counter(t)
        
        # Filter s to only characters in t
        filtered_s = [(i, char) for i, char in enumerate(s) if char in required]
        
        if not filtered_s:
            return ""
        
        left = right = 0
        formed = 0
        window_counts = defaultdict(int)
        
        min_len = float('inf')
        min_left = 0
        
        while right < len(filtered_s):
            char = filtered_s[right][1]
            window_counts[char] += 1
            
            if window_counts[char] == required[char]:
                formed += 1
            
            while formed == len(required):
                start_idx = filtered_s[left][0]
                end_idx = filtered_s[right][0]
                
                if end_idx - start_idx + 1 < min_len:
                    min_len = end_idx - start_idx + 1
                    min_left = start_idx
                
                left_char = filtered_s[left][1]
                window_counts[left_char] -= 1
                if window_counts[left_char] < required[left_char]:
                    formed -= 1
                
                left += 1
            
            right += 1
        
        return "" if min_len == float('inf') else s[min_left:min_left + min_len]
    ```

=== "📝 Tips & Insights"

    **🎯 Key Insights:**
    - Two-pointer sliding window technique
    - Track when window becomes valid/invalid
    - Contract window while maintaining validity
    
    **💡 Interview Tips:**
    - Draw window expansion/contraction
    - Explain the formed counter logic
    - Mention filtering optimization
    
    **⚠️ Common Pitfalls:**
    - Not handling character frequencies correctly
    - Infinite loop in window contraction
    - Off-by-one errors in substring extraction

---

## Problem 7: Maximum Rectangle

**Difficulty:** Hard  
**Pattern:** Stack + Histogram  
**Time:** O(m×n) | **Space:** O(n)

### Problem Statement

Given a binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area.

**Example:**
```text
Input: matrix = [
  ["1","0","1","0","0"],
  ["1","0","1","1","1"],
  ["1","1","1","1","1"],
  ["1","0","0","1","0"]
]
Output: 6
```

=== "💡 Optimal Solution"

    ```python
    def maximal_rectangle(matrix):
        """
        Convert to largest rectangle in histogram for each row.
        
        Key insight: Each row can be treated as base of histogram
        where height[j] = consecutive 1's ending at current row.
        """
        if not matrix or not matrix[0]:
            return 0
        
        rows, cols = len(matrix), len(matrix[0])
        heights = [0] * cols
        max_area = 0
        
        for i in range(rows):
            # Update histogram heights
            for j in range(cols):
                if matrix[i][j] == '1':
                    heights[j] += 1
                else:
                    heights[j] = 0
            
            # Find max rectangle in current histogram
            max_area = max(max_area, largest_rectangle_histogram(heights))
        
        return max_area
    
    def largest_rectangle_histogram(heights):
        """
        Find largest rectangle in histogram using stack.
        """
        stack = []
        max_area = 0
        index = 0
        
        while index < len(heights):
            if not stack or heights[index] >= heights[stack[-1]]:
                stack.append(index)
                index += 1
            else:
                top = stack.pop()
                width = index if not stack else index - stack[-1] - 1
                area = heights[top] * width
                max_area = max(max_area, area)
        
        while stack:
            top = stack.pop()
            width = index if not stack else index - stack[-1] - 1
            area = heights[top] * width
            max_area = max(max_area, area)
        
        return max_area
    ```

=== "🔄 DP Approach"

    ```python
    def maximal_rectangle_dp(matrix):
        """
        Dynamic programming approach tracking left, right, height.
        """
        if not matrix or not matrix[0]:
            return 0
        
        rows, cols = len(matrix), len(matrix[0])
        heights = [0] * cols
        left = [0] * cols    # Leftmost position where height >= heights[j]
        right = [cols] * cols # Rightmost position where height >= heights[j]
        max_area = 0
        
        for i in range(rows):
            current_left = 0
            current_right = cols
            
            # Update heights
            for j in range(cols):
                if matrix[i][j] == '1':
                    heights[j] += 1
                else:
                    heights[j] = 0
            
            # Update left boundary
            for j in range(cols):
                if matrix[i][j] == '1':
                    left[j] = max(left[j], current_left)
                else:
                    left[j] = 0
                    current_left = j + 1
            
            # Update right boundary
            for j in range(cols - 1, -1, -1):
                if matrix[i][j] == '1':
                    right[j] = min(right[j], current_right)
                else:
                    right[j] = cols
                    current_right = j
            
            # Calculate max area
            for j in range(cols):
                max_area = max(max_area, heights[j] * (right[j] - left[j]))
        
        return max_area
    ```

=== "📝 Tips & Insights"

    **🎯 Key Insights:**
    - Transform 2D problem to multiple 1D histogram problems
    - Use stack for efficient histogram rectangle calculation
    - Heights array tracks consecutive 1's ending at current row
    
    **💡 Interview Tips:**
    - Start with histogram subproblem explanation
    - Show how matrix transforms to histogram
    - Mention DP alternative approach
    
    **⚠️ Common Pitfalls:**
    - Not resetting heights for 0's
    - Stack implementation errors
    - Off-by-one in width calculations

---

## Problem 8: Largest Rectangle in Histogram

**Difficulty:** Hard  
**Pattern:** Monotonic Stack  
**Time:** O(n) | **Space:** O(n)

### Problem Statement

Given an array of integers `heights` representing histogram bar heights, return the area of the largest rectangle.

**Example:**
```text
Input: heights = [2,1,5,6,2,3]
Output: 10
Explanation: Rectangle with height 5 and width 2.
```

=== "💡 Optimal Solution"

    ```python
    def largest_rectangle_area(heights):
        """
        Monotonic stack to track increasing heights.
        
        Key insight: For each bar, find the maximum width
        where it can be the shortest bar in the rectangle.
        """
        stack = []
        max_area = 0
        index = 0
        
        while index < len(heights):
            # If current bar is higher, push to stack
            if not stack or heights[index] >= heights[stack[-1]]:
                stack.append(index)
                index += 1
            else:
                # Pop and calculate area with popped bar as smallest
                top = stack.pop()
                
                # Width calculation:
                # - If stack empty: width = current index
                # - Else: width = current index - stack top - 1
                width = index if not stack else index - stack[-1] - 1
                area = heights[top] * width
                max_area = max(max_area, area)
        
        # Process remaining bars in stack
        while stack:
            top = stack.pop()
            width = index if not stack else index - stack[-1] - 1
            area = heights[top] * width
            max_area = max(max_area, area)
        
        return max_area
    ```

=== "🔄 Cleaner Implementation"

    ```python
    def largest_rectangle_area_clean(heights):
        """
        Add sentinel values for cleaner implementation.
        """
        # Add sentinels: 0 at start and end
        heights = [0] + heights + [0]
        stack = []
        max_area = 0
        
        for i, h in enumerate(heights):
            while stack and heights[stack[-1]] > h:
                height = heights[stack.pop()]
                width = i - stack[-1] - 1
                max_area = max(max_area, height * width)
            stack.append(i)
        
        return max_area
    ```

=== "📊 Divide & Conquer"

    ```python
    def largest_rectangle_divide_conquer(heights):
        """
        Divide and conquer approach - O(n log n) average, O(n²) worst.
        """
        def helper(start, end):
            if start > end:
                return 0
            
            # Find minimum height in range
            min_idx = start
            for i in range(start, end + 1):
                if heights[i] < heights[min_idx]:
                    min_idx = i
            
            # Maximum area is either:
            # 1. Rectangle using min height across entire range
            # 2. Max rectangle in left part
            # 3. Max rectangle in right part
            min_area = heights[min_idx] * (end - start + 1)
            left_area = helper(start, min_idx - 1)
            right_area = helper(min_idx + 1, end)
            
            return max(min_area, left_area, right_area)
        
        return helper(0, len(heights) - 1)
    ```

=== "📝 Tips & Insights"

    **🎯 Key Insights:**
    - Stack maintains increasing height sequence
    - When popping, calculate rectangle with popped height
    - Width determined by current position and stack top
    
    **💡 Interview Tips:**
    - Draw stack state changes
    - Explain why we pop when height decreases
    - Show width calculation logic
    
    **⚠️ Common Pitfalls:**
    - Wrong width calculation
    - Not handling empty stack case
    - Forgetting to process remaining stack

---

## Problem 9: Merge k Sorted Arrays

**Difficulty:** Hard  
**Pattern:** Divide & Conquer/Heap  
**Time:** O(n log k) | **Space:** O(log k)

### Problem Statement

Merge k sorted arrays into one sorted array.

**Example:**
```text
Input: arrays = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
```

=== "💡 Optimal: Divide & Conquer"

    ```python
    def merge_k_sorted_arrays(arrays):
        """
        Divide and conquer approach - merge pairs recursively.
        
        Key insight: Merge pairs of arrays until one remains.
        Time: O(n log k), Space: O(log k) for recursion.
        """
        if not arrays:
            return []
        
        def merge_two(arr1, arr2):
            """Merge two sorted arrays."""
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
        
        # Divide and conquer
        while len(arrays) > 1:
            merged_arrays = []
            
            # Merge pairs
            for i in range(0, len(arrays), 2):
                arr1 = arrays[i]
                arr2 = arrays[i + 1] if i + 1 < len(arrays) else []
                merged_arrays.append(merge_two(arr1, arr2))
            
            arrays = merged_arrays
        
        return arrays[0]
    ```

=== "🔄 Min Heap Approach"

    ```python
    import heapq
    
    def merge_k_sorted_heap(arrays):
        """
        Use min heap to always get the smallest element.
        """
        if not arrays:
            return []
        
        heap = []
        result = []
        
        # Initialize heap with first element from each array
        for i, array in enumerate(arrays):
            if array:
                heapq.heappush(heap, (array[0], i, 0))  # (value, array_idx, element_idx)
        
        while heap:
            val, array_idx, element_idx = heapq.heappop(heap)
            result.append(val)
            
            # Add next element from the same array
            if element_idx + 1 < len(arrays[array_idx]):
                next_val = arrays[array_idx][element_idx + 1]
                heapq.heappush(heap, (next_val, array_idx, element_idx + 1))
        
        return result
    ```

=== "📊 Iterative Merging"

    ```python
    def merge_k_sorted_iterative(arrays):
        """
        Merge arrays one by one - less efficient but simpler.
        """
        if not arrays:
            return []
        
        result = arrays[0]
        
        for i in range(1, len(arrays)):
            result = merge_two_arrays(result, arrays[i])
        
        return result
    
    def merge_two_arrays(arr1, arr2):
        """Helper to merge two sorted arrays."""
        result = []
        i = j = 0
        
        while i < len(arr1) and j < len(arr2):
            if arr1[i] <= arr2[j]:
                result.append(arr1[i])
                i += 1
            else:
                result.append(arr2[j])
                j += 1
        
        result.extend(arr1[i:])
        result.extend(arr2[j:])
        return result
    ```

=== "📝 Tips & Insights"

    **🎯 Key Insights:**
    - Divide & conquer gives optimal O(n log k) complexity
    - Heap approach good for streaming/online scenarios
    - Each element processed exactly once
    
    **💡 Interview Tips:**
    - Compare heap vs divide & conquer approaches
    - Explain time complexity analysis
    - Mention space optimization possibilities
    
    **⚠️ Common Pitfalls:**
    - Not handling empty arrays
    - Heap index management errors
    - Forgetting to extend remaining elements

---

## Problem 10: Count of Smaller Numbers After Self

**Difficulty:** Hard  
**Pattern:** Merge Sort/Fenwick Tree  
**Time:** O(n log n) | **Space:** O(n)

### Problem Statement

Given an integer array `nums`, return an integer array `counts` where `counts[i]` is the number of smaller elements to the right of `nums[i]`.

**Example:**
```text
Input: nums = [5,2,6,1]
Output: [2,1,1,0]
```

=== "💡 Optimal: Merge Sort"

    ```python
    def count_smaller(nums):
        """
        Modified merge sort to count inversions.
        
        Key insight: During merge, count how many elements
        from right array are smaller than left array elements.
        """
        def merge_sort(arr):
            if len(arr) <= 1:
                return arr
            
            mid = len(arr) // 2
            left = merge_sort(arr[:mid])
            right = merge_sort(arr[mid:])
            
            return merge(left, right)
        
        def merge(left, right):
            result = []
            i = j = 0
            
            while i < len(left) and j < len(right):
                if left[i][0] > right[j][0]:
                    result.append(right[j])
                    j += 1
                else:
                    # All elements in right[:j] are smaller than left[i]
                    counts[left[i][1]] += j
                    result.append(left[i])
                    i += 1
            
            # Add remaining elements
            while i < len(left):
                counts[left[i][1]] += j  # All of right array is smaller
                result.append(left[i])
                i += 1
            
            result.extend(right[j:])
            return result
        
        # Create array of (value, original_index) pairs
        indexed_nums = [(nums[i], i) for i in range(len(nums))]
        counts = [0] * len(nums)
        
        merge_sort(indexed_nums)
        return counts
    ```

=== "🔄 Fenwick Tree (BIT)"

    ```python
    class FenwickTree:
        def __init__(self, size):
            self.size = size
            self.tree = [0] * (size + 1)
        
        def update(self, i, delta):
            while i <= self.size:
                self.tree[i] += delta
                i += i & (-i)
        
        def query(self, i):
            result = 0
            while i > 0:
                result += self.tree[i]
                i -= i & (-i)
            return result
    
    def count_smaller_fenwick(nums):
        """
        Use Fenwick Tree for range sum queries.
        """
        if not nums:
            return []
        
        # Coordinate compression
        sorted_nums = sorted(set(nums))
        rank = {v: i + 1 for i, v in enumerate(sorted_nums)}
        
        ft = FenwickTree(len(sorted_nums))
        counts = []
        
        # Process from right to left
        for i in range(len(nums) - 1, -1, -1):
            # Query for numbers smaller than current
            smaller_count = ft.query(rank[nums[i]] - 1)
            counts.append(smaller_count)
            
            # Add current number to tree
            ft.update(rank[nums[i]], 1)
        
        return counts[::-1]
    ```

=== "📊 Brute Force"

    ```python
    def count_smaller_brute(nums):
        """
        Check each element against all elements to its right.
        """
        counts = []
        
        for i in range(len(nums)):
            count = 0
            for j in range(i + 1, len(nums)):
                if nums[j] < nums[i]:
                    count += 1
            counts.append(count)
        
        return counts
    ```

=== "📝 Tips & Insights"

    **🎯 Key Insights:**
    - Merge sort naturally counts inversions during merge
    - Fenwick Tree good for range queries after coordinate compression
    - Process right-to-left for accumulative counting
    
    **💡 Interview Tips:**
    - Start with brute force O(n²) solution
    - Explain inversion counting concept
    - Show coordinate compression for Fenwick Tree
    
    **⚠️ Common Pitfalls:**
    - Not preserving original indices during merge sort
    - Fenwick Tree indexing errors (1-based)
    - Forgetting coordinate compression

---

## Problem 11: Best Time to Buy and Sell Stock with Cooldown

**Difficulty:** Hard  
**Pattern:** Dynamic Programming  
**Time:** O(n) | **Space:** O(1)

### Problem Statement

You can buy and sell stocks with unlimited transactions, but must cooldown for one day after selling.

**Example:**
```text
Input: prices = [1,2,3,0,2]
Output: 3
Explanation: buy[0] -> sell[1] -> cooldown -> buy[3] -> sell[4]
```

=== "💡 Optimal Solution"

    ```python
    def max_profit_cooldown(prices):
        """
        State machine DP with three states.
        
        States:
        - hold: Currently holding stock
        - sold: Just sold stock (must cooldown)
        - rest: Not holding stock and can buy
        """
        if len(prices) < 2:
            return 0
        
        # Initialize states
        hold = -prices[0]  # Bought on day 0
        sold = 0           # Can't sell on day 0
        rest = 0           # Starting state
        
        for i in range(1, len(prices)):
            prev_hold, prev_sold, prev_rest = hold, sold, rest
            
            # Transitions
            hold = max(prev_hold, prev_rest - prices[i])  # Keep holding or buy today
            sold = prev_hold + prices[i]                  # Sell today
            rest = max(prev_rest, prev_sold)              # Rest or continue resting
        
        # Return max of sold or rest (don't want to end holding)
        return max(sold, rest)
    ```

=== "🔄 Alternative DP"

    ```python
    def max_profit_cooldown_alt(prices):
        """
        Two-state DP: buy and sell.
        """
        if len(prices) < 2:
            return 0
        
        n = len(prices)
        buy = [0] * n
        sell = [0] * n
        
        buy[0] = -prices[0]
        buy[1] = max(-prices[0], -prices[1])
        sell[1] = max(0, prices[1] - prices[0])
        
        for i in range(2, n):
            # Either don't buy today or buy today (after cooldown)
            buy[i] = max(buy[i-1], sell[i-2] - prices[i])
            # Either don't sell today or sell today
            sell[i] = max(sell[i-1], buy[i-1] + prices[i])
        
        return sell[n-1]
    ```

=== "📊 Space Optimized"

    ```python
    def max_profit_cooldown_optimized(prices):
        """
        Space-optimized version using variables.
        """
        if len(prices) < 2:
            return 0
        
        # Variables represent: buy[i-1], sell[i-1], sell[i-2]
        buy, sell, prev_sell = -prices[0], 0, 0
        
        for i in range(1, len(prices)):
            prev_buy = buy
            buy = max(buy, prev_sell - prices[i])
            prev_sell = sell
            sell = max(sell, prev_buy + prices[i])
        
        return sell
    ```

=== "📝 Tips & Insights"

    **🎯 Key Insights:**
    - Model as state machine with transitions
    - Cooldown affects when you can buy again
    - Track multiple states simultaneously
    
    **💡 Interview Tips:**
    - Draw state diagram first
    - Explain each state transition
    - Show space optimization technique
    
    **⚠️ Common Pitfalls:**
    - Not handling cooldown constraint properly
    - Wrong state transitions
    - Edge cases with few days

---

## Problem 12: Maximal Rectangle

**Difficulty:** Hard  
**Pattern:** Stack + DP  
**Time:** O(m×n) | **Space:** O(n)

### Problem Statement

Given a `rows x cols` binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's.

*Note: This is similar to Problem 7 but worth reinforcing the concept.*

=== "💡 Optimal Solution"

    ```python
    def maximal_rectangle(matrix):
        """
        Convert each row to histogram and find max rectangle.
        """
        if not matrix or not matrix[0]:
            return 0
        
        rows, cols = len(matrix), len(matrix[0])
        heights = [0] * cols
        max_area = 0
        
        for row in matrix:
            # Update histogram heights
            for j in range(cols):
                heights[j] = heights[j] + 1 if row[j] == '1' else 0
            
            # Find max rectangle in current histogram
            max_area = max(max_area, largest_rectangle_in_histogram(heights))
        
        return max_area
    
    def largest_rectangle_in_histogram(heights):
        """Find largest rectangle using stack."""
        stack = []
        max_area = 0
        
        for i, h in enumerate(heights + [0]):  # Add sentinel
            while stack and heights[stack[-1]] > h:
                height = heights[stack.pop()]
                width = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, height * width)
            stack.append(i)
        
        return max_area
    ```

=== "📝 Tips & Insights"

    **🎯 Key Insights:**
    - Reduce 2D problem to multiple 1D problems
    - Each row becomes base of histogram
    - Stack efficiently finds max rectangle
    
    **💡 Interview Tips:**
    - Connect to histogram problem
    - Show height array evolution
    - Explain stack-based rectangle finding

---

## Problem 13: Russian Doll Envelopes

**Difficulty:** Hard  
**Pattern:** DP + Binary Search  
**Time:** O(n log n) | **Space:** O(n)

### Problem Statement

You have envelopes with widths and heights. One envelope can fit into another if both width and height are strictly smaller.

**Example:**
```text
Input: envelopes = [[5,4],[6,4],[6,7],[2,3]]
Output: 3
Explanation: [2,3] -> [5,4] -> [6,7]
```

=== "💡 Optimal Solution"

    ```python
    def max_envelopes(envelopes):
        """
        Sort by width, then find LIS on heights.
        
        Key insight: Sort by width ascending, height descending
        to handle same widths correctly.
        """
        if not envelopes:
            return 0
        
        # Sort: width ascending, height descending
        envelopes.sort(key=lambda x: (x[0], -x[1]))
        
        # Extract heights and find LIS
        heights = [env[1] for env in envelopes]
        return length_of_lis(heights)
    
    def length_of_lis(nums):
        """Find length of longest increasing subsequence using binary search."""
        if not nums:
            return 0
        
        tails = []  # tails[i] = smallest tail of increasing subsequence of length i+1
        
        for num in nums:
            # Binary search for position
            left, right = 0, len(tails)
            while left < right:
                mid = (left + right) // 2
                if tails[mid] < num:
                    left = mid + 1
                else:
                    right = mid
            
            # If num is larger than all elements, append
            if left == len(tails):
                tails.append(num)
            else:
                tails[left] = num  # Replace with smaller value
        
        return len(tails)
    ```

=== "🔄 DP Approach"

    ```python
    def max_envelopes_dp(envelopes):
        """
        Standard DP approach - O(n²) time.
        """
        if not envelopes:
            return 0
        
        envelopes.sort()
        n = len(envelopes)
        dp = [1] * n
        
        for i in range(1, n):
            for j in range(i):
                if (envelopes[j][0] < envelopes[i][0] and 
                    envelopes[j][1] < envelopes[i][1]):
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)
    ```

=== "📝 Tips & Insights"

    **🎯 Key Insights:**
    - Reduce to Longest Increasing Subsequence problem
    - Clever sorting handles 2D constraint
    - Binary search optimizes LIS to O(n log n)
    
    **💡 Interview Tips:**
    - Explain sorting strategy for same widths
    - Connect to classic LIS problem
    - Show binary search optimization
    
    **⚠️ Common Pitfalls:**
    - Wrong sorting order for equal widths
    - Not handling strict inequality
    - Binary search implementation errors

---

## Problem 14: Count Inversions in Array

**Difficulty:** Hard  
**Pattern:** Merge Sort  
**Time:** O(n log n) | **Space:** O(n)

### Problem Statement

Count the number of inversions in an array. An inversion is when `i < j` but `arr[i] > arr[j]`.

**Example:**
```text
Input: arr = [8, 4, 2, 1]
Output: 6
Explanation: (8,4), (8,2), (8,1), (4,2), (4,1), (2,1)
```

=== "💡 Optimal Solution"

    ```python
    def count_inversions(arr):
        """
        Modified merge sort to count inversions.
        
        Key insight: During merge, when taking from right array,
        count how many elements from left array are greater.
        """
        def merge_sort_count(arr, temp, left, right):
            inv_count = 0
            if left < right:
                mid = (left + right) // 2
                
                inv_count += merge_sort_count(arr, temp, left, mid)
                inv_count += merge_sort_count(arr, temp, mid + 1, right)
                inv_count += merge_count(arr, temp, left, mid, right)
            
            return inv_count
        
        def merge_count(arr, temp, left, mid, right):
            i, j, k = left, mid + 1, left
            inv_count = 0
            
            while i <= mid and j <= right:
                if arr[i] <= arr[j]:
                    temp[k] = arr[i]
                    i += 1
                else:
                    temp[k] = arr[j]
                    # All elements from arr[i] to arr[mid] are > arr[j]
                    inv_count += (mid - i + 1)
                    j += 1
                k += 1
            
            # Copy remaining elements
            while i <= mid:
                temp[k] = arr[i]
                i += 1
                k += 1
            
            while j <= right:
                temp[k] = arr[j]
                j += 1
                k += 1
            
            # Copy back to original array
            for i in range(left, right + 1):
                arr[i] = temp[i]
            
            return inv_count
        
        temp = [0] * len(arr)
        return merge_sort_count(arr.copy(), temp, 0, len(arr) - 1)
    ```

=== "🔄 Brute Force"

    ```python
    def count_inversions_brute(arr):
        """
        Check all pairs - O(n²) time.
        """
        count = 0
        n = len(arr)
        
        for i in range(n):
            for j in range(i + 1, n):
                if arr[i] > arr[j]:
                    count += 1
        
        return count
    ```

=== "📝 Tips & Insights"

    **🎯 Key Insights:**
    - Merge sort naturally exposes inversions
    - Count inversions during merge step
    - Efficient O(n log n) vs brute force O(n²)
    
    **💡 Interview Tips:**
    - Start with brute force approach
    - Explain merge sort modification
    - Show inversion counting logic
    
    **⚠️ Common Pitfalls:**
    - Wrong inversion count during merge
    - Not copying arrays correctly
    - Off-by-one errors in ranges

---

## Problem 15: Maximum Subarray Product

**Difficulty:** Hard  
**Pattern:** Dynamic Programming  
**Time:** O(n) | **Space:** O(1)

### Problem Statement

Given an integer array `nums`, find a contiguous subarray that has the largest product.

**Example:**
```text
Input: nums = [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.
```

=== "💡 Optimal Solution"

    ```python
    def max_product(nums):
        """
        Track both maximum and minimum products.
        
        Key insight: Negative numbers can turn min into max,
        so we need to track both extremes.
        """
        if not nums:
            return 0
        
        max_prod = min_prod = result = nums[0]
        
        for i in range(1, len(nums)):
            num = nums[i]
            
            # If current number is negative, swap max and min
            if num < 0:
                max_prod, min_prod = min_prod, max_prod
            
            # Update max and min products
            max_prod = max(num, max_prod * num)
            min_prod = min(num, min_prod * num)
            
            # Update global result
            result = max(result, max_prod)
        
        return result
    ```

=== "🔄 Alternative Implementation"

    ```python
    def max_product_alt(nums):
        """
        More explicit handling of three cases.
        """
        if not nums:
            return 0
        
        max_prod = min_prod = result = nums[0]
        
        for i in range(1, len(nums)):
            num = nums[i]
            
            # Calculate new max and min
            candidates = [num, max_prod * num, min_prod * num]
            new_max = max(candidates)
            new_min = min(candidates)
            
            max_prod = new_max
            min_prod = new_min
            result = max(result, max_prod)
        
        return result
    ```

=== "📊 Prefix/Suffix Approach"

    ```python
    def max_product_prefix_suffix(nums):
        """
        Calculate max product using prefix and suffix products.
        """
        if not nums:
            return 0
        
        n = len(nums)
        prefix = suffix = 1
        max_prod = float('-inf')
        
        for i in range(n):
            # Reset if product becomes 0
            prefix = prefix * nums[i] if prefix != 0 else nums[i]
            suffix = suffix * nums[n - 1 - i] if suffix != 0 else nums[n - 1 - i]
            
            max_prod = max(max_prod, prefix, suffix)
        
        return max_prod
    ```

=== "📝 Tips & Insights"

    **🎯 Key Insights:**
    - Negative numbers can flip max/min products
    - Need to track both maximum and minimum
    - Zero resets the product calculation
    
    **💡 Interview Tips:**
    - Explain why we need both max and min
    - Show how negative numbers affect products
    - Mention prefix/suffix alternative approach
    
    **⚠️ Common Pitfalls:**
    - Not handling negative numbers correctly
    - Forgetting about zero elements
    - Not tracking minimum product

---

## 🎯 Summary & Next Steps

### ✅ What You've Mastered

- **Binary Search Variants:** Complex search spaces and optimization problems
- **Advanced Stack Applications:** Monotonic stacks and histogram problems
- **Sophisticated DP:** Multi-dimensional states and optimization
- **String/Array Hybrid Problems:** Complex pattern matching and window techniques

### 🚀 Advanced Interview Strategies

1. **Pattern Recognition:** Quickly identify the underlying algorithm pattern
2. **Optimization Thinking:** Always consider time/space tradeoffs
3. **Edge Case Handling:** Systematic approach to boundary conditions
4. **Multiple Solutions:** Present brute force, then optimize

### 📈 Performance Targets

- **Hard Problems:** 25-35 minutes including explanation
- **Code Quality:** Clean, readable, well-commented solutions
- **Testing:** Always verify with multiple test cases
- **Communication:** Clear explanation of approach and complexity

Master these hard problems and you'll be well-prepared for the most challenging technical interviews! 🎯

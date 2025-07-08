# Longest Increasing Subsequence Pattern

## Introduction

The Longest Increasing Subsequence (LIS) pattern is a fundamental dynamic programming pattern that deals with finding a subsequence of elements that are in strictly increasing order.

=== "Overview"
    **Core Idea**: Find a subsequence (not necessarily contiguous) of elements that are in increasing order.
    
    **When to Use**:
    
    - When you need to find a subsequence with elements in a specific order
    - When looking for the longest chain of elements with a certain property
    - When the problem involves finding an optimal subsequence with ordering constraints
    
    **Recurrence Relation**: `dp[i] = max(dp[j] + 1)` for all `j < i` where `nums[j] < nums[i]`
    
    **Real-World Applications**:
    
    - Finding the longest chain of activities that can be performed in order
    - Determining the maximum length of nested boxes or envelopes
    - Identifying the longest sequence of increasing temperatures
    - Planning project dependencies that must be executed in order

=== "Example Problems"
    - **Longest Increasing Subsequence**: Find the length of the longest subsequence that is strictly increasing
      - Problem: Given an array, find a subsequence where each element is larger than the previous
      - DP Solution: `dp[i]` represents the length of LIS ending at index i
    
    - **Maximum Sum Increasing Subsequence**: Find the increasing subsequence with the maximum possible sum
      - Problem: Like LIS, but we want to maximize the sum rather than just the length
      - Recurrence: `dp[i] = max(dp[j] + nums[i])` for all `j < i` where `nums[j] < nums[i]`
    
    - **Longest Chain of Pairs**: Given pairs of numbers (a,b), find the longest chain such that b of one pair < a of next pair
      - Problem: Sort by second element, then apply LIS pattern
      - Demonstrates how LIS can be applied to more complex ordering relationships
    
    - **Russian Doll Envelopes**: Nest envelopes inside each other based on width and height
      - Problem: Sort by width, then find LIS based on height
      - Shows how 2D ordering constraints can be reduced to 1D LIS

=== "Visualization"
    For the LIS problem with array `[10, 9, 2, 5, 3, 7, 101, 18]`:
    
    ```
    Array: [10, 9, 2, 5, 3, 7, 101, 18]
    dp[0] = 1 (just the element 10)
    dp[1] = 1 (just the element 9)
    dp[2] = 1 (just the element 2)
    dp[3] = 2 (elements [2, 5])
    dp[4] = 2 (elements [2, 3])
    dp[5] = 3 (elements [2, 3, 7])
    dp[6] = 4 (elements [2, 3, 7, 101])
    dp[7] = 4 (elements [2, 3, 7, 18])
    ```
    
    The LIS is of length 4 (e.g., [2, 3, 7, 101] or [2, 3, 7, 18])
    
    ![LIS Pattern Visualization](https://i.imgur.com/UCM2RuT.png)

=== "Implementation"
    **Standard O(n²) Implementation**:
    
    ```python
    def lengthOfLIS(nums):
        if not nums:
            return 0
        
        n = len(nums)
        dp = [1] * n  # Minimum length is 1
        
        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)
    ```
    
    **Binary Search Optimization (O(n log n))**:
    
    ```python
    def lengthOfLIS_optimized(nums):
        if not nums:
            return 0
        
        tails = []  # tails[i] = smallest end value of all LIS of length i+1
        
        for num in nums:
            # Binary search to find position to insert/replace
            left, right = 0, len(tails) - 1
            while left <= right:
                mid = (left + right) // 2
                if tails[mid] < num:
                    left = mid + 1
                else:
                    right = mid - 1
                    
            # Insert new element or replace existing one
            if left == len(tails):
                tails.append(num)
            else:
                tails[left] = num
        
        return len(tails)
    ```

=== "Tips and Insights"
    - **State Definition**: `dp[i]` typically represents the length of the LIS ending at index i
    - **Pattern Recognition**: Look for problems where you need to maintain an order or increasing relationship
    - **Variation**: The "increasing" condition can be replaced with other comparisons
    - **Optimization**: The binary search approach is not intuitive but provides significant performance gains
    - **Reconstruction**: To reconstruct the actual subsequence, maintain a parent/predecessor array
    - **Two-Dimensional**: Some problems require sorting by one dimension and applying LIS on another
    - **Beyond Numbers**: The pattern works for any objects with a well-defined ordering relation
    - **Multiple Constraints**: Can be extended to handle multiple constraints simultaneously
    - **Alternative Metric**: Instead of length, you might optimize for sum, product, or other metrics

# Dynamic Programming - Medium Problems

## 🎯 Learning Objectives

Master intermediate DP patterns including 2D DP, knapsack variants, and string algorithms:

- Advanced state transitions and optimization
- 2D dynamic programming techniques
- Knapsack and coin change variants
- String manipulation and subsequence problems
- Grid path problems and constraints

---

## Problem 1: Unique Paths

**Difficulty**: 🟡 Medium  
**Pattern**: 2D Grid DP  
**Time**: O(m×n), **Space**: O(n) optimized

=== "Problem"

    A robot is located at the top-left corner of a `m x n` grid. The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid.

    How many possible unique paths are there?

    **Example 1:**
    ```
    Input: m = 3, n = 7
    Output: 28
    ```

    **Example 2:**
    ```
    Input: m = 3, n = 2
    Output: 3
    Explanation: From top-left to bottom-right, there are 3 ways:
    1. Right -> Down -> Down
    2. Down -> Down -> Right
    3. Down -> Right -> Down
    ```

=== "Solution"

    ```python
    def uniquePaths(m, n):
        """
        2D DP approach - build table of paths to each cell.
        
        Time: O(m×n), Space: O(m×n)
        """
        # Create DP table
        dp = [[1] * n for _ in range(m)]
        
        # Fill the DP table
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        
        return dp[m-1][n-1]
    
    def uniquePaths_optimized(m, n):
        """
        Space-optimized: only keep previous row.
        
        Time: O(m×n), Space: O(n)
        """
        prev = [1] * n
        
        for i in range(1, m):
            curr = [1] * n
            for j in range(1, n):
                curr[j] = prev[j] + curr[j-1]
            prev = curr
        
        return prev[n-1]
    
    def uniquePaths_math(m, n):
        """
        Mathematical approach using combinations.
        
        Time: O(min(m,n)), Space: O(1)
        """
        # Total moves: (m-1) down + (n-1) right = m+n-2
        # Choose (m-1) positions for down moves: C(m+n-2, m-1)
        import math
        return math.comb(m + n - 2, m - 1)
    ```

=== "Insights"

    **Key Concepts:**
    
    - **State Definition**: `dp[i][j]` = number of paths to cell (i,j)
    - **Recurrence**: `dp[i][j] = dp[i-1][j] + dp[i][j-1]`
    - **Base Cases**: First row and column are all 1s
    - **Space Optimization**: Only need previous row to compute current row
    
    **Mathematical Insight:**
    The problem is equivalent to arranging (m-1) down moves and (n-1) right moves, which is a combinations problem.

---

## Problem 2: Coin Change

**Difficulty**: 🟡 Medium  
**Pattern**: Unbounded Knapsack  
**Time**: O(amount × coins), **Space**: O(amount)

=== "Problem"

    You are given an integer array `coins` representing coins of different denominations and an integer `amount` representing a total amount of money.

    Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return `-1`.

    **Example 1:**
    ```
    Input: coins = [1,3,4], amount = 6
    Output: 2
    Explanation: 6 = 3 + 3
    ```

    **Example 2:**
    ```
    Input: coins = [2], amount = 3
    Output: -1
    ```

=== "Solution"

    ```python
    def coinChange(coins, amount):
        """
        Bottom-up DP approach.
        
        Time: O(amount × len(coins))
        Space: O(amount)
        """
        # Initialize DP array
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0  # Base case: 0 coins needed for amount 0
        
        # Fill DP table
        for i in range(1, amount + 1):
            for coin in coins:
                if coin <= i:
                    dp[i] = min(dp[i], dp[i - coin] + 1)
        
        return dp[amount] if dp[amount] != float('inf') else -1
    
    def coinChange_top_down(coins, amount):
        """
        Top-down memoization approach.
        
        Time: O(amount × len(coins))
        Space: O(amount) for recursion + memoization
        """
        memo = {}
        
        def dp(remaining):
            if remaining == 0:
                return 0
            if remaining < 0:
                return float('inf')
            
            if remaining in memo:
                return memo[remaining]
            
            min_coins = float('inf')
            for coin in coins:
                min_coins = min(min_coins, dp(remaining - coin) + 1)
            
            memo[remaining] = min_coins
            return min_coins
        
        result = dp(amount)
        return result if result != float('inf') else -1
    
    def coinChange_bfs(coins, amount):
        """
        BFS approach - find shortest path to amount.
        
        Time: O(amount × len(coins))
        Space: O(amount)
        """
        if amount == 0:
            return 0
        
        from collections import deque
        queue = deque([0])
        visited = {0}
        level = 0
        
        while queue:
            level += 1
            for _ in range(len(queue)):
                current = queue.popleft()
                
                for coin in coins:
                    next_amount = current + coin
                    
                    if next_amount == amount:
                        return level
                    
                    if next_amount < amount and next_amount not in visited:
                        visited.add(next_amount)
                        queue.append(next_amount)
        
        return -1
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Unbounded Knapsack**: Can use each coin multiple times
    - **State Definition**: `dp[i]` = minimum coins needed for amount i
    - **Recurrence**: `dp[i] = min(dp[i-coin] + 1)` for all valid coins
    - **Base Case**: `dp[0] = 0` (zero coins for zero amount)
    
    **Multiple Approaches:**
    1. **Bottom-up DP**: Build solution from smaller amounts
    2. **Top-down Memoization**: Recursive with caching
    3. **BFS**: Treat as shortest path problem

---

## Problem 3: Longest Increasing Subsequence

**Difficulty**: 🟡 Medium  
**Pattern**: Sequence DP + Binary Search  
**Time**: O(n log n), **Space**: O(n)

=== "Problem"

    Given an integer array `nums`, return the length of the longest strictly increasing subsequence.

    **Example 1:**
    ```
    Input: nums = [10,22,9,33,21,50,41,60]
    Output: 5
    Explanation: The longest increasing subsequence is [10,22,33,50,60].
    ```

    **Example 2:**
    ```
    Input: nums = [0,1,0,3,2,3]
    Output: 4
    ```

=== "Solution"

    ```python
    def lengthOfLIS(nums):
        """
        DP approach with binary search optimization.
        
        Time: O(n log n), Space: O(n)
        """
        if not nums:
            return 0
        
        # tails[i] = smallest ending element of all increasing subsequences of length i+1
        tails = []
        
        for num in nums:
            # Binary search for the position to insert/replace
            left, right = 0, len(tails)
            
            while left < right:
                mid = (left + right) // 2
                if tails[mid] < num:
                    left = mid + 1
                else:
                    right = mid
            
            # If num is larger than all elements in tails, append it
            if left == len(tails):
                tails.append(num)
            else:
                # Replace the first element that is >= num
                tails[left] = num
        
        return len(tails)
    
    def lengthOfLIS_dp(nums):
        """
        Classic DP approach.
        
        Time: O(n²), Space: O(n)
        """
        if not nums:
            return 0
        
        n = len(nums)
        dp = [1] * n  # dp[i] = length of LIS ending at index i
        
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)
    
    def lengthOfLIS_with_sequence(nums):
        """
        Return both length and actual sequence.
        """
        if not nums:
            return 0, []
        
        n = len(nums)
        dp = [1] * n
        parent = [-1] * n
        
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i] and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j
        
        # Find the index with maximum LIS length
        max_length = max(dp)
        max_index = dp.index(max_length)
        
        # Reconstruct the sequence
        sequence = []
        current = max_index
        while current != -1:
            sequence.append(nums[current])
            current = parent[current]
        
        sequence.reverse()
        return max_length, sequence
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Patience Sorting**: Maintain smallest tail for each length
    - **Binary Search**: Find position to insert/replace efficiently
    - **Greedy Choice**: Always keep smallest possible tail
    - **State Definition**: `dp[i]` = length of LIS ending at index i
    
    **Algorithm Intuition:**
    Keep track of the smallest ending element for each possible LIS length. When processing a new element, either extend the longest sequence or replace an element to keep tails small.

---

## Problem 4: Edit Distance

**Difficulty**: 🟡 Medium  
**Pattern**: 2D String DP  
**Time**: O(m×n), **Space**: O(m×n)

=== "Problem"

    Given two strings `word1` and `word2`, return the minimum number of operations required to convert `word1` to `word2`.

    You have the following three operations permitted on a word:
    - Insert a character
    - Delete a character
    - Replace a character

    **Example 1:**
    ```
    Input: word1 = "horse", word2 = "ros"
    Output: 3
    Explanation: 
    horse -> rorse (replace 'h' with 'r')
    rorse -> rose (remove 'r')
    rose -> ros (remove 'e')
    ```

=== "Solution"

    ```python
    def minDistance(word1, word2):
        """
        2D DP approach for edit distance.
        
        Time: O(m×n), Space: O(m×n)
        """
        m, n = len(word1), len(word2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i  # Delete all characters from word1
        
        for j in range(n + 1):
            dp[0][j] = j  # Insert all characters of word2
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]  # Characters match, no operation needed
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # Delete from word1
                        dp[i][j-1],    # Insert into word1
                        dp[i-1][j-1]   # Replace in word1
                    )
        
        return dp[m][n]
    
    def minDistance_optimized(word1, word2):
        """
        Space-optimized version using only two rows.
        
        Time: O(m×n), Space: O(n)
        """
        m, n = len(word1), len(word2)
        
        # Use only two rows
        prev = list(range(n + 1))
        curr = [0] * (n + 1)
        
        for i in range(1, m + 1):
            curr[0] = i
            
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    curr[j] = prev[j-1]
                else:
                    curr[j] = 1 + min(prev[j], curr[j-1], prev[j-1])
            
            prev, curr = curr, prev
        
        return prev[n]
    
    def minDistance_with_operations(word1, word2):
        """
        Return both distance and actual operations.
        """
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        # Reconstruct operations
        operations = []
        i, j = m, n
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and word1[i-1] == word2[j-1]:
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                operations.append(f"Replace '{word1[i-1]}' with '{word2[j-1]}'")
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                operations.append(f"Delete '{word1[i-1]}'")
                i -= 1
            else:
                operations.append(f"Insert '{word2[j-1]}'")
                j -= 1
        
        operations.reverse()
        return dp[m][n], operations
    ```

=== "Insights"

    **Key Concepts:**
    
    - **2D State Space**: `dp[i][j]` = edit distance between first i chars of word1 and first j chars of word2
    - **Three Operations**: Insert, delete, replace each cost 1
    - **Optimal Substructure**: Optimal solution contains optimal solutions to subproblems
    - **Base Cases**: Empty string conversions
    
    **Recurrence Relation:**
    ```
    If word1[i-1] == word2[j-1]:
        dp[i][j] = dp[i-1][j-1]
    Else:
        dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    ```

---

## Problem 5: Word Break

**Difficulty**: 🟡 Medium  
**Pattern**: String DP + Dictionary  
**Time**: O(n³), **Space**: O(n)

=== "Problem"

    Given a string `s` and a dictionary of strings `wordDict`, return `true` if `s` can be segmented into a space-separated sequence of one or more dictionary words.

    **Example 1:**
    ```
    Input: s = "leetcode", wordDict = ["leet","code"]
    Output: true
    Explanation: Return true because "leetcode" can be segmented as "leet code".
    ```

    **Example 2:**
    ```
    Input: s = "applepenapple", wordDict = ["apple","pen"]
    Output: true
    Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
    ```

=== "Solution"

    ```python
    def wordBreak(s, wordDict):
        """
        DP approach - check if string can be broken at each position.
        
        Time: O(n³) - nested loops and substring operations
        Space: O(n) - DP array
        """
        word_set = set(wordDict)  # O(1) lookup
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True  # Empty string can always be segmented
        
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
        
        return dp[n]
    
    def wordBreak_optimized(s, wordDict):
        """
        Optimized version - check from current position backwards.
        
        Time: O(n × m × k) where m = avg word length, k = dict size
        Space: O(n)
        """
        word_set = set(wordDict)
        max_len = max(len(word) for word in wordDict) if wordDict else 0
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        
        for i in range(1, n + 1):
            for j in range(max(0, i - max_len), i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
        
        return dp[n]
    
    def wordBreak_with_memoization(s, wordDict):
        """
        Top-down memoization approach.
        """
        word_set = set(wordDict)
        memo = {}
        
        def can_break(start):
            if start == len(s):
                return True
            
            if start in memo:
                return memo[start]
            
            for end in range(start + 1, len(s) + 1):
                if s[start:end] in word_set and can_break(end):
                    memo[start] = True
                    return True
            
            memo[start] = False
            return False
        
        return can_break(0)
    
    def wordBreak_with_path(s, wordDict):
        """
        Return actual word segmentation if possible.
        """
        word_set = set(wordDict)
        n = len(s)
        dp = [False] * (n + 1)
        parent = [-1] * (n + 1)  # To reconstruct path
        
        dp[0] = True
        
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    parent[i] = j
                    break
        
        if not dp[n]:
            return False, []
        
        # Reconstruct path
        result = []
        i = n
        while i > 0:
            j = parent[i]
            result.append(s[j:i])
            i = j
        
        result.reverse()
        return True, result
    ```

=== "Insights"

    **Key Concepts:**
    
    - **State Definition**: `dp[i]` = true if s[0:i] can be segmented
    - **Recurrence**: `dp[i] = dp[j] && s[j:i] in wordDict` for some j < i
    - **Optimization**: Limit search range using maximum word length
    - **Base Case**: Empty string can always be segmented
    
    **Algorithm Strategy:**
    For each position, check all possible word endings that could complete a valid segmentation from a previous valid position.

---

## Problem 6: Decode Ways

**Difficulty**: 🟡 Medium  
**Pattern**: Linear DP with Constraints  
**Time**: O(n), **Space**: O(1)

=== "Problem"

    A message containing letters from A-Z can be encoded into numbers using the mapping:
    'A' -> "1", 'B' -> "2", ..., 'Z' -> "26"

    To decode an encoded message, all the digits must be grouped then mapped back into letters using the reverse of the mapping above. Given a string `s` containing only digits, return the number of ways to decode it.

    **Example 1:**
    ```
    Input: s = "12"
    Output: 2
    Explanation: "12" could be decoded as "AB" (1 2) or "L" (12).
    ```

    **Example 2:**
    ```
    Input: s = "226"
    Output: 3
    Explanation: "226" could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).
    ```

=== "Solution"

    ```python
    def numDecodings(s):
        """
        DP approach with space optimization.
        
        Time: O(n), Space: O(1)
        """
        if not s or s[0] == '0':
            return 0
        
        n = len(s)
        if n == 1:
            return 1
        
        # dp[i] represents number of ways to decode s[:i]
        prev2 = 1  # dp[0] = 1 (empty string)
        prev1 = 1  # dp[1] = 1 (first character if valid)
        
        for i in range(2, n + 1):
            current = 0
            
            # Check single digit
            if s[i-1] != '0':
                current += prev1
            
            # Check two digits
            two_digit = int(s[i-2:i])
            if 10 <= two_digit <= 26:
                current += prev2
            
            prev2 = prev1
            prev1 = current
        
        return prev1
    
    def numDecodings_full_dp(s):
        """
        Full DP array approach for clarity.
        
        Time: O(n), Space: O(n)
        """
        if not s or s[0] == '0':
            return 0
        
        n = len(s)
        dp = [0] * (n + 1)
        dp[0] = 1  # Empty string
        dp[1] = 1  # First character
        
        for i in range(2, n + 1):
            # Single digit
            if s[i-1] != '0':
                dp[i] += dp[i-1]
            
            # Two digits
            two_digit = int(s[i-2:i])
            if 10 <= two_digit <= 26:
                dp[i] += dp[i-2]
        
        return dp[n]
    
    def numDecodings_memoization(s):
        """
        Top-down memoization approach.
        """
        memo = {}
        
        def decode(index):
            if index == len(s):
                return 1
            
            if s[index] == '0':
                return 0
            
            if index in memo:
                return memo[index]
            
            # Decode as single digit
            result = decode(index + 1)
            
            # Decode as two digits
            if index + 1 < len(s):
                two_digit = int(s[index:index+2])
                if two_digit <= 26:
                    result += decode(index + 2)
            
            memo[index] = result
            return result
        
        return decode(0)
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Constraint Handling**: Only digits 1-26 are valid
    - **Edge Cases**: Leading zeros, invalid combinations
    - **Fibonacci-like**: Each position can combine previous 1 or 2 positions
    - **State Definition**: `dp[i]` = ways to decode string up to position i
    
    **Critical Cases:**
    - '0': Cannot be decoded alone, must be part of "10" or "20"
    - "30"-"99": Invalid two-digit combinations
    - Leading zeros: Make entire string invalid

---

## Problem 7: House Robber II

**Difficulty**: 🟡 Medium  
**Pattern**: Linear DP with Circular Constraint  
**Time**: O(n), **Space**: O(1)

=== "Problem"

    You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are **arranged in a circle**. That means the first house is the neighbor of the last one.

    Given an integer array `nums` representing the amount of money of each house, return the maximum amount of money you can rob tonight **without alerting the police**.

    **Example 1:**
    ```
    Input: nums = [2,3,2]
    Output: 3
    Explanation: You cannot rob house 1 (money = 2) and house 3 (money = 2) simultaneously.
    ```

=== "Solution"

    ```python
    def rob(nums):
        """
        Handle circular constraint by considering two scenarios.
        
        Time: O(n), Space: O(1)
        """
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums)
        
        # Two scenarios: rob first house or rob last house
        # Scenario 1: Rob houses 0 to n-2 (excluding last)
        max1 = rob_linear(nums[:-1])
        
        # Scenario 2: Rob houses 1 to n-1 (excluding first)
        max2 = rob_linear(nums[1:])
        
        return max(max1, max2)
    
    def rob_linear(nums):
        """
        Classic house robber for linear arrangement.
        """
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        
        prev2 = 0  # Max money up to 2 houses ago
        prev1 = nums[0]  # Max money up to 1 house ago
        
        for i in range(1, len(nums)):
            current = max(prev1, prev2 + nums[i])
            prev2 = prev1
            prev1 = current
        
        return prev1
    
    def rob_detailed(nums):
        """
        Detailed version showing which houses are robbed.
        """
        if not nums:
            return 0, []
        if len(nums) == 1:
            return nums[0], [0]
        
        def rob_linear_with_path(arr, start_idx):
            if not arr:
                return 0, []
            if len(arr) == 1:
                return arr[0], [start_idx]
            
            # DP with path tracking
            dp = [0] * len(arr)
            path = [[] for _ in range(len(arr))]
            
            dp[0] = arr[0]
            path[0] = [start_idx]
            
            dp[1] = max(arr[0], arr[1])
            path[1] = [start_idx] if arr[0] >= arr[1] else [start_idx + 1]
            
            for i in range(2, len(arr)):
                if dp[i-1] > dp[i-2] + arr[i]:
                    dp[i] = dp[i-1]
                    path[i] = path[i-1]
                else:
                    dp[i] = dp[i-2] + arr[i]
                    path[i] = path[i-2] + [start_idx + i]
            
            return dp[-1], path[-1]
        
        # Two scenarios
        money1, houses1 = rob_linear_with_path(nums[:-1], 0)
        money2, houses2 = rob_linear_with_path(nums[1:], 1)
        
        if money1 >= money2:
            return money1, houses1
        else:
            return money2, houses2
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Circular Constraint**: First and last houses are adjacent
    - **Problem Reduction**: Split into two linear subproblems
    - **Scenario Analysis**: Either rob first house or don't rob first house
    - **Edge Cases**: Handle small arrays separately
    
    **Algorithm Strategy:**
    Since robbing both first and last house is illegal, consider two cases:
    1. Rob houses 0 to n-2 (can rob first, not last)
    2. Rob houses 1 to n-1 (can rob last, not first)

---

## Problem 8: Maximum Product Subarray

**Difficulty**: 🟡 Medium  
**Pattern**: State Tracking DP  
**Time**: O(n), **Space**: O(1)

=== "Problem"

    Given an integer array `nums`, find a contiguous non-empty subarray within the array that has the largest product.

    **Example 1:**
    ```
    Input: nums = [2,3,-2,4]
    Output: 6
    Explanation: [2,3] has the largest product 6.
    ```

    **Example 2:**
    ```
    Input: nums = [-2,0,-1]
    Output: 0
    Explanation: The result cannot be 2, because [-2,-1] is not a subarray.
    ```

=== "Solution"

    ```python
    def maxProduct(nums):
        """
        Track both maximum and minimum products ending at each position.
        
        Time: O(n), Space: O(1)
        """
        if not nums:
            return 0
        
        max_so_far = min_so_far = result = nums[0]
        
        for i in range(1, len(nums)):
            num = nums[i]
            
            # Calculate all possibilities
            temp_max = max(num, max_so_far * num, min_so_far * num)
            min_so_far = min(num, max_so_far * num, min_so_far * num)
            max_so_far = temp_max
            
            result = max(result, max_so_far)
        
        return result
    
    def maxProduct_detailed(nums):
        """
        More explicit version showing the logic.
        """
        if not nums:
            return 0
        
        max_ending_here = nums[0]
        min_ending_here = nums[0]
        max_so_far = nums[0]
        
        for i in range(1, len(nums)):
            current = nums[i]
            
            # When current is negative, max and min swap roles
            if current < 0:
                max_ending_here, min_ending_here = min_ending_here, max_ending_here
            
            # Update max and min ending at current position
            max_ending_here = max(current, max_ending_here * current)
            min_ending_here = min(current, min_ending_here * current)
            
            # Update global maximum
            max_so_far = max(max_so_far, max_ending_here)
        
        return max_so_far
    
    def maxProduct_with_indices(nums):
        """
        Return both maximum product and the subarray indices.
        """
        if not nums:
            return 0, (0, 0)
        
        max_ending_here = min_ending_here = nums[0]
        max_so_far = nums[0]
        start = end = 0
        temp_start = 0
        
        for i in range(1, len(nums)):
            current = nums[i]
            
            # Track when we might start a new subarray
            if max_ending_here == current:
                temp_start = i
            
            # Calculate new max and min
            candidates = [current, max_ending_here * current, min_ending_here * current]
            new_max = max(candidates)
            new_min = min(candidates)
            
            # Update tracking
            if new_max > max_so_far:
                max_so_far = new_max
                start = temp_start
                end = i
            
            max_ending_here = new_max
            min_ending_here = new_min
        
        return max_so_far, (start, end)
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Dual State Tracking**: Keep both maximum and minimum products
    - **Sign Consideration**: Negative numbers can flip max/min
    - **Reset Strategy**: Sometimes better to start fresh from current element
    - **Kadane's Variant**: Similar to maximum sum subarray but with multiplication
    
    **Why Track Minimum?**
    A large negative number (minimum) can become maximum when multiplied by another negative number.

---

## Problem 9: Palindromic Substrings

**Difficulty**: 🟡 Medium  
**Pattern**: 2D DP or Expand Around Centers  
**Time**: O(n²), **Space**: O(n²) or O(1)

=== "Problem"

    Given a string `s`, return the number of palindromic substrings in it.

    A string is a palindrome when it reads the same backward as forward.

    **Example 1:**
    ```
    Input: s = "abc"
    Output: 3
    Explanation: Three palindromic strings: "a", "b", "c".
    ```

    **Example 2:**
    ```
    Input: s = "aaa"
    Output: 6
    Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".
    ```

=== "Solution"

    ```python
    def countSubstrings(s):
        """
        Expand around centers approach.
        
        Time: O(n²), Space: O(1)
        """
        count = 0
        
        def expand_around_center(left, right):
            nonlocal count
            while left >= 0 and right < len(s) and s[left] == s[right]:
                count += 1
                left -= 1
                right += 1
        
        for i in range(len(s)):
            # Odd length palindromes (center at i)
            expand_around_center(i, i)
            
            # Even length palindromes (center between i and i+1)
            expand_around_center(i, i + 1)
        
        return count
    
    def countSubstrings_dp(s):
        """
        2D DP approach.
        
        Time: O(n²), Space: O(n²)
        """
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        count = 0
        
        # Single characters are palindromes
        for i in range(n):
            dp[i][i] = True
            count += 1
        
        # Check for length 2
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                dp[i][i + 1] = True
                count += 1
        
        # Check for lengths 3 and above
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j] and dp[i + 1][j - 1]:
                    dp[i][j] = True
                    count += 1
        
        return count
    
    def countSubstrings_with_list(s):
        """
        Return count and list of all palindromic substrings.
        """
        palindromes = []
        
        def expand_around_center(left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                palindromes.append(s[left:right + 1])
                left -= 1
                right += 1
        
        for i in range(len(s)):
            expand_around_center(i, i)      # Odd length
            expand_around_center(i, i + 1)  # Even length
        
        return len(palindromes), palindromes
    
    def countSubstrings_manacher(s):
        """
        Manacher's algorithm for linear time.
        
        Time: O(n), Space: O(n)
        """
        # Transform string to handle even length palindromes
        transformed = '#'.join('^{}$'.format(s))
        n = len(transformed)
        centers = [0] * n
        center = right = 0
        count = 0
        
        for i in range(1, n - 1):
            # Mirror of i with respect to center
            mirror = 2 * center - i
            
            if i < right:
                centers[i] = min(right - i, centers[mirror])
            
            # Try to expand palindrome centered at i
            try:
                while transformed[i + centers[i] + 1] == transformed[i - centers[i] - 1]:
                    centers[i] += 1
            except IndexError:
                pass
            
            # If palindrome centered at i extends past right, adjust center and right
            if i + centers[i] > right:
                center, right = i, i + centers[i]
            
            # Count palindromes
            count += (centers[i] + 1) // 2
        
        return count
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Center Expansion**: Check both odd and even length palindromes
    - **2D DP**: `dp[i][j]` = true if substring s[i:j+1] is palindrome
    - **Manacher's Algorithm**: Linear time algorithm using previously computed results
    - **Space Trade-off**: O(1) space with expansion vs O(n²) with DP
    
    **Algorithm Comparison:**
    1. **Expand Around Centers**: Simple, O(1) space
    2. **2D DP**: Clear logic, easier to modify for variations
    3. **Manacher's**: Optimal O(n) time complexity

---

## Problem 10: Target Sum

**Difficulty**: 🟡 Medium  
**Pattern**: Subset Sum DP  
**Time**: O(n × sum), **Space**: O(sum)

=== "Problem"

    You are given an integer array `nums` and an integer `target`.

    You want to build an expression out of nums by adding one of the symbols '+' and '-' before each integer in nums and then concatenate all the integers.

    Return the number of different expressions that you can build, which evaluates to target.

    **Example 1:**
    ```
    Input: nums = [1,1,1,1,1], target = 3
    Output: 5
    Explanation: There are 5 ways to assign symbols:
    -1 + 1 + 1 + 1 + 1 = 3
    +1 - 1 + 1 + 1 + 1 = 3
    +1 + 1 - 1 + 1 + 1 = 3
    +1 + 1 + 1 - 1 + 1 = 3
    +1 + 1 + 1 + 1 - 1 = 3
    ```

=== "Solution"

    ```python
    def findTargetSumWays(nums, target):
        """
        Transform to subset sum problem.
        
        Time: O(n × sum), Space: O(sum)
        """
        total = sum(nums)
        
        # Check if target is achievable
        if target > total or target < -total or (target + total) % 2 == 1:
            return 0
        
        # Transform problem: find subset with sum = (target + total) / 2
        subset_sum = (target + total) // 2
        
        # DP for counting subsets with given sum
        dp = [0] * (subset_sum + 1)
        dp[0] = 1  # One way to get sum 0: choose nothing
        
        for num in nums:
            for j in range(subset_sum, num - 1, -1):
                dp[j] += dp[j - num]
        
        return dp[subset_sum]
    
    def findTargetSumWays_backtrack(nums, target):
        """
        Backtracking approach with memoization.
        
        Time: O(n × sum), Space: O(n × sum)
        """
        memo = {}
        
        def backtrack(index, current_sum):
            if index == len(nums):
                return 1 if current_sum == target else 0
            
            if (index, current_sum) in memo:
                return memo[(index, current_sum)]
            
            # Try both + and - for current number
            add = backtrack(index + 1, current_sum + nums[index])
            subtract = backtrack(index + 1, current_sum - nums[index])
            
            memo[(index, current_sum)] = add + subtract
            return memo[(index, current_sum)]
        
        return backtrack(0, 0)
    
    def findTargetSumWays_2d_dp(nums, target):
        """
        2D DP approach for clarity.
        
        Time: O(n × sum), Space: O(n × sum)
        """
        total = sum(nums)
        if target > total or target < -total:
            return 0
        
        # Offset to handle negative indices
        offset = total
        n = len(nums)
        
        # dp[i][j] = ways to get sum (j - offset) using first i numbers
        dp = [[0] * (2 * total + 1) for _ in range(n + 1)]
        dp[0][offset] = 1  # Base case: 0 numbers give sum 0
        
        for i in range(1, n + 1):
            for j in range(2 * total + 1):
                # Don't include current number (impossible, but for completeness)
                # Include current number with +
                if j + nums[i-1] <= 2 * total:
                    dp[i][j] += dp[i-1][j + nums[i-1]]
                
                # Include current number with -
                if j - nums[i-1] >= 0:
                    dp[i][j] += dp[i-1][j - nums[i-1]]
        
        return dp[n][target + offset]
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Problem Transformation**: Convert to subset sum problem
    - **Mathematical Insight**: P - N = target, P + N = total → P = (target + total) / 2
    - **Subset Sum DP**: Count ways to achieve specific sum
    - **Memoization**: Cache results for (index, current_sum) pairs
    
    **Transformation Logic:**
    Let P = sum of positive numbers, N = sum of negative numbers
    - P - N = target
    - P + N = total
    - Solving: P = (target + total) / 2

---

## 📝 Summary

### Core Medium DP Patterns

| **Pattern** | **Key Technique** | **Example Problems** |
|-------------|-------------------|---------------------|
| **2D Grid DP** | Build table row by row | Unique Paths, Edit Distance |
| **Knapsack Variants** | Choose/don't choose decisions | Coin Change, Target Sum |
| **Sequence DP** | Extend or start new | LIS, Maximum Product |
| **String DP** | Character matching/transformation | Word Break, Decode Ways |
| **Constraint DP** | Handle special restrictions | House Robber II |

### Essential Templates

```python
# 2D Grid DP
dp = [[0] * n for _ in range(m)]
for i in range(m):
    for j in range(n):
        dp[i][j] = function(dp[i-1][j], dp[i][j-1])

# Knapsack DP
dp = [0] * (target + 1)
for item in items:
    for j in range(target, item - 1, -1):
        dp[j] = max(dp[j], dp[j - item] + value)

# String DP
dp = [[0] * (n + 1) for _ in range(m + 1)]
for i in range(1, m + 1):
    for j in range(1, n + 1):
        if s1[i-1] == s2[j-1]:
            dp[i][j] = dp[i-1][j-1] + 1
        else:
            dp[i][j] = max(dp[i-1][j], dp[i][j-1])
```

### Space Optimization Techniques

1. **Rolling Array**: Use only previous row for 2D DP
2. **Two Variables**: Track only necessary previous states
3. **Reverse Iteration**: Avoid overwriting needed values
4. **Mathematical Formula**: Direct calculation when possible

### Problem-Solving Strategy

1. **Identify State**: What parameters define subproblems?
2. **Find Recurrence**: How do states relate?
3. **Handle Base Cases**: What are the simplest cases?
4. **Choose Approach**: Top-down vs bottom-up
5. **Optimize Space**: Can we reduce memory usage?

---

## 🎯 Next Steps

- **[Hard DP Problems](hard-problems.md)** - Advanced patterns and optimizations
- **[2D DP Deep Dive](2d-dp.md)** - Matrix and grid problems
- **[String DP Algorithms](string-dp.md)** - Advanced string manipulation

These medium problems bridge the gap between basic DP concepts and complex optimization challenges. Master these patterns to tackle any DP problem with confidence!

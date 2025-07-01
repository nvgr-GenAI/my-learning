# Dynamic Programming

Dynamic Programming (DP) is a method for solving complex problems by breaking them down into simpler subproblems. It is particularly useful for optimization problems where the same subproblems occur multiple times.

## Key Concepts

### What is Dynamic Programming?
- **Overlapping Subproblems**: The problem can be broken down into subproblems which are reused several times
- **Optimal Substructure**: An optimal solution can be constructed from optimal solutions of its subproblems
- **Memoization**: Store the results of expensive function calls and return the cached result when the same inputs occur again

### Types of DP Approaches
1. **Top-Down (Memoization)**: Start with the original problem and recursively break it down
2. **Bottom-Up (Tabulation)**: Start with the smallest subproblems and build up to the original problem

## Classic DP Problems

### 1. Fibonacci Sequence

=== "Recursive (Naive)"
    ```python
    def fibonacci_naive(n):
        """Time: O(2^n), Space: O(n)"""
        if n <= 1:
            return n
        return fibonacci_naive(n-1) + fibonacci_naive(n-2)
    ```

=== "Memoization (Top-Down)"
    ```python
    def fibonacci_memo(n, memo={}):
        """Time: O(n), Space: O(n)"""
        if n in memo:
            return memo[n]
        if n <= 1:
            return n
        memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
        return memo[n]
    ```

=== "Tabulation (Bottom-Up)"
    ```python
    def fibonacci_dp(n):
        """Time: O(n), Space: O(n)"""
        if n <= 1:
            return n
        
        dp = [0] * (n + 1)
        dp[1] = 1
        
        for i in range(2, n + 1):
            dp[i] = dp[i-1] + dp[i-2]
        
        return dp[n]
    ```

=== "Space Optimized"
    ```python
    def fibonacci_optimized(n):
        """Time: O(n), Space: O(1)"""
        if n <= 1:
            return n
        
        prev2, prev1 = 0, 1
        for i in range(2, n + 1):
            current = prev1 + prev2
            prev2, prev1 = prev1, current
        
        return prev1
    ```

### 2. Longest Common Subsequence (LCS)

```python
def lcs_length(text1, text2):
    """
    Find the length of longest common subsequence
    Time: O(m*n), Space: O(m*n)
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

def lcs_string(text1, text2):
    """Return the actual LCS string"""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill the dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Reconstruct the LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if text1[i-1] == text2[j-1]:
            lcs.append(text1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    return ''.join(reversed(lcs))
```

### 3. 0/1 Knapsack Problem

```python
def knapsack_01(weights, values, capacity):
    """
    0/1 Knapsack Problem
    Time: O(n*W), Space: O(n*W)
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Don't include current item
            dp[i][w] = dp[i-1][w]
            
            # Include current item if it fits
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], 
                              dp[i-1][w - weights[i-1]] + values[i-1])
    
    return dp[n][capacity]

def knapsack_optimized(weights, values, capacity):
    """Space optimized version - O(W) space"""
    dp = [0] * (capacity + 1)
    
    for i in range(len(weights)):
        # Traverse backwards to avoid using updated values
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]
```

### 4. Coin Change Problem

=== "Minimum Coins"
    ```python
    def coin_change_min(coins, amount):
        """
        Find minimum number of coins to make amount
        Time: O(amount * len(coins)), Space: O(amount)
        """
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] = min(dp[i], dp[i - coin] + 1)
        
        return dp[amount] if dp[amount] != float('inf') else -1
    ```

=== "Number of Ways"
    ```python
    def coin_change_ways(coins, amount):
        """
        Find number of ways to make amount
        Time: O(amount * len(coins)), Space: O(amount)
        """
        dp = [0] * (amount + 1)
        dp[0] = 1
        
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] += dp[i - coin]
        
        return dp[amount]
    ```

### 5. Longest Increasing Subsequence (LIS)

```python
def lis_length(nums):
    """
    Find length of longest increasing subsequence
    Time: O(n^2), Space: O(n)
    """
    if not nums:
        return 0
    
    n = len(nums)
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

def lis_optimized(nums):
    """
    Optimized version using binary search
    Time: O(n log n), Space: O(n)
    """
    if not nums:
        return 0
    
    from bisect import bisect_left
    
    tails = []
    for num in nums:
        pos = bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    
    return len(tails)
```

### 6. Edit Distance (Levenshtein Distance)

```python
def edit_distance(word1, word2):
    """
    Find minimum edit distance between two strings
    Time: O(m*n), Space: O(m*n)
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )
    
    return dp[m][n]
```

### 7. Maximum Subarray Sum (Kadane's Algorithm)

```python
def max_subarray_sum(nums):
    """
    Find maximum sum of contiguous subarray
    Time: O(n), Space: O(1)
    """
    max_sum = current_sum = nums[0]
    
    for i in range(1, len(nums)):
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)
    
    return max_sum

def max_subarray_with_indices(nums):
    """Return max sum and the subarray indices"""
    max_sum = current_sum = nums[0]
    start = end = temp_start = 0
    
    for i in range(1, len(nums)):
        if current_sum < 0:
            current_sum = nums[i]
            temp_start = i
        else:
            current_sum += nums[i]
        
        if current_sum > max_sum:
            max_sum = current_sum
            start = temp_start
            end = i
    
    return max_sum, start, end
```

## Advanced DP Patterns

### 1. State Machine DP

```python
def max_profit_with_cooldown(prices):
    """
    Stock trading with cooldown period
    States: hold, sold, rest
    """
    if not prices:
        return 0
    
    # hold[i]: max profit on day i if holding stock
    # sold[i]: max profit on day i if sold stock
    # rest[i]: max profit on day i if resting
    
    hold = -prices[0]
    sold = 0
    rest = 0
    
    for i in range(1, len(prices)):
        prev_hold = hold
        prev_sold = sold
        prev_rest = rest
        
        hold = max(prev_hold, prev_rest - prices[i])
        sold = prev_hold + prices[i]
        rest = max(prev_rest, prev_sold)
    
    return max(sold, rest)
```

### 2. Interval DP

```python
def min_cost_to_merge_stones(stones, k):
    """
    Minimum cost to merge stones into one pile
    """
    n = len(stones)
    if (n - 1) % (k - 1) != 0:
        return -1
    
    # Prefix sum for range sum queries
    prefix_sum = [0]
    for stone in stones:
        prefix_sum.append(prefix_sum[-1] + stone)
    
    # dp[i][j][p] = min cost to merge stones[i:j+1] into p piles
    dp = [[[float('inf')] * (k + 1) for _ in range(n)] for _ in range(n)]
    
    # Base case: single stone forms one pile with 0 cost
    for i in range(n):
        dp[i][i][1] = 0
    
    # Fill the DP table
    for length in range(2, n + 1):  # length of subarray
        for i in range(n - length + 1):
            j = i + length - 1
            
            # Merge into p piles (2 <= p <= k)
            for p in range(2, k + 1):
                for mid in range(i, j, k - 1):
                    dp[i][j][p] = min(dp[i][j][p],
                                    dp[i][mid][1] + dp[mid + 1][j][p - 1])
            
            # Merge k piles into 1 pile
            dp[i][j][1] = dp[i][j][k] + prefix_sum[j + 1] - prefix_sum[i]
    
    return dp[0][n - 1][1]
```

### 3. Tree DP

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def rob_tree(root):
    """
    House robber in binary tree
    Returns (rob_root, not_rob_root)
    """
    if not root:
        return 0, 0
    
    left_rob, left_not_rob = rob_tree(root.left)
    right_rob, right_not_rob = rob_tree(root.right)
    
    # If we rob current node, we can't rob children
    rob_current = root.val + left_not_rob + right_not_rob
    
    # If we don't rob current, we can choose optimally from children
    not_rob_current = max(left_rob, left_not_rob) + max(right_rob, right_not_rob)
    
    return rob_current, not_rob_current

def max_path_sum(root):
    """Maximum path sum in binary tree"""
    def helper(node):
        if not node:
            return 0, float('-inf')
        
        left_gain, left_max = helper(node.left)
        right_gain, right_max = helper(node.right)
        
        # Maximum gain if we extend the path through current node
        current_gain = node.val + max(0, left_gain, right_gain)
        
        # Maximum path sum considering current node as the highest point
        current_max = node.val + max(0, left_gain) + max(0, right_gain)
        
        # Overall maximum so far
        overall_max = max(left_max, right_max, current_max)
        
        return current_gain, overall_max
    
    return helper(root)[1]
```

## DP Optimization Techniques

### 1. Space Optimization

```python
def unique_paths_optimized(m, n):
    """
    Space optimized version of unique paths
    Time: O(m*n), Space: O(n)
    """
    dp = [1] * n
    
    for i in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j-1]
    
    return dp[n-1]
```

### 2. Rolling Array

```python
def min_path_sum_rolling(grid):
    """
    Using rolling array to save space
    """
    m, n = len(grid), len(grid[0])
    dp = [float('inf')] * n
    dp[0] = 0
    
    for i in range(m):
        dp[0] += grid[i][0]
        for j in range(1, n):
            dp[j] = min(dp[j], dp[j-1]) + grid[i][j]
    
    return dp[n-1]
```

## Common DP Problems by Category

### Linear DP
- Fibonacci Numbers
- Climbing Stairs
- House Robber
- Maximum Subarray
- Longest Increasing Subsequence

### Grid DP
- Unique Paths
- Minimum Path Sum
- Maximal Square
- Dungeon Game

### String DP
- Longest Common Subsequence
- Edit Distance
- Palindromic Subsequence
- Word Break

### Tree DP
- Binary Tree Maximum Path Sum
- House Robber III
- Diameter of Binary Tree

### Interval DP
- Matrix Chain Multiplication
- Burst Balloons
- Merge Stones

## Study Tips

1. **Identify the Pattern**: Look for overlapping subproblems and optimal substructure
2. **Define State**: Clearly define what each DP state represents
3. **State Transition**: Find the recurrence relation
4. **Base Cases**: Handle edge cases properly
5. **Space Optimization**: Consider if you can reduce space complexity
6. **Practice Categories**: Focus on one category at a time

## Resources

- [Dynamic Programming Patterns](https://leetcode.com/discuss/general-discussion/458695/dynamic-programming-patterns)
- [DP for Beginners](https://leetcode.com/discuss/general-discussion/662866/dp-for-beginners-problems-patterns-sample-solutions)
- [Geeks for Geeks DP](https://www.geeksforgeeks.org/dynamic-programming/)

---

*This page covers the essential concepts and patterns in Dynamic Programming. Practice these problems and understand the underlying patterns to master DP!*

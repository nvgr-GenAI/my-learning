# Dynamic Programming - Easy Problems

## 🎯 Learning Objectives

Master basic DP patterns and build intuition for more complex problems. These 15 problems cover the most important DP patterns asked in technical interviews.

=== "📋 Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | Climbing Stairs | Linear DP (Fibonacci) | Easy | O(n) | O(1) |
    | 2 | House Robber | Linear DP | Easy | O(n) | O(1) |
    | 3 | Min Cost Climbing Stairs | Linear DP | Easy | O(n) | O(1) |
    | 4 | N-th Tribonacci Number | Linear DP | Easy | O(n) | O(1) |
    | 5 | Fibonacci Number | Linear DP | Easy | O(n) | O(1) |
    | 6 | Pascal's Triangle | 2D DP | Easy | O(n²) | O(n²) |
    | 7 | Pascal's Triangle II | 1D Space Optimized | Easy | O(n²) | O(n) |
    | 8 | Is Subsequence | Two Pointers/DP | Easy | O(n) | O(1) |
    | 9 | Range Sum Query - Immutable | Prefix Sum DP | Easy | O(1) | O(n) |
    | 10 | Counting Bits | Bit DP | Easy | O(n) | O(n) |
    | 11 | Best Time to Buy/Sell Stock | Linear DP | Easy | O(n) | O(1) |
    | 12 | Maximum Subarray (Kadane's) | Linear DP | Easy | O(n) | O(1) |
    | 13 | Unique Paths | 2D DP | Easy | O(mn) | O(mn) |
    | 14 | Minimum Path Sum | 2D DP | Easy | O(mn) | O(mn) |
    | 15 | Coin Change II (Count Ways) | 1D DP | Easy | O(amount×coins) | O(amount) |

=== "🎯 Core DP Patterns"

    **🔢 Linear DP:**
    - State depends only on previous few states
    - Examples: Fibonacci, climbing stairs, house robber
    
    **📊 Grid DP:**
    - 2D state space (often matrix traversal)
    - Examples: Pascal's triangle, unique paths
    
    **🎒 Knapsack-style:**
    - Choose/don't choose decisions
    - Examples: Coin change, house robber
    
    **📈 Optimization DP:**
    - Find minimum/maximum value
    - Examples: Min cost climbing, maximum subarray

=== "⚡ Interview Strategy"

    **💡 Problem Recognition:**
    
    - **Optimal substructure**: Solution can be built from subproblems
    - **Overlapping subproblems**: Same subproblems appear multiple times
    - **Decision making**: Choose between multiple options at each step
    
    **🔄 Solution Steps:**
    
    1. **Define state**: What does dp[i] represent?
    2. **Find recurrence**: How does dp[i] relate to previous states?
    3. **Identify base cases**: Smallest valid inputs
    4. **Implement and optimize**: Start with tabulation, then optimize space

---

## Problem 1: Climbing Stairs

**Difficulty**: 🟢 Easy  
**Pattern**: Linear DP (Fibonacci variant)  
**Time**: O(n), **Space**: O(1) optimized

=== "Problem Statement"

    You're climbing a staircase. It takes n steps to reach the top. Each time you can climb 1 or 2 steps. In how many distinct ways can you climb to the top?

    **Example:**
    ```text
    Input: n = 3
    Output: 3
    Explanation: Three ways to climb to top:
    1. 1 step + 1 step + 1 step
    2. 1 step + 2 steps  
    3. 2 steps + 1 step
    ```

=== "Optimal Solution"

    ```python
    def climb_stairs_optimized(n):
        """
        Space-optimized DP - O(1) space, O(n) time.
        Ways to reach step n = ways to reach (n-1) + ways to reach (n-2).
        """
        if n <= 2:
            return n
        
        prev2 = 1  # ways to reach step 1
        prev1 = 2  # ways to reach step 2
        
        for i in range(3, n + 1):
            current = prev1 + prev2
            prev2 = prev1
            prev1 = current
        
        return prev1

    # Test
    test_cases = [1, 2, 3, 4, 5, 8]
    for n in test_cases:
        result = climb_stairs_optimized(n)
        print(f"n={n}: {result} ways")
    ```

=== "Tabulation Approach"

    ```python
    def climb_stairs_tabulation(n):
        """
        Bottom-up DP with array - O(n) space, O(n) time.
        """
        if n <= 2:
            return n
        
        dp = [0] * (n + 1)
        dp[1] = 1
        dp[2] = 2
        
        for i in range(3, n + 1):
            dp[i] = dp[i-1] + dp[i-2]
        
        return dp[n]
    ```

---

## Problem 2: House Robber

**Difficulty**: 🟢 Easy  
**Pattern**: Linear DP with Skip Decision  
**Time**: O(n), **Space**: O(1) optimized

=== "Problem Statement"

    You are a robber planning to rob houses along a street. Each house has a certain amount of money. You cannot rob two adjacent houses. Find the maximum amount you can rob.

    **Example:**
    ```text
    Input: nums = [2,7,9,3,1]
    Output: 12
    Explanation: Rob houses 0, 2, and 4 (2 + 9 + 1 = 12)
    ```

=== "Optimal Solution"

    ```python
    def rob_optimized(nums):
        """
        Space-optimized DP - O(1) space, O(n) time.
        Track max money with and without robbing current house.
        """
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        
        prev2 = nums[0]  # Max money up to house i-2
        prev1 = max(nums[0], nums[1])  # Max money up to house i-1
        
        for i in range(2, len(nums)):
            current = max(prev1, prev2 + nums[i])
            prev2 = prev1
            prev1 = current
        
        return prev1

    # Test
    test_cases = [
        [2,7,9,3,1],  # Expected: 12
        [2,1,1,2],    # Expected: 4
        [5],          # Expected: 5
        [1,2]         # Expected: 2
    ]
    
    for nums in test_cases:
        result = rob_optimized(nums)
        print(f"Houses {nums}: Max money = {result}")
    ```

=== "Tabulation Approach"

    ```python
    def rob_tabulation(nums):
        """
        Bottom-up DP with array - O(n) space, O(n) time.
        """
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        
        dp = [0] * len(nums)
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        
        for i in range(2, len(nums)):
            dp[i] = max(dp[i-1], dp[i-2] + nums[i])
        
        return dp[-1]
    ```

---

## Problem 3: Min Cost Climbing Stairs

**Difficulty**: 🟢 Easy  
**Pattern**: Linear DP with Choice  
**Time**: O(n), **Space**: O(1) optimized

=== "Problem Statement"

    You are given an array where each element represents the cost of climbing a stair. You can start from step 0 or 1. At each step, you can climb 1 or 2 steps. Find the minimum cost to reach the top.

    **Example:**
    ```text
    Input: cost = [10,15,20]
    Output: 15
    Explanation: Start at index 1, pay 15, climb 2 steps to reach top
    ```

=== "Optimal Solution"

    ```python
    def min_cost_climbing_optimized(cost):
        """
        Space-optimized DP - O(1) space, O(n) time.
        """
        prev2 = cost[0]  # Min cost to reach step 0
        prev1 = cost[1]  # Min cost to reach step 1
        
        for i in range(2, len(cost)):
            current = cost[i] + min(prev1, prev2)
            prev2 = prev1
            prev1 = current
        
        # Can start from either step 0 or 1, so take minimum
        return min(prev1, prev2)

    # Test
    test_cases = [
        [10, 15, 20],        # Expected: 15
        [1, 100, 1, 1, 1],   # Expected: 6
        [0, 0, 1, 1]         # Expected: 1
    ]
    
    for cost in test_cases:
        result = min_cost_climbing_optimized(cost)
        print(f"Cost {cost}: Min cost = {result}")
    ```

---

## Problem 4: N-th Tribonacci Number

**Difficulty**: 🟢 Easy  
**Pattern**: Linear DP (Tribonacci sequence)  
**Time**: O(n), **Space**: O(1) optimized

=== "Problem Statement"

    The Tribonacci sequence is defined as:
    T(0) = 0, T(1) = 1, T(2) = 1
    T(n) = T(n-1) + T(n-2) + T(n-3) for n >= 3
    
    Given n, return the value of T(n).

=== "Optimal Solution"

    ```python
    def tribonacci(n):
        """
        Space-optimized DP - O(1) space, O(n) time.
        """
        if n == 0:
            return 0
        if n <= 2:
            return 1
        
        a, b, c = 0, 1, 1  # T(0), T(1), T(2)
        
        for i in range(3, n + 1):
            next_val = a + b + c
            a, b, c = b, c, next_val
        
        return c

    # Test
    test_cases = [0, 1, 2, 3, 4, 5, 25]
    for n in test_cases:
        result = tribonacci(n)
        print(f"T({n}) = {result}")
    ```

---

## Problem 5: Fibonacci Number

**Difficulty**: 🟢 Easy  
**Pattern**: Linear DP (Classic Fibonacci)  
**Time**: O(n), **Space**: O(1) optimized

=== "Problem Statement"

    The Fibonacci numbers form a sequence where each number is the sum of the two preceding ones. F(0) = 0, F(1) = 1, F(n) = F(n-1) + F(n-2).

=== "Optimal Solution"

    ```python
    def fibonacci(n):
        """
        Space-optimized DP - O(1) space, O(n) time.
        """
        if n <= 1:
            return n
        
        prev2 = 0  # F(0)
        prev1 = 1  # F(1)
        
        for i in range(2, n + 1):
            current = prev1 + prev2
            prev2 = prev1
            prev1 = current
        
        return prev1
    ```

---

## Problem 6: Pascal's Triangle

**Difficulty**: 🟢 Easy  
**Pattern**: 2D DP  
**Time**: O(n²), **Space**: O(n²)

=== "Problem Statement"

    Generate the first numRows of Pascal's triangle. Each row has one more element than the previous row.

=== "Optimal Solution"

    ```python
    def generate_pascal_triangle(numRows):
        """
        Generate Pascal's triangle using DP.
        """
        triangle = []
        
        for i in range(numRows):
            row = [1] * (i + 1)  # Initialize row with 1s
            
            # Fill middle elements
            for j in range(1, i):
                row[j] = triangle[i-1][j-1] + triangle[i-1][j]
            
            triangle.append(row)
        
        return triangle
    ```

---

## Problem 7: Pascal's Triangle II

**Difficulty**: 🟢 Easy  
**Pattern**: 1D Space Optimized DP  
**Time**: O(n²), **Space**: O(n)

=== "Problem Statement"

    Given row index, return the indexth row of Pascal's triangle using only O(n) extra space.

=== "Optimal Solution"

    ```python
    def get_row(rowIndex):
        """
        Space-optimized: Generate only the required row.
        """
        row = [1] * (rowIndex + 1)
        
        for i in range(1, rowIndex + 1):
            # Update from right to left to avoid overwriting needed values
            for j in range(i - 1, 0, -1):
                row[j] += row[j - 1]
        
        return row
    ```

---

## Problem 8: Is Subsequence

**Difficulty**: 🟢 Easy  
**Pattern**: Two Pointers/DP  
**Time**: O(n), **Space**: O(1)

=== "Problem Statement"

    Given strings s and t, return true if s is a subsequence of t.

=== "Optimal Solution"

    ```python
    def is_subsequence(s, t):
        """
        Two pointers approach - O(n) time, O(1) space.
        """
        i = j = 0
        
        while i < len(s) and j < len(t):
            if s[i] == t[j]:
                i += 1
            j += 1
        
        return i == len(s)
    ```

---

## Problem 9: Range Sum Query - Immutable

**Difficulty**: 🟢 Easy  
**Pattern**: Prefix Sum DP  
**Time**: O(1) query, **Space**: O(n)

=== "Problem Statement"

    Design a data structure to find the sum of elements between indices i and j (inclusive).

=== "Optimal Solution"

    ```python
    class NumArray:
        def __init__(self, nums):
            """
            Precompute prefix sums for O(1) range queries.
            """
            self.prefix_sums = [0]
            for num in nums:
                self.prefix_sums.append(self.prefix_sums[-1] + num)
        
        def sumRange(self, left, right):
            """
            Return sum from left to right inclusive.
            """
            return self.prefix_sums[right + 1] - self.prefix_sums[left]
    ```

---

## Problem 10: Counting Bits

**Difficulty**: 🟢 Easy  
**Pattern**: Bit DP  
**Time**: O(n), **Space**: O(n)

=== "Problem Statement"

    Given integer n, return an array where ans[i] is the number of 1's in the binary representation of i.

=== "Optimal Solution"

    ```python
    def count_bits(n):
        """
        DP approach: ans[i] = ans[i >> 1] + (i & 1)
        """
        ans = [0] * (n + 1)
        
        for i in range(1, n + 1):
            ans[i] = ans[i >> 1] + (i & 1)
        
        return ans
    ```

---

## Problem 11: Best Time to Buy and Sell Stock

**Difficulty**: 🟢 Easy  
**Pattern**: Linear DP  
**Time**: O(n), **Space**: O(1)

=== "Problem Statement"

    Find the maximum profit from buying and selling stock once.

=== "Optimal Solution"

    ```python
    def max_profit(prices):
        """
        Track minimum price seen so far and maximum profit.
        """
        min_price = float('inf')
        max_profit = 0
        
        for price in prices:
            min_price = min(min_price, price)
            max_profit = max(max_profit, price - min_price)
        
        return max_profit
    ```

---

## Problem 12: Maximum Subarray (Kadane's Algorithm)

**Difficulty**: 🟢 Easy  
**Pattern**: Linear DP  
**Time**: O(n), **Space**: O(1)

=== "Problem Statement"

    Find the contiguous subarray with the largest sum.

=== "Optimal Solution"

    ```python
    def max_subarray(nums):
        """
        Kadane's algorithm - extend subarray or start new one.
        """
        max_ending_here = max_so_far = nums[0]
        
        for i in range(1, len(nums)):
            max_ending_here = max(nums[i], max_ending_here + nums[i])
            max_so_far = max(max_so_far, max_ending_here)
        
        return max_so_far
    ```

---

## Problem 13: Unique Paths

**Difficulty**: 🟢 Easy  
**Pattern**: 2D DP  
**Time**: O(mn), **Space**: O(mn)

=== "Problem Statement"

    Find number of possible unique paths from top-left to bottom-right in an m x n grid.

=== "Optimal Solution"

    ```python
    def unique_paths(m, n):
        """
        DP: paths[i][j] = paths[i-1][j] + paths[i][j-1]
        """
        dp = [[1] * n for _ in range(m)]
        
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        
        return dp[m-1][n-1]
    ```

---

## Problem 14: Minimum Path Sum

**Difficulty**: 🟢 Easy  
**Pattern**: 2D DP  
**Time**: O(mn), **Space**: O(mn)

=== "Problem Statement"

    Find a path from top left to bottom right with minimum sum.

=== "Optimal Solution"

    ```python
    def min_path_sum(grid):
        """
        DP: min_sum[i][j] = grid[i][j] + min(up, left)
        """
        m, n = len(grid), len(grid[0])
        
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                elif i == 0:
                    grid[i][j] += grid[i][j-1]
                elif j == 0:
                    grid[i][j] += grid[i-1][j]
                else:
                    grid[i][j] += min(grid[i-1][j], grid[i][j-1])
        
        return grid[m-1][n-1]
    ```

---

## Problem 15: Coin Change II (Count Ways)

**Difficulty**: 🟢 Easy  
**Pattern**: 1D DP (Unbounded Knapsack)  
**Time**: O(amount × coins), **Space**: O(amount)

=== "Problem Statement"

    Given coins and an amount, return the number of combinations that make up that amount.

=== "Optimal Solution"

    ```python
    def change(amount, coins):
        """
        DP: dp[i] = number of ways to make amount i
        """
        dp = [0] * (amount + 1)
        dp[0] = 1  # One way to make amount 0: use no coins
        
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] += dp[i - coin]
        
        return dp[amount]
    ```

---

## 🎯 Practice Summary

### Key Patterns Mastered

1. **Linear DP**: Fibonacci variations (stairs, tribonacci)
2. **Choice DP**: House robber, min cost climbing
3. **Grid DP**: Pascal's triangle, unique paths
4. **Optimization DP**: Maximum subarray, best time to buy/sell
5. **Counting DP**: Coin change variations
6. **Prefix Sum DP**: Range queries, cumulative sums
7. **Bit DP**: Counting set bits with DP relations

### Space Optimization Techniques

- **Rolling Variables**: Keep only last 1-3 values (O(1) space)
- **In-place Updates**: Modify input when allowed
- **Mathematical Insights**: Direct formulas for some sequences

### Interview Success Strategy

1. **Pattern Recognition**: Identify DP by optimal substructure
2. **State Definition**: Clearly define what dp[i] represents
3. **Recurrence Relation**: How current state depends on previous
4. **Base Cases**: Handle edge cases (empty, single element)
5. **Space Optimization**: Reduce from O(n) to O(1) when possible

### Common Mistakes to Avoid

1. **Index errors**: Off-by-one in loops and base cases
2. **State confusion**: Unclear dp state definition
3. **Missing edge cases**: Empty input, single elements
4. **Suboptimal space**: Using O(n) when O(1) possible

### Next Steps

Ready for more challenges? Try **[Medium DP Problems](medium-problems.md)** to explore:

- 2D DP problems (Edit Distance, Longest Common Subsequence)
- Advanced knapsack variants (Partition, Target Sum)
- String DP problems (Word Break, Palindrome Partitioning)
- Tree DP problems (House Robber III, Binary Tree Maximum Path Sum)

---

*These easy problems build the foundation for more complex DP patterns. Master the intuition here before moving to harder challenges!*

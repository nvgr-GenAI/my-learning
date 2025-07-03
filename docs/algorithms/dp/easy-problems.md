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
    | 11 | Min Steps to One | Linear DP | Easy | O(n) | O(n) |
    | 12 | Decode Ways (Simple) | Linear DP | Easy | O(n) | O(1) |
    | 13 | Perfect Squares | DP + BFS | Easy | O(n√n) | O(n) |
    | 14 | Coin Change (Min Coins) | 1D DP | Easy | O(amount×coins) | O(amount) |
    | 15 | Maximum Product of Three | Array DP | Easy | O(n) | O(1) |

=== "🎯 Core DP Patterns"

    **🔢 Linear DP:**
    - State depends only on previous few states
    - Examples: Fibonacci, climbing stairs, house robber
    
    **📊 Grid DP:**
    - 2D state space (often matrix traversal)
    - Examples: Pascal's triangle, path counting
    
    **🎒 Knapsack-style:**
    - Choose/don't choose decisions
    - Examples: Coin change, perfect squares
    
    **📈 Optimization DP:**
    - Find minimum/maximum value
    - Examples: Min cost climbing, min steps to one

=== "⚡ Interview Strategy"

    **💡 Problem Recognition:**
    
    - **Optimal substructure**: Solution can be built from subproblems
    - **Overlapping subproblems**: Same subproblems appear multiple times
    - **Decision making**: Choose between multiple options at each step
    
    **🎪 Solution Approach:**
    
    1. **Identify state**: What parameters uniquely define a subproblem?
    2. **Find recurrence**: How does current state relate to previous states?
    3. **Handle base cases**: What are the simplest cases?
    4. **Optimize space**: Can we reduce space complexity?

---

## Problem 1: Climbing Stairs

**Difficulty**: 🟢 Easy  
**Pattern**: Linear DP (Fibonacci-like)  
**Time**: O(n), **Space**: O(1) optimized

=== "Problem Statement"

    You're climbing a staircase with `n` steps. Each time you can climb either 1 or 2 steps. In how many distinct ways can you climb to the top?

    **Example:**
    ```text
    Input: n = 3
    Output: 3
    Explanation: 
    1. 1 step + 1 step + 1 step
    2. 1 step + 2 steps  
    3. 2 steps + 1 step
    ```

=== "Optimal Solution"

    ```python
    def climb_stairs_optimized(n):
        """
        Space-optimized approach - O(1) space, O(n) time.
        Only keep track of previous two values.
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
    for n in range(1, 6):
        result = climb_stairs_optimized(n)
        print(f"n={n}: {result} ways")
    ```

=== "Tabulation (Bottom-up)"

    ```python
    def climb_stairs_tabulation(n):
        """
        Bottom-up DP approach - O(n) space, O(n) time.
        Build solution from smaller subproblems.
        """
        if n <= 2:
            return n
        
        dp = [0] * (n + 1)
        dp[1] = 1
        dp[2] = 2
        
        for i in range(3, n + 1):
            dp[i] = dp[i-1] + dp[i-2]
        
        return dp[n]

    # Test with visualization
    n = 5
    result = climb_stairs_tabulation(n)
    print(f"Ways to climb {n} stairs: {result}")
    ```

=== "Memoization (Top-down)"

    ```python
    def climb_stairs_memo(n, memo=None):
        """
        Top-down DP with memoization - O(n) space, O(n) time.
        Solve subproblems and cache results.
        """
        if memo is None:
            memo = {}
        
        if n in memo:
            return memo[n]
        
        if n <= 2:
            return n
        
        memo[n] = climb_stairs_memo(n-1, memo) + climb_stairs_memo(n-2, memo)
        return memo[n]

    # Test
    n = 10
    result = climb_stairs_memo(n)
    print(f"Memoized result for n={n}: {result}")
    ```

=== "Naive Recursive"

    ```python
    def climb_stairs_recursive(n):
        """
        Naive recursive approach - O(2^n) time, O(n) space.
        Included for learning - too slow for large n.
        """
        if n <= 2:
            return n
        return climb_stairs_recursive(n-1) + climb_stairs_recursive(n-2)

    # Test (only for small n)
    for n in range(1, 8):
        result = climb_stairs_recursive(n)
        print(f"Recursive n={n}: {result}")
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Recurrence relation: `f(n) = f(n-1) + f(n-2)` (Fibonacci pattern)
    - Each step can be reached from step n-1 (take 1 step) or n-2 (take 2 steps)
    - Space optimization reduces O(n) to O(1) by keeping only last two values
    
    **⚡ Interview Tips:**
    
    - Start with recursive solution to establish recurrence
    - Always mention space optimization opportunity
    - Explain why it's similar to Fibonacci sequence
    
    **🔍 Extensions:**
    
    - Climbing stairs with 1, 2, or 3 steps allowed
    - Climbing stairs with some steps broken (obstacles)
    - Minimum cost climbing stairs (next problem)

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
        
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        
        for i in range(2, n):
            # Either rob current house + max from i-2, or skip current house
            dp[i] = max(dp[i-1], dp[i-2] + nums[i])
        
        return dp[n-1]

    # Test with trace
    nums = [2,7,9,3,1]
    result = rob_tabulation(nums)
    print(f"Tabulation result: {result}")
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Decision at each house: rob it (can't rob previous) or skip it
    - Recurrence: `dp[i] = max(dp[i-1], dp[i-2] + nums[i])`
    - Space optimization: only need previous two values
    
    **⚡ Interview Tips:**
    
    - Explain the constraint clearly (no adjacent houses)
    - Show how it reduces to a choice at each step
    - Mention circular house robber as follow-up
    
    **🔍 Variations:**
    
    - House Robber II (circular street)
    - House Robber III (binary tree)
    - Delete and earn (similar constraint pattern)

---

## Problem 3: Min Cost Climbing Stairs

**Difficulty**: 🟢 Easy  
**Pattern**: Linear DP with Cost Minimization  
**Time**: O(n), **Space**: O(1)

=== "Problem Statement"

    You can climb 1 or 2 steps at a time. Each step has a cost. Find the minimum cost to reach the top (beyond the last step).

    **Example:**
    ```text
    Input: cost = [10,15,20]
    Output: 15
    Explanation: Start at index 1, pay 15, climb 2 steps to reach top
    ```

=== "Optimal Solution"

    ```python
    def min_cost_climbing_stairs(cost):
        """
        Space-optimized DP - O(1) space, O(n) time.
        Can start from step 0 or step 1 for free.
        """
        n = len(cost)
        
        # Base cases: cost to reach step 0 and step 1
        prev2 = 0  # Cost to reach step 0 (can start here for free)
        prev1 = 0  # Cost to reach step 1 (can start here for free)
        
        # Calculate min cost to reach each step
        for i in range(2, n + 1):
            # To reach step i, either come from i-1 or i-2
            current = min(prev1 + cost[i-1], prev2 + cost[i-2])
            prev2 = prev1
            prev1 = current
        
        return prev1

    # Test cases
    test_cases = [
        [10, 15, 20],           # Expected: 15
        [1, 100, 1, 1, 1, 100, 1, 1, 100, 1],  # Expected: 6
        [0, 0, 0, 1],          # Expected: 0
    ]
    
    for cost in test_cases:
        result = min_cost_climbing_stairs(cost)
        print(f"Cost array {cost}: Min cost = {result}")
    ```

=== "Tabulation with Trace"

    ```python
    def min_cost_climbing_stairs_trace(cost):
        """
        DP with detailed trace for understanding.
        """
        n = len(cost)
        dp = [0] * (n + 1)  # dp[i] = min cost to reach step i
        
        # Base cases
        dp[0] = 0  # Can start at step 0 for free
        dp[1] = 0  # Can start at step 1 for free
        
        print(f"Cost array: {cost}")
        print(f"DP array initialization: {dp}")
        
        for i in range(2, n + 1):
            dp[i] = min(dp[i-1] + cost[i-1], dp[i-2] + cost[i-2])
            print(f"Step {i}: min({dp[i-1]} + {cost[i-1]}, {dp[i-2]} + {cost[i-2]}) = {dp[i]}")
        
        return dp[n]

    # Test with visualization
    cost = [10, 15, 20]
    result = min_cost_climbing_stairs_trace(cost)
    print(f"Final result: {result}")
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Can start at step 0 or step 1 for free (important detail)
    - Goal is to reach beyond the last step, not the last step itself
    - Pay cost when leaving a step, not when arriving
    
    **⚡ Interview Tips:**
    
    - Clarify: "When do we pay the cost?" (When leaving the step)
    - Ask: "Can we start at step 0 or 1?" (Usually both)
    - Distinguish from regular climbing stairs (has cost component)
    
    **🔍 Pattern Recognition:**
    
    - Similar to climbing stairs but with optimization objective
    - Foundation for more complex cost optimization problems
    - Related to minimum path sum in grids

---
- **Base cases**: `dp[1] = 1, dp[2] = 2`
- **Space optimization**: Only need last two values

---

## Problem 2: House Robber

**Difficulty**: 🟢 Easy  
**Pattern**: Linear DP with Choice  
**Time**: O(n), **Space**: O(1) optimized

### Problem Description

You are a robber planning to rob houses along a street. Each house has money, but you cannot rob two adjacent houses. What is the maximum amount you can rob?

```python
Input: nums = [2,7,9,3,1]
Output: 12
Explanation: Rob house 0 (money = 2), house 2 (money = 9) and house 4 (money = 1).
Total = 2 + 9 + 1 = 12.
```

### Solution

```python
def rob_recursive(nums, i=0, memo=None):
    """
    Recursive approach with memoization
    """
    if memo is None:
        memo = {}
    
    if i in memo:
        return memo[i]
    
    if i >= len(nums):
        return 0
    
    # Choice: rob current house or skip it
    rob_current = nums[i] + rob_recursive(nums, i + 2, memo)
    skip_current = rob_recursive(nums, i + 1, memo)
    
    memo[i] = max(rob_current, skip_current)
    return memo[i]

def rob_tabulation(nums):
    """
    Bottom-up tabulation approach
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    
    for i in range(2, n):
        # Choice: rob current house + dp[i-2] or skip (dp[i-1])
        dp[i] = max(dp[i-1], dp[i-2] + nums[i])
    
    return dp[n-1]

def rob_optimized(nums):
    """
    Space-optimized O(1) approach
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    prev2 = nums[0]              # max money up to house i-2
    prev1 = max(nums[0], nums[1])  # max money up to house i-1
    
    for i in range(2, len(nums)):
        current = max(prev1, prev2 + nums[i])
        prev2 = prev1
        prev1 = current
    
    return prev1

def rob_even_simpler(nums):
    """
    Most elegant solution
    """
    rob_prev = 0    # max money if we rob previous house
    not_rob_prev = 0  # max money if we don't rob previous house
    
    for money in nums:
        current_rob = not_rob_prev + money  # rob current house
        not_rob_prev = max(rob_prev, not_rob_prev)  # don't rob current
        rob_prev = current_rob
    
    return max(rob_prev, not_rob_prev)

# Test
nums = [2, 7, 9, 3, 1]
print(f"Maximum money: {rob_optimized(nums)}")  # 12
```

### Key Insights

- **State**: `dp[i]` = maximum money robbed from first i houses
- **Choice**: Rob house i (get `nums[i] + dp[i-2]`) or skip (get `dp[i-1]`)
- **Recurrence**: `dp[i] = max(dp[i-1], dp[i-2] + nums[i])`

---

## Problem 3: Maximum Subarray (Kadane's Algorithm)

**Difficulty**: 🟢 Easy  
**Pattern**: Linear DP  
**Time**: O(n), **Space**: O(1)

### Problem Description

Given an integer array `nums`, find the contiguous subarray with the largest sum and return its sum.

```python
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
```

### Solution

```python
def max_subarray_dp(nums):
    """
    DP approach: dp[i] = maximum sum ending at index i
    """
    if not nums:
        return 0
    
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    max_sum = dp[0]
    
    for i in range(1, n):
        # Choice: extend previous subarray or start new one
        dp[i] = max(nums[i], dp[i-1] + nums[i])
        max_sum = max(max_sum, dp[i])
    
    return max_sum

def max_subarray_optimized(nums):
    """
    Kadane's algorithm - space optimized
    """
    if not nums:
        return 0
    
    current_sum = nums[0]
    max_sum = nums[0]
    
    for i in range(1, len(nums)):
        # If current sum becomes negative, start fresh
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)
    
    return max_sum

def max_subarray_with_indices(nums):
    """
    Return maximum sum and the actual subarray
    """
    if not nums:
        return 0, []
    
    current_sum = nums[0]
    max_sum = nums[0]
    start = 0
    end = 0
    temp_start = 0
    
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
    
    return max_sum, nums[start:end+1]

# Test
nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
result, subarray = max_subarray_with_indices(nums)
print(f"Maximum sum: {result}")  # 6
print(f"Subarray: {subarray}")   # [4, -1, 2, 1]
```

### Key Insights

- **State**: Maximum sum ending at current position
- **Decision**: Extend current subarray or start new one
- **Core idea**: If current sum becomes negative, start fresh

---

## Problem 4: Min Cost Climbing Stairs

**Difficulty**: 🟢 Easy  
**Pattern**: Linear DP with Cost  
**Time**: O(n), **Space**: O(1)

### Problem Description

You can start from step 0 or 1. On each step, you can climb 1 or 2 steps. Each step has a cost. Find minimum cost to reach the top.

```python
Input: cost = [10,15,20]
Output: 15
Explanation: Start at index 1, pay 15, climb 2 steps to reach top.
```

### Solution

```python
def min_cost_climbing_stairs(cost):
    """
    DP approach: minimum cost to reach each step
    """
    n = len(cost)
    if n <= 2:
        return min(cost)
    
    # dp[i] = minimum cost to reach step i
    dp = [0] * (n + 1)
    dp[0] = cost[0]  # start from step 0
    dp[1] = cost[1]  # start from step 1
    
    for i in range(2, n):
        # Can reach step i from step i-1 or i-2
        dp[i] = cost[i] + min(dp[i-1], dp[i-2])
    
    # To reach top, we can come from last or second-last step
    return min(dp[n-1], dp[n-2])

def min_cost_climbing_stairs_optimized(cost):
    """
    Space-optimized O(1) solution
    """
    n = len(cost)
    if n <= 2:
        return min(cost)
    
    # Only need last two values
    prev2 = cost[0]
    prev1 = cost[1]
    
    for i in range(2, n):
        current = cost[i] + min(prev1, prev2)
        prev2 = prev1
        prev1 = current
    
    return min(prev1, prev2)

def min_cost_alternative(cost):
    """
    Alternative thinking: cost to reach beyond array
    """
    n = len(cost)
    
    # Add two more positions (top of stairs)
    cost = cost + [0, 0]
    
    for i in range(2, n + 2):
        cost[i] += min(cost[i-1], cost[i-2])
    
    return cost[n + 1]

# Test
cost = [10, 15, 20]
print(f"Minimum cost: {min_cost_climbing_stairs_optimized(cost)}")  # 15
```

### Key Insights

- **State**: Minimum cost to reach step i
- **Transition**: Can reach step i from i-1 or i-2
- **Final answer**: Minimum of reaching from last two steps

---

## Problem 5: Fibonacci Number

**Difficulty**: 🟢 Easy  
**Pattern**: Linear DP (Classic)  
**Time**: O(n), **Space**: O(1)

### Problem Description

The Fibonacci numbers form a sequence where each number is the sum of the two preceding ones.

```python
F(0) = 0, F(1) = 1
F(n) = F(n - 1) + F(n - 2) for n > 1
```

### Solution

```python
def fibonacci_recursive(n):
    """Naive recursive - exponential time"""
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

def fibonacci_memo(n, memo={}):
    """Memoization approach"""
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]

def fibonacci_dp(n):
    """Bottom-up tabulation"""
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]

def fibonacci_optimized(n):
    """Space-optimized O(1)"""
    if n <= 1:
        return n
    
    prev2, prev1 = 0, 1
    
    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
    
    return prev1

def fibonacci_matrix(n):
    """Matrix exponentiation - O(log n)"""
    if n <= 1:
        return n
    
    def matrix_multiply(A, B):
        return [[A[0][0]*B[0][0] + A[0][1]*B[1][0],
                 A[0][0]*B[0][1] + A[0][1]*B[1][1]],
                [A[1][0]*B[0][0] + A[1][1]*B[1][0],
                 A[1][0]*B[0][1] + A[1][1]*B[1][1]]]
    
    def matrix_power(mat, power):
        if power == 1:
            return mat
        if power % 2 == 0:
            half = matrix_power(mat, power // 2)
            return matrix_multiply(half, half)
        else:
            return matrix_multiply(mat, matrix_power(mat, power - 1))
    
    base = [[1, 1], [1, 0]]
    result = matrix_power(base, n)
    return result[0][1]

# Test
n = 10
print(f"Fibonacci({n}) = {fibonacci_optimized(n)}")  # 55
```

### All Approaches Comparison

| Approach | Time | Space | Notes |
|----------|------|-------|-------|
| Recursive | O(2^n) | O(n) | Exponential, impractical |
| Memoization | O(n) | O(n) | Top-down, intuitive |
| Tabulation | O(n) | O(n) | Bottom-up, iterative |
| Optimized | O(n) | O(1) | Space-efficient |
| Matrix | O(log n) | O(1) | Advanced technique |

---

## Problem 6: N-th Tribonacci Number

**Difficulty**: 🟢 Easy  
**Pattern**: Linear DP Extension  
**Time**: O(n), **Space**: O(1)

### Problem Description

Tribonacci sequence: T(n) = T(n-1) + T(n-2) + T(n-3) for n >= 3, with T(0) = 0, T(1) = 1, T(2) = 1.

### Solution

```python
def tribonacci(n):
    """Space-optimized tribonacci"""
    if n == 0:
        return 0
    if n <= 2:
        return 1
    
    # Keep track of last three values
    prev3, prev2, prev1 = 0, 1, 1
    
    for i in range(3, n + 1):
        current = prev1 + prev2 + prev3
        prev3, prev2, prev1 = prev2, prev1, current
    
    return prev1

def tribonacci_dp(n):
    """Tabulation approach"""
    if n == 0:
        return 0
    if n <= 2:
        return 1
    
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 1
    
    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2] + dp[i-3]
    
    return dp[n]

# Test
n = 7
print(f"Tribonacci({n}) = {tribonacci(n)}")  # 24
```

---

## 🎯 Practice Summary

### Key Patterns Learned

1. **Fibonacci Pattern**: `dp[i] = dp[i-1] + dp[i-2]`
2. **Choice Pattern**: `dp[i] = max/min(option1, option2)`
3. **Subarray Pattern**: Extend current or start new
4. **Cost Pattern**: Add current cost to optimal previous

### Space Optimization Techniques

- **Rolling Variables**: Keep only necessary previous values
- **In-place Updates**: Modify input array if allowed
- **Mathematical Formula**: Closed-form solutions when available

### Common Mistakes to Avoid

1. **Index errors**: Off-by-one in base cases
2. **State definition**: Unclear what dp[i] represents
3. **Base case handling**: Missing edge cases
4. **Space optimization**: Updating variables in wrong order

### Next Steps

Ready for more challenges? Try **[Medium DP Problems](medium-problems.md)** to explore:

- 2D DP problems (Unique Paths, Edit Distance)
- Knapsack variants (Coin Change, Partition)
- String DP problems (Word Break, Palindromes)

---

*These easy problems build the foundation for more complex DP patterns. Master the intuition here before moving to harder challenges!*

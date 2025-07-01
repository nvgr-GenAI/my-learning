# Dynamic Programming - Easy Problems

## 🎯 Learning Objectives

Master basic DP patterns and build intuition:

- Linear DP with simple recurrence relations
- Basic memoization and tabulation techniques
- Space optimization for 1D problems
- Pattern recognition for common DP scenarios

---

## Problem 1: Climbing Stairs

**Difficulty**: 🟢 Easy  
**Pattern**: Linear DP (Fibonacci-like)  
**Time**: O(n), **Space**: O(1) optimized

### Problem Description

You're climbing a staircase with `n` steps. Each time you can climb either 1 or 2 steps. In how many distinct ways can you climb to the top?

```python
Input: n = 3
Output: 3
Explanation: 
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps  
3. 2 steps + 1 step
```

### Solution Approaches

```python
def climb_stairs_recursive(n):
    """
    Naive recursive approach - exponential time
    """
    if n <= 2:
        return n
    return climb_stairs_recursive(n-1) + climb_stairs_recursive(n-2)

def climb_stairs_memo(n, memo=None):
    """
    Memoization approach - top-down DP
    """
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    
    if n <= 2:
        return n
    
    memo[n] = climb_stairs_memo(n-1, memo) + climb_stairs_memo(n-2, memo)
    return memo[n]

def climb_stairs_tabulation(n):
    """
    Tabulation approach - bottom-up DP
    """
    if n <= 2:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2
    
    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]

def climb_stairs_optimized(n):
    """
    Space-optimized approach - O(1) space
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
n = 5
print(f"Ways to climb {n} stairs: {climb_stairs_optimized(n)}")  # 8
```

### Key Insights

- **Recurrence**: `dp[i] = dp[i-1] + dp[i-2]` (Fibonacci pattern)
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

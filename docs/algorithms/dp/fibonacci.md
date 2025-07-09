# Fibonacci Sequence Pattern

## Introduction

The Fibonacci sequence pattern is one of the most fundamental dynamic programming patterns. It involves problems where each state depends on the sum or some function of a fixed number of previous states.

=== "Overview"
    **Core Idea**: Current state depends on the sum or some function of a fixed number of previous states.
    
    **When to Use**:
    
    - When each new value in a sequence depends on the previous k values
    - When calculating the nth term of a sequence with fixed recurrence relation
    - When there are multiple ways to reach the current state by combining previous states
    
    **Recurrence Relation**: `dp[i] = dp[i-1] + dp[i-2]` (for classical Fibonacci)
    
    **Real-World Applications**:
    
    - Growth of a rabbit population where adults produce new pairs
    - Calculating ways to ascend stairs taking 1 or 2 steps at a time
    - Determining ways to place dominos on a board of size n
    - Financial modeling of compound growth scenarios

=== "Example Problems"
    - **Fibonacci Numbers**: Calculate the nth Fibonacci number
      - Problem: F(0) = 0, F(1) = 1, F(n) = F(n-1) + F(n-2) for n > 1
      - DP solution directly implements the recurrence relation
    
    - **Climbing Stairs**: Count ways to reach the top taking 1 or 2 steps at a time
      - Problem: To reach stair n, you can come from stair n-1 (taking 1 step) or from stair n-2 (taking 2 steps)
      - Recurrence: `dp[i] = dp[i-1] + dp[i-2]`
    
    - **House Robber**: Maximum money you can rob without taking from adjacent houses
      - Problem: You can't rob consecutive houses, so you need to decide which ones to rob
      - Recurrence: `dp[i] = max(dp[i-1], dp[i-2] + nums[i])` (either skip current house or rob it and skip previous)
    
    - **Tribonacci Numbers**: Each number is the sum of the three preceding ones
      - Variation: F(n) = F(n-1) + F(n-2) + F(n-3)
      - Shows how the pattern extends to more previous states

=== "Visualization"
    For the Climbing Stairs problem with n = 5 stairs:
    
    ```
    n = 1: 1 way (take one step)
    n = 2: 2 ways (take two 1-steps or take one 2-step)
    n = 3: 3 ways (1+1+1, 1+2, 2+1)
    n = 4: 5 ways (1+1+1+1, 1+1+2, 1+2+1, 2+1+1, 2+2)
    n = 5: 8 ways (sum of ways for n=4 and n=3)
    ```
    
    The pattern builds like:
    
    ![Fibonacci Pattern Visualization](https://i.imgur.com/FmOEybj.png)
    
    Decision tree for House Robber problem:
    
    ![House Robber Decision Tree](https://i.imgur.com/DqWhJtM.png)

=== "Implementation"
    **Naive Recursive Approach** (inefficient):
    
    ```python
    def fibonacci_recursive(n):
        if n <= 1:
            return n
        return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)
    
    # Time Complexity: O(2^n)
    # Space Complexity: O(n) - recursion stack
    ```
    
    **Top-Down DP (Memoization)**:
    
    ```python
    def fibonacci_memoization(n, memo={}):
        if n in memo:
            return memo[n]
        if n <= 1:
            return n
            
        memo[n] = fibonacci_memoization(n - 1, memo) + fibonacci_memoization(n - 2, memo)
        return memo[n]
    
    # Time Complexity: O(n)
    # Space Complexity: O(n)
    ```
    
    **Bottom-Up DP (Tabulation)**:
    
    ```python
    def fibonacci_tabulation(n):
        if n <= 1:
            return n
        
        dp = [0] * (n + 1)
        dp[0], dp[1] = 0, 1
        
        for i in range(2, n + 1):
            dp[i] = dp[i-1] + dp[i-2]
        
        return dp[n]
    
    # Time Complexity: O(n)
    # Space Complexity: O(n)
    ```
    
    **Space-Optimized Version**:
    
    ```python
    def fibonacci_optimized(n):
        if n <= 1:
            return n
        
        prev2, prev1 = 0, 1
        
        for _ in range(2, n + 1):
            curr = prev1 + prev2
            prev2, prev1 = prev1, curr
        
        return prev1
    
    # Time Complexity: O(n)
    # Space Complexity: O(1)
    ```

=== "Tips and Insights"
    - **Pattern Recognition**: Look for problems where each state depends on a fixed number of previous states
    - **State Definition**: Define dp[i] as the answer to the problem for a specific size i
    - **Space Optimization**: Often you only need to keep track of the last few states, not the entire array
    - **Extension**: The basic pattern can be extended to depend on more than two previous states
    - **Variant**: Watch for variations where the relation isn't a simple sum but involves min, max, or other operations
    - **Hidden Fibonacci**: Some problems don't explicitly mention sequences but follow the same recurrence pattern
    - **Initialization**: Pay careful attention to base cases (usually dp[0] and dp[1])
    - **Matrix Exponentiation**: For very large n values, consider matrix exponentiation for O(log n) time complexity

### Memoization (Top-Down DP)

```python
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]

# Time Complexity: O(n)
# Space Complexity: O(n)
```

### Tabulation (Bottom-Up DP)

```python
def fibonacci_dp(n):
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]

# Time Complexity: O(n)
# Space Complexity: O(n)
```

### Space Optimized

```python
def fibonacci_optimized(n):
    if n <= 1:
        return n
    
    prev2 = 0
    prev1 = 1
    
    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1

# Time Complexity: O(n)
# Space Complexity: O(1)
```

## Variations

### Climbing Stairs

```python
def climb_stairs(n):
    """
    You can climb 1 or 2 steps at a time.
    How many ways to reach the top?
    """
    if n <= 2:
        return n
    
    prev2 = 1
    prev1 = 2
    
    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1
```

### House Robber

```python
def rob(nums):
    """
    Rob houses but can't rob adjacent houses.
    Maximum money that can be robbed.
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    prev2 = 0
    prev1 = nums[0]
    
    for i in range(1, len(nums)):
        current = max(prev1, prev2 + nums[i])
        prev2 = prev1
        prev1 = current
    
    return prev1
```

### Tribonacci

```python
def tribonacci(n):
    """
    T(n) = T(n-1) + T(n-2) + T(n-3)
    """
    if n == 0:
        return 0
    if n <= 2:
        return 1
    
    a, b, c = 0, 1, 1
    
    for i in range(3, n + 1):
        next_val = a + b + c
        a, b, c = b, c, next_val
    
    return c
```

## Analysis

### Recurrence Relations

- **Fibonacci**: F(n) = F(n-1) + F(n-2)
- **Climbing Stairs**: Same as Fibonacci
- **House Robber**: dp[i] = max(dp[i-1], dp[i-2] + nums[i])

### Time Complexity Comparison

| Approach | Time | Space |
|----------|------|-------|
| Recursive | O(2^n) | O(n) |
| Memoization | O(n) | O(n) |
| Tabulation | O(n) | O(n) |
| Optimized | O(n) | O(1) |

## Pattern Recognition

The Fibonacci pattern appears when:

1. **State depends on previous 1-2 states**
2. **Building solution from smaller subproblems**
3. **Linear recurrence relation**

## Practice Problems

- [ ] Fibonacci Number
- [ ] Climbing Stairs
- [ ] House Robber
- [ ] Min Cost Climbing Stairs
- [ ] N-th Tribonacci Number
- [ ] Delete and Earn
- [ ] Jump Game
- [ ] Jump Game II
- [ ] Maximum Alternating Subsequence Sum
- [ ] Unique Paths
- [ ] Unique Binary Search Trees
- [ ] Decode Ways
- [ ] Coin Change
- [ ] Domino and Tromino Tiling
- [ ] Paint House

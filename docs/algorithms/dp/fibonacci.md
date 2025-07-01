# Fibonacci Sequence

## Overview

The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones: 0, 1, 1, 2, 3, 5, 8, 13, 21, ...

## Approaches

### Naive Recursive Approach

```python
def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

# Time Complexity: O(2^n)
# Space Complexity: O(n) - recursion stack
```

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
